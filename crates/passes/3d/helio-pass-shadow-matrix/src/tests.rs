use super::*;

#[test]
fn empty_sparse_rows_cannot_write_shadow_slot_zero() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None, force_fallback_adapter: false, apply_limit_buckets: false,
        }).await.expect("GPU required");
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: adapter.limits(), ..Default::default()
        }).await.unwrap();
        device.on_uncaptured_error(std::sync::Arc::new(|error| panic!("{error:?}")));
        let buffer = |size| device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size, mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let lights = buffer(1025 * 128);
        let matrices = buffer(6 * 64);
        let camera = buffer(736);
        let dirty = buffer(4);
        let hashes = buffer(4);
        let pass = ShadowMatrixPass::new(&device, &lights, &matrices, &camera, &dirty, &hashes, 1024);
        queue.write_buffer(&pass.uniform_buf, 0, bytemuck::bytes_of(&ShadowMatrixUniforms {
            light_count: 1025, shadow_atlas_size: 1024, _pad: [0; 2],
        }));
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: 388, mapped_at_creation: false,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        });
        for live in [false, true] {
            if live {
                let mut words = [0u32; 32];
                words[3] = 10.0f32.to_bits(); // range
                words[11] = 1.0f32.to_bits(); // intensity
                words[13] = 1; // point light, shadow base=0
                queue.write_buffer(&lights, 1024 * 128, bytemuck::cast_slice(&words));
            }
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut compute = encoder.begin_compute_pass(&Default::default());
                compute.set_pipeline(&pass.pipeline);
                compute.set_bind_group(0, &pass.bind_group, &[]);
                compute.dispatch_workgroups(1025u32.div_ceil(64), 1, 1);
            }
            encoder.copy_buffer_to_buffer(&matrices, 0, &staging, 0, 384);
            encoder.copy_buffer_to_buffer(&dirty, 0, &staging, 384, 4);
            queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            staging.slice(..).map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            rx.recv().unwrap().unwrap();
            {
                let bytes = staging.slice(..).get_mapped_range().unwrap();
                let values: &[f32] = bytemuck::cast_slice(&bytes[..384]);
                assert!(values.iter().all(|v| v.is_finite()));
                if live {
                    assert!(values.iter().any(|v| *v != 0.0));
                    assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes[384..])[0], 1);
                } else {
                    assert!(bytes.iter().all(|v| *v == 0), "unused rows wrote a shadow matrix or dirty flag");
                }
            }
            staging.unmap();
        }
    });
}

/// GPU allocation ranks local requests by camera visibility, projected
/// influence and distance. Directionals remain highest priority, and existing
/// winners receive hysteresis to stabilize slots at the budget edge.
#[test]
fn caster_allocation_assigns_the_most_important_requests() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None, force_fallback_adapter: false, apply_limit_buckets: false,
        }).await.expect("GPU required");
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: adapter.limits(), ..Default::default()
        }).await.unwrap();
        device.on_uncaptured_error(std::sync::Arc::new(|error| panic!("{error:?}")));
        let buffer = |size| device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size, mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        const ROWS: u64 = 600; // More than one row per thread.
        let lights = buffer(ROWS * 128);
        // Twelve faces: two casters.
        let pass = ShadowMatrixPass::new(&device, &lights, &buffer(12 * 64), &buffer(736), &buffer(8), &buffer(8), 1024);
        assert_eq!(pass.caster_capacity(), 2);
        queue.write_buffer(&pass.caster_params_buf, 0, bytemuck::bytes_of(&CasterParams {
            row_count: ROWS as u32, caster_capacity: 2, nonce: 1, _pad: 0,
        }));

        // Identity view-projection: x=4 is outside the view. The strong light
        // there must lose to the weaker visible request.
        let mut camera_words = [0u32; 184];
        for diagonal in [0usize, 5, 10, 15] { camera_words[32 + diagonal] = 1.0f32.to_bits(); }
        camera_words[67] = 0.1f32.to_bits();
        queue.write_buffer(&pass.camera_buf, 0, bytemuck::cast_slice(&camera_words));

        // (row, light_type, intensity, range, x, shadow_index as authored)
        let authored = [
            (3u64, 1u32, 4.9f32, 1.0f32, 0.0f32, 0u32),
            (305, 1, 100.0, 0.5, 4.0, 0),
            (410, 0, 0.1, 0.0, 0.0, 0),
            (450, 1, 1e6, 10.0, 0.0, u32::MAX),
            (599, 1, 5.0, 1.0, 0.1, 0),
        ];
        for &(row, light_type, intensity, range, x, shadow_index) in &authored {
            let mut words = [0u32; 32];
            words[0] = x.to_bits();
            words[3] = range.to_bits();
            words[11] = intensity.to_bits();
            words[12] = shadow_index;
            words[13] = light_type;
            queue.write_buffer(&lights, row * 128, bytemuck::cast_slice(&words));
        }

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: ROWS * 128, mapped_at_creation: false,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        });
        for run in 0..2 {
            if run == 1 {
                // Raise the challenger to the incumbent's base score. The
                // incumbent bonus should prevent a slot swap at the boundary.
                let mut words = [0u32; 32];
                words[3] = 1.0f32.to_bits();
                words[11] = 5.0f32.to_bits();
                words[12] = 0;
                words[13] = 1;
                queue.write_buffer(&lights, 3 * 128, bytemuck::cast_slice(&words));
            }
            let mut encoder = device.create_command_encoder(&Default::default());
            pass.record_caster_allocation(&mut encoder);
            encoder.copy_buffer_to_buffer(&lights, 0, &staging, 0, ROWS * 128);
            queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            staging.slice(..).map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            rx.recv().unwrap().unwrap();
            {
                let bytes = staging.slice(..).get_mapped_range().unwrap();
                let words: &[u32] = bytemuck::cast_slice(&bytes);
                let slot = |row: usize| words[row * 32 + 12];
                let pad = |row: usize| words[row * 32 + 15];
                // Visible row 599 wins the local slot; the brighter row 305
                // is outside the view. The sun gets the next slot.
                assert_eq!(slot(599), 0, "run {run}");
                assert_eq!(slot(410), 6, "run {run}");
                assert_eq!(slot(3), u32::MAX, "run {run}");
                assert_eq!(slot(305), u32::MAX, "offscreen light is not selected (run {run})");
                assert_eq!(slot(450), u32::MAX, "unrequested lights never take a slot (run {run})");
                // Request kept (bit 2), row marked (bit 3), legacy ray-traced
                // intent pinned (bits 0-1) for requesting and other lights.
                assert_eq!(pad(3), 0b1111, "run {run}");
                assert_eq!(pad(599), 0b1_1111, "winner is retained for hysteresis (run {run})");
                assert_eq!(pad(410), 0b1_1111, "directional winner is retained (run {run})");
                assert_eq!(pad(450), 0b1001, "run {run}");
                // Vacant rows are untouched.
                assert!(words[..32].iter().all(|w| *w == 0), "run {run}");
            }
            staging.unmap();
        }
    });
}
