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
