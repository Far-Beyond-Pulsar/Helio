//! Checks the vendored wgpu patch behind Helio#311: a command buffer encoded
//! once can be submitted again, interleaved with `queue.write_buffer`, and
//! each submission really executes.
//!
//! Skips when no adapter exists or the backend cannot reuse command buffers
//! (anything but Vulkan and D3D12).

const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;
@compute @workgroup_size(1)
fn main() { atomicAdd(&counter, 1u); }
"#;

fn read_u32(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer) -> u32 {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reusable readback"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(src, 0, &readback, 0, 4);
    queue.submit([encoder.finish()]);
    readback.slice(..).map_async(wgpu::MapMode::Read, |result| result.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let mapped = readback.slice(..).get_mapped_range().expect("readback is mapped");
    let value = u32::from_le_bytes(mapped[..4].try_into().unwrap());
    drop(mapped);
    readback.unmap();
    value
}

#[test]
fn reusable_command_buffers_resubmit_and_interleave_with_queue_writes() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: reusable command buffers");
            return;
        };
        let backend = adapter.get_info().backend;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("available adapter must create a device");

        let counter = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reusable counter"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reusable counter shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("reusable counter pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reusable counter bind group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: counter.as_entire_binding(),
            }],
        });

        // An unmarked command buffer is handed back untouched.
        let plain = device.create_command_encoder(&Default::default()).finish();
        let plain = plain
            .into_reusable()
            .expect_err("an unmarked command buffer must not become reusable");
        queue.submit([plain]);

        let mut encoder = device.create_command_encoder(&Default::default());
        if !encoder.mark_reusable() {
            assert!(
                !matches!(backend, wgpu::Backend::Vulkan | wgpu::Backend::Dx12),
                "{backend:?} must support reusable command buffers"
            );
            eprintln!("reusable command buffers: unsupported on {backend:?}, skipping");
            return;
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        let increment = encoder
            .finish()
            .into_reusable()
            .expect("a marked command buffer must become reusable");

        queue.write_buffer(&counter, 0, &0u32.to_le_bytes());
        for _ in 0..3 {
            queue.submit_mixed([wgpu::SubmitItem::Reusable(&increment)]);
        }
        assert_eq!(read_u32(&device, &queue, &counter), 3, "three replays");

        // A queue write between replays must land before the next replay.
        queue.write_buffer(&counter, 0, &10u32.to_le_bytes());
        let tail = device.create_command_encoder(&Default::default()).finish();
        queue.submit_mixed([
            wgpu::SubmitItem::Reusable(&increment),
            wgpu::SubmitItem::Once(tail),
            wgpu::SubmitItem::Reusable(&increment),
        ]);
        assert_eq!(read_u32(&device, &queue, &counter), 12, "write, then two replays");

        // Dropping it while nothing is in flight, and again after reuse, is fine.
        drop(increment);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    });
}
