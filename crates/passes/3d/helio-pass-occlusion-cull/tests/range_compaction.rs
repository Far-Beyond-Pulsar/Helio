mod support;

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn make_buffer(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

fn read_u32(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer) -> Vec<u32> {
    let readback = make_buffer(
        device,
        "Range compaction readback",
        source.size(),
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    );
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(source, 0, &readback, 0, source.size());
    queue.submit([encoder.finish()]);
    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range().unwrap();
    let result = bytemuck::cast_slice::<u8, u32>(&mapped).to_vec();
    drop(mapped);
    readback.unmap();
    result
}

#[test]
fn compacts_surviving_indirect_draws_and_writes_per_range_count() {
    pollster::block_on(async {
        let Some((device, queue)) = support::request_test_device("GPU Range Compaction").await else {
            eprintln!("skipping range compaction test: no GPU adapter available");
            return;
        };

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Range compaction test BGL"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, false),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, true),
                storage_entry(6, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Range compaction test PL"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Range compaction test shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/compact_ranges.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Range compaction test pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compact_ranges"),
            compilation_options: Default::default(),
            cache: None,
        });

        let source = make_buffer(
            &device,
            "Compaction source indirect",
            4 * 20,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        let compacted = make_buffer(
            &device,
            "Compaction output indirect",
            4 * 20,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        let range_counts = make_buffer(
            &device,
            "Compaction range counts",
            16,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let ranges = make_buffer(
            &device,
            "Compaction range tables",
            20,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let draw_counts = make_buffer(
            &device,
            "Compaction draw counts",
            16 * 4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        let params = make_buffer(
            &device,
            "Compaction params",
            3 * 256,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let commands: [u32; 20] = [
            3, 1, 0, 0, 10,
            3, 0, 0, 0, 11,
            3, 1, 0, 0, 12,
            3, 0, 0, 0, 13,
        ];
        queue.write_buffer(&source, 0, bytemuck::cast_slice(&commands));
        queue.write_buffer(&range_counts, 0, bytemuck::cast_slice(&[1u32, 0, 0, 1]));
        queue.write_buffer(&ranges, 0, bytemuck::cast_slice(&[0u32, 0, 0, 0, 4]));
        for bucket in 0..3u32 {
            queue.write_buffer(&params, bucket as u64 * 256, bytemuck::cast_slice(&[4u32, bucket]));
        }
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Range compaction test BG"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: source.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: compacted.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: range_counts.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: ranges.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: ranges.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: ranges.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: draw_counts.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params,
                        offset: 0,
                        size: std::num::NonZeroU64::new(8),
                    }),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&source, 0, &compacted, 0, source.size());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            for bucket in 0..3u32 {
                pass.set_bind_group(0, &bg, &[bucket * 256]);
                pass.dispatch_workgroups(1, 1, 1);
            }
        }
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

        let args = read_u32(&device, &queue, &compacted);
        assert_eq!(args[1], 1);
        assert_eq!(args[6], 1);
        assert_eq!(args[11], 0);
        assert_eq!(args[16], 0);
        let mut first_instances = [args[4], args[9]];
        first_instances.sort_unstable();
        assert_eq!(first_instances, [10, 12], "surviving args are packed at range head");
        let counts = read_u32(&device, &queue, &draw_counts);
        assert_eq!(counts[4], 2);
    });
}
