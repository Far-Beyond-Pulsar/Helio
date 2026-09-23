use super::{exposure_groups, PostProcessPass};

#[test]
#[ignore = "requires a GPU"]
fn exposure_reduction_matches_cpu_for_multiple_groups_and_partial_edges() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(
            wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
        );
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let shader_source = PostProcessPass::build_shader_source(&[]);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Exposure reduction test shader"),
            source: wgpu::ShaderSource::Wgsl(
                helio_core::shader::resolve(&shader_source).into_owned().into(),
            ),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Exposure reduction test layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 15,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Exposure reduction test pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let sample = pipeline("cs_exposure");
        let reduce = pipeline("cs_exposure_reduce");

        for (width, height) in [(1u32, 1u32), (137, 83), (384, 140)] {
            let pixels: Vec<u8> = (0..height)
                .flat_map(|y| {
                    (0..width).flat_map(move |x| {
                        [
                            ((x * 13 + y * 5) % 250 + 1) as u8,
                            ((x * 7 + y * 17) % 250 + 1) as u8,
                            ((x * 19 + y * 11) % 250 + 1) as u8,
                            255,
                        ]
                    })
                })
                .collect();
            let expected = {
                let mut sum = 0.0f64;
                let mut count = 0usize;
                for y in (0..height).step_by(4) {
                    for x in (0..width).step_by(4) {
                        let i = ((y * width + x) * 4) as usize;
                        let rgb = [pixels[i] as f64, pixels[i + 1] as f64, pixels[i + 2] as f64];
                        let luminance = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255.0;
                        sum += luminance.max(0.0001).log2();
                        count += 1;
                    }
                }
                (sum / count as f64) as f32
            };
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Exposure reduction test input"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                texture.as_image_copy(),
                &pixels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 4),
                    rows_per_image: Some(height),
                },
                texture.size(),
            );
            let (gx, gy) = exposure_groups(width, height);
            let partials = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Exposure reduction test partials"),
                size: u64::from(gx) * u64::from(gy) * 8,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let output = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Exposure reduction test output"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Exposure reduction test readback"),
                size: 4,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Exposure reduction test bindings"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&texture.create_view(&Default::default())) },
                    wgpu::BindGroupEntry { binding: 11, resource: output.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 15, resource: partials.as_entire_binding() },
                ],
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            for (pipeline, x, y) in [(&sample, gx, gy), (&reduce, 1, 1)] {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(x, y, 1);
            }
            encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 4);
            queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            readback.slice(..).map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            rx.recv().unwrap().unwrap();
            let bytes = readback.slice(..).get_mapped_range().unwrap();
            let actual = f32::from_le_bytes(bytes[..4].try_into().unwrap());
            assert!((actual - expected).abs() < 0.0005, "{width}x{height}: {actual} vs {expected}");
            drop(bytes);
            readback.unmap();
        }
    });
}
