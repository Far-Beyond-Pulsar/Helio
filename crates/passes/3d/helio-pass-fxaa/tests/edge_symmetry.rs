#[test]
#[ignore = "requires a GPU"]
fn edges_are_reflection_symmetric_and_constant_borders_stay_constant() {
    pollster::block_on(async {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        // Invoke the production pixel function at exact texel centers. The
        // asymmetric triangle and its reflection expose directional edge bias.
        let source = include_str!("../shaders/fxaa.wgsl")
            .replace("@fragment", "")
            .replace("-> @location(0) vec4<f32>", "-> vec4<f32>");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(format!("{source}\n\
                @group(0) @binding(2) var<storage,read_write> output: array<vec4<f32>>;\n\
                @compute @workgroup_size(8,8) fn check(@builtin(global_invocation_id) id: vec3<u32>) {{\n\
                  var pixel: VertexOutput;\n\
                  pixel.position=vec4<f32>(vec2<f32>(id.xy)+0.5,0.0,1.0);\n\
                  pixel.uv=(vec2<f32>(id.xy)+0.5)/64.0;\n\
                  output[id.y*64u+id.x]=fs_main(pixel);\n\
                }}").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("check"),
            compilation_options: Default::default(),
            cache: None,
        });
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 65536,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 65536,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut results = Vec::new();
        for mode in 0..4 {
            let mut pixels = Vec::new();
            for y in 0..64 {
                for x in 0..64 {
                    let xx = if mode == 1 { 63 - x } else { x };
                    let yy = if mode == 2 { 63 - y } else { y };
                    let value = if mode == 3 || (xx > 8 && yy > 9 && 2 * xx + yy < 104) {
                        204u8
                    } else {
                        26u8
                    };
                    pixels.extend_from_slice(&[value, value, value, 255]);
                }
            }
            queue.write_texture(
                texture.as_image_copy(),
                &pixels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(64),
                },
                texture.size(),
            );
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(8, 8, 1);
            }
            encoder.copy_buffer_to_buffer(&output, 0, &read, 0, 65536);
            queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            read.slice(..)
                .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            rx.recv().unwrap().unwrap();
            let data = read.slice(..).get_mapped_range().unwrap();
            results.push(bytemuck::cast_slice::<u8, f32>(&data).to_vec());
            drop(data);
            read.unmap();
        }
        let blended = results[0]
            .chunks_exact(4)
            .filter(|p| p[0] > 0.11 && p[0] < 0.79)
            .count();
        assert!(
            blended > 20,
            "diagonal was not antialiased: {blended} blended pixels"
        );
        for y in 0..64 {
            for x in 0..64 {
                let a = results[0][(y * 64 + x) * 4];
                let b = results[1][(y * 64 + 63 - x) * 4];
                let c = results[2][((63 - y) * 64 + x) * 4];
                assert!(
                    (a - b).abs() < 0.002 && (a - c).abs() < 0.002,
                    "directional bias at {x},{y}: {a} {b} {c}"
                );
                assert!(
                    (results[3][(y * 64 + x) * 4] - 0.8).abs() < 0.0001,
                    "constant border changed"
                );
            }
        }
    });
}
