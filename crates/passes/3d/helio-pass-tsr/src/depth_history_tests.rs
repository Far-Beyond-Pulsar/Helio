use super::*;
use wgpu::util::DeviceExt;

#[test]
fn resolve_rejects_wrong_surface_and_writes_current_linear_depth() {
    pollster::block_on(async {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        let pass = TsrPass::new(
            &device,
            64,
            8,
            64,
            8,
            wgpu::TextureFormat::Rgba8Unorm,
            TsrQuality::Native,
        );
        let extent = wgpu::Extent3d {
            width: 64,
            height: 8,
            depth_or_array_layers: 1,
        };
        let current = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let coverage_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test transparency coverage"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let coverage_view = coverage_texture.create_view(&Default::default());
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let current_view = current.create_view(&Default::default());
        let depth_view = depth.create_view(&Default::default());
        let mut pixels = Vec::new();
        for _y in 0..8 {
            for x in 0..64 {
                let v = 60 + x * 2;
                pixels.extend_from_slice(&[v as u8, v as u8, v as u8, 255]);
            }
        }
        queue.write_texture(
            current.as_image_copy(),
            &pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(8),
            },
            extent,
        );
        // Half-float 0.5 in history, while the current image is a gray ramp.
        // Neighborhood clamping still leaves an observable accepted history term.
        let history: Vec<u16> = (0..512)
            .flat_map(|_| [0x3800, 0x3800, 0x3800, 0x3c00])
            .collect();
        queue.write_texture(
            pass.history_texture.as_image_copy(),
            bytemuck::cast_slice(&history),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(512),
                rows_per_image: Some(8),
            },
            extent,
        );
        let mut camera = helio_core::GpuCameraUniforms::zeroed();
        let mut identity = [0.0; 16];
        for i in [0, 5, 10, 15] {
            identity[i] = 1.0;
        }
        camera.view = identity;
        camera.inv_view_proj = identity;
        camera.inv_view_proj[14] = -10.5;
        camera.prev_view_proj = identity;
        camera.prev_view_proj[14] = 10.5;
        let cameras = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[camera, camera]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pass.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&current_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&pass.history_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&pass.linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&pass.point_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: cameras.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: pass.uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&coverage_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&pass.history_depth_view),
                },
            ],
        });
        let probe = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                r#"
            @group(0) @binding(0) var color: texture_2d<f32>;
            @group(0) @binding(1) var depth: texture_2d<f32>;
            @group(0) @binding(2) var<storage,read_write> result: array<vec4<f32>,2>;
            @compute @workgroup_size(1) fn read_pixel() {
                result[0]=textureLoad(color,vec2<i32>(32,4),0);
                result[1]=vec4<f32>(textureLoad(depth,vec2<i32>(32,4),0).r);
            }
        "#
                .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &probe,
            entry_point: Some("read_pixel"),
            compilation_options: Default::default(),
            cache: None,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 32,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 32,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let probe_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&pass.output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&pass.output_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut results = Vec::new();
        for (stored_depth, flags, reactivity, time_delta, coverage, previous_coverage) in [
            (10.0f32, 0u32, 0.0, 1.0 / 60.0, 0u8, 0u16),
            (2.0, 0, 0.0, 1.0 / 60.0, 0, 0),
            (10.0, 1, 0.0, 1.0 / 60.0, 0, 0),
            (10.0, 0, 1.0, 1.0 / 120.0, 0, 0),
            (10.0, 0, 1.0, 1.0 / 60.0, 0, 0),
            (10.0, 0, 1.0, 1.0 / 15.0, 0, 0),
            (10.0, 0, 1.0, 0.5, 0, 0),
            (10.0, 2, 0.0, 1.0 / 60.0, 0, 0), // opaque pixels unchanged
            (10.0, 2, 0.0, 1.0 / 60.0, 255, 0), // arriving glass
            (10.0, 2, 0.0, 1.0 / 60.0, 0, 0x3c00), // departing glass
            (10.0, 2, 0.0, 1.0 / 60.0, 128, 0), // partial coverage
            (10.0, 3, 0.0, 1.0 / 60.0, 128, 0), // reset still stores coverage
            (10.0, 0, 0.0, 1.0 / 60.0, 255, 0x3c00), // disabled ignores coverage
        ] {
            queue.write_texture(
                coverage_texture.as_image_copy(),
                &vec![coverage; 512],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(64),
                    rows_per_image: Some(8),
                },
                extent,
            );
            let history: Vec<u16> = (0..512)
                .flat_map(|_| [0x3800, 0x3800, 0x3800, previous_coverage])
                .collect();
            queue.write_texture(
                pass.history_texture.as_image_copy(),
                bytemuck::cast_slice(&history),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(512),
                    rows_per_image: Some(8),
                },
                extent,
            );
            queue.write_texture(
                pass.history_depth.as_image_copy(),
                bytemuck::cast_slice(&vec![stored_depth; 512]),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(8),
                },
                extent,
            );
            let uniform = TsrUniform {
                jitter_offset: [0.0; 2],
                reactivity,
                flags,
                time_delta,
                tap_radius: 2,
                previous_jitter_uv: [0.0; 2],
                previous_view: identity,
            };
            queue.write_buffer(&pass.uniform_buf, 0, bytemuck::bytes_of(&uniform));
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0.5),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
            }
            {
                let attachments = [
                    Some(wgpu::RenderPassColorAttachment {
                        view: &pass.output_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &pass.output_depth_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ];
                let mut draw = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &attachments,
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                draw.set_pipeline(&pass.pipeline);
                draw.set_bind_group(0, &bind, &[]);
                draw.draw(0..3, 0..1);
            }
            {
                let mut compute = encoder.begin_compute_pass(&Default::default());
                compute.set_pipeline(&pipeline);
                compute.set_bind_group(0, &probe_bind, &[]);
                compute.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&output, 0, &read, 0, 32);
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
        assert_eq!(
            results[1], results[2],
            "wrong-surface history must match a reset"
        );
        assert!(
            (results[0][0] - results[2][0]).abs() > 0.0001,
            "matching history must accumulate: {results:?}"
        );
        for result in &results[3..7] {
            assert_eq!(
                result, &results[2],
                "full reactivity must discard history at every frame rate"
            );
        }
        assert_eq!(
            &results[7][..3],
            &results[0][..3],
            "opaque RGB must be unchanged"
        );
        assert_eq!(results[12], results[0], "disabled coverage must be ignored");
        for i in [8, 9, 11] {
            assert_eq!(
                &results[i][..3],
                &results[2][..3],
                "local full reactivity/reset must discard history"
            );
        }
        assert_eq!(results[8][3], 1.0);
        assert_eq!(results[9][3], 0.0, "departed coverage must not persist");
        assert_eq!(results[7][3], 0.0);
        for i in [10, 11] {
            assert!((results[i][3] - 128.0 / 255.0).abs() < 0.001);
        }
        assert!(
            (results[10][0] - results[7][0]).abs() > 0.0001,
            "partial coverage must reduce history"
        );
        for result in results {
            assert!(
                (result[4] - 10.0).abs() < 0.00001,
                "stored depth: {result:?}"
            );
        }
    });
}
