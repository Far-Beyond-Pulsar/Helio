use super::*;
use glam::{DMat4, DVec4, Mat4, Vec3};
use wgpu::util::DeviceExt;

/// Stored TSR images are centered. Compare GPU history coordinates against
/// unjittered f64 transforms for static and moving cameras at several depths.
#[test]
fn history_reprojection_removes_both_projection_jitters() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        let source = format!(
            "{}\n{}",
            include_str!("../shaders/tsr_main.wgsl"),
            r#"
@group(0) @binding(10) var<storage,read_write> test_uv:array<vec4<f32>>;
@compute @workgroup_size(64)
fn validate_history(@builtin(global_invocation_id) id:vec3<u32>) {
    let uv=(vec2<f32>(f32(id.x%8u),f32(id.x/8u))+0.5)/8.0;
    let depth=array<f32,4>(0.5,0.9,0.99,0.9999)[id.x%4u];
    var result=reproject_history(uv,depth,vec2<f32>(960.0,540.0));
    if tsr.reset!=0u {result=reproject_sample_motion(uv,depth,vec2<f32>(960.0,540.0)).xy;}
    test_uv[id.x]=vec4<f32>(result,depth,1.0);
}
"#
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("validate_history"),
            compilation_options: Default::default(),
            cache: None,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 1024,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 1024,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        #[allow(deprecated)]
        let projection = Mat4::perspective_rh(60_f32.to_radians(), 960.0 / 540.0, 0.1, 10000.0);
        let to_double = |m: Mat4| DMat4::from_cols_array(&m.to_cols_array().map(f64::from));
        for (moving, sampled) in [(false, false), (true, false), (false, true), (true, true)] {
            let view = if moving {
                Mat4::from_rotation_y(0.006) * Mat4::from_translation(Vec3::new(-0.002, 0.001, 0.0))
            } else {
                Mat4::IDENTITY
            };
            for frame in [1, 2, 17, 1024] {
                let jitter = r1_r2_jitter(frame);
                let old_jitter = r1_r2_jitter(frame - 1);
                let translate = |j: [f32; 2]| {
                    Mat4::from_translation(Vec3::new(j[0] * 2.0 / 960.0, j[1] * 2.0 / 540.0, 0.0))
                };
                let camera = helio_core::GpuCameraUniforms::new(
                    view,
                    translate(jitter) * projection,
                    Vec3::ZERO,
                    0.1,
                    10000.0,
                    frame as u32,
                    [jitter[0] * 2.0 / 960.0, jitter[1] * 2.0 / 540.0],
                    translate(old_jitter) * projection,
                );
                let params = TsrUniform {
                    jitter_offset: jitter,
                    previous_jitter: old_jitter,
                    reactivity: 0.0,
                    reset: u32::from(sampled),
                    time_delta: 1.0 / 60.0,
                    tap_radius: 1,
                    clip_to_previous: clip_to_previous(&camera).unwrap(),
                };
                let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[(6, &uniform), (10, &output)].map(|(binding, b)| {
                        wgpu::BindGroupEntry {
                            binding,
                            resource: b.as_entire_binding(),
                        }
                    }),
                });
                let mut encoder = device.create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &group, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
                encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, 1024);
                queue.submit(Some(encoder.finish()));
                let (tx, rx) = std::sync::mpsc::channel();
                staging
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                rx.recv().unwrap().unwrap();
                {
                    let mapped = staging.slice(..).get_mapped_range().unwrap();
                    let actual: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
                    for (i, result) in actual.iter().enumerate() {
                        let uv = [(i % 8) as f64 / 8.0 + 0.0625, (i / 8) as f64 / 8.0 + 0.0625];
                        let sample_uv = if sampled {
                            std::array::from_fn(|axis| {
                                let offset = f64::from(jitter[axis]) * [1.0, -1.0][axis];
                                ((uv[axis] * [960.0, 540.0][axis] + offset).floor() + 0.5 - offset)
                                    / [960.0, 540.0][axis]
                            })
                        } else {
                            uv
                        };
                        let clip = DVec4::new(
                            sample_uv[0] * 2.0 - 1.0,
                            1.0 - sample_uv[1] * 2.0,
                            f64::from(result[2]),
                            1.0,
                        );
                        let previous =
                            to_double(projection) * to_double(projection * view).inverse() * clip;
                        let expected = [
                            previous.x / previous.w * 0.5 + 0.5 + uv[0] - sample_uv[0],
                            0.5 - previous.y / previous.w * 0.5 + uv[1] - sample_uv[1],
                        ];
                        for axis in 0..2 {
                            let error = (f64::from(result[axis]) - expected[axis]).abs()
                                * [960.0, 540.0][axis];
                            assert!(error<0.005, "history shifted {error} pixels: moving={moving}, sampled={sampled}, frame={frame}, pixel={i}");
                        }
                    }
                }
                staging.unmap();
            }
        }
    });
}

/// Render the production resolve with old color inside the current neighborhood
/// bounds. Only geometric history rejection can remove that old color. Camera
/// translation makes matching previous depth differ from current depth.
#[test]
fn resolve_rejects_revealed_surfaces_and_retains_matching_previous_projection() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        let size = wgpu::Extent3d {
            width: 8,
            height: 8,
            depth_or_array_layers: 1,
        };
        let make = |format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
        };
        let current = make(wgpu::TextureFormat::Rgba16Float);
        let current_view = current.create_view(&Default::default());
        let depth = make(wgpu::TextureFormat::Depth32Float);
        let depth_view = depth.create_view(&Default::default());
        let final_color = make(wgpu::TextureFormat::Rgba16Float);
        let final_view = final_color.create_view(&Default::default());
        let pass = TsrPass::new(
            &device,
            8,
            8,
            8,
            8,
            wgpu::TextureFormat::Rgba16Float,
            TsrQuality::Native,
        );
        let pixels: Vec<u16> = (0..64)
            .flat_map(|i| {
                let c = if (i % 8 + i / 8) % 2 == 0 { 0 } else { 0x3c00 };
                [c, c, c, 0x3c00]
            })
            .collect();
        queue.write_texture(
            current.as_image_copy(),
            bytemuck::cast_slice(&pixels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(64),
                rows_per_image: Some(8),
            },
            size,
        );
        #[allow(deprecated)]
        let projection = Mat4::perspective_rh(60_f32.to_radians(), 1.0, 0.1, 1000.0);
        let previous = projection * Mat4::from_translation(Vec3::new(0.0, 0.0, -2.0));
        let camera = helio_core::GpuCameraUniforms::new(
            Mat4::IDENTITY,
            projection,
            Vec3::ZERO,
            0.1,
            1000.0,
            1,
            [0.0; 2],
            previous,
        );
        let cameras = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[camera; 2]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let u = TsrUniform {
            jitter_offset: [0.0; 2],
            previous_jitter: [0.0; 2],
            reactivity: 0.0,
            reset: 0,
            time_delta: 1.0 / 60.0,
            tap_radius: 2,
            clip_to_previous: clip_to_previous(&camera).unwrap(),
        };
        queue.write_buffer(&pass.uniform_buf, 0, bytemuck::bytes_of(&u));
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&pass.history_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&pass.moments[0].history_view),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(&pass.moments[1].history_view),
                },
            ],
        });
        let projected_depth = |meters: f32| {
            let clip = projection * glam::Vec4::new(0.0, 0.0, -meters, 1.0);
            clip.z / clip.w
        };
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 10240,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let depth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("depth footprint fixture"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n@fragment fn fs_depth()->@builtin(frag_depth) f32 {{return {:.9};}}",
                    BLIT_WGSL,
                    projected_depth(20.0)
                )
                .into(),
            ),
        });
        let depth_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                module: &depth_shader,
                entry_point: Some("vs_blit"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &depth_shader,
                entry_point: Some("fs_depth"),
                targets: &[],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        // Matching surface, foreground occluder, old sky, and mixed edge coverage.
        // The mixed case has an invalid nearest depth but valid neighboring taps.
        for (stored, accept, mixed_edge, depth_boundary, fixture) in [
            (projected_depth(12.0), true, false, false, 0),
            (projected_depth(1.0), false, false, false, 0),
            (1.0, false, false, false, 0),
            (projected_depth(12.0), true, true, false, 0),
            (projected_depth(12.0), false, false, true, 0),
            (projected_depth(12.0), false, false, false, 1),
            (projected_depth(12.0), false, false, false, 2),
            (projected_depth(12.0), false, false, false, 3),
            (projected_depth(12.0), true, false, false, 4),
            (projected_depth(1.0), false, false, false, 5),
            (projected_depth(12.0), false, false, false, 6),
        ] {
            let mut frame_uniform = u;
            if fixture >= 4 {
                // A black current sample does not invalidate a previously
                // measured noisy surface. A depth change or explicit reset
                // must still discard both colour and its moment history.
                frame_uniform.reset = u32::from(fixture == 6);
                let values = vec![[0u16, 0, 0, 0x3c00]; 64];
                queue.write_texture(current.as_image_copy(), bytemuck::cast_slice(&values),
                    wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(64), rows_per_image: Some(8) }, size);
            } else if fixture != 0 {
                frame_uniform.reset = u32::from(fixture != 3);
                // HDR constant (4, 0.5, 0.25) or a midtone checker. Neither
                // may be clipped/sharpened in the image retained as history.
                let values: Vec<u16> = (0..64).flat_map(|i| {
                    if fixture != 2 { [0x4400, 0x3800, 0x3400, 0x3c00] }
                    else {
                        let c = if (i % 8 + i / 8) % 2 == 0 { 0x3400 } else { 0x3800 };
                        [c, c, c, 0x3c00]
                    }
                }).collect();
                queue.write_texture(current.as_image_copy(), bytemuck::cast_slice(&values),
                    wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(64), rows_per_image: Some(8) }, size);
            }
            if depth_boundary {
                frame_uniform.jitter_offset = [0.25, 0.0];
                frame_uniform.reset = 1;
            }
            queue.write_buffer(&pass.uniform_buf, 0, bytemuck::bytes_of(&frame_uniform));
            let mut depths = [[stored; 2]; 64];
            if mixed_edge {
                depths[3 * 8 + 3] = [projected_depth(1.0); 2];
            }
            queue.write_texture(
                pass.history_depth_texture.as_image_copy(),
                bytemuck::cast_slice(&depths),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(64),
                    rows_per_image: Some(8),
                },
                size,
            );
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mean = if fixture >= 4 { wgpu::Color { r: 0.25, g: 0.0, b: 0.0, a: 32.0 } } else { wgpu::Color::BLACK };
                let variance = if fixture >= 4 { wgpu::Color { r: 0.0625, g: 0.0, b: 0.0, a: 0.0 } } else { wgpu::Color::BLACK };
                let attachments = [(&pass.moments[0].history_view,mean),(&pass.moments[1].history_view,variance)]
                    .map(|(view,value)| Some(wgpu::RenderPassColorAttachment { view, resolve_target:None, depth_slice:None,
                        ops:wgpu::Operations {load:wgpu::LoadOp::Clear(value),store:wgpu::StoreOp::Store} }));
                let _clear=encoder.begin_render_pass(&wgpu::RenderPassDescriptor {color_attachments:&attachments,..Default::default()});
            }
            {
                let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &pass.history_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.5,
                                g: 0.5,
                                b: 0.5,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(projected_depth(10.0)),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });
            }
            if depth_boundary {
                let mut draw = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });
                draw.set_pipeline(&depth_pipeline);
                draw.set_scissor_rect(4, 0, 4, 8);
                draw.draw(0..3, 0..1);
            }
            {
                let attachments = [&pass.output_view, &pass.output_depth_view,
                    &pass.moments[0].output_view, &pass.moments[1].output_view].map(|view| {
                    Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })
                });
                let mut draw = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &attachments,
                    ..Default::default()
                });
                draw.set_pipeline(&pass.pipeline);
                draw.set_bind_group(0, &group, &[]);
                draw.draw(0..3, 0..1);
            }
            {
                let mut draw = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &final_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
                draw.set_pipeline(&pass.blit_pipeline);
                draw.set_bind_group(0, &pass.blit_bind_group, &[]);
                draw.draw(0..3, 0..1);
            }
            for (texture, offset) in [(&final_color, 0), (&pass.output_depth_texture, 2048), (&pass.output_texture, 4096),
                (&pass.moments[0].output,6144),(&pass.moments[1].output,8192)] {
                encoder.copy_texture_to_buffer(
                    texture.as_image_copy(),
                    wgpu::TexelCopyBufferInfo {
                        buffer: &staging,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset,
                            bytes_per_row: Some(256),
                            rows_per_image: Some(8),
                        },
                    },
                    size,
                );
            }
            queue.submit(Some(encoder.finish()));
            let (tx, rx) = std::sync::mpsc::channel();
            staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            rx.recv().unwrap().unwrap();
            {
                let bytes = staging.slice(..).get_mapped_range().unwrap();
                let decode = |offset| {
                    let value = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
                    let exponent = (value >> 10) & 31;
                    if exponent == 0 { (value & 1023) as f32 * 2.0_f32.powi(-24) }
                    else { f32::from_bits(((value as u32 & 0x8000) << 16) | ((exponent as u32 + 112) << 23) | ((value as u32 & 1023) << 13)) }
                };
                let displayed = decode(3 * 256 + 3 * 8);
                let retained = decode(4096 + 3 * 256 + 3 * 8);
                let pixel = displayed * 255.0;
                if fixture >= 4 {
                    let mean=decode(6144+3*256+3*8);
                    let count=decode(6144+3*256+3*8+6);
                    let variance=decode(8192+3*256+3*8);
                    if fixture==4 {
                        // This fixture also translates the camera, so the
                        // existing large-motion blend shortens accumulation.
                        assert!(retained>0.25 && retained<0.5,"valid measured coverage was clipped: {retained}");
                        assert!(mean>0.23 && mean<0.25 && variance>0.05 && variance<0.07);
                        assert_eq!(count,32.0);
                    } else {
                        assert_eq!((retained,mean,variance,count),(0.0,0.0,0.0,1.0),"disocclusion/reset retained noisy history");
                    }
                } else if fixture != 0 {
                    assert_eq!(retained, if fixture != 2 { 4.0 } else { 0.25 },
                        "history must preserve the unsharpened HDR current sample on reset");
                    if fixture != 2 { assert_eq!(displayed, 4.0, "display sharpening must preserve HDR constants"); }
                } else if depth_boundary {
                    assert!(
                        pixel > 0.0 && pixel < 255.0,
                        "boundary color should include both current samples"
                    );
                } else if accept {
                    assert!(pixel > 50.0, "valid translated history was rejected: {pixel}");
                } else {
                    assert!(
                        pixel < 3.0,
                        "revealed pixel retained stale history: {pixel}, old depth {stored}"
                    );
                }
                let offset = 2048 + 3 * 256 + 3 * 8;
                let saved = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                let saved_max =
                    f32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap());
                assert_eq!(
                    saved_max.to_bits(),
                    projected_depth(if depth_boundary { 20.0 } else { 10.0 }).to_bits(),
                    "depth bounds must cover the actual bilinear footprint"
                );
                assert_eq!(
                    saved.to_bits(),
                    projected_depth(10.0).to_bits(),
                    "depth history lost precision"
                );
            }
            staging.unmap();
        }
    });
}
