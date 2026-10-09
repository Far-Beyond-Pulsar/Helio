use super::*;
use wgpu::util::DeviceExt;

#[test]
fn resolve_rejects_wrong_surface_and_writes_current_linear_depth() {
    run_depth_history_fixture(false);
}

#[test]
fn translated_origin_prepare_preserves_history_and_rejects_invalid_views() {
    run_depth_history_fixture(true);
}

fn run_depth_history_fixture(translated_origin: bool) {
    pollster::block_on(async {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let Ok(adapter) = instance.request_adapter(&Default::default()).await else {
            eprintln!("SKIP: no GPU adapter for TSR depth-history fixture");
            return;
        };
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        let mut pass = TsrPass::new(
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
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let create_bind = |pass: &TsrPass| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            })
        };
        let mut bind = create_bind(&pass);
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
        let create_probe_bind = |pass: &TsrPass| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            })
        };
        let mut probe_bind = create_probe_bind(&pass);
        let mut results = Vec::new();
        let cases = if translated_origin {
            vec![
                (10.0f32, 0u32, 0.0, 1.0 / 60.0, 0u8, 0u16), // valid translated history
                (2.0, 0, 0.0, 1.0 / 60.0, 0, 0),             // wrong surface
                (10.0, 1, 0.0, 1.0 / 60.0, 0, 0),            // explicit camera cut
                (10.0, 0, 1.0, 1.0 / 60.0, 0, 0),            // full reactivity
                (10.0, 2, 0.0, 1.0 / 60.0, 255, 0),          // arriving glass
                (10.0, 2, 0.0, 1.0 / 60.0, 0, 0x3c00),       // departing glass
                (10.0, 0, 0.0, 1.0 / 60.0, 0, 0),            // reverse origin translation
                (2.0, 0, 0.0, 1.0 / 60.0, 0, 0),
                (10.0, 1, 0.0, 1.0 / 60.0, 0, 0),
                // Changed active views must reset even with matching geometry.
                // Mode-switch depths match the stale, unrebased view, so a
                // failure to reset cannot hide behind depth rejection.
                (10.0, 0, 0.0, 1.0 / 60.0, 0, 0), // changed active view ID
                (12.5, 0, 0.0, 1.0 / 60.0, 0, 0), // Some origin -> None
                (12.5, 0, 0.0, 1.0 / 60.0, 0, 0), // None -> Some origin
                (10.0, 0, 0.0, 1.0 / 60.0, 0, 0), // rotated positive-pole origin, nonzero local eye
                (2.0, 0, 0.0, 1.0 / 60.0, 0, 0),
                (10.0, 1, 0.0, 1.0 / 60.0, 0, 0),
                (12.5, 0, 0.0, 1.0 / 60.0, 0, 0), // third upload after two successive shifts
                (2.0, 0, 0.0, 1.0 / 60.0, 0, 0),
                (12.5, 1, 0.0, 1.0 / 60.0, 0, 0),
                (10.0, 0, 0.0, 1.0 / 60.0, 0, 0), // rotated negative-pole origin
                (2.0, 0, 0.0, 1.0 / 60.0, 0, 0),
                (10.0, 1, 0.0, 1.0 / 60.0, 0, 0),
                (12.5, 0, 0.0, 1.0 / 60.0, 0, 0), // resize invalidates matched history
            ]
        } else {
            vec![
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
            ]
        };
        for (case, (stored_depth, flags, reactivity, time_delta, coverage, previous_coverage)) in
            cases.into_iter().enumerate()
        {
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
            let mut raster_depth = 0.5;
            if translated_origin {
                // Physical orthographic scene: camera starts at world origin,
                // looking down -Z at a plane 10m away. The next camera follows
                // an origin shift by 2.5m along Z, so CURRENT depth is12.5m,
                // while the same surface's PREVIOUS depth remains10m.
                // XY translation produces2 display pixels of real reprojection
                // (UV.03125, above the large-motion threshold) without changing
                // the uniform history colour used to observe acceptance.
                let reverse = (6..=8).contains(&case);
                let delta: [f64; 3] = if reverse {
                    [-0.0625, 0.0, -2.5]
                } else {
                    [0.0625, 0.0, 2.5]
                };
                let sign = if (18..=20).contains(&case) { -1.0 } else { 1.0 };
                let origin: [f64; 3] = [0.25, sign * 6_371_000.125, 0.375];
                let world_delta = if case >= 12 {
                    [delta[0], sign * delta[2], -sign * delta[1]]
                } else {
                    delta
                };
                let current_origin: [f64; 3] =
                    std::array::from_fn(|axis| origin[axis] + world_delta[axis]);
                let mut previous = camera;
                previous.view = identity;
                previous.proj = identity;
                previous.proj[10] = -0.01;
                previous.view_proj = previous.proj;
                previous.inv_view_proj = identity;
                previous.inv_view_proj[10] = -100.0;
                previous.prev_view_proj = previous.proj;
                if case >= 12 {
                    // Looking towards the centre from either pole: worldY
                    // maps to signed viewZ. A nonzero local eye tests translation
                    // in addition to the orientation of the origin correction.
                    let q = sign as f32;
                    let eye = [0.375, -0.5, 0.625];
                    previous.view = [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        q,
                        0.0,
                        0.0,
                        -q,
                        0.0,
                        0.0,
                        -eye[0],
                        q * eye[2],
                        -q * eye[1],
                        1.0,
                    ];
                    previous.view_proj = [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        -0.01 * q,
                        0.0,
                        0.0,
                        -q,
                        0.0,
                        0.0,
                        -eye[0],
                        q * eye[2],
                        0.01 * q * eye[1],
                        1.0,
                    ];
                    previous.inv_view_proj = [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        -q,
                        0.0,
                        0.0,
                        -100.0 * q,
                        0.0,
                        0.0,
                        eye[0],
                        eye[1],
                        eye[2],
                        1.0,
                    ];
                    previous.prev_view_proj = previous.view_proj;
                    previous.position_near[..3].copy_from_slice(&eye);
                }
                previous.jitter_frame = [0.0; 4];
                pass.set_transparency_reactivity(flags & 2 != 0);
                pass.set_reactivity(reactivity);
                let scene_buffers = helio_core::SceneBufferProjection::empty();
                let registry = helio_core::ResourceRegistry::empty();
                let previous_origin = if case == 11 {
                    None
                } else {
                    Some(origin.into())
                };
                pass.prepare(&PrepareContext {
                    device: &device,
                    queue: &queue,
                    camera: &cameras,
                    camera_data: &previous,
                    camera_generation: 0,
                    scene_buffers: &scene_buffers,
                    registry: &registry,
                    resize: false,
                    width: 64,
                    height: 8,
                    frame_num: 0,
                    delta_time: time_delta,
                    world_origin: previous_origin,
                })
                .unwrap();
                let mut current = previous;
                current.prev_view_proj[12] += delta[0] as f32;
                current.prev_view_proj[13] += delta[1] as f32;
                current.prev_view_proj[14] += -0.01 * delta[2] as f32;
                if case == 9 {
                    current.jitter_frame[3] = f32::from_bits(1);
                }
                if flags & 1 != 0 {
                    pass.reset_history();
                }
                let next_origin = if case == 10 {
                    None
                } else {
                    Some(current_origin.into())
                };
                pass.prepare(&PrepareContext {
                    device: &device,
                    queue: &queue,
                    camera: &cameras,
                    camera_data: &current,
                    camera_generation: 1,
                    scene_buffers: &scene_buffers,
                    registry: &registry,
                    resize: false,
                    width: 64,
                    height: 8,
                    frame_num: 1,
                    delta_time: time_delta,
                    world_origin: next_origin,
                })
                .unwrap();
                let successive = (15..=17).contains(&case);
                if successive {
                    // History now belongs to the middle camera, independently
                    // supplied at12.5m. Reapplying both earlier origin shifts
                    // would incorrectly expect10m and reject this third frame.
                    let final_origin: [f64; 3] =
                        std::array::from_fn(|axis| origin[axis] + 2.0 * world_delta[axis]);
                    if flags & 1 != 0 {
                        pass.reset_history();
                    }
                    pass.prepare(&PrepareContext {
                        device: &device,
                        queue: &queue,
                        camera: &cameras,
                        camera_data: &current,
                        camera_generation: 2,
                        scene_buffers: &scene_buffers,
                        registry: &registry,
                        resize: false,
                        width: 64,
                        height: 8,
                        frame_num: 2,
                        delta_time: time_delta,
                        world_origin: Some(final_origin.into()),
                    })
                    .unwrap();
                }
                if case == 21 {
                    pass.on_resize(&device, 64, 8);
                    bind = create_bind(&pass);
                    probe_bind = create_probe_bind(&pass);
                    // Seed the replacement texture with otherwise valid history.
                    // This distinguishes reset from incidental cleared-depth rejection.
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
                    pass.prepare(&PrepareContext {
                        device: &device,
                        queue: &queue,
                        camera: &cameras,
                        camera_data: &current,
                        camera_generation: 2,
                        scene_buffers: &scene_buffers,
                        registry: &registry,
                        resize: true,
                        width: 64,
                        height: 8,
                        frame_num: 2,
                        delta_time: time_delta,
                        world_origin: next_origin,
                    })
                    .unwrap();
                }
                queue.write_buffer(&cameras, 0, bytemuck::cast_slice(&[current, current]));
                raster_depth =
                    (10.0 + delta[2] as f32 * if successive { 2.0 } else { 1.0 }) / 100.0;
            }
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(raster_depth),
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
        if translated_origin {
            for index in [1, 3, 4, 5, 9, 10, 11, 13, 14, 19, 20, 21] {
                assert_eq!(
                    &results[index][..3],
                    &results[2][..3],
                    "wrong depth/cut/coverage/view-mode change must reject history: case{index}"
                );
            }
            assert_eq!(
                &results[7][..3],
                &results[8][..3],
                "reverse-motion wrong depth must reject history"
            );
            for (accepted, reset) in [(0, 2), (6, 8), (12, 14), (15, 17), (18, 20)] {
                assert!((results[accepted][0]-results[reset][0]).abs()>0.0001,
                    "actual prepare discarded a correct translated-origin history sample: case{accepted}, {results:?}");
            }
            assert_eq!(
                &results[16][..3],
                &results[17][..3],
                "successive-shift wrong history depth must reject"
            );
            for (index, result) in results.iter().enumerate() {
                let current_depth = if (6..=8).contains(&index) {
                    7.5
                } else if (15..=17).contains(&index) {
                    15.0
                } else {
                    12.5
                };
                assert!((result[4]-current_depth).abs()<0.00001,
                    "origin translation changed actual current view depth at case{index}:{result:?}");
            }
            eprintln!("TSR translated-origin actual prepare: both motion directions and rotated pole origins retain correct10m history; successive shifts retain12.5m history; wrong depth, cuts, coverage, view/mode switches and resize reject");
        } else {
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
        }
    });
}

#[test]
fn previous_view_rebase_preserves_physical_coordinates_and_f64_cancellation() {
    // A90-degree view rotation maps (x,y,z) to(-z,y,x), plus translation.
    let view = [
        0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, -3.0, 1.0,
    ];
    let rebased = rebase_previous_view(view, [2.5, -1.25, 7.0]);
    assert_eq!(&rebased[..12], &view[..12]);
    assert_eq!(&rebased[12..], &[-6.0, 0.75, -0.5, 1.0]);
    let mut translated = [0.0; 16];
    for i in [0, 5, 10, 15] {
        translated[i] = 1.0;
    }
    translated[14] = -16_777_216.0;
    assert_eq!(
        rebase_previous_view(translated, [0.0, 0.0, 16_777_216.25])[14],
        0.25,
        "casting the origin shift to f32 before composition loses the physical residual"
    );
}
