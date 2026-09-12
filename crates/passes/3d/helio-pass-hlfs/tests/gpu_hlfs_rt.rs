//! The first executable RT control: opaque world visibility, not a performance claim.
#![allow(dead_code)]
mod support;
use helio_core::{BlasGeometry, TlasInstanceInput};
use helio_pass_hlfs::{HlfsConfig, HlfsDebugMode, HlfsMode, HlfsPass};
use support::{mean, point, Fixture};
use wgpu::util::DeviceExt;

fn blocker(f: &mut Fixture, x: f32) {
    // Entirely offscreen to the right of the fixture's orthographic camera.
    let vertices = f
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("offscreen RT caster"),
            contents: bytemuck::cast_slice(&[
                x, -100.0f32, -100.0, x, 100.0, -100.0, x, 0.0, 100.0,
            ]),
            usage: wgpu::BufferUsages::BLAS_INPUT,
        });
    let mut encoder = f.device.create_command_encoder(&Default::default());
    f.scene
        .blas_manager
        .build_from_buffers(
            1,
            &mut encoder,
            BlasGeometry {
                revision: 0,
                vertices: &vertices,
                first_vertex: 0,
                vertex_count: 3,
                vertex_stride: 12,
                indices: None,
                first_index: 0,
                index_count: 0,
            },
        )
        .unwrap();
    f.scene
        .tlas_manager
        .build(
            &mut encoder,
            &[TlasInstanceInput {
                mesh_id: 1,
                transform: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            }],
            &f.scene.blas_manager,
        )
        .unwrap();
    f.queue.submit([encoder.finish()]);
}

fn empty_scene(f: &mut Fixture) {
    let mut encoder = f.device.create_command_encoder(&Default::default());
    f.scene
        .tlas_manager
        .build(&mut encoder, &[], &f.scene.blas_manager)
        .unwrap();
    f.queue.submit([encoder.finish()]);
}

fn light() -> libhelio::GpuLight {
    let mut light = point([10.0, 0.0, 2.0], [1.0, 1.0, 1.0], 10000.0);
    light.position_range[3] = 100.0;
    light.shadow_index = u32::MAX;
    light.set_ray_traced_shadows(true);
    light
}

#[test]
#[ignore = "requires Vulkan hardware ray queries"]
fn offscreen_shadow_does_not_require_an_atlas_slot_and_removal_clears_it() {
    pollster::block_on(async {
        let mut f = Fixture::new_rt(65, 49).await;
        f.config(HlfsConfig {
            mode: HlfsMode::RayTraced,
            debug_mode: HlfsDebugMode::Reference,
            ..Default::default()
        });
        f.lights(vec![light()]);
        empty_scene(&mut f);
        f.frame();
        let lit = mean(&f.read());
        blocker(&mut f, 5.0);
        f.frame();
        let shadow = mean(&f.read());
        assert!(
            lit > 1.0 && shadow < lit * 0.05,
            "offscreen shadow without atlas: {lit} -> {shadow}"
        );
        // Moving the blocker beyond the light must not occlude the finite segment.
        blocker(&mut f, 15.0);
        f.frame();
        let beyond = mean(&f.read());
        assert!(
            (beyond / lit - 1.0).abs() < 0.01,
            "blocker beyond light: {beyond} vs {lit}"
        );
        empty_scene(&mut f);
        f.frame();
        assert!((mean(&f.read()) / lit - 1.0).abs() < 0.01);
        blocker(&mut f, 5.0);
        let mut unshadowed = light();
        unshadowed.set_ray_traced_shadows(false);
        f.lights(vec![unshadowed]);
        f.frame();
        assert!(
            (mean(&f.read()) / lit - 1.0).abs() < 0.01,
            "explicit unshadowed intent"
        );
    });
}

#[test]
#[ignore = "requires Vulkan hardware ray queries"]
fn reduced_resolution_repair_uses_hardware_visibility_in_every_phase() {
    pollster::block_on(async {
        let mut f = Fixture::new_rt(65, 49).await;
        f.config(HlfsConfig {
            mode: HlfsMode::RayTraced,
            ..HlfsConfig::compact()
        });
        f.lights(vec![light()]);
        let center = 24 * 65 + 32;
        let mut depths = vec![1.0; 65 * 49];
        depths[center] = (3.0 - 0.1) / (10.0 - 0.1);
        f.depth_values(&depths);
        empty_scene(&mut f);
        f.frame();
        let lit = mean(&[f.read()[center]]);
        blocker(&mut f, 5.0);
        for phase in 0..4 {
            f.graph
                .find_pass_mut::<HlfsPass>()
                .unwrap()
                .invalidate_history();
            f.frame();
            let shadow = mean(&[f.read()[center]]);
            assert!(
                lit > 1.0 && shadow < lit * 0.05,
                "phase {phase}: lit {lit}, shadow {shadow}"
            );
        }
    });
}

#[test]
#[ignore = "requires a GPU; ray-query support is intentionally not requested"]
fn unsupported_mode_change_is_rejected_without_changing_the_pass() {
    pollster::block_on(async {
        let mut f = Fixture::new(8, 8).await;
        let pass = f.graph.find_pass_mut::<HlfsPass>().unwrap();
        let output = pass.output_texture().clone();
        assert!(pass
            .try_set_config(
                &f.device,
                HlfsConfig {
                    mode: HlfsMode::RayTraced,
                    ..Default::default()
                }
            )
            .is_err());
        assert_eq!(pass.config().mode, HlfsMode::ScreenSpace);
        assert_eq!(pass.output_texture(), &output);
        f.frame();
    });
}

#[test]
#[ignore = "requires Vulkan hardware ray queries"]
fn missing_tlas_errors_and_mode_switch_preserves_output() {
    pollster::block_on(async {
        let mut f = Fixture::new_rt(8, 8).await;
        let output = f
            .graph
            .find_pass::<HlfsPass>()
            .unwrap()
            .output_texture()
            .clone();
        f.config(HlfsConfig {
            mode: HlfsMode::RayTraced,
            ..Default::default()
        });
        assert!(
            f.try_frame().is_err(),
            "capable device with missing TLAS must not render unoccluded"
        );
        empty_scene(&mut f);
        f.frame();
        f.config(HlfsConfig::default());
        f.frame();
        assert_eq!(
            f.graph.find_pass::<HlfsPass>().unwrap().output_texture(),
            &output
        );
    });
}

fn receiver_plane(f: &mut Fixture, z: f32) {
    let vertices = f
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("perspective receiver"),
            contents: bytemuck::cast_slice(&[
                -500.0f32, -500.0, z, 500.0, -500.0, z, 0.0, 500.0, z,
            ]),
            usage: wgpu::BufferUsages::BLAS_INPUT,
        });
    let mut encoder = f.device.create_command_encoder(&Default::default());
    f.scene
        .blas_manager
        .build_from_buffers(
            1,
            &mut encoder,
            BlasGeometry {
                revision: 0,
                vertices: &vertices,
                first_vertex: 0,
                vertex_count: 3,
                vertex_stride: 12,
                indices: None,
                first_index: 0,
                index_count: 0,
            },
        )
        .unwrap();
    f.scene
        .tlas_manager
        .build(
            &mut encoder,
            &[TlasInstanceInput {
                mesh_id: 1,
                transform: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            }],
            &f.scene.blas_manager,
        )
        .unwrap();
    f.queue.submit([encoder.finish()]);
}

#[test]
#[ignore = "requires Vulkan hardware ray queries"]
fn perspective_depth_error_does_not_shadow_the_receiver_or_erase_nearby_blockers() {
    pollster::block_on(async {
        let mut f = Fixture::new_rt(32, 32).await;
        f.config(HlfsConfig {
            mode: HlfsMode::RayTraced,
            debug_mode: HlfsDebugMode::Reference,
            ..Default::default()
        });
        let proj = glam::Mat4::perspective_rh(0.8, 1.0, 0.1, 200.0);
        for camera in [glam::Vec3::ZERO, glam::Vec3::new(2.0, 2.0, 24.0)] {
            let view = if camera == glam::Vec3::ZERO {
                glam::Mat4::IDENTITY
            } else {
                glam::Mat4::look_at_rh(camera, glam::Vec3::new(0.0, 5.0, -20.0), glam::Vec3::Y)
            };
            let vp = proj * view;
            f.scene.camera.update(libhelio::GpuCameraUniforms::new(
                view, proj, camera, 0.1, 200.0, 0, [0.0; 2], vp,
            ));
            let mut light = point(camera.to_array(), [1.0; 3], 100000.0);
            light.position_range[3] = 200.0;
            light.set_ray_traced_shadows(true);
            f.lights(vec![light]);
            // Independently intersect camera rays with the known plane in f64,
            // then quantize the projected depth to the actual Depth32 format.
            let inverse = vp.as_dmat4().inverse();
            for distance in [10.0f32, 30.0, 60.0, 83.7, 100.0, 150.0] {
                let z = camera.z - distance;
                let mut depths = Vec::new();
                for y in 0..32 {
                    for x in 0..32 {
                        let h = inverse
                            * glam::DVec4::new(
                                (x as f64 + 0.5) / 16.0 - 1.0,
                                1.0 - (y as f64 + 0.5) / 16.0,
                                0.5,
                                1.0,
                            );
                        let direction = h.truncate() / h.w - camera.as_dvec3();
                        let world = camera.as_dvec3()
                            + direction * ((z as f64 - camera.z as f64) / direction.z);
                        let clip = vp.as_dmat4() * world.extend(1.0);
                        depths.push((clip.z / clip.w) as f32);
                    }
                }
                f.depth_values(&depths);
                empty_scene(&mut f);
                f.frame();
                let lit = mean(&f.read());
                receiver_plane(&mut f, z);
                f.frame();
                let receiver = mean(&f.read());
                eprintln!("camera={camera:?} distance={distance} lit={lit} receiver={receiver}");
                assert!(
                    (receiver - lit).abs() < lit * 0.01,
                    "self shadow at distance {distance}, camera {camera:?}"
                );
                receiver_plane(&mut f, z + 0.02);
                f.frame();
                let blocked = mean(&f.read());
                assert!(
                    blocked < lit * 0.05,
                    "lost 2cm blocker at distance {distance}, camera {camera:?}: {blocked}/{lit}"
                );
            }
        }
    });
}

// Preliminary synthetic work accounting, not the frozen game-scene acceptance
// protocol: 1,024 moving lights, 256 moving triangle instances, cached BLAS.
#[test]
#[ignore = "explicit RT GPU benchmark; run alone with --ignored --nocapture"]
fn benchmark_rt_resolution_and_acceleration() {
    pollster::block_on(async {
        for (width, height, scale, candidates) in [
            (2560, 1440, 1, 8),
            (2560, 1440, 2, 8),
            (2560, 1440, 2, 4),
            (2560, 1440, 2, 2),
            (3840, 2160, 2, 8),
            (3840, 2160, 2, 2),
        ] {
            let mut f = Fixture::new_rt(width, height).await;
            f.compact_output();
            f.config(HlfsConfig {
                mode: HlfsMode::RayTraced,
                sample_scale: scale,
                candidates_per_sample: candidates,
                samples_per_pixel: 2,
                ..Default::default()
            });
            assert!(f
                .graph
                .find_pass_mut::<HlfsPass>()
                .unwrap()
                .enable_timing(&f.device));
            let vertices = f
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("benchmark caster"),
                    contents: bytemuck::cast_slice(&[
                        -0.1f32, -0.1, 0.5, 0.1, -0.1, 0.5, 0.0, 0.1, 0.5,
                    ]),
                    usage: wgpu::BufferUsages::BLAS_INPUT,
                });
            let mut encoder = f.device.create_command_encoder(&Default::default());
            f.scene
                .blas_manager
                .build_from_buffers(
                    1,
                    &mut encoder,
                    BlasGeometry {
                        revision: 0,
                        vertices: &vertices,
                        first_vertex: 0,
                        vertex_count: 3,
                        vertex_stride: 12,
                        indices: None,
                        first_index: 0,
                        index_count: 0,
                    },
                )
                .unwrap();
            f.queue.submit([encoder.finish()]);
            let query = f.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("TLAS benchmark"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            });
            let resolve = f.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 16,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let read = f.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 16,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut rows = Vec::new();
            for frame in 0..56 {
                let cpu = std::time::Instant::now();
                let shift = (frame as f32 * 0.1).sin() * 0.05;
                f.lights(
                    (0..1024)
                        .map(|i| {
                            let mut light = point(
                                [
                                    (i % 32) as f32 / 8.0 - 2.0 + shift,
                                    (i / 32) as f32 / 8.0 - 2.0,
                                    2.0,
                                ],
                                [1.0, 0.8, 0.5],
                                16.0 / 1024.0,
                            );
                            light.set_ray_traced_shadows(true);
                            light
                        })
                        .collect(),
                );
                let instances: Vec<_> = (0..256)
                    .map(|i| TlasInstanceInput {
                        mesh_id: 1,
                        transform: [
                            1.0,
                            0.0,
                            0.0,
                            (i % 16) as f32 / 4.0 - 2.0 + shift,
                            0.0,
                            1.0,
                            0.0,
                            (i / 16) as f32 / 4.0 - 2.0,
                            0.0,
                            0.0,
                            1.0,
                            0.0,
                        ],
                    })
                    .collect();
                let mut encoder = f.device.create_command_encoder(&Default::default());
                encoder.write_timestamp(&query, 0);
                f.scene
                    .tlas_manager
                    .build(&mut encoder, &instances, &f.scene.blas_manager)
                    .unwrap();
                encoder.write_timestamp(&query, 1);
                encoder.resolve_query_set(&query, 0..2, &resolve, 0);
                encoder.copy_buffer_to_buffer(&resolve, 0, &read, 0, 16);
                f.queue.submit([encoder.finish()]);
                f.frame();
                let cpu_ms = cpu.elapsed().as_secs_f64() * 1000.0;
                let stages = f.stage_milliseconds();
                // compact_output replaces the outer graph timestamp markers.
                // Its six internal stage intervals remain valid and cover HLFS.
                let graph_ms: f64 = stages.iter().sum();
                assert!(
                    graph_ms.is_finite() && graph_ms > 0.0,
                    "missing HLFS stage timestamps"
                );
                let (tx, rx) = std::sync::mpsc::channel();
                read.slice(..)
                    .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
                f.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                rx.recv().unwrap().unwrap();
                let bytes = read.slice(..).get_mapped_range().unwrap();
                let start = u64::from_le_bytes(bytes[..8].try_into().unwrap());
                let end = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
                let tlas_ms = (end - start) as f64 * f.queue.get_timestamp_period() as f64 / 1e6;
                drop(bytes);
                read.unmap();
                if frame >= 16 {
                    rows.push((graph_ms + tlas_ms, tlas_ms, graph_ms, cpu_ms, stages));
                }
            }
            let median = |mut values: Vec<f64>| {
                values.sort_by(f64::total_cmp);
                values[values.len() / 2]
            };
            let mut totals: Vec<_> = rows.iter().map(|r| r.0).collect();
            totals.sort_by(f64::total_cmp);
            let stages: [f64; 6] =
                std::array::from_fn(|i| median(rows.iter().map(|r| r.4[i]).collect()));
            eprintln!("RT_PROBE resolution={width}x{height} scale={scale} spp=2 candidates={candidates} lights=1024 moving_instances=256 median_gpu_ms={:.4} p95_gpu_ms={:.4} tlas_median_ms={:.4} hlfs_median_ms={:.4} cpu_submit_median_ms={:.4} stages={stages:?}",totals[20],totals[38],median(rows.iter().map(|r|r.1).collect()),median(rows.iter().map(|r|r.2).collect()),median(rows.iter().map(|r|r.3).collect()));
        }
    });
}
