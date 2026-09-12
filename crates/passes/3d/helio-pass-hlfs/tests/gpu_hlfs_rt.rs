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
