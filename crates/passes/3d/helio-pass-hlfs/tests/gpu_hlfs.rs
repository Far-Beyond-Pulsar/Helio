//! GPU validation is intentional: a Rust build does not validate WGSL or bindings.
mod support;

#[test]
#[ignore = "requires a GPU adapter; run explicitly with --ignored"]
fn rendered_energy_survives_light_growth_overflow_and_removal() {
    use helio_pass_hlfs::{HlfsConfig, HlfsDebugMode};
    use support::*;
    pollster::block_on(async {
        let mut f = Fixture::new(65, 49).await;
        for count in [0usize, 1, 2, 65, 257, 1024] {
            f.lights(
                (0..count)
                    .map(|_| point([0.0, 0.0, 2.0], [1.0, 0.8, 0.5], 8.0 / count.max(1) as f32))
                    .collect(),
            );
            f.config(HlfsConfig {
                debug_mode: HlfsDebugMode::Reference,
                ..Default::default()
            });
            f.frame();
            let reference = f.read();
            assert!(reference.iter().flatten().all(|v| v.is_finite()));
            if count > 0 {
                assert!(mean(&reference) > 0.05, "nonzero direct light required");
            }
            f.config(HlfsConfig::default());
            for _ in 0..32 {
                f.frame();
            }
            let sampled = f.read();
            assert!(sampled.iter().flatten().all(|v| v.is_finite()));
            let error = (mean(&sampled) - mean(&reference)).abs() / mean(&reference).max(0.001);
            eprintln!(
                "lights={count} reference={} sampled={} relative mean error={error}",
                mean(&reference),
                mean(&sampled)
            );
            assert!(
                error < 0.08,
                "lighting energy drift at {count} lights: {error}"
            );
        }
        f.lights(vec![]);
        f.frame();
        let dark = f.read();
        assert!(
            mean(&dark) < 0.02,
            "removed lights persist in history: {}",
            mean(&dark)
        );
    });
}

#[test]
#[ignore = "requires a GPU adapter; run explicitly with --ignored"]
fn mixed_lighting_converges_at_full_and_half_resolution() {
    use helio_pass_hlfs::{HlfsConfig, HlfsDebugMode};
    use support::*;
    pollster::block_on(async {
        let mut f = Fixture::new(129, 97).await;
        let mut lights: Vec<_> = (0..128)
            .map(|i| {
                let x = (i % 16) as f32 / 4.0 - 2.0;
                let y = (i / 16) as f32 / 2.0 - 2.0;
                point(
                    [x, y, 0.6 + (i % 3) as f32],
                    [
                        (i % 3 == 0) as u8 as f32,
                        (i % 3 == 1) as u8 as f32,
                        (i % 3 == 2) as u8 as f32,
                    ],
                    0.2,
                )
            })
            .collect();
        lights.push(libhelio::GpuLight {
            light_type: 0,
            direction_outer: [0.0, 0.0, -1.0, 0.0],
            color_intensity: [1.0, 0.8, 0.5, 3.0],
            ..Default::default()
        });
        f.lights(lights);
        f.config(HlfsConfig {
            debug_mode: HlfsDebugMode::Reference,
            ..Default::default()
        });
        f.frame();
        let reference = f.read();
        f.config(HlfsConfig {
            debug_mode: HlfsDebugMode::Unfiltered,
            ..Default::default()
        });
        let mut raw_mean = 0.0;
        for _ in 0..64 {
            f.frame();
            raw_mean += mean(&f.read()) / 64.0;
        }
        eprintln!(
            "reference mean={}, unfiltered mean={raw_mean}",
            mean(&reference)
        );
        for scale in [1, 2] {
            f.config(HlfsConfig {
                sample_scale: scale,
                ..Default::default()
            });
            for _ in 0..64 {
                f.frame();
            }
            let sampled = f.read();
            let error = (mean(&sampled) - mean(&reference)).abs() / mean(&reference);
            let rmse = (sampled
                .iter()
                .zip(&reference)
                .flat_map(|(a, b)| a.iter().zip(b).map(|(a, b)| (a - b) * (a - b)))
                .sum::<f32>()
                / (sampled.len() * 3) as f32)
                .sqrt()
                / mean(&reference);
            eprintln!(
                "mixed lights scale={scale}: relative mean error={error}, normalized RMSE={rmse}"
            );
            if let Ok(dir) = std::env::var("HLFS_CAPTURE_DIR") {
                std::fs::create_dir_all(&dir).unwrap();
                let save = |name: &str, pixels: &[[f32; 3]]| {
                    let mut image = image::RgbImage::new(f.width, f.height);
                    for (out, p) in image.pixels_mut().zip(pixels) {
                        *out =
                            image::Rgb(p.map(|v| ((v / (1.0 + v)).powf(1.0 / 2.2) * 255.0) as u8));
                    }
                    image.save(std::path::Path::new(&dir).join(name)).unwrap();
                };
                save("mixed-reference.png", &reference);
                save(&format!("mixed-scale-{scale}.png"), &sampled);
            }
            assert!(
                error < 0.08 && rmse < 0.2,
                "mixed lighting fails to converge"
            );
        }
        // Same population, changed intensities: temporal rejection must work
        // without relying on a light-count change to reset the pass.
        f.lights(
            (0..129)
                .map(|_| point([0.0, 0.0, 2.0], [1.0; 3], 0.0))
                .collect(),
        );
        f.frame();
        assert!(
            mean(&f.read()) < 0.02,
            "extinguished lights survive photometric rejection"
        );
    });
}

#[test]
#[ignore = "requires a GPU adapter; run explicitly with --ignored"]
fn creates_pipelines_on_available_adapter() {
    pollster::block_on(async {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("GPU adapter required for HLFS validation");
        eprintln!("HLFS adapter: {:?}", adapter.get_info());
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .unwrap();
        let _pass = helio_pass_hlfs::HlfsPass::new(
            &device,
            &queue,
            128,
            96,
            wgpu::TextureFormat::Rgba16Float,
        );
    });
}

#[test]
#[ignore = "requires a GPU adapter; run explicitly with --ignored"]
fn shadows_use_atlas_layers_independently_of_light_index() {
    use helio_pass_hlfs::{HlfsConfig, HlfsDebugMode};
    use support::*;
    pollster::block_on(async {
        let mut f = Fixture::new(33, 25).await;
        let mut lights = vec![point([0.0, 0.0, 2.0], [1.0; 3], 0.0); 65];
        lights[64] = point([0.0, 0.0, 2.0], [1.0; 3], 8.0);
        lights[64].shadow_index = 0;
        f.lights(lights);
        f.config(HlfsConfig {
            debug_mode: HlfsDebugMode::Reference,
            ..Default::default()
        });
        f.constant_shadow(1.0);
        f.frame();
        assert!(mean(&f.read()) > 0.1);
        f.constant_shadow(0.0);
        f.frame();
        assert!(mean(&f.read()) < 0.02, "light 64 ignores its shadow map");
    });
}

#[test]
#[ignore = "requires a GPU adapter; run explicitly with --ignored"]
fn offscreen_lights_and_minimal_targets_remain_valid() {
    use helio_pass_hlfs::{HlfsConfig, HlfsDebugMode, HlfsPass};
    use support::*;
    pollster::block_on(async {
        let mut f = Fixture::new(65, 49).await;
        let mut lights = vec![point([1000.0, 1000.0, 2.0], [1.0; 3], 10000.0); 64];
        lights[63] = point([-1.0, -1.0, 0.5], [1.0, 0.5, 0.2], 2.0);
        lights[63].position_range[3] = 1.2;
        f.lights(lights);
        f.config(HlfsConfig {
            debug_mode: HlfsDebugMode::Reference,
            ..Default::default()
        });
        f.frame();
        let reference = f.read();
        f.config(HlfsConfig::default());
        for _ in 0..16 {
            f.frame();
        }
        let sampled = f.read();
        let error = (mean(&sampled) - mean(&reference)).abs() / mean(&reference);
        assert!(error < 0.03, "spatial light culling loses energy: {error}");
        let mut tiny = Fixture::new(1, 1).await;
        tiny.lights(vec![point([0.0, 0.0, 2.0], [1.0; 3], 8.0)]);
        for scale in [2, 1, 2] {
            tiny.config(HlfsConfig {
                sample_scale: scale,
                ..Default::default()
            });
            tiny.graph
                .find_pass_mut::<HlfsPass>()
                .unwrap()
                .resize(&tiny.device, 0, 0);
            tiny.frame();
            assert!(mean(&tiny.read()).is_finite() && mean(&tiny.read()) > 0.1);
        }
    });
}

#[test]
#[ignore = "explicit GPU benchmark; run with --ignored --nocapture"]
fn benchmark_gpu_light_scaling() {
    use helio_pass_hlfs::{HlfsConfig, HlfsDebugMode, HlfsPass};
    use support::*;
    pollster::block_on(async {
        let mut f = Fixture::new(640, 360).await;
        f.constant_shadow(1.0);
        for shadowed in [false, true] {
            for count in [64, 256, 1024] {
                f.lights(
                    (0..count)
                        .map(|i| {
                            let mut light = point(
                                [
                                    (i % 16) as f32 / 4.0 - 2.0,
                                    (i / 16 % 16) as f32 / 4.0 - 2.0,
                                    2.0,
                                ],
                                [1.0, 0.8, 0.5],
                                8.0 / count as f32,
                            );
                            if shadowed {
                                light.shadow_index = 0;
                            }
                            light
                        })
                        .collect(),
                );
                for mode in [HlfsDebugMode::Reference, HlfsDebugMode::Final] {
                    f.config(HlfsConfig {
                        debug_mode: mode,
                        ..Default::default()
                    });
                    for _ in 0..16 {
                        f.frame();
                    }
                    let mut times = Vec::new();
                    for _ in 0..40 {
                        f.frame();
                        times.push(f.milliseconds());
                    }
                    times.sort_by(f64::total_cmp);
                    eprintln!("GPU lights={count} shadowed={shadowed} mode={mode:?} median_ms={:.3} p95_ms={:.3} allocation_bytes={}",times[20],times[38],f.graph.find_pass::<HlfsPass>().unwrap().allocation_bytes());
                }
            }
        }
    });
}

#[test]
#[ignore = "requires a GPU adapter; run explicitly with --ignored"]
fn screen_contacts_follow_current_depth_and_clear_after_motion() {
    use helio_pass_hlfs::{HlfsConfig, HlfsDebugMode};
    use support::*;
    pollster::block_on(async {
        let mut f = Fixture::new(129, 65).await;
        let mut light = point([1.0, 0.0, 0.1], [1.0; 3], 1.0);
        light.shadow_index = 0;
        f.lights(vec![light]);
        f.constant_shadow(1.0);
        let depths: Vec<_> = (0..65)
            .flat_map(|_| {
                (0..129).map(|x| {
                    let wx = (x as f32 + 0.5) / 129.0 * 4.0 - 2.0;
                    let z = if wx >= 0.0 && wx < 0.2 { 0.05 } else { 0.0 };
                    (3.0 - z - 0.1) / 9.9
                })
            })
            .collect();
        f.depth_values(&depths);
        f.config(HlfsConfig {
            screen_trace_distance: 0.0,
            debug_mode: HlfsDebugMode::Reference,
            ..Default::default()
        });
        f.frame();
        let unshadowed = f.read();
        f.config(HlfsConfig {
            screen_trace_distance: 0.5,
            debug_mode: HlfsDebugMode::Reference,
            ..Default::default()
        });
        f.frame();
        let contact = f.read();
        let region = |pixels: &[[f32; 3]]| {
            (25..40)
                .flat_map(|y| (52..64).map(move |x| pixels[y * 129 + x][0]))
                .sum::<f32>()
        };
        eprintln!(
            "Contact region unshadowed={} occluded={}",
            region(&unshadowed),
            region(&contact)
        );
        assert!(
            region(&contact) < region(&unshadowed) * 0.8,
            "current-depth contact shadow missing"
        );
        f.depth_values(&vec![(3.0 - 0.1) / 9.9; 129 * 65]);
        f.frame();
        assert!(
            region(&f.read()) > region(&unshadowed) * 0.95,
            "moving contact occluder leaves stale shadow"
        );
    });
}

#[test]
#[ignore = "requires a GPU adapter; run explicitly with --ignored"]
fn half_resolution_keeps_isolated_one_pixel_geometry_lit() {
    use helio_pass_hlfs::{HlfsConfig, HlfsDebugMode};
    use support::*;
    pollster::block_on(async {
        let mut f = Fixture::new(65, 49).await;
        f.lights(vec![point([0.0, 0.0, 2.0], [1.0; 3], 8.0)]);
        let index = 20 * 65 + 21;
        let mut depths = vec![1.0; 65 * 49];
        depths[index] = (3.0 - 0.1) / 9.9;
        f.depth_values(&depths);
        f.config(HlfsConfig {
            debug_mode: HlfsDebugMode::Reference,
            ..Default::default()
        });
        f.frame();
        let reference = f.read()[index];
        assert!(reference[0] > 0.1);
        f.config(HlfsConfig {
            sample_scale: 2,
            ..Default::default()
        });
        for phase in 0..4 {
            f.scene.frame_count = phase;
            f.frame();
            let pixels = f.read();
            assert!(
                (pixels[index][0] - reference[0]).abs() < 0.005,
                "thin surface loses illumination in phase {phase}: {:?}",
                pixels[index]
            );
            assert!(
                pixels
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != index)
                    .all(|(_, p)| p[0] == 0.0),
                "lighting leaks into sky"
            );
        }
    });
}
