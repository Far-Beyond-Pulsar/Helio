//! The denoiser against fields whose truth is known: the tests are the
//! producer, writing an irradiance signal (`helio_frame_api::denoise`) traced
//! over an analytic G-buffer (the history crate's fixture), and compare the
//! denoised irradiance at each pixel's own normal with the field there.
//! Skipped without a GPU. Readiness is a real signal (the history's length
//! read back) under a wall-clock guard, never a count of frames.

#[path = "../../helio-pass-history/tests/support/mod.rs"]
mod support;

use glam::{DVec2, DVec3};
use helio_frame_api::denoise::{self, Signal, TraceBlock};
use helio_frame_api::history;
use helio_pass_denoise::{DenoisePass, DenoiseSettings};
use support::{Fixture, Quad, Scene, View};

const SIZE: [u32; 2] = [160, 112];
const MAX_HISTORY: f32 = 32.0;

/// A field of irradiance: luminance `a(x) + b . n`, a fixed colour.
#[derive(Clone, Copy)]
struct Field {
    a: f64,
    gradient: DVec3,
    b: DVec3,
}

/// The colour of every field (its luminance 1).
const COLOUR: [f64; 3] = [1.1, 1.0, 0.8];

fn luminance(c: [f64; 3]) -> f64 {
    0.25 * c[0] + 0.5 * c[1] + 0.25 * c[2]
}

impl Field {
    fn flat(a: f64) -> Self {
        Self {
            a,
            gradient: DVec3::ZERO,
            b: DVec3::ZERO,
        }
    }

    fn at(&self, x: DVec3, n: DVec3) -> f64 {
        self.a + self.gradient.dot(x) + self.b.dot(n)
    }
}

/// The producer: one sample at each block's traced pixel.
struct Producer {
    signal: Signal,
    textures: [wgpu::Texture; 2],
    views: [wgpu::TextureView; 2],
    /// Relative standard deviation of each sample (uniform noise).
    noise: f64,
    hit_distance: f32,
}

/// A uniform in [0, 1) from `a` and `b` (PCG).
fn random(a: u32, b: u32) -> f64 {
    let mut s = a
        .wrapping_mul(747_796_405)
        .wrapping_add(b.wrapping_mul(2_891_336_453))
        ^ 0x9e37_79b9;
    s = s.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((s >> ((s >> 28) + 4)) ^ s).wrapping_mul(277_803_737);
    f64::from((word >> 22) ^ word) / 4_294_967_296.0
}

impl Producer {
    fn new(gpu: &support::Gpu, signal: Signal, noise: f64) -> Self {
        let size = signal.traced_size(SIZE);
        let textures = [0, 1].map(|i| {
            gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(signal.inputs[i]),
                size: wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: signal.format(i).to_wgpu(),
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        });
        let views = textures
            .each_ref()
            .map(|t| t.create_view(&Default::default()));
        Self {
            signal,
            textures,
            views,
            noise,
            hit_distance: 3.0,
        }
    }

    /// This frame's samples of `field` (by surface: `fields[i]` for hits on
    /// quad `i`'s depth band), `confidence` each.
    fn write(&self, f: &Fixture, field: impl Fn(&support::Hit) -> Field, confidence: f32) {
        let side = self.signal.block.side();
        let traced = self.signal.traced_size(SIZE);
        let mut texels = [Vec::new(), Vec::new()];
        let colour = COLOUR.map(|c| c / luminance(COLOUR));
        for by in 0..traced[1] {
            for bx in 0..traced[0] {
                let t = denoise::traced_pixel([bx, by], f.frame as u32, side, SIZE);
                let hit = f.texels[(t[1] * SIZE[0] + t[0]) as usize].hit;
                let Some(hit) = hit else {
                    texels[0].push([0.0; 4]);
                    texels[1].push([0.0; 4]);
                    continue;
                };
                let e = field(&hit).at(hit.point, hit.normal);
                let u = random(f.frame as u32, by * traced[0] + bx) * 2.0 - 1.0;
                let scale = 1.0 + self.noise * 3f64.sqrt() * u;
                let y = e * scale;
                let rgb = colour.map(|c| c * y);
                let co = 0.5 * (rgb[0] - rgb[2]);
                let cg = -0.25 * rgb[0] + 0.5 * rgb[1] - 0.25 * rgb[2];
                // The luminance-weighted direction whose first-order term is
                // the field's b (DENOISE_IRRADIANCE_SLOPE = 3/4).
                let d = field(&hit).b / 0.75 * scale;
                texels[0].push([y as f32, co as f32, cg as f32, self.hit_distance]);
                texels[1].push([d.x as f32, d.y as f32, d.z as f32, confidence]);
            }
        }
        for (texture, data) in self.textures.iter().zip(&texels) {
            support::write_half(f.gpu, texture, data);
        }
    }

    fn seeds(&self) -> [(&'static str, &wgpu::TextureView); 2] {
        [
            (self.signal.inputs[0], &self.views[0]),
            (self.signal.inputs[1], &self.views[1]),
        ]
    }
}

fn fixture(gpu: &'static support::Gpu, signal: Signal) -> Fixture {
    Fixture::new(
        gpu,
        SIZE,
        |env| {
            history::offer(env, helio_pass_history::publisher());
            denoise::offer(env, helio_pass_denoise::denoiser());
            env.insert_setting(DenoiseSettings {
                max_history: MAX_HISTORY,
            });
        },
        move |build| {
            let filtered = denoise::filter(build, &signal)?;
            assert_eq!(filtered.denoiser, Some("helio"));
            assert_eq!(filtered.textures, signal.outputs);
            Ok(())
        },
    )
}

/// Draws, produces and runs one frame.
fn frame(
    f: &mut Fixture,
    p: &Producer,
    scene: &Scene,
    view: &View,
    origin: DVec3,
    field: impl Fn(&support::Hit) -> Field,
    confidence: f32,
) {
    f.draw(scene, view, origin);
    p.write(f, field, confidence);
    f.run(&p.seeds());
}

fn pass(f: &Fixture) -> &DenoisePass {
    f.graph
        .find_pass::<DenoisePass>()
        .expect("the denoiser built")
}

/// The history's length at each pixel.
fn lengths(f: &Fixture) -> Vec<f32> {
    let (history, _) = pass(f).textures().expect("a frame ran");
    support::read(f.gpu, &history[1])
        .iter()
        .map(|t| t[3])
        .collect()
}

/// The denoised luminance at each pixel's own normal (`None`: sky).
fn denoised(f: &Fixture) -> Vec<Option<f64>> {
    let (_, out) = pass(f).textures().expect("a frame ran");
    let a = support::read(f.gpu, &out[0]);
    let b = support::read(f.gpu, &out[1]);
    f.texels
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let hit = t.hit?;
            let n = hit.normal;
            let y = f64::from(a[i][0])
                + f64::from(b[i][0]) * n.x
                + f64::from(b[i][1]) * n.y
                + f64::from(b[i][2]) * n.z;
            // The chroma came through as the field's.
            let chroma = [a[i][1], a[i][2]];
            let c = COLOUR.map(|c| c / luminance(COLOUR));
            let expected = [0.5 * (c[0] - c[2]), -0.25 * c[0] + 0.5 * c[1] - 0.25 * c[2]];
            if y > 1e-3 && a[i][3] >= 0.0 {
                for (got, want) in chroma.iter().zip(expected) {
                    assert!((f64::from(*got) - want).abs() < 0.01, "chroma {chroma:?}");
                }
            }
            Some(y.max(0.0))
        })
        .collect()
}

/// Runs frames (`step`) until every pixel that sees a surface holds the full
/// history.
fn converge(f: &mut Fixture, mut step: impl FnMut(&mut Fixture)) {
    support::until(180, "the denoiser's history fills", || {
        step(f);
        lengths(f)
            .iter()
            .zip(hits(f))
            .all(|(l, hit)| !hit || *l >= MAX_HISTORY - 1e-3)
    });
}

fn wall_scene() -> Scene {
    Scene::new(vec![Quad::wall(
        DVec3::new(0.0, 0.0, -4.0),
        DVec2::new(20.0, 20.0),
    )])
}

fn front_view() -> View {
    View::looking(DVec3::new(0.0, 0.0, 0.0), DVec3::new(0.0, 0.0, -1.0))
}

fn hits(f: &Fixture) -> Vec<bool> {
    f.texels.iter().map(|t| t.hit.is_some()).collect()
}

/// A sample of standard deviation sigma (relative) a pixel a frame: the
/// running mean of at most N frames has a variance of at most sigma^2 / N
/// (equal weights up to N; the capped exponential mean after has
/// sigma^2 / (2N - 1)); the spatial stage only averages further. Bound: the
/// relative RMS error over the image is within sigma / sqrt(N), plus 15% for
/// the RMS estimated over the image's few independent regions; the mean
/// within a tenth of that (a bias).
#[test]
fn a_noisy_signal_converges_to_its_field_within_the_bound() {
    let Some(gpu) = support::gpu() else { return };
    let field = Field {
        a: 1.0,
        gradient: DVec3::new(0.05, 0.03, 0.0),
        b: DVec3::new(0.0, 0.0, 0.25),
    };
    let sigma = 0.5;
    for block in [TraceBlock::One, TraceBlock::Sixteen] {
        let signal = Signal::irradiance("test_irradiance", block);
        let mut f = fixture(gpu, signal);
        let p = Producer::new(gpu, signal, sigma);
        let scene = wall_scene();
        let view = front_view();
        converge(&mut f, |f| {
            frame(f, &p, &scene, &view, DVec3::ZERO, |_| field, 1.0)
        });
        // As many frames again: the mean forgets the start.
        for _ in 0..MAX_HISTORY as usize {
            frame(&mut f, &p, &scene, &view, DVec3::ZERO, |_| field, 1.0);
        }
        let out = denoised(&f);
        let (mut sum, mut sum_sq, mut n) = (0.0, 0.0, 0.0);
        for (i, y) in out.iter().enumerate() {
            let (Some(y), Some(hit)) = (y, f.texels[i].hit) else {
                continue;
            };
            let truth = field.at(hit.point, hit.normal);
            let e = y / truth - 1.0;
            sum += e;
            sum_sq += e * e;
            n += 1.0;
        }
        let (mean, rms) = (sum / n, (sum_sq / n).sqrt());
        let bound = sigma / f64::from(MAX_HISTORY).sqrt() * 1.15;
        println!(
            "{block:?}: relative rms error {rms:.4} (single sample {sigma}, bound {bound:.4}), mean {mean:+.4}"
        );
        assert!(rms < bound, "{block:?}: rms {rms} against {bound}");
        assert!(mean.abs() < bound / 10.0, "{block:?}: mean {mean}");
    }
}

/// A wall (1) behind a box (4) that moves by itself, then a camera that
/// moves: no pixel shows the other surface's light. Noise free, so what
/// remains is the half floats' rounding: 2% bounds a ghost of the other
/// surface at a hundredth of the difference.
#[test]
fn a_disocclusion_and_a_camera_move_do_not_ghost() {
    let Some(gpu) = support::gpu() else { return };
    for block in [TraceBlock::One, TraceBlock::Sixteen] {
        let signal = Signal::irradiance("test_ghost", block);
        let mut f = fixture(gpu, signal);
        let p = Producer::new(gpu, signal, 0.0);
        let wall = Quad::wall(DVec3::new(0.0, 0.0, -4.0), DVec2::new(20.0, 20.0));
        let box_at = |x: f64, moved: f64| Quad {
            moved: DVec3::new(moved, 0.0, 0.0),
            ..Quad::wall(DVec3::new(x, 0.0, -2.0), DVec2::new(0.35, 0.35))
        };
        let field = |hit: &support::Hit| Field::flat(if hit.point.z > -3.0 { 4.0 } else { 1.0 });
        let mut view = front_view();
        let still = Scene::new(vec![wall, box_at(-0.3, 0.0)]);
        converge(&mut f, |f| {
            frame(f, &p, &still, &view, DVec3::ZERO, field, 1.0)
        });
        let check = |f: &Fixture, what: &str| {
            let out = denoised(f);
            let mut worst: f64 = 0.0;
            for (i, y) in out.iter().enumerate() {
                let (Some(y), Some(hit)) = (y, f.texels[i].hit) else {
                    continue;
                };
                let truth = field(&hit).a;
                worst = worst.max((y / truth - 1.0).abs());
            }
            println!("{block:?} {what}: worst relative error {worst:.4}");
            assert!(worst < 0.02, "{block:?} {what}: {worst}");
        };
        // The box moves right by about 10 pixels a frame for three frames.
        let mut x = -0.3;
        for _ in 0..3 {
            x += 0.15;
            let moving = Scene::new(vec![wall, box_at(x, 0.15)]);
            frame(&mut f, &p, &moving, &view, DVec3::ZERO, field, 1.0);
            check(&f, "moving box");
        }
        // The camera pans and turns for four frames.
        let still = Scene::new(vec![wall, box_at(x, 0.0)]);
        for _ in 0..4 {
            view.eye += DVec3::new(0.12, 0.05, 0.0);
            view.forward = (view.forward + DVec3::new(-0.02, 0.0, 0.0)).normalize();
            frame(&mut f, &p, &still, &view, DVec3::ZERO, field, 1.0);
            check(&f, "moving camera");
        }
    }
}

/// A light that changes, said by the producer's confidence 0: the next
/// frame holds the new light alone. Without a confidence the history keeps
/// (1 - 1/N) of the old light, which the test shows it would have kept.
#[test]
fn confidence_zero_restarts_a_pixel() {
    let Some(gpu) = support::gpu() else { return };
    for with_confidence in [true, false] {
        let mut signal = Signal::irradiance("test_confidence", TraceBlock::One);
        if with_confidence {
            signal = signal.with_confidence();
        }
        let mut f = fixture(gpu, signal);
        let p = Producer::new(gpu, signal, 0.0);
        let scene = wall_scene();
        let view = front_view();
        converge(&mut f, |f| {
            frame(f, &p, &scene, &view, DVec3::ZERO, |_| Field::flat(1.0), 1.0)
        });
        frame(
            &mut f,
            &p,
            &scene,
            &view,
            DVec3::ZERO,
            |_| Field::flat(2.0),
            0.0,
        );
        let out = denoised(&f);
        let values: Vec<f64> = out.iter().flatten().copied().collect();
        let worst = values
            .iter()
            .map(|y| (y / 2.0 - 1.0).abs())
            .fold(0.0, f64::max);
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        println!("confidence {with_confidence}: mean {mean:.4} after the change");
        if with_confidence {
            assert!(worst < 0.01, "{worst}");
            // The history restarted: one frame's weight.
            for (l, hit) in lengths(&f).iter().zip(hits(&f)) {
                assert!(!hit || (*l - 1.0).abs() < 1e-3, "length {l}");
            }
        } else {
            let lagged = 1.0 + 1.0 / f64::from(MAX_HISTORY);
            assert!((mean - lagged).abs() < 0.01, "{mean} against {lagged}");
        }
    }
}

/// A signal traced at a quarter and a sixteenth of the pixels over a
/// bumpy surface: every pixel takes the light by its own normal. The field
/// is a + b . n with a constant: what remains is the half floats' rounding
/// in the input, the history and three spatial iterations (2^-11 each:
/// 0.5% with margin). Reading each pixel at its traced neighbour's normal
/// instead would err by b . (n - n0), which the test shows is far above it.
#[test]
fn quarter_and_sixteenth_signals_resolve_to_internal_resolution() {
    let Some(gpu) = support::gpu() else { return };
    let field = Field {
        a: 1.0,
        gradient: DVec3::ZERO,
        b: DVec3::new(0.3, 0.2, 0.4),
    };
    for block in [TraceBlock::Four, TraceBlock::Sixteen] {
        let signal = Signal::irradiance("test_resolve", block);
        let mut f = fixture(gpu, signal);
        let p = Producer::new(gpu, signal, 0.0);
        let mut scene = wall_scene();
        scene.bumps = Some(std::sync::Arc::new(|x: DVec3, n: DVec3| {
            n + DVec3::new(0.35 * (x.x * 9.0).sin(), 0.35 * (x.y * 7.0).cos(), 0.0)
        }));
        let view = front_view();
        converge(&mut f, |f| {
            frame(f, &p, &scene, &view, DVec3::ZERO, |_| field, 1.0)
        });
        let out = denoised(&f);
        let (mut worst, mut spread): (f64, f64) = (0.0, 0.0);
        for (i, y) in out.iter().enumerate() {
            let (Some(y), Some(hit)) = (y, f.texels[i].hit) else {
                continue;
            };
            let truth = field.at(hit.point, hit.normal);
            worst = worst.max((y / truth - 1.0).abs());
            // The light at the flat normal, what a block's one value gives.
            spread = spread.max((field.at(hit.point, DVec3::Z) / truth - 1.0).abs());
        }
        println!("{block:?}: worst relative error {worst:.4}; one value a block would err by {spread:.3}");
        assert!(spread > 0.05);
        assert!(worst < 0.005, "{block:?}: {worst}");
    }
}

/// Two runs of the same noisy signal, one moving the frame's origin by
/// whole cells mid-run with the camera still: their outputs differ only by
/// the f32 rounding of the reprojection (a millionth of a pixel), which can
/// flip a stochastic rounding of a stored half float by one step (2^-11):
/// once in the history (the running mean keeps no more than it) and once in
/// each of the spatial stage's four iterations. Bound: five steps at any
/// pixel, and no bias (the mean difference within one step).
#[test]
fn a_frame_origin_change_mid_run_changes_nothing() {
    let Some(gpu) = support::gpu() else { return };
    let field = Field {
        a: 1.0,
        gradient: DVec3::new(0.05, 0.0, 0.0),
        b: DVec3::new(0.1, 0.0, 0.2),
    };
    let far = DVec3::new(6_371_000.0, 0.0, 0.0);
    let scene = Scene::new(vec![Quad::wall(
        far + DVec3::new(0.0, 0.0, -4.0),
        DVec2::new(20.0, 20.0),
    )]);
    let view = View::looking(far, DVec3::new(0.0, 0.0, -1.0));
    let signal = Signal::irradiance("test_origin", TraceBlock::Sixteen);
    let mut outputs = Vec::new();
    for jump in [false, true] {
        let mut f = fixture(gpu, signal);
        let p = Producer::new(gpu, signal, 0.5);
        let origin = DVec3::new(6_370_816.0, 0.0, 0.0);
        let field = |_: &support::Hit| Field {
            gradient: field.gradient,
            // The field in the world, not the frame.
            a: field.a - field.gradient.dot(far),
            ..field
        };
        for i in 0..40 {
            let moved = jump && i >= 20;
            let origin = origin
                + if moved {
                    DVec3::new(1024.0, 0.0, -2048.0)
                } else {
                    DVec3::ZERO
                };
            frame(&mut f, &p, &scene, &view, origin, |h| field(h), 1.0);
        }
        outputs.push(denoised(&f));
    }
    let (mut worst, mut sum, mut n): (f64, f64, f64) = (0.0, 0.0, 0.0);
    for (a, b) in outputs[0].iter().zip(&outputs[1]) {
        if let (Some(a), Some(b)) = (a, b) {
            let d = (b - a) / a.max(1e-3);
            worst = worst.max(d.abs());
            sum += d;
            n += 1.0;
        }
    }
    let step = 2f64.powi(-11);
    println!(
        "origin change: worst relative difference {worst:.6}, mean {:+.6}",
        sum / n
    );
    assert!(worst <= 5.0 * step, "{worst}");
    assert!((sum / n).abs() <= step, "{}", sum / n);
}

/// The stages' GPU cost at 1920x1080 internal, pipelined (two frames in
/// flight, as a renderer runs), from the graph's GPU timings: the history's
/// pixels pass and each denoiser stage, medians over 200 frames after 100.
/// A measurement, not a check: run on demand
/// (`--ignored --nocapture the_stages_cost`).
#[test]
#[ignore = "a measurement at 1920x1080, run on demand"]
fn the_stages_cost_at_1080p() {
    let Some(gpu) = support::gpu() else { return };
    let size = [1920u32, 1080];
    for block in [TraceBlock::One, TraceBlock::Four, TraceBlock::Sixteen] {
        let signal = Signal::irradiance("test_cost", block);
        let mut f = Fixture::new(
            gpu,
            size,
            |env| {
                history::offer(env, helio_pass_history::publisher());
                denoise::offer(env, helio_pass_denoise::denoiser());
            },
            move |build| denoise::filter(build, &signal).map(|_| ()),
        );
        // Ground and a wall with a box: disocclusion at its edges, every
        // pixel a surface.
        let scene = Scene::new(vec![
            Quad::ground(DVec3::new(0.0, -1.0, -10.0), DVec2::new(100.0, 100.0)),
            Quad::wall(DVec3::new(0.0, 0.0, -30.0), DVec2::new(100.0, 40.0)),
            Quad::wall(DVec3::new(0.5, 0.0, -4.0), DVec2::new(0.5, 0.5)),
        ]);
        let view = View::looking(DVec3::new(0.0, 0.5, 0.0), DVec3::new(0.0, -0.1, -1.0));
        f.draw(&scene, &view, DVec3::ZERO);
        let traced = signal.traced_size(size);
        let texels = (traced[0] * traced[1]) as usize;
        let textures = [0, 1].map(|i| {
            gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: traced[0],
                    height: traced[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: signal.format(i).to_wgpu(),
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        });
        // A noisy signal, a new draw of its noise each frame (four, cycled):
        // the stages see a pixel's variance, as under a real producer.
        let draws: Vec<Vec<u16>> = (0..4u32)
            .map(|k| {
                (0..texels)
                    .flat_map(|i| [0.5 + random(i as u32, k) as f32, 0.05, 0.02, 2.0])
                    .map(support::f32_to_f16)
                    .collect()
            })
            .collect();
        let s1 = vec![[0.0f32, 0.1, 0.3, 1.0]; texels];
        support::write_half(gpu, &textures[1], &s1);
        let views = textures
            .each_ref()
            .map(|t| t.create_view(&Default::default()));
        let seeds = [(signal.inputs[0], &views[0]), (signal.inputs[1], &views[1])];
        let done = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let mut samples: std::collections::BTreeMap<&'static str, Vec<f64>> = Default::default();
        for i in 0..300 {
            gpu.queue.write_texture(
                textures[0].as_image_copy(),
                bytemuck::cast_slice(&draws[i % 4]),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(traced[0] * 8),
                    rows_per_image: Some(traced[1]),
                },
                textures[0].size(),
            );
            f.run_pipelined(&seeds, &done, 2);
            if i >= 100 {
                for t in f.graph.profiler().get_gpu_timings() {
                    if t.name.starts_with("Denoise") || t.name.starts_with("History") {
                        samples
                            .entry(t.name)
                            .or_default()
                            .push(t.duration_ns as f64 / 1e6);
                    }
                }
            }
        }
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        for (name, mut ms) in samples {
            ms.sort_by(f64::total_cmp);
            println!(
                "{block:?} {name}: median {:.3} ms (p10 {:.3}, p90 {:.3}, {} samples)",
                ms[ms.len() / 2],
                ms[ms.len() / 10],
                ms[ms.len() * 9 / 10],
                ms.len()
            );
        }
    }
}
