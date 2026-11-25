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

/// How a producer's samples scatter about the field (each a factor of it,
/// of mean 1).
#[derive(Clone, Copy, Debug)]
enum Noise {
    /// Uniform, of this relative standard deviation.
    Uniform(f64),
    /// One block of each group of `k` (2x2 for 4, 4x4 for 16), a new one
    /// each frame, carries the light of all `k`; the others none (a few
    /// rays that find a bright source among many that find nothing).
    OneIn(u32),
    /// Log-normal of this standard deviation of the logarithm, each 4x4
    /// group of blocks normalised to its mean.
    LogNormal(f64),
}

/// The hit distance a sample reports, from the hit it was traced at, the
/// frame, the block's index and the sample's factor of the field (`None`:
/// 3 m everywhere).
type HitDistance = std::sync::Arc<dyn Fn(&support::Hit, u64, u32, f64) -> f32>;

/// The producer: one sample at each block's traced pixel.
struct Producer {
    signal: Signal,
    textures: [wgpu::Texture; 2],
    views: [wgpu::TextureView; 2],
    noise: Noise,
    hit_distance: Option<HitDistance>,
}

/// Each block's factor of the field on `frame` (`Noise`). Over a group the
/// factors of `OneIn` and `LogNormal` average exactly 1 every frame, so a
/// whole image's input mean is the field's, frame by frame: what a test of
/// the denoiser's mean compares with.
fn factors(noise: Noise, traced: [u32; 2], frame: u32) -> Vec<f64> {
    let index = |bx: u32, by: u32| by * traced[0] + bx;
    let mut out = vec![1.0; (traced[0] * traced[1]) as usize];
    match noise {
        Noise::Uniform(sigma) => {
            for (i, f) in out.iter_mut().enumerate() {
                let u = random(frame, i as u32) * 2.0 - 1.0;
                *f = 1.0 + sigma * 3f64.sqrt() * u;
            }
        }
        Noise::OneIn(k) => {
            let g = if k == 4 { 2 } else { 4 };
            assert!(traced[0].is_multiple_of(g) && traced[1].is_multiple_of(g));
            for gy in 0..traced[1] / g {
                for gx in 0..traced[0] / g {
                    let pick =
                        (random(frame ^ 0x51ed_270b, gy * traced[0] + gx) * f64::from(k)) as u32;
                    for slot in 0..k {
                        let i = index(gx * g + slot % g, gy * g + slot / g);
                        out[i as usize] = if slot == pick { f64::from(k) } else { 0.0 };
                    }
                }
            }
        }
        Noise::LogNormal(sigma) => {
            assert!(traced[0].is_multiple_of(4) && traced[1].is_multiple_of(4));
            for gy in 0..traced[1] / 4 {
                for gx in 0..traced[0] / 4 {
                    let slots: Vec<u32> = (0..16)
                        .map(|s| index(gx * 4 + s % 4, gy * 4 + s / 4))
                        .collect();
                    // Box-Muller from two uniforms a block.
                    let v: Vec<f64> = slots
                        .iter()
                        .map(|&i| {
                            let u1 = random(frame.wrapping_mul(3) + 1, i).max(1e-12);
                            let u2 = random(frame.wrapping_mul(3) + 2, i);
                            let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                            (sigma * z).exp()
                        })
                        .collect();
                    let mean = v.iter().sum::<f64>() / 16.0;
                    for (&i, v) in slots.iter().zip(&v) {
                        out[i as usize] = v / mean;
                    }
                }
            }
        }
    }
    out
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
        Self::with_noise(gpu, signal, Noise::Uniform(noise))
    }

    fn with_noise(gpu: &support::Gpu, signal: Signal, noise: Noise) -> Self {
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
            hit_distance: None,
        }
    }

    /// This frame's samples of `field` (by surface: `fields[i]` for hits on
    /// quad `i`'s depth band), `confidence` each.
    fn write(&self, f: &Fixture, field: impl Fn(&support::Hit) -> Field, confidence: f32) {
        let side = self.signal.block.side();
        let traced = self.signal.traced_size(SIZE);
        let mut texels = [Vec::new(), Vec::new()];
        let colour = COLOUR.map(|c| c / luminance(COLOUR));
        let factors = factors(self.noise, traced, f.frame as u32);
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
                let index = by * traced[0] + bx;
                let scale = factors[index as usize];
                let hit_distance = self
                    .hit_distance
                    .as_ref()
                    .map_or(3.0, |h| h(&hit, f.frame, index, scale));
                let y = e * scale;
                let rgb = colour.map(|c| c * y);
                let co = 0.5 * (rgb[0] - rgb[2]);
                let cg = -0.25 * rgb[0] + 0.5 * rgb[1] - 0.25 * rgb[2];
                // The luminance-weighted direction whose first-order term is
                // the field's b (DENOISE_IRRADIANCE_SLOPE = 3/4).
                let d = field(&hit).b / 0.75 * scale;
                texels[0].push([y as f32, co as f32, cg as f32, hit_distance]);
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
    // M: (length).
    support::read(f.gpu, &history[2])
        .iter()
        .map(|t| t[0])
        .collect()
}

/// The denoised luminance at each pixel's own normal (`None`: sky); the
/// chroma came through as the field's.
fn denoised(f: &Fixture) -> Vec<Option<f64>> {
    let (out, chroma) = denoised_and_chroma(f);
    assert!(chroma < 0.01, "chroma off by {chroma}");
    out
}

/// The denoised luminance at each pixel's own normal (`None`: sky), and
/// the worst difference of a lit pixel's chroma ratios from the field's.
fn denoised_and_chroma(f: &Fixture) -> (Vec<Option<f64>>, f64) {
    let (_, out) = pass(f).textures().expect("a frame ran");
    let a = support::read(f.gpu, &out[0]);
    let b = support::read(f.gpu, &out[1]);
    let c = COLOUR.map(|c| c / luminance(COLOUR));
    let expected = [0.5 * (c[0] - c[2]), -0.25 * c[0] + 0.5 * c[1] - 0.25 * c[2]];
    let mut worst: f64 = 0.0;
    let out = f
        .texels
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let hit = t.hit?;
            let n = hit.normal;
            let y = f64::from(a[i][0])
                + f64::from(b[i][0]) * n.x
                + f64::from(b[i][1]) * n.y
                + f64::from(b[i][2]) * n.z;
            if y > 1e-3 && a[i][3] >= 0.0 {
                for (got, want) in [a[i][1], a[i][2]].iter().zip(expected) {
                    worst = worst.max((f64::from(*got) - want).abs());
                }
            }
            Some(y.max(0.0))
        })
        .collect();
    (out, worst)
}

/// The mean denoised luminance over the pixels that see a surface.
fn image_mean(out: &[Option<f64>]) -> f64 {
    let lit: Vec<f64> = out.iter().flatten().copied().collect();
    lit.iter().sum::<f64>() / lit.len() as f64
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
    for block in [TraceBlock::One, TraceBlock::Four, TraceBlock::Sixteen] {
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
    for block in [TraceBlock::One, TraceBlock::Four, TraceBlock::Sixteen] {
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
    let cases = [
        (TraceBlock::One, true),
        (TraceBlock::Four, true),
        (TraceBlock::Sixteen, true),
        (TraceBlock::One, false),
    ];
    for (block, with_confidence) in cases {
        let mut signal = Signal::irradiance("test_confidence", block);
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
        println!("{block:?} confidence {with_confidence}: mean {mean:.4} after the change");
        if with_confidence {
            assert!(worst < 0.01, "{block:?}: {worst}");
            // The history restarted: one frame's weight (traced at every
            // pixel, 1; else the tents of the four nearest blocks' traced
            // pixels, each at most 1), where a kept history would hold 32.
            for (l, hit) in lengths(&f).iter().zip(hits(&f)) {
                let one = if block == TraceBlock::One {
                    (*l - 1.0).abs() < 1e-3
                } else {
                    *l <= 4.0 + 1e-3
                };
                assert!(!hit || one, "{block:?}: length {l}");
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
    for block in [TraceBlock::One, TraceBlock::Four, TraceBlock::Sixteen] {
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
    for block in [TraceBlock::One, TraceBlock::Four, TraceBlock::Sixteen] {
        let signal = Signal::irradiance("test_origin", block);
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
            "{block:?} origin change: worst relative difference {worst:.6}, mean {:+.6}",
            sum / n
        );
        assert!(worst <= 5.0 * step, "{worst}");
        assert!((sum / n).abs() <= step, "{}", sum / n);
    }
}

/// The noise kinds of the mean tests: one block in 4 and in 16 carrying all
/// the light (relative deviations 1.7 and 3.9), log-normal of log-deviation
/// 0.75 and 1.5.
const HEAVY_TAILED: [Noise; 4] = [
    Noise::OneIn(4),
    Noise::OneIn(16),
    Noise::LogNormal(0.75),
    Noise::LogNormal(1.5),
];

/// Heavy-tailed positive signals over a uniformly lit wall, whose input's
/// image mean is the field's every frame (`factors`): once the history has
/// settled, the denoiser's output keeps that mean, its luminance within 2%
/// (a 32-frame mean of the image mean: the output's own noise in it is under
/// a tenth of that), and its colour (chroma within 0.01, as every test
/// asks). Once more with each sample's hit distance going with its light
/// (the bright ones short). A weight that falls with a sample's own
/// estimated variance or its own ray's length, or a chroma averaged
/// without its luminance, fails here.
#[test]
fn a_heavy_tailed_signal_keeps_its_mean() {
    let Some(gpu) = support::gpu() else { return };
    let scene = wall_scene();
    let view = front_view();
    let mut failures = Vec::new();
    let cases = HEAVY_TAILED
        .map(|noise| (noise, false))
        .into_iter()
        .chain([(Noise::OneIn(16), true)]);
    for (noise, short_bright) in cases {
        for block in [TraceBlock::One, TraceBlock::Four, TraceBlock::Sixteen] {
            let signal = Signal::irradiance("test_mean", block);
            let mut f = fixture(gpu, signal);
            let mut p = Producer::with_noise(gpu, signal, noise);
            if short_bright {
                // A sample's ray length going with its light: the bright
                // ones met a source 5 cm away, the others nothing within
                // 5 m (a gather's rays beside a lamp).
                p.hit_distance = Some(std::sync::Arc::new(
                    |_: &support::Hit, _, _, factor: f64| {
                        if factor > 1.0 {
                            0.05
                        } else {
                            5.0
                        }
                    },
                ));
            }
            let noise = format!(
                "{noise:?}{}",
                if short_bright {
                    " (bright rays short)"
                } else {
                    ""
                }
            );
            let field = |_: &support::Hit| Field::flat(1.0);
            converge(&mut f, |f| {
                frame(f, &p, &scene, &view, DVec3::ZERO, field, 1.0)
            });
            for _ in 0..MAX_HISTORY as usize {
                frame(&mut f, &p, &scene, &view, DVec3::ZERO, field, 1.0);
            }
            let (mut sum, mut chroma): (f64, f64) = (0.0, 0.0);
            let frames = 32;
            for _ in 0..frames {
                frame(&mut f, &p, &scene, &view, DVec3::ZERO, field, 1.0);
                let (out, c) = denoised_and_chroma(&f);
                sum += image_mean(&out) / f64::from(frames);
                chroma = chroma.max(c);
            }
            println!("{noise} {block:?}: output mean over input mean {sum:.4}, chroma off by {chroma:.4}");
            if (sum - 1.0).abs() >= 0.02 || chroma >= 0.01 {
                failures.push(format!("{noise} {block:?}: {sum:.4} (chroma {chroma:.4})"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

/// The output reports the samples' worth behind each estimate (output 0's
/// w), what a reader leans its prior on where it is short of
/// `denoise::SETTLED_SAMPLES`: on the plain wall with a full history every
/// pixel rests on at least that many; on the first frame after the wall
/// jumps from 4 to 6 m (its history rejected everywhere) the worth is what
/// this frame's samples and the spatial stage's widening give, less than
/// the settled worth.
#[test]
fn the_output_reports_the_samples_it_rests_on() {
    let Some(gpu) = support::gpu() else { return };
    let near = wall_scene();
    let far = Scene::new(vec![Quad::wall(
        DVec3::new(0.0, 0.0, -6.0),
        DVec2::new(30.0, 30.0),
    )]);
    let view = front_view();
    let field = |_: &support::Hit| Field::flat(1.0);
    for block in [TraceBlock::One, TraceBlock::Four, TraceBlock::Sixteen] {
        let signal = Signal::irradiance("test_samples", block);
        let mut f = fixture(gpu, signal);
        let p = Producer::new(gpu, signal, 0.5);
        converge(&mut f, |f| {
            frame(f, &p, &near, &view, DVec3::ZERO, field, 1.0)
        });
        let worth = |f: &Fixture| -> Vec<f32> {
            let (_, out) = pass(f).textures().expect("a frame ran");
            support::read(f.gpu, &out[0])
                .iter()
                .zip(hits(f))
                .filter(|(_, hit)| *hit)
                .map(|(t, _)| t[3])
                .collect()
        };
        let settled = worth(&f);
        frame(&mut f, &p, &far, &view, DVec3::ZERO, field, 1.0);
        // Pixels with light (a pixel none of whose samples hold its
        // surface yet reports none: below 0).
        let young: Vec<f32> = worth(&f).into_iter().filter(|w| *w >= 0.0).collect();
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let least = |v: &[f32]| v.iter().copied().fold(f32::MAX, f32::min);
        println!(
            "{block:?}: settled worth mean {:.2} least {:.2}; after the jump mean {:.2} least {:.2}",
            mean(&settled),
            least(&settled),
            mean(&young),
            least(&young)
        );
        assert!(
            least(&settled) >= denoise::SETTLED_SAMPLES - 1e-2,
            "{block:?}: a settled pixel rests on {}",
            least(&settled)
        );
        assert!(mean(&young) < mean(&settled), "{block:?}");
    }
}

/// The same signals in the first frames after a disocclusion: the wall
/// jumps between 4 and 6 m every 12 frames, so its history is rejected
/// everywhere (checked: no pixel keeps more than one frame's weight) and
/// the output rests on one to four frames of samples, filtered wide. 32
/// jumps a case; the output's image mean on each of the four frames after
/// a jump, averaged over the jumps, within 3% of the input's: a frame's
/// stratified samples have the field's mean exactly, and what the filter's
/// renormalisation at the image's edges and its residual noise leave in an
/// image mean of one to four frames' samples is under that.
#[test]
fn a_heavy_tailed_signal_keeps_its_mean_after_a_disocclusion() {
    let Some(gpu) = support::gpu() else { return };
    let near = wall_scene();
    let far = Scene::new(vec![Quad::wall(
        DVec3::new(0.0, 0.0, -6.0),
        DVec2::new(30.0, 30.0),
    )]);
    let view = front_view();
    let field = |_: &support::Hit| Field::flat(1.0);
    let mut failures = Vec::new();
    for noise in [Noise::OneIn(16), Noise::LogNormal(1.5)] {
        for block in [TraceBlock::One, TraceBlock::Four, TraceBlock::Sixteen] {
            let signal = Signal::irradiance("test_mean_short", block);
            let mut f = fixture(gpu, signal);
            let p = Producer::with_noise(gpu, signal, noise);
            converge(&mut f, |f| {
                frame(f, &p, &near, &view, DVec3::ZERO, field, 1.0)
            });
            let jumps = 32;
            let mut ratios = vec![Vec::new(); 4];
            for jump in 0..jumps {
                let scene = if jump % 2 == 0 { &far } else { &near };
                for (k, ratio) in ratios.iter_mut().enumerate() {
                    frame(&mut f, &p, scene, &view, DVec3::ZERO, field, 1.0);
                    if k == 0 {
                        let kept = lengths(&f)
                            .iter()
                            .zip(hits(&f))
                            .filter(|(_, hit)| *hit)
                            .map(|(l, _)| *l)
                            .fold(0.0f32, f32::max);
                        assert!(kept <= 4.0 + 1e-3, "{block:?}: history kept ({kept})");
                    }
                    ratio.push(image_mean(&denoised_and_chroma(&f).0));
                }
                for _ in 0..8 {
                    frame(&mut f, &p, scene, &view, DVec3::ZERO, field, 1.0);
                }
            }
            for (k, r) in ratios.iter().enumerate() {
                let mean = r.iter().sum::<f64>() / r.len() as f64;
                let sd = (r.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (r.len() - 1) as f64)
                    .sqrt();
                println!("{noise:?} {block:?}: frame {} after the jump, output over input {mean:.4} (jumps' sd {sd:.4})", k + 1);
                if (mean - 1.0).abs() >= 0.03 {
                    failures.push(format!("{noise:?} {block:?} frame {}: {mean:.4}", k + 1));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

/// An occluder's foot: a wall 1 m tall and 2 cm thick standing on a floor,
/// seen from right above it, so the floor either side of it lies two or
/// three pixels apart in the image (the wall's top between them), on the
/// same plane with the same normal: the geometry weight cannot tell them
/// apart. The floor before the wall is lit (1), behind it dark (0.2), the
/// wall 0.5; 50% noise a sample. The producer reports each sample's hit
/// distance as a gather does: one cosine-distributed ray traced into the
/// scene, no nearer than a block's footprint. Measured on the denoiser's
/// own output over 256 frames, so a level common to a side does not count:
/// the light carried behind (the floor within 10 cm behind the wall, the
/// reach of a block's footprint and its filter there, against the floor
/// more than 60 cm behind, as a share of the two far floors' difference)
/// and the dark carried before it, each under 2% (a ghost at a fiftieth of
/// the difference, as the disocclusion test bounds one; the bands' own
/// noise is under a third of that), and the far floors' noise
/// reduced as on the plain wall (relative RMS within sigma / sqrt(N), plus
/// 15%).
#[test]
fn light_is_not_carried_past_an_occluder_s_foot() {
    let Some(gpu) = support::gpu() else { return };
    let (wall_z, height, half_thickness) = (-3.0, 1.0, 0.01);
    let scene = Scene::new(vec![
        Quad::ground(DVec3::new(0.0, 0.0, -10.0), DVec2::new(30.0, 30.0)),
        // Its top, and its two faces (seen edge-on from the eye above it).
        Quad::ground(
            DVec3::new(0.0, height, wall_z),
            DVec2::new(30.0, half_thickness),
        ),
        Quad::wall(
            DVec3::new(0.0, 0.5 * height, wall_z + half_thickness),
            DVec2::new(30.0, 0.5 * height),
        ),
        Quad::wall(
            DVec3::new(0.0, 0.5 * height, wall_z - half_thickness),
            DVec2::new(30.0, 0.5 * height),
        ),
    ]);
    let view = View::looking(DVec3::new(0.0, 2.5, wall_z), DVec3::new(0.0, -1.0, -0.15));
    let (lit, dark, wall) = (1.0, 0.2, 0.5);
    let field = move |hit: &support::Hit| {
        Field::flat(if hit.quad != 0 {
            wall
        } else if hit.point.z > wall_z {
            lit
        } else {
            dark
        })
    };
    let sigma = 0.5;
    let mut failures = Vec::new();
    for block in [TraceBlock::One, TraceBlock::Four, TraceBlock::Sixteen] {
        let signal = Signal::irradiance("test_foot", block);
        let mut f = fixture(gpu, signal);
        let mut p = Producer::new(gpu, signal, sigma);
        // A block's footprint at the hit: its side in pixels at the view's
        // pixel width a metre of depth.
        let traced_scene = scene.clone();
        let side = f64::from(block.side());
        let pixel = 2.0 * (view.fov * 0.5).tan() / f64::from(SIZE[1]);
        let (eye, forward) = (view.eye, view.forward);
        p.hit_distance = Some(std::sync::Arc::new(
            move |hit: &support::Hit, frame: u64, index: u32, _: f64| {
                let n = hit.normal;
                let t = if n.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
                let u = n.cross(t).normalize();
                let v = n.cross(u);
                let r1 = random(frame as u32 ^ 0x2545_f491, index);
                let r2 = random(frame as u32 ^ 0x9e37_79b9, index);
                let (r, phi) = (r1.sqrt(), std::f64::consts::TAU * r2);
                let dir = u * (r * phi.cos()) + v * (r * phi.sin()) + n * (1.0 - r1).sqrt();
                let origin = hit.point + n * 1e-6;
                let footprint = side * pixel * (hit.point - eye).dot(forward);
                let t = traced_scene
                    .cast(origin, dir.normalize())
                    .map_or(1e9, |h| (h.point - origin).length());
                t.max(footprint) as f32
            },
        ));
        converge(&mut f, |f| {
            frame(f, &p, &scene, &view, DVec3::ZERO, field, 1.0)
        });
        for _ in 0..MAX_HISTORY as usize {
            frame(&mut f, &p, &scene, &view, DVec3::ZERO, field, 1.0);
        }
        // Sums and counts: behind near, behind far, before near, before
        // far; the far floors' squared relative error.
        let mut bands = [(0.0, 0.0); 4];
        let (mut sum_sq, mut n) = (0.0, 0.0);
        for _ in 0..256 {
            frame(&mut f, &p, &scene, &view, DVec3::ZERO, field, 1.0);
            let out = denoised(&f);
            for (y, t) in out.iter().zip(&f.texels) {
                let (Some(y), Some(hit)) = (y, t.hit) else {
                    continue;
                };
                if hit.quad != 0 {
                    continue;
                }
                let d = hit.point.z - wall_z;
                let band = match d {
                    d if d < 0.0 && d > -0.1 => 0,
                    d if d < -0.6 => 1,
                    d if d > 0.0 && d < 0.1 => 2,
                    d if d > 0.6 => 3,
                    _ => continue,
                };
                bands[band].0 += y;
                bands[band].1 += 1.0;
                if band == 1 || band == 3 {
                    let truth = if band == 1 { dark } else { lit };
                    sum_sq += (y / truth - 1.0).powi(2);
                    n += 1.0;
                }
            }
        }
        let mean = bands.map(|(s, n)| s / n);
        let difference = mean[3] - mean[1];
        let carried = (mean[0] - mean[1]) / difference;
        let darkened = (mean[3] - mean[2]) / difference;
        let rms = (sum_sq / n).sqrt();
        let bound = sigma / f64::from(MAX_HISTORY).sqrt() * 1.15;
        println!(
            "{block:?}: carried behind the wall {carried:+.4}, dark carried before it {darkened:+.4} (far floors {:.4} / {:.4}; pixel-frames {:?}), far floors' relative rms {rms:.4} (bound {bound:.4})",
            mean[1],
            mean[3],
            bands.map(|b| b.1)
        );
        if carried.abs() >= 0.02 || darkened.abs() >= 0.02 || rms >= bound {
            failures.push(format!(
                "{block:?}: {carried:+.4} {darkened:+.4} rms {rms:.4}"
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
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
