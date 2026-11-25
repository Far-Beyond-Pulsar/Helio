//! "Where a pixel was" against positions computed in f64 on the CPU: a still
//! camera (its jitter moving), a moving camera, an object that moves by
//! itself, and a frame-origin change. Skipped without a GPU.
//!
//! The bound: a previous position is stored as an offset in a half float,
//! whose step is 2^-10 of its magnitude (2^-7 px under 16 px of motion); the
//! reprojection is the f64 composition rounded to f32 and evaluated in f32
//! (1e-6 of the raster, 1e-4 px at 128). 0.02 px holds both with margin and
//! is a fiftieth of what TSR or a denoiser would notice.

mod support;

use glam::{DVec2, DVec3};
use helio_frame_api::history::{self, pixel_flags, Need};
use support::{Fixture, Quad, Scene, View};

const SIZE: [u32; 2] = [96, 64];
const BOUND: f64 = 0.02;

fn fixture(gpu: &'static support::Gpu) -> Fixture {
    Fixture::new(
        gpu,
        SIZE,
        |env| history::offer(env, helio_pass_history::publisher()),
        |build| {
            assert!(history::request(build, Need::Pixels)?);
            // A second reader adds nothing.
            assert!(history::request(build, Need::Reprojection)?);
            Ok(())
        },
    )
}

/// The `PIXELS` texture of the frame just run.
fn pixels(f: &Fixture) -> Vec<[f32; 4]> {
    let pass = f
        .graph
        .find_pass::<helio_pass_history::HistoryPixelsPass>()
        .expect("the graph holds the pixels pass");
    support::read(f.gpu, &pass.pixels_texture().expect("a frame ran"))
}

/// Checks every pixel of the frame just run against `expected` (given the
/// pixel, its surface and the previous position the GPU found: the previous
/// position of its surface in pixels; whether the 2x2 the GPU read there
/// should all hold the same surface (`Some(true)`), none of it
/// (`Some(false)`) or is mixed (`None`); whether it moves by itself),
/// returning the worst error. The taps are judged around the GPU's position:
/// a pixel centre reprojected onto itself lies on a texel boundary, where
/// rounding picks either 2x2.
fn check(
    f: &Fixture,
    mut expected: impl FnMut([u32; 2], &support::Hit, DVec2) -> (DVec2, Option<bool>, bool),
    explain: impl Fn([u32; 2], DVec2) -> String,
) -> f64 {
    let texels = pixels(f);
    let mut worst = 0.0f64;
    let mut checked = 0;
    for y in 0..SIZE[1] {
        for x in 0..SIZE[0] {
            let i = (y * SIZE[0] + x) as usize;
            let Some(hit) = f.texels[i].hit else {
                continue;
            };
            let t = texels[i];
            let got = DVec2::new(
                f64::from(x) + 0.5 + f64::from(t[0]),
                f64::from(y) + 0.5 + f64::from(t[1]),
            );
            let (previous, all_taps, moves) = expected([x, y], &hit, got);
            let flags = t[3] as u32;
            assert_eq!(
                flags & pixel_flags::MOVES != 0,
                moves,
                "pixel {x},{y}: flags {flags}"
            );
            assert!(
                flags & pixel_flags::HAS_PREVIOUS != 0,
                "pixel {x},{y}: no previous"
            );
            let inside = previous.x > 1.0
                && previous.y > 1.0
                && previous.x < f64::from(SIZE[0]) - 1.0
                && previous.y < f64::from(SIZE[1]) - 1.0;
            if inside {
                worst = worst.max(got.distance(previous));
                let want = match all_taps {
                    Some(true) => Some(0xf),
                    Some(false) => Some(0),
                    None => None,
                };
                if let Some(want) = want {
                    assert_eq!(
                        t[2] as u32,
                        want,
                        "pixel {x},{y} (quad {}, previous {previous}, gpu {got}): taps {}; {}",
                        hit.quad,
                        t[2],
                        explain([x, y], got)
                    );
                }
                checked += 1;
            }
        }
    }
    assert!(checked > 1000, "only {checked} pixels checked");
    worst
}

/// Ground and a wall, a camera looking down at them at an angle.
fn scene() -> Scene {
    Scene::new(vec![
        Quad::ground(DVec3::new(0.0, 0.0, -10.0), DVec2::new(50.0, 50.0)),
        Quad::wall(DVec3::new(0.0, 2.0, -14.0), DVec2::new(20.0, 4.0)),
    ])
}

fn camera(eye: DVec3, jitter: DVec2) -> View {
    View::looking(eye, DVec3::new(0.0, -0.4, -1.0)).with_jitter(jitter)
}

/// Whether the four texels around `p` (the previous position the GPU found)
/// lie on the surface (quad) the pixel's does: `None` where the 2x2 straddles
/// two surfaces or the sky.
fn interior(previous: &View, p: DVec2, hit: &support::Hit, scene: &Scene) -> Option<bool> {
    let base = (p - 0.5).floor();
    for i in 0..4u32 {
        let t = base + DVec2::new(f64::from(i & 1), f64::from(i >> 1));
        let dir = support::ray(previous, SIZE, [t.x as u32, t.y as u32]);
        if scene.cast(previous.eye, dir)?.quad != hit.quad {
            return None;
        }
    }
    Some(true)
}

/// What the previous frame held around `got` and at the pixel's
/// neighbours: each tap's quad and view depth along the previous camera,
/// the pixel's own previous depth and its neighbours' quads (a failure's
/// message).
fn explain(f: &Fixture, previous: &View, scene: &Scene, px: [u32; 2], got: DVec2) -> String {
    let mut out = String::new();
    let depth = |p: DVec3| (p - previous.eye).dot(previous.forward);
    let base = (got - 0.5).floor();
    for i in 0..4u32 {
        let t = base + DVec2::new(f64::from(i & 1), f64::from(i >> 1));
        let dir = support::ray(previous, SIZE, [t.x as u32, t.y as u32]);
        if let Some(h) = scene.cast(previous.eye, dir) {
            out += &format!("tap {t}: quad {} depth {:.5}; ", h.quad, depth(h.point));
        }
    }
    let at = |x: i64, y: i64| {
        let i = (y * i64::from(SIZE[0]) + x) as usize;
        f.texels[i].hit
    };
    let (x, y) = (i64::from(px[0]), i64::from(px[1]));
    if let Some(h) = at(x, y) {
        out += &format!("own previous depth {:.5}; ", depth(h.point));
    }
    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
        if let Some(h) = at(x + dx, y + dy) {
            out += &format!(
                "neighbour {dx},{dy}: quad {} previous depth {:.5}; ",
                h.quad,
                depth(h.point)
            );
        }
    }
    out
}

#[test]
fn a_still_camera_finds_each_pixel_where_its_jitter_left_it() {
    let Some(gpu) = support::gpu() else { return };
    let mut f = fixture(gpu);
    let scene = scene();
    let eye = DVec3::new(0.3, 3.0, 2.0);
    let jitters = [DVec2::new(0.25, -0.125), DVec2::new(-0.375, 0.3125)];
    f.draw(&scene, &camera(eye, jitters[0]), DVec3::ZERO);
    f.run(&[]);
    let previous = camera(eye, jitters[0]);
    let now = camera(eye, jitters[1]);
    f.draw(&scene, &now, DVec3::ZERO);
    f.run(&[]);
    let worst = check(
        &f,
        |_, hit, got| {
            let (p, _) = support::project(&previous, SIZE, hit.point);
            (p, interior(&previous, got, hit, &scene), false)
        },
        |px, got| explain(&f, &previous, &scene, px, got),
    );
    println!("still camera: worst {worst} px");
    assert!(worst < BOUND, "{worst}");
}

#[test]
fn a_moving_camera_finds_each_pixel_by_its_depth() {
    let Some(gpu) = support::gpu() else { return };
    let mut f = fixture(gpu);
    let scene = scene();
    let before = camera(DVec3::new(0.3, 3.0, 2.0), DVec2::new(0.25, -0.125));
    let mut after = camera(DVec3::new(0.9, 2.7, 1.4), DVec2::new(-0.375, 0.3125));
    after.forward = DVec3::new(0.15, -0.35, -1.0).normalize();
    f.draw(&scene, &before, DVec3::ZERO);
    f.run(&[]);
    f.draw(&scene, &after, DVec3::ZERO);
    f.run(&[]);
    let worst = check(
        &f,
        |_, hit, got| {
            let (p, _) = support::project(&before, SIZE, hit.point);
            (p, interior(&before, got, hit, &scene), false)
        },
        |px, got| explain(&f, &before, &scene, px, got),
    );
    println!("moving camera: worst {worst} px");
    assert!(worst < BOUND, "{worst}");
}

#[test]
fn an_object_that_moves_by_itself_is_found_by_its_motion_and_uncovers_what_it_hid() {
    let Some(gpu) = support::gpu() else { return };
    let mut f = fixture(gpu);
    let eye = DVec3::new(0.0, 1.0, 4.0);
    let view = View::looking(eye, DVec3::new(0.0, 0.0, -1.0));
    let wall = Quad::wall(DVec3::new(0.0, 1.0, -6.0), DVec2::new(30.0, 20.0));
    let box_at = |x: f64, moved: DVec3| Quad {
        moved,
        ..Quad::wall(DVec3::new(x, 1.0, 0.0), DVec2::new(0.6, 0.6))
    };
    let step = DVec3::new(0.35, 0.0, 0.0);
    let before = Scene::new(vec![wall, box_at(-0.2, DVec3::ZERO)]);
    let after = Scene::new(vec![wall, box_at(-0.2 + step.x, step)]);
    f.draw(&before, &view, DVec3::ZERO);
    f.run(&[]);
    f.draw(&after, &view, DVec3::ZERO);
    f.run(&[]);
    let mut uncovered = 0;
    let worst = check(
        &f,
        |_, hit, got| match hit.previous {
            Some(was) => {
                let (q, _) = support::project(&view, SIZE, was);
                (q, None, true)
            }
            None => {
                let (q, _) = support::project(&view, SIZE, hit.point);
                // Wall the box covered last frame (all four taps on it).
                let covered = (0..4u32).all(|i| {
                    let t = (got - 0.5).floor() + DVec2::new(f64::from(i & 1), f64::from(i >> 1));
                    let dir = support::ray(&view, SIZE, [t.x as u32, t.y as u32]);
                    before
                        .cast(eye, dir)
                        .is_some_and(|h| (h.point.z - 0.0).abs() < 1e-6)
                });
                let free = (0..4u32).all(|i| {
                    let t = (got - 0.5).floor() + DVec2::new(f64::from(i & 1), f64::from(i >> 1));
                    let dir = support::ray(&view, SIZE, [t.x as u32, t.y as u32]);
                    before
                        .cast(eye, dir)
                        .is_some_and(|h| (h.point.z + 6.0).abs() < 1e-6)
                });
                if covered {
                    uncovered += 1;
                }
                (
                    q,
                    if covered {
                        Some(false)
                    } else if free {
                        Some(true)
                    } else {
                        None
                    },
                    false,
                )
            }
        },
        |px, got| explain(&f, &view, &before, px, got),
    );
    println!("moving object: worst {worst} px, {uncovered} pixels uncovered");
    assert!(uncovered > 50, "{uncovered}");
    assert!(worst < BOUND, "{worst}");
}

#[test]
fn a_frame_origin_change_moves_no_pixel() {
    let Some(gpu) = support::gpu() else { return };
    let mut f = fixture(gpu);
    let scene = scene();
    // Far from the world's origin, as on a planet: the frame origin jumps
    // by whole cells between the frames while the camera holds still.
    let eye = DVec3::new(6_371_000.3, 3.0, 2.0);
    let scene = Scene::new(
        scene
            .quads
            .iter()
            .map(|q| Quad {
                centre: q.centre + DVec3::new(6_371_000.0, 0.0, 0.0),
                ..*q
            })
            .collect(),
    );
    let jitters = [DVec2::new(0.25, -0.125), DVec2::new(-0.375, 0.3125)];
    let origin = DVec3::new(6_370_816.0, 0.0, 0.0);
    f.draw(&scene, &camera(eye, jitters[0]), origin);
    f.run(&[]);
    let previous = camera(eye, jitters[0]);
    f.draw(
        &scene,
        &camera(eye, jitters[1]),
        origin + DVec3::new(1024.0, 0.0, -2048.0),
    );
    f.run(&[]);
    let worst = check(
        &f,
        |_, hit, got| {
            let (p, _) = support::project(&previous, SIZE, hit.point);
            (p, interior(&previous, got, hit, &scene), false)
        },
        |px, got| explain(&f, &previous, &scene, px, got),
    );
    println!("origin change: worst {worst} px");
    assert!(worst < BOUND, "{worst}");
}

/// A view's first frame has no previous one: nothing matches.
#[test]
fn a_first_frame_has_no_history() {
    let Some(gpu) = support::gpu() else { return };
    let mut f = fixture(gpu);
    f.draw(
        &scene(),
        &camera(DVec3::new(0.3, 3.0, 2.0), DVec2::ZERO),
        DVec3::ZERO,
    );
    f.run(&[]);
    for t in pixels(&f) {
        assert_eq!(t[2], 0.0);
        assert_eq!(t[3] as u32 & pixel_flags::HAS_PREVIOUS, 0);
    }
}
