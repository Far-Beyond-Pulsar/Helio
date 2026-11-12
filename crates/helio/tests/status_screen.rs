//! The status screen as the output frame shows it: a near-black field and one
//! red dialog, centred, inside 90% of the frame, with no glow around it.
//!
//! Each state is rendered to an offscreen target and read back. The tests look
//! at pixels only, so they check the shader and the layout together, the way
//! a person looks at the frame. With `HELIO_STATUS_PNG_DIR` set they also write
//! each frame there as a PNG, for review against the design.
//!
//! Needs a GPU device; without one the tests skip.

#[macro_use]
mod common;

use std::path::PathBuf;
use std::sync::Arc;

use common::{gpu_serial, is_void, Frame, Gpu};
use helio::{GraphRebuilder, NamedPipeline, PassBuildContext, PipelineFailure};
use helio_core::RenderGraph;

/// A rebuilder that never builds: it is what makes a failure "rebuildable".
fn rebuilder() -> GraphRebuilder {
    Arc::new(|_, _, _, _, _, _, _| Err(PipelineFailure::new("still failing")))
}

#[derive(Clone, Copy, Debug)]
enum State {
    None,
    NodesMissing,
    BuildFailed,
    Stress,
}

const STATES: [(State, &str); 4] = [
    (State::None, "none"),
    (State::NodesMissing, "nodes_missing"),
    (State::BuildFailed, "build_failed"),
    (State::Stress, "stress_46"),
];

fn failure(state: State) -> Option<(&'static str, PipelineFailure)> {
    match state {
        State::None => None,
        State::NodesMissing => Some((
            "assets/pipelines/forest.rendergraph",
            PipelineFailure::new("the graph does not validate: 2 nodes are not available.")
                .with_errors([
                    "missing node `helio.pass.voxel_chunks` (plugin `pulsar.voxel` disabled)"
                        .to_owned(),
                    "missing node `helio.pass.water_surface` (plugin `pulsar.water` disabled)"
                        .to_owned(),
                ])
                .with_warnings([
                    "`Foliage: Cull` reads `voxel_ground`, which nothing writes".to_owned(),
                ])
                .with_rebuilder(rebuilder()),
        )),
        State::BuildFailed => Some((
            "preview:12",
            PipelineFailure::new(
                "the GPU rejected what the pipeline built (an invalid shader or pipeline state)",
            )
            .with_errors([
                "wgpu validation: `Device::create_compute_pipeline`, label `SpriteRows`"
                    .to_owned(),
                "Not enough memory resources are available to complete this operation (0x8007000E)"
                    .to_owned(),
            ])
            .with_notes([
                "adapter `NVIDIA GeForce RTX 3070`, 6.1 GiB of 8.0 GiB in use".to_owned(),
                "build took 4.2 s, 31 of 52 passes were built".to_owned(),
            ])
            .with_rebuilder(rebuilder()),
        )),
        State::Stress => Some((
            "assets/pipelines/legacy/everything-in-one-graph-with-a-very-long-name-that-has-to-wrap.rendergraph",
            PipelineFailure::new("the graph does not validate: 46 problems.")
                .with_details((0..46).map(|i| {
                    if i % 7 == 0 {
                        format!(
                            "warning: routing: `pass {}` writes `scene_color` and nothing reads it",
                            200 + i
                        )
                    } else {
                        format!(
                            "error: missing node `helio.pass.layer_{}` (plugin `pulsar.legacy` disabled)",
                            200 + i
                        )
                    }
                }))
                .with_rebuilder(rebuilder()),
        )),
    }
}

fn render(gpu: &Gpu, state: State, size: [u32; 2], format: wgpu::TextureFormat) -> Frame {
    let mut renderer = gpu.renderer_in(size, None, format);
    if let Some((id, failure)) = failure(state) {
        renderer.fail_pipeline(id, failure);
    }
    gpu.frame_in(&mut renderer, format)
}

fn save(frame: &Frame, name: &str) {
    let Ok(dir) = std::env::var("HELIO_STATUS_PNG_DIR") else {
        return;
    };
    let dir = PathBuf::from(dir);
    std::fs::create_dir_all(&dir).expect("the PNG directory is made");
    image::save_buffer(
        dir.join(format!("{name}.png")),
        &frame.pixels,
        frame.width,
        frame.height,
        image::ExtendedColorType::Rgba8,
    )
    .expect("the PNG is written");
}

/// The dialog is the only red thing: its fill is far redder than the field,
/// the shadow (black) or the text (white or pale).
fn is_panel(p: [u8; 4]) -> bool {
    p[0] >= 20 && u32::from(p[0]) > 2 * u32::from(p[1].max(p[2]))
}

/// The rectangle of red pixels, as (left, top, right, bottom) inclusive.
fn dialog_rect(frame: &Frame) -> Option<[u32; 4]> {
    let mut rect: Option<[u32; 4]> = None;
    for y in 0..frame.height {
        for x in 0..frame.width {
            if is_panel(frame.at(x, y)) {
                rect = Some(match rect {
                    None => [x, y, x, y],
                    Some([l, t, r, b]) => [l.min(x), t.min(y), r.max(x), b.max(y)],
                });
            }
        }
    }
    rect
}

fn assert_design(frame: &Frame, label: &str) -> [u32; 4] {
    let (w, h) = (frame.width, frame.height);

    // Mostly black: the corners, the middles of the edges.
    for (x, y) in [
        (0, 0),
        (w - 1, 0),
        (0, h - 1),
        (w - 1, h - 1),
        (w / 2, 0),
        (w / 2, h - 1),
        (0, h / 2),
        (w - 1, h / 2),
    ] {
        assert!(
            is_void(frame.at(x, y)),
            "{label}: ({x}, {y}) is {:?}, not near-black",
            frame.at(x, y)
        );
    }
    let black = frame
        .pixels
        .chunks_exact(4)
        .filter(|p| is_void([p[0], p[1], p[2], p[3]]))
        .count();
    assert!(
        black as f32 > 0.4 * (w * h) as f32,
        "{label}: only {black} of {} pixels are black",
        w * h
    );

    // A red dialog around the centre.
    let [l, t, r, b] = dialog_rect(frame).unwrap_or_else(|| panic!("{label}: no red dialog"));
    let centre = frame.at(w / 2, h / 2);
    assert!(
        centre[0] > centre[2] && centre[0] > 15,
        "{label}: the centre is {centre:?}"
    );
    // Inside the frame, inside 90%, centred.
    let (dw, dh) = (r - l + 1, b - t + 1);
    assert!(dw as f32 <= 0.9 * w as f32 + 2.0, "{label}: width {dw} of {w}");
    assert!(dh as f32 <= 0.9 * h as f32 + 2.0, "{label}: height {dh} of {h}");
    let (cx, cy) = ((l + r) as f32 / 2.0, (t + b) as f32 / 2.0);
    assert!(
        (cx - (w - 1) as f32 / 2.0).abs() <= 2.0 && (cy - (h - 1) as f32 / 2.0).abs() <= 2.0,
        "{label}: dialog [{l}, {t}, {r}, {b}] is not centred in {w}x{h}"
    );

    // No glow: past a margin round the dialog nothing is red. The shadow
    // only darkens the black field.
    let margin = 6;
    for y in 0..h {
        for x in 0..w {
            let outside = x + margin < l || x > r + margin || y + margin < t || y > b + margin;
            if outside {
                let p = frame.at(x, y);
                assert!(
                    p[0] <= 14 && p[1] <= 14 && p[2] <= 14,
                    "{label}: ({x}, {y}) is {p:?} outside the dialog [{l}, {t}, {r}, {b}]"
                );
            }
        }
    }
    [l, t, r, b]
}

#[test]
fn every_state_is_a_red_dialog_on_black_at_1280_by_720() {
    let _serial = gpu_serial();
    let gpu = gpu_or_skip!("status screen frames");
    let mut heights = Vec::new();
    for (state, name) in STATES {
        let frame = render(&gpu, state, [1280, 720], common::FORMAT);
        save(&frame, &format!("status_{name}_1280x720"));
        let rect = assert_design(&frame, name);
        heights.push((name, rect[3] - rect[1] + 1));
    }
    // The dialog is sized to its content: the short message is shorter than
    // the stress list, which is the tallest the frame allows.
    let height = |n: &str| heights.iter().find(|(name, _)| *name == n).unwrap().1;
    assert!(height("none") < height("nodes_missing"));
    assert!(height("nodes_missing") < height("stress_46"));
    assert!(height("stress_46") as f32 <= 0.9 * 720.0 + 2.0);
}

#[test]
fn the_stress_list_fits_smaller_frames() {
    let _serial = gpu_serial();
    let gpu = gpu_or_skip!("status screen small frames");
    for size in [[720, 405], [420, 320], [1920, 1080]] {
        for (state, name) in STATES {
            let frame = render(&gpu, state, size, common::FORMAT);
            save(&frame, &format!("status_{name}_{}x{}", size[0], size[1]));
            assert_design(&frame, &format!("{name} {}x{}", size[0], size[1]));
        }
    }
}

#[test]
fn an_srgb_target_looks_the_same_as_a_linear_one() {
    let _serial = gpu_serial();
    let gpu = gpu_or_skip!("status screen sRGB");
    let linear = render(&gpu, State::NodesMissing, [640, 360], wgpu::TextureFormat::Rgba8Unorm);
    for format in [
        wgpu::TextureFormat::Rgba8UnormSrgb,
        wgpu::TextureFormat::Bgra8UnormSrgb,
    ] {
        let srgb = render(&gpu, State::NodesMissing, [640, 360], format);
        let worst = linear
            .pixels
            .iter()
            .zip(&srgb.pixels)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        assert!(
            worst <= 3,
            "{format:?} differs from Rgba8Unorm by {worst} (the shader converts for sRGB targets)"
        );
    }
}

#[test]
fn the_frame_is_drawn_whatever_was_in_the_target() {
    // The pass writes every pixel, so the host does not clear the target:
    // what it held before must not show.
    let _serial = gpu_serial();
    let gpu = gpu_or_skip!("status screen overwrites");
    let mut renderer = gpu.renderer([320, 180], None);
    let first = gpu.frame(&mut renderer);
    let second = gpu.frame(&mut renderer);
    assert_eq!(first.pixels, second.pixels, "the same screen, frame after frame");
    // A pipeline in between does not leave its clear colour behind.
    renderer.set_pipeline(Some(NamedPipeline::new(
        "a",
        |ctx: PassBuildContext<'_>| -> RenderGraph {
            let mut graph = RenderGraph::new_with_external_device(ctx.device, ctx.queue);
            graph.lock(ctx.config.width, ctx.config.height);
            graph
        },
    )));
    gpu.frame(&mut renderer);
    renderer.set_pipeline(None);
    let third = gpu.frame(&mut renderer);
    assert_eq!(first.pixels, third.pixels, "none, then a pipeline, then none");
}
