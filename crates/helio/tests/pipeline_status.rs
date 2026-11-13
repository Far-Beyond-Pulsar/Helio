//! A renderer needs no pipeline, swaps pipelines at a frame boundary, and
//! shows why when it has none or the one it has does not build. The checks
//! read the output frame back (in tests only): the status screen is drawn, so
//! its pixels exist, and no other graph stands in for a failed pipeline.
//!
//! Needs a GPU device; without one the tests skip.

#[macro_use]
mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use common::{gpu, gpu_serial, is_void, near, Gpu, FORMAT, PIPELINE_CLEAR};
use helio::{
    GraphRebuilder, IntoGraph, NamedPipeline, PassBuildContext, PipelineFailure, PipelineStatus,
    Renderer,
};
use helio_core::RenderGraph;
use helio_pass_debug_overlay::{DebugOverlayPass, DebugOverlayState};
use pulsar_scenedb::gpu::GpuMirrorHandle;

/// A pipeline that draws `text` with the overlay pass over the clear colour.
/// Counts its builds, and its graph's drops through `alive`.
struct TextPipeline {
    text: &'static str,
    builds: Arc<AtomicUsize>,
}

impl IntoGraph for TextPipeline {
    fn into_graph(
        self: Box<Self>,
        ctx: PassBuildContext<'_>,
    ) -> Result<RenderGraph, PipelineFailure> {
        self.builds.fetch_add(1, Ordering::SeqCst);
        let mut graph = RenderGraph::new_with_external_device(ctx.device, ctx.queue);
        let state = DebugOverlayState::new();
        {
            let text = self.text;
            let mut state = state.lock().unwrap();
            state.enabled = true;
            state.populate = Some(Box::new(move |s| {
                s.clear();
                s.write_text(2, 2, text);
            }));
        }
        graph.add_pass(Box::new(DebugOverlayPass::new(
            ctx.device,
            ctx.queue,
            state,
            ctx.config.surface_format,
            ctx.config.width,
            ctx.config.height,
        )));
        graph.lock(ctx.config.width, ctx.config.height);
        Ok(graph)
    }
}

fn text_pipeline(id: &str, text: &'static str) -> (NamedPipeline, Arc<AtomicUsize>) {
    let builds = Arc::new(AtomicUsize::new(0));
    let pipeline = NamedPipeline::new(
        id,
        TextPipeline {
            text,
            builds: Arc::clone(&builds),
        },
    );
    (pipeline, builds)
}

/// A pipeline whose build fails with `failure`.
struct FailingPipeline(Box<dyn Fn() -> PipelineFailure>);

impl IntoGraph for FailingPipeline {
    fn into_graph(
        self: Box<Self>,
        _: PassBuildContext<'_>,
    ) -> Result<RenderGraph, PipelineFailure> {
        Err((self.0)())
    }
}

#[test]
fn a_renderer_with_no_pipeline_builds_and_says_so() {
    let _serial = gpu_serial();
    let gpu = gpu_or_skip!("no pipeline");
    let mut renderer = gpu.renderer([640, 360], None);
    assert_eq!(renderer.pipeline_status(), &PipelineStatus::None);
    assert_eq!(renderer.pipeline_id(), None);

    let frame = gpu.frame(&mut renderer);
    assert!(
        frame.text_pixels() > 200,
        "the status screen is drawn text, not a black frame ({} text pixels)",
        frame.text_pixels()
    );
    // The frame is the status screen's near-black field, not the pipeline's
    // clear colour.
    assert!(is_void(frame.corner()), "{:?}", frame.corner());
}

#[test]
fn the_hint_the_host_gives_is_on_the_screen_text() {
    let _serial = gpu_serial();
    let gpu = gpu_or_skip!("status hint");
    let mut renderer = gpu.renderer([640, 360], None);
    let before = gpu.frame(&mut renderer);
    renderer
        .set_status_hint("Choose one under World Settings, Render Pipeline, in the level editor.");
    let after = gpu.frame(&mut renderer);
    assert_ne!(
        before.pixels, after.pixels,
        "a screen already showing is drawn again"
    );
}

#[test]
fn an_invalid_pipeline_shows_its_reason_and_is_in_the_status() {
    let _serial = gpu_serial();
    let gpu = gpu_or_skip!("invalid pipeline");
    let failing = NamedPipeline::new(
        "game/post.rendergraph",
        FailingPipeline(Box::new(|| {
            PipelineFailure::new("missing node `voxel.terrain` (plugin voxel disabled)")
                .with_details(["error: terrain: node kind is not registered".to_owned()])
        })),
    );
    let mut renderer = gpu.renderer([640, 360], Some(failing));
    assert_eq!(
        renderer.pipeline_status(),
        &PipelineStatus::Invalid {
            id: "game/post.rendergraph".into(),
            reason: "missing node `voxel.terrain` (plugin voxel disabled)".into(),
            details: vec!["error: terrain: node kind is not registered".into()],
        }
    );
    assert_eq!(renderer.pipeline_id(), Some("game/post.rendergraph"));

    let frame = gpu.frame(&mut renderer);
    assert!(frame.text_pixels() > 200, "{}", frame.text_pixels());
    assert!(is_void(frame.corner()), "{:?}", frame.corner());
    // The dialog is red around the centre of the frame.
    let centre = frame.at(frame.width / 2, frame.height / 2);
    assert!(
        centre[0] > 15 && centre[0] > 2 * centre[1].max(centre[2]),
        "centre {centre:?}"
    );
}

#[test]
fn a_pipeline_that_panics_is_invalid_not_a_crash() {
    let _serial = gpu_serial();
    let gpu = gpu_or_skip!("panicking pipeline");
    let panicking = NamedPipeline::new("boom", |_: PassBuildContext<'_>| -> RenderGraph {
        panic!("shader `post.wgsl` failed to compile")
    });
    let mut renderer = gpu.renderer([320, 180], Some(panicking));
    match renderer.pipeline_status() {
        PipelineStatus::Invalid {
            id,
            reason,
            details,
        } => {
            assert_eq!(id, "boom");
            assert!(reason.contains("panicked"), "{reason}");
            assert!(
                details.iter().any(|d| d.contains("post.wgsl")),
                "{details:?}"
            );
        }
        other => panic!("expected Invalid, got {other:?}"),
    }
    assert!(gpu.frame(&mut renderer).text_pixels() > 100);
}

#[test]
fn pipelines_swap_at_the_frame_boundary_through_none_and_back() {
    let _serial = gpu_serial();
    let gpu = gpu_or_skip!("swap sequence");
    let (a, a_builds) = text_pipeline("a", "PIPELINE A");
    let mut renderer = gpu.renderer([640, 360], Some(a));
    assert_eq!(renderer.pipeline_status(), &PipelineStatus::Ready);
    assert_eq!(renderer.pipeline_id(), Some("a"));
    let frame = gpu.frame(&mut renderer);
    assert!(near(frame.corner(), PIPELINE_CLEAR));
    let a_text = frame.text_pixels();
    assert!(a_text > 50, "{a_text}");

    // A -> B: waits for the frame boundary.
    let (b, _) = text_pipeline("b", "A DIFFERENT PIPELINE B");
    renderer.set_pipeline(Some(b));
    assert_eq!(renderer.pipeline_id(), Some("a"), "not before the boundary");
    let frame = gpu.frame(&mut renderer);
    assert_eq!(renderer.pipeline_id(), Some("b"));
    assert_eq!(renderer.pipeline_status(), &PipelineStatus::Ready);
    assert_ne!(frame.text_pixels(), a_text, "B drew its own text");

    // B -> none.
    renderer.set_pipeline(None);
    let frame = gpu.frame(&mut renderer);
    assert_eq!(renderer.pipeline_status(), &PipelineStatus::None);
    assert_eq!(renderer.pipeline_id(), None);
    assert!(is_void(frame.corner()), "{:?}", frame.corner());
    assert!(frame.text_pixels() > 200);

    // none -> A again, a new instance built from scratch.
    let (a2, a2_builds) = text_pipeline("a", "PIPELINE A");
    renderer.set_pipeline(Some(a2));
    let frame = gpu.frame(&mut renderer);
    assert_eq!(renderer.pipeline_status(), &PipelineStatus::Ready);
    assert_eq!(frame.text_pixels(), a_text);
    assert!(near(frame.corner(), PIPELINE_CLEAR));
    assert_eq!(a_builds.load(Ordering::SeqCst), 1);
    assert_eq!(a2_builds.load(Ordering::SeqCst), 1);

    // A later call before the boundary replaces an earlier one.
    renderer.set_pipeline(None);
    let (c, c_builds) = text_pipeline("c", "C");
    renderer.set_pipeline(Some(c));
    gpu.frame(&mut renderer);
    assert_eq!(renderer.pipeline_id(), Some("c"));
    assert_eq!(c_builds.load(Ordering::SeqCst), 1);
}

#[test]
fn swapping_pipelines_does_not_grow_gpu_memory() {
    let _serial = gpu_serial();
    let gpu = gpu_or_skip!("swap leak");
    let mut renderer = gpu.renderer([640, 360], None);
    let cycle = |renderer: &mut Renderer| {
        let (a, _) = text_pipeline("a", "PIPELINE A");
        renderer.set_pipeline(Some(a));
        gpu.frame(renderer);
        let (b, _) = text_pipeline("b", "PIPELINE B");
        renderer.set_pipeline(Some(b));
        gpu.frame(renderer);
        renderer.set_pipeline(None);
        gpu.frame(renderer);
        renderer.set_pipeline(Some(NamedPipeline::new(
            "bad",
            FailingPipeline(Box::new(|| PipelineFailure::new("nope"))),
        )));
        gpu.frame(renderer);
    };
    let allocated = || {
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        gpu.device
            .generate_allocator_report()
            .map(|report| report.total_allocated_bytes)
    };
    // Warm up: lazily created pools and caches settle.
    for _ in 0..2 {
        cycle(&mut renderer);
    }
    let Some(settled) = allocated() else {
        eprintln!("GPU_ALLOCATOR_REPORT_UNAVAILABLE: swap leak");
        return;
    };
    for _ in 0..6 {
        cycle(&mut renderer);
    }
    let after = allocated().unwrap();
    assert!(
        after <= settled,
        "six more swap cycles grew device memory from {settled} to {after} bytes"
    );
}

#[test]
fn resize_keeps_working_in_every_state() {
    let _serial = gpu_serial();
    let gpu = gpu_or_skip!("resize");
    let mut renderer = gpu.renderer([640, 360], None);

    renderer.set_render_size(800, 450);
    let frame = gpu.frame(&mut renderer);
    assert_eq!((frame.width, frame.height), (800, 450));
    assert!(frame.text_pixels() > 200, "none, resized");

    renderer.set_pipeline(Some(NamedPipeline::new(
        "bad",
        FailingPipeline(Box::new(|| PipelineFailure::new("nope"))),
    )));
    renderer.set_render_size(512, 288);
    let frame = gpu.frame(&mut renderer);
    assert_eq!((frame.width, frame.height), (512, 288));
    assert!(frame.text_pixels() > 100, "invalid, resized");
    assert!(is_void(frame.corner()), "{:?}", frame.corner());

    let (a, _) = text_pipeline("a", "PIPELINE A");
    renderer.set_pipeline(Some(a));
    renderer.set_render_size(1024, 576);
    let frame = gpu.frame(&mut renderer);
    assert_eq!((frame.width, frame.height), (1024, 576));
    assert!(near(frame.corner(), PIPELINE_CLEAR));
    assert!(frame.text_pixels() > 50, "ready, resized");

    renderer.set_render_size(640, 360);
    let frame = gpu.frame(&mut renderer);
    assert!(near(frame.corner(), PIPELINE_CLEAR));
    assert!(frame.text_pixels() > 50, "ready, resized back");
}

/// A failed pipeline that can be built again becomes the pipeline when it
/// does, and a pipeline that stops building shows why (no stand-in graph).
#[test]
fn a_rebuild_retries_a_failure_and_reports_a_new_one() {
    let _serial = gpu_serial();
    let gpu = gpu_or_skip!("rebuild");
    // Fails until `fixed` is set; then the rebuilder yields a graph; set it back
    // and the same rebuilder fails again.
    let fixed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let rebuilder_fixed = Arc::clone(&fixed);
    let rebuilder: GraphRebuilder = Arc::new(move |device, queue, config, _, _, _, _| {
        if !rebuilder_fixed.load(Ordering::SeqCst) {
            return Err(PipelineFailure::new("the plugin is still disabled"));
        }
        let builds = Arc::new(AtomicUsize::new(0));
        Box::new(TextPipeline {
            text: "BACK",
            builds,
        })
        .into_graph(PassBuildContext {
            device,
            queue,
            config: config.clone(),
            debug_state: Default::default(),
            camera_buffer: &queue_scratch(device),
            debug_camera_buffer: &queue_scratch(device),
            cull_stats_buffer: &queue_scratch(device),
            owns_device: false,
            scene_db: scene_db_for(device, queue),
        })
    });
    let start = Arc::clone(&rebuilder);
    let pipeline = NamedPipeline::new(
        "needs-plugin",
        FailingPipeline(Box::new(move || {
            PipelineFailure::new("the plugin is disabled").with_rebuilder(Arc::clone(&start))
        })),
    );
    let mut renderer = gpu.renderer([320, 180], Some(pipeline));
    assert!(matches!(
        renderer.pipeline_status(),
        PipelineStatus::Invalid { reason, .. } if reason == "the plugin is disabled"
    ));

    // Still failing: a different reason replaces the old one.
    renderer.request_graph_rebuild();
    gpu.frame(&mut renderer);
    assert!(matches!(
        renderer.pipeline_status(),
        PipelineStatus::Invalid { reason, .. } if reason == "the plugin is still disabled"
    ));

    // Fixed: the rebuild brings the pipeline.
    fixed.store(true, Ordering::SeqCst);
    renderer.request_graph_rebuild();
    let frame = gpu.frame(&mut renderer);
    assert_eq!(renderer.pipeline_status(), &PipelineStatus::Ready);
    assert_eq!(renderer.pipeline_id(), Some("needs-plugin"));
    assert!(near(frame.corner(), PIPELINE_CLEAR));

    // Broken again: invalid, with no graph kept in its place.
    fixed.store(false, Ordering::SeqCst);
    renderer.request_graph_rebuild();
    let frame = gpu.frame(&mut renderer);
    assert!(matches!(
        renderer.pipeline_status(),
        PipelineStatus::Invalid { reason, .. } if reason == "the plugin is still disabled"
    ));
    assert!(is_void(frame.corner()), "{:?}", frame.corner());
}

#[test]
fn a_host_can_report_a_pipeline_it_could_not_make() {
    let _serial = gpu_serial();
    let gpu = gpu_or_skip!("fail_pipeline");
    let mut renderer = gpu.renderer([320, 180], None);
    renderer.fail_pipeline(
        "missing.rendergraph",
        PipelineFailure::new("no such pipeline is registered")
            .with_details(["registered: helio.deferred, helio.forward".to_owned()]),
    );
    let frame = gpu.frame(&mut renderer);
    assert!(matches!(
        renderer.pipeline_status(),
        PipelineStatus::Invalid { id, .. } if id == "missing.rendergraph"
    ));
    assert!(frame.text_pixels() > 100);
}

// The rebuilder above runs on a bare device and queue; these make what a build
// context needs.
fn queue_scratch(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scratch"),
        size: 256,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::UNIFORM
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn scene_db_for(device: &Arc<wgpu::Device>, queue: &Arc<wgpu::Queue>) -> GpuMirrorHandle {
    Gpu {
        device: Arc::clone(device),
        queue: Arc::clone(queue),
    }
    .mirror()
}
