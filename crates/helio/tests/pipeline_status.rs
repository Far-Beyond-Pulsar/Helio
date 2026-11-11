//! A renderer needs no pipeline, swaps pipelines at a frame boundary, and
//! shows why when it has none or the one it has does not build. The checks
//! read the output frame back (in tests only): the status screen is drawn text,
//! so its pixels exist, and no other graph stands in for a failed pipeline.
//!
//! Needs a GPU device; without one the tests skip.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use glam::Vec3;
use helio::{
    Camera, GraphRebuilder, IntoGraph, NamedPipeline, PassBuildContext, PipelineFailure,
    PipelineStatus, Renderer, RendererBuilder, RendererConfig,
};
use helio_core::RenderGraph;
use helio_pass_debug_overlay::{DebugOverlayPass, DebugOverlayState};
use pulsar_scenedb::gpu::{EngineGpuContext, GpuMirrorHandle, SceneGpuConfig, SceneGpuStore};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// The renderer's own clear colour (what a rendering pipeline's frame starts as).
const PIPELINE_CLEAR: [f32; 4] = [0.0, 0.2, 0.0, 1.0];

/// One test at a time may hold a device (see Pulsar-Native#1647).
static GPU_DEVICE: Mutex<()> = Mutex::new(());

fn gpu_serial() -> std::sync::MutexGuard<'static, ()> {
    GPU_DEVICE.lock().unwrap_or_else(|e| e.into_inner())
}

struct Gpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

fn gpu() -> Option<Gpu> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
        let adapter = instance.request_adapter(&Default::default()).await.ok()?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: helio::required_wgpu_features(adapter.features()),
                required_limits: helio::required_wgpu_limits(adapter.limits()),
                experimental_features: helio::required_experimental_features(adapter.features()),
                ..Default::default()
            })
            .await
            .ok()?;
        Some(Gpu {
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    })
}

impl Gpu {
    fn mirror(&self) -> GpuMirrorHandle {
        let context = EngineGpuContext::new(Arc::clone(&self.device), Arc::clone(&self.queue));
        let store = SceneGpuStore::new(
            &context,
            SceneGpuConfig {
                classes: Vec::new(),
                tombstone_headroom: 0,
                max_cells_metadata: 0,
            },
        );
        GpuMirrorHandle::new(Arc::new(store), Arc::clone(&self.queue))
    }

    /// A renderer with the pipeline the builder is given (or none).
    fn renderer(&self, size: [u32; 2], pipeline: Option<NamedPipeline>) -> Renderer {
        let mut builder =
            RendererBuilder::new(RendererConfig::new(size[0], size[1], FORMAT), self.mirror())
                .with_external_device()
                .with_clear_color(PIPELINE_CLEAR);
        if let Some(NamedPipeline { id, pipeline }) = pipeline {
            builder = builder.with_pipeline(id, BoxedPipeline(pipeline));
        }
        builder.build(
            Arc::clone(&self.device),
            Arc::clone(&self.queue),
            size[0],
            size[1],
            FORMAT,
        )
    }

    /// Renders one frame and reads it back as RGBA8.
    fn frame(&self, renderer: &mut Renderer) -> Frame {
        let (width, height) = (renderer.output_width(), renderer.output_height());
        let target = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());
        let camera = Camera::perspective_look_at(
            Vec3::new(0.0, 2.0, 5.0),
            Vec3::ZERO,
            Vec3::Y,
            1.0,
            width as f32 / height as f32,
            0.1,
            100.0,
        );
        renderer.render(&camera, &view).expect("the frame renders");

        let row = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test readback"),
            size: u64::from(row * height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.unwrap());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let mapped = readback.slice(..).get_mapped_range().unwrap();
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            let start = (y * row) as usize;
            pixels.extend_from_slice(&mapped[start..start + (width * 4) as usize]);
        }
        drop(mapped);
        readback.unmap();
        Frame {
            width,
            height,
            pixels,
        }
    }
}

struct Frame {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl Frame {
    fn at(&self, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * self.width + x) * 4) as usize;
        self.pixels[i..i + 4].try_into().unwrap()
    }

    /// Pixels the text drew: bright, in any channel.
    fn text_pixels(&self) -> usize {
        self.pixels
            .chunks_exact(4)
            .filter(|p| p[0] > 150 && p[1] > 150 && p[2] > 150)
            .count()
    }

    fn corner(&self) -> [u8; 4] {
        self.at(self.width - 1, self.height - 1)
    }
}

fn near(pixel: [u8; 4], color: [f32; 4]) -> bool {
    (0..3).all(|c| (i32::from(pixel[c]) - (color[c] * 255.0).round() as i32).abs() <= 2)
}

/// `Box<dyn IntoGraph>` as an `IntoGraph` for `with_pipeline`.
struct BoxedPipeline(Box<dyn IntoGraph>);

impl IntoGraph for BoxedPipeline {
    fn into_graph(
        self: Box<Self>,
        ctx: PassBuildContext<'_>,
    ) -> Result<RenderGraph, PipelineFailure> {
        self.0.into_graph(ctx)
    }
}

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

fn status_clear(status: &PipelineStatus) -> [f32; 4] {
    match status {
        PipelineStatus::None => [0.03, 0.04, 0.08, 1.0],
        PipelineStatus::Invalid { .. } => [0.10, 0.03, 0.03, 1.0],
        PipelineStatus::Ready => PIPELINE_CLEAR,
    }
}

macro_rules! gpu_or_skip {
    ($name:literal) => {{
        match gpu() {
            Some(gpu) => gpu,
            None => {
                eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: {}", $name);
                return;
            }
        }
    }};
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
    // The frame is the status screen's colour, not the pipeline's clear colour.
    assert!(near(frame.corner(), status_clear(&PipelineStatus::None)));
}

#[test]
fn the_hint_the_host_gives_is_on_the_screen_text() {
    let _serial = gpu_serial();
    let gpu = gpu_or_skip!("status hint");
    let mut renderer = gpu.renderer([640, 360], None);
    let before = gpu.frame(&mut renderer).text_pixels();
    renderer
        .set_status_hint("Choose one under World Settings, Render Pipeline, in the level editor.");
    let after = gpu.frame(&mut renderer).text_pixels();
    assert_ne!(before, after, "a screen already showing is drawn again");
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
    assert!(near(
        frame.corner(),
        status_clear(renderer.pipeline_status())
    ));
    // The accent bar is the error colour.
    let accent = frame.at(2, 200);
    assert!(accent[0] > 200 && accent[1] < 100, "accent {accent:?}");
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
    assert!(near(frame.corner(), status_clear(&PipelineStatus::None)));
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
    assert!(near(
        frame.corner(),
        status_clear(renderer.pipeline_status())
    ));

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
    assert!(near(
        frame.corner(),
        status_clear(renderer.pipeline_status())
    ));
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
