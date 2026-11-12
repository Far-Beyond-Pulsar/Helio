//! A graph asset that does not build is the pipeline's failure, shown on the
//! output and held in the renderer's status, with no other graph in its
//! place: a node whose plugin is disabled, a cycle, a node that fails to
//! build. A valid asset renders, and swaps with the failing ones.
//!
//! Needs a GPU device; without one the tests skip.

use std::sync::{Arc, Mutex};

use glam::Vec3;
use helio::{Camera, NamedPipeline, PipelineStatus, Renderer, RendererBuilder, RendererConfig};
use helio_feature_api::{NodeDescriptor, NodeRegistry, PinDesc};
use helio_graph_asset::core_nodes::{ON_FRAME_START, OUTPUT};
use helio_graph_asset::{
    core_registry, Connection, GraphAsset, GraphAssetPipeline, NodeInstance, PinRef,
};
use pulsar_scenedb::gpu::{EngineGpuContext, GpuMirrorHandle, SceneGpuConfig, SceneGpuStore};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
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
            builder = builder.with_pipeline(id, Boxed(pipeline));
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

struct Boxed(Box<dyn helio::IntoGraph>);

impl helio::IntoGraph for Boxed {
    fn into_graph(
        self: Box<Self>,
        ctx: helio::PassBuildContext<'_>,
    ) -> Result<helio_core::RenderGraph, helio::PipelineFailure> {
        self.0.into_graph(ctx)
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

fn node(id: &str, kind: &str) -> NodeInstance {
    NodeInstance {
        id: id.into(),
        kind: kind.into(),
        params: Default::default(),
        position: [0.0; 2],
    }
}

fn wire(from: &str, to: &str) -> Connection {
    Connection {
        from: PinRef::new(from, "out"),
        to: PinRef::new(to, "in"),
    }
}

/// On Frame Start, `middle` in a chain, the Output.
fn chain(name: &str, middle: &[(&str, &str)]) -> GraphAsset {
    let mut asset = GraphAsset {
        name: name.into(),
        ..Default::default()
    };
    asset.nodes.push(node("start", ON_FRAME_START));
    let mut previous = "start".to_owned();
    for (id, kind) in middle {
        asset.nodes.push(node(id, kind));
        asset.connections.push(wire(&previous, id));
        previous = (*id).to_owned();
    }
    asset.nodes.push(node("output", OUTPUT));
    asset.connections.push(wire(&previous, "output"));
    asset
}

fn flow(
    kind: &str,
    build: impl Fn() -> Result<(), String> + Send + Sync + 'static,
) -> NodeDescriptor {
    NodeDescriptor::new(kind, "Flow", "Test", "", move |_| build())
        .with_pins(vec![PinDesc::flow("in")], vec![PinDesc::flow("out")])
}

fn registry() -> NodeRegistry {
    let mut registry = core_registry();
    registry.register(flow("t.flow", || Ok(())));
    registry.register(flow("t.broken", || {
        Err("the pass constructor failed: shader `post.wgsl` has an error".into())
    }));
    // A plugin's node, absent because the plugin is disabled.
    registry.mark_absent("ghost.pass", "plugin ghost disabled");
    registry
}

fn pipeline(id: &str, asset: GraphAsset) -> NamedPipeline {
    NamedPipeline::new(
        id,
        GraphAssetPipeline::new(Arc::new(asset), Arc::new(registry())),
    )
}

fn invalid(renderer: &Renderer) -> (&str, &str, &[String]) {
    match renderer.pipeline_status() {
        PipelineStatus::Invalid {
            id,
            reason,
            details,
        } => (id, reason, details),
        other => panic!("expected an invalid pipeline, got {other:?}"),
    }
}

/// The status screen's field: near-black, with its vignette and grain.
fn is_void(pixel: [u8; 4]) -> bool {
    pixel[..3].iter().all(|&c| c <= 12)
}

#[test]
fn a_node_of_a_disabled_plugin_names_the_plugin() {
    let _serial = gpu_serial();
    let gpu = gpu_or_skip!("absent node");
    let asset = chain("post", &[("pass", "t.flow"), ("terrain", "ghost.pass")]);
    let mut renderer = gpu.renderer([640, 360], Some(pipeline("game/post.rendergraph", asset)));
    let (id, reason, details) = invalid(&renderer);
    assert_eq!(id, "game/post.rendergraph");
    assert!(reason.contains("`post`"), "{reason}");
    let details = details.join("\n");
    assert!(
        details.contains("missing node `ghost.pass` (plugin ghost disabled)"),
        "{details}"
    );
    let frame = gpu.frame(&mut renderer);
    assert!(frame.text_pixels() > 200);
    assert!(is_void(frame.corner()), "{:?}", frame.corner());
    // A graph diagnostic carries the severity the status screen marks.
    let (_, _, details) = invalid(&renderer);
    assert!(
        details.iter().any(|d| d.starts_with("error: ")),
        "{details:?}"
    );
}

#[test]
fn a_cycle_is_reported() {
    let _serial = gpu_serial();
    let gpu = gpu_or_skip!("cycle");
    let mut asset = chain("loop", &[("a", "t.flow"), ("b", "t.flow")]);
    asset.connections.push(wire("b", "a"));
    let mut renderer = gpu.renderer([640, 360], Some(pipeline("loop.rendergraph", asset)));
    let (_, _, details) = invalid(&renderer);
    assert!(details.iter().any(|d| d.contains("cycle")), "{details:?}");
    assert!(gpu.frame(&mut renderer).text_pixels() > 200);
}

#[test]
fn a_node_that_fails_to_build_reports_its_error() {
    let _serial = gpu_serial();
    let gpu = gpu_or_skip!("build error");
    let asset = chain("broken", &[("bad", "t.broken")]);
    let mut renderer = gpu.renderer([640, 360], Some(pipeline("broken.rendergraph", asset)));
    let (_, _, details) = invalid(&renderer);
    assert!(
        details
            .iter()
            .any(|d| d.contains("bad") && d.contains("post.wgsl` has an error")),
        "{details:?}"
    );
    assert!(gpu.frame(&mut renderer).text_pixels() > 200);
}

#[test]
fn an_asset_swaps_with_failing_ones_and_none_with_no_stand_in() {
    let _serial = gpu_serial();
    let gpu = gpu_or_skip!("asset swaps");
    let good = || pipeline("good.rendergraph", chain("good", &[("pass", "t.flow")]));
    let bad = || pipeline("bad.rendergraph", chain("bad", &[("bad", "t.broken")]));

    let mut renderer = gpu.renderer([640, 360], Some(good()));
    assert_eq!(renderer.pipeline_status(), &PipelineStatus::Ready);
    let frame = gpu.frame(&mut renderer);
    assert!(near(frame.corner(), PIPELINE_CLEAR));
    assert_eq!(frame.text_pixels(), 0, "a valid empty graph draws nothing");

    for next in [Some(bad()), None, Some(good()), Some(bad()), Some(good())] {
        let expected_ready = next.as_ref().is_some_and(|n| n.id == "good.rendergraph");
        renderer.set_pipeline(next);
        let frame = gpu.frame(&mut renderer);
        if expected_ready {
            assert_eq!(renderer.pipeline_status(), &PipelineStatus::Ready);
            assert!(near(frame.corner(), PIPELINE_CLEAR));
            assert_eq!(frame.text_pixels(), 0);
        } else {
            assert!(!renderer.pipeline_status().is_ready());
            assert!(frame.text_pixels() > 200, "the reason is on the frame");
            assert!(is_void(frame.corner()), "{:?}", frame.corner());
        }
    }
}
