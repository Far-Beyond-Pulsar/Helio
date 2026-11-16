//! A pass renders secondary views (`helio_core::view`) through the graph's
//! view source: each view graph runs with the request's camera and the
//! requester's resources, inline views nest and compose their maps and slots
//! in the frame's view table, and the frame's budget bounds how deep.

mod support;

use std::sync::{Arc, Mutex};

use glam::{Mat4, Vec3, Vec4};
use helio_core::view::{VIEW_CLIPPED, VIEW_MIRRORED, VIEW_REMAPS_RESOURCE};
use helio_core::{
    GpuCameraUniforms, PassContext, RenderGraph, RenderPass, ResourceKey, Result as HelioResult,
    ViewBudget, ViewBuild, ViewContext, ViewDescription, ViewGraphSource, ViewKind, ViewRequest,
};

/// What the requester's graph published before its views ran.
const PARENT_VALUE: ResourceKey<u32> = ResourceKey::new("views_test_parent_value");

#[derive(Clone, Debug)]
struct Seen {
    built_for: ViewDescription,
    camera: GpuCameraUniforms,
    inherited: Option<u32>,
    table: bool,
}

type Log = Arc<Mutex<Vec<Seen>>>;

fn target(
    device: &wgpu::Device,
) -> (
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Texture,
    wgpu::TextureView,
) {
    let make = |format, usage| {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("view test target"),
            size: wgpu::Extent3d {
                width: 32,
                height: 32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    };
    let (color, color_view) = make(
        wgpu::TextureFormat::Rgba16Float,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
    );
    let (depth, depth_view) = make(
        wgpu::TextureFormat::Depth32Float,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
    );
    (color, color_view, depth, depth_view)
}

/// The pass a view graph runs: it records what it saw and, in an inline
/// view, asks for a view inside it (a portal seen through a portal).
struct Probe {
    built_for: ViewDescription,
    log: Log,
    targets: (
        wgpu::Texture,
        wgpu::TextureView,
        wgpu::Texture,
        wgpu::TextureView,
    ),
}

impl RenderPass for Probe {
    fn name(&self) -> &'static str {
        "ViewProbe"
    }

    fn renders_views(&self) -> bool {
        self.built_for.kind == ViewKind::Inline
    }

    fn render_views(&mut self, views: &mut ViewContext<'_>) -> HelioResult<()> {
        let map = Mat4::from_translation(Vec3::new(0.0, 0.0, -10.0));
        let request = ViewRequest {
            key: 7,
            kind: ViewKind::Inline,
            mirrored: false,
            camera: views.camera_data.through(map, map),
            content_to_drawn: map,
            width: 32,
            height: 32,
            target: views.target,
            depth: &self.targets.3,
            depth_texture: &self.targets.2,
            imports: &[],
            seeds: &[],
        };
        views.render_view(&request)?;
        Ok(())
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a helio_core::ResourceRegistry<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        self.log.lock().unwrap().push(Seen {
            built_for: self.built_for,
            camera: *ctx.camera_data,
            inherited: ctx.registry.get(PARENT_VALUE),
            table: ctx.registry.contains(VIEW_REMAPS_RESOURCE),
        });
        Ok(())
    }
}

struct Source {
    log: Log,
}

impl ViewGraphSource for Source {
    fn build(&self, build: &ViewBuild<'_>) -> Result<RenderGraph, String> {
        let mut graph = RenderGraph::new_with_external_device(build.device, build.queue);
        graph.add_pass(Box::new(Probe {
            built_for: build.description,
            log: Arc::clone(&self.log),
            targets: target(build.device),
        }));
        Ok(graph)
    }
}

/// The frame's own pass: a mirrored capture and a portal's inline view.
struct Requester {
    rendered: Arc<Mutex<Vec<bool>>>,
    capture: (
        wgpu::Texture,
        wgpu::TextureView,
        wgpu::Texture,
        wgpu::TextureView,
    ),
    inline: (
        wgpu::Texture,
        wgpu::TextureView,
        wgpu::Texture,
        wgpu::TextureView,
    ),
}

impl RenderPass for Requester {
    fn name(&self) -> &'static str {
        "ViewRequester"
    }

    fn renders_views(&self) -> bool {
        true
    }

    fn render_views(&mut self, views: &mut ViewContext<'_>) -> HelioResult<()> {
        let capture = ViewRequest {
            key: 1,
            kind: ViewKind::Capture,
            mirrored: true,
            camera: views
                .camera_data
                .with_clip_plane(Vec4::new(0.0, 1.0, 0.0, 0.0)),
            content_to_drawn: Mat4::IDENTITY,
            width: 32,
            height: 32,
            target: &self.capture.1,
            depth: &self.capture.3,
            depth_texture: &self.capture.2,
            imports: &[],
            seeds: &[],
        };
        let map = Mat4::from_translation(Vec3::new(100.0, 0.0, 0.0));
        let inline = ViewRequest {
            key: 2,
            kind: ViewKind::Inline,
            mirrored: false,
            camera: views
                .camera_data
                .through(map, map)
                .with_clip_plane(Vec4::new(1.0, 0.0, 0.0, 0.0)),
            content_to_drawn: map,
            width: 32,
            height: 32,
            target: &self.inline.1,
            depth: &self.inline.3,
            depth_texture: &self.inline.2,
            imports: &[],
            seeds: &[],
        };
        let mut rendered = self.rendered.lock().unwrap();
        rendered.push(views.render_view(&capture)?);
        rendered.push(views.render_view(&inline)?);
        Ok(())
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a helio_core::ResourceRegistry<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn execute(&mut self, _ctx: &mut PassContext) -> HelioResult<()> {
        Ok(())
    }
}

#[test]
fn views_run_their_graphs_nest_and_stop_at_the_budget() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
        let Some(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok()
        else {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: views");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("a device");
        let (device, queue) = (Arc::new(device), Arc::new(queue));
        let log: Log = Arc::default();
        let rendered = Arc::new(Mutex::new(Vec::new()));

        let mut graph = RenderGraph::new_with_external_device(&device, &queue);
        graph.set_view_source(Arc::new(Source {
            log: Arc::clone(&log),
        }));
        graph.set_view_budget(ViewBudget {
            max_depth: 2,
            max_views: 8,
        });
        graph.add_pass(Box::new(Requester {
            rendered: Arc::clone(&rendered),
            capture: target(&device),
            inline: target(&device),
        }));
        graph.lock(32, 32);

        let (_, frame_target, _, frame_depth) = target(&device);
        for frame in 0..2u64 {
            let mut scene =
                support::SceneInputAdapter::new(Arc::clone(&device), Arc::clone(&queue));
            scene.frame_count = frame;
            log.lock().unwrap().clear();
            rendered.lock().unwrap().clear();
            let mut registry = helio_core::ResourceRegistry::empty();
            registry.write(PARENT_VALUE, 42, "test");
            graph
                .execute_with_registry(&scene, &frame_target, &frame_depth, &mut registry)
                .expect("the frame renders");
            let _ = device.poll(wgpu::PollType::wait_indefinitely());

            assert_eq!(*rendered.lock().unwrap(), [true, true], "frame {frame}");
            let seen = log.lock().unwrap().clone();
            // The capture, the inline view, and the view inside the inline
            // view (depth 2); the depth-3 view the last one asks for is past
            // the budget and does not run.
            assert_eq!(seen.len(), 3, "frame {frame}: {seen:#?}");
            for view in &seen {
                assert_eq!(
                    view.inherited,
                    Some(42),
                    "a view reads its requester's resources"
                );
                assert!(view.table, "the frame's view table is published");
            }
            let capture = seen
                .iter()
                .find(|v| v.built_for.kind == ViewKind::Capture)
                .expect("the capture ran");
            assert_eq!(capture.built_for.kind, ViewKind::Capture);
            assert!(capture.built_for.mirrored && capture.built_for.clipped);
            assert_eq!(
                capture.camera.view_info,
                [0, VIEW_MIRRORED | VIEW_CLIPPED, 1, 0],
                "a capture owns its G-buffer: no inline slot"
            );
            // The view inside the inline view runs before the inline view's
            // own draw (its pass renders views first).
            let at = |depth: u32| {
                seen.iter()
                    .find(|v| v.built_for.kind == ViewKind::Inline && v.built_for.depth == depth)
                    .expect("the inline view at that depth ran")
            };
            let (outer, inner) = (at(1), at(2));
            assert_eq!(outer.built_for.depth, 1);
            assert_eq!(inner.built_for.depth, 2);
            // Each inline view has its slot in the frame's table, and the
            // nested one draws through both maps.
            assert_ne!(outer.camera.view_info[0], 0);
            assert_ne!(inner.camera.view_info[0], 0);
            assert_ne!(outer.camera.view_info[0], inner.camera.view_info[0]);
            let composed = Mat4::from_translation(Vec3::new(100.0, 0.0, -10.0));
            let drawn = Mat4::from_cols_array(&inner.camera.content_to_drawn);
            assert!(drawn.abs_diff_eq(composed, 1e-4), "{drawn} vs {composed}");
        }
    });
}
