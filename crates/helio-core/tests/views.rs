//! A pass renders captures (`helio_core::view`) through the graph's view
//! source: each view graph runs with the request's camera and the
//! requester's resources, a capture can ask for one inside it, and the
//! frame's budget bounds how deep.

mod support;

use std::sync::{Arc, Mutex};

use glam::Vec4;
use helio_core::view::{VIEW_CLIPPED, VIEW_MIRRORED};
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
}

type Log = Arc<Mutex<Vec<Seen>>>;

type Targets = (
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Texture,
    wgpu::TextureView,
);

fn target(device: &wgpu::Device) -> Targets {
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

fn capture<'r>(key: u64, mirrored: bool, camera: GpuCameraUniforms, targets: &'r Targets) -> ViewRequest<'r> {
    ViewRequest {
        key,
        mirrored,
        camera,
        width: 32,
        height: 32,
        target: &targets.1,
        depth: &targets.3,
        depth_texture: &targets.2,
        seeds: &[],
    }
}

/// The pass a view graph runs: it records what it saw and asks for a
/// capture inside it (a reflection seen in a reflection).
struct Probe {
    built_for: ViewDescription,
    log: Log,
    targets: Targets,
}

impl RenderPass for Probe {
    fn name(&self) -> &'static str {
        "ViewProbe"
    }

    fn renders_views(&self) -> bool {
        true
    }

    fn render_views(&mut self, views: &mut ViewContext<'_>) -> HelioResult<()> {
        let request = capture(7, false, *views.camera_data, &self.targets);
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

/// The frame's own pass: a mirrored, clipped capture.
struct Requester {
    rendered: Arc<Mutex<Vec<bool>>>,
    capture: Targets,
}

impl RenderPass for Requester {
    fn name(&self) -> &'static str {
        "ViewRequester"
    }

    fn renders_views(&self) -> bool {
        true
    }

    fn render_views(&mut self, views: &mut ViewContext<'_>) -> HelioResult<()> {
        let camera = views
            .camera_data
            .with_clip_plane(Vec4::new(0.0, 1.0, 0.0, 0.0));
        let request = capture(1, true, camera, &self.capture);
        let ran = views.render_view(&request)?;
        self.rendered.lock().unwrap().push(ran);
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
fn captures_run_their_graphs_nest_and_stop_at_the_budget() {
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

            assert_eq!(*rendered.lock().unwrap(), [true], "frame {frame}");
            let seen = log.lock().unwrap().clone();
            // The capture and the capture inside it (depth 2); the depth-3
            // capture the last one asks for is past the budget and does not
            // run.
            assert_eq!(seen.len(), 2, "frame {frame}: {seen:#?}");
            for view in &seen {
                assert_eq!(view.built_for.kind, ViewKind::Capture);
                assert_eq!(
                    view.inherited,
                    Some(42),
                    "a view reads its requester's resources"
                );
            }
            let at = |depth: u32| {
                seen.iter()
                    .find(|v| v.built_for.depth == depth)
                    .expect("the capture at that depth ran")
            };
            let (outer, inner) = (at(1), at(2));
            assert!(outer.built_for.mirrored && outer.built_for.clipped);
            assert_eq!(
                outer.camera.view_info,
                [0, VIEW_MIRRORED | VIEW_CLIPPED, 1, 0]
            );
            assert!(!inner.built_for.mirrored);
            assert_eq!(inner.camera.view_info[2], 2);
        }
    });
}
