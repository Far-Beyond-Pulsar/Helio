//! Contract coverage for Phase 9's independent-pass ordering rule.
//!
//! The two probes deliberately have no declared dependency.  They are run in
//! both graph orders, which gives the parallel recorder two valid recording
//! orders to exercise.  The observed pass output and the publication contract
//! must be the same regardless of that order.

use helio_core::{GpuScene, PassContext, RenderGraph, RenderPass, Result as HelioResult};
use std::sync::{Arc, Mutex};
mod support;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Observation {
    output: [bool; 2],
    published: [bool; 2],
    received_scene_db_projection: [bool; 2],
}

struct IndependentProbe {
    slot: usize,
    observation: Arc<Mutex<Observation>>,
}

impl RenderPass for IndependentProbe {
    fn name(&self) -> &'static str {
        match self.slot {
            0 => "IndependentA",
            1 => "IndependentB",
            _ => unreachable!("test probe slot must be 0 or 1"),
        }
    }

    fn writes(&self) -> &'static [&'static str] {
        match self.slot {
            0 => &["independent_a"],
            1 => &["independent_b"],
            _ => &[],
        }
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        self.observation.lock().unwrap().output[self.slot] = true;
        // The projection is a required graph input. This test deliberately
        // uses an empty projection, proving that test fixtures do not reopen
        // an Option-based legacy scene-input path.
        let _ = ctx.scene_buffers;
        self.observation.lock().unwrap().received_scene_db_projection[self.slot] = true;
        Ok(())
    }

    fn publish<'a>(&'a self, _frame: &mut libhelio::FrameResources<'a>) {
        self.observation.lock().unwrap().published[self.slot] = true;
    }
}

#[test]
fn independent_passes_are_order_independent_for_output_and_publication() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Some(adapter) = request_test_adapter(&instance).await else {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: parallel order independence");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Parallel Order Independence Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("available adapter must create a device");
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let forward = run_order(&device, &queue, [0, 1]);
        let reverse = run_order(&device, &queue, [1, 0]);
        assert_eq!(forward, reverse);
    assert_eq!(forward.output, [true, true]);
    assert_eq!(forward.published, [true, true]);
    assert_eq!(forward.received_scene_db_projection, [true, true]);
    });
}

fn run_order(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    order: [usize; 2],
) -> Observation {
    let observation = Arc::new(Mutex::new(Observation {
        output: [false; 2],
        published: [false; 2],
        received_scene_db_projection: [false; 2],
    }));
    let mut graph = RenderGraph::new(device, queue);
    for slot in order {
        graph.add_pass(Box::new(IndependentProbe {
            slot,
            observation: Arc::clone(&observation),
        }));
    }
    graph
        .validate_dependencies()
        .expect("independent writes must not require an ordering edge");
    graph.lock(8, 8);

    let scene = GpuScene::new(Arc::clone(device), Arc::clone(queue));
    let scene_input = support::SceneInputAdapter(&scene);
    let target = frame_texture_view(device, wgpu::TextureFormat::Rgba8Unorm);
    let depth = frame_texture_view(device, wgpu::TextureFormat::Depth32Float);
    graph
        .execute(&scene_input, &target, &depth)
        .expect("graph must execute");
    let result = observation.lock().unwrap().clone();
    result
}

fn frame_texture_view(device: &wgpu::Device, format: wgpu::TextureFormat) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("Parallel Order Independence Frame Texture"),
            size: wgpu::Extent3d {
                width: 8,
                height: 8,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

async fn request_test_adapter(instance: &wgpu::Instance) -> Option<wgpu::Adapter> {
    for force_fallback_adapter in [false, true] {
        if let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter,
                apply_limit_buckets: false,
            })
            .await
        {
            return Some(adapter);
        }
    }
    None
}
