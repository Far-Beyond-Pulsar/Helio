use helio_core::graph::{ResourceBuilder, ResourceFormat, ResourceSize};
use helio_core::{
    GpuScene, PassContext, PrepareContext, RenderGraph, RenderPass, Result as HelioResult,
};
use std::sync::{Arc, Mutex};

struct ResizeProbePass {
    observations: Arc<Mutex<Vec<(bool, u32, u32)>>>,
}

struct InternalAttachmentPass;

impl RenderPass for InternalAttachmentPass {
    fn name(&self) -> &'static str {
        "InternalAttachment"
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.write_color(
            "pre_aa",
            ResourceFormat::Rgba8Unorm,
            ResourceSize::MatchSurface,
        );
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        depth: &'a wgpu::TextureView,
        resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        let pre_aa = resources
            .pre_aa
            .read(self.name())
            .expect("graph must route its resized internal color attachment");
        let color_attachments = Box::leak(Box::new([Some(wgpu::RenderPassColorAttachment {
            view: pre_aa,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })]));
        Some(wgpu::RenderPassDescriptor {
            label: Some("Internal Attachment Resize Contract"),
            color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn execute(&mut self, _ctx: &mut PassContext) -> HelioResult<()> {
        Ok(())
    }
}

impl RenderPass for ResizeProbePass {
    fn name(&self) -> &'static str {
        "ResizeProbe"
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        self.observations
            .lock()
            .expect("resize observations must not be poisoned")
            .push((ctx.resize, ctx.width, ctx.height));
        Ok(())
    }

    fn execute(&mut self, _ctx: &mut PassContext) -> HelioResult<()> {
        Ok(())
    }
}

#[test]
fn prepare_receives_one_resize_pulse_after_graph_resize() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Some(adapter) = request_test_adapter(&instance).await else {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: render graph resize contract");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Render Graph Resize Contract Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("available adapter must create a resize-contract device");
        device.on_uncaptured_error(Arc::new(|error| {
            panic!("render graph resize validation error: {error:?}");
        }));

        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let scene = GpuScene::new(Arc::clone(&device), Arc::clone(&queue));
        let observations = Arc::new(Mutex::new(Vec::new()));
        let mut graph = RenderGraph::new(&device, &queue);
        graph.add_pass(Box::new(ResizeProbePass {
            observations: Arc::clone(&observations),
        }));
        graph.lock(32, 24);

        let (target, depth) = frame_views(&device, 32, 24);
        graph
            .execute(&scene, &target, &depth)
            .expect("initial graph frame must execute");

        graph.set_render_size(64, 48);
        let (target, depth) = frame_views(&device, 64, 48);
        graph
            .execute(&scene, &target, &depth)
            .expect("first resized graph frame must execute");
        graph
            .execute(&scene, &target, &depth)
            .expect("steady graph frame must execute");

        assert_eq!(
            observations
                .lock()
                .expect("resize observations must not be poisoned")
                .as_slice(),
            &[(false, 32, 24), (true, 64, 48), (false, 64, 48)]
        );
    });
}

#[test]
fn graph_owned_internal_attachment_matches_depth_after_resize() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Some(adapter) = request_test_adapter(&instance).await else {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: render graph attachment resize contract");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Render Graph Attachment Resize Contract Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("available adapter must create an attachment resize-contract device");
        device.on_uncaptured_error(Arc::new(|error| {
            panic!("render graph attachment resize validation error: {error:?}");
        }));

        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let scene = GpuScene::new(Arc::clone(&device), Arc::clone(&queue));
        let mut graph = RenderGraph::new(&device, &queue);
        graph.add_pass(Box::new(InternalAttachmentPass));
        graph.lock(32, 24);

        let (target, depth) = frame_views(&device, 32, 24);
        graph
            .execute(&scene, &target, &depth)
            .expect("initial graph attachment frame must execute");

        graph.set_render_size(64, 48);
        let (target, depth) = frame_views(&device, 64, 48);
        graph
            .execute(&scene, &target, &depth)
            .expect("resized graph attachments must have matching extents");
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("attachment resize validation work must complete");
    });
}

fn frame_views(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::TextureView, wgpu::TextureView) {
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Resize Contract Target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Resize Contract Depth"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    (
        target.create_view(&wgpu::TextureViewDescriptor::default()),
        depth.create_view(&wgpu::TextureViewDescriptor::default()),
    )
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
