use helio_core::graph::{ResourceBuilder, ResourceFormat, ResourceSize};
use helio_core::{GpuScene, PassContext, RenderGraph, RenderPass, Result as HelioResult};
use helio_pass_water_sim::WaterSimPass;
use std::sync::Arc;

struct PreAaProducer;

impl RenderPass for PreAaProducer {
    fn name(&self) -> &'static str {
        "PreAaProducer"
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
        _depth: &'a wgpu::TextureView,
        resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        color_only_descriptor(
            "Pre-AA Producer",
            resources
                .pre_aa
                .read(self.name())
                .expect("pre_aa is routed"),
        )
    }

    fn execute(&mut self, _ctx: &mut PassContext) -> HelioResult<()> {
        Ok(())
    }
}

struct PreAaDepthConsumer;

impl RenderPass for PreAaDepthConsumer {
    fn name(&self) -> &'static str {
        "PreAaDepthConsumer"
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read("pre_aa");
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
            .expect("pre_aa is published");
        let color_attachments = Box::leak(Box::new([Some(wgpu::RenderPassColorAttachment {
            view: pre_aa,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })]));
        Some(wgpu::RenderPassDescriptor {
            label: Some("Pre-AA Depth Consumer"),
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

#[test]
fn cached_water_output_is_reacquired_after_graph_resize() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Some(adapter) = request_test_adapter(&instance).await else {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: water resize contract");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Water Resize Contract Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("available adapter must create a water resize-contract device");
        device.on_uncaptured_error(Arc::new(|error| {
            panic!("water resize validation error: {error:?}");
        }));

        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let scene = GpuScene::new(Arc::clone(&device), Arc::clone(&queue));
        let camera = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Water Resize Contract Camera"),
            size: 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut graph = RenderGraph::new(&device, &queue);
        graph.add_pass(Box::new(PreAaProducer));
        graph.add_pass(Box::new(WaterSimPass::new(
            &device,
            &camera,
            32,
            24,
            wgpu::TextureFormat::Rgba8Unorm,
        )));
        graph.add_pass(Box::new(PreAaDepthConsumer));
        graph.lock(32, 24);

        let (target, depth) = frame_views(&device, 32, 24);
        graph
            .execute(&scene, &target, &depth)
            .expect("initial water graph frame must execute");

        graph.set_render_size(64, 48);
        let (target, depth) = frame_views(&device, 64, 48);
        graph
            .execute(&scene, &target, &depth)
            .expect("water output and resized depth must have matching extents");
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("water resize validation work must complete");
    });
}

fn color_only_descriptor<'a>(
    label: &'static str,
    view: &'a wgpu::TextureView,
) -> Option<wgpu::RenderPassDescriptor<'a>> {
    let color_attachments = Box::leak(Box::new([Some(wgpu::RenderPassColorAttachment {
        view,
        resolve_target: None,
        depth_slice: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            store: wgpu::StoreOp::Store,
        },
    })]));
    Some(wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments,
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

fn frame_views(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::TextureView, wgpu::TextureView) {
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Water Resize Contract Target"),
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
        label: Some("Water Resize Contract Depth"),
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
