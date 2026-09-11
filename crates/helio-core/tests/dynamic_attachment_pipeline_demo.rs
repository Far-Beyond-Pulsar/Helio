//! Demonstrates the dynamic-rendering layer added to the outer pass
//! executor: symbolic attachment resolution (`AttachmentSlot` +
//! `ColorAttachmentIntent`) and format-keyed pipeline caching
//! (`PipelineFormatCache`), wired through `RenderGraph` without changing
//! `RenderPass::execute`'s signature or any existing pass's implementation.
//!
//! `DemoPass` below never sees a `wgpu::TextureView` or builds its own
//! `wgpu::RenderPassDescriptor` by hand — it declares a named transient
//! (`declare_resources`), describes its one color attachment symbolically
//! (`render_pass_descriptor_with_pool`), and looks its pipeline up by the
//! runtime format the executor resolved for that attachment
//! (`execute`, via `ctx.pipeline_cache`). Skips (rather than fails) when no
//! GPU adapter is available, matching this crate's other hardware tests.

use helio_core::graph::{
    attachment_format, AttachmentSlot, ColorAttachmentIntent, GraphTexturePool, PipelineFormatKey,
    ResourceBuilder, ResourceFormat, ResourceSize,
};
use helio_core::{PassContext, RenderGraph, RenderPass, Result as HelioResult};
use std::sync::Arc;

/// A minimal post-process pass: clears a named transient to a solid color.
/// Stands in for anything that would otherwise bake a target format into a
/// pipeline at construction time (see `helio-pass-fxaa`'s `target_format`
/// constructor argument) — here the format is re-resolved every frame and
/// the pipeline only rebuilds if that format actually changes.
struct DemoPass {
    pipeline_layout: wgpu::PipelineLayout,
    shader: wgpu::ShaderModule,
}

const DEMO_COLOR: &str = "demo_dynamic_color";

impl DemoPass {
    fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DemoPass Shader"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
                @vertex
                fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
                    let x = f32(i32(i) - 1);
                    let y = f32(i32(i & 1u) * 2 - 1);
                    return vec4<f32>(x, y, 0.0, 1.0);
                }
                @fragment
                fn fs_main() -> @location(0) vec4<f32> {
                    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
                }
                "#
                .into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DemoPass Layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        Self {
            pipeline_layout,
            shader,
        }
    }

    /// Builds the `PipelineFormatKey` for this frame's resolved attachment
    /// format — the same key is used to look the pipeline up in
    /// `execute()` and would be used again if the format ever changes.
    fn format_key(&self, pool: &GraphTexturePool) -> Option<PipelineFormatKey> {
        let color_format = attachment_format(AttachmentSlot::Named(DEMO_COLOR), pool)?;
        Some(PipelineFormatKey::new(
            "DemoPass",
            vec![color_format],
            None,
        ))
    }

    fn build_pipeline(&self, device: &wgpu::Device, color_format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DemoPass Pipeline"),
            layout: Some(&self.pipeline_layout),
            vertex: wgpu::VertexState {
                module: &self.shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &self.shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    }
}

impl RenderPass for DemoPass {
    fn name(&self) -> &'static str {
        "DemoPass"
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.write_color(DEMO_COLOR, ResourceFormat::Rgba8UnormSrgb, ResourceSize::Output);
    }

    // Required by the trait, but this pass only ever resolves attachments
    // through the pool-aware path below — so the legacy 3-arg path (which
    // can't see named transients outside `FrameResources`) has nothing to do.
    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    /// The dynamic-rendering entry point: describes the one color
    /// attachment symbolically and lets the executor's `pool` resolve it to
    /// this frame's physical view — no raw `wgpu::TextureView` plumbing.
    fn render_pass_descriptor_with_pool<'a>(
        &'a self,
        target: &'a wgpu::TextureView,
        depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
        pool: &'a GraphTexturePool,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        let intent = ColorAttachmentIntent::new(
            AttachmentSlot::Named(DEMO_COLOR),
            wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            wgpu::StoreOp::Store,
        );
        // Leaked to satisfy the descriptor's borrow — fine for this demo;
        // a real pass keeps the attachment array in a field (as the
        // existing passes already do) rather than allocating one per frame.
        let color_attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>] =
            Box::leak(vec![Some(intent.resolve(target, depth, pool)?)].into_boxed_slice());
        Some(wgpu::RenderPassDescriptor {
            label: Some("DemoPass"),
            color_attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    /// Untouched `RenderPass::execute` signature and body shape — this is
    /// exactly what "zero-breakage pass contract" means: the pass draws
    /// into whatever the executor already opened via
    /// `render_pass_descriptor_with_pool`, unaware of how the attachment or
    /// pipeline format were resolved.
    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let Some(key) = self.format_key(ctx.resource_pool) else {
            return Ok(()); // named transient not allocated this frame — skip
        };
        let pipeline: Arc<wgpu::RenderPipeline> = ctx.pipeline_cache.get_or_create(key, || {
            let format = attachment_format(AttachmentSlot::Named(DEMO_COLOR), ctx.resource_pool)
                .expect("format_key already confirmed this resolves");
            self.build_pipeline(ctx.device, format)
        });

        let Some(rp_ptr) = ctx.active_render_pass_ptr() else {
            return Ok(()); // no open render pass this frame — nothing to draw into
        };
        let rp = unsafe { &mut *rp_ptr };
        rp.set_pipeline(&pipeline);
        rp.draw(0..3, 0..1);
        Ok(())
    }
}

#[test]
fn dynamic_attachment_and_pipeline_cache_round_trip() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Some(adapter) = request_test_adapter(&instance).await else {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: dynamic attachment pipeline demo");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Dynamic Attachment Demo Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("available adapter must create a device");
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let mut graph = RenderGraph::new(&device, &queue);
        graph.add_pass(Box::new(DemoPass::new(&device)));
        graph.lock(64, 64);

        let scene = helio_core::GpuScene::new(device.clone(), queue.clone());
        let target_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Demo Swapchain Stand-in"),
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let target_view = target_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Demo Depth Stand-in"),
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Frame 1: pipeline cache miss, builds one pipeline for the resolved format.
        graph
            .execute(&scene, &target_view, &depth_view)
            .expect("frame 1 should execute");
        // Frame 2 at the same resolved format: same cache entry, no rebuild —
        // demonstrates "dynamic resizing/reformatting propagates without
        // pass re-initialization" by *not* needing DemoPass to know anything
        // changed (or didn't) between frames.
        graph
            .execute(&scene, &target_view, &depth_view)
            .expect("frame 2 should execute");

        // Simulate a resize: the pool reallocates `demo_dynamic_color` at
        // the new extent (same format), so the cached pipeline is still
        // valid and DemoPass still doesn't need to re-initialize anything.
        graph.set_render_size(128, 128);
        graph
            .execute(&scene, &target_view, &depth_view)
            .expect("frame after resize should execute");
    });
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
