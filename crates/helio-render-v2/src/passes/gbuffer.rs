//! G-buffer write pass.
//!
//! Renders scene geometry into four Rgba G-buffer textures + the shared depth
//! buffer.  No lighting is evaluated here — that happens in the subsequent
//! `DeferredLightingPass`.
//!
//! Records directly into the primary `CommandEncoder` using the sort order
//! computed by `DepthPrepassPass` (shared via `shared_sorted`).  No bundles.
//!
//! G-buffer formats:
//!   albedo   Rgba8Unorm
//!   normals  Rgba16Float
//!   ORM      Rgba8Unorm
//!   emissive Rgba16Float
//!   depth    Depth32Float (loaded from depth-prepass, not cleared)

use std::sync::{Arc, Mutex};
use crate::graph::{RenderPass, PassContext, PassResourceBuilder, ResourceHandle};
use crate::mesh::DrawCall;
use crate::passes::depth_prepass::record_opaque_draws;
use crate::Result;

/// The four G-buffer render targets.  Wrapped in `Arc<Mutex<…>>` so the Renderer
/// can swap in new views when the window is resized without rebuilding the graph.
pub struct GBufferTargets {
    pub albedo_view:   wgpu::TextureView,   // Rgba8Unorm
    pub normal_view:   wgpu::TextureView,   // Rgba16Float
    pub orm_view:      wgpu::TextureView,   // Rgba8Unorm
    pub emissive_view: wgpu::TextureView,   // Rgba16Float
}

pub struct GBufferPass {
    targets:       Arc<Mutex<GBufferTargets>>,
    pipeline:      Arc<wgpu::RenderPipeline>,
    draw_list:     Arc<Mutex<Vec<DrawCall>>>,
    /// Pre-computed sort shared from `DepthPrepassPass` (same material order).
    shared_sorted: Arc<Mutex<Vec<usize>>>,
}

impl GBufferPass {
    pub fn new(
        targets:       Arc<Mutex<GBufferTargets>>,
        pipeline:      Arc<wgpu::RenderPipeline>,
        draw_list:     Arc<Mutex<Vec<DrawCall>>>,
        shared_sorted: Arc<Mutex<Vec<usize>>>,
    ) -> Self {
        Self { targets, pipeline, draw_list, shared_sorted }
    }
}

impl RenderPass for GBufferPass {
    fn name(&self) -> &str { "gbuffer" }

    fn declare_resources(&self, builder: &mut PassResourceBuilder) {
        builder.read(ResourceHandle::named("shadow_atlas"));
        builder.read(ResourceHandle::named("rc_cascade0"));
        builder.read(ResourceHandle::named("sky_layer"));
        builder.write(ResourceHandle::named("gbuffer"));
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let draw_calls = self.draw_list.lock().unwrap();
        let sorted     = self.shared_sorted.lock().unwrap();
        let targets    = self.targets.lock().unwrap();

        let color_attachments = [
            Some(wgpu::RenderPassColorAttachment {
                view: &targets.albedo_view,
                resolve_target: None, depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: &targets.normal_view,
                resolve_target: None, depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: &targets.orm_view,
                resolve_target: None, depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: &targets.emissive_view,
                resolve_target: None, depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            }),
        ];

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("G-Buffer Write Pass"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: ctx.depth_view,
                // Load depth written by DepthPrepassPass — GPU rejects occluded GBuffer
                // fragments before running the material shader.
                depth_ops: Some(wgpu::Operations {
                    load:  wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, ctx.global_bind_group, &[]);
        pass.set_bind_group(2, ctx.lighting_bind_group, &[]);

        {
            record_opaque_draws(&mut pass, &draw_calls, &sorted);
        }

        Ok(())
    }
}

