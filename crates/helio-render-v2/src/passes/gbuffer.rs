//! G-buffer write pass.
//!
//! Renders scene geometry into four Rgba G-buffer textures + the shared depth
//! buffer.  No lighting is evaluated here — that happens in the subsequent
//! DeferredLightingPass.

use std::sync::{Arc, Mutex};
use crate::graph::{RenderPass, PassContext, PassResourceBuilder, ResourceHandle};
use crate::mesh::DrawCall;
use crate::Result;

/// The four G-buffer render targets.  Wrapped in Arc<Mutex<…>> so the Renderer
/// can swap in new views when the window is resized without rebuilding the graph.
pub struct GBufferTargets {
    pub albedo_view:   wgpu::TextureView,   // Rgba8Unorm
    pub normal_view:   wgpu::TextureView,   // Rgba16Float
    pub orm_view:      wgpu::TextureView,   // Rgba8Unorm
    pub emissive_view: wgpu::TextureView,   // Rgba16Float
}

pub struct GBufferPass {
    targets:   Arc<Mutex<GBufferTargets>>,
    pipeline:  Arc<wgpu::RenderPipeline>,
    draw_list: Arc<Mutex<Vec<DrawCall>>>,
}

impl GBufferPass {
    pub fn new(
        targets:   Arc<Mutex<GBufferTargets>>,
        pipeline:  Arc<wgpu::RenderPipeline>,
        draw_list: Arc<Mutex<Vec<DrawCall>>>,
    ) -> Self {
        Self { targets, pipeline, draw_list }
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
        let draw_calls: Vec<DrawCall> = self.draw_list.lock().unwrap().clone();

        let targets = self.targets.lock().unwrap();

        let color_attachments = [
            // target 0: albedo
            Some(wgpu::RenderPassColorAttachment {
                view: &targets.albedo_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }),
            // target 1: normals
            Some(wgpu::RenderPassColorAttachment {
                view: &targets.normal_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }),
            // target 2: ORM
            Some(wgpu::RenderPassColorAttachment {
                view: &targets.orm_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }),
            // target 3: emissive
            Some(wgpu::RenderPassColorAttachment {
                view: &targets.emissive_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }),
        ];

        let depth_attachment = Some(wgpu::RenderPassDepthStencilAttachment {
            view: ctx.depth_view,
            depth_ops: Some(wgpu::Operations {
                load:  wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        });

        // Need an explicit encoder::begin_render_pass here since we have 4 targets
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("G-Buffer Write Pass"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: depth_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, ctx.global_bind_group, &[]);
        // group 2 is NOT set here — gbuffer.wgsl has no group 2 bindings

        for dc in &draw_calls {
            if dc.transparent_blend {
                continue;
            }
            pass.set_bind_group(1, Some(dc.material_bind_group.as_ref()), &[]);
            pass.set_vertex_buffer(0, dc.vertex_buffer.slice(..));
            pass.set_index_buffer(dc.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..dc.index_count, 0, 0..1);
        }

        Ok(())
    }
}
