//! Main geometry pass – renders all queued draw calls

use crate::graph::{RenderPass, PassContext, PassResourceBuilder, ResourceHandle};
use crate::mesh::DrawCall;
use crate::Result;
use std::sync::{Arc, Mutex};

/// Renders all geometry draw calls submitted via `Renderer::draw_mesh()`
pub struct GeometryPass {
    pipeline: Option<Arc<wgpu::RenderPipeline>>,
    draw_list: Arc<Mutex<Vec<DrawCall>>>,
}

impl GeometryPass {
    pub fn with_draw_list(draw_list: Arc<Mutex<Vec<DrawCall>>>) -> Self {
        Self { pipeline: None, draw_list }
    }

    pub fn set_pipeline(&mut self, pipeline: Arc<wgpu::RenderPipeline>) {
        self.pipeline = Some(pipeline);
    }
}

impl RenderPass for GeometryPass {
    fn name(&self) -> &str { "geometry" }

    fn declare_resources(&self, builder: &mut PassResourceBuilder) {
        // Reads shadow atlas written by ShadowPass → enforces shadow-before-geometry order
        builder.read(ResourceHandle::named("shadow_atlas"));
        // Reads RC cascade 0 written by RadianceCascadesPass → enforces RC-before-geometry order
        builder.read(ResourceHandle::named("rc_cascade0"));
        // Reads sky layer written by SkyPass → enforces sky-before-geometry order
        builder.read(ResourceHandle::named("sky_layer"));
        // Writes color target read by BillboardPass → enforces geometry-before-billboard order
        builder.write(ResourceHandle::named("color_target"));
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let target      = ctx.target;
        let depth_view  = ctx.depth_view;
        let global_bg   = ctx.global_bind_group;
        let lighting_bg = ctx.lighting_bind_group;

        // Borrow draw calls directly to avoid a full per-frame Vec clone.
        let draw_calls = self.draw_list.lock().unwrap();

        // When SkyPass has already filled the color buffer, load it instead of clearing.
        let color_load = if ctx.has_sky {
            wgpu::LoadOp::Load
        } else {
            let [r, g, b] = ctx.sky_color.map(|c| c as f64);
            wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a: 1.0 })
        };
        let color_attachment = Some(wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: color_load,
                store: wgpu::StoreOp::Store,
            },
        });

        let depth_attachment = Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        });

        // If no pipeline, just clear so features/billboards can still draw
        let Some(pipeline) = self.pipeline.as_ref() else {
            let _pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Geometry Pass (clear only)"),
                color_attachments: &[color_attachment],
                depth_stencil_attachment: depth_attachment,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            return Ok(());
        };

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Geometry Pass"),
            color_attachments: &[color_attachment],
            depth_stencil_attachment: depth_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, global_bg, &[]);
        pass.set_bind_group(2, lighting_bg, &[]);

        for dc in draw_calls.iter() {
            pass.set_bind_group(1, Some(dc.material_bind_group.as_ref()), &[]);
            pass.set_vertex_buffer(0, dc.vertex_buffer.slice(..));
            pass.set_index_buffer(dc.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..dc.index_count, dc.pool_base_vertex, dc.slot..dc.slot + 1);
        }

        Ok(())
    }
}

impl Default for GeometryPass {
    fn default() -> Self {
        Self::with_draw_list(Arc::new(Mutex::new(Vec::new())))
    }
}
