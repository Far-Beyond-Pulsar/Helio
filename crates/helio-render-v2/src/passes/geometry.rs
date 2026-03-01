//! Main geometry pass – renders all queued draw calls

use crate::graph::{RenderPass, PassContext};
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

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let target      = ctx.target;
        let global_bg   = ctx.global_bind_group;
        let lighting_bg = ctx.lighting_bind_group;

        // Snapshot the draw list (cheap Arc clone, cheap lock)
        let draw_calls: Vec<DrawCall> = self.draw_list.lock().unwrap().clone();

        let [r, g, b] = ctx.sky_color.map(|c| c as f64);
        let color_attachment = Some(wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a: 1.0 }),
                store: wgpu::StoreOp::Store,
            },
        });

        // If no pipeline, just clear so features/billboards can still draw
        let Some(pipeline) = self.pipeline.as_ref() else {
            let _pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Geometry Pass (clear only)"),
                color_attachments: &[color_attachment],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            return Ok(());
        };

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Geometry Pass"),
            color_attachments: &[color_attachment],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, global_bg, &[]);
        pass.set_bind_group(2, lighting_bg, &[]);

        for dc in &draw_calls {
            pass.set_bind_group(1, &dc.material_bind_group, &[]);
            pass.set_vertex_buffer(0, dc.vertex_buffer.slice(..));
            pass.set_index_buffer(dc.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..dc.index_count, 0, 0..1);
        }

        Ok(())
    }
}

impl Default for GeometryPass {
    fn default() -> Self {
        Self::with_draw_list(Arc::new(Mutex::new(Vec::new())))
    }
}
