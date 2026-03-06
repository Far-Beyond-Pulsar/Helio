//! Forward transparent pass for alpha-blended materials.
//!
//! Draws objects flagged with `transparent_blend` after deferred lighting,
//! using the same PBR shader as forward geometry (`geometry.wgsl`) so lights,
//! shadows, environment and GI all affect transparent surfaces.

use std::sync::{Arc, Mutex};

use crate::graph::{PassContext, PassResourceBuilder, RenderPass, ResourceHandle};
use crate::mesh::DrawCall;
use crate::Result;

pub struct TransparentPass {
    pipeline: Arc<wgpu::RenderPipeline>,
    draw_list: Arc<Mutex<Vec<DrawCall>>>,
}

impl TransparentPass {
    pub fn new(
        pipeline: Arc<wgpu::RenderPipeline>,
        draw_list: Arc<Mutex<Vec<DrawCall>>>,
    ) -> Self {
        Self { pipeline, draw_list }
    }
}

impl RenderPass for TransparentPass {
    fn name(&self) -> &str {
        "transparent"
    }

    fn declare_resources(&self, builder: &mut PassResourceBuilder) {
        builder.read(ResourceHandle::named("color_target"));
        builder.read(ResourceHandle::named("gbuffer"));
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let mut draws: Vec<DrawCall> = self
            .draw_list
            .lock()
            .unwrap()
            .iter()
            .filter(|dc| dc.transparent_blend)
            .cloned()
            .collect();

        if draws.is_empty() {
            return Ok(());
        }

        // Back-to-front for standard alpha blending.
        let cam = ctx.camera_position;
        draws.sort_by(|a, b| {
            let da = {
                let dx = a.bounds_center[0] - cam.x;
                let dy = a.bounds_center[1] - cam.y;
                let dz = a.bounds_center[2] - cam.z;
                dx * dx + dy * dy + dz * dz
            };
            let db = {
                let dx = b.bounds_center[0] - cam.x;
                let dy = b.bounds_center[1] - cam.y;
                let dz = b.bounds_center[2] - cam.z;
                dx * dx + dy * dy + dz * dz
            };
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Transparent Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: ctx.depth_view,
                depth_ops: None,
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, ctx.global_bind_group, &[]);
        pass.set_bind_group(2, ctx.lighting_bind_group, &[]);

        for dc in &draws {
            pass.set_bind_group(1, Some(dc.material_bind_group.as_ref()), &[]);
            pass.set_vertex_buffer(0, dc.vertex_buffer.slice(..));
            pass.set_index_buffer(dc.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..dc.index_count, 0, 0..1);
        }

        Ok(())
    }
}
