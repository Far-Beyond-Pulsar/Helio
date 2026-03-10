//! Forward transparent pass for alpha-blended materials.
//!
//! Draws objects flagged with `transparent_blend` after deferred lighting,
//! using the same PBR shader as forward geometry (`geometry.wgsl`).
//!
//! Per-frame back-to-front sort + inline draw_indexed with pool VB/IB.
//! No RenderBundle overhead — pool draws are cheap at O(transparent_draws) CPU.

use std::sync::{Arc, Mutex};

use crate::graph::{PassContext, PassResourceBuilder, RenderPass, ResourceHandle};
use crate::mesh::DrawCall;
use crate::Result;

pub struct TransparentPass {
    pipeline:          Arc<wgpu::RenderPipeline>,
    draw_list:         Arc<Mutex<Vec<DrawCall>>>,
    pool_vertex_buffer: Arc<wgpu::Buffer>,
    pool_index_buffer:  Arc<wgpu::Buffer>,
}

impl TransparentPass {
    pub fn new(
        pipeline:           Arc<wgpu::RenderPipeline>,
        draw_list:          Arc<Mutex<Vec<DrawCall>>>,
        pool_vertex_buffer: Arc<wgpu::Buffer>,
        pool_index_buffer:  Arc<wgpu::Buffer>,
    ) -> Self {
        Self { pipeline, draw_list, pool_vertex_buffer, pool_index_buffer }
    }
}

impl RenderPass for TransparentPass {
    fn name(&self) -> &str { "transparent" }

    fn declare_resources(&self, builder: &mut PassResourceBuilder) {
        builder.read(ResourceHandle::named("color_target"));
        builder.write(ResourceHandle::named("color_target"));
        builder.read(ResourceHandle::named("gbuffer"));
        builder.write(ResourceHandle::named("transparent_done"));
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let draw_calls = self.draw_list.lock().unwrap();

        // Collect transparent draw indices.
        let ts = ctx.transparent_start.min(draw_calls.len());
        let mut indices: Vec<usize> = (ts..draw_calls.len()).collect();
        if indices.is_empty() { return Ok(()); }

        // Back-to-front sort by camera-space depth for correct blending.
        let cam = ctx.camera_position;
        let fwd = ctx.camera_forward;
        indices.sort_unstable_by(|&ia, &ib| {
            let da = (glam::Vec3::from(draw_calls[ia].bounds_center) - cam).dot(fwd);
            let db = (glam::Vec3::from(draw_calls[ib].bounds_center) - cam).dot(fwd);
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
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
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

        // Pool-only: one VB/IB bind for all transparent draws.
        pass.set_vertex_buffer(0, self.pool_vertex_buffer.slice(..));
        pass.set_index_buffer(self.pool_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        let mut last_mat: Option<usize> = None;
        for &idx in &indices {
            let dc = &draw_calls[idx];
            let mat_ptr = Arc::as_ptr(&dc.material_bind_group) as usize;
            if last_mat != Some(mat_ptr) {
                pass.set_bind_group(1, Some(dc.material_bind_group.as_ref()), &[]);
                last_mat = Some(mat_ptr);
            }
            pass.draw_indexed(
                dc.pool_first_index..dc.pool_first_index + dc.index_count,
                dc.pool_base_vertex,
                dc.slot..dc.slot + 1,
            );
        }

        Ok(())
    }
}

