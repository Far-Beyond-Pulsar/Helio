//! Depth prepass - renders depth-only before GBuffer for early fragment rejection.
//!
//! This pass renders all opaque geometry to write depth without color output.
//! GBuffer later loads this depth and skips fragments already in shadow/occluded,
//! saving expensive shading work on invisible pixels (typically 50-80%).

use std::sync::{Arc, Mutex};
use crate::graph::{RenderPass, PassContext, PassResourceBuilder, ResourceHandle};
use crate::mesh::DrawCall;
use crate::Result;

/// Depth-only pass that writes depth for all opaque draw calls
pub struct DepthPrepassPass {
    pipeline: Arc<wgpu::RenderPipeline>,
    draw_list: Arc<Mutex<Vec<DrawCall>>>,
    sorted_opaque_indices: Vec<usize>,
}

impl DepthPrepassPass {
    pub fn new(
        pipeline: Arc<wgpu::RenderPipeline>,
        draw_list: Arc<Mutex<Vec<DrawCall>>>,
    ) -> Self {
        Self {
            pipeline,
            draw_list,
            sorted_opaque_indices: Vec::new(),
        }
    }
}

impl RenderPass for DepthPrepassPass {
    fn name(&self) -> &str { "depth_prepass" }

    fn declare_resources(&self, builder: &mut PassResourceBuilder) {
        // Depth prepass writes to the shared depth buffer; GBuffer will load it
        builder.write(ResourceHandle::named("depth"));
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let draw_calls = self.draw_list.lock().unwrap();
        let cam = ctx.camera_position;
        let fwd = ctx.camera_forward;

        // Sort opaque draws front-to-back for early-Z rejection.
        // This maximizes the benefit of Z-test culling: nearby opaque geometry
        // writes depth first, rejecting all geometry behind it.
        self.sorted_opaque_indices.clear();
        self.sorted_opaque_indices.reserve(
            draw_calls
                .len()
                .saturating_sub(self.sorted_opaque_indices.capacity()),
        );
        for (idx, dc) in draw_calls.iter().enumerate() {
            // Depth prepass only renders opaque geometry
            if !dc.transparent_blend {
                self.sorted_opaque_indices.push(idx);
            }
        }

        self.sorted_opaque_indices.sort_by(|&ia, &ib| {
            let a = &draw_calls[ia];
            let b = &draw_calls[ib];

            // Front-to-back: closest first (smallest depth)
            let za = (glam::Vec3::from(a.bounds_center) - cam).dot(fwd) - a.bounds_radius;
            let zb = (glam::Vec3::from(b.bounds_center) - cam).dot(fwd) - b.bounds_radius;

            za.partial_cmp(&zb)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    let ma = Arc::as_ptr(&a.material_bind_group) as usize;
                    let mb = Arc::as_ptr(&b.material_bind_group) as usize;
                    ma.cmp(&mb)
                })
        });

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Depth Prepass"),
            color_attachments: &[],  // No color output — depth only
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: ctx.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),  // Far plane = 1.0
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

        let mut last_material: Option<usize> = None;
        for &idx in &self.sorted_opaque_indices {
            let dc = &draw_calls[idx];
            let mat_ptr = Arc::as_ptr(&dc.material_bind_group) as usize;
            if last_material != Some(mat_ptr) {
                pass.set_bind_group(1, Some(dc.material_bind_group.as_ref()), &[]);
                last_material = Some(mat_ptr);
            }
            pass.set_vertex_buffer(0, dc.vertex_buffer.slice(..));
            pass.set_index_buffer(dc.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..dc.index_count, 0, 0..1);
        }

        Ok(())
    }
}
