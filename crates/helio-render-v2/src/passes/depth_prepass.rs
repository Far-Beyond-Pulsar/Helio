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
    /// Sorted opaque index list shared with GBufferPass so the front-to-back
    /// ordering is computed once here and read directly by GBuffer.
    pub shared_sorted: Arc<Mutex<Vec<usize>>>,
}

impl DepthPrepassPass {
    pub fn new(
        pipeline: Arc<wgpu::RenderPipeline>,
        draw_list: Arc<Mutex<Vec<DrawCall>>>,
    ) -> (Self, Arc<Mutex<Vec<usize>>>) {
        let shared_sorted = Arc::new(Mutex::new(Vec::new()));
        let pass = Self {
            pipeline,
            draw_list,
            sorted_opaque_indices: Vec::new(),
            shared_sorted: shared_sorted.clone(),
        };
        (pass, shared_sorted)
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
        self.sorted_opaque_indices.reserve(draw_calls.len());
        for (idx, dc) in draw_calls.iter().enumerate() {
            // Depth prepass only renders opaque geometry
            if !dc.transparent_blend {
                self.sorted_opaque_indices.push(idx);
            }
        }

        // sort_unstable_by: in-place introsort, no heap allocation vs sort_by's merge sort.
        self.sorted_opaque_indices.sort_unstable_by(|&ia, &ib| {
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

        // Publish sorted order to GBufferPass so it skips its own identical sort.
        {
            let mut shared = self.shared_sorted.lock().unwrap();
            shared.clear();
            shared.extend_from_slice(&self.sorted_opaque_indices);
        }

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
        pass.set_bind_group(2, ctx.lighting_bind_group, &[]);

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
