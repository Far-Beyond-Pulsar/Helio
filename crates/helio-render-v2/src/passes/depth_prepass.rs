//! Depth prepass - renders depth-only before GBuffer for early fragment rejection.
//!
//! This pass renders all opaque geometry to write depth without color output.
//! GBuffer later loads this depth and skips fragments already in shadow/occluded,
//! saving expensive shading work on invisible pixels (typically 50-80%).

use std::sync::{Arc, Mutex};
use crate::graph::{RenderPass, PassContext, PassResourceBuilder, ResourceHandle};
use crate::mesh::{DrawCall, INSTANCE_STRIDE};
use crate::Result;

/// Depth-only pass that writes depth for all opaque draw calls
pub struct DepthPrepassPass {
    pipeline: Arc<wgpu::RenderPipeline>,
    draw_list: Arc<Mutex<Vec<DrawCall>>>,
    sorted_opaque_indices: Vec<usize>,
    /// Sorted opaque index list shared with GBufferPass so the material-order
    /// sort is computed once per structural change here and read by GBuffer.
    pub shared_sorted: Arc<Mutex<Vec<usize>>>,
    /// Order-independent (XOR) hash of the opaque draw set identity.
    /// Re-sorts and publishes to GBufferPass only when this changes.
    /// Using a commutative accumulator means HashMap iteration order cannot
    /// trigger false positives (the previous order-dependent hash fired on
    /// every scene rebuild due to non-deterministic HashMap ordering).
    sort_geom_hash: u64,
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
            sort_geom_hash: u64::MAX,
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

        // Order-independent (commutative XOR) hash of the opaque draw set identity.
        // Using XOR means HashMap/Vec iteration order cannot trigger false positives —
        // the previous order-dependent FNV hash fired on every scene rebuild caused by
        // non-deterministic HashMap iteration, forcing a re-sort every frame.
        // Only changes when the SET of batches changes structurally (add/remove/resize).
        let new_sort_hash = {
            let mut h: u64 = draw_calls.iter().filter(|dc| !dc.transparent_blend).count() as u64;
            for dc in draw_calls.iter() {
                if dc.transparent_blend || dc.instance_buffer.is_none() { continue; }
                let entry = (Arc::as_ptr(&dc.vertex_buffer) as u64)
                    .wrapping_mul(0x9e3779b97f4a7c15)
                    ^ (Arc::as_ptr(dc.instance_buffer.as_ref().unwrap()) as u64)
                        .wrapping_mul(0x517cc1b727220a95)
                    ^ (dc.instance_count as u64).wrapping_mul(0x6c62272e07bb0142);
                h ^= entry.wrapping_mul(0xbf58476d1ce4e5b9);
            }
            h
        };

        // Re-sort by material pointer (state-change minimisation) and publish to
        // GBufferPass only when the draw set changes identity.
        if new_sort_hash != self.sort_geom_hash {
            self.sorted_opaque_indices.clear();
            self.sorted_opaque_indices.reserve(draw_calls.len());
            for (idx, dc) in draw_calls.iter().enumerate() {
                if !dc.transparent_blend { self.sorted_opaque_indices.push(idx); }
            }
            self.sorted_opaque_indices.sort_unstable_by_key(|&i| {
                Arc::as_ptr(&draw_calls[i].material_bind_group) as usize
            });
            {
                let mut shared = self.shared_sorted.lock().unwrap();
                shared.clear();
                shared.extend_from_slice(&self.sorted_opaque_indices);
            }
            self.sort_geom_hash = new_sort_hash;
        }

        // Direct-encode all opaque draws into the depth prepass.
        // No RenderBundle compilation step — direct encoding costs ~1-2 ms at
        // 5 000 calls but is constant every frame, vs the previous bundle approach
        // which cost 4-6 ms per rebuild and was rebuilding nearly every frame.
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Depth Prepass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: ctx.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
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
        pass.set_bind_group(3, ctx.gpu_scene_bind_group, &[]);
        let mut last_material: Option<usize> = None;
        for &idx in &self.sorted_opaque_indices {
            let dc = &draw_calls[idx];
            if dc.instance_buffer.is_none() { continue; }
            let mat_ptr = Arc::as_ptr(&dc.material_bind_group) as usize;
            if last_material != Some(mat_ptr) {
                pass.set_bind_group(1, Some(dc.material_bind_group.as_ref()), &[]);
                last_material = Some(mat_ptr);
            }
            pass.set_vertex_buffer(0, dc.vertex_buffer.slice(..));
            let inst_start = dc.instance_buffer_offset;
            let inst_end   = inst_start + dc.instance_count as u64 * INSTANCE_STRIDE;
            pass.set_vertex_buffer(1, dc.instance_buffer.as_ref().unwrap().slice(inst_start..inst_end));
            pass.set_index_buffer(dc.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..dc.index_count, 0, 0..dc.instance_count);
        }
        Ok(())
    }
}
