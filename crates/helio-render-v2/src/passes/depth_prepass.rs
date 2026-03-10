//! Depth prepass — depth-only early-Z pass before the G-buffer.
//!
//! GPU-driven path: when the indirect buffer is available (populated by
//! `IndirectDispatchPass`), issues a single `multi_draw_indexed_indirect` call
//! for all opaque geometry using the unified pool vertex/index buffers.
//! O(1) CPU cost regardless of draw count.
//!
//! Legacy fallback: when the indirect buffer is not available (non-pool meshes),
//! falls back to per-draw `draw_indexed` with per-draw VB/IB binds.

use std::sync::{Arc, Mutex};
use crate::gpu_scene::MaterialRange;
use crate::graph::{RenderPass, PassContext, PassResourceBuilder, ResourceHandle};
use crate::mesh::DrawCall;
use crate::Result;

/// Sort opaque draw call indices by (material, slot).
pub fn sort_opaque_indices(draw_calls: &[DrawCall]) -> Vec<usize> {
    let mut indices: Vec<usize> = Vec::with_capacity(draw_calls.len());
    for (idx, dc) in draw_calls.iter().enumerate() {
        if !dc.transparent_blend {
            indices.push(idx);
        }
    }
    indices.sort_unstable_by(|&ia, &ib| {
        let a = &draw_calls[ia];
        let b = &draw_calls[ib];
        (Arc::as_ptr(&a.material_bind_group) as usize)
            .cmp(&(Arc::as_ptr(&b.material_bind_group) as usize))
            .then_with(|| a.slot.cmp(&b.slot))
    });
    indices
}

pub struct DepthPrepassPass {
    pipeline: Arc<wgpu::RenderPipeline>,
    // ── Legacy path (non-pool meshes) ──────────────────────────────────────────
    draw_list: Arc<Mutex<Vec<DrawCall>>>,
    sorted_opaque_indices: Vec<usize>,
    pub shared_sorted: Arc<Mutex<Vec<usize>>>,
    sort_generation: u64,
    // ── GPU-driven path ────────────────────────────────────────────────────────
    /// Pool vertex buffer (shared with all geometry passes).
    pool_vertex_buffer: Arc<wgpu::Buffer>,
    /// Pool index buffer (shared with all geometry passes).
    pool_index_buffer: Arc<wgpu::Buffer>,
    /// Indirect buffer written by IndirectDispatchPass. `None` until first dispatch.
    shared_indirect: Arc<Mutex<Option<Arc<wgpu::Buffer>>>>,
    /// Per-material draw ranges into the indirect buffer (opaque only).
    shared_material_ranges: Arc<Mutex<Vec<MaterialRange>>>,
    /// Default material BG set for the depth-only pipeline (depth shader ignores it).
    default_material_bg: Arc<wgpu::BindGroup>,
}

impl DepthPrepassPass {
    pub fn new(
        pipeline: Arc<wgpu::RenderPipeline>,
        draw_list: Arc<Mutex<Vec<DrawCall>>>,
        pool_vertex_buffer: Arc<wgpu::Buffer>,
        pool_index_buffer: Arc<wgpu::Buffer>,
        shared_indirect: Arc<Mutex<Option<Arc<wgpu::Buffer>>>>,
        shared_material_ranges: Arc<Mutex<Vec<MaterialRange>>>,
        default_material_bg: Arc<wgpu::BindGroup>,
    ) -> (Self, Arc<Mutex<Vec<usize>>>) {
        let shared_sorted = Arc::new(Mutex::new(Vec::new()));
        let pass = Self {
            pipeline,
            draw_list,
            sorted_opaque_indices: Vec::new(),
            shared_sorted: shared_sorted.clone(),
            sort_generation: u64::MAX,
            pool_vertex_buffer,
            pool_index_buffer,
            shared_indirect,
            shared_material_ranges,
            default_material_bg,
        };
        (pass, shared_sorted)
    }
}

impl RenderPass for DepthPrepassPass {
    fn name(&self) -> &str { "depth_prepass" }

    fn declare_resources(&self, builder: &mut PassResourceBuilder) {
        builder.write(ResourceHandle::named("depth"));
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Depth Prepass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: ctx.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(1.0),
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

        // ── GPU-driven path ────────────────────────────────────────────────
        let indirect_buf = self.shared_indirect.lock().unwrap().clone();
        if let Some(indirect_buf) = indirect_buf {
            let draw_count = {
                let ranges = self.shared_material_ranges.lock().unwrap();
                if ranges.is_empty() { return Ok(()); }
                // For depth-only: one multi_draw_indexed_indirect for ALL draws.
                // Count = sum of all range counts.
                ranges.iter().map(|r| r.count).sum::<u32>()
            };
            if draw_count == 0 { return Ok(()); }

            // Set a dummy material BG so the pipeline layout is satisfied.
            // The depth-only shader doesn't sample any material textures.
            pass.set_bind_group(1, Some(self.default_material_bg.as_ref()), &[]);
            pass.set_vertex_buffer(0, self.pool_vertex_buffer.slice(..));
            pass.set_index_buffer(self.pool_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            // Single call for all opaque draws — compute shader already culled them.
            pass.multi_draw_indexed_indirect(&indirect_buf, 0, draw_count);
            return Ok(());
        }

        // ── Legacy fallback (non-pool meshes) ─────────────────────────────
        let draw_calls = self.draw_list.lock().unwrap();
        if ctx.draw_list_generation != self.sort_generation {
            self.sorted_opaque_indices = sort_opaque_indices(&draw_calls);
            {
                let mut shared = self.shared_sorted.lock().unwrap();
                shared.clear();
                shared.extend_from_slice(&self.sorted_opaque_indices);
            }
            self.sort_generation = ctx.draw_list_generation;
        }
        record_opaque_draws(&mut pass, &draw_calls, &self.sorted_opaque_indices);
        Ok(())
    }
}

/// Record opaque draw calls into an active render pass (legacy fallback path).
///
/// Uses per-draw VB/IB binds. Only used when pool meshes are not available.
pub(crate) fn record_opaque_draws(
    pass: &mut wgpu::RenderPass<'_>,
    draw_calls: &[DrawCall],
    sorted: &[usize],
) {
    let mut last_mat:  Option<usize> = None;
    let mut last_vbuf: Option<usize> = None;
    let mut last_ibuf: Option<usize> = None;

    for &idx in sorted {
        let dc = &draw_calls[idx];
        let mat_ptr  = Arc::as_ptr(&dc.material_bind_group) as usize;
        let vbuf_ptr = Arc::as_ptr(&dc.vertex_buffer)        as usize;
        let ibuf_ptr = Arc::as_ptr(&dc.index_buffer)         as usize;

        if last_mat != Some(mat_ptr) {
            pass.set_bind_group(1, Some(dc.material_bind_group.as_ref()), &[]);
            last_mat = Some(mat_ptr);
        }
        if last_vbuf != Some(vbuf_ptr) {
            pass.set_vertex_buffer(0, dc.vertex_buffer.slice(..));
            last_vbuf = Some(vbuf_ptr);
        }
        if last_ibuf != Some(ibuf_ptr) {
            pass.set_index_buffer(dc.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            last_ibuf = Some(ibuf_ptr);
        }

        pass.draw_indexed(
            dc.pool_first_index..dc.pool_first_index + dc.index_count,
            dc.pool_base_vertex,
            dc.slot..dc.slot + 1,
        );
    }
}

