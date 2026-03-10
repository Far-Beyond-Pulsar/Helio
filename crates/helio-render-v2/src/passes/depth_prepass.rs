//! Depth prepass — depth-only early-Z pass before the G-buffer.
//!
//! Records directly into the primary `CommandEncoder` each frame (no bundles).
//! Draw calls are sorted by (material, mesh) on scene-generation change and
//! consecutive same-mesh+material draws are merged into a single instanced call.
//!
//! The sort result is published to `shared_sorted` so `GBufferPass` can skip
//! re-sorting (same material order is optimal for both passes).
//!
//! ## Why no RenderBundles
//!
//! wgpu `RenderBundle`s map to D3D12 bundles / Vulkan secondary command buffers.
//! Replaying them via `execute_bundles()` is O(1) in the *primary* encoder, but:
//!   - They require a parallel compile step that `thread::scope`-joins on the main
//!     thread whenever the scene changes (chunk streaming) — adding latency spikes.
//!   - The driver still validates and inlines them at submit time.
//!
//! Direct recording is simpler, has no compile overhead, and lets the primary
//! encoder pipeline state flow naturally.  At ~200 ns/command on DX12/Vulkan,
//! 15 000 opaque draws ≈ 3 ms — acceptable and constant (no spikes on movement).

use std::sync::{Arc, Mutex};
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

/// Depth-only pass that writes depth for all opaque draw calls.
///
/// Sorts draw calls by (material, mesh) once per scene-generation change,
/// merges contiguous same-mesh+material runs into instanced draws, then
/// records them directly into the primary `CommandEncoder`.
pub struct DepthPrepassPass {
    pipeline: Arc<wgpu::RenderPipeline>,
    draw_list: Arc<Mutex<Vec<DrawCall>>>,
    /// Opaque draw indices sorted by (material, vertex_buffer, index_buffer, offset).
    /// Rebuilt when `draw_list_generation` changes; shared with `GBufferPass`.
    sorted_opaque_indices: Vec<usize>,
    /// Sorted index list published for `GBufferPass` after each rebuild.
    pub shared_sorted: Arc<Mutex<Vec<usize>>>,
    /// `draw_list_generation` value at the time `sorted_opaque_indices` was last built.
    sort_generation: u64,
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
            sort_generation: u64::MAX,
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
        let draw_calls = self.draw_list.lock().unwrap();

        // Re-sort only when the scene composition changes (add/remove object).
        // Camera movement does NOT trigger this — zero re-sort cost while flying.
        if ctx.draw_list_generation != self.sort_generation {
            let t = std::time::Instant::now();
            self.sorted_opaque_indices = sort_opaque_indices(&draw_calls);
            {
                let mut shared = self.shared_sorted.lock().unwrap();
                shared.clear();
                shared.extend_from_slice(&self.sorted_opaque_indices);
            }
            self.sort_generation = ctx.draw_list_generation;
            let sort_ms = t.elapsed().as_secs_f32() * 1000.0;
            if sort_ms > 0.5 {
                eprintln!(
                    "⚠️ [DepthPrepass] Sort: {} opaque draws — {:.2}ms (gen {})",
                    self.sorted_opaque_indices.len(), sort_ms, ctx.draw_list_generation,
                );
            }
        }

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

        {
            record_opaque_draws(&mut pass, &draw_calls, &self.sorted_opaque_indices);
        }

        Ok(())
    }
}

/// Record opaque draw calls into an active render pass.
///
/// Minimises state changes: bind groups and VB/IB are only re-set when they
/// actually change between draws.  Each draw uses `first_instance = dc.slot`
/// so the vertex shader fetches its transform from the instance_data storage
/// buffer — no per-draw vertex-attribute instance buffers.
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
