//! Depth prepass - renders depth-only before GBuffer for early fragment rejection.
//!
//! This pass renders all opaque geometry to write depth without color output.
//! GBuffer later loads this depth and skips fragments already in shadow/occluded,
//! saving expensive shading work on invisible pixels (typically 50-80%).
//!
//! ## Performance design
//!
//! The hot path at steady state (no scene changes) is:
//!   1. Check `draw_list_generation` — unchanged, skip everything below.
//!   2. `pass.execute_bundles([&bundle])` — single wgpu command, ~0 µs CPU.
//!
//! On first use or after a scene change the bundle is (re)compiled.  Compilation
//! encodes all draws with:
//!   - `set_bind_group` only on material changes  (state-change minimisation)
//!   - `set_vertex_buffer(0)` / `set_index_buffer` only on mesh changes
//!   - Contiguous-slot runs of the same mesh+material merged into one
//!     `draw_indexed` with `instance_count = N`  (GPU instancing)
//! Typical cost ≈ 3–6 ms once; every subsequent frame ≈ 0 ms.

use std::sync::{Arc, Mutex};
use crate::graph::{RenderPass, PassContext, PassResourceBuilder, ResourceHandle};
use crate::mesh::{DrawCall, INSTANCE_STRIDE};
use crate::Result;

/// Shared inbox for delivering pre-compiled bundles from parallel compilation.
///
/// Modelled after Unreal's `FParallelCommandListSet`: the draw list is split
/// into N chunks (one per worker thread) and each chunk is recorded into its
/// own `RenderBundle` in parallel.  The pass replays all N bundles in order
/// via `execute_bundles(&bundles)`.  This is how UE achieves <2 ms command
/// recording for scenes with tens of thousands of draw calls.
pub struct BundleInbox {
    pub depth_prepass: Mutex<Option<PrecompiledBundles>>,
    pub gbuffer: Mutex<Option<PrecompiledBundles>>,
}

/// A set of pre-compiled RenderBundles (one per chunk) ready for a pass to adopt.
///
/// Equivalent to Unreal's completed `FParallelCommandListSet`:
/// N command lists recorded in parallel, ready for `ExecuteCommandLists()`.
pub struct PrecompiledBundles {
    /// One RenderBundle per chunk, recorded in parallel.  Replayed in order.
    pub bundles: Vec<wgpu::RenderBundle>,
    /// Buffer references that must stay alive while bundles are in-flight.
    pub kept_arcs: Vec<Arc<wgpu::Buffer>>,
    /// Full sorted opaque index list (shared with GBufferPass for fallback).
    pub sorted_indices: Vec<usize>,
    pub generation: u64,
    pub global_bg_ptr: usize,
    pub lighting_bg_ptr: usize,
}

/// Sort opaque draw call indices by (material, vertex_buffer, index_buffer, offset).
/// Used by the parallel pre-compilation step.
pub fn sort_opaque_indices(draw_calls: &[DrawCall]) -> Vec<usize> {
    let mut indices: Vec<usize> = Vec::with_capacity(draw_calls.len());
    for (idx, dc) in draw_calls.iter().enumerate() {
        if !dc.transparent_blend && dc.instance_buffer.is_some() {
            indices.push(idx);
        }
    }
    indices.sort_unstable_by(|&ia, &ib| {
        let a = &draw_calls[ia];
        let b = &draw_calls[ib];
        (Arc::as_ptr(&a.material_bind_group) as usize)
            .cmp(&(Arc::as_ptr(&b.material_bind_group) as usize))
            .then_with(|| (Arc::as_ptr(&a.vertex_buffer) as usize)
                .cmp(&(Arc::as_ptr(&b.vertex_buffer) as usize)))
            .then_with(|| (Arc::as_ptr(&a.index_buffer) as usize)
                .cmp(&(Arc::as_ptr(&b.index_buffer) as usize)))
            .then_with(|| a.instance_buffer_offset.cmp(&b.instance_buffer_offset))
    });
    indices
}

/// Build a depth-prepass RenderBundle from pre-sorted draw indices.
/// Thread-safe: only needs shared references to device, pipeline, and bind groups.
pub fn build_depth_bundle(
    device: &wgpu::Device,
    pipeline: &wgpu::RenderPipeline,
    draw_calls: &[DrawCall],
    sorted: &[usize],
    global_bg: &wgpu::BindGroup,
    lighting_bg: &wgpu::BindGroup,
) -> (wgpu::RenderBundle, Vec<Arc<wgpu::Buffer>>) {
    let mut enc = device.create_render_bundle_encoder(
        &wgpu::RenderBundleEncoderDescriptor {
            label: Some("depth_prepass_bundle"),
            color_formats: &[],
            depth_stencil: Some(wgpu::RenderBundleDepthStencil {
                format: wgpu::TextureFormat::Depth32Float,
                depth_read_only: false,
                stencil_read_only: true,
            }),
            sample_count: 1,
            multiview: None,
        },
    );
    enc.set_pipeline(pipeline);
    enc.set_bind_group(0, global_bg, &[]);
    enc.set_bind_group(2, lighting_bg, &[]);

    let mut kept: Vec<Arc<wgpu::Buffer>> = Vec::with_capacity(sorted.len());
    let mut last_mat:  Option<usize> = None;
    let mut last_vbuf: Option<usize> = None;
    let mut last_ibuf: Option<usize> = None;
    let mut batch_start: u64  = 0;
    let mut batch_count: u32  = 0;
    let mut batch_idx:   usize = 0;

    for &idx in sorted {
        let dc = &draw_calls[idx];
        let mat_ptr  = Arc::as_ptr(&dc.material_bind_group) as usize;
        let vbuf_ptr = Arc::as_ptr(&dc.vertex_buffer)        as usize;
        let ibuf_ptr = Arc::as_ptr(&dc.index_buffer)         as usize;

        if batch_count > 0
            && last_mat  == Some(mat_ptr)
            && last_vbuf == Some(vbuf_ptr)
            && last_ibuf == Some(ibuf_ptr)
            && batch_start + batch_count as u64 * INSTANCE_STRIDE == dc.instance_buffer_offset
        {
            batch_count += 1;
            continue;
        }

        if batch_count > 0 {
            let bdc      = &draw_calls[batch_idx];
            let inst_end = batch_start + batch_count as u64 * INSTANCE_STRIDE;
            enc.set_vertex_buffer(1, bdc.instance_buffer.as_ref().unwrap().slice(batch_start..inst_end));
            enc.draw_indexed(0..bdc.index_count, 0, 0..batch_count);
        }

        if last_mat != Some(mat_ptr) {
            enc.set_bind_group(1, Some(dc.material_bind_group.as_ref()), &[]);
            last_mat = Some(mat_ptr);
        }
        if last_vbuf != Some(vbuf_ptr) {
            enc.set_vertex_buffer(0, dc.vertex_buffer.slice(..));
            last_vbuf = Some(vbuf_ptr);
        }
        if last_ibuf != Some(ibuf_ptr) {
            enc.set_index_buffer(dc.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            last_ibuf = Some(ibuf_ptr);
        }

        if let Some(buf) = &dc.instance_buffer { kept.push(Arc::clone(buf)); }
        batch_start = dc.instance_buffer_offset;
        batch_count = 1;
        batch_idx   = idx;
    }

    if batch_count > 0 {
        let bdc      = &draw_calls[batch_idx];
        let inst_end = batch_start + batch_count as u64 * INSTANCE_STRIDE;
        enc.set_vertex_buffer(1, bdc.instance_buffer.as_ref().unwrap().slice(batch_start..inst_end));
        enc.draw_indexed(0..bdc.index_count, 0, 0..batch_count);
    }

    let bundle = enc.finish(&wgpu::RenderBundleDescriptor { label: Some("depth_prepass_bundle") });
    (bundle, kept)
}

/// Depth-only pass that writes depth for all opaque draw calls.
///
/// Uses Unreal-style `FParallelCommandListSet` approach: the sorted opaque draw
/// list is split into N chunks at pre-compile time, each chunk encoded into its
/// own `RenderBundle` on a worker thread.  At replay time all N bundles execute
/// in order via a single `execute_bundles()` call.
pub struct DepthPrepassPass {
    pipeline: Arc<wgpu::RenderPipeline>,
    device: Arc<wgpu::Device>,
    draw_list: Arc<Mutex<Vec<DrawCall>>>,
    /// Opaque draw indices sorted by (material, vertex_buffer, index_buffer, offset).
    /// Rebuilt when draw_list_generation changes; shared with GBufferPass.
    sorted_opaque_indices: Vec<usize>,
    /// Sorted index list published for GBufferPass after each rebuild.
    pub shared_sorted: Arc<Mutex<Vec<usize>>>,
    /// draw_list_generation value at the time sorted_opaque_indices was last built.
    sort_generation: u64,
    /// Pre-compiled RenderBundles (one per chunk); replayed each frame.
    cached_bundles: Vec<wgpu::RenderBundle>,
    /// draw_list_generation when cached_bundles were last compiled.
    bundle_gen: u64,
    /// Raw pointer identity of global / lighting bind groups embedded in the
    /// bundle.  If either changes (window resize → new bind group) the bundle
    /// must be recompiled so it references the correct object.
    bundle_global_bg_ptr: usize,
    bundle_lighting_bg_ptr: usize,
    /// Keeps `Arc<wgpu::Buffer>` clones alive while the bundle may be in flight.
    kept_arcs: Vec<Arc<wgpu::Buffer>>,
    /// Shared inbox for receiving pre-compiled bundles from parallel compilation.
    inbox: Arc<BundleInbox>,
}

impl DepthPrepassPass {
    pub fn new(
        pipeline: Arc<wgpu::RenderPipeline>,
        draw_list: Arc<Mutex<Vec<DrawCall>>>,
        device: Arc<wgpu::Device>,
        inbox: Arc<BundleInbox>,
    ) -> (Self, Arc<Mutex<Vec<usize>>>) {
        let shared_sorted = Arc::new(Mutex::new(Vec::new()));
        let pass = Self {
            pipeline,
            device,
            draw_list,
            sorted_opaque_indices: Vec::new(),
            shared_sorted: shared_sorted.clone(),
            sort_generation: u64::MAX,
            cached_bundles: Vec::new(),
            bundle_gen: u64::MAX,
            bundle_global_bg_ptr: 0,
            bundle_lighting_bg_ptr: 0,
            kept_arcs: Vec::new(),
            inbox,
        };
        (pass, shared_sorted)
    }

    /// (Re)compile the depth-prepass RenderBundle from the current sorted draw list.
    ///
    /// Sorts draws by (material, vertex_buffer, index_buffer, instance_buffer_offset)
    /// to minimise per-command state changes, then merges consecutive draws that share
    /// the same mesh + material and have contiguous instance-buffer slots into a single
    /// instanced draw call.
    fn compile_bundle(
        &mut self,
        draw_calls: &[DrawCall],
        global_bg: &wgpu::BindGroup,
        lighting_bg: &wgpu::BindGroup,
        generation: u64,
    ) {
        // ── Sort ──────────────────────────────────────────────────────────────
        self.sorted_opaque_indices.clear();
        self.sorted_opaque_indices.reserve(draw_calls.len());
        for (idx, dc) in draw_calls.iter().enumerate() {
            if !dc.transparent_blend && dc.instance_buffer.is_some() {
                self.sorted_opaque_indices.push(idx);
            }
        }
        self.sorted_opaque_indices.sort_unstable_by(|&ia, &ib| {
            let a = &draw_calls[ia];
            let b = &draw_calls[ib];
            (Arc::as_ptr(&a.material_bind_group) as usize)
                .cmp(&(Arc::as_ptr(&b.material_bind_group) as usize))
                .then_with(|| (Arc::as_ptr(&a.vertex_buffer) as usize)
                    .cmp(&(Arc::as_ptr(&b.vertex_buffer) as usize)))
                .then_with(|| (Arc::as_ptr(&a.index_buffer) as usize)
                    .cmp(&(Arc::as_ptr(&b.index_buffer) as usize)))
                .then_with(|| a.instance_buffer_offset.cmp(&b.instance_buffer_offset))
        });

        // Publish sort order for GBufferPass.
        {
            let mut shared = self.shared_sorted.lock().unwrap();
            shared.clear();
            shared.extend_from_slice(&self.sorted_opaque_indices);
        }
        self.sort_generation = generation;

        // ── Build bundle ──────────────────────────────────────────────────────
        let mut enc = self.device.create_render_bundle_encoder(
            &wgpu::RenderBundleEncoderDescriptor {
                label: Some("depth_prepass_bundle"),
                color_formats: &[],
                depth_stencil: Some(wgpu::RenderBundleDepthStencil {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_read_only: false,
                    stencil_read_only: true,
                }),
                sample_count: 1,
                multiview: None,
            },
        );
        enc.set_pipeline(&self.pipeline);
        enc.set_bind_group(0, global_bg, &[]);
        enc.set_bind_group(2, lighting_bg, &[]);

        let mut kept: Vec<Arc<wgpu::Buffer>> = Vec::with_capacity(self.sorted_opaque_indices.len());
        let mut last_mat:  Option<usize> = None;
        let mut last_vbuf: Option<usize> = None;
        let mut last_ibuf: Option<usize> = None;
        // Batch state: tracks a run of draws with the same mesh+material+contiguous offsets.
        let mut batch_start: u64  = 0;
        let mut batch_count: u32  = 0;
        let mut batch_idx:   usize = 0;

        for &idx in &self.sorted_opaque_indices {
            let dc = &draw_calls[idx];
            let mat_ptr  = Arc::as_ptr(&dc.material_bind_group) as usize;
            let vbuf_ptr = Arc::as_ptr(&dc.vertex_buffer)        as usize;
            let ibuf_ptr = Arc::as_ptr(&dc.index_buffer)         as usize;

            // Extend current batch: same mesh + material + contiguous instance slot.
            if batch_count > 0
                && last_mat  == Some(mat_ptr)
                && last_vbuf == Some(vbuf_ptr)
                && last_ibuf == Some(ibuf_ptr)
                && batch_start + batch_count as u64 * INSTANCE_STRIDE == dc.instance_buffer_offset
            {
                batch_count += 1;
                continue;
            }

            // Flush accumulated batch.
            if batch_count > 0 {
                let bdc      = &draw_calls[batch_idx];
                let inst_end = batch_start + batch_count as u64 * INSTANCE_STRIDE;
                enc.set_vertex_buffer(
                    1,
                    bdc.instance_buffer.as_ref().unwrap().slice(batch_start..inst_end),
                );
                enc.draw_indexed(0..bdc.index_count, 0, 0..batch_count);
            }

            // Emit state changes only when they actually differ.
            if last_mat != Some(mat_ptr) {
                enc.set_bind_group(1, Some(dc.material_bind_group.as_ref()), &[]);
                last_mat = Some(mat_ptr);
            }
            if last_vbuf != Some(vbuf_ptr) {
                enc.set_vertex_buffer(0, dc.vertex_buffer.slice(..));
                last_vbuf = Some(vbuf_ptr);
            }
            if last_ibuf != Some(ibuf_ptr) {
                enc.set_index_buffer(dc.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                last_ibuf = Some(ibuf_ptr);
            }

            if let Some(buf) = &dc.instance_buffer { kept.push(Arc::clone(buf)); }
            batch_start = dc.instance_buffer_offset;
            batch_count = 1;
            batch_idx   = idx;
        }

        // Flush final batch.
        if batch_count > 0 {
            let bdc      = &draw_calls[batch_idx];
            let inst_end = batch_start + batch_count as u64 * INSTANCE_STRIDE;
            enc.set_vertex_buffer(
                1,
                bdc.instance_buffer.as_ref().unwrap().slice(batch_start..inst_end),
            );
            enc.draw_indexed(0..bdc.index_count, 0, 0..batch_count);
        }

        let bundle = enc.finish(&wgpu::RenderBundleDescriptor { label: Some("depth_prepass_bundle") });
        eprintln!(
            "⚠️ [DepthPrepass] Bundle compiled (inline fallback): {} opaque draws (gen {})",
            self.sorted_opaque_indices.len(), generation,
        );

        self.cached_bundles = vec![bundle];
        self.kept_arcs     = kept;
        self.bundle_gen    = generation;
        self.bundle_global_bg_ptr   = global_bg   as *const _ as usize;
        self.bundle_lighting_bg_ptr = lighting_bg as *const _ as usize;
    }
}

impl RenderPass for DepthPrepassPass {
    fn name(&self) -> &str { "depth_prepass" }

    fn declare_resources(&self, builder: &mut PassResourceBuilder) {
        // Depth prepass writes to the shared depth buffer; GBuffer will load it.
        builder.write(ResourceHandle::named("depth"));
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let global_bg_ptr   = ctx.global_bind_group   as *const _ as usize;
        let lighting_bg_ptr = ctx.lighting_bind_group as *const _ as usize;

        // Recompile if the draw set or any embedded bind group changed.
        let need_rebuild = ctx.draw_list_generation != self.bundle_gen
            || global_bg_ptr   != self.bundle_global_bg_ptr
            || lighting_bg_ptr != self.bundle_lighting_bg_ptr;

        if need_rebuild {
            // Check inbox first — the Renderer may have pre-compiled in parallel
            // (Unreal-style FParallelCommandListSet).
            let precompiled = self.inbox.depth_prepass.lock().unwrap().take();
            if let Some(pre) = precompiled {
                self.cached_bundles     = pre.bundles;
                self.kept_arcs          = pre.kept_arcs;
                self.sorted_opaque_indices = pre.sorted_indices.clone();
                {
                    let mut shared = self.shared_sorted.lock().unwrap();
                    shared.clear();
                    shared.extend_from_slice(&pre.sorted_indices);
                }
                self.sort_generation        = pre.generation;
                self.bundle_gen             = pre.generation;
                self.bundle_global_bg_ptr   = pre.global_bg_ptr;
                self.bundle_lighting_bg_ptr = pre.lighting_bg_ptr;
                eprintln!(
                    "⚠️ [DepthPrepass] {} bundles from inbox: {} opaque draws (gen {})",
                    self.cached_bundles.len(), self.sorted_opaque_indices.len(), pre.generation,
                );
            } else {
                crate::profile_scope!("depth_prepass/compile");
                // Fallback: compile inline as a single bundle (no parallel pre-compile).
                let draw_calls: Vec<DrawCall> = self.draw_list.lock().unwrap().clone();
                self.compile_bundle(
                    &draw_calls,
                    ctx.global_bind_group,
                    ctx.lighting_bind_group,
                    ctx.draw_list_generation,
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

        {
            crate::profile_scope!("depth_prepass/replay");
            if !self.cached_bundles.is_empty() {
                pass.execute_bundles(&self.cached_bundles);
            }
        }

        Ok(())
    }
}
