//! G-buffer write pass.
//!
//! Renders scene geometry into four Rgba G-buffer textures + the shared depth
//! buffer.  No lighting is evaluated here — that happens in the subsequent
//! DeferredLightingPass.
//!
//! ## Performance design
//!
//! Same RenderBundle caching strategy as DepthPrepassPass: the bundle is
//! compiled once per scene change and replayed at ~0 µs CPU cost every other
//! frame.  The sorted draw order is shared from DepthPrepassPass (already
//! computed this frame when DepthPrepass runs first in the graph).
//!
//! G-buffer formats:
//!   albedo   Rgba8Unorm
//!   normals  Rgba16Float
//!   ORM      Rgba8Unorm
//!   emissive Rgba16Float
//!   depth    Depth32Float (loaded from depth-prepass, not cleared)

use std::sync::{Arc, Mutex};
use crate::graph::{RenderPass, PassContext, PassResourceBuilder, ResourceHandle};
use crate::mesh::{DrawCall, INSTANCE_STRIDE};
use crate::passes::depth_prepass::BundleInbox;
use crate::Result;

/// The four G-buffer render targets.  Wrapped in Arc<Mutex<…>> so the Renderer
/// can swap in new views when the window is resized without rebuilding the graph.
pub struct GBufferTargets {
    pub albedo_view:   wgpu::TextureView,   // Rgba8Unorm
    pub normal_view:   wgpu::TextureView,   // Rgba16Float
    pub orm_view:      wgpu::TextureView,   // Rgba8Unorm
    pub emissive_view: wgpu::TextureView,   // Rgba16Float
}

/// Build a G-buffer RenderBundle from pre-sorted draw indices (chunk).
/// Thread-safe: only needs shared references.
/// Called from the parallel pre-compile step (Unreal-style FParallelCommandListSet).
pub fn build_gbuffer_bundle(
    device: &wgpu::Device,
    pipeline: &wgpu::RenderPipeline,
    draw_calls: &[DrawCall],
    sorted: &[usize],
    global_bg: &wgpu::BindGroup,
) -> (wgpu::RenderBundle, Vec<Arc<wgpu::Buffer>>) {
    let mut enc = device.create_render_bundle_encoder(
        &wgpu::RenderBundleEncoderDescriptor {
            label: Some("gbuffer_bundle"),
            color_formats: &[
                Some(wgpu::TextureFormat::Rgba8Unorm),   // albedo
                Some(wgpu::TextureFormat::Rgba16Float),  // normals
                Some(wgpu::TextureFormat::Rgba8Unorm),   // ORM
                Some(wgpu::TextureFormat::Rgba16Float),  // emissive
            ],
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

    let bundle = enc.finish(&wgpu::RenderBundleDescriptor { label: Some("gbuffer_bundle") });
    (bundle, kept)
}

pub struct GBufferPass {
    targets:   Arc<Mutex<GBufferTargets>>,
    pipeline:  Arc<wgpu::RenderPipeline>,
    device:    Arc<wgpu::Device>,
    draw_list: Arc<Mutex<Vec<DrawCall>>>,
    /// Pre-computed sort shared from DepthPrepassPass (same sort order).
    shared_sorted: Arc<Mutex<Vec<usize>>>,
    /// draw_list_generation value at the time cached_bundles were compiled.
    bundle_gen: u64,
    /// Raw pointer of the global bind group embedded in the bundle.
    bundle_global_bg_ptr: usize,
    /// Pre-compiled RenderBundles (one per chunk) replayed each frame.
    cached_bundles: Vec<wgpu::RenderBundle>,
    /// Keeps `Arc<wgpu::Buffer>` clones alive while the bundle may be in flight.
    kept_arcs: Vec<Arc<wgpu::Buffer>>,
    /// Shared inbox for receiving pre-compiled bundles from parallel compilation.
    inbox: Arc<BundleInbox>,
}

impl GBufferPass {
    pub fn new(
        targets:       Arc<Mutex<GBufferTargets>>,
        pipeline:      Arc<wgpu::RenderPipeline>,
        draw_list:     Arc<Mutex<Vec<DrawCall>>>,
        shared_sorted: Arc<Mutex<Vec<usize>>>,
        device:        Arc<wgpu::Device>,
        inbox:         Arc<BundleInbox>,
    ) -> Self {
        Self {
            targets,
            pipeline,
            device,
            draw_list,
            shared_sorted,
            bundle_gen: u64::MAX,
            bundle_global_bg_ptr: 0,
            cached_bundles: Vec::new(),
            kept_arcs: Vec::new(),
            inbox,
        }
    }

    /// (Re)compile the G-buffer RenderBundle using the draw order already
    /// published by DepthPrepassPass into `shared_sorted`.  Falls back to a
    /// self-built sort when shared_sorted is empty (first frame / graph race).
    fn compile_bundle(
        &mut self,
        draw_calls: &[DrawCall],
        global_bg: &wgpu::BindGroup,
        generation: u64,
    ) {
        // ── Get sorted indices (from DepthPrepass or fall-back self-sort) ────
        let sorted: Vec<usize> = {
            let shared = self.shared_sorted.lock().unwrap();
            if !shared.is_empty() {
                shared.clone()
            } else {
                // Fallback: sort ourselves (same key as DepthPrepassPass).
                let mut indices: Vec<usize> = (0..draw_calls.len())
                    .filter(|&i| !draw_calls[i].transparent_blend && draw_calls[i].instance_buffer.is_some())
                    .collect();
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
        };

        // ── Build bundle ──────────────────────────────────────────────────────
        let mut enc = self.device.create_render_bundle_encoder(
            &wgpu::RenderBundleEncoderDescriptor {
                label: Some("gbuffer_bundle"),
                color_formats: &[
                    Some(wgpu::TextureFormat::Rgba8Unorm),   // albedo
                    Some(wgpu::TextureFormat::Rgba16Float),  // normals
                    Some(wgpu::TextureFormat::Rgba8Unorm),   // ORM
                    Some(wgpu::TextureFormat::Rgba16Float),  // emissive
                ],
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

        let mut kept: Vec<Arc<wgpu::Buffer>> = Vec::with_capacity(sorted.len());
        let mut last_mat:  Option<usize> = None;
        let mut last_vbuf: Option<usize> = None;
        let mut last_ibuf: Option<usize> = None;
        let mut batch_start: u64  = 0;
        let mut batch_count: u32  = 0;
        let mut batch_idx:   usize = 0;

        for &idx in &sorted {
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

            // State changes only when they differ.
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

        let bundle = enc.finish(&wgpu::RenderBundleDescriptor { label: Some("gbuffer_bundle") });
        eprintln!(
            "⚠️ [GBuffer] Bundle compiled (inline fallback): {} opaque draws (gen {})",
            sorted.len(), generation,
        );

        self.cached_bundles     = vec![bundle];
        self.kept_arcs          = kept;
        self.bundle_gen         = generation;
        self.bundle_global_bg_ptr = global_bg as *const _ as usize;
    }
}

impl RenderPass for GBufferPass {
    fn name(&self) -> &str { "gbuffer" }

    fn declare_resources(&self, builder: &mut PassResourceBuilder) {
        builder.read(ResourceHandle::named("shadow_atlas"));
        builder.read(ResourceHandle::named("rc_cascade0"));
        builder.read(ResourceHandle::named("sky_layer"));
        builder.write(ResourceHandle::named("gbuffer"));
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let global_bg_ptr = ctx.global_bind_group as *const _ as usize;

        // Recompile if draw set or embedded bind group changed.
        let need_rebuild = ctx.draw_list_generation != self.bundle_gen
            || global_bg_ptr != self.bundle_global_bg_ptr;

        if need_rebuild {
            // Check inbox first — the Renderer may have pre-compiled in parallel
            // (Unreal-style FParallelCommandListSet).
            let precompiled = self.inbox.gbuffer.lock().unwrap().take();
            if let Some(pre) = precompiled {
                self.cached_bundles     = pre.bundles;
                self.kept_arcs          = pre.kept_arcs;
                self.bundle_gen         = pre.generation;
                self.bundle_global_bg_ptr = pre.global_bg_ptr;
                eprintln!(
                    "⚠️ [GBuffer] {} bundles from inbox (gen {})",
                    self.cached_bundles.len(), pre.generation,
                );
            } else {
                crate::profile_scope!("gbuffer/compile");
                // Fallback: compile inline as a single bundle (no parallel pre-compile).
                let draw_calls: Vec<DrawCall> = self.draw_list.lock().unwrap().clone();
                self.compile_bundle(&draw_calls, ctx.global_bind_group, ctx.draw_list_generation);
            }
        }

        let targets = self.targets.lock().unwrap();

        let color_attachments = [
            Some(wgpu::RenderPassColorAttachment {
                view: &targets.albedo_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: &targets.normal_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: &targets.orm_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: &targets.emissive_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }),
        ];

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("G-Buffer Write Pass"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: ctx.depth_view,
                // Load depth written by DepthPrepassPass so the GPU can reject
                // occluded GBuffer fragments before running the material shader.
                depth_ops: Some(wgpu::Operations {
                    load:  wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        {
            crate::profile_scope!("gbuffer/replay");
            if !self.cached_bundles.is_empty() {
                pass.execute_bundles(&self.cached_bundles);
            }
        }

        Ok(())
    }
}
