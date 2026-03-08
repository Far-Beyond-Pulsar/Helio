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
    device: Arc<wgpu::Device>,
    pipeline: Arc<wgpu::RenderPipeline>,
    draw_list: Arc<Mutex<Vec<DrawCall>>>,
    sorted_opaque_indices: Vec<usize>,
    /// Sorted opaque index list shared with GBufferPass so the front-to-back
    /// ordering is computed once here and read directly by GBuffer.
    pub shared_sorted: Arc<Mutex<Vec<usize>>>,
    /// Cached RenderBundle.  A bundle pre-compiles all draw commands so the
    /// main encoder only records a single execute_bundles command, reducing
    /// encoder.finish() cost from O(N draws) to O(1).
    bundle_cache: Option<wgpu::RenderBundle>,
    /// FNV hash of the opaque draw set (vertex_buf_ptr × instance_buf_ptr ×
    /// instance_count per call).  The bundle is valid when this matches the
    /// current draw list.  Safe because all draw calls reference per-batch
    /// stable Arc<Buffer> at offset 0 — the hash only changes on structural
    /// identity changes, not on transform data updates (write_buffer handles those).
    bundle_geom_hash: u64,
    /// Incremented each execute() call.  Used to rate-limit re-encodes so a
    /// burst of new chunks during streaming doesn't force a full rebuild every
    /// single frame.
    frame_counter: u64,
    /// frame_counter value when the bundle was last fully re-encoded.
    bundle_last_rebuild: u64,
}

impl DepthPrepassPass {
    pub fn new(
        device: Arc<wgpu::Device>,
        pipeline: Arc<wgpu::RenderPipeline>,
        draw_list: Arc<Mutex<Vec<DrawCall>>>,
    ) -> (Self, Arc<Mutex<Vec<usize>>>) {
        let shared_sorted = Arc::new(Mutex::new(Vec::new()));
        let pass = Self {
            device,
            pipeline,
            draw_list,
            sorted_opaque_indices: Vec::new(),
            shared_sorted: shared_sorted.clone(),
            bundle_cache: None,
            bundle_geom_hash: u64::MAX,
            frame_counter: 0,
            bundle_last_rebuild: 0,
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
        self.frame_counter += 1;
        let draw_calls = self.draw_list.lock().unwrap();

        // Compute geometric hash of the current opaque draw set.
        // Covers (vertex_buf_ptr, instance_buf_ptr, instance_count) — NOT transform
        // values.  Changes only when the opaque set identity changes (new batch, removed
        // batch, instance count resized).  Safe to use because draw calls reference
        // per-batch stable Arc<Buffer> at offset 0: existing buffer slots are never
        // shifted by the addition of unrelated batches.
        let new_geom_hash = {
            let mut h: u64 = 0xcbf29ce484222325;
            for dc in draw_calls.iter() {
                if dc.transparent_blend { continue; }
                h ^= Arc::as_ptr(&dc.vertex_buffer) as u64;
                h = h.wrapping_mul(0x100000001b3);
                h ^= dc.instance_count as u64;
                h = h.wrapping_mul(0x100000001b3);
                if let Some(b) = &dc.instance_buffer {
                    h ^= Arc::as_ptr(b) as u64;
                    h = h.wrapping_mul(0x100000001b3);
                }
            }
            h
        };

        let set_changed = new_geom_hash != self.bundle_geom_hash;

        // Re-sort and publish to GBufferPass whenever the opaque set changes.
        // Sort key is material-pointer (camera-independent, O(N log N) on cheap key).
        if set_changed {
            self.sorted_opaque_indices.clear();
            self.sorted_opaque_indices.reserve(draw_calls.len());
            for (idx, dc) in draw_calls.iter().enumerate() {
                if !dc.transparent_blend {
                    self.sorted_opaque_indices.push(idx);
                }
            }
            self.sorted_opaque_indices.sort_unstable_by_key(|&i| {
                Arc::as_ptr(&draw_calls[i].material_bind_group) as usize
            });
            {
                let mut shared = self.shared_sorted.lock().unwrap();
                shared.clear();
                shared.extend_from_slice(&self.sorted_opaque_indices);
            }
        }

        // Rate-limited bundle re-encode.  With per-batch stable buffers the stale
        // bundle is safe to replay — existing batches' Arc<Buffer> references remain
        // valid.  New chunks added since the last rebuild are absent from the bundle
        // for at most ~4 frames (≈67 ms at 60 fps), which is imperceptible.
        let can_rebuild = self.bundle_cache.is_none()
            || self.frame_counter.saturating_sub(self.bundle_last_rebuild) >= 4;

        if set_changed && can_rebuild {
            let _bundle_t = std::time::Instant::now();
            let mut benc = self.device.create_render_bundle_encoder(
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
            benc.set_pipeline(&self.pipeline);
            benc.set_bind_group(0, ctx.global_bind_group, &[]);
            benc.set_bind_group(2, ctx.lighting_bind_group, &[]);
            let mut last_material: Option<usize> = None;
            for &idx in &self.sorted_opaque_indices {
                let dc = &draw_calls[idx];
                let mat_ptr = Arc::as_ptr(&dc.material_bind_group) as usize;
                if last_material != Some(mat_ptr) {
                    benc.set_bind_group(1, Some(dc.material_bind_group.as_ref()), &[]);
                    last_material = Some(mat_ptr);
                }
                benc.set_vertex_buffer(0, dc.vertex_buffer.slice(..));
                let inst_start = dc.instance_buffer_offset;
                let inst_end   = inst_start + dc.instance_count as u64 * INSTANCE_STRIDE;
                benc.set_vertex_buffer(1, dc.instance_buffer.as_ref().unwrap().slice(inst_start..inst_end));
                benc.set_index_buffer(dc.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                benc.draw_indexed(0..dc.index_count, 0, 0..dc.instance_count);
            }
            self.bundle_cache        = Some(benc.finish(&wgpu::RenderBundleDescriptor { label: None }));
            self.bundle_geom_hash    = new_geom_hash;
            self.bundle_last_rebuild = self.frame_counter;
            eprintln!(
                "⚠️ [DepthPrepass] Bundle rebuild: {} draw calls — {:.2}ms",
                self.sorted_opaque_indices.len(),
                _bundle_t.elapsed().as_secs_f32() * 1000.0,
            );
        }
        drop(draw_calls);

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
        if let Some(bundle) = &self.bundle_cache {
            pass.execute_bundles(std::iter::once(bundle));
        }
        Ok(())
    }
}
