//! Forward transparent pass for alpha-blended materials.
//!
//! Draws objects flagged with `transparent_blend` after deferred lighting,
//! using the same PBR shader as forward geometry (`geometry.wgsl`) so lights,
//! shadows, environment and GI all affect transparent surfaces.
//!
//! ## Why no RenderBundle here
//!
//! Transparent objects must be sorted back-to-front from the camera every frame
//! for correct alpha blending.  Rebuilding a RenderBundle on every camera-movement
//! frame carries two costs that dominate in practice:
//!
//!   1. `benc.finish()` inside the pass execute — O(N transparent draws) CPU work
//!      every camera frame, showing up in `graph_ms` but invisible to GPU scopes.
//!   2. Driver re-validation of the new bundle object showing up in `encoder.finish()`
//!      / `queue.submit()` as erratic spikes only when the camera moves.
//!
//! After hardware-instancing the transparent draw count is typically tiny (one draw
//! per unique transparent material × texture combo), so recording them directly into
//! the RenderPass is O(~5–50) encoder commands per frame — negligible cost.

use std::sync::{Arc, Mutex};

use crate::graph::{PassContext, PassResourceBuilder, RenderPass, ResourceHandle};
use crate::mesh::{DrawCall, INSTANCE_STRIDE};
use crate::Result;

pub struct TransparentPass {
    pipeline: Arc<wgpu::RenderPipeline>,
    draw_list: Arc<Mutex<Vec<DrawCall>>>,
    sorted_transparent_indices: Vec<usize>,
    /// Camera position and forward at the time of the last sort.
    last_sort_cam_pos: glam::Vec3,
    last_sort_cam_fwd: glam::Vec3,
    /// draw_list_generation value when sorted_transparent_indices was last computed.
    last_sort_generation: u64,
    /// draw_calls.len() at the time sorted_transparent_indices was last computed.
    /// Guards against index-OOB when draw_list shrinks without a generation bump
    /// (e.g. frustum-visibility-only changes that don't add/evict proxies).
    last_sort_draw_len: usize,
}

impl TransparentPass {
    pub fn new(
        pipeline: Arc<wgpu::RenderPipeline>,
        draw_list: Arc<Mutex<Vec<DrawCall>>>,
    ) -> Self {
        Self {
            pipeline,
            draw_list,
            sorted_transparent_indices: Vec::new(),
            last_sort_cam_pos: glam::Vec3::ZERO,
            last_sort_cam_fwd: glam::Vec3::NEG_Z,
            last_sort_generation: u64::MAX,
            last_sort_draw_len: usize::MAX,
        }
    }
}

impl RenderPass for TransparentPass {
    fn name(&self) -> &str {
        "transparent"
    }

    fn declare_resources(&self, builder: &mut PassResourceBuilder) {
        builder.read(ResourceHandle::named("color_target"));
        builder.write(ResourceHandle::named("color_target"));
        builder.read(ResourceHandle::named("gbuffer"));
        // Ordering token consumed by overlay passes (billboards/debug overlays)
        // to ensure they execute after transparent composition.
        builder.write(ResourceHandle::named("transparent_done"));
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let draw_calls = self.draw_list.lock().unwrap();

        // Re-sort when the draw list structure or camera changes.  The sort
        // operates on a small index list (one entry per unique transparent
        // material batch, not per instance — so typically 5-30 entries for a
        // Minecraft-like world).  O(N log N) on 30 items is < 1 µs.
        let cam = ctx.camera_position;
        let fwd = ctx.camera_forward;
        let cam_moved = (cam - self.last_sort_cam_pos).length_squared() > 0.01
            || (fwd - self.last_sort_cam_fwd).length_squared() > 0.0001;
        // Re-sort if generation changed (visible set composition or proxy add/evict),
        // camera moved (back-to-front order changed), or draw_list has a different
        // number of entries than at last sort (safety net against index-OOB).
        let need_sort = ctx.draw_list_generation != self.last_sort_generation
            || cam_moved
            || draw_calls.len() != self.last_sort_draw_len;

        if need_sort {
            let _sort_t = std::time::Instant::now();
            self.sorted_transparent_indices.clear();
            self.sorted_transparent_indices.reserve(draw_calls.len());
            for (idx, dc) in draw_calls.iter().enumerate() {
                if dc.transparent_blend {
                    self.sorted_transparent_indices.push(idx);
                }
            }

            // Back-to-front for correct alpha blending.
            self.sorted_transparent_indices.sort_by(|&ia, &ib| {
                let a = &draw_calls[ia];
                let b = &draw_calls[ib];
                let da = (glam::Vec3::from(a.bounds_center) - cam).dot(fwd);
                let db = (glam::Vec3::from(b.bounds_center) - cam).dot(fwd);
                db.partial_cmp(&da)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        let pa = Arc::as_ptr(&a.vertex_buffer) as usize;
                        let pb = Arc::as_ptr(&b.vertex_buffer) as usize;
                        pa.cmp(&pb)
                    })
            });

            self.last_sort_cam_pos = cam;
            self.last_sort_cam_fwd = fwd;
            self.last_sort_generation = ctx.draw_list_generation;
            self.last_sort_draw_len  = draw_calls.len();
            let _sort_ms = _sort_t.elapsed().as_secs_f32() * 1000.0;
            if _sort_ms > 0.1 {
                eprintln!(
                    "⚠️ [Transparent] Sort: {} items — {:.2}ms",
                    self.sorted_transparent_indices.len(),
                    _sort_ms,
                );
            }
        }

        if self.sorted_transparent_indices.is_empty() {
            return Ok(());
        }

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

        let mut last_material: Option<usize> = None;
        for &idx in &self.sorted_transparent_indices {
            let dc = &draw_calls[idx];
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
