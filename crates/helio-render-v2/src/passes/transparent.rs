//! Forward transparent pass for alpha-blended materials.
//!
//! Draws objects flagged with `transparent_blend` after deferred lighting,
//! using the same PBR shader as forward geometry (`geometry.wgsl`) so lights,
//! shadows, environment and GI all affect transparent surfaces.

use std::sync::{Arc, Mutex};

use crate::graph::{PassContext, PassResourceBuilder, RenderPass, ResourceHandle};
use crate::mesh::DrawCall;
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

        // Skip re-sort when neither the draw list nor the camera has changed.
        // Back-to-front sort on 1300+ indices runs ~0.7 ms; skipping it on
        // static frames cuts per-frame CPU overhead significantly.
        let cam = ctx.camera_position;
        let fwd = ctx.camera_forward;
        let cam_moved = (cam - self.last_sort_cam_pos).length_squared() > 0.01
            || (fwd - self.last_sort_cam_fwd).length_squared() > 0.0001;
        let need_sort = ctx.draw_list_generation != self.last_sort_generation || cam_moved;

        if need_sort {
            self.sorted_transparent_indices.clear();
            self.sorted_transparent_indices.reserve(draw_calls.len());
            for (idx, dc) in draw_calls.iter().enumerate() {
                if dc.transparent_blend {
                    self.sorted_transparent_indices.push(idx);
                }
            }

            if self.sorted_transparent_indices.is_empty() {
                self.last_sort_cam_pos = cam;
                self.last_sort_cam_fwd = fwd;
                self.last_sort_generation = ctx.draw_list_generation;
                return Ok(());
            }

            // Back-to-front for standard alpha blending using view depth.
            self.sorted_transparent_indices.sort_by(|&ia, &ib| {
                let a = &draw_calls[ia];
                let b = &draw_calls[ib];
                let oa = glam::Vec3::from(a.bounds_center) - cam;
                let ob = glam::Vec3::from(b.bounds_center) - cam;

                let da = oa.dot(fwd);
                let db = ob.dot(fwd);

                db.partial_cmp(&da)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    // Deterministic fallback when depths match closely.
                    .then_with(|| {
                        let pa = Arc::as_ptr(&a.vertex_buffer) as usize;
                        let pb = Arc::as_ptr(&b.vertex_buffer) as usize;
                        pa.cmp(&pb)
                    })
            });

            self.last_sort_cam_pos = cam;
            self.last_sort_cam_fwd = fwd;
            self.last_sort_generation = ctx.draw_list_generation;
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
                // Transparent pass must still depth-test against opaque depth.
                // We load existing depth and keep it unchanged (pipeline has
                // depth_write_enabled = false).
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
            pass.set_vertex_buffer(1, dc.instance_buffer.as_ref().unwrap().slice(..));
            pass.set_index_buffer(dc.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..dc.index_count, 0, 0..dc.instance_count);
        }

        Ok(())
    }
}
