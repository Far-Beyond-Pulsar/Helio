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
        self.sorted_transparent_indices.clear();
        self.sorted_transparent_indices.reserve(
            draw_calls
                .len()
                .saturating_sub(self.sorted_transparent_indices.capacity()),
        );
        for (idx, dc) in draw_calls.iter().enumerate() {
            if dc.transparent_blend {
                self.sorted_transparent_indices.push(idx);
            }
        }

        if self.sorted_transparent_indices.is_empty() {
            return Ok(());
        }

        // Back-to-front for standard alpha blending using view depth.
        let cam = ctx.camera_position;
        let fwd = ctx.camera_forward;
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
            pass.set_index_buffer(dc.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..dc.index_count, 0, 0..1);
        }

        Ok(())
    }
}
