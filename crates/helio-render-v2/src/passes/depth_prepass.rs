//! Depth prepass — depth-only early-Z pass before the G-buffer.
//!
//! GPU-driven: single `multi_draw_indexed_indirect` for all opaque geometry
//! using the unified pool vertex/index buffers. O(1) CPU cost.

use std::sync::{Arc, Mutex};
use crate::buffer_pool::SharedPoolBuffer;
use crate::gpu_scene::MaterialRange;
use crate::graph::{RenderPass, PassContext, PassResourceBuilder, ResourceHandle};
use crate::Result;

pub struct DepthPrepassPass {
    pipeline: Arc<wgpu::RenderPipeline>,
    pool_vertex_buffer: SharedPoolBuffer,
    pool_index_buffer: SharedPoolBuffer,
    shared_indirect: Arc<Mutex<Option<Arc<wgpu::Buffer>>>>,
    shared_material_ranges: Arc<Mutex<Vec<MaterialRange>>>,
    default_material_bg: Arc<wgpu::BindGroup>,
    has_multi_draw: bool,
}

impl DepthPrepassPass {
    pub fn new(
        pipeline: Arc<wgpu::RenderPipeline>,
        pool_vertex_buffer: SharedPoolBuffer,
        pool_index_buffer: SharedPoolBuffer,
        shared_indirect: Arc<Mutex<Option<Arc<wgpu::Buffer>>>>,
        shared_material_ranges: Arc<Mutex<Vec<MaterialRange>>>,
        default_material_bg: Arc<wgpu::BindGroup>,
        has_multi_draw: bool,
    ) -> Self {
        Self { pipeline, pool_vertex_buffer, pool_index_buffer, shared_indirect, shared_material_ranges, default_material_bg, has_multi_draw }
    }
}

impl RenderPass for DepthPrepassPass {
    fn name(&self) -> &str { "depth_prepass" }

    fn declare_resources(&self, builder: &mut PassResourceBuilder) {
        builder.write(ResourceHandle::named("depth"));
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        // Always clear the depth buffer to 1.0, even when there are no mesh
        // draw calls.  Other passes (e.g. SDF ray march) load the depth buffer
        // with CompareFunction::Less and depend on a clean far-plane value.
        let indirect_buf = self.shared_indirect.lock().unwrap().clone();
        let draw_count = indirect_buf.as_ref().map(|_| {
            let ranges = self.shared_material_ranges.lock().unwrap();
            ranges.iter().map(|r| r.count).sum::<u32>()
        }).unwrap_or(0);

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

        if let Some(indirect_buf) = &indirect_buf {
            if draw_count > 0 {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, ctx.global_bind_group, &[]);
                pass.set_bind_group(1, Some(self.default_material_bg.as_ref()), &[]);
                pass.set_bind_group(2, ctx.lighting_bind_group, &[]);
                let vb = self.pool_vertex_buffer.lock().unwrap().clone();
                let ib = self.pool_index_buffer .lock().unwrap().clone();
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                if self.has_multi_draw {
                    pass.multi_draw_indexed_indirect(indirect_buf, 0, draw_count);
                } else {
                    for j in 0..draw_count {
                        pass.draw_indexed_indirect(indirect_buf, j as u64 * 20);
                    }
                }
            }
        }
        Ok(())
    }
}

