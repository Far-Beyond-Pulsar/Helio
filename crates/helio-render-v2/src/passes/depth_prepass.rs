//! Depth prepass — depth-only early-Z pass before the G-buffer.
//!
//! GPU-driven: single `multi_draw_indexed_indirect` for all opaque geometry
//! using the unified pool vertex/index buffers. O(1) CPU cost.

use std::sync::{Arc, Mutex};
use crate::gpu_scene::MaterialRange;
use crate::graph::{RenderPass, PassContext, PassResourceBuilder, ResourceHandle};
use crate::Result;

pub struct DepthPrepassPass {
    pipeline: Arc<wgpu::RenderPipeline>,
    pool_vertex_buffer: Arc<wgpu::Buffer>,
    pool_index_buffer: Arc<wgpu::Buffer>,
    shared_indirect: Arc<Mutex<Option<Arc<wgpu::Buffer>>>>,
    shared_material_ranges: Arc<Mutex<Vec<MaterialRange>>>,
    default_material_bg: Arc<wgpu::BindGroup>,
}

impl DepthPrepassPass {
    pub fn new(
        pipeline: Arc<wgpu::RenderPipeline>,
        pool_vertex_buffer: Arc<wgpu::Buffer>,
        pool_index_buffer: Arc<wgpu::Buffer>,
        shared_indirect: Arc<Mutex<Option<Arc<wgpu::Buffer>>>>,
        shared_material_ranges: Arc<Mutex<Vec<MaterialRange>>>,
        default_material_bg: Arc<wgpu::BindGroup>,
    ) -> Self {
        Self { pipeline, pool_vertex_buffer, pool_index_buffer, shared_indirect, shared_material_ranges, default_material_bg }
    }
}

impl RenderPass for DepthPrepassPass {
    fn name(&self) -> &str { "depth_prepass" }

    fn declare_resources(&self, builder: &mut PassResourceBuilder) {
        builder.write(ResourceHandle::named("depth"));
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let indirect_buf = self.shared_indirect.lock().unwrap().clone();
        let Some(indirect_buf) = indirect_buf else { return Ok(()); };

        let draw_count = {
            let ranges = self.shared_material_ranges.lock().unwrap();
            if ranges.is_empty() { return Ok(()); }
            ranges.iter().map(|r| r.count).sum::<u32>()
        };
        if draw_count == 0 { return Ok(()); }
        println!("[DepthPrepass] encoding {} draw_indexed_indirect calls", draw_count);

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
        pass.set_bind_group(1, Some(self.default_material_bg.as_ref()), &[]);
        pass.set_bind_group(2, ctx.lighting_bind_group, &[]);
        pass.set_vertex_buffer(0, self.pool_vertex_buffer.slice(..));
        pass.set_index_buffer(self.pool_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        // Use per-call draw_indexed_indirect rather than multi_draw_indexed_indirect.
        // multi_draw_indexed_indirect requires the multiDrawIndirect Vulkan device feature
        // which must be explicitly requested at device creation.
        for j in 0..draw_count {
            pass.draw_indexed_indirect(&indirect_buf, j as u64 * 20);
        }
        Ok(())
    }
}

