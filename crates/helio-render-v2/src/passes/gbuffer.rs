//! G-buffer write pass — GPU-driven, pool-only.
//!
//! One `multi_draw_indexed_indirect` per unique material range. O(unique_materials) CPU cost.

use std::sync::{Arc, Mutex};
use crate::gpu_scene::MaterialRange;
use crate::graph::{RenderPass, PassContext, PassResourceBuilder, ResourceHandle};
use crate::Result;

pub struct GBufferTargets {
    pub albedo_view:   wgpu::TextureView,
    pub normal_view:   wgpu::TextureView,
    pub orm_view:      wgpu::TextureView,
    pub emissive_view: wgpu::TextureView,
}

pub struct GBufferPass {
    targets:               Arc<Mutex<GBufferTargets>>,
    pipeline:              Arc<wgpu::RenderPipeline>,
    pool_vertex_buffer:    Arc<wgpu::Buffer>,
    pool_index_buffer:     Arc<wgpu::Buffer>,
    shared_indirect:       Arc<Mutex<Option<Arc<wgpu::Buffer>>>>,
    shared_material_ranges: Arc<Mutex<Vec<MaterialRange>>>,
}

impl GBufferPass {
    pub fn new(
        targets:               Arc<Mutex<GBufferTargets>>,
        pipeline:              Arc<wgpu::RenderPipeline>,
        pool_vertex_buffer:    Arc<wgpu::Buffer>,
        pool_index_buffer:     Arc<wgpu::Buffer>,
        shared_indirect:       Arc<Mutex<Option<Arc<wgpu::Buffer>>>>,
        shared_material_ranges: Arc<Mutex<Vec<MaterialRange>>>,
    ) -> Self {
        Self { targets, pipeline, pool_vertex_buffer, pool_index_buffer, shared_indirect, shared_material_ranges }
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
        let indirect_buf = self.shared_indirect.lock().unwrap().clone();
        let Some(indirect_buf) = indirect_buf else {
            return Ok(());
        };

        let ranges = self.shared_material_ranges.lock().unwrap();

        let targets = self.targets.lock().unwrap();
        let color_attachments = [
            Some(wgpu::RenderPassColorAttachment {
                view: &targets.albedo_view, resolve_target: None, depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: &targets.normal_view, resolve_target: None, depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: &targets.orm_view, resolve_target: None, depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: &targets.emissive_view, resolve_target: None, depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            }),
        ];

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("G-Buffer Write Pass"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: ctx.depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, ctx.global_bind_group, &[]);
        pass.set_vertex_buffer(0, self.pool_vertex_buffer.slice(..));
        pass.set_index_buffer(self.pool_index_buffer.slice(..), wgpu::IndexFormat::Uint32);

        for range in ranges.iter() {
            let byte_offset = range.start as u64 * 20;
            pass.set_bind_group(1, Some(range.bind_group.as_ref()), &[]);
            // Use per-call draw_indexed_indirect rather than multi_draw_indexed_indirect.
            // multi_draw_indexed_indirect requires the multiDrawIndirect Vulkan device feature
            // which must be explicitly requested at device creation.
            for j in 0..range.count {
                pass.draw_indexed_indirect(&indirect_buf, byte_offset + j as u64 * 20);
            }
        }
        Ok(())
    }
}

