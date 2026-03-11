//! SDF fullscreen ray march render pass

use crate::Result;
use crate::graph::{RenderPass, PassContext, PassResourceBuilder, ResourceHandle};
use std::sync::Arc;

/// Fullscreen render pass that sphere-traces through the SDF 3D texture.
pub struct SdfRayMarchPass {
    pipeline: Arc<wgpu::RenderPipeline>,
    bind_group: Arc<wgpu::BindGroup>,
}

impl SdfRayMarchPass {
    pub fn new(
        pipeline: Arc<wgpu::RenderPipeline>,
        bind_group: Arc<wgpu::BindGroup>,
    ) -> Self {
        Self { pipeline, bind_group }
    }

    /// Replace the bind group (called when volumes change)
    pub fn set_bind_group(&mut self, bind_group: Arc<wgpu::BindGroup>) {
        self.bind_group = bind_group;
    }
}

impl RenderPass for SdfRayMarchPass {
    fn name(&self) -> &str { "sdf_ray_march" }

    fn declare_resources(&self, builder: &mut PassResourceBuilder) {
        builder.read(ResourceHandle::named("sdf_volume"));
        // "transparent_done" is a single-writer ordering token written by TransparentPass.
        // Using it guarantees this pass runs AFTER both deferred_lighting and transparent,
        // even though features register before those passes in the graph build order.
        // (Multi-writer resources like "color_target" only resolve from prior registrations.)
        builder.read(ResourceHandle::named("transparent_done"));
        builder.write(ResourceHandle::named("sdf_layer"));
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SDF Ray March"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
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
        pass.set_bind_group(1, &*self.bind_group, &[]);
        pass.draw(0..3, 0..1); // fullscreen triangle

        Ok(())
    }
}
