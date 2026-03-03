//! Sky pass – renders a physically-based atmospheric sky before geometry.
//!
//! Uses Nishita single-scatter Rayleigh+Mie scattering and optional FBM
//! volumetric clouds. The pass is a full-screen triangle (no vertex buffer)
//! that writes to the colour attachment only (depth untouched so the geometry
//! pass can clear depth normally on top).

use crate::graph::{RenderPass, PassContext, PassResourceBuilder, ResourceHandle};
use crate::Result;
use std::sync::Arc;

/// Renders a physical sky into the colour target before the geometry pass.
pub struct SkyPass {
    pipeline:       Arc<wgpu::RenderPipeline>,
    /// Bind group (group 1) carrying the SkyUniform buffer.  Updated each frame
    /// by the renderer via a shared `Arc`.
    sky_bind_group: Arc<wgpu::BindGroup>,
}

impl SkyPass {
    pub fn new(
        pipeline:       Arc<wgpu::RenderPipeline>,
        sky_bind_group: Arc<wgpu::BindGroup>,
    ) -> Self {
        Self { pipeline, sky_bind_group }
    }
}

impl RenderPass for SkyPass {
    fn name(&self) -> &str { "sky" }

    fn declare_resources(&self, builder: &mut PassResourceBuilder) {
        // sky_layer is an ordering token read by GeometryPass
        builder.write(ResourceHandle::named("sky_layer"));
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        // Skip entirely when no atmosphere is configured this frame
        if !ctx.has_sky { return Ok(()); }

        let color_attachment = Some(wgpu::RenderPassColorAttachment {
            view:           ctx.target,
            resolve_target: None,
            depth_slice:    None,
            ops: wgpu::Operations {
                // Sky IS the clear – we write every pixel
                load:  wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        });

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label:                    Some("Sky Pass"),
            color_attachments:        &[color_attachment],
            // No depth attachment – geometry pass clears depth on its own
            depth_stencil_attachment: None,
            timestamp_writes:         None,
            occlusion_query_set:      None,
            multiview_mask:           None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, ctx.global_bind_group, &[]);
        pass.set_bind_group(1, Some(self.sky_bind_group.as_ref()), &[]);

        // Full-screen triangle, no vertex buffer required
        pass.draw(0..3, 0..1);

        Ok(())
    }
}
