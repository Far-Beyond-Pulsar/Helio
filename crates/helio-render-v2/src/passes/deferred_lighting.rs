//! Deferred lighting pass.
//!
//! Runs a fullscreen triangle that reads the G-buffer written by GBufferPass
//! and evaluates all lights, shadows, RC GI and tonemapping in a single
//! screen-space draw.  The G-buffer bind group (group 1) is held behind
//! `Arc<Mutex<Arc<wgpu::BindGroup>>>` so the Renderer can swap in a new
//! bind group when the window is resized without rebuilding the graph.

use std::sync::{Arc, Mutex};
use crate::graph::{RenderPass, PassContext, PassResourceBuilder, ResourceHandle};
use crate::Result;

pub struct DeferredLightingPass {
    /// Group 1: G-buffer textures + depth.  Swapped on resize.
    gbuffer_bg: Arc<Mutex<Arc<wgpu::BindGroup>>>,
    pipeline:   Arc<wgpu::RenderPipeline>,
}

impl DeferredLightingPass {
    pub fn new(
        gbuffer_bg: Arc<Mutex<Arc<wgpu::BindGroup>>>,
        pipeline:   Arc<wgpu::RenderPipeline>,
    ) -> Self {
        Self { gbuffer_bg, pipeline }
    }
}

impl RenderPass for DeferredLightingPass {
    fn name(&self) -> &str { "deferred_lighting" }

    fn declare_resources(&self, builder: &mut PassResourceBuilder) {
        // Reads G-buffer written by GBufferPass
        builder.read(ResourceHandle::named("gbuffer"));
        // Reads lighting data (already written before geometry)
        builder.read(ResourceHandle::named("shadow_atlas"));
        builder.read(ResourceHandle::named("rc_cascade0"));
        // Reads sky background (LoadOp::Load preserves it)
        builder.read(ResourceHandle::named("sky_layer"));
        // Writes to the final colour target
        builder.write(ResourceHandle::named("color_target"));
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let gbuffer_bg = self.gbuffer_bg.lock().unwrap();

        // Preserve the sky/background rendered by SkyPass via LoadOp::Load.
        // Pixels that map to depth=1 (sky) are discarded in the shader so the
        // sky background is untouched.
        let color_attach = Some(wgpu::RenderPassColorAttachment {
            view: ctx.target,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load:  wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        });

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Deferred Lighting Pass"),
            color_attachments: &[color_attach],
            depth_stencil_attachment: None,  // reads depth from G-buffer; never writes
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, ctx.global_bind_group,   &[]);   // camera + globals
        pass.set_bind_group(1, Some(&**gbuffer_bg),    &[]);   // G-buffer textures
        pass.set_bind_group(2, ctx.lighting_bind_group,  &[]);   // lights, shadows, env

        // Draw the fullscreen triangle.  The vertex shader generates positions
        // procedurally from vertex_index — no vertex buffer needed.
        pass.draw(0..3, 0..1);

        Ok(())
    }
}
