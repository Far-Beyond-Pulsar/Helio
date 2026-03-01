//! Shadow depth pass - renders depth into the shadow atlas

use crate::graph::{RenderPass, PassContext};
use crate::Result;
use std::sync::Arc;

/// Depth-only pass that fills shadow atlas layers
///
/// One render sub-pass per shadow-casting light.  Each sub-pass clears and
/// re-draws all shadow casters into that light's atlas layer.
///
/// Currently this just clears the atlas to ensure no stale depth values;
/// actual shadow-caster geometry submission will be wired up alongside the
/// scene mesh system.
pub struct ShadowPass {
    /// One view per shadow-casting light (D2, single array layer each)
    layer_views: Vec<Arc<wgpu::TextureView>>,
}

impl ShadowPass {
    pub fn new(layer_views: Vec<Arc<wgpu::TextureView>>) -> Self {
        Self { layer_views }
    }
}

impl RenderPass for ShadowPass {
    fn name(&self) -> &str {
        "shadow"
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        for (i, view) in self.layer_views.iter().enumerate() {
            let _pass = ctx.begin_render_pass(
                &format!("Shadow Pass Layer {i}"),
                &[],
                Some(wgpu::RenderPassDepthStencilAttachment {
                    view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
            );
            // TODO: draw shadow-caster geometry per light
        }
        Ok(())
    }
}
