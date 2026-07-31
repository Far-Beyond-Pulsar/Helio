//! Stub — caustics rendering is handled by helio-pass-water-sim.
use helio_core::{PassContext, RenderPass, Result as HelioResult};

pub struct WaterCausticsPass;

impl RenderPass for WaterCausticsPass {
    fn name(&self) -> &'static str {
        "WaterCaustics(stub)"
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn execute(&mut self, _ctx: &mut PassContext) -> HelioResult<()> {
        Ok(())
    }
}
