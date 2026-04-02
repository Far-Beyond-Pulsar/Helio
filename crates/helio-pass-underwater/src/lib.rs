//! Stub — underwater rendering is handled by helio-pass-water-sim.
use helio_v3::{PassContext, RenderPass, Result as HelioResult};

pub struct UnderwaterPass;

impl RenderPass for UnderwaterPass {
    fn name(&self) -> &'static str {
        "Underwater(stub)"
    }
    fn execute(&mut self, _ctx: &mut PassContext) -> HelioResult<()> {
        Ok(())
    }
}
