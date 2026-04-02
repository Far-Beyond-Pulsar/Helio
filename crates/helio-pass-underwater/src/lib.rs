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
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}
