//! Stub — caustics rendering is handled by helio-pass-water-sim.
use helio_v3::{PassContext, RenderPass, Result as HelioResult};

pub struct WaterCausticsPass;

impl RenderPass for WaterCausticsPass {
    fn name(&self) -> &'static str {
        "WaterCaustics(stub)"
    }
    fn execute(&mut self, _ctx: &mut PassContext) -> HelioResult<()> {
        Ok(())
    }
}
