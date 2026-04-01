//! Stub — water surface rendering is handled by helio-pass-water-sim.
use helio_v3::{PassContext, RenderPass, Result as HelioResult};

pub struct WaterSurfacePass;

impl RenderPass for WaterSurfacePass {
    fn name(&self) -> &'static str {
        "WaterSurface(stub)"
    }
    fn execute(&mut self, _ctx: &mut PassContext) -> HelioResult<()> {
        Ok(())
    }
}
