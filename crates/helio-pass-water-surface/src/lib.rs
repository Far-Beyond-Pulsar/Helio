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
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}
