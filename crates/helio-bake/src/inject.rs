use std::sync::Arc;

use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use libhelio::FrameResources;

use crate::data::BakedData;

/// A render pass that publishes pre-baked GPU resources into `FrameResources` each frame.
///
/// This pass does **zero GPU work** — it purely stores `Arc`-wrapped references and
/// writes them into the frame resource bus in `publish()`.  
/// It is added to the render graph by the `Renderer` after a successful bake.
///
/// **Ordering**: insert this pass before `SsaoPass` and `DeferredLightPass` so that
/// those passes see the baked data in their `execute()` context.
pub struct BakeInjectPass {
    data: Arc<BakedData>,
}

impl BakeInjectPass {
    pub fn new(data: Arc<BakedData>) -> Self {
        Self { data }
    }

    /// Returns the underlying baked data (for post-bake injection into specific passes).
    pub fn baked_data(&self) -> &Arc<BakedData> {
        &self.data
    }
}

impl RenderPass for BakeInjectPass {
    fn name(&self) -> &'static str {
        "BakeInject"
    }

    fn publish<'a>(&'a self, frame: &mut FrameResources<'a>) {
        // AO — replaces SSAO slot so downstream passes (DeferredLight) see baked AO
        if let Some(ref view) = self.data.ao_view {
            frame.baked_ao = Some(view.as_ref());
        }
        if let Some(ref sampler) = self.data.ao_sampler {
            frame.baked_ao_sampler = Some(sampler.as_ref());
        }

        // Lightmap atlas
        if let Some(ref view) = self.data.lightmap_view {
            frame.baked_lightmap = Some(view.as_ref());
        }
        if let Some(ref sampler) = self.data.lightmap_sampler {
            frame.baked_lightmap_sampler = Some(sampler.as_ref());
        }

        // Reflection cubemap
        if let Some(ref view) = self.data.reflection_view {
            frame.baked_reflection = Some(view.as_ref());
        }
        if let Some(ref sampler) = self.data.reflection_sampler {
            frame.baked_reflection_sampler = Some(sampler.as_ref());
        }

        // Irradiance SH GPU buffer
        if let Some(ref buf) = self.data.irradiance_sh_buf {
            frame.baked_irradiance_sh = Some(buf.as_ref());
        }

        // PVS — CPU-side bitfield for visibility queries
        if let Some(ref pvs) = self.data.pvs {
            frame.baked_pvs = Some(libhelio::BakedPvsRef {
                world_min: pvs.world_min,
                world_max: pvs.world_max,
                grid_dims: pvs.grid_dims,
                cell_size: pvs.cell_size,
                cell_count: pvs.cell_count,
                words_per_cell: pvs.words_per_cell,
                bits: &pvs.bits,
            });
        }
    }

    fn prepare(&mut self, _ctx: &PrepareContext) -> HelioResult<()> {
        Ok(()) // nothing to upload
    }

    fn execute(&mut self, _ctx: &mut PassContext) -> HelioResult<()> {
        Ok(()) // no GPU commands
    }
}
