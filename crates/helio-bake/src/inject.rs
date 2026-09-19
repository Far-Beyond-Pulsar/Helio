use std::sync::Arc;

use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use helio_core::ResourceRegistry;

use crate::data::BakedData;

/// A render pass that publishes pre-baked GPU resources into `ResourceRegistry` each frame.
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

/// Extends a borrow from `self`'s own lifetime to the registry's `'a`.
///
/// Sound here specifically because `BakeInjectPass` only ever borrows out of
/// `self.data: Arc<BakedData>`, which is baking-run-lived (owned by the
/// `Renderer`/graph across many frames), never frame-scoped — so any single
/// frame's `'a` is always shorter than the data's real lifetime. The trait's
/// `publish(&self, ..)` (not `&'a self`) cannot express that relationship, so
/// the borrow checker sees an unrelated, shorter lifetime `'1` here instead;
/// this is the same lifetime-extension idiom `ResourceRegistry::
/// write_texture_binding` already uses for exactly this reason.
unsafe fn extend_lifetime<'a, T: ?Sized>(value: &T) -> &'a T {
    &*(value as *const T)
}

impl RenderPass for BakeInjectPass {
    fn name(&self) -> &'static str {
        "BakeInject"
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a helio_core::ResourceRegistry<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn publish<'a>(&self, frame: &mut ResourceRegistry<'a>) {
        // SAFETY: every borrow below is extended per `extend_lifetime`'s doc —
        // out of `self.data: Arc<BakedData>`, which outlives any single frame.
        // AO — replaces SSAO slot so downstream passes (DeferredLight) see baked AO
        if let Some(ref view) = self.data.ao_view {
            frame.write(helio_core::ResourceKey::new("baked_ao"), unsafe { extend_lifetime(view.as_ref()) }, "BakeInject");
        }
        if let Some(ref sampler) = self.data.ao_sampler {
            frame.write(helio_core::ResourceKey::new("baked_ao_sampler"), unsafe { extend_lifetime(sampler.as_ref()) }, "BakeInject");
        }

        // Lightmap atlas
        if let Some(ref view) = self.data.lightmap_view {
            frame.write(helio_core::ResourceKey::new("baked_lightmap"), unsafe { extend_lifetime(view.as_ref()) }, "BakeInject");
        }
        if let Some(ref sampler) = self.data.lightmap_sampler {
            frame.write(helio_core::ResourceKey::new("baked_lightmap_sampler"), unsafe { extend_lifetime(sampler.as_ref()) }, "BakeInject");
        }

        // Reflection cubemap
        if let Some(ref view) = self.data.reflection_view {
            frame.write(helio_core::ResourceKey::new("baked_reflection"), unsafe { extend_lifetime(view.as_ref()) }, "BakeInject");
        }
        if let Some(ref sampler) = self.data.reflection_sampler {
            frame.write(helio_core::ResourceKey::new("baked_reflection_sampler"), unsafe { extend_lifetime(sampler.as_ref()) }, "BakeInject");
        }

        // Irradiance SH GPU buffer
        if let Some(ref buf) = self.data.irradiance_sh_buf {
            frame.write(helio_core::ResourceKey::new("baked_irradiance_sh"), unsafe { extend_lifetime(buf.as_ref()) }, "BakeInject");
        }

        // PVS — CPU-side bitfield for visibility queries
        if let Some(ref pvs) = self.data.pvs {
            frame.write(helio_core::ResourceKey::new("baked_pvs"),
                helio_bake_types::BakedPvsRef {
                    world_min: pvs.world_min,
                    world_max: pvs.world_max,
                    grid_dims: pvs.grid_dims,
                    cell_size: pvs.cell_size,
                    cell_count: pvs.cell_count,
                    words_per_cell: pvs.words_per_cell,
                    bits: unsafe { extend_lifetime(&pvs.bits) },
                },
                "BakeInject",
            );
        }
    }

    fn prepare(&mut self, _ctx: &PrepareContext) -> HelioResult<()> {
        Ok(()) // nothing to upload
    }

    fn execute(&mut self, _ctx: &mut PassContext) -> HelioResult<()> {
        Ok(()) // no GPU commands
    }
}
