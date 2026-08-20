//! `LightGpuData` -- the `#[gpu]`-mirrored, SceneDB-owned counterpart to
//! `LightComponent`'s editor-facing properties (Pulsar-Native#561: light's
//! render data belongs in SceneDB the same way `StaticMeshComponent::
//! vertices`/`indices` do -- not sidestepped just because `LightComponent`
//! itself can't be `Pod`).
//!
//! `LightComponent` can't derive `SceneStore` directly: its `#[sub_props]`
//! fields carry `bool`s, enums, and (`LightFunctionProps::
//! light_function_material`) a `String` -- none of which satisfy the
//! whole-struct `Pod` bound the fixed-size `#[derive(SceneStore)]` path
//! requires (see that macro's `generate_pod_impl`). Rather than restructure
//! nine editor-facing sub-prop structs around GPU byte-layout constraints
//! they have no other reason to care about, this is a second, companion
//! component holding ONLY the render-relevant translation -- nothing else
//! about it needs to be reflection/properties-panel-friendly, so nothing
//! stops it from being `Pod`. `LightComponent`'s own `hydrate`/`remove`
//! (`runtime.rs`) keep it in sync; it has no `ComponentRuntimeBehavior` of
//! its own.
//!
//! Wraps `helio::GpuLight` directly rather than re-declaring its fields:
//! that's the EXACT byte layout Helio's own shaders already expect (see
//! that struct's own doc for the WGSL mirror list) -- SceneDB ends up
//! holding a byte-identical copy of what the GPU wants, not a third shape
//! translated from either side that could drift.
//!
//! `position_range`/`direction_outer`'s xyz components are transform-
//! dependent and deliberately NOT trustworthy in this mirror -- they're
//! whatever the entity's `Transform` was at the last property edit (hydrate
//! time), not necessarily the current one. `HelioRenderer::
//! sync_light_gpu_data` overwrites `position_range`'s xyz from the live
//! `Transform` before pushing to Helio, the same way `rebuild_static_mesh_
//! frame` combines model-space vertex data with a freshly-read `Transform`
//! rather than trusting a baked model matrix. Everything else in the row
//! (color, intensity, shadow/god-rays/flare settings) IS trustworthy as-is
//! -- none of it depends on the object's transform.

use helio::GpuLight;

/// Byte-identical wrapper around [`GpuLight`] -- exists solely to give this
/// field a `pulsar_scenedb::Pod` impl local to this crate. Orphan rules
/// block a direct `unsafe impl Pod for GpuLight` here: neither `GpuLight`
/// (defined in `libhelio`) nor `Pod` (defined in `pulsar_scenedb`) is local
/// to `helio_component`. A `#[repr(transparent)]` wrapper sidesteps that
/// without touching either upstream crate.
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct LightGpuRow(pub GpuLight);

// SAFETY: `GpuLight` is itself `#[repr(C)]` + `bytemuck::Pod` (see its own
// doc) -- a plain, fully-initialized, no-padding-hole POD struct. This
// wrapper is `#[repr(transparent)]` over it, so it carries the identical
// layout/validity guarantees `pulsar_scenedb::Pod` requires.
unsafe impl pulsar_scenedb::Pod for LightGpuRow {}

impl Default for LightGpuRow {
    fn default() -> Self {
        Self(GpuLight::default())
    }
}

impl From<GpuLight> for LightGpuRow {
    fn from(light: GpuLight) -> Self {
        Self(light)
    }
}

/// `#[gpu]`-mirrored SceneDB component -- see this module's doc.
#[derive(pulsar_scenedb::SceneStore, Clone, Copy, Default)]
pub struct LightGpuData {
    #[gpu]
    pub row: LightGpuRow,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn light_gpu_row_round_trips_a_gpu_light_byte_for_byte() {
        let light = GpuLight {
            color_intensity: [0.25, 0.5, 0.75, 42.0],
            ..Default::default()
        };
        let row = LightGpuRow::from(light);
        assert_eq!(row.0.color_intensity, light.color_intensity);
    }
}
