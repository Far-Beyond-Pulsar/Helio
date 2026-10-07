//! The light a `LightComponent` instance casts, as the renderer's scene join
//! reads it (Pulsar-Native#1035, Phase 2).
//!
//! A second GPU registration on [`LightComponent`]: SceneDB derives this row
//! from the authored value on every insert, guarded write, removal, despawn
//! and mirror replay, through the same `LightComponent` -> `GpuLight`
//! mapping the companion row uses ([`super::LightComponentGpuMirror::
//! to_helio_gpu_light`]). It is the light in its own space: position and
//! direction are zero, because they come from the owner object's transform,
//! which the scene join applies on the GPU together with the owner's
//! visibility and the instance's enabled state.
//!
//! Layout: `helio::GpuLight` field for field, except that the slot GpuLight
//! keeps as padding carries the authored `general.enabled` flag. The join
//! writes it back to zero in its output.

use pulsar_scenedb::gpu::GpuMirrorHandle;
use pulsar_scenedb_derive::SceneStore;

use super::LightComponent;
use pulsar_world_registry::GpuMirrored;

/// Buffer the rows register under; keyed by the light instance entity.
pub const LIGHT_SOURCES_BUFFER: &str = "light_sources";

#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "light_sources")]
pub struct LightSourceRow {
    #[gpu]
    pub position_range: [f32; 4],
    #[gpu]
    pub direction_outer: [f32; 4],
    #[gpu]
    pub color_intensity: [f32; 4],
    #[gpu]
    pub shadow_index: u32,
    #[gpu]
    pub light_type: u32,
    #[gpu]
    pub inner_angle: f32,
    /// 1 when the authored light is enabled, 0 when not (`GpuLight::_pad`).
    #[gpu]
    pub enabled: u32,
    #[gpu]
    pub god_rays_enabled: u32,
    #[gpu]
    pub god_rays_density: f32,
    #[gpu]
    pub god_rays_weight: f32,
    #[gpu]
    pub god_rays_decay: f32,
    #[gpu]
    pub god_rays_exposure: f32,
    #[gpu]
    pub flare_enabled: u32,
    #[gpu]
    pub flare_type: u32,
    #[gpu]
    pub flare_intensity: f32,
    #[gpu]
    pub flare_scale: f32,
    #[gpu]
    pub flare_tint_r: f32,
    #[gpu]
    pub flare_tint_g: f32,
    #[gpu]
    pub flare_tint_b: f32,
    #[gpu]
    pub ies_profile_index: i32,
    #[gpu]
    pub light_function_index: i32,
    #[gpu]
    pub ies_angle_scale: f32,
    #[gpu]
    pub ies_angle_offset: f32,
}

impl LightSourceRow {
    pub fn of(light: &LightComponent) -> Self {
        let mut row: Self = bytemuck::cast(light.to_gpu_mirror().to_helio_gpu_light());
        row.enabled = u32::from(light.general.enabled);
        row
    }
}

fn light_source_dispatch(mirror: &GpuMirrorHandle, row: u32, data: *const (), is_new_insert: bool) {
    // SAFETY: SceneDB reaches this only through `LightComponent`'s own
    // `ComponentId`, with a pointer to a live `LightComponent`.
    let light = unsafe { &*(data as *const LightComponent) };
    pulsar_scenedb::gpu::write_derived_row(mirror, row, &LightSourceRow::of(light), is_new_insert);
}

fn light_source_clear(mirror: &GpuMirrorHandle, row: u32) {
    pulsar_scenedb::gpu::clear_derived_row::<LightSourceRow>(mirror, row);
}

pulsar_scenedb::pulsar_reflection::inventory::submit! {
    pulsar_scenedb::gpu::GpuMirrorRegistration {
        component_id: pulsar_scenedb::component_id::<LightComponent>,
        dispatch: light_source_dispatch,
    }
}

pulsar_scenedb::pulsar_reflection::inventory::submit! {
    pulsar_scenedb::gpu::GpuClearRegistration {
        component_id: pulsar_scenedb::component_id::<LightComponent>,
        clear: light_source_clear,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_row_is_a_gpu_light_with_the_enabled_flag_in_its_padding() {
        assert_eq!(
            std::mem::size_of::<LightSourceRow>(),
            std::mem::size_of::<helio::GpuLight>()
        );
        let mut light = LightComponent::default();
        light.general.enabled = true;
        let row = LightSourceRow::of(&light);
        assert_eq!(row.enabled, 1);
        let mut expected = light.to_gpu_mirror().to_helio_gpu_light();
        expected._pad = 1;
        assert_eq!(bytemuck::bytes_of(&row), bytemuck::bytes_of(&expected));
        light.general.enabled = false;
        assert_eq!(LightSourceRow::of(&light).enabled, 0);
    }
}
