use helio::{GpuLight, LightType as HelioLightType};
use serde_json::Value;
use std::collections::HashMap;

use super::{LightComponent, LightComponentGpuMirror, LightType};

impl LightComponentGpuMirror {
    /// Translate this `#[gpu]`-mirrored companion into Helio's GPU light
    /// representation -- single source of truth for the `LightComponent` ->
    /// `GpuLight` mapping (Pulsar-Native#561).
    ///
    /// Reads ONLY the fields `runtime.rs`'s sub-props marked `#[gpu]`
    /// (`general.light_type`, `intensity.intensity`, `color.color`,
    /// `attenuation.{range,inner_cone_angle,outer_cone_angle}`, `shadows.
    /// cast_shadows`) -- every other `LightComponent` property (falloff
    /// curve shaping, volumetrics, light functions, performance/mobile
    /// settings, ...) has no `GpuLight` field to land in and stays plain
    /// `#[property]`-only, absent from this mirror entirely.
    ///
    /// `position_range`'s xyz is always `[0.0; 0.0; 0.0]` here -- `GpuLight`
    /// bakes in world-space position, which is transform-dependent and has
    /// no business being cached in a mirror that only updates on a genuine
    /// property edit (an object can move without any `LightComponent`
    /// property changing, which would never re-trigger a mirror rebuild).
    /// `HelioRenderer::rebuild_light_frame` overwrites those three
    /// components from the live `Transform` before ever using the result --
    /// the same "model-space payload + per-frame transform combine" split
    /// `rebuild_static_mesh_frame` already uses for mesh geometry.
    ///
    /// Called only for an ENABLED light -- `runtime.rs`'s hydrate checks
    /// `general.enabled` itself and skips `sync_gpu_mirror` (so this mirror
    /// is never even inserted) when it's `false`, the same "disabled means
    /// absent, not zeroed" contract this type has always had.
    pub fn to_helio_gpu_light(&self) -> GpuLight {
        let helio_type = match self.general.light_type.0 {
            LightType::Directional => HelioLightType::Directional,
            LightType::Point => HelioLightType::Point,
            LightType::Spot => HelioLightType::Spot,
            LightType::Area => HelioLightType::Point, // helio has no Area; nearest equivalent
        };

        GpuLight {
            position_range: [0.0, 0.0, 0.0, self.attenuation.range.0],
            direction_outer: [0.0, -1.0, 0.0, self.attenuation.outer_cone_angle.0.to_radians()],
            color_intensity: [
                self.color.color.0[0],
                self.color.color.0[1],
                self.color.color.0[2],
                self.intensity.intensity.0,
            ],
            shadow_index: if self.shadows.cast_shadows.0 { 0 } else { u32::MAX },
            light_type: helio_type as u32,
            inner_angle: self.attenuation.inner_cone_angle.0.to_radians(),
            _pad: 0,
            ..Default::default()
        }
    }
}

impl LightComponent {
    pub fn from_component_data(data: &Value) -> Self {
        let mut light = Self::default();
        if let Some(obj) = data.as_object() {
            light.general.apply_from_component_data(obj);
            light.intensity.apply_from_component_data(obj);
            light.color.apply_from_component_data(obj);
            light.attenuation.apply_from_component_data(obj);
            light.shadows.apply_from_component_data(obj);
            light.volumetrics.apply_from_component_data(obj);
            light.light_function.apply_from_component_data(obj);
            light.performance.apply_from_component_data(obj);
            light.advanced.apply_from_component_data(obj);
        }
        light
    }

    pub fn to_scene_props(&self) -> HashMap<String, Value> {
        let mut out = HashMap::new();
        self.general.apply_to_scene_props(&mut out);
        self.intensity.apply_to_scene_props(&mut out);
        self.color.apply_to_scene_props(&mut out);
        self.attenuation.apply_to_scene_props(&mut out);
        self.shadows.apply_to_scene_props(&mut out);
        self.volumetrics.apply_to_scene_props(&mut out);
        self.light_function.apply_to_scene_props(&mut out);
        self.performance.apply_to_scene_props(&mut out);
        self.advanced.apply_to_scene_props(&mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pulsar_world_registry::GpuMirrored;

    // NOTE: "a disabled light produces no GpuLight" is no longer this
    // function's responsibility -- `to_helio_gpu_light` always translates
    // whatever mirror it's given; "disabled means absent" now lives one
    // level up, in `runtime.rs`'s hydrate (which skips `sync_gpu_mirror`
    // entirely for a disabled light, so no `LightComponentGpuMirror` --
    // and therefore no `to_helio_gpu_light` call -- ever happens). See
    // `runtime.rs`'s `hydrate_omits_light_gpu_data_for_a_disabled_light`
    // for that coverage.

    #[test]
    fn mirror_carries_color_and_intensity_with_a_zeroed_position_placeholder() {
        let mut light = LightComponent::default();
        light.color.color = [0.25, 0.5, 0.75, 1.0];
        light.intensity.intensity = 42.0;
        light.attenuation.range = 10.0;

        let gpu = light.to_gpu_mirror().to_helio_gpu_light();

        // xyz is always the zeroed placeholder here -- HelioRenderer::
        // rebuild_light_frame is the one place that overwrites it from the
        // live Transform (see this fn's own doc).
        assert_eq!(gpu.position_range, [0.0, 0.0, 0.0, 10.0]);
        assert_eq!(gpu.color_intensity, [0.25, 0.5, 0.75, 42.0]);
    }

    #[test]
    fn light_type_maps_onto_the_matching_helio_discriminant() {
        let cases = [
            (LightType::Directional, HelioLightType::Directional),
            (LightType::Point, HelioLightType::Point),
            (LightType::Spot, HelioLightType::Spot),
            // helio has no Area light type; Area falls back to Point.
            (LightType::Area, HelioLightType::Point),
        ];
        for (editor_type, expected) in cases {
            let mut light = LightComponent::default();
            light.general.light_type = editor_type;
            let gpu = light.to_gpu_mirror().to_helio_gpu_light();
            assert_eq!(
                gpu.light_type, expected as u32,
                "{editor_type:?} should map to {expected:?}"
            );
        }
    }

    #[test]
    fn cast_shadows_toggle_maps_to_the_shadow_index_sentinel() {
        let mut light = LightComponent::default();

        light.shadows.cast_shadows = true;
        assert_eq!(light.to_gpu_mirror().to_helio_gpu_light().shadow_index, 0);

        light.shadows.cast_shadows = false;
        assert_eq!(
            light.to_gpu_mirror().to_helio_gpu_light().shadow_index,
            u32::MAX
        );
    }
}
