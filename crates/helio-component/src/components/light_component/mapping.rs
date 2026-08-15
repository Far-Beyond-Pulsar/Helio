use helio::{GpuLight, LightType as HelioLightType};
use serde_json::Value;
use std::collections::HashMap;

use super::{LightComponent, LightType};

impl LightComponent {
    /// Translate this editor-facing component into Helio's GPU light representation.
    ///
    /// Single source of truth for the `LightComponent` -> `GpuLight` mapping --
    /// factored out of what used to be inline logic in `runtime.rs`'s
    /// `sync_component` so that path and any future SceneDB-driven bridge (the
    /// Pulsar-Native#561 "set us up" step -- see `helio_scenedb::components::
    /// LightComponent`'s doc for why lights can't be a simple `#[gpu(buffer=...)]`
    /// mirrored field, and `helio_scenedb::HelioRenderSubsystem` for the seam this
    /// is meant to eventually feed) call the exact same code and can never
    /// silently drift apart.
    ///
    /// Returns `None` when the light is disabled (`general.enabled == false`) --
    /// the caller is responsible for removing any previously-inserted Helio light
    /// in that case; this function only describes what a currently-enabled light
    /// should look like on the GPU side.
    pub fn to_gpu_light(&self, position: [f32; 3]) -> Option<GpuLight> {
        if !self.general.enabled {
            return None;
        }

        let helio_type = match self.general.light_type {
            LightType::Directional => HelioLightType::Directional,
            LightType::Point => HelioLightType::Point,
            LightType::Spot => HelioLightType::Spot,
            LightType::Area => HelioLightType::Point, // helio has no Area; nearest equivalent
        };

        let [px, py, pz] = position;

        Some(GpuLight {
            position_range: [px, py, pz, self.attenuation.range],
            direction_outer: [0.0, -1.0, 0.0, self.attenuation.outer_cone_angle.to_radians()],
            color_intensity: [
                self.color.color[0],
                self.color.color[1],
                self.color.color[2],
                self.intensity.intensity,
            ],
            shadow_index: if self.shadows.cast_shadows { 0 } else { u32::MAX },
            light_type: helio_type as u32,
            inner_angle: self.attenuation.inner_cone_angle.to_radians(),
            _pad: 0,
            ..Default::default()
        })
    }

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

    #[test]
    fn disabled_light_translates_to_no_gpu_light() {
        let mut light = LightComponent::default();
        light.general.enabled = false;
        assert!(light.to_gpu_light([1.0, 2.0, 3.0]).is_none());
    }

    #[test]
    fn enabled_light_carries_position_color_and_intensity() {
        let mut light = LightComponent::default();
        light.color.color = [0.25, 0.5, 0.75, 1.0];
        light.intensity.intensity = 42.0;
        light.attenuation.range = 10.0;

        let gpu = light
            .to_gpu_light([1.0, 2.0, 3.0])
            .expect("enabled light must produce a GpuLight");

        assert_eq!(gpu.position_range, [1.0, 2.0, 3.0, 10.0]);
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
            let gpu = light.to_gpu_light([0.0, 0.0, 0.0]).unwrap();
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
        assert_eq!(light.to_gpu_light([0.0, 0.0, 0.0]).unwrap().shadow_index, 0);

        light.shadows.cast_shadows = false;
        assert_eq!(
            light.to_gpu_light([0.0, 0.0, 0.0]).unwrap().shadow_index,
            u32::MAX
        );
    }
}
