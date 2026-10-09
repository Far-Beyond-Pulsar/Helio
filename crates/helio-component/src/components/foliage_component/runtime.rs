//! Foliage reaches the foliage passes through its derived source row
//! (`environment_rows::FoliageSourceRow`), built from the mappings here.

use engine_class_derive::{register_world_component};

use super::FoliageComponent;

pub(crate) fn gpu_type(
    component: &FoliageComponent,
) -> helio_pass_foliage_place::components::FoliageTypeComponent {
    use helio_pass_foliage_place::{pack_kind_and_flags, FoliageKind};

    let flags = u32::from(component.rendering.two_sided)
        * helio_pass_foliage_place::FOLIAGE_FLAG_TWO_SIDED
        | u32::from(component.rendering.casts_shadow)
            * helio_pass_foliage_place::FOLIAGE_FLAG_CASTS_SHADOW
        | u32::from(component.interaction.receives_interaction)
            * helio_pass_foliage_place::FOLIAGE_FLAG_RECEIVES_INTERACTION;
    let [red, green, blue, _] = component.rendering.base_color;
    let [base_color, roughness_metallic] = helio_pass_foliage_place::FoliageMaterial {
        base_color: [red, green, blue],
        roughness: component.rendering.roughness,
        metallic: component.rendering.metallic,
    }
    .pack();
    helio_pass_foliage_place::components::FoliageTypeComponent {
        density: component.general.density,
        height_range: [
            component.placement.height_min,
            component.placement.height_max,
        ],
        width_range: [component.placement.width_min, component.placement.width_max],
        slope_range: [
            component.placement.slope_max_degrees.to_radians().cos(),
            component.placement.slope_min_degrees.to_radians().cos(),
        ],
        altitude_range: [
            component.placement.altitude_min,
            component.placement.altitude_max,
        ],
        lod_distances: [
            component.rendering.lod_distance_0,
            component.rendering.lod_distance_1,
            component.rendering.lod_distance_2,
            component.rendering.lod_distance_3,
        ],
        wind_response: [
            component.wind.trunk_sway,
            component.wind.branch_flutter,
            component.wind.leaf_jitter,
        ],
        interaction_stiffness: component.wind.interaction_stiffness,
        // The authored colour, roughness and metallic travel in the row itself
        // (`base_color`, `roughness_metallic`). Slot zero stays the default until
        // foliage resolves materials through the material table.
        material_id: 0u32,
        density_layer: component.general.density_layer as u32,
        kind_and_flags: pack_kind_and_flags(FoliageKind::Blade, flags),
        mesh_or_impostor_id: u32::MAX,
        base_color,
        roughness_metallic,
        _pad: 0,
    }
}

/// The component's own wind. Its `_pad[0]` is 1 when the component opts
/// out of the level's global wind (`use_global_wind` off): the environment
/// join gives such a component's wind precedence over the global one. The
/// wind clock (`time_prev_time`) is unused: the foliage passes read the
/// renderer's frame clock.
pub(crate) fn wind(component: &FoliageComponent) -> helio_pass_foliage_place::GpuWind {
    let mut wind = helio_pass_foliage_place::Wind {
        direction: glam::Vec3::from_array(component.wind.wind_direction),
        speed: if component.wind.wind_enabled {
            component.wind.wind_speed
        } else {
            0.0
        },
        gust_amplitude: component.wind.gust_amplitude,
        gust_frequency: component.wind.gust_frequency,
        turbulence_scale: component.wind.turbulence_scale,
        ..Default::default()
    }
    .to_gpu();
    wind._pad[0] = if component.wind.use_global_wind {
        0.0
    } else {
        1.0
    };
    wind
}

#[register_world_component]
impl FoliageComponent {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn registers_under_its_class_name() {
        assert_eq!(
            pulsar_world_registry::component_id_for_class("FoliageComponent"),
            Some(pulsar_scenedb::component_id::<FoliageComponent>())
        );
    }
}
