//! Foliage reaches the foliage passes through its derived source row
//! (`environment_rows::FoliageSourceRow`), built from the mappings here.

use engine_class_derive::{register_runtime_behavior, register_world_component};
use pulsar_reflection::{ComponentRuntimeBehavior, ComponentRuntimeContext, RuntimeComponentOwner};

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
        // Material projections remain a separate scene domain. Slot zero is the
        // stable default until foliage materials receive their own SceneDB column.
        material_id: 0u32,
        density_layer: component.general.density_layer as u32,
        kind_and_flags: pack_kind_and_flags(FoliageKind::Blade, flags),
        mesh_or_impostor_id: u32::MAX,
        _pad: [0; 3],
    }
}

pub(crate) fn wind(component: &FoliageComponent) -> helio_pass_foliage_place::GpuWind {
    helio_pass_foliage_place::Wind {
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
    .to_gpu()
}

#[register_world_component]
#[register_runtime_behavior]
impl ComponentRuntimeBehavior for FoliageComponent {
    const CLASS_NAME: &'static str = "FoliageComponent";

    fn sync_component(
        _owner: &RuntimeComponentOwner,
        _component_index: usize,
        _component: &Self,
        _context: &mut dyn ComponentRuntimeContext,
    ) {
        // Foliage reaches its passes through its derived source row; there
        // is nothing to sync.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn runtime_behavior_has_the_reflected_component_name() {
        assert_eq!(
            <FoliageComponent as ComponentRuntimeBehavior>::CLASS_NAME,
            "FoliageComponent"
        );
    }
}
