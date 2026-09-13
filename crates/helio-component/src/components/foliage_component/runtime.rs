use engine_class_derive::{register_runtime_behavior, register_world_component};
use helio::MaterialId;
use pulsar_reflection::{
    get_subsystem, ComponentRuntimeBehavior, ComponentRuntimeContext, LiveKeySet,
    RuntimeComponentOwner,
};

use super::FoliageComponent;
use crate::subsystems::PendingWorldWrites;

fn gpu_type(component: &FoliageComponent) -> helio_pass_foliage_place::components::FoliageTypeComponent {
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
        material_id: MaterialId::from_raw(0, 0).slot(),
        density_layer: component.general.density_layer as u32,
        kind_and_flags: pack_kind_and_flags(FoliageKind::Blade, flags),
        mesh_or_impostor_id: u32::MAX,
        _pad: [0; 3],
    }
}

fn wind(component: &FoliageComponent) -> libhelio::GpuWind {
    libhelio::Wind {
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
        owner: &RuntimeComponentOwner,
        component_index: usize,
        component: &Self,
        context: &mut dyn ComponentRuntimeContext,
    ) {
        get_subsystem!(context, LiveKeySet)
            .insert(format!("{}:{component_index}", owner.scene_object_id));
        let Some(entity) = context
            .subsystems_mut()
            .get_mut::<pulsar_scenedb::Entity>()
            .copied()
        else {
            return;
        };
        let writes = get_subsystem!(context, PendingWorldWrites);
        if !component.general.enabled {
            writes.push(move |world| {
                world.remove::<helio_pass_foliage_place::components::FoliageTypeComponent>(entity);
                world.remove::<helio_pass_foliage_place::components::FoliageLayerComponent>(entity);
                world.remove::<helio_pass_foliage_place::components::FoliageInteractorComponent>(
                    entity,
                );
                world.remove::<helio_pass_foliage_place::components::FoliageWindComponent>(entity);
            });
            return;
        }

        let gpu_type = gpu_type(component);
        let half = component.placement.layer_extent;
        let [x, _y, z] = owner.position;
        let gpu_layer = helio_pass_foliage_place::components::FoliageLayerComponent {
            bounds_min: [x - half, component.placement.altitude_min, z - half, 0.0],
            bounds_max: [
                x + half,
                component.placement.altitude_max,
                z + half,
                component.placement.has_infinite_extent as u32 as f32,
            ],
        };
        let position = if component.interaction.interactor_enabled {
            owner.position
        } else {
            [0.0, -100_000.0, 0.0]
        };
        let gpu_interactor = helio_pass_foliage_place::components::FoliageInteractorComponent {
            position_radius: [
                position[0],
                position[1],
                position[2],
                if component.interaction.interactor_enabled {
                    component.interaction.interactor_radius.max(0.0)
                } else {
                    0.0
                },
            ],
            velocity: [0.0; 4],
        };
        let gpu_wind =
            helio_pass_foliage_place::components::FoliageWindComponent::from(wind(component));
        writes.push(move |world| {
            world.insert(entity, gpu_type);
            world.insert(entity, gpu_layer);
            world.insert(entity, gpu_interactor);
            world.insert(entity, gpu_wind);
        });
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
