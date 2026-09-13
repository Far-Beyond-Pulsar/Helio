//! SceneDB-backed portal authoring component.

use engine_class_derive::{engine_class, register_runtime_behavior, register_world_component};
use pulsar_reflection::{
    get_subsystem, ComponentRuntimeBehavior, ComponentRuntimeContext, LiveKeySet,
    RuntimeComponentOwner,
};

use crate::subsystems::PendingWorldWrites;

pub const PORTAL_CLASS_NAME: &str = "PortalComponent";

#[engine_class(category = "Rendering", clone, debug, serialize, deserialize)]
#[category("General", category_color = "#6EC5FF")]
pub struct PortalComponent {
    #[property]
    pub enabled: bool,
    #[property]
    pub portal_id: i32,
    #[property(min = 0.1, max = 1000.0, step = 0.1, category = "General")]
    pub width: f32,
    #[property(min = 0.1, max = 1000.0, step = 0.1, category = "General")]
    pub height: f32,
}

impl Default for PortalComponent {
    fn default() -> Self {
        Self {
            enabled: true,
            portal_id: 0,
            width: 2.0,
            height: 3.0,
        }
    }
}

#[register_world_component]
#[register_runtime_behavior]
impl ComponentRuntimeBehavior for PortalComponent {
    const CLASS_NAME: &'static str = PORTAL_CLASS_NAME;

    fn sync_component(
        owner: &RuntimeComponentOwner,
        _component_index: usize,
        component: &Self,
        context: &mut dyn ComponentRuntimeContext,
    ) {
        let Some(entity) = context
            .subsystems_mut()
            .get_mut::<pulsar_scenedb::Entity>()
            .copied()
        else {
            return;
        };
        let key = format!("portal:{}:{}", component.portal_id, owner.scene_object_id);
        let live = get_subsystem!(context, LiveKeySet);
        if component.enabled {
            live.insert(key);
        }
        let writes = get_subsystem!(context, PendingWorldWrites);
        if !component.enabled {
            writes.push(move |world| {
                world.remove::<helio_pass_portal_cull::components::PortalViewComponent>(entity);
                world.remove::<helio_pass_portal_cull::components::PortalChainComponent>(entity);
            });
            return;
        }

        let q = glam::Quat::from_euler(
            glam::EulerRot::YXZ,
            owner.rotation[1].to_radians(),
            owner.rotation[0].to_radians(),
            owner.rotation[2].to_radians(),
        );
        let forward = q * glam::Vec3::NEG_Z;
        let pose = helio::portal_pose_facing(
            glam::Vec3::from_array(owner.position),
            glam::Vec3::from_array(owner.position) + forward,
            q * glam::Vec3::Y,
        );
        let view = helio_pass_portal_cull::components::PortalViewComponent {
            transform: pose.transform.to_cols_array(),
            inverse_transform: pose.transform.inverse().to_cols_array(),
            half_extent: [component.width * 0.5, component.height * 0.5],
            // Coordinate-space projection is a separate transient scene service;
            // zero is the identity/default space for a standalone authored side.
            coordinate_space: 0,
            _pad: 0,
        };
        let chain = helio_pass_portal_cull::components::PortalChainComponent {
            portals: [entity.index(), 0, 0],
            depth: 1,
        };
        writes.push(move |world| {
            world.insert(entity, view);
            world.insert(entity, chain);
        });
    }
}
