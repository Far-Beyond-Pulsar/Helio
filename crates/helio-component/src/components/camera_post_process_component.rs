//! Per-view PP baseline, stored in the post-process pass's SceneDB schema.
use engine_class_derive::{engine_class, register_runtime_behavior, register_world_component};
use pulsar_reflection::{get_subsystem, ComponentRuntimeBehavior, ComponentRuntimeContext, RuntimeComponentOwner};
use crate::subsystems::PendingWorldWrites;
use super::PostProcessSettingsProps;

pub const CAMERA_POST_PROCESS_CLASS_NAME: &str = "CameraPostProcessComponent";

#[engine_class(category = "Rendering", clone, debug, serialize, deserialize)]
#[category("Camera", category_color = "#3AA0FF")]
#[serde(default)]
pub struct CameraPostProcessComponent {
    #[property(category = "Camera")]
    pub enabled: bool,
    /// Generic render view identifier, independent of any editor camera type.
    #[property(category = "Camera")]
    pub view_id: u32,
    #[sub_props]
    #[serde(flatten)]
    pub settings: PostProcessSettingsProps,
}

impl Default for CameraPostProcessComponent {
    fn default() -> Self {
        Self { enabled: true, view_id: 0, settings: PostProcessSettingsProps::default() }
    }
}

fn remove_camera(world: &mut pulsar_scenedb::World, entity: pulsar_scenedb::Entity) {
    world.remove::<CameraPostProcessComponent>(entity);
    world.remove::<helio_pass_postprocess::CameraPostProcessComponent>(entity);
}

#[register_world_component(remove = remove_camera)]
#[register_runtime_behavior]
impl ComponentRuntimeBehavior for CameraPostProcessComponent {
    const CLASS_NAME: &'static str = CAMERA_POST_PROCESS_CLASS_NAME;

    fn sync_component(
        _owner: &RuntimeComponentOwner,
        _component_index: usize,
        component: &Self,
        context: &mut dyn ComponentRuntimeContext,
    ) {
        let Some(entity) = context.subsystems_mut().get_mut::<pulsar_scenedb::Entity>().copied() else { return; };
        let writes = get_subsystem!(context, PendingWorldWrites);
        if !component.enabled {
            writes.push(move |world| { world.remove::<helio_pass_postprocess::CameraPostProcessComponent>(entity); });
            return;
        }
        let packed = helio_pass_postprocess::CameraPostProcessComponent::new(
            component.view_id, &component.settings.to_settings(),
        );
        writes.push(move |world| { world.insert(entity, packed); });
    }
}
