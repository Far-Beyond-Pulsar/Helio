use engine_class_derive::{register_runtime_behavior, register_world_component};
use helio::{Renderer, SceneActor};
use pulsar_reflection::{
    get_subsystem, scene_id_to_tag, ComponentRuntimeBehavior, ComponentRuntimeContext,
    RuntimeComponentOwner,
};

use super::LightComponent;

// Phase B5 (Pulsar-Native#556).
#[register_world_component]
#[register_runtime_behavior]
impl ComponentRuntimeBehavior for LightComponent {
    const CLASS_NAME: &'static str = "LightComponent";

    fn sync_component(
        owner: &RuntimeComponentOwner,
        _component_index: usize,
        component: &Self,
        context: &mut dyn ComponentRuntimeContext,
    ) {
        let tag = scene_id_to_tag(owner.scene_object_id);
        let scene = get_subsystem!(context, Renderer).scene_mut();

        // `to_gpu_light` is the single source of truth for this translation --
        // see its doc for why this exists as a standalone method rather than
        // inline logic here.
        match component.to_gpu_light(owner.position) {
            None => {
                // Disabled — drop the light we previously inserted, if any.
                if let Some(id) = scene.light_by_tag(tag) {
                    let _ = scene.remove_light(id);
                }
            }
            Some(gpu) => {
                // Helio owns the scene: ask it whether we already have a light for
                // this object rather than tracking handles editor-side.
                match scene.light_by_tag(tag) {
                    Some(id) => {
                        let _ = scene.update_light(id, gpu);
                    }
                    None => {
                        scene.insert_actor(SceneActor::light_with_tag(gpu, tag));
                    }
                }
            }
        }
    }
}
