use engine_class_derive::{register_runtime_behavior, register_world_component};
use helio::{Renderer, SceneActor};
use pulsar_reflection::{
    get_subsystem, scene_id_to_tag, ComponentRuntimeBehavior, ComponentRuntimeContext,
    RuntimeComponentOwner,
};

use super::LightComponent;

/// Drops `owner`'s Helio light, if it has one -- the single teardown path
/// shared by [`ComponentRuntimeBehavior::sync_component`]'s disabled-light
/// branch and this class's `on_removed` hook (component removed, or its
/// owning object despawned; see `#[register_world_component(on_removed =
/// ...)]` below and `pulsar_world_registry::WorldComponentRegistration::
/// on_removed`'s doc for why a *hook* is what removal needs -- SceneDB has
/// no idea what a `LightId` is, only this component does). Idempotent: a
/// tag with no light is simply not found, not an error -- safe to call
/// speculatively (`dispatch_component_removals`'s whole-object sweep calls
/// every registered class's `on_removed` for every despawned object,
/// whether or not that object ever actually had a `LightComponent`).
fn remove_light_by_tag(owner: &RuntimeComponentOwner, context: &mut dyn ComponentRuntimeContext) {
    let tag = scene_id_to_tag(owner.scene_object_id);
    let scene = get_subsystem!(context, Renderer).scene_mut();
    if let Some(id) = scene.light_by_tag(tag) {
        let _ = scene.remove_light(id);
    }
}

// Phase B5 (Pulsar-Native#556).
#[register_world_component(on_removed = remove_light_by_tag)]
#[register_runtime_behavior]
impl ComponentRuntimeBehavior for LightComponent {
    const CLASS_NAME: &'static str = "LightComponent";

    fn sync_component(
        owner: &RuntimeComponentOwner,
        _component_index: usize,
        component: &Self,
        context: &mut dyn ComponentRuntimeContext,
    ) {
        // `to_gpu_light` is the single source of truth for this translation --
        // see its doc for why this exists as a standalone method rather than
        // inline logic here.
        match component.to_gpu_light(owner.position) {
            None => {
                // Disabled — same teardown as a real removal (see
                // `remove_light_by_tag`'s doc).
                remove_light_by_tag(owner, context);
            }
            Some(gpu) => {
                let tag = scene_id_to_tag(owner.scene_object_id);
                let scene = get_subsystem!(context, Renderer).scene_mut();
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
