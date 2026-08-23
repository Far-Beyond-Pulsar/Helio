use engine_class_derive::{register_runtime_behavior, register_world_component};
use pulsar_reflection::{ComponentRuntimeBehavior, ComponentRuntimeContext, RuntimeComponentOwner};
use pulsar_world_registry::GpuMirrored;

use super::LightComponent;

/// Custom hydrate (Pulsar-Native#561, normalized onto the generic auto-
/// mirror system): parses `LightComponent` as the auto-generated hydrate
/// would, AND keeps its auto-generated `#[gpu]`-mirrored companion
/// (`LightComponentGpuMirror`, produced by `#[engine_class]` from the
/// `#[gpu]`-marked fields in `general`/`intensity`/`color`/`attenuation`/
/// `shadows`, `component.rs`/`sub_props/*.rs`) in step -- inserted (via
/// `GpuMirrored::sync_gpu_mirror`) when the light is enabled, removed (via
/// `GpuMirrored::remove_gpu_mirror`) when it isn't. This custom hydrate
/// still exists ONLY for that conditional: `sync_gpu_mirror`'s own default
/// always inserts unconditionally (see that trait's doc), and "disabled
/// means absent, not present-with-meaningless-values" is business logic
/// specific to this component, not something any generic default could
/// know -- everything else about how the mirror itself is built and
/// attached goes through the exact same `GpuMirrored` trait every other
/// `#[gpu]`-marked component uses, no bespoke insert/remove of a hand-
/// rolled companion type anymore.
fn hydrate_light_component(
    world: &mut pulsar_scenedb::World,
    entity: pulsar_scenedb::Entity,
    data: &serde_json::Value,
) -> Result<(), String> {
    let parsed: LightComponent = serde_json::from_value(data.clone()).map_err(|error| error.to_string())?;

    if parsed.general.enabled {
        parsed.sync_gpu_mirror(world, entity);
    } else {
        LightComponent::remove_gpu_mirror(world, entity);
    }

    world.insert(entity, parsed);
    Ok(())
}

/// Custom remove (Pulsar-Native#561): drops both halves -- the plain
/// `world.remove::<LightComponent>(entity)` `#[register_world_component]`
/// would otherwise generate leaves the GPU mirror orphaned (a real, if
/// GPU-storage-only, leak: nothing else ever removes it, since it isn't
/// itself a registered class `dispatch_component_removals` knows to sweep).
fn remove_light_component(world: &mut pulsar_scenedb::World, entity: pulsar_scenedb::Entity) {
    let _ = world.remove::<LightComponent>(entity);
    LightComponent::remove_gpu_mirror(world, entity);
}

/// `refresh_gpu_mirror` override (Pulsar-Native#561: the properties-panel
/// live-edit bug -- editing a light's color/intensity/enabled in the editor
/// had no visible effect on the rendered scene). Root cause: `sync_gpu_
/// mirror` was previously only ever called from `hydrate_light_component`
/// above, i.e. the JSON-hydrate path -- but the properties panel's real
/// write path (`update_live_component_property`, `ui_level_editor`) mutates
/// the live `LightComponent` directly and never re-hydrates, so nothing
/// ever told `LightComponentGpuMirror` (the only thing `HelioRenderer::
/// rebuild_light_frame` actually reads) a field had changed after the
/// object's first hydrate.
///
/// `HelioRenderer::sync_scene`/`sync_scene_delta` (`engine_backend`) now
/// call `pulsar_world_registry::refresh_world_component_gpu_mirror_for_
/// class` once per COMPONENTS/PROPS-dirty entity per sync pass, generically,
/// for every `#[register_world_component]`-registered class -- this is
/// `LightComponent`'s hook into that, mirroring `hydrate_light_component`'s
/// own enabled-check exactly (the bare `gpu_mirror` flag's generated
/// default can't express "disabled means absent, not present-with-
/// meaningless-values", the same reason `hydrate`/`remove` already need
/// their own custom fns above).
///
/// Can't borrow `world` immutably (to read `LightComponent`) and mutably
/// (to `world.insert` the mirror) at once, so this re-borrows twice rather
/// than holding one `&LightComponent` across both steps -- same shape
/// `#[register_world_component(gpu_mirror)]`'s own generated default uses.
fn refresh_light_gpu_mirror(world: &mut pulsar_scenedb::World, entity: pulsar_scenedb::Entity) {
    let Some(enabled) = world.get::<LightComponent>(entity).map(|light| light.general.enabled) else {
        return;
    };
    if enabled {
        let mirror = world.get::<LightComponent>(entity).map(GpuMirrored::to_gpu_mirror);
        if let Some(mirror) = mirror {
            world.insert(entity, mirror);
        }
    } else {
        LightComponent::remove_gpu_mirror(world, entity);
    }
}

// Phase B5 (Pulsar-Native#556). No `on_removed` hook: `HelioRenderer::
// rebuild_light_frame` (`renderer.rs`) rebuilds Helio's ENTIRE light list
// from a fresh SceneDB query every frame, so a removed/disabled light
// simply has no `LightComponentGpuMirror` row for that query to find --
// absence is the removal signal, nothing left for a teardown hook to do
// (unlike before this component was normalized, when Helio held a
// persistent `LightId` actor that needed an explicit `remove_light` call).
#[register_world_component(
    hydrate = hydrate_light_component,
    remove = remove_light_component,
    refresh_gpu_mirror = refresh_light_gpu_mirror
)]
#[register_runtime_behavior]
impl ComponentRuntimeBehavior for LightComponent {
    const CLASS_NAME: &'static str = "LightComponent";

    fn sync_component(
        _owner: &RuntimeComponentOwner,
        _component_index: usize,
        _component: &Self,
        _context: &mut dyn ComponentRuntimeContext,
    ) {
        // Deliberately empty (Pulsar-Native#561, mirroring
        // `StaticMeshComponent::sync_component`'s own doc for why). This
        // used to translate `component` into a `GpuLight` and push it into
        // Helio directly -- that translation now happens once, at hydrate
        // time, into the auto-generated `#[gpu]`-mirrored
        // `LightComponentGpuMirror` companion (`hydrate_light_component`
        // above), which SceneDB keeps in sync automatically. Resolving
        // every entity's already-hydrated mirror into Helio's actual light
        // list happens once per frame, for every light at once
        // (`HelioRenderer::rebuild_light_frame`, `renderer.rs`) -- this
        // trait's `&Self`-only, one-component-at-a-time signature has no
        // way to do that, deliberately (see `StaticMeshComponent::
        // sync_component`'s doc for the same structural reason).
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::LightComponentGpuMirror;

    fn light_json(enabled: bool) -> serde_json::Value {
        let mut light = LightComponent::default();
        light.general.enabled = enabled;
        serde_json::to_value(&light).expect("LightComponent must serialize")
    }

    #[test]
    fn hydrate_inserts_the_gpu_mirror_for_an_enabled_light() {
        let mut world = pulsar_scenedb::World::new();
        let entity = world.spawn();

        hydrate_light_component(&mut world, entity, &light_json(true)).unwrap();

        assert!(world.get::<LightComponent>(entity).is_some());
        assert!(
            world.get::<LightComponentGpuMirror>(entity).is_some(),
            "an enabled light must get its #[gpu]-mirrored companion"
        );
    }

    #[test]
    fn hydrate_omits_the_gpu_mirror_for_a_disabled_light() {
        let mut world = pulsar_scenedb::World::new();
        let entity = world.spawn();

        hydrate_light_component(&mut world, entity, &light_json(false)).unwrap();

        assert!(world.get::<LightComponent>(entity).is_some());
        assert!(
            world.get::<LightComponentGpuMirror>(entity).is_none(),
            "a disabled light must not carry a GPU-mirrored row"
        );
    }

    #[test]
    fn re_hydrating_as_disabled_drops_a_previously_mirrored_row() {
        let mut world = pulsar_scenedb::World::new();
        let entity = world.spawn();

        hydrate_light_component(&mut world, entity, &light_json(true)).unwrap();
        assert!(world.get::<LightComponentGpuMirror>(entity).is_some());

        hydrate_light_component(&mut world, entity, &light_json(false)).unwrap();
        assert!(
            world.get::<LightComponentGpuMirror>(entity).is_none(),
            "toggling enabled -> disabled must remove the stale mirrored row, not just stop updating it"
        );
    }

    #[test]
    fn refresh_gpu_mirror_picks_up_a_live_edit_hydrate_never_saw() {
        let mut world = pulsar_scenedb::World::new();
        let entity = world.spawn();
        hydrate_light_component(&mut world, entity, &light_json(true)).unwrap();

        let before = world.get::<LightComponentGpuMirror>(entity).unwrap().intensity.intensity;

        // The properties panel's real live-edit path: mutate the World-
        // resident LightComponent directly, no JSON, no re-hydrate --
        // exactly what `update_live_component_property` does.
        world.get_mut::<LightComponent>(entity).unwrap().intensity.intensity = 4242.0;

        // Sanity: this is the bug as originally reported -- the live value
        // changed, but the mirror `HelioRenderer::rebuild_light_frame`
        // actually reads is still whatever hydrate saw.
        assert_eq!(
            world.get::<LightComponentGpuMirror>(entity).unwrap().intensity.intensity,
            before,
            "sanity: a plain live edit must NOT auto-propagate to the mirror by itself"
        );

        refresh_light_gpu_mirror(&mut world, entity);

        assert_eq!(
            world.get::<LightComponentGpuMirror>(entity).unwrap().intensity.intensity.0,
            4242.0,
            "refresh_light_gpu_mirror must re-derive the mirror from the CURRENT live value"
        );
    }

    #[test]
    fn refresh_gpu_mirror_removes_the_mirror_when_a_live_edit_disables_the_light() {
        let mut world = pulsar_scenedb::World::new();
        let entity = world.spawn();
        hydrate_light_component(&mut world, entity, &light_json(true)).unwrap();
        assert!(world.get::<LightComponentGpuMirror>(entity).is_some());

        // Live-disable, same path as above -- not a re-hydrate.
        world.get_mut::<LightComponent>(entity).unwrap().general.enabled = false;
        refresh_light_gpu_mirror(&mut world, entity);

        assert!(
            world.get::<LightComponentGpuMirror>(entity).is_none(),
            "a live edit to enabled=false must remove the mirror on the next refresh, \
             not leave the last-synced (now-stale) value rendering forever"
        );
    }

    #[test]
    fn remove_drops_both_the_component_and_its_gpu_mirror() {
        let mut world = pulsar_scenedb::World::new();
        let entity = world.spawn();
        hydrate_light_component(&mut world, entity, &light_json(true)).unwrap();
        assert!(world.get::<LightComponent>(entity).is_some());
        assert!(world.get::<LightComponentGpuMirror>(entity).is_some());

        remove_light_component(&mut world, entity);

        assert!(world.get::<LightComponent>(entity).is_none());
        assert!(
            world.get::<LightComponentGpuMirror>(entity).is_none(),
            "remove must not orphan the GPU mirror -- nothing else ever sweeps it"
        );
    }
}
