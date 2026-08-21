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

// Phase B5 (Pulsar-Native#556). No `on_removed` hook: `HelioRenderer::
// rebuild_light_frame` (`renderer.rs`) rebuilds Helio's ENTIRE light list
// from a fresh SceneDB query every frame, so a removed/disabled light
// simply has no `LightComponentGpuMirror` row for that query to find --
// absence is the removal signal, nothing left for a teardown hook to do
// (unlike before this component was normalized, when Helio held a
// persistent `LightId` actor that needed an explicit `remove_light` call).
#[register_world_component(hydrate = hydrate_light_component, remove = remove_light_component)]
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
