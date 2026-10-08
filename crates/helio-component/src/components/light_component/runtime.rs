use engine_class_derive::{register_world_component};
use pulsar_reflection::{ComponentRuntimeBehavior, ComponentRuntimeContext, RuntimeComponentOwner};

use super::LightComponent;

/// `LightComponent`'s JSON boundary decoder. Levels authored before the
/// sub-props migration store the flat property shape the editor once used
/// (`intensity` as a bare number); the Rust representation groups those same
/// properties into sub-props (`general`, `intensity`, etc.). Accept both, so
/// those projects load, and decode once into the typed value.
///
/// Nothing else is light-specific about storing one: the decoded value is
/// inserted through SceneDB's erased insert, and its generated
/// `LightComponentGpuMirror` row (including `general.enabled`) follows every
/// write through SceneDB's mirror dispatch.
fn decode_light_component(data: &serde_json::Value) -> Result<LightComponent, String> {
    if data.get("general").is_some() {
        serde_json::from_value(data.clone()).map_err(|error| error.to_string())
    } else {
        Ok(LightComponent::from_component_data(data))
    }
}

// Phase B5 (Pulsar-Native#556). No `on_removed` hook: Helio holds no
// persistent light actor. Removing the component clears its GPU row through
// SceneDB, and a disabled light's row says so (`general.enabled` is part of
// it).
#[register_world_component(decode = decode_light_component)]
impl ComponentRuntimeBehavior for LightComponent {
    const CLASS_NAME: &'static str = "LightComponent";

    fn sync_component(
        _owner: &RuntimeComponentOwner,
        _component_index: usize,
        _component: &Self,
        _context: &mut dyn ComponentRuntimeContext,
    ) {
        // Deliberately empty (Pulsar-Native#561, mirroring
        // `StaticMeshComponent::sync_component`'s own doc for why). The
        // `GpuLight` translation is the generated `LightComponentGpuMirror`
        // row, which SceneDB writes on every write of this component. Resolving
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

    #[test]
    fn decodes_the_nested_shape() {
        let mut light = LightComponent::default();
        light.general.enabled = false;
        light.intensity.intensity = 77.0;
        let json = serde_json::to_value(&light).expect("LightComponent must serialize");

        let decoded = decode_light_component(&json).unwrap();
        assert!(!decoded.general.enabled);
        assert_eq!(decoded.intensity.intensity, 77.0);
    }

    #[test]
    fn decodes_the_legacy_flat_shape() {
        // The shape that failed to hydrate in the 2026-10-04 editor log.
        let json = serde_json::json!({ "enabled": true, "intensity": 1002.0 });
        let decoded = decode_light_component(&json).unwrap();
        assert!(decoded.general.enabled);
        assert_eq!(decoded.intensity.intensity, 1002.0);
    }
}
