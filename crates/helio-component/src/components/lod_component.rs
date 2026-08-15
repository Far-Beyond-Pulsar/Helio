//! Level of Detail (LOD) component for performance optimization

use engine_class_derive::{engine_class, register_runtime_behavior, register_world_component};
use pulsar_reflection::{
    ComponentRuntimeBehavior, ComponentRuntimeContext, ReflectError, ReflectResult,
    RuntimeComponentOwner, pulsar_type,
};

/// LOD component for managing mesh detail based on distance
///
/// This component demonstrates Vec<T> properties where each LOD level
/// can be added/removed dynamically in the UI.
#[engine_class(category = "Rendering", default, clone, debug, serialize, deserialize)]
pub struct LODComponent {
    /// LOD levels with their distance thresholds
    /// Users can add/remove LOD levels with +/- buttons
    #[property]
    pub lod_levels: Vec<LODLevel>,

    /// Whether to animate transitions between LOD levels
    #[property]
    pub smooth_transitions: bool,

    /// Transition duration in seconds
    #[property(min = 0.0, max = 2.0, step = 0.1)]
    pub transition_duration: f32,

    /// Bias to prefer higher or lower LOD (negative = higher quality, positive = lower quality)
    #[property(min = -2.0, max = 2.0, step = 0.1)]
    pub lod_bias: f32,
}

// Pulsar-Native#561 (editor-side "everything is a real World component, no
// ComponentDb-only flat JSON left"): `LODComponent` never drove any Helio
// behavior directly (LOD selection is a renderer-internal runtime decision,
// not a "sync this component to a Helio scene object" operation), so this
// stays a no-op `sync_component` -- same free-win shape already established
// for `RigidbodyComponent`/`PhysicsComponent` (Phase B5). What matters here
// isn't the dispatch (there's nothing to dispatch), it's that
// `#[register_world_component]` makes this class hydrate into a real typed
// `World` value, reachable through the same live-edit path
// (`SceneDatabase::update_live_component_property`/
// `read_live_component_property`) as every other component, instead of
// living only in `ComponentDb`'s flat JSON.
#[register_world_component]
#[register_runtime_behavior]
impl ComponentRuntimeBehavior for LODComponent {
    const CLASS_NAME: &'static str = "LODComponent";

    fn sync_component(
        _owner: &RuntimeComponentOwner,
        _component_index: usize,
        _component: &Self,
        _context: &mut dyn ComponentRuntimeContext,
    ) {
        // No Helio-side behavior -- LOD selection happens at render time by
        // reading this component's data directly, not via a sync push.
    }
}

/// Single LOD level descriptor
#[engine_class(no_register, default, clone, debug, serialize, deserialize)]
pub struct LODLevel {
    /// Distance from camera where this LOD becomes active
    #[property(min = 0.0, max = 10000.0, step = 1.0)]
    pub distance_threshold: f32,

    /// Screen space coverage percentage (0-100) where this LOD becomes active
    #[property(min = 0.0, max = 100.0, step = 0.1)]
    pub screen_coverage: f32,

    /// Mesh asset path for this LOD level
    /// In a full implementation, this would be a proper asset reference
    #[property]
    pub mesh_path: String,
}

fn serialize_lod_level_json(value: &LODLevel) -> ReflectResult<serde_json::Value> {
    serde_json::to_value(value).map_err(|e| ReflectError::SerializationFailed(e.to_string()))
}

fn deserialize_lod_level_json(value: serde_json::Value) -> ReflectResult<LODLevel> {
    serde_json::from_value(value).map_err(|e| ReflectError::DeserializationFailed(e.to_string()))
}

#[pulsar_type(
    serialize_json_with = serialize_lod_level_json,
    deserialize_json_with = deserialize_lod_level_json
)]
pub type RegisteredLodLevel = LODLevel;
