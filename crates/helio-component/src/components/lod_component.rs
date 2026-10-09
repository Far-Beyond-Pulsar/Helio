//! Level of Detail (LOD) component for performance optimization

use engine_class_derive::{engine_class, register_world_component};
use pulsar_reflection::{pulsar_type, ReflectError, ReflectResult};

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

// A World-resident typed component (Pulsar-Native#561), edited through the
// same live path as every other component. Nothing consumes it yet; it is
// declared unfinished below (#1053).
#[register_world_component]
impl LODComponent {}

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

// Reported unfinished (Pulsar-Native#1035, Phase 4; tracked in #1053): the
// properties card shows the reason and issue, and attaching one logs them once.
pulsar_world_registry::declare_unfinished_component!(
    "LODComponent",
    "nothing consumes LOD settings; meshes always draw their imported level of detail",
    "https://github.com/Far-Beyond-Pulsar/Pulsar-Native/issues/1053",
);
