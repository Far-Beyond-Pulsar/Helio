//! Reflected, SceneDB-owned authoring components for voxel objects and terrain.
//!
//! These structs contain reflected configuration and runtime-only live data.
//! Voxel encodings, edit batches, generation behavior, and GPU/cache
//! representations belong to the dedicated voxel pass and its SceneDB-facing
//! data API. No persistence behavior is implied by the runtime data fields.

use engine_class_derive::engine_class;
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

/// Generic live payload storage owned by a voxel component row.
///
/// Keys are intentionally opaque to this crate; voxel chunk/key semantics are
/// defined by the voxel pass. The tuple contains the live data revision and
/// payload map so a batch can publish both under one lock. Values are immutable
/// so readers can retain a cheap snapshot while a writer replaces one entry.
/// Component clones create fresh stores; callers can explicitly clone the
/// `Arc` when shared access is intended. Four opaque words allow collision-free
/// pass-defined keys (for example signed XYZ bit patterns plus an LOD word).
pub type VoxelPayloadKey = [u64; 4];
pub type VoxelPayloadStore = Arc<RwLock<(u64, HashMap<VoxelPayloadKey, Arc<[u8]>>)>>;

fn empty_payload_store() -> VoxelPayloadStore {
    Arc::new(RwLock::new((0, HashMap::new())))
}

fn clone_payload_store(store: &VoxelPayloadStore) -> VoxelPayloadStore {
    // `Clone` must preserve component value semantics for SceneDB snapshots
    // and transactions. Share immutable payload allocations, but copy the
    // mutable index/revision so two cloned entries never alias future edits.
    let state = store
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    Arc::new(RwLock::new((state.0, state.1.clone())))
}

/// Authoring component for a small deformable voxel object.
///
/// A newly-created object describes a cubic 16³ voxel volume. Its editable
/// voxel contents are SceneDB-owned data managed through the voxel data API;
/// this component stores configuration and the SceneDB material-ID palette.
#[engine_class(category = "Voxel", debug, serialize, deserialize)]
#[category("Volume", category_color = "#8F8F8F")]
#[category("Materials", category_color = "#D1A73F")]
#[category("Editing", category_color = "#D18F6F")]
#[category("Surface", category_color = "#2FA88A")]
pub struct VoxelComponent {
    /// Runtime-only live payloads and data revision. Not an inspector
    /// property, serialized configuration, or GPU-mirrored field. Component
    /// clones start with an independent empty store.
    #[serde(skip)]
    pub payloads: VoxelPayloadStore,
    /// Whether the object participates in voxel rendering and queries.
    #[property]
    pub enabled: bool,
    /// Edge length of one base-resolution voxel in world units.
    #[property(min = 0.0001, max = 10000.0, step = 0.01, category = "Volume")]
    pub voxel_size: f64,
    /// Initial cubic grid dimensions. The default is 16 × 16 × 16.
    #[property(category = "Volume")]
    pub dimensions: [u32; 3],
    /// Palette of IDs into Helio's existing SceneDB material records.
    #[property(category = "Materials")]
    pub material_ids: Vec<u32>,
    /// Palette slot used for newly initialized voxels.
    #[property(category = "Materials")]
    pub default_material_slot: u32,
    /// Whether external callers may submit deformation/edit batches.
    #[property(category = "Editing")]
    pub editable: bool,
    /// Selects blocky (0) or smooth (1) surface presentation. Interpretation
    /// and validation are owned by the voxel pass.
    #[property(category = "Surface")]
    pub smooth_surface: bool,
}

impl Default for VoxelComponent {
    fn default() -> Self {
        Self {
            payloads: empty_payload_store(),
            enabled: true,
            voxel_size: 1.0,
            dimensions: [16; 3],
            material_ids: Vec::new(),
            default_material_slot: 0,
            editable: true,
            smooth_surface: false,
        }
    }
}

impl VoxelComponent {
    /// Clone the live SceneDB-owned data handle for batch access or
    /// caller-controlled snapshot/exfiltration. The returned handle contains
    /// no persistence policy.
    pub fn payload_store(&self) -> VoxelPayloadStore {
        Arc::clone(&self.payloads)
    }
}

// Cloning a component preserves its live value while isolating future map
// mutations. The immutable Arc payload allocations remain shared until one
// entry replaces/deletes them.
impl Clone for VoxelComponent {
    fn clone(&self) -> Self {
        Self {
            payloads: clone_payload_store(&self.payloads),
            enabled: self.enabled,
            voxel_size: self.voxel_size,
            dimensions: self.dimensions,
            material_ids: self.material_ids.clone(),
            default_material_slot: self.default_material_slot,
            editable: self.editable,
            smooth_surface: self.smooth_surface,
        }
    }
}

/// General-purpose voxel terrain authoring configuration.
///
/// `domain_mode` and `shape_mode` are stable primitive discriminants so this
/// component crate does not define voxel-specific helper enums. Their values
/// are interpreted and validated by the voxel pass. The intended initial
/// values are domain 0 = bounded, 1 = unbounded and shape 0 = plane, 1 = sphere.
#[engine_class(category = "Voxel/Terrain", debug, serialize, deserialize)]
#[category("Domain", category_color = "#8F8F8F")]
#[category("Generation", category_color = "#D1A73F")]
#[category("Materials", category_color = "#D1A73F")]
#[category("Surface", category_color = "#2FA88A")]
#[category("LOD", category_color = "#7C6FD1")]
#[category("Streaming", category_color = "#3AA0FF")]
#[category("Editing", category_color = "#D18F6F")]
pub struct VoxelTerrainComponent {
    /// Runtime-only live payloads and data revision. Not an inspector
    /// property, serialized configuration, or GPU-mirrored field. Component
    /// clones start with an independent empty store.
    #[serde(skip)]
    pub payloads: VoxelPayloadStore,
    /// Whether this terrain source participates in rendering and queries.
    #[property]
    pub enabled: bool,
    /// Domain discriminant: 0 bounded, 1 unbounded.
    #[property(category = "Domain")]
    pub domain_mode: u32,
    /// Shape discriminant: 0 plane, 1 sphere/planet; other values are
    /// reserved for pass-registered/custom source kinds.
    #[property(category = "Domain")]
    pub shape_mode: u32,
    /// Finite-domain minimum and maximum on each axis. Ignored for unbounded domains.
    #[property(category = "Domain")]
    pub bounds_min_x: f64,
    #[property(category = "Domain")]
    pub bounds_min_y: f64,
    #[property(category = "Domain")]
    pub bounds_min_z: f64,
    #[property(category = "Domain")]
    pub bounds_max_x: f64,
    #[property(category = "Domain")]
    pub bounds_max_y: f64,
    #[property(category = "Domain")]
    pub bounds_max_z: f64,
    /// Sphere radius in world units; used when `shape_mode` selects a sphere.
    #[property(min = 0.0, max = 1.0e15, step = 1.0, category = "Domain")]
    pub planet_radius: f64,
    /// Edge length of a base-resolution voxel in world units.
    #[property(min = 0.0001, max = 10000.0, step = 0.01, category = "Generation")]
    pub voxel_size: f64,
    /// Stable registered generator/source identifier. Empty means externally
    /// supplied data only; generator implementation is not stored here.
    #[property(category = "Generation")]
    pub generator_id: String,
    /// Seed supplied to the registered generator.
    #[property(category = "Generation")]
    pub seed: u64,
    /// Opaque serialized parameters consumed by the registered generator.
    #[property(category = "Generation")]
    pub generator_parameters: String,
    /// Palette of IDs into Helio's existing SceneDB material records.
    #[property(category = "Materials")]
    pub material_ids: Vec<u32>,
    /// Select smooth rather than blocky surface extraction when supported.
    #[property(category = "Surface")]
    pub smooth_surface: bool,
    /// Screen-space target error used by terrain LOD selection.
    #[property(min = 0.1, max = 64.0, step = 0.1, category = "LOD")]
    pub target_error_pixels: f32,
    /// Preferred detail distance in world units; zero leaves selection to the
    /// pass's view-driven policy.
    #[property(min = 0.0, max = 1.0e15, step = 1.0, category = "LOD")]
    pub detail_distance: f64,
    /// Scheduling priority among terrain entries; this is not a residency cap.
    #[property(min = -1000000.0, max = 1000000.0, step = 1.0, category = "Streaming")]
    pub priority: i32,
    /// Whether external callers may submit canonical live edit/data batches.
    #[property(category = "Editing")]
    pub editable: bool,
    /// Generator/configuration revision, separate from the runtime chunk-data
    /// revision held with `payloads`. It is serialized with authored config
    /// but omitted from the property editor.
    pub source_revision: u64,
}

impl Default for VoxelTerrainComponent {
    fn default() -> Self {
        Self {
            payloads: empty_payload_store(),
            enabled: true,
            domain_mode: 1,
            shape_mode: 0,
            bounds_min_x: 0.0,
            bounds_min_y: 0.0,
            bounds_min_z: 0.0,
            bounds_max_x: 0.0,
            bounds_max_y: 0.0,
            bounds_max_z: 0.0,
            planet_radius: 1.0,
            voxel_size: 1.0,
            generator_id: String::new(),
            seed: 0,
            generator_parameters: String::new(),
            material_ids: Vec::new(),
            smooth_surface: false,
            target_error_pixels: 1.0,
            detail_distance: 0.0,
            priority: 0,
            editable: true,
            source_revision: 0,
        }
    }
}

impl VoxelTerrainComponent {
    /// Clone the live SceneDB-owned data handle for batch access or
    /// caller-controlled snapshot/exfiltration. The returned handle contains
    /// no persistence policy.
    pub fn payload_store(&self) -> VoxelPayloadStore {
        Arc::clone(&self.payloads)
    }
}

// Clone keeps a value-preserving independent live chunk index for SceneDB
// snapshots/transactions; immutable payload allocations remain shared.
impl Clone for VoxelTerrainComponent {
    fn clone(&self) -> Self {
        Self {
            payloads: clone_payload_store(&self.payloads),
            enabled: self.enabled,
            domain_mode: self.domain_mode,
            shape_mode: self.shape_mode,
            bounds_min_x: self.bounds_min_x,
            bounds_min_y: self.bounds_min_y,
            bounds_min_z: self.bounds_min_z,
            bounds_max_x: self.bounds_max_x,
            bounds_max_y: self.bounds_max_y,
            bounds_max_z: self.bounds_max_z,
            planet_radius: self.planet_radius,
            voxel_size: self.voxel_size,
            generator_id: self.generator_id.clone(),
            seed: self.seed,
            generator_parameters: self.generator_parameters.clone(),
            material_ids: self.material_ids.clone(),
            smooth_surface: self.smooth_surface,
            target_error_pixels: self.target_error_pixels,
            detail_distance: self.detail_distance,
            priority: self.priority,
            editable: self.editable,
            source_revision: self.source_revision,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pulsar_reflection::EngineClass;

    fn assert_runtime_storage<T>(component: &T, payloads: &VoxelPayloadStore)
    where
        T: EngineClass + serde::Serialize + for<'de> serde::Deserialize<'de>,
    {
        assert!(payloads.read().unwrap().1.is_empty());
        let serialized = serde_json::to_value(component).unwrap();
        assert!(serialized.get("payloads").is_none());

        let restored: T = serde_json::from_value(serialized).unwrap();
        let restored_json = serde_json::to_value(restored).unwrap();
        assert!(restored_json.get("payloads").is_none());
    }

    #[test]
    fn voxel_component_runtime_payloads_are_empty_hidden_and_not_serialized() {
        let component = VoxelComponent::default();
        assert_runtime_storage(&component, &component.payloads);
        assert!(!component
            .get_properties()
            .iter()
            .any(|property| property.name == "payloads"));
    }

    #[test]
    fn terrain_component_runtime_payloads_are_empty_hidden_and_not_serialized() {
        let component = VoxelTerrainComponent::default();
        assert_runtime_storage(&component, &component.payloads);
        assert!(!component
            .get_properties()
            .iter()
            .any(|property| property.name == "payloads"));
    }

    #[test]
    fn cloned_component_preserves_live_state_without_aliasing_future_mutations() {
        let original = VoxelTerrainComponent::default();
        let payload: Arc<[u8]> = Arc::from([1, 2, 3]);
        original
            .payloads
            .write()
            .unwrap()
            .1
            .insert([u64::MAX, 0, 42, 7], payload.clone());
        let clone = original.clone();

        assert!(!Arc::ptr_eq(&original.payloads, &clone.payloads));
        assert_eq!(original.payloads.read().unwrap().1.len(), 1);
        assert_eq!(clone.payloads.read().unwrap().1.len(), 1);
        assert_eq!(clone.payloads.read().unwrap().0, 0);

        clone
            .payloads
            .write()
            .unwrap()
            .1
            .remove(&[u64::MAX, 0, 42, 7]);
        assert_eq!(original.payloads.read().unwrap().1.len(), 1);

        // Shared snapshots are explicit and retain immutable bytes cheaply.
        let shared_handle = Arc::clone(&original.payloads);
        let shared_payload = shared_handle.read().unwrap().1[&[u64::MAX, 0, 42, 7]].clone();
        assert!(Arc::ptr_eq(&payload, &shared_payload));
    }
}
