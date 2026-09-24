//! Reflected, SceneDB-owned authoring components for voxel objects and terrain.
//!
//! These structs contain reflected configuration and runtime-only live data.
//! Voxel encodings, edit batches, and generation behavior belong to the
//! renderer-independent data API. No persistence behavior is implied by the
//! runtime data fields.

use engine_class_derive::engine_class;
use helio_voxel_data::VoxelStoredPayload;
pub use helio_voxel_data::{VoxelPayloadKey, VoxelPayloadStore};
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

/// Generic live payload storage owned by a voxel component row.
///
/// Keys are intentionally opaque to this crate; voxel chunk/key semantics are
/// defined by the voxel data contract. The tuple contains the live data revision and
/// payload map so a batch can publish both under one lock. Values are immutable
/// so readers can retain a cheap snapshot while a writer replaces one entry.
/// Component clones create fresh stores; callers can explicitly clone the
/// `Arc` when shared access is intended. Four opaque words allow collision-free
/// chunk keys (for example signed XYZ bit patterns plus an LOD word).
fn empty_payload_store() -> VoxelPayloadStore {
    Arc::new(RwLock::new((0, HashMap::new())))
}

fn default_voxel_generator_version() -> u32 {
    1
}

fn default_chunk_edge_voxels() -> u32 {
    8
}

fn default_max_chunk_lod() -> u32 {
    16
}

fn default_lod_scale() -> u32 {
    2
}

fn filled_cube_payload_store(dimensions: [u32; 3], slot: u8) -> VoxelPayloadStore {
    let mut chunks = HashMap::new();
    for z in 0..dimensions[2].div_ceil(8) {
        for y in 0..dimensions[1].div_ceil(8) {
            for x in 0..dimensions[0].div_ceil(8) {
                let mut samples = [0u8; 8 * 8 * 8];
                for local_z in 0..8 {
                    for local_y in 0..8 {
                        for local_x in 0..8 {
                            if x * 8 + local_x < dimensions[0]
                                && y * 8 + local_y < dimensions[1]
                                && z * 8 + local_z < dimensions[2]
                            {
                                samples[(local_z * 64 + local_y * 8 + local_x) as usize] = slot;
                            }
                        }
                    }
                }
                chunks.insert(
                    [u64::from(x), u64::from(y), u64::from(z), 0],
                    VoxelStoredPayload::raw_material(samples),
                );
            }
        }
    }
    Arc::new(RwLock::new((0, chunks)))
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
pub struct VoxelComponent {
    /// Runtime-only live payloads and data revision. Not an inspector
    /// property, serialized configuration, or GPU-mirrored field. Component
    /// clones copy the index/revision and share immutable payload allocations.
    #[serde(skip)]
    payloads: VoxelPayloadStore,
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
}

impl Default for VoxelComponent {
    fn default() -> Self {
        Self {
            payloads: filled_cube_payload_store([16; 3], 1),
            enabled: true,
            voxel_size: 1.0,
            dimensions: [16; 3],
            material_ids: vec![0],
            default_material_slot: 1,
            editable: true,
        }
    }
}

impl VoxelComponent {
    /// Construct a filled, editable volume in the canonical 8³ material-chunk
    /// encoding. Partial edge chunks are zero-padded; material slot zero is air.
    pub fn filled_cube(
        dimensions: [u32; 3],
        material_ids: Vec<u32>,
        slot: u8,
    ) -> Result<Self, &'static str> {
        if dimensions.iter().any(|&size| size == 0 || size > 256) {
            return Err("cube dimensions must be within 1..=256 on each axis");
        }
        if material_ids.len() > 255 || slot == 0 || usize::from(slot) > material_ids.len() {
            return Err("cube material slot must reference one of at most 255 material IDs");
        }
        Ok(Self {
            payloads: filled_cube_payload_store(dimensions, slot),
            enabled: true,
            voxel_size: 1.0,
            dimensions,
            material_ids,
            default_material_slot: u32::from(slot),
            editable: true,
        })
    }
    /// Low-level live SceneDB data capability. Normal
    /// producers should mutate through `VoxelSourceWriter` so validation and
    /// revision checks are preserved; scripts should export via data snapshots.
    /// This handle carries no persistence policy.
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
        }
    }
}

/// General-purpose voxel terrain authoring configuration.
///
/// `domain_mode` describes the chunk-key domain. Generation behavior is
/// identified by an opaque source ID and parameters, not by a built-in shape.
#[engine_class(category = "Voxel/Terrain", debug, serialize, deserialize)]
#[category("Domain", category_color = "#8F8F8F")]
#[category("Generation", category_color = "#D1A73F")]
#[category("Materials", category_color = "#D1A73F")]
#[category("Editing", category_color = "#D18F6F")]
pub struct VoxelTerrainComponent {
    /// Runtime-only live payloads and data revision. Not an inspector
    /// property, serialized configuration, or GPU-mirrored field. Component
    /// clones copy the index/revision and share immutable payload allocations.
    #[serde(skip)]
    payloads: VoxelPayloadStore,
    /// Whether this terrain source participates in rendering and queries.
    #[property]
    pub enabled: bool,
    /// Domain discriminant: 0 bounded, 1 unbounded.
    #[property(category = "Domain")]
    pub domain_mode: u32,
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
    /// Edge length of a base-resolution voxel in world units.
    #[property(min = 0.0001, max = 10000.0, step = 0.01, category = "Generation")]
    pub voxel_size: f64,
    /// Number of base-resolution voxels covered by one chunk key on each
    /// axis at LOD zero. The payload format can encode that region as samples,
    /// a hierarchy, a compressed field, or another registered representation.
    #[serde(default = "default_chunk_edge_voxels")]
    #[property(category = "Generation")]
    pub chunk_edge_voxels: u32,
    /// Highest chunk LOD accepted for this terrain source.
    #[serde(default = "default_max_chunk_lod")]
    #[property(category = "Generation")]
    pub max_chunk_lod: u32,
    /// Spatial scale between adjacent chunk LODs. Two means each coarser
    /// chunk covers twice the width of a finer chunk along each axis.
    #[serde(default = "default_lod_scale")]
    #[property(category = "Generation")]
    pub lod_scale: u32,
    /// Stable registered generator/source identifier. Empty means externally
    /// supplied data only; generator implementation is not stored here.
    #[property(category = "Generation")]
    pub generator_id: String,
    /// Stable implementation version. Changing it invalidates generated
    /// output without serializing executable generator code.
    #[serde(default = "default_voxel_generator_version")]
    #[property(category = "Generation")]
    pub generator_version: u32,
    /// Seed supplied to the registered generator.
    #[property(category = "Generation")]
    pub seed: u64,
    /// Opaque serialized parameters consumed by the registered generator.
    #[property(category = "Generation")]
    pub generator_parameters: String,
    /// Palette of IDs into Helio's existing SceneDB material records.
    #[property(category = "Materials")]
    pub material_ids: Vec<u32>,
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
            bounds_min_x: 0.0,
            bounds_min_y: 0.0,
            bounds_min_z: 0.0,
            bounds_max_x: 0.0,
            bounds_max_y: 0.0,
            bounds_max_z: 0.0,
            voxel_size: 1.0,
            chunk_edge_voxels: default_chunk_edge_voxels(),
            max_chunk_lod: default_max_chunk_lod(),
            lod_scale: default_lod_scale(),
            generator_id: String::new(),
            generator_version: 1,
            seed: 0,
            generator_parameters: String::new(),
            material_ids: vec![0],
            editable: true,
            source_revision: 0,
        }
    }
}

impl VoxelTerrainComponent {
    /// Low-level live SceneDB data capability. Normal
    /// producers should mutate through `VoxelSourceWriter` so validation and
    /// revision checks are preserved; scripts should export via data snapshots.
    /// This handle carries no persistence policy.
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
            bounds_min_x: self.bounds_min_x,
            bounds_min_y: self.bounds_min_y,
            bounds_min_z: self.bounds_min_z,
            bounds_max_x: self.bounds_max_x,
            bounds_max_y: self.bounds_max_y,
            bounds_max_z: self.bounds_max_z,
            voxel_size: self.voxel_size,
            chunk_edge_voxels: self.chunk_edge_voxels,
            max_chunk_lod: self.max_chunk_lod,
            lod_scale: self.lod_scale,
            generator_id: self.generator_id.clone(),
            generator_version: self.generator_version,
            seed: self.seed,
            generator_parameters: self.generator_parameters.clone(),
            material_ids: self.material_ids.clone(),
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
    fn voxel_component_starts_as_eight_filled_chunks_hidden_from_serialization() {
        let component = VoxelComponent::default();
        let store = component.payload_store();
        let state = store.read().unwrap();
        assert_eq!(state.1.len(), 8);
        assert!(
            state
                .1
                .values()
                .all(|bytes| bytes.len() == 512 && bytes.iter().all(|&slot| slot == 1))
        );
        drop(state);
        let serialized = serde_json::to_value(&component).unwrap();
        assert!(serialized.get("payloads").is_none());
        assert!(
            !component
                .get_properties()
                .iter()
                .any(|property| property.name == "payloads")
        );
    }

    #[test]
    fn filled_cube_zero_pads_partial_edges_and_checks_palette() {
        let cube = VoxelComponent::filled_cube([9, 1, 1], vec![7], 1).unwrap();
        let store = cube.payload_store();
        let state = store.read().unwrap();
        assert_eq!(state.1.len(), 2);
        assert_eq!(state.1[&[0, 0, 0, 0]][0], 1);
        assert_eq!(state.1[&[0, 0, 0, 0]][1], 1);
        assert_eq!(state.1[&[0, 0, 0, 0]][8], 0);
        assert_eq!(state.1[&[1, 0, 0, 0]][0], 1);
        assert_eq!(state.1[&[1, 0, 0, 0]][1], 0);
        assert!(VoxelComponent::filled_cube([0, 1, 1], vec![7], 1).is_err());
        assert!(VoxelComponent::filled_cube([1, 1, 1], vec![], 1).is_err());
    }

    #[test]
    fn terrain_component_runtime_payloads_are_empty_hidden_and_not_serialized() {
        let component = VoxelTerrainComponent::default();
        assert_runtime_storage(&component, &component.payloads);
        assert!(
            !component
                .get_properties()
                .iter()
                .any(|property| property.name == "payloads")
        );
    }

    #[test]
    fn older_terrain_config_defaults_generator_version_and_chunk_layout() {
        let mut value = serde_json::to_value(VoxelTerrainComponent::default()).unwrap();
        value.as_object_mut().unwrap().remove("generator_version");
        value.as_object_mut().unwrap().remove("chunk_edge_voxels");
        value.as_object_mut().unwrap().remove("max_chunk_lod");
        value.as_object_mut().unwrap().remove("lod_scale");
        let restored: VoxelTerrainComponent = serde_json::from_value(value).unwrap();
        assert_eq!(restored.generator_version, 1);
        assert_eq!(restored.chunk_edge_voxels, 8);
        assert_eq!(restored.max_chunk_lod, 16);
        assert_eq!(restored.lod_scale, 2);
    }

    #[test]
    fn cloned_component_preserves_live_state_without_aliasing_future_mutations() {
        let original = VoxelTerrainComponent::default();
        let payload: Arc<[u8]> = Arc::from([1, 2, 3]);
        original.payloads.write().unwrap().1.insert(
            [u64::MAX, 0, 42, 7],
            VoxelStoredPayload::raw_material(payload.clone()),
        );
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
        let shared_payload = shared_handle.read().unwrap().1[&[u64::MAX, 0, 42, 7]]
            .bytes
            .clone();
        assert!(Arc::ptr_eq(&payload, &shared_payload));
    }
}
