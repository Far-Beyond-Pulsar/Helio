//! Reflected, SceneDB-owned authoring components for voxel objects and terrain.
//!
//! These structs contain reflected configuration and runtime-only live data.
//! Voxel encodings, edit batches, and generation behavior belong to the
//! renderer-independent data API. No persistence behavior is implied by the
//! runtime data fields.

use engine_class_derive::engine_class;
use helio_voxel_data::{
    VoxelEditJournal,
    VoxelStoredPayload, VOXEL_TERRAIN_GENERATOR, VOXEL_TERRAIN_GENERATOR_VERSION,
};
pub use helio_voxel_data::{VoxelPayloadKey, VoxelPayloadStore};
use pulsar_scene_model::components::Transform;
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

/// Files without a version use the generator's registered version.
fn default_voxel_generator_version() -> u32 {
    0
}

/// The registered terrain generator that fills a world: its id and output
/// version. Serialized flat into the terrain component (`generator_id`,
/// `generator_version`); the editor shows it as a searchable picker of the
/// registered generators.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct VoxelGeneratorRef {
    /// Stable generator id; empty means externally supplied data only.
    #[serde(rename = "generator_id", default)]
    pub id: String,
    /// Output version; a new version may generate different terrain.
    #[serde(rename = "generator_version", default = "default_voxel_generator_version")]
    pub version: u32,
}

impl VoxelGeneratorRef {
    pub fn new(id: impl Into<String>, version: u32) -> Self {
        Self { id: id.into(), version }
    }
}

impl Default for VoxelGeneratorRef {
    /// The layered terrain generator.
    fn default() -> Self {
        Self::new(VOXEL_TERRAIN_GENERATOR, VOXEL_TERRAIN_GENERATOR_VERSION)
    }
}

fn serialize_generator_ref_json(value: &VoxelGeneratorRef) -> pulsar_reflection::ReflectResult<serde_json::Value> {
    serde_json::to_value(value).map_err(|e| pulsar_reflection::ReflectError::SerializationFailed(e.to_string()))
}

fn deserialize_generator_ref_json(value: serde_json::Value) -> pulsar_reflection::ReflectResult<VoxelGeneratorRef> {
    serde_json::from_value(value).map_err(|e| pulsar_reflection::ReflectError::DeserializationFailed(e.to_string()))
}

/// Registered for reflection; the picker editor is registered by the host
/// that knows the generator registry.
#[pulsar_reflection::pulsar_type(
    serialize_json_with = serialize_generator_ref_json,
    deserialize_json_with = deserialize_generator_ref_json
)]
#[allow(dead_code)]
type RegisteredVoxelGeneratorRef = VoxelGeneratorRef;

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
#[category("Rendering", category_color = "#7C9DC9")]
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
    /// Stable renderer backend identifier. Empty leaves selection to the host.
    #[serde(default)]
    #[property(category = "Rendering")]
    pub renderer_id: String,
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
            renderer_id: String::new(),
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
            renderer_id: String::new(),
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
            renderer_id: self.renderer_id.clone(),
            material_ids: self.material_ids.clone(),
            default_material_slot: self.default_material_slot,
            editable: self.editable,
        }
    }
}

/// Overall form of a voxel world.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize, pulsar_reflection::Reflectable)]
pub enum VoxelWorldShape {
    /// A planet centred on the entity origin (`planet_radius`).
    Sphere,
    /// A square plane of `plane_size` metres centred on the entity origin.
    #[default]
    Plane,
    /// A plane without edges within reach (about 13 400 km across).
    InfinitePlane,
}

fn default_planet_radius() -> f64 {
    6_371_000.0
}

fn default_plane_size() -> f64 {
    4_096.0
}

/// The base component of every voxel world, whatever generates it.
///
/// It owns what all terrains share: the world's shape and size, the voxel
/// size, which registered generator fills it (with a seed), the material
/// palette, editability, and the world's edits (an ordered brush journal
/// plus per-sample payload chunks). A generator's own settings live in its
/// settings component on the same entity (for example
/// [`VoxelTerrainLayersComponent`] for the built-in generator), or in the
/// opaque `generator_parameters` for generators without one. `domain_mode`
/// describes the chunk-key domain of live sample data.
#[engine_class(category = "Voxel/Terrain", debug, serialize, deserialize)]
#[category("World", category_color = "#6FA86F")]
#[category("Generation", category_color = "#D1A73F")]
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
    /// Planet, finite plane or infinite plane.
    #[serde(default)]
    #[property(category = "World")]
    pub shape: VoxelWorldShape,
    /// Planet radius in metres (sphere worlds).
    #[serde(default = "default_planet_radius")]
    #[property(min = 1000.0, max = 50000000.0, step = 1000.0, category = "World", label = "Planet radius (m)")]
    pub planet_radius: f64,
    /// Edge length of a finite plane in metres.
    #[serde(default = "default_plane_size")]
    #[property(min = 16.0, max = 13000000.0, step = 16.0, category = "World", label = "Plane size (m)")]
    pub plane_size: f64,
    /// Edge length of a base-resolution voxel in metres (0.1 to 1 for
    /// streamed terrain).
    #[property(min = 0.1, max = 1.0, step = 0.05, category = "World", label = "Voxel size (m)")]
    pub voxel_size: f64,
    /// The registered terrain generator that fills the world. Its settings
    /// live in its settings component on the same entity.
    #[serde(flatten)]
    #[property(category = "Generation")]
    pub generator: VoxelGeneratorRef,
    /// Seed supplied to the generator.
    #[property(category = "Generation")]
    pub seed: u64,
    /// Settings (JSON) for a generator without a settings component; a
    /// settings component on the entity replaces them.
    #[serde(default)]
    pub generator_parameters: String,

    /// Renderer appearance JSON, independent of generator data and edits.
    #[serde(default)]
    #[property(category = "Editing", label = "Terrain appearance (JSON)")]
    pub appearance_parameters: String,

    // The fields below describe the chunk domain of live sample data
    // (`payload_store`). Streamed terrain derives its own layout and LOD, so
    // they are serialized but not shown in the inspector.
    /// Domain discriminant: 0 bounded, 1 unbounded.
    pub domain_mode: u32,
    /// Finite-domain minimum and maximum on each axis. Ignored for unbounded domains.
    pub bounds_min_x: f64,
    pub bounds_min_y: f64,
    pub bounds_min_z: f64,
    pub bounds_max_x: f64,
    pub bounds_max_y: f64,
    pub bounds_max_z: f64,
    /// Number of base-resolution voxels covered by one chunk key on each
    /// axis at LOD zero. The payload format can encode that region as samples,
    /// a hierarchy, a compressed field, or another registered representation.
    #[serde(default = "default_chunk_edge_voxels")]
    pub chunk_edge_voxels: u32,
    /// Highest chunk LOD accepted for this terrain source.
    #[serde(default = "default_max_chunk_lod")]
    pub max_chunk_lod: u32,
    /// Spatial scale between adjacent chunk LODs. Two means each coarser
    /// chunk covers twice the width of a finer chunk along each axis.
    #[serde(default = "default_lod_scale")]
    pub lod_scale: u32,
    /// Stable renderer backend identifier. Empty selects the unique backend
    /// that supports the generator.
    #[serde(default)]
    pub renderer_id: String,
    /// Palette of IDs into Helio's existing SceneDB material records.
    pub material_ids: Vec<u32>,
    /// Whether external callers may submit canonical live edit/data batches.
    #[property(category = "Editing")]
    pub editable: bool,
    /// Generator/configuration revision, separate from the runtime chunk-data
    /// revision held with `payloads`. It is serialized with authored config
    /// but omitted from the property editor.
    pub source_revision: u64,
    /// Ordered shape edits (destruction and construction), saved with the
    /// level. Append through the sculpt tool or the scripting methods, which
    /// also advance `source_revision`.
    #[serde(default)]
    pub edits: VoxelEditJournal,
}

impl Default for VoxelTerrainComponent {
    fn default() -> Self {
        Self {
            payloads: empty_payload_store(),
            enabled: true,
            shape: VoxelWorldShape::default(),
            planet_radius: default_planet_radius(),
            plane_size: default_plane_size(),
            domain_mode: 1,
            bounds_min_x: 0.0,
            bounds_min_y: 0.0,
            bounds_min_z: 0.0,
            bounds_max_x: 0.0,
            bounds_max_y: 0.0,
            bounds_max_z: 0.0,
            voxel_size: 0.1,
            chunk_edge_voxels: default_chunk_edge_voxels(),
            max_chunk_lod: default_max_chunk_lod(),
            lod_scale: default_lod_scale(),
            renderer_id: String::new(),
            generator: VoxelGeneratorRef::default(),
            seed: 0,
            generator_parameters: String::new(),
            appearance_parameters: String::new(),
            material_ids: vec![0],
            editable: true,
            source_revision: 0,
            edits: VoxelEditJournal::default(),
        }
    }
}

impl VoxelTerrainComponent {
    /// A planet of `radius` metres with Helio's terrain generator and 0.1 m
    /// voxels. Add a [`VoxelTerrainLayersComponent`] to shape its continents and
    /// mountains.
    pub fn planet(radius: f64) -> Self {
        Self { shape: VoxelWorldShape::Sphere, planet_radius: radius, ..Self::default() }
    }
    /// A square plane of `size` metres with Helio's terrain generator.
    pub fn plane(size: f64) -> Self {
        Self { shape: VoxelWorldShape::Plane, plane_size: size, ..Self::default() }
    }
    /// A plane without edges within reach, with Helio's terrain generator.
    pub fn infinite_plane() -> Self {
        Self { shape: VoxelWorldShape::InfinitePlane, ..Self::default() }
    }

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
            shape: self.shape,
            planet_radius: self.planet_radius,
            plane_size: self.plane_size,
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
            renderer_id: self.renderer_id.clone(),
            generator: self.generator.clone(),
            seed: self.seed,
            generator_parameters: self.generator_parameters.clone(),
            appearance_parameters: self.appearance_parameters.clone(),
            material_ids: self.material_ids.clone(),
            editable: self.editable,
            source_revision: self.source_revision,
            edits: self.edits.clone(),
        }
    }
}

/// A solid terrain material.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize, pulsar_reflection::Reflectable)]
pub enum VoxelTerrainMaterial {
    #[default]
    Grass,
    Dirt,
    Stone,
    Sand,
    Snow,
    Water,
    Gravel,
    Sandstone,
    DarkStone,
    Wood,
    Leaves,
    Clay,
    Brick,
    Planks,
    Cobble,
}

/// What a terrain layer adds (`helio_pass_voxel_planet::layers::LayerKind`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize, pulsar_reflection::Reflectable)]
pub enum VoxelLayerKind {
    /// Rolling relief, bent by the warp (hills, highlands).
    #[default]
    Hills,
    /// Bends the layers after it (coastlines, ridge lines); first only.
    Warp,
    /// Continents and ocean basins: Height is the ocean depth, Base the
    /// lowland height. Provides the land masks.
    Continents,
    /// Ridged mountain ranges inside regions (Coverage, Region size).
    Mountains,
    /// Metre-scale detail; Ratio is its amplitude per wavelength.
    Roughness,
    /// Branching gullies down the slope of the coarser layers; full depth
    /// on slopes steeper than Ratio.
    Erosion,
    /// Crater sizes from Scale down: Coverage is the density of the
    /// largest, Persistence its growth per size, Ratio depth/diameter,
    /// Ratio 2 rim/depth, Ratio 3 the share of fresh craters.
    Craters,
    /// Smooth low plains covering Coverage of the surface, Height deep;
    /// they flatten the layers before them.
    Basins,
    /// A constant height (a flat world is one plateau).
    Plateau,
}

/// Where a terrain layer applies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize, pulsar_reflection::Reflectable)]
pub enum VoxelLayerMask {
    #[default]
    Everywhere,
    /// Land only (needs a Continents layer).
    Land,
    /// Everywhere but fading out under deep sea (needs a Continents layer).
    AboveDeepSea,
}

/// How generated cells get their materials.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize, pulsar_reflection::Reflectable)]
pub enum VoxelMaterialStyle {
    /// Meadows, dry lands, rock, strata and snow above the snowline.
    #[default]
    Earthlike,
    /// Regolith over bedrock, dark basins, bright young ejecta.
    Lunar,
    /// The surface material over Soil depth of soil over rock.
    Layered,
    /// The first material rule that holds, else the rock material: a
    /// game's own biomes.
    Rules,
}

/// One material rule: the material of a ground cell where every condition
/// holds (ranges are inclusive). Rules are tried in order.
#[engine_class(no_register, clone, debug, serialize, deserialize)]
#[serde(default)]
pub struct VoxelMaterialRule {
    #[property]
    pub material: VoxelTerrainMaterial,
    /// Column height above the datum.
    #[property(min = -100000.0, max = 100000.0, step = 10.0, label = "Min height (m)")]
    pub min_height_m: f64,
    #[property(min = -100000.0, max = 1000000.0, step = 10.0, label = "Max height (m)")]
    pub max_height_m: f64,
    /// Ground slope, rise over run (1 is 45 degrees).
    #[property(min = 0.0, max = 1000.0, step = 0.05)]
    pub min_slope: f64,
    #[property(min = 0.0, max = 1000.0, step = 0.05)]
    pub max_slope: f64,
    /// Depth below the column top; 0 is the exposed surface.
    #[property(min = 0.0, max = 1000000.0, step = 0.1, label = "Min depth (m)")]
    pub min_depth_m: f64,
    #[property(min = 0.0, max = 1000000.0, step = 0.1, label = "Max depth (m)")]
    pub max_depth_m: f64,
    /// Moisture, 0 (dry) to 1 (wet).
    #[property(min = 0.0, max = 1.0, step = 0.01)]
    pub min_moisture: f64,
    #[property(min = 0.0, max = 1.0, step = 0.01)]
    pub max_moisture: f64,
    /// Erosion, -1 (gully floors) to 1 (the ribs between gullies).
    #[property(min = -1.0, max = 1.0, step = 0.05)]
    pub min_erosion: f64,
    #[property(min = -1.0, max = 1.0, step = 0.05)]
    pub max_erosion: f64,
    /// Patches of this size (0: none) covering Patch share.
    #[property(min = 0.0, max = 10000.0, step = 0.01, label = "Patch size (km)")]
    pub patch_km: f64,
    #[property(min = 0.0, max = 1.0, step = 0.01)]
    pub patch_share: f64,
    /// Strata bands of this thickness (0: none), odd or even ones.
    #[property(min = 0.0, max = 10000.0, step = 0.1, label = "Band (m)")]
    pub band_m: f64,
    #[property]
    pub odd_bands: bool,
    /// Share of the cells as single-cell specks (1: all cells).
    #[property(min = 0.0, max = 1.0, step = 0.01)]
    pub speck_share: f64,
}

impl Default for VoxelMaterialRule {
    fn default() -> Self {
        Self {
            material: VoxelTerrainMaterial::Stone,
            min_height_m: -1.0e6,
            max_height_m: 1.0e6,
            min_slope: 0.0,
            max_slope: 1.0e3,
            min_depth_m: 0.0,
            max_depth_m: 1.0e6,
            min_moisture: 0.0,
            max_moisture: 1.0,
            min_erosion: -1.0,
            max_erosion: 1.0,
            patch_km: 0.0,
            patch_share: 0.5,
            band_m: 0.0,
            odd_bands: false,
            speck_share: 1.0,
        }
    }
}

impl PartialEq for VoxelMaterialRule {
    fn eq(&self, other: &Self) -> bool {
        serde_json::to_value(self).ok() == serde_json::to_value(other).ok()
    }
}

impl VoxelMaterialRule {
    /// `material` everywhere (narrow it with the fields).
    pub fn new(material: VoxelTerrainMaterial) -> Self {
        Self { material, ..Self::default() }
    }
}

fn serialize_material_rule_json(value: &VoxelMaterialRule) -> pulsar_reflection::ReflectResult<serde_json::Value> {
    serde_json::to_value(value).map_err(|e| pulsar_reflection::ReflectError::SerializationFailed(e.to_string()))
}

fn deserialize_material_rule_json(value: serde_json::Value) -> pulsar_reflection::ReflectResult<VoxelMaterialRule> {
    serde_json::from_value(value).map_err(|e| pulsar_reflection::ReflectError::DeserializationFailed(e.to_string()))
}

#[pulsar_reflection::pulsar_type(
    serialize_json_with = serialize_material_rule_json,
    deserialize_json_with = deserialize_material_rule_json
)]
#[allow(dead_code)]
type RegisteredVoxelMaterialRule = VoxelMaterialRule;

/// One layer of a terrain stack. Fields mean what the layer's kind says;
/// unused ones are ignored.
#[engine_class(no_register, clone, debug, serialize, deserialize)]
#[serde(default)]
pub struct VoxelTerrainLayer {
    #[property]
    pub kind: VoxelLayerKind,
    #[property]
    pub enabled: bool,
    #[property]
    pub mask: VoxelLayerMask,
    /// Main height: first octave amplitude, basin or ocean depth, plateau height.
    #[property(min = -100000.0, max = 100000.0, step = 1.0, label = "Height (m)")]
    pub height_m: f64,
    /// Secondary height: the lowlands of Continents.
    #[property(min = -100000.0, max = 100000.0, step = 1.0, label = "Base (m)")]
    pub base_m: f64,
    /// Wavelength of the first octave or the largest feature.
    #[property(min = 0.001, max = 100000.0, step = 0.1, label = "Scale (km)")]
    pub scale_km: f64,
    #[property(min = 0.0, max = 16.0, step = 1.0)]
    pub octaves: u32,
    /// Amplitude ratio between octaves (craters: density growth).
    #[property(min = 0.0, max = 3.0, step = 0.01)]
    pub persistence: f64,
    /// Share of the surface covered (mountain regions, basins, craters).
    #[property(min = 0.0, max = 1.0, step = 0.01)]
    pub coverage: f64,
    /// Size of the regions the layer occupies.
    #[property(min = 0.1, max = 100000.0, step = 1.0, label = "Region size (km)")]
    pub region_km: f64,
    #[property(min = 0.0, max = 4.0, step = 0.005)]
    pub ratio: f64,
    #[property(min = 0.0, max = 4.0, step = 0.01, label = "Ratio 2")]
    pub ratio2: f64,
    #[property(min = 0.0, max = 1.0, step = 0.01, label = "Ratio 3")]
    pub ratio3: f64,
}

impl Default for VoxelTerrainLayer {
    fn default() -> Self {
        Self {
            kind: VoxelLayerKind::Hills,
            enabled: true,
            mask: VoxelLayerMask::Everywhere,
            height_m: 100.0,
            base_m: 0.0,
            scale_km: 10.0,
            octaves: 4,
            persistence: 0.5,
            coverage: 0.5,
            region_km: 100.0,
            ratio: 0.0,
            ratio2: 0.0,
            ratio3: 0.0,
        }
    }
}

impl PartialEq for VoxelTerrainLayer {
    fn eq(&self, other: &Self) -> bool {
        serde_json::to_value(self).ok() == serde_json::to_value(other).ok()
    }
}

impl VoxelTerrainLayer {
    /// A layer of `kind` with the default parameters.
    pub fn new(kind: VoxelLayerKind) -> Self {
        Self { kind, ..Self::default() }
    }
}

fn serialize_terrain_layer_json(value: &VoxelTerrainLayer) -> pulsar_reflection::ReflectResult<serde_json::Value> {
    serde_json::to_value(value).map_err(|e| pulsar_reflection::ReflectError::SerializationFailed(e.to_string()))
}

fn deserialize_terrain_layer_json(value: serde_json::Value) -> pulsar_reflection::ReflectResult<VoxelTerrainLayer> {
    serde_json::from_value(value).map_err(|e| pulsar_reflection::ReflectError::DeserializationFailed(e.to_string()))
}

#[pulsar_reflection::pulsar_type(
    serialize_json_with = serialize_terrain_layer_json,
    deserialize_json_with = deserialize_terrain_layer_json
)]
#[allow(dead_code)]
type RegisteredVoxelTerrainLayer = VoxelTerrainLayer;

/// Generated caves: tunnels and caverns inside cave regions.
#[engine_class(no_register, clone, debug, serialize, deserialize)]
#[category("Caves", category_color = "#7A6A9E")]
#[derive(pulsar_reflection::Reflectable)]
#[serde(default)]
pub struct VoxelCaves {
    #[property(category = "Caves")]
    pub enabled: bool,
    /// Deepest cave cell below the local surface.
    #[property(min = 0.0, max = 170.0, step = 1.0, category = "Caves", label = "Depth (m)")]
    pub depth_m: f64,
    /// Rough share of the land inside cave regions.
    #[property(min = 0.0, max = 1.0, step = 0.01, category = "Caves", label = "Regions (share)")]
    pub share: f64,
    #[property(min = 0.1, max = 500.0, step = 0.1, category = "Caves", label = "Region size (km)")]
    pub region_km: f64,
    #[property(min = 0.0, max = 20.0, step = 0.1, category = "Caves", label = "Tunnel radius (m)")]
    pub tunnel_radius_m: f64,
    #[property(min = 4.0, max = 2000.0, step = 1.0, category = "Caves", label = "Tunnel winding (m)")]
    pub tunnel_wavelength_m: f64,
    #[property(min = 8.0, max = 4000.0, step = 1.0, category = "Caves", label = "Cavern size (m)")]
    pub cavern_wavelength_m: f64,
    /// Rough share of the cave volume opened as caverns.
    #[property(min = 0.0, max = 1.0, step = 0.01, category = "Caves", label = "Caverns (share)")]
    pub cavern_share: f64,
    /// Rock kept above caverns (tunnels may open into hillsides).
    #[property(min = 0.0, max = 100.0, step = 0.5, category = "Caves", label = "Cavern cover (m)")]
    pub cover_m: f64,
}

impl Default for VoxelCaves {
    fn default() -> Self {
        Self {
            enabled: true,
            depth_m: 120.0,
            share: 0.45,
            region_km: 6.0,
            tunnel_radius_m: 2.5,
            tunnel_wavelength_m: 160.0,
            cavern_wavelength_m: 160.0,
            cavern_share: 0.04,
            cover_m: 4.0,
        }
    }
}

/// Generated overhangs and arches: the surface folded in 3D.
#[engine_class(no_register, clone, debug, serialize, deserialize)]
#[category("Overhangs", category_color = "#A8826F")]
#[derive(pulsar_reflection::Reflectable)]
#[serde(default)]
pub struct VoxelOverhangs {
    #[property(category = "Overhangs")]
    pub enabled: bool,
    /// Largest displacement over the heightfield.
    #[property(min = 0.0, max = 20.0, step = 0.5, category = "Overhangs", label = "Height (m)")]
    pub height_m: f64,
    #[property(min = 4.0, max = 500.0, step = 1.0, category = "Overhangs", label = "Size (m)")]
    pub wavelength_m: f64,
    #[property(min = 0.1, max = 500.0, step = 0.1, category = "Overhangs", label = "Region size (km)")]
    pub region_km: f64,
    #[property(min = 0.0, max = 1.0, step = 0.01, category = "Overhangs", label = "Regions (share)")]
    pub share: f64,
}

impl Default for VoxelOverhangs {
    fn default() -> Self {
        Self { enabled: true, height_m: 6.0, wavelength_m: 24.0, region_km: 3.0, share: 0.3 }
    }
}

/// An ordered terrain layer stack, its caves and overhangs and its
/// materials: the `helio.terrain` settings, edited as one value (the
/// inspector's stack editor), so a preset replaces all of it at once.
#[engine_class(no_register, clone, debug, serialize, deserialize)]
#[serde(default)]
pub struct VoxelTerrainStack {
    /// Applied in order; at most eight enabled.
    #[property]
    pub layers: Vec<VoxelTerrainLayer>,
    #[property]
    pub caves: VoxelCaves,
    #[property]
    pub overhangs: VoxelOverhangs,
    #[property(label = "Material style")]
    pub materials: VoxelMaterialStyle,
    /// Earthlike: flat ground above this height is snow.
    #[property(min = -10000.0, max = 100000.0, step = 10.0, label = "Snowline (m)")]
    pub snowline_m: f64,
    /// Depth of the soil (Earthlike, Layered) or regolith (Lunar).
    #[property(min = 0.0, max = 1000.0, step = 0.1, label = "Soil depth (m)")]
    pub soil_depth_m: f64,
    /// Layered: the surface, soil and rock materials (rock is also the
    /// Rules style's fallback).
    #[property]
    pub surface: VoxelTerrainMaterial,
    #[property]
    pub soil: VoxelTerrainMaterial,
    #[property]
    pub rock: VoxelTerrainMaterial,
    /// Rules style: tried in order, at most 16.
    #[property]
    pub rules: Vec<VoxelMaterialRule>,
}

impl Default for VoxelTerrainStack {
    fn default() -> Self {
        Self::earth()
    }
}

impl PartialEq for VoxelTerrainStack {
    fn eq(&self, other: &Self) -> bool {
        serde_json::to_value(self).ok() == serde_json::to_value(other).ok()
    }
}

fn serialize_terrain_stack_json(value: &VoxelTerrainStack) -> pulsar_reflection::ReflectResult<serde_json::Value> {
    serde_json::to_value(value).map_err(|e| pulsar_reflection::ReflectError::SerializationFailed(e.to_string()))
}

fn deserialize_terrain_stack_json(value: serde_json::Value) -> pulsar_reflection::ReflectResult<VoxelTerrainStack> {
    serde_json::from_value(value).map_err(|e| pulsar_reflection::ReflectError::DeserializationFailed(e.to_string()))
}

/// Registered for reflection; its editor (`voxel_stack_editor`) is
/// registered there.
#[pulsar_reflection::pulsar_type(
    serialize_json_with = serialize_terrain_stack_json,
    deserialize_json_with = deserialize_terrain_stack_json
)]
#[allow(dead_code)]
type RegisteredVoxelTerrainStack = VoxelTerrainStack;

/// Settings of the layered terrain generator (`helio.terrain`): an ordered
/// stack of layers (continents, mountains, erosion, hills, craters,
/// basins, plateaus), caves, overhangs and materials. Any world is a stack:
/// an Earth-like planet, a cratered moon, a desert, a flat block world, or
/// one a game builds from a seed. It configures the [`VoxelTerrainComponent`]
/// on the same entity, whose seed varies it; its serialized form (the
/// stack's fields, flattened) is the generator's settings JSON.
#[engine_class(category = "Voxel/Terrain", clone, debug, serialize, deserialize)]
#[category("Terrain", category_color = "#6FA86F")]
#[serde(default)]
pub struct VoxelTerrainLayersComponent {
    #[serde(flatten)]
    #[property(category = "Terrain", label = "Terrain stack")]
    pub stack: VoxelTerrainStack,
}

impl Default for VoxelTerrainLayersComponent {
    fn default() -> Self {
        Self::earth()
    }
}

impl VoxelTerrainLayersComponent {
    pub fn earth() -> Self {
        Self { stack: VoxelTerrainStack::earth() }
    }
    pub fn moon() -> Self {
        Self { stack: VoxelTerrainStack::moon() }
    }
    pub fn desert() -> Self {
        Self { stack: VoxelTerrainStack::desert() }
    }
    pub fn flat(height_m: f64) -> Self {
        Self { stack: VoxelTerrainStack::flat(height_m) }
    }
}

impl VoxelTerrainStack {
    /// The presets by name: "earth", "moon", "desert" and "flat".
    pub const PRESETS: [&'static str; 4] = ["earth", "moon", "desert", "flat"];

    /// A preset by name (case-insensitive).
    pub fn preset(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "earth" => Some(Self::earth()),
            "moon" => Some(Self::moon()),
            "desert" => Some(Self::desert()),
            "flat" => Some(Self::flat(0.0)),
            _ => None,
        }
    }

    /// Continents and oceans, ridged mountains with branching erosion,
    /// hills and roughness; caves, overhangs, meadows, rock and snow.
    pub fn earth() -> Self {
        use VoxelLayerKind::*;
        let layer = |kind, f: &dyn Fn(&mut VoxelTerrainLayer)| {
            let mut l = VoxelTerrainLayer::new(kind);
            f(&mut l);
            l
        };
        Self {
            layers: vec![
                layer(Warp, &|l| l.scale_km = 40.0),
                layer(Continents, &|l| {
                    l.scale_km = 3_000.0;
                    l.height_m = 2_400.0;
                    l.base_m = 180.0;
                }),
                layer(Mountains, &|l| {
                    l.mask = VoxelLayerMask::Land;
                    l.height_m = 2_400.0;
                    l.scale_km = 20.0;
                    l.octaves = 7;
                    l.persistence = 0.47;
                    l.region_km = 240.0;
                    l.coverage = 0.45;
                }),
                layer(Erosion, &|l| {
                    l.height_m = 40.0;
                    l.scale_km = 1.6;
                    l.octaves = 6;
                    l.ratio = 0.5;
                }),
                layer(Hills, &|l| {
                    l.mask = VoxelLayerMask::AboveDeepSea;
                    l.height_m = 140.0;
                    l.scale_km = 9.0;
                }),
                layer(Roughness, &|l| {
                    l.mask = VoxelLayerMask::AboveDeepSea;
                    l.scale_km = 0.512;
                    l.octaves = 9;
                    l.ratio = 0.035;
                }),
            ],
            caves: VoxelCaves::default(),
            overhangs: VoxelOverhangs::default(),
            materials: VoxelMaterialStyle::Earthlike,
            snowline_m: 3_000.0,
            soil_depth_m: 0.7,
            surface: VoxelTerrainMaterial::Grass,
            soil: VoxelTerrainMaterial::Dirt,
            rock: VoxelTerrainMaterial::Stone,
            rules: Vec::new(),
        }
    }

    /// Dunes and mesas with materials from rules: sand on gentle ground,
    /// sandstone and clay strata, gravel in gullies, stone patches and dark
    /// stone specks.
    pub fn desert() -> Self {
        use VoxelTerrainMaterial as M;
        let mut earth = Self::earth();
        earth.layers.retain(|l| l.kind != VoxelLayerKind::Continents);
        for l in &mut earth.layers {
            l.mask = VoxelLayerMask::Everywhere;
        }
        earth.layers.insert(1, VoxelTerrainLayer { height_m: 400.0, ..VoxelTerrainLayer::new(VoxelLayerKind::Plateau) });
        Self {
            materials: VoxelMaterialStyle::Rules,
            rules: vec![
                VoxelMaterialRule { max_depth_m: 0.3, max_erosion: -0.4, min_slope: 0.2, ..VoxelMaterialRule::new(M::Gravel) },
                VoxelMaterialRule { max_depth_m: 1.5, max_slope: 0.6, ..VoxelMaterialRule::new(M::Sand) },
                VoxelMaterialRule { max_depth_m: 0.0, speck_share: 0.06, ..VoxelMaterialRule::new(M::DarkStone) },
                VoxelMaterialRule { patch_km: 0.05, patch_share: 0.15, ..VoxelMaterialRule::new(M::Stone) },
                VoxelMaterialRule { band_m: 2.5, odd_bands: true, ..VoxelMaterialRule::new(M::Clay) },
            ],
            rock: M::Sandstone,
            snowline_m: 1.0e5,
            ..earth
        }
    }

    /// Cratered highlands and dark basalt plains over regolith, no caves.
    pub fn moon() -> Self {
        use VoxelLayerKind::*;
        Self {
            layers: vec![
                VoxelTerrainLayer { height_m: 1_500.0, scale_km: 250.0, octaves: 3, ..VoxelTerrainLayer::new(Hills) },
                VoxelTerrainLayer { height_m: 1_200.0, scale_km: 900.0, coverage: 0.3, ..VoxelTerrainLayer::new(Basins) },
                VoxelTerrainLayer {
                    scale_km: 40.0,
                    octaves: 10,
                    coverage: 0.3,
                    persistence: 1.25,
                    ratio: 0.2,
                    ratio2: 0.3,
                    ratio3: 0.15,
                    ..VoxelTerrainLayer::new(Craters)
                },
            ],
            caves: VoxelCaves { enabled: false, ..VoxelCaves::default() },
            overhangs: VoxelOverhangs { enabled: false, ..VoxelOverhangs::default() },
            materials: VoxelMaterialStyle::Lunar,
            soil_depth_m: 4.0,
            ..Self::earth()
        }
    }

    /// Level ground at `height_m`: grass over a metre of dirt over stone.
    pub fn flat(height_m: f64) -> Self {
        Self {
            layers: vec![VoxelTerrainLayer { height_m, ..VoxelTerrainLayer::new(VoxelLayerKind::Plateau) }],
            caves: VoxelCaves { enabled: false, ..VoxelCaves::default() },
            overhangs: VoxelOverhangs { enabled: false, ..VoxelOverhangs::default() },
            materials: VoxelMaterialStyle::Layered,
            soil_depth_m: 1.0,
            ..Self::earth()
        }
    }
}

// SceneDB component methods are the stable scripting/registry surface for
// individual edits and bounded brush batches. The editor uses
// `VoxelSourceSession` for asynchronous, generation-checked brush strokes.
#[pulsar_scenedb::component_methods]
impl VoxelComponent {
    #[world_method]
    fn paint_sample(
        world: &mut pulsar_scenedb::World,
        entity: pulsar_scenedb::Entity,
        x: i64,
        y: i64,
        z: i64,
        material_slot: u8,
    ) -> Result<(), String> {
        validate_paint_slot(material_slot)?;
        edit_object_sample(world, entity, [x, y, z], material_slot)
    }

    #[world_method]
    fn paint_samples(
        world: &mut pulsar_scenedb::World,
        entity: pulsar_scenedb::Entity,
        samples: Vec<[i64; 3]>,
        material_slot: u8,
    ) -> Result<(), String> {
        validate_paint_slot(material_slot)?;
        edit_object_samples(world, entity, &samples, material_slot)
    }

    #[world_method]
    fn erase_sample(
        world: &mut pulsar_scenedb::World,
        entity: pulsar_scenedb::Entity,
        x: i64,
        y: i64,
        z: i64,
    ) -> Result<(), String> {
        edit_object_sample(world, entity, [x, y, z], 0)
    }

    #[world_method]
    fn erase_samples(
        world: &mut pulsar_scenedb::World,
        entity: pulsar_scenedb::Entity,
        samples: Vec<[i64; 3]>,
    ) -> Result<(), String> {
        edit_object_samples(world, entity, &samples, 0)
    }
}

#[pulsar_scenedb::component_methods]
impl VoxelTerrainComponent {
    #[world_method]
    fn paint_sample(
        world: &mut pulsar_scenedb::World,
        entity: pulsar_scenedb::Entity,
        x: i64,
        y: i64,
        z: i64,
        material_slot: u8,
    ) -> Result<(), String> {
        validate_paint_slot(material_slot)?;
        edit_terrain_sample(world, entity, [x, y, z], material_slot)
    }

    #[world_method]
    fn paint_samples(
        world: &mut pulsar_scenedb::World,
        entity: pulsar_scenedb::Entity,
        samples: Vec<[i64; 3]>,
        material_slot: u8,
    ) -> Result<(), String> {
        validate_paint_slot(material_slot)?;
        edit_terrain_samples(world, entity, &samples, material_slot)
    }

    #[world_method]
    fn erase_sample(
        world: &mut pulsar_scenedb::World,
        entity: pulsar_scenedb::Entity,
        x: i64,
        y: i64,
        z: i64,
    ) -> Result<(), String> {
        edit_terrain_sample(world, entity, [x, y, z], 0)
    }

    #[world_method]
    fn erase_samples(
        world: &mut pulsar_scenedb::World,
        entity: pulsar_scenedb::Entity,
        samples: Vec<[i64; 3]>,
    ) -> Result<(), String> {
        edit_terrain_samples(world, entity, &samples, 0)
    }
}

fn edit_object_sample(
    world: &mut pulsar_scenedb::World,
    entity: pulsar_scenedb::Entity,
    xyz: [i64; 3],
    material_slot: u8,
) -> Result<(), String> {
    edit_object_samples(world, entity, &[xyz], material_slot)
}

fn edit_object_samples(
    world: &mut pulsar_scenedb::World,
    entity: pulsar_scenedb::Entity,
    samples: &[[i64; 3]],
    material_slot: u8,
) -> Result<(), String> {
    let component = world
        .get::<VoxelComponent>(entity)
        .ok_or_else(|| "voxel object component is absent".to_string())?;
    if !component.enabled || !component.editable {
        return Err("voxel object is disabled or not editable".into());
    }
    if component
        .dimensions
        .iter()
        .any(|&size| size == 0 || size > 256)
    {
        return Err("voxel object dimensions must be between 1 and 256 samples".into());
    }
    if samples.iter().any(|sample| {
        sample
            .iter()
            .enumerate()
            .any(|(axis, &coord)| coord < 0 || coord >= i64::from(component.dimensions[axis]))
    }) {
        return Err("voxel sample is outside the object dimensions".into());
    }
    publish_samples(
        component.payload_store(),
        helio_voxel_data::VoxelTerrainId(u128::from(entity.bits())),
        samples,
        material_slot,
        &component.material_ids,
        helio_voxel_data::VoxelDomain::Bounded {
            min: [0; 3],
            max: component
                .dimensions
                .map(|size| i64::from(size.div_ceil(8) - 1)),
            max_lod: 0,
        },
    )
}

fn edit_terrain_sample(
    world: &mut pulsar_scenedb::World,
    entity: pulsar_scenedb::Entity,
    xyz: [i64; 3],
    material_slot: u8,
) -> Result<(), String> {
    edit_terrain_samples(world, entity, &[xyz], material_slot)
}

fn edit_terrain_samples(
    world: &mut pulsar_scenedb::World,
    entity: pulsar_scenedb::Entity,
    samples: &[[i64; 3]],
    material_slot: u8,
) -> Result<(), String> {
    let component = world
        .get::<VoxelTerrainComponent>(entity)
        .ok_or_else(|| "voxel terrain component is absent".to_string())?;
    if !component.enabled || !component.editable {
        return Err("voxel terrain is disabled or not editable".into());
    }
    if super::voxel_world::is_generated(component) {
        // Generated terrain keeps its edits in the journal: sample (x, y, z)
        // is the block at ((x, y, z) + 0.5) voxels from the origin, and the
        // slot is a terrain material id.
        let planet = super::voxel_world::terrain_world(world, entity)?;
        let voxel = planet.grid().voxel_size();
        let edits = samples
            .iter()
            .map(|s| {
                let p = glam::DVec3::new(s[0] as f64 + 0.5, s[1] as f64 + 0.5, s[2] as f64 + 0.5) * voxel;
                super::voxel_world::block_edit(&planet, p, u32::from(material_slot))
            })
            .collect();
        return super::voxel_world::append_edits(world, entity, edits);
    }
    if component.chunk_edge_voxels != 8 {
        return Err("sample edits require the built-in 8-voxel chunk layout".into());
    }
    if component.domain_mode != 0 && component.domain_mode != 1 {
        return Err("voxel terrain domain_mode must be bounded (0) or unbounded (1)".into());
    }
    if !component.voxel_size.is_finite() || component.voxel_size <= 0.0 || component.lod_scale == 0
    {
        return Err("voxel terrain requires a positive finite voxel size and LOD scale".into());
    }
    if component.material_ids.len() > usize::from(u8::MAX) {
        return Err("voxel terrain material palette exceeds 255 IDs".into());
    }
    let transform = world.get::<Transform>(entity).copied().unwrap_or_default();
    let [sx, sy, sz] = transform.scale;
    if transform
        .rotation
        .iter()
        .any(|value| !value.is_finite() || value.abs() > 1.0e-5)
        || !sx.is_finite()
        || sx <= 0.0
        || !sy.is_finite()
        || !sz.is_finite()
        || (sx - sy).abs() > 1.0e-5
        || (sx - sz).abs() > 1.0e-5
        || transform.position.iter().any(|value| !value.is_finite())
    {
        return Err("voxel terrain edits require an unrotated transform with finite positive uniform scale".into());
    }
    let max_lod = u8::try_from(component.max_chunk_lod)
        .map_err(|_| "max_chunk_lod must fit in a chunk key".to_string())?;
    let domain = if component.domain_mode == 1 {
        helio_voxel_data::VoxelDomain::Unbounded { max_lod }
    } else {
        let min = [
            component.bounds_min_x,
            component.bounds_min_y,
            component.bounds_min_z,
        ];
        let max = [
            component.bounds_max_x,
            component.bounds_max_y,
            component.bounds_max_z,
        ];
        if (0..3)
            .any(|axis| !min[axis].is_finite() || !max[axis].is_finite() || min[axis] >= max[axis])
        {
            return Err("bounded terrain requires finite increasing bounds".into());
        }
        let chunk_size = component.voxel_size * f64::from(sx) * 8.0;
        if !chunk_size.is_finite() || chunk_size <= 0.0 {
            return Err("voxel_size must be finite and positive".into());
        }
        let chunk_min = std::array::from_fn(|axis| {
            ((min[axis] - f64::from(transform.position[axis])) / chunk_size).floor() as i64
        });
        let chunk_max = std::array::from_fn(|axis| {
            (((max[axis] - f64::from(transform.position[axis])) / chunk_size).ceil() as i64)
                .saturating_sub(1)
        });
        helio_voxel_data::VoxelDomain::BoundedBase {
            min: chunk_min,
            max: chunk_max,
            max_lod,
            lod_scale: component.lod_scale,
        }
    };
    publish_samples(
        component.payload_store(),
        helio_voxel_data::VoxelTerrainId(u128::from(entity.bits())),
        samples,
        material_slot,
        &component.material_ids,
        domain,
    )
}

fn publish_samples(
    store: VoxelPayloadStore,
    terrain: helio_voxel_data::VoxelTerrainId,
    samples: &[[i64; 3]],
    material_slot: u8,
    material_ids: &[u32],
    domain: helio_voxel_data::VoxelDomain,
) -> Result<(), String> {
    if samples.is_empty() {
        return Ok(());
    }
    if samples.len() > helio_voxel_data::VOXEL_EDIT_MAX_SAMPLES_PER_JOB {
        return Err(format!(
            "voxel sample batch exceeds the {} sample limit",
            helio_voxel_data::VOXEL_EDIT_MAX_SAMPLES_PER_JOB
        ));
    }
    if usize::from(material_slot) > material_ids.len() {
        return Err(format!(
            "material slot {material_slot} is outside the palette"
        ));
    }
    let writer = helio_voxel_data::VoxelSourceWriter::new(
        terrain,
        helio_voxel_data::VoxelSourceId(0),
        store,
    );
    let edits: Vec<_> = samples
        .iter()
        .copied()
        .map(|xyz| helio_voxel_data::VoxelSampleEdit {
            xyz,
            lod: 0,
            material_slot,
        })
        .collect();
    writer
        .publish_sample_edits(&edits, domain, material_ids)
        .map(|_| ())
        .map_err(|error| format!("voxel sample edit failed: {error:?}"))
}

fn validate_paint_slot(material_slot: u8) -> Result<(), String> {
    if material_slot == 0 {
        Err("material slot 0 is reserved for erase operations".into())
    } else {
        Ok(())
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
        assert!(state
            .1
            .values()
            .all(|bytes| bytes.len() == 512 && bytes.iter().all(|&slot| slot == 1)));
        drop(state);
        let serialized = serde_json::to_value(&component).unwrap();
        assert!(serialized.get("payloads").is_none());
        assert!(!component
            .get_properties()
            .iter()
            .any(|property| property.name == "payloads"));
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
        assert!(!component
            .get_properties()
            .iter()
            .any(|property| property.name == "payloads"));
    }

    #[test]
    fn older_terrain_config_defaults_generator_version_and_chunk_layout() {
        let mut value = serde_json::to_value(VoxelTerrainComponent::default()).unwrap();
        value.as_object_mut().unwrap().remove("generator_version");
        value.as_object_mut().unwrap().remove("chunk_edge_voxels");
        value.as_object_mut().unwrap().remove("max_chunk_lod");
        value.as_object_mut().unwrap().remove("lod_scale");
        let restored: VoxelTerrainComponent = serde_json::from_value(value).unwrap();
        assert_eq!(restored.generator.version, 0);
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
