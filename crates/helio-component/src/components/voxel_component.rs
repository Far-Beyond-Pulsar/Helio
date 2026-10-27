//! Reflected, SceneDB-owned authoring components for voxel terrain.
//!
//! These structs contain reflected configuration and runtime-only live data.
//! Voxel encodings, edit batches, and generation behavior belong to the
//! renderer-independent data API. No persistence behavior is implied by the
//! runtime data fields.

use engine_class_derive::{component_events, engine_class};
use super::VoxelFeatureSpawn;
use helio_voxel_data::{VOXEL_TERRAIN_GENERATOR, VOXEL_TERRAIN_GENERATOR_VERSION, VoxelEditJournal};

/// Typed payload for block lifecycle events. Coordinates are the exact
/// world-space centre of the base cell; `material` is the material present
/// when the event occurred (before a break, or after a placement).
#[engine_class(no_register)]
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BlockData {
    #[property]
    pub x: f64,
    #[property]
    pub y: f64,
    #[property]
    pub z: f64,
    #[property]
    pub material: u32,
}

/// Typed payload for a terrain cell whose material changed from one solid
/// material to another.
#[engine_class(no_register)]
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BlockMaterialChange {
    #[property]
    pub x: f64,
    #[property]
    pub y: f64,
    #[property]
    pub z: f64,
    #[property]
    pub previous_material: u32,
    #[property]
    pub material: u32,
}

#[component_events(class = "VoxelTerrainComponent")]
pub trait TerrainEvents {
    #[bp_event]
    fn block_broken() -> BlockData {}
    #[bp_event]
    fn block_placed() -> BlockData {}
    #[bp_event]
    fn block_material_changed() -> BlockMaterialChange {}
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
    #[serde(
        rename = "generator_version",
        default = "default_voxel_generator_version"
    )]
    pub version: u32,
}

impl VoxelGeneratorRef {
    pub fn new(id: impl Into<String>, version: u32) -> Self {
        Self {
            id: id.into(),
            version,
        }
    }
}

impl Default for VoxelGeneratorRef {
    /// The layered terrain generator.
    fn default() -> Self {
        Self::new(VOXEL_TERRAIN_GENERATOR, VOXEL_TERRAIN_GENERATOR_VERSION)
    }
}

fn serialize_generator_ref_json(
    value: &VoxelGeneratorRef,
) -> pulsar_reflection::ReflectResult<serde_json::Value> {
    serde_json::to_value(value)
        .map_err(|e| pulsar_reflection::ReflectError::SerializationFailed(e.to_string()))
}

fn deserialize_generator_ref_json(
    value: serde_json::Value,
) -> pulsar_reflection::ReflectResult<VoxelGeneratorRef> {
    serde_json::from_value(value)
        .map_err(|e| pulsar_reflection::ReflectError::DeserializationFailed(e.to_string()))
}

/// Registered for reflection; the picker editor is registered by the host
/// that knows the generator registry.
#[pulsar_reflection::pulsar_type(
    serialize_json_with = serialize_generator_ref_json,
    deserialize_json_with = deserialize_generator_ref_json
)]
#[allow(dead_code)]
type RegisteredVoxelGeneratorRef = VoxelGeneratorRef;

/// Overall form of a voxel world.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    pulsar_reflection::Reflectable,
)]
pub enum VoxelWorldShape {
    /// A planet centred on the entity origin (`planet_radius`).
    Sphere,
    /// A square plane of `plane_size` metres centred on the entity origin.
    #[default]
    Plane,
    /// A plane without edges within reach (about 13 400 km across).
    InfinitePlane,
}

/// The air of a voxel world (`VoxelTerrainComponent::air`): the sky the
/// level shows, its light on the ground and the haze in the distance.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    pulsar_reflection::Reflectable,
)]
pub enum VoxelAir {
    /// The entity's `AtmosphereComponent` as authored, every property.
    #[default]
    Custom,
    /// No air (a moon, an asteroid): a black sky and only the sun's light.
    None,
    /// Earth's air: a blue sky.
    Earthlike,
    /// Thin dusty air (Mars): a pale butterscotch sky, blue sunsets.
    Thin,
    /// Thick hazy air (Titan): a deep, murky sky.
    Dense,
}

impl VoxelAir {
    /// Names of the presets, as scripts give them.
    pub const NAMES: [&'static str; 5] = ["custom", "none", "earthlike", "thin", "dense"];

    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "custom" => Self::Custom,
            "none" => Self::None,
            "earthlike" | "earth" => Self::Earthlike,
            "thin" | "mars" => Self::Thin,
            "dense" | "titan" => Self::Dense,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        Self::NAMES[self as usize]
    }
}

fn default_planet_radius() -> f64 {
    6_371_000.0
}

fn default_air_scale() -> f32 {
    1.0
}

fn default_air_tint() -> [f32; 3] {
    [1.0; 3]
}

fn default_plane_size() -> f64 {
    4_096.0
}

/// A voxel world: a planet, a plane or an endless plane, whatever generates
/// it. Every property is the world's whole description, and every one is a
/// blueprint method too (`world_shape`/`set_world_shape`, `air`/`set_air`,
/// ...), so a game can build random worlds.
///
/// - **World**: shape and size, voxel size.
/// - **Generation**: the registered generator, its seed; its own settings
///   live in its settings component on the same entity (for the built-in
///   generator, [`VoxelTerrainLayersComponent`]), or in
///   `generator_parameters` for generators without one.
/// - **Features**: what grows and lies on the ground (trees, bushes,
///   shrubs, cacti, boulders, craters) and where, grown with the seed
///   ([`VoxelFeatureSpawn`]).
/// - **Environment**: the world's air. A preset derives the entity's
///   `AtmosphereComponent` (the level's sky; its clouds stay as authored)
///   and its `GlobalFogComponent` (a ground haze); `Custom` keeps both as
///   authored.
/// - **Appearance**: the renderer's look of its materials.
/// - **Editing**: whether it can be dug and built, and its edits: an
///   ordered brush journal saved with the level.
#[engine_class(category = "Voxel/Terrain", debug, serialize, deserialize)]
#[category("World", category_color = "#6FA86F")]
#[category("Generation", category_color = "#D1A73F")]
#[category("Features", category_color = "#7FAF5F")]
#[category("Environment", category_color = "#6F9FD1")]
#[category("Appearance", category_color = "#B07FD1")]
#[category("Editing", category_color = "#D18F6F")]
pub struct VoxelTerrainComponent {
    /// Runtime-only events committed by world methods and drained after the
    /// method releases its World borrow. Clones start with an empty outbox.
    #[serde(skip)]
    pub(crate) pending_block_broken: Vec<BlockData>,
    #[serde(skip)]
    pub(crate) pending_block_placed: Vec<BlockData>,
    #[serde(skip)]
    pub(crate) pending_block_material_changed: Vec<BlockMaterialChange>,
    /// Whether this terrain source participates in rendering and queries.
    #[property]
    pub enabled: bool,
    /// Planet, finite plane or infinite plane.
    #[serde(default)]
    #[property(category = "World")]
    pub shape: VoxelWorldShape,
    /// Planet radius in metres (sphere worlds).
    #[serde(default = "default_planet_radius")]
    #[property(
        min = 1000.0,
        max = 50000000.0,
        step = 1000.0,
        category = "World",
        label = "Planet radius (m)"
    )]
    pub planet_radius: f64,
    /// Edge length of a finite plane in metres.
    #[serde(default = "default_plane_size")]
    #[property(
        min = 16.0,
        max = 13000000.0,
        step = 16.0,
        category = "World",
        label = "Plane size (m)"
    )]
    pub plane_size: f64,
    /// Edge length of a base-resolution voxel in metres (0.1 to 1 for
    /// streamed terrain).
    #[property(
        min = 0.1,
        max = 1.0,
        step = 0.05,
        category = "World",
        label = "Voxel size (m)"
    )]
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
    /// What grows on the ground and lies on it, spawn by spawn from the
    /// widest spacing to the narrowest: the engine's flora grown with the
    /// seed in cells of the voxel size. Part of the ground: edits made
    /// before a change to them are dropped.
    #[serde(default)]
    #[property(category = "Features")]
    pub features: Vec<VoxelFeatureSpawn>,

    /// The world's air: a preset, or `Custom` for the entity's
    /// `AtmosphereComponent` as authored.
    #[serde(default)]
    #[property(category = "Environment")]
    pub air: VoxelAir,
    /// How much air (1: the preset's; 0: none). It scales every scattering
    /// coefficient of the preset.
    #[serde(default = "default_air_scale")]
    #[property(
        min = 0.0,
        max = 10.0,
        step = 0.05,
        category = "Environment",
        label = "Air density"
    )]
    pub air_density: f32,
    /// The sky's colour: the preset's molecular scattering per channel.
    #[serde(default = "default_air_tint")]
    #[property(category = "Environment", label = "Sky tint")]
    pub air_tint: [f32; 3],
    /// Haze and dust: the preset's aerosol scattering and its ground haze,
    /// times this.
    #[serde(default = "default_air_scale")]
    #[property(
        min = 0.0,
        max = 20.0,
        step = 0.05,
        category = "Environment",
        label = "Haze"
    )]
    pub air_haze: f32,

    /// Renderer appearance JSON, independent of generator data and edits.
    #[serde(default)]
    #[property(category = "Appearance", label = "Terrain appearance (JSON)")]
    pub appearance_parameters: String,

    /// Whether it can be dug and built (the sculpt tool, block methods).
    #[property(category = "Editing")]
    pub editable: bool,
    /// Generator/configuration revision. It is serialized with authored
    /// config but omitted from the property editor.
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
            pending_block_broken: Vec::new(),
            pending_block_placed: Vec::new(),
            pending_block_material_changed: Vec::new(),
            enabled: true,
            shape: VoxelWorldShape::default(),
            planet_radius: default_planet_radius(),
            plane_size: default_plane_size(),
            voxel_size: 0.1,
            generator: VoxelGeneratorRef::default(),
            seed: 0,
            generator_parameters: String::new(),
            features: Vec::new(),
            air: VoxelAir::default(),
            air_density: default_air_scale(),
            air_tint: default_air_tint(),
            air_haze: default_air_scale(),
            appearance_parameters: String::new(),
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
        Self {
            shape: VoxelWorldShape::Sphere,
            planet_radius: radius,
            ..Self::default()
        }
    }
    /// A square plane of `size` metres with Helio's terrain generator.
    pub fn plane(size: f64) -> Self {
        Self {
            shape: VoxelWorldShape::Plane,
            plane_size: size,
            ..Self::default()
        }
    }
    /// A plane without edges within reach, with Helio's terrain generator.
    pub fn infinite_plane() -> Self {
        Self {
            shape: VoxelWorldShape::InfinitePlane,
            ..Self::default()
        }
    }
}

// Clone starts with an empty event outbox.
impl Clone for VoxelTerrainComponent {
    fn clone(&self) -> Self {
        Self {
            pending_block_broken: Vec::new(),
            pending_block_placed: Vec::new(),
            pending_block_material_changed: Vec::new(),
            enabled: self.enabled,
            shape: self.shape,
            planet_radius: self.planet_radius,
            plane_size: self.plane_size,
            voxel_size: self.voxel_size,
            generator: self.generator.clone(),
            seed: self.seed,
            generator_parameters: self.generator_parameters.clone(),
            features: self.features.clone(),
            air: self.air,
            air_density: self.air_density,
            air_tint: self.air_tint,
            air_haze: self.air_haze,
            appearance_parameters: self.appearance_parameters.clone(),
            editable: self.editable,
            source_revision: self.source_revision,
            edits: self.edits.clone(),
        }
    }
}

/// A solid terrain material.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    pulsar_reflection::Reflectable,
)]
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
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    pulsar_reflection::Reflectable,
)]
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
    /// Escarpments: inside regions of Scale covering Coverage, the layers
    /// before it step into terraces Height tall, joined by cliffs over the
    /// last Ratio of each step.
    Cliffs,
    /// Rivers and lakes from the planet's drainage: valleys cut into the
    /// layers before it, from sources draining Region size squared of wet
    /// land to the sea. At a source the valley sinks Height, its
    /// floodplain is Scale wide with a channel Ratio 2 of it, Base deep;
    /// walls slope Ratio, shoulders rounded over Ratio 3 metres. Coverage
    /// is how much the drainage has worn the planet (ranges lowered,
    /// basins filled with sediment); Octaves refines it into streams that
    /// many levels (at most three: creeks).
    Rivers,
}

/// Where a terrain layer applies.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    pulsar_reflection::Reflectable,
)]
pub enum VoxelLayerMask {
    #[default]
    Everywhere,
    /// Land only (needs a Continents layer).
    Land,
    /// Everywhere but fading out under deep sea (needs a Continents layer).
    AboveDeepSea,
}

/// How generated cells get their materials.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    pulsar_reflection::Reflectable,
)]
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

/// Where a material rule holds on river beds
/// (`helio_pass_voxel_planet::layers::RiverBed`).
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    pulsar_reflection::Reflectable,
)]
pub enum VoxelRiverBed {
    #[default]
    Any,
    /// Only in rivers' channels.
    Only,
    /// Never in them.
    Never,
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
    /// Moisture, 0 (dry) to 1 (wet): the climate's.
    #[property(min = 0.0, max = 1.0, step = 0.01)]
    pub min_moisture: f64,
    #[property(min = 0.0, max = 1.0, step = 0.01)]
    pub max_moisture: f64,
    /// Annual mean temperature at the cell (the climate's, cooler higher).
    #[property(min = -1000.0, max = 1000.0, step = 0.5, label = "Min temperature (C)")]
    pub min_temperature_c: f64,
    #[property(min = -1000.0, max = 1000.0, step = 0.5, label = "Max temperature (C)")]
    pub max_temperature_c: f64,
    /// Where river beds count.
    #[property(label = "River beds")]
    pub river_bed: VoxelRiverBed,
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
            min_temperature_c: -1.0e3,
            max_temperature_c: 1.0e3,
            river_bed: VoxelRiverBed::Any,
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
        Self {
            material,
            ..Self::default()
        }
    }
}

fn serialize_material_rule_json(
    value: &VoxelMaterialRule,
) -> pulsar_reflection::ReflectResult<serde_json::Value> {
    serde_json::to_value(value)
        .map_err(|e| pulsar_reflection::ReflectError::SerializationFailed(e.to_string()))
}

fn deserialize_material_rule_json(
    value: serde_json::Value,
) -> pulsar_reflection::ReflectResult<VoxelMaterialRule> {
    serde_json::from_value(value)
        .map_err(|e| pulsar_reflection::ReflectError::DeserializationFailed(e.to_string()))
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
    /// A layer of `kind` with that kind's defaults (the generator's own
    /// table, `layers::Layer::new`): what the inspector shows when a layer
    /// is added or changes kind.
    pub fn new(kind: VoxelLayerKind) -> Self {
        use helio_pass_voxel_planet::layers::{Layer, LayerKind};
        serde_json::to_value(kind)
            .and_then(serde_json::from_value::<LayerKind>)
            .and_then(|kind| serde_json::to_value(Layer::new(kind)))
            .and_then(serde_json::from_value)
            .unwrap_or(Self {
                kind,
                ..Self::default()
            })
    }
}

fn serialize_terrain_layer_json(
    value: &VoxelTerrainLayer,
) -> pulsar_reflection::ReflectResult<serde_json::Value> {
    serde_json::to_value(value)
        .map_err(|e| pulsar_reflection::ReflectError::SerializationFailed(e.to_string()))
}

fn deserialize_terrain_layer_json(
    value: serde_json::Value,
) -> pulsar_reflection::ReflectResult<VoxelTerrainLayer> {
    serde_json::from_value(value)
        .map_err(|e| pulsar_reflection::ReflectError::DeserializationFailed(e.to_string()))
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
    #[property(
        min = 0.0,
        max = 170.0,
        step = 1.0,
        category = "Caves",
        label = "Depth (m)"
    )]
    pub depth_m: f64,
    /// Rough share of the land inside cave regions.
    #[property(
        min = 0.0,
        max = 1.0,
        step = 0.01,
        category = "Caves",
        label = "Regions (share)"
    )]
    pub share: f64,
    #[property(
        min = 0.1,
        max = 500.0,
        step = 0.1,
        category = "Caves",
        label = "Region size (km)"
    )]
    pub region_km: f64,
    #[property(
        min = 0.0,
        max = 20.0,
        step = 0.1,
        category = "Caves",
        label = "Tunnel radius (m)"
    )]
    pub tunnel_radius_m: f64,
    #[property(
        min = 4.0,
        max = 2000.0,
        step = 1.0,
        category = "Caves",
        label = "Tunnel winding (m)"
    )]
    pub tunnel_wavelength_m: f64,
    #[property(
        min = 8.0,
        max = 4000.0,
        step = 1.0,
        category = "Caves",
        label = "Cavern size (m)"
    )]
    pub cavern_wavelength_m: f64,
    /// Rough share of the cave volume opened as caverns.
    #[property(
        min = 0.0,
        max = 1.0,
        step = 0.01,
        category = "Caves",
        label = "Caverns (share)"
    )]
    pub cavern_share: f64,
    /// Rock kept over every cave: tunnels and caverns close towards it
    /// instead of breaking the surface, except at entrances.
    #[property(
        min = 0.0,
        max = 100.0,
        step = 0.5,
        category = "Caves",
        label = "Cover (m)"
    )]
    pub cover_m: f64,
    /// Rough share of a cave region where tunnels open to the surface.
    #[property(
        min = 0.0,
        max = 1.0,
        step = 0.01,
        category = "Caves",
        label = "Entrances (share)"
    )]
    pub entrance_share: f64,
    #[property(
        min = 4.0,
        max = 2000.0,
        step = 1.0,
        category = "Caves",
        label = "Entrance spacing (m)"
    )]
    pub entrance_spacing_m: f64,
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
            cover_m: 6.0,
            entrance_share: 0.05,
            entrance_spacing_m: 80.0,
        }
    }
}

/// A world's climate (`helio_pass_voxel_planet::layers::Climate`):
/// temperature from latitude and height, moisture from the rain belts, the
/// way to the sea and rain shadows; baked over the whole planet with Maps,
/// else Latitude's everywhere.
#[engine_class(no_register, clone, debug, serialize, deserialize)]
#[category("Climate", category_color = "#5E9EA8")]
#[derive(pulsar_reflection::Reflectable)]
#[serde(default)]
pub struct VoxelClimate {
    /// Bake planet maps: climate over the whole planet, and the drainage a
    /// Rivers layer carves.
    #[property(category = "Climate", label = "Maps")]
    pub maps: bool,
    /// The planet's axis (world direction): latitudes run around it.
    #[property(category = "Climate")]
    pub pole: [f32; 3],
    /// Latitude of a flat world's origin (north is -Z), and of an unmapped
    /// world's climate.
    #[property(min = -90.0, max = 90.0, step = 1.0, category = "Climate", label = "Latitude (deg)")]
    pub latitude_deg: f64,
    /// Annual mean temperatures at sea level.
    #[property(min = -100.0, max = 100.0, step = 0.5, category = "Climate", label = "Equator (C)")]
    pub equator_c: f64,
    #[property(min = -100.0, max = 100.0, step = 0.5, category = "Climate", label = "Poles (C)")]
    pub pole_c: f64,
    /// Cooling with height.
    #[property(min = 0.0, max = 20.0, step = 0.1, category = "Climate", label = "Lapse rate (C/km)")]
    pub lapse_c_per_km: f64,
    /// Ground colder than this keeps its snow.
    #[property(min = -50.0, max = 50.0, step = 0.5, category = "Climate", label = "Snow (C)")]
    pub snow_c: f64,
    /// Scale of the rain (1: Earth's); an unmapped world's moisture is its
    /// latitude's rain belt times it.
    #[property(min = 0.0, max = 4.0, step = 0.05, category = "Climate")]
    pub wetness: f64,
    /// How far inland, upwind, the rain halves.
    #[property(min = 1.0, max = 20000.0, step = 10.0, category = "Climate", label = "Continentality (km)")]
    pub continentality_km: f64,
    /// Height of ground upwind that halves the rain behind it.
    #[property(min = 1.0, max = 20000.0, step = 10.0, category = "Climate", label = "Rain shadow (m)")]
    pub rain_shadow_m: f64,
    /// Ground under the datum is sea.
    #[property(category = "Climate")]
    pub ocean: bool,
    /// Size of the maps' texels at face centres.
    #[property(min = 0.1, max = 1000.0, step = 1.0, category = "Climate", label = "Map texel (km)")]
    pub texel_km: f64,
}

impl VoxelClimate {
    /// The generator's climate as the component holds it.
    fn from_generator(climate: helio_pass_voxel_planet::layers::Climate) -> Self {
        serde_json::to_value(climate)
            .and_then(serde_json::from_value)
            .expect("climate settings mirror the generator's")
    }
    /// Earth's: warm tropics, frozen poles, oceans, rain belts.
    pub fn earth() -> Self {
        Self::from_generator(helio_pass_voxel_planet::layers::Climate::earth())
    }
    /// `latitude_deg`'s climate everywhere, unbaked.
    pub fn uniform(latitude_deg: f64, wetness: f64) -> Self {
        Self::from_generator(helio_pass_voxel_planet::layers::Climate::uniform(
            latitude_deg,
            wetness,
        ))
    }
}

impl Default for VoxelClimate {
    fn default() -> Self {
        Self::earth()
    }
}

/// Generated overhangs: steep ground leaning over, a continuous
/// deformation of the heightfield (no floating rock).
#[engine_class(no_register, clone, debug, serialize, deserialize)]
#[category("Overhangs", category_color = "#A8826F")]
#[derive(pulsar_reflection::Reflectable)]
#[serde(default)]
pub struct VoxelOverhangs {
    #[property(category = "Overhangs")]
    pub enabled: bool,
    /// Largest horizontal lean of the ground.
    #[property(
        min = 0.0,
        max = 20.0,
        step = 0.5,
        category = "Overhangs",
        label = "Height (m)"
    )]
    pub height_m: f64,
    /// Vertical wavelength of the lean (ledge spacing).
    #[property(
        min = 2.0,
        max = 500.0,
        step = 1.0,
        category = "Overhangs",
        label = "Ledge spacing (m)"
    )]
    pub wavelength_m: f64,
    #[property(
        min = 0.1,
        max = 500.0,
        step = 0.1,
        category = "Overhangs",
        label = "Region size (km)"
    )]
    pub region_km: f64,
    #[property(
        min = 0.0,
        max = 1.0,
        step = 0.01,
        category = "Overhangs",
        label = "Regions (share)"
    )]
    pub share: f64,
}

impl Default for VoxelOverhangs {
    fn default() -> Self {
        Self {
            enabled: true,
            height_m: 6.0,
            wavelength_m: 6.0,
            region_km: 3.0,
            share: 0.3,
        }
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
    /// Temperature and moisture: the Earthlike snow and dry lands, rules'
    /// and features' climate ranges, and the maps rivers come from.
    #[property]
    pub climate: VoxelClimate,
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

fn serialize_terrain_stack_json(
    value: &VoxelTerrainStack,
) -> pulsar_reflection::ReflectResult<serde_json::Value> {
    serde_json::to_value(value)
        .map_err(|e| pulsar_reflection::ReflectError::SerializationFailed(e.to_string()))
}

fn deserialize_terrain_stack_json(
    value: serde_json::Value,
) -> pulsar_reflection::ReflectResult<VoxelTerrainStack> {
    serde_json::from_value(value)
        .map_err(|e| pulsar_reflection::ReflectError::DeserializationFailed(e.to_string()))
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
        Self {
            stack: VoxelTerrainStack::earth(),
        }
    }
    pub fn moon() -> Self {
        Self {
            stack: VoxelTerrainStack::moon(),
        }
    }
    pub fn desert() -> Self {
        Self {
            stack: VoxelTerrainStack::desert(),
        }
    }
    pub fn flat(height_m: f64) -> Self {
        Self {
            stack: VoxelTerrainStack::flat(height_m),
        }
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
    /// hills, river valleys and lakes, and roughness; caves, overhangs;
    /// Earth's climate over meadows, dry lands, rock and snow.
    pub fn earth() -> Self {
        use VoxelLayerKind::*;
        Self {
            layers: vec![
                VoxelTerrainLayer::new(Warp),
                VoxelTerrainLayer::new(Continents),
                VoxelTerrainLayer {
                    mask: VoxelLayerMask::Land,
                    ..VoxelTerrainLayer::new(Mountains)
                },
                VoxelTerrainLayer::new(Erosion),
                Self::earth_hills(),
                VoxelTerrainLayer::new(Rivers),
                VoxelTerrainLayer {
                    mask: VoxelLayerMask::AboveDeepSea,
                    ..VoxelTerrainLayer::new(Roughness)
                },
            ],
            caves: VoxelCaves::default(),
            overhangs: VoxelOverhangs::default(),
            materials: VoxelMaterialStyle::Earthlike,
            climate: VoxelClimate::earth(),
            soil_depth_m: 0.7,
            surface: VoxelTerrainMaterial::Grass,
            soil: VoxelTerrainMaterial::Dirt,
            rock: VoxelTerrainMaterial::Stone,
            rules: Vec::new(),
        }
    }

    /// Earth's hills: 140 m over 9 km down to knolls a kilometre across.
    fn earth_hills() -> VoxelTerrainLayer {
        VoxelTerrainLayer {
            mask: VoxelLayerMask::AboveDeepSea,
            octaves: 5,
            persistence: 0.6,
            ..VoxelTerrainLayer::new(VoxelLayerKind::Hills)
        }
    }

    /// Dunes and mesas with materials from rules: sand on gentle ground,
    /// sandstone and clay strata, gravel in gullies, stone patches and dark
    /// stone specks.
    pub fn desert() -> Self {
        use VoxelTerrainMaterial as M;
        let mut earth = Self::earth();
        earth.layers.retain(|l| {
            l.kind != VoxelLayerKind::Continents && l.kind != VoxelLayerKind::Rivers
        });
        // The desert's hills stay gentle under its dunes.
        for l in earth
            .layers
            .iter_mut()
            .filter(|l| l.kind == VoxelLayerKind::Hills)
        {
            *l = VoxelTerrainLayer {
                mask: l.mask,
                ..VoxelTerrainLayer::new(VoxelLayerKind::Hills)
            };
        }
        for l in &mut earth.layers {
            l.mask = VoxelLayerMask::Everywhere;
        }
        earth
            .layers
            .insert(1, VoxelTerrainLayer::new(VoxelLayerKind::Plateau));
        Self {
            materials: VoxelMaterialStyle::Rules,
            rules: vec![
                VoxelMaterialRule {
                    max_depth_m: 0.3,
                    max_erosion: -0.4,
                    min_slope: 0.2,
                    ..VoxelMaterialRule::new(M::Gravel)
                },
                VoxelMaterialRule {
                    max_depth_m: 1.5,
                    max_slope: 0.6,
                    ..VoxelMaterialRule::new(M::Sand)
                },
                VoxelMaterialRule {
                    max_depth_m: 0.0,
                    speck_share: 0.06,
                    ..VoxelMaterialRule::new(M::DarkStone)
                },
                VoxelMaterialRule {
                    patch_km: 0.05,
                    patch_share: 0.15,
                    ..VoxelMaterialRule::new(M::Stone)
                },
                VoxelMaterialRule {
                    band_m: 2.5,
                    odd_bands: true,
                    ..VoxelMaterialRule::new(M::Clay)
                },
            ],
            rock: M::Sandstone,
            climate: VoxelClimate::uniform(25.0, 0.2),
            ..earth
        }
    }

    /// Cratered highlands and dark basalt plains over regolith, no caves.
    pub fn moon() -> Self {
        use VoxelLayerKind::*;
        Self {
            layers: vec![
                VoxelTerrainLayer {
                    height_m: 1_500.0,
                    scale_km: 250.0,
                    octaves: 3,
                    ..VoxelTerrainLayer::new(Hills)
                },
                VoxelTerrainLayer {
                    height_m: 1_200.0,
                    scale_km: 900.0,
                    coverage: 0.3,
                    ..VoxelTerrainLayer::new(Basins)
                },
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
            caves: VoxelCaves {
                enabled: false,
                ..VoxelCaves::default()
            },
            overhangs: VoxelOverhangs {
                enabled: false,
                ..VoxelOverhangs::default()
            },
            materials: VoxelMaterialStyle::Lunar,
            climate: VoxelClimate::uniform(0.0, 0.0),
            soil_depth_m: 4.0,
            ..Self::earth()
        }
    }

    /// Level ground at `height_m`: grass over a metre of dirt over stone.
    pub fn flat(height_m: f64) -> Self {
        Self {
            layers: vec![VoxelTerrainLayer {
                height_m,
                ..VoxelTerrainLayer::new(VoxelLayerKind::Plateau)
            }],
            caves: VoxelCaves {
                enabled: false,
                ..VoxelCaves::default()
            },
            overhangs: VoxelOverhangs {
                enabled: false,
                ..VoxelOverhangs::default()
            },
            materials: VoxelMaterialStyle::Layered,
            climate: VoxelClimate::uniform(45.0, 1.0),
            soil_depth_m: 1.0,
            ..Self::earth()
        }
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
    // Sample (x, y, z) is the block at ((x, y, z) + 0.5) voxels from the
    // origin, and the slot is a terrain material id: an edit in the journal.
    let planet = super::voxel_world::terrain_world(world, entity)?;
    let voxel = planet.grid().voxel_size();
    let edits = samples
        .iter()
        .map(|s| {
            let p =
                glam::DVec3::new(s[0] as f64 + 0.5, s[1] as f64 + 0.5, s[2] as f64 + 0.5) * voxel;
            super::voxel_world::block_edit(&planet, p, u32::from(material_slot))
        })
        .collect();
    super::voxel_world::append_edits(world, entity, edits)
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

    #[test]
    fn older_terrains_load_and_drop_their_chunk_fields() {
        // A level saved before the world description: chunk-domain fields,
        // a renderer and a palette.
        let mut value = serde_json::to_value(VoxelTerrainComponent::planet(1_737_400.0)).unwrap();
        let object = value.as_object_mut().unwrap();
        for field in ["air", "air_density", "air_tint", "air_haze"] {
            object.remove(field);
        }
        object.insert("domain_mode".into(), 1.into());
        object.insert("chunk_edge_voxels".into(), 8.into());
        object.insert("max_chunk_lod".into(), 16.into());
        object.insert("lod_scale".into(), 2.into());
        object.insert("renderer_id".into(), "helio.voxel-terrain".into());
        object.insert("material_ids".into(), serde_json::json!([0, 1]));
        object.insert("bounds_max_x".into(), 64.0.into());
        let restored: VoxelTerrainComponent = serde_json::from_value(value).unwrap();
        assert_eq!(restored.planet_radius, 1_737_400.0);
        assert_eq!(restored.air, VoxelAir::Custom, "the authored sky stays");
        assert_eq!(
            (restored.air_density, restored.air_tint, restored.air_haze),
            (1.0, [1.0; 3], 1.0)
        );
        let saved = serde_json::to_value(&restored).unwrap();
        for field in [
            "domain_mode",
            "chunk_edge_voxels",
            "max_chunk_lod",
            "lod_scale",
            "renderer_id",
            "material_ids",
            "bounds_max_x",
            "payloads",
        ] {
            assert!(saved.get(field).is_none(), "{field} is not saved");
        }
        let properties = restored.get_properties();
        for field in [
            "air",
            "air_density",
            "air_tint",
            "air_haze",
            "shape",
            "planet_radius",
            "voxel_size",
            "seed",
            "editable",
        ] {
            assert!(
                properties.iter().any(|property| property.name == field),
                "{field} is a property"
            );
        }
    }

    #[test]
    fn air_names_round_trip() {
        for name in VoxelAir::NAMES {
            assert_eq!(VoxelAir::from_name(name).unwrap().name(), name);
        }
        assert_eq!(VoxelAir::from_name("Mars"), Some(VoxelAir::Thin));
        assert_eq!(VoxelAir::from_name("vacuum"), None);
    }
}
