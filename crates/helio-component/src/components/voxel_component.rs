//! Reflected, SceneDB-owned authoring components for voxel objects and terrain.
//!
//! These structs contain durable configuration only. Voxel encodings, edit
//! batches, generation behavior, and GPU/cache representations belong to the
//! dedicated voxel pass and its SceneDB-facing data API.

use engine_class_derive::engine_class;

/// Authoring component for a small deformable voxel object.
///
/// A newly-created object describes a cubic 16³ voxel volume. Its editable
/// voxel contents are SceneDB-owned data managed through the voxel data API;
/// this component stores configuration and the SceneDB material-ID palette.
#[engine_class(category = "Voxel", clone, debug, serialize, deserialize)]
#[category("Volume", category_color = "#8F8F8F")]
#[category("Materials", category_color = "#D1A73F")]
#[category("Editing", category_color = "#D18F6F")]
#[category("Surface", category_color = "#2FA88A")]
pub struct VoxelComponent {
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

/// General-purpose voxel terrain authoring configuration.
///
/// `domain_mode` and `shape_mode` are stable primitive discriminants so this
/// component crate does not define voxel-specific helper enums. Their values
/// are interpreted and validated by the voxel pass. The intended initial
/// values are domain 0 = bounded, 1 = unbounded and shape 0 = plane, 1 = sphere.
#[engine_class(category = "Voxel/Terrain", clone, debug, serialize, deserialize)]
#[category("Domain", category_color = "#8F8F8F")]
#[category("Generation", category_color = "#D1A73F")]
#[category("Materials", category_color = "#D1A73F")]
#[category("Surface", category_color = "#2FA88A")]
#[category("LOD", category_color = "#7C6FD1")]
#[category("Streaming", category_color = "#3AA0FF")]
#[category("Editing", category_color = "#D18F6F")]
pub struct VoxelTerrainComponent {
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
    /// Whether external callers may submit persistent edit/data batches.
    #[property(category = "Editing")]
    pub editable: bool,
    /// Service-managed source/configuration revision. It is serialized with the
    /// component but deliberately omitted from the property editor; only the
    /// voxel batch API may advance it.
    pub source_revision: u64,
}

impl Default for VoxelTerrainComponent {
    fn default() -> Self {
        Self {
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
