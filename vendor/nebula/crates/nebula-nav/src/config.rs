use serde::{Deserialize, Serialize};
use nebula_core::traits::BakeInput;

/// Configuration for navigation mesh baking.
///
/// The algorithm follows the Recast voxelisation + region-growing pipeline
/// (the same approach used by Recast Navigation and Unreal Engine's NavMesh
/// system) implemented here in pure Rust with rayon parallelism.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavConfig {
    /// Agent cylinder radius (world units).  Geometry within this distance
    /// from a wall is considered un-walkable.
    pub agent_radius: f32,

    /// Agent standing height (world units).  Areas with clearance less than
    /// this value are marked un-walkable.
    pub agent_height: f32,

    /// Maximum walkable step height (world units).  Vertical transitions
    /// smaller than this are considered passable.
    pub max_step_height: f32,

    /// Maximum slope angle (degrees) considered walkable.
    pub max_slope_deg: f32,

    /// Voxel cell size in the X/Z plane (world units).  Smaller values give
    /// higher-fidelity meshes at the cost of memory and bake time.
    pub cell_size: f32,

    /// Voxel cell height (Y axis, world units).
    pub cell_height: f32,

    /// Minimum connected-region area in voxels (regions smaller than this
    /// are merged with neighbours or pruned).
    pub min_region_area: u32,

    /// Merge threshold for region simplification (voxels).
    pub merge_region_area: u32,

    /// Maximum edge length in the simplified contour polygon (world units).
    /// Shorter values produce more accurate edge shapes.
    pub max_edge_length: f32,

    /// Maximum distance the simplified contour may deviate from the raw
    /// voxel boundary (world units).
    pub max_edge_error: f32,

    /// Number of layers in the detail mesh (height-field sample rate).
    /// 0 disables the detail mesh step.
    pub detail_sample_dist: f32,

    /// Maximum detail mesh surface error (world units).
    pub detail_sample_max_error: f32,

    /// Optional world-space AABB to bake.  `None` uses the full scene AABB.
    pub bake_aabb: Option<([f32;3], [f32;3])>,
}

impl Default for NavConfig {
    fn default() -> Self {
        Self {
            agent_radius:          0.4,
            agent_height:          1.8,
            max_step_height:       0.4,
            max_slope_deg:         45.0,
            cell_size:             0.3,
            cell_height:           0.2,
            min_region_area:       8,
            merge_region_area:     20,
            max_edge_length:       12.0,
            max_edge_error:        1.3,
            detail_sample_dist:    6.0,
            detail_sample_max_error: 1.0,
            bake_aabb:             None,
        }
    }
}

impl NavConfig {
    /// Coarse fast-preview preset.
    pub fn fast() -> Self {
        Self { cell_size: 1.0, cell_height: 0.5, min_region_area: 4, max_edge_length: 24.0, ..Default::default() }
    }

    /// High-precision production preset.
    pub fn ultra() -> Self {
        Self { cell_size: 0.15, cell_height: 0.1, min_region_area: 16, max_edge_length: 6.0, max_edge_error: 0.5, ..Default::default() }
    }
}

impl BakeInput for NavConfig {}
