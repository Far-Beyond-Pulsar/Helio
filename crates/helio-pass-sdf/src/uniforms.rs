//! GPU-mapped uniform structs for the SDF clipmap system.
//!
//! All structs are `bytemuck::Pod` + `Zeroable` and match their WGSL counterparts.

use bytemuck::{Pod, Zeroable};

/// Parameters for a single sparse SDF grid level (matches WGSL `GridParams`).
/// Size: 80 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SdfGridParams {
    /// World-space origin of the grid.
    pub world_min: [f32; 3],
    /// Voxel size (world units per voxel).
    pub voxel_size: f32,
    /// Grid dimensions in bricks (x, y, z).
    pub grid_dim: [u32; 3],
    /// Brick size in voxels (e.g., 8).
    pub brick_size: u32,
    /// Total number of active bricks (set by CPU before dispatch).
    pub active_brick_count: u32,
    /// Atlas capacity in bricks.
    pub atlas_capacity: u32,
    /// Total number of SDF edits.
    pub edit_count: u32,
    /// Whether terrain is enabled (0 or 1).
    pub terrain_enabled: u32,
    /// Atlas dimensions (bricks per axis).
    pub atlas_dim: [u32; 3],
    /// Padding.
    pub _pad: u32,
}

impl SdfGridParams {
    /// Construct params for a sparse grid at `world_min`/`world_max`.
    pub fn new_sparse(
        world_min: [f32; 3],
        voxel_size: f32,
        grid_dim: u32,
        brick_size: u32,
        atlas_dim: u32,
        active_brick_count: u32,
        edit_count: u32,
        terrain_enabled: bool,
    ) -> Self {
        Self {
            world_min,
            voxel_size,
            grid_dim: [grid_dim; 3],
            brick_size,
            active_brick_count,
            atlas_capacity: atlas_dim * atlas_dim * atlas_dim,
            edit_count,
            terrain_enabled: terrain_enabled as u32,
            atlas_dim: [atlas_dim; 3],
            _pad: 0,
        }
    }
}

/// Per-level clip-map state uploaded to GPU (matches WGSL `ClipLevel`).
/// Size: 64 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuClipLevel {
    /// World-space origin of this level's grid.
    pub world_min: [f32; 3],
    /// Voxel size for this level.
    pub voxel_size: f32,
    /// Grid dimensions in voxels per side.
    pub grid_dim: u32,
    /// Brick dimensions per side.
    pub brick_dim: u32,
    /// Offset into the global brick-indices array (packed bricks for this level).
    pub brick_index_offset: u32,
    /// Number of active bricks for this level.
    pub active_brick_count: u32,
    /// Toroidal origin (in brick coords) for this level.
    pub toroidal_origin: [i32; 3],
    /// Padding.
    pub _pad0: u32,
    /// Atlas grid dimensions for this level.
    pub atlas_dim: [u32; 3],
    /// Padding.
    pub _pad1: u32,
}

/// Full clip-map parameters uploaded to GPU (matches WGSL `ClipMapParams`).
/// Size: 16 + 8 * 64 = 528 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SdfClipMapParams {
    /// Number of active clip levels.
    pub level_count: u32,
    pub _pad: [u32; 3],
    pub levels: [GpuClipLevel; 8],
}
