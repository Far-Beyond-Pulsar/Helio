//! GPU uniform structs for the SDF system.

// Compile-time assertion: SdfGridParams must be exactly 96 bytes.
const _: () = assert!(
    std::mem::size_of::<SdfGridParams>() == 96,
    "SdfGridParams must be 96 bytes to match sdf_evaluate.wgsl",
);

/// Parameters for the SDF evaluation grid (96 bytes, 16-byte aligned).
///
/// Layout must match the WGSL `SdfGridParams` struct in `sdf_evaluate.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SdfGridParams {
    pub volume_min: [f32; 3],
    pub _pad0: f32,
    pub volume_max: [f32; 3],
    pub _pad1: f32,
    pub grid_dim: u32,
    pub edit_count: u32,
    pub voxel_size: f32,
    pub max_march_dist: f32,
    pub brick_size: u32,
    pub brick_grid_dim: u32,
    /// Clip-map level index (0 = finest).  Was `active_brick_count` in the old layout.
    pub level_idx: u32,
    pub atlas_bricks_per_axis: u32,
    pub grid_origin: [f32; 3],
    pub debug_flags: u32,
    /// Number of bricks per level (`brick_grid_dim^3`).
    pub bricks_per_level: u32,
    pub _pad2: u32,
    pub _pad3: u32,
    pub _pad4: u32,
}

impl SdfGridParams {
    pub fn new_sparse(
        volume_min: [f32; 3],
        volume_max: [f32; 3],
        grid_dim: u32,
        edit_count: u32,
        brick_size: u32,
        level_idx: u32,
        atlas_bricks_per_axis: u32,
    ) -> Self {
        let range_x = volume_max[0] - volume_min[0];
        let range_y = volume_max[1] - volume_min[1];
        let range_z = volume_max[2] - volume_min[2];
        let max_range = range_x.max(range_y).max(range_z);
        let brick_grid_dim = grid_dim / brick_size;
        let bricks_per_level = brick_grid_dim * brick_grid_dim * brick_grid_dim;
        let voxel_size = max_range / grid_dim as f32;
        let max_march_dist = max_range * 2.0;

        Self {
            volume_min,
            _pad0: 0.0,
            volume_max,
            _pad1: 0.0,
            grid_dim,
            edit_count,
            voxel_size,
            max_march_dist,
            brick_size,
            brick_grid_dim,
            level_idx,
            atlas_bricks_per_axis,
            grid_origin: [0.0; 3],
            debug_flags: 0,
            bricks_per_level,
            _pad2: 0,
            _pad3: 0,
            _pad4: 0,
        }
    }
}
