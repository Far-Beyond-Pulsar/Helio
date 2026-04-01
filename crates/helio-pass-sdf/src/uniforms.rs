//! GPU uniform structs for the SDF system.

/// Parameters for the SDF evaluation grid (80 bytes, 16-byte aligned).
///
/// Layout must match the WGSL `SdfGridParams` struct.
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
    pub active_brick_count: u32,
    pub atlas_bricks_per_axis: u32,
    pub grid_origin: [f32; 3],
    pub debug_flags: u32,
}

impl SdfGridParams {
    pub fn new_sparse(
        volume_min: [f32; 3],
        volume_max: [f32; 3],
        grid_dim: u32,
        edit_count: u32,
        brick_size: u32,
        active_brick_count: u32,
        atlas_bricks_per_axis: u32,
    ) -> Self {
        let range_x = volume_max[0] - volume_min[0];
        let range_y = volume_max[1] - volume_min[1];
        let range_z = volume_max[2] - volume_min[2];
        let max_range = range_x.max(range_y).max(range_z);
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
            brick_grid_dim: grid_dim / brick_size,
            active_brick_count,
            atlas_bricks_per_axis,
            grid_origin: [0.0; 3],
            debug_flags: 0,
        }
    }
}
