//! GPU uniform structs for the SDF system

/// Parameters for the SDF evaluation grid (48 bytes, 16-byte aligned)
///
/// Layout must match the WGSL `SdfGridParams` struct.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SdfGridParams {
    pub volume_min: [f32; 3],
    pub grid_dim: u32,
    pub volume_max: [f32; 3],
    pub edit_count: u32,
    pub voxel_size: f32,
    pub max_march_dist: f32,
    pub _pad: [f32; 2],
}

impl SdfGridParams {
    pub fn new(
        volume_min: [f32; 3],
        volume_max: [f32; 3],
        grid_dim: u32,
        edit_count: u32,
    ) -> Self {
        let range_x = volume_max[0] - volume_min[0];
        let range_y = volume_max[1] - volume_min[1];
        let range_z = volume_max[2] - volume_min[2];
        let max_range = range_x.max(range_y).max(range_z);
        let voxel_size = max_range / grid_dim as f32;
        let max_march_dist = max_range * 2.0;

        Self {
            volume_min,
            grid_dim,
            volume_max,
            edit_count,
            voxel_size,
            max_march_dist,
            _pad: [0.0; 2],
        }
    }
}
