//! GPU uniform structs for the SDF system

/// Parameters for the SDF evaluation grid (80 bytes, 16-byte aligned)
///
/// Layout must match the WGSL `SdfGridParams` struct.
/// NOTE: vec3 + pad pattern avoids the vec3 alignment trap where
/// scalars packed into the vec3 tail read incorrectly on some drivers.
///
/// The first 48 bytes are used by the dense path. The sparse/clip-map
/// paths also read the Phase 3 fields (bytes 48..64). Debug fields at 64..80.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SdfGridParams {
    pub volume_min: [f32; 3],
    pub _pad0: f32,           // explicit padding after vec3
    pub volume_max: [f32; 3],
    pub _pad1: f32,           // explicit padding after vec3
    pub grid_dim: u32,
    pub edit_count: u32,
    pub voxel_size: f32,
    pub max_march_dist: f32,
    // Phase 3 additions (16 bytes)
    pub brick_size: u32,            // voxels per brick edge (8)
    pub brick_grid_dim: u32,        // grid_dim / brick_size (16)
    pub active_brick_count: u32,    // number of active bricks this frame
    pub atlas_bricks_per_axis: u32, // how many bricks fit along each atlas axis
    // Debug (16 bytes)
    pub debug_flags: u32,           // 0 = normal, 1 = debug visualization
    pub _pad2: u32,
    pub _pad3: u32,
    pub _pad4: u32,
}

impl SdfGridParams {
    /// Create params for the dense path (Phase 3 fields zeroed).
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
            _pad0: 0.0,
            volume_max,
            _pad1: 0.0,
            grid_dim,
            edit_count,
            voxel_size,
            max_march_dist,
            brick_size: 0,
            brick_grid_dim: 0,
            active_brick_count: 0,
            atlas_bricks_per_axis: 0,
            debug_flags: 0,
            _pad2: 0,
            _pad3: 0,
            _pad4: 0,
        }
    }

    /// Create params for the sparse brick path.
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
            debug_flags: 0,
            _pad2: 0,
            _pad3: 0,
            _pad4: 0,
        }
    }

    /// Set the debug flags on an existing params struct.
    pub fn with_debug(mut self, enabled: bool) -> Self {
        self.debug_flags = if enabled { 1 } else { 0 };
        self
    }
}
