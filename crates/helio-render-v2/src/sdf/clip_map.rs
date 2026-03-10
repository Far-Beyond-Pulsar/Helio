//! Geometry Clip Maps — multi-resolution SDF with camera-centered LOD
//!
//! Nests multiple `BrickMap` levels centered on the camera, each with
//! progressively coarser voxel sizes. Level 0 is finest, Level 3 is coarsest.

use super::brick::BrickMap;
use super::edit_list::SdfEdit;
use super::uniforms::SdfGridParams;
use super::brick::DEFAULT_BRICK_SIZE;
use std::sync::Arc;

/// Default number of clip levels
pub const DEFAULT_CLIP_LEVELS: u32 = 4;

/// Each level multiplies the voxel size by this factor
pub const CLIP_VOXEL_SCALE: f32 = 2.0;

/// Default atlas bricks per axis per level
const DEFAULT_ATLAS_BRICKS_PER_AXIS: u32 = 16;

/// GPU-side per-level clip data (48 bytes, 16-byte aligned).
/// Must match WGSL `GpuClipLevel` exactly.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuClipLevel {
    pub volume_min: [f32; 3],
    pub _pad0: f32,
    pub volume_max: [f32; 3],
    pub _pad1: f32,
    pub voxel_size: f32,
    pub brick_size: u32,
    pub brick_grid_dim: u32,
    pub atlas_bricks_per_axis: u32,
}

/// GPU-side clip map parameters (208 bytes, 16-byte aligned).
/// Must match WGSL `SdfClipMapParams` exactly.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SdfClipMapParams {
    pub level_count: u32,
    pub grid_dim: u32,
    pub max_march_dist: f32,
    pub debug_flags: u32,
    pub levels: [GpuClipLevel; 4],
}

/// A single clip level with its own BrickMap.
pub struct ClipLevel {
    pub level_index: u32,
    pub voxel_size: f32,
    pub volume_min: [f32; 3],
    pub volume_max: [f32; 3],
    pub brick_map: BrickMap,
    pub last_snapped_center: [f32; 3],
    pub params_buffer: Option<Arc<wgpu::Buffer>>,
}

/// Multi-level clip map managing several `BrickMap`s centered on the camera.
pub struct SdfClipMap {
    levels: Vec<ClipLevel>,
    grid_dim: u32,
    base_voxel_size: f32,
    level_count: u32,
    pub clip_params_buffer: Option<Arc<wgpu::Buffer>>,
}

impl SdfClipMap {
    /// Create a new clip map with the given base voxel size and level count.
    pub fn new(grid_dim: u32, base_voxel_size: f32, level_count: u32) -> Self {
        let mut levels = Vec::with_capacity(level_count as usize);

        for i in 0..level_count {
            let scale = CLIP_VOXEL_SCALE.powi(i as i32);
            let voxel_size = base_voxel_size * scale;
            let half_extent = voxel_size * grid_dim as f32 * 0.5;

            levels.push(ClipLevel {
                level_index: i,
                voxel_size,
                volume_min: [-half_extent, -half_extent, -half_extent],
                volume_max: [half_extent, half_extent, half_extent],
                brick_map: BrickMap::new(grid_dim, DEFAULT_BRICK_SIZE, DEFAULT_ATLAS_BRICKS_PER_AXIS),
                last_snapped_center: [f32::NAN, f32::NAN, f32::NAN], // force first update
                params_buffer: None,
            });
        }

        Self {
            levels,
            grid_dim,
            base_voxel_size,
            level_count,
            clip_params_buffer: None,
        }
    }

    /// Create GPU resources for all levels. Called once from `SdfFeature::register()`.
    pub fn create_gpu_resources(&mut self, device: &wgpu::Device) {
        // Clip params uniform buffer
        let clip_params_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Clip Map Params"),
            size: std::mem::size_of::<SdfClipMapParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.clip_params_buffer = Some(clip_params_buffer);

        // Per-level resources
        for (i, level) in self.levels.iter_mut().enumerate() {
            level.brick_map.create_gpu_resources(device);

            let params_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("SDF Clip Level {} Params", i)),
                size: std::mem::size_of::<SdfGridParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            level.params_buffer = Some(params_buffer);
        }
    }

    /// Update volume centers based on a focal point (e.g. volume center or camera).
    /// Returns a bitmask of which levels have dirty bricks.
    pub fn update_center(&mut self, center: glam::Vec3) -> u32 {
        let mut dirty_mask = 0u32;

        for level in self.levels.iter_mut() {
            let vs = level.voxel_size;
            let half_extent = vs * self.grid_dim as f32 * 0.5;

            // Snap center to voxel grid to prevent swimming
            let snapped = [
                (center.x / vs).floor() * vs,
                (center.y / vs).floor() * vs,
                (center.z / vs).floor() * vs,
            ];

            if snapped != level.last_snapped_center {
                level.volume_min = [
                    snapped[0] - half_extent,
                    snapped[1] - half_extent,
                    snapped[2] - half_extent,
                ];
                level.volume_max = [
                    snapped[0] + half_extent,
                    snapped[1] + half_extent,
                    snapped[2] + half_extent,
                ];
                level.last_snapped_center = snapped;
                dirty_mask |= 1 << level.level_index;
            }
        }

        dirty_mask
    }

    /// Classify active bricks for all dirty levels.
    pub fn classify_dirty_levels(
        &mut self,
        dirty_mask: u32,
        edits: &[SdfEdit],
    ) {
        for level in self.levels.iter_mut() {
            if dirty_mask & (1 << level.level_index) != 0 {
                level.brick_map.classify(
                    edits,
                    level.volume_min,
                    level.volume_max,
                );
            }
        }
    }

    /// Upload all buffers for dirty levels. Returns per-level active brick counts.
    pub fn upload_dirty_levels(
        &self,
        dirty_mask: u32,
        queue: &wgpu::Queue,
        edit_count: u32,
    ) -> Vec<u32> {
        let mut counts = Vec::with_capacity(self.levels.len());
        for level in &self.levels {
            if dirty_mask & (1 << level.level_index) != 0 {
                level.brick_map.upload(queue);

                // Upload per-level SdfGridParams
                if let Some(buf) = &level.params_buffer {
                    let params = SdfGridParams::new_sparse(
                        level.volume_min,
                        level.volume_max,
                        self.grid_dim,
                        edit_count,
                        level.brick_map.brick_size(),
                        level.brick_map.active_count(),
                        level.brick_map.atlas_bricks_per_axis(),
                    );
                    queue.write_buffer(buf, 0, bytemuck::bytes_of(&params));
                }
            }
            counts.push(level.brick_map.active_count());
        }
        counts
    }

    /// Build the GPU clip map params uniform.
    pub fn build_clip_params(&self) -> SdfClipMapParams {
        let mut gpu_levels = [GpuClipLevel {
            volume_min: [0.0; 3],
            _pad0: 0.0,
            volume_max: [0.0; 3],
            _pad1: 0.0,
            voxel_size: 0.0,
            brick_size: 0,
            brick_grid_dim: 0,
            atlas_bricks_per_axis: 0,
        }; 4];

        for (i, level) in self.levels.iter().enumerate() {
            if i >= 4 { break; }
            gpu_levels[i] = GpuClipLevel {
                volume_min: level.volume_min,
                _pad0: 0.0,
                volume_max: level.volume_max,
                _pad1: 0.0,
                voxel_size: level.voxel_size,
                brick_size: level.brick_map.brick_size(),
                brick_grid_dim: level.brick_map.brick_grid_dim(),
                atlas_bricks_per_axis: level.brick_map.atlas_bricks_per_axis(),
            };
        }

        // max_march_dist from the coarsest level
        let coarsest = &self.levels[self.levels.len() - 1];
        let range_x = coarsest.volume_max[0] - coarsest.volume_min[0];
        let range_y = coarsest.volume_max[1] - coarsest.volume_min[1];
        let range_z = coarsest.volume_max[2] - coarsest.volume_min[2];
        let max_range = range_x.max(range_y).max(range_z);

        SdfClipMapParams {
            level_count: self.level_count,
            grid_dim: self.grid_dim,
            max_march_dist: max_range * 2.0,
            debug_flags: 0,
            levels: gpu_levels,
        }
    }

    /// Build concatenated brick indices for all levels (for the ray march shader).
    /// Returns a Vec<u32> with level_count * brick_grid_dim^3 entries.
    pub fn build_all_brick_indices(&self) -> Vec<u32> {
        let mut all = Vec::new();
        for level in &self.levels {
            all.extend_from_slice(level.brick_map.brick_index_slice());
        }
        all
    }

    pub fn levels(&self) -> &[ClipLevel] {
        &self.levels
    }

    pub fn levels_mut(&mut self) -> &mut [ClipLevel] {
        &mut self.levels
    }

    pub fn level_count(&self) -> u32 {
        self.level_count
    }

    pub fn grid_dim(&self) -> u32 {
        self.grid_dim
    }
}
