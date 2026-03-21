//! Multi-resolution SDF clip-map: 8 nested levels with toroidal scrolling.
//!
//! Level 0 is the finest (smallest voxel_size), level 7 is the coarsest.
//! When the camera moves, only the newly-visible shell of each level is
//! reclassified — O(1) CPU work per frame.

use glam::Vec3;
use bytemuck;
use crate::brick::BrickMap;
use crate::edit_bvh::{Aabb, EditBvh};
use crate::terrain::TerrainConfig;
use crate::edit_list::GpuSdfEdit;
use crate::uniforms::{GpuClipLevel, SdfClipMapParams};

/// Number of clip-map levels.
pub const LEVEL_COUNT: usize = 8;
/// Grid dimension in bricks per axis per level.
pub const GRID_DIM: u32 = 16;
/// Brick size in voxels.
pub const BRICK_SIZE: u32 = 8;
/// Atlas dimension in bricks per axis.
pub const ATLAS_DIM: u32 = 8; // 8^3 = 512 bricks per level

pub struct ClipLevel {
    pub brick_map: BrickMap,
    /// Camera-aligned world-space center for this level (snapped to brick grid).
    pub center: Vec3,
    /// Level index (0 = finest).
    pub level_idx: usize,
}

impl ClipLevel {
    pub fn new(level_idx: usize, base_voxel_size: f32, center: Vec3) -> Self {
        let voxel_size = base_voxel_size * (1 << level_idx) as f32;
        let (world_min, toroidal_origin) = Self::compute_origin(center, voxel_size);
        let mut bm = BrickMap::new(world_min, voxel_size, GRID_DIM, BRICK_SIZE, ATLAS_DIM);
        bm.toroidal_origin = toroidal_origin;
        ClipLevel { brick_map: bm, center, level_idx }
    }

    fn compute_origin(center: Vec3, voxel_size: f32) -> (Vec3, [i32; 3]) {
        let bs = (BRICK_SIZE as f32) * voxel_size;
        let half = (GRID_DIM as f32) * bs * 0.5;
        // Snap center to brick grid.
        let snapped = Vec3::new(
            (center.x / bs).floor() * bs,
            (center.y / bs).floor() * bs,
            (center.z / bs).floor() * bs,
        );
        let world_min = snapped - Vec3::splat(half);
        let origin = [
            (world_min.x / bs) as i32,
            (world_min.y / bs) as i32,
            (world_min.z / bs) as i32,
        ];
        (world_min, origin)
    }

    /// Move the level center to `new_center`. Returns the previous toroidal origin.
    pub fn update_center(&mut self, new_center: Vec3) -> [i32; 3] {
        let voxel_size = self.brick_map.voxel_size;
        let (world_min, new_origin) = Self::compute_origin(new_center, voxel_size);
        let prev_origin = self.brick_map.toroidal_origin;
        self.center = new_center;
        self.brick_map.world_min = world_min;
        self.brick_map.toroidal_origin = new_origin;
        prev_origin
    }
}

/// Full multi-resolution SDF clip-map.
pub struct SdfClipMap {
    pub levels: Vec<ClipLevel>,
    pub base_voxel_size: f32,

    /// GPU buffer for SdfClipMapParams (uploaded each frame if dirty).
    pub clip_params_buf: Option<wgpu::Buffer>,
    /// CPU-side concatenated active brick indices across all levels.
    pub cached_all_indices: Vec<u32>,
    /// GPU buffer for the concatenated all-brick-indices.
    pub all_brick_indices_buf: Option<wgpu::Buffer>,
}

impl SdfClipMap {
    pub fn new(base_voxel_size: f32, center: Vec3) -> Self {
        let levels = (0..LEVEL_COUNT)
            .map(|i| ClipLevel::new(i, base_voxel_size, center))
            .collect();
        Self {
            levels,
            base_voxel_size,
            clip_params_buf: None,
            cached_all_indices: Vec::new(),
            all_brick_indices_buf: None,
        }
    }

    /// Create GPU buffers for the clip-map level data and all-brick-indices.
    pub fn create_gpu_resources(&mut self, device: &wgpu::Device) {
        for level in &mut self.levels {
            level.brick_map.create_gpu_resources(device);
        }
        self.clip_params_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SdfClipMapParams"),
            size: std::mem::size_of::<SdfClipMapParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        // Max size: LEVEL_COUNT * GRID_DIM^3 * 4 bytes.
        let max_indices = LEVEL_COUNT * (GRID_DIM as usize).pow(3);
        self.all_brick_indices_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF All Brick Indices"),
            size: (max_indices * 4).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
    }

    /// Full classify: reclassify all bricks in all levels against the BVH.
    pub fn classify_all(
        &mut self,
        bvh: &EditBvh,
        edits: &[GpuSdfEdit],
        terrain: Option<&TerrainConfig>,
    ) {
        for level in &mut self.levels {
            level.brick_map.classify_bvh(bvh, edits, terrain);
        }
    }

    /// Update camera center and return a bitmask of which levels scrolled.
    pub fn update_center(&mut self, new_center: Vec3) -> u8 {
        let mut dirty_mask = 0u8;
        for (i, level) in self.levels.iter_mut().enumerate() {
            let voxel_size = level.brick_map.voxel_size;
            let bs = (BRICK_SIZE as f32) * voxel_size;
            let old_snap = Vec3::new(
                (level.center.x / bs).floor() * bs,
                (level.center.y / bs).floor() * bs,
                (level.center.z / bs).floor() * bs,
            );
            let new_snap = Vec3::new(
                (new_center.x / bs).floor() * bs,
                (new_center.y / bs).floor() * bs,
                (new_center.z / bs).floor() * bs,
            );
            if old_snap.distance_squared(new_snap) > 0.0 {
                dirty_mask |= 1 << i;
                level.update_center(new_center);
            }
        }
        dirty_mask
    }

    /// Toroidal reclassify for dirty levels.
    pub fn classify_toroidal_levels(
        &mut self,
        dirty_mask: u8,
        prev_origins: &[[i32; 3]; LEVEL_COUNT],
        bvh: &EditBvh,
        edits: &[GpuSdfEdit],
        terrain: Option<&TerrainConfig>,
    ) {
        for (i, level) in self.levels.iter_mut().enumerate() {
            if dirty_mask & (1 << i) != 0 {
                let prev = prev_origins[i];
                let curr = level.brick_map.toroidal_origin;
                level.brick_map.classify_toroidal(prev, curr, bvh, edits, terrain);
            }
        }
    }

    /// Upload dirty GPU buffers for all levels.
    pub fn upload_dirty(&mut self, dirty_mask: u8, queue: &wgpu::Queue) {
        for (i, level) in self.levels.iter_mut().enumerate() {
            if dirty_mask & (1 << i) != 0 || level.brick_map.dirty.iter().any(|d| *d) {
                level.brick_map.upload_dirty(queue);
            }
        }
    }

    /// Rebuild the cached concatenated all-brick-indices and upload it.
    pub fn update_cached_indices(&mut self, queue: &wgpu::Queue) {
        self.cached_all_indices.clear();
        for level in &self.levels {
            let active: Vec<u32> = level.brick_map.states.iter().enumerate()
                .filter_map(|(i, s)| {
                    if matches!(s, crate::brick::BrickState::Active(_)) { Some(i as u32) } else { None }
                })
                .collect();
            self.cached_all_indices.extend_from_slice(&active);
        }
        if let Some(buf) = &self.all_brick_indices_buf {
            if !self.cached_all_indices.is_empty() {
                queue.write_buffer(buf, 0, bytemuck::cast_slice(&self.cached_all_indices));
            }
        }
    }

    /// Upload SdfClipMapParams to GPU buffer.
    pub fn upload_clip_params(&self, queue: &wgpu::Queue) {
        let params = self.build_clip_params();
        if let Some(buf) = &self.clip_params_buf {
            queue.write_buffer(buf, 0, bytemuck::bytes_of(&params));
        }
    }

    /// Build the SdfClipMapParams struct.
    pub fn build_clip_params(&self) -> SdfClipMapParams {
        let mut levels = [GpuClipLevel::zeroed(); LEVEL_COUNT];
        let mut brick_index_offset = 0u32;
        for (i, level) in self.levels.iter().enumerate() {
            let bm = &level.brick_map;
            levels[i] = GpuClipLevel {
                world_min: bm.world_min.to_array(),
                voxel_size: bm.voxel_size,
                grid_dim: bm.grid_dim,
                brick_dim: bm.brick_size,
                brick_index_offset,
                active_brick_count: bm.active_count(),
                toroidal_origin: bm.toroidal_origin,
                _pad0: 0,
                atlas_dim: [bm.atlas_dim; 3],
                _pad1: 0,
            };
            brick_index_offset += bm.active_count();
        }
        SdfClipMapParams {
            level_count: LEVEL_COUNT as u32,
            _pad: [0; 3],
            levels,
        }
    }
}

use bytemuck::Zeroable;
