//! Sparse brick map for a single SDF clip-map level.
//!
//! Maintains a 3D grid of `BrickState` values; manages GPU atlas allocation
//! and toroidal scrolling when the camera moves.

use std::collections::HashMap;
use glam::Vec3;
use crate::edit_bvh::{Aabb, EditBvh};
use crate::terrain::TerrainConfig;
use crate::noise::{terrain_sdf, terrain_height_range};
use crate::edit_list::GpuSdfEdit;
use crate::uniforms::SdfGridParams;

/// State of one brick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrickState {
    /// Brick is outside any SDF surface — skip during ray march.
    Empty,
    /// Brick overlaps a surface — occupies an atlas slot.
    Active(u32), // atlas index
    /// Brick was active, turned empty, atlas slot freed.
    Freed,
}

/// CPU+GPU brick map for one level.
pub struct BrickMap {
    /// World-space origin of the grid.
    pub world_min: Vec3,
    /// Voxel size for this level.
    pub voxel_size: f32,
    /// Grid dimensions in bricks per axis.
    pub grid_dim: u32,
    /// Brick size in voxels.
    pub brick_size: u32,
    /// Atlas capacity (bricks per axis).
    pub atlas_dim: u32,

    /// Per-brick state: grid_dim^3 entries. Index: z*grid_dim^2 + y*grid_dim + x.
    pub states: Vec<BrickState>,
    /// Free-list of atlas indices.
    free_atlas: Vec<u32>,
    /// Next atlas slot to allocate (when free_atlas is exhausted).
    next_atlas: u32,
    /// Dirty flags: which bricks need re-evaluation.
    pub dirty: Vec<bool>,

    /// Toroidal origin in brick coordinates.
    pub toroidal_origin: [i32; 3],

    // ---- GPU resources (created lazily via `create_gpu_resources`) ----
    /// Atlas storage buffer (f32 SDF values, quantized u8→u32 packed).
    pub atlas_buffer: Option<wgpu::Buffer>,
    /// Active-brick list: linear array of flat brick indices.
    pub active_brick_buf: Option<wgpu::Buffer>,
    /// Brick-index map: maps flat grid idx → atlas idx (u32::MAX if empty).
    pub brick_index_buf: Option<wgpu::Buffer>,
    /// Per-level grid params uniform buffer.
    pub params_buf: Option<wgpu::Buffer>,

    /// Per-brick edit-list offsets GPU buffer (one per brick).
    pub edit_list_offsets_buf: Option<wgpu::Buffer>,
    /// Per-brick edit-list data GPU buffer (per-brick packed u32 edit indices).
    pub edit_list_data_buf: Option<wgpu::Buffer>,
    /// CPU-side: sorted edit lists per brick. Dense vec of (offset, count) pairs.
    pub edit_list_offsets: Vec<u32>,
    pub edit_list_data: Vec<u32>,

    /// CPU-side: flat brick-index map (grid_idx → atlas_idx).
    pub brick_index_cpu: Vec<u32>,
}

const EMPTY_ATLAS: u32 = u32::MAX;

impl BrickMap {
    pub fn new(
        world_min: Vec3,
        voxel_size: f32,
        grid_dim: u32,
        brick_size: u32,
        atlas_dim: u32,
    ) -> Self {
        let n = (grid_dim * grid_dim * grid_dim) as usize;
        let atlas_cap = (atlas_dim * atlas_dim * atlas_dim) as usize;
        Self {
            world_min,
            voxel_size,
            grid_dim,
            brick_size,
            atlas_dim,
            states: vec![BrickState::Empty; n],
            free_atlas: Vec::new(),
            next_atlas: 0,
            dirty: vec![false; n],
            toroidal_origin: [0; 3],
            atlas_buffer: None,
            active_brick_buf: None,
            brick_index_buf: None,
            params_buf: None,
            edit_list_offsets_buf: None,
            edit_list_data_buf: None,
            edit_list_offsets: vec![0u32; n + 1],
            edit_list_data: Vec::new(),
            brick_index_cpu: vec![EMPTY_ATLAS; n],
        }
    }

    /// Allocate all GPU buffers.
    pub fn create_gpu_resources(&mut self, device: &wgpu::Device) {
        let n = (self.grid_dim * self.grid_dim * self.grid_dim) as usize;
        let voxels_per_brick = (self.brick_size * self.brick_size * self.brick_size) as usize;
        let atlas_cap = (self.atlas_dim * self.atlas_dim * self.atlas_dim) as usize;
        // 4 voxels packed per u32 (u8 values).
        let atlas_bytes = (atlas_cap * voxels_per_brick / 4) as u64 * 4;

        self.atlas_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Atlas"),
            size: atlas_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.active_brick_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Active Bricks"),
            size: (n * 4).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.brick_index_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Brick Index"),
            size: (n * 4).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.params_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Grid Params"),
            size: std::mem::size_of::<SdfGridParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.edit_list_offsets_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Edit List Offsets"),
            size: ((n + 1) * 4).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        // Edit list data preallocated for up to 64 edits per brick.
        self.edit_list_data_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Edit List Data"),
            size: (n * 4 * 64).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
    }

    fn brick_world_aabb(&self, bx: i32, by: i32, bz: i32) -> Aabb {
        let bs = (self.brick_size as f32) * self.voxel_size;
        let min = self.world_min + Vec3::new(bx as f32 * bs, by as f32 * bs, bz as f32 * bs);
        Aabb::new(min, min + Vec3::splat(bs))
    }

    /// Classify all bricks against the BVH of edits + optional terrain.
    /// Marks dirty=true for any brick that changes state.
    /// Returns a list of newly-active brick flat indices (for atlas alloc).
    pub fn classify_bvh(
        &mut self,
        bvh: &EditBvh,
        edits: &[GpuSdfEdit],
        terrain: Option<&TerrainConfig>,
    ) {
        let gd = self.grid_dim as i32;
        for bz in 0..gd {
            for by in 0..gd {
                for bx in 0..gd {
                    let flat = (bz * gd * gd + by * gd + bx) as usize;
                    let aabb = self.brick_world_aabb(bx, by, bz);
                    let occupied = self.brick_overlaps_surface(&aabb, bvh, edits, terrain);
                    self.set_brick_occupied(flat, occupied);
                }
            }
        }
        self.rebuild_edit_lists(bvh, edits);
    }

    /// Classify only the bricks within the toroidal shell that was scrolled in.
    pub fn classify_toroidal(
        &mut self,
        prev_origin: [i32; 3],
        new_origin: [i32; 3],
        bvh: &EditBvh,
        edits: &[GpuSdfEdit],
        terrain: Option<&TerrainConfig>,
    ) {
        let gd = self.grid_dim as i32;
        // Determine which brick coords changed (the shell).
        let delta = [
            new_origin[0] - prev_origin[0],
            new_origin[1] - prev_origin[1],
            new_origin[2] - prev_origin[2],
        ];
        for bz in 0..gd {
            for by in 0..gd {
                for bx in 0..gd {
                    // World brick coord = toroidal + grid index
                    let wx = new_origin[0] + bx;
                    let wy = new_origin[1] + by;
                    let wz = new_origin[2] + bz;
                    let in_shell = Self::in_scroll_shell(bx, by, bz, delta, gd);
                    if !in_shell { continue; }
                    let tx = ((bx - new_origin[0]).rem_euclid(gd)) as usize;
                    let ty = ((by - new_origin[1]).rem_euclid(gd)) as usize;
                    let tz = ((bz - new_origin[2]).rem_euclid(gd)) as usize;
                    let flat = tz * (gd as usize * gd as usize) + ty * gd as usize + tx;
                    let aabb = self.brick_world_aabb(wx, wy, wz);
                    let occupied = self.brick_overlaps_surface(&aabb, bvh, edits, terrain);
                    self.set_brick_occupied(flat, occupied);
                }
            }
        }
        self.rebuild_edit_lists(bvh, edits);
    }

    fn in_scroll_shell(bx: i32, by: i32, bz: i32, delta: [i32; 3], gd: i32) -> bool {
        (delta[0] != 0 && (bx == 0 || bx == gd - 1)) ||
        (delta[1] != 0 && (by == 0 || by == gd - 1)) ||
        (delta[2] != 0 && (bz == 0 || bz == gd - 1))
    }

    fn brick_overlaps_surface(
        &self,
        aabb: &Aabb,
        bvh: &EditBvh,
        edits: &[GpuSdfEdit],
        terrain: Option<&TerrainConfig>,
    ) -> bool {
        // Check terrain intersection.
        if let Some(tc) = terrain {
            let (min_h, max_h) = terrain_height_range(aabb.min, aabb.max, tc);
            if aabb.min.y <= max_h && aabb.max.y >= min_h {
                return true;
            }
        }
        // Check edits via BVH.
        let mut results = Vec::new();
        bvh.query_aabb(aabb, &mut results);
        !results.is_empty()
    }

    fn set_brick_occupied(&mut self, flat: usize, occupied: bool) {
        match (self.states[flat], occupied) {
            (BrickState::Empty, true) => {
                let atlas_idx = self.alloc_atlas();
                self.states[flat] = BrickState::Active(atlas_idx);
                self.brick_index_cpu[flat] = atlas_idx;
                self.dirty[flat] = true;
            }
            (BrickState::Active(idx), false) => {
                self.free_atlas.push(idx);
                self.states[flat] = BrickState::Empty;
                self.brick_index_cpu[flat] = EMPTY_ATLAS;
                self.dirty[flat] = false; // no need to re-evaluate
            }
            _ => {}
        }
    }

    fn alloc_atlas(&mut self) -> u32 {
        if let Some(idx) = self.free_atlas.pop() {
            idx
        } else {
            let idx = self.next_atlas;
            self.next_atlas += 1;
            idx
        }
    }

    /// Mark all bricks that overlap the AABB of a changed edit as dirty.
    pub fn mark_edit_dirty(&mut self, aabb: &Aabb) {
        let gd = self.grid_dim as i32;
        let bs = (self.brick_size as f32) * self.voxel_size;
        let bx0 = ((aabb.min.x - self.world_min.x) / bs).floor() as i32;
        let by0 = ((aabb.min.y - self.world_min.y) / bs).floor() as i32;
        let bz0 = ((aabb.min.z - self.world_min.z) / bs).floor() as i32;
        let bx1 = ((aabb.max.x - self.world_min.x) / bs).ceil() as i32;
        let by1 = ((aabb.max.y - self.world_min.y) / bs).ceil() as i32;
        let bz1 = ((aabb.max.z - self.world_min.z) / bs).ceil() as i32;
        for bz in bz0.max(0)..bz1.min(gd) {
            for by in by0.max(0)..by1.min(gd) {
                for bx in bx0.max(0)..bx1.min(gd) {
                    let flat = (bz * gd * gd + by * gd + bx) as usize;
                    if matches!(self.states[flat], BrickState::Active(_)) {
                        self.dirty[flat] = true;
                    }
                }
            }
        }
    }

    /// Rebuild the per-brick edit lists CPU-side using the BVH.
    fn rebuild_edit_lists(&mut self, bvh: &EditBvh, edits: &[GpuSdfEdit]) {
        let n = (self.grid_dim * self.grid_dim * self.grid_dim) as usize;
        let gd = self.grid_dim as i32;
        self.edit_list_data.clear();
        self.edit_list_offsets.resize(n + 1, 0);
        let mut offset = 0u32;
        for flat in 0..n {
            self.edit_list_offsets[flat] = offset;
            if !matches!(self.states[flat], BrickState::Active(_)) {
                continue;
            }
            let bx = (flat % self.grid_dim as usize) as i32;
            let by = ((flat / self.grid_dim as usize) % self.grid_dim as usize) as i32;
            let bz = (flat / (self.grid_dim as usize * self.grid_dim as usize)) as i32;
            let aabb = self.brick_world_aabb(bx, by, bz);
            let mut overlapping = Vec::new();
            bvh.query_aabb(&aabb, &mut overlapping);
            for idx in &overlapping {
                self.edit_list_data.push(*idx as u32);
            }
            offset += overlapping.len() as u32;
        }
        self.edit_list_offsets[n] = offset;
    }

    /// Upload all dirty bricks' GPU buffers (active list + brick index).
    pub fn upload(&self, queue: &wgpu::Queue) {
        if let Some(buf) = &self.active_brick_buf {
            let active: Vec<u32> = self.states.iter().enumerate().filter_map(|(i, s)| {
                if matches!(s, BrickState::Active(_)) { Some(i as u32) } else { None }
            }).collect();
            queue.write_buffer(buf, 0, bytemuck::cast_slice(&active));
        }
        if let Some(buf) = &self.brick_index_buf {
            queue.write_buffer(buf, 0, bytemuck::cast_slice(&self.brick_index_cpu));
        }
        if let Some(buf) = &self.edit_list_offsets_buf {
            queue.write_buffer(buf, 0, bytemuck::cast_slice(&self.edit_list_offsets));
        }
        if let Some(buf) = &self.edit_list_data_buf {
            if !self.edit_list_data.is_empty() {
                queue.write_buffer(buf, 0, bytemuck::cast_slice(&self.edit_list_data));
            }
        }
    }

    /// Upload only dirty bricks (re-upload active list + index map entirely; they're small).
    pub fn upload_dirty(&mut self, queue: &wgpu::Queue) {
        self.upload(queue);
        // Clear dirty flags.
        for d in &mut self.dirty { *d = false; }
    }

    /// How many active bricks this level currently has.
    pub fn active_count(&self) -> u32 {
        self.states.iter().filter(|s| matches!(s, BrickState::Active(_))).count() as u32
    }

    /// Build SdfGridParams for this level.
    pub fn build_grid_params(&self, edit_count: u32, terrain_enabled: bool) -> SdfGridParams {
        SdfGridParams::new_sparse(
            self.world_min.to_array(),
            self.voxel_size,
            self.grid_dim,
            self.brick_size,
            self.atlas_dim,
            self.active_count(),
            edit_count,
            terrain_enabled,
        )
    }
}
