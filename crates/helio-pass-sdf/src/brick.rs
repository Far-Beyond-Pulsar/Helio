//! Sparse Brick Map — CPU-side classification and GPU buffer management.
//!
//! Divides the SDF volume into a grid of 8^3 bricks. Only bricks whose AABB
//! intersects at least one edit's bounding sphere are allocated atlas slots.

use crate::edit_bvh::{Aabb, EditBvh};
use crate::edit_list::SdfEdit;
use crate::primitives::SdfShapeType;
use crate::terrain::TerrainConfig;
use std::collections::HashMap;

/// Voxels per brick edge.
pub const DEFAULT_BRICK_SIZE: u32 = 8;

/// Padded brick size for ghost voxels (1-voxel border for seamless trilinear).
pub const PADDED_BRICK_SIZE: u32 = DEFAULT_BRICK_SIZE + 1;

/// Sentinel value in the brick index meaning "no brick allocated".
pub const EMPTY_BRICK: u32 = 0xFFFF_FFFF;

/// Pack signed world brick coordinates into a single u32.
/// 10 bits per axis with bias 512 → range [-512, 511].
pub fn pack_world_brick_coord(wx: i32, wy: i32, wz: i32) -> u32 {
    let bias = 512i32;
    (((wx + bias) as u32) & 0x3FF)
        | ((((wy + bias) as u32) & 0x3FF) << 10)
        | ((((wz + bias) as u32) & 0x3FF) << 20)
}

/// Per-brick edit list: variable-length list of edit indices.
pub type EditList = Vec<u16>;

/// CPU-side brick map managing classification and atlas slot allocation.
pub struct BrickMap {
    brick_size: u32,
    brick_grid_dim: u32,
    atlas_bricks_per_axis: u32,

    brick_index: Vec<u32>,
    active_bricks: Vec<u32>,
    edit_lists: Vec<EditList>,
    dirty_bricks: Vec<u32>,
    dirty_edit_lists: Vec<EditList>,
    atlas_next_slot: u32,
    atlas_free_list: Vec<u32>,

    brick_world_coords: Vec<[i32; 3]>,
    stored_edit_lists: Vec<EditList>,

    pub brick_index_buffer: Option<wgpu::Buffer>,
    pub active_bricks_buffer: Option<wgpu::Buffer>,
    pub edit_list_offsets_buffer: Option<wgpu::Buffer>,
    pub edit_list_data_buffer: Option<wgpu::Buffer>,
    pub atlas_buffer: Option<wgpu::Buffer>,
    atlas_dim: u32,
}

impl BrickMap {
    pub fn new(grid_dim: u32, brick_size: u32, atlas_bricks_per_axis: u32) -> Self {
        let brick_grid_dim = grid_dim / brick_size;
        let total_bricks = (brick_grid_dim * brick_grid_dim * brick_grid_dim) as usize;

        Self {
            brick_size,
            brick_grid_dim,
            atlas_bricks_per_axis,
            brick_index: vec![EMPTY_BRICK; total_bricks],
            active_bricks: Vec::new(),
            edit_lists: Vec::new(),
            dirty_bricks: Vec::new(),
            dirty_edit_lists: Vec::new(),
            atlas_next_slot: 0,
            atlas_free_list: Vec::new(),
            brick_world_coords: vec![[i32::MAX; 3]; total_bricks],
            stored_edit_lists: vec![Vec::new(); total_bricks],
            brick_index_buffer: None,
            active_bricks_buffer: None,
            edit_list_offsets_buffer: None,
            edit_list_data_buffer: None,
            atlas_buffer: None,
            atlas_dim: atlas_bricks_per_axis * PADDED_BRICK_SIZE,
        }
    }

    pub fn create_gpu_resources(&mut self, device: &wgpu::Device) {
        let bgd = self.brick_grid_dim;
        let total_bricks = bgd * bgd * bgd;
        let max_atlas_bricks = self.atlas_bricks_per_axis.pow(3);

        self.brick_index_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Brick Index"),
            size: (total_bricks as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.active_bricks_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Active Bricks"),
            size: (max_atlas_bricks as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.edit_list_offsets_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Edit List Offsets"),
            size: (max_atlas_bricks as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let edit_list_data_size = (max_atlas_bricks as u64) * (1 + 64) * 4;
        self.edit_list_data_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Edit List Data"),
            size: edit_list_data_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let atlas_dim = self.atlas_bricks_per_axis * PADDED_BRICK_SIZE;
        let total_voxels = (atlas_dim as u64) * (atlas_dim as u64) * (atlas_dim as u64);
        let atlas_buffer_size = ((total_voxels + 3) / 4) * 4;
        self.atlas_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Brick Atlas Buffer"),
            size: atlas_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.atlas_dim = atlas_dim;
    }

    // ── Toroidal addressing ──────────────────────────────────────────────────

    fn world_to_grid(&self, wx: i32, wy: i32, wz: i32) -> usize {
        let bgd = self.brick_grid_dim as i32;
        let gx = ((wx % bgd) + bgd) % bgd;
        let gy = ((wy % bgd) + bgd) % bgd;
        let gz = ((wz % bgd) + bgd) % bgd;
        (gx + gy * bgd + gz * bgd * bgd) as usize
    }

    /// Toroidal classify: populates the brick_index using modular addressing.
    pub fn classify_toroidal(
        &mut self,
        _edits: &[SdfEdit],
        volume_min: [f32; 3],
        volume_max: [f32; 3],
        terrain: Option<&TerrainConfig>,
        bvh: Option<&EditBvh>,
        cached_bounds: &[(glam::Vec3, f32)],
    ) {
        let bgd = self.brick_grid_dim;
        let voxel_size = (volume_max[0] - volume_min[0]) / (bgd * self.brick_size) as f32;
        let brick_world_size = voxel_size * self.brick_size as f32;

        let wb_min = [
            (volume_min[0] / brick_world_size).floor() as i32,
            (volume_min[1] / brick_world_size).floor() as i32,
            (volume_min[2] / brick_world_size).floor() as i32,
        ];
        let wb_max = [
            wb_min[0] + bgd as i32 - 1,
            wb_min[1] + bgd as i32 - 1,
            wb_min[2] + bgd as i32 - 1,
        ];

        self.active_bricks.clear();
        self.edit_lists.clear();
        self.dirty_bricks.clear();
        self.dirty_edit_lists.clear();

        let mut used_slots = vec![false; (bgd * bgd * bgd) as usize];
        let mut hit_buf = Vec::new();

        for wz in wb_min[2]..=wb_max[2] {
            for wy in wb_min[1]..=wb_max[1] {
                for wx in wb_min[0]..=wb_max[0] {
                    let grid_idx = self.world_to_grid(wx, wy, wz);
                    let brick_min = glam::Vec3::new(
                        wx as f32 * brick_world_size,
                        wy as f32 * brick_world_size,
                        wz as f32 * brick_world_size,
                    );
                    let brick_max = brick_min + glam::Vec3::splat(brick_world_size);

                    let elist = Self::build_edit_list(
                        brick_min,
                        brick_max,
                        cached_bounds,
                        bvh,
                        &mut hit_buf,
                    );

                    let terrain_active = terrain.map_or(false, |cfg| {
                        Self::brick_intersects_terrain(brick_min, brick_max, cfg)
                    });

                    if !elist.is_empty() || terrain_active {
                        used_slots[grid_idx] = true;
                        let world_coord = [wx, wy, wz];
                        let existing_coord = self.brick_world_coords[grid_idx];
                        let is_same_world_pos = existing_coord == world_coord;
                        let has_slot = self.brick_index[grid_idx] != EMPTY_BRICK;

                        if has_slot && is_same_world_pos {
                            if elist != self.stored_edit_lists[grid_idx] {
                                self.dirty_bricks.push(pack_world_brick_coord(wx, wy, wz));
                                self.dirty_edit_lists.push(elist.clone());
                            }
                        } else {
                            if has_slot {
                                self.atlas_free_list.push(self.brick_index[grid_idx]);
                            }
                            let slot = self.alloc_atlas_slot();
                            self.brick_index[grid_idx] = slot;
                            self.brick_world_coords[grid_idx] = world_coord;
                            self.dirty_bricks.push(pack_world_brick_coord(wx, wy, wz));
                            self.dirty_edit_lists.push(elist.clone());
                        }

                        self.stored_edit_lists[grid_idx] = elist.clone();
                        self.active_bricks.push(grid_idx as u32);
                        self.edit_lists.push(elist);
                    } else {
                        used_slots[grid_idx] = true;
                        if self.brick_index[grid_idx] != EMPTY_BRICK {
                            self.atlas_free_list.push(self.brick_index[grid_idx]);
                            self.brick_index[grid_idx] = EMPTY_BRICK;
                            self.brick_world_coords[grid_idx] = [i32::MAX; 3];
                            self.stored_edit_lists[grid_idx].clear();
                        }
                    }
                }
            }
        }

        for i in 0..(bgd * bgd * bgd) as usize {
            if !used_slots[i] && self.brick_index[i] != EMPTY_BRICK {
                self.atlas_free_list.push(self.brick_index[i]);
                self.brick_index[i] = EMPTY_BRICK;
                self.brick_world_coords[i] = [i32::MAX; 3];
                self.stored_edit_lists[i].clear();
            }
        }
    }

    /// Toroidal scroll: only process newly-exposed edge bricks.
    pub fn scroll_toroidal(
        &mut self,
        old_volume_min: [f32; 3],
        new_volume_min: [f32; 3],
        new_volume_max: [f32; 3],
        _edits: &[SdfEdit],
        terrain: Option<&TerrainConfig>,
        bvh: Option<&EditBvh>,
        cached_bounds: &[(glam::Vec3, f32)],
    ) -> u32 {
        let bgd = self.brick_grid_dim;
        let voxel_size = (new_volume_max[0] - new_volume_min[0]) / (bgd * self.brick_size) as f32;
        let brick_world_size = voxel_size * self.brick_size as f32;

        let old_wb_min = [
            (old_volume_min[0] / brick_world_size).floor() as i32,
            (old_volume_min[1] / brick_world_size).floor() as i32,
            (old_volume_min[2] / brick_world_size).floor() as i32,
        ];
        let new_wb_min = [
            (new_volume_min[0] / brick_world_size).floor() as i32,
            (new_volume_min[1] / brick_world_size).floor() as i32,
            (new_volume_min[2] / brick_world_size).floor() as i32,
        ];
        let new_wb_max = [
            new_wb_min[0] + bgd as i32 - 1,
            new_wb_min[1] + bgd as i32 - 1,
            new_wb_min[2] + bgd as i32 - 1,
        ];
        let old_wb_max = [
            old_wb_min[0] + bgd as i32 - 1,
            old_wb_min[1] + bgd as i32 - 1,
            old_wb_min[2] + bgd as i32 - 1,
        ];

        self.dirty_bricks.clear();
        self.dirty_edit_lists.clear();

        // Phase 1: Free bricks that left the volume
        let mut removed_grid_indices = std::collections::HashSet::new();
        for wz in old_wb_min[2]..=old_wb_max[2] {
            for wy in old_wb_min[1]..=old_wb_max[1] {
                for wx in old_wb_min[0]..=old_wb_max[0] {
                    if wx >= new_wb_min[0]
                        && wx <= new_wb_max[0]
                        && wy >= new_wb_min[1]
                        && wy <= new_wb_max[1]
                        && wz >= new_wb_min[2]
                        && wz <= new_wb_max[2]
                    {
                        continue;
                    }
                    let grid_idx = self.world_to_grid(wx, wy, wz);
                    if self.brick_index[grid_idx] != EMPTY_BRICK
                        && self.brick_world_coords[grid_idx] == [wx, wy, wz]
                    {
                        self.atlas_free_list.push(self.brick_index[grid_idx]);
                        self.brick_index[grid_idx] = EMPTY_BRICK;
                        self.brick_world_coords[grid_idx] = [i32::MAX; 3];
                        self.stored_edit_lists[grid_idx].clear();
                        removed_grid_indices.insert(grid_idx as u32);
                    }
                }
            }
        }

        if !removed_grid_indices.is_empty() {
            let mut write = 0;
            for read in 0..self.active_bricks.len() {
                if !removed_grid_indices.contains(&self.active_bricks[read]) {
                    if write != read {
                        self.active_bricks[write] = self.active_bricks[read];
                        self.edit_lists[write] = std::mem::take(&mut self.edit_lists[read]);
                    }
                    write += 1;
                }
            }
            self.active_bricks.truncate(write);
            self.edit_lists.truncate(write);
        }

        // Phase 2: Process newly-exposed shell bricks
        let mut hit_buf = Vec::new();

        for wz in new_wb_min[2]..=new_wb_max[2] {
            let in_old_z = wz >= old_wb_min[2] && wz <= old_wb_max[2];
            for wy in new_wb_min[1]..=new_wb_max[1] {
                let in_old_yz = in_old_z && wy >= old_wb_min[1] && wy <= old_wb_max[1];
                for wx in new_wb_min[0]..=new_wb_max[0] {
                    if in_old_yz && wx >= old_wb_min[0] && wx <= old_wb_max[0] {
                        continue;
                    }

                    let grid_idx = self.world_to_grid(wx, wy, wz);
                    let brick_min = glam::Vec3::new(
                        wx as f32 * brick_world_size,
                        wy as f32 * brick_world_size,
                        wz as f32 * brick_world_size,
                    );
                    let brick_max = brick_min + glam::Vec3::splat(brick_world_size);

                    let elist = Self::build_edit_list(
                        brick_min,
                        brick_max,
                        cached_bounds,
                        bvh,
                        &mut hit_buf,
                    );
                    let terrain_active = terrain.map_or(false, |cfg| {
                        Self::brick_intersects_terrain(brick_min, brick_max, cfg)
                    });

                    if !elist.is_empty() || terrain_active {
                        let world_coord = [wx, wy, wz];
                        let has_slot = self.brick_index[grid_idx] != EMPTY_BRICK;
                        let existing_coord = self.brick_world_coords[grid_idx];
                        let is_same = has_slot && existing_coord == world_coord;

                        if has_slot && !is_same {
                            self.atlas_free_list.push(self.brick_index[grid_idx]);
                        }
                        if !is_same {
                            let slot = self.alloc_atlas_slot();
                            self.brick_index[grid_idx] = slot;
                            self.brick_world_coords[grid_idx] = world_coord;
                        }
                        self.stored_edit_lists[grid_idx] = elist.clone();
                        self.dirty_bricks.push(pack_world_brick_coord(wx, wy, wz));
                        self.dirty_edit_lists.push(elist.clone());
                        self.active_bricks.push(grid_idx as u32);
                        self.edit_lists.push(elist);
                    } else if self.brick_index[grid_idx] != EMPTY_BRICK
                        && self.brick_world_coords[grid_idx] == [wx, wy, wz]
                    {
                        self.atlas_free_list.push(self.brick_index[grid_idx]);
                        self.brick_index[grid_idx] = EMPTY_BRICK;
                        self.brick_world_coords[grid_idx] = [i32::MAX; 3];
                        self.stored_edit_lists[grid_idx].clear();
                    }
                }
            }
        }

        self.dirty_bricks.len() as u32
    }

    /// Mark bricks overlapping an edit's bounding sphere as dirty.
    pub fn mark_edit_dirty(
        &mut self,
        edit: &SdfEdit,
        _all_edits: &[SdfEdit],
        volume_min: [f32; 3],
        volume_max: [f32; 3],
        terrain: Option<&TerrainConfig>,
        bvh: Option<&EditBvh>,
        cached_bounds: &[(glam::Vec3, f32)],
    ) -> bool {
        let (center, radius) = Self::edit_bounding_sphere(edit);
        let bgd = self.brick_grid_dim;
        let voxel_size = (volume_max[0] - volume_min[0]) / (bgd * self.brick_size) as f32;
        let brick_world_size = voxel_size * self.brick_size as f32;

        let wb_min = [
            ((center.x - radius) / brick_world_size).floor() as i32,
            ((center.y - radius) / brick_world_size).floor() as i32,
            ((center.z - radius) / brick_world_size).floor() as i32,
        ];
        let wb_max = [
            ((center.x + radius) / brick_world_size).ceil() as i32,
            ((center.y + radius) / brick_world_size).ceil() as i32,
            ((center.z + radius) / brick_world_size).ceil() as i32,
        ];

        let vol_wb_min = [
            (volume_min[0] / brick_world_size).floor() as i32,
            (volume_min[1] / brick_world_size).floor() as i32,
            (volume_min[2] / brick_world_size).floor() as i32,
        ];
        let vol_wb_max = [
            vol_wb_min[0] + bgd as i32 - 1,
            vol_wb_min[1] + bgd as i32 - 1,
            vol_wb_min[2] + bgd as i32 - 1,
        ];

        self.dirty_bricks.clear();
        self.dirty_edit_lists.clear();

        let mut changed = false;
        let mut hit_buf = Vec::new();

        for wz in wb_min[2]..=wb_max[2] {
            for wy in wb_min[1]..=wb_max[1] {
                for wx in wb_min[0]..=wb_max[0] {
                    if wx < vol_wb_min[0]
                        || wx > vol_wb_max[0]
                        || wy < vol_wb_min[1]
                        || wy > vol_wb_max[1]
                        || wz < vol_wb_min[2]
                        || wz > vol_wb_max[2]
                    {
                        continue;
                    }

                    let grid_idx = self.world_to_grid(wx, wy, wz);
                    let brick_min = glam::Vec3::new(
                        wx as f32 * brick_world_size,
                        wy as f32 * brick_world_size,
                        wz as f32 * brick_world_size,
                    );
                    let brick_max = brick_min + glam::Vec3::splat(brick_world_size);

                    let elist = Self::build_edit_list(
                        brick_min,
                        brick_max,
                        cached_bounds,
                        bvh,
                        &mut hit_buf,
                    );
                    let terrain_active = terrain.map_or(false, |cfg| {
                        Self::brick_intersects_terrain(brick_min, brick_max, cfg)
                    });

                    if !elist.is_empty() || terrain_active {
                        let world_coord = [wx, wy, wz];
                        let was_empty = self.brick_index[grid_idx] == EMPTY_BRICK;
                        if was_empty {
                            let slot = self.alloc_atlas_slot();
                            self.brick_index[grid_idx] = slot;
                            self.brick_world_coords[grid_idx] = world_coord;
                            self.active_bricks.push(grid_idx as u32);
                            self.edit_lists.push(elist.clone());
                        }
                        self.stored_edit_lists[grid_idx] = elist.clone();
                        self.dirty_bricks.push(pack_world_brick_coord(wx, wy, wz));
                        self.dirty_edit_lists.push(elist);
                        changed = true;

                        // Update the edit list in active_bricks for this grid slot
                        if !was_empty {
                            if let Some(pos) = self
                                .active_bricks
                                .iter()
                                .position(|&g| g == grid_idx as u32)
                            {
                                if pos < self.edit_lists.len() {
                                    self.edit_lists[pos] = self.stored_edit_lists[grid_idx].clone();
                                }
                            }
                        }
                    }
                }
            }
        }

        changed
    }

    // ── Upload ──────────────────────────────────────────────────────────────

    /// Upload dirty-only bricks for compute dispatch.
    pub fn upload_dirty(&self, queue: &wgpu::Queue) {
        if let Some(buf) = &self.brick_index_buffer {
            queue.write_buffer(buf, 0, bytemuck::cast_slice(&self.brick_index));
        }
        if let Some(buf) = &self.active_bricks_buffer {
            if !self.dirty_bricks.is_empty() {
                queue.write_buffer(buf, 0, bytemuck::cast_slice(&self.dirty_bricks));
            }
        }
        self.upload_packed_edit_lists(&self.dirty_edit_lists, queue);
    }

    fn upload_packed_edit_lists(&self, lists: &[EditList], queue: &wgpu::Queue) {
        if lists.is_empty() {
            return;
        }

        let mut offsets = Vec::with_capacity(lists.len());
        let mut data = Vec::new();

        for elist in lists {
            offsets.push(data.len() as u32);
            data.push(elist.len() as u32);
            for &idx in elist {
                data.push(idx as u32);
            }
        }

        if let Some(buf) = &self.edit_list_offsets_buffer {
            queue.write_buffer(buf, 0, bytemuck::cast_slice(&offsets));
        }
        if let Some(buf) = &self.edit_list_data_buffer {
            queue.write_buffer(buf, 0, bytemuck::cast_slice(&data));
        }
    }

    // ── Accessors ───────────────────────────────────────────────────────────

    pub fn active_count(&self) -> u32 {
        self.active_bricks.len() as u32
    }
    pub fn brick_size(&self) -> u32 {
        self.brick_size
    }
    pub fn brick_grid_dim(&self) -> u32 {
        self.brick_grid_dim
    }
    pub fn atlas_bricks_per_axis(&self) -> u32 {
        self.atlas_bricks_per_axis
    }
    pub fn atlas_dim(&self) -> u32 {
        self.atlas_dim
    }
    pub fn brick_index_slice(&self) -> &[u32] {
        &self.brick_index
    }
    pub fn dirty_count(&self) -> u32 {
        self.dirty_bricks.len() as u32
    }
    pub fn clear_dirty(&mut self) {
        self.dirty_bricks.clear();
        self.dirty_edit_lists.clear();
    }

    // ── Private helpers ─────────────────────────────────────────────────────

    fn alloc_atlas_slot(&mut self) -> u32 {
        if let Some(slot) = self.atlas_free_list.pop() {
            slot
        } else {
            let slot = self.atlas_next_slot;
            let max_slots = self.atlas_bricks_per_axis.pow(3);
            if slot >= max_slots {
                log::warn!(
                    "Brick atlas overflow: {} slots exhausted, recycling last slot.",
                    max_slots
                );
                return max_slots - 1;
            }
            self.atlas_next_slot += 1;
            slot
        }
    }

    fn build_edit_list(
        brick_min: glam::Vec3,
        brick_max: glam::Vec3,
        bounds: &[(glam::Vec3, f32)],
        bvh: Option<&EditBvh>,
        hit_buf: &mut Vec<usize>,
    ) -> EditList {
        let mut elist = EditList::new();
        if let Some(bvh) = bvh {
            let brick_aabb = Aabb::new(brick_min, brick_max);
            bvh.query_region(&brick_aabb, hit_buf);
            for &edit_idx in hit_buf.iter() {
                let (center, radius) = bounds[edit_idx];
                if Self::sphere_aabb_intersect(center, radius, brick_min, brick_max) {
                    elist.push(edit_idx as u16);
                }
            }
        } else {
            for (edit_idx, &(center, radius)) in bounds.iter().enumerate() {
                if Self::sphere_aabb_intersect(center, radius, brick_min, brick_max) {
                    elist.push(edit_idx as u16);
                }
            }
        }
        elist
    }

    pub fn edit_bounding_sphere(edit: &SdfEdit) -> (glam::Vec3, f32) {
        let center = glam::Vec3::new(
            edit.transform.w_axis.x,
            edit.transform.w_axis.y,
            edit.transform.w_axis.z,
        );

        let local_extent = match edit.shape {
            SdfShapeType::Sphere => edit.params.param0,
            SdfShapeType::Cube => {
                let hx = edit.params.param0;
                let hy = edit.params.param1;
                let hz = edit.params.param2;
                (hx * hx + hy * hy + hz * hz).sqrt()
            }
            SdfShapeType::Capsule => edit.params.param0 + edit.params.param1,
            SdfShapeType::Torus => edit.params.param0 + edit.params.param1,
            SdfShapeType::Cylinder => {
                let r = edit.params.param0;
                let h = edit.params.param1;
                (r * r + h * h).sqrt()
            }
        };

        let sx = edit.transform.x_axis.truncate().length();
        let sy = edit.transform.y_axis.truncate().length();
        let sz = edit.transform.z_axis.truncate().length();
        let max_scale = sx.max(sy).max(sz);

        let world_radius = local_extent * max_scale + edit.blend_radius;
        (center, world_radius)
    }

    fn sphere_aabb_intersect(
        center: glam::Vec3,
        radius: f32,
        aabb_min: glam::Vec3,
        aabb_max: glam::Vec3,
    ) -> bool {
        let closest = center.clamp(aabb_min, aabb_max);
        let dist_sq = (closest - center).length_squared();
        dist_sq <= radius * radius
    }

    fn brick_intersects_terrain(
        brick_min: glam::Vec3,
        brick_max: glam::Vec3,
        config: &TerrainConfig,
    ) -> bool {
        let margin = (brick_max - brick_min).length() * 0.5;
        let (min_h, max_h) = crate::noise::terrain_height_range(brick_min, brick_max, config);
        let band_min = min_h - margin;
        let band_max = max_h + margin;
        brick_max.y >= band_min && brick_min.y <= band_max
    }
}
