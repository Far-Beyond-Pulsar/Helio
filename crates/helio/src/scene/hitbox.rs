//! Water hitbox management methods for Scene.
//!
//! Water hitboxes are per-frame AABB descriptors that displace the water
//! heightfield simulation, producing realistic wave effects when objects
//! enter or leave the water surface.

use crate::handles::WaterHitboxId;
use crate::scene::actor::WaterHitboxDescriptor;
use crate::arena::DenseRemove;
use crate::scene::errors::{invalid, Result};
use crate::scene::types::WaterHitboxRecord;
use crate::scene::Scene;
use bytemuck;
use libhelio::GpuWaterHitbox;

impl Scene {
    /// Insert a water hitbox into the scene.
    ///
    /// A hitbox records where an object was (old bounds) and is (new bounds)
    /// so the simulation can compute realistic displacement waves.
    ///
    /// # Performance
    /// - CPU cost: O(1) insertion into dense arena
    /// - GPU cost: Deferred — uploaded once per frame in `renderer_impl.rs`
    pub fn insert_water_hitbox(&mut self, desc: WaterHitboxDescriptor) -> Result<WaterHitboxId> {
        let gpu = desc.to_gpu();
        let record = WaterHitboxRecord { gpu };
        let (id, index) = self.water_hitboxes.insert(record);
        self.water_hitboxes_dirty = true;
        self.water_hitboxes_dirty_range = Some((index, index + 1));
        Ok(id)
    }

    /// Remove a water hitbox from the scene.
    pub fn remove_water_hitbox(&mut self, id: WaterHitboxId) -> Result<()> {
        let DenseRemove { dense_index, moved, .. } = self
            .water_hitboxes
            .remove(id)
            .ok_or_else(|| invalid("water hitbox"))?;
        self.water_hitboxes_dirty = true;
        if let Some((_, moved_index)) = moved {
            let start = dense_index.min(moved_index);
            let end = dense_index.max(moved_index) + 1;
            self.water_hitboxes_dirty_range = Some((start, end));
        } else {
            self.water_hitboxes_dirty_range = Some((dense_index, dense_index + 1));
        }
        Ok(())
    }

    /// Update an existing water hitbox's bounds.
    ///
    /// Call each frame to advance `old_min/max` → previous `new_min/max`, and
    /// set `new_min/max` to the object's current world-space AABB.
    pub fn update_water_hitbox(
        &mut self,
        id: WaterHitboxId,
        desc: WaterHitboxDescriptor,
    ) -> Result<()> {
        let (index, record) = self
            .water_hitboxes
            .get_mut_with_index(id)
            .ok_or_else(|| invalid("water hitbox"))?;
        record.gpu = desc.to_gpu();
        self.water_hitboxes_dirty = true;
        match self.water_hitboxes_dirty_range {
            Some((start, end)) => {
                self.water_hitboxes_dirty_range = Some((start.min(index), (end.max(index + 1))));
            }
            None => self.water_hitboxes_dirty_range = Some((index, index + 1)),
        }
        Ok(())
    }

    /// Collect GPU-side data for all hitboxes (called by renderer each frame).
    pub fn get_water_hitboxes_gpu(&self) -> Vec<GpuWaterHitbox> {
        (0..self.water_hitboxes.dense_len())
            .filter_map(|i| self.water_hitboxes.get_dense(i))
            .map(|record| record.gpu)
            .collect()
    }

    /// Get a zero-allocation view of the GPU hitbox array.
    ///
    /// This avoids constructing a temporary `Vec` each frame when hitboxes
    /// are uploaded to the GPU from the renderer.
    pub fn get_water_hitboxes_gpu_slice(&self) -> &[GpuWaterHitbox] {
        bytemuck::cast_slice(self.water_hitboxes.dense.as_slice())
    }

    /// Returns the current dirty hitbox upload range, if any.
    pub fn water_hitboxes_dirty_range(&self) -> Option<(usize, usize)> {
        self.water_hitboxes_dirty_range
    }

    /// Consume the current dirty hitbox range and clear it.
    pub(crate) fn consume_water_hitboxes_dirty_range(&mut self) -> Option<(usize, usize)> {
        self.water_hitboxes_dirty_range.take()
    }

    /// Number of active water hitboxes.
    pub fn water_hitboxes_count(&self) -> u32 {
        self.water_hitboxes.dense_len() as u32
    }

    /// Whether hitboxes have changed since last GPU upload.
    pub fn water_hitboxes_dirty(&self) -> bool {
        self.water_hitboxes_dirty
    }

    /// Clear the hitboxes dirty flag (called by renderer after upload).
    pub(crate) fn clear_water_hitboxes_dirty(&mut self) {
        self.water_hitboxes_dirty = false;
    }
}
