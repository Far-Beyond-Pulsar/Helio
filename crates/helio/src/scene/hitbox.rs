//! Water hitbox management methods for Scene.
//!
//! Water hitboxes are per-frame AABB descriptors that displace the water
//! heightfield simulation, producing realistic wave effects when objects
//! enter or leave the water surface.

use crate::handles::WaterHitboxId;
use crate::scene::actor::WaterHitboxDescriptor;
use crate::scene::errors::{invalid, Result};
use crate::scene::types::WaterHitboxRecord;
use crate::scene::Scene;
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
        let (id, _) = self.water_hitboxes.insert(record);
        self.water_hitboxes_dirty = true;
        Ok(id)
    }

    /// Remove a water hitbox from the scene.
    pub fn remove_water_hitbox(&mut self, id: WaterHitboxId) -> Result<()> {
        self.water_hitboxes
            .remove(id)
            .ok_or_else(|| invalid("water hitbox"))?;
        self.water_hitboxes_dirty = true;
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
        let (_index, record) = self
            .water_hitboxes
            .get_mut_with_index(id)
            .ok_or_else(|| invalid("water hitbox"))?;
        record.gpu = desc.to_gpu();
        self.water_hitboxes_dirty = true;
        Ok(())
    }

    /// Collect GPU-side data for all hitboxes (called by renderer each frame).
    pub fn get_water_hitboxes_gpu(&self) -> Vec<GpuWaterHitbox> {
        (0..self.water_hitboxes.dense_len())
            .filter_map(|i| self.water_hitboxes.get_dense(i))
            .map(|record| record.gpu)
            .collect()
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
