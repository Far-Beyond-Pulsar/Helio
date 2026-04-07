//! Water volume management methods for Scene.
//!
//! This module provides methods for inserting, updating, and removing water volumes
//! from the scene, as well as querying water volume data for rendering.

use crate::handles::WaterVolumeId;
use crate::scene::actor::WaterVolumeDescriptor;
use crate::scene::errors::{invalid, Result};
use crate::scene::types::WaterVolumeRecord;
use crate::scene::Scene;
use libhelio::GpuWaterVolume;

impl Scene {
    /// Insert a water volume into the scene.
    ///
    /// # Parameters
    /// - `desc`: Water volume configuration descriptor
    ///
    /// # Returns
    /// A unique handle to the inserted water volume, or an error if insertion fails.
    ///
    /// # Performance
    /// - CPU cost: O(1) insertion into dense arena
    /// - GPU cost: Deferred to next `flush()` call
    ///
    /// # Example
    /// ```ignore
    /// use helio::WaterVolumeDescriptor;
    ///
    /// let ocean = WaterVolumeDescriptor::ocean();
    /// let volume_id = scene.insert_water_volume(ocean)?;
    /// ```
    pub fn insert_water_volume(&mut self, desc: WaterVolumeDescriptor) -> Result<WaterVolumeId> {
        let gpu = desc.to_gpu();
        let record = WaterVolumeRecord { gpu };
        let (id, _) = self.water_volumes.insert(record);
        self.water_volumes_dirty = true;
        Ok(id)
    }

    /// Remove a water volume from the scene.
    ///
    /// # Parameters
    /// - `id`: Handle to the water volume to remove
    ///
    /// # Returns
    /// `Ok(())` if the volume was removed, or an error if the handle is invalid.
    ///
    /// # Performance
    /// - CPU cost: O(1) removal from dense arena (swap-remove)
    /// - GPU cost: Deferred to next `flush()` call
    ///
    /// # Example
    /// ```ignore
    /// scene.remove_water_volume(volume_id)?;
    /// ```
    pub fn remove_water_volume(&mut self, id: WaterVolumeId) -> Result<()> {
        self.water_volumes
            .remove(id)
            .ok_or_else(|| invalid("water volume"))?;
        self.water_volumes_dirty = true;
        Ok(())
    }

    /// Update an existing water volume's parameters.
    ///
    /// # Parameters
    /// - `id`: Handle to the water volume to update
    /// - `desc`: New water volume configuration
    ///
    /// # Returns
    /// `Ok(())` if the volume was updated, or an error if the handle is invalid.
    ///
    /// # Performance
    /// - CPU cost: O(1) lookup and update
    /// - GPU cost: Deferred to next `flush()` call
    ///
    /// # Example
    /// ```ignore
    /// let mut ocean = WaterVolumeDescriptor::ocean();
    /// ocean.wave_amplitude = 1.0; // Increase wave height
    /// scene.update_water_volume(volume_id, ocean)?;
    /// ```
    pub fn update_water_volume(
        &mut self,
        id: WaterVolumeId,
        desc: WaterVolumeDescriptor,
    ) -> Result<()> {
        let (_index, record) = self
            .water_volumes
            .get_mut_with_index(id)
            .ok_or_else(|| invalid("water volume"))?;
        record.gpu = desc.to_gpu();
        self.water_volumes_dirty = true;
        Ok(())
    }

    /// Get GPU-side water volume data for all volumes.
    ///
    /// Returns a vector of GPU water volume descriptors suitable for uploading
    /// to a storage buffer for rendering.
    ///
    /// # Returns
    /// Vector of all water volumes' GPU representations.
    ///
    /// # Performance
    /// - CPU cost: O(N) where N is the number of water volumes
    /// - Allocates a new vector each call
    ///
    /// # Example
    /// ```ignore
    /// let gpu_volumes = scene.get_water_volumes_gpu();
    /// // Upload to GPU storage buffer
    /// ```
    pub fn get_water_volumes_gpu(&self) -> Vec<GpuWaterVolume> {
        (0..self.water_volumes.dense_len())
            .filter_map(|i| self.water_volumes.get_dense(i))
            .map(|record| record.gpu)
            .collect()
    }

    /// Get the number of water volumes in the scene.
    ///
    /// # Returns
    /// The count of active water volumes.
    ///
    /// # Performance
    /// - CPU cost: O(1)
    ///
    /// # Example
    /// ```ignore
    /// if scene.water_volumes_count() > 0 {
    ///     // Enable water rendering passes
    /// }
    /// ```
    pub fn water_volumes_count(&self) -> u32 {
        self.water_volumes.dense_len() as u32
    }

    /// Check if the water volumes have been modified since the last flush.
    ///
    /// # Returns
    /// `true` if water volumes have been added, removed, or updated.
    ///
    /// # Performance
    /// - CPU cost: O(1)
    pub fn water_volumes_dirty(&self) -> bool {
        self.water_volumes_dirty
    }

    /// Clear the water volumes dirty flag.
    ///
    /// This should be called after uploading water volume data to the GPU.
    ///
    /// # Performance
    /// - CPU cost: O(1)
    pub(crate) fn clear_water_volumes_dirty(&mut self) {
        self.water_volumes_dirty = false;
    }

    /// Force-mark water volumes as dirty so their parameters are re-applied to
    /// the render pass on the next frame. Call this after the render graph is
    /// rebuilt (e.g. on resize) to ensure the new `WaterSimPass` receives the
    /// current wind/sim settings from the volume descriptor.
    pub(crate) fn mark_water_volumes_dirty(&mut self) {
        self.water_volumes_dirty = true;
    }
}
