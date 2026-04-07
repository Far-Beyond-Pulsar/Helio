//! Water volume management methods for Scene.
//!
//! This module provides methods for inserting, updating, and removing water volumes
//! from the scene, as well as querying water volume data for rendering.

use crate::handles::WaterVolumeId;
use crate::scene::actor::WaterVolumeDescriptor;
use crate::arena::DenseRemove;
use crate::scene::errors::{invalid, Result};
use crate::scene::types::WaterVolumeRecord;
use crate::scene::Scene;
use bytemuck;
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
        let (id, index) = self.water_volumes.insert(record);
        self.water_volumes_dirty = true;
        self.water_volumes_dirty_range = Some((index, index + 1));
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
        let DenseRemove { dense_index, moved, .. } = self
            .water_volumes
            .remove(id)
            .ok_or_else(|| invalid("water volume"))?;
        self.water_volumes_dirty = true;
        if let Some((_, moved_index)) = moved {
            let start = dense_index.min(moved_index);
            let end = dense_index.max(moved_index) + 1;
            self.water_volumes_dirty_range = Some((start, end));
        } else {
            self.water_volumes_dirty_range = Some((dense_index, dense_index + 1));
        }
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
        let (index, record) = self
            .water_volumes
            .get_mut_with_index(id)
            .ok_or_else(|| invalid("water volume"))?;
        record.gpu = desc.to_gpu();
        self.water_volumes_dirty = true;
        match self.water_volumes_dirty_range {
            Some((start, end)) => {
                self.water_volumes_dirty_range = Some((start.min(index), end.max(index + 1)));
            }
            None => self.water_volumes_dirty_range = Some((index, index + 1)),
        }
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

    /// Get a zero-allocation view of the GPU water volume array.
    pub fn get_water_volumes_gpu_slice(&self) -> &[GpuWaterVolume] {
        bytemuck::cast_slice(self.water_volumes.dense.as_slice())
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

    /// Returns the current dirty water volume upload range, if any.
    pub fn water_volumes_dirty_range(&self) -> Option<(usize, usize)> {
        self.water_volumes_dirty_range
    }

    /// Consume the current dirty water volume range and clear it.
    pub(crate) fn consume_water_volumes_dirty_range(&mut self) -> Option<(usize, usize)> {
        self.water_volumes_dirty_range.take()
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
