//! Light resource management for the scene.
//!
//! Lights are stored in a dense arena and uploaded to GPU storage buffers.
//! Unlike other resources, lights have no reference counting (they exist
//! independently of objects).

use helio_v3::GpuLight;

use crate::handles::LightId;

use super::super::errors::{invalid, Result};
use super::super::types::LightRecord;

impl super::super::Scene {
    /// Insert a light into the scene.
    ///
    /// Adds the light to the dense arena and uploads it to the GPU light storage buffer.
    ///
    /// # Parameters
    /// - `light`: GPU light parameters:
    ///   - Position (for point/spot lights)
    ///   - Direction (for directional/spot lights)
    ///   - Color and intensity
    ///   - Light type (point, directional, spot)
    ///   - Shadow settings (shadow_index, shadow resolution)
    ///
    /// # Returns
    /// A [`LightId`] handle that can be used to update or remove the light.
    ///
    /// # Performance
    /// - CPU cost: O(1) insertion into dense arena
    /// - GPU cost: Pushes light data to GPU storage buffer
    /// - Memory: Lights are stored in a dense GPU storage buffer
    ///
    /// # Shadow Casting Limits
    /// The scene supports up to 42 shadow-casting lights (42 × 6 = 252 shadow atlas layers).
    /// Additional shadow-casting lights will have shadows disabled automatically.
    ///
    /// # Example
    /// ```ignore
    /// let light_id = scene.insert_light(GpuLight {
    ///     position: [0.0, 5.0, 0.0],
    ///     color: [1.0, 1.0, 1.0],
    ///     intensity: 100.0,
    ///     light_type: LightType::Point as u32,
    ///     shadow_index: 0, // Enable shadows (assigned automatically in flush())
    ///     ..Default::default()
    /// });
    /// ```
    pub(in crate::scene) fn insert_light(&mut self, light: GpuLight) -> LightId {
        self.insert_light_with_movability(light, None)
    }

    /// Insert a light into the scene with explicit movability.
    pub(in crate::scene) fn insert_light_with_movability(
        &mut self,
        light: GpuLight,
        movability: Option<libhelio::Movability>,
    ) -> LightId {
        let (id, dense_index) = self.lights.insert(LightRecord {
            gpu: light,
            movability: movability.unwrap_or_default(), // Default to Static
        });
        let pushed = self.gpu_scene.lights.push(light);
        debug_assert_eq!(pushed, dense_index);
        id
    }

    /// Update a light's parameters.
    ///
    /// Modifies the light's GPU parameters and updates the GPU storage buffer.
    ///
    /// # Parameters
    /// - `id`: Light handle
    /// - `light`: New GPU light parameters
    ///
    /// # Errors
    /// - [`SceneError::InvalidHandle`](super::super::SceneError::InvalidHandle) if the light ID is invalid
    ///
    /// # Returns
    /// `Ok(())` if the light was successfully updated.
    ///
    /// # Performance
    /// - CPU cost: O(1)
    /// - GPU cost: Updates light storage buffer slot
    ///
    /// # Example
    /// ```ignore
    /// // Animate light intensity
    /// let mut light = scene.get_light(light_id)?;
    /// light.intensity = 200.0; // Brighten
    /// scene.update_light(light_id, light)?;
    /// ```
    pub fn update_light(&mut self, id: LightId, light: GpuLight) -> Result<()> {
        let Some((dense_index, record)) = self.lights.get_mut_with_index(id) else {
            return Err(invalid("light"));
        };
        // Enforce movability: Static lights cannot have position/direction updated
        if !record.movability.can_move() {
            let old_pos = record.gpu.position_range;
            let new_pos = light.position_range;
            let old_dir = record.gpu.direction_outer;
            let new_dir = light.direction_outer;

            // Check if position or direction changed
            let position_changed = old_pos != new_pos;
            let direction_changed = old_dir != new_dir;

            if position_changed || direction_changed {
                log::warn!(
                    "Attempted to update position/direction on Static light {:?}. Set movability to Movable to allow updates.",
                    id
                );
                return Ok(()); // No-op instead of error
            }
        }
        record.gpu = light;

        // Increment generation counter for movable lights (for shadow cache invalidation)
        // Only increment if the light can actually move
        if record.movability.can_move() {
            self.movable_lights_generation += 1;
            self.gpu_scene.movable_lights_generation = self.movable_lights_generation;
        }

        let updated = self.gpu_scene.lights.update(dense_index, light);
        debug_assert!(updated);
        Ok(())
    }

    /// Remove a light from the scene.
    ///
    /// Removes the light from the dense arena and GPU storage buffer using swap-remove
    /// (the last light is moved to the removed light's slot for O(1) removal).
    ///
    /// # Parameters
    /// - `id`: Light handle
    ///
    /// # Errors
    /// - [`SceneError::InvalidHandle`](super::super::SceneError::InvalidHandle) if the light ID is invalid
    ///
    /// # Returns
    /// `Ok(())` if the light was successfully removed.
    ///
    /// # Performance
    /// - CPU cost: O(1) swap-remove from dense arena
    /// - GPU cost: Swap-removes light from GPU storage buffer
    ///
    /// # Example
    /// ```ignore
    /// scene.remove_light(light_id)?;
    /// ```
    pub fn remove_light(&mut self, id: LightId) -> Result<()> {
        let removed = self.lights.remove(id).ok_or_else(|| invalid("light"))?;
        let gpu_removed = self.gpu_scene.lights.swap_remove(removed.dense_index);
        debug_assert!(gpu_removed.is_some());
        Ok(())
    }
}

