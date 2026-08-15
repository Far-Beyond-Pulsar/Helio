//! Light resource management for the scene.
//!
//! Lights are stored in a dense arena and uploaded to GPU storage buffers.
//! Unlike other resources, lights have no reference counting (they exist
//! independently of objects).

use helio_core::GpuLight;

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
    pub fn insert_light(&mut self, light: GpuLight) -> LightId {
        self.insert_light_with_movability(light, None, 0)
    }

    /// Insert a light into the scene with explicit movability and user tag.
    pub fn insert_light_with_movability(
        &mut self,
        light: GpuLight,
        movability: Option<libhelio::Movability>,
        user_tag: u64,
    ) -> LightId {
        // Default lights to Movable (most common case for real-time lighting).
        // Static lights are opt-in for baking scenarios.
        let movability = movability.unwrap_or(libhelio::Movability::Movable);
        let gpu_index = self.gpu_scene.lights.push(light) as u32;
        let (id, dense_index) = self.lights.insert(LightRecord {
            gpu: light,
            movability,
            user_tag,
            gpu_index,
        });
        debug_assert_eq!(gpu_index as usize, dense_index);

        // Index by application tag so the owner can find this light again
        // without keeping its own id map. Tag 0 means "untagged".
        if user_tag != 0 {
            self.lights_by_tag.insert(user_tag, id);
        }

        // Invalidate any previous bake if this is a static/stationary light
        if !movability.can_move() {
            self.bake_invalidated = true;
        }

        // A newly-inserted light is pushed onto `gpu_scene.lights` as-is (including
        // if it's static), mirroring the full dense arena 1:1 for now. The next
        // flush() must run its movable-only filter/rebuild to correct this, so mark
        // the buffer dirty regardless of movability.
        self.lights_dirty = true;

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
        let Some((_dense_index, record)) = self.lights.get_mut_with_index(id) else {
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

        let gpu_index = record.gpu_index as usize;
        let updated = self.gpu_scene.lights.update(gpu_index, light);
        debug_assert!(updated, "GPU light index {} out of bounds (len {}) — flush may have mis-indexed this light", gpu_index, self.gpu_scene.lights.len());

        // The direct `.update()` above already patches the correct slot in the GPU
        // buffer in place, so this doesn't strictly need a full flush-time rebuild.
        // Still mark dirty (matching the previous unconditional-every-flush behavior
        // for any frame that touched a light) so the movable-only filter/reindex in
        // flush() stays exactly as authoritative as it was before this dirty-gate was
        // introduced -- correctness over the marginal extra win of skipping it here.
        self.lights_dirty = true;
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
        let user_tag = self.lights.get(id).map(|r| r.user_tag).unwrap_or(0);
        let removed = self.lights.remove(id).ok_or_else(|| invalid("light"))?;
        let gpu_removed = self.gpu_scene.lights.swap_remove(removed.dense_index);
        debug_assert!(gpu_removed.is_some());

        // Drop the tag index entry, but only if it still points at *this*
        // light — a newer light may have since claimed the same tag.
        if user_tag != 0 && self.lights_by_tag.get(&user_tag) == Some(&id) {
            self.lights_by_tag.remove(&user_tag);
        }

        // Update gpu_index for the element that was swap-moved into the vacated slot
        if let Some((moved_handle, new_dense_index)) = removed.moved {
            if let Some((_, record)) = self.lights.get_mut_with_index(moved_handle) {
                record.gpu_index = new_dense_index as u32;
            }
        }

        // Removal changes movable-light membership/count, so the next flush() must
        // re-run its filter/reindex pass.
        self.lights_dirty = true;

        Ok(())
    }
}

