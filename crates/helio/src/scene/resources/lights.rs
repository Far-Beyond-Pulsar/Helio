//! Light resource management for the scene.
//!
//! Lights are stored on the CPU archetype ECS (`Scene::world`, see
//! `docs/scenedb_object_storage_migration.md`) and uploaded to GPU storage
//! buffers. Unlike other resources, lights have no reference counting (they
//! exist independently of objects).
//!
//! `LightRecord::gpu_index` is the **sole** authority for a light's current
//! slot in `gpu_scene.lights` — see the doc on that field and on
//! [`Scene::insert_light_with_movability`] for why: an earlier version of
//! this file additionally trusted the CPU dense-array position as a proxy
//! for the GPU slot (`debug_assert_eq!(gpu_index, dense_index)` at insert
//! time), which only held until the first `flush()` filtered any static
//! light out of the buffer — after that, `remove_light`'s swap-remove used
//! the wrong index for any scene with static lights, and `update_light`
//! could write a static light's data into an unrelated (now-reused) GPU
//! slot. Fixed here by never letting a static light's `gpu_index` claim a
//! buffer slot it doesn't actually hold.

use helio_core::GpuLight;

use crate::handles::LightId;

use super::super::errors::{invalid, Result};
use super::super::types::LightRecord;

impl super::super::Scene {
    /// Insert a light into the scene.
    ///
    /// Adds the light to the CPU ECS. Movable lights are also pushed to the
    /// GPU light storage buffer immediately; static/stationary lights are
    /// not — see [`Self::insert_light_with_movability`].
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
    /// - CPU cost: O(1) insertion into the ECS
    /// - GPU cost: O(1) push to GPU storage buffer (movable lights only)
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
    ///
    /// Static/stationary lights are baked and `flush()`'s lights-filter step
    /// excludes them from `gpu_scene.lights` on every frame regardless (real-time
    /// lighting only shades movable lights) — so a static light is never given a
    /// real GPU buffer slot here; `gpu_index` is `u32::MAX` ("not a buffer
    /// resident") until/unless its movability changes to something that can move.
    pub fn insert_light_with_movability(
        &mut self,
        light: GpuLight,
        movability: Option<libhelio::Movability>,
        user_tag: u64,
    ) -> LightId {
        // Default lights to Movable (most common case for real-time lighting).
        // Static lights are opt-in for baking scenarios.
        let movability = movability.unwrap_or(libhelio::Movability::Movable);
        let gpu_index = if movability.can_move() {
            self.gpu_scene.lights.push(light) as u32
        } else {
            u32::MAX
        };

        let entity = self.world.spawn();
        self.world.insert(
            entity,
            LightRecord {
                gpu: light,
                movability,
                user_tag,
                gpu_index,
            },
        );
        let id = LightId::from_entity(entity);

        // Index by application tag so the owner can find this light again
        // without keeping its own id map. Tag 0 means "untagged".
        if user_tag != 0 {
            self.lights_by_tag.insert(user_tag, id);
        }

        // Invalidate any previous bake if this is a static/stationary light
        if !movability.can_move() {
            self.bake_invalidated = true;
        }

        id
    }

    /// Update a light's parameters.
    ///
    /// Modifies the light's GPU parameters. Writes through to the GPU storage
    /// buffer only for movable lights — a static light has no buffer slot to
    /// write (see [`Self::insert_light_with_movability`]); its CPU record is
    /// still updated, so the new values take effect if it's later baked or
    /// its movability changes.
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
    /// - GPU cost: O(1) storage buffer slot write (movable lights only)
    ///
    /// # Example
    /// ```ignore
    /// // Animate light intensity
    /// let mut light = scene.get_light(light_id)?;
    /// light.intensity = 200.0; // Brighten
    /// scene.update_light(light_id, light)?;
    /// ```
    pub fn update_light(&mut self, id: LightId, light: GpuLight) -> Result<()> {
        let Some(record) = self.world.get_mut::<LightRecord>(id.entity()) else {
            return Err(invalid("light"));
        };

        if !record.movability.can_move() {
            // Enforce movability: Static lights cannot have position/direction updated.
            let position_changed = record.gpu.position_range != light.position_range;
            let direction_changed = record.gpu.direction_outer != light.direction_outer;
            if position_changed || direction_changed {
                log::warn!(
                    "Attempted to update position/direction on Static light {:?}. Set movability to Movable to allow updates.",
                    id
                );
                return Ok(()); // No-op instead of error
            }
            // No GPU slot to write (see module doc) — CPU record only.
            record.gpu = light;
            return Ok(());
        }

        record.gpu = light;

        // Increment generation counter for movable lights (for shadow cache invalidation).
        self.movable_lights_generation += 1;
        self.gpu_scene.movable_lights_generation = self.movable_lights_generation;

        let gpu_index = record.gpu_index as usize;
        let updated = self.gpu_scene.lights.update(gpu_index, light);
        debug_assert!(updated, "GPU light index {} out of bounds (len {}) — flush may have mis-indexed this light", gpu_index, self.gpu_scene.lights.len());
        Ok(())
    }

    /// Remove a light from the scene.
    ///
    /// Removes the light from the CPU ECS. If it was movable (and therefore
    /// held a real slot in `gpu_scene.lights`), swap-removes it from the GPU
    /// storage buffer too and patches whichever other light got swapped into
    /// the vacated slot so its `gpu_index` stays accurate before the next
    /// `flush()`. Static lights hold no GPU slot (see
    /// [`Self::insert_light_with_movability`]) — nothing to swap-remove.
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
    /// - CPU cost: O(1) removal from the ECS
    /// - GPU cost: O(1) swap-remove, plus an O(N) scan over lights to patch
    ///   the swapped-in record's `gpu_index` (movable lights only — same
    ///   O(N) class `flush()` already pays every frame regardless)
    ///
    /// # Example
    /// ```ignore
    /// scene.remove_light(light_id)?;
    /// ```
    pub fn remove_light(&mut self, id: LightId) -> Result<()> {
        let (user_tag, movability, gpu_index) = {
            let r = self
                .world
                .get::<LightRecord>(id.entity())
                .ok_or_else(|| invalid("light"))?;
            (r.user_tag, r.movability, r.gpu_index)
        };

        if !self.world.despawn(id.entity()) {
            return Err(invalid("light"));
        }

        // Drop the tag index entry, but only if it still points at *this*
        // light — a newer light may have since claimed the same tag.
        if user_tag != 0 && self.lights_by_tag.get(&user_tag) == Some(&id) {
            self.lights_by_tag.remove(&user_tag);
        }

        if movability.can_move() {
            let gpu_removed = self.gpu_scene.lights.swap_remove(gpu_index as usize);
            debug_assert!(gpu_removed.is_some());

            // `swap_remove` moved whatever was at the old last index into
            // `gpu_index` (unless we removed the last element, in which case
            // nothing needs patching and this loop simply finds no match).
            // `GrowableBuffer::swap_remove` doesn't report identity, so find
            // the CPU record that still claims the vacated slot.
            let old_last_index = self.gpu_scene.lights.len() as u32; // already shrank by one
            for (_, other) in self.world.query::<&mut LightRecord>() {
                if other.gpu_index == old_last_index {
                    other.gpu_index = gpu_index;
                    break;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::scene::Scene;
    use bytemuck::Zeroable;
    use helio_core::GpuLight;
    use libhelio::Movability;

    fn create_test_device() -> (std::sync::Arc<wgpu::Device>, std::sync::Arc<wgpu::Queue>) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::from_env().unwrap_or(wgpu::Backends::PRIMARY),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("No adapter found");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        }))
        .expect("Failed to create device");
        (std::sync::Arc::new(device), std::sync::Arc::new(queue))
    }

    fn light_at(x: f32) -> GpuLight {
        GpuLight {
            position_range: [x, 0.0, 0.0, 10.0],
            shadow_index: u32::MAX,
            ..GpuLight::zeroed()
        }
    }

    /// Regression test for the gpu_index bug this migration fixed: a static
    /// light's slot in `gpu_scene.lights` used to go stale the moment
    /// `flush()` filtered it out (real slots are movable-only), so calling
    /// `update_light`/`remove_light` on it afterward could write into or
    /// swap-remove a *different* light's GPU slot. `gpu_index` is now the
    /// sole authority — `u32::MAX` means "not a resident", checked before
    /// any GPU write.
    #[test]
    fn static_lights_never_hold_a_gpu_slot_and_updating_one_does_not_corrupt_others() {
        let (device, queue) = create_test_device();
        let mut scene = Scene::new(device, queue);

        let a = scene.insert_light_with_movability(light_at(1.0), Some(Movability::Movable), 0);
        let b = scene.insert_light_with_movability(light_at(2.0), Some(Movability::Static), 0);
        let c = scene.insert_light_with_movability(light_at(3.0), Some(Movability::Movable), 0);

        scene.flush();
        // Only the two movable lights (A, C) actually hold a GPU slot.
        assert_eq!(scene.gpu_scene().resources().light_count, 2);

        // This used to write into whatever stale GPU slot B's gpu_index
        // still pointed at (A's or C's) instead of doing nothing.
        scene
            .update_light(b, light_at(99.0))
            .expect("update_light on a static light must not error or corrupt another light");
        assert_eq!(scene.gpu_scene().resources().light_count, 2);
        // A and C's data must be untouched by B's update.
        assert_eq!(scene.get_light(a).unwrap().position_range[0], 1.0);
        assert_eq!(scene.get_light(c).unwrap().position_range[0], 3.0);

        // Removing A swap-removes it from gpu_scene.lights and must patch
        // whichever light (C) got swapped into A's freed slot.
        scene.remove_light(a).expect("remove_light");
        assert_eq!(scene.gpu_scene().resources().light_count, 1);

        // If C's gpu_index weren't patched correctly this would either trip
        // update_light's debug_assert (out-of-bounds slot) or silently write
        // the wrong slot; asserting the round-trip catches both.
        scene
            .update_light(c, light_at(30.0))
            .expect("update_light on the swapped-in light");
        assert_eq!(scene.get_light(c).unwrap().position_range[0], 30.0);
    }
}
