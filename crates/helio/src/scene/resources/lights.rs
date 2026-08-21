//! Light resource management for the scene.
//!
//! Lights are stored in a dense arena and uploaded to GPU storage buffers.
//! Unlike other resources, lights have no reference counting (they exist
//! independently of objects).

use helio_core::GpuLight;

use crate::arena::DenseArena;
use crate::handles::LightId;

use super::super::errors::{invalid, Result};
use super::super::types::{LightRecord, LightRenderInput};

impl super::super::Scene {
    /// Wholesale-replaces the scene's light list from a transient, SceneDB-
    /// owned snapshot -- the light equivalent of [`Scene::
    /// rebuild_static_mesh_instances`] (Pulsar-Native#561: `LightComponent`
    /// fully normalized onto SceneDB's `#[gpu]` mirror). No per-entity
    /// `insert_light`/`update_light`/`remove_light`/`lights_by_tag` lookup
    /// bookkeeping happens here or needs to happen on the caller's side --
    /// every light present in `inputs` this call IS the complete real-time
    /// light set for this frame, full stop. A light absent from `inputs`
    /// this call (its `LightComponent` was disabled, removed, or its owning
    /// object despawned) simply doesn't get re-inserted -- no explicit
    /// teardown signal needed, the same "absence is removal" property
    /// `rebuild_static_mesh_instances` already has for objects.
    ///
    /// Every light inserted this way is [`libhelio::Movability::Movable`] --
    /// `LightComponent` has no movability/static-baking concept of its own,
    /// and this call's whole point is BEING the per-frame rebuild, so a
    /// "baked once, never touched again" light has no way to reach it in the
    /// first place. [`Scene::insert_light_with_movability`] (unchanged) is
    /// still how a non-SceneDB-driven caller (an editor tool, a baking pass)
    /// inserts a genuinely static light into a `Scene` that ISN'T also
    /// driven by this call.
    ///
    /// `flush()`'s shadow-atlas importance-scoring/movability-filter/
    /// per-caster dirty-hash logic needs no changes at all: it already reads
    /// from `self.lights` (this call's target) regardless of how that arena
    /// got populated -- `flush()` doesn't know or care that this call
    /// replaced it wholesale instead of individual `insert_light`/
    /// `update_light`/`remove_light` calls accumulating it over time.
    ///
    /// # Warning
    ///
    /// Calling this on a `Scene` that ALSO has independently-inserted
    /// lights (via [`Scene::insert_light`]/[`Scene::
    /// insert_light_with_movability`] from some other caller) wholesale-
    /// replaces those too -- same caveat `rebuild_static_mesh_instances`
    /// already carries for objects. Don't mix the two insertion models on
    /// one `Scene`.
    pub fn rebuild_light_instances(&mut self, inputs: &[LightRenderInput]) {
        let mut lights: DenseArena<LightRecord, LightId> = DenseArena::new();
        let mut lights_by_tag = std::collections::HashMap::new();
        for input in inputs {
            let (handle, _dense_index) = lights.insert(LightRecord {
                gpu: input.light,
                movability: libhelio::Movability::Movable,
                user_tag: input.user_tag,
                // Recomputed correctly by flush()'s movability-filter pass
                // (every record here is Movable, so it always survives that
                // filter) -- an initial value here is never actually read
                // before flush() overwrites it.
                gpu_index: 0,
            });
            if input.user_tag != 0 {
                lights_by_tag.insert(input.user_tag, handle);
            }
        }
        self.lights = lights;
        self.lights_by_tag = lights_by_tag;
        self.lights_dirty = true;
    }
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

#[cfg(test)]
mod tests {
    use super::super::super::Scene;
    use super::LightRenderInput;
    use helio_core::GpuLight;

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

    fn light(color: [f32; 4]) -> GpuLight {
        GpuLight { color_intensity: color, ..Default::default() }
    }

    #[test]
    fn rebuild_populates_lights_readable_by_iter_and_tag() {
        let (device, queue) = create_test_device();
        let mut scene = Scene::new(device, queue);

        scene.rebuild_light_instances(&[
            LightRenderInput { light: light([1.0, 0.0, 0.0, 10.0]), user_tag: 111 },
            LightRenderInput { light: light([0.0, 1.0, 0.0, 20.0]), user_tag: 222 },
        ]);

        let mut seen: Vec<(u64, [f32; 4])> = scene.iter_lights().map(|(_, l, tag)| (tag, l.color_intensity)).collect();
        seen.sort_by_key(|&(tag, _)| tag);
        assert_eq!(seen, vec![(111, [1.0, 0.0, 0.0, 10.0]), (222, [0.0, 1.0, 0.0, 20.0])]);

        let id = scene.light_by_tag(222).expect("tag must resolve via light_by_tag");
        assert_eq!(scene.get_light(id).unwrap().color_intensity, [0.0, 1.0, 0.0, 20.0]);
    }

    #[test]
    fn a_light_absent_from_the_next_rebuild_is_gone_no_explicit_removal_needed() {
        let (device, queue) = create_test_device();
        let mut scene = Scene::new(device, queue);

        scene.rebuild_light_instances(&[
            LightRenderInput { light: light([1.0, 0.0, 0.0, 10.0]), user_tag: 111 },
            LightRenderInput { light: light([0.0, 1.0, 0.0, 20.0]), user_tag: 222 },
        ]);
        assert_eq!(scene.iter_lights().count(), 2);

        // Tag 111's light disappears (disabled/removed on the SceneDB side) --
        // the next rebuild simply doesn't include it, same as a despawned
        // entity vanishing from a World::query result.
        scene.rebuild_light_instances(&[LightRenderInput { light: light([0.0, 1.0, 0.0, 20.0]), user_tag: 222 }]);

        assert!(scene.light_by_tag(111).is_none(), "absence from the rebuild must be enough, no explicit remove_light call needed");
        assert!(scene.light_by_tag(222).is_some());
        assert_eq!(scene.iter_lights().count(), 1);
    }

    #[test]
    fn flush_still_runs_shadow_assignment_correctly_against_a_rebuilt_light_list() {
        // The real proof this integrates cleanly with the PRE-EXISTING
        // shadow-atlas importance-scoring logic in flush() -- that logic
        // reads from `self.lights` regardless of whether individual
        // insert_light calls populated it (the old way) or one
        // rebuild_light_instances call did (this test).
        let (device, queue) = create_test_device();
        let mut scene = Scene::new(device, queue);

        // A bright, large-range light -- must win a shadow atlas slot.
        let mut bright = light([1.0, 1.0, 1.0, 1000.0]);
        bright.position_range[3] = 100.0; // range
        bright.shadow_index = 0; // requests shadows (anything != u32::MAX)

        scene.rebuild_light_instances(&[LightRenderInput { light: bright, user_tag: 42 }]);
        scene.flush();

        let id = scene.light_by_tag(42).expect("light must still be present after flush");
        let after = scene.get_light(id).expect("light must be readable after flush");
        assert_ne!(after.shadow_index, u32::MAX, "a lone, shadow-requesting light must win a shadow atlas slot");
    }
}

