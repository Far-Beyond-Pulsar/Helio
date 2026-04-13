//! Object update operations.
//!
//! Provides methods for updating object transforms, materials, and bounds with
//! O(1) performance in both persistent and optimized modes.

use glam::Mat4;

use crate::handles::{MaterialId, MeshId, ObjectId};

use super::super::errors::{invalid, Result};
use super::super::helpers::{normal_matrix, sphere_to_aabb};
use super::super::types::PickableObject;

impl super::super::Scene {
    /// Update an object's world transform.
    ///
    /// Modifies the object's model matrix and recomputes the normal matrix for correct
    /// normal vector transformation in shaders.
    ///
    /// # Parameters
    /// - `id`: Object handle
    /// - `transform`: New world-space model matrix (column-major)
    ///
    /// # Errors
    /// - [`SceneError::InvalidHandle`](super::super::SceneError::InvalidHandle) if the object ID is invalid
    ///
    /// # Returns
    /// `Ok(())` if the transform was successfully updated.
    ///
    /// # Performance (Both Modes)
    /// - CPU cost: O(1) - updates CPU-side record and GPU buffer slot
    /// - GPU cost: O(1) - writes to single GPU buffer slot via cached slot index
    /// - Memory: No allocations
    ///
    /// # Normal Matrix Computation
    ///
    /// The normal matrix is the inverse-transpose of the model matrix's upper-left 3×3 block.
    /// This ensures correct normal vector transformation when the model matrix includes
    /// non-uniform scaling.
    ///
    /// Computing this on the CPU (once per transform update) is more efficient than
    /// computing it per-vertex in the shader.
    ///
    /// # Mode Behavior
    ///
    /// **Persistent mode:**
    /// - Updates CPU-side record
    /// - Writes directly to GPU slot (if not dirty)
    /// - GPU slot = dense_index (stable)
    ///
    /// **Optimized mode:**
    /// - Updates CPU-side record
    /// - Writes directly to GPU slot (if not dirty)
    /// - GPU slot assigned during last rebuild (stable until next rebuild)
    ///
    /// **Pending rebuild:**
    /// - Updates CPU-side record only
    /// - New transform will be included in next rebuild automatically
    ///
    /// # Example
    /// ```ignore
    /// use glam::{Mat4, Vec3};
    ///
    /// // Translate object
    /// let new_transform = Mat4::from_translation(Vec3::new(10.0, 0.0, 5.0));
    /// scene.update_object_transform(obj_id, new_transform)?;
    ///
    /// // Rotate object (preserves position)
    /// let rotation = Mat4::from_rotation_y(std::f32::consts::PI / 4.0);
    /// let current_transform = scene.get_object_transform(obj_id)?;
    /// scene.update_object_transform(obj_id, rotation * current_transform)?;
    /// ```
    pub fn update_object_transform(&mut self, id: ObjectId, transform: Mat4) -> Result<()> {
        let Some((_, record)) = self.objects.get_mut_with_index(id) else {
            return Err(invalid("object"));
        };
        // Enforce movability: Static objects cannot have transforms updated
        if !record.movability.can_move() {
            log::warn!(
                "Attempted to update transform on Static object {:?}. Set movability to Movable to allow transform updates.",
                id
            );
            return Ok(()); // No-op instead of error
        }
        record.instance.model = transform.to_cols_array();
        record.instance.normal_mat = normal_matrix(transform);

        // Increment generation counter for movable objects (for shadow cache invalidation)
        self.movable_objects_generation += 1;
        self.gpu_scene.movable_objects_generation = self.movable_objects_generation;

        // If the GPU layout is stable (no pending rebuild), update the slot in-place.
        // If a rebuild is pending the new data will be included in it automatically.
        if !self.objects_dirty {
            let slot = record.draw.first_instance as usize;
            self.gpu_scene.instances.update(slot, record.instance);
        }
        Ok(())
    }

    /// Update an object's material reference.
    ///
    /// Changes which material an object uses. Decrements the old material's reference
    /// count and increments the new material's reference count.
    ///
    /// # Parameters
    /// - `id`: Object handle
    /// - `material`: New material handle
    ///
    /// # Errors
    /// - [`SceneError::InvalidHandle`](super::super::SceneError::InvalidHandle) if the object or material ID is invalid
    ///
    /// # Returns
    /// `Ok(())` if the material was successfully updated.
    ///
    /// # Performance
    ///
    /// **Persistent mode:**
    /// - CPU cost: O(1) - updates CPU record and GPU slot
    /// - GPU cost: O(1) - writes to single GPU buffer slot
    ///
    /// **Optimized mode:**
    /// - CPU cost: O(1) + invalidates optimization (marks dirty)
    /// - GPU cost: Deferred to next `flush()` when rebuild occurs
    /// - Trade-off: Material change breaks instancing groups, triggers rebuild
    ///
    /// # Mode Behavior
    ///
    /// **Persistent mode:**
    /// - Updates material reference and slot index
    /// - Writes directly to GPU instance buffer slot
    /// - No rebuild needed
    ///
    /// **Optimized mode:**
    /// - Invalidates optimization (sets `objects_layout_optimized = false`)
    /// - Marks scene dirty for rebuild on next `flush()`
    /// - Material change breaks instancing groups (mesh+material batching)
    ///
    /// # Reference Counting
    ///
    /// - Decrements old material's ref count (may allow removal)
    /// - Increments new material's ref count (prevents removal)
    ///
    /// # Example
    /// ```ignore
    /// // Swap material for glowing effect
    /// let emissive_material = scene.insert_material(GpuMaterial {
    ///     emissive: [1.0, 0.5, 0.0, 1.0], // Orange glow
    ///     ..Default::default()
    /// });
    /// scene.update_object_material(obj_id, emissive_material)?;
    /// ```
    pub fn update_object_material(&mut self, id: ObjectId, material: MaterialId) -> Result<()> {
        let new_slot = {
            let (slot, new_material) = self
                .materials
                .get_mut_with_slot(material)
                .ok_or_else(|| invalid("material"))?;
            new_material.ref_count += 1;
            slot
        };
        let Some((_, record)) = self.objects.get_mut_with_index(id) else {
            return Err(invalid("object"));
        };
        let old_material_id = record.material;
        record.material = material;
        record.instance.material_id = new_slot as u32;
        if let Some((_, old_material)) = self.materials.get_mut_with_slot(old_material_id) {
            old_material.ref_count = old_material.ref_count.saturating_sub(1);
        }

        if self.objects_layout_optimized {
            // Material change breaks instancing groups - invalidate
            self.objects_layout_optimized = false;
            self.objects_dirty = true;
        } else {
            // Persistent mode - update in place
            let slot = record.gpu_slot as usize;
            self.gpu_scene.instances.update(slot, record.instance);
        }

        Ok(())
    }

    /// Update an object's bounding sphere.
    ///
    /// Changes the object's bounding volume used for GPU frustum culling.
    /// The sphere is converted to an AABB for GPU-side culling tests.
    ///
    /// # Parameters
    /// - `id`: Object handle
    /// - `bounds`: New bounding sphere `[center.x, center.y, center.z, radius]`
    ///
    /// # Errors
    /// - [`SceneError::InvalidHandle`](super::super::SceneError::InvalidHandle) if the object ID is invalid
    ///
    /// # Returns
    /// `Ok(())` if the bounds were successfully updated.
    ///
    /// # Performance (Both Modes)
    /// - CPU cost: O(1) - updates CPU record and GPU buffer slots
    /// - GPU cost: O(1) - writes to instance and AABB buffer slots
    /// - Memory: No allocations
    ///
    /// # Important
    ///
    /// The bounding sphere must accurately enclose the mesh after transformation,
    /// or the object will be incorrectly culled (disappear when it should be visible).
    ///
    /// # Mode Behavior
    ///
    /// Bounds updates **do not** invalidate optimization because they don't affect
    /// instancing groups (mesh+material batching). The update is applied in-place
    /// in both modes.
    ///
    /// # Example
    /// ```ignore
    /// // Expand bounding sphere after scaling mesh
    /// let new_bounds = [0.0, 1.5, 0.0, 2.5]; // Larger radius
    /// scene.update_object_bounds(obj_id, new_bounds)?;
    /// ```
    pub fn update_object_bounds(&mut self, id: ObjectId, bounds: [f32; 4]) -> Result<()> {
        let Some((_, record)) = self.objects.get_mut_with_index(id) else {
            return Err(invalid("object"));
        };
        record.instance.bounds = bounds;
        record.aabb = sphere_to_aabb(bounds);
        // Bounds don't affect the instancing group, so update in-place when layout is stable.
        if !self.objects_dirty {
            let slot = record.draw.first_instance as usize;
            self.gpu_scene.instances.update(slot, record.instance);
            self.gpu_scene.aabbs.update(slot, record.aabb);
        }
        Ok(())
    }

    /// Update lightmap indices for all static objects based on baked lightmap atlas regions.
    ///
    /// Called automatically by the renderer after baking completes. Maps each static object's
    /// mesh_id to its corresponding lightmap atlas region index.
    ///
    /// # Parameters
    /// - `regions`: Lightmap atlas regions from `BakedData::lightmap_atlas_regions()`
    ///
    /// # Algorithm
    /// 1. Build a `mesh_slot → region_index` hashmap from the atlas regions
    /// 2. Iterate all objects in the scene
    /// 3. For each Static object, look up its mesh slot in the map
    /// 4. Update `instance.lightmap_index` to the found index (or 0xFFFFFFFF if not found)
    /// 5. Upload updated instance data to GPU
    ///
    /// # Performance
    /// - CPU cost: O(N + M) where N = object count, M = region count
    /// - GPU cost: O(N) writes to instance buffer (only for static objects)
    #[cfg(feature = "bake")]
    pub fn update_lightmap_indices(
        &mut self,
        regions: &[helio_bake::CachedAtlasRegion],
    ) {
        use std::collections::HashMap;

        // Build mesh_slot -> region_index lookup map
        // mesh_id is stored as [slot_u64, 0] where slot_u64 encodes the Helio MeshId slot
        let region_map: HashMap<u32, u32> = regions
            .iter()
            .enumerate()
            .filter_map(|(idx, r)| {
                // Extract mesh slot from UUID [slot_u64, 0]
                let mesh_slot = r.mesh_id[0] as u32;
                if r.mesh_id[1] == 0 {
                    Some((mesh_slot, idx as u32))
                } else {
                    None // Skip entries with non-zero second component
                }
            })
            .collect();

        let mut updated_count = 0;

        // Iterate all objects and update lightmap indices for static objects.
        // DenseArena exposes its `dense: Vec<T>` as a public field — iterate directly.
        for record in self.objects.dense.iter_mut() {
            // Only non-movable objects are baked (Static + Stationary).
            // build_static_bake_scene includes both, so lightmap indices must
            // be assigned for both here to avoid a silent mismatch.
            if record.movability == libhelio::Movability::Movable {
                continue;
            }

            // Get mesh slot from MeshId
            let mesh_slot = record.mesh.slot();

            // Lookup region index for this mesh
            let lightmap_index = region_map.get(&mesh_slot).copied().unwrap_or(0xFFFFFFFF);

            // Update instance data
            record.instance.lightmap_index = lightmap_index;

            // Upload to GPU if layout is stable
            if !self.objects_dirty {
                let slot = record.gpu_slot as usize;
                self.gpu_scene.instances.update(slot, record.instance);
            }

            updated_count += 1;
        }

        log::info!(
            "[Scene] Updated lightmap indices for {} static objects ({} regions in atlas)",
            updated_count,
            regions.len()
        );
    }

    // ── Editor query API ─────────────────────────────────────────────────────

    /// Return the world-space model matrix of an object.
    ///
    /// Returns `Err` if the handle is invalid.
    pub fn get_object_transform(&self, id: ObjectId) -> Result<Mat4> {
        let Some((_, record)) = self.objects.get_with_index(id) else {
            return Err(invalid("object"));
        };
        Ok(Mat4::from_cols_array(&record.instance.model))
    }

    /// Return the world-space bounding sphere `[cx, cy, cz, radius]` of an object.
    ///
    /// Returns `Err` if the handle is invalid.
    pub fn get_object_bounds(&self, id: ObjectId) -> Result<[f32; 4]> {
        let Some((_, record)) = self.objects.get_with_index(id) else {
            return Err(invalid("object"));
        };
        Ok(record.instance.bounds)
    }

    /// Iterate every live object, yielding `(id, world_transform, bounds_sphere)`.
    ///
    /// `bounds_sphere` is `[cx, cy, cz, radius]` in world space — suitable for
    /// ray-sphere picking.
    ///
    /// This iterator is O(N) in the number of live objects; do not call it per-vertex.
    pub fn iter_objects_for_editor(
        &self,
    ) -> impl Iterator<Item = (ObjectId, Mat4, [f32; 4])> + '_ {
        self.objects.iter_with_handles().map(|(id, rec)| {
            let transform = Mat4::from_cols_array(&rec.instance.model);
            (id, transform, rec.instance.bounds)
        })
    }

    /// Return an [`crate::ObjectDescriptor`] that can be passed straight back to
    /// [`crate::Scene::insert_object`] to create an identical copy.
    ///
    /// Returns `Err` if the handle is invalid.
    pub fn get_object_descriptor(&self, id: ObjectId) -> Result<crate::scene::types::ObjectDescriptor> {
        use crate::scene::types::ObjectDescriptor;
        use crate::groups::GroupMask;
        let Some((_, record)) = self.objects.get_with_index(id) else {
            return Err(invalid("object"));
        };
        Ok(ObjectDescriptor {
            mesh:        record.mesh,
            material:    record.material,
            transform:   Mat4::from_cols_array(&record.instance.model),
            bounds:      record.instance.bounds,
            flags:       record.instance.flags,
            groups:      GroupMask(record.groups.0),
            movability:  Some(record.movability),
        })
    }

    /// Iterate every live object, yielding a [`PickableObject`] descriptor.
    ///
    /// The [`crate::ScenePicker`] uses this to sync its instance list after objects
    /// are added or removed.  O(N) over live objects — call on scene change, not
    /// per frame.
    pub fn iter_pickable_objects(&self) -> impl Iterator<Item = PickableObject> + '_ {
        self.objects.iter_with_handles().map(|(id, rec)| PickableObject {
            id,
            mesh_id: rec.mesh,
            transform: Mat4::from_cols_array(&rec.instance.model),
        })
    }
}

