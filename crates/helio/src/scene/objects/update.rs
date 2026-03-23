//! Object update operations.
//!
//! Provides methods for updating object transforms, materials, and bounds with
//! O(1) performance in both persistent and optimized modes.

use glam::Mat4;

use crate::handles::{MaterialId, ObjectId};

use super::super::errors::{Result, invalid};
use super::super::helpers::{normal_matrix, sphere_to_aabb};

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
        record.instance.model = transform.to_cols_array();
        record.instance.normal_mat = normal_matrix(transform);
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
}
