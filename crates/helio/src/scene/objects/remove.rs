//! Object removal operations.
//!
//! Provides the [`Scene::remove_object`](crate::Scene::remove_object) method for removing
//! renderable objects from the scene.

use crate::handles::ObjectId;

use super::super::errors::{invalid, Result};

impl super::super::Scene {
    /// Remove an object from the scene.
    ///
    /// Removes the object from the dense arena and decrements mesh and material reference
    /// counts. GPU buffers are rebuilt on the next flush with automatic instancing.
    ///
    /// # Parameters
    /// - `id`: Object handle returned by [`insert_object`](crate::Scene::insert_object)
    ///
    /// # Errors
    /// - [`SceneError::InvalidHandle`](super::super::SceneError::InvalidHandle) if the object ID is invalid
    ///
    /// # Returns
    /// `Ok(())` if the object was successfully removed.
    ///
    /// # Performance
    /// - CPU cost: O(1) removal from dense arena
    /// - GPU cost: Full optimized rebuild deferred to next `flush()`
    ///
    /// # Reference Counting
    ///
    /// Decrements reference counts for the mesh and material. If the reference count
    /// reaches zero, the mesh/material can be removed with [`remove_mesh`](crate::Scene::remove_mesh)
    /// or [`remove_material`](crate::Scene::remove_material).
    ///
    /// # Example
    /// ```ignore
    /// // Remove object
    /// scene.remove_object(obj_id)?;
    ///
    /// // Now mesh and material may be removable (if no other objects use them)
    /// if mesh_ref_count == 0 {
    ///     scene.remove_mesh(mesh_id)?;
    /// }
    /// if material_ref_count == 0 {
    ///     scene.remove_material(material_id)?;
    /// }
    /// ```
    pub fn remove_object(&mut self, id: ObjectId) -> Result<()> {
        // Capture handles and movability before removal.
        let (mesh_id, material_id, is_static) = {
            let (_, r) = self
                .objects
                .get_with_index(id)
                .ok_or_else(|| invalid("object"))?;
            (r.mesh, r.material, !r.movability.can_move())
        };

        // Remove from CPU-side arena only.
        // GPU buffers will be rebuilt with automatic instancing on next flush.
        self.objects.remove(id).ok_or_else(|| invalid("object"))?;

        // Decrement ref counts
        if let Some(material) = self
            .materials
            .get_mut_with_slot(material_id)
            .map(|(_, m)| m)
        {
            material.ref_count = material.ref_count.saturating_sub(1);
        }
        if let Some(mesh) = self.mesh_pool.get_mut(mesh_id) {
            mesh.ref_count = mesh.ref_count.saturating_sub(1);
        }

        // Mark for full optimized rebuild on next flush.
        self.objects_dirty = true;

        // Cascade: auto-free mesh and material when their ref counts hit zero.
        if self
            .mesh_pool
            .get(mesh_id)
            .map_or(false, |r| r.ref_count == 0)
        {
            let _ = self.remove_mesh(mesh_id);
        }
        if self
            .materials
            .get(material_id)
            .map_or(false, |r| r.ref_count == 0)
        {
            let _ = self.remove_material(material_id);
        }

        // After removal: mark static atlas dirty if a static object was removed
        if is_static {
            self.static_objects_dirty = true;
        }

        Ok(())
    }
}
