//! Object removal operations.
//!
//! Provides the [`Scene::remove_object`](crate::Scene::remove_object) method for removing
//! renderable objects from the scene with O(1) performance in persistent mode.

use helio_v3::DrawIndexedIndirectArgs;

use crate::arena::DenseRemove;
use crate::handles::ObjectId;

use super::super::errors::{invalid, Result};

impl super::super::Scene {
    /// Remove an object from the scene.
    ///
    /// Removes the object from the dense arena, decrements mesh and material reference
    /// counts, and updates GPU buffers.
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
    /// # Performance (Persistent Mode - Default)
    /// - CPU cost: O(1) swap-remove from dense arena + O(1) GPU buffer swap-removes
    /// - GPU cost: Swap-removes from instance, AABB, draw call, indirect, and visibility buffers
    /// - Memory: No allocations
    ///
    /// # Performance (Optimized Mode)
    /// - CPU cost: O(1) removal + invalidates optimization (next remove triggers rebuild)
    /// - GPU cost: Deferred to next `flush()` when rebuild occurs
    /// - Trade-off: Reverts to persistent mode after remove
    ///
    /// # Mode Behavior
    ///
    /// **Persistent mode (default):**
    /// - Swap-removes object from dense arena
    /// - Swap-removes from all GPU buffers (O(1) operations)
    /// - Updates moved object's GPU slot (if swap occurred)
    /// - Decrements mesh and material reference counts
    ///
    /// **Optimized mode (when active):**
    /// - Invalidates optimization (sets `objects_layout_optimized = false`)
    /// - Marks scene dirty for rebuild on next `flush()`
    /// - Removes from CPU-side arena only
    /// - GPU buffers updated during rebuild
    ///
    /// # Swap-Remove Semantics
    ///
    /// In persistent mode, removal uses **swap-remove** for O(1) performance:
    /// 1. The removed object is deleted
    /// 2. The last object in the array moves to the removed object's slot
    /// 3. The moved object's GPU slot is updated to reflect its new position
    ///
    /// If the removed object was the last object, no swap occurs.
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
        if self.objects_layout_optimized {
            // Optimization active - invalidate and mark for rebuild
            self.objects_layout_optimized = false;
            self.objects_dirty = true;

            // Still remove from CPU-side arena
            let DenseRemove { removed, .. } =
                self.objects.remove(id).ok_or_else(|| invalid("object"))?;

            // Decrement ref counts
            if let Some(material) = self
                .materials
                .get_mut_with_slot(removed.material)
                .map(|(_, m)| m)
            {
                material.ref_count = material.ref_count.saturating_sub(1);
            }
            if let Some(mesh) = self.mesh_pool.get_mut(removed.mesh) {
                mesh.ref_count = mesh.ref_count.saturating_sub(1);
            }
        } else {
            // Persistent mode - swap_remove from GPU buffers
            let DenseRemove {
                removed,
                dense_index,
                moved,
            } = self.objects.remove(id).ok_or_else(|| invalid("object"))?;

            // Swap-remove from GPU buffers (O(1) operations)
            self.gpu_scene.instances.swap_remove(dense_index);
            self.gpu_scene.aabbs.swap_remove(dense_index);
            self.gpu_scene.draw_calls.swap_remove(dense_index);
            self.gpu_scene.indirect.swap_remove(dense_index);
            self.gpu_scene.visibility.swap_remove(dense_index);

            // Update moved object's GPU slot
            if let Some((moved_id, new_index)) = moved {
                if let Some((_, record)) = self.objects.get_mut_with_index(moved_id) {
                    record.gpu_slot = new_index as u32;
                    record.draw.first_instance = new_index as u32;

                    // Update draw_call in GPU buffer
                    let mut draw = record.draw;
                    draw.first_instance = new_index as u32;
                    self.gpu_scene.draw_calls.update(new_index, draw);

                    // Update indirect in GPU buffer
                    let indirect_args = DrawIndexedIndirectArgs {
                        index_count: draw.index_count,
                        instance_count: 1,
                        first_index: draw.first_index,
                        base_vertex: draw.vertex_offset,
                        first_instance: new_index as u32,
                    };
                    self.gpu_scene.indirect.update(new_index, indirect_args);
                }
            }

            // Decrement ref counts
            if let Some(material) = self
                .materials
                .get_mut_with_slot(removed.material)
                .map(|(_, m)| m)
            {
                material.ref_count = material.ref_count.saturating_sub(1);
            }
            if let Some(mesh) = self.mesh_pool.get_mut(removed.mesh) {
                mesh.ref_count = mesh.ref_count.saturating_sub(1);
            }
        }

        Ok(())
    }
}

