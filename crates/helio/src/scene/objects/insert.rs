//! Object insertion operations.
//!
//! Provides the [`Scene::insert_object`](crate::Scene::insert_object) method for adding
//! renderable objects to the scene with O(1) performance in persistent mode.

use helio_v3::{DrawIndexedIndirectArgs, GpuDrawCall};

use crate::handles::ObjectId;

use super::super::errors::{invalid, Result};
use super::super::helpers::{object_gpu_data, object_is_visible};
use super::super::types::ObjectDescriptor;

impl super::super::Scene {
    /// Insert a renderable object into the scene.
    ///
    /// Creates a new object that references a mesh and material, with a world-space
    /// transform and optional group membership.
    ///
    /// # Parameters
    /// - `desc`: Object descriptor containing:
    ///   - `mesh`: Mesh handle from [`insert_mesh`](crate::Scene::insert_mesh)
    ///   - `material`: Material handle from [`insert_material`](crate::Scene::insert_material)
    ///   - `transform`: World-space model matrix (column-major)
    ///   - `bounds`: Bounding sphere `[center.x, center.y, center.z, radius]`
    ///   - `flags`: Render flags (bit 0 = casts shadow, bit 1 = receives shadow)
    ///   - `groups`: Group membership mask for batch visibility control
    ///
    /// # Errors
    /// - [`SceneError::InvalidHandle`](super::super::SceneError::InvalidHandle) if the mesh or material ID is invalid
    ///
    /// # Returns
    /// An [`ObjectId`] handle that can be used to update or remove the object.
    ///
    /// # Performance (Persistent Mode - Default)
    /// - CPU cost: O(1) insertion into dense arena + O(1) GPU buffer pushes
    /// - GPU cost: Delta uploads to instance, AABB, draw call, indirect, and visibility buffers
    /// - Memory: No allocations (uses pre-allocated buffer capacity)
    ///
    /// # Performance (Optimized Mode)
    /// - CPU cost: O(1) insertion + invalidates optimization (next add triggers rebuild)
    /// - GPU cost: Deferred to next `flush()` when rebuild occurs
    /// - Trade-off: Reverts to persistent mode after insert
    ///
    /// # Mode Behavior
    ///
    /// **Persistent mode (default):**
    /// - Object gets GPU slot = dense_index
    /// - One draw call per object (instance_count = 1)
    /// - Delta updates to GPU buffers (O(1) operations)
    /// - No sorting or instancing
    ///
    /// **Optimized mode (when active):**
    /// - Invalidates optimization (sets `objects_layout_optimized = false`)
    /// - Marks scene dirty for rebuild on next `flush()`
    /// - Next rebuild will include the new object
    ///
    /// # Example
    /// ```ignore
    /// use helio::{ObjectDescriptor, GroupMask};
    /// use glam::Mat4;
    ///
    /// let obj_id = scene.insert_object(ObjectDescriptor {
    ///     mesh: mesh_id,
    ///     material: material_id,
    ///     transform: Mat4::from_translation([0.0, 1.5, 0.0].into()),
    ///     bounds: [0.0, 1.5, 0.0, 1.6],  // Sphere at (0, 1.5, 0) with radius 1.6
    ///     flags: 0b11,                    // Casts and receives shadows
    ///     groups: GroupMask::NONE,        // Always visible
    /// })?;
    /// ```
    ///
    /// # Reference Counting
    ///
    /// Increments the reference count for the mesh and material. They cannot be removed
    /// while this object exists. Call [`remove_object`](crate::Scene::remove_object) to
    /// decrement reference counts.
    pub(in crate::scene) fn insert_object(&mut self, desc: ObjectDescriptor) -> Result<ObjectId> {
        let mesh_slice = {
            let mesh = self
                .mesh_pool
                .get(desc.mesh)
                .ok_or_else(|| invalid("mesh"))?;
            mesh.slice
        };
        let material_slot = {
            let (slot, material) = self
                .materials
                .get_mut_with_slot(desc.material)
                .ok_or_else(|| invalid("material"))?;
            material.ref_count += 1;
            slot
        };
        self.mesh_pool
            .get_mut(desc.mesh)
            .ok_or_else(|| invalid("mesh"))?
            .ref_count += 1;

        let record = object_gpu_data(desc.mesh, material_slot, desc, mesh_slice);
        let (id, dense_index) = self.objects.insert(record);

        // Track static topology changes for shadow atlas caching
        let inserted_movability = self.objects.get_dense(dense_index)
            .map(|r| r.movability)
            .unwrap_or_default();
        if !inserted_movability.can_move() {
            self.static_objects_dirty = true;
        }

        if self.objects_layout_optimized {
            // Optimization active - invalidate and mark for rebuild
            self.objects_layout_optimized = false;
            self.objects_dirty = true;
        } else {
            // Persistent mode - delta update
            let record = self.objects.get_dense_mut(dense_index).unwrap();
            record.gpu_slot = dense_index as u32;
            record.draw.first_instance = dense_index as u32;

            // Push to GPU buffers (O(1) operations)
            self.gpu_scene.instances.push(record.instance);
            self.gpu_scene.aabbs.push(record.aabb);

            let draw_call = GpuDrawCall {
                index_count: record.draw.index_count,
                first_index: record.draw.first_index,
                vertex_offset: record.draw.vertex_offset,
                first_instance: dense_index as u32,
                instance_count: 1,
            };
            self.gpu_scene.draw_calls.push(draw_call);

            let indirect_args = DrawIndexedIndirectArgs {
                index_count: record.draw.index_count,
                instance_count: 1,
                first_index: record.draw.first_index,
                base_vertex: record.draw.vertex_offset,
                first_instance: dense_index as u32,
            };
            self.gpu_scene.indirect.push(indirect_args);

            let vis = if object_is_visible(record.groups, self.group_hidden) {
                1u32
            } else {
                0u32
            };
            self.gpu_scene.visibility.push(vis);
        }

        Ok(id)
    }
}

