//! Object insertion operations.
//!
//! Provides the [`Scene::insert_object`](crate::Scene::insert_object) method for adding
//! renderable objects to the scene. Objects are automatically batched into instanced
//! draw calls on the next flush.

use crate::handles::ObjectId;

use super::super::errors::{invalid, Result};
use super::super::helpers::object_gpu_data;
use super::super::types::ObjectDescriptor;

impl super::super::Scene {
    /// Insert a renderable object into the scene.
    ///
    /// Creates a new object that references a mesh and material, with a world-space
    /// transform and optional group membership. Objects sharing the same mesh and
    /// material are automatically batched into instanced draw calls on the next flush.
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
    /// # Performance
    /// - CPU cost: O(1) insertion into dense arena
    /// - GPU cost: Full optimized rebuild deferred to next `flush()` — includes
    ///   automatic instancing (objects with the same mesh+material share a draw call)
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
    pub fn insert_object(&mut self, desc: ObjectDescriptor) -> Result<ObjectId> {
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

        let user_tag = desc.user_tag;
        let record = object_gpu_data(desc.mesh, material_slot, desc, mesh_slice);
        let is_static = !record.movability.can_move();

        let entity = self.world.spawn();
        self.world.insert(entity, record);
        let id = ObjectId::from_entity(entity);

        // Index by application tag so the owner can find this object again
        // without keeping its own id map. Tag 0 means "untagged".
        if user_tag != 0 {
            self.objects_by_tag.insert(user_tag, id);
        }

        // Track static topology changes for shadow atlas caching
        if is_static {
            self.static_objects_dirty = true;
            self.bake_invalidated = true;
        }

        // Mark for full optimized rebuild on next flush — this automatically
        // batches objects with the same mesh+material into instanced draw calls.
        self.objects_dirty = true;

        Ok(id)
    }
}
