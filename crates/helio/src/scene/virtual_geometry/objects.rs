//! Virtual object instancing and transform updates.
//!
//! Provides methods for creating instances of virtual meshes and updating their transforms.

use glam::Mat4;
use helio_v3::GpuInstanceData;

use crate::handles::VirtualObjectId;
use crate::vg::VirtualObjectDescriptor;

use super::super::errors::{invalid, Result};
use super::super::helpers::normal_matrix;
use super::super::types::VirtualObjectRecord;

impl super::super::Scene {
    /// Place an instance of a virtual mesh into the scene.
    ///
    /// Creates a new virtual object that references a virtual mesh with a world-space
    /// transform and material.
    ///
    /// # Parameters
    /// - `desc`: Virtual object descriptor containing:
    ///   - `virtual_mesh`: Virtual mesh handle from [`insert_virtual_mesh`](crate::Scene::insert_virtual_mesh)
    ///   - `transform`: World-space model matrix
    ///   - `bounds`: Bounding sphere `[center.x, center.y, center.z, radius]`
    ///   - `material_id`: Material slot index (u32, not MaterialId)
    ///   - `flags`: Render flags (bit 0 = casts shadow, bit 1 = receives shadow)
    ///   - `groups`: Group membership mask
    ///
    /// # Errors
    /// - [`SceneError::InvalidHandle`](super::super::SceneError::InvalidHandle) if the virtual mesh ID is invalid
    ///
    /// # Returns
    /// A [`VirtualObjectId`] handle that can be used to update or remove the virtual object.
    ///
    /// # Performance
    /// - CPU cost: O(1) insertion into dense arena + marks VG dirty
    /// - GPU cost: Deferred to next `flush()` when VG buffers are rebuilt
    /// - Memory: No allocations (uses pre-allocated arena capacity)
    ///
    /// # Reference Counting
    ///
    /// Increments the virtual mesh's reference count. The mesh cannot be removed
    /// while this object exists.
    ///
    /// # Example
    /// ```ignore
    /// use helio::{VirtualObjectDescriptor, GroupMask};
    /// use glam::Mat4;
    ///
    /// let vg_obj_id = scene.insert_virtual_object(VirtualObjectDescriptor {
    ///     virtual_mesh: vg_mesh_id,
    ///     transform: Mat4::from_translation([10.0, 0.0, 5.0].into()),
    ///     bounds: [10.0, 0.0, 5.0, 5.0], // Sphere at (10, 0, 5) with radius 5
    ///     material_id: 0,                // Material slot 0
    ///     flags: 0b11,                   // Casts and receives shadows
    ///     groups: GroupMask::NONE,       // Always visible
    /// })?;
    /// ```
    pub(in crate::scene) fn insert_virtual_object(
        &mut self,
        desc: VirtualObjectDescriptor,
    ) -> Result<VirtualObjectId> {
        let record = self
            .vg_meshes
            .get_mut(&desc.virtual_mesh)
            .ok_or_else(|| invalid("virtual_mesh"))?;
        record.ref_count += 1;

        let instance = GpuInstanceData {
            model: desc.transform.to_cols_array(),
            normal_mat: normal_matrix(desc.transform),
            bounds: desc.bounds,
            mesh_id: record.mesh_ids[0].slot(),
            material_id: desc.material_id,
            flags: desc.flags,
            _pad: 0,
        };
        let (id, _) = self.vg_objects.insert(VirtualObjectRecord {
            virtual_mesh: desc.virtual_mesh,
            groups: desc.groups,
            movability: desc.movability.unwrap_or_default(),
            instance,
        });
        self.vg_objects_dirty = true;
        Ok(id)
    }

    /// Update the world transform of a virtual object.
    ///
    /// Modifies the object's model matrix and recomputes the normal matrix.
    /// The change is reflected in the next VG buffer rebuild (on next `flush()`).
    ///
    /// # Parameters
    /// - `id`: Virtual object handle
    /// - `transform`: New world-space model matrix
    ///
    /// # Errors
    /// - [`SceneError::InvalidHandle`](super::super::SceneError::InvalidHandle) if the virtual object ID is invalid
    ///
    /// # Returns
    /// `Ok(())` if the transform was successfully updated.
    ///
    /// # Performance
    /// - CPU cost: O(1) - updates CPU-side record, marks VG dirty
    /// - GPU cost: Deferred to next `flush()` when VG buffers are rebuilt
    /// - Memory: No allocations
    ///
    /// # Deferred Updates
    ///
    /// Unlike regular objects (which support O(1) in-place GPU updates), virtual
    /// geometry transforms are always applied via full buffer rebuild. This is
    /// acceptable because VG is designed for static or infrequently-updated geometry.
    ///
    /// # Example
    /// ```ignore
    /// use glam::{Mat4, Vec3};
    ///
    /// // Move virtual object
    /// let new_transform = Mat4::from_translation(Vec3::new(20.0, 0.0, 10.0));
    /// scene.update_virtual_object_transform(vg_obj_id, new_transform)?;
    ///
    /// // Change takes effect on next flush()
    /// scene.flush();
    /// ```
    pub fn update_virtual_object_transform(
        &mut self,
        id: VirtualObjectId,
        transform: Mat4,
    ) -> Result<()> {
        let Some((_, record)) = self.vg_objects.get_mut_with_index(id) else {
            return Err(invalid("virtual_object"));
        };
        // Enforce movability: Static objects cannot have transforms updated
        if !record.movability.can_move() {
            log::warn!(
                "Attempted to update transform on Static virtual object {:?}. Set movability to Movable to allow transform updates.",
                id
            );
            return Ok(()); // No-op instead of error
        }
        record.instance.model = transform.to_cols_array();
        record.instance.normal_mat = normal_matrix(transform);

        // Increment generation counter for movable objects (for shadow cache invalidation)
        self.movable_objects_generation += 1;
        self.gpu_scene.movable_objects_generation = self.movable_objects_generation;

        // Mark dirty so vg_frame_data() picks up the new transform.
        self.vg_objects_dirty = true;
        Ok(())
    }

    /// Remove a virtual object from the scene.
    ///
    /// Removes the virtual object from the dense arena and decrements the virtual
    /// mesh's reference count.
    ///
    /// # Parameters
    /// - `id`: Virtual object handle
    ///
    /// # Errors
    /// - [`SceneError::InvalidHandle`](super::super::SceneError::InvalidHandle) if the virtual object ID is invalid
    ///
    /// # Returns
    /// `Ok(())` if the virtual object was successfully removed.
    ///
    /// # Performance
    /// - CPU cost: O(1) removal from dense arena + marks VG dirty
    /// - GPU cost: Deferred to next `flush()` when VG buffers are rebuilt
    ///
    /// # Reference Counting
    ///
    /// Decrements the virtual mesh's reference count. If the count reaches zero,
    /// the mesh can be removed with [`remove_virtual_mesh`](crate::Scene::remove_virtual_mesh).
    ///
    /// # Example
    /// ```ignore
    /// scene.remove_virtual_object(vg_obj_id)?;
    ///
    /// // If this was the last object using the mesh, it can now be removed
    /// scene.remove_virtual_mesh(vg_mesh_id)?;
    /// ```
    pub fn remove_virtual_object(&mut self, id: VirtualObjectId) -> Result<()> {
        let removed = self
            .vg_objects
            .remove(id)
            .ok_or_else(|| invalid("virtual_object"))?;
        if let Some(mesh_record) = self.vg_meshes.get_mut(&removed.removed.virtual_mesh) {
            mesh_record.ref_count = mesh_record.ref_count.saturating_sub(1);
        }
        self.vg_objects_dirty = true;
        Ok(())
    }
}

