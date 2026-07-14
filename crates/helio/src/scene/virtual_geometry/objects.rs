//! Virtual object instancing and transform updates.
//!
//! Provides methods for creating instances of virtual meshes and updating their transforms.

use glam::Mat4;
use helio_core::GpuInstanceData;

use crate::handles::VirtualObjectId;
use crate::vg::VirtualObjectDescriptor;

use super::super::errors::{invalid, Result};
use super::super::helpers::normal_matrix;
use super::super::types::VirtualObjectRecord;

fn include_dirty_instance(range: &mut Option<(usize, usize)>, index: usize) {
    *range = Some(match *range {
        Some((start, end)) => (start.min(index), end.max(index + 1)),
        None => (index, index + 1),
    });
}

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
        let mesh_id = record
            .mesh_ids
            .first()
            .copied()
            .ok_or_else(|| invalid("virtual_mesh_geometry"))?;
        record.ref_count += 1;

        let instance = GpuInstanceData {
            model: desc.transform.to_cols_array(),
            normal_mat: normal_matrix(desc.transform),
            bounds: desc.bounds,
            mesh_id: mesh_id.slot(),
            material_id: desc.material_id,
            flags: desc.flags,
            lightmap_index: 0xFFFFFFFF,  // Virtual geometry doesn't use lightmaps
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
    /// The change is reflected by a bounded instance-buffer upload on next `flush()`.
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
    /// - CPU cost: O(1) - updates the record and expands one dirty range
    /// - GPU cost: Uploads only the contiguous dirty instance range on next `flush()`
    /// - Memory: No allocations
    ///
    /// # Deferred Updates
    ///
    /// Meshlet descriptors, object LOD ranges, and work spans remain immutable.
    /// Insertions/removals still rebuild topology, but transform-only changes do not.
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
        let Some((dense_index, record)) = self.vg_objects.get_mut_with_index(id) else {
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
        let updated_instance = record.instance;

        // Increment generation counter for movable objects (for shadow cache invalidation)
        self.movable_objects_generation += 1;
        self.gpu_scene.movable_objects_generation = self.movable_objects_generation;

        // Topology rebuilds republish every instance. Otherwise keep the CPU
        // mirror in place and publish only the affected range on the next flush.
        if !self.vg_objects_dirty {
            if let Some(instance) = self.vg_cpu_instances.get_mut(dense_index) {
                *instance = updated_instance;
                include_dirty_instance(&mut self.vg_instance_dirty_range, dense_index);
            } else {
                // A missing mirror means topology has not been published yet.
                self.vg_objects_dirty = true;
            }
        }
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use glam::{Mat4, Vec3};
    use libhelio::Movability;

    use crate::{
        groups::GroupMask,
        mesh::PackedVertex,
        vg::{VirtualMeshUpload, VirtualObjectDescriptor},
        Scene,
    };

    use super::include_dirty_instance;

    fn create_test_scene() -> Scene {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("no adapter found");
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                ..Default::default()
            },
        ))
        .expect("failed to create device");
        Scene::new(Arc::new(device), Arc::new(queue))
    }

    #[test]
    fn transform_dirty_range_is_end_exclusive_and_coalesced() {
        let mut range = None;
        include_dirty_instance(&mut range, 7);
        assert_eq!(range, Some((7, 8)));
        include_dirty_instance(&mut range, 3);
        include_dirty_instance(&mut range, 11);
        include_dirty_instance(&mut range, 5);
        assert_eq!(range, Some((3, 12)));
    }

    #[test]
    fn transform_only_flush_keeps_topology_version_and_publishes_one_instance() {
        let mut scene = create_test_scene();
        let vertices = vec![
            PackedVertex::from_components(
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0, 0.0],
                1.0,
            ),
            PackedVertex::from_components(
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0, 0.0],
                1.0,
            ),
            PackedVertex::from_components(
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0, 0.0],
                1.0,
            ),
        ];
        let mesh = scene.insert_virtual_mesh(VirtualMeshUpload {
            vertices,
            indices: vec![0, 1, 2],
        });
        let object = scene
            .insert_virtual_object(VirtualObjectDescriptor {
                virtual_mesh: mesh,
                material_id: 0,
                transform: Mat4::IDENTITY,
                bounds: [0.5, 0.0, 0.5, 1.0],
                flags: 0,
                groups: GroupMask::NONE,
                movability: Some(Movability::Movable),
            })
            .expect("insert virtual object");
        scene.flush();

        let topology_version = scene.vg_buffer_version;
        let instance_version = scene.vg_instance_version;
        let moved = Mat4::from_translation(Vec3::new(4.0, 5.0, 6.0));
        scene
            .update_virtual_object_transform(object, moved)
            .expect("update virtual object transform");

        assert!(!scene.vg_objects_dirty);
        assert_eq!(scene.vg_instance_dirty_range, Some((0, 1)));
        scene.flush();

        assert_eq!(scene.vg_buffer_version, topology_version);
        assert_eq!(scene.vg_instance_version, instance_version + 1);
        assert_eq!(scene.vg_published_instance_dirty_range, Some((0, 1)));
        assert_eq!(scene.vg_cpu_instances[0].model, moved.to_cols_array());
        let frame = scene.vg_frame_data().expect("VG frame data");
        assert_eq!(frame.instance_dirty_start, 0);
        assert_eq!(frame.instance_dirty_count, 1);
    }
}
