//! Multi-material (sectioned) mesh support.
//!
//! Mirrors Unreal Engine's Static Mesh section model: a single asset with one shared
//! vertex buffer and N index ranges, each drawn with an independent material.
//! All sections of a placed instance share the same world-space transform.
//!
//! # GPU architecture
//!
//! - **One vertex buffer region** per sectioned mesh asset (uploaded once).
//! - **N index buffer regions**, one per section (one draw call each).
//! - **N `ObjectId`s** per placed instance, pooled under a single [`SectionedInstanceId`].
//!
//! Moving/removing a [`SectionedInstanceId`] updates all N draw calls atomically, so
//! the object behaves as a single unit from the caller's perspective.  The scene also
//! maintains a reverse map so the picker can automatically resolve a section hit back
//! to its parent [`SectionedInstanceId`].

use glam::Mat4;

use crate::arena::SparsePool;
use crate::groups::GroupMask;
use crate::handles::{MaterialId, MeshId, MultiMeshId, ObjectId, SectionedInstanceId};
use crate::mesh::SectionedMeshUpload;
use crate::scene::types::ObjectDescriptor;

use super::errors::{invalid, Result};

// ─── Internal instance record ─────────────────────────────────────────────────

/// Internal record for a placed sectioned mesh instance.
///
/// Stored under [`SectionedInstanceId`] in the scene's `sectioned_instances` pool.
/// The reverse mapping `ObjectId → SectionedInstanceId` lives in `section_to_instance`.
pub(crate) struct SectionedInstanceRecord {
    /// One draw-call `ObjectId` per material section (same order as the `materials`
    /// slice passed to [`Scene::insert_sectioned_object`]).
    pub section_objects: Vec<ObjectId>,
    /// Asset handle — used when decrementing the ref-count on removal.
    pub multi_mesh: MultiMeshId,
}

// ─── Scene methods ────────────────────────────────────────────────────────────

impl super::Scene {
    // ── Asset queries ─────────────────────────────────────────────────────────

    /// Return the per-section [`MeshId`]s for a previously uploaded sectioned mesh.
    ///
    /// Needed when registering the mesh geometry with [`crate::ScenePicker`] so
    /// that each section participates in BVH ray-picking.  The returned slice is
    /// in the same order as the `sections` array passed to [`insert_sectioned_mesh`].
    ///
    /// Returns `None` if the handle is stale or has already been removed.
    pub fn sectioned_section_mesh_ids(&self, id: MultiMeshId) -> Option<&[MeshId]> {
        self.multi_meshes
            .get(id)
            .map(|r| r.section_mesh_ids.as_slice())
    }

    // ── Asset upload / removal ────────────────────────────────────────────────

    /// Upload a multi-material mesh to the GPU.
    ///
    /// Vertices are pushed **once** into the shared vertex buffer.
    /// Each element of `upload.sections` is an independent index list that will be
    /// rendered as a separate draw call with its own material.
    ///
    /// Returns a [`MultiMeshId`] asset handle. The asset persists until
    /// [`remove_sectioned_mesh`](Self::remove_sectioned_mesh) is called.
    pub fn insert_sectioned_mesh(&mut self, upload: SectionedMeshUpload) -> MultiMeshId {
        let record = self.mesh_pool.insert_sectioned(upload);
        let (id, _, _) = self.multi_meshes.insert(record);
        id
    }

    /// Remove a multi-material mesh asset.
    ///
    /// Fails if any live instances still reference this mesh.
    /// The underlying GPU vertex/index buffer space is not reclaimed (append-only pool).
    pub fn remove_sectioned_mesh(&mut self, id: MultiMeshId) -> Result<()> {
        let ref_count = self
            .multi_meshes
            .get(id)
            .ok_or_else(|| invalid("multi_mesh"))?
            .ref_count;
        if ref_count > 0 {
            return Err(invalid("multi_mesh still referenced by live instances"));
        }
        let section_ids = self
            .multi_meshes
            .get(id)
            .map(|r| r.section_mesh_ids.clone())
            .unwrap_or_default();
        for mesh_id in section_ids {
            self.mesh_pool.remove(mesh_id);
        }
        self.multi_meshes.remove(id);
        Ok(())
    }

    // ── Instance placement ────────────────────────────────────────────────────

    /// Place a multi-material mesh instance into the scene.
    ///
    /// Creates **one GPU draw call per section**, all sharing the same `transform`.
    /// The number of `materials` must exactly match the number of sections the
    /// mesh was uploaded with.
    ///
    /// Returns a [`SectionedInstanceId`] — a lightweight `Copy` handle that the
    /// scene stores internally.  Pass it to [`update_sectioned_object_transform`],
    /// [`remove_sectioned_object`], or the editor.  The picker automatically maps
    /// any section hit back to this handle.
    ///
    /// # Errors
    /// - `InvalidHandle` if `multi_mesh` is not a valid handle.
    /// - `InvalidHandle` if `materials.len()` ≠ section count.
    /// - `InvalidHandle` if any `MaterialId` in `materials` is invalid.
    pub fn insert_sectioned_object(
        &mut self,
        multi_mesh: MultiMeshId,
        materials: &[MaterialId],
        transform: Mat4,
        bounds: [f32; 4],
        movability: Option<libhelio::Movability>,
    ) -> Result<SectionedInstanceId> {
        // Snapshot the section mesh IDs — avoids holding a borrow into multi_meshes
        // while we mutably call insert_object.
        let section_mesh_ids = {
            let record = self
                .multi_meshes
                .get(multi_mesh)
                .ok_or_else(|| invalid("multi_mesh"))?;
            if record.section_mesh_ids.len() != materials.len() {
                return Err(invalid("material count must match mesh section count"));
            }
            record.section_mesh_ids.clone()
        };

        let mut section_objects = Vec::with_capacity(section_mesh_ids.len());
        for (&mesh_id, &material_id) in section_mesh_ids.iter().zip(materials.iter()) {
            let obj_id = self.insert_object(ObjectDescriptor {
                mesh: mesh_id,
                material: material_id,
                transform,
                bounds,
                flags: 0b11, // casts + receives shadows
                groups: GroupMask::NONE,
                movability,
            })?;
            section_objects.push(obj_id);
        }

        // Increment the asset ref-count.
        if let Some((_, r)) = self.multi_meshes.get_mut_with_slot(multi_mesh) {
            r.ref_count += 1;
        }

        // Store the instance in the pool and build the reverse map.
        let record = SectionedInstanceRecord {
            section_objects: section_objects.clone(),
            multi_mesh,
        };
        let (inst_id, _, _) = self.sectioned_instances.insert(record);
        for obj_id in &section_objects {
            self.section_to_instance.insert(*obj_id, inst_id);
        }

        Ok(inst_id)
    }

    /// Update the world transform of all sections in a placed instance.
    ///
    /// O(N) where N = section count (typically 2–8).
    pub fn update_sectioned_object_transform(
        &mut self,
        id: SectionedInstanceId,
        transform: Mat4,
    ) -> Result<()> {
        let section_objects = self
            .sectioned_instances
            .get(id)
            .ok_or_else(|| invalid("sectioned_instance"))?
            .section_objects
            .clone();
        for obj_id in section_objects {
            self.update_object_transform(obj_id, transform)?;
        }
        Ok(())
    }

    /// Remove a placed sectioned mesh instance.
    ///
    /// Removes all GPU draw calls, cleans up the reverse map, and decrements the
    /// asset ref-count.  The [`MultiMeshId`] asset is unaffected.
    pub fn remove_sectioned_object(&mut self, id: SectionedInstanceId) -> Result<()> {
        let record = self
            .sectioned_instances
            .get(id)
            .ok_or_else(|| invalid("sectioned_instance"))?
            .section_objects
            .clone();

        for obj_id in &record {
            self.section_to_instance.remove(obj_id);
            self.remove_object(*obj_id)?;
        }

        // Decrement asset ref-count — need to look up multi_mesh from the pool first.
        let multi_mesh = self
            .sectioned_instances
            .get(id)
            .map(|r| r.multi_mesh);
        if let Some(mm) = multi_mesh {
            if let Some((_, r)) = self.multi_meshes.get_mut_with_slot(mm) {
                r.ref_count = r.ref_count.saturating_sub(1);
            }
        }

        self.sectioned_instances.remove(id);
        Ok(())
    }

    // ── Instance queries (used by picker + editor) ────────────────────────────

    /// Return the `SectionedInstanceId` that owns the given section `ObjectId`,
    /// or `None` if the object is not part of any sectioned instance.
    pub fn section_instance_for_object(&self, id: ObjectId) -> Option<SectionedInstanceId> {
        self.section_to_instance.get(&id).copied()
    }

    /// Return the world transform of a sectioned instance (taken from section 0).
    ///
    /// Returns `None` if the handle is stale.
    pub fn get_sectioned_instance_transform(&self, id: SectionedInstanceId) -> Option<Mat4> {
        let first = *self.sectioned_instances.get(id)?.section_objects.first()?;
        self.get_object_transform(first).ok()
    }

    /// Return the bounding sphere `[cx, cy, cz, radius]` of a sectioned instance
    /// (taken from section 0 — all sections share the same bounds).
    ///
    /// Returns `None` if the handle is stale.
    pub fn get_sectioned_instance_bounds(&self, id: SectionedInstanceId) -> Option<[f32; 4]> {
        let first = *self.sectioned_instances.get(id)?.section_objects.first()?;
        self.get_object_bounds(first).ok()
    }

    /// Duplicate a placed sectioned mesh instance, preserving its transform and materials.
    ///
    /// Returns the [`SectionedInstanceId`] of the new copy, or an error if the
    /// source handle is stale.
    pub fn duplicate_sectioned_object(&mut self, id: SectionedInstanceId) -> Result<SectionedInstanceId> {
        // Snapshot what we need before any mutable borrows.
        let (multi_mesh, section_objects) = {
            let rec = self.sectioned_instances.get(id).ok_or_else(|| invalid("sectioned_instance"))?;
            (rec.multi_mesh, rec.section_objects.clone())
        };

        // Collect per-section descriptors (material + bounds) from section 0 for bounds.
        let mut materials: Vec<MaterialId> = Vec::with_capacity(section_objects.len());
        let mut bounds = [0.0f32; 4];
        let mut transform = Mat4::IDENTITY;
        let mut movability = None;
        for (i, &obj_id) in section_objects.iter().enumerate() {
            let desc = self.get_object_descriptor(obj_id)?;
            materials.push(desc.material);
            if i == 0 {
                bounds     = desc.bounds;
                transform  = desc.transform;
                movability = desc.movability;
            }
        }

        self.insert_sectioned_object(multi_mesh, &materials, transform, bounds, movability)
    }
}

