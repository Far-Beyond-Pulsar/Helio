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
//! - **N `ObjectId`s** per placed instance, grouped inside a [`SectionedObjectId`].
//!
//! Moving/removing a [`SectionedObjectId`] updates all N draw calls atomically, so
//! the object behaves as a single unit from the caller's perspective.

use glam::Mat4;

use crate::groups::GroupMask;
use crate::handles::{MaterialId, MultiMeshId, ObjectId};
use crate::mesh::SectionedMeshUpload;
use crate::scene::types::ObjectDescriptor;

use super::errors::{invalid, Result};

// ─── Public type ─────────────────────────────────────────────────────────────

/// A placed instance of a multi-material (sectioned) mesh.
///
/// Contains one [`ObjectId`] per material section. All sections share the same
/// vertex buffer and world-space transform; updating one updates all.
///
/// Created by [`Scene::insert_sectioned_object`].
/// Consumed by [`Scene::remove_sectioned_object`].
#[derive(Debug, Clone)]
pub struct SectionedObjectId {
    /// One draw-call object per material section (same order as `materials` supplied
    /// to [`Scene::insert_sectioned_object`]).
    pub section_objects: Vec<ObjectId>,
}

// ─── Scene methods ────────────────────────────────────────────────────────────

impl super::Scene {
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
    /// Fails if any [`SectionedObjectId`] instances are still alive for this mesh.
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
        // Free each section's MeshId slot (mesh pool ref counts are already at zero
        // because remove_sectioned_object decremented them via remove_object).
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

    /// Place a multi-material mesh instance into the scene.
    ///
    /// Creates **one GPU draw call per section**, all sharing the same `transform`.
    /// The number of `materials` must exactly match the number of sections the
    /// mesh was uploaded with.
    ///
    /// Returns a [`SectionedObjectId`] that moves and removes all sections atomically.
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
    ) -> Result<SectionedObjectId> {
        // Snapshot the section mesh IDs while we still have an immutable borrow.
        let section_mesh_ids = {
            let record = self
                .multi_meshes
                .get(multi_mesh)
                .ok_or_else(|| invalid("multi_mesh"))?;
            if record.section_mesh_ids.len() != materials.len() {
                return Err(invalid(
                    "material count must match mesh section count",
                ));
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

        // Increment the asset's instance ref count.
        if let Some((_, r)) = self.multi_meshes.get_mut_with_slot(multi_mesh) {
            r.ref_count += 1;
        }

        Ok(SectionedObjectId { section_objects })
    }

    /// Update the world transform of all sections in a sectioned object.
    ///
    /// O(N) where N = section count (typically a small constant, e.g. 2–8).
    pub fn update_sectioned_object_transform(
        &mut self,
        id: &SectionedObjectId,
        transform: Mat4,
    ) -> Result<()> {
        for &obj_id in &id.section_objects {
            self.update_object_transform(obj_id, transform)?;
        }
        Ok(())
    }

    /// Remove all section draw-call objects for a placed sectioned mesh instance.
    ///
    /// The [`MultiMeshId`] **asset** is not affected; only this instance is removed.
    /// Decrements the asset's reference count so it can later be freed with
    /// [`remove_sectioned_mesh`](Self::remove_sectioned_mesh).
    pub fn remove_sectioned_object(
        &mut self,
        id: SectionedObjectId,
        multi_mesh: MultiMeshId,
    ) -> Result<()> {
        for obj_id in id.section_objects {
            self.remove_object(obj_id)?;
        }
        if let Some((_, r)) = self.multi_meshes.get_mut_with_slot(multi_mesh) {
            r.ref_count = r.ref_count.saturating_sub(1);
        }
        Ok(())
    }
}
