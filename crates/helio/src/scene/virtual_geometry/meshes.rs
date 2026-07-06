//! Virtual mesh upload and management.
//!
//! Handles meshletization, LOD generation, and virtual mesh lifecycle.

use crate::handles::MeshId;
use crate::mesh::MeshUpload;
use crate::vg::{
    generate_lod_meshes, meshletize_with_indices, VirtualMeshId, VirtualMeshUpload,
};

use super::super::errors::{invalid, Result, SceneError};
use super::super::types::VirtualMeshRecord;

impl super::super::Scene {
    /// Upload a high-resolution mesh and decompose it into GPU meshlets for virtual
    /// geometry rendering.
    ///
    /// Uses meshoptimizer for LOD simplification and meshlet building. Generates
    /// 8 LOD levels and decomposes each into spatially coherent meshlets with
    /// tight bounding spheres and backface cones.
    pub(in crate::scene) fn insert_virtual_mesh(&mut self, upload: VirtualMeshUpload) -> VirtualMeshId {
        // Generate 8 LOD levels via meshopt simplification.
        let lod_meshes = generate_lod_meshes(&upload.vertices, &upload.indices);

        let mut all_meshlets: Vec<libhelio::GpuMeshletEntry> = Vec::new();
        let mut mesh_ids: Vec<MeshId> = Vec::new();

        for (lod_level, (lod_verts, lod_indices)) in lod_meshes.into_iter().enumerate() {
            // Build meshlets with meshoptimizer — this produces a reordered
            // index buffer that groups spatially coherent triangles.
            let (mut meshlets, meshlet_indices) =
                meshletize_with_indices(&lod_verts, &lod_indices, 0, 0);

            // Upload the vertices + meshlet-reordered indices to the mega-buffer.
            let mesh_id = self.mesh_pool.insert(MeshUpload {
                vertices: lod_verts,
                indices: meshlet_indices,
            });
            let slice = self.mesh_pool.get(mesh_id).unwrap().slice;

            for m in &mut meshlets {
                m.first_index += slice.first_index;
                m.vertex_offset += slice.first_vertex as i32;
                m.lod_error = lod_level as f32;
            }
            all_meshlets.extend(meshlets);
            mesh_ids.push(mesh_id);
        }

        let id = VirtualMeshId(self.vg_next_mesh_id);
        self.vg_next_mesh_id += 1;
        self.vg_meshes.insert(
            id,
            VirtualMeshRecord {
                mesh_ids,
                meshlets: all_meshlets,
                ref_count: 0,
            },
        );
        id
    }

    /// Remove a virtual mesh.
    ///
    /// Also removes all underlying mesh data from the mesh pool for each LOD level.
    ///
    /// # Parameters
    /// - `id`: Virtual mesh handle
    ///
    /// # Errors
    /// - [`SceneError::InvalidHandle`](super::super::SceneError::InvalidHandle) if the virtual mesh ID is invalid
    /// - [`SceneError::ResourceInUse`] if any virtual objects are still using this mesh
    ///
    /// # Returns
    /// `Ok(())` if the mesh was successfully removed.
    ///
    /// # Example
    /// ```ignore
    /// // Remove all virtual objects using this mesh first
    /// for obj_id in vg_objects_using_mesh {
    ///     scene.remove_virtual_object(obj_id)?;
    /// }
    ///
    /// // Now the mesh can be removed
    /// scene.remove_virtual_mesh(vg_mesh_id)?;
    /// ```
    pub fn remove_virtual_mesh(&mut self, id: VirtualMeshId) -> Result<()> {
        let ref_count = {
            let record = self
                .vg_meshes
                .get(&id)
                .ok_or_else(|| invalid("virtual_mesh"))?;
            record.ref_count
        };
        if ref_count != 0 {
            return Err(SceneError::ResourceInUse {
                resource: "virtual_mesh",
            });
        }
        if let Some(record) = self.vg_meshes.remove(&id) {
            for mesh_id in record.mesh_ids {
                // Ignore the return value to avoid altering observable behavior if
                // `remove_mesh` returns a Result or other value.
                let _ = self.remove_mesh(mesh_id);
            }
        }
        Ok(())
    }
}

