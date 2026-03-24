//! Virtual mesh upload and management.
//!
//! Handles meshletization, LOD generation, and virtual mesh lifecycle.

use std::collections::HashMap;

use crate::handles::MeshId;
use crate::mesh::MeshUpload;
use crate::vg::{
    generate_lod_meshes, meshletize, sort_triangles_spatially, VirtualMeshId, VirtualMeshUpload,
};

use super::super::errors::{invalid, Result, SceneError};
use super::super::types::VirtualMeshRecord;

impl super::super::Scene {
    /// Upload a high-resolution mesh and decompose it into GPU meshlets for virtual
    /// geometry rendering.
    ///
    /// This method:
    /// 1. Generates 3 LOD levels (full, medium, coarse) via vertex clustering
    /// 2. Spatially sorts triangles for optimal cluster bounds
    /// 3. Decomposes each LOD into meshlets (~64 triangles each)
    /// 4. Uploads vertex/index data to the shared mesh pool
    /// 5. Stores meshlet descriptors for GPU-driven rendering
    ///
    /// # Parameters
    /// - `upload`: High-resolution mesh data (vertices and indices)
    ///
    /// # Returns
    /// A [`VirtualMeshId`] that can be used with [`insert_virtual_object`](crate::Scene::insert_virtual_object).
    ///
    /// # Performance
    /// - CPU cost: O(N log N) spatial sort + O(N) meshletization per LOD
    /// - GPU cost: Uploads vertex/index data for all 3 LODs
    /// - Memory: Stores meshlet descriptors in CPU-side vector
    ///
    /// # LOD Generation
    ///
    /// Three LOD levels are generated automatically:
    /// - **LOD 0 (Full):** Original mesh (100% detail)
    /// - **LOD 1 (Medium):** ~50% vertex count via clustering
    /// - **LOD 2 (Coarse):** ~25% vertex count via clustering
    ///
    /// GPU shader selects LOD based on distance and screen-space error.
    ///
    /// # Meshletization
    ///
    /// Each LOD is decomposed into small clusters (meshlets) for GPU-driven culling:
    /// - Target: ~64 triangles per meshlet
    /// - Each meshlet has tight bounding sphere for culling
    /// - Meshlets reference global vertex/index buffers
    ///
    /// # Example
    /// ```ignore
    /// use helio::VirtualMeshUpload;
    ///
    /// let vg_mesh_id = scene.insert_virtual_mesh(VirtualMeshUpload {
    ///     vertices: high_res_vertices, // Vec<Vertex>
    ///     indices: high_res_indices,   // Vec<u32>
    /// });
    ///
    /// // Now instance this mesh multiple times
    /// for transform in level.transforms {
    ///     scene.insert_virtual_object(VirtualObjectDescriptor {
    ///         virtual_mesh: vg_mesh_id,
    ///         transform,
    ///         ..Default::default()
    ///     })?;
    /// }
    /// ```
    pub fn insert_virtual_mesh(&mut self, upload: VirtualMeshUpload) -> VirtualMeshId {
        // Generate three LOD levels (full, medium, coarse) via vertex clustering.
        let lod_meshes = generate_lod_meshes(&upload.vertices, &upload.indices);

        let mut all_meshlets: Vec<libhelio::GpuMeshletEntry> = Vec::new();
        let mut mesh_ids: Vec<MeshId> = Vec::new();

        for (lod_level, (lod_verts, lod_indices)) in lod_meshes.into_iter().enumerate() {
            // Spatially sort triangles before uploading so the mega-buffer index
            // data matches what meshletize expects (sorted = tight cluster bounds).
            let sorted_indices = sort_triangles_spatially(&lod_verts, &lod_indices);

            // First generate meshlets in the local (per-mesh) index/vertex space.
            // We use zero-based offsets here and will patch them after inserting
            // into the global mesh pool once we know the slice offsets.
            let mut meshlets = meshletize(&lod_verts, &sorted_indices, 0, 0);

            // Now upload the data without cloning; ownership moves into MeshUpload.
            let mesh_id = self.mesh_pool.insert(MeshUpload {
                vertices: lod_verts,
                indices: sorted_indices,
            });
            let slice = self.mesh_pool.get(mesh_id).unwrap().slice;

            // Patch meshlet offsets to account for their location in the mega-buffer.
            for m in &mut meshlets {
                m.first_index += slice.first_index;
                m.vertex_offset += slice.first_vertex as i32;
            }
            // Tag with LOD level so the cull shader can select by distance.
            for m in &mut meshlets {
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

