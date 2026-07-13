//! Virtual mesh upload and management.
//!
//! Handles meshletization, LOD generation, and virtual mesh lifecycle.

use crate::handles::MeshId;
use crate::mesh::MeshUpload;
use crate::vg::{
    generate_lod_meshes, meshletize_with_indices, GeneratedLodMesh, VirtualMeshId,
    VirtualMeshUpload,
};

use super::super::errors::{invalid, Result, SceneError};
use super::super::types::VirtualMeshRecord;

fn referenced_bounds(
    vertices: &[crate::mesh::PackedVertex],
    indices: &[u32],
) -> Option<[f32; 4]> {
    let mut min = glam::Vec3::splat(f32::INFINITY);
    let mut max = glam::Vec3::splat(f32::NEG_INFINITY);

    for &index in indices {
        let position = glam::Vec3::from_array(vertices.get(index as usize)?.position);
        if !position.is_finite() {
            return None;
        }
        min = min.min(position);
        max = max.max(position);
    }

    if indices.is_empty() {
        return None;
    }

    let center = (min + max) * 0.5;
    let mut radius = 0.0_f32;
    for &index in indices {
        let position = glam::Vec3::from_array(vertices[index as usize].position);
        radius = radius.max(position.distance(center));
    }

    Some([center.x, center.y, center.z, radius])
}

impl super::super::Scene {
    /// Upload a high-resolution mesh and decompose it into GPU meshlets for virtual
    /// geometry rendering.
    ///
    /// Uses meshoptimizer for LOD simplification and meshlet building. Generates
    /// up to 8 distinct LOD levels and decomposes each into spatially coherent
    /// meshlets with tight bounding spheres and backface cones. Small or rigid
    /// assets may expose fewer levels rather than duplicated placeholder LODs.
    pub(in crate::scene) fn insert_virtual_mesh(&mut self, upload: VirtualMeshUpload) -> VirtualMeshId {
        let local_bounds = referenced_bounds(&upload.vertices, &upload.indices).unwrap_or([0.0; 4]);

        // Generate the asset's distinct LOD chain via meshopt simplification.
        let lod_meshes = generate_lod_meshes(&upload.vertices, &upload.indices);

        let mut all_meshlets: Vec<libhelio::GpuMeshletEntry> = Vec::new();
        let mut mesh_ids: Vec<MeshId> = Vec::new();
        let mut lod_errors = [0.0; libhelio::VG_LOD_LEVELS];
        let mut lod_first_meshlets = [0; libhelio::VG_LOD_LEVELS];
        let mut lod_meshlet_counts = [0; libhelio::VG_LOD_LEVELS];
        let mut max_meshlet_count = 0;

        for (lod_level, lod_mesh) in lod_meshes.into_iter().enumerate() {
            let GeneratedLodMesh {
                vertices: lod_verts,
                indices: lod_indices,
                error,
            } = lod_mesh;
            let first_meshlet = u32::try_from(all_meshlets.len())
                .expect("virtual mesh exceeds the u32 descriptor address space");
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
                m.lod_error = error;
            }

            let meshlet_count = u32::try_from(meshlets.len())
                .expect("virtual-mesh LOD exceeds the u32 descriptor address space");
            lod_errors[lod_level] = error;
            lod_first_meshlets[lod_level] = first_meshlet;
            lod_meshlet_counts[lod_level] = meshlet_count;
            max_meshlet_count = max_meshlet_count.max(meshlet_count);
            all_meshlets.extend(meshlets);
            mesh_ids.push(mesh_id);
        }

        let lod_count = u32::try_from(mesh_ids.len()).expect("LOD count must fit in u32");

        let id = VirtualMeshId(self.vg_next_mesh_id);
        self.vg_next_mesh_id += 1;
        self.vg_meshes.insert(
            id,
            VirtualMeshRecord {
                mesh_ids,
                meshlets: all_meshlets,
                local_bounds,
                lod_count,
                lod_errors,
                lod_first_meshlets,
                lod_meshlet_counts,
                max_meshlet_count,
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

#[cfg(test)]
mod tests {
    use super::referenced_bounds;
    use crate::mesh::PackedVertex;

    fn vertex(position: [f32; 3]) -> PackedVertex {
        PackedVertex::from_components(
            position,
            [0.0, 1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0, 0.0],
            1.0,
        )
    }

    #[test]
    fn bounds_only_include_referenced_vertices() {
        let vertices = [
            vertex([-1.0, -1.0, -1.0]),
            vertex([1.0, 1.0, 1.0]),
            vertex([10_000.0, 10_000.0, 10_000.0]),
        ];
        let bounds = referenced_bounds(&vertices, &[0, 1, 0]).unwrap();

        assert_eq!(&bounds[..3], &[0.0, 0.0, 0.0]);
        assert!((bounds[3] - 3.0_f32.sqrt()).abs() < 1.0e-6);
    }

    #[test]
    fn bounds_reject_missing_or_non_finite_geometry() {
        assert!(referenced_bounds(&[], &[]).is_none());
        assert!(referenced_bounds(&[vertex([0.0; 3])], &[1]).is_none());
        assert!(referenced_bounds(&[vertex([f32::NAN, 0.0, 0.0])], &[0]).is_none());
    }
}
