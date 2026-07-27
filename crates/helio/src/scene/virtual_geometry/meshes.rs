//! Virtual mesh upload and management.
//!
//! Handles meshletization, LOD generation, and virtual mesh lifecycle.

use glam::Vec3;
use libhelio::GpuMeshletVertex;

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
    pub fn insert_virtual_mesh(&mut self, upload: VirtualMeshUpload) -> VirtualMeshId {
        let local_bounds = referenced_bounds(&upload.vertices, &upload.indices).unwrap_or([0.0; 4]);

        let lod_meshes = generate_lod_meshes(&upload.vertices, &upload.indices);

        let mut all_meshlets: Vec<libhelio::GpuMeshletEntry> = Vec::new();
        let mut all_meshlet_vertices: Vec<GpuMeshletVertex> = Vec::new();
        let mut all_meshlet_indices: Vec<u16> = Vec::new();
        // Per-LOD meshlet counts (needed for parent-link building).
        let mut lod_meshlet_counts: Vec<u32> = Vec::new();

        // Collect per-LOD errors from the simplification chain.
        let mut lod_errors: Vec<f32> = Vec::new();

        for lod_mesh in lod_meshes.into_iter() {
            let GeneratedLodMesh {
                vertices: lod_verts,
                indices: lod_indices,
                error,
            } = lod_mesh;

            let vertex_base = all_meshlet_vertices.len() as u32;
            let index_base = all_meshlet_indices.len() as u32;

            let (mut meshlets, meshlet_verts, meshlet_idxs) =
                meshletize_with_indices(&lod_verts, &lod_indices);

            // Adjust offsets from per-LOD-local to per-mesh-global.
            // lod_error is filled below — we need the NEXT coarser level's error.
            for m in &mut meshlets {
                m.meshlet_vertex_offset += vertex_base;
                m.meshlet_index_offset += index_base;
                // parent_cluster_id filled in post-processing below.
            }

            let meshlet_count = u32::try_from(meshlets.len())
                .expect("virtual-mesh LOD exceeds the u32 descriptor address space");
            lod_errors.push(error);
            lod_meshlet_counts.push(meshlet_count);
            all_meshlets.extend(meshlets);
            all_meshlet_vertices.extend(meshlet_verts);
            all_meshlet_indices.extend(meshlet_idxs);
        }

        // ── Fix up lod_error ──────────────────────────────────────────────────
        // Each meshlet carries the CUMULATIVE simplification error of its own
        // LOD level vs the original mesh.  The DAG traversal starts at the
        // root (coarsest, largest error) and walks toward finer levels:
        // if error <= threshold → good enough, emit this level.
        let lod_count = u32::try_from(lod_meshlet_counts.len()).expect("LOD count must fit in u32");
        let mut meshlet_cursor = 0usize;
        for level in 0..lod_count as usize {
            let count = lod_meshlet_counts[level] as usize;
            let cumulative_error = lod_errors[level];
            for m in &mut all_meshlets[meshlet_cursor..meshlet_cursor + count] {
                m.lod_error = cumulative_error;
            }
            meshlet_cursor += count;
        }

        // ── Build flat DAG parent links ──────────────────────────────────────
        // For each LOD level `l` (child, finer), find the overlapping meshlet
        // in LOD `l+1` (parent, coarser).  Use bounding-sphere overlap:
        // a child belongs to the parent whose center the child's center is
        // closest to, provided their spheres overlap.
        //
        // The coarsest LOD meshlets have no parent (u32::MAX).
        for level in 0..lod_count.saturating_sub(1) as usize {
            let child_start = lod_meshlet_counts[..level].iter().copied().sum::<u32>() as usize;
            let child_count = lod_meshlet_counts[level] as usize;
            let parent_start = child_start + child_count;
            let parent_count = lod_meshlet_counts[level + 1] as usize;

            for ci in child_start..child_start + child_count {
                let child = &all_meshlets[ci];
                let c_center = Vec3::from_array(child.center);
                let c_radius = child.radius;

                let mut best_parent = u32::MAX;
                let mut best_dist = f32::MAX;

                for pi in parent_start..parent_start + parent_count {
                    let parent = &all_meshlets[pi];
                    let p_center = Vec3::from_array(parent.center);
                    let p_radius = parent.radius;

                    let dist = c_center.distance(p_center);
                    if dist <= c_radius + p_radius + 1e-6 && dist < best_dist {
                        best_parent = pi as u32;
                        best_dist = dist;
                    }
                }

                // If no parent overlaps, find the closest parent by center distance.
                if best_parent == u32::MAX {
                    for pi in parent_start..parent_start + parent_count {
                        let parent = &all_meshlets[pi];
                        let p_center = Vec3::from_array(parent.center);
                        let dist = c_center.distance(p_center);
                        if dist < best_dist {
                            best_parent = pi as u32;
                            best_dist = dist;
                        }
                    }
                }

                all_meshlets[ci].parent_cluster_id = best_parent;
            }
        }

        let total_meshlet_count = u32::try_from(all_meshlets.len())
            .expect("virtual mesh exceeds the u32 descriptor address space");

        let id = VirtualMeshId(self.vg_next_mesh_id);
        self.vg_next_mesh_id += 1;
        self.vg_meshes.insert(
            id,
            VirtualMeshRecord {
                meshlets: all_meshlets,
                meshlet_vertices: all_meshlet_vertices,
                meshlet_indices: all_meshlet_indices,
                local_bounds,
                lod_count,
                total_meshlet_count,
                ref_count: 0,
            },
        );
        id
    }

    /// Remove a virtual mesh.
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
        self.vg_meshes.remove(&id);
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
