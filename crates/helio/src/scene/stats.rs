//! Scene statistics and query methods.
//!
//! This module contains methods for querying scene statistics such as mesh counts,
//! light counts and mesh occupancy.

use libhelio::GpuLight;

use crate::handles::LightId;
use crate::scene::Scene;

impl Scene {
    /// Get read-only access to the GPU scene resources.
    ///
    /// Returns a reference to the internal [`GpuScene`] containing all GPU buffers,
    /// bind groups, and render state. Used by the renderer to access GPU resources.
    ///
    /// # Returns
    /// A reference to the [`GpuScene`].
    pub fn gpu_scene(&self) -> &helio_v3::GpuScene {
        &self.gpu_scene
    }

    /// Iterate over all live lights, yielding the handle, GPU light data, and user tag.
    pub fn iter_lights(&self) -> impl Iterator<Item = (LightId, &GpuLight, u64)> + '_ {
        self.lights
            .iter_with_handles()
            .map(|(id, record)| (id, &record.gpu, record.user_tag))
    }

    /// Get the GPU light data for a single light by its handle.
    pub fn get_light(&self, id: LightId) -> Option<GpuLight> {
        self.lights.get_with_index(id).map(|(_, record)| record.gpu)
    }

    /// Aggregate mesh statistics for the scene: total vertices, total triangles,
    /// and the number of unique mesh records currently live in the pool.
    /// These reflect the GPU buffer occupancy (unique geometry, not instanced totals).
    pub fn mesh_stats(&self) -> (usize, usize, usize) {
        let verts = self.mesh_pool.total_vertex_count();
        let tris = self.mesh_pool.total_index_count() / 3;
        let meshes = self.mesh_pool.unique_mesh_count();
        (verts, tris, meshes)
    }

    /// Counts drawn geometry by summing index/vertex counts across all live object
    /// instances. Returns `(drawn_vertices, drawn_triangles)`.
    ///
    /// Unlike `mesh_stats()`, this accounts for instancing: a mesh referenced by
    /// 1,000 objects contributes 1,000× its vertex/triangle count to the totals.
    pub fn drawn_mesh_stats(&self) -> (usize, usize) {
        let mut drawn_verts: usize = 0;
        let mut drawn_tris: usize = 0;
        for i in 0..self.objects.dense_len() {
            let Some(obj) = self.objects.get_dense(i) else {
                continue;
            };
            drawn_tris += (obj.draw.index_count / 3) as usize;
            if let Some(rec) = self.mesh_pool.get(obj.mesh) {
                drawn_verts += rec.slice.vertex_count as usize;
            }
        }
        (drawn_verts, drawn_tris)
    }
}
