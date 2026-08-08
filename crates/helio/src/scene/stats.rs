//! Scene statistics and query methods.
//!
//! This module contains methods for querying scene statistics such as mesh counts,
//! light counts, and bake invalidation state.

use libhelio::GpuLight;

use crate::handles::LightId;
use crate::scene::types::{LightRecord, ObjectRecord};
use crate::scene::Scene;

impl Scene {
    /// Get read-only access to the GPU scene resources.
    ///
    /// Returns a reference to the internal [`GpuScene`] containing all GPU buffers,
    /// bind groups, and render state. Used by the renderer to access GPU resources.
    ///
    /// # Returns
    /// A reference to the [`GpuScene`].
    pub fn gpu_scene(&self) -> &helio_core::GpuScene {
        &self.gpu_scene
    }

    /// Iterate over all live lights, yielding the handle, GPU light data, and user tag.
    pub fn iter_lights(&self) -> impl Iterator<Item = (LightId, &GpuLight, u64)> + '_ {
        self.world
            .query::<&LightRecord>()
            .map(|(entity, record)| (LightId::from_entity(entity), &record.gpu, record.user_tag))
    }

    /// Get the GPU light data for a single light by its handle.
    pub fn get_light(&self, id: LightId) -> Option<GpuLight> {
        self.world.get::<LightRecord>(id.entity()).map(|record| record.gpu)
    }

    /// Look up an object by the application-defined `user_tag` it was
    /// inserted with.
    ///
    /// This is what lets an application drive the scene from its own entity
    /// ids without maintaining a parallel `entity -> ObjectId` map: tag each
    /// actor on insert, then find it again here. Returns `None` for tag `0`
    /// (the untagged default) or if no live object carries the tag.
    pub fn object_by_tag(&self, user_tag: u64) -> Option<crate::handles::ObjectId> {
        self.objects_by_tag.get(&user_tag).copied()
    }

    /// Look up a light by the application-defined `user_tag` it was inserted
    /// with. See [`Scene::object_by_tag`].
    pub fn light_by_tag(&self, user_tag: u64) -> Option<LightId> {
        self.lights_by_tag.get(&user_tag).copied()
    }

    /// Returns true if static geometry or lights have been added since the last bake.
    ///
    /// When this returns true after a bake has been configured, the baked lighting
    /// is out of date and `auto_bake()` should be called again to rebake with the
    /// new static content.
    pub fn is_bake_invalidated(&self) -> bool {
        self.bake_invalidated
    }

    /// Aggregate mesh statistics for the scene: total vertices, total triangles,
    /// and the number of unique mesh records currently live in the pool.
    /// These reflect the GPU buffer occupancy (unique geometry, not instanced totals).
    pub fn mesh_stats(&self) -> (usize, usize, usize) {
        let verts = self.mesh_pool.total_vertex_count();
        let tris  = self.mesh_pool.total_index_count() / 3;
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
        for (_, obj) in self.world.query::<&ObjectRecord>() {
            drawn_tris += (obj.draw.index_count / 3) as usize;
            if let Some(rec) = self.mesh_pool.get(obj.mesh) {
                drawn_verts += rec.slice.vertex_count as usize;
            }
        }
        (drawn_verts, drawn_tris)
    }
}
