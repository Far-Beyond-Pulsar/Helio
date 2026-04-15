//! Mesh resource management for the scene.
//!
//! Meshes are stored in a shared `MeshPool` and reference-counted. Multiple objects
//! can reference the same mesh. Meshes cannot be removed while objects are using them.

use crate::handles::MeshId;
use crate::mesh::{MeshBuffers, MeshUpload};

use super::super::errors::{invalid, Result, SceneError};

impl super::super::Scene {
    /// Insert a mesh into the scene's mesh pool.
    ///
    /// Uploads vertex and index data to GPU memory and returns a handle that can be
    /// referenced by objects.
    ///
    /// # Parameters
    /// - `mesh`: Mesh upload data containing vertices and indices
    ///
    /// # Returns
    /// A [`MeshId`] handle that can be used with [`insert_object`](crate::Scene::insert_object).
    ///
    /// # Performance
    /// - CPU cost: O(1) handle allocation
    /// - GPU cost: Uploads vertices and indices to growable GPU buffers
    /// - Memory: Vertices and indices are stored in shared mega-buffers
    ///
    /// # Example
    /// ```ignore
    /// let mesh_id = scene.insert_mesh(MeshUpload {
    ///     vertices: vec![/* vertex data */],
    ///     indices: vec![/* index data */],
    /// });
    /// ```
    pub(in crate::scene) fn insert_mesh(&mut self, mesh: MeshUpload) -> MeshId {
        self.mesh_pool.insert(mesh)
    }

    /// Remove a mesh from the scene's mesh pool.
    ///
    /// # Errors
    /// - [`SceneError::InvalidHandle`] if the mesh ID is invalid
    /// - [`SceneError::ResourceInUse`] if any objects are still using this mesh
    ///
    /// # Returns
    /// `Ok(())` if the mesh was successfully removed.
    ///
    /// # Example
    /// ```ignore
    /// // Remove all objects using the mesh first
    /// for obj_id in objects_using_mesh {
    ///     scene.remove_object(obj_id)?;
    /// }
    ///
    /// // Now the mesh can be removed
    /// scene.remove_mesh(mesh_id)?;
    /// ```
    pub fn remove_mesh(&mut self, id: MeshId) -> Result<()> {
        let Some(record) = self.mesh_pool.get(id) else {
            return Err(invalid("mesh"));
        };
        if record.ref_count != 0 {
            return Err(SceneError::ResourceInUse { resource: "mesh" });
        }
        self.mesh_pool.remove(id).ok_or_else(|| invalid("mesh"))?;
        Ok(())
    }

    /// Get read-only access to the mesh pool's GPU buffers.
    ///
    /// Returns buffer views for vertex data, index data, and mesh metadata.
    /// Used by the renderer to bind mesh buffers for drawing.
    ///
    /// # Returns
    /// A [`MeshBuffers`] struct containing references to:
    /// - Vertex buffer (shared for all meshes)
    /// - Index buffer (shared for all meshes)
    /// - Mesh metadata buffer (slice offsets per mesh)
    ///
    /// # Example
    /// ```ignore
    /// let buffers = scene.mesh_buffers();
    /// render_pass.set_vertex_buffer(0, buffers.vertices.slice(..));
    /// render_pass.set_index_buffer(buffers.indices.slice(..), IndexFormat::Uint32);
    /// ```
    pub fn mesh_buffers(&self) -> MeshBuffers<'_> {
        self.mesh_pool.buffers()
    }

    /// Aggregate mesh statistics for the scene: total vertices, total triangles,
    /// and the number of unique mesh records currently live in the pool.
    pub fn mesh_stats(&self) -> (usize, usize, usize) {
        let verts = self.mesh_pool.total_vertex_count();
        let tris  = self.mesh_pool.total_index_count() / 3;
        let meshes = self.mesh_pool.unique_mesh_count();
        (verts, tris, meshes)
    }
}

