//! Object insertion operations (SceneDB-backed).

use crate::handles::ObjectId;

use super::super::errors::{invalid, Result};
use super::super::helpers::object_gpu_data;
use super::super::types::ObjectDescriptor;

impl super::super::Scene {
    /// Insert a renderable object into the scene (SceneDB-backed).
    ///
    /// Spawns an entity in [`Scene::world`](crate::Scene) with
    /// [`HelioGpuInstance`] (GPU-mirrored render data) and
    /// [`HelioCpuInstance`] (CPU bookkeeping). The GPU fields are automatically
    /// synced by `flush_gpu_mirror` — no manual buffer rebuild needed for the
    /// per-object data (the sorted instance/draw-call buffers are still rebuilt
    /// by `flush` for instancing).
    pub fn insert_object(&mut self, desc: ObjectDescriptor) -> Result<ObjectId> {
        let mesh_slice = {
            let mesh = self
                .mesh_pool
                .get(desc.mesh)
                .ok_or_else(|| invalid("mesh"))?;
            mesh.slice
        };
        let material_slot = {
            let (slot, material) = self
                .materials
                .get_mut_with_slot(desc.material)
                .ok_or_else(|| invalid("material"))?;
            material.ref_count += 1;
            slot
        };
        self.mesh_pool
            .get_mut(desc.mesh)
            .ok_or_else(|| invalid("mesh"))?
            .ref_count += 1;

        let movability = desc.movability.unwrap_or_default();
        let is_static = !movability.can_move();
        let user_tag = desc.user_tag;
        let entity = self.world.spawn();
        self.world.insert(entity, object_gpu_data(desc.mesh, material_slot, desc, mesh_slice));
        let id = ObjectId::from_entity(entity);

        // Index by application tag so the owner can find this object again.
        if user_tag != 0 {
            self.objects_by_tag.insert(user_tag, id);
        }

        if is_static {
            self.static_objects_dirty = true;
            self.bake_invalidated = true;
        }

        self.objects_dirty = true;

        Ok(id)
    }
}
