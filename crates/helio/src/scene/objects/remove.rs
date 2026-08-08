//! Object removal operations (SceneDB-backed).

use crate::handles::ObjectId;
use crate::scene::types::ObjectRecord;

use super::super::errors::{invalid, Result};

impl super::super::Scene {
    /// Remove an object from the scene (SceneDB-backed).
    pub fn remove_object(&mut self, id: ObjectId) -> Result<()> {
        let entity = id.entity();
        let (mesh_id, material_id, is_static, user_tag) = {
            let r = self
                .world
                .query::<(&ObjectRecord,)>()
                .find(|(e, _)| *e == entity)
                .map(|(_, (r,))| r)
                .ok_or_else(|| invalid("object"))?;
            (r.mesh, r.material, !r.movability.can_move(), r.user_tag)
        };

        self.world.despawn(entity);

        if user_tag != 0 && self.objects_by_tag.get(&user_tag) == Some(&id) {
            self.objects_by_tag.remove(&user_tag);
        }

        if let Some(material) = self
            .materials
            .get_mut_with_slot(material_id)
            .map(|(_, m)| m)
        {
            material.ref_count = material.ref_count.saturating_sub(1);
        }
        if let Some(mesh) = self.mesh_pool.get_mut(mesh_id) {
            mesh.ref_count = mesh.ref_count.saturating_sub(1);
        }

        self.objects_dirty = true;

        if self
            .mesh_pool
            .get(mesh_id)
            .map_or(false, |r| r.ref_count == 0)
        {
            let _ = self.remove_mesh(mesh_id);
        }
        if self
            .materials
            .get(material_id)
            .map_or(false, |r| r.ref_count == 0)
        {
            let _ = self.remove_material(material_id);
        }

        if is_static {
            self.static_objects_dirty = true;
        }

        Ok(())
    }
}
