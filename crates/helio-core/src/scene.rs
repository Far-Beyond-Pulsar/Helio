use crate::{Camera, Transform, Aabb};
use std::collections::HashMap;

pub type EntityId = u64;

#[derive(Clone, Debug)]
pub struct Entity {
    pub id: EntityId,
    pub transform: Transform,
    pub mesh: Option<crate::MeshHandle>,
    pub material: Option<u32>,
    pub visible: bool,
    pub cast_shadows: bool,
    pub receive_shadows: bool,
}

impl Entity {
    pub fn new(id: EntityId) -> Self {
        Self {
            id,
            transform: Transform::default(),
            mesh: None,
            material: None,
            visible: true,
            cast_shadows: true,
            receive_shadows: true,
        }
    }
}

pub struct Scene {
    pub camera: Camera,
    pub entities: HashMap<EntityId, Entity>,
    pub bounds: Aabb,
    next_entity_id: EntityId,
}

impl Scene {
    pub fn new(camera: Camera) -> Self {
        Self {
            camera,
            entities: HashMap::new(),
            bounds: Aabb::default(),
            next_entity_id: 1,
        }
    }

    pub fn add_entity(&mut self, entity: Entity) -> EntityId {
        let id = entity.id;
        self.entities.insert(id, entity);
        self.update_bounds();
        id
    }

    pub fn create_entity(&mut self) -> EntityId {
        let id = self.next_entity_id;
        self.next_entity_id += 1;
        let entity = Entity::new(id);
        self.add_entity(entity)
    }

    pub fn remove_entity(&mut self, id: EntityId) -> Option<Entity> {
        let entity = self.entities.remove(&id);
        if entity.is_some() {
            self.update_bounds();
        }
        entity
    }

    pub fn get_entity(&self, id: EntityId) -> Option<&Entity> {
        self.entities.get(&id)
    }

    pub fn get_entity_mut(&mut self, id: EntityId) -> Option<&mut Entity> {
        self.entities.get_mut(&id)
    }

    pub fn visible_entities(&self) -> impl Iterator<Item = &Entity> {
        self.entities.values().filter(|e| e.visible)
    }

    fn update_bounds(&mut self) {
        if self.entities.is_empty() {
            self.bounds = Aabb::default();
            return;
        }

        let mut min = glam::Vec3::splat(f32::MAX);
        let mut max = glam::Vec3::splat(f32::MIN);

        for entity in self.entities.values() {
            let pos = entity.transform.position;
            min = min.min(pos);
            max = max.max(pos);
        }

        self.bounds = Aabb::new(min, max);
    }

    pub fn clear(&mut self) {
        self.entities.clear();
        self.bounds = Aabb::default();
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new(Camera::default())
    }
}
