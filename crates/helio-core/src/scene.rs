use crate::{Camera, Transform};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

pub type EntityId = u64;

#[derive(Debug, Clone)]
pub struct Entity {
    pub id: EntityId,
    pub transform: Transform,
    pub visible: bool,
    pub cast_shadows: bool,
    pub receive_shadows: bool,
    pub static_object: bool,
    pub lod_bias: f32,
}

impl Entity {
    pub fn new(id: EntityId) -> Self {
        Self {
            id,
            transform: Transform::default(),
            visible: true,
            cast_shadows: true,
            receive_shadows: true,
            static_object: false,
            lod_bias: 1.0,
        }
    }
}

pub struct Scene {
    entities: Arc<RwLock<HashMap<EntityId, Entity>>>,
    next_entity_id: Arc<RwLock<EntityId>>,
    pub camera: Camera,
    pub ambient_light: glam::Vec3,
    pub environment_intensity: f32,
    pub fog_enabled: bool,
    pub fog_color: glam::Vec3,
    pub fog_density: f32,
    pub fog_start: f32,
    pub fog_end: f32,
}

impl Scene {
    pub fn new(camera: Camera) -> Self {
        Self {
            entities: Arc::new(RwLock::new(HashMap::new())),
            next_entity_id: Arc::new(RwLock::new(0)),
            camera,
            ambient_light: glam::Vec3::splat(0.03),
            environment_intensity: 1.0,
            fog_enabled: false,
            fog_color: glam::Vec3::new(0.5, 0.6, 0.7),
            fog_density: 0.02,
            fog_start: 10.0,
            fog_end: 100.0,
        }
    }
    
    pub fn add_entity(&mut self, mut entity: Entity) -> EntityId {
        let mut next_id = self.next_entity_id.write();
        entity.id = *next_id;
        *next_id += 1;
        
        let id = entity.id;
        self.entities.write().insert(id, entity);
        id
    }
    
    pub fn remove_entity(&mut self, id: EntityId) -> Option<Entity> {
        self.entities.write().remove(&id)
    }
    
    pub fn get_entity(&self, id: EntityId) -> Option<Entity> {
        self.entities.read().get(&id).cloned()
    }
    
    pub fn get_entity_mut<F, R>(&self, id: EntityId, f: F) -> Option<R>
    where
        F: FnOnce(&mut Entity) -> R,
    {
        let mut entities = self.entities.write();
        entities.get_mut(&id).map(f)
    }
    
    pub fn for_each_entity<F>(&self, mut f: F)
    where
        F: FnMut(&Entity),
    {
        let entities = self.entities.read();
        for entity in entities.values() {
            f(entity);
        }
    }
    
    pub fn visible_entities(&self) -> Vec<Entity> {
        self.entities
            .read()
            .values()
            .filter(|e| e.visible)
            .cloned()
            .collect()
    }
    
    pub fn entity_count(&self) -> usize {
        self.entities.read().len()
    }
}
