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
    pub entities: Vec<Entity>,
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
            entities: Vec::new(),
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
    
    pub fn add_entity(&mut self, entity: Entity) {
        self.entities.push(entity);
    }
    
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }
}
