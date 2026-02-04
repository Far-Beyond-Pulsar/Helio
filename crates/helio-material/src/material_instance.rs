use crate::Material;
use glam::{Vec2, Vec3, Vec4};
use std::collections::HashMap;

pub struct MaterialInstance {
    pub material: Material,
    pub parameters: HashMap<String, MaterialParameter>,
}

#[derive(Debug, Clone)]
pub enum MaterialParameter {
    Float(f32),
    Vec2(Vec2),
    Vec3(Vec3),
    Vec4(Vec4),
    Texture(u32),
}

impl MaterialInstance {
    pub fn new(material: Material) -> Self {
        Self {
            material,
            parameters: HashMap::new(),
        }
    }
    
    pub fn set_parameter(&mut self, name: impl Into<String>, value: MaterialParameter) {
        self.parameters.insert(name.into(), value);
    }
    
    pub fn get_parameter(&self, name: &str) -> Option<&MaterialParameter> {
        self.parameters.get(name)
    }
}
