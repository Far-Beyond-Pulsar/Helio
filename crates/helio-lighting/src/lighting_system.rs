use crate::*;

pub struct LightingSystem {
    pub directional_lights: Vec<DirectionalLight>,
    pub point_lights: Vec<PointLight>,
    pub spot_lights: Vec<SpotLight>,
    pub area_lights: Vec<AreaLight>,
    pub mode: LightingMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightingMode {
    Forward,
    Deferred,
    ForwardPlus,
}

impl LightingSystem {
    pub fn new(mode: LightingMode) -> Self {
        Self {
            directional_lights: Vec::new(),
            point_lights: Vec::new(),
            spot_lights: Vec::new(),
            area_lights: Vec::new(),
            mode,
        }
    }
    
    pub fn add_directional_light(&mut self, light: DirectionalLight) {
        self.directional_lights.push(light);
    }
    
    pub fn add_point_light(&mut self, light: PointLight) {
        self.point_lights.push(light);
    }
    
    pub fn add_spot_light(&mut self, light: SpotLight) {
        self.spot_lights.push(light);
    }
    
    pub fn add_area_light(&mut self, light: AreaLight) {
        self.area_lights.push(light);
    }
    
    pub fn total_light_count(&self) -> usize {
        self.directional_lights.len() 
            + self.point_lights.len() 
            + self.spot_lights.len() 
            + self.area_lights.len()
    }
}
