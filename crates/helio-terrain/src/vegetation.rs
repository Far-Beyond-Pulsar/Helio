use glam::Vec3;

pub struct Vegetation {
    pub positions: Vec<Vec3>,
    pub density: f32,
    pub lod_distances: Vec<f32>,
}

pub struct GrassSystem {
    pub blade_count: u32,
    pub blade_width: f32,
    pub blade_height: f32,
    pub wind_strength: f32,
    pub wind_speed: f32,
}

impl GrassSystem {
    pub fn new() -> Self {
        Self {
            blade_count: 1000000,
            blade_width: 0.02,
            blade_height: 0.5,
            wind_strength: 1.0,
            wind_speed: 1.0,
        }
    }
}

impl Default for GrassSystem {
    fn default() -> Self {
        Self::new()
    }
}
