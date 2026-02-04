use glam::Vec3;

pub struct VolumetricClouds {
    pub enabled: bool,
    pub coverage: f32,
    pub density: f32,
    pub altitude_min: f32,
    pub altitude_max: f32,
    pub wind_speed: Vec3,
    pub detail_scale: f32,
    pub erosion_scale: f32,
    pub march_steps: u32,
    pub light_steps: u32,
}

impl Default for VolumetricClouds {
    fn default() -> Self {
        Self {
            enabled: true,
            coverage: 0.5,
            density: 0.3,
            altitude_min: 1500.0,
            altitude_max: 4000.0,
            wind_speed: Vec3::new(10.0, 0.0, 5.0),
            detail_scale: 0.5,
            erosion_scale: 0.3,
            march_steps: 64,
            light_steps: 6,
        }
    }
}
