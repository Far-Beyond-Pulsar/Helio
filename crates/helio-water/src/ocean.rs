use glam::Vec3;

pub struct Ocean {
    pub enabled: bool,
    pub water_level: f32,
    pub color_shallow: Vec3,
    pub color_deep: Vec3,
    pub extinction_coefficient: Vec3,
    pub roughness: f32,
    pub metallic: f32,
    pub ior: f32,
    pub foam_intensity: f32,
    pub caustics_enabled: bool,
}

impl Default for Ocean {
    fn default() -> Self {
        Self {
            enabled: true,
            water_level: 0.0,
            color_shallow: Vec3::new(0.0, 0.8, 0.9),
            color_deep: Vec3::new(0.0, 0.2, 0.4),
            extinction_coefficient: Vec3::new(0.45, 0.029, 0.018),
            roughness: 0.1,
            metallic: 0.0,
            ior: 1.333,
            foam_intensity: 0.5,
            caustics_enabled: true,
        }
    }
}
