use glam::Vec3;

pub struct Bloom {
    pub enabled: bool,
    pub threshold: f32,
    pub intensity: f32,
    pub scatter: f32,
    pub radius: f32,
    pub iterations: u32,
    pub tint: Vec3,
}

impl Default for Bloom {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: 1.0,
            intensity: 0.2,
            scatter: 0.7,
            radius: 2.0,
            iterations: 5,
            tint: Vec3::ONE,
        }
    }
}
