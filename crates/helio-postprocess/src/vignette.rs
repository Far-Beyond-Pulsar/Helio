use glam::Vec3;

pub struct Vignette {
    pub enabled: bool,
    pub intensity: f32,
    pub smoothness: f32,
    pub roundness: f32,
    pub color: Vec3,
}

impl Default for Vignette {
    fn default() -> Self {
        Self {
            enabled: false,
            intensity: 0.4,
            smoothness: 0.5,
            roundness: 1.0,
            color: Vec3::ZERO,
        }
    }
}
