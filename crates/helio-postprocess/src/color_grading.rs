use glam::{Vec3, Vec4};

pub struct ColorGrading {
    pub enabled: bool,
    pub temperature: f32,
    pub tint: f32,
    pub saturation: f32,
    pub contrast: f32,
    pub gamma: f32,
    pub gain: Vec3,
    pub offset: Vec3,
    pub shadows: Vec3,
    pub midtones: Vec3,
    pub highlights: Vec3,
    pub shadows_max: f32,
    pub highlights_min: f32,
}

impl Default for ColorGrading {
    fn default() -> Self {
        Self {
            enabled: true,
            temperature: 0.0,
            tint: 0.0,
            saturation: 1.0,
            contrast: 1.0,
            gamma: 1.0,
            gain: Vec3::ONE,
            offset: Vec3::ZERO,
            shadows: Vec3::ONE,
            midtones: Vec3::ONE,
            highlights: Vec3::ONE,
            shadows_max: 0.3,
            highlights_min: 0.7,
        }
    }
}

pub struct LUT {
    pub texture: Option<u32>,
    pub intensity: f32,
}

impl Default for LUT {
    fn default() -> Self {
        Self {
            texture: None,
            intensity: 1.0,
        }
    }
}
