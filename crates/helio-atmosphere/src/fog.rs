use glam::Vec3;

pub struct AtmosphericFog {
    pub enabled: bool,
    pub density: f32,
    pub height_falloff: f32,
    pub start_distance: f32,
    pub inscattering_color: Vec3,
    pub directional_inscattering: f32,
    pub directional_exponent: f32,
}

impl Default for AtmosphericFog {
    fn default() -> Self {
        Self {
            enabled: true,
            density: 0.02,
            height_falloff: 0.2,
            start_distance: 0.0,
            inscattering_color: Vec3::new(0.447, 0.639, 1.0),
            directional_inscattering: 0.5,
            directional_exponent: 4.0,
        }
    }
}
