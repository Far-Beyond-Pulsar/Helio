use glam::Vec3;

pub struct AtmosphericScattering {
    pub rayleigh_coefficient: Vec3,
    pub mie_coefficient: f32,
    pub rayleigh_scale_height: f32,
    pub mie_scale_height: f32,
    pub planet_radius: f32,
    pub atmosphere_radius: f32,
    pub sun_intensity: f32,
}

impl Default for AtmosphericScattering {
    fn default() -> Self {
        Self {
            rayleigh_coefficient: Vec3::new(5.8e-6, 13.5e-6, 33.1e-6),
            mie_coefficient: 21e-6,
            rayleigh_scale_height: 8000.0,
            mie_scale_height: 1200.0,
            planet_radius: 6371000.0,
            atmosphere_radius: 6471000.0,
            sun_intensity: 22.0,
        }
    }
}

pub struct SkyDome {
    pub atmospheric_scattering: AtmosphericScattering,
    pub sun_direction: Vec3,
    pub sun_color: Vec3,
    pub moon_direction: Vec3,
    pub moon_color: Vec3,
    pub stars_intensity: f32,
}

impl Default for SkyDome {
    fn default() -> Self {
        Self {
            atmospheric_scattering: AtmosphericScattering::default(),
            sun_direction: Vec3::new(0.3, 0.7, 0.2).normalize(),
            sun_color: Vec3::ONE,
            moon_direction: Vec3::new(-0.3, 0.5, -0.2).normalize(),
            moon_color: Vec3::new(0.8, 0.8, 1.0),
            stars_intensity: 0.5,
        }
    }
}
