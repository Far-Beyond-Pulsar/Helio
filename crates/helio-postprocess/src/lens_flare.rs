use glam::Vec3;

pub struct LensFlare {
    pub enabled: bool,
    pub intensity: f32,
    pub ghosts: u32,
    pub ghost_dispersal: f32,
    pub halo_width: f32,
    pub chromatic_distortion: f32,
    pub tint: Vec3,
}

impl Default for LensFlare {
    fn default() -> Self {
        Self {
            enabled: false,
            intensity: 1.0,
            ghosts: 8,
            ghost_dispersal: 0.3,
            halo_width: 0.4,
            chromatic_distortion: 2.0,
            tint: Vec3::ONE,
        }
    }
}
