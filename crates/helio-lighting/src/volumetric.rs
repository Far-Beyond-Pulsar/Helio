use helio_core::gpu;

pub struct VolumetricSystem {
    pub fog_density: f32,
    pub fog_height_falloff: f32,
    pub fog_inscattering_color: glam::Vec3,
    pub volumetric_texture: Option<gpu::Texture>,
}

impl VolumetricSystem {
    pub fn new() -> Self {
        Self {
            fog_density: 0.01,
            fog_height_falloff: 0.2,
            fog_inscattering_color: glam::Vec3::new(0.5, 0.6, 0.7),
            volumetric_texture: None,
        }
    }
}

impl Default for VolumetricSystem {
    fn default() -> Self {
        Self::new()
    }
}
