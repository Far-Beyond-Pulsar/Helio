use glam::Vec3;

pub struct VolumetricLighting {
    pub enabled: bool,
    pub scattering: f32,
    pub extinction: f32,
    pub anisotropy: f32,
    pub slice_count: u32,
    pub sample_count: u32,
    pub temporal_reprojection: bool,
}

impl VolumetricLighting {
    pub fn new() -> Self {
        Self {
            enabled: true,
            scattering: 0.1,
            extinction: 0.01,
            anisotropy: 0.3,
            slice_count: 64,
            sample_count: 8,
            temporal_reprojection: true,
        }
    }
}

impl Default for VolumetricLighting {
    fn default() -> Self {
        Self::new()
    }
}

pub struct VolumetricFog {
    pub enabled: bool,
    pub density: f32,
    pub height_falloff: f32,
    pub max_opacity: f32,
    pub start_distance: f32,
    pub albedo: Vec3,
}

impl VolumetricFog {
    pub fn new() -> Self {
        Self {
            enabled: true,
            density: 0.01,
            height_falloff: 0.1,
            max_opacity: 1.0,
            start_distance: 0.0,
            albedo: Vec3::splat(0.5),
        }
    }
}

impl Default for VolumetricFog {
    fn default() -> Self {
        Self::new()
    }
}
