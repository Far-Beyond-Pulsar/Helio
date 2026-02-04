use helio_core::gpu;
use glam::Vec3;

pub struct DDGIVolume {
    pub position: Vec3,
    pub extents: Vec3,
    pub probe_counts: [u32; 3],
    pub irradiance_texture: Option<gpu::Texture>,
    pub depth_texture: Option<gpu::Texture>,
}

pub struct GISystem {
    pub volumes: Vec<DDGIVolume>,
    pub enabled: bool,
    pub intensity: f32,
}

impl GISystem {
    pub fn new() -> Self {
        Self {
            volumes: Vec::new(),
            enabled: true,
            intensity: 1.0,
        }
    }
}

impl Default for GISystem {
    fn default() -> Self {
        Self::new()
    }
}
