use helio_core::gpu;
use glam::Vec3;

pub struct ReflectionProbe {
    pub position: Vec3,
    pub extent: Vec3,
    pub cubemap: Option<gpu::Texture>,
    pub importance: f32,
}

pub struct ProbeSystem {
    pub probes: Vec<ReflectionProbe>,
}

impl ProbeSystem {
    pub fn new() -> Self {
        Self {
            probes: Vec::new(),
        }
    }
}

impl Default for ProbeSystem {
    fn default() -> Self {
        Self::new()
    }
}
