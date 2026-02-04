use glam::{Vec3, Mat4};
use helio_core::TextureCube;

pub struct ReflectionProbe {
    pub position: Vec3,
    pub cubemap: Option<u32>,
    pub radius: f32,
    pub intensity: f32,
    pub box_projection: bool,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
}

impl ReflectionProbe {
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            cubemap: None,
            radius: 10.0,
            intensity: 1.0,
            box_projection: false,
            bounds_min: position - Vec3::splat(5.0),
            bounds_max: position + Vec3::splat(5.0),
        }
    }
    
    pub fn with_box_projection(mut self, bounds_min: Vec3, bounds_max: Vec3) -> Self {
        self.box_projection = true;
        self.bounds_min = bounds_min;
        self.bounds_max = bounds_max;
        self
    }
}

pub struct ReflectionProbeSystem {
    pub probes: Vec<ReflectionProbe>,
    pub resolution: u32,
    pub enable_parallax_correction: bool,
}

impl ReflectionProbeSystem {
    pub fn new() -> Self {
        Self {
            probes: Vec::new(),
            resolution: 256,
            enable_parallax_correction: true,
        }
    }
    
    pub fn add_probe(&mut self, probe: ReflectionProbe) {
        self.probes.push(probe);
    }
}

impl Default for ReflectionProbeSystem {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ScreenSpaceReflections {
    pub enabled: bool,
    pub max_steps: u32,
    pub binary_search_steps: u32,
    pub max_distance: f32,
    pub thickness: f32,
    pub fallback_to_cubemap: bool,
}

impl ScreenSpaceReflections {
    pub fn new() -> Self {
        Self {
            enabled: true,
            max_steps: 64,
            binary_search_steps: 8,
            max_distance: 50.0,
            thickness: 0.5,
            fallback_to_cubemap: true,
        }
    }
}

impl Default for ScreenSpaceReflections {
    fn default() -> Self {
        Self::new()
    }
}
