use glam::Vec3;

pub struct RayTracingConfig {
    pub enabled: bool,
    pub max_ray_depth: u32,
    pub samples_per_pixel: u32,
    pub use_hardware_rt: bool,
}

impl Default for RayTracingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_ray_depth: 4,
            samples_per_pixel: 1,
            use_hardware_rt: true,
        }
    }
}

pub struct RayTracedShadows {
    pub enabled: bool,
    pub samples: u32,
    pub max_distance: f32,
}

pub struct RayTracedReflections {
    pub enabled: bool,
    pub max_bounces: u32,
    pub samples: u32,
    pub denoiser_enabled: bool,
}

pub struct RayTracedAmbientOcclusion {
    pub enabled: bool,
    pub radius: f32,
    pub samples: u32,
}

pub struct RayTracedGlobalIllumination {
    pub enabled: bool,
    pub bounces: u32,
    pub samples: u32,
    pub irradiance_cache: bool,
}

impl Default for RayTracedShadows {
    fn default() -> Self {
        Self {
            enabled: false,
            samples: 1,
            max_distance: 100.0,
        }
    }
}

impl Default for RayTracedReflections {
    fn default() -> Self {
        Self {
            enabled: false,
            max_bounces: 1,
            samples: 1,
            denoiser_enabled: true,
        }
    }
}

impl Default for RayTracedAmbientOcclusion {
    fn default() -> Self {
        Self {
            enabled: false,
            radius: 1.0,
            samples: 1,
        }
    }
}

impl Default for RayTracedGlobalIllumination {
    fn default() -> Self {
        Self {
            enabled: false,
            bounces: 1,
            samples: 1,
            irradiance_cache: true,
        }
    }
}
