use glam::{Vec3, UVec3};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GITechnique {
    None,
    Lightmaps,
    LightProbes,
    SSGI,
    RTGI,
    DDGI,
    Lumen,
    SVGF,
    RTXGI,
    Voxel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GIProbeResolution {
    Low,
    Medium,
    High,
    Ultra,
}

pub struct GlobalIllumination {
    pub technique: GITechnique,
    pub enabled: bool,
    pub intensity: f32,
    pub bounce_count: u32,
    pub quality: f32,
    pub probe_count: UVec3,
    pub probe_spacing: Vec3,
}

impl GlobalIllumination {
    pub fn new(resolution: GIProbeResolution) -> Self {
        let probe_count = match resolution {
            GIProbeResolution::Low => UVec3::new(8, 4, 8),
            GIProbeResolution::Medium => UVec3::new(16, 8, 16),
            GIProbeResolution::High => UVec3::new(32, 16, 32),
            GIProbeResolution::Ultra => UVec3::new(64, 32, 64),
        };
        
        Self {
            technique: GITechnique::DDGI,
            enabled: true,
            intensity: 1.0,
            bounce_count: 1,
            quality: 1.0,
            probe_count,
            probe_spacing: Vec3::splat(2.0),
        }
    }
}

impl Default for GlobalIllumination {
    fn default() -> Self {
        Self::new(GIProbeResolution::Medium)
    }
}

pub struct DDGI {
    pub probe_spacing: Vec3,
    pub probe_count: (u32, u32, u32),
    pub rays_per_probe: u32,
    pub irradiance_resolution: u32,
    pub visibility_resolution: u32,
    pub hysteresis: f32,
}

impl DDGI {
    pub fn new() -> Self {
        Self {
            probe_spacing: Vec3::splat(2.0),
            probe_count: (16, 8, 16),
            rays_per_probe: 256,
            irradiance_resolution: 8,
            visibility_resolution: 16,
            hysteresis: 0.98,
        }
    }
}

impl Default for DDGI {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Lumen {
    pub screen_trace_quality: f32,
    pub final_gather_quality: f32,
    pub scene_lighting_update_speed: f32,
    pub max_trace_distance: f32,
    pub software_raytracing: bool,
    pub hardware_raytracing: bool,
}

impl Lumen {
    pub fn new() -> Self {
        Self {
            screen_trace_quality: 1.0,
            final_gather_quality: 1.0,
            scene_lighting_update_speed: 1.0,
            max_trace_distance: 100.0,
            software_raytracing: true,
            hardware_raytracing: false,
        }
    }
}

impl Default for Lumen {
    fn default() -> Self {
        Self::new()
    }
}

pub struct VoxelGI {
    pub voxel_size: f32,
    pub grid_dimensions: (u32, u32, u32),
    pub mipmap_levels: u32,
    pub cone_tracing_steps: u32,
}

impl VoxelGI {
    pub fn new() -> Self {
        Self {
            voxel_size: 0.5,
            grid_dimensions: (128, 128, 128),
            mipmap_levels: 6,
            cone_tracing_steps: 16,
        }
    }
}

impl Default for VoxelGI {
    fn default() -> Self {
        Self::new()
    }
}
