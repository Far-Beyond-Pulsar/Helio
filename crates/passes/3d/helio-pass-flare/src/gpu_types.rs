//! Wire contract only. Authoring and camera/volume resolution belong to PP.

pub const LENS_BLOCK_OFFSET: u64 = 464;
pub const LENS_BLOCK_SIZE: u64 = 128;
pub const POSTPROCESS_BINDING_SIZE: u64 = LENS_BLOCK_OFFSET + LENS_BLOCK_SIZE;

/// A read-only mirror of the resolved PP tail, never a second settings store.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLensResponse {
    pub enabled: u32,
    pub quality: u32,
    pub profile: u32,
    pub ghost_count: u32,
    pub intensity: f32,
    pub threshold: f32,
    pub soft_knee: f32,
    pub ghost_intensity: f32,
    pub halo_intensity: f32,
    pub glare_intensity: f32,
    pub streak_intensity: f32,
    pub dispersion: f32,
    pub aperture_f_number: f32,
    pub focal_length_mm: f32,
    pub sensor_width_mm: f32,
    pub vignette: f32,
    pub starburst_intensity: f32,
    pub starburst_length: f32,
    pub aperture_blades: u32,
    pub aperture_rotation: f32,
    pub coating_strength: f32,
    pub ghost_rim: f32,
    pub dirt_intensity: f32,
    pub light_sources: u32,
    pub light_intensity: f32,
    pub field_margin: f32,
    pub response_time: f32,
    pub _pad: [f32; 5],
}

const _: () = assert!(std::mem::size_of::<GpuLensResponse>() == LENS_BLOCK_SIZE as usize);

/// Analytic lens sources written by `cs_sources`: a count and 32 entries.
pub const MAX_LENS_SOURCES: u64 = 32;
pub const LENS_SOURCE_SIZE: u64 = 32;
pub const LENS_SOURCES_SIZE: u64 = 16 + MAX_LENS_SOURCES * LENS_SOURCE_SIZE;
