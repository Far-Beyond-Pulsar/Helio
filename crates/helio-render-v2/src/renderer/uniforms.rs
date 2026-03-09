//! GPU uniform types uploaded each frame.

/// Globals uniform data
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct GlobalsUniform {
    pub frame: u32,
    pub delta_time: f32,
    pub light_count: u32,
    pub ambient_intensity: f32,
    pub ambient_color: [f32; 4],  // w unused, ensures alignment
    pub rc_world_min: [f32; 4],   // xyz = RC probe grid world AABB min, w unused
    pub rc_world_max: [f32; 4],   // xyz = RC probe grid world AABB max, w unused
    /// View-space distance at each CSM cascade boundary (4 cascades → 3 splits + sentinel).
    pub csm_splits: [f32; 4],
}

/// Sky uniform data – 112 bytes, must exactly match SkyUniforms in sky.wgsl
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct SkyUniform {
    pub sun_direction:      [f32; 3],  // offset   0 (12 bytes)
    pub sun_intensity:      f32,       // offset  12
    pub rayleigh_scatter:   [f32; 3],  // offset  16
    pub rayleigh_h_scale:   f32,       // offset  28
    pub mie_scatter:        f32,       // offset  32
    pub mie_h_scale:        f32,       // offset  36
    pub mie_g:              f32,       // offset  40
    pub sun_disk_cos:       f32,       // offset  44
    pub earth_radius:       f32,       // offset  48
    pub atm_radius:         f32,       // offset  52
    pub exposure:           f32,       // offset  56
    pub clouds_enabled:     u32,       // offset  60
    pub cloud_coverage:     f32,       // offset  64
    pub cloud_density:      f32,       // offset  68
    pub cloud_base:         f32,       // offset  72
    pub cloud_top:          f32,       // offset  76
    pub cloud_wind_x:       f32,       // offset  80
    pub cloud_wind_z:       f32,       // offset  84
    pub cloud_speed:        f32,       // offset  88
    pub time_sky:           f32,       // offset  92
    pub skylight_intensity: f32,       // offset  96
    pub _pad0:              f32,       // offset 100
    pub _pad1:              f32,       // offset 104
    pub _pad2:              f32,       // offset 108 → total 112 (multiple of 16)
}

/// GPU shadow light-space matrix (must match WGSL LightMatrix struct = 64 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuShadowMatrix {
    pub mat: [f32; 16],
}
