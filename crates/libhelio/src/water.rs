//! Water volume and hitbox GPU types.
//!
//! This module defines the GPU-side representations for:
//! - [`GpuWaterVolume`]: per-volume rendering parameters (waves, caustics, etc.)
//! - [`GpuWaterHitbox`]: per-frame AABB hitbox that displaces the water heightfield

use bytemuck::{Pod, Zeroable};

/// GPU water volume descriptor (256 bytes, 16-byte aligned).
///
/// Defines a water volume's bounds, wave parameters, visual properties,
/// and rendering settings. Stored in GPU storage buffers for efficient
/// access by water rendering shaders.
///
/// # Memory Layout
/// - Total size: 256 bytes (16 × vec4<f32>)
/// - Alignment: 16 bytes
///
/// # Field layout (slot → name → WGSL vec4 index)
/// 0  bounds_min              → vec4 containing (x,y,z,_)
/// 1  bounds_max              → vec4 containing (x,y,z,_)
/// 2  wave_params             → (amplitude, frequency, speed, steepness)
/// 3  wave_direction          → (dx, dz, _, _)
/// 4  water_color             → (r, g, b, foam_threshold)
/// 5  extinction              → (r, g, b, foam_amount)  [Beer-Lambert per-channel]
/// 6  reflection_refraction   → (refl_strength, refr_strength, fresnel_power, _)
/// 7  caustics_params         → (enabled, intensity, scale, speed)
/// 8  fog_params              → (density, god_rays, _, _)
/// 9  sim_params              → (ior, caustic_intensity, fresnel_min, density)
/// 10 shadow_params           → (rim, hitbox_shadow, ao, _)
/// 11 sun_direction           → (dx, dy, dz, _)
/// 12..15  _pad3.._pad6       → reserved
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuWaterVolume {
    /// Minimum bounds (xyz) + padding
    pub bounds_min: [f32; 4],

    /// Maximum bounds (xyz) + surface height in w component
    pub bounds_max: [f32; 4],

    /// Wave parameters: (amplitude, frequency, speed, steepness)
    pub wave_params: [f32; 4],

    /// Wave direction (dx, dz) + padding
    pub wave_direction: [f32; 4],

    /// Water base color (rgb) + foam_threshold (w)
    pub water_color: [f32; 4],

    /// Color absorption per meter (rgb) + foam_amount (w)  [Beer-Lambert]
    pub extinction: [f32; 4],

    /// Reflection strength, refraction strength, fresnel power, padding
    pub reflection_refraction: [f32; 4],

    /// Caustics: enabled (0/1), intensity, scale, speed
    pub caustics_params: [f32; 4],

    /// Fog density, god rays intensity, padding, padding
    pub fog_params: [f32; 4],

    /// Heightfield simulation surface params: (ior, caustic_intensity, fresnel_min, density)
    pub sim_params: [f32; 4],

    /// Shadow parameters: (rim_light, hitbox_shadow, ambient_occlusion, padding)
    pub shadow_params: [f32; 4],

    /// Sun/dominant light direction (dx, dy, dz) + padding
    pub sun_direction: [f32; 4],

    pub _pad3: [f32; 4],
    pub _pad4: [f32; 4],
    pub _pad5: [f32; 4],
    pub _pad6: [f32; 4],
}

impl GpuWaterVolume {
    /// Ocean preset with heightfield sim defaults.
    pub fn default_ocean() -> Self {
        Self {
            bounds_min: [-100.0, -10.0, -100.0, 0.0],
            bounds_max: [100.0, 50.0, 100.0, 0.0],
            wave_params: [0.5, 0.3, 1.5, 0.5],
            wave_direction: [1.0, 0.0, 0.0, 0.0],
            water_color: [0.0, 0.2, 0.4, 0.8],
            extinction: [0.1, 0.05, 0.02, 0.6],
            reflection_refraction: [0.8, 0.2, 5.0, 0.0],
            caustics_params: [1.0, 1.5, 5.0, 0.5],
            fog_params: [0.03, 1.0, 0.0, 0.0],
            // ior=1.333, caustic_intensity=1.5, fresnel_min=0.1, density=0.03
            sim_params: [1.333, 1.5, 0.1, 0.03],
            // rim=1.0, hitbox_shadow=0.0, ao=1.0
            shadow_params: [1.0, 0.0, 1.0, 0.0],
            // normalized sun direction
            sun_direction: [0.408_248_3, 0.816_496_6, 0.408_248_3, 0.0],
            _pad3: [0.0; 4],
            _pad4: [0.0; 4],
            _pad5: [0.0; 4],
            _pad6: [0.0; 4],
        }
    }

    /// Lake/pool preset with heightfield sim defaults.
    pub fn default_lake() -> Self {
        Self {
            bounds_min: [-50.0, -5.0, -50.0, 0.0],
            bounds_max: [50.0, 20.0, 50.0, 0.0],
            wave_params: [0.2, 0.5, 0.8, 0.3],
            wave_direction: [1.0, 0.0, 0.0, 0.0],
            water_color: [0.1, 0.3, 0.2, 0.7],
            extinction: [0.2, 0.1, 0.08, 0.5],
            reflection_refraction: [0.6, 0.3, 4.0, 0.0],
            caustics_params: [1.0, 1.2, 4.0, 0.4],
            fog_params: [0.05, 0.5, 0.0, 0.0],
            sim_params: [1.333, 1.2, 0.1, 0.05],
            shadow_params: [1.0, 0.0, 1.0, 0.0],
            sun_direction: [0.408_248_3, 0.816_496_6, 0.408_248_3, 0.0],
            _pad3: [0.0; 4],
            _pad4: [0.0; 4],
            _pad5: [0.0; 4],
            _pad6: [0.0; 4],
        }
    }
}

// Compile-time size verification
const _: () = assert!(
    std::mem::size_of::<GpuWaterVolume>() == 256,
    "GpuWaterVolume must be exactly 256 bytes"
);

const _: () = assert!(
    std::mem::align_of::<GpuWaterVolume>() <= 16,
    "GpuWaterVolume alignment must be 16 bytes or less for GPU compatibility"
);

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Water Hitbox
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// GPU-side AABB hitbox that displaces the water heightfield simulation.
///
/// Each hitbox records where an object *was* (old bounds) and where it *is* (new bounds).
/// The simulation shader displaces water upward where the hitbox moved away and
/// downward where it moved to, producing realistic wave entry effects.
///
/// # Memory Layout
/// - Total size: 80 bytes (5 × vec4<f32>)
/// - Alignment: 16 bytes
///
/// # Algorithm
/// The displacement uses a smooth Gaussian-like falloff within the AABB:
/// - Rise = volume_in_box(old_center, old_half_extent, position)
/// - Fall = volume_in_box(new_center, new_half_extent, position)
/// - delta_height += strength * (rise - fall)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuWaterHitbox {
    /// Previous frame AABB minimum (xyz) + padding
    pub old_min: [f32; 4],
    /// Previous frame AABB maximum (xyz) + padding
    pub old_max: [f32; 4],
    /// Current frame AABB minimum (xyz) + padding
    pub new_min: [f32; 4],
    /// Current frame AABB maximum (xyz) + padding
    pub new_max: [f32; 4],
    /// (edge_softness, strength, padding, padding)
    /// - edge_softness: controls Gaussian falloff at AABB edges (0.5 = sharp, 2.0 = very soft)
    /// - strength: displacement multiplier (default 1.0)
    pub params: [f32; 4],
}

const _: () = assert!(
    std::mem::size_of::<GpuWaterHitbox>() == 80,
    "GpuWaterHitbox must be exactly 80 bytes"
);
