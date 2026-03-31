//! Water volume GPU types and utilities.
//!
//! This module defines the GPU-side representation of water volumes for
//! realistic water rendering with waves, caustics, and underwater effects.

use bytemuck::{Pod, Zeroable};

/// GPU water volume descriptor (256 bytes, 16-byte aligned).
///
/// Defines a water volume's bounds, wave parameters, visual properties,
/// and rendering settings. Stored in GPU storage buffers for efficient
/// access by water rendering shaders.
///
/// # Memory Layout
/// - Total size: 256 bytes
/// - Alignment: 16 bytes (vec4<f32> in WGSL)
/// - Padding: Explicit padding to meet alignment requirements
///
/// # Fields Organization
/// - `bounds_min/max`: AABB defining water volume extents
/// - `wave_params`: Gerstner wave parameters (amplitude, frequency, speed, steepness)
/// - `water_color`: Base water color and foam settings
/// - `extinction`: Beer-Lambert absorption coefficients per wavelength
/// - `reflection_refraction`: Surface rendering parameters
/// - `caustics_params`: Caustics generation settings
/// - `fog_params`: Underwater volumetric fog settings
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuWaterVolume {
    /// Minimum bounds (xyz) + padding
    pub bounds_min: [f32; 4],

    /// Maximum bounds (xyz) + surface height in w component
    pub bounds_max: [f32; 4],

    /// Wave parameters: (amplitude, frequency, speed, steepness)
    pub wave_params: [f32; 4],

    /// Wave direction (xy) + padding
    pub wave_direction: [f32; 4],

    /// Water base color (rgb) + foam_threshold (w)
    pub water_color: [f32; 4],

    /// Color absorption per meter (rgb) + foam_amount (w)
    pub extinction: [f32; 4],

    /// Reflection strength, refraction strength, fresnel power, padding
    pub reflection_refraction: [f32; 4],

    /// Caustics: enabled (0/1), intensity, scale, speed
    pub caustics_params: [f32; 4],

    /// Fog density, god rays intensity, padding, padding
    pub fog_params: [f32; 4],

    // Explicit padding to 256 bytes (9 used + 7 padding = 16 vec4s = 256 bytes)
    pub _pad0: [f32; 4],
    pub _pad1: [f32; 4],
    pub _pad2: [f32; 4],
    pub _pad3: [f32; 4],
    pub _pad4: [f32; 4],
    pub _pad5: [f32; 4],
    pub _pad6: [f32; 4],
}

impl GpuWaterVolume {
    /// Creates a default GPU water volume with typical ocean parameters.
    ///
    /// # Returns
    /// A water volume configured for a realistic ocean surface with:
    /// - Medium amplitude waves (0.5m)
    /// - Blue-green water color
    /// - Moderate extinction (clearer water)
    /// - Enabled caustics
    /// - Moderate fog density
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
            _pad0: [0.0; 4],
            _pad1: [0.0; 4],
            _pad2: [0.0; 4],
            _pad3: [0.0; 4],
            _pad4: [0.0; 4],
            _pad5: [0.0; 4],
            _pad6: [0.0; 4],
        }
    }

    /// Creates a default GPU water volume with typical lake parameters.
    ///
    /// # Returns
    /// A water volume configured for a calm lake with:
    /// - Small amplitude waves (0.2m)
    /// - Green-tinted water color
    /// - Higher extinction (murkier water)
    /// - Disabled caustics (not needed for calm water)
    /// - Light fog
    pub fn default_lake() -> Self {
        Self {
            bounds_min: [-50.0, -5.0, -50.0, 0.0],
            bounds_max: [50.0, 20.0, 50.0, 0.0],
            wave_params: [0.2, 0.5, 0.8, 0.3],
            wave_direction: [1.0, 0.0, 0.0, 0.0],
            water_color: [0.1, 0.3, 0.2, 0.7],
            extinction: [0.2, 0.1, 0.08, 0.5],
            reflection_refraction: [0.6, 0.3, 4.0, 0.0],
            caustics_params: [0.0, 0.0, 0.0, 0.0],
            fog_params: [0.05, 0.5, 0.0, 0.0],
            _pad0: [0.0; 4],
            _pad1: [0.0; 4],
            _pad2: [0.0; 4],
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
