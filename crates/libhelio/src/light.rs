//! GPU light types.

use bytemuck::{Pod, Zeroable};

/// GPU light type discriminant.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightType {
    Directional = 0,
    Point = 1,
    Spot = 2,
    Area = 3,
}

/// Per-light GPU data. 80 bytes.
///
/// # WGSL equivalent
/// ```wgsl
/// struct GpuLight {
///     position:       vec4<f32>,  // xyz = position, w = range
///     direction:      vec4<f32>,  // xyz = direction, w = spot outer angle cos
///     color:          vec4<f32>,  // xyz = linear RGB, w = intensity
///     shadow_index:   u32,        // -1 if no shadow
///     light_type:     u32,        // LightType enum
///     inner_angle:    f32,        // spot inner angle cos
///     _pad:           u32,
/// }
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuLight {
    /// World-space position (xyz) + effective range (w)
    pub position_range: [f32; 4],
    /// Direction (xyz, normalized) + spot outer cos angle (w)
    pub direction_outer: [f32; 4],
    /// Linear RGB color (xyz) + intensity (w, in candela for point/spot, lux for directional)
    pub color_intensity: [f32; 4],
    /// Shadow map slice index (-1u32 = no shadow)
    pub shadow_index: u32,
    /// LightType discriminant
    pub light_type: u32,
    /// Spot inner cos angle
    pub inner_angle: f32,
    pub _pad: u32,
}

/// Per-light shadow matrix for the shadow map atlas.
/// One matrix per shadow-casting light (directional: 4 cascades, spot/point: 1–6 faces).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuShadowMatrix {
    /// Light-space view-projection matrix
    pub light_view_proj: [f32; 16],
    /// Shadow atlas UV rect (x, y, width, height) in [0,1]
    pub atlas_rect: [f32; 4],
    /// Depth bias + normal bias + cascade split (w) + padding
    pub bias_split: [f32; 4],
}
