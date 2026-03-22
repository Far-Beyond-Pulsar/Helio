//! GPU material types. Must match `helio-render-v2` layout for asset compat.

use bytemuck::{Pod, Zeroable};

/// Material workflow discriminant.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialWorkflow {
    Metallic = 0,
    Specular = 1,
}

/// GPU material data. 96 bytes (matches helio-render-v2).
///
/// All texture indices reference the global bindless texture array.
/// If a texture is not present, the index is u32::MAX.
///
/// # WGSL equivalent
/// ```wgsl
/// struct GpuMaterial {
///     base_color:           vec4<f32>,
///     emissive:             vec4<f32>,
///     roughness_metallic:   vec4<f32>,   // x=roughness, y=metallic, z=ior, w=specular
///     tex_base_color:       u32,
///     tex_normal:           u32,
///     tex_roughness:        u32,
///     tex_emissive:         u32,
///     tex_occlusion:        u32,
///     workflow:             u32,
///     flags:                u32,
///     _pad:                 u32,
/// }
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuMaterial {
    /// Base color (RGBA linear)
    pub base_color: [f32; 4],
    /// Emissive color (RGB) + unused (w)
    pub emissive: [f32; 4],
    /// x=roughness, y=metallic/specular_strength, z=IOR, w=specular_tint
    pub roughness_metallic: [f32; 4],
    /// Bindless texture indices
    pub tex_base_color: u32,
    pub tex_normal: u32,
    pub tex_roughness: u32,
    pub tex_emissive: u32,
    pub tex_occlusion: u32,
    /// MaterialWorkflow discriminant
    pub workflow: u32,
    /// Flags (bit 0 = double-sided, bit 1 = alpha-blend, bit 2 = alpha-test)
    pub flags: u32,
    pub _pad: u32,
}

impl GpuMaterial {
    /// Index used to indicate "no texture bound"
    pub const NO_TEXTURE: u32 = u32::MAX;
}
