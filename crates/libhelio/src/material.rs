//! GPU material types. Must match `helio-render-v2` layout for asset compat.

use bytemuck::{Pod, Zeroable};

/// Material workflow discriminant.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialWorkflow {
    Metallic = 0,
    Specular = 1,
}

/// Feature flags for [`GpuMaterial::flags`].
///
/// Each flag toggles a warp-uniform branch in the generated WGSL; disabled
/// features cost zero instructions via constant-condition elimination.
pub const FLAG_DOUBLE_SIDED: u32 = 1 << 0;
pub const FLAG_ALPHA_BLEND: u32 = 1 << 1;
pub const FLAG_ALPHA_TEST: u32 = 1 << 2;
pub const FLAG_HAS_NORMAL_MAP: u32 = 1 << 3;
pub const FLAG_HAS_CLEAR_COAT: u32 = 1 << 4;
pub const FLAG_HAS_SUBSURFACE: u32 = 1 << 5;
pub const FLAG_HAS_ANISOTROPY: u32 = 1 << 6;
pub const FLAG_HAS_CUSTOM_SHADER: u32 = 1 << 7;

/// Material class shader archetypes.
pub const MATERIAL_CLASS_DEFAULT: u32 = 0;
pub const MATERIAL_CLASS_CLEAR_COAT: u32 = 1;
pub const MATERIAL_CLASS_SUBSURFACE: u32 = 2;
pub const MATERIAL_CLASS_ANISOTROPIC: u32 = 3;
pub const MATERIAL_CLASS_SKIN: u32 = 4;
pub const MATERIAL_CLASS_CUSTOM: u32 = 0xFFFF;

/// GPU material data. 112 bytes.
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
///     material_class:       u32,
///     class_params:         vec4<f32>,
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
    /// Feature flags (see `FLAG_*` constants)
    pub flags: u32,
    /// Material class selector (see `MATERIAL_CLASS_*` constants)
    pub material_class: u32,
    /// Class-specific parameters interpreted by the active Radiant template.
    /// The default PBR template ignores these; custom templates can use them
    /// for any purpose (e.g. clear-coat strength, subsurface colour, anisotropy direction).
    pub class_params: [f32; 4],
}

impl GpuMaterial {
    /// Index used to indicate "no texture bound"
    pub const NO_TEXTURE: u32 = u32::MAX;
}

