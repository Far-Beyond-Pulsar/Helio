//! Shared materials/shading contract for Helio.
//!
//! GPU material layout, binding-mode selection (bindless vs. baseline-WebGPU),
//! the PBR/BRDF evaluation library, and the Radiant hybrid material-template
//! system all live here so every material-consuming pass (gbuffer,
//! forward-lit, transparent, decal, virtual-geometry, portal-instances) and
//! the host `helio::Renderer` (which must pick a binding mode before creating
//! the device) depend on one shared crate for this shape, rather than any one
//! pass owning it centrally or the host reaching into a specific pass.
//!
//! Holds no scene storage of its own: persistent material authoring is a
//! SceneDB component (`engine_backend::scene::MaterialComponent` /
//! `helio_pass_gbuffer::MaterialComponent`); this crate only defines the GPU
//! row layout and shading/template machinery those components project into.

pub mod material;
pub mod radiant;
pub mod shader;

pub use material::{
    GpuMaterial, MaterialBindingConfig, MaterialBindingMode, MaterialTextureBindings,
    MaterialWorkflow, BINDLESS_MATERIAL_FEATURES, EXPANDED_MATERIAL_TEXTURE_RESERVE,
    FLAG_ALPHA_BLEND, FLAG_ALPHA_TEST, FLAG_DOUBLE_SIDED, FLAG_FORWARD_SHADING,
    FLAG_HAS_ANISOTROPY, FLAG_HAS_CLEAR_COAT, FLAG_HAS_CUSTOM_SHADER, FLAG_HAS_NORMAL_MAP,
    FLAG_HAS_SUBSURFACE, FLAG_TRANSPARENT_ONLY, MATERIAL_CLASS_ANISOTROPIC,
    MATERIAL_CLASS_CLEAR_COAT, MATERIAL_CLASS_CUSTOM, MATERIAL_CLASS_DEFAULT,
    MATERIAL_CLASS_SKIN, MATERIAL_CLASS_SUBSURFACE, MAX_MATERIAL_TEXTURES,
};
pub use radiant::{
    RadiantGraphRegistry, RadiantShaderCache, RadiantShaderKey, RadiantTemplate,
    RadiantTemplateRegistry, SharedTemplateRegistry,
};
pub use shader::{
    apply_webgpu_decal_bindings, apply_webgpu_material_bindings, PBR_EVAL, PBR_EVAL_SNIPPET,
    PBR_MARKER,
};
