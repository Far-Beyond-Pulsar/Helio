//! Per-frame transient resource views.
//!
//! `FrameResources` holds borrowed references to the transient textures that the
//! `RenderGraph` owns. These are passed into `PassContext` and `PrepareContext` so
//! passes can read outputs of earlier passes without any allocation or locking.

/// Views into the GBuffer textures.
///
/// Produced by `GBufferPass`, consumed by `DeferredLightingPass`, `SsaoPass`, etc.
pub struct GBufferViews<'a> {
    /// Albedo (RGB) + AO (A) — `Rgba8Unorm`
    pub albedo_ao: &'a wgpu::TextureView,
    /// Normal (RGB, encoded) + roughness (A) — `Rgba16Float`
    pub normal_roughness: &'a wgpu::TextureView,
    /// Emissive (RGB) + metallic (A) — `Rgba8Unorm`
    pub emissive_metallic: &'a wgpu::TextureView,
    /// Velocity (RG) — `Rg16Float`
    pub velocity: &'a wgpu::TextureView,
}

/// All transient per-frame texture references.
///
/// The `RenderGraph` creates the actual `wgpu::Texture` objects and passes
/// borrowed views through this struct. Zero allocations in the hot path.
pub struct FrameResources<'a> {
    /// GBuffer textures (populated after GBufferPass)
    pub gbuffer: Option<GBufferViews<'a>>,
    /// Shadow atlas (2D array texture view) — populated after ShadowPass
    pub shadow_atlas: Option<&'a wgpu::TextureView>,
    /// Shadow atlas sampler (comparison sampler)
    pub shadow_sampler: Option<&'a wgpu::Sampler>,
    /// Hi-Z pyramid (mip chain of depth, for occlusion culling)
    pub hiz: Option<&'a wgpu::TextureView>,
    /// Hi-Z sampler (min reduction sampler)
    pub hiz_sampler: Option<&'a wgpu::Sampler>,
    /// Atmospheric sky LUT (transmittance + aerial perspective)
    pub sky_lut: Option<&'a wgpu::TextureView>,
    /// Sky LUT sampler (linear, clamp)
    pub sky_lut_sampler: Option<&'a wgpu::Sampler>,
    /// SSAO result texture
    pub ssao: Option<&'a wgpu::TextureView>,
    /// Pre-AA HDR color buffer (input to TAA/FXAA/SMAA)
    pub pre_aa: Option<&'a wgpu::TextureView>,
    /// Sky context (has_sky, state_changed, sky_color)
    pub sky: crate::SkyContext,
}

impl<'a> FrameResources<'a> {
    /// Creates an empty (all-None) frame resources for the start of a frame.
    pub fn empty() -> Self {
        Self {
            gbuffer: None,
            shadow_atlas: None,
            shadow_sampler: None,
            hiz: None,
            hiz_sampler: None,
            sky_lut: None,
            sky_lut_sampler: None,
            ssao: None,
            pre_aa: None,
            sky: crate::SkyContext::default(),
        }
    }
}
