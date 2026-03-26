//! Per-frame transient resource views.
//!
//! `FrameResources` holds borrowed references to the transient textures that the
//! `RenderGraph` owns. These are passed into `PassContext` and `PrepareContext` so
//! passes can read outputs of earlier passes without any allocation or locking.

/// Per-frame billboard instance data, provided by the high-level `Renderer`.
///
/// The high-level renderer stores a `Vec<BillboardInstance>` and populates this
/// struct each frame so that `BillboardPass::prepare()` can upload the data to
/// the GPU without any extra allocation.
#[derive(Clone, Copy)]
pub struct BillboardFrameData<'a> {
    /// Raw bytes of a `BillboardInstance` array (must be `Pod`-compatible).
    pub instances: &'a [u8],
    /// Number of valid instances in the slice.
    pub count: u32,
}

/// Views into the GBuffer textures.
///
/// Produced by `GBufferPass`, consumed by `DeferredLightingPass`, `SsaoPass`, etc.
#[derive(Clone, Copy)]
pub struct GBufferViews<'a> {
    /// Albedo (RGB) + alpha (A) — `Rgba8Unorm`
    pub albedo: &'a wgpu::TextureView,
    /// World normal (RGB) + F0.r (A) — `Rgba16Float`
    pub normal: &'a wgpu::TextureView,
    /// AO, roughness, metallic, F0.g — `Rgba8Unorm`
    pub orm: &'a wgpu::TextureView,
    /// Emissive (RGB) + F0.b (A) — `Rgba16Float`
    pub emissive: &'a wgpu::TextureView,
}

/// Borrowed mesh buffers for passes that render scene geometry directly.
#[derive(Clone, Copy)]
pub struct MeshBuffers<'a> {
    pub vertices: &'a wgpu::Buffer,
    pub indices: &'a wgpu::Buffer,
}

/// Borrowed material-texture state for passes that sample Helio's texture table.
#[derive(Clone, Copy)]
pub struct MaterialTextureBindings<'a> {
    pub material_textures: &'a wgpu::Buffer,
    pub texture_views: &'a [&'a wgpu::TextureView],
    pub samplers: &'a [&'a wgpu::Sampler],
    pub version: u64,
}

/// Frame-local scene inputs for the high-level Helio renderer.
#[derive(Clone, Copy)]
pub struct MainSceneResources<'a> {
    pub mesh_buffers: MeshBuffers<'a>,
    pub material_textures: MaterialTextureBindings<'a>,
    pub clear_color: [f32; 4],
    pub ambient_color: [f32; 3],
    pub ambient_intensity: f32,
    /// Radiance Cascades volume bounds (AAA dual-tier GI: RC near, ambient far).
    /// RC active within these bounds, simpler ambient fallback outside.
    pub rc_world_min: [f32; 3],
    pub rc_world_max: [f32; 3],
}

/// All transient per-frame texture references.
///
/// The `RenderGraph` creates the actual `wgpu::Texture` objects and passes
/// borrowed views through this struct. Zero allocations in the hot path.
#[derive(Clone, Copy)]
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
    /// Tiled light lists buffer (populated by LightCullPass, consumed by DeferredLightPass).
    /// Layout: `tile_light_lists[tile_idx * MAX_LIGHTS_PER_TILE + i] = light_index`.
    pub tile_light_lists: Option<&'a wgpu::Buffer>,
    /// Tiled light counts buffer: one u32 per tile giving the number of lights.
    pub tile_light_counts: Option<&'a wgpu::Buffer>,
    /// Full-resolution depth view — only present when render_scale < 1.0.
    /// Post-upscale passes (e.g. BillboardPass) that render to the native-resolution
    /// `ctx.target` must use this instead of `ctx.depth` (which is at internal res)
    /// to avoid a render-pass attachment size mismatch.
    pub full_res_depth: Option<&'a wgpu::TextureView>,
    /// High-level Helio scene resources used by wrapper-owned passes.
    pub main_scene: Option<MainSceneResources<'a>>,
    /// Sky context (has_sky, state_changed, sky_color)
    pub sky: crate::sky::SkyContext,
    /// Billboards to render this frame (uploaded by the high-level Renderer).
    pub billboards: Option<BillboardFrameData<'a>>,
    /// Virtual geometry meshlet + instance data for this frame.
    pub vg: Option<VgFrameData<'a>>,
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
            tile_light_lists: None,
            tile_light_counts: None,
            full_res_depth: None,
            main_scene: None,
            sky: crate::sky::SkyContext::default(),
            billboards: None,
            vg: None,
        }
    }
}

/// Per-frame virtual geometry data: CPU-side meshlet and instance byte slices.
///
/// The `VirtualGeometryPass` uploads these slices to its owned GPU buffers on the
/// first frame and whenever `buffer_version` advances (topology or transform change).
#[derive(Clone, Copy)]
pub struct VgFrameData<'a> {
    /// Raw bytes of a `GpuMeshletEntry` array.
    pub meshlets: &'a [u8],
    /// Raw bytes of a `GpuInstanceData` array (one entry per VG object).
    pub instances: &'a [u8],
    /// Total number of meshlets across all VG objects.
    pub meshlet_count: u32,
    /// Number of VG object instances.
    pub instance_count: u32,
    /// Version counter incremented each time meshlet or instance data changes.
    /// The pass re-uploads GPU buffers only when this advances.
    pub buffer_version: u64,
}

