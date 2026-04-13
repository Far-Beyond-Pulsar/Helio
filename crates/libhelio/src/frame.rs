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
    /// Monotonic generation incremented only when billboard content changes.
    pub generation: u64,
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
    /// Radiance Cascades volume bounds (dual-tier GI: RC near, ambient far).
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
    /// GBuffer lightmap UV texture (Rg16Float) populated by GBufferPass.
    /// Contains per-pixel lightmap atlas UVs for sampling baked_lightmap.
    pub gbuffer_lightmap_uv: Option<&'a wgpu::TextureView>,
    /// Shadow atlas (2D array texture view) — populated after ShadowPass (dynamic/Movable objects)
    pub shadow_atlas: Option<&'a wgpu::TextureView>,
    /// Static shadow atlas (2D array texture view) — cached until Static/Stationary topology changes.
    /// Combined with `shadow_atlas` in the lighting shader: a pixel is shadowed if either atlas occludes it.
    pub static_shadow_atlas: Option<&'a wgpu::TextureView>,
    /// Shadow atlas sampler (comparison sampler)
    pub shadow_sampler: Option<&'a wgpu::Sampler>,
    /// Hi-Z pyramid (mip chain of depth, for occlusion culling)
    pub hiz: Option<&'a wgpu::TextureView>,
    /// Hi-Z sampler (min reduction sampler)
    pub hiz_sampler: Option<&'a wgpu::Sampler>,
    /// Static HiZ: Pre-baked 3D voxel occlusion grid for static geometry (camera-independent)
    pub static_hiz: Option<&'a wgpu::TextureView>,
    /// Static HiZ sampler (linear, clamp)
    pub static_hiz_sampler: Option<&'a wgpu::Sampler>,
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

    /// Full-resolution depth texture object for compute passes that need raw texture access.
    pub full_res_depth_texture: Option<&'a wgpu::Texture>,
    /// High-level Helio scene resources used by wrapper-owned passes.
    pub main_scene: Option<MainSceneResources<'a>>,
    /// Sky context (has_sky, state_changed, sky_color)
    pub sky: crate::sky::SkyContext,
    /// Billboards to render this frame (uploaded by the high-level Renderer).
    pub billboards: Option<BillboardFrameData<'a>>,
    /// Virtual geometry meshlet + instance data for this frame.
    pub vg: Option<VgFrameData<'a>>,

    /// Water caustics texture (populated by WaterCausticsPass)
    pub water_caustics: Option<&'a wgpu::TextureView>,

    /// Water volumes buffer (populated by Scene)
    pub water_volumes: Option<&'a wgpu::Buffer>,

    /// Number of water volumes in the buffer
    pub water_volume_count: u32,

    /// Water heightfield simulation texture (Rgba16Float 256×256, ping-pong current)
    /// R=height, G=velocity, B=normal.x, A=normal.z
    /// Populated by `WaterSimPass::publish()`.
    pub water_sim_texture: Option<&'a wgpu::TextureView>,

    /// Linear clamp sampler for water_sim_texture (set by WaterSimPass)
    pub water_sim_sampler: Option<&'a wgpu::Sampler>,

    /// Water hitboxes storage buffer (populated by Renderer each frame)
    pub water_hitboxes: Option<&'a wgpu::Buffer>,

    /// Number of hitboxes in water_hitboxes
    pub water_hitbox_count: u32,

    /// Main depth texture (for passes that need to copy/sample it)
    pub depth_texture: Option<&'a wgpu::Texture>,

    // ── Pre-baked data (populated by BakeInjectPass when baking is enabled) ──

    /// Pre-baked ambient occlusion texture (R8Unorm, same format as SSAO output).
    ///
    /// When present, `SsaoPass` skips runtime computation and publishes this texture
    /// instead of its own SSAO result.
    pub baked_ao: Option<&'a wgpu::TextureView>,

    /// Sampler for [`baked_ao`](Self::baked_ao).
    pub baked_ao_sampler: Option<&'a wgpu::Sampler>,

    /// Pre-baked lightmap atlas (RGBA32F or RGBA16F).
    ///
    /// Contains direct + multi-bounce indirect illumination for static geometry.
    /// Indexed by per-mesh UV atlas regions stored in the baked data.
    pub baked_lightmap: Option<&'a wgpu::TextureView>,

    /// Sampler for [`baked_lightmap`](Self::baked_lightmap).
    pub baked_lightmap_sampler: Option<&'a wgpu::Sampler>,

    /// Pre-baked reflection cubemap (Rgba32Float or Rgba8Unorm RGBE, 6 faces + mip chain).
    ///
    /// First probe only; closest-probe blending is future work.
    pub baked_reflection: Option<&'a wgpu::TextureView>,

    /// Sampler for [`baked_reflection`](Self::baked_reflection) (trilinear).
    pub baked_reflection_sampler: Option<&'a wgpu::Sampler>,

    /// Pre-baked irradiance spherical harmonics (L2, 9 RGB coefficients = 27 × f32).
    ///
    /// Stored as a uniform buffer (`wgpu::BufferUsages::UNIFORM`).
    pub baked_irradiance_sh: Option<&'a wgpu::Buffer>,

    /// Pre-baked potentially-visible set for CPU-side visibility culling.
    ///
    /// Use [`BakedPvsRef::is_visible`] to test cell-to-cell visibility before
    /// submitting draw calls. Returns `None` when PVS baking was not configured.
    pub baked_pvs: Option<BakedPvsRef<'a>>,
}

// ── PVS CPU reference ──────────────────────────────────────────────────────────

/// Borrowed reference into the pre-baked PVS bitfield grid.
///
/// Zero-copy view — `bits` borrows directly from the `BakedData` owned by
/// `BakeInjectPass`. Valid for the duration of the frame.
#[derive(Clone, Copy)]
pub struct BakedPvsRef<'a> {
    pub world_min: [f32; 3],
    pub world_max: [f32; 3],
    pub grid_dims: [u32; 3],
    pub cell_size: f32,
    pub cell_count: u32,
    pub words_per_cell: u32,
    /// Packed bitfield: `bits[from * words_per_cell + to/64] >> (to%64) & 1 == 1` means
    /// cell `to` is potentially visible from cell `from`.
    pub bits: &'a [u64],
}

impl<'a> BakedPvsRef<'a> {
    /// Returns `true` if cell `to_cell` is potentially visible from cell `from_cell`.
    #[inline]
    pub fn is_visible(&self, from_cell: usize, to_cell: usize) -> bool {
        let idx = from_cell * self.words_per_cell as usize + to_cell / 64;
        if idx >= self.bits.len() { return true; } // conservative default
        (self.bits[idx] >> (to_cell % 64)) & 1 == 1
    }

    /// Returns the grid-cell index at world position `p`, or `None` if out of bounds.
    #[inline]
    pub fn cell_at(&self, p: [f32; 3]) -> Option<usize> {
        let [gx, gy, gz] = self.grid_dims;
        let dx = ((p[0] - self.world_min[0]) / self.cell_size) as i32;
        let dy = ((p[1] - self.world_min[1]) / self.cell_size) as i32;
        let dz = ((p[2] - self.world_min[2]) / self.cell_size) as i32;
        if dx < 0 || dy < 0 || dz < 0
            || dx >= gx as i32 || dy >= gy as i32 || dz >= gz as i32
        {
            return None;
        }
        Some(dx as usize + dy as usize * gx as usize + dz as usize * gx as usize * gy as usize)
    }
}

// ── Owned PVS data (lives in BakedData, referenced by BakedPvsRef) ────────────

/// Owned CPU-side PVS data stored in [`BakedData`].
///
/// Published as a zero-copy [`BakedPvsRef`] into `FrameResources` each frame.
pub struct BakedPvsData {
    pub world_min: [f32; 3],
    pub world_max: [f32; 3],
    pub grid_dims: [u32; 3],
    pub cell_size: f32,
    pub cell_count: u32,
    pub words_per_cell: u32,
    pub bits: Vec<u64>,
}

impl<'a> FrameResources<'a> {
    /// Creates an empty (all-None) frame resources for the start of a frame.
    pub fn empty() -> Self {
        Self {
            gbuffer: None,
            gbuffer_lightmap_uv: None,
            shadow_atlas: None,
            static_shadow_atlas: None,
            shadow_sampler: None,
            hiz: None,
            hiz_sampler: None,
            static_hiz: None,
            static_hiz_sampler: None,
            sky_lut: None,
            sky_lut_sampler: None,
            ssao: None,
            pre_aa: None,
            tile_light_lists: None,
            tile_light_counts: None,
            full_res_depth: None,
            full_res_depth_texture: None,
            main_scene: None,
            sky: crate::sky::SkyContext::default(),
            billboards: None,
            vg: None,
            water_caustics: None,
            water_volumes: None,
            water_volume_count: 0,
            water_sim_texture: None,
            water_sim_sampler: None,
            water_hitboxes: None,
            water_hitbox_count: 0,
            depth_texture: None,
            baked_ao: None,
            baked_ao_sampler: None,
            baked_lightmap: None,
            baked_lightmap_sampler: None,
            baked_reflection: None,
            baked_reflection_sampler: None,
            baked_irradiance_sh: None,
            baked_pvs: None,
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

