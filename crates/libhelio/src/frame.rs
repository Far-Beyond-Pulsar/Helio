//! Per-frame transient resource views.
//!
//! `FrameResources` holds borrowed references to the transient textures that the
//! `RenderGraph` owns. These are passed into `PassContext` and `PrepareContext` so
//! passes can read outputs of earlier passes without any allocation or locking.

use crate::wind::GpuWind;
use crate::CoronaEmitterFrameData;

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
///
/// Static geometry (terrain, buildings, props) lives in `vertices`/`indices`.
/// Dynamic geometry (skinned characters, morphed meshes) lives in
/// `dynamic_vertices`/`dynamic_indices`. Each pair must be bound separately
/// around the corresponding draw calls.
#[derive(Clone, Copy)]
pub struct MeshBuffers<'a> {
    /// Vertex buffer for upload-once static geometry.
    pub vertices: &'a wgpu::Buffer,
    /// Index buffer for upload-once static geometry.
    pub indices: &'a wgpu::Buffer,
    /// Vertex buffer for per-frame-updatable dynamic geometry.
    pub dynamic_vertices: &'a wgpu::Buffer,
    /// Index buffer for per-frame-updatable dynamic geometry.
    pub dynamic_indices: &'a wgpu::Buffer,
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
    /// Hardware ray tracing TLAS, if available. None on non-RT hardware or WASM.
    pub tlas: Option<&'a wgpu::Tlas>,
}

/// Debug-tracked resource slot.
///
/// In debug builds, records which pass wrote the value so we can detect
/// when a pass reads a resource that no prior pass wrote this frame.
/// In release builds, compiles down to a plain `Option<T>` with zero overhead.
#[derive(Clone, Copy)]
pub struct Tracked<T> {
    value: Option<T>,
    #[cfg(debug_assertions)]
    written_by: Option<&'static str>,
}

impl<T: Copy> Tracked<T> {
    /// Creates an empty (unwritten) slot.
    pub const fn empty() -> Self {
        Self {
            value: None,
            #[cfg(debug_assertions)]
            written_by: None,
        }
    }

    /// Creates a slot with a pre-set value (no writer recorded).
    /// Used for renderer-provided fields that are available from the start.
    pub const fn with_value(value: T) -> Self {
        Self {
            value: Some(value),
            #[cfg(debug_assertions)]
            written_by: None,
        }
    }

    /// Writes a value, recording the writer pass name in debug builds.
    pub fn write(&mut self, value: T, _pass_name: &'static str) {
        self.value = Some(value);
        #[cfg(debug_assertions)]
        {
            self.written_by = Some(_pass_name);
        }
    }

    /// Reads the value. Panics in debug builds if the slot was never written
    /// (i.e. no prior pass called `write` on it this frame).
    ///
    /// Returns `None` when the slot was explicitly written but set to `None`
    /// (the panics only fire for unwritten slots, not empty-but-written ones).
    pub fn read(&self, _reader_pass: &'static str) -> Option<T> {
        #[cfg(debug_assertions)]
        if self.written_by.is_none() && self.value.is_some() {
            // This state shouldn't happen if write() is always used,
            // but just in case, don't panic if there's actually a value.
        }
        #[cfg(debug_assertions)]
        if self.value.is_none() && self.written_by.is_none() {
            panic!(
                "[RenderGraph] pass '{reader}' read resource that was never written this frame",
                reader = _reader_pass
            );
        }
        self.value
    }

    /// Reads without debug tracking (for optional resources that legitimately
    /// may be `None`, e.g. `full_res_depth`).
    pub fn get(&self) -> Option<T> {
        self.value
    }

    /// Returns true if this slot was written this frame (debug builds only).
    /// Always returns `true` in release builds.
    #[inline]
    pub fn was_written(&self) -> bool {
        #[cfg(debug_assertions)]
        {
            self.written_by.is_some()
        }
        #[cfg(not(debug_assertions))]
        {
            self.value.is_some()
        }
    }
}

impl<T> Tracked<T> {
    /// Returns `true` if the slot has a value (regardless of tracking state).
    pub fn is_some(&self) -> bool {
        self.value.is_some()
    }

    /// Returns `true` if the slot has no value.
    pub fn is_none(&self) -> bool {
        self.value.is_none()
    }

    /// Converts to `Option<&T>`.
    pub fn as_ref(&self) -> Option<&T> {
        self.value.as_ref()
    }
}

/// All transient per-frame texture references.
///
/// The `RenderGraph` creates the actual `wgpu::Texture` objects and passes
/// borrowed views through this struct. Zero allocations in the hot path.
#[derive(Clone, Copy)]
pub struct FrameResources<'a> {
    /// GBuffer textures (populated after GBufferPass)
    pub gbuffer: Tracked<GBufferViews<'a>>,
    /// GBuffer lightmap UV texture (Rg16Float) populated by GBufferPass.
    /// Contains per-pixel lightmap atlas UVs for sampling baked_lightmap.
    pub gbuffer_lightmap_uv: Tracked<&'a wgpu::TextureView>,
    /// GBuffer SSS data (Rgba16Float): subsurface_color.rgb, subsurface_radius.
    /// Populated by GBufferPass, consumed by DeferredLightPass and SssBlurPass.
    pub gbuffer_sss: Tracked<&'a wgpu::TextureView>,
    /// GBuffer extra surface data (Rgba16Float): roughness_aniso_x, roughness_aniso_y,
    /// aniso_rotation, bitcast<f32>(surface_flags).
    /// Populated by GBufferPass, consumed by DeferredLightPass.
    pub gbuffer_extra: Tracked<&'a wgpu::TextureView>,
    /// GBuffer velocity buffer (Rg16Float): screen-space motion in pixels/frame.
    /// Populated by GBufferPass, consumed by PostProcessPass (motion blur) and TaAPass.
    pub gbuffer_velocity: Tracked<&'a wgpu::TextureView>,
    /// Shadow atlas (2D array texture view) — populated after ShadowPass (dynamic/Movable objects)
    pub shadow_atlas: Tracked<&'a wgpu::TextureView>,
    /// Static shadow atlas (2D array texture view) — cached until Static/Stationary topology changes.
    /// Combined with `shadow_atlas` in the lighting shader: a pixel is shadowed if either atlas occludes it.
    pub static_shadow_atlas: Tracked<&'a wgpu::TextureView>,
    /// Shadow atlas sampler (comparison sampler)
    pub shadow_sampler: Tracked<&'a wgpu::Sampler>,
    /// Hi-Z pyramid (mip chain of depth, for occlusion culling)
    pub hiz: Tracked<&'a wgpu::TextureView>,
    /// Hi-Z sampler (min reduction sampler)
    pub hiz_sampler: Tracked<&'a wgpu::Sampler>,
    /// Static HiZ: Pre-baked 3D voxel occlusion grid for static geometry (camera-independent)
    pub static_hiz: Tracked<&'a wgpu::TextureView>,
    /// Static HiZ sampler (linear, clamp)
    pub static_hiz_sampler: Tracked<&'a wgpu::Sampler>,
    /// Atmospheric sky LUT (transmittance + aerial perspective)
    pub sky_lut: Tracked<&'a wgpu::TextureView>,
    /// Sky LUT sampler (linear, clamp)
    pub sky_lut_sampler: Tracked<&'a wgpu::Sampler>,
    /// SSAO result texture
    pub ssao: Tracked<&'a wgpu::TextureView>,
    /// Volumetric fog accumulation, internal resolution (or a divisor of it).
    /// rgb = in-scattered radiance, a = transmittance to the surface.
    pub fog_accum: Tracked<&'a wgpu::TextureView>,
    /// Pre-AA HDR color buffer (input to TAA/FXAA/SMAA)
    pub pre_aa: Tracked<&'a wgpu::TextureView>,
    /// Tiled light lists buffer (populated by LightCullPass, consumed by DeferredLightPass).
    /// Layout: `tile_light_lists[tile_idx * MAX_LIGHTS_PER_TILE + i] = light_index`.
    pub tile_light_lists: Tracked<&'a wgpu::Buffer>,
    /// Tiled light counts buffer: one u32 per tile giving the number of lights.
    pub tile_light_counts: Tracked<&'a wgpu::Buffer>,
    /// Full-resolution depth view — only present when render_scale < 1.0.
    /// Post-upscale passes (e.g. BillboardPass) that render to the native-resolution
    /// `ctx.target` must use this instead of `ctx.depth` (which is at internal res)
    /// to avoid a render-pass attachment size mismatch.
    pub full_res_depth: Tracked<&'a wgpu::TextureView>,

    /// Full-resolution depth texture object for compute passes that need raw texture access.
    pub full_res_depth_texture: Tracked<&'a wgpu::Texture>,
    /// High-level Helio scene resources used by wrapper-owned passes.
    pub main_scene: Tracked<MainSceneResources<'a>>,
    /// Sky context (has_sky, state_changed, sky_color)
    pub sky: crate::sky::SkyContext,
    /// Billboards to render this frame (uploaded by the high-level Renderer).
    pub billboards: Tracked<BillboardFrameData<'a>>,
    /// Virtual geometry meshlet + instance data for this frame.
    pub vg: Tracked<VgFrameData<'a>>,

    /// Water caustics texture (populated by WaterCausticsPass)
    pub water_caustics: Tracked<&'a wgpu::TextureView>,

    /// Water volumes buffer (populated by Scene)
    pub water_volumes: Tracked<&'a wgpu::Buffer>,

    /// Number of water volumes in the buffer
    pub water_volume_count: u32,

    /// Post-process volumes storage buffer (populated by Renderer)
    pub pp_volumes: Tracked<&'a wgpu::Buffer>,

    /// Number of post-process volumes in the buffer
    pub pp_volume_count: u32,

    /// Water heightfield simulation texture (Rgba16Float 256×256, ping-pong current)
    /// R=height, G=velocity, B=normal.x, A=normal.z
    /// Populated by `WaterSimPass::publish()`.
    pub water_sim_texture: Tracked<&'a wgpu::TextureView>,

    /// Linear clamp sampler for water_sim_texture (set by WaterSimPass)
    pub water_sim_sampler: Tracked<&'a wgpu::Sampler>,

    /// Water hitboxes storage buffer (populated by Renderer each frame)
    pub water_hitboxes: Tracked<&'a wgpu::Buffer>,

    /// Number of hitboxes in water_hitboxes
    pub water_hitbox_count: u32,

    // ── Foliage ──────────────────────────────────────────────────────────────

    /// Foliage type/layer tables plus the global wind uniform for this frame.
    ///
    /// Published by the high-level `Renderer`; read by every foliage pass. Left
    /// unwritten when no foliage types are registered, which is how the foliage passes
    /// early-out of `prepare()` and record zero commands — see the zero-overhead
    /// guarantees in the foliage plan. Do not "helpfully" write an empty
    /// [`FoliageFrameData`] instead: that turns the free path into a per-frame upload of
    /// two empty buffers plus four zero-instance indirect draws.
    pub foliage: Tracked<FoliageFrameData<'a>>,

    /// Top-down terrain capture over the active foliage ring, written by
    /// `FoliageTerrainPass` and read by placement, interaction and the far-ring
    /// terrain-shading fallback.
    ///
    /// The capture is the *only* thing those consumers know about the ground: it is what
    /// lets foliage work over voxel meshes, heightmaps and planetary voxel pages without
    /// the placement shader branching on terrain type.
    pub foliage_terrain: Tracked<FoliageTerrainViews<'a>>,

    /// Camera-relative foliage interaction field (`Rgba16Float`), written by
    /// `FoliageInteractionPass`.
    ///
    /// RG = horizontal displacement direction × magnitude, B = vertical crush,
    /// A = recovery timer. Foliage vertex shaders sample it and bend proportional to
    /// normalised height along the blade.
    pub foliage_interaction: Tracked<&'a wgpu::TextureView>,

    /// Sampler for [`foliage_interaction`](Self::foliage_interaction).
    ///
    /// Must be linear **clamp-to-edge**: the field is camera-relative and snapped to its
    /// texel grid, so a repeating address mode wraps trampled grass from one edge of the
    /// field to the opposite edge, 64 m away.
    pub foliage_interaction_sampler: Tracked<&'a wgpu::Sampler>,

    /// Foliage interactor storage buffer (populated by the Renderer each frame).
    ///
    /// Splatted into the interaction field by `FoliageInteractionPass`. Follows the
    /// `water_hitboxes` contract exactly: the buffer may be over-allocated, and
    /// [`foliage_interactor_count`](Self::foliage_interactor_count) — not the buffer
    /// size — is the authority on how many entries are live this frame.
    pub foliage_interactors: Tracked<&'a wgpu::Buffer>,

    /// Number of interactors in foliage_interactors
    pub foliage_interactor_count: u32,

    /// Radiance Cascades cascade atlas texture view
    pub rc_view: Tracked<&'a wgpu::TextureView>,

    /// Main depth texture (for passes that need to copy/sample it)
    pub depth_texture: Tracked<&'a wgpu::Texture>,
    /// Single-layer `D2` view of the *render-target* depth for sampling passes.
    ///
    /// In multiview (XR) mode the render passes write depth into a 2-layer
    /// array and the depth-stencil attachment (`ctx.depth`) is a `D2Array`
    /// view, which cannot be bound to `texture_depth_2d` / `D2` bind group
    /// entries. This field carries a layer-0 `D2` view of the same texture so
    /// HiZ, lens flare etc. can sample the depth that was actually rendered.
    /// On desktop it is simply the plain depth view.
    pub depth_sampler_view: Tracked<&'a wgpu::TextureView>,

    // ── Pre-baked data (populated by BakeInjectPass when baking is enabled) ──

    /// Pre-baked ambient occlusion texture (R8Unorm, same format as SSAO output).
    ///
    /// When present, `SsaoPass` skips runtime computation and publishes this texture
    /// instead of its own SSAO result.
    pub baked_ao: Tracked<&'a wgpu::TextureView>,

    /// Sampler for [`baked_ao`](Self::baked_ao).
    pub baked_ao_sampler: Tracked<&'a wgpu::Sampler>,

    /// Pre-baked lightmap atlas (RGBA32F or RGBA16F).
    ///
    /// Contains direct + multi-bounce indirect illumination for static geometry.
    /// Indexed by per-mesh UV atlas regions stored in the baked data.
    pub baked_lightmap: Tracked<&'a wgpu::TextureView>,

    /// Sampler for [`baked_lightmap`](Self::baked_lightmap).
    pub baked_lightmap_sampler: Tracked<&'a wgpu::Sampler>,

    /// Pre-baked reflection cubemap (Rgba32Float or Rgba8Unorm RGBE, 6 faces + mip chain).
    ///
    /// First probe only; closest-probe blending is future work.
    pub baked_reflection: Tracked<&'a wgpu::TextureView>,

    /// Sampler for [`baked_reflection`](Self::baked_reflection) (trilinear).
    pub baked_reflection_sampler: Tracked<&'a wgpu::Sampler>,

    /// Pre-baked irradiance spherical harmonics (L2, 9 RGB coefficients = 27 × f32).
    ///
    /// Stored as a uniform buffer (`wgpu::BufferUsages::UNIFORM`).
    pub baked_irradiance_sh: Tracked<&'a wgpu::Buffer>,

    /// Pre-baked potentially-visible set for CPU-side visibility culling.
    ///
    /// Use [`BakedPvsRef::is_visible`] to test cell-to-cell visibility before
    /// submitting draw calls. Returns `None` when PVS baking was not configured.
    pub baked_pvs: Tracked<BakedPvsRef<'a>>,

    /// Cluster light grid for forward rendering (populated by LightCullPass).
    pub cluster_light_grid: Tracked<ClusterLightGrid<'a>>,

    /// Corona particle emitter definitions (uploaded by the Renderer each frame)
    pub corona_emitters: Tracked<CoronaEmitterFrameData<'a>>,

    /// Post-process uniform buffer (written by the Renderer, read by PostProcessPass).
    /// Points to the pass's own `GpuPostProcessUniforms` buffer.
    pub postprocess_uniforms: Tracked<&'a wgpu::Buffer>,
    /// 3D colour grading LUT texture (Rgba16Float, optional).
    /// Populated by LUT builder pass, consumed by PostProcessPass.
    pub color_grading_lut: Tracked<&'a wgpu::TextureView>,
    /// IES light profile texture array (R8Unorm, 256×256 per slice, optional).
    /// Populated by Renderer, consumed by DeferredLightPass.
    pub ies_textures: Tracked<&'a wgpu::TextureView>,

    /// Reflection capture storage buffer
    pub reflection_captures: Tracked<&'a wgpu::Buffer>,

    /// Number of reflection captures in the buffer
    pub reflection_capture_count: u32,

    /// SSR texture (Rgba16Float, full resolution) — SSR output.
    /// RGB = reflected colour, A = hit confidence (0 = no hit, 1 = confident).
    /// Written by SsrPass, read by DeferredLightPass.
    pub ssr_trace: Tracked<&'a wgpu::TextureView>,

    /// Planar reflection texture (Rgba16Float, full resolution).
    /// RGB = reflected colour from the nearest planar reflector, A = confidence.
    /// Written by PlanarReflectionPass, read by DeferredLightPass.
    pub planar_reflection: Tracked<&'a wgpu::TextureView>,

    /// Sampler for planar_reflection.
    pub planar_reflection_sampler: Tracked<&'a wgpu::Sampler>,

    // ── HLFS resources (populated by HLFS pass) ──

    /// Clip-stack read views for the shade pass (4 levels of 128³ RGBA16F).
    ///
    /// Populated by `HlfsPass::publish()` after the clip-stack buffer swap.
    pub hlfs_clip_stack: Option<[&'a wgpu::TextureView; 4]>,

    /// HLFS-specific globals uniform buffer (HlfsGlobals layout).
    pub hlfs_globals: Option<&'a wgpu::Buffer>,

    // ── Secondary views (portals + sublevels) ───────────────────────────

    /// Per-frame secondary-view descriptors for portal eyes and sublevel cameras.
    ///
    /// Published by the high-level `Renderer`; read by `SecondaryGBufferPass` and
    /// `ProxyCompositePass`. Left unwritten when no portals or sublevels are
    /// registered, which is how those passes early-out of `prepare()` and record
    /// zero commands. Do not "helpfully" write an empty `SecondaryFrameData`
    /// instead — an unwritten slot is a zero-cost path; a written one is not.
    pub secondary: Tracked<SecondaryFrameData<'a>>,

    // ── DOF resources (populated by PostProcessPass / DofPass) ──

    /// Pre-DOF HDR colour — output of the post-process uber-shader before
    /// depth-of-field is applied. Written by PostProcessPass when DofPass
    /// follows in the graph; read by DofPass.
    pub pre_dof: Tracked<&'a wgpu::TextureView>,

    /// Post-DOF final colour — output of DofPass (or identity when not present).
    pub post_dof: Tracked<&'a wgpu::TextureView>,
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

/// Cluster light grid bindings for forward rendering.
///
/// Produced by LightCullPass, consumed by forward-lit passes and transparent pass.
#[derive(Clone, Copy)]
pub struct ClusterLightGrid<'a> {
    pub tile_light_lists: &'a wgpu::Buffer,
    pub tile_light_counts: &'a wgpu::Buffer,
    pub num_tiles_x: u32,
    pub num_tiles_y: u32,
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
    /// Creates an empty (all-Tracked::empty) frame resources for the start of a frame.
    pub fn empty() -> Self {
        Self {
            gbuffer: Tracked::empty(),
            gbuffer_lightmap_uv: Tracked::empty(),
            gbuffer_sss: Tracked::empty(),
            gbuffer_extra: Tracked::empty(),
            gbuffer_velocity: Tracked::empty(),
            shadow_atlas: Tracked::empty(),
            static_shadow_atlas: Tracked::empty(),
            shadow_sampler: Tracked::empty(),
            hiz: Tracked::empty(),
            hiz_sampler: Tracked::empty(),
            static_hiz: Tracked::empty(),
            static_hiz_sampler: Tracked::empty(),
            sky_lut: Tracked::empty(),
            sky_lut_sampler: Tracked::empty(),
            ssao: Tracked::empty(),
            fog_accum: Tracked::empty(),
            pre_aa: Tracked::empty(),
            tile_light_lists: Tracked::empty(),
            tile_light_counts: Tracked::empty(),
            full_res_depth: Tracked::empty(),
            full_res_depth_texture: Tracked::empty(),
            main_scene: Tracked::empty(),
            sky: crate::sky::SkyContext::default(),
            billboards: Tracked::empty(),
            vg: Tracked::empty(),
            water_caustics: Tracked::empty(),
            water_volumes: Tracked::empty(),
            water_volume_count: 0,
            pp_volumes: Tracked::empty(),
            pp_volume_count: 0,
            water_sim_texture: Tracked::empty(),
            water_sim_sampler: Tracked::empty(),
            water_hitboxes: Tracked::empty(),
            water_hitbox_count: 0,
            foliage: Tracked::empty(),
            foliage_terrain: Tracked::empty(),
            foliage_interaction: Tracked::empty(),
            foliage_interaction_sampler: Tracked::empty(),
            foliage_interactors: Tracked::empty(),
            foliage_interactor_count: 0,
            depth_texture: Tracked::empty(),
            depth_sampler_view: Tracked::empty(),
            rc_view: Tracked::empty(),
            baked_ao: Tracked::empty(),
            baked_ao_sampler: Tracked::empty(),
            baked_lightmap: Tracked::empty(),
            baked_lightmap_sampler: Tracked::empty(),
            baked_reflection: Tracked::empty(),
            baked_reflection_sampler: Tracked::empty(),
            baked_irradiance_sh: Tracked::empty(),
            baked_pvs: Tracked::empty(),
            cluster_light_grid: Tracked::empty(),
            corona_emitters: Tracked::empty(),
            postprocess_uniforms: Tracked::empty(),
            color_grading_lut: Tracked::empty(),
            ies_textures: Tracked::empty(),
            reflection_captures: Tracked::empty(),
            reflection_capture_count: 0,
            ssr_trace: Tracked::empty(),
            planar_reflection: Tracked::empty(),
            planar_reflection_sampler: Tracked::empty(),
            secondary: Tracked::empty(),
            hlfs_clip_stack: None,
            hlfs_globals: None,
            pre_dof: Tracked::empty(),
            post_dof: Tracked::empty(),
        }
    }

    /// Resets debug tracking markers so that fields written in a previous
    /// frame don't satisfy the "was written this frame" check.
    ///
    /// Fields that have a value are re-marked with the given `_writer` name
    /// (e.g. `"Renderer"`).  In release builds this is a no-op.
    pub fn reset_tracking(&mut self, _writer: &'static str) {
        #[cfg(debug_assertions)]
        {
            macro_rules! reset_field {
                ($field:ident) => {
                    if self.$field.value.is_some() {
                        self.$field.written_by = Some(_writer);
                    } else {
                        self.$field.written_by = None;
                    }
                };
            }
            reset_field!(gbuffer);
            reset_field!(gbuffer_lightmap_uv);
            reset_field!(gbuffer_sss);
            reset_field!(gbuffer_extra);
            reset_field!(gbuffer_velocity);
            reset_field!(shadow_atlas);
            reset_field!(static_shadow_atlas);
            reset_field!(shadow_sampler);
            reset_field!(hiz);
            reset_field!(hiz_sampler);
            reset_field!(static_hiz);
            reset_field!(static_hiz_sampler);
            reset_field!(sky_lut);
            reset_field!(sky_lut_sampler);
            reset_field!(ssao);
            reset_field!(pre_aa);
            reset_field!(tile_light_lists);
            reset_field!(tile_light_counts);
            reset_field!(full_res_depth);
            reset_field!(full_res_depth_texture);
            reset_field!(main_scene);
            reset_field!(billboards);
            reset_field!(vg);
            reset_field!(water_caustics);
            reset_field!(water_volumes);
            reset_field!(pp_volumes);
            reset_field!(water_sim_texture);
            reset_field!(water_sim_sampler);
            reset_field!(water_hitboxes);
            // `foliage_interactor_count` is a plain u32, not a `Tracked` slot, so it gets
            // no line here — same as `water_hitbox_count` directly above.
            reset_field!(foliage);
            reset_field!(foliage_terrain);
            reset_field!(foliage_interaction);
            reset_field!(foliage_interaction_sampler);
            reset_field!(foliage_interactors);
            reset_field!(depth_texture);
            reset_field!(depth_sampler_view);
            reset_field!(rc_view);
            reset_field!(baked_ao);
            reset_field!(baked_ao_sampler);
            reset_field!(baked_lightmap);
            reset_field!(baked_lightmap_sampler);
            reset_field!(baked_reflection);
            reset_field!(baked_reflection_sampler);
            reset_field!(baked_irradiance_sh);
            reset_field!(baked_pvs);
            reset_field!(cluster_light_grid);
            reset_field!(corona_emitters);
            reset_field!(postprocess_uniforms);
            reset_field!(color_grading_lut);
            reset_field!(ies_textures);
            reset_field!(reflection_captures);
            reset_field!(secondary);
            reset_field!(ssr_trace);
            reset_field!(planar_reflection);
            reset_field!(planar_reflection_sampler);
        }
    }
}

/// Per-frame virtual geometry data: immutable mesh/object slices and dirty-tracked instances.
///
/// The `VirtualGeometryPass` uploads these slices to its owned GPU buffers on the
/// first frame and whenever `buffer_version` advances. Transform-only changes
/// advance `instance_version` and upload only `instance_dirty_start..+count`.
#[derive(Clone, Copy)]
pub struct VgFrameData<'a> {
    /// Raw bytes of a `GpuMeshletEntry` array.
    pub meshlets: &'a [u8],
    /// Raw bytes of a `GpuVgObject` array (one entry per VG object).
    pub objects: &'a [u8],
    /// Raw bytes of a `GpuInstanceData` array (one entry per VG object).
    pub instances: &'a [u8],
    /// Raw bytes of immutable `GpuVgWorkItem` expansion records.
    pub work_items: &'a [u8],
    /// Number of unique meshlet descriptors across all referenced virtual meshes.
    pub meshlet_count: u32,
    /// Number of VG objects (and corresponding instance entries).
    pub object_count: u32,
    /// Number of second-stage 64-meshlet work spans.
    pub work_item_count: u32,
    /// Exact worst-case number of indirect draws after selecting one LOD per object.
    pub max_draw_count: u32,
    /// Version counter incremented when meshlet, object, or work-item topology changes.
    pub buffer_version: u64,
    /// Version counter incremented when a transform-only dirty range is published.
    pub instance_version: u64,
    /// First dirty instance in the current transform-only publication.
    pub instance_dirty_start: u32,
    /// Number of dirty instances; zero when `buffer_version` owns the update.
    pub instance_dirty_count: u32,
}

/// Per-frame foliage data: immutable type/layer tables plus the per-frame wind clock.
///
/// Carried as raw byte slices for the same reason [`VgFrameData`] is: `libhelio` holds the
/// inter-pass contract and must not depend on the crate that defines `GpuFoliageType` /
/// `GpuFoliageLayer`, or every pass crate would be forced to link the foliage crate to see
/// `FrameResources`. The producer and the consuming passes agree on the element type; this
/// struct only carries bytes and counts. Publishing a slice whose length is not
/// `count * size_of::<element>()` is therefore undetectable here and shows up as garbage
/// densities and blades placed under the world — bytemuck-cast on the publishing side, do
/// not hand-roll the slice.
///
/// The `FoliagePlacePass` uploads the tables on the first frame and whenever `generation`
/// advances, mirroring `VgFrameData::buffer_version`.
#[derive(Clone, Copy)]
pub struct FoliageFrameData<'a> {
    /// Raw bytes of a `GpuFoliageType` array — one entry per authored foliage type.
    pub types: &'a [u8],
    /// Raw bytes of a `GpuFoliageLayer` array — one entry per authored foliage layer.
    pub layers: &'a [u8],
    /// Number of valid entries in `types`. Foliage type ids index this array directly, so
    /// a stale count silently reads past the end of the table on the GPU.
    pub type_count: u32,
    /// Number of valid entries in `layers`.
    pub layer_count: u32,

    /// Global wind state for this frame, including both timestamps.
    ///
    /// Lives here rather than in its own `Tracked` slot because wind is only ever
    /// meaningful when there is foliage to move, and because every foliage pass that
    /// needs it already reads this struct. See [`GpuWind::time_prev_time`] for why the
    /// second timestamp cannot be dropped.
    pub wind: GpuWind,

    /// Version counter incremented when the type or layer tables change.
    ///
    /// **Wind must not advance this.** `wind` changes every single frame; if the
    /// publisher folds it into the generation, the type and layer tables are re-uploaded
    /// every frame and the residency cache's whole point — that steady-state foliage costs
    /// nothing on the CPU — is lost. Tables change on authoring edits only.
    pub generation: u64,
}

/// Per-frame secondary-view descriptors for portal eyes and sublevel cameras.
///
/// Carried as a raw byte slice for the same reason [`FoliageFrameData`] is: `libhelio`
/// holds the inter-pass contract and must not depend on the crate that defines
/// `GpuSecondaryView`. The producer and the consuming passes agree on the layout — an
/// array of `GpuSecondaryView` entries — and interpret the bytes identically.
#[derive(Clone, Copy)]
pub struct SecondaryFrameData<'a> {
    /// Raw bytes of a `[GpuSecondaryView; MAX_SECONDARY_VIEWS]` array.
    /// Only the first `view_count` entries are meaningful.
    pub view_bytes: &'a [u8],
    /// Number of active secondary views this frame.
    pub view_count: u32,
}

/// Views into the top-down foliage terrain capture.
///
/// Produced by `FoliageTerrainPass` over the active foliage ring, consumed by
/// `FoliagePlacePass`, `FoliageInteractionPass` and the far-ring terrain-shading fallback.
///
/// Both textures cover the same camera-relative, texel-snapped ring extent (default 256 m
/// at 4 texels/m) and must be sampled with the same transform. They are re-rendered only
/// for tiles whose residency or generation changed; the snap is what stops the capture
/// swimming under camera motion, so any consumer that derives its own unsnapped UVs
/// reintroduces exactly the shimmer the snapping exists to remove.
#[derive(Clone, Copy)]
pub struct FoliageTerrainViews<'a> {
    /// Terrain height (R) + slope (G) — `Rg16Float`.
    ///
    /// Slope is stored as `cos(angle)` so the placement shader tests a foliage type's
    /// `slope_range` acceptance band with two compares and no trig.
    pub height_slope: &'a wgpu::TextureView,
    /// Packed world normal (RGB) + material id (A) — `Rgba8Unorm`.
    ///
    /// The material id is what lets procedural density rules key off the surface
    /// (e.g. the voxel `MAT_GRASS` palette entry) without the placement shader knowing
    /// which terrain representation produced it.
    pub normal_material: &'a wgpu::TextureView,
}

