//! Per-frame transient resource views.
//!
//! `FrameResources` holds borrowed references to the transient textures that the
//! `RenderGraph` owns. These are passed into `PassContext` and `PrepareContext` so
//! passes can read outputs of earlier passes without any allocation or locking.

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
    /// Atmospheric sky LUT (transmittance + aerial perspective)
    pub sky_lut: Tracked<&'a wgpu::TextureView>,
    /// Sky LUT sampler (linear, clamp)
    pub sky_lut_sampler: Tracked<&'a wgpu::Sampler>,
    /// SSAO result texture
    pub ssao: Tracked<&'a wgpu::TextureView>,
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

    /// Radiance Cascades cascade atlas texture view
    pub rc_view: Tracked<&'a wgpu::TextureView>,

    /// Main depth texture (for passes that need to copy/sample it)
    pub depth_texture: Tracked<&'a wgpu::Texture>,

    /// Corona particle emitter definitions (uploaded by the Renderer each frame)
    pub corona_emitters: Tracked<CoronaEmitterFrameData<'a>>,
}

impl<'a> FrameResources<'a> {
    /// Creates an empty (all-Tracked::empty) frame resources for the start of a frame.
    pub fn empty() -> Self {
        Self {
            gbuffer: Tracked::empty(),
            shadow_atlas: Tracked::empty(),
            static_shadow_atlas: Tracked::empty(),
            shadow_sampler: Tracked::empty(),
            hiz: Tracked::empty(),
            hiz_sampler: Tracked::empty(),
            sky_lut: Tracked::empty(),
            sky_lut_sampler: Tracked::empty(),
            ssao: Tracked::empty(),
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
            water_sim_texture: Tracked::empty(),
            water_sim_sampler: Tracked::empty(),
            water_hitboxes: Tracked::empty(),
            water_hitbox_count: 0,
            depth_texture: Tracked::empty(),
            rc_view: Tracked::empty(),
            corona_emitters: Tracked::empty(),
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
            reset_field!(shadow_atlas);
            reset_field!(static_shadow_atlas);
            reset_field!(shadow_sampler);
            reset_field!(hiz);
            reset_field!(hiz_sampler);
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
            reset_field!(water_sim_texture);
            reset_field!(water_sim_sampler);
            reset_field!(water_hitboxes);
            reset_field!(depth_texture);
            reset_field!(rc_view);
            reset_field!(corona_emitters);
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
