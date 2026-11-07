//! GPU-native post-processing pipeline for Helio.
//!
//! The tone map reads the frame's post settings ([`POST_SETTINGS`], a
//! `GpuPostProcessUniforms` buffer) and the scene colour the colour chain
//! hands it. Whoever resolves the scene's settings (the post-process volumes
//! feature) publishes them early in the frame, so every reader sees the
//! same values; without a publisher the tone map uses the defaults.
//!
//! Sub-stages (execution order in `execute()`):
//!   1. `cs_exposure`/`cs_exposure_reduce`/`cs_exposure_adapt` — metered log luminance and eye adaptation (compute)
//!   1b. `cs_local_exposure_grid`/`_blur`/`_curve` — bilateral grid of log luminance and a tone curve per cell, for local exposure (compute)
//!   2. `cs_bloom_down_extract` — extract brights from HDR → bloom mip 0 (compute)
//!   3. `cs_bloom_down`         — 2x downsample mip chain, 4 passes (compute)
//!   3b. `cs_bloom_upsample`    — progressive upsample-and-add, mip 3 → mip 0 (compute)
//!   4. `fs_uber`               — global and local exposure, bloom + lens composite, grade, tonemap, vignette, grain (render)
//!
//! Bind groups:
//!   Main BGLs (group 0): uniforms, samplers, hdr/depth, bloom, noise, custom, volumes, blend output
//!   Bloom BGL (group 1): per-dispatch bloom src (sampled) + dst (storage write)
//!   Bloom upsample BGL (group 2): mip i + coarser sum (sampled) + sum i (storage write)
//!
//! See also `postprocess.wgsl` for shader-level injection points:
//!   INJECTION_POINT_0 (pre-blend), INJECTION_POINT_1 (post-tonemap),
//!   INJECTION_POINT_2 (post-grain), INJECTION_POINT_3 (final)

use bytemuck;
use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

pub mod gpu_types;
pub use gpu_types::*;

mod lut_builder;
pub use lut_builder::LutBuilder;

#[cfg(test)]
mod abi_tests;
#[cfg(test)]
mod exposure_tests;
#[cfg(test)]
mod local_exposure_tests;
#[cfg(test)]
mod tonemap_tests;

const BLOOM_MIPS: u32 = 5;
const WG_BLOOM: u32 = 8;
const WG_EXPOSURE_X: u32 = 16;
const WG_EXPOSURE_Y: u32 = 16;
const EXPOSURE_STRIDE: u32 = 4;
fn exposure_groups(width: u32, height: u32) -> (u32, u32) {
    (
        width.max(1).div_ceil(EXPOSURE_STRIDE * WG_EXPOSURE_X),
        height.max(1).div_ceil(EXPOSURE_STRIDE * WG_EXPOSURE_Y),
    )
}
fn exposure_partial_buffer(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Buffer {
    let (gx, gy) = exposure_groups(width, height);
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PostProcess Exposure Partials"),
        // A 64-bin histogram a workgroup (`cs_exposure`).
        size: u64::from(gx) * u64::from(gy) * 64 * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}
/// Local exposure's bilateral grid: screen cells per column (rows follow the
/// aspect ratio, so cells stay square at any resolution), luminance bins (1 EV
/// each; `LE_BINS` in the shader) and the blur's workgroup size.
const LE_GRID_ROWS: u32 = 32;
const LE_BINS: u32 = 24;
const WG_LE_BLUR: u32 = 8;

/// The grid's screen cells for an analysed image of `width` x `height`.
fn local_exposure_grid_dims(width: u32, height: u32) -> (u32, u32) {
    let columns = (LE_GRID_ROWS as f32 * width.max(1) as f32 / height.max(1) as f32).round() as u32;
    (columns.max(1), LE_GRID_ROWS)
}

/// Local exposure's bilateral grid: gathered into `raw` by
/// `cs_local_exposure_grid`, blurred into `blurred` by `cs_local_exposure_blur`,
/// made a tone curve per cell in `texture` by `cs_local_exposure_curve`, and
/// sampled by `fs_uber`. Recreated with the analysed size.
struct LocalExposureGrid {
    cells: (u32, u32),
    raw: wgpu::Buffer,
    blurred: wgpu::Buffer,
    _texture: wgpu::Texture,
    storage_view: wgpu::TextureView,
    sampled_view: wgpu::TextureView,
}

impl LocalExposureGrid {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let cells = local_exposure_grid_dims(width, height);
        // (sum of EV, weight) a bin.
        let grid = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: u64::from(cells.0 * cells.1 * LE_BINS) * 8,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let raw = grid("PostProcess Local Exposure Grid");
        let blurred = grid("PostProcess Local Exposure Blurred Grid");
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("PostProcess Local Exposure Curves"),
            size: wgpu::Extent3d {
                width: cells.0,
                height: cells.1,
                depth_or_array_layers: LE_BINS,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let storage_view = texture.create_view(&Default::default());
        let sampled_view = texture.create_view(&Default::default());
        Self {
            cells,
            raw,
            blurred,
            _texture: texture,
            storage_view,
            sampled_view,
        }
    }

    /// Workgroups of `cs_local_exposure_grid` (one a screen cell), then of
    /// `cs_local_exposure_blur` and `cs_local_exposure_curve`.
    fn workgroups(&self) -> [[u32; 3]; 3] {
        let (x, y) = (
            self.cells.0.div_ceil(WG_LE_BLUR),
            self.cells.1.div_ceil(WG_LE_BLUR),
        );
        [[self.cells.0, self.cells.1, 1], [x, y, LE_BINS], [x, y, 1]]
    }
}

/// Graph key of the post settings the tone map reads this frame (a
/// `GpuPostProcessUniforms` buffer, uniform and storage): the post-settings
/// contract. Whoever resolves the scene's post settings (post-process
/// volumes) publishes it; without it the tone map uses the host's
/// [`POST_BASELINE`].
pub const POST_SETTINGS: &str = "postprocess_uniforms";

/// The host's post settings: what the frame's post settings are where
/// nothing overrides them. The tone map's stand-in for [`POST_SETTINGS`],
/// and the base a publisher of those (post-process volumes) blends over.
/// Without it, `PostProcessSettings::default()`.
pub const POST_BASELINE: helio_core::PayloadContract<PostProcessSettings> =
    helio_core::PayloadContract::new("post_settings_baseline");

/// The camera's 3D colour-grading LUT (a renderer config's `features`):
/// `Rgba16Float`, `TextureDimension::D3`. Without it grading uses the
/// identity LUT.
#[derive(Clone, Debug)]
pub struct ColorGradingLut(pub wgpu::TextureView);

/// `bool` registry key: false only when no source the post settings' publisher blends from
/// (defaults, camera rows, volumes) can enable bloom, so the resolved
/// `bloom_enabled` is certainly 0 this frame and nothing samples the bloom
/// mips. Hosts register these columns up front, so the check reads the rows
/// back when they change (`SceneBufferLiveness`) and is true until they have
/// been read.
pub const BLOOM_MAYBE_ACTIVE: &str = "bloom_maybe_active";

/// `bool` registry key: false only when no source the post settings' publisher blends from
/// can select auto exposure, so the resolved `exposure_mode` is certainly
/// manual this frame and nothing reads the metered luminance. Tracked like
/// [`BLOOM_MAYBE_ACTIVE`].
pub const AUTO_EXPOSURE_MAYBE_ACTIVE: &str = "auto_exposure_maybe_active";

/// `GpuPostProcessUniforms` in WGSL, for the post settings' publishers and
/// readers ([`with_post_settings`]).
pub const POST_SETTINGS_WGSL: &str = include_str!("../shaders/post_settings.wgsl");

/// `source` (a shader opting into Helio's prelude) with the post settings'
/// struct after its first line.
pub fn with_post_settings(source: &str) -> String {
    match source.split_once('\n') {
        Some((first, rest)) => format!("{first}\n{POST_SETTINGS_WGSL}\n{rest}"),
        None => format!("{POST_SETTINGS_WGSL}\n{source}"),
    }
}

/// Maximum active volumes blended per frame. SceneDB buffers can be larger:
/// rows are addressed by sparse entity index, so shaders scan the full buffer
/// before compacting into this fixed active-volume budget.
pub const MAX_PP_VOLUMES: u32 = 64;
const _: () = assert!(MAX_PP_VOLUMES == 64);

/// A host's WGSL effects compiled into the post-process shader (see
/// [`PostProcessPass::new_with_user_effects`]), as a feature setting: a
/// graph's post-process node builds with it.
#[derive(Clone, Copy, Debug)]
pub struct UserEffects(pub &'static str);

/// Position in the uber-shader effect chain where a user effect is injected.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UserEffectPosition {
    PreBlend = 0,    // before exposure / bloom / color grade
    PostTonemap = 1, // after tonemap, before vignette / grain
    PostGrain = 2,   // after grain, before motion blur
    Final = 3,       // after all built-in effects
}

/// A single user-defined effect entry in the shader injection chain.
#[derive(Clone, Debug)]
pub struct UserEffectEntry {
    /// Which injection point to splice into.
    pub position: UserEffectPosition,
    /// Complete WGSL function body (e.g. `return color * 0.5;`).
    /// The function receives `(color: vec3<f32>, uv: vec2<f32>, dims: vec2<f32>) -> vec3<f32>`.
    pub body: String,
}

pub struct PostProcessPass {
    color_input: &'static str,
    color_alias: Option<&'static str>,
    /// The glare an optical effect published (`helio_frame_api::color::glare`),
    /// composited with the scene's exposure; black without one.
    glare: Option<&'static str>,
    /// The host's [`POST_BASELINE`] (`PostProcessSettings::default()`
    /// without one): the post settings while nothing publishes
    /// [`POST_SETTINGS`].
    default_settings: wgpu::Buffer,
    /// The baseline `default_settings` holds.
    baseline: GpuPostProcessUniforms,
    uber_timing_query: Option<wgpu::QuerySet>,
    compute_timing_query: Option<wgpu::QuerySet>,
    avg_luminance_buf: wgpu::Buffer,
    exposure_partials_buf: wgpu::Buffer,

    exposure_pipeline: wgpu::ComputePipeline,
    exposure_reduce_pipeline: wgpu::ComputePipeline,
    exposure_adapt_pipeline: wgpu::ComputePipeline,
    local_exposure_grid_pipeline: wgpu::ComputePipeline,
    local_exposure_blur_pipeline: wgpu::ComputePipeline,
    local_exposure_curve_pipeline: wgpu::ComputePipeline,
    local_exposure: LocalExposureGrid,
    bloom_extract_pipeline: wgpu::ComputePipeline,
    bloom_down_pipeline: wgpu::ComputePipeline,
    bloom_upsample_pipeline: wgpu::ComputePipeline,
    /// The image's pipeline, and its debug-view variant (`DEBUG_VIEW`).
    uber_pipelines: [wgpu::RenderPipeline; 2],
    /// A renderer debug view is on: the scene colour is shown as it is.
    debug_view: bool,

    // Separate BGLs for compute vs render
    compute_main_bgl: wgpu::BindGroupLayout,
    render_main_bgl: wgpu::BindGroupLayout,
    bloom_compute_bgl: wgpu::BindGroupLayout,
    bloom_upsample_bgl: wgpu::BindGroupLayout,

    compute_main_bg: Option<wgpu::BindGroup>,
    render_main_bg: Option<wgpu::BindGroup>,
    main_bg_key: Option<(
        wgpu::TextureView,
        wgpu::TextureView,
        wgpu::Buffer,
        wgpu::Buffer,
        Option<wgpu::TextureView>,
        Option<wgpu::TextureView>,
        Option<wgpu::TextureView>,
    )>,

    // Bloom BGs
    bloom_extract_bg: Option<(wgpu::TextureView, wgpu::BindGroup)>,
    bloom_down_bgs: Vec<wgpu::BindGroup>,
    /// One per upsample level, `[i]` writing `bloom_sums[i]`.
    bloom_upsample_bgs: Vec<wgpu::BindGroup>,

    bloom_textures: Vec<wgpu::Texture>,
    bloom_sampled_views: Vec<wgpu::TextureView>,
    bloom_storage_views: Vec<wgpu::TextureView>,
    /// `bloom_sums[i]` = mip i plus the B-spline upsample of `bloom_sums[i + 1]`
    /// (mip 4 for the last); `bloom_sums[0]` is fs_uber's bloom input.
    bloom_sums: Vec<BloomSum>,

    linear_sampler: wgpu::Sampler,
    point_sampler: wgpu::Sampler,

    width: u32,
    /// Size of the analysed input, which bloom mips and exposure partials follow.
    analysis_size: (u32, u32),
    height: u32,
    format: wgpu::TextureFormat,

    /// Adapted exposure in `avg_luminance_buf[1]` is valid history. Cleared on
    /// construction and resize so the first metered frame snaps.
    exposure_history: bool,

    // ── Bloom gating ───────────────────────────────────────────────────────
    bloom_active: bool,

    // ── Custom effect infrastructure ───────────────────────────────────────
    noise_texture: wgpu::Texture,
    noise_view: wgpu::TextureView,
    noise_sampler: wgpu::Sampler,
    /// 1x1 black stand-in bound at b17 when nothing published glare.
    fallback_lens_view: wgpu::TextureView,
    /// 1x1 (0,0) stand-in bound at b18 when the graph has no velocity pass.
    fallback_velocity_view: wgpu::TextureView,
    /// 1x1x1 identity LUT bound at b19 when no LUT is loaded.
    fallback_lut_view: wgpu::TextureView,
    /// The grading LUT ([`ColorGradingLut`]); identity without.
    color_grading_lut: Option<wgpu::TextureView>,
    custom_params_buf: wgpu::Buffer,
    custom_params: Vec<[f32; 4]>,

    uber_pl: wgpu::PipelineLayout,

    // Current user shader snippet (the one baked into uber_pipelines).
    user_shader_snippet: Option<String>,
    // Pending snippet queued by set_user_shader — applied in prepare().
    pending_shader_snippet: Option<String>,
    // Multi-effect chain entries.
    user_effect_entries: Vec<UserEffectEntry>,
    // Cached built shader source to avoid rebuilding identical configs.
    cached_shader_source: Option<String>,
}

/// The uber pipeline for `format`, and its debug-view variant: a renderer
/// debug view (buffers, GI, voxel levels) shows the scene colour as the
/// view wrote it, without the camera's exposure, grade or tone curve.
fn uber_pipelines(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    format: wgpu::TextureFormat,
) -> [wgpu::RenderPipeline; 2] {
    [false, true].map(|debug_view| {
        helio_core::shader::create_render_pipeline(
            device,
            wgpu::RenderPipelineDescriptor {
                label: Some(if debug_view {
                    "PostProcess Uber (debug view)"
                } else {
                    "PostProcess Uber"
                }),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("vs_fullscreen"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some("fs_uber"),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &[("DEBUG_VIEW", f64::from(u8::from(debug_view)))],
                        ..Default::default()
                    },
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        )
    })
}

impl PostProcessPass {
    /// Select the linear HDR input published by the preceding graph pass.
    /// Grades with `lut` instead of the identity LUT.
    pub fn with_color_grading_lut(mut self, lut: Option<wgpu::TextureView>) -> Self {
        self.color_grading_lut = lut;
        self
    }

    pub fn with_color_input(mut self, key: &'static str) -> Self {
        self.color_input = key;
        self
    }

    /// The image the colour input may be a view of
    /// (`helio_frame_api::color::alias`), read too so it stays alive.
    pub fn with_color_alias(mut self, alias: Option<&'static str>) -> Self {
        self.color_alias = alias;
        self
    }

    /// The glare to composite (`helio_frame_api::color::glare`): light an
    /// optical effect adds inside the tone map with the scene's exposure.
    pub fn with_glare(mut self, glare: Option<&'static str>) -> Self {
        self.glare = glare;
        self
    }
    /// Optional GPU timestamps around the full-screen tonemap and effects pass.
    pub fn enable_uber_timing(&mut self, device: &wgpu::Device) -> bool {
        if !device.features().contains(
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
        ) {
            return false;
        }
        self.uber_timing_query = Some(device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Postprocess uber timings"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        }));
        true
    }

    pub fn uber_timing_query(&self) -> Option<&wgpu::QuerySet> {
        self.uber_timing_query.as_ref()
    }

    /// Optional GPU timestamps around the compute work: [0] start, [1] after
    /// exposure, [2] after local exposure, [3] after bloom.
    pub fn enable_compute_timing(&mut self, device: &wgpu::Device) -> bool {
        if !device.features().contains(
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
        ) {
            return false;
        }
        self.compute_timing_query = Some(device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Postprocess compute timings"),
            ty: wgpu::QueryType::Timestamp,
            count: 4,
        }));
        true
    }

    pub fn compute_timing_query(&self) -> Option<&wgpu::QuerySet> {
        self.compute_timing_query.as_ref()
    }

    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        Self::new_with_user_effects(device, queue, width, height, format, None)
    }

    pub fn new_with_user_effects(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        user_effects_fn: Option<&str>,
    ) -> Self {
        let initial_entries = user_effects_fn
            .map(|body| {
                vec![UserEffectEntry {
                    // Legacy API: inject at FINAL position to match old pass-through
                    // behavior where user_effects was the only thing running.
                    position: UserEffectPosition::Final,
                    body: body.to_string(),
                }]
            })
            .unwrap_or_default();

        let initial_src = Self::build_shader_source(&initial_entries);
        let shader = helio_core::shader::module(device, "PostProcess Shader", &initial_src);

        let avg_luminance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PostProcess Avg Luminance"),
            // measured log2, adapted log2, delta seconds, history valid.
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let exposure_partials_buf = exposure_partial_buffer(device, width, height);

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("PostProcess Linear Sampler"),
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let point_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("PostProcess Point Sampler"),
            min_filter: wgpu::FilterMode::Nearest,
            mag_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let (bloom_textures, bloom_sampled_views, bloom_storage_views) =
            Self::create_bloom_mips(device, width, height);
        let bloom_sums = BloomSum::chain(device, width, height);
        let local_exposure = LocalExposureGrid::new(device, width, height);

        // ── Shared BGL entry helpers ────────────────────────────────────────

        let uniform_entry = |binding: u32, vis: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let sampled_tex_entry =
            |binding: u32, vis: wgpu::ShaderStages, depth: bool| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: vis,
                ty: wgpu::BindingType::Texture {
                    sample_type: if depth {
                        wgpu::TextureSampleType::Depth
                    } else {
                        wgpu::TextureSampleType::Float { filterable: true }
                    },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            };
        let sampler_entry =
            |binding: u32, vis: wgpu::ShaderStages, filtering: bool| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: vis,
                ty: wgpu::BindingType::Sampler(if filtering {
                    wgpu::SamplerBindingType::Filtering
                } else {
                    wgpu::SamplerBindingType::NonFiltering
                }),
                count: None,
            };
        let storage_buf_entry =
            |binding: u32, vis: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: vis,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            };
        let storage_ro_entry = |binding: u32, vis: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let cv = wgpu::ShaderStages::COMPUTE;
        let fv = wgpu::ShaderStages::FRAGMENT;
        let cfv = wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT;

        // ── compute_main_bgl: exposure partials and shared postprocess inputs ──
        let compute_main_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PostProcess Compute Main BGL"),
            entries: &[
                uniform_entry(0, cfv),
                storage_ro_entry(1, cfv),
                sampled_tex_entry(2, cfv, false),
                // Depth: metering skips sky and space (`cs_exposure`).
                sampled_tex_entry(3, cfv, true),
                sampler_entry(4, cfv, true),
                sampler_entry(5, fv, false),
                storage_buf_entry(11, cfv),
                storage_buf_entry(15, cv),
                sampled_tex_entry(12, cfv, false),
                sampler_entry(13, cfv, false),
                storage_ro_entry(14, cfv),
                storage_buf_entry(23, cv),
                storage_buf_entry(26, cv),
                wgpu::BindGroupLayoutEntry {
                    binding: 24,
                    visibility: cv,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
            ],
        });

        // ── render_main_bgl: b0-b14 (bloom sampled at b6-b10, mips 1-4 summed at b21) ─
        let render_main_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PostProcess Render Main BGL"),
            entries: &[
                uniform_entry(0, fv),
                storage_ro_entry(1, fv),
                sampled_tex_entry(2, fv, false),
                sampled_tex_entry(3, fv, true),
                sampler_entry(4, fv, true),
                sampler_entry(5, fv, false),
                sampled_tex_entry(6, fv, false),
                sampled_tex_entry(7, fv, false),
                sampled_tex_entry(8, fv, false),
                sampled_tex_entry(9, fv, false),
                sampled_tex_entry(10, fv, false),
                sampled_tex_entry(21, fv, false),
                storage_buf_entry(11, fv),
                sampled_tex_entry(12, fv, false),
                sampler_entry(13, fv, false),
                storage_ro_entry(14, fv),
                // Reduced-resolution scene-linear lens response.
                sampled_tex_entry(17, fv, false),
                // Velocity buffer (Rg16Float) for per-pixel motion blur
                wgpu::BindGroupLayoutEntry {
                    binding: 18,
                    visibility: fv,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 3D LUT texture for colour grading
                wgpu::BindGroupLayoutEntry {
                    binding: 19,
                    visibility: fv,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // Local exposure's blurred bilateral grid.
                wgpu::BindGroupLayoutEntry {
                    binding: 25,
                    visibility: fv,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        // ── bloom_compute_bgl: per-dispatch src + dst ──────────────────────
        let bloom_compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PostProcess Bloom Compute BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: cv,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: cv,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // ── bloom_upsample_bgl: mip i + coarser sum sampled, sum i dst ────
        let bloom_upsample_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("PostProcess Bloom Upsample BGL"),
                entries: &[
                    sampled_tex_entry(0, cv, false),
                    sampled_tex_entry(1, cv, false),
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: cv,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });
        let bloom_upsample_bgs = Self::make_bloom_upsample_bgs(
            device,
            &bloom_upsample_bgl,
            &bloom_sampled_views,
            &bloom_sums,
        );

        // Precompute bloom_down BGs
        let bloom_down_bgs = Self::make_bloom_down_bgs(
            device,
            &bloom_compute_bgl,
            &bloom_sampled_views,
            &bloom_storage_views,
        );

        // ── Pipeline layouts ──────────────────────────────────────────────────
        let exposure_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PostProcess Exposure PL"),
            bind_group_layouts: &[Some(&compute_main_bgl)],
            immediate_size: 0,
        });
        let bloom_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PostProcess Bloom PL"),
            bind_group_layouts: &[Some(&compute_main_bgl), Some(&bloom_compute_bgl)],
            immediate_size: 0,
        });
        // Group 0 for the shared linear sampler; group 1 (per-dispatch
        // down-sample src/dst) is unused by cs_bloom_upsample.
        let bloom_upsample_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PostProcess Bloom Upsample PL"),
            bind_group_layouts: &[Some(&compute_main_bgl), None, Some(&bloom_upsample_bgl)],
            immediate_size: 0,
        });
        let render_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PostProcess Render PL"),
            bind_group_layouts: &[Some(&render_main_bgl)],
            immediate_size: 0,
        });

        let mk_compute = |label: &str, entry: &str, layout: &wgpu::PipelineLayout| {
            helio_core::shader::create_compute_pipeline(
                &device,
                wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(layout),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                },
            )
        };

        let exposure_pipeline = mk_compute("PostProcess Exposure", "cs_exposure", &exposure_pl);
        let exposure_reduce_pipeline = mk_compute(
            "PostProcess Exposure Reduce",
            "cs_exposure_reduce",
            &exposure_pl,
        );
        let exposure_adapt_pipeline = mk_compute(
            "PostProcess Exposure Adapt",
            "cs_exposure_adapt",
            &exposure_pl,
        );
        // Zeroes its own bins: naga's zeroing would do it from one invocation.
        let local_exposure_grid_pipeline = helio_core::shader::create_compute_pipeline(
            &device,
            wgpu::ComputePipelineDescriptor {
                label: Some("PostProcess Local Exposure Grid"),
                layout: Some(&exposure_pl),
                module: &shader,
                entry_point: Some("cs_local_exposure_grid"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    zero_initialize_workgroup_memory: false,
                    ..Default::default()
                },
                cache: None,
            },
        );
        let local_exposure_blur_pipeline = mk_compute(
            "PostProcess Local Exposure Blur",
            "cs_local_exposure_blur",
            &exposure_pl,
        );
        let local_exposure_curve_pipeline = mk_compute(
            "PostProcess Local Exposure Curve",
            "cs_local_exposure_curve",
            &exposure_pl,
        );
        let bloom_extract_pipeline = mk_compute(
            "PostProcess Bloom Extract",
            "cs_bloom_down_extract",
            &bloom_pl,
        );
        let bloom_down_pipeline = mk_compute("PostProcess Bloom Down", "cs_bloom_down", &bloom_pl);
        let bloom_upsample_pipeline = mk_compute(
            "PostProcess Bloom Upsample",
            "cs_bloom_upsample",
            &bloom_upsample_pl,
        );

        let uber_pipelines = uber_pipelines(&device, &render_pl, &shader, format);

        // ── Noise texture ──────────────────────────────────────────────────
        let noise_size = 64u32;
        let noise_data: Vec<u8> = {
            let mut rng: u64 = 987654321;
            let mut data = Vec::with_capacity((noise_size * noise_size) as usize);
            for _ in 0..noise_size * noise_size {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                data.push((rng >> 40) as u8);
            }
            data
        };
        let noise_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("PostProcess Noise"),
            size: wgpu::Extent3d {
                width: noise_size,
                height: noise_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &noise_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &noise_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(noise_size),
                rows_per_image: Some(noise_size),
            },
            wgpu::Extent3d {
                width: noise_size,
                height: noise_size,
                depth_or_array_layers: 1,
            },
        );
        let noise_view = noise_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let noise_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("PostProcess Noise Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        // Stand-in for the glare when nothing published one.
        // Textures are zero-initialised, and black adds nothing.
        let fallback_lens_view = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("PostProcess Lens Fallback"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());

        // 1x1 zero-velocity fallback for when the graph has no GBuffer pass.
        let fallback_velocity_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("PostProcess Velocity Fallback"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &fallback_velocity_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[0u8; 4],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let fallback_velocity_view =
            fallback_velocity_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // 1x1x1 identity LUT fallback (maps (0,0,0)→black, (1,1,1)→white)
        let fallback_lut_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("PostProcess LUT Fallback"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // Identity: (0,0,0,1) = black input maps to black, but for LUT we use
        // the identity where coord (r,g,b) returns (r,g,b). 1x1x1 can't hold
        // a true identity but it won't be sampled when lut_platform == 0.
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &fallback_lut_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[
                0x00, 0x00, 0x80, 0x3C, 0x00, 0x00, 0x80, 0x3C, 0x00, 0x00, 0x80, 0x3C, 0x00, 0x00,
                0x80, 0x3C,
            ],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(16),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let fallback_lut_view = fallback_lut_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });

        let custom_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PostProcess Custom Params"),
            size: 64 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let stored_snippet = user_effects_fn.map(|s| s.to_string());
        let default_settings = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("PostProcess Default Settings"),
                contents: bytemuck::bytes_of(&PostProcessSettings::default().to_gpu()),
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST,
            },
        );

        Self {
            color_input: helio_frame_api::color::SCENE_COLOR,
            color_alias: None,
            glare: None,
            default_settings,
            baseline: PostProcessSettings::default().to_gpu(),
            uber_timing_query: None,
            compute_timing_query: None,
            avg_luminance_buf,
            exposure_partials_buf,
            exposure_pipeline,
            exposure_reduce_pipeline,
            exposure_adapt_pipeline,
            local_exposure_grid_pipeline,
            local_exposure_blur_pipeline,
            local_exposure_curve_pipeline,
            local_exposure,
            bloom_extract_pipeline,
            bloom_down_pipeline,
            bloom_upsample_pipeline,
            uber_pipelines,
            debug_view: false,
            compute_main_bgl,
            render_main_bgl,
            bloom_compute_bgl,
            bloom_upsample_bgl,
            compute_main_bg: None,
            render_main_bg: None,
            main_bg_key: None,
            bloom_extract_bg: None,
            bloom_down_bgs,
            bloom_upsample_bgs,
            bloom_textures,
            bloom_sampled_views,
            bloom_storage_views,
            bloom_sums,
            linear_sampler,
            point_sampler,
            width,
            analysis_size: (width, height),
            height,
            format,
            exposure_history: false,
            bloom_active: true,
            noise_texture,
            noise_view,
            noise_sampler,
            fallback_lens_view,
            fallback_velocity_view,
            fallback_lut_view,
            color_grading_lut: None,
            custom_params_buf,
            custom_params: Vec::new(),
            user_shader_snippet: stored_snippet,
            pending_shader_snippet: None,
            user_effect_entries: initial_entries,
            cached_shader_source: Some(initial_src),
            uber_pl: render_pl,
        }
    }

    // ── Shader source builder ────────────────────────────────────────────────

    /// Build the complete WGSL source by splicing user effect entries into the
    /// base shader at `//%P0` through `//%P3` markers.
    ///
    /// Each entry is either:
    /// - A complete `fn user_effects(...)` definition (old API via `new_with_user_effects`)
    ///   → placed verbatim at module scope; a call is emitted at the marker.
    /// - A bare expression body (new API via `add_user_effect`)
    ///   → wrapped in a generated `fn` and placed at module scope; a call emitted at the marker.
    fn build_shader_source(entries: &[UserEffectEntry]) -> String {
        // Fetched as text: user effects are spliced into it below, so it is
        // not handed to `module` as an `include_wgsl!` whole.
        let base = helio_core::shader::source_text(
            "PostProcess Shader",
            helio_core::include_wgsl!("../shaders/postprocess.wgsl"),
        );
        let mut result = with_post_settings(&base);

        // Collect module-scope definitions and per-position calls.
        let mut defs = String::new();
        let mut calls_by_pos: [Vec<String>; 4] = [vec![], vec![], vec![], vec![]];
        let mut entry_index = 0usize;

        for e in entries {
            let pos = e.position as usize;
            if pos >= 4 {
                continue;
            }

            let trimmed = e.body.trim();

            if trimmed.starts_with("fn ") {
                // Old API: one or more complete function definitions.
                // The last `fn` is usually the main `user_effects(...)` entry point.
                // Extract its name for the call site.
                let fn_name = "user_effects";
                defs.push_str(&format!("{}\n", trimmed));
                calls_by_pos[pos].push(format!("    color = {}(color, uv, dims);\n", fn_name));
            } else {
                // New API: bare expression body — wrap in a generated function.
                let fn_name = format!("userfx_{}", entry_index);
                entry_index += 1;
                defs.push_str(&format!(
                    "fn {}(color: vec3<f32>, uv: vec2<f32>, dims: vec2<f32>) -> vec3<f32> {{ return {}; }}\n",
                    fn_name, trimmed
                ));
                calls_by_pos[pos].push(format!("    color = {}(color, uv, dims);\n", fn_name));
            }
        }

        // Replace markers with calls, then append definitions at module scope.
        for (pos, calls) in calls_by_pos.iter().enumerate() {
            let marker = format!("//%P{}", pos);
            if calls.is_empty() {
                result = result.replace(&marker, &format!("//%P{} (empty)", pos));
            } else {
                let splice: String = calls.iter().flat_map(|c| c.chars()).collect();
                result = result.replace(&marker, &splice);
            }
        }

        if !defs.is_empty() {
            result.push_str("\n// ── Injected user effects ──\n");
            result.push_str(&defs);
        }

        result
    }

    fn rebuild_uber_from_entries(&mut self, device: &wgpu::Device) {
        let source = Self::build_shader_source(&self.user_effect_entries);
        if self.cached_shader_source.as_deref() == Some(&source) {
            return; // identical — skip rebuild
        }
        let shader_mod = helio_core::shader::module(device, "PostProcess Shader", &source);
        self.uber_pipelines = uber_pipelines(device, &self.uber_pl, &shader_mod, self.format);
        self.cached_shader_source = Some(source);
    }

    // ── Public API ───────────────────────────────────────────────────────────

    /// Gate bloom compute dispatches on/off.
    /// Call each frame from the renderer when the blended bloom settings are known.
    pub fn set_bloom_active(&mut self, active: bool) {
        self.bloom_active = active;
    }

    /// Queue a new user shader snippet to be applied at the start of the next frame.
    /// The pipeline rebuild happens in `prepare()`, not on the calling thread.
    /// Pass `None` to restore the default no-op.
    pub fn set_user_shader(&mut self, wgsl: Option<&str>) {
        self.pending_shader_snippet = wgsl.map(|s| s.to_string());
    }

    /// Add a user effect entry at the given position in the effect chain.
    /// Effects are applied in the order they are added.
    /// Call `commit_user_effects()` to rebuild the uber-pipeline.
    pub fn add_user_effect(&mut self, position: UserEffectPosition, body: &str) {
        self.user_effect_entries.push(UserEffectEntry {
            position,
            body: body.to_string(),
        });
    }

    /// Remove all user effect entries and rebuild the pipeline.
    pub fn clear_user_effects(&mut self, device: &wgpu::Device) {
        self.user_effect_entries.clear();
        self.rebuild_uber_from_entries(device);
    }

    /// Rebuild the uber-pipeline with the current set of user effect entries.
    /// Called automatically if `set_user_shader()` is used (legacy path).
    pub fn commit_user_effects(&mut self, device: &wgpu::Device) {
        self.rebuild_uber_from_entries(device);
    }

    /// Upload custom float4 parameters that the shader reads from `pp_custom`.
    pub fn set_custom_params(&mut self, params: &[[f32; 4]]) {
        self.custom_params.clear();
        self.custom_params.extend_from_slice(params);
    }

    // ── Internal helpers ────────────────────────────────────────────────────

    fn create_bloom_mips(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (
        Vec<wgpu::Texture>,
        Vec<wgpu::TextureView>,
        Vec<wgpu::TextureView>,
    ) {
        let mut textures = Vec::with_capacity(BLOOM_MIPS as usize);
        let mut sampled_views = Vec::with_capacity(BLOOM_MIPS as usize);
        let mut storage_views = Vec::with_capacity(BLOOM_MIPS as usize);

        for i in 0..BLOOM_MIPS {
            let mw = (width >> (i + 1)).max(1);
            let mh = (height >> (i + 1)).max(1);
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Bloom Mip {}", i)),
                size: wgpu::Extent3d {
                    width: mw,
                    height: mh,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            sampled_views.push(tex.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("Bloom Mip {} Sampled", i)),
                ..Default::default()
            }));
            storage_views.push(tex.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("Bloom Mip {} Storage", i)),
                ..Default::default()
            }));
            textures.push(tex);
        }
        (textures, sampled_views, storage_views)
    }

    fn make_bloom_down_bgs(
        device: &wgpu::Device,
        bloom_compute_bgl: &wgpu::BindGroupLayout,
        bloom_sampled_views: &[wgpu::TextureView],
        bloom_storage_views: &[wgpu::TextureView],
    ) -> Vec<wgpu::BindGroup> {
        (1..BLOOM_MIPS as usize)
            .map(|i| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("PostProcess Bloom Down BG mip{}", i)),
                    layout: bloom_compute_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &bloom_sampled_views[i - 1],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&bloom_storage_views[i]),
                        },
                    ],
                })
            })
            .collect()
    }

    fn make_bloom_upsample_bgs(
        device: &wgpu::Device,
        bloom_upsample_bgl: &wgpu::BindGroupLayout,
        bloom_sampled_views: &[wgpu::TextureView],
        sums: &[BloomSum],
    ) -> Vec<wgpu::BindGroup> {
        fn view(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: wgpu::BindingResource::TextureView(view),
            }
        }
        (0..sums.len())
            .map(|i| {
                // The coarsest sum adds the last mip, which has nothing below it.
                let coarser = sums
                    .get(i + 1)
                    .map_or(&bloom_sampled_views[i + 1], |sum| &sum.sampled_view);
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("PostProcess Bloom Upsample BG {i}")),
                    layout: bloom_upsample_bgl,
                    entries: &[
                        view(0, &bloom_sampled_views[i]),
                        view(1, coarser),
                        view(2, &sums[i].storage_view),
                    ],
                })
            })
            .collect()
    }

    fn rebuild_bind_groups(
        &mut self,
        device: &wgpu::Device,
        postprocess_buf: &wgpu::Buffer,
        pre_aa_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera_buf: &wgpu::Buffer,
        lens_view: Option<&wgpu::TextureView>,
        velocity_view: Option<&wgpu::TextureView>,
        lut_view: Option<&wgpu::TextureView>,
    ) {
        let lens_view = lens_view.unwrap_or(&self.fallback_lens_view);
        self.compute_main_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PostProcess Compute Main BG"),
            layout: &self.compute_main_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: postprocess_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(pre_aa_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.point_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: self.avg_luminance_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: self.exposure_partials_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::TextureView(&self.noise_view),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: wgpu::BindingResource::Sampler(&self.noise_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: self.custom_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 23,
                    resource: self.local_exposure.raw.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 26,
                    resource: self.local_exposure.blurred.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 24,
                    resource: wgpu::BindingResource::TextureView(&self.local_exposure.storage_view),
                },
            ],
        }));

        let velocity_view = velocity_view.unwrap_or(&self.fallback_velocity_view);
        let lut_view = lut_view.unwrap_or(&self.fallback_lut_view);

        self.render_main_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PostProcess Render Main BG"),
            layout: &self.render_main_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: postprocess_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(pre_aa_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.point_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&self.bloom_sampled_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&self.bloom_sampled_views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&self.bloom_sampled_views[2]),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(&self.bloom_sampled_views[3]),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::TextureView(&self.bloom_sampled_views[4]),
                },
                wgpu::BindGroupEntry {
                    binding: 21,
                    resource: wgpu::BindingResource::TextureView(&self.bloom_sums[0].sampled_view),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: self.avg_luminance_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::TextureView(&self.noise_view),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: wgpu::BindingResource::Sampler(&self.noise_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: self.custom_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: wgpu::BindingResource::TextureView(lens_view),
                },
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: wgpu::BindingResource::TextureView(velocity_view),
                },
                wgpu::BindGroupEntry {
                    binding: 19,
                    resource: wgpu::BindingResource::TextureView(lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 25,
                    resource: wgpu::BindingResource::TextureView(&self.local_exposure.sampled_view),
                },
            ],
        }));
    }

    fn mip_dims(&self, mip: u32) -> (u32, u32) {
        (
            (self.analysis_size.0 >> (mip + 1)).max(1),
            (self.analysis_size.1 >> (mip + 1)).max(1),
        )
    }
}

impl RenderPass for PostProcessPass {
    fn name(&self) -> &'static str {
        "PostProcess"
    }

    fn set_debug_mode(&mut self, mode: u32) {
        self.debug_view = mode != 0;
    }

    fn reads(&self) -> &'static [&'static str] {
        // The scene colour it tone maps is the chain's (`declare_resources`).
        match self.glare {
            Some(glare) => helio_frame_api::static_names(&[POST_SETTINGS, glare]),
            None => &[POST_SETTINGS],
        }
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a helio_core::ResourceRegistry<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read(self.color_input);
        // The input may be a view of an earlier image
        // (`helio_frame_api::color::alias`); keep the pool from reusing it.
        if let Some(alias) = self.color_alias {
            builder.read(alias);
        }
        // The frame's post settings; the defaults without a publisher.
        builder.read(POST_SETTINGS);
        // Optional: without a published glare the uber shader binds a 1x1
        // black fallback.
        if let Some(glare) = self.glare {
            builder.read(glare);
        }
        // Optional: graphs without a GBuffer pass or velocity pass fall back to
        // a 1x1 zero-velocity texture (no per-object motion blur).
        builder.read(helio_frame_api::gbuffer::VELOCITY);
    }

    fn on_resize(&mut self, device: &wgpu::Device, size: helio_core::RenderSize) {
        let [width, height] = size.output;
        self.width = width;
        self.height = height;
        self.resize_analysis(device, width, height);
        self.exposure_history = false;
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        self.prepare_frame(ctx)
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        self.execute_frame(ctx)
    }
}

impl PostProcessPass {
    /// Size bloom mips and exposure partials from the image they analyse. The
    /// input can differ from the output size (internal-resolution FXAA input,
    /// dynamic resolution); a mismatch scales bloom about the image corner.
    fn resize_analysis(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.analysis_size = (width, height);
        self.exposure_partials_buf = exposure_partial_buffer(device, width, height);
        let (textures, sampled_views, storage_views) =
            Self::create_bloom_mips(device, width, height);
        self.bloom_textures = textures;
        self.bloom_sampled_views = sampled_views;
        self.bloom_storage_views = storage_views;
        self.bloom_down_bgs = Self::make_bloom_down_bgs(
            device,
            &self.bloom_compute_bgl,
            &self.bloom_sampled_views,
            &self.bloom_storage_views,
        );
        self.bloom_sums = BloomSum::chain(device, width, height);
        self.bloom_upsample_bgs = Self::make_bloom_upsample_bgs(
            device,
            &self.bloom_upsample_bgl,
            &self.bloom_sampled_views,
            &self.bloom_sums,
        );
        self.local_exposure = LocalExposureGrid::new(device, width, height);
        self.compute_main_bg = None;
        self.render_main_bg = None;
        self.main_bg_key = None;
        self.bloom_extract_bg = None;
    }

    fn prepare_frame(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        // Adaptation inputs for cs_exposure_adapt: frame time and whether the
        // adapted value is valid history (otherwise the first meter snaps).
        let adaptation = [
            ctx.delta_time.max(0.0),
            if self.exposure_history { 1.0 } else { 0.0 },
        ];
        ctx.write_buffer(
            &self.avg_luminance_buf,
            8,
            bytemuck::cast_slice(&adaptation),
        );
        self.exposure_history = true;

        let baseline = ctx
            .scene_buffers
            .payload(&POST_BASELINE)
            .map_or_else(|| PostProcessSettings::default().to_gpu(), |b| b.to_gpu());
        if bytemuck::bytes_of(&baseline) != bytemuck::bytes_of(&self.baseline) {
            ctx.write_buffer(&self.default_settings, 0, bytemuck::bytes_of(&baseline));
            self.baseline = baseline;
        }

        // Deferred shader rebuild: if a snippet was queued, apply it now.
        if self.pending_shader_snippet.is_some() {
            let pending = self.pending_shader_snippet.take().unwrap();
            let changed = self.user_shader_snippet.as_deref() != Some(pending.as_str());
            if changed {
                // Convert legacy single-snippet API into a single Final entry
                // (matching old pass-through behavior).
                self.user_effect_entries
                    .retain(|e| e.position != UserEffectPosition::Final);
                self.user_effect_entries.push(UserEffectEntry {
                    position: UserEffectPosition::Final,
                    body: pending.clone(),
                });
                self.rebuild_uber_from_entries(ctx.device);
                self.user_shader_snippet = Some(pending);
            }
        }

        // Upload custom params
        if !self.custom_params.is_empty() {
            ctx.write_buffer(
                &self.custom_params_buf,
                0,
                bytemuck::cast_slice(&self.custom_params),
            );
        }

        Ok(())
    }

    fn execute_frame(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let input_key = self.color_input;
        let pre_aa_view = match ctx
            .registry
            .get::<&wgpu::TextureView>(helio_core::ResourceKey::new(input_key))
        {
            Some(v) => v,
            None => return Ok(()),
        };
        let input_size = (
            pre_aa_view.texture().width(),
            pre_aa_view.texture().height(),
        );
        if input_size != self.analysis_size {
            self.resize_analysis(ctx.device, input_size.0, input_size.1);
        }
        let postprocess_buf = ctx
            .registry
            .get::<&wgpu::Buffer>(helio_core::ResourceKey::new(POST_SETTINGS))
            .map_or_else(|| self.default_settings.clone(), |buffer| (*buffer).clone());
        let postprocess_buf = &postprocess_buf;

        let camera_buf = ctx.camera;

        // None when nothing published glare; rebuild_bind_groups then
        // binds the 1x1 black fallback. Part of the key so that a glare pass
        // being added, removed, or resized rebuilds the group instead of
        // leaving b17 pointing at a stale view.
        let lens_view = self.glare.and_then(|glare| {
            ctx.registry
                .get::<&wgpu::TextureView>(helio_core::ResourceKey::new(glare))
        });
        let velocity_view = ctx
            .registry
            .get::<&wgpu::TextureView>(helio_core::ResourceKey::new(
                helio_frame_api::gbuffer::VELOCITY,
            ));
        let lut_view = self.color_grading_lut.clone();
        let lut_view = lut_view.as_ref();

        let bg_key = (
            pre_aa_view.clone(),
            ctx.depth.clone(),
            camera_buf.clone(),
            postprocess_buf.clone(),
            lens_view.cloned(),
            velocity_view.cloned(),
            lut_view.cloned(),
        );
        if self.main_bg_key.as_ref() != Some(&bg_key) {
            self.rebuild_bind_groups(
                ctx.device,
                postprocess_buf,
                pre_aa_view,
                ctx.depth,
                camera_buf,
                lens_view,
                velocity_view,
                lut_view,
            );
            self.main_bg_key = Some(bg_key);
        }

        // Bloom extract BG
        if self.bloom_extract_bg.as_ref().map(|(k, _)| k) != Some(pre_aa_view) {
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PostProcess Bloom Extract BG"),
                layout: &self.bloom_compute_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.bloom_sampled_views[1]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.bloom_storage_views[0]),
                    },
                ],
            });
            self.bloom_extract_bg = Some((pre_aa_view.clone(), bg));
        }

        let compute_bg = self.compute_main_bg.as_ref().unwrap();
        let render_bg = self.render_main_bg.as_ref().unwrap();
        let extract_bg = &self.bloom_extract_bg.as_ref().unwrap().1;
        // Exposure and bloom read this frame's HDR input. GBuffer, HLFS and
        // TSR record to the graphics encoder, so these dependent dispatches
        // must follow them on that same encoder.
        let mut cmds = ctx.graphics_cmds();

        // 0. Volume blending runs in the post plugin's volume blend pass,
        //    scheduled ahead of this pass (it publishes POST_SETTINGS).

        // 1. Sampled log luminance: one sample per 4x4 block, then one final
        // workgroup reduction. Each partial has a unique writer. Only auto
        // exposure reads the result, so metering is skipped while no settings
        // source can select it (user effects may read it themselves). The
        // adapted value then goes stale, so the next metered frame snaps.
        let metering = !self.user_effect_entries.is_empty()
            || ctx
                .registry
                .get::<bool>(helio_core::ResourceKey::new(AUTO_EXPOSURE_MAYBE_ACTIVE))
                .unwrap_or(true);
        if !metering {
            self.exposure_history = false;
        }
        if let Some(query) = &self.compute_timing_query {
            cmds.write_timestamp(query, 0);
        }
        if metering {
            let mut cpass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PostProcess Exposure"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.exposure_pipeline);
            cpass.set_bind_group(0, compute_bg, &[]);
            let (gx, gy) = exposure_groups(self.analysis_size.0, self.analysis_size.1);
            cpass.dispatch_workgroups(gx, gy, 1);
        }
        if metering {
            let mut cpass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PostProcess Exposure Reduce"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.exposure_reduce_pipeline);
            cpass.set_bind_group(0, compute_bg, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }
        if metering {
            let mut cpass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PostProcess Exposure Adapt"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.exposure_adapt_pipeline);
            cpass.set_bind_group(0, compute_bg, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }
        if let Some(query) = &self.compute_timing_query {
            cmds.write_timestamp(query, 1);
        }

        // 1b. Local exposure's bilateral grid, from this frame's input and
        // exposure. The kernels return at once while the resolved settings
        // leave it off (fs_uber then skips it too).
        {
            let mut cpass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PostProcess Local Exposure"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, compute_bg, &[]);
            let pipelines = [
                &self.local_exposure_grid_pipeline,
                &self.local_exposure_blur_pipeline,
                &self.local_exposure_curve_pipeline,
            ];
            for (pipeline, [x, y, z]) in pipelines.into_iter().zip(self.local_exposure.workgroups())
            {
                cpass.set_pipeline(pipeline);
                cpass.dispatch_workgroups(x, y, z);
            }
        }
        if let Some(query) = &self.compute_timing_query {
            cmds.write_timestamp(query, 2);
        }

        // 2. Bloom, unless no settings source can enable it this frame: then
        // fs_uber never samples the mips, and they are rebuilt from scratch
        // on the first frame bloom may be on. User effects may sample the
        // mips themselves, so they always keep them.
        let bloom_maybe_active = !self.user_effect_entries.is_empty()
            || ctx
                .registry
                .get::<bool>(helio_core::ResourceKey::new(BLOOM_MAYBE_ACTIVE))
                .unwrap_or(true);
        if self.bloom_active && bloom_maybe_active {
            // 2a. Bloom extract: HDR → mip 0
            {
                let mut cpass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("PostProcess Bloom Extract"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.bloom_extract_pipeline);
                cpass.set_bind_group(0, compute_bg, &[]);
                cpass.set_bind_group(1, extract_bg, &[]);
                let (mw, mh) = self.mip_dims(0);
                cpass.dispatch_workgroups(
                    (mw + WG_BLOOM - 1) / WG_BLOOM,
                    (mh + WG_BLOOM - 1) / WG_BLOOM,
                    1,
                );
            }

            // 2b. Bloom downsample
            for i in 0..(BLOOM_MIPS as usize - 1) {
                let (mw, mh) = self.mip_dims(i as u32 + 1);
                let mut cpass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("PostProcess Bloom Down mip{}", i + 1)),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.bloom_down_pipeline);
                cpass.set_bind_group(0, compute_bg, &[]);
                cpass.set_bind_group(1, &self.bloom_down_bgs[i], &[]);
                cpass.dispatch_workgroups(
                    (mw + WG_BLOOM - 1) / WG_BLOOM,
                    (mh + WG_BLOOM - 1) / WG_BLOOM,
                    1,
                );
            }

            // 2c. Progressive upsample-and-add, coarse to fine, so the uber
            // pass samples one texture that holds every mip.
            for i in (0..self.bloom_sums.len()).rev() {
                let (mw, mh) = self.mip_dims(i as u32);
                let mut cpass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("PostProcess Bloom Upsample"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.bloom_upsample_pipeline);
                cpass.set_bind_group(0, compute_bg, &[]);
                cpass.set_bind_group(2, &self.bloom_upsample_bgs[i], &[]);
                cpass.dispatch_workgroups(mw.div_ceil(WG_BLOOM), mh.div_ceil(WG_BLOOM), 1);
            }
        }

        // 3. Uber render pass to the final target.
        if let Some(query) = &self.uber_timing_query {
            cmds.write_timestamp(query, 0);
        }
        if let Some(query) = &self.compute_timing_query {
            cmds.write_timestamp(query, 3);
        }
        {
            let mut pass = cmds.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("PostProcess Uber"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: ctx.target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.uber_pipelines[usize::from(self.debug_view)]);
            pass.set_bind_group(0, render_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        if let Some(query) = &self.uber_timing_query {
            cmds.write_timestamp(query, 1);
        }

        Ok(())
    }
}

/// One level of the bloom upsample chain, written by `cs_bloom_upsample`:
/// sum `i` is bloom mip `i` plus the upsampled sum below it.
struct BloomSum {
    // Kept alive for the views below.
    _texture: wgpu::Texture,
    sampled_view: wgpu::TextureView,
    storage_view: wgpu::TextureView,
}

impl BloomSum {
    /// Sums for mips 0..BLOOM_MIPS-1; the last mip is its own sum.
    fn chain(device: &wgpu::Device, width: u32, height: u32) -> Vec<Self> {
        (0..BLOOM_MIPS - 1)
            .map(|mip| Self::new(device, mip, width, height))
            .collect()
    }

    fn new(device: &wgpu::Device, mip: u32, width: u32, height: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("Bloom Sum {mip}")),
            size: wgpu::Extent3d {
                width: (width >> (mip + 1)).max(1),
                height: (height >> (mip + 1)).max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let sampled_view = texture.create_view(&Default::default());
        let storage_view = texture.create_view(&Default::default());
        Self {
            _texture: texture,
            sampled_view,
            storage_view,
        }
    }
}
