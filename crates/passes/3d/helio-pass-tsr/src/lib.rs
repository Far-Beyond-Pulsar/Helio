//! Temporal Super-Resolution (TSR) pass.
//!
//! Temporal upscaling and anti-aliasing. Given the pre-AA HDR frame at
//! internal resolution, jittered by the camera, and its own history at output
//! resolution, it produces a stable, sharp image at display resolution.
//! `shaders/tsr_main.wgsl` documents the algorithm.
//!
//! ## Quality presets
//!
//! [`TsrQuality`] selects the recommended `render_scale`.  Select a preset via
//! [`RendererConfig::with_tsr_quality`] when building the render graph.
//!
//! ## Reactivity API
//!
//! [`TsrPass::reset_history`] discards the temporal history (call on camera cuts,
//! level loads, teleports).  [`TsrPass::set_reactivity`] drops a fraction of the
//! history's confidence every frame.
//!
//! ## Producer masks
//!
//! Like FSR 2's reactive and composition masks, passes can tell the resolve
//! about their shading: `transparency_reactivity` (coverage that must not
//! accumulate, [`TsrPass::with_transparency_reactivity`]) and
//! [`STOCHASTIC_MASK`] (shading sampled stochastically per frame that relies
//! on accumulation, [`TsrPass::with_stochastic_mask`]). [`MOTION_VECTORS`]
//! give each texel's motion, so moving objects reproject with themselves
//! ([`TsrPass::with_motion_vectors`]); without them the camera's motion is
//! used for every texel.
//!
//! ## O(1) guarantee
//!
//! `execute()` records exactly:
//! 1. With motion vectors, one buffer clear and one compute dispatch over
//!    the input texels (TSR reconstruct: where moving objects were last
//!    frame, for disocclusion).
//! 2. One compute dispatch over 8x8 tiles of output pixels (TSR resolve)
//!    into one of two ping-pong histories.
//! 3. One fullscreen draw (RCAS sharpening of that resolve) into
//!    `output_texture`, published as `tsr_color`.
//! 4. Optional fullscreen blit into `ctx.target`; intermediate mode leaves
//!    `tsr_color` to a following post-process pass.
//!
//! All are constant-time GPU operations regardless of scene complexity.

use bytemuck::{Pod, Zeroable};
use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

// ── Passthrough blit shader ───────────────────────────────────────────────────

/// Passthrough blit that copies `blit_tex` to the render target.
/// No sharpening — all processing happens in the resolve and RCAS.
const BLIT_WGSL: &str = "
@group(0) @binding(0) var blit_tex:     texture_2d<f32>;
@group(0) @binding(1) var blit_sampler: sampler;

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
}

@vertex fn vs_blit(@builtin(vertex_index) vi: u32) -> VertexOut {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    return VertexOut(vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0), vec2<f32>(x, y));
}

@fragment fn fs_blit(in: VertexOut) -> @location(0) vec4<f32> {
    return textureSampleLevel(blit_tex, blit_sampler, in.uv, 0.0);
}
";

#[cfg(test)]
mod camera_sampling_tests;
#[cfg(test)]
mod depth_history_tests;
#[cfg(test)]
mod temporal_quality_tests;

// ── Quality presets ───────────────────────────────────────────────────────────

/// Temporal Super-Resolution quality presets: the internal / display
/// resolution ratio. The resolve is the same for all of them; fewer input
/// samples per output pixel make it integrate proportionally more frames.
///
/// Set `RendererConfig::render_scale` to `preset.render_scale()` for the
/// recommended operating point.
///
/// ## Example
/// ```rust,ignore
/// let quality = TsrQuality::Quality;
/// let config  = RendererConfig::new(1920, 1080, surface_format)
///     .with_render_scale(quality.render_scale())
///     .with_tsr_quality(quality);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TsrQuality {
    /// 0.50× internal resolution: maximum performance uplift.
    Performance,

    /// 0.67× internal resolution: suitable for mid-range GPUs.
    Balanced,

    /// 0.75× internal resolution, matching UE5 TSR "Quality". **Default.**
    #[default]
    Quality,

    /// 1.0× internal resolution: no upscaling, TSR still runs for temporal AA.
    Native,
}

impl TsrQuality {
    /// Recommended `render_scale` (internal / display) for this preset.
    pub fn render_scale(self) -> f32 {
        match self {
            Self::Performance => 0.50,
            Self::Balanced => 0.67,
            Self::Quality => 0.75,
            Self::Native => 1.00,
        }
    }
}

// ── Inputs ────────────────────────────────────────────────────────────────────

/// Graph resource a lighting pass writes to say how much of a pixel's shading
/// it sampled stochastically this frame and relies on temporal accumulation
/// to integrate (FSR 2's reactive and transparency-and-composition masks are
/// the same kind of producer-written hint, for the opposite need).
///
/// Contract: a filterable single-channel texture at the internal (pre-AA)
/// resolution, read as `r` in `[0, 1]`: the fraction of the pixel's
/// luminance that comes from per-frame stochastic samples, for example a sun
/// shadow ray shared by a 2x2 block whose pixel rotates each frame, or sky
/// visibility sampled at one rotating pixel of a 4x4 block. `0` is
/// deterministic shading (the default when no pass writes it). The value
/// must not itself change with the sampling pattern. It is read only when
/// [`TsrPass::with_stochastic_mask`] is set.
///
/// Where it is set, history may lie anywhere within the range of this
/// frame's neighbourhood, not only within its deviation: a frame whose
/// samples all landed in shadow (or all in light) does not reject what the
/// other frames saw. Change detection is unaffected, so a light switching
/// still drops history.
pub const STOCHASTIC_MASK: &str = "tsr_stochastic";

/// Graph resource of per-texel motion vectors, which the geometry passes
/// write with the G-buffer (`gbuffer_velocity`).
///
/// Contract: an `Rgba16Float` texture at the internal (pre-AA) resolution.
/// `xy` is the texel's motion since last frame in pixels: its raster position
/// this frame minus the same surface point's last frame, both with that
/// frame's camera jitter (fragment position minus the point projected by the
/// previous, jittered view-projection and its previous transform or
/// animation). Bit 0 of `w` (as an integer) is set for geometry that moves
/// by itself: a moving or skinned instance, animated foliage. Elsewhere
/// (static geometry, passes that write none, the cleared target) the resolve
/// derives the camera's motion from depth, which is exact where a writer's
/// camera-relative matrices round (UE's velocity buffer holds only dynamic
/// geometry the same way). The rest of `w` and `z` are the writers' own.
/// Read only when [`TsrPass::with_motion_vectors`] is set.
pub const MOTION_VECTORS: &str = "gbuffer_velocity";

// ── GPU uniform ───────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TsrUniform {
    jitter_offset: [f32; 2], // this frame's jitter in input pixels
    reactivity: f32,         // fraction of history confidence dropped per frame
    flags: u32,              // bit 0: reset, bit 1: transparency coverage, bit 2: motion vectors
    previous_jitter_uv: [f32; 2],
    _pad: [f32; 2],
    reprojection: Reprojection,
}

/// How the resolve carries a point of this frame's clip space to last
/// frame's (`TsrUniform` in tsr_main.wgsl): clip to previous clip, composed
/// in f64 from this frame's view-projection, last frame's and last frame's
/// view (UE's ClipToPrevClip). Through world space in f32 instead, a near
/// point moves by rounding of the camera's position (hundreds of metres off
/// the frame origin), frame to frame, and history accumulating over dozens
/// of frames smears by it.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
struct Reprojection {
    clip_to_previous_clip: [f32; 16],
    clip_to_world_w: [f32; 4],
    eye_to_previous_clip: [f32; 4],
    clip_to_previous_view_z: [f32; 4],
}

impl Reprojection {
    /// From this frame's view-projection (`view_proj`) and eye, last frame's
    /// view-projection and view, in this frame's coordinates (all
    /// column-major).
    fn new(
        view_proj: [f32; 16],
        eye: [f32; 3],
        previous_view_proj: [f32; 16],
        previous_view: [f32; 16],
    ) -> Self {
        let matrix = |m: [f32; 16]| glam::DMat4::from_cols_array(&m.map(f64::from));
        let clip_to_world = matrix(view_proj).inverse();
        let previous_view_proj = matrix(previous_view_proj);
        let clip_to_previous_clip = previous_view_proj * clip_to_world;
        let clip_to_previous_view = matrix(previous_view) * clip_to_world;
        let eye = glam::DVec4::new(f64::from(eye[0]), f64::from(eye[1]), f64::from(eye[2]), 1.0);
        let f32s = |v: glam::DVec4| v.to_array().map(|x| x as f32);
        Self {
            clip_to_previous_clip: clip_to_previous_clip.to_cols_array().map(|x| x as f32),
            clip_to_world_w: f32s(clip_to_world.row(3)),
            eye_to_previous_clip: f32s(previous_view_proj * eye),
            clip_to_previous_view_z: f32s(clip_to_previous_view.row(2)),
        }
    }
}

// Express previous view-space depth in the current floating origin. Current
// local points are old local points minus origin_shift, so compose V * T(shift).
// Matrices use column-major GPU layout; only the translation column changes.
// Compose in f64, matching the renderer's previous-projection rebasing.
fn rebase_previous_view(previous: [f32; 16], origin_shift: [f64; 3]) -> [f32; 16] {
    assert!(
        origin_shift.iter().all(|v| v.is_finite()),
        "camera origin shift must be finite"
    );
    let mut rebased = previous;
    for row in 0..4 {
        rebased[12 + row] = (f64::from(previous[row]) * origin_shift[0]
            + f64::from(previous[4 + row]) * origin_shift[1]
            + f64::from(previous[8 + row]) * origin_shift[2]
            + f64::from(previous[12 + row])) as f32;
    }
    rebased
}

// ── History ───────────────────────────────────────────────────────────────────

/// Every history target: written as storage by the resolve, so a core
/// storage format (`Rg16Float` is not one).
const HISTORY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// `tsr_color`: the sharpened resolve, scene-linear HDR.
const OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// One frame's resolve, read back by the next: accumulated linear HDR colour
/// (alpha: transparency coverage), view depths for disocclusion (dilated and
/// the pixel's own), accumulation state (weight, thin-feature lock, the fine
/// change detector's fast mean and spread), the learnt deviations (the
/// input's about history, the coarse detector's) with the fine detector's
/// slow mean and the shading-change reference, and the coarse detector's
/// mean log luminances, which share a target with the depths.
/// `shaders/tsr_main.wgsl` documents each channel. Four targets at 32 bytes a
/// pixel. Half precision depth (0.05 %) is well inside the disocclusion
/// tolerance (0.4 %). The views keep the textures alive; the handles are for
/// tests to seed and read history.
#[cfg_attr(not(test), allow(dead_code))]
struct History {
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    depth_coarse: wgpu::Texture,
    depth_coarse_view: wgpu::TextureView,
    state: wgpu::Texture,
    state_view: wgpu::TextureView,
    noise: wgpu::Texture,
    noise_view: wgpu::TextureView,
}

impl History {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let make = |label| {
            let texture = output_texture(
                device,
                label,
                width,
                height,
                HISTORY_FORMAT,
                wgpu::TextureUsages::STORAGE_BINDING,
            );
            let view = texture.create_view(&Default::default());
            (texture, view)
        };
        let (color, color_view) = make("TSR History");
        let (depth_coarse, depth_coarse_view) = make("TSR History Depth and Coarse Luma");
        let (state, state_view) = make("TSR History State");
        let (noise, noise_view) = make("TSR History Noise");
        Self {
            color,
            color_view,
            depth_coarse,
            depth_coarse_view,
            state,
            state_view,
            noise,
            noise_view,
        }
    }
}

fn output_texture(
    device: &wgpu::Device,
    label: &'static str,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    written: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        // Copies: tests seed and read back history and output.
        usage: written
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

// ── Pass ──────────────────────────────────────────────────────────────────────

/// What the resolve and reconstruction bind groups were made for: colour,
/// depth, coverage, stochastic mask and motion vectors.
type ResolveKey = (
    wgpu::TextureView,
    wgpu::TextureView,
    wgpu::TextureView,
    wgpu::TextureView,
    wgpu::TextureView,
);

/// The reconstruction's buffer for `texels` input texels.
fn nearest_buffer(device: &wgpu::Device, texels: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("TSR Reconstructed Nearest Depth"),
        size: texels.max(1) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Temporal Super-Resolution pass.
///
/// Placed **in place of** `TaaPass` in the render graph when TSR is enabled.
/// Reads `"pre_aa"` from [`ResourceRegistry`](helio_core::ResourceRegistry) and
/// publishes the upsampled, temporally accumulated image as `tsr_color`.
/// By default it also blits to `ctx.target`.
pub struct TsrPass {
    /// A private second instance used only by the right eye in sequential XR.
    /// It owns independent history textures and camera/jitter tracking; the
    /// normal instance remains the mono and left-eye path.
    xr_right: Option<Box<TsrPass>>,
    active_xr_right: bool,
    xr_mode: bool,
    color_input: &'static str,
    intermediate_output: bool,
    timing_query: Option<wgpu::QuerySet>,
    // ── Resolve ───────────────────────────────────────────────────────────────
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    /// Per write index: the resolve writing `histories[i]` reads the other.
    bind_groups: [Option<wgpu::BindGroup>; 2],
    bind_group_key: Option<ResolveKey>,
    _reactivity_fallback: wgpu::Texture,
    reactivity_fallback_view: wgpu::TextureView,
    transparency_reactivity: bool,
    /// Read [`STOCHASTIC_MASK`] (zero when off).
    stochastic_mask: bool,
    /// Read [`MOTION_VECTORS`] (none when off: the camera's motion).
    motion_vectors: bool,
    /// The uniform's flags as `prepare` set them; `execute` adds bit 2 when
    /// this frame's graph published motion vectors.
    flags: u32,
    _motion_fallback: wgpu::Texture,
    motion_fallback_view: wgpu::TextureView,
    uniform_buf: wgpu::Buffer,

    // ── Reconstruction (last frame's nearest depth per input texel) ──────────
    reconstruct_pipeline: wgpu::ComputePipeline,
    reconstruct_bgl: wgpu::BindGroupLayout,
    reconstruct_bind_group: Option<wgpu::BindGroup>,
    /// One word per input texel, at least; grown with the internal size.
    nearest: wgpu::Buffer,

    // ── Sharpening (history → output_texture) ─────────────────────────────────
    sharpen_pipeline: wgpu::RenderPipeline,
    sharpen_bgl: wgpu::BindGroupLayout,
    sharpen_bind_groups: [wgpu::BindGroup; 2],

    // ── Blit pipeline (output_texture → ctx.target) ───────────────────────────
    blit_pipeline: wgpu::RenderPipeline,
    blit_bgl: wgpu::BindGroupLayout,
    blit_bind_group: wgpu::BindGroup,

    // ── Temporal history (ping-pong) ──────────────────────────────────────────
    histories: [History; 2],
    /// The history this frame's resolve writes; it reads the other one.
    write: usize,
    /// This frame's sharpened resolve at display resolution (`tsr_color`).
    pub output_texture: wgpu::Texture,
    pub output_view: wgpu::TextureView,

    // ── Samplers ──────────────────────────────────────────────────────────────
    linear_sampler: wgpu::Sampler,

    // ── Dimensions ────────────────────────────────────────────────────────────
    output_width: u32,
    output_height: u32,

    // ── Reactivity state ──────────────────────────────────────────────────────
    /// `true` until the first frame (or after `reset_history()`).
    first_frame: bool,
    previous_jitter_uv: [f32; 2],
    previous_view: [f32; 16],
    previous_world_origin: Option<[f64; 3]>,
    previous_view_id: Option<u32>,
    /// Fraction of history confidence dropped per frame (`0` = none, `1` = all).
    reactivity: f32,

    quality: TsrQuality,
    surface_format: wgpu::TextureFormat,
}

impl TsrPass {
    /// Select the scene-linear image reconstructed by this pass.
    pub fn with_color_input(mut self, key: &'static str) -> Self {
        self.color_input = key;
        if let Some(right) = &mut self.xr_right {
            right.color_input = key;
        }
        self
    }
    /// Publish the HDR resolve for post-processing without a redundant surface blit.
    pub fn with_intermediate_output(mut self) -> Self {
        self.intermediate_output = true;
        if let Some(right) = &mut self.xr_right {
            right.intermediate_output = true;
        }
        self
    }

    /// Opt in to R8 transparency coverage and retain it in history alpha.
    /// The resolved RGB is unchanged for zero current/previous coverage.
    pub fn with_transparency_reactivity(mut self) -> Self {
        self.transparency_reactivity = true;
        if let Some(right) = &mut self.xr_right {
            right.transparency_reactivity = true;
        }
        self
    }

    /// Create a new TSR pass.
    ///
    /// - `internal_*` — pre-AA (geometry) render resolution; the resolve reads
    ///   it from the bound textures, so it may change without a rebuild.
    /// - `output_*`   — display resolution; history/output textures live here.
    /// - `format`     — swapchain / surface format for the final blit.
    /// - `quality`    — the preset this pass was configured for.
    pub fn new(
        device: &wgpu::Device,
        _internal_width: u32,
        _internal_height: u32,
        output_width: u32,
        output_height: u32,
        format: wgpu::TextureFormat,
        quality: TsrQuality,
    ) -> Self {
        Self::new_inner(device, output_width, output_height, format, quality)
    }

    fn new_inner(
        device: &wgpu::Device,
        output_width: u32,
        output_height: u32,
        format: wgpu::TextureFormat,
        quality: TsrQuality,
    ) -> Self {
        let tsr_shader = helio_core::shader::module(
            device,
            "TSR Shader",
            helio_core::include_wgsl!("../shaders/tsr_main.wgsl"),
        );
        let blit_shader = helio_core::shader::module(device, "TSR Blit Shader", BLIT_WGSL);

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TSR Uniform"),
            size: std::mem::size_of::<TsrUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("TSR Linear Sampler"),
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let output_width = output_width.max(1);
        let output_height = output_height.max(1);
        let histories = [
            History::new(device, output_width, output_height),
            History::new(device, output_width, output_height),
        ];
        let output_texture = output_texture(
            device,
            "TSR Output",
            output_width,
            output_height,
            OUTPUT_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let output_view = output_texture.create_view(&Default::default());

        // Zero-initialized fallback keeps graphs without transparency unchanged.
        let reactivity_fallback = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("TSR zero reactivity"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let reactivity_fallback_view = reactivity_fallback.create_view(&Default::default());
        // Zero: no texel has a motion vector (w = 0), all move with the camera.
        let motion_fallback = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("TSR no motion vectors"),
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
        });
        let motion_fallback_view = motion_fallback.create_view(&Default::default());

        // ── TSR BGL ───────────────────────────────────────────────────────────
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TSR BGL"),
            entries: &[
                tex_entry(0, wgpu::TextureSampleType::Float { filterable: true }), // current_frame
                tex_entry(1, wgpu::TextureSampleType::Float { filterable: true }), // history_frame
                tex_entry(2, wgpu::TextureSampleType::Depth),                      // depth_tex
                sampler_entry(3, wgpu::SamplerBindingType::Filtering),             // linear_sampler
                camera_storage_entry(5),                                           // cameras
                uniform_entry(6),                                                  // tsr
                tex_entry(7, wgpu::TextureSampleType::Float { filterable: true }), // history_depth_coarse
                tex_entry(8, wgpu::TextureSampleType::Float { filterable: true }), // transparency_reactivity
                tex_entry(9, wgpu::TextureSampleType::Float { filterable: true }), // history_state
                tex_entry(11, wgpu::TextureSampleType::Float { filterable: true }), // history_noise
                tex_entry(13, wgpu::TextureSampleType::Float { filterable: true }), // stochastic_mask
                tex_entry(14, wgpu::TextureSampleType::Float { filterable: true }), // motion_vectors
                buffer_entry(15, true), // previous_nearest
                storage_entry(16),      // out_color
                storage_entry(17),      // out_depth_coarse
                storage_entry(18),      // out_state
                storage_entry(19),      // out_noise
            ],
        });
        let reconstruct_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TSR Reconstruct BGL"),
            entries: &[
                tex_entry(2, wgpu::TextureSampleType::Depth), // depth_tex
                uniform_entry(6),                             // tsr
                tex_entry(14, wgpu::TextureSampleType::Float { filterable: true }), // motion_vectors
                buffer_entry(20, false), // reconstructed_nearest
            ],
        });
        let sharpen_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TSR Sharpen BGL"),
            entries: &[tex_entry(
                10,
                wgpu::TextureSampleType::Float { filterable: true },
            )],
        });
        let sharpen_bind_groups = histories
            .each_ref()
            .map(|history| make_sharpen_bg(device, &sharpen_bgl, history));

        // ── Blit BGL ──────────────────────────────────────────────────────────
        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TSR Blit BGL"),
            entries: &[
                tex_entry(0, wgpu::TextureSampleType::Float { filterable: true }),
                sampler_entry(1, wgpu::SamplerBindingType::Filtering),
            ],
        });

        let blit_bind_group = make_blit_bg(device, &blit_bgl, &output_view, &linear_sampler);

        // ── Pipelines ─────────────────────────────────────────────────────────
        let fullscreen = |label: &'static str,
                          layout: &wgpu::BindGroupLayout,
                          module: &wgpu::ShaderModule,
                          vs: &'static str,
                          fs: &'static str,
                          targets: &[Option<wgpu::ColorTargetState>]| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
            helio_core::shader::create_render_pipeline(
                &device,
                wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module,
                        entry_point: Some(vs),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module,
                        entry_point: Some(fs),
                        compilation_options: Default::default(),
                        targets,
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
        };
        let target = |format| {
            Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })
        };
        let pipeline = helio_core::shader::create_compute_pipeline(
            &device,
            wgpu::ComputePipelineDescriptor {
                label: Some("TSR Resolve Pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("TSR Resolve Pipeline"),
                        bind_group_layouts: &[Some(&bgl)],
                        immediate_size: 0,
                    }),
                ),
                module: &tsr_shader,
                entry_point: Some("cs_main"),
                // Every invocation fills its share of the tile before the barrier;
                // naga would otherwise zero it from a single invocation first.
                compilation_options: wgpu::PipelineCompilationOptions {
                    zero_initialize_workgroup_memory: false,
                    ..Default::default()
                },
                cache: None,
            },
        );
        let reconstruct_pipeline = helio_core::shader::create_compute_pipeline(
            &device,
            wgpu::ComputePipelineDescriptor {
                label: Some("TSR Reconstruct Pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("TSR Reconstruct Pipeline"),
                        bind_group_layouts: &[Some(&reconstruct_bgl)],
                        immediate_size: 0,
                    }),
                ),
                module: &tsr_shader,
                entry_point: Some("cs_reconstruct"),
                compilation_options: Default::default(),
                cache: None,
            },
        );
        let sharpen_pipeline = fullscreen(
            "TSR Sharpen Pipeline",
            &sharpen_bgl,
            &tsr_shader,
            "vs_main",
            "fs_sharpen",
            &[target(OUTPUT_FORMAT)],
        );
        let blit_pipeline = fullscreen(
            "TSR Blit Pipeline",
            &blit_bgl,
            &blit_shader,
            "vs_blit",
            "fs_blit",
            &[target(format)],
        );

        let pass = Self {
            xr_right: None,
            active_xr_right: false,
            xr_mode: false,
            color_input: "pre_aa",
            intermediate_output: false,
            timing_query: None,
            pipeline,
            bgl,
            bind_groups: [None, None],
            bind_group_key: None,
            _reactivity_fallback: reactivity_fallback,
            reactivity_fallback_view,
            transparency_reactivity: false,
            stochastic_mask: false,
            motion_vectors: false,
            flags: 0,
            _motion_fallback: motion_fallback,
            motion_fallback_view,
            uniform_buf,
            reconstruct_pipeline,
            reconstruct_bgl,
            reconstruct_bind_group: None,
            nearest: nearest_buffer(device, u64::from(output_width) * u64::from(output_height)),
            sharpen_pipeline,
            sharpen_bgl,
            sharpen_bind_groups,
            blit_pipeline,
            blit_bgl,
            blit_bind_group,
            histories,
            write: 0,
            output_texture,
            output_view,
            linear_sampler,
            output_width,
            output_height,
            first_frame: true,
            previous_jitter_uv: [0.0; 2],
            previous_view: [0.0; 16],
            previous_world_origin: None,
            previous_view_id: None,
            reactivity: 0.0,
            quality,
            surface_format: format,
        };
        pass
    }

    // ── Public reactivity API ─────────────────────────────────────────────────

    /// Discard all temporal history.
    ///
    /// Call this on camera cuts, level loads, teleports, or any sudden scene
    /// change where stale history would cause visible ghosting.
    pub fn reset_history(&mut self) {
        self.first_frame = true;
        if let Some(right) = &mut self.xr_right {
            right.reset_history();
        }
    }

    /// Optional GPU timestamps: before the resolve, after it, after the
    /// sharpening draw and after the final blit.
    pub fn enable_timing(&mut self, device: &wgpu::Device) -> bool {
        if !device.features().contains(
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
        ) {
            return false;
        }
        self.timing_query = Some(device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("TSR timings"),
            ty: wgpu::QueryType::Timestamp,
            count: 4,
        }));
        if let Some(right) = &mut self.xr_right {
            right.enable_timing(device);
        }
        true
    }

    pub fn timing_query(&self) -> Option<&wgpu::QuerySet> {
        self.timing_query.as_ref()
    }

    /// Read [`STOCHASTIC_MASK`], which a graph's lighting passes write. Off,
    /// the resolve reads zero (all shading deterministic).
    pub fn with_stochastic_mask(mut self) -> Self {
        self.stochastic_mask = true;
        if let Some(right) = &mut self.xr_right {
            right.stochastic_mask = true;
        }
        self
    }

    /// Enable or disable reading [`STOCHASTIC_MASK`]. History keeps its meaning
    /// either way, so it is not reset.
    pub fn set_stochastic_mask(&mut self, enabled: bool) {
        if self.stochastic_mask != enabled {
            self.stochastic_mask = enabled;
            self.bind_group_key = None;
            if let Some(right) = &mut self.xr_right {
                right.set_stochastic_mask(enabled);
            }
        }
    }

    /// Read [`MOTION_VECTORS`], which a graph's geometry passes write. Off,
    /// every texel moves with the camera.
    pub fn with_motion_vectors(mut self) -> Self {
        self.motion_vectors = true;
        if let Some(right) = &mut self.xr_right {
            right.motion_vectors = true;
        }
        self
    }

    /// Enable coverage history for a graph providing `transparency_reactivity`.
    /// Changing modes resets history because its alpha channel changes meaning.
    pub fn set_transparency_reactivity(&mut self, enabled: bool) {
        if self.transparency_reactivity != enabled {
            self.transparency_reactivity = enabled;
            self.reset_history();
            self.bind_group_key = None;
            if let Some(right) = &mut self.xr_right {
                right.set_transparency_reactivity(enabled);
            }
        }
    }

    /// Set a per-frame reactivity bias.
    ///
    /// `factor` in `[0.0, 1.0]` is the fraction of the history's accumulated
    /// confidence dropped every frame:
    /// - `0.0` — full history (maximum stability). **Default.**
    /// - `1.0` — current frame only (maximum reactivity, every frame is fresh).
    ///
    /// Expose this via post-process volumes so different regions of the level
    /// can have different TSR responsiveness.
    pub fn set_reactivity(&mut self, factor: f32) {
        self.reactivity = factor.clamp(0.0, 1.0);
        if let Some(right) = &mut self.xr_right {
            right.set_reactivity(factor);
        }
    }

    /// The quality preset this pass was constructed with.
    pub fn quality(&self) -> TsrQuality {
        self.quality
    }
}

// ── BGL entry helpers ─────────────────────────────────────────────────────────

// The resolve is a compute shader; sharpening and blit are fragment shaders.
const VISIBILITY: wgpu::ShaderStages =
    wgpu::ShaderStages::COMPUTE.union(wgpu::ShaderStages::FRAGMENT);

fn tex_entry(binding: u32, sample_type: wgpu::TextureSampleType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: VISIBILITY,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32, ty: wgpu::SamplerBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: VISIBILITY,
        ty: wgpu::BindingType::Sampler(ty),
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

// The engine-wide camera buffer (`GpuCameraBuffer`, label "Camera Storage") is a
// storage buffer sized for 2 cameras (mono/stereo), matching every other pass's
// `var<storage, read> cameras: array<CameraUniforms, 2>`. Bindings that reference
// it must use this, not `uniform_entry`.
fn camera_storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn buffer_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: HISTORY_FORMAT,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn make_blit_bg(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("TSR Blit BG"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

fn make_sharpen_bg(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    history: &History,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("TSR Sharpen BG"),
        layout: bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 10,
            resource: wgpu::BindingResource::TextureView(&history.color_view),
        }],
    })
}

/// The resolve bind group reading `previous` and writing `next`.
#[allow(clippy::too_many_arguments)]
fn make_resolve_bg(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    previous: &History,
    next: &History,
    color: &wgpu::TextureView,
    depth: &wgpu::TextureView,
    reactive: &wgpu::TextureView,
    stochastic: &wgpu::TextureView,
    motion: &wgpu::TextureView,
    nearest: &wgpu::Buffer,
    sampler: &wgpu::Sampler,
    camera: &wgpu::Buffer,
    uniform: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let view = wgpu::BindingResource::TextureView;
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("TSR BG"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: view(color),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: view(&previous.color_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: view(depth),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: camera.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: view(&previous.depth_coarse_view),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: view(reactive),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: view(&previous.state_view),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: view(&previous.noise_view),
            },
            wgpu::BindGroupEntry {
                binding: 13,
                resource: view(stochastic),
            },
            wgpu::BindGroupEntry {
                binding: 14,
                resource: view(motion),
            },
            wgpu::BindGroupEntry {
                binding: 15,
                resource: nearest.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 16,
                resource: view(&next.color_view),
            },
            wgpu::BindGroupEntry {
                binding: 17,
                resource: view(&next.depth_coarse_view),
            },
            wgpu::BindGroupEntry {
                binding: 18,
                resource: view(&next.state_view),
            },
            wgpu::BindGroupEntry {
                binding: 19,
                resource: view(&next.noise_view),
            },
        ],
    })
}

/// The reconstruction bind group, writing `nearest`.
fn make_reconstruct_bg(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    depth: &wgpu::TextureView,
    motion: &wgpu::TextureView,
    nearest: &wgpu::Buffer,
    uniform: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let view = wgpu::BindingResource::TextureView;
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("TSR Reconstruct BG"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 2,
                resource: view(depth),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 14,
                resource: view(motion),
            },
            wgpu::BindGroupEntry {
                binding: 20,
                resource: nearest.as_entire_binding(),
            },
        ],
    })
}

/// Input texels per reconstruction workgroup (`cs_reconstruct`).
const RECONSTRUCT_GROUP: [u32; 2] = [8, 8];

/// Output pixels per resolve workgroup (`TILE` in tsr_main.wgsl).
const RESOLVE_TILE: [u32; 2] = [8, 8];

/// The resolve over `width` x `height` output pixels, in tiles.
fn dispatch_resolve(
    cmds: &mut helio_core::CommandRecorder<'_>,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    width: u32,
    height: u32,
) {
    let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("TSR Resolve"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(
        width.div_ceil(RESOLVE_TILE[0]),
        height.div_ceil(RESOLVE_TILE[1]),
        1,
    );
}

/// Fullscreen draw of `pipeline` into `targets` (all cleared first).
fn draw_fullscreen(
    cmds: &mut helio_core::CommandRecorder<'_>,
    label: &'static str,
    targets: &[&wgpu::TextureView],
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
) {
    let attachments: Vec<_> = targets
        .iter()
        .map(|view| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })
        })
        .collect();
    let mut pass = cmds.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &attachments,
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..3, 0..1);
}

// ── RenderPass impl ───────────────────────────────────────────────────────────

impl RenderPass for TsrPass {
    fn set_frame_inputs(&mut self, inputs: &helio_core::RenderFrameInputs<'_>) {
        self.xr_mode = inputs.xr_mode;
    }

    fn name(&self) -> &'static str {
        "TSR"
    }

    fn writes(&self) -> &'static [&'static str] {
        &["tsr_color"]
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        // The pass owns this view for the graph lifetime; frame registries
        // cannot outlive it. Resize replaces it before the next publication.
        let view: &'a wgpu::TextureView = unsafe {
            std::mem::transmute(if self.active_xr_right {
                &self
                    .xr_right
                    .as_ref()
                    .expect("right-eye TSR state")
                    .output_view
            } else {
                &self.output_view
            })
        };
        frame.write(helio_core::ResourceKey::new("tsr_color"), view, "TSR");
    }

    fn requires_camera_jitter(&self) -> bool {
        true
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
        // FogCompositePass may publish pre_aa's view as fogged_hdr (no fog
        // possible); reading pre_aa keeps the pool from reusing it before us.
        if self.color_input == "fogged_hdr" {
            builder.read("pre_aa");
        }
        if self.transparency_reactivity {
            builder.read("transparency_reactivity");
        }
        if self.stochastic_mask {
            builder.read(STOCHASTIC_MASK);
        }
        if self.motion_vectors {
            builder.read(MOTION_VECTORS);
        }
    }

    fn on_resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if let Some(right) = &mut self.xr_right {
            right.on_resize(device, width, height);
        }
        self.output_width = width.max(1);
        self.output_height = height.max(1);

        self.histories = [
            History::new(device, self.output_width, self.output_height),
            History::new(device, self.output_width, self.output_height),
        ];
        self.output_texture = output_texture(
            device,
            "TSR Output",
            self.output_width,
            self.output_height,
            OUTPUT_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        self.output_view = self.output_texture.create_view(&Default::default());
        self.sharpen_bind_groups = self
            .histories
            .each_ref()
            .map(|history| make_sharpen_bg(device, &self.sharpen_bgl, history));
        self.blit_bind_group = make_blit_bg(
            device,
            &self.blit_bgl,
            &self.output_view,
            &self.linear_sampler,
        );

        // The resolve bind groups reference the replaced histories.
        self.bind_groups = [None, None];
        self.bind_group_key = None;

        // History is stale after resize — prime the next frame.
        self.first_frame = true;
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        self.active_xr_right = self.xr_mode && ctx.camera_data.jitter_frame[3].to_bits() == 2;
        if self.active_xr_right {
            if self.xr_right.is_none() {
                let mut right = Self::new_inner(
                    ctx.device,
                    self.output_width,
                    self.output_height,
                    self.surface_format,
                    self.quality,
                );
                right.color_input = self.color_input;
                right.intermediate_output = self.intermediate_output;
                right.transparency_reactivity = self.transparency_reactivity;
                right.stochastic_mask = self.stochastic_mask;
                right.motion_vectors = self.motion_vectors;
                right.reactivity = self.reactivity;
                if self.timing_query.is_some() {
                    right.enable_timing(ctx.device);
                }
                self.xr_right = Some(Box::new(right));
            }
            if let Some(right) = &mut self.xr_right {
                return right.prepare(ctx);
            }
        }
        // The camera is the sample authority. Reconstructing a separate
        // sequence here breaks manual offsets and externally owned frame clocks.
        let ndc = ctx.camera_data.jitter_frame;
        let jitter = [
            ndc[0] * ctx.width as f32 * 0.5,
            ndc[1] * ctx.height as f32 * 0.5,
        ];
        let world_origin = ctx.world_origin.to_array();
        let view_id = ndc[3].to_bits();
        let view_changed = self
            .previous_view_id
            .is_some_and(|previous| previous != view_id);
        let reset = u32::from(self.first_frame || view_changed);
        self.first_frame = false;
        let previous_view = if reset != 0 {
            ctx.camera_data.view
        } else {
            match self.previous_world_origin {
                Some(previous) => rebase_previous_view(
                    self.previous_view,
                    [
                        world_origin[0] - previous[0],
                        world_origin[1] - previous[1],
                        world_origin[2] - previous[2],
                    ],
                ),
                None => self.previous_view,
            }
        };

        let u = TsrUniform {
            jitter_offset: jitter,
            reactivity: self.reactivity,
            flags: reset | (u32::from(self.transparency_reactivity) << 1),
            previous_jitter_uv: self.previous_jitter_uv,
            _pad: [0.0; 2],
            reprojection: Reprojection::new(
                ctx.camera_data.view_proj,
                [
                    ctx.camera_data.position_near[0],
                    ctx.camera_data.position_near[1],
                    ctx.camera_data.position_near[2],
                ],
                ctx.camera_data.prev_view_proj,
                previous_view,
            ),
        };

        ctx.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u));
        self.flags = u.flags;
        self.previous_jitter_uv = [ndc[0] * 0.5, -ndc[1] * 0.5];
        // Retain the raw matrix and its origin, not this frame's rebased
        // history matrix, so successive shifts are never applied twice.
        self.previous_view = ctx.camera_data.view;
        self.previous_world_origin = Some(world_origin);
        self.previous_view_id = Some(view_id);
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if self.active_xr_right {
            if let Some(right) = &mut self.xr_right {
                return right.execute(ctx);
            }
        }
        // ── 1. Lazy bind groups ────────────────────────────────────────────────
        let pre_aa_view: &wgpu::TextureView = ctx
            .registry
            .read(helio_core::ResourceKey::new(self.color_input), "TSR")
            .ok_or_else(|| {
                helio_core::Error::InvalidPassConfig(
                    "TsrPass requires frame.pre_aa (published by DeferredLightPass)".into(),
                )
            })?;

        let reactive_view = if self.transparency_reactivity {
            ctx.registry
                .read_texture_view(
                    helio_core::ResourceKey::new("transparency_reactivity"),
                    "TSR",
                )
                .ok_or_else(|| {
                    helio_core::Error::InvalidPassConfig(
                        "TSR coverage mode requires transparency_reactivity".into(),
                    )
                })?
        } else {
            &self.reactivity_fallback_view
        };
        // A frame no pass published the mask for (its writer has nothing
        // drawn yet) reads zero: deterministic shading.
        let stochastic_view = if self.stochastic_mask {
            ctx.registry
                .texture_view(helio_core::ResourceKey::new(STOCHASTIC_MASK))
                .unwrap_or(&self.reactivity_fallback_view)
        } else {
            &self.reactivity_fallback_view
        };
        let motion = self
            .motion_vectors
            .then(|| {
                ctx.registry
                    .read_texture_view(helio_core::ResourceKey::new(MOTION_VECTORS), "TSR")
            })
            .flatten();
        let motion_bound = motion.is_some();
        let motion_view = motion.unwrap_or(&self.motion_fallback_view);
        // Queue writes land before this frame's commands run.
        ctx.queue.write_buffer(
            &self.uniform_buf,
            std::mem::offset_of!(TsrUniform, flags) as u64,
            bytemuck::bytes_of(&(self.flags | (u32::from(motion_bound) << 2))),
        );
        // The reconstruction holds a word per input texel.
        let texels = u64::from(ctx.width) * u64::from(ctx.height);
        if self.nearest.size() < texels * 4 {
            self.nearest = nearest_buffer(ctx.device, texels);
            self.bind_group_key = None;
        }
        let key = (
            pre_aa_view.clone(),
            ctx.depth.clone(),
            reactive_view.clone(),
            stochastic_view.clone(),
            motion_view.clone(),
        );
        if self.bind_group_key.as_ref() != Some(&key) {
            self.bind_groups = [1, 0].map(|read| {
                Some(make_resolve_bg(
                    ctx.device,
                    &self.bgl,
                    &self.histories[read],
                    &self.histories[read ^ 1],
                    pre_aa_view,
                    ctx.depth,
                    reactive_view,
                    stochastic_view,
                    motion_view,
                    &self.nearest,
                    &self.linear_sampler,
                    ctx.camera,
                    &self.uniform_buf,
                ))
            });
            self.reconstruct_bind_group = Some(make_reconstruct_bg(
                ctx.device,
                &self.reconstruct_bgl,
                ctx.depth,
                motion_view,
                &self.nearest,
                &self.uniform_buf,
            ));
            self.bind_group_key = Some(key);
        }
        let write = self.write;
        // The next frame reads what this one writes.
        self.write ^= 1;

        let mut cmds = ctx.graphics_cmds();
        if let Some(query) = &self.timing_query {
            cmds.write_timestamp(query, 0);
        }

        // ── 2. Reconstruct where moving objects were ──────────────────────────
        if motion_bound {
            ctx.begin_gpu_scope(&mut cmds, "TSR: reconstruct");
            cmds.clear_buffer(&self.nearest, 0, Some(texels * 4));
            {
                let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("TSR Reconstruct"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.reconstruct_pipeline);
                pass.set_bind_group(0, self.reconstruct_bind_group.as_ref().unwrap(), &[]);
                pass.dispatch_workgroups(
                    ctx.width.div_ceil(RECONSTRUCT_GROUP[0]),
                    ctx.height.div_ceil(RECONSTRUCT_GROUP[1]),
                    1,
                );
            }
            ctx.end_gpu_scope(&mut cmds, "TSR: reconstruct");
        }

        // ── 3. Resolve → histories[write] ─────────────────────────────────────
        ctx.begin_gpu_scope(&mut cmds, "TSR: resolve");
        dispatch_resolve(
            &mut cmds,
            &self.pipeline,
            self.bind_groups[write].as_ref().unwrap(),
            self.output_width,
            self.output_height,
        );
        ctx.end_gpu_scope(&mut cmds, "TSR: resolve");
        if let Some(query) = &self.timing_query {
            cmds.write_timestamp(query, 1);
        }

        // ── 4. Sharpen → output_view (tsr_color) ──────────────────────────────
        ctx.begin_gpu_scope(&mut cmds, "TSR: sharpen");
        draw_fullscreen(
            &mut cmds,
            "TSR Sharpen",
            &[&self.output_view],
            &self.sharpen_pipeline,
            &self.sharpen_bind_groups[write],
        );
        ctx.end_gpu_scope(&mut cmds, "TSR: sharpen");
        if let Some(query) = &self.timing_query {
            cmds.write_timestamp(query, 2);
        }

        // ── 5. Blit output_view → ctx.target ──────────────────────────────────
        if !self.intermediate_output {
            ctx.begin_gpu_scope(&mut cmds, "TSR: blit");
            draw_fullscreen(
                &mut cmds,
                "TSR Blit",
                &[ctx.target],
                &self.blit_pipeline,
                &self.blit_bind_group,
            );
            ctx.end_gpu_scope(&mut cmds, "TSR: blit");
        }

        if let Some(query) = &self.timing_query {
            cmds.write_timestamp(query, 3);
        }

        Ok(())
    }
}
