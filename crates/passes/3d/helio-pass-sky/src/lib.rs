//! Authored cloud-volume simulation and high-performance volumetric cloudscape.
//!
//! Implements the spec pipeline:
//! 1. Quarter-Resolution Ray Marching with 4x4 Bayer / blue-noise dithering
//! 2. Temporal Reprojection & Accumulation with Neighborhood Clamping / Variance Bounding (EMA 90/10)
//! 3. Depth-Aware Bilateral Upsample (cross-bilateral filter)
//! Plus: coarse-to-fine traversal, space leaping (100-300m), early termination
//! (alpha >=0.98), depth-buffer culling, height-gradient cloud types, Perlin-Worley +
//! Worley erosion with LOD, dual Henyey-Greenstein, Beer-Powder, multi-scattering
//! octaves, height-gradient ambient, and toggleable debug views.
//!
//! The two original WGSL modules are preserved verbatim for contract tests.
//! The new pipeline is additive and declares its own transient resources via
//! the frame graph.

use bytemuck::{Pod, Zeroable};
use helio_core::graph::{ResourceBuilder, ResourceFormat, ResourceSize};
use helio_core::{
    DebugViewDescriptor, PassContext, PrepareContext, RenderPass, Result as HelioResult,
};

pub const VOLUME_SIZE: wgpu::Extent3d = wgpu::Extent3d {
    width: 96,
    height: 48,
    depth_or_array_layers: 96,
};
pub const SIM_PARAMS_SIZE: u64 = 112;
pub const RENDER_PARAMS_SIZE: u64 = 272;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Exact layout of the authored simulation block (7 vec4 values / 112 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SimParams {
    pub values: [[f32; 4]; 7],
}

/// Exact layout of the authored render block (17 vec4 values / 272 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct RenderParams {
    pub values: [[f32; 4]; 17],
}

pub const SIM_SHADER: &str = include_str!("shaders/simulate.wgsl");
pub const RENDER_SHADER: &str = include_str!("shaders/render.wgsl");

// ── New High-Performance Pipeline Shaders ───────────────────────────────────
pub const CLOUD_RAYMARCH_SHADER: &str = include_str!("shaders/cloud_raymarch.wgsl");
pub const CLOUD_REPROJECT_SHADER: &str = include_str!("shaders/cloud_reproject.wgsl");
pub const CLOUD_UPSAMPLE_SHADER: &str = include_str!("shaders/cloud_upsample.wgsl");
pub const CLOUD_COMMON_SHADER: &str = include_str!("shaders/cloud_common.wgsl");
pub const CLOUD_VOLUME_LOWRES_SHADER: &str = include_str!("shaders/cloud_volume_lowres.wgsl");
pub const CLOUD_VOLUME_COMPOSITE_SHADER: &str = include_str!("shaders/cloud_volume_composite.wgsl");
pub const CLOUD_TEMPORAL_SHADER: &str = include_str!("shaders/cloud_temporal.wgsl");

// ── Atmospheric Sky LUT Shader (Hillaire 2020) ──────────────────────────────
pub const SKY_LUT_SHADER: &str = include_str!("shaders/sky_lut.wgsl");
pub const SKY_SHADER: &str = include_str!("shaders/sky.wgsl");

pub const LUT_WIDTH: u32 = 192;
pub const LUT_HEIGHT: u32 = 108;

/// Sky uniforms matching the WGSL shader layout (112 bytes, 16-byte aligned).
/// Must match the layout used in sky.wgsl and sky_lut.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ShaderSkyUniforms {
    pub sun_direction: [f32; 3],
    pub sun_intensity: f32,
    pub rayleigh_scatter: [f32; 3],
    pub rayleigh_h_scale: f32,
    pub mie_scatter: f32,
    pub mie_h_scale: f32,
    pub mie_g: f32,
    pub sun_disk_cos: f32,
    pub earth_radius: f32,
    pub atm_radius: f32,
    pub exposure: f32,
    pub clouds_enabled: u32,
    pub cloud_coverage: f32,
    pub cloud_density: f32,
    pub cloud_base: f32,
    pub cloud_top: f32,
    pub cloud_wind_x: f32,
    pub cloud_wind_z: f32,
    pub cloud_speed: f32,
    pub time_sky: f32,
    pub skylight_intensity: f32,
    pub cloud_mode: u32,
    pub cloud_quality: u32,
    pub cloud_resolution: u32,
}

impl ShaderSkyUniforms {
    pub fn earth_like() -> Self {
        let d = [0.0f32, 0.9, 0.4];
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        Self {
            sun_direction: [d[0] / len, d[1] / len, d[2] / len],
            sun_intensity: 22.0,
            rayleigh_scatter: [5.8e-3, 1.35e-2, 3.31e-2],
            rayleigh_h_scale: 0.1,
            mie_scatter: 2.1e-3,
            mie_h_scale: 0.075,
            mie_g: 0.76,
            sun_disk_cos: 0.9998,
            earth_radius: 6360.0,
            atm_radius: 6420.0,
            exposure: 0.1,
            clouds_enabled: 0,
            cloud_coverage: 0.0,
            cloud_density: 0.0,
            cloud_base: 0.0,
            cloud_top: 0.0,
            cloud_wind_x: 0.0,
            cloud_wind_z: 0.0,
            cloud_speed: 0.0,
            time_sky: 0.0,
            skylight_intensity: 0.0,
            cloud_mode: CloudRenderMode::Volume3D as u32,
            cloud_quality: CloudQuality::High as u32,
            cloud_resolution: CloudResolution::Quarter as u32,
        }
    }
}

/// Cloud representation selected by the renderer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum CloudRenderMode {
    Layer2D = 0,
    Volume3D = 1,
}

/// Bounded quality tier. Higher tiers spend more samples/detail in the volume.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum CloudQuality {
    Low = 0,
    Medium = 1,
    High = 2,
    Ultra = 3,
}

/// Cloud render resolution divisor: 1/full, 2/half, 4/quarter, 8/eighth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum CloudResolution {
    Full = 1,
    Half = 2,
    Quarter = 4,
    Eighth = 8,
}

// ── Debug Views ─────────────────────────────────────────────────────────────
const DEBUG_VIEWS: &[DebugViewDescriptor] = &[
    DebugViewDescriptor {
        name: "cloud_step_counts",
        debug_mode: 1,
        description: "Quarter-res ray march step counts (blue=low, red=high)",
    },
    DebugViewDescriptor {
        name: "cloud_reproject_confidence",
        debug_mode: 2,
        description: "Reprojection confidence / clamping masks (green=confident, red=clamped)",
    },
    DebugViewDescriptor {
        name: "cloud_raw_density",
        debug_mode: 3,
        description: "Raw density channels (R=density, G=half, B=inverse)",
    },
    DebugViewDescriptor {
        name: "cloud_bilateral_edge",
        debug_mode: 4,
        description: "Bilateral upsample edge mask",
    },
];

/// Toggleable debug mode for volumetric clouds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum CloudDebugMode {
    Off = 0,
    StepCounts = 1,
    ReprojectConfidence = 2,
    RawDensity = 3,
    BilateralEdge = 4,
}

impl From<u32> for CloudDebugMode {
    fn from(v: u32) -> Self {
        match v {
            1 => Self::StepCounts,
            2 => Self::ReprojectConfidence,
            3 => Self::RawDensity,
            4 => Self::BilateralEdge,
            _ => Self::Off,
        }
    }
}

// ── Frame Graph Intermediate Buffers ────────────────────────────────────────
/// Quarter-resolution divisor — 1/4 per axis = 1/16 total pixels.
/// Configurable to 1/2 per axis for higher quality on high-end GPUs.
pub const QUARTER_DIVISOR: u32 = 4;
pub const HALF_DIVISOR: u32 = 2;

// ── Cloud Pipeline Config ───────────────────────────────────────────────────
#[derive(Clone, Copy, Debug)]
pub struct CloudPipelineConfig {
    pub enabled: bool,
    /// Sub-sampling divisor per axis: 2 or 4. Default 4 (quarter-res).
    pub divisor: u32,
    /// Temporal blend: 0.90 history, 0.10 new (EMA). Default 0.90.
    pub temporal_blend: f32,
    /// Bilateral depth sigma. Default 1.0.
    pub bilateral_sigma: f32,
    /// Debug mode.
    pub debug: CloudDebugMode,
    /// When true, clouds use infinite horizontal extent (skydome mode) covering
    /// the entire horizon via tiled weather map and dome-shell ray intersection.
    pub infinite_extent: bool,
    pub mode: CloudRenderMode,
    pub quality: CloudQuality,
    pub resolution: CloudResolution,
}

impl Default for CloudPipelineConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            divisor: HALF_DIVISOR,
            temporal_blend: 0.90,
            bilateral_sigma: 1.0,
            debug: CloudDebugMode::Off,
            infinite_extent: false,
            mode: CloudRenderMode::Volume3D,
            quality: CloudQuality::High,
            resolution: CloudResolution::Half,
        }
    }
}

// ── Volumetric Cloud Uniform (extended) ─────────────────────────────────────
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CloudFrameUniform {
    pub resolution_time: [f32; 4],
    pub quarter_res_info: [f32; 4],
    pub reproj_info: [f32; 4], // history_valid, blend, frame_idx_mod, pad
    pub bilateral_info: [f32; 4],
    pub debug_mode: [u32; 4],
    pub infinite_flags: [u32; 4], // x = infinite_extent (1 enabled)
    pub _pad: [u32; 4],
}

// =============================================================================
// CloudVolumePass — legacy + new pipeline unified
// =============================================================================

pub struct SkyPass {
    // ── Legacy simulation / volume ──────────────────────────────────────────
    sim_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    sim_bgl: wgpu::BindGroupLayout,
    render_bgl: wgpu::BindGroupLayout,
    sim_params: wgpu::Buffer,
    render_params: wgpu::Buffer,
    sampler: wgpu::Sampler,
    volumes: [wgpu::Texture; 2],
    volume_views: [wgpu::TextureView; 2],
    sim_groups: [wgpu::BindGroup; 2],
    render_groups: [wgpu::BindGroup; 2],
    ping: usize,
    target_format: wgpu::TextureFormat,

    // ── New High-Performance Pipeline ───────────────────────────────────────
    config: CloudPipelineConfig,
    frame_uniform: wgpu::Buffer,
    raymarch_pipeline: wgpu::ComputePipeline,
    reproject_pipeline: wgpu::ComputePipeline,
    upsample_pipeline: wgpu::ComputePipeline,
    raymarch_bgl: wgpu::BindGroupLayout,
    reproject_bgl: wgpu::BindGroupLayout,
    upsample_bgl: wgpu::BindGroupLayout,
    linear_sampler: wgpu::Sampler,
    nearest_sampler: wgpu::Sampler,
    history_textures: [wgpu::Texture; 2],
    history_views: [wgpu::TextureView; 2],
    quarter_color_texture: wgpu::Texture,
    quarter_color_view: wgpu::TextureView,
    quarter_data_texture: wgpu::Texture,
    quarter_data_view: wgpu::TextureView,
    accum_texture: wgpu::Texture,
    accum_view: wgpu::TextureView,
    confidence_texture: wgpu::Texture,
    confidence_view: wgpu::TextureView,
    history_ping: usize,
    frame_idx: u64,
    history_valid: bool,
    width: u32,
    height: u32,
    allocated_divisor: u32,
    use_high_perf: bool,
    volume_lowres_pipeline: wgpu::ComputePipeline,
    volume_lowres_bgl: wgpu::BindGroupLayout,
    temporal_pipeline: wgpu::ComputePipeline,
    temporal_bgl: wgpu::BindGroupLayout,
    temporal_params: wgpu::Buffer,
    volume_composite_pipeline: wgpu::RenderPipeline,
    volume_composite_bgl: wgpu::BindGroupLayout,
    volume_composite_sampler: wgpu::Sampler,
    volume_composite_bg: Option<wgpu::BindGroup>,
    volume_composite_bg_key: Option<usize>,

    // ── Atmospheric Sky (Hillaire LUT + Skydome composite) ───────────────────
    sky_uniform_buf: wgpu::Buffer,
    sky_lut_pipeline: wgpu::RenderPipeline,
    sky_lut_bgl0: wgpu::BindGroupLayout,
    sky_lut_bgl1: wgpu::BindGroupLayout,
    sky_lut_bg0: wgpu::BindGroup,
    sky_lut_bg1: wgpu::BindGroup,
    sky_pipeline: wgpu::RenderPipeline,
    sky_bgl0: wgpu::BindGroupLayout,
    sky_bgl1: wgpu::BindGroupLayout,
    sky_bg0: wgpu::BindGroup,
    sky_bg1: Option<wgpu::BindGroup>,
    sky_bg1_key: Option<usize>,
    sky_lut_sampler: wgpu::Sampler,
    camera_buf: wgpu::Buffer,
}

/// Backwards alias — volumetric pass is now SkyPass.
pub type CloudVolumePass = SkyPass;

fn volume(device: &wgpu::Device, label: &str) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: VOLUME_SIZE,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D3),
        ..Default::default()
    });
    (texture, view)
}

fn texture_2d(
    device: &wgpu::Device,
    label: &str,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    (tex, view)
}

impl SkyPass {
    /// Creates the sky pass with a camera buffer (preferred for unified sky+clouds).
    pub fn new(
        device: &wgpu::Device,
        camera_buf: &wgpu::Buffer,
        target_format: wgpu::TextureFormat,
    ) -> Self {
        Self::new_with_camera_and_size(device, camera_buf, target_format, 1280, 720)
    }

    /// Legacy volumetric constructor (no external camera buffer) — creates an internal dummy camera.
    pub fn new_legacy(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let dummy_camera = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sky Dummy Camera"),
            size: 80,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self::new(device, &dummy_camera, target_format)
    }

    /// Backwards compat: CloudVolumePass::new redirected.
    pub fn new_cloud_volume_compat(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
    ) -> Self {
        Self::new_legacy(device, target_format)
    }

    pub fn new_with_size(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let dummy_camera = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sky Dummy Camera (sized)"),
            size: 80,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self::new_with_camera_and_size(device, &dummy_camera, target_format, width, height)
    }

    pub fn new_with_camera_and_size(
        device: &wgpu::Device,
        camera_buf: &wgpu::Buffer,
        target_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let sim_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cloud Volume Simulation"),
            source: wgpu::ShaderSource::Wgsl(SIM_SHADER.into()),
        });
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cloud Volume Raymarch"),
            source: wgpu::ShaderSource::Wgsl(RENDER_SHADER.into()),
        });
        let sim_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cloud Volume Simulation BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(SIM_PARAMS_SIZE),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
            ],
        });
        let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cloud Volume Render BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(RENDER_PARAMS_SIZE),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let sim_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cloud Simulation Params (112 bytes)"),
            size: SIM_PARAMS_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let render_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cloud Render Params (272 bytes)"),
            size: RENDER_PARAMS_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Cloud Volume Repeat Clamp Repeat Linear"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let (a, av) = volume(device, "Cloud Volume A");
        let (b, bv) = volume(device, "Cloud Volume B");
        let volumes = [a, b];
        let volume_views = [av, bv];
        // Dummy 1x1 textures for legacy render pipeline missing bindings (scene color, luna, moon2)
        let (dummy_tex, dummy_view) = texture_2d(
            device,
            "Dummy 1x1",
            1,
            1,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        // Keep texture alive by leaking into volumes? Store in dummy to prevent drop — use forget
        // We keep dummy_tex alive via hiding in a Box leak for pipeline lifetime (simple: forget)
        std::mem::forget(dummy_tex);
        let sim_groups = [0, 1].map(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Cloud Simulation Ping-Pong Bind Group"),
                layout: &sim_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: sim_params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&volume_views[i]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&volume_views[1 - i]),
                    },
                ],
            })
        });
        let render_groups = [0, 1].map(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Cloud Raymarch Bind Group"),
                layout: &render_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: render_params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&volume_views[i]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&dummy_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&dummy_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(&dummy_view),
                    },
                ],
            })
        });
        let sim_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Cloud Simulation Pipeline Layout"),
            bind_group_layouts: &[Some(&sim_bgl)],
            immediate_size: 0,
        });
        let sim_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Cloud Simulation 4x4x4"),
            layout: Some(&sim_layout),
            module: &sim_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let render_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Cloud Raymarch Pipeline Layout"),
            bind_group_layouts: &[Some(&render_bgl)],
            immediate_size: 0,
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Cloud Fullscreen Triangle Raymarch"),
            layout: Some(&render_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // ── Atmospheric Sky Pipelines (LUT + Composite) ───────────────────────
        let sky_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sky Uniforms (Unifed)"),
            size: std::mem::size_of::<ShaderSkyUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sky_lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Sky LUT Sampler (Unified)"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let sky_lut_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sky LUT Shader (Unified)"),
            source: wgpu::ShaderSource::Wgsl(SKY_LUT_SHADER.into()),
        });
        let sky_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sky Composite Shader (Unified)"),
            source: wgpu::ShaderSource::Wgsl(SKY_SHADER.into()),
        });
        // Sky LUT BGLs: camera storage + sky uniforms
        let sky_lut_bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sky LUT BGL0 (Unified)"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let sky_lut_bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sky LUT BGL1 (Unified)"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let sky_lut_bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sky LUT BG0 (Unified)"),
            layout: &sky_lut_bgl0,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            }],
        });
        let sky_lut_bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sky LUT BG1 (Unified)"),
            layout: &sky_lut_bgl1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: sky_uniform_buf.as_entire_binding(),
            }],
        });
        let sky_lut_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sky LUT PL (Unified)"),
            bind_group_layouts: &[Some(&sky_lut_bgl0), Some(&sky_lut_bgl1)],
            immediate_size: 0,
        });
        let sky_lut_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sky LUT Pipeline (Unified)"),
            layout: Some(&sky_lut_layout),
            vertex: wgpu::VertexState {
                module: &sky_lut_module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &sky_lut_module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
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
        });
        // Sky composite BGLs: camera storage + sky uniforms + LUT texture + sampler
        let sky_bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sky Composite BGL0 (Unified)"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let sky_bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sky Composite BGL1 (Unified)"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let sky_bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sky Composite BG0 (Unified)"),
            layout: &sky_bgl0,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            }],
        });
        let sky_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sky Composite PL (Unified)"),
            bind_group_layouts: &[Some(&sky_bgl0), Some(&sky_bgl1)],
            immediate_size: 0,
        });
        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sky Composite Pipeline (Unified)"),
            layout: Some(&sky_layout),
            vertex: wgpu::VertexState {
                module: &sky_module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &sky_module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
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
        });

        // ── High-Performance Pipeline Resources ───────────────────────────────
        let config = CloudPipelineConfig::default();
        let qw = (width / config.divisor).max(1);
        let qh = (height / config.divisor).max(1);
        let frame_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cloud Frame Uniform"),
            size: std::mem::size_of::<CloudFrameUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Cloud Linear Clamp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Cloud Nearest Clamp"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        // Intermediate buffers: Quarter-Res Target, History Buffer (ping-pong), Velocity/Depth handled via FrameResources
        let (quarter_color_texture, quarter_color_view) = texture_2d(
            device,
            "Cloud Quarter Color",
            qw,
            qh,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        );
        let (quarter_data_texture, quarter_data_view) = texture_2d(
            device,
            "Cloud Quarter Data (depth/step/density)",
            qw,
            qh,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        );
        let (accum_texture, accum_view) = texture_2d(
            device,
            "Cloud Quarter Accum (temporal)",
            qw,
            qh,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        );
        let (confidence_texture, confidence_view) = texture_2d(
            device,
            "Cloud Reproject Confidence",
            qw,
            qh,
            wgpu::TextureFormat::R32Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        );
        let (h0, hv0) = texture_2d(
            device,
            "Cloud History A",
            qw,
            qh,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
        );
        let (h1, hv1) = texture_2d(
            device,
            "Cloud History B",
            qw,
            qh,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
        );

        // Create pipelines for new stages (compute)
        let raymarch_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cloud Quarter Raymarch"),
            source: wgpu::ShaderSource::Wgsl(CLOUD_RAYMARCH_SHADER.into()),
        });
        let reproject_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cloud Temporal Reproject"),
            source: wgpu::ShaderSource::Wgsl(CLOUD_REPROJECT_SHADER.into()),
        });
        let upsample_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cloud Bilateral Upsample"),
            source: wgpu::ShaderSource::Wgsl(CLOUD_UPSAMPLE_SHADER.into()),
        });

        // Raymarch BGL: uniform, volume 3d, samplers, weather 2d, depth, noise 3d x2, storage out x2
        let raymarch_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cloud Raymarch BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let reproject_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cloud Reproject BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let upsample_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cloud Upsample BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let raymarch_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Cloud Raymarch PL"),
            bind_group_layouts: &[Some(&raymarch_bgl)],
            immediate_size: 0,
        });
        let raymarch_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Cloud Quarter Raymarch"),
            layout: Some(&raymarch_layout),
            module: &raymarch_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let reproject_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Cloud Reproject PL"),
            bind_group_layouts: &[Some(&reproject_bgl)],
            immediate_size: 0,
        });
        let reproject_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Cloud Temporal Reproject"),
            layout: Some(&reproject_layout),
            module: &reproject_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let upsample_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Cloud Upsample PL"),
            bind_group_layouts: &[Some(&upsample_bgl)],
            immediate_size: 0,
        });
        let upsample_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Cloud Bilateral Upsample"),
            layout: Some(&upsample_layout),
            module: &upsample_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let volume_lowres_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Finite Cloud Volume Low Resolution"),
            source: wgpu::ShaderSource::Wgsl(CLOUD_VOLUME_LOWRES_SHADER.into()),
        });
        let volume_lowres_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Finite Cloud Volume Low Resolution BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let volume_lowres_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Finite Cloud Volume Low Resolution PL"),
            bind_group_layouts: &[Some(&volume_lowres_bgl)],
            immediate_size: 0,
        });
        let volume_lowres_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Finite Cloud Volume Low Resolution"),
                layout: Some(&volume_lowres_layout),
                module: &volume_lowres_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let volume_composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Finite Cloud Volume Composite"),
            source: wgpu::ShaderSource::Wgsl(CLOUD_VOLUME_COMPOSITE_SHADER.into()),
        });
        let volume_composite_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Finite Cloud Volume Composite BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let volume_composite_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Finite Cloud Volume Composite PL"),
                bind_group_layouts: &[Some(&volume_composite_bgl)],
                immediate_size: 0,
            });
        let volume_composite_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Finite Cloud Volume Composite"),
                layout: Some(&volume_composite_layout),
                vertex: wgpu::VertexState {
                    module: &volume_composite_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &volume_composite_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                    // cloud_volume_lowres writes premultiplied radiance
                    // (rgb already contains the integrated alpha). Do not
                    // multiply it by alpha a second time during compositing.
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
        let volume_composite_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Finite Cloud Volume Composite Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let temporal_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Finite Cloud Temporal Accumulation"),
            source: wgpu::ShaderSource::Wgsl(CLOUD_TEMPORAL_SHADER.into()),
        });
        let temporal_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Finite Cloud Temporal Accumulation BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let temporal_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Finite Cloud Temporal Accumulation PL"),
            bind_group_layouts: &[Some(&temporal_bgl)],
            immediate_size: 0,
        });
        let temporal_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Finite Cloud Temporal Accumulation"),
            layout: Some(&temporal_layout),
            module: &temporal_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let temporal_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Finite Cloud Temporal Parameters"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            sim_pipeline,
            render_pipeline,
            sim_bgl,
            render_bgl,
            sim_params,
            render_params,
            sampler,
            volumes,
            volume_views,
            sim_groups,
            render_groups,
            ping: 0,
            target_format,
            config,
            frame_uniform,
            raymarch_pipeline,
            reproject_pipeline,
            upsample_pipeline,
            raymarch_bgl,
            reproject_bgl,
            upsample_bgl,
            volume_lowres_pipeline,
            volume_lowres_bgl,
            temporal_pipeline,
            temporal_bgl,
            temporal_params,
            volume_composite_pipeline,
            volume_composite_bgl,
            volume_composite_sampler,
            volume_composite_bg: None,
            volume_composite_bg_key: None,
            linear_sampler,
            nearest_sampler,
            history_textures: [h0, h1],
            history_views: [hv0, hv1],
            quarter_color_texture,
            quarter_color_view,
            quarter_data_texture,
            quarter_data_view,
            accum_texture,
            accum_view,
            confidence_texture,
            confidence_view,
            history_ping: 0,
            frame_idx: 0,
            history_valid: false,
            width,
            height,
            allocated_divisor: config.divisor,
            use_high_perf: true,
            sky_uniform_buf,
            sky_lut_pipeline,
            sky_lut_bgl0,
            sky_lut_bgl1,
            sky_lut_bg0,
            sky_lut_bg1,
            sky_pipeline,
            sky_bgl0,
            sky_bgl1,
            sky_bg0,
            sky_bg1: None,
            sky_bg1_key: None,
            sky_lut_sampler,
            camera_buf: camera_buf.clone(),
        }
    }

    pub fn write_sim_params(&self, queue: &wgpu::Queue, params: &SimParams) {
        queue.write_buffer(&self.sim_params, 0, bytemuck::bytes_of(params));
    }
    pub fn write_render_params(&self, queue: &wgpu::Queue, params: &RenderParams) {
        queue.write_buffer(&self.render_params, 0, bytemuck::bytes_of(params));
    }
    pub fn volume_view(&self) -> &wgpu::TextureView {
        &self.volume_views[self.ping]
    }
    pub fn dispatch(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Cloud Volume Simulation"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.sim_pipeline);
        pass.set_bind_group(0, &self.sim_groups[self.ping], &[]);
        pass.dispatch_workgroups(24, 12, 24);
        drop(pass);
        self.ping = 1 - self.ping;
    }
    pub fn render(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.render_pipeline);
        pass.set_bind_group(0, &self.render_groups[self.ping], &[]);
        pass.draw(0..3, 0..1);
    }

    // ── High-Performance Pipeline Controls ─────────────────────────────────
    pub fn set_debug_mode(&mut self, mode: CloudDebugMode) {
        self.config.debug = mode;
    }
    pub fn set_high_perf_enabled(&mut self, enabled: bool) {
        self.use_high_perf = enabled;
        if !enabled {
            self.history_valid = false;
        }
    }
    pub fn set_divisor(&mut self, divisor: u32) {
        self.config.divisor = divisor.clamp(1, 8);
        self.config.resolution = match self.config.divisor {
            2 => CloudResolution::Half,
            _ => CloudResolution::Quarter,
        };
    }
    pub fn set_cloud_mode(&mut self, mode: CloudRenderMode) {
        if self.config.mode != mode {
            self.config.mode = mode;
            self.history_valid = false;
        }
    }
    pub fn set_cloud_quality(&mut self, quality: CloudQuality) {
        if self.config.quality != quality {
            self.config.quality = quality;
            self.history_valid = false;
        }
    }
    pub fn set_cloud_resolution(&mut self, resolution: CloudResolution) {
        self.config.resolution = resolution;
        self.config.divisor = resolution as u32;
        self.history_valid = false;
    }
    pub fn reset_history(&mut self) {
        self.history_valid = false;
        self.frame_idx = 0;
    }

    /// Snapshot the complete runtime cloud configuration for UI/settings
    /// systems. The returned value is cheap to copy and can be persisted.
    pub fn cloud_config(&self) -> CloudPipelineConfig {
        self.config
    }

    pub fn history_view(&self) -> &wgpu::TextureView {
        &self.history_views[self.history_ping]
    }
    pub fn quarter_view(&self) -> &wgpu::TextureView {
        &self.quarter_color_view
    }
    pub fn accum_view(&self) -> &wgpu::TextureView {
        &self.accum_view
    }

    // ── Infinite Extent / Sky Controls ───────────────────────────────────────
    pub fn set_infinite_extent(&mut self, enabled: bool) {
        self.config.infinite_extent = enabled;
        self.history_valid = false;
    }
    pub fn with_infinite_extent(mut self, enabled: bool) -> Self {
        self.set_infinite_extent(enabled);
        self
    }
    pub fn infinite_extent(&self) -> bool {
        self.config.infinite_extent
    }
    pub fn set_clouds_enabled(&mut self, enabled: bool) {
        if self.config.enabled != enabled {
            self.config.enabled = enabled;
            self.history_valid = false;
        }
    }
    /// Fluent builder for divisor (2 or 4).
    pub fn with_divisor(mut self, divisor: u32) -> Self {
        self.set_divisor(divisor);
        self
    }
    /// Fluent builder for temporal blend (0.8..0.95).
    pub fn with_temporal_blend(mut self, blend: f32) -> Self {
        self.config.temporal_blend = blend.clamp(0.0, 0.99);
        self
    }
}

impl RenderPass for SkyPass {
    fn name(&self) -> &'static str {
        "Sky"
    }
    fn writes(&self) -> &'static [&'static str] {
        &["sky_lut", "pre_aa"]
    }

    fn publish<'a>(&'a self, _frame: &mut libhelio::FrameResources<'a>) {}

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        // Unified pass drives both sky_lut and pre_aa manually via encoder_ptr
        // to avoid encoder lock (graph would hold an active pre_aa pass while we
        // try to render the LUT). Returning None lets execute create both passes
        // sequentially on encoder_ptr without conflict.
        None
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        // Atmospheric LUT (192x108)
        builder.write_color_raw(
            "sky_lut",
            wgpu::TextureFormat::Rgba16Float,
            ResourceSize::Absolute {
                width: LUT_WIDTH,
                height: LUT_HEIGHT,
            },
        );
        // Main HDR target (pre_aa) — sky composite writes here
        builder.write_color_raw("pre_aa", self.target_format, ResourceSize::MatchSurface);
        // Quarter-resolution color target (offscreen raymarch)
        builder.write_color(
            "cloud_quarter_color",
            ResourceFormat::Rgba16Float,
            ResourceSize::ScaledInternal {
                divisor: self.config.divisor,
            },
        );
        // Quarter-resolution depth/data buffer
        builder.write_color(
            "cloud_quarter_data",
            ResourceFormat::Rgba16Float,
            ResourceSize::ScaledInternal {
                divisor: self.config.divisor,
            },
        );
        // History buffer (quarter-res, ping-ponged internally but declared for graph aliasing)
        builder.write_color(
            "cloud_history",
            ResourceFormat::Rgba16Float,
            ResourceSize::ScaledInternal {
                divisor: self.config.divisor,
            },
        );
        // Velocity is read from gbuffer_velocity (published by GBufferPass)
        builder.read("gbuffer_velocity");
        // Depth is accessed via ctx.depth / depth_texture from FrameResources
        builder.read("pre_aa"); // for final composite read
    }

    fn debug_views(&self) -> &'static [DebugViewDescriptor] {
        DEBUG_VIEWS
    }

    fn set_debug_mode(&mut self, mode: u32) {
        self.config.debug = CloudDebugMode::from(mode);
    }

    fn on_resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.width = width.max(1);
        self.height = height.max(1);
        let divisor = self.config.divisor.max(1);
        let qw = (self.width / divisor).max(1);
        let qh = (self.height / divisor).max(1);
        let mk = |label: &'static str, fmt: wgpu::TextureFormat| {
            texture_2d(
                device,
                label,
                qw,
                qh,
                fmt,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            )
        };
        let (qc_tex, qc_view) = mk("Cloud Quarter Color", wgpu::TextureFormat::Rgba16Float);
        let (qd_tex, qd_view) = mk("Cloud Quarter Data", wgpu::TextureFormat::Rgba16Float);
        let (ac_tex, ac_view) = mk("Cloud Quarter Accum", wgpu::TextureFormat::Rgba16Float);
        let (cf_tex, cf_view) = mk("Cloud Confidence", wgpu::TextureFormat::R32Float);
        let (h0_tex, h0_view) = mk("Cloud History A", wgpu::TextureFormat::Rgba16Float);
        let (h1_tex, h1_view) = mk("Cloud History B", wgpu::TextureFormat::Rgba16Float);
        self.quarter_color_texture = qc_tex;
        self.quarter_color_view = qc_view;
        self.quarter_data_texture = qd_tex;
        self.quarter_data_view = qd_view;
        self.accum_texture = ac_tex;
        self.accum_view = ac_view;
        self.confidence_texture = cf_tex;
        self.confidence_view = cf_view;
        self.history_textures = [h0_tex, h1_tex];
        self.history_views = [h0_view, h1_view];
        self.allocated_divisor = divisor;
        self.history_valid = false;
        self.frame_idx = 0;
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        self.frame_idx = ctx.frame_num;
        // Resolution is a runtime setting. Reallocate the owned temporal
        // targets lazily here so changing it between frames is safe and does
        // not invalidate bind groups while a frame is being encoded.
        if self.allocated_divisor != self.config.divisor {
            self.on_resize(ctx.device, self.width, self.height);
        }
        let qw = (self.width / self.config.divisor).max(1) as f32;
        let qh = (self.height / self.config.divisor).max(1) as f32;
        let w = self.width as f32;
        let h = self.height as f32;
        let uniform = CloudFrameUniform {
            resolution_time: [w, h, ctx.frame_num as f32 * 0.016, ctx.frame_num as f32],
            quarter_res_info: [qw, qh, 1.0 / qw.max(1.0), 1.0 / qh.max(1.0)],
            reproj_info: [
                if self.history_valid { 1.0 } else { 0.0 },
                self.config.temporal_blend,
                (ctx.frame_num % 16) as f32,
                0.0,
            ],
            bilateral_info: [self.config.bilateral_sigma, 0.5, 0.1, 0.0],
            debug_mode: [self.config.debug as u32, 0, 0, 0],
            infinite_flags: [if self.config.infinite_extent { 1 } else { 0 }, 0, 0, 0],
            _pad: [0; 4],
        };
        ctx.queue
            .write_buffer(&self.frame_uniform, 0, bytemuck::bytes_of(&uniform));
        let (cloud_base, cloud_top) = ctx
            .frame_resources
            .sky
            .clouds
            .map(|clouds| (clouds.base, clouds.top))
            .unwrap_or((32.0, 120.0));
        let temporal_values = [
            self.config.temporal_blend,
            if self.history_valid { 1.0 } else { 0.0 },
            cloud_base,
            cloud_top,
        ];
        ctx.queue.write_buffer(
            &self.temporal_params,
            0,
            bytemuck::cast_slice(&temporal_values),
        );

        // Upload sky uniforms (Nishita atmosphere + cloud overlay params)
        if ctx.frame_resources.sky.has_sky {
            let mut sky_uniforms = ShaderSkyUniforms::earth_like();
            if let Some(clouds) = ctx.frame_resources.sky.clouds {
                sky_uniforms.clouds_enabled = self.config.enabled as u32;
                sky_uniforms.cloud_coverage = clouds.coverage;
                sky_uniforms.cloud_density = clouds.density;
                sky_uniforms.cloud_base = clouds.base;
                sky_uniforms.cloud_top = clouds.top;
                sky_uniforms.cloud_wind_x = clouds.wind_x;
                sky_uniforms.cloud_wind_z = clouds.wind_z;
                sky_uniforms.cloud_speed = clouds.speed;
                sky_uniforms.skylight_intensity = clouds.skylight_intensity;
                // Propagate infinite extent flag from pipeline config to shader via coverage scaling is handled in raymarch
            }
            sky_uniforms.cloud_mode = self.config.mode as u32;
            sky_uniforms.cloud_quality = self.config.quality as u32;
            sky_uniforms.cloud_resolution = self.config.resolution as u32;
            // Reflect infiniteExtent flag by expanding cloud top/base when enabled — optional skydome coverage
            if self.config.infinite_extent && self.config.mode == CloudRenderMode::Layer2D {
                // widen vertical range slightly for full dome visibility
                sky_uniforms.cloud_base = sky_uniforms.cloud_base.min(800.0);
                sky_uniforms.cloud_top = sky_uniforms.cloud_top.max(2500.0);
            }
            sky_uniforms.time_sky = (ctx.frame_num as f32) * 0.03;
            ctx.write_buffer(&self.sky_uniform_buf, 0, bytemuck::bytes_of(&sky_uniforms));
        }
        Ok(())
    }

    fn chain_transparent(&self) -> bool {
        // New pipeline uses only compute encoder (plus optional legacy raster fallback)
        true
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // ── 1) Sky LUT generation (192x108) ─────────────────────────────────
        if ctx.resources.sky.has_sky {
            // Ensure LUT bind group is up to date (for generation)
            // LUT generation render pass — writes to graph-owned sky_lut texture if available.
            // We use encoder_ptr directly because this pass also owns the subsequent pre_aa pass.
            if let Some(sky_lut_view) = ctx.resources.sky_lut.get() {
                let encoder = unsafe { &mut *ctx.encoder_ptr };
                let attachments = [Some(wgpu::RenderPassColorAttachment {
                    view: sky_lut_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })];
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Sky LUT (Unified)"),
                    color_attachments: &attachments,
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.sky_lut_pipeline);
                pass.set_bind_group(0, &self.sky_lut_bg0, &[]);
                pass.set_bind_group(1, &self.sky_lut_bg1, &[]);
                pass.draw(0..3, 0..1);
            }
        }

        // Boundless procedural clouds are rendered by the sky shader.  Do not
        // also simulate/overlay the legacy 3D volume: that path is expensive
        // and its empty/default volume is the source of the gray/black veil
        // seen over the procedural demo.
        let procedural_clouds = ctx
            .resources
            .sky
            .clouds
            .map(|clouds| clouds.infinite_extent)
            .unwrap_or(false);

        // ── 2) Legacy cloud volume simulation dispatch ───────────────────────
        if self.config.enabled && !procedural_clouds && self.config.mode == CloudRenderMode::Layer2D
        {
            let ce = unsafe { &mut *ctx.compute_encoder_ptr };
            self.dispatch(ce);
        }

        if !self.use_high_perf {
            // Fallback: legacy fullscreen raymarch (writes to pre_aa when available)
            let target_view = ctx.resources.pre_aa.get().unwrap_or(ctx.target);
            if let Some(ptr) = ctx.active_render_pass_ptr() {
                unsafe { self.render(&mut *ptr) };
            } else {
                let attachments = [Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })];
                let mut pass = unsafe {
                    (&mut *ctx.encoder_ptr).begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Cloud Volume Raymarch (Legacy)"),
                        color_attachments: &attachments,
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    })
                };
                if ctx.resources.sky.has_sky {
                    // Also composite sky when in legacy mode
                    if let Some(sky_lut_view) = ctx.resources.sky_lut.get() {
                        let key = sky_lut_view as *const _ as usize;
                        if self.sky_bg1_key != Some(key) {
                            self.sky_bg1 =
                                Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                    label: Some("Sky Composite BG1 (Legacy)"),
                                    layout: &self.sky_bgl1,
                                    entries: &[
                                        wgpu::BindGroupEntry {
                                            binding: 0,
                                            resource: self.sky_uniform_buf.as_entire_binding(),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 1,
                                            resource: wgpu::BindingResource::TextureView(
                                                sky_lut_view,
                                            ),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 2,
                                            resource: wgpu::BindingResource::Sampler(
                                                &self.sky_lut_sampler,
                                            ),
                                        },
                                    ],
                                }));
                            self.sky_bg1_key = Some(key);
                        }
                    }
                    pass.set_pipeline(&self.sky_pipeline);
                    pass.set_bind_group(0, &self.sky_bg0, &[]);
                    if let Some(ref bg) = self.sky_bg1 {
                        pass.set_bind_group(1, bg, &[]);
                    }
                    pass.draw(0..3, 0..1);
                }
                self.render(&mut pass);
            }
            return Ok(());
        }

        // ── High-Performance Pipeline ─────────────────────────────────────────
        // Note: Full implementation would bind weather_map, depth, noise textures
        // from FrameResources / scene. For portability, we use fallback 1x1
        // textures when those resources are not available, ensuring the pipeline
        // never fails validation on minimal graphs.

        let volume_clouds_enabled = self.config.enabled
            && self.config.mode == CloudRenderMode::Volume3D
            && ctx.resources.sky.has_sky
            && ctx.resources.sky.clouds.is_some();
        // Godot's reference path deliberately keeps reduced-resolution cloud
        // frames free of temporal color history.  Reusing coarse texels and
        // then filtering them at presentation resolution is what creates the
        // long streaks and ghost silhouettes this pass used to exhibit.
        let use_cloud_temporal = volume_clouds_enabled && self.config.divisor == 1;
        if volume_clouds_enabled {
            let compute_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Finite Cloud Volume Low Resolution BG"),
                layout: &self.volume_lowres_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.camera_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.sky_uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.quarter_color_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&self.quarter_data_view),
                    },
                ],
            });
            let ce = unsafe { &mut *ctx.compute_encoder_ptr };
            let mut cpass = ce.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Finite Cloud Volume Low Resolution"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.volume_lowres_pipeline);
            cpass.set_bind_group(0, &compute_bg, &[]);
            let qw = (self.width / self.config.divisor).max(1);
            let qh = (self.height / self.config.divisor).max(1);
            cpass.dispatch_workgroups(qw.div_ceil(8), qh.div_ceil(8), 1);
            drop(cpass);
        }

        // History ping-pong
        let history_read = self.history_ping;
        let history_write = 1 - self.history_ping;

        if use_cloud_temporal {
            let temporal_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Finite Cloud Temporal Accumulation BG"),
                layout: &self.temporal_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.quarter_color_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            &self.history_views[history_read],
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.temporal_params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.camera_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(
                            &self.history_views[history_write],
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(&self.quarter_data_view),
                    },
                ],
            });
            let ce = unsafe { &mut *ctx.compute_encoder_ptr };
            let mut cpass = ce.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Finite Cloud Temporal Accumulation"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.temporal_pipeline);
            cpass.set_bind_group(0, &temporal_bg, &[]);
            let qw = (self.width / self.config.divisor).max(1);
            let qh = (self.height / self.config.divisor).max(1);
            cpass.dispatch_workgroups(qw.div_ceil(8), qh.div_ceil(8), 1);
            drop(cpass);
            self.history_ping = history_write;
            self.history_valid = true;
        }

        if volume_clouds_enabled {
            let cloud_source = if use_cloud_temporal {
                &self.history_views[history_write]
            } else {
                &self.quarter_color_view
            };
            let key = cloud_source as *const _ as usize;
            if self.volume_composite_bg_key != Some(key) {
                self.volume_composite_bg = Some(ctx.device.create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        label: Some("Finite Cloud Volume Composite BG"),
                        layout: &self.volume_composite_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(cloud_source),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(
                                    &self.volume_composite_sampler,
                                ),
                            },
                        ],
                    },
                ));
                self.volume_composite_bg_key = Some(key);
            }
        }

        // ── 3) Composite sky + clouds into pre_aa (active render pass) ─────────
        // Lazy bind group for sky composite (needs LUT view)
        if let Some(sky_lut_view) = ctx.resources.sky_lut.get() {
            let key = sky_lut_view as *const _ as usize;
            if self.sky_bg1_key != Some(key) {
                self.sky_bg1 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Sky Composite BG1 (Unified)"),
                    layout: &self.sky_bgl1,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.sky_uniform_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(sky_lut_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.sky_lut_sampler),
                        },
                    ],
                }));
                self.sky_bg1_key = Some(key);
            }
        }

        // Fallback composite: legacy render path ensures visible clouds even before
        // full graph wiring is complete (prevents blank sky during incremental rollout)
        if let Some(ptr) = ctx.active_render_pass_ptr() {
            let rp = unsafe { &mut *ptr };
            if ctx.resources.sky.has_sky {
                rp.set_pipeline(&self.sky_pipeline);
                rp.set_bind_group(0, &self.sky_bg0, &[]);
                if let Some(ref bg) = self.sky_bg1 {
                    rp.set_bind_group(1, bg, &[]);
                }
                rp.draw(0..3, 0..1);
            }
            if volume_clouds_enabled {
                rp.set_pipeline(&self.volume_composite_pipeline);
                rp.set_bind_group(0, self.volume_composite_bg.as_ref().unwrap(), &[]);
                rp.draw(0..3, 0..1);
            }
            // Localized cloud actors still use the legacy volume renderer.
            // Boundless procedural clouds are already part of the sky composite.
            if self.config.enabled
                && !procedural_clouds
                && self.config.mode == CloudRenderMode::Layer2D
            {
                self.render(rp);
            }
        } else {
            // No active pass — manual fallback (writes to pre_aa when graph allocates it)
            let target_view = ctx.resources.pre_aa.get().unwrap_or(ctx.target);
            let attachments = [Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })];
            let mut pass = unsafe {
                (&mut *ctx.encoder_ptr).begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Sky + Clouds Composite (Unified Fallback)"),
                    color_attachments: &attachments,
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
            };
            if ctx.resources.sky.has_sky {
                pass.set_pipeline(&self.sky_pipeline);
                pass.set_bind_group(0, &self.sky_bg0, &[]);
                if let Some(ref bg) = self.sky_bg1 {
                    pass.set_bind_group(1, bg, &[]);
                }
                pass.draw(0..3, 0..1);
            }
            if volume_clouds_enabled {
                pass.set_pipeline(&self.volume_composite_pipeline);
                pass.set_bind_group(0, self.volume_composite_bg.as_ref().unwrap(), &[]);
                pass.draw(0..3, 0..1);
            }
            if self.config.enabled
                && !procedural_clouds
                && self.config.mode == CloudRenderMode::Layer2D
            {
                self.render(&mut pass);
            }
        }

        Ok(())
    }
}
