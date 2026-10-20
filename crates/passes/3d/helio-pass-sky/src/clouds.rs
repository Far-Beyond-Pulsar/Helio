//! Volumetric clouds, part of the atmosphere: a layer of clouds in a band of
//! altitudes over the planet's surface (or flat ground), described by the
//! [`AtmosphereComponent`](crate::AtmosphereComponent) row's cloud fields
//! ([`CloudLayer`](crate::atmosphere::CloudLayer)). Their density is noise
//! (`shaders/clouds_common.wgsl`, after Schneider's Nubis): the weather
//! decides where clouds gather, billows shape them, wisps erode their edges,
//! and the wind carries all three. They are lit by the sun through the air
//! (the atmosphere's transmittance), with multiple scattering by octaves
//! (Wrenninge, as Hillaire 2016), and by the sky's light from the
//! atmosphere's sky table.
//!
//! [`AtmospherePass`](crate::AtmospherePass) builds the noise once and,
//! each frame, the clouds' shadow: an orthographic map along the sun around
//! the camera ([`CLOUD_SHADOW_MAP`], a Beer shadow map, and
//! [`CLOUD_SHADOW_FRAME`]). Passes lighting anything under the clouds opt
//! into [`CLOUD_SHADOW_SNIPPET`] (`//!use cloud_shadow`) and multiply the
//! sun by `cloud_shadow(p)`, as the volumetric fog's shafts do.
//!
//! [`CloudTracePass`] runs once the depth is complete. It writes the sun's
//! visibility at each pixel ([`SUN_VISIBILITY`]: the terrain's
//! `directional_visibility` times the clouds' shadow), which deferred
//! lighting multiplies into the sun, and marches the clouds seen from the
//! camera at half the lighting target's resolution, a quarter of its pixels
//! each frame, reconstructing the rest from reprojected history
//! ([`ATMOSPHERE_CLOUDS`], [`ATMOSPHERE_CLOUD_DEPTH`]). The atmosphere
//! composite puts them behind geometry, with the air in front of them.

use bytemuck::{Pod, Zeroable};
use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, PrepareContext, RenderPass, ResourceKey, Result as HelioResult};

use crate::atmosphere::{
    ATMOSPHERE_AERIAL, ATMOSPHERE_FRAME, ATMOSPHERE_MULTI_SCATTERING, ATMOSPHERE_SKY_TABLE,
    ATMOSPHERE_SKY_TABLE_SNIPPET, ATMOSPHERE_SNIPPET, ATMOSPHERE_TRANSMITTANCE,
};

/// Graph key of the clouds' shadow map (texture, [`CLOUD_SHADOW_FORMAT`]).
pub const CLOUD_SHADOW_MAP: &str = "cloud_shadow_map";
/// Graph key of the map's frame (uniform buffer, `CloudShadowFrame` in
/// `shaders/cloud_shadow.wgsl`).
pub const CLOUD_SHADOW_FRAME: &str = "cloud_shadow_frame";
pub const CLOUD_SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const CLOUD_SHADOW_FRAME_BYTES: u64 = 64;
/// Texels along each side of the shadow map.
pub const CLOUD_SHADOW_SIDE: u32 = 256;
/// Marker opting a shader into [`CLOUD_SHADOW_SNIPPET`].
pub const CLOUD_SHADOW_MARKER: &str = "//!use cloud_shadow";
/// `CloudShadowFrame` and `cloud_shadow(p)` (`shaders/cloud_shadow.wgsl`).
pub const CLOUD_SHADOW_SNIPPET: helio_core::shader::ShaderSnippet =
    helio_core::wgsl_snippet!(CLOUD_SHADOW_MARKER, "shaders/cloud_shadow.wgsl");
/// Marker opting a shader into [`CLOUDS_SNIPPET`].
pub const CLOUDS_MARKER: &str = "//!use atmosphere_clouds";
/// The clouds' density and phase (`shaders/clouds_common.wgsl`).
pub const CLOUDS_SNIPPET: helio_core::shader::ShaderSnippet =
    helio_core::wgsl_snippet!(CLOUDS_MARKER, "shaders/clouds_common.wgsl");

/// Graph keys of the clouds seen from the camera (half resolution,
/// `Rgba16Float`): colour (radiance, opacity: one minus the transmittance)
/// and depth (the clouds'
/// front and the footprint's geometry, km).
pub const ATMOSPHERE_CLOUDS: &str = "atmosphere_clouds";
pub const ATMOSPHERE_CLOUD_DEPTH: &str = "atmosphere_cloud_depth";
/// Graph key of the sun's visibility at each pixel (full resolution,
/// `Rgba16Float`: x the visibility, yzw the sun's direction it holds for),
/// as the voxel terrain's `directional_visibility`, which it includes.
pub const SUN_VISIBILITY: &str = "sun_visibility";
/// Graph key of the terrain's sun visibility a voxel pass publishes.
pub const DIRECTIONAL_VISIBILITY: &str = "directional_visibility";
/// Graph keys of the clouds' noise volumes.
pub const ATMOSPHERE_CLOUD_WEATHER: &str = "atmosphere_cloud_weather";
pub const ATMOSPHERE_CLOUD_SHAPE: &str = "atmosphere_cloud_shape";
pub const ATMOSPHERE_CLOUD_DETAIL: &str = "atmosphere_cloud_detail";

const WEATHER_SIDE: u32 = 64;
const SHAPE_SIDE: u32 = 128;
const DETAIL_SIDE: u32 = 32;
const NOISE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const CLOUD_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

fn entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None }
}

fn sampled(binding: u32, dimension: wgpu::TextureViewDimension) -> wgpu::BindGroupLayoutEntry {
    entry(binding, wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Float { filterable: true },
        view_dimension: dimension,
        multisampled: false,
    })
}

fn storage_texture(binding: u32, format: wgpu::TextureFormat, dimension: wgpu::TextureViewDimension) -> wgpu::BindGroupLayoutEntry {
    entry(binding, wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format, view_dimension: dimension })
}

fn buffer(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    entry(binding, wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None })
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    entry(binding, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering))
}

fn compute_pipeline(device: &wgpu::Device, module: &wgpu::ShaderModule, layout: &wgpu::BindGroupLayout, entry_point: &str) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(entry_point),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry_point),
        layout: Some(&pipeline_layout),
        module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn bind_group(device: &wgpu::Device, label: &str, layout: &wgpu::BindGroupLayout, resources: &[(u32, wgpu::BindingResource)]) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &resources
            .iter()
            .map(|(binding, resource)| wgpu::BindGroupEntry { binding: *binding, resource: resource.clone() })
            .collect::<Vec<_>>(),
    })
}

/// The sampler the noise is read with: linear, repeating, mipmapped.
fn noise_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Cloud noise"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    })
}

/// A dispatch of one kernel.
struct Dispatch {
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    groups: [u32; 3],
}

/// The clouds' noise and shadow, owned by the atmosphere pass.
pub(crate) struct CloudVolumes {
    weather: wgpu::TextureView,
    shape: wgpu::TextureView,
    detail: wgpu::TextureView,
    /// The noise's kernels, recorded once.
    build: Vec<Dispatch>,
    built: bool,
    shadow_map: wgpu::TextureView,
    shadow_frame: wgpu::Buffer,
    shadow: [Dispatch; 2],
}

impl CloudVolumes {
    pub(crate) fn new(device: &wgpu::Device, frame: &wgpu::Buffer) -> Self {
        let noise = helio_core::shader::module_with(
            device,
            "Cloud noise",
            helio_core::include_wgsl!("shaders/cloud_noise.wgsl"),
            &[],
        );
        let d3 = wgpu::TextureViewDimension::D3;
        let generate_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cloud noise"),
            entries: &[storage_texture(0, NOISE_FORMAT, d3)],
        });
        let downsample_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cloud noise mips"),
            entries: &[storage_texture(0, NOISE_FORMAT, d3), sampled(1, d3)],
        });
        let downsample = compute_pipeline(device, &noise, &downsample_layout, "downsample");
        let mut build = Vec::new();
        let mut volume = |label: &str, side: u32, mips: u32, kernel: &str| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: side, height: side, depth_or_array_layers: side },
                mip_level_count: mips,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: NOISE_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            let level = |mip: u32| {
                texture.create_view(&wgpu::TextureViewDescriptor { base_mip_level: mip, mip_level_count: Some(1), ..Default::default() })
            };
            let top = level(0);
            build.push(Dispatch {
                pipeline: compute_pipeline(device, &noise, &generate_layout, kernel),
                group: bind_group(device, label, &generate_layout, &[(0, wgpu::BindingResource::TextureView(&top))]),
                groups: [side.div_ceil(4); 3],
            });
            for mip in 1..mips {
                let (out, input) = (level(mip), level(mip - 1));
                build.push(Dispatch {
                    pipeline: downsample.clone(),
                    group: bind_group(device, label, &downsample_layout, &[
                        (0, wgpu::BindingResource::TextureView(&out)),
                        (1, wgpu::BindingResource::TextureView(&input)),
                    ]),
                    groups: [(side >> mip).max(1).div_ceil(4); 3],
                });
            }
            texture.create_view(&Default::default())
        };
        let weather = volume("Cloud weather", WEATHER_SIDE, 1, "weather_kernel");
        let shape = volume("Cloud shape", SHAPE_SIDE, SHAPE_SIDE.ilog2() + 1, "shape_kernel");
        let detail = volume("Cloud detail", DETAIL_SIDE, DETAIL_SIDE.ilog2() + 1, "detail_kernel");

        let shadow_module = helio_core::shader::module_with(
            device,
            "Cloud shadow map",
            helio_core::include_wgsl!("shaders/cloud_shadow_map.wgsl"),
            &[ATMOSPHERE_SNIPPET, CLOUDS_SNIPPET, CLOUD_SHADOW_SNIPPET],
        );
        let shadow_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cloud shadow map"),
            entries: &[
                buffer(0, wgpu::BufferBindingType::Storage { read_only: true }),
                sampled(1, d3),
                sampled(2, d3),
                sampled(3, d3),
                sampler_entry(4),
                storage_texture(5, CLOUD_SHADOW_FORMAT, wgpu::TextureViewDimension::D2),
                buffer(6, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let shadow_map = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("Cloud shadow map"),
                size: wgpu::Extent3d { width: CLOUD_SHADOW_SIDE, height: CLOUD_SHADOW_SIDE, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: CLOUD_SHADOW_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let shadow_frame = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cloud shadow frame"),
            size: CLOUD_SHADOW_FRAME_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = noise_sampler(device);
        let view = wgpu::BindingResource::TextureView;
        let shadow_group = bind_group(device, "Cloud shadow map", &shadow_layout, &[
            (0, frame.as_entire_binding()),
            (1, view(&weather)),
            (2, view(&shape)),
            (3, view(&detail)),
            (4, wgpu::BindingResource::Sampler(&sampler)),
            (5, view(&shadow_map)),
            (6, shadow_frame.as_entire_binding()),
        ]);
        let shadow = [
            Dispatch {
                pipeline: compute_pipeline(device, &shadow_module, &shadow_layout, "cloud_shadow_resolve"),
                group: shadow_group.clone(),
                groups: [1, 1, 1],
            },
            Dispatch {
                pipeline: compute_pipeline(device, &shadow_module, &shadow_layout, "cloud_shadow_kernel"),
                group: shadow_group,
                groups: [CLOUD_SHADOW_SIDE.div_ceil(8), CLOUD_SHADOW_SIDE.div_ceil(8), 1],
            },
        ];
        Self { weather, shape, detail, build, built: false, shadow_map, shadow_frame, shadow }
    }

    /// The map's frame, zeroed while no atmosphere is active.
    pub(crate) fn shadow_frame(&self) -> &wgpu::Buffer {
        &self.shadow_frame
    }

    /// Builds the noise (the first time) and this frame's shadow map, after
    /// the atmosphere's frame is resolved.
    pub(crate) fn record(&mut self, pass: &mut helio_core::cmd::ComputeCmds<'_>) {
        if !self.built {
            for dispatch in &self.build {
                pass.set_pipeline(&dispatch.pipeline);
                pass.set_bind_group(0, Some(&dispatch.group), &[]);
                pass.dispatch_workgroups(dispatch.groups[0], dispatch.groups[1], dispatch.groups[2]);
            }
            self.built = true;
        }
        for dispatch in &self.shadow {
            pass.set_pipeline(&dispatch.pipeline);
            pass.set_bind_group(0, Some(&dispatch.group), &[]);
            pass.dispatch_workgroups(dispatch.groups[0], dispatch.groups[1], dispatch.groups[2]);
        }
    }

    pub(crate) fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        frame.write_texture_binding(CLOUD_SHADOW_MAP, &self.shadow_map, "Atmosphere");
        frame.write_buffer_binding(CLOUD_SHADOW_FRAME, &self.shadow_frame, "Atmosphere");
        frame.write_texture_binding(ATMOSPHERE_CLOUD_WEATHER, &self.weather, "Atmosphere");
        frame.write_texture_binding(ATMOSPHERE_CLOUD_SHAPE, &self.shape, "Atmosphere");
        frame.write_texture_binding(ATMOSPHERE_CLOUD_DETAIL, &self.detail, "Atmosphere");
    }
}

/// Placeholder resources for a consumer of the clouds' shadow while nothing
/// publishes it: a 1x1 map and a zeroed frame (no clouds).
pub fn cloud_shadow_fallback(device: &wgpu::Device) -> (wgpu::TextureView, wgpu::Buffer) {
    let map = device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("No cloud shadow"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: CLOUD_SHADOW_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default());
    let frame = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("No cloud shadow frame"),
        size: CLOUD_SHADOW_FRAME_BYTES,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });
    (map, frame)
}

/// `TraceParams` (`cloud_trace.wgsl`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TraceParams {
    full: [u32; 2],
    half: [u32; 2],
    quarter: [u32; 2],
    sub: [u32; 2],
    frame: u32,
    history: u32,
    _pad: [u32; 2],
}

/// The traced pixel of each 2x2 block, frame by frame.
const SUB_PIXELS: [[u32; 2]; 4] = [[0, 0], [1, 1], [1, 0], [0, 1]];

struct Targets {
    full: [u32; 2],
    half: [u32; 2],
    quarter: [u32; 2],
    trace_color: wgpu::TextureView,
    trace_depth: wgpu::TextureView,
    color: [wgpu::TextureView; 2],
    depth: [wgpu::TextureView; 2],
    sun_visibility: wgpu::TextureView,
}

impl Targets {
    fn new(device: &wgpu::Device, full: [u32; 2]) -> Self {
        let half = full.map(|v| v.div_ceil(2).max(1));
        let quarter = half.map(|v| v.div_ceil(2).max(1));
        let target = |label: &str, size: [u32; 2]| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: CLOUD_FORMAT,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        Self {
            full,
            half,
            quarter,
            trace_color: target("Cloud trace", quarter),
            trace_depth: target("Cloud trace depth", quarter),
            color: [target("Clouds A", half), target("Clouds B", half)],
            depth: [target("Cloud depth A", half), target("Cloud depth B", half)],
            sun_visibility: target("Sun visibility", full),
        }
    }
}

/// The clouds seen from the camera (see the module docs): marched at half
/// resolution, a quarter of the pixels each frame, the rest reconstructed.
/// Records nothing without an atmosphere; with one but no clouds, it
/// publishes clear (transmittance 1) targets.
pub struct CloudTracePass {
    trace_layout: wgpu::BindGroupLayout,
    reconstruct_layout: wgpu::BindGroupLayout,
    visibility_layout: wgpu::BindGroupLayout,
    trace: wgpu::ComputePipeline,
    reconstruct: wgpu::ComputePipeline,
    visibility: wgpu::ComputePipeline,
    visibility_group: Option<(Vec<wgpu::TextureView>, Vec<wgpu::Buffer>, wgpu::BindGroup)>,
    /// A 1x1 zero texture: no terrain visibility.
    no_terrain: wgpu::TextureView,
    params: wgpu::Buffer,
    noise_sampler: wgpu::Sampler,
    lut_sampler: wgpu::Sampler,
    targets: Option<Targets>,
    /// The bind groups and the resources they were built from.
    trace_group: Option<(Vec<wgpu::TextureView>, Vec<wgpu::Buffer>, wgpu::BindGroup)>,
    reconstruct_groups: Option<(wgpu::TextureView, [wgpu::BindGroup; 2])>,
    frame: u64,
    /// Whether the last output is a history for this frame.
    history: bool,
    active: bool,
}

impl CloudTracePass {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = helio_core::shader::module_with(
            device,
            "Cloud trace",
            helio_core::include_wgsl!("shaders/cloud_trace.wgsl"),
            &[ATMOSPHERE_SNIPPET, CLOUDS_SNIPPET, ATMOSPHERE_SKY_TABLE_SNIPPET, CLOUD_SHADOW_SNIPPET],
        );
        let d2 = wgpu::TextureViewDimension::D2;
        let d3 = wgpu::TextureViewDimension::D3;
        let read_only = wgpu::BufferBindingType::Storage { read_only: true };
        let depth = entry(6, wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: d2,
            multisampled: false,
        });
        let trace_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cloud trace"),
            entries: &[
                buffer(0, read_only),
                buffer(1, read_only),
                buffer(2, read_only),
                sampled(3, d2),
                sampled(4, d2),
                sampled(5, d3),
                depth,
                sampled(7, d3),
                sampled(8, d3),
                sampled(9, d3),
                sampler_entry(10),
                sampler_entry(11),
                buffer(12, wgpu::BufferBindingType::Uniform),
                storage_texture(13, CLOUD_FORMAT, d2),
                storage_texture(14, CLOUD_FORMAT, d2),
            ],
        });
        let reconstruct_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cloud reconstruct"),
            entries: &[
                buffer(0, read_only),
                depth,
                sampler_entry(11),
                buffer(12, wgpu::BufferBindingType::Uniform),
                sampled(15, d2),
                sampled(16, d2),
                sampled(17, d2),
                sampled(18, d2),
                storage_texture(19, CLOUD_FORMAT, d2),
                storage_texture(20, CLOUD_FORMAT, d2),
            ],
        });
        let visibility_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sun visibility"),
            entries: &[
                buffer(0, read_only),
                buffer(1, read_only),
                depth,
                buffer(12, wgpu::BufferBindingType::Uniform),
                buffer(21, wgpu::BufferBindingType::Uniform),
                sampled(22, d2),
                sampled(23, d2),
                storage_texture(24, CLOUD_FORMAT, d2),
            ],
        });
        let visibility = compute_pipeline(device, &module, &visibility_layout, "sun_visibility");
        let no_terrain = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("No terrain sun visibility"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: CLOUD_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let trace = compute_pipeline(device, &module, &trace_layout, "trace");
        let reconstruct = compute_pipeline(device, &module, &reconstruct_layout, "reconstruct");
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cloud trace params"),
            size: std::mem::size_of::<TraceParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Cloud trace LUT"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            trace_layout,
            reconstruct_layout,
            visibility_layout,
            trace,
            reconstruct,
            visibility,
            visibility_group: None,
            no_terrain,
            params,
            noise_sampler: noise_sampler(device),
            lut_sampler,
            targets: None,
            trace_group: None,
            reconstruct_groups: None,
            frame: 0,
            history: false,
            active: false,
        }
    }

    /// Forget the clouds' history: the next frame reconstructs from its
    /// traces alone.
    pub fn reset_history(&mut self) {
        self.history = false;
    }

    fn current(&self) -> usize {
        (self.frame & 1) as usize
    }
}

impl RenderPass for CloudTracePass {
    fn name(&self) -> &'static str {
        "CloudTrace"
    }

    fn reads(&self) -> &'static [&'static str] {
        &[
            "depth",
            ATMOSPHERE_FRAME,
            ATMOSPHERE_SKY_TABLE,
            ATMOSPHERE_TRANSMITTANCE,
            ATMOSPHERE_MULTI_SCATTERING,
            ATMOSPHERE_AERIAL,
            ATMOSPHERE_CLOUD_WEATHER,
            ATMOSPHERE_CLOUD_SHAPE,
            ATMOSPHERE_CLOUD_DETAIL,
            CLOUD_SHADOW_MAP,
            CLOUD_SHADOW_FRAME,
            DIRECTIONAL_VISIBILITY,
        ]
    }

    fn writes(&self) -> &'static [&'static str] {
        &[ATMOSPHERE_CLOUDS, ATMOSPHERE_CLOUD_DEPTH, SUN_VISIBILITY]
    }

    fn declare_resources(&self, _builder: &mut ResourceBuilder) {}

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a helio_core::ResourceRegistry<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let full = [ctx.width.max(1), ctx.height.max(1)];
        if self.targets.as_ref().is_none_or(|t| t.full != full) {
            self.targets = Some(Targets::new(ctx.device, full));
            self.trace_group = None;
            self.reconstruct_groups = None;
            self.visibility_group = None;
            self.history = false;
        }
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let registry = ctx.registry;
        self.active = registry.get::<bool>(ResourceKey::new("atmosphere_active")).unwrap_or(false);
        let resources = (
            registry.get::<&wgpu::Buffer>(ResourceKey::new(ATMOSPHERE_FRAME)),
            registry.get::<&wgpu::Buffer>(ResourceKey::new(ATMOSPHERE_SKY_TABLE)),
            registry.texture_view(ResourceKey::new(ATMOSPHERE_TRANSMITTANCE)),
            registry.texture_view(ResourceKey::new(ATMOSPHERE_MULTI_SCATTERING)),
            registry.texture_view(ResourceKey::new(ATMOSPHERE_AERIAL)),
            registry.texture_binding(ATMOSPHERE_CLOUD_WEATHER),
            registry.texture_binding(ATMOSPHERE_CLOUD_SHAPE),
            registry.texture_binding(ATMOSPHERE_CLOUD_DETAIL),
            registry.texture_binding(CLOUD_SHADOW_MAP),
            registry.buffer_binding(CLOUD_SHADOW_FRAME),
        );
        let (
            Some(frame),
            Some(sky_table),
            Some(transmittance),
            Some(multi),
            Some(aerial),
            Some(weather),
            Some(shape),
            Some(detail),
            Some(shadow_map),
            Some(shadow_frame),
        ) = resources
        else {
            self.active = false;
            return Ok(());
        };
        if !self.active {
            self.history = false;
            return Ok(());
        }
        let Some(targets) = &self.targets else { return Ok(()) };
        self.frame += 1;
        let current = self.current();
        let params = TraceParams {
            full: targets.full,
            half: targets.half,
            quarter: targets.quarter,
            sub: SUB_PIXELS[(self.frame % 4) as usize],
            frame: self.frame as u32,
            history: u32::from(self.history),
            _pad: [0; 2],
        };
        helio_core::upload::write_buffer(ctx.queue, &self.params, 0, bytemuck::bytes_of(&params));

        let views = vec![ctx.depth.clone(), transmittance.clone(), multi.clone(), aerial.clone(), weather.clone(), shape.clone(), detail.clone()];
        let buffers = vec![ctx.camera.clone(), frame.clone(), sky_table.clone()];
        if self.trace_group.as_ref().is_none_or(|(v, b, _)| *v != views || *b != buffers) {
            let view = wgpu::BindingResource::TextureView;
            let group = bind_group(ctx.device, "Cloud trace", &self.trace_layout, &[
                (0, ctx.camera.as_entire_binding()),
                (1, frame.as_entire_binding()),
                (2, sky_table.as_entire_binding()),
                (3, view(transmittance)),
                (4, view(multi)),
                (5, view(aerial)),
                (6, view(ctx.depth)),
                (7, view(weather)),
                (8, view(shape)),
                (9, view(detail)),
                (10, wgpu::BindingResource::Sampler(&self.noise_sampler)),
                (11, wgpu::BindingResource::Sampler(&self.lut_sampler)),
                (12, self.params.as_entire_binding()),
                (13, view(&targets.trace_color)),
                (14, view(&targets.trace_depth)),
            ]);
            self.trace_group = Some((views, buffers, group));
            self.reconstruct_groups = None;
        }
        if self.reconstruct_groups.as_ref().is_none_or(|(depth, _)| depth != ctx.depth) {
            let view = wgpu::BindingResource::TextureView;
            let group = |out: usize| {
                bind_group(ctx.device, "Cloud reconstruct", &self.reconstruct_layout, &[
                    (0, ctx.camera.as_entire_binding()),
                    (6, view(ctx.depth)),
                    (11, wgpu::BindingResource::Sampler(&self.lut_sampler)),
                    (12, self.params.as_entire_binding()),
                    (15, view(&targets.trace_color)),
                    (16, view(&targets.trace_depth)),
                    (17, view(&targets.color[1 - out])),
                    (18, view(&targets.depth[1 - out])),
                    (19, view(&targets.color[out])),
                    (20, view(&targets.depth[out])),
                ])
            };
            self.reconstruct_groups = Some((ctx.depth.clone(), [group(0), group(1)]));
        }
        let terrain = registry.texture_binding(DIRECTIONAL_VISIBILITY).unwrap_or(&self.no_terrain);
        let views = vec![ctx.depth.clone(), shadow_map.clone(), terrain.clone()];
        let buffers = vec![ctx.camera.clone(), frame.clone(), shadow_frame.clone()];
        if self.visibility_group.as_ref().is_none_or(|(v, b, _)| *v != views || *b != buffers) {
            let view = wgpu::BindingResource::TextureView;
            let group = bind_group(ctx.device, "Sun visibility", &self.visibility_layout, &[
                (0, ctx.camera.as_entire_binding()),
                (1, frame.as_entire_binding()),
                (6, view(ctx.depth)),
                (12, self.params.as_entire_binding()),
                (21, shadow_frame.as_entire_binding()),
                (22, view(shadow_map)),
                (23, view(terrain)),
                (24, view(&targets.sun_visibility)),
            ]);
            self.visibility_group = Some((views, buffers, group));
        }
        let (Some((_, _, trace_group)), Some((_, reconstruct_groups)), Some((_, _, visibility_group))) =
            (&self.trace_group, &self.reconstruct_groups, &self.visibility_group)
        else {
            return Ok(());
        };
        let mut cmds = ctx.graphics_cmds();
        let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Cloud trace"), timestamp_writes: None });
        pass.set_pipeline(&self.visibility);
        pass.set_bind_group(0, Some(visibility_group), &[]);
        pass.dispatch_workgroups(targets.full[0].div_ceil(8), targets.full[1].div_ceil(8), 1);
        pass.set_pipeline(&self.trace);
        pass.set_bind_group(0, Some(trace_group), &[]);
        pass.dispatch_workgroups(targets.quarter[0].div_ceil(8), targets.quarter[1].div_ceil(8), 1);
        pass.set_pipeline(&self.reconstruct);
        pass.set_bind_group(0, Some(&reconstruct_groups[current]), &[]);
        pass.dispatch_workgroups(targets.half[0].div_ceil(8), targets.half[1].div_ceil(8), 1);
        drop(pass);
        self.history = true;
        Ok(())
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        let Some(targets) = &self.targets else { return };
        if !self.active {
            return;
        }
        let current = self.current();
        frame.write_texture_binding(ATMOSPHERE_CLOUDS, &targets.color[current], "CloudTrace");
        frame.write_texture_binding(ATMOSPHERE_CLOUD_DEPTH, &targets.depth[current], "CloudTrace");
        frame.write_texture_binding(SUN_VISIBILITY, &targets.sun_visibility, "CloudTrace");
    }

    fn on_resize(&mut self, _device: &wgpu::Device, _width: u32, _height: u32) {
        self.history = false;
    }
}
