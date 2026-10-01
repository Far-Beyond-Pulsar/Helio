const _RC_TRACE_WGSL: &str = include_str!("../shaders/rc_trace.wgsl");

use bytemuck::{Pod, Zeroable};
use helio_core::graph::{ResourceBuilder, ResourceSize};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

/// Radiance Cascades volume settings. The pass owns these because both the
/// traced volume and its bounds are specific to this GI technique.
#[derive(Debug, Clone, Copy)]
pub struct GiConfig {
    /// Half extent of the camera-centered volume in world units.
    pub rc_radius: f32,
    /// Fade margin used by consumers when blending to ambient GI.
    pub rc_fade_margin: f32,
}

impl Default for GiConfig {
    fn default() -> Self {
        Self { rc_radius: 80.0, rc_fade_margin: 20.0 }
    }
}

impl GiConfig {
    pub fn ambient_only() -> Self { Self { rc_radius: 0.0, rc_fade_margin: 0.0 } }

    pub fn large_radius(radius: f32) -> Self {
        Self { rc_radius: radius, rc_fade_margin: radius * 0.25 }
    }
}

/// Radiance-cascades GI volume extent (dual-tier GI: RC near, ambient far).
///
/// Published by [`RadianceCascadesPass`] from its [`GiConfig`] and the frame's
/// camera, before any pass prepares, under the well-known
/// `"radiance_cascades_volume"` [`helio_core::ResourceKey`], separate from the generic `RenderEnvironment`
/// resource (clear color, ambient fallback, TLAS): this bounds volume is
/// specific to the radiance-cascades GI technique, not a property every
/// shading pass's environment has, so it is this pass's own resource, not a
/// smuggled field on a core-owned type.
#[derive(Clone, Copy)]
pub struct RadianceCascadesVolume {
    pub world_min: [f32; 3],
    pub world_max: [f32; 3],
}

/// Resource-registry key for [`RadianceCascadesVolume`].
pub const RADIANCE_CASCADES_VOLUME: helio_core::ResourceKey<RadianceCascadesVolume> =
    helio_core::ResourceKey::new("radiance_cascades_volume");

const PROBE_DIM: u32 = 8;
const DIR_DIM: u32 = 4;
const ATLAS_W: u32 = PROBE_DIM * DIR_DIM;
const ATLAS_H: u32 = PROBE_DIM * PROBE_DIM * DIR_DIM;

const WORKGROUP_SIZE_X: u32 = 8;
const WORKGROUP_SIZE_Y: u32 = 8;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RCDynamic {
    world_min: [f32; 4],
    world_max: [f32; 4],
    frame: u32,
    light_count: u32,
    _pad0: u32,
    _pad1: u32,
    sky_color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RCStatic {
    cascade_index: u32,
    probe_dim: u32,
    dir_dim: u32,
    t_max_bits: u32,
    parent_probe_dim: u32,
    parent_dir_dim: u32,
    _pad0: u32,
    _pad1: u32,
}

pub struct RadianceCascadesPass {
    gi_config: GiConfig,
    /// Fallback pipeline (no RT).
    fb_pipeline: wgpu::ComputePipeline,
    /// RT pipeline (real rc_trace.wgsl).
    rt_pipeline: Option<wgpu::ComputePipeline>,
    fb_bgl: wgpu::BindGroupLayout,
    rt_bgl: Option<wgpu::BindGroupLayout>,
    fb_bind_group: Option<wgpu::BindGroup>,
    fb_bg_key: Option<(usize, usize, usize)>,
    uniform_buf: wgpu::Buffer,
    static_buf: Option<wgpu::Buffer>,
    use_rt: bool,
    /// Live light list for the RT trace (`RC_COMPACT_WGSL`). RT only.
    compact: Option<LiveLightCompaction>,
    /// RT only: textures the trace reads that must not alias what it writes
    /// in the same dispatch (Helio#304).
    rt_targets: Option<RtTargets>,
}

/// History ping-pong and the parent-cascade input for the RT trace. Reading
/// and storing one texture in a single dispatch is a usage conflict wgpu
/// rejects, so each frame reads last frame's history and writes the other.
struct RtTargets {
    history: [wgpu::TextureView; 2],
    /// Index of the history texture read this frame; the other is written.
    read: usize,
    /// Input for the parent-cascade merge. Only cascade 0 is dispatched
    /// (`parent_dir_dim = 0`, so the shader never reads it); a real merge
    /// binds the coarser cascade's output here.
    parent_placeholder: wgpu::TextureView,
}

impl RtTargets {
    fn new(device: &wgpu::Device) -> Self {
        let texture = |label, width, height, usage| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let history_usage =
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING;
        Self {
            history: [
                texture("RC History A", ATLAS_W, ATLAS_H, history_usage),
                texture("RC History B", ATLAS_W, ATLAS_H, history_usage),
            ],
            read: 0,
            parent_placeholder: texture("RC Parent Placeholder", 1, 1, wgpu::TextureUsages::TEXTURE_BINDING),
        }
    }
}

/// Builds `[count, row...]` of live `"scene_lights"` rows so each ray hit
/// sums the lights that exist, not every allocated row (#838). Rebuilt only
/// when the rows change.
struct LiveLightCompaction {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    live_buf: wgpu::Buffer,
    /// `(epoch, content_generation, row_capacity)` the list was built from.
    key: Option<(u64, u64, u32)>,
    /// Set by `prepare` when `key` is stale.
    pending: Option<(u64, u64, u32)>,
}

/// Unordered compaction is fine: the trace only sums over the list.
const RC_COMPACT_WGSL: &str = r#"
struct GpuLight {
    position_range:  vec4<f32>,
    direction_outer: vec4<f32>,
    color_intensity: vec4<f32>,
    _rest:           array<vec4<f32>, 5>,
}
struct Params { row_count: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(1) var<storage, read_write> live: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn compact(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= min(params.row_count, arrayLength(&lights)) { return; }
    let c = lights[i].color_intensity;
    if c.w <= 0.0 || all(c.rgb <= vec3<f32>(0.0)) { return; }
    let slot = atomicAdd(&live[0], 1u);
    if slot + 1u < arrayLength(&live) { atomicStore(&live[slot + 1u], i); }
}
"#;

const FALLBACK_WGSL: &str = r#"
struct RCDynamic {
    world_min:   vec4<f32>,
    world_max:   vec4<f32>,
    frame:       u32,
    light_count: u32,
    _pad0:       u32,
    _pad1:       u32,
    sky_color:   vec4<f32>,
}

struct Camera {
    view:           mat4x4<f32>,
    proj:           mat4x4<f32>,
    view_proj:      mat4x4<f32>,
    view_proj_inv:  mat4x4<f32>,
    position_near:  vec4<f32>,
    forward_far:    vec4<f32>,
    jitter_frame:   vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}

@group(0) @binding(0) var cascade_out:   texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var<uniform>  rc_dyn:      RCDynamic;
@group(0) @binding(2) var depth_tex:    texture_2d<f32>;
@group(0) @binding(3) var scene_color:  texture_2d<f32>;
@group(0) @binding(4) var<storage, read> cameras: array<Camera, 2>;

const PROBE_DIM:   u32 = 8u;
const DIR_DIM:     u32 = 4u;
const MAX_RAY_DIST: f32 = 100.0;
const MARCH_STEPS:  u32 = 32u;

fn oct_decode(uv: vec2<f32>) -> vec3<f32> {
    let f  = uv * 2.0 - 1.0;
    let af = abs(f);
    let l  = af.x + af.y;
    var n: vec3<f32>;
    if l > 1.0 {
        let sx = select(-1.0, 1.0, f.x >= 0.0);
        let sz = select(-1.0, 1.0, f.y >= 0.0);
        n = vec3<f32>((1.0 - af.y) * sx, 1.0 - l, (1.0 - af.x) * sz);
    } else {
        n = vec3<f32>(f.x, 1.0 - l, f.y);
    }
    return normalize(n);
}

fn helio_ndc_to_uv(ndc: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let atlas_w = PROBE_DIM * DIR_DIM;
    let atlas_h = PROBE_DIM * PROBE_DIM * DIR_DIM;

    if gid.x >= atlas_w || gid.y >= atlas_h { return; }

    let dx = gid.x % DIR_DIM;
    let px = gid.x / DIR_DIM;
    let dy = gid.y % DIR_DIM;
    let pyz = gid.y / DIR_DIM;
    let pz = pyz % PROBE_DIM;
    let py = pyz / PROBE_DIM;

    let dir_uv = (vec2<f32>(f32(dx), f32(dy)) + 0.5) / f32(DIR_DIM);
    let dir = oct_decode(dir_uv);

    let t = (vec3<f32>(f32(px), f32(py), f32(pz)) + 0.5) / f32(PROBE_DIM);
    let world_size = rc_dyn.world_max.xyz - rc_dyn.world_min.xyz;
    let probe_pos = rc_dyn.world_min.xyz + t * world_size;

    let start_world = probe_pos;
    let end_world   = start_world + dir * MAX_RAY_DIST;

    let clip_start = cameras[0].view_proj * vec4<f32>(start_world, 1.0);
    let clip_end   = cameras[0].view_proj * vec4<f32>(end_world, 1.0);

    if clip_start.w <= 0.0 {
        textureStore(cascade_out, vec2<i32>(i32(gid.x), i32(gid.y)),
            vec4<f32>(rc_dyn.sky_color.rgb, 0.0));
        return;
    }

    let ndc_start = clip_start.xyz / clip_start.w;
    let ndc_end   = clip_end.xyz   / clip_end.w;

    let uv_start    = helio_ndc_to_uv(ndc_start.xy);
    let uv_end      = helio_ndc_to_uv(ndc_end.xy);
    let depth_start = ndc_start.z;
    let depth_end   = ndc_end.z;

    let delta_uv    = uv_end - uv_start;
    let delta_depth = depth_end - depth_start;

    let depth_dims = textureDimensions(depth_tex);
    let scene_dims = vec2<f32>(textureDimensions(scene_color));
    var radiance = vec3<f32>(0.0);
    var hit = false;

    for (var i: u32 = 1u; i <= MARCH_STEPS; i++) {
        let t_step = f32(i) / f32(MARCH_STEPS);
        let uv = uv_start + delta_uv * t_step;
        let d  = depth_start + delta_depth * t_step;

        if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) { break; }

        // Hi-Z mip 0 is an R32Float copy of the previous frame's depth. Reading
        // it as a color texture avoids the invalid combined shadow sampler that
        // Naga emits for sampled depth textures on OpenGL/GLES.
        let depth_coord = min(
            vec2<i32>(uv * vec2<f32>(depth_dims)),
            vec2<i32>(depth_dims) - vec2<i32>(1),
        );
        let scene_d = textureLoad(depth_tex, depth_coord, 0).x;

        if scene_d >= 1.0 { continue; }

        if d >= scene_d {
            let scene_coord = min(
                vec2<i32>(uv * scene_dims),
                vec2<i32>(textureDimensions(scene_color)) - vec2<i32>(1),
            );
            radiance = textureLoad(scene_color,
                scene_coord, 0).rgb;
            hit = true;
            break;
        }
    }

    if !hit {
        radiance = rc_dyn.sky_color.rgb;
    }

    textureStore(cascade_out, vec2<i32>(i32(gid.x), i32(gid.y)),
        vec4<f32>(radiance, 0.0));
}
"#;

impl LiveLightCompaction {
    fn new(device: &wgpu::Device) -> Self {
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RC Live Lights BGL"),
            entries: &[
                storage(0, true),
                storage(1, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RC Live Lights"),
            source: wgpu::ShaderSource::Wgsl(RC_COMPACT_WGSL.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RC Live Lights PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RC Live Lights"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("compact"),
            compilation_options: Default::default(),
            cache: None,
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RC Live Lights Params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            bgl,
            params_buf,
            live_buf: Self::live_buffer(device, 1),
            key: None,
            pending: None,
        }
    }

    fn live_buffer(device: &wgpu::Device, rows: u32) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RC Live Lights"),
            size: (u64::from(rows) + 1) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Rebuild the list if the light rows changed since it was built.
    /// Queue a rebuild if the light rows changed since the list was built:
    /// grows the list and uploads the row count, so `record` only dispatches.
    fn prepare(&mut self, ctx: &PrepareContext) {
        let key = ctx
            .scene_buffers
            .get(helio_core::BufferKey::of("scene_lights"))
            .map_or((0, 0, 0), |l| (l.epoch, l.content_generation, l.row_capacity()));
        self.pending = (self.key != Some(key)).then_some(key);
        let Some((_, _, rows)) = self.pending else { return };
        if self.live_buf.size() < (u64::from(rows) + 1) * 4 {
            self.live_buf = Self::live_buffer(ctx.device, rows.next_power_of_two());
        }
        ctx.queue.write_buffer(&self.params_buf, 0, bytemuck::cast_slice(&[rows, 0u32, 0, 0]));
    }

    /// Rebuild the list if `prepare` found it stale.
    fn record(&mut self, ctx: &mut PassContext, lights_buf: &wgpu::Buffer) {
        let Some(key) = self.pending.take() else { return };
        let rows = key.2;
        let mut encoder = ctx.graphics_cmds();
        encoder.clear_buffer(&self.live_buf, 0, Some(4));
        if rows > 0 {
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("RC Live Lights BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: lights_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.live_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.params_buf.as_entire_binding() },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RC Live Lights"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(rows.div_ceil(256), 1, 1);
        }
        self.key = Some(key);
    }
}

impl RadianceCascadesPass {
    pub fn new(device: &wgpu::Device, lights_buf: &wgpu::Buffer) -> Self {
        let _ = lights_buf;

        let use_rt = device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);

        // ── Uniform buffers ────────────────────────────────────────────
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RC Dynamic Uniform"),
            size: std::mem::size_of::<RCDynamic>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let static_buf = use_rt.then(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("RC Static Uniform"),
                size: std::mem::size_of::<RCStatic>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        // ── Fallback BGL & pipeline ────────────────────────────────────
        let fb_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RC Fallback BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let fb_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RC Fallback Shader"),
            source: wgpu::ShaderSource::Wgsl(FALLBACK_WGSL.into()),
        });

        let fb_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RC Fallback PL"),
            bind_group_layouts: &[Some(&fb_bgl)],
            immediate_size: 0,
        });

        let fb_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RC Fallback Pipeline"),
            layout: Some(&fb_pl),
            module: &fb_shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ── RT BGL & pipeline (if supported) ───────────────────────────
        let (rt_bgl, rt_pipeline) = if use_rt {
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("RC Trace BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::AccelerationStructure {
                            vertex_return: false,
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
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

            let rt_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("RC Trace Shader"),
                source: wgpu::ShaderSource::Wgsl(_RC_TRACE_WGSL.into()),
            });

            let rt_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("RC Trace PL"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });

            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("RC Trace Pipeline"),
                layout: Some(&rt_pl),
                module: &rt_shader,
                entry_point: Some("cs_trace"),
                compilation_options: Default::default(),
                cache: None,
            });

            (Some(bgl), Some(pipeline))
        } else {
            (None, None)
        };
        let compact = use_rt.then(|| LiveLightCompaction::new(device));
        let rt_targets = use_rt.then(|| RtTargets::new(device));

        Self {
            gi_config: GiConfig::default(),
            fb_pipeline,
            rt_pipeline,
            fb_bgl,
            rt_bgl,
            fb_bind_group: None,
            fb_bg_key: None,
            uniform_buf,
            static_buf,
            use_rt,
            compact,
            rt_targets,
        }
    }

    /// Configure the camera-centered RC volume.
    pub fn set_gi_config(&mut self, config: GiConfig) {
        self.gi_config = config;
    }

    /// The camera-centred volume this pass traces, and the one it publishes
    /// for consumers, so the two can never disagree.
    fn volume(&self, camera_position: [f32; 4]) -> RadianceCascadesVolume {
        let radius = self.gi_config.rc_radius.max(0.0);
        let [x, y, z, _] = camera_position;
        RadianceCascadesVolume {
            world_min: [x - radius, y - radius, z - radius],
            world_max: [x + radius, y + radius, z + radius],
        }
    }

    /// The RT trace's bind group (binding order matches the BGL). It stores
    /// into `cascade_out` and this frame's history texture, and reads the
    /// other history texture and the parent placeholder, so no texture is
    /// both read and stored in the dispatch (Helio#304).
    fn trace_bind_group(
        &self,
        device: &wgpu::Device,
        cascade_out: &wgpu::TextureView,
        tlas: &wgpu::Tlas,
        lights: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let targets = self.rt_targets.as_ref().expect("RT path always has its targets");
        let view = |view| wgpu::BindingResource::TextureView(view);
        let entries = [
            wgpu::BindGroupEntry { binding: 0, resource: view(cascade_out) },
            wgpu::BindGroupEntry { binding: 1, resource: view(&targets.parent_placeholder) },
            wgpu::BindGroupEntry { binding: 2, resource: self.uniform_buf.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: self.static_buf.as_ref().expect("RT static uniforms").as_entire_binding(),
            },
            wgpu::BindGroupEntry { binding: 4, resource: tlas.as_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: lights.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: view(&targets.history[targets.read]) },
            wgpu::BindGroupEntry { binding: 7, resource: view(&targets.history[1 - targets.read]) },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: self
                    .compact
                    .as_ref()
                    .expect("RT path always has a live-light list")
                    .live_buf
                    .as_entire_binding(),
            },
        ];
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RC Trace BG"),
            layout: self.rt_bgl.as_ref().expect("RT layout"),
            entries: &entries,
        })
    }
}

impl RenderPass for RadianceCascadesPass {
    fn name(&self) -> &'static str {
        "RadianceCascades"
    }

    fn reads(&self) -> &'static [&'static str] {
        &["hiz", "pre_aa", "render_environment"]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.write_color_raw(
            "rc_cascades",
            wgpu::TextureFormat::Rgba16Float,
            ResourceSize::Absolute {
                width: ATLAS_W,
                height: ATLAS_H,
            },
        );
        builder.with_extra_usage(wgpu::TextureUsages::STORAGE_BINDING);
        // RT history lives in `RtTargets` (a pass-owned ping-pong pair):
        // one pool texture cannot be read and written in the same dispatch.
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a helio_core::ResourceRegistry<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn publish_frame_inputs<'a>(
        &self,
        camera: &helio_core::GpuCameraUniforms,
        frame: &mut helio_core::ResourceRegistry<'a>,
    ) {
        // Before any pass prepares: GBuffer and deferred lighting read the
        // volume ahead of this pass in graph order.
        frame.write(
            RADIANCE_CASCADES_VOLUME,
            self.volume(camera.position_near),
            "RadianceCascades",
        );
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let light_count = ctx
            .scene_buffers
            .get(helio_core::BufferKey::of("scene_lights"))
            .map_or(0, |lights| lights.row_capacity());
        let sky = ctx
            .registry
            .get::<helio_pass_sky::SkyContext>(helio_core::ResourceKey::new("sky"))
            .map(|sky| sky.sky_color)
            .unwrap_or([0.0, 0.0, 0.0]);
        let volume = self.volume(ctx.camera_data.position_near);
        let dyn_data = RCDynamic {
            world_min: [volume.world_min[0], volume.world_min[1], volume.world_min[2], 0.0],
            world_max: [volume.world_max[0], volume.world_max[1], volume.world_max[2], 0.0],
            frame: ctx.frame_num as u32,
            light_count,
            _pad0: 0,
            _pad1: 0,
            sky_color: [sky[0], sky[1], sky[2], 0.0],
        };
        ctx.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&dyn_data));

        if let Some(ref static_buf) = self.static_buf {
            let static_data = RCStatic {
                cascade_index: 0,
                probe_dim: PROBE_DIM,
                dir_dim: DIR_DIM,
                t_max_bits: f32::MAX.to_bits(),
                parent_probe_dim: 0,
                parent_dir_dim: 0,
                _pad0: 0,
                _pad1: 0,
            };
            ctx.write_buffer(static_buf, 0, bytemuck::bytes_of(&static_data));
        }
        if let Some(compact) = self.compact.as_mut() {
            compact.prepare(ctx);
        }

        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if self.use_rt {
            self.execute_rt(ctx)
        } else {
            self.execute_fallback(ctx)
        }
    }
}

impl RadianceCascadesPass {
    fn execute_fallback(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let tex = ctx
            .resource_pool
            .get_texture("rc_cascades")
            .ok_or_else(|| {
                helio_core::Error::InvalidPassConfig(
                    "RadianceCascades: missing rc_cascades texture".into(),
                )
            })?;
        let depth_view = match ctx.registry.get(helio_core::ResourceKey::new("hiz")) {
            Some(v) => v,
            None => return Ok(()),
        };
        let pre_aa_view = match ctx.registry.get(helio_core::ResourceKey::new("pre_aa")) {
            Some(v) => v,
            None => return Ok(()),
        };

        // `rc_cascades` is a persistent resource-pool texture that only
        // changes on resize, and Hi-Z/pre_aa views are stable for the life
        // of the frame graph — recreating the view + bind group every frame
        // was pure CPU/driver overhead. Cache by resource identity instead.
        let key = (
            tex as *const wgpu::Texture as usize,
            depth_view as *const wgpu::TextureView as usize,
            pre_aa_view as *const wgpu::TextureView as usize,
        );
        if self.fb_bg_key != Some(key) {
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.fb_bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("RC Fallback BG"),
                layout: &self.fb_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(pre_aa_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: ctx.camera.as_entire_binding(),
                    },
                ],
            }));
            self.fb_bg_key = Some(key);
        }

        let wg_x = ATLAS_W.div_ceil(WORKGROUP_SIZE_X);
        let wg_y = ATLAS_H.div_ceil(WORKGROUP_SIZE_Y);

        let desc = wgpu::ComputePassDescriptor {
            label: Some("RadianceCascades (Fallback)"),
            timestamp_writes: None,
        };
        let mut cmds = ctx.graphics_cmds();
        let mut pass = cmds.begin_compute_pass(&desc);
        pass.set_pipeline(&self.fb_pipeline);
        pass.set_bind_group(0, self.fb_bind_group.as_ref().unwrap(), &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
        Ok(())
    }

    fn execute_rt(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let rt_bgl = self.rt_bgl.as_ref().unwrap();
        let rt_pipeline = self.rt_pipeline.as_ref().unwrap();

        let cascade_out = ctx
            .resource_pool
            .get_texture("rc_cascades")
            .ok_or_else(|| {
                helio_core::Error::InvalidPassConfig(
                    "RadianceCascades: missing rc_cascades texture".into(),
                )
            })?;
        let cascade_out_view = cascade_out.create_view(&wgpu::TextureViewDescriptor::default());

        let lights_buf = ctx
            .scene_buffers
            .get(helio_core::BufferKey::of("scene_lights"))
            .map_or_else(|| ctx.camera.clone(), |handle| handle.buffer.clone());
        let lights_buf = &lights_buf;
        if let Some(compact) = self.compact.as_mut() {
            compact.record(ctx, lights_buf);
        }

        // Get TLAS from frame resources (set by the renderer from GpuScene)
        let environment = ctx.registry.read::<helio_core::RenderEnvironment>(helio_core::resource_keys::render_environment(), "RadianceCascades");
        let tlas = environment.and_then(|value| value.tlas);

        let Some(tlas) = tlas else {
            // No TLAS — fall back to the ambient-only fallback shader.
            return self.execute_fallback(ctx);
        };

        let bind_group = self.trace_bind_group(ctx.device, &cascade_out_view, tlas, lights_buf);

        let wg_x = ATLAS_W.div_ceil(WORKGROUP_SIZE_X);
        let wg_y = ATLAS_H.div_ceil(WORKGROUP_SIZE_Y);

        let desc = wgpu::ComputePassDescriptor {
            label: Some("RadianceCascades (RT)"),
            timestamp_writes: None,
        };
        let mut cmds = ctx.graphics_cmds();
        let mut pass = cmds.begin_compute_pass(&desc);
        pass.set_pipeline(rt_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
        drop(pass);
        // This frame's output is next frame's history.
        if let Some(targets) = self.rt_targets.as_mut() {
            targets.read = 1 - targets.read;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{RadianceCascadesPass, FALLBACK_WGSL};
    use naga::back::glsl;

    /// Helio#304: the RT trace must never read a texture it also stores to.
    /// Two frames through the real bind group (history ping-ponged between
    /// them) must raise no validation error. Needs a ray-query adapter.
    #[test]
    fn rt_trace_binds_no_texture_it_also_writes() {
        pollster::block_on(async {
            let instance =
                wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
            let Ok(adapter) = instance.request_adapter(&Default::default()).await else {
                eprintln!("skipping: no GPU adapter");
                return;
            };
            if !adapter.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) {
                eprintln!("skipping: adapter has no ray query");
                return;
            }
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
                    required_limits: adapter.limits(),
                    // Explicit GPU test: acknowledges the experimental ray API.
                    experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                    ..Default::default()
                })
                .await
                .unwrap();
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let lights = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rc test lights"),
                size: 128 * 2,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut pass = RadianceCascadesPass::new(&device, &lights);
            assert!(pass.use_rt, "ray-query device must take the RT path");
            let cascade_out = device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("rc test cascades"),
                    size: wgpu::Extent3d {
                        width: super::ATLAS_W,
                        height: super::ATLAS_H,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default());
            let tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
                label: Some("rc test tlas"),
                max_instances: 1,
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.build_acceleration_structures(
                std::iter::empty::<&wgpu::BlasBuildEntry>(),
                std::iter::once(&tlas),
            );
            for _frame in 0..2 {
                let bind_group = pass.trace_bind_group(&device, &cascade_out, &tlas, &lights);
                {
                    let mut compute = encoder.begin_compute_pass(&Default::default());
                    compute.set_pipeline(pass.rt_pipeline.as_ref().unwrap());
                    compute.set_bind_group(0, &bind_group, &[]);
                    compute.dispatch_workgroups(
                        super::ATLAS_W.div_ceil(super::WORKGROUP_SIZE_X),
                        super::ATLAS_H.div_ceil(super::WORKGROUP_SIZE_Y),
                        1,
                    );
                }
                let targets = pass.rt_targets.as_mut().unwrap();
                targets.read = 1 - targets.read;
            }
            queue.submit([encoder.finish()]);
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            let error = scope.pop().await;
            assert!(error.is_none(), "RT trace validation failed: {error:?}");
        });
    }

    /// The ray-query trace and the live-light compaction must parse and
    /// validate: the trace is only compiled on RT hardware otherwise.
    #[test]
    fn trace_and_live_light_shaders_validate() {
        for (name, source) in [
            ("rc_trace", super::_RC_TRACE_WGSL),
            ("live lights", super::RC_COMPACT_WGSL),
        ] {
            let module = naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|e| panic!("{name} must parse: {}", e.emit_to_string(source)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name} must validate: {e:?}"));
        }
    }

    #[test]
    fn fallback_shader_translates_to_gles() {
        let module = naga::front::wgsl::parse_str(FALLBACK_WGSL)
            .expect("Radiance Cascades fallback WGSL must parse");
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .subgroup_stages(naga::valid::ShaderStages::all())
        .subgroup_operations(naga::valid::SubgroupOperationSet::all())
        .validate(&module)
        .expect("Radiance Cascades fallback WGSL must validate");

        let mut output = String::new();
        glsl::Writer::new(
            &mut output,
            &module,
            &info,
            &glsl::Options::default(),
            &glsl::PipelineOptions {
                shader_stage: naga::ShaderStage::Compute,
                entry_point: "cs_main".into(),
                multiview: None,
            },
            naga::proc::BoundsCheckPolicies::default(),
        )
        .expect("Radiance Cascades fallback must lower to GLES")
        .write()
        .expect("Radiance Cascades fallback must emit GLES");

        assert!(output.contains("void main()"));
        assert!(output.contains("texelFetch"));
        assert!(
            !output.contains("sampler2DShadow"),
            "fallback depth must lower as an ordinary R32Float texture"
        );
    }

    async fn compile_on_available_backends() -> usize {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;

        for adapter in &adapters {
            let info = adapter.get_info();
            let backend = format!("{:?}", info.backend);
            let (device, _queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("Radiance Cascades Portability Test Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: adapter.limits(),
                    ..Default::default()
                })
                .await
                .unwrap_or_else(|error| panic!("{backend} adapter must create a device: {error}"));
            device.on_uncaptured_error(Arc::new(move |error| {
                panic!("Radiance Cascades {backend} validation error: {error:?}");
            }));

            let lights = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Radiance Cascades Portability Lights"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let _pass = RadianceCascadesPass::new(&device, &lights);
        }

        adapters.len()
    }

    #[test]
    fn fallback_pipeline_compiles_on_every_available_backend() {
        if pollster::block_on(compile_on_available_backends()) == 0 {
            eprintln!("skipping Radiance Cascades portability test: no GPU adapter available");
        }
    }
}
