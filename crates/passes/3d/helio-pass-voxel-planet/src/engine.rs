//! Helio integration: GPU residency, exact traversal and GBuffer output.
use crate::field::FieldConstants;
use crate::grid::Cell;
use crate::planet::Planet;
use crate::residency::{Capacity, FrameWork, Residency, NONE};
use bytemuck::{Pod, Zeroable};
use glam::{DVec3, Vec3};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;

pub const GBUFFER_FORMATS: [wgpu::TextureFormat; 8] = [
    wgpu::TextureFormat::Rgba8Unorm,
    wgpu::TextureFormat::Rgba16Float,
    wgpu::TextureFormat::Rgba8Unorm,
    wgpu::TextureFormat::Rgba16Float,
    wgpu::TextureFormat::Rg16Float,
    wgpu::TextureFormat::Rgba16Float,
    wgpu::TextureFormat::Rgba16Float,
    wgpu::TextureFormat::Rgba16Float,
];

/// One published view of the planet. `eye` is the planet-centred camera
/// position; the host renders Helio with its world origin at `eye`.
#[derive(Clone)]
pub struct PlanetFrame {
    pub eye: DVec3,
    pub planet: Arc<Planet>,
    /// Direction towards the sun (planet-centred frame).
    pub sun: Vec3,
    pub shadows: bool,
}

pub type SharedPlanetFrame = Arc<Mutex<Option<PlanetFrame>>>;

#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// Level cells project to this many pixels where their range starts.
    pub lod_pixels: f32,
    /// Relative width of the stochastic level transition.
    pub lod_dither: f32,
    /// Column jobs per frame.
    pub job_budget: usize,
    /// Start primary rays at the conservative per-tile beam distance.
    pub beam: bool,
    /// End rising eye rays at the directional sky bound.
    pub horizon: bool,
    /// Diagnostics: skip residency planning (no jobs, windows or evictions)
    /// so several renders see identical GPU state.
    pub freeze_residency: bool,
    pub capacity: Capacity,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            lod_pixels: 1.0,
            lod_dither: std::env::var("HELIO_VOXEL_LOD_DITHER").ok().and_then(|v| v.parse().ok()).unwrap_or(0.25),
            job_budget: 12_288,
            beam: std::env::var_os("HELIO_VOXEL_NO_BEAM").is_none(),
            horizon: std::env::var_os("HELIO_VOXEL_NO_HORIZON").is_none(),
            freeze_residency: false,
            capacity: Capacity::default(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct FaceGpu {
    m_a: [f32; 4],
    q_a: [f32; 4],
    m_b: [f32; 4],
    q_b: [f32; 4],
    index: [i32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct FrameGpu {
    faces: [FaceGpu; 6],
    eye: [f32; 4],
    layer: [f32; 4],
    layer_i: [i32; 4],
    lod: [f32; 4],
    screen: [f32; 4],
    sun: [f32; 4],
    counts: [u32; 4],
    neighbours: [[u32; 4]; 6],
    extra: [u32; 4],
    /// Per level: angular distance from the eye within which the level's
    /// summary blocks cannot be reached by eye rays above `lod.z`.
    ring: [[f32; 4]; 8],
}

/// Public per-frame statistics.
#[derive(Clone, Copy, Debug, Default)]
pub struct PlanetStats {
    pub ready: bool,
    pub resident_columns: usize,
    pub pending_columns: usize,
    pub jobs: usize,
    pub evictions: usize,
    pub failed_jobs: usize,
    pub overflow_columns: usize,
    pub free_pages: i32,
    pub pool_pages: u32,
    pub active_levels: u32,
    pub finest_level: u32,
    pub plan_cpu_ms: f64,
    pub upload_cpu_ms: f64,
    pub encode_cpu_ms: f64,
    pub window_rebuild_ms: f64,
    pub lod0_distance: f64,
    pub logical_bytes: u64,
}

struct Readback {
    buffer: wgpu::Buffer,
    keys: Vec<u64>,
    state: Arc<AtomicBool>,
    stage: u8, // 0 free, 1 encoded, 2 mapping
}

const PROBE_BYTES: u64 = 128;

fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn source(access: &str, parts: &[&str]) -> String {
    let mut s = String::from(include_str!("../shaders/field.wgsl"));
    // Generation updates the summaries atomically; traversal reads plain values.
    let level_top = if parts.iter().any(|p| p.contains("fn level_suffix")) { "atomic<i32>" } else { "i32" };
    s.push_str(&include_str!("../shaders/common.wgsl").replace("ACCESS", access).replace("LEVEL_TOP", level_top));
    for part in parts {
        s.push_str(&part.replace("ACCESS", access));
    }
    s
}

struct Pipelines {
    gen_layout: wgpu::BindGroupLayout,
    trace_layout: wgpu::BindGroupLayout,
    render_layout: wgpu::BindGroupLayout,
    camera_layout: wgpu::BindGroupLayout,
    patch: wgpu::ComputePipeline,
    patch_blocks: wgpu::ComputePipeline,
    evict: wgpu::ComputePipeline,
    generate: wgpu::ComputePipeline,
    count: wgpu::ComputePipeline,
    refill: wgpu::ComputePipeline,
    allocate: wgpu::ComputePipeline,
    fixup: wgpu::ComputePipeline,
    publish: wgpu::ComputePipeline,
    level_suffix: wgpu::ComputePipeline,
    primary: wgpu::ComputePipeline,
    beam: wgpu::ComputePipeline,
    horizon_clear: wgpu::ComputePipeline,
    horizon_blocks: wgpu::ComputePipeline,
    horizon_suffix: wgpu::ComputePipeline,
    shade: wgpu::ComputePipeline,
    sunlight: wgpu::ComputePipeline,
    gbuffer: wgpu::RenderPipeline,
}

impl Pipelines {
    fn new(device: &wgpu::Device) -> Self {
        let gen_entries: Vec<_> = [
            uniform(0),
            uniform(1),
            storage(2, false),
            storage(3, false),
            storage(4, false),
            storage(5, true),
            storage(6, true),
            storage(7, true),
            storage(8, false),
            storage(9, false),
            storage(10, false),
            storage(11, false),
            storage(12, false),
            storage(13, true),
            storage(14, false),
            storage(15, false),
        ]
        .into();
        let gen_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("planet generation"),
            entries: &gen_entries,
        });
        let mut trace_entries: Vec<_> = vec![
            uniform(0),
            uniform(1),
            storage(2, false),
            storage(3, false),
            storage(4, false),
            storage(5, true),
            storage(6, true),
            storage(7, false),
            storage(8, false),
            storage(14, false),
            storage(15, false),
            storage(16, false),
            storage(17, false),
            storage(18, false),
            storage(19, true),
        ];
        trace_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 9,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba16Float,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        });
        trace_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 10,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        let trace_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("planet trace"),
            entries: &trace_entries,
        });
        let render_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("planet gbuffer"),
            entries: &[uniform(0), storage(8, true)],
        });
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("planet camera"),
            entries: &[uniform(0)],
        });
        let module = |label, src: String| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };
        let gen_module = module("planet generation", source("read_write", &[include_str!("../shaders/generate.wgsl")]));
        let trace_src = [
            include_str!("../shaders/view.wgsl"),
            include_str!("../shaders/horizon.wgsl"),
            include_str!("../shaders/trace.wgsl"),
            include_str!("../shaders/beam.wgsl"),
            include_str!("../shaders/surface.wgsl"),
        ];
        let trace_module = module("planet trace", source("read_write", &trace_src));
        let render_module = module(
            "planet gbuffer",
            source("read", &[include_str!("../shaders/view.wgsl"), include_str!("../shaders/gbuffer.wgsl")]),
        );
        let gen_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("planet generation"),
            bind_group_layouts: &[Some(&gen_layout)],
            immediate_size: 0,
        });
        let trace_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("planet trace"),
            bind_group_layouts: &[Some(&trace_layout), Some(&camera_layout)],
            immediate_size: 0,
        });
        let render_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("planet gbuffer"),
            bind_group_layouts: &[Some(&render_layout), Some(&camera_layout)],
            immediate_size: 0,
        });
        let compute = |layout: &wgpu::PipelineLayout, module: &wgpu::ShaderModule, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(layout),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let gbuffer = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("planet gbuffer"),
            layout: Some(&render_pl),
            vertex: wgpu::VertexState {
                module: &render_module,
                entry_point: Some("fullscreen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_module,
                entry_point: Some("gbuffer"),
                compilation_options: Default::default(),
                targets: &GBUFFER_FORMATS.map(|format| {
                    Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })
                }),
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            patch: compute(&gen_pl, &gen_module, "patch_table"),
            patch_blocks: compute(&gen_pl, &gen_module, "patch_blocks"),
            evict: compute(&gen_pl, &gen_module, "evict"),
            generate: compute(&gen_pl, &gen_module, "generate"),
            count: compute(&gen_pl, &gen_module, "count"),
            refill: compute(&gen_pl, &gen_module, "refill"),
            allocate: compute(&gen_pl, &gen_module, "allocate"),
            fixup: compute(&gen_pl, &gen_module, "fixup"),
            publish: compute(&gen_pl, &gen_module, "publish"),
            level_suffix: compute(&gen_pl, &gen_module, "level_suffix"),
            primary: compute(&trace_pl, &trace_module, "primary"),
            beam: compute(&trace_pl, &trace_module, "beam"),
            horizon_clear: compute(&trace_pl, &trace_module, "horizon_clear"),
            horizon_blocks: compute(&trace_pl, &trace_module, "horizon_blocks"),
            horizon_suffix: compute(&trace_pl, &trace_module, "horizon_suffix"),
            shade: compute(&trace_pl, &trace_module, "shade"),
            sunlight: compute(&trace_pl, &trace_module, "sunlight"),
            gbuffer,
            gen_layout,
            trace_layout,
            render_layout,
            camera_layout,
        }
    }
}

struct Buffers {
    frame: wgpu::Buffer,
    field: wgpu::Buffer,
    table: wgpu::Buffer,
    records: wgpu::Buffer,
    pool: wgpu::Buffer,
    brushes: wgpu::Buffer,
    edit_refs: wgpu::Buffer,
    jobs: wgpu::Buffer,
    job_out: wgpu::Buffer,
    scratch: wgpu::Buffer,
    alloc: wgpu::Buffer,
    free_runs: wgpu::Buffer,
    free_pages: wgpu::Buffer,
    evictions: wgpu::Buffer,
    level_tops: wgpu::Buffer,
    block_state: wgpu::Buffer,
    /// Directional sky bound: accumulated and suffix tables.
    horizon_acc: wgpu::Buffer,
    horizon: wgpu::Buffer,
    /// Live tier-1 block slots (grows).
    live_blocks: wgpu::Buffer,
    brush_capacity: u32,
    bytes: u64,
}

const JOB_OUT_BYTES: u64 = 96;
/// Must match `SECTORS` and `BUCKETS` in horizon.wgsl.
const HORIZON_SECTORS: u32 = 256;
const HORIZON_BUCKETS: u32 = 32;
const HORIZON_GROUPS: u32 = 16;

impl Buffers {
    fn new(device: &wgpu::Device, cap: &Capacity, field: &FieldConstants) -> Self {
        let mut bytes = 0u64;
        let mut make = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            bytes += size;
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(16),
                usage,
                mapped_at_creation: false,
            })
        };
        let st = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let pages = cap.pool_units / 512;
        let frame = make("planet frame", std::mem::size_of::<FrameGpu>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let records = make("planet records", u64::from(cap.records) * 32, st);
        let pool = make("planet pool", u64::from(cap.pool_units) * 64, st);
        let edit_refs = make("planet edit refs", u64::from(cap.edit_words) * 4, st);
        let jobs = make("planet jobs", u64::from(cap.max_jobs) * 32, st);
        let job_out = make("planet job results", u64::from(cap.max_jobs) * JOB_OUT_BYTES, st | wgpu::BufferUsages::COPY_SRC);
        let scratch = make("planet scratch", u64::from(cap.scratch_units) * 64, st);
        let free_runs = make("planet free runs", u64::from(cap.pool_units) * 8, st);
        let block_state = make(
            "planet block summaries",
            u64::from(crate::residency::block_region()) * 6 * 24 * 16,
            st,
        );
        let evictions = make("planet evictions", (u64::from(cap.max_evictions) * 3 + u64::from(cap.max_jobs) * 2) * 4, st);
        let horizon_acc = make("planet horizon accumulation", u64::from((HORIZON_SECTORS + HORIZON_GROUPS) * HORIZON_BUCKETS) * 4, st);
        let horizon = make(
            "planet horizon bound",
            u64::from(2 * (HORIZON_SECTORS + 1) * HORIZON_BUCKETS) * 4,
            st | wgpu::BufferUsages::COPY_SRC,
        );
        let live_blocks = make("planet live summary blocks", 65_536 * 4, st);
        let brush_capacity = 65_536;
        let brushes = make("planet brushes", u64::from(brush_capacity) * 32, st | wgpu::BufferUsages::COPY_SRC);
        let table_init = vec![NONE; 1 << cap.table_bits];
        bytes += (table_init.len() * 4) as u64;
        let table = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("planet column table"),
            contents: bytemuck::cast_slice(&table_init),
            usage: st,
        });
        let pages_init: Vec<u32> = (0..pages).collect();
        bytes += u64::from(pages) * 4;
        let free_pages = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("planet free pages"),
            contents: bytemuck::cast_slice(&pages_init),
            usage: st,
        });
        let mut alloc_init = [0i32; 32];
        alloc_init[30] = pages as i32;
        let alloc = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("planet allocator"),
            contents: bytemuck::cast_slice(&alloc_init),
            usage: st | wgpu::BufferUsages::COPY_SRC,
        });
        let level_tops = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("planet level tops"),
            contents: bytemuck::cast_slice(&[i32::MIN / 2; 64]),
            usage: st,
        });
        let field = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("planet field"),
            contents: bytemuck::bytes_of(field),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        Self {
            frame,
            field,
            table,
            records,
            pool,
            brushes,
            edit_refs,
            jobs,
            job_out,
            scratch,
            alloc,
            free_runs,
            free_pages,
            evictions,
            level_tops,
            block_state,
            horizon_acc,
            horizon,
            live_blocks,
            brush_capacity,
            bytes,
        }
    }
}

struct Screen {
    size: [u32; 2],
    hits: wgpu::Buffer,
    surfaces: wgpu::Buffer,
    beams: wgpu::Buffer,
    sun: wgpu::Texture,
    sun_view: wgpu::TextureView,
}

impl Screen {
    fn new(device: &wgpu::Device, size: [u32; 2]) -> Self {
        let pixels = u64::from(size[0].max(1)) * u64::from(size[1].max(1));
        let hits = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planet hits"),
            size: pixels * 32,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let surfaces = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planet surfaces"),
            size: pixels * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let tiles = u64::from(size[0].div_ceil(4)) * u64::from(size[1].div_ceil(4));
        let beams = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planet beam distances"),
            size: tiles * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let sun = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("planet directional visibility"),
            size: wgpu::Extent3d {
                width: size[0].max(1),
                height: size[1].max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let sun_view = sun.create_view(&Default::default());
        Self {
            size,
            hits,
            surfaces,
            beams,
            sun,
            sun_view,
        }
    }
}

/// Active GPU state for one planet.
pub struct PlanetRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Pipelines,
    buffers: Buffers,
    screen: Screen,
    residency: Residency,
    planet: Arc<Planet>,
    settings: Settings,
    gen_group: wgpu::BindGroup,
    camera_group: Option<(usize, wgpu::BindGroup)>,
    readbacks: Vec<Readback>,
    frame_index: u32,
    stats: PlanetStats,
    sun_active: bool,
    profiler: Option<helio_core::profiling::GpuProfiler>,
    initial_complete: bool,
    /// Measured GPU generation cost per column job (EMA) and last job count.
    ms_per_job: f64,
    last_jobs: usize,
    last_eye: Option<DVec3>,
}

impl PlanetRenderer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, planet: Arc<Planet>, settings: Settings, size: [u32; 2]) -> Self {
        let pipelines = Pipelines::new(device);
        let buffers = Buffers::new(device, &settings.capacity, planet.field());
        let gen_group = Self::gen_group(device, &pipelines, &buffers);
        let readbacks = (0..4)
            .map(|_| Readback {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("planet status readback"),
                    size: u64::from(settings.capacity.max_jobs) * JOB_OUT_BYTES + PROBE_BYTES,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                keys: Vec::new(),
                state: Arc::new(AtomicBool::new(false)),
                stage: 0,
            })
            .collect();
        Self {
            device: device.clone(),
            queue: queue.clone(),
            screen: Screen::new(device, size),
            residency: Residency::with_worker(*planet.grid(), settings.capacity),
            planet,
            settings,
            gen_group,
            camera_group: None,
            readbacks,
            frame_index: 0,
            stats: PlanetStats::default(),
            sun_active: false,
            profiler: None,
            initial_complete: false,
            ms_per_job: 0.0013,
            last_jobs: 0,
            last_eye: None,
            pipelines,
            buffers,
        }
    }

    fn gen_group(device: &wgpu::Device, p: &Pipelines, b: &Buffers) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = [
            &b.frame, &b.field, &b.table, &b.records, &b.pool, &b.brushes, &b.edit_refs, &b.jobs, &b.job_out,
            &b.scratch, &b.alloc, &b.free_runs, &b.free_pages, &b.evictions, &b.level_tops, &b.block_state,
        ]
        .iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect();
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet generation"),
            layout: &p.gen_layout,
            entries: &entries,
        })
    }

    pub fn planet(&self) -> &Arc<Planet> {
        &self.planet
    }
    pub fn stats(&self) -> PlanetStats {
        self.stats
    }
    pub fn settings_mut(&mut self) -> &mut Settings {
        &mut self.settings
    }
    pub fn hit_buffer(&self) -> &wgpu::Buffer {
        &self.screen.hits
    }
    /// Directional sky bound table (diagnostics): `[bucket][sector]` suffix
    /// maxima in base layers, then one all-sector row per bucket.
    pub fn horizon_buffer(&self) -> &wgpu::Buffer {
        &self.buffers.horizon
    }
    pub fn surface_buffer(&self) -> &wgpu::Buffer {
        &self.screen.surfaces
    }
    pub fn screen_size(&self) -> [u32; 2] {
        self.screen.size
    }
    pub fn sun_texture(&self) -> &wgpu::Texture {
        &self.screen.sun
    }
    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiler = enabled.then(|| helio_core::profiling::GpuProfiler::new(&self.device, &self.queue));
    }
    pub fn profiler(&self) -> Option<&helio_core::profiling::GpuProfiler> {
        self.profiler.as_ref()
    }
    /// Wait for and return this renderer's most recent stage timings
    /// (`planet_*` scopes, milliseconds). For synchronized diagnostics only.
    pub fn stage_timings_blocking(&mut self) -> Vec<(&'static str, f64)> {
        let device = self.device.clone();
        self.profiler.as_mut().map_or_else(Vec::new, |p| {
            p.read_timestamps_blocking(&device)
                .iter()
                .map(|t| (t.name, t.duration_ns as f64 / 1.0e6))
                .collect()
        })
    }
    /// Residency has issued and completed every window column.
    pub fn settled(&self) -> bool {
        self.residency.idle() && self.readbacks.iter().all(|r| r.stage == 0)
    }

    fn frame_uniform(&self, eye: DVec3, size: [u32; 2], lod0: f64, jobs: u32, evictions: u32, sun: Vec3, shadows: bool) -> FrameGpu {
        let planet = &self.planet;
        let grid = planet.grid();
        let mut frame = FrameGpu::default();
        for face in 0..6u8 {
            let f = grid.face_frame(face, eye);
            let v = |d: DVec3, w: f64| [d.x as f32, d.y as f32, d.z as f32, w as f32];
            frame.faces[face as usize] = FaceGpu {
                m_a: v(f.m[0], f.rho[0]),
                q_a: v(f.q[0], f.fraction[0]),
                m_b: v(f.m[1], f.rho[1]),
                q_b: v(f.q[1], f.fraction[1]),
                index: [
                    f.index[0].clamp(i32::MIN as i64, i32::MAX as i64) as i32,
                    f.index[1].clamp(i32::MIN as i64, i32::MAX as i64) as i32,
                    i32::from(f.valid),
                    0,
                ],
            };
            let n = grid.cells();
            let edges = [
                Cell::new(face, 0, n / 2, 0),
                Cell::new(face, n - 1, n / 2, 0),
                Cell::new(face, n / 2, 0, 0),
                Cell::new(face, n / 2, n - 1, 0),
            ];
            for (e, (cell, (axis, step))) in edges.iter().zip([(0, -1), (0, 1), (1, -1), (1, 1)]).enumerate() {
                frame.neighbours[face as usize][e] = u32::from(grid.neighbour(*cell, axis, step).face);
            }
        }
        let rho = eye.length();
        let dir = eye / rho;
        let s = grid.voxel_size();
        let layer = (rho - grid.radius()) / s;
        let k = layer.floor();
        frame.eye = [dir.x as f32, dir.y as f32, dir.z as f32, rho as f32];
        frame.layer = [(layer - k) as f32, s as f32, grid.delta() as f32, (planet.outer_radius() - rho) as f32];
        frame.layer_i = [k.clamp(i32::MIN as f64, i32::MAX as f64) as i32, grid.cells(), grid.levels() as i32, i32::from(crate::grid::face_of(eye))];
        // Directional sky bound cut height (relative to the eye radius): the
        // bound applies to rising rays above it, so later hits lie in
        // [cut, outer]. See `horizon.wgsl`.
        const SKY_CUT_M: f64 = 100.0;
        frame.lod = [lod0 as f32, self.settings.lod_dither, -SKY_CUT_M as f32, (rho + grid.radius() * 3.0) as f32];
        // Nearest ray distance at which each level may fall back to coarser
        // data: chord bound for points past its fallback angle at radius
        // >= the cut radius.
        let r_lo = rho - SKY_CUT_M;
        let fallback: Vec<f64> = self
            .residency
            .fallback_angles(dir)
            .iter()
            .map(|a| if *a >= std::f64::consts::PI { f64::INFINITY } else { 2.0 * (rho * r_lo).sqrt() * (a * 0.5).sin() })
            .collect();
        let rings = sky_rings(lod0, f64::from(self.settings.lod_dither), rho, r_lo, planet.outer_radius(), &fallback);
        for (level, phi) in rings.iter().enumerate().take(32) {
            frame.ring[level / 4][level % 4] = *phi as f32;
        }
        let flags = u32::from(self.settings.beam) | (u32::from(self.settings.horizon) << 1);
        frame.screen = [size[0] as f32, size[1] as f32, (self.frame_index % 1024) as f32, flags as f32];
        let sun = sun.normalize_or_zero();
        frame.sun = [sun.x, sun.y, sun.z, if shadows { 1.0 } else { 0.0 }];
        frame.counts = [jobs, evictions, (1u32 << self.settings.capacity.table_bits) - 1, self.settings.capacity.pool_units];
        frame
    }

    /// Upload this frame's residency changes. Returns (table patches,
    /// summary block patches) appended after the eviction list.
    fn upload(&mut self, work: &FrameWork) -> (u32, u32) {
        let q = &self.queue;
        if work.full_table {
            q.write_buffer(&self.buffers.table, 0, bytemuck::cast_slice(self.residency.table()));
        }
        for (index, brush) in &work.brush_writes {
            if *index >= self.buffers.brush_capacity {
                self.grow_brushes(*index + 1);
            }
            self.queue.write_buffer(&self.buffers.brushes, u64::from(*index) * 32, bytemuck::bytes_of(brush));
        }
        for (base, words) in &work.edit_writes {
            self.queue.write_buffer(&self.buffers.edit_refs, u64::from(*base) * 4, bytemuck::cast_slice(words));
        }
        if !work.jobs.is_empty() {
            self.queue.write_buffer(&self.buffers.jobs, 0, bytemuck::cast_slice(&work.jobs));
        }
        // Evictions followed by table patches (slot, value) pairs.
        let mut words: Vec<u32> = work.evictions.clone();
        let patches = if work.full_table { 0 } else { work.table_writes.len() as u32 };
        if !work.full_table {
            for (slot, value) in &work.table_writes {
                words.push(*slot);
                words.push(*value);
            }
        }
        for (slot, bi, bj) in &work.block_inits {
            words.extend([*slot, *bi as u32, *bj as u32]);
        }
        if !words.is_empty() {
            let bytes = (words.len() * 4) as u64;
            if bytes > self.buffers.evictions.size() {
                self.buffers.evictions = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("planet evictions"),
                    size: bytes.next_power_of_two(),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.gen_group = Self::gen_group(&self.device, &self.pipelines, &self.buffers);
            }
            self.queue.write_buffer(&self.buffers.evictions, 0, bytemuck::cast_slice(&words));
        }
        (patches, work.block_inits.len() as u32)
    }

    fn grow_brushes(&mut self, needed: u32) {
        let capacity = needed.next_power_of_two().max(self.buffers.brush_capacity * 2);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planet brushes"),
            size: u64::from(capacity) * 32,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.buffers.brushes, 0, &buffer, 0, u64::from(self.buffers.brush_capacity) * 32);
        self.queue.submit([encoder.finish()]);
        self.buffers.bytes += u64::from(capacity - self.buffers.brush_capacity) * 32;
        self.buffers.brushes = buffer;
        self.buffers.brush_capacity = capacity;
        self.gen_group = Self::gen_group(&self.device, &self.pipelines, &self.buffers);
    }

    fn poll_readbacks(&mut self) {
        let _ = self.device.poll(wgpu::PollType::Poll);
        let mut failed = Vec::new();
        for r in &mut self.readbacks {
            if r.stage == 2 && r.state.load(Ordering::Acquire) {
                {
                    let data = r.buffer.slice(..).get_mapped_range().unwrap();
                    for (index, key) in r.keys.iter().enumerate() {
                        let at = index * JOB_OUT_BYTES as usize;
                        let status = u32::from_le_bytes(data[at..at + 4].try_into().unwrap());
                        if status != 0 {
                            failed.push((*key, status));
                        }
                    }
                    let probe = u64::from(self.settings.capacity.max_jobs) * JOB_OUT_BYTES;
                    let at = probe as usize;
                    self.stats.free_pages = i32::from_le_bytes(data[at + 120..at + 124].try_into().unwrap());
                }
                r.buffer.unmap();
                r.stage = 0;
                r.state.store(false, Ordering::Release);
            }
        }
        self.stats.failed_jobs += failed.iter().filter(|(_, s)| *s != 1).count();
        self.stats.overflow_columns += failed.iter().filter(|(_, s)| *s == 1).count();
        self.residency.requeue(failed);
        // Start mapping readbacks encoded in earlier frames.
        for r in &mut self.readbacks {
            if r.stage == 1 {
                let state = r.state.clone();
                r.buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                    if result.is_ok() {
                        state.store(true, Ordering::Release);
                    }
                });
                r.stage = 2;
            }
        }
    }

    fn dispatch(pass: &mut wgpu::ComputePass<'_>, pipeline: &wgpu::ComputePipeline, groups: [u32; 3]) {
        if groups.iter().all(|g| *g > 0) {
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
        }
    }

    /// Encode one frame: residency, primary visibility, shading and GBuffer.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        camera: &wgpu::Buffer,
        camera_data: &helio_core::GpuCameraUniforms,
        frame: &PlanetFrame,
        size: [u32; 2],
        gbuffer: [&wgpu::TextureView; 8],
        depth: &wgpu::TextureView,
        frame_num: u64,
    ) {
        if let Some(p) = &mut self.profiler {
            let residency: f64 = p
                .read_timestamps_deferred()
                .iter()
                .filter(|t| t.name == "planet_residency")
                .map(|t| t.duration_ns as f64 / 1.0e6)
                .sum();
            if self.last_jobs >= 256 && residency > 0.0 {
                let sample = residency / self.last_jobs as f64;
                self.ms_per_job = self.ms_per_job * 0.7 + sample * 0.3;
            }
        }
        self.frame_index = self.frame_index.wrapping_add(1);
        if self.screen.size != size {
            self.screen = Screen::new(&self.device, size);
        }
        self.poll_readbacks();
        let tan_half = 1.0 / f64::from(camera_data.proj[5]).abs().max(1e-6);
        let lod0 = Residency::lod_distance(self.planet.grid(), tan_half, size[1], f64::from(self.settings.lod_pixels));
        let started = std::time::Instant::now();
        // Generation budget: small while the view moves (frame pacing), large
        // when it is still (fast convergence), from the measured job cost.
        let moving = self.last_eye.is_none_or(|e| e.distance(frame.eye) > 0.01);
        self.last_eye = Some(frame.eye);
        let target_ms = if moving { 1.5 } else { 6.0 };
        let budget = ((target_ms / self.ms_per_job.max(1e-5)) as usize)
            .clamp(256, self.settings.job_budget.min(self.settings.capacity.max_jobs as usize));
        let work = if self.settings.freeze_residency {
            FrameWork::default()
        } else {
            self.residency.plan(&self.planet, frame.eye, lod0, budget)
        };
        self.last_jobs = work.jobs.len();
        self.stats.plan_cpu_ms = started.elapsed().as_secs_f64() * 1000.0;
        let uploading = std::time::Instant::now();
        let (patches, block_patches) = self.upload(&work);
        if let Some(live) = self.residency.take_live_blocks() {
            let bytes = (live.len() * 4) as u64;
            if bytes > self.buffers.live_blocks.size() {
                self.buffers.live_blocks = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("planet live summary blocks"),
                    size: bytes.next_power_of_two(),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            if !live.is_empty() {
                self.queue.write_buffer(&self.buffers.live_blocks, 0, bytemuck::cast_slice(live));
            }
        }
        let live_blocks = self.residency.live_block_count() as u32;
        self.stats.upload_cpu_ms = uploading.elapsed().as_secs_f64() * 1000.0;
        let encoding = std::time::Instant::now();
        let jobs = work.jobs.len() as u32;
        let evictions = work.evictions.len() as u32;
        let mut uniform = self.frame_uniform(frame.eye, size, lod0, jobs, evictions, frame.sun, frame.shadows);
        uniform.extra[0] = patches;
        uniform.extra[1] = crate::residency::block_region();
        uniform.extra[2] = block_patches;
        uniform.extra[3] = live_blocks;
        self.queue.write_buffer(&self.buffers.frame, 0, bytemuck::bytes_of(&uniform));
        let camera_key = camera as *const _ as usize;
        if self.camera_group.as_ref().is_none_or(|(k, _)| *k != camera_key) {
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("planet camera"),
                layout: &self.pipelines.camera_layout,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera.as_entire_binding() }],
            });
            self.camera_group = Some((camera_key, group));
        }
        let trace_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet trace"),
            layout: &self.pipelines.trace_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.buffers.frame.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buffers.field.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buffers.table.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buffers.records.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.buffers.pool.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.buffers.brushes.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.buffers.edit_refs.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.screen.hits.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: self.screen.surfaces.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&self.screen.sun_view) },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(depth) },
                wgpu::BindGroupEntry { binding: 14, resource: self.buffers.level_tops.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 15, resource: self.buffers.block_state.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 16, resource: self.screen.beams.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 17, resource: self.buffers.horizon_acc.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 18, resource: self.buffers.horizon.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 19, resource: self.buffers.live_blocks.as_entire_binding() },
            ],
        });
        let render_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet gbuffer"),
            layout: &self.pipelines.render_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.buffers.frame.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: self.screen.surfaces.as_entire_binding() },
            ],
        });
        let camera_group = &self.camera_group.as_ref().unwrap().1;
        if let Some(p) = &mut self.profiler {
            p.begin_pass(encoder, "planet_residency");
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &self.gen_group, &[]);
            let wg = |n: u32| n.div_ceil(64);
            if evictions > 0 {
                Self::dispatch(&mut pass, &self.pipelines.evict, [wg(evictions), 1, 1]);
            }
            if patches > 0 {
                // `patch_table` reads pairs after the eviction list.
                pass.set_pipeline(&self.pipelines.patch);
                pass.dispatch_workgroups(wg(patches), 1, 1);
            }
            if block_patches > 0 {
                pass.set_pipeline(&self.pipelines.patch_blocks);
                pass.dispatch_workgroups(wg(block_patches), 1, 1);
            }
            if jobs > 0 {
                let groups = [jobs.min(32_768), jobs.div_ceil(32_768), 1];
                Self::dispatch(&mut pass, &self.pipelines.generate, groups);
                Self::dispatch(&mut pass, &self.pipelines.count, [wg(jobs), 1, 1]);
                Self::dispatch(&mut pass, &self.pipelines.refill, [1, 1, 1]);
                Self::dispatch(&mut pass, &self.pipelines.allocate, [wg(jobs), 1, 1]);
                Self::dispatch(&mut pass, &self.pipelines.fixup, [1, 1, 1]);
                Self::dispatch(&mut pass, &self.pipelines.publish, groups);
                Self::dispatch(&mut pass, &self.pipelines.level_suffix, [1, 1, 1]);
            }
        }
        if let Some(p) = &mut self.profiler {
            p.end_pass(encoder, "planet_residency");
        }
        if jobs > 0 {
            if let Some(r) = self.readbacks.iter_mut().find(|r| r.stage == 0) {
                encoder.copy_buffer_to_buffer(&self.buffers.job_out, 0, &r.buffer, 0, u64::from(jobs) * JOB_OUT_BYTES);
                let probe = u64::from(self.settings.capacity.max_jobs) * JOB_OUT_BYTES;
                encoder.copy_buffer_to_buffer(&self.buffers.alloc, 0, &r.buffer, probe, PROBE_BYTES);
                r.keys = work.job_keys.clone();
                r.stage = 1;
            } else {
                // No readback slot: assume success; failures stay invisible
                // until the column is regenerated.
            }
        }
        let groups = [size[0].div_ceil(8), size[1].div_ceil(8), 1];
        if let Some(p) = &mut self.profiler {
            p.begin_pass(encoder, "planet_horizon");
        }
        {
            // Directional sky bound from this frame's summary blocks.
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &trace_group, &[]);
            pass.set_bind_group(1, camera_group, &[]);
            Self::dispatch(&mut pass, &self.pipelines.horizon_clear, [((HORIZON_SECTORS + HORIZON_GROUPS) * HORIZON_BUCKETS).div_ceil(64), 1, 1]);
            Self::dispatch(&mut pass, &self.pipelines.horizon_blocks, [live_blocks.div_ceil(64), 1, 1]);
            Self::dispatch(&mut pass, &self.pipelines.horizon_suffix, [1, 1, 1]);
        }
        if let Some(p) = &mut self.profiler {
            p.end_pass(encoder, "planet_horizon");
            p.begin_pass(encoder, "planet_primary");
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &trace_group, &[]);
            pass.set_bind_group(1, camera_group, &[]);
            Self::dispatch(&mut pass, &self.pipelines.beam, [size[0].div_ceil(32), size[1].div_ceil(32), 1]);
            Self::dispatch(&mut pass, &self.pipelines.primary, groups);
        }
        if let Some(p) = &mut self.profiler {
            p.end_pass(encoder, "planet_primary");
            p.begin_pass(encoder, "planet_shade");
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &trace_group, &[]);
            pass.set_bind_group(1, camera_group, &[]);
            Self::dispatch(&mut pass, &self.pipelines.shade, groups);
        }
        if let Some(p) = &mut self.profiler {
            p.end_pass(encoder, "planet_shade");
            p.begin_pass(encoder, "planet_gbuffer");
        }
        {
            let attachments = gbuffer.map(|view| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("planet gbuffer"),
                color_attachments: &attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipelines.gbuffer);
            pass.set_bind_group(0, &render_group, &[]);
            pass.set_bind_group(1, camera_group, &[]);
            pass.draw(0..3, 0..1);
        }
        if let Some(p) = &mut self.profiler {
            p.end_pass(encoder, "planet_gbuffer");
        }
        self.sun_active = frame.shadows;
        if frame.shadows {
            if let Some(p) = &mut self.profiler {
                p.begin_pass(encoder, "planet_sunlight");
            }
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_bind_group(0, &trace_group, &[]);
                pass.set_bind_group(1, camera_group, &[]);
                Self::dispatch(&mut pass, &self.pipelines.sunlight, [size[0].div_ceil(16), size[1].div_ceil(16), 1]);
            }
            if let Some(p) = &mut self.profiler {
                p.end_pass(encoder, "planet_sunlight");
            }
        }
        if let Some(p) = &mut self.profiler {
            p.resolve_queries(encoder, frame_num);
        }
        self.stats.encode_cpu_ms = encoding.elapsed().as_secs_f64() * 1000.0;
        let rs = self.residency.stats;
        self.stats.resident_columns = rs.resident_columns;
        self.stats.pending_columns = rs.pending_columns;
        self.stats.jobs = rs.jobs;
        self.stats.evictions = rs.evictions;
        self.stats.active_levels = rs.active_levels;
        self.stats.finest_level = rs.finest_level;
        self.stats.window_rebuild_ms = rs.window_rebuild_ms;
        self.stats.lod0_distance = lod0;
        self.stats.pool_pages = self.settings.capacity.pool_units / 512;
        self.stats.logical_bytes = self.buffers.bytes + u64::from(size[0]) * u64::from(size[1]) * (32 + 16 + 8);
        if self.residency.idle() {
            self.initial_complete = true;
        }
        self.stats.ready = self.initial_complete;
    }

    pub fn sun_view(&self) -> Option<&wgpu::TextureView> {
        self.sun_active.then_some(&self.screen.sun_view)
    }
}

/// Per level, the angular distance from the eye within which every point at
/// radius `[r_lo, r_hi]` is nearer than the first ray distance the level can
/// serve. Level L serves distances from its dithered ring start, or where a
/// finer level may be missing (rays fall back to coarser columns there):
/// from that level's ring start or its fallback distance, whichever is
/// farther.
fn sky_rings(lod0: f64, dither: f64, rho: f64, r_lo: f64, r_hi: f64, fallback: &[f64]) -> Vec<f64> {
    let start = |level: usize| {
        if level == 0 { 0.0 } else { lod0 * f64::from(1u32 << (level - 1).min(30)) / (1.0 + dither * 0.5) * 0.999 }
    };
    // Angular distance where a point at radius r is exactly t away (0 when a
    // point straight above or below the eye already is).
    let phi = |t: f64, r: f64| {
        let q = (t * t - (rho - r) * (rho - r)) / (4.0 * rho * r);
        if q <= 0.0 { 0.0 } else { 2.0 * q.sqrt().min(1.0).asin() }
    };
    let mut earliest = f64::INFINITY;
    fallback
        .iter()
        .enumerate()
        .map(|(level, from)| {
            let t = start(level).min(earliest);
            earliest = earliest.min(start(level).max(*from));
            let mut p = phi(t, r_lo).min(phi(t, r_hi));
            let r_star = (rho * rho - t * t).max(0.0).sqrt();
            if r_star > r_lo && r_star < r_hi {
                p = p.min(phi(t, r_star));
            }
            p * 0.999
        })
        .collect()
}

/// Graph entry that allocates GPU residency once a planet frame is published.
pub struct PlanetPass {
    source: SharedPlanetFrame,
    settings: Settings,
    active: Option<PlanetRenderer>,
    profiling: bool,
}

impl PlanetPass {
    pub fn new(source: SharedPlanetFrame) -> Self {
        Self::with_settings(source, Settings::default())
    }
    pub fn with_settings(source: SharedPlanetFrame, settings: Settings) -> Self {
        Self {
            source,
            settings,
            active: None,
            profiling: false,
        }
    }
    pub fn renderer(&self) -> Option<&PlanetRenderer> {
        self.active.as_ref()
    }
    pub fn renderer_mut(&mut self) -> Option<&mut PlanetRenderer> {
        self.active.as_mut()
    }
    pub fn stats(&self) -> Option<PlanetStats> {
        self.active.as_ref().map(PlanetRenderer::stats)
    }
    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiling = enabled;
        if let Some(r) = &mut self.active {
            r.set_profiling(enabled);
        }
    }
    /// A host viewport should keep rendering until residency settles.
    pub fn needs_frame(&self) -> bool {
        let has_source = self.source.try_lock().map_or(true, |s| s.is_some());
        has_source && self.active.as_ref().is_some_and(|r| !r.settled())
    }
}

const READS: &[&str] = &["gbuffer", "gbuffer_lightmap_uv", "gbuffer_sss", "gbuffer_extra", "gbuffer_velocity"];
const WRITES: &[&str] = &[
    "gbuffer",
    "gbuffer_lightmap_uv",
    "gbuffer_sss",
    "gbuffer_extra",
    "gbuffer_velocity",
    "directional_visibility",
];

impl RenderPass for PlanetPass {
    fn name(&self) -> &'static str {
        "VoxelPlanet"
    }
    fn inherit_persistent_state(&mut self, previous: &mut dyn RenderPass) -> bool {
        let Some(previous) = previous.as_any_mut().downcast_mut::<Self>() else {
            return false;
        };
        if !Arc::ptr_eq(&self.source, &previous.source) {
            return false;
        }
        self.active = previous.active.take();
        self.active.is_some()
    }
    fn reads(&self) -> &'static [&'static str] {
        READS
    }
    fn writes(&self) -> &'static [&'static str] {
        WRITES
    }
    fn declare_resources(&self, builder: &mut helio_core::graph::ResourceBuilder) {
        for name in READS {
            builder.read(name);
        }
    }
    fn render_pass_descriptor<'a>(
        &'a self,
        _: &'a wgpu::TextureView,
        _: &'a wgpu::TextureView,
        _: &'a helio_core::ResourceRegistry<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }
    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let frame = self
            .source
            .lock()
            .map_err(|_| helio_core::Error::InvalidPassConfig("planet frame source poisoned".into()))?
            .clone();
        match frame {
            None => self.active = None,
            Some(frame) => {
                let same = self.active.as_ref().is_some_and(|r| {
                    Arc::ptr_eq(r.planet(), &frame.planet)
                        || (r.planet().recipe() == frame.planet.recipe())
                });
                if !same {
                    let mut renderer = PlanetRenderer::new(ctx.device, ctx.queue, frame.planet.clone(), self.settings, [ctx.width, ctx.height]);
                    renderer.set_profiling(self.profiling);
                    self.active = Some(renderer);
                } else if let Some(r) = &mut self.active {
                    // Same recipe: adopt the newer edit state without rebuilding.
                    r.planet = frame.planet.clone();
                }
            }
        }
        Ok(())
    }
    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let Some(renderer) = &mut self.active else { return Ok(()) };
        let Some(frame) = self.source.lock().ok().and_then(|f| f.clone()) else { return Ok(()) };
        let missing = |name: &str| helio_core::Error::ResourceNotFound(name.into());
        let g = ctx
            .registry
            .get::<helio_core::ViewGroup<'_, 4>>(helio_core::ResourceKey::new("gbuffer"))
            .ok_or_else(|| missing("gbuffer"))?;
        macro_rules! view {
            ($name:literal) => {
                ctx.registry
                    .get::<&wgpu::TextureView>(helio_core::ResourceKey::new($name))
                    .ok_or_else(|| missing($name))?
            };
        }
        let targets = [
            g.views[0],
            g.views[1],
            g.views[2],
            g.views[3],
            view!("gbuffer_lightmap_uv"),
            view!("gbuffer_sss"),
            view!("gbuffer_extra"),
            view!("gbuffer_velocity"),
        ];
        let encoder = unsafe { &mut *ctx.encoder_ptr };
        renderer.encode(
            encoder,
            ctx.camera,
            ctx.camera_data,
            &frame,
            [ctx.width, ctx.height],
            targets,
            ctx.depth,
            ctx.frame_num,
        );
        Ok(())
    }
    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        if let Some(view) = self.active.as_ref().and_then(PlanetRenderer::sun_view) {
            frame.write_texture_binding("directional_visibility", view, self.name());
        }
    }
}
