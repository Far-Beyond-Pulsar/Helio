//! Helio integration: GPU residency, exact traversal and GBuffer output.
use crate::grid::Cell;
use crate::planet::Planet;
use crate::residency::{Capacity, FrameWork, Residency, NONE};
use crate::terrain::TerrainProgram;
use bytemuck::{Pod, Zeroable};
use glam::{DVec3, IVec4, Mat4, Vec3, Vec4};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
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
/// position. The pass traces relative to this precise eye internally; the
/// host's shared camera and scene geometry can remain in world space.
#[derive(Clone)]
pub struct PlanetFrame {
    pub eye: DVec3,
    pub planet: Arc<Planet>,
    /// Direction towards the sun (planet-centred frame).
    pub sun: Vec3,
    pub shadows: bool,
}

pub type SharedPlanetFrame = Arc<Mutex<Option<PlanetFrame>>>;

/// Art controls, independent of occupancy, terrain recipes and edit journals.
/// Palette RGB is sRGB in [0,1]; W is perceptual roughness.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct TerrainAppearance {
    pub palette: [[f32; 4]; 16],
    /// Dry, meadow and lush grass colours (sRGB).
    pub grass: [[f32; 4]; 3],
    /// Grass patch contrast, voxel pigment contrast, edge occlusion strength.
    pub detail: [f32; 4],
}

impl Default for TerrainAppearance {
    fn default() -> Self {
        let colours = [
            [200,0,200], [91,125,65], [120,87,61], [133,139,142],
            [203,188,151], [217,228,236], [28,72,92], [116,111,102],
            [185,142,104], [82,88,95], [101,75,53], [59,102,52],
            [155,113,89], [148,77,63], [158,119,79], [121,126,130],
        ];
        let roughness = [0.9,0.94,0.96,0.84,0.93,0.78,0.35,0.9,0.88,0.82,0.97,0.94,0.92,0.86,0.86,0.85];
        Self {
            palette: std::array::from_fn(|i| [colours[i][0] as f32 / 255.0, colours[i][1] as f32 / 255.0, colours[i][2] as f32 / 255.0, roughness[i]]),
            grass: [[137.0/255.0,143.0/255.0,91.0/255.0,0.0], [91.0/255.0,125.0/255.0,65.0/255.0,0.0], [55.0/255.0,99.0/255.0,58.0/255.0,0.0]],
            detail: [0.75,0.18,0.08,0.0],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// Level cells project to this many pixels where their range starts.
    pub lod_pixels: f32,
    /// Relative width of the stochastic level transition.
    pub lod_dither: f32,
    /// Column jobs per frame.
    pub job_budget: usize,
    /// End rising eye rays at the directional sky bound.
    pub horizon: bool,
    /// Skip hash lookups of columns the summary blocks prove absent.
    pub residency_hints: bool,
    /// Reuse a coarse climate height only when Landform bounds prove that
    /// every canonical height gives the same material. Disable for audits.
    pub climate_height_reuse: bool,
    /// Preserve sub-cell radial relief in unedited coarse columns.
    /// Set before generating columns; resident columns retain their format.
    pub coarse_relief: bool,
    /// Reconstruct distant slope lighting from existing raw climate samples,
    /// blending by pixel footprint independently of the current clipmap level.
    pub far_relief: bool,
    /// Diagnostics: skip residency planning (no jobs, windows or evictions)
    /// so several renders see identical GPU state.
    pub freeze_residency: bool,
    /// Diagnostics: fixed frame index for the sunlight representative pattern.
    pub frame_override: Option<u32>,
    pub capacity: Capacity,
    pub appearance: TerrainAppearance,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            lod_pixels: 1.0,
            lod_dither: std::env::var("HELIO_VOXEL_LOD_DITHER").ok().and_then(|v| v.parse().ok()).unwrap_or(0.25),
            job_budget: 12_288,
            horizon: std::env::var_os("HELIO_VOXEL_NO_HORIZON").is_none(),
            residency_hints: true,
            climate_height_reuse: true,
            coarse_relief: std::env::var("HELIO_VOXEL_COARSE_RELIEF").ok().is_none_or(|v| v != "0"),
            far_relief: std::env::var("HELIO_VOXEL_FAR_RELIEF").ok().is_none_or(|v| v != "0"),
            freeze_residency: false,
            frame_override: None,
            capacity: Capacity::default(),
            appearance: TerrainAppearance::default(),
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
    /// x: tier-1 summary blocks prove column absence (`blocks_exact`).
    hints: [u32; 4],
    palette: [[f32; 4]; 16],
    grass: [[f32; 4]; 3],
    detail: [f32; 4],
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
    /// Completed generation results by status (ok, band, scratch, pool, skipped).
    pub jobs_by_status: [usize; 5],
    pub overflow_columns: usize,
    pub free_pages: i32,
    pub pool_pages: u32,
    /// Available runs in each power-of-two allocation class.
    pub free_runs_by_class: [u32; 10],
    pub free_pool_units: u64,
    pub reclaimable_pages: u32,
    pub recycled_pages: u32,
    pub active_levels: u32,
    pub finest_level: u32,
    pub plan_cpu_ms: f64,
    pub upload_cpu_ms: f64,
    pub encode_cpu_ms: f64,
    pub window_rebuild_ms: f64,
    pub lod0_distance: f64,
    pub logical_bytes: u64,
    /// Measured GPU generation cost per column job (microseconds) and this
    /// frame's job budget.
    pub us_per_job: f64,
    pub job_budget: usize,
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

/// `World` of world.wgsl: the grid mapping and coarse-level bounds.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct WorldGpu {
    /// reference cells, layer thickness (mm), grid cells, level offset.
    grid: [i32; 4],
    /// domain scale (Q24), pad.
    scale: [u32; 4],
    bounds: [[i32; 4]; 6],
}

impl WorldGpu {
    fn new(planet: &Planet) -> Self {
        let g = planet.grid();
        let m = planet.field().bound_margins();
        Self {
            grid: [g.reference_cells(), g.layer_mm() as i32, g.cells(), g.level_offset() as i32],
            scale: [g.domain_scale(), 0, 0, 0],
            bounds: std::array::from_fn(|i| std::array::from_fn(|j| m[i * 4 + j])),
        }
    }
}

/// Timestamps written from command encoders (what the stage profiler uses).
fn timestamps_supported(device: &wgpu::Device) -> bool {
    device.features().contains(wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS)
}

/// Terrain constants padded to a whole uniform (16-byte multiple).
fn terrain_bytes(program: &TerrainProgram) -> Vec<u8> {
    let mut bytes = program.constants.clone();
    bytes.resize(bytes.len().max(16).next_multiple_of(16), 0);
    bytes
}

/// Shader source: the noise library, world helpers and the terrain program,
/// then the engine parts.
fn source(access: &str, parts: &[&str], plane: bool, program: &TerrainProgram) -> String {
    let mut s = String::from(include_str!("../shaders/noise.wgsl"));
    s.push_str(include_str!("../shaders/world.wgsl"));
    s.push_str(&program.wgsl);
    // Generation updates the summaries atomically; traversal reads plain values.
    // Traversal reads a summary block entry as one vector load.
    let generation = parts.iter().any(|p| p.contains("fn level_suffix"));
    let (level_top, block_entry) = if generation { ("atomic<i32>", "atomic<i32>") } else { ("i32", "vec4<i32>") };
    s.push_str(
        &include_str!("../shaders/common.wgsl")
            .replace("ACCESS", access)
            .replace("LEVEL_TOP", level_top)
            .replace("BLOCK_ENTRY", block_entry)
            .replace("SHAPE_ID", if plane { "1u" } else { "0u" }),
    );
    if program.key == "helio.landform/1" {
        s.push_str(include_str!("../shaders/landform_climate.wgsl"));
    } else {
        // Custom generators keep their full query; only Landform's material
        // classification and symmetric bounds justify the shortcut.
        s.push_str("fn climate_height_reusable(top: i32, level: u32) -> bool { return false; }\n");
    }
    for part in parts {
        s.push_str(&part.replace("ACCESS", access));
    }
    s
}

struct Pipelines {
    plane: bool,
    program: String,
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
    recycle_layout: wgpu::BindGroupLayout,
    reclaim_pages: wgpu::ComputePipeline,
    compact_runs: wgpu::ComputePipeline,
    finish_recycle: wgpu::ComputePipeline,
    publish: wgpu::ComputePipeline,
    level_suffix: wgpu::ComputePipeline,
    primary: wgpu::ComputePipeline,
    horizon_clear: wgpu::ComputePipeline,
    horizon_blocks: wgpu::ComputePipeline,
    horizon_suffix: wgpu::ComputePipeline,
    shade: wgpu::ComputePipeline,
    shade_relief: OnceLock<wgpu::ComputePipeline>,
    shade_module: wgpu::ShaderModule,
    shade_layout: wgpu::PipelineLayout,
    climate: wgpu::ComputePipeline,
    sunlight: wgpu::ComputePipeline,
    gbuffer: wgpu::RenderPipeline,
}

impl Pipelines {
    fn shade_for(&self, device: &wgpu::Device, relief: bool) -> &wgpu::ComputePipeline {
        if !relief {
            return &self.shade;
        }
        self.shade_relief.get_or_init(|| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("planet shade relief"),
                layout: Some(&self.shade_layout),
                module: &self.shade_module,
                entry_point: Some("shade"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("FAR_RELIEF", 1.0)],
                    ..Default::default()
                },
                cache: None,
            })
        })
    }

    /// Whether these pipelines serve a world of this shape and program.
    fn serve(&self, plane: bool, program: &TerrainProgram) -> bool {
        self.plane == plane && self.program == program.key
    }

    fn new(device: &wgpu::Device, plane: bool, program: &TerrainProgram) -> Self {
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
            uniform(16),
            storage(17, false),
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
            uniform(16),
            storage(17, false),
            storage(18, false),
            storage(19, true),
            storage(20, false),
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
        // Composed from several files plus the terrain program in Rust, so it
        // goes through `module` as plain text (not hot reloadable).
        let module = |label: &str, src: String| helio_core::shader::module(device, label, &src);
        let gen_module = module("planet generation", source("read_write", &[include_str!("../shaders/generate.wgsl")], plane, program));
        let trace_src = [
            include_str!("../shaders/view.wgsl"),
            include_str!("../shaders/horizon.wgsl"),
            include_str!("../shaders/trace.wgsl"),
            include_str!("../shaders/surface.wgsl"),
        ];
        let trace_module = module("planet trace", source("read_write", &trace_src, plane, program));
        let render_module = module(
            "planet gbuffer",
            source("read", &[include_str!("../shaders/view.wgsl"), include_str!("../shaders/gbuffer.wgsl")], plane, program),
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
        let recycle_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("planet allocator recycling"),
            entries: &[storage(0, false), storage(1, false), storage(2, true),
                storage(3, false), storage(4, false), storage(5, false)],
        });
        let recycle_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("planet allocator recycling"),
            bind_group_layouts: &[Some(&recycle_layout)], immediate_size: 0,
        });
        let recycle_module = module("planet allocator recycling", include_str!("../shaders/allocator_recycle.wgsl").to_owned());
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
            plane,
            program: program.key.to_string(),
            patch: compute(&gen_pl, &gen_module, "patch_table"),
            patch_blocks: compute(&gen_pl, &gen_module, "patch_blocks"),
            evict: compute(&gen_pl, &gen_module, "evict"),
            generate: compute(&gen_pl, &gen_module, "generate"),
            count: compute(&gen_pl, &gen_module, "count"),
            refill: compute(&gen_pl, &gen_module, "refill"),
            allocate: compute(&gen_pl, &gen_module, "allocate"),
            fixup: compute(&gen_pl, &gen_module, "fixup"),
            reclaim_pages: compute(&recycle_pl, &recycle_module, "reclaim_pages"),
            compact_runs: compute(&recycle_pl, &recycle_module, "compact_runs"),
            finish_recycle: compute(&recycle_pl, &recycle_module, "finish_recycle"),
            recycle_layout,
            publish: compute(&gen_pl, &gen_module, "publish"),
            level_suffix: compute(&gen_pl, &gen_module, "level_suffix"),
            primary: compute(&trace_pl, &trace_module, "primary"),
            horizon_clear: compute(&trace_pl, &trace_module, "horizon_clear"),
            horizon_blocks: compute(&trace_pl, &trace_module, "horizon_blocks"),
            horizon_suffix: compute(&trace_pl, &trace_module, "horizon_suffix"),
            shade: compute(&trace_pl, &trace_module, "shade"),
            climate: compute(&trace_pl, &trace_module, "climate"),
            sunlight: compute(&trace_pl, &trace_module, "sunlight"),
            shade_relief: OnceLock::new(),
            shade_module: trace_module,
            shade_layout: trace_pl,
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
    world: wgpu::Buffer,
    terrain: wgpu::Buffer,
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
    free_runs_spare: Option<wgpu::Buffer>,
    page_meta: wgpu::Buffer,
    recycle_counts: wgpu::Buffer,
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
    fn new(device: &wgpu::Device, cap: &Capacity, planet: &Planet) -> Self {
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
        let records = make("planet records", u64::from(cap.records) * 32, st | wgpu::BufferUsages::COPY_SRC);
        let pool = make("planet pool", u64::from(cap.pool_units) * 64, st | wgpu::BufferUsages::COPY_SRC);
        let edit_refs = make("planet edit refs", u64::from(cap.edit_words) * 4, st);
        let jobs = make("planet jobs", u64::from(cap.max_jobs) * 32, st);
        let job_out = make("planet job results", u64::from(cap.max_jobs) * JOB_OUT_BYTES, st | wgpu::BufferUsages::COPY_SRC);
        let scratch = make("planet scratch", u64::from(cap.scratch_units) * 64, st);
        let free_runs = make("planet free runs", u64::from(cap.pool_units) * 8, st);
        // Unassigned zeroed pages have no free runs; refill initializes both
        // fields before a page is visible to allocation.
        let page_meta = make("planet allocation pages", u64::from(pages) * 8, st);
        let recycle_counts = make("planet recycled class counts", 64, st);
        let block_state = make(
            "planet block summaries",
            u64::from(crate::residency::block_region()) * 6 * 24 * 16,
            st | wgpu::BufferUsages::COPY_SRC,
        );
        let evictions = make("planet evictions", (u64::from(cap.max_evictions) * 3 + u64::from(cap.max_jobs) * 2) * 4, st);
        let horizon_acc = make("planet horizon accumulation", u64::from((HORIZON_SECTORS + HORIZON_GROUPS) * HORIZON_BUCKETS) * 4, st);
        let horizon = make(
            "planet horizon bound",
            u64::from((HORIZON_SECTORS + 1) * HORIZON_BUCKETS) * 4,
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
            usage: st | wgpu::BufferUsages::COPY_SRC,
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
        let world = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("planet world"),
            contents: bytemuck::bytes_of(&WorldGpu::new(planet)),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let terrain = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("planet terrain constants"),
            contents: &terrain_bytes(&planet.field().program()),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        Self {
            frame,
            world,
            terrain,
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
            free_runs_spare: None,
            page_meta,
            recycle_counts,
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
    climate: wgpu::Buffer,
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
        let climate = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planet climate height"),
            size: pixels * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        Self {
            size,
            hits,
            surfaces,
            climate,
            sun,
            sun_view,
        }
    }
}

/// Active GPU state for one planet.
pub struct PlanetRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Arc<Pipelines>,
    buffers: Buffers,
    screen: Screen,
    residency: Residency,
    planet: Arc<Planet>,
    settings: Settings,
    gen_group: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,
    camera_group: wgpu::BindGroup,
    /// Last local projection and precise eye, for motion in the shared GBuffer.
    camera_history: Option<(u64, u32, DVec3, Mat4)>,
    readbacks: Vec<Readback>,
    frame_index: u32,
    stats: PlanetStats,
    sun_active: bool,
    profiler: Option<helio_core::profiling::GpuProfiler>,
    initial_complete: bool,
    /// Measured GPU generation cost per column job (EMA) and last job count.
    ms_per_job: f64,
    last_jobs: usize,
    /// Jobs issued per recent frame number, and the frame whose timestamps
    /// last updated `ms_per_job`.
    frame_jobs: std::collections::VecDeque<(u64, usize)>,
    costed_frame: Option<u64>,
    last_eye: Option<(DVec3, std::time::Instant)>,
    last_frame_num: u64,
    recycle_pending: bool,
}

impl PlanetRenderer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, planet: Arc<Planet>, settings: Settings, size: [u32; 2]) -> Self {
        Self::replacing(None, device, queue, planet, settings, size)
    }

    /// A renderer for `planet` that takes over `previous`'s compiled
    /// pipelines when they serve the same shape and terrain program, so a
    /// world rebuilt with new generator settings compiles no shaders.
    pub fn replacing(
        previous: Option<&PlanetRenderer>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        planet: Arc<Planet>,
        settings: Settings,
        size: [u32; 2],
    ) -> Self {
        let plane = planet.grid().is_plane();
        let program = planet.field().program();
        let pipelines = previous
            .map(|r| &r.pipelines)
            .filter(|p| p.serve(plane, &program))
            .cloned()
            .unwrap_or_else(|| Arc::new(Pipelines::new(device, plane, &program)));
        let buffers = Buffers::new(device, &settings.capacity, &planet);
        let gen_group = Self::gen_group(device, &pipelines, &buffers);
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planet local camera"),
            size: std::mem::size_of::<helio_core::GpuCameraUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet local camera"),
            layout: &pipelines.camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });
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
            camera_buffer,
            camera_group,
            camera_history: None,
            readbacks,
            frame_index: 0,
            stats: PlanetStats::default(),
            sun_active: false,
            // Stage timestamps are not only diagnostics: the generation
            // budget divides a time target by the measured cost per column.
            // Without them it stays at the conservative default (the editor
            // streamed 3x slower than the harness, which enabled profiling).
            profiler: timestamps_supported(device).then(|| helio_core::profiling::GpuProfiler::new(device, queue)),
            initial_complete: false,
            ms_per_job: 0.0013,
            frame_jobs: std::collections::VecDeque::new(),
            costed_frame: None,
            last_jobs: 0,
            last_eye: None,
            last_frame_num: 0,
            recycle_pending: false,
            pipelines,
            buffers,
        }
    }

    fn gen_group(device: &wgpu::Device, p: &Pipelines, b: &Buffers) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = [
            &b.frame, &b.world, &b.table, &b.records, &b.pool, &b.brushes, &b.edit_refs, &b.jobs, &b.job_out,
            &b.scratch, &b.alloc, &b.free_runs, &b.free_pages, &b.evictions, &b.level_tops, &b.block_state, &b.terrain, &b.page_meta,
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
    /// GPU column hash table and the CPU table it must equal (diagnostics).
    pub fn column_table(&self) -> (&wgpu::Buffer, &[u32]) {
        (&self.buffers.table, self.residency.table())
    }
    /// Column records, brick pool and summary blocks (diagnostics).
    pub fn residency_buffers(&self) -> [&wgpu::Buffer; 3] {
        [&self.buffers.records, &self.buffers.pool, &self.buffers.block_state]
    }
    /// Directional sky bound table (diagnostics): `[bucket][sector]` lowest
    /// clearing elevations (radians), then one all-sector row.
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
    /// Stage timestamps are always recorded where the device supports them
    /// (they size the generation budget); this only creates the profiler if
    /// it is missing. Disabling is a no-op.
    pub fn set_profiling(&mut self, enabled: bool) {
        if enabled && self.profiler.is_none() {
            self.profiler = Some(helio_core::profiling::GpuProfiler::new(&self.device, &self.queue));
        }
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
    /// Most recently completed stage timings with the frame number they
    /// belong to (non-blocking; results trail submission by a frame or two).
    pub fn stage_timings_deferred(&mut self) -> Option<(u64, Vec<(&'static str, f64)>)> {
        let p = self.profiler.as_mut()?;
        let timings = p.read_timestamps_deferred().iter().map(|t| (t.name, t.duration_ns as f64 / 1.0e6)).collect();
        p.last_completed_frame().map(|frame| (frame, timings))
    }
    /// Frame number of the last encoded frame (as passed to `encode`).
    pub fn frame_number(&self) -> u64 {
        self.last_frame_num
    }
    /// Residency has issued and completed every window column.
    pub fn settled(&self) -> bool {
        self.residency.idle() && self.readbacks.iter().all(|r| r.stage == 0)
    }

    fn frame_uniform(&self, eye: DVec3, size: [u32; 2], lod0: f64, jobs: u32, evictions: u32, sun: Vec3, shadows: bool) -> FrameGpu {
        let planet = &self.planet;
        let grid = planet.grid();
        let mut frame = FrameGpu::default();
        let clean = |v: f32| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
        // Public appearance values stay sRGB; convert the three colour
        // channels once on upload instead of evaluating pow per shaded pixel.
        let linear = |row: [f32; 4]| [clean(row[0]).powf(2.2), clean(row[1]).powf(2.2), clean(row[2]).powf(2.2), clean(row[3])];
        frame.palette = self.settings.appearance.palette.map(linear);
        frame.grass = self.settings.appearance.grass.map(linear);
        frame.detail = self.settings.appearance.detail.map(clean);
        // Material-equivalent quantized tops are not equivalent derivatives.
        // The relief prototype needs raw heights at the existing 2x2 anchors.
        frame.hints[1] = u32::from(self.settings.climate_height_reuse && !self.settings.far_relief);
        frame.hints[2] = u32::from(self.settings.far_relief);
        // Summary tops from an older journal cannot prune live edits.
        frame.hints[3] = (if self.residency.pending_edits() { 4 } else { 0 })
            | (if self.settings.coarse_relief { 8 } else { 0 });
        for face in 0..6u8 {
            // A plane has one face; the others keep default frames.
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
                frame.neighbours[face as usize][e] = if grid.is_plane() { u32::from(face) } else { u32::from(grid.neighbour(*cell, axis, step).face) };
            }
        }
        let rho = grid.radial(eye);
        let dir = grid.up(eye);
        let s = grid.voxel_size();
        let layer = (rho - grid.radius()) / s;
        let k = layer.floor();
        frame.eye = [dir.x as f32, dir.y as f32, dir.z as f32, rho as f32];
        frame.layer = [(layer - k) as f32, s as f32, grid.delta() as f32, (planet.outer_radius() - rho) as f32];
        frame.layer_i = [k.clamp(i32::MIN as f64, i32::MAX as f64) as i32, grid.cells(), grid.levels() as i32, i32::from(if grid.is_plane() { crate::grid::PLANE_FACE } else { crate::grid::face_of(eye) })];
        // Directional sky bound cut depth below the eye radius: the bound
        // covers points above it (see `horizon.wgsl`), and each level's ring
        // is where no such point can use the level. Any depth is exact. A
        // shallow cut keeps coarse blocks near the eye out of the table (their
        // rounded-up tops would block low rays on the ground); in the air it
        // deepens with the clearance, so rays aimed down start near the
        // terrain instead of crossing every level's ring through air.
        const SKY_CUT_M: f64 = 100.0;
        let cut = SKY_CUT_M.max(0.75 * planet.air_clearance(eye));
        // Farthest ray distance: past the far side of a planet, or across a plane.
        let far = if grid.is_plane() { f64::from(grid.cells()) * s * 2.0 + rho.abs() } else { rho + grid.radius() * 3.0 };
        frame.lod = [lod0 as f32, self.settings.lod_dither, -cut as f32, far as f32];
        let (_, rings) = self.sky_rings(eye, lod0, cut);
        for (level, phi) in rings.iter().enumerate().take(32) {
            frame.ring[level / 4][level % 4] = *phi as f32;
        }
        // Bit 1: directional sky bound; bit 2: sky-bound fail-safe disabled
        // (HELIO_VOXEL_NO_FAILSAFE, for A/B timing).
        let flags = (u32::from(self.settings.horizon) << 1) | (u32::from(std::env::var_os("HELIO_VOXEL_NO_FAILSAFE").is_some()) << 2);
        let index = self.settings.frame_override.unwrap_or(self.frame_index % 1024);
        frame.screen = [size[0] as f32, size[1] as f32, index as f32, flags as f32];
        let sun = sun.normalize_or_zero();
        frame.sun = [sun.x, sun.y, sun.z, if shadows { 1.0 } else { 0.0 }];
        frame.counts = [jobs, evictions, (1u32 << self.settings.capacity.table_bits) - 1, self.settings.capacity.pool_units];
        frame
    }

    /// Per level, the sky-bound fallback distance (ray distance at which the
    /// level may fall back to coarser data) and ring (angular distance, or
    /// metres on a plane, where the level's blocks start to serve rays), for
    /// an eye and cut depth. `frame_uniform` uses these; diagnostics may call
    /// it after a frame (the residency state is the frame's until the next
    /// plan).
    pub fn sky_rings(&self, eye: DVec3, lod0: f64, cut: f64) -> (Vec<f64>, Vec<f64>) {
        let planet = &self.planet;
        let grid = planet.grid();
        let rho = grid.radial(eye);
        // Nearest ray distance at which each level may fall back to coarser
        // data: chord bound for points past its fallback angle at radius
        // >= the cut radius.
        let r_lo = rho - cut;
        let fallback: Vec<f64> = self
            .residency
            .fallback_distances(eye)
            .iter()
            .map(|distance| {
                if grid.is_plane() {
                    // A point that far away horizontally is at least that far.
                    return *distance;
                }
                let a = distance / grid.radius();
                if a >= std::f64::consts::PI { f64::INFINITY } else { 2.0 * (rho * r_lo).sqrt() * (a * 0.5).sin() }
            })
            .collect();
        let rings = if grid.is_plane() {
            // Points in [eye - cut, outer] differ in height by at most dz.
            let dz = cut.max(planet.outer_radius() - rho);
            plane_sky_rings(lod0, f64::from(self.settings.lod_dither), dz, &fallback)
        } else {
            sky_rings(lod0, f64::from(self.settings.lod_dither), rho, r_lo, planet.outer_radius(), &fallback)
        };
        (fallback, rings)
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
        // Evictions followed by table patches (slot, value) pairs. The GPU
        // applies patches in parallel, and backward-shift deletion writes a
        // slot several times in a frame: each slot is sent once, with its
        // final value (an earlier value winning left an empty slot inside a
        // probe run, hiding every column past it).
        let mut words: Vec<u32> = work.evictions.clone();
        let mut patches = 0;
        if !work.full_table {
            let table = self.residency.table();
            let mut sent = rustc_hash::FxHashSet::default();
            for (slot, _) in &work.table_writes {
                if sent.insert(*slot) {
                    words.push(*slot);
                    words.push(table[*slot as usize]);
                    patches += 1;
                }
            }
        }
        // A slot released and re-acquired in one frame must end in its last
        // state; the GPU patches entries in parallel.
        let mut last = rustc_hash::FxHashMap::default();
        for (index, (slot, _, _)) in work.block_inits.iter().enumerate() {
            last.insert(*slot, index);
        }
        let block_inits: Vec<_> = work.block_inits.iter().enumerate().filter(|(i, (slot, _, _))| last[slot] == *i).map(|(_, b)| *b).collect();
        for (slot, bi, bj) in &block_inits {
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
        (patches, block_inits.len() as u32)
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
        let mut completed = Vec::new();
        for r in &mut self.readbacks {
            if r.stage == 2 && r.state.load(Ordering::Acquire) {
                {
                    let data = r.buffer.slice(..).get_mapped_range().unwrap();
                    for (index, key) in r.keys.iter().enumerate() {
                        let at = index * JOB_OUT_BYTES as usize;
                        let status = u32::from_le_bytes(data[at..at + 4].try_into().unwrap());
                        completed.push((*key, status));
                    }
                    let probe = u64::from(self.settings.capacity.max_jobs) * JOB_OUT_BYTES;
                    let at = probe as usize;
                    self.stats.free_pages = i32::from_le_bytes(data[at + 120..at + 124].try_into().unwrap());
                    self.stats.reclaimable_pages = i32::from_le_bytes(data[at + 104..at + 108].try_into().unwrap()).max(0) as u32;
                    self.stats.recycled_pages = i32::from_le_bytes(data[at + 108..at + 112].try_into().unwrap()).max(0) as u32;
                    for class in 0..10 {
                        self.stats.free_runs_by_class[class] = i32::from_le_bytes(
                            data[at + class * 4..at + class * 4 + 4].try_into().unwrap(),
                        ).max(0) as u32;
                    }
                    self.stats.free_pool_units = self.stats.free_pages.max(0) as u64 * 512
                        + self.stats.free_runs_by_class.iter().enumerate()
                            .map(|(class, count)| u64::from(*count) << class).sum::<u64>();
                }
                r.buffer.unmap();
                r.stage = 0;
                r.state.store(false, Ordering::Release);
            }
        }
        self.stats.failed_jobs += completed.iter().filter(|(_, s)| *s != 0 && *s != 1).count();
        for (_, status) in &completed {
            if let Some(count) = self.stats.jobs_by_status.get_mut(*status as usize) { *count += 1; }
        }
        if completed.iter().any(|(_, status)| *status == 3) && self.stats.reclaimable_pages > 0 {
            self.recycle_pending = true;
        }
        self.stats.overflow_columns += completed.iter().filter(|(_, s)| *s == 1).count();
        self.residency.complete_jobs(completed);
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
    /// The shared camera supplies orientation, projection and jitter. Its
    /// translation is replaced only in this pass's private camera buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        camera_data: &helio_core::GpuCameraUniforms,
        frame: &PlanetFrame,
        size: [u32; 2],
        gbuffer: [&wgpu::TextureView; 8],
        depth: &wgpu::TextureView,
        frame_num: u64,
    ) {
        // An edit-only publication keeps the recipe/pipelines but changes
        // the authoritative journal. Direct renderer users need the same
        // synchronization as PlanetPass's mailbox path.
        if !Arc::ptr_eq(&self.planet, &frame.planet) {
            assert_eq!(self.planet.recipe(), frame.planet.recipe(), "recreate PlanetRenderer after a recipe change");
            self.planet = frame.planet.clone();
        }
        if let Some(p) = &mut self.profiler {
            // Timestamps arrive frames late and the same sample is returned
            // until a newer one completes: each sample is used once, with the
            // job count of the frame it measured. (Dividing by the last
            // frame's jobs overestimated the cost 2-7x, most in the editor.)
            let residency: f64 = p
                .read_timestamps_deferred()
                .iter()
                .filter(|t| t.name == "planet_residency")
                .map(|t| t.duration_ns as f64 / 1.0e6)
                .sum();
            let completed = p.last_completed_frame();
            if completed.is_some() && completed != self.costed_frame {
                self.costed_frame = completed;
                let jobs = self.frame_jobs.iter().find(|(f, _)| Some(*f) == completed).map_or(0, |(_, j)| *j);
                if jobs >= 256 && residency > 0.0 {
                    let sample = residency / jobs as f64;
                    self.ms_per_job = self.ms_per_job * 0.7 + sample * 0.3;
                }
            }
        }
        self.frame_index = self.frame_index.wrapping_add(1);
        self.last_frame_num = frame_num;
        if self.screen.size != size {
            self.screen = Screen::new(&self.device, size);
        }
        self.poll_readbacks();
        let tan_half = 1.0 / f64::from(camera_data.proj[5]).abs().max(1e-6);
        let lod0 = Residency::lod_distance(self.planet.grid(), tan_half, size[1], f64::from(self.settings.lod_pixels));
        let started = std::time::Instant::now();
        // Generation budget: small while the view moves (frame pacing), large
        // when it is still (fast convergence), from the measured job cost.
        let now = std::time::Instant::now();
        let moving = self.last_eye.is_none_or(|(eye, _)| eye.distance(frame.eye) > 0.01);
        let ground_clearance = self.planet.ground_height(frame.eye);
        let predicted = self.last_eye.and_then(|(eye, when)| {
            let dt = now.duration_since(when).as_secs_f64();
            if !moving || dt > 0.25 { return None; }
            crate::windows::motion_forecast(self.planet.grid(), frame.eye, eye, dt,
                lod0, ground_clearance)
        });
        let coverage = self.last_eye.and_then(|(eye, when)| {
            crate::windows::window_forecast(frame.eye, eye, now.duration_since(when).as_secs_f64(), ground_clearance)
        });
        self.residency.set_prefetch_eye(coverage);
        self.residency.set_priority_eye(predicted);
        let forward = DVec3::new(f64::from(camera_data.forward_far[0]),
            f64::from(camera_data.forward_far[1]), f64::from(camera_data.forward_far[2]));
        let ground_radial = self.planet.grid().radial(frame.eye) - ground_clearance;
        let camera_far = f64::from(camera_data.forward_far[3]);
        let focus_far = if camera_far.is_finite() && camera_far > 0.0 { camera_far }
            else if self.planet.grid().is_plane() {
                f64::from(self.planet.grid().cells()) * self.planet.grid().voxel_size() * std::f64::consts::SQRT_2
            } else { self.planet.grid().radius() * 4.0 };
        let focus = crate::windows::visible_focus(self.planet.grid(), frame.eye, forward, ground_radial, focus_far);
        self.residency.set_view_focus(focus);
        self.last_eye = Some((frame.eye, now));
        // Spend additional generation time when visible detail is catching up,
        // rather than withholding it until the camera stops moving.
        let backlog = (self.residency.stats.pending_columns as f64 / 80_000.0).min(1.0);
        let target_ms = if moving { 1.5 + 1.5 * backlog } else { 6.0 };
        // CPU for applying window diffs and admitting columns: small while
        // moving (a big diff spreads over frames instead of freezing one),
        // growing to 4 ms with the backlog (a new region streams in ~2x
        // faster; admission costs ~0.3 us per column, diffs about as much).
        let cpu_ms = if moving { 1.5 + 2.5 * backlog } else { 4.0 };
        self.residency.set_cpu_budget(Some(std::time::Duration::from_secs_f64(cpu_ms * 1.0e-3)));
        let desired_budget = ((target_ms / self.ms_per_job.max(1e-5)) as usize)
            .clamp(256, self.settings.job_budget.min(self.settings.capacity.max_jobs as usize));
        // Every publication needs completion feedback before its old journal
        // storage can be reused. Apply backpressure rather than issuing jobs
        // whose outcomes cannot be observed.
        let budget = if self.readbacks.iter().any(|r| r.stage == 0) { desired_budget } else { 0 };
        let work = if self.settings.freeze_residency {
            FrameWork::default()
        } else {
            self.residency.plan(&self.planet, frame.eye, lod0, budget)
        };
        self.last_jobs = work.jobs.len();
        self.stats.us_per_job = self.ms_per_job * 1000.0;
        self.stats.job_budget = budget;
        if self.frame_jobs.len() == 16 {
            self.frame_jobs.pop_front();
        }
        self.frame_jobs.push_back((frame_num, work.jobs.len()));
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
        uniform.hints[0] = u32::from(self.settings.residency_hints && self.residency.blocks_exact());
        self.queue.write_buffer(&self.buffers.frame, 0, bytemuck::bytes_of(&uniform));
        // All terrain rays (including sunlight rays reconstructed from mesh
        // depth) are offsets from PlanetFrame::eye. Using the scene's world
        // position here would apply that translation twice. A local view has
        // the same clip/depth coordinates as the shared world-space view.
        let mut local_camera = *camera_data;
        let mut view = Mat4::from_cols_array(&camera_data.view);
        view.w_axis = Vec4::W;
        let view_proj = Mat4::from_cols_array(&camera_data.proj) * view;
        let view_id = camera_data.jitter_frame[3].to_bits();
        let previous = match self.camera_history {
            Some((number, id, eye, projection))
                if number.wrapping_add(1) == frame_num && id == view_id =>
            {
                helio_core::temporal::rebase_previous_projection(projection, frame.eye - eye)
            }
            _ => view_proj,
        };
        local_camera.view = view.to_cols_array();
        local_camera.view_proj = view_proj.to_cols_array();
        local_camera.inv_view_proj = view_proj.inverse().to_cols_array();
        local_camera.position_near[..3].fill(0.0);
        local_camera.prev_view_proj = previous.to_cols_array();
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&local_camera));
        self.camera_history = Some((frame_num, view_id, frame.eye, view_proj));
        let trace_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet trace"),
            layout: &self.pipelines.trace_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.buffers.frame.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buffers.world.as_entire_binding() },
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
                wgpu::BindGroupEntry { binding: 16, resource: self.buffers.terrain.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 17, resource: self.buffers.horizon_acc.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 18, resource: self.buffers.horizon.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 19, resource: self.buffers.live_blocks.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 20, resource: self.screen.climate.as_entire_binding() },
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
        let camera_group = &self.camera_group;
        // Recycle only under observed pool pressure with wholly free pages.
        // The spare stack is allocated lazily; normal frames do no pool scan.
        let recycle_groups = if self.recycle_pending && self.stats.reclaimable_pages > 0 {
            self.recycle_pending = false;
            if self.buffers.free_runs_spare.is_none() {
                let bytes = self.buffers.free_runs.size();
                self.buffers.free_runs_spare = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("planet recycled free runs"), size: bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
                }));
                self.buffers.bytes += bytes;
            }
            let b = &self.buffers;
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("planet allocator recycling"), layout: &self.pipelines.recycle_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: b.alloc.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: b.page_meta.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: b.free_runs.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: b.free_runs_spare.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: b.free_pages.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: b.recycle_counts.as_entire_binding() },
                ],
            });
            encoder.clear_buffer(&self.buffers.recycle_counts, 0, None);
            std::mem::swap(&mut self.buffers.free_runs, self.buffers.free_runs_spare.as_mut().unwrap());
            let new_group = Self::gen_group(&self.device, &self.pipelines, &self.buffers);
            let old_group = std::mem::replace(&mut self.gen_group, new_group);
            Some((old_group, group))
        } else { None };
        if let Some(p) = &mut self.profiler {
            p.begin_pass(encoder, "planet_residency");
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, recycle_groups.as_ref().map_or(&self.gen_group, |groups| &groups.0), &[]);
            let wg = |n: u32| n.div_ceil(64);
            if evictions > 0 {
                Self::dispatch(&mut pass, &self.pipelines.evict, [wg(evictions), 1, 1]);
            }
            if let Some((_, group)) = &recycle_groups {
                pass.set_bind_group(0, group, &[]);
                Self::dispatch(&mut pass, &self.pipelines.reclaim_pages, [self.stats.pool_pages.div_ceil(128), 1, 1]);
                let groups = (self.settings.capacity.pool_units * 2).div_ceil(256);
                Self::dispatch(&mut pass, &self.pipelines.compact_runs, [groups.min(32768), groups.div_ceil(32768), 1]);
                Self::dispatch(&mut pass, &self.pipelines.finish_recycle, [1, 1, 1]);
                pass.set_bind_group(0, &self.gen_group, &[]);
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
            Self::dispatch(&mut pass, &self.pipelines.climate,
                [self.screen.size[0].div_ceil(16), self.screen.size[1].div_ceil(16), 1]);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &trace_group, &[]);
            pass.set_bind_group(1, camera_group, &[]);
            Self::dispatch(&mut pass, self.pipelines.shade_for(&self.device, self.settings.far_relief), groups);
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
        self.stats.logical_bytes = self.buffers.bytes
            + u64::from(size[0]) * u64::from(size[1]) * (32 + 16 + 8)
            + self.screen.climate.size();
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
/// serve. Level L serves distances from its dithered ring start, or through
/// fallback: a missing column moves a ray one level coarser, so L is reached
/// that way only where L - 1 is in use and may be missing (beyond both
/// L - 1's first use and its fallback distance).
fn sky_rings(lod0: f64, dither: f64, rho: f64, r_lo: f64, r_hi: f64, fallback: &[f64]) -> Vec<f64> {
    // Angular distance where a point at radius r is exactly t away (0 when a
    // point straight above or below the eye already is).
    let phi = |t: f64, r: f64| {
        let q = (t * t - (rho - r) * (rho - r)) / (4.0 * rho * r);
        if q <= 0.0 { 0.0 } else { 2.0 * q.sqrt().min(1.0).asin() }
    };
    rings(lod0, dither, fallback, |t| {
        let mut p = phi(t, r_lo).min(phi(t, r_hi));
        let r_star = (rho * rho - t * t).max(0.0).sqrt();
        if r_star > r_lo && r_star < r_hi {
            p = p.min(phi(t, r_star));
        }
        p
    })
}

/// [`sky_rings`] on a plane: horizontal distances (metres) for points whose
/// height differs from the eye's by at most `dz`.
fn plane_sky_rings(lod0: f64, dither: f64, dz: f64, fallback: &[f64]) -> Vec<f64> {
    rings(lod0, dither, fallback, |t| (t * t - dz * dz).max(0.0).sqrt())
}

/// Per level: the nearest ray distance `t` the level can serve (its dithered
/// ring start, or where the finer level may fall back), mapped by `ground`
/// to the ground distance within which no point can be that far away.
fn rings(lod0: f64, dither: f64, fallback: &[f64], ground: impl Fn(f64) -> f64) -> Vec<f64> {
    let start = |level: usize| {
        if level == 0 { 0.0 } else { lod0 * f64::from(1u32 << (level - 1).min(30)) / (1.0 + dither * 0.5) * 0.999 }
    };
    let mut finer: Option<(f64, f64)> = None;
    fallback
        .iter()
        .enumerate()
        .map(|(level, from)| {
            let t = match finer {
                None => 0.0,
                Some((used, missing)) => start(level).min(used.max(missing)),
            };
            finer = Some((t, *from));
            ground(t) * 0.999
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
    /// Changes art without recreating residency or altering the canonical world.
    /// Returns true when temporal colour history needs invalidating.
    pub fn set_appearance(&mut self, appearance: TerrainAppearance) -> bool {
        if self.settings.appearance == appearance { return false; }
        self.settings.appearance = appearance;
        if let Some(renderer) = &mut self.active { renderer.settings.appearance = appearance; }
        true
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
        self.settings.appearance = previous.settings.appearance;
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
                    let mut renderer = PlanetRenderer::replacing(
                        self.active.as_ref(),
                        ctx.device,
                        ctx.queue,
                        frame.planet.clone(),
                        self.settings,
                        [ctx.width, ctx.height],
                    );
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

/// Compare a world's terrain program on the GPU with its CPU field at
/// `samples` pseudo-random columns (every level, face edges included) and
/// ground-material inputs. Generator authors run this in their tests; it
/// returns the first disagreement.
pub fn verify_field(device: &wgpu::Device, queue: &wgpu::Queue, planet: &Planet, samples: u32) -> Result<(), String> {
    let grid = *planet.grid();
    let field = planet.field();
    let program = field.program();
    let mut rng = 0x1234_5678u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let (lo, hi) = field.height_range();
    let span = (i64::from(hi) - i64::from(lo) + 1).max(1) as u64;
    let mut inputs = Vec::new();
    let mut extra = Vec::new();
    for s in 0..samples {
        let level = (next() % u64::from(grid.levels())) as u32;
        let cells = (grid.cells() >> level).max(1) as u64;
        let face = if grid.is_plane() { i32::from(crate::grid::PLANE_FACE) } else { (next() % 6) as i32 };
        let (i, j) = if s % 4 == 0 {
            ((next() % 2) as i32 * (cells as i32 - 1), (next() % cells) as i32)
        } else {
            ((next() % cells) as i32, (next() % cells) as i32)
        };
        inputs.push(IVec4::new(face, i, j, level as i32));
        extra.push(IVec4::new(
            (i64::from(lo) + (next() % span) as i64) as i32,
            (next() % 40) as i32,
            (next() % 40) as i32,
            (next() % 200_000) as i32 - 100_000,
        ));
    }
    let kernel = "
@group(0) @binding(20) var<storage, read> verify_in: array<vec4<i32>>;
@group(0) @binding(21) var<storage, read> verify_extra: array<vec4<i32>>;
@group(0) @binding(22) var<storage, read_write> verify_out: array<vec2<i32>>;
@compute @workgroup_size(64) fn verify(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&verify_in) { return; }
    let a = verify_in[id.x];
    let e = verify_extra[id.x];
    let p = domain_point(u32(a.x), a.y, a.z, u32(a.w));
    verify_out[id.x] = vec2<i32>(field_height(u32(a.x), a.y, a.z, u32(a.w)), i32(ground_material(p, e.x, e.y, e.z, e.w)));
}
";
    let module = helio_core::shader::module(
        device,
        "terrain verification",
        &source("read", &[kernel], grid.is_plane(), &program),
    );
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("terrain verification"),
        layout: None,
        module: &module,
        entry_point: Some("verify"),
        compilation_options: Default::default(),
        cache: None,
    });
    let init = |label, contents: &[u8], usage| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage });
    let world = init("verify world", bytemuck::bytes_of(&WorldGpu::new(planet)), wgpu::BufferUsages::UNIFORM);
    let terrain = init("verify terrain", &terrain_bytes(&program), wgpu::BufferUsages::UNIFORM);
    let ins = init("verify inputs", bytemuck::cast_slice(&inputs), wgpu::BufferUsages::STORAGE);
    let ext = init("verify extra", bytemuck::cast_slice(&extra), wgpu::BufferUsages::STORAGE);
    let bytes = u64::from(samples) * 8;
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("verify out"),
        size: bytes.max(8),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("verify readback"),
        size: bytes.max(8),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("terrain verification"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 1, resource: world.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 16, resource: terrain.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 20, resource: ins.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 21, resource: ext.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 22, resource: out.as_entire_binding() },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(samples.div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out, 0, &read, 0, bytes.max(8));
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    read.slice(..).map_async(wgpu::MapMode::Read, move |r| drop(tx.send(r)));
    device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| e.to_string())?;
    rx.recv().map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
    let data = read.slice(..).get_mapped_range().map_err(|e| e.to_string())?;
    let gpu: &[[i32; 2]] = bytemuck::cast_slice(&data[..bytes as usize]);
    for ((a, e), g) in inputs.iter().zip(&extra).zip(gpu) {
        let p = grid.domain_point(a.x as u8, a.y, a.z, a.w as u32);
        let cpu = [
            field.height(p, a.w as u32 + grid.level_offset()),
            field.ground_material(p, e.x, e.y, e.z, e.w) as i32,
        ];
        if *g != cpu {
            return Err(format!("column {a} with inputs {e}: GPU {g:?}, CPU {cpu:?}"));
        }
    }
    Ok(())
}
