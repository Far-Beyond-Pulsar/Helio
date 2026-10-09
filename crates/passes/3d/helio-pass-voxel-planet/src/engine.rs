//! Helio integration: GPU residency, exact traversal and GBuffer output.
use crate::grid::Cell;
use crate::planet::Planet;
use crate::residency::{Capacity, FrameWork, JobBudget, PlanRequest, PlanResult, Residency, ResidencyWorker, NONE};
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
    /// Where tools ask for the terrain hit under points of the view.
    pub picks: Option<SharedPicks>,
}

pub type SharedPlanetFrame = Arc<Mutex<Option<PlanetFrame>>>;

/// A request for the terrain hit under view point `uv` (0..1, from the top
/// left), answered a frame or two later in [`Picks::results`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PickRequest {
    pub id: u64,
    pub uv: [f32; 2],
}

/// The answer to a [`PickRequest`]: the distance from the eye of the first
/// terrain hit along that pixel's ray and the size of the cell that drew it
/// (how far the exact surface can be from it), or `None` (sky, loading).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PickResult {
    pub id: u64,
    pub hit: Option<PickHit>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PickHit {
    pub distance: f64,
    pub cell_m: f64,
}

/// Pick requests and answers shared between a tool and the pass.
#[derive(Default, Debug)]
pub struct Picks {
    pub requests: Vec<PickRequest>,
    pub results: Vec<PickResult>,
}

pub type SharedPicks = Arc<Mutex<Picks>>;

/// Picks per readback (one frame's requests beyond it wait a frame).
const MAX_PICKS: usize = 8;
/// Bytes of a `Hit` (trace.wgsl).
const HIT_BYTES: u64 = 32;

/// One frame's copied hits for pick requests.
struct PickSlot {
    buffer: wgpu::Buffer,
    requests: Vec<u64>,
    sink: Option<SharedPicks>,
    state: Arc<AtomicBool>,
    stage: u8, // 0 free, 1 copied, 2 mapping
}

pub use crate::terrain::{MaterialAppearance, TerrainAppearance, MATERIALS};

/// `MaterialGpu` of common.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct MaterialGpu {
    colour: [f32; 4],
    patches: [[f32; 4]; 3],
    links: [u32; 4],
}

#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// Level cells project to this many pixels where their range starts.
    pub lod_pixels: f32,
    /// Relative width of the stochastic level transition.
    pub lod_dither: f32,
    /// Most column jobs a frame (the job and readback buffers); the GPU
    /// work itself is budgeted in work units (`PlanetStats::unit_budget`).
    pub job_budget: usize,
    /// End rising eye rays at the directional sky bound.
    pub horizon: bool,
    /// Occlude the sky's ambient light by the terrain around each point
    /// (`skylight` in surface.wgsl).
    pub sky_occlusion: bool,
    /// Skip hash lookups of columns the summary blocks prove absent.
    pub residency_hints: bool,
    /// Diagnostics (`HELIO_VOXEL_DEBUG`): 1 colours pixels by level, brighter
    /// where shading is filtered; 2 also lights them with the vertical (only
    /// sun shadows stay dark); 3 colours by the level the distance asks for;
    /// 4 colours by column kind (generated volume, edit topology, relief, plain).
    pub debug_view: u32,
    /// Preserve sub-cell radial relief in unedited coarse columns.
    /// Set before generating columns; resident columns retain their format.
    pub coarse_relief: bool,
    /// Generate coarse Landform columns from the ridge-envelope display
    /// height (unresolved ridges keep their mean mass). Disable to audit
    /// coarse tops against the canonical field. Set before generating.
    pub ridge_display: bool,
    /// Diagnostics: skip residency planning (no jobs, windows or evictions)
    /// so several renders see identical GPU state.
    pub freeze_residency: bool,
    /// Diagnostics: fixed frame index for the sunlight representative pattern.
    pub frame_override: Option<u32>,
    /// Diagnostics: keep a copy of the CPU column table as uploaded (see
    /// `PlanetRenderer::column_table`; copies 32 MB per plan).
    pub table_snapshots: bool,
    pub capacity: Capacity,
    /// Material table and detail; `None` uses the terrain generator's own
    /// ([`crate::terrain::TerrainField::appearance`]).
    pub appearance: Option<TerrainAppearance>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            lod_pixels: 1.0,
            lod_dither: std::env::var("HELIO_VOXEL_LOD_DITHER").ok().and_then(|v| v.parse().ok()).unwrap_or(0.25),
            job_budget: 12_288,
            horizon: std::env::var_os("HELIO_VOXEL_NO_HORIZON").is_none(),
            sky_occlusion: std::env::var_os("HELIO_VOXEL_NO_SKY_OCCLUSION").is_none(),
            residency_hints: true,
            debug_view: std::env::var("HELIO_VOXEL_DEBUG").ok().and_then(|v| v.parse().ok()).unwrap_or(0),
            coarse_relief: std::env::var("HELIO_VOXEL_COARSE_RELIEF").ok().is_none_or(|v| v != "0"),
            ridge_display: std::env::var("HELIO_VOXEL_RIDGE_DISPLAY").ok().is_none_or(|v| v != "0"),
            freeze_residency: false,
            frame_override: None,
            table_snapshots: false,
            capacity: Capacity::default(),
            appearance: None,
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
    /// Linear appearance (see `TerrainAppearance`).
    materials: [MaterialGpu; MATERIALS],
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
    /// Jobs retried because one frame's jobs outgrew the generation
    /// scratch (tall clipped bands); the job budget then shrinks.
    pub scratch_retries: usize,
    /// Columns published with a band clipped to the window around the eye
    /// (deep digs, deep caves), reported as they are generated.
    pub clipped_columns: usize,
    pub free_pages: i32,
    /// Free brick pool units (unassigned pages and free runs of all classes).
    pub free_units: u64,
    /// Pool page recycles so far (each under pool pressure).
    pub recycles: u32,
    /// Level-0 distance divisor keeping demand inside capacity (1 = none).
    pub lod_pressure: f64,
    /// Columns skipped because their table slot was out of GPU probe reach.
    pub table_refused: usize,
    pub pool_pages: u32,
    pub active_levels: u32,
    pub finest_level: u32,
    /// Residency worker CPU time of the last uploaded plan.
    pub plan_cpu_ms: f64,
    /// Frames whose residency plan was not ready (cumulative; such a frame
    /// uploads nothing).
    pub late_plans: usize,
    /// Pending columns re-ranked against a moved eye (cumulative).
    pub reranked: usize,
    pub upload_cpu_ms: f64,
    pub encode_cpu_ms: f64,
    pub window_rebuild_ms: f64,
    pub lod0_distance: f64,
    pub logical_bytes: u64,
    /// Measured GPU generation cost per work unit (microseconds; a unit is
    /// a heightfield column, `residency::job_units`), the plan's unit budget
    /// and the units this frame's jobs carry.
    pub us_per_unit: f64,
    pub unit_budget: f64,
    pub units: f64,
    /// Edit data on the GPU: baked brick slots in use and the pool's slots,
    /// and edit block words in use.
    pub baked_bricks: u32,
    pub baked_pool: u32,
    pub edit_words: u32,
    /// Distinct face brushes the resident columns reference.
    pub brushes: u32,
}

/// Copy of the allocator counters and the failed jobs since the last copy.
struct Readback {
    buffer: wgpu::Buffer,
    /// Failure entries copied (at most the jobs issued since the last copy).
    entries: u32,
    state: Arc<AtomicBool>,
    stage: u8, // 0 free, 1 encoded, 2 mapping, 3 reserved for an in-flight plan's jobs
}

const PROBE_BYTES: u64 = 128;
/// Bytes per failed-job entry (key0, key1, status, pad).
const FAILURE_BYTES: u64 = 16;
/// Allocator word counting failed jobs (`A_FAILS` in generate.wgsl).
const A_FAILS: u64 = 10;
/// Frames between pool page recycles while the pool is under pressure.
const RECYCLE_INTERVAL: u64 = 60;

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
    /// sphere domain constants (`Grid::sphere_constants`); w: columns store
    /// a surface word per cell.
    sphere: [u32; 4],
}

impl WorldGpu {
    fn new(planet: &Planet) -> Self {
        let g = planet.grid();
        let m = planet.field().render_bound_margins();
        Self {
            grid: [g.reference_cells(), g.layer_mm() as i32, g.cells(), g.level_offset() as i32],
            scale: {
                let (inv, shift, layer_q16) = g.volume_constants();
                [g.domain_scale(), inv, shift, layer_q16]
            },
            bounds: std::array::from_fn(|i| std::array::from_fn(|j| m[i * 4 + j])),
            sphere: {
                let mut sphere = g.sphere_constants();
                sphere[3] = u32::from(planet.field().program().wgsl.contains("fn terrain_surface"));
                sphere
            },
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

/// The pass's composed shaders as compiled for a world form and terrain
/// program: generation, trace and gbuffer.
pub fn shader_sources(plane: bool, program: &TerrainProgram) -> [(&'static str, String); 3] {
    let view = include_str!("../shaders/view.wgsl");
    [
        ("planet generation", source("read_write", &[include_str!("../shaders/generate.wgsl")], plane, program)),
        (
            "planet trace",
            source(
                "read_write",
                &[view, include_str!("../shaders/horizon.wgsl"), include_str!("../shaders/trace.wgsl"), include_str!("../shaders/surface.wgsl")],
                plane,
                program,
            ),
        ),
        ("planet gbuffer", source("read", &[view, include_str!("../shaders/gbuffer.wgsl")], plane, program)),
    ]
}

/// [`shader_sources`] for both world forms with the Earth terrain program
/// (shader validation without a device).
pub fn validation_sources() -> Vec<(String, String)> {
    let grid = crate::grid::Grid::new(6_371_000.0, 0.1).expect("Earth grid");
    let field = crate::layers::TerrainLayers::earth().field(&grid, 1).expect("Earth terrain");
    let program = crate::terrain::TerrainField::program(&field);
    [false, true]
        .into_iter()
        .flat_map(|plane| {
            let form = if plane { "plane" } else { "sphere" };
            shader_sources(plane, &program).map(|(label, source)| (format!("{label} ({form})"), source))
        })
        .collect()
}

/// Shader source: the noise library, world helpers and the terrain program,
/// then the engine parts.
fn source(access: &str, parts: &[&str], plane: bool, program: &TerrainProgram) -> String {
    let mut s = String::from(include_str!("../shaders/noise.wgsl"));
    s.push_str(include_str!("../shaders/world.wgsl"));
    s.push_str(&program.wgsl);
    if !program.wgsl.contains("fn terrain_surface") {
        // No surface word: columns store none (`World::sphere.w`).
        s.push_str("fn terrain_surface(p: vec3<i32>, level: u32, height: i32) -> u32 { return 0u; }\n");
    }
    if !program.wgsl.contains("fn terrain_density") {
        // Heightfield programs: no volumetric terms (`TerrainField::extent`, `density`).
        s.push_str("fn terrain_extent(p: vec3<i32>, level: u32) -> vec2<i32> { return vec2<i32>(0); }\n");
        s.push_str("fn terrain_density(p: vec3<i32>, q: vec3<i32>, level: u32, top: i32, height: i32, lean_height: i32, k: i32) -> i32 { return heightfield_density(top, k); }\n");
    }
    if !program.wgsl.contains("fn terrain_lean") {
        // No lean (`TerrainField::lean`).
        s.push_str("fn terrain_lean(level: u32) -> vec2<i32> { return vec2<i32>(0); }\n");
        s.push_str("fn terrain_lean_offset(p: vec3<i32>, i: i32, j: i32, k: i32, level: u32) -> vec2<i32> { return vec2<i32>(0); }\n");
    }
    // Generation updates the summaries atomically; traversal reads plain values.
    // Traversal reads a summary block entry as one vector load.
    let generation = parts.iter().any(|p| p.contains("fn level_suffix"));
    let (level_top, block_entry) = if generation { ("atomic<i32>", "atomic<i32>") } else { ("i32", "vec4<i32>") };
    s.push_str(
        &include_str!("../shaders/common.wgsl")
            .replace("ACCESS", access)
            .replace("LEVEL_TOP", level_top)
            .replace("BLOCK_ENTRY", block_entry)
            .replace("SHAPE_ID", if plane { "1u" } else { "0u" })
            // Diagnostics: a different probe limit per salt misses every
            // driver shader cache, to measure cold pipeline compiles.
            .replace(
                "const MAX_PROBES: u32 = 64u;",
                &std::env::var("HELIO_VOXEL_SHADER_SALT")
                    .ok()
                    .and_then(|v| v.parse::<u32>().ok())
                    .map_or_else(|| "const MAX_PROBES: u32 = 64u;".to_string(), |salt| format!("const MAX_PROBES: u32 = {}u;", 64 + salt % 64)),
            ),
    );
    // Generation evaluates a column's height and surface word at one call
    // site (`terrain_column`): compilers inline every call, and each copy
    // of a large program is compile time.
    if program.wgsl.contains("fn terrain_column") {
        if generation {
            s.push_str("fn generation_column(face:u32,i:i32,j:i32,level:u32,display:bool)->vec2<i32> { return terrain_column(domain_point(face,i,j,level),level+u32(world.grid.w),display); }\n");
        }
    } else {
        if generation {
            s.push_str("fn generation_column(face:u32,i:i32,j:i32,level:u32,display:bool)->vec2<i32> { let h = field_height(face,i,j,level); return vec2<i32>(h, i32(terrain_surface(domain_point(face,i,j,level),level+u32(world.grid.w),h))); }\n");
        }
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
    publish: wgpu::ComputePipeline,
    level_suffix: wgpu::ComputePipeline,
    recycle_layout: wgpu::BindGroupLayout,
    reclaim: wgpu::ComputePipeline,
    compact: wgpu::ComputePipeline,
    finish_recycle: wgpu::ComputePipeline,
    primary: wgpu::ComputePipeline,
    horizon_clear: wgpu::ComputePipeline,
    horizon_blocks: wgpu::ComputePipeline,
    horizon_suffix: wgpu::ComputePipeline,
    shade: wgpu::ComputePipeline,
    sunlight: wgpu::ComputePipeline,
    skylight: wgpu::ComputePipeline,
    gbuffer: wgpu::RenderPipeline,
}

impl Pipelines {
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
            storage(18, false),
            storage(20, true),
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
            storage(20, true),
            storage(7, false),
            storage(8, false),
            storage(14, false),
            storage(15, false),
            uniform(16),
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
        // Composed from several files plus the terrain program in Rust, so it
        // goes through `module` as plain text (not hot reloadable).
        let module = |label: &str, src: String| helio_core::shader::module(device, label, &src);
        let [(_, gen_src), (_, trace_src), (_, render_src)] = shader_sources(plane, program);
        let gen_module = module("planet generation", gen_src);
        let trace_module = module("planet trace", trace_src);
        let render_module = module("planet gbuffer", render_src);
        let recycle_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("planet pool recycling"),
            entries: &[storage(0, false), storage(1, false), storage(2, true), storage(3, false), storage(4, false), storage(5, false)],
        });
        let recycle_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("planet pool recycling"),
            bind_group_layouts: &[Some(&recycle_layout)],
            immediate_size: 0,
        });
        let recycle_module = helio_core::shader::module(device, "planet pool recycling", include_str!("../shaders/allocator_recycle.wgsl"));
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
        let times = std::env::var_os("HELIO_VOXEL_PIPELINE_TIMES").is_some();
        let compute = |layout: &wgpu::PipelineLayout, module: &wgpu::ShaderModule, entry: &str| {
            let started = std::time::Instant::now();
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(layout),
                module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: if entry == "generate" && program.key == crate::landform::DISPLAY_PROGRAM {
                        &[("RIDGE_DISPLAY_GENERATION", 1.0)]
                    } else {
                        &[]
                    },
                    ..Default::default()
                },
                cache: None,
            });
            if times {
                eprintln!("PIPELINE {entry} {:.0} ms", started.elapsed().as_secs_f64() * 1e3);
            }
            pipeline
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
        // The large programs (the terrain generator, shading with the
        // material rules, the two tracers) compile on their own threads:
        // drivers compile pipelines in parallel, and these took ~12 s in a
        // row cold (90 s on a loaded editor start).
        std::thread::scope(|scope| {
            let generate = scope.spawn(|| compute(&gen_pl, &gen_module, "generate"));
            let shade = scope.spawn(|| compute(&trace_pl, &trace_module, "shade"));
            let sunlight = scope.spawn(|| compute(&trace_pl, &trace_module, "sunlight"));
            let skylight = scope.spawn(|| compute(&trace_pl, &trace_module, "skylight"));
            let primary = scope.spawn(|| compute(&trace_pl, &trace_module, "primary"));
            let join = |h: std::thread::ScopedJoinHandle<'_, wgpu::ComputePipeline>| h.join().expect("pipeline compile thread");
            Self {
                plane,
                program: program.key.to_string(),
                patch: compute(&gen_pl, &gen_module, "patch_table"),
                patch_blocks: compute(&gen_pl, &gen_module, "patch_blocks"),
                evict: compute(&gen_pl, &gen_module, "evict"),
                count: compute(&gen_pl, &gen_module, "count"),
                refill: compute(&gen_pl, &gen_module, "refill"),
                allocate: compute(&gen_pl, &gen_module, "allocate"),
                fixup: compute(&gen_pl, &gen_module, "fixup"),
                publish: compute(&gen_pl, &gen_module, "publish"),
                level_suffix: compute(&gen_pl, &gen_module, "level_suffix"),
                reclaim: compute(&recycle_pl, &recycle_module, "reclaim"),
                compact: compute(&recycle_pl, &recycle_module, "compact"),
                finish_recycle: compute(&recycle_pl, &recycle_module, "finish"),
                recycle_layout,
                horizon_clear: compute(&trace_pl, &trace_module, "horizon_clear"),
                horizon_blocks: compute(&trace_pl, &trace_module, "horizon_blocks"),
                horizon_suffix: compute(&trace_pl, &trace_module, "horizon_suffix"),
                generate: join(generate),
                shade: join(shade),
                sunlight: join(sunlight),
                skylight: join(skylight),
                primary: join(primary),
                gbuffer,
                gen_layout,
                trace_layout,
                render_layout,
                camera_layout,
            }
        })
    }
}

struct Buffers {
    frame: wgpu::Buffer,
    world: wgpu::Buffer,
    terrain: wgpu::Buffer,
    table: wgpu::Buffer,
    records: wgpu::Buffer,
    pool: wgpu::Buffer,
    /// Baked brick slots (`BAKED_BRICK_BYTES` each), grown to the
    /// residency's high-water mark up to `Capacity::baked_bricks`.
    baked: wgpu::Buffer,
    baked_slots: u32,
    /// The shared face brush table (`FACE_BRUSH_BYTES` each), grown to the
    /// residency's high-water mark up to `Capacity::brushes`.
    brushes: wgpu::Buffer,
    brush_slots: u32,
    edit_refs: wgpu::Buffer,
    jobs: wgpu::Buffer,
    job_out: wgpu::Buffer,
    scratch: wgpu::Buffer,
    alloc: wgpu::Buffer,
    free_runs: wgpu::Buffer,
    free_pages: wgpu::Buffer,
    page_meta: wgpu::Buffer,
    failures: wgpu::Buffer,
    /// Recycling: compacted free run stacks (created on first use) and
    /// per-class counts.
    compacted_runs: Option<wgpu::Buffer>,
    recycle_counts: wgpu::Buffer,
    evictions: wgpu::Buffer,
    level_tops: wgpu::Buffer,
    block_state: wgpu::Buffer,
    /// Directional sky bound: accumulated and suffix tables.
    horizon_acc: wgpu::Buffer,
    horizon: wgpu::Buffer,
    /// Live tier-1 block slots (grows).
    live_blocks: wgpu::Buffer,
    bytes: u64,
}

const JOB_OUT_BYTES: u64 = 104;
/// Bytes per face brush (`edits::FaceBrush`).
const FACE_BRUSH_BYTES: u64 = std::mem::size_of::<crate::edits::FaceBrush>() as u64;
/// Bytes per baked brick slot: 512 cells of 16 bits.
const BAKED_BRICK_BYTES: u64 = crate::edit_store::BRICK_CELLS as u64 * 2;
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
        let free_runs = make("planet free runs", u64::from(cap.pool_units) * 8, st | wgpu::BufferUsages::COPY_DST);
        let page_meta = make("planet pool pages", u64::from(pages) * 8, st);
        let failures = make("planet failed jobs", u64::from(cap.max_jobs) * FAILURE_BYTES, st | wgpu::BufferUsages::COPY_SRC);
        let recycle_counts = make("planet recycled run counts", 64, st);
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
        // A thirtieth of the budget to start; it grows with destruction.
        let baked_slots = (cap.baked_bricks / 32).max(64).min(cap.baked_bricks);
        let baked = make("planet baked edits", u64::from(baked_slots) * BAKED_BRICK_BYTES, st | wgpu::BufferUsages::COPY_SRC);
        let brush_slots = (cap.brushes / 64).max(256).min(cap.brushes.max(1));
        let brushes = make("planet brushes", u64::from(brush_slots) * FACE_BRUSH_BYTES, st | wgpu::BufferUsages::COPY_SRC);
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
            baked,
            baked_slots,
            brushes,
            brush_slots,
            edit_refs,
            jobs,
            job_out,
            scratch,
            alloc,
            free_runs,
            free_pages,
            page_meta,
            failures,
            compacted_runs: None,
            recycle_counts,
            evictions,
            level_tops,
            block_state,
            horizon_acc,
            horizon,
            live_blocks,
            bytes,
        }
    }
}

struct Screen {
    size: [u32; 2],
    hits: wgpu::Buffer,
    surfaces: wgpu::Buffer,
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
        Self {
            size,
            hits,
            surfaces,
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
    residency: ResidencyWorker,
    /// The residency as the GPU holds it: the last uploaded plan's result
    /// (its work taken).
    plan: PlanResult,
    /// Job outcomes read back, for the next plan request.
    failed: Vec<(u64, u32, i32)>,
    /// Readback reserved for the in-flight plan's jobs.
    plan_readback: Option<usize>,
    /// Planet, eye and level-0 distance of the last plan request.
    submitted: Option<(Arc<Planet>, DVec3, f64)>,
    /// Mean interval between encoded frames (ms), for the worker's budget.
    frame_ms: f64,
    last_encode: Option<std::time::Instant>,
    /// `residency_health` was asked for: scan the table in the next plan.
    want_probe: AtomicBool,
    planet: Arc<Planet>,
    settings: Settings,
    gen_group: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,
    camera_group: wgpu::BindGroup,
    /// Last local projection and precise eye, for motion in the shared GBuffer.
    camera_history: Option<(u64, u32, DVec3, Mat4)>,
    readbacks: Vec<Readback>,
    picks: Vec<PickSlot>,
    frame_index: u32,
    stats: PlanetStats,
    sun_active: bool,
    profiler: Option<helio_core::profiling::GpuProfiler>,
    initial_complete: bool,
    /// GPU generation cost per work unit (ms, `unit_cost`).
    ms_per_unit: f64,
    /// Recent (work units, generate GPU ms) of frames that generated: the
    /// `generate` dispatch alone, timed on its own. The residency stage's
    /// other work (evictions, table patches, allocation, publication) does
    /// not scale with the jobs, and a cave column costs tens of times a
    /// heightfield one: a regression of residency time over job counts swung
    /// from 0.1 to 50 us a job (12k-job frames, 100-250 ms generation
    /// spikes, then 256-job crawls).
    cost_samples: std::collections::VecDeque<(f64, f64)>,
    /// Work units issued per recent frame number, and the frame whose
    /// timestamps last updated `ms_per_unit`.
    frame_units: std::collections::VecDeque<(u64, f64)>,
    costed_frame: Option<u64>,
    /// Divides the level-0 distance while demand exceeds the record or pool
    /// capacity (>= 1; see `update_lod_pressure`).
    lod_pressure: f64,
    /// Failed jobs counted at the last pressure step.
    pressure_failed_jobs: usize,
    /// Job budget scale under scratch pressure (1 without).
    scratch_scale: f64,
    last_pressure_update: u64,
    /// The pool ran short (from readbacks); last frame that recycled pages.
    pool_pressure: bool,
    last_recycle: u64,
    last_eye: Option<DVec3>,
    last_frame_num: u64,
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
        Self::with_pipelines(pipelines, device, queue, planet, settings, size)
    }

    /// A renderer for `planet` drawn with already compiled `pipelines`
    /// (which must serve its shape and terrain program).
    fn with_pipelines(
        pipelines: Arc<Pipelines>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        planet: Arc<Planet>,
        settings: Settings,
        size: [u32; 2],
    ) -> Self {
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
        let readbacks = (0..8)
            .map(|_| Readback {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("planet status readback"),
                    size: PROBE_BYTES + u64::from(settings.capacity.max_jobs) * FAILURE_BYTES,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                entries: 0,
                state: Arc::new(AtomicBool::new(false)),
                stage: 0,
            })
            .collect();
        Self {
            device: device.clone(),
            queue: queue.clone(),
            screen: Screen::new(device, size),
            residency: ResidencyWorker::start(*planet.grid(), settings.capacity),
            plan: PlanResult::initial(*planet.grid()),
            failed: Vec::new(),
            plan_readback: None,
            submitted: None,
            frame_ms: 16.7,
            last_encode: None,
            want_probe: AtomicBool::new(false),
            planet,
            settings,
            gen_group,
            camera_buffer,
            camera_group,
            camera_history: None,
            readbacks,
            picks: (0..3)
                .map(|_| PickSlot {
                    buffer: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("planet pick readback"),
                        size: MAX_PICKS as u64 * HIT_BYTES,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    }),
                    requests: Vec::new(),
                    sink: None,
                    state: Arc::new(AtomicBool::new(false)),
                    stage: 0,
                })
                .collect(),
            frame_index: 0,
            stats: PlanetStats::default(),
            sun_active: false,
            // Stage timestamps are not only diagnostics: the generation
            // budget divides a time target by the measured cost per column.
            // Without them it stays at the conservative default (the editor
            // streamed 3x slower than the harness, which enabled profiling).
            profiler: timestamps_supported(device).then(|| helio_core::profiling::GpuProfiler::new(device, queue)),
            initial_complete: false,
            ms_per_unit: 0.0013,
            frame_units: std::collections::VecDeque::new(),
            costed_frame: None,
            cost_samples: std::collections::VecDeque::new(),
            pool_pressure: false,
            last_recycle: 0,
            lod_pressure: 1.0,
            pressure_failed_jobs: 0,
            scratch_scale: 1.0,
            last_pressure_update: 0,
            last_eye: None,
            last_frame_num: 0,
            pipelines,
            buffers,
        }
    }

    fn gen_group(device: &wgpu::Device, p: &Pipelines, b: &Buffers) -> wgpu::BindGroup {
        let mut entries: Vec<wgpu::BindGroupEntry> = [
            &b.frame, &b.world, &b.table, &b.records, &b.pool, &b.baked, &b.edit_refs, &b.jobs, &b.job_out,
            &b.scratch, &b.alloc, &b.free_runs, &b.free_pages, &b.evictions, &b.level_tops, &b.block_state, &b.terrain,
            &b.failures, &b.page_meta,
        ]
        .iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect();
        entries.push(wgpu::BindGroupEntry { binding: 20, resource: b.brushes.as_entire_binding() });
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
    /// GPU column hash table and the CPU table it must equal once the last
    /// encoded frame has executed (diagnostics; empty unless
    /// `Settings::table_snapshots` was set before that frame's plan).
    pub fn column_table(&self) -> (&wgpu::Buffer, &[u32]) {
        (&self.buffers.table, self.plan.table.as_deref().unwrap_or(&[]))
    }
    /// Longest column table probe run, entries beyond the GPU probe limit,
    /// and queued window diffs (diagnostics). The worker scans the table on
    /// request, so the probe figures trail the call by a frame or two.
    pub fn residency_health(&self) -> (u32, usize, usize) {
        self.want_probe.store(true, Ordering::Relaxed);
        let (longest, beyond) = self.plan.probe.unwrap_or_default();
        (longest, beyond, self.plan.queued_diffs)
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
    /// Residency has issued and completed every window column. A readback
    /// of the allocator counters alone (sampled every 30 frames) carries no
    /// job outcome: counting it kept hosts that render until settled (the
    /// editor viewport) rendering forever, each sample restarting their
    /// settling.
    pub fn settled(&self) -> bool {
        self.plan.idle
            && !self.residency.in_flight()
            && self.failed.is_empty()
            && self.readbacks.iter().all(|r| r.stage == 0 || (r.stage != 3 && r.entries == 0))
    }

    fn frame_uniform(&self, eye: DVec3, size: [u32; 2], lod0: f64, jobs: u32, evictions: u32, sun: Vec3, shadows: bool) -> FrameGpu {
        let planet = &self.planet;
        let grid = planet.grid();
        let mut frame = FrameGpu::default();
        // Appearance is public in sRGB; convert once per frame, not per pixel.
        let clean = |v: f32| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
        let linear = |c: [f32; 4]| [clean(c[0]).powf(2.2), clean(c[1]).powf(2.2), clean(c[2]).powf(2.2), clean(c[3])];
        let appearance = self.settings.appearance.unwrap_or_else(|| planet.field().appearance());
        frame.materials = std::array::from_fn(|id| {
            let m = &appearance.materials[id];
            let link = |other: Option<u8>| other.filter(|&o| usize::from(o) < MATERIALS).map_or(id as u32, u32::from);
            let rgb = |c: [f32; 3]| linear([c[0], c[1], c[2], 1.0]);
            let mut patches = [[0.0; 4]; 3];
            if let Some(p) = m.patches {
                patches = p.map(rgb);
                patches[0][3] = 1.0;
            }
            let (fleck, share) = m.fleck.map_or((None, 0.0), |(f, s)| (Some(f), clean(s)));
            MaterialGpu {
                colour: linear(m.colour),
                patches,
                links: [link(m.lip), link(fleck), link(m.speck_host), (share * 65_536.0) as u32],
            }
        });
        frame.detail = appearance.detail.map(clean);
        frame.hints[3] = (if self.settings.coarse_relief { 8 } else { 0 }) | (if self.settings.ridge_display { 16 } else { 0 }) | (self.settings.debug_view << 8);
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
            .plan
            .coverage
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
        if work.baked_slots > self.buffers.baked_slots {
            self.grow_baked(work.baked_slots);
        }
        if work.brush_slots > self.buffers.brush_slots {
            self.grow_brushes(work.brush_slots);
        }
        for (slot, brush) in &work.brush_writes {
            self.queue.write_buffer(&self.buffers.brushes, u64::from(*slot) * FACE_BRUSH_BYTES, bytemuck::bytes_of(brush));
        }
        for (slot, brick) in &work.baked_writes {
            let cells: Vec<u16> = brick.cells.iter().map(|c| c.0).collect();
            self.queue.write_buffer(&self.buffers.baked, u64::from(*slot) * BAKED_BRICK_BYTES, bytemuck::cast_slice(&cells));
        }
        for (base, words) in &work.edit_writes {
            self.queue.write_buffer(&self.buffers.edit_refs, u64::from(*base) * 4, bytemuck::cast_slice(words));
        }
        if !work.jobs.is_empty() {
            self.queue.write_buffer(&self.buffers.jobs, 0, bytemuck::cast_slice(&work.jobs));
        }
        // Evictions followed by table patches (slot, value) pairs and summary
        // block patches. The GPU applies patches in parallel; the residency
        // sends each slot once, with its final value.
        let mut words: Vec<u32> = work.evictions.clone();
        for (slot, value) in &work.table_writes {
            words.extend([*slot, *value]);
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
        (work.table_writes.len() as u32, work.block_inits.len() as u32)
    }

    /// Grow the baked brick pool to hold `needed` slots (doubling, within
    /// the budget), keeping the bricks already uploaded.
    fn grow_baked(&mut self, needed: u32) {
        let slots = needed.next_power_of_two().max(self.buffers.baked_slots * 2).min(self.settings.capacity.baked_bricks.max(needed));
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planet baked edits"),
            size: u64::from(slots) * BAKED_BRICK_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.buffers.baked, 0, &buffer, 0, u64::from(self.buffers.baked_slots) * BAKED_BRICK_BYTES);
        self.queue.submit([encoder.finish()]);
        self.buffers.bytes += u64::from(slots - self.buffers.baked_slots) * BAKED_BRICK_BYTES;
        self.buffers.baked = buffer;
        self.buffers.baked_slots = slots;
        self.gen_group = Self::gen_group(&self.device, &self.pipelines, &self.buffers);
    }

    /// Grow the shared brush table to hold `needed` slots (doubling, within
    /// the budget), keeping the brushes already uploaded.
    fn grow_brushes(&mut self, needed: u32) {
        let slots = needed.next_power_of_two().max(self.buffers.brush_slots * 2).min(self.settings.capacity.brushes.max(needed));
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planet brushes"),
            size: u64::from(slots) * FACE_BRUSH_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.buffers.brushes, 0, &buffer, 0, u64::from(self.buffers.brush_slots) * FACE_BRUSH_BYTES);
        self.queue.submit([encoder.finish()]);
        self.buffers.bytes += u64::from(slots - self.buffers.brush_slots) * FACE_BRUSH_BYTES;
        self.buffers.brushes = buffer;
        self.buffers.brush_slots = slots;
        self.gen_group = Self::gen_group(&self.device, &self.pipelines, &self.buffers);
    }

    /// Copy the hits under this frame's pick requests for readback.
    fn copy_picks(slots: &mut [PickSlot], hits: &wgpu::Buffer, encoder: &mut wgpu::CommandEncoder, picks: &SharedPicks, size: [u32; 2]) {
        let Some(slot) = slots.iter_mut().find(|slot| slot.stage == 0) else { return };
        let requests: Vec<PickRequest> = {
            let Ok(mut shared) = picks.lock() else { return };
            let n = shared.requests.len().min(MAX_PICKS);
            shared.requests.drain(..n).collect()
        };
        if requests.is_empty() {
            return;
        }
        for (n, request) in requests.iter().enumerate() {
            let x = ((request.uv[0].clamp(0.0, 1.0) * size[0] as f32) as u32).min(size[0].max(1) - 1);
            let y = ((request.uv[1].clamp(0.0, 1.0) * size[1] as f32) as u32).min(size[1].max(1) - 1);
            let pixel = u64::from(x) + u64::from(y) * u64::from(size[0]);
            encoder.copy_buffer_to_buffer(hits, pixel * HIT_BYTES, &slot.buffer, n as u64 * HIT_BYTES, HIT_BYTES);
        }
        slot.requests = requests.iter().map(|r| r.id).collect();
        slot.sink = Some(picks.clone());
        slot.stage = 1;
    }

    /// Answer picks whose hits arrived; start mapping last frame's copies.
    fn poll_picks(&mut self) {
        let voxel = self.planet.grid().voxel_size();
        for slot in &mut self.picks {
            if slot.stage == 2 && slot.state.load(Ordering::Acquire) {
                let results: Vec<PickResult> = {
                    let data = slot.buffer.slice(..).get_mapped_range().unwrap();
                    slot.requests
                        .iter()
                        .enumerate()
                        .map(|(n, &id)| {
                            let at = n * HIT_BYTES as usize;
                            let t = f32::from_le_bytes(data[at..at + 4].try_into().unwrap());
                            let info = u32::from_le_bytes(data[at + 16..at + 20].try_into().unwrap());
                            let hit = (info & 3 == 1 && t.is_finite() && t > 0.0)
                                .then(|| PickHit { distance: f64::from(t), cell_m: voxel * f64::from(1u32 << ((info >> 5) & 31)) });
                            PickResult { id, hit }
                        })
                        .collect()
                };
                slot.buffer.unmap();
                if let Some(Ok(mut shared)) = slot.sink.take().as_ref().map(|s| s.lock()) {
                    shared.results.extend(results);
                }
                slot.requests.clear();
                slot.stage = 0;
                slot.state.store(false, Ordering::Release);
            }
        }
        for slot in &mut self.picks {
            if slot.stage == 1 {
                let state = slot.state.clone();
                slot.buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                    if result.is_ok() {
                        state.store(true, Ordering::Release);
                    }
                });
                slot.stage = 2;
            }
        }
    }

    fn poll_readbacks(&mut self) {
        let _ = self.device.poll(wgpu::PollType::Poll);
        let mut failed = Vec::new();
        let pages = self.settings.capacity.pool_units / 512;
        for r in &mut self.readbacks {
            if r.stage == 2 && r.state.load(Ordering::Acquire) {
                {
                    let data = r.buffer.slice(..).get_mapped_range().unwrap();
                    let word = |i: usize| i32::from_le_bytes(data[i * 4..i * 4 + 4].try_into().unwrap());
                    let entries = (word(A_FAILS as usize).max(0) as u32).min(r.entries) as usize;
                    for e in 0..entries {
                        let at = PROBE_BYTES as usize / 4 + e * 4;
                        let key = u64::from(word(at) as u32) | (u64::from(word(at + 1) as u32) << 32);
                        failed.push((key, word(at + 2) as u32, word(at + 3)));
                    }
                    self.stats.free_pages = word(30);
                    // Pool pressure: generation failed for want of space, or
                    // few pages are left to give to a size class.
                    if failed.iter().any(|(_, status, _)| *status == 3) || (word(30).max(0) as u32) < pages / 16 {
                        self.pool_pressure = true;
                    }
                    // Free runs of every size class plus unassigned pages.
                    self.stats.free_units = (0..10).map(|c| u64::from(word(c).max(0) as u32) << c).sum::<u64>()
                        + u64::from(word(30).max(0) as u32) * 512;
                }
                r.buffer.unmap();
                r.stage = 0;
                r.state.store(false, Ordering::Release);
            }
        }
        // Scratch overflow: halve the job budget; recover while it does not
        // recur (bands of 256 bricks take 256 scratch units per job).
        let scratch = failed.iter().filter(|(_, s, _)| *s == 2).count();
        if scratch > 0 {
            self.scratch_scale = (self.scratch_scale * 0.5).max(1.0 / 64.0);
        } else if self.scratch_scale < 1.0 {
            self.scratch_scale = (self.scratch_scale * 1.1).min(1.0);
        }
        self.stats.scratch_retries += scratch;
        self.stats.failed_jobs += failed.iter().filter(|(_, s, _)| *s != crate::residency::STATUS_CLIPPED && *s != 2).count();
        self.stats.clipped_columns += failed.iter().filter(|(_, s, _)| *s == crate::residency::STATUS_CLIPPED).count();
        self.failed.extend(failed);
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

    /// Keep demand inside what capacity and admission throughput can serve
    /// instead of stalling at a limit. Resident columns grow with the pixel
    /// count (a 1440p ground view wants ~1.7M, 2.6x the 720p set): once
    /// records or pool units ran out, admission stopped and the view stayed
    /// coarse; and after a dive, admission (CPU-bound, ~1500 columns per
    /// frame) needed ~25 s to refill it while rays fell back several levels
    /// (cells 100+ px wide). Above 85% of records or pool, or with more than
    /// 10% of the wanted columns (and 50k) outstanding (pending or still in
    /// queued window diffs), the level-0 distance
    /// shrinks 10% (25% above 25% pending; cells a little wider, demand and
    /// churn ~20% lower per 10%): the view stays
    /// complete one step coarser and refines as the backlog drains. Below
    /// 65% of both and 2% outstanding it recovers 5%. Steps are 30 frames apart
    /// so each window replan settles first.
    fn update_lod_pressure(&mut self, frame_num: u64, moving: bool) {
        if frame_num < self.last_pressure_update + 30 {
            return;
        }
        let cap = &self.settings.capacity;
        let rs = &self.plan.stats;
        let records = (rs.resident_columns + rs.pending_columns) as f64 / f64::from(cap.records);
        // `free_units` is 0 until the first allocator readback.
        let pool = if self.stats.free_units == 0 { 0.0 } else { 1.0 - self.stats.free_units as f64 / f64::from(cap.pool_units) };
        // Edit data counts as pool: baked brick slots and edit block words
        // (a destroyed region's columns hold more of both).
        let edits = (f64::from(rs.baked_bricks) / f64::from(cap.baked_bricks.max(1))).max(f64::from(rs.edit_words) / f64::from(cap.edit_words.max(1)));
        let pool = pool.max(edits);
        // No free page and jobs waiting to retry: the free units left belong
        // to other size classes, so the pool is full for the columns wanted
        // (counting units alone left a fragmented pool failing forever).
        let starved = self.stats.free_pages == 0 && self.stats.free_units != 0 && self.stats.failed_jobs > self.pressure_failed_jobs;
        self.pressure_failed_jobs = self.stats.failed_jobs;
        // Only wanted columns count: removals a pressure step itself queues
        // must not raise it further. A still camera's backlog is loading,
        // not churn: it never raises pressure and never blocks recovery.
        let outstanding = rs.pending_columns + self.plan.queued_adds;
        let backlog = if moving { outstanding as f64 / (rs.resident_columns + outstanding).max(1) as f64 } else { 0.0 };
        let pressure = if records > 0.85 || pool > 0.85 || starved || (backlog > 0.1 && outstanding > 50_000) {
            // Churn falls with the square of the pressure; a deep backlog
            // (fast flight at high resolution) takes bigger steps.
            (self.lod_pressure * if backlog > 0.25 { 1.25 } else { 1.1 }).min(4.0)
        } else if records < 0.65 && pool < 0.65 && !starved && backlog < 0.02 {
            (self.lod_pressure / 1.05).max(1.0)
        } else {
            self.lod_pressure
        };
        if pressure != self.lod_pressure {
            self.lod_pressure = pressure;
            self.last_pressure_update = frame_num;
        }
        self.stats.lod_pressure = self.lod_pressure;
    }

    /// Return wholly free pool pages from their size classes to the free
    /// page stack (allocator_recycle.wgsl).
    fn recycle_pages(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let b = &mut self.buffers;
        let compacted = b.compacted_runs.get_or_insert_with(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("planet compacted free runs"),
                size: b.free_runs.size(),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        });
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet pool recycling"),
            layout: &self.pipelines.recycle_layout,
            entries: &[&b.alloc, &b.page_meta, &b.free_runs, &*compacted, &b.free_pages, &b.recycle_counts]
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry { binding: binding as u32, resource: buffer.as_entire_binding() })
                .collect::<Vec<_>>(),
        });
        encoder.clear_buffer(&b.recycle_counts, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &group, &[]);
            let pages = self.settings.capacity.pool_units / 512;
            Self::dispatch(&mut pass, &self.pipelines.reclaim, [pages.div_ceil(64), 1, 1]);
            let groups = (self.settings.capacity.pool_units * 2).div_ceil(256);
            Self::dispatch(&mut pass, &self.pipelines.compact, [groups.min(32_768), groups.div_ceil(32_768), 1]);
            Self::dispatch(&mut pass, &self.pipelines.finish_recycle, [1, 1, 1]);
        }
        encoder.copy_buffer_to_buffer(compacted, 0, &b.free_runs, 0, b.free_runs.size());
        self.stats.recycles += 1;
    }

    fn dispatch(pass: &mut wgpu::ComputePass<'_>, pipeline: &wgpu::ComputePipeline, groups: [u32; 3]) {
        if groups.iter().all(|g| *g > 0) {
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
        }
    }

    /// Dispatch boundaries are the finest portable GPU timestamp granularity.
    /// Separate compute passes allow encoder timestamps without requiring the
    /// optional TIMESTAMP_QUERY_INSIDE_PASSES device feature. The untimed path
    /// below keeps the original single compute pass.
    #[allow(clippy::too_many_arguments)]
    fn encode_residency_detailed(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &mut PassContext<'_>,
        jobs: u32,
        evictions: u32,
        patches: u32,
        block_patches: u32,
    ) {
        macro_rules! scope {
            ($path:literal, $body:block) => {{
                ctx.begin_gpu_scope(encoder, concat!("VoxelPlanet::residency::", $path));
                $body
                ctx.end_gpu_scope(encoder, concat!("VoxelPlanet::residency::", $path));
            }};
        }
        macro_rules! dispatch {
            ($path:literal, $pipeline:ident, $groups:expr) => {
                scope!($path, {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(concat!("VoxelPlanet::residency::", $path)),
                        ..Default::default()
                    });
                    pass.set_bind_group(0, &self.gen_group, &[]);
                    Self::dispatch(&mut pass, &self.pipelines.$pipeline, $groups);
                });
            };
        }
        let wg = |n: u32| n.div_ceil(64);
        if evictions > 0 || patches > 0 || block_patches > 0 {
            scope!("maintenance", {
                if evictions > 0 {
                    dispatch!("maintenance::evict", evict, [wg(evictions), 1, 1]);
                }
                if patches > 0 || block_patches > 0 {
                    scope!("maintenance::patch", {
                        if patches > 0 {
                            dispatch!("maintenance::patch::table", patch, [wg(patches), 1, 1]);
                        }
                        if block_patches > 0 {
                            dispatch!("maintenance::patch::summary_blocks", patch_blocks, [wg(block_patches), 1, 1]);
                        }
                    });
                }
            });
        }
        if jobs > 0 {
            let groups = [jobs.min(32_768), jobs.div_ceil(32_768), 1];
            scope!("admission", {
                if let Some(p) = &mut self.profiler {
                    p.begin_pass(encoder, "planet_generate");
                }
                dispatch!("admission::generate", generate, groups);
                if let Some(p) = &mut self.profiler {
                    p.end_pass(encoder, "planet_generate");
                }
                scope!("admission::allocation", {
                    dispatch!("admission::allocation::count", count, [wg(jobs), 1, 1]);
                    dispatch!("admission::allocation::refill", refill, [1, 1, 1]);
                    dispatch!("admission::allocation::allocate", allocate, [wg(jobs), 1, 1]);
                    dispatch!("admission::allocation::fixup", fixup, [1, 1, 1]);
                });
                scope!("admission::publication", {
                    dispatch!("admission::publication::publish", publish, groups);
                    dispatch!("admission::publication::level_suffix", level_suffix, [1, 1, 1]);
                });
            });
        }
    }

    /// Take the finished residency plan, whose work this frame uploads, and
    /// request the next one, which the worker plans while this frame is
    /// encoded and executed. Returns the work and the readback reserved for
    /// its jobs.
    fn exchange_plan(&mut self, eye: DVec3, lod0: f64, moving: bool) -> (FrameWork, Option<usize>) {
        let mut work = FrameWork::default();
        let mut readback = None;
        if let Some(mut result) = self.residency.try_take() {
            work = std::mem::take(&mut result.work);
            readback = self.plan_readback.take();
            self.stats.plan_cpu_ms = result.plan_ms;
            result.probe = result.probe.or(self.plan.probe);
            self.plan = result;
        } else if self.residency.in_flight() {
            self.stats.late_plans += 1;
        }
        if !self.residency.in_flight() && self.wants_plan(eye, lod0) {
            // Generation budget: small while the view moves (frame pacing),
            // large when it is still (fast convergence), from the measured
            // job cost. Every job's outcome must reach the CPU (a failure the
            // CPU never sees leaves a resident hole that is never retried):
            // without a free readback to reserve, the plan issues no jobs.
            let target_ms = if moving { 3.0 } else { 12.0 };
            let free = self.readbacks.iter().position(|r| r.stage == 0);
            // Work buys generation time at its measured cost (`unit_cost`);
            // the floor (a few cave columns or a few dozen heightfield ones)
            // keeps refinement going whatever the estimate; only scratch
            // pressure lowers it. The job count only caps the buffers.
            let units = (target_ms / self.ms_per_unit.max(1e-5) * self.scratch_scale).max(32.0 * self.scratch_scale);
            let budget = JobBudget {
                units: if free.is_some() { units } else { 0.0 },
                jobs: self.settings.job_budget.min(self.settings.capacity.max_jobs as usize),
            };
            if let Some(index) = free {
                self.readbacks[index].stage = 3;
                self.plan_readback = Some(index);
            }
            // Most of the time until the next frame takes the result (a late
            // result costs a frame without uploads).
            let cpu_ms = (self.frame_ms * 0.6).clamp(if moving { 1.5 } else { 4.0 }, 12.0);
            self.stats.unit_budget = budget.units;
            self.residency.submit(PlanRequest {
                planet: self.planet.clone(),
                eye,
                lod0,
                lod_dither: f64::from(self.settings.lod_dither),
                budget,
                cpu_budget: std::time::Duration::from_secs_f64(cpu_ms * 1.0e-3),
                failed: std::mem::take(&mut self.failed),
                table: self.settings.table_snapshots,
                probe: self.want_probe.swap(false, Ordering::Relaxed),
            });
            self.submitted = Some((self.planet.clone(), eye, lod0));
        }
        (work, readback)
    }

    /// Whether a plan can change anything: residency is still busy, job
    /// outcomes wait, or the edits or view changed since the last request
    /// (the residency replans windows once the eye moves two voxels).
    fn wants_plan(&self, eye: DVec3, lod0: f64) -> bool {
        !self.plan.idle
            || !self.failed.is_empty()
            || self.submitted.as_ref().is_none_or(|(planet, at, requested)| {
                !Arc::ptr_eq(planet, &self.planet)
                    || at.distance(eye) > self.planet.grid().voxel_size()
                    || (requested - lod0).abs() > lod0 * 0.005
            })
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
        self.encode_profiled(encoder, camera_data, frame, size, gbuffer, depth, frame_num, None);
    }

    // Graph scopes share the enclosing VoxelPlanet pass's query set/readback.
    // The private profiler remains responsible for generation budgeting.
    #[allow(clippy::too_many_arguments)]
    fn encode_profiled(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        camera_data: &helio_core::GpuCameraUniforms,
        frame: &PlanetFrame,
        size: [u32; 2],
        gbuffer: [&wgpu::TextureView; 8],
        depth: &wgpu::TextureView,
        frame_num: u64,
        mut graph_context: Option<&mut PassContext<'_>>,
    ) {
        macro_rules! begin_stage {
            ($stage:literal) => {
                if let Some(p) = &mut self.profiler {
                    p.begin_pass(encoder, concat!("planet_", $stage));
                }
                if let Some(ctx) = graph_context.as_deref_mut() {
                    ctx.begin_gpu_scope(encoder, concat!("VoxelPlanet::", $stage));
                }
            };
        }
        macro_rules! end_stage {
            ($stage:literal) => {
                if let Some(ctx) = graph_context.as_deref_mut() {
                    ctx.end_gpu_scope(encoder, concat!("VoxelPlanet::", $stage));
                }
                if let Some(p) = &mut self.profiler {
                    p.end_pass(encoder, concat!("planet_", $stage));
                }
            };
        }
        if let Some(p) = &mut self.profiler {
            // Timestamps arrive frames late and the same sample is returned
            // until a newer one completes: each sample is used once, with the
            // job count of the frame it measured. (Dividing by the last
            // frame's jobs overestimated the cost 2-7x, most in the editor.)
            let generate: f64 = p
                .read_timestamps_deferred()
                .iter()
                .filter(|t| t.name == "planet_generate")
                .map(|t| t.duration_ns as f64 / 1.0e6)
                .sum();
            let completed = p.last_completed_frame();
            if completed.is_some() && completed != self.costed_frame {
                self.costed_frame = completed;
                let units = self.frame_units.iter().find(|(f, _)| Some(*f) == completed).map_or(0.0, |(_, u)| *u);
                if units > 0.0 && generate > 0.0 {
                    if self.cost_samples.len() == 32 {
                        self.cost_samples.pop_front();
                    }
                    self.cost_samples.push_back((units, generate));
                    if let Some(cost) = unit_cost(&self.cost_samples) {
                        self.ms_per_unit = cost;
                    }
                }
            }
        }
        // An edit-only publication keeps the recipe/pipelines but changes
        // the authoritative journal. Direct renderer users need the same
        // synchronization as PlanetPass's mailbox path.
        if !Arc::ptr_eq(&self.planet, &frame.planet) {
            assert_eq!(self.planet.recipe(), frame.planet.recipe(), "recreate PlanetRenderer after a recipe change");
            self.planet = frame.planet.clone();
        }
        self.frame_index = self.frame_index.wrapping_add(1);
        self.last_frame_num = frame_num;
        if self.screen.size != size {
            self.screen = Screen::new(&self.device, size);
        }
        self.poll_readbacks();
        self.poll_picks();
        let tan_half = 1.0 / f64::from(camera_data.proj[5]).abs().max(1e-6);
        let lod0 = Residency::lod_distance(self.planet.grid(), tan_half, size[1], f64::from(self.settings.lod_pixels)) / self.lod_pressure;
        let now = std::time::Instant::now();
        if let Some(last) = self.last_encode {
            // Idle editor gaps are not frame time.
            let ms = (now - last).as_secs_f64() * 1000.0;
            self.frame_ms = self.frame_ms * 0.9 + ms.min(50.0) * 0.1;
        }
        self.last_encode = Some(now);
        let moving = self.last_eye.is_none_or(|e| e.distance(frame.eye) > 0.01);
        self.last_eye = Some(frame.eye);
        self.update_lod_pressure(frame_num, moving);
        // Frozen: no plan is taken or started, so renders see identical GPU
        // state (an in-flight plan waits).
        let (work, plan_readback) = if self.settings.freeze_residency {
            (FrameWork::default(), None)
        } else {
            self.exchange_plan(frame.eye, lod0, moving)
        };
        self.stats.us_per_unit = self.ms_per_unit * 1000.0;
        self.stats.units = work.units;
        if self.frame_units.len() == 16 {
            self.frame_units.pop_front();
        }
        self.frame_units.push_back((frame_num, work.units));
        let uploading = std::time::Instant::now();
        let (patches, block_patches) = self.upload(&work);
        if let Some(live) = self.plan.live_blocks.take() {
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
                self.queue.write_buffer(&self.buffers.live_blocks, 0, bytemuck::cast_slice(&live));
            }
        }
        let live_blocks = self.plan.live_block_count as u32;
        self.stats.upload_cpu_ms = uploading.elapsed().as_secs_f64() * 1000.0;
        let encoding = std::time::Instant::now();
        let jobs = work.jobs.len() as u32;
        let evictions = work.evictions.len() as u32;
        let mut uniform = self.frame_uniform(frame.eye, size, lod0, jobs, evictions, frame.sun, frame.shadows);
        uniform.extra[0] = patches;
        uniform.extra[1] = crate::residency::block_region();
        uniform.extra[2] = block_patches;
        uniform.extra[3] = live_blocks;
        uniform.hints[0] = u32::from(self.settings.residency_hints && self.plan.blocks_exact);
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
                wgpu::BindGroupEntry { binding: 5, resource: self.buffers.baked.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.buffers.edit_refs.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 20, resource: self.buffers.brushes.as_entire_binding() },
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
        // Before the timed residency stage: its time measures job cost.
        if self.pool_pressure && frame_num >= self.last_recycle + RECYCLE_INTERVAL {
            self.pool_pressure = false;
            self.last_recycle = frame_num;
            self.recycle_pages(encoder);
        }
        begin_stage!("residency");
        if let Some(ctx) = graph_context.as_deref_mut().filter(|ctx| ctx.gpu_scopes_enabled()) {
            self.encode_residency_detailed(encoder, ctx, jobs, evictions, patches, block_patches);
        } else {
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
                // Generation in its own pass: its timestamps size the budget.
                drop(pass);
                let groups = [jobs.min(32_768), jobs.div_ceil(32_768), 1];
                if let Some(p) = &mut self.profiler {
                    p.begin_pass(encoder, "planet_generate");
                }
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_bind_group(0, &self.gen_group, &[]);
                    Self::dispatch(&mut pass, &self.pipelines.generate, groups);
                }
                if let Some(p) = &mut self.profiler {
                    p.end_pass(encoder, "planet_generate");
                }
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_bind_group(0, &self.gen_group, &[]);
                Self::dispatch(&mut pass, &self.pipelines.count, [wg(jobs), 1, 1]);
                Self::dispatch(&mut pass, &self.pipelines.refill, [1, 1, 1]);
                Self::dispatch(&mut pass, &self.pipelines.allocate, [wg(jobs), 1, 1]);
                Self::dispatch(&mut pass, &self.pipelines.fixup, [1, 1, 1]);
                Self::dispatch(&mut pass, &self.pipelines.publish, groups);
                Self::dispatch(&mut pass, &self.pipelines.level_suffix, [1, 1, 1]);
            }
        }
        end_stage!("residency");
        let camera_group = &self.camera_group;
        // Allocator counters and this frame's failed jobs (into the readback
        // reserved when the plan was requested; without a free one its job
        // budget was 0), then the failure list restarts. Counters alone are
        // sampled every 30 frames.
        let readback = match plan_readback {
            Some(index) if jobs > 0 => Some(index),
            reserved => {
                if let Some(index) = reserved {
                    self.readbacks[index].stage = 0;
                }
                self.readbacks.iter().position(|r| r.stage == 0).filter(|_| frame_num % 30 == 0)
            }
        };
        debug_assert!(jobs == 0 || readback.is_some(), "jobs without a readback");
        if let Some(r) = readback.map(|index| &mut self.readbacks[index]) {
            encoder.copy_buffer_to_buffer(&self.buffers.alloc, 0, &r.buffer, 0, PROBE_BYTES);
            if jobs > 0 {
                encoder.copy_buffer_to_buffer(&self.buffers.failures, 0, &r.buffer, PROBE_BYTES, u64::from(jobs) * FAILURE_BYTES);
                encoder.clear_buffer(&self.buffers.alloc, A_FAILS * 4, Some(4));
            }
            r.entries = jobs;
            r.stage = 1;
        }
        let groups = [size[0].div_ceil(8), size[1].div_ceil(8), 1];
        begin_stage!("horizon");
        {
            // Directional sky bound from this frame's summary blocks.
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &trace_group, &[]);
            pass.set_bind_group(1, camera_group, &[]);
            Self::dispatch(&mut pass, &self.pipelines.horizon_clear, [((HORIZON_SECTORS + HORIZON_GROUPS) * HORIZON_BUCKETS).div_ceil(64), 1, 1]);
            Self::dispatch(&mut pass, &self.pipelines.horizon_blocks, [live_blocks.div_ceil(64), 1, 1]);
            Self::dispatch(&mut pass, &self.pipelines.horizon_suffix, [1, 1, 1]);
        }
        end_stage!("horizon");
        begin_stage!("primary");
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &trace_group, &[]);
            pass.set_bind_group(1, camera_group, &[]);
            Self::dispatch(&mut pass, &self.pipelines.primary, groups);
        }
        if let Some(picks) = &frame.picks {
            Self::copy_picks(&mut self.picks, &self.screen.hits, encoder, picks, size);
        }
        end_stage!("primary");
        begin_stage!("shade");
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &trace_group, &[]);
            pass.set_bind_group(1, camera_group, &[]);
            Self::dispatch(&mut pass, &self.pipelines.shade, groups);
        }
        end_stage!("shade");
        if self.settings.sky_occlusion {
            begin_stage!("skylight");
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_bind_group(0, &trace_group, &[]);
                pass.set_bind_group(1, camera_group, &[]);
                Self::dispatch(&mut pass, &self.pipelines.skylight, [size[0].div_ceil(32), size[1].div_ceil(32), 1]);
            }
            end_stage!("skylight");
        }
        begin_stage!("gbuffer");
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
        end_stage!("gbuffer");
        self.sun_active = frame.shadows;
        if frame.shadows {
            begin_stage!("sunlight");
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_bind_group(0, &trace_group, &[]);
                pass.set_bind_group(1, camera_group, &[]);
                Self::dispatch(&mut pass, &self.pipelines.sunlight, [size[0].div_ceil(16), size[1].div_ceil(16), 1]);
            }
            end_stage!("sunlight");
        }
        if let Some(p) = &mut self.profiler {
            p.resolve_queries(encoder, frame_num);
        }
        self.stats.encode_cpu_ms = encoding.elapsed().as_secs_f64() * 1000.0;
        let rs = self.plan.stats;
        self.stats.resident_columns = rs.resident_columns;
        self.stats.pending_columns = rs.pending_columns;
        self.stats.jobs = jobs as usize;
        self.stats.evictions = evictions as usize;
        self.stats.active_levels = rs.active_levels;
        self.stats.finest_level = rs.finest_level;
        self.stats.window_rebuild_ms = rs.window_rebuild_ms;
        self.stats.table_refused = rs.table_refused;
        self.stats.reranked = rs.reranked;
        self.stats.baked_bricks = rs.baked_bricks;
        self.stats.baked_pool = self.buffers.baked_slots;
        self.stats.brushes = rs.brushes;
        self.stats.edit_words = rs.edit_words;
        self.stats.lod0_distance = lod0;
        self.stats.pool_pages = self.settings.capacity.pool_units / 512;
        self.stats.logical_bytes = self.buffers.bytes + u64::from(size[0]) * u64::from(size[1]) * (32 + 16 + 8 + 4);
        if self.plan.idle {
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
    /// The frame `active` draws: the source's while it serves the same
    /// recipe, else the last one it did (the previous terrain stays on
    /// screen while the new one's pipelines compile).
    shown: Option<PlanetFrame>,
    /// Pipelines compiling on a worker thread for a shape and program.
    compiling: Option<CompilingPipelines>,
    /// The last compiled pipelines, kept while no world is shown, so a
    /// world of the same program comes back without compiling.
    compiled: Option<Arc<Pipelines>>,
    profiling: bool,
}

/// Pipelines for a world shape and terrain program, compiled on a worker
/// thread: cold, they take seconds, which used to freeze the host at start
/// and on every terrain change.
struct CompilingPipelines {
    plane: bool,
    program: String,
    done: Arc<OnceLock<Arc<Pipelines>>>,
}

impl CompilingPipelines {
    fn start(device: &wgpu::Device, plane: bool, program: crate::TerrainProgram) -> Self {
        let done = Arc::new(OnceLock::new());
        let (device, cell) = (device.clone(), done.clone());
        let key = program.key.to_string();
        std::thread::Builder::new()
            .name("voxel pipelines".into())
            .spawn(move || {
                let _ = cell.set(Arc::new(Pipelines::new(&device, plane, &program)));
            })
            .expect("spawn the voxel pipeline compiler");
        Self { plane, program: key, done }
    }
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
            shown: None,
            compiling: None,
            compiled: None,
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
    /// Change the art without touching residency or the world (`None`: the
    /// terrain generator's own). Returns true when it changed (temporal
    /// colour history should be reset).
    pub fn set_appearance(&mut self, appearance: Option<TerrainAppearance>) -> bool {
        if self.settings.appearance == appearance {
            return false;
        }
        self.settings.appearance = appearance;
        if let Some(renderer) = &mut self.active {
            renderer.settings.appearance = appearance;
        }
        true
    }
    /// A host viewport should keep rendering until residency settles.
    pub fn needs_frame(&self) -> bool {
        let has_source = self.source.try_lock().map_or(true, |s| s.is_some());
        has_source && (self.compiling.is_some() || self.active.as_ref().is_some_and(|r| !r.settled()))
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
        self.shown = previous.shown.take();
        self.compiling = previous.compiling.take();
        self.compiled = previous.compiled.take();
        // A graph rebuild must not revert the host's art settings.
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
        let Some(frame) = frame else {
            self.active = None;
            self.shown = None;
            self.compiling = None;
            return Ok(());
        };
        let same = self.active.as_ref().is_some_and(|r| {
            Arc::ptr_eq(r.planet(), &frame.planet) || r.planet().recipe() == frame.planet.recipe()
        });
        if same {
            // Same recipe: adopt the newer edit state without rebuilding.
            if let Some(r) = &mut self.active {
                r.planet = frame.planet.clone();
            }
            self.compiling = None;
            self.shown = Some(frame);
            return Ok(());
        }
        let plane = frame.planet.grid().is_plane();
        let program = frame.planet.field().program();
        let ready = match (&self.compiled, &self.compiling) {
            (Some(p), _) if p.serve(plane, &program) => Some(p.clone()),
            (_, Some(c)) if c.plane == plane && c.program == program.key => c.done.get().cloned(),
            _ => {
                self.compiling = Some(CompilingPipelines::start(ctx.device, plane, program));
                None
            }
        };
        match ready {
            Some(pipelines) => {
                let mut renderer = PlanetRenderer::with_pipelines(
                    pipelines.clone(),
                    ctx.device,
                    ctx.queue,
                    frame.planet.clone(),
                    self.settings,
                    [ctx.width, ctx.height],
                );
                renderer.set_profiling(self.profiling);
                self.active = Some(renderer);
                self.compiling = None;
                self.compiled = Some(pipelines);
                self.shown = Some(frame);
            }
            // Until then the previous terrain (if any) stays, seen from
            // the new frame's eye.
            None => {
                self.shown = self.active.as_ref().map(|r| PlanetFrame { planet: r.planet().clone(), ..frame });
            }
        }
        Ok(())
    }
    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let Some(renderer) = &mut self.active else { return Ok(()) };
        let Some(frame) = self.shown.clone() else { return Ok(()) };
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
        renderer.encode_profiled(
            encoder,
            ctx.camera_data,
            &frame,
            [ctx.width, ctx.height],
            targets,
            ctx.depth,
            ctx.frame_num,
            Some(ctx),
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
/// Diagnostics: GPU time (ns) per column of the terrain program's height and
/// surface word, over a `side` x `side` patch of level-0 columns around
/// `eye`'s cell, dispatched `repeats` times in one submission.
pub fn time_field(device: &wgpu::Device, queue: &wgpu::Queue, planet: &Planet, eye: glam::DVec3, side: u32, repeats: u32) -> f64 {
    let grid = *planet.grid();
    let program = planet.field().program();
    let (cell, _) = grid.locate(eye);
    let kernel = "
@group(0) @binding(20) var<storage, read> time_in: array<vec4<i32>>;
@group(0) @binding(22) var<storage, read_write> time_out: array<i32>;
@compute @workgroup_size(64) fn time_field(@builtin(global_invocation_id) id: vec3<u32>) {
    let a = time_in[0];
    let side = u32(a.w);
    if id.x >= side * side { return; }
    let i = a.y + i32(id.x % side);
    let j = a.z + i32(id.x / side);
    let p = domain_point(u32(a.x), i, j, 0u);
    let height = field_height(u32(a.x), i, j, 0u);
    let surface = terrain_surface(p, u32(world.grid.w), height);
    time_out[id.x] = height ^ i32(surface);
}
";
    let module = helio_core::shader::module(device, "terrain timing", &source("read", &[kernel], grid.is_plane(), &program));
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("terrain timing"),
        layout: None,
        module: &module,
        entry_point: Some("time_field"),
        compilation_options: Default::default(),
        cache: None,
    });
    let init = |label, contents: &[u8], usage| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage });
    let world = init("time world", bytemuck::bytes_of(&WorldGpu::new(planet)), wgpu::BufferUsages::UNIFORM);
    let terrain = init("time terrain", &terrain_bytes(&program), wgpu::BufferUsages::UNIFORM);
    let half = side as i32 / 2;
    let input = IVec4::new(i32::from(cell.face), cell.i - half, cell.j - half, side as i32);
    let ins = init("time input", bytemuck::bytes_of(&input), wgpu::BufferUsages::STORAGE);
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("time out"),
        size: u64::from(side * side) * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("terrain timing"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 1, resource: world.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 16, resource: terrain.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 20, resource: ins.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 22, resource: out.as_entire_binding() },
        ],
    });
    let run = |count: u32| {
        let mut encoder = device.create_command_encoder(&Default::default());
        for _ in 0..count {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((side * side).div_ceil(64), 1, 1);
        }
        let started = std::time::Instant::now();
        queue.submit([encoder.finish()]);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        started.elapsed().as_secs_f64()
    };
    run(2); // warm up (compile, clocks)
    let seconds = run(repeats);
    seconds * 1e9 / (f64::from(side * side) * f64::from(repeats))
}

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
@group(0) @binding(22) var<storage, read_write> verify_out: array<vec4<i32>>;
@compute @workgroup_size(64) fn verify(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&verify_in) { return; }
    let a = verify_in[id.x];
    let e = verify_extra[id.x];
    let level = u32(a.w);
    let p = domain_point(u32(a.x), a.y, a.z, level);
    let height = field_height(u32(a.x), a.y, a.z, level);
    // A layer near the column top (inside any volumetric extent).
    let top = top_cells(height, level);
    let extent = terrain_extent(p, level);
    let k = top - extent.x - 1 + rem_floor(e.w, max(extent.x + extent.y + 2, 1));
    let q = volume_point(u32(a.x), a.y, a.z, k, level);
    // A leaning height around the surface (exercising the overhang clamp).
    let density = terrain_density(p, q, level, top, height, height + (e.z & 8191) - 4096, k);
    let lean = terrain_lean(level);
    let offset = terrain_lean_offset(p, a.y, a.z, k, level);
    let surface = terrain_surface(p, level + u32(world.grid.w), height) & 0xffu;
    verify_out[id.x] = vec4<i32>(height, i32(ground_material(p, surface, e.x, e.y, e.z, e.w)),
        density, (q.x ^ q.y ^ q.z) + i32(surface) * 7919 + extent.x * 65599 + extent.y * 257
            + lean.x * 7 + lean.y * 13 + offset.x * 31 + offset.y * 131);
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
    let bytes = u64::from(samples) * 16;
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
    let gpu: &[[i32; 4]] = bytemuck::cast_slice(&data[..bytes as usize]);
    for ((a, e), g) in inputs.iter().zip(&extra).zip(gpu) {
        let level = a.w as u32;
        let p = grid.domain_point(a.x as u8, a.y, a.z, level);
        let height = field.height(p, level + grid.level_offset());
        let top = crate::terrain::top_cells(&grid, height, level);
        let (below, above) = field.extent(p, level);
        let k = top - below - 1 + e.w.rem_euclid((below + above + 2).max(1));
        let q = grid.volume_point(a.x as u8, a.y, a.z, k, level);
        let density = field.density(p, q, level, top, height, height.wrapping_add(e.z & 8191).wrapping_sub(4096), k);
        let lean = field.lean(level);
        let offset = field.lean_offset(p, a.y, a.z, k, level);
        let surface = field.surface(p, level + grid.level_offset(), height) & 0xff;
        let cpu = [
            height,
            field.ground_material(p, surface, e.x, e.y, e.z, e.w) as i32,
            density,
            (q.x ^ q.y ^ q.z).wrapping_add(surface as i32 * 7919).wrapping_add(below.wrapping_mul(65599)).wrapping_add(above.wrapping_mul(257))
                .wrapping_add(lean.0 * 7).wrapping_add(lean.1 * 13).wrapping_add(offset.0.wrapping_mul(31)).wrapping_add(offset.1.wrapping_mul(131)),
        ];
        if *g != cpu {
            return Err(format!("column {a} with inputs {e}: GPU {g:?}, CPU {cpu:?}"));
        }
    }
    Ok(())
}

/// GPU generation cost per work unit (ms) from recent (units, generate ms)
/// samples: the window's time over its units, raised at once to the latest
/// frame's when that one cost more a unit (work the unit model misjudges
/// must shrink the next budget now, not after the window averaged it in).
fn unit_cost(samples: &std::collections::VecDeque<(f64, f64)>) -> Option<f64> {
    let units: f64 = samples.iter().map(|s| s.0).sum();
    if samples.len() < 4 || units < 256.0 {
        return None;
    }
    let average = samples.iter().map(|s| s.1).sum::<f64>() / units;
    let latest = samples.back().filter(|s| s.0 >= 64.0).map_or(0.0, |s| s.1 / s.0);
    Some(average.max(latest).clamp(1.0e-4, 0.05))
}

#[cfg(test)]
mod cost_tests {
    use super::unit_cost;
    use std::collections::VecDeque;

    #[test]
    fn unit_cost_is_the_window_average_raised_by_a_costlier_latest_frame() {
        // Too few jobs to judge.
        assert!(unit_cost(&VecDeque::from([(10.0, 0.01); 8])).is_none());
        // Steady heightfield columns: 1 us a unit.
        let mut samples: VecDeque<(f64, f64)> = (0..8).map(|i| (1000.0 + 100.0 * f64::from(i), (1000.0 + 100.0 * f64::from(i)) * 0.001)).collect();
        assert!((unit_cost(&samples).unwrap() - 0.001).abs() < 1e-9);
        // A frame the unit model underestimated (4 us a unit) raises it at once.
        samples.push_back((1000.0, 4.0));
        assert!((unit_cost(&samples).unwrap() - 0.004).abs() < 1e-9);
        // A cheap frame afterwards keeps the window's (raised) average.
        samples.push_back((1000.0, 0.5));
        let cost = unit_cost(&samples).unwrap();
        let average = samples.iter().map(|s| s.1).sum::<f64>() / samples.iter().map(|s| s.0).sum::<f64>();
        assert!((cost - average).abs() < 1e-12 && cost > 0.001);
        // A handful of units cannot raise it on their own.
        samples.push_back((8.0, 0.2));
        assert!(unit_cost(&samples).unwrap() < 0.01);
    }
}
