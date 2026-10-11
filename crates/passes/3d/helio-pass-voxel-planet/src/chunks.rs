//! Chunk trees (`docs/chunk-trees.md`), phase 1: generation of chunk trees
//! from the terrain contract and primary visibility through rasterized
//! chunk boxes, measured against the span-column renderer.
//!
//! A tile is the 64x64 columns of one level under a stack of chunks; a
//! chunk is 64^3 cells of one level stored as a 64-tree. [`ChunkWorld::build`]
//! selects the cut of the tile quadtree whose cells project to at most
//! `lod_pixels` pixels and generates every chunk under it; [`ChunkWorld::draw`]
//! rasterizes the chunks' boxes to seed each pixel with the nearest box its
//! ray enters, then traces each pixel's ray through that chunk's tree in a
//! compute pass (shaders/chunk_draw.wgsl).

use crate::edits::FaceBrush;
use crate::engine::{terrain_bytes, terrain_source, WorldGpu};
use crate::grid::Grid;
use crate::planet::Planet;
use bytemuck::{Pod, Zeroable};
use glam::{DMat3, DVec3, Mat4, Vec3};
use wgpu::util::DeviceExt;

/// Cells along a chunk edge (and columns along a tile edge).
pub const CHUNK: i32 = 64;
const TILE_COLUMNS: u64 = 4096;
/// Tiles generated per dispatch (their columns are scratch until their
/// chunks are built).
const TILE_BATCH: usize = 512;
/// Chunks generated per dispatch (column masks are scratch).
const CHUNK_BATCH: usize = 512;
/// Chunk slots.
pub const MAX_CHUNKS: usize = 131072;
/// Leaves of every chunk together (8 bytes each).
const LEAF_CAPACITY: u64 = 32 << 20;
/// Chunk table slots (at least twice the chunks).
const TABLE_SLOTS: usize = 2 * MAX_CHUNKS;
/// Draws whose timings can be pending at once.
pub const TIMING_RING: u64 = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TileJob {
    face: u32,
    level: u32,
    cx: i32,
    cy: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ChunkJob {
    tile: u32,
    cz: i32,
    slot: u32,
    brushes: u32,
    brush_count: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct ChunkMeta {
    root_occ: [u32; 2],
    root_full: [u32; 2],
    z: [i32; 2],
    leaves: u32,
    status: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ChunkDraw {
    m: [[f32; 4]; 3],
    inv: [[f32; 4]; 3],
    /// Lowest and one past the highest occupied layer, slot (bits), flags
    /// (bits; 1: mirror x).
    bounds: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct DrawParams {
    view_proj: [[f32; 4]; 4],
    inverse_view_proj: [[f32; 4]; 4],
    /// Direction to the sun, near plane.
    sun: [f32; 4],
    /// x: colour by level (`HELIO_VOXEL_CHUNKS_LEVELS`).
    debug: [u32; 4],
    /// Chunk table slots - 1, base cells per face edge.
    table: [u32; 4],
}

/// A tile of the quadtree: 64x64 columns of `level` on `face`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileKey {
    pub face: u8,
    pub level: u32,
    pub cx: i32,
    pub cy: i32,
}

/// A chunk's affine frame: world = origin + m * cell coordinates.
#[derive(Clone, Copy, Debug)]
struct Frame {
    origin: DVec3,
    m: DMat3,
    inv: DMat3,
}

/// The affine frame of chunk `cz` over `tile`: the grid mapping's central
/// differences across the chunk, about its centre. Within a chunk it is
/// linear to `512 x voxel / radius` of a cell (docs/chunk-trees.md).
fn chunk_frame(grid: &Grid, tile: TileKey, cz: i32) -> Frame {
    let s = f64::from(1u32 << tile.level);
    let half = f64::from(CHUNK / 2) * s;
    let centre = [f64::from(tile.cx * CHUNK) * s + half, f64::from(tile.cy * CHUNK) * s + half, f64::from(cz * CHUNK) * s + half];
    let at = |d: [f64; 3]| grid.position(tile.face, [centre[0] + d[0], centre[1] + d[1], centre[2] + d[2]]);
    let n = f64::from(CHUNK);
    let u = (at([half, 0.0, 0.0]) - at([-half, 0.0, 0.0])) / n;
    let v = (at([0.0, half, 0.0]) - at([0.0, -half, 0.0])) / n;
    let w = (at([0.0, 0.0, half]) - at([0.0, 0.0, -half])) / n;
    let m = DMat3::from_cols(u, v, w);
    Frame { origin: at([0.0; 3]) - m * DVec3::splat(n / 2.0), m, inv: m.inverse() }
}

/// Distance from `eye` to the tile's columns between layers `range` (level
/// cells): to the nearest point in index space when the eye is over the
/// tile's face, else to its centre less its half-diagonal.
fn tile_distance(grid: &Grid, tile: TileKey, range: (i32, i32), eye: DVec3) -> f64 {
    let s = f64::from(1u32 << tile.level);
    let (i0, j0) = (f64::from(tile.cx * CHUNK) * s, f64::from(tile.cy * CHUNK) * s);
    let w = f64::from(CHUNK) * s;
    let (k0, k1) = (f64::from(range.0) * s, f64::from(range.1) * s);
    if let Some([ei, ej, ek]) = grid.face_coords(tile.face, eye) {
        let p = grid.position(tile.face, [ei.clamp(i0, i0 + w), ej.clamp(j0, j0 + w), ek.clamp(k0, k1)]);
        return (p - eye).length();
    }
    let c = grid.position(tile.face, [i0 + w / 2.0, j0 + w / 2.0, (k0 + k1) / 2.0]);
    let corner = grid.position(tile.face, [i0, j0, k0]);
    ((c - eye).length() - (corner - c).length()).max(0.0)
}

/// Whether `fb` can touch cells of `level` with half-cell bounds `i`, `j`,
/// `k` (inclusive).
fn brush_meets(fb: &FaceBrush, face: u8, level: u32, i: (i64, i64), j: (i64, i64), k: (i64, i64)) -> bool {
    if fb.face() != face || fb.radius_half < (1u32 << level) {
        return false;
    }
    let w = i64::from(fb.center[3]);
    let (ci, cj) = (i64::from(fb.center[0]), i64::from(fb.center[1]));
    ci + w >= i.0 && ci - w <= i.1 && cj + w >= j.0 && cj - w <= j.1 && i64::from(fb.k_hi) >= k.0 && i64::from(fb.k_lo) <= k.1
}

struct Chunk {
    tile: TileKey,
    cz: i32,
    slot: u32,
    frame: Frame,
    z: (i32, i32),
}

/// Results of [`ChunkWorld::build`].
#[derive(Clone, Copy, Debug, Default)]
pub struct BuildStats {
    pub tiles: usize,
    pub chunks: usize,
    pub empty_chunks: usize,
    pub leaves: u64,
    /// Bytes of the stored trees: leaves, mid nodes, roots, materials.
    pub bytes: u64,
    pub seconds: f64,
    pub per_level: [u32; 24],
    /// Tiles kept (not refined) per level, and the layers their chunk
    /// stacks span in all.
    pub tiles_per_level: [u32; 24],
    pub span_per_level: [u64; 24],
    pub overflow: bool,
}

/// Chunk trees for one planet and one view (phase 1: built at once for a
/// still eye).
pub struct ChunkWorld {
    device: wgpu::Device,
    queue: wgpu::Queue,
    gen_layout: wgpu::BindGroupLayout,
    tile_columns: wgpu::ComputePipeline,
    chunk_columns: wgpu::ComputePipeline,
    chunk_build: wgpu::ComputePipeline,
    draw_layout: wgpu::BindGroupLayout,
    trace_layout: wgpu::BindGroupLayout,
    seed_layout: wgpu::BindGroupLayout,
    resolve_layout: wgpu::BindGroupLayout,
    seed_outside: wgpu::RenderPipeline,
    seed_inside: wgpu::RenderPipeline,
    trace: wgpu::ComputePipeline,
    resolve: wgpu::RenderPipeline,
    targets: Option<Targets>,
    /// Rounds of seed and trace per draw.
    rounds: u32,
    world: wgpu::Buffer,
    terrain: wgpu::Buffer,
    tile_jobs: wgpu::Buffer,
    tile_data: wgpu::Buffer,
    tile_ranges: wgpu::Buffer,
    chunk_jobs: wgpu::Buffer,
    column_masks: wgpu::Buffer,
    chunk_meta: wgpu::Buffer,
    mids: wgpu::Buffer,
    mid_leaves: wgpu::Buffer,
    leaves: wgpu::Buffer,
    leaf_alloc: wgpu::Buffer,
    column_materials: wgpu::Buffer,
    leaf_materials: wgpu::Buffer,
    mid_materials: wgpu::Buffer,
    params: wgpu::Buffer,
    palette: wgpu::Buffer,
    /// Chunk keys by slot, the chunk table and its slots - 1, and each
    /// slot's index among the drawn chunks.
    chunk_keys: wgpu::Buffer,
    chunk_table: wgpu::Buffer,
    table_mask: u32,
    /// Base cells along a face edge.
    face_cells: u32,
    slot_instance: wgpu::Buffer,
    instances: wgpu::Buffer,
    /// Timestamps of the last `TIMING_RING` draws: queries, resolve
    /// buffer, one readback buffer per draw.
    queries: Option<(wgpu::QuerySet, wgpu::Buffer, Vec<wgpu::Buffer>)>,
    /// Fragment, step and hit counts (`HELIO_VOXEL_CHUNKS_COUNT`).
    counters: wgpu::Buffer,
    chunks: Vec<Chunk>,
    drawn: (u32, u32),
}

/// The draw's targets: the seed (each pixel's nearest box) and the trace's
/// outputs.
pub struct Targets {
    pub size: [u32; 2],
    seed: wgpu::Texture,
    /// Box entries during the seed, then the hits' depth (Depth32Float).
    pub depth: wgpu::Texture,
    /// Chunk slot (u32::MAX: none) and cell, face (Rg32Uint).
    pub visibility: wgpu::Texture,
    /// Plain colour (Rgba8Unorm).
    pub colour: wgpu::Texture,
    /// Distance from the eye to the hit, 0 for none (R32Float).
    pub distance: wgpu::Texture,
    /// Where each ray left its last box without a hit (R32Float).
    exits: wgpu::Texture,
}

const SEED_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

impl Targets {
    fn new(device: &wgpu::Device, size: [u32; 2]) -> Self {
        let texture = |label: &str, format: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: usage | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let a = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let s = wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;
        Self {
            size,
            seed: texture("chunk seed", SEED_FORMAT, a),
            depth: texture("chunk depth", DEPTH_FORMAT, a),
            visibility: texture("chunk visibility", wgpu::TextureFormat::Rg32Uint, s),
            colour: texture("chunk colour", wgpu::TextureFormat::Rgba8Unorm, s),
            distance: texture("chunk distance", wgpu::TextureFormat::R32Float, s),
            exits: texture("chunk exits", wgpu::TextureFormat::R32Float, s | wgpu::TextureUsages::RENDER_ATTACHMENT),
        }
    }
}

fn view(texture: &wgpu::Texture) -> wgpu::TextureView {
    texture.create_view(&Default::default())
}

fn storage(binding: u32, read_only: bool, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

fn entry(binding: u32, b: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: b.as_entire_binding() }
}

fn texture_entry(binding: u32, sample_type: wgpu::TextureSampleType, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture { sample_type, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
        count: None,
    }
}

fn storage_texture(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format, view_dimension: wgpu::TextureViewDimension::D2 },
        count: None,
    }
}

fn uniform(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

/// The chunk shaders as compiled for a world form and terrain program:
/// generation, then drawing (with diagnostic counters when `counting`).
pub fn shader_sources(plane: bool, program: &crate::terrain::TerrainProgram, wide64: bool, counting: bool) -> [(&'static str, String); 2] {
    // Counters only in a counting build: side effects in a fragment shader
    // can turn early depth tests off.
    let count = |n: u32| if counting { format!("atomicAdd(&counters[{n}], 1u);") } else { String::new() };
    let draw = include_str!("../shaders/chunk_draw.wgsl")
        .replace("COUNTERS", if counting { "@group(0) @binding(9) var<storage, read_write> counters: array<atomic<u32>, 4>;" } else { "" })
        .replace("COUNT(0)", &count(0))
        .replace("COUNT(1)", &count(1))
        .replace("COUNT(2)", &count(2))
        .replace("COUNT(3)", &count(3));
    [("chunk generation", terrain_source(plane, program, wide64) + include_str!("../shaders/chunk_generate.wgsl")), ("chunk draw", draw)]
}

impl ChunkWorld {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, planet: &Planet) -> Self {
        let program = planet.field().program();
        let wide64 = device.features().contains(wgpu::Features::SHADER_INT64);
        let [(_, gen_source), (_, draw_source)] = shader_sources(planet.grid().is_plane(), &program, wide64, std::env::var_os("HELIO_VOXEL_CHUNKS_COUNT").is_some());
        let gen_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("chunk generation"), source: wgpu::ShaderSource::Wgsl(gen_source.into()) });
        let c = wgpu::ShaderStages::COMPUTE;
        let gen_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("chunk generation"),
            entries: &[
                uniform(1, c),
                uniform(16, c),
                storage(2, true, c),
                storage(3, false, c),
                storage(4, false, c),
                storage(5, true, c),
                storage(6, false, c),
                storage(7, true, c),
                storage(8, true, c),
                storage(9, false, c),
                storage(10, false, c),
                storage(11, false, c),
                storage(12, false, c),
                storage(13, false, c),
                storage(14, false, c),
                storage(15, false, c),
                storage(17, false, c),
            ],
        });
        let gen_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&gen_layout)], immediate_size: 0 });
        let compute = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&gen_pipeline_layout),
                module: &gen_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let tile_columns = compute("tile_columns_main");
        let chunk_columns = compute("chunk_columns_main");
        let chunk_build = compute("chunk_build_main");

        let draw_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("chunk draw"), source: wgpu::ShaderSource::Wgsl(draw_source.into()) });
        let all = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE;
        let draw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("chunk draw"),
            entries: &[
                uniform(0, all),
                storage(1, true, all),
                storage(2, true, all),
                storage(3, true, all),
                storage(4, true, all),
                storage(5, true, all),
                storage(6, true, all),
                uniform(7, all),
                storage(8, true, all),
                storage(9, false, wgpu::ShaderStages::COMPUTE),
                storage(10, true, all),
                storage(11, true, all),
                storage(12, true, all),
            ],
        });
        let trace_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("chunk trace"),
            entries: &[
                texture_entry(0, wgpu::TextureSampleType::Uint, wgpu::ShaderStages::COMPUTE),
                storage_texture(1, wgpu::TextureFormat::Rg32Uint),
                storage_texture(2, wgpu::TextureFormat::Rgba8Unorm),
                storage_texture(3, wgpu::TextureFormat::R32Float),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::ReadWrite, format: wgpu::TextureFormat::R32Float, view_dimension: wgpu::TextureViewDimension::D2 },
                    count: None,
                },
            ],
        });
        let seed_layout_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("chunk seed exits"),
            entries: &[texture_entry(5, wgpu::TextureSampleType::Float { filterable: false }, wgpu::ShaderStages::FRAGMENT)],
        });
        let resolve_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("chunk depth resolve"),
            entries: &[texture_entry(4, wgpu::TextureSampleType::Float { filterable: false }, wgpu::ShaderStages::FRAGMENT)],
        });
        let layout = |second: &wgpu::BindGroupLayout| device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&draw_layout), Some(second)], immediate_size: 0 });
        let seed_layout = layout(&seed_layout_1);
        let depth = |compare: wgpu::CompareFunction| {
            Some(wgpu::DepthStencilState { format: DEPTH_FORMAT, depth_write_enabled: Some(true), depth_compare: Some(compare), stencil: Default::default(), bias: Default::default() })
        };
        let seed = |fragment: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(fragment),
                layout: Some(&seed_layout),
                vertex: wgpu::VertexState { module: &draw_module, entry_point: Some("seed_vertex"), compilation_options: Default::default(), buffers: &[] },
                primitive: wgpu::PrimitiveState { cull_mode: Some(wgpu::Face::Back), front_face: wgpu::FrontFace::Ccw, ..Default::default() },
                depth_stencil: depth(wgpu::CompareFunction::Less),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState { module: &draw_module, entry_point: Some(fragment), compilation_options: Default::default(), targets: &[Some(SEED_FORMAT.into())] }),
                multiview_mask: None,
                cache: None,
            })
        };
        let seed_outside = seed("seed_fragment");
        let seed_inside = seed("seed_fragment_inside");
        let trace = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("chunk trace"),
            layout: Some(&layout(&trace_layout)),
            module: &draw_module,
            entry_point: Some("trace"),
            compilation_options: Default::default(),
            cache: None,
        });
        let resolve = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("chunk depth resolve"),
            layout: Some(&layout(&resolve_layout)),
            vertex: wgpu::VertexState { module: &draw_module, entry_point: Some("resolve_vertex"), compilation_options: Default::default(), buffers: &[] },
            primitive: Default::default(),
            depth_stencil: depth(wgpu::CompareFunction::Always),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState { module: &draw_module, entry_point: Some("resolve_depth"), compilation_options: Default::default(), targets: &[] }),
            multiview_mask: None,
            cache: None,
        });

        let buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
        let s = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let world = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("chunk world"), contents: bytemuck::bytes_of(&WorldGpu::new(planet)), usage: wgpu::BufferUsages::UNIFORM });
        let terrain = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("chunk terrain"), contents: &terrain_bytes(&program), usage: wgpu::BufferUsages::UNIFORM });
        let appearance = planet.field().appearance();
        let palette: Vec<[f32; 4]> = appearance.materials.iter().map(|m| [m.colour[0].powf(2.2), m.colour[1].powf(2.2), m.colour[2].powf(2.2), 1.0]).collect();
        let palette = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("chunk palette"), contents: bytemuck::cast_slice(&palette), usage: wgpu::BufferUsages::UNIFORM });
        let queries = device.features().contains(wgpu::Features::TIMESTAMP_QUERY).then(|| {
            (
                device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("chunk draw"), ty: wgpu::QueryType::Timestamp, count: 3 * TIMING_RING as u32 }),
                buffer("chunk timestamps", u64::from(wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT) * TIMING_RING, wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC),
                (0..TIMING_RING).map(|_| buffer("chunk timestamps read", 24, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST)).collect(),
            )
        });
        Self {
            device: device.clone(),
            queue: queue.clone(),
            gen_layout,
            tile_columns,
            chunk_columns,
            chunk_build,
            draw_layout,
            trace_layout,
            seed_layout: seed_layout_1,
            resolve_layout,
            seed_outside,
            seed_inside,
            trace,
            resolve,
            targets: None,
            rounds: std::env::var("HELIO_VOXEL_CHUNKS_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(1),
            world,
            terrain,
            tile_jobs: buffer("chunk tile jobs", (TILE_BATCH * 16) as u64, s),
            tile_data: buffer("chunk tile columns", TILE_BATCH as u64 * TILE_COLUMNS * 16, wgpu::BufferUsages::STORAGE),
            tile_ranges: buffer("chunk tile ranges", (TILE_BATCH * 16) as u64, s | wgpu::BufferUsages::COPY_SRC),
            chunk_jobs: buffer("chunk jobs", (CHUNK_BATCH * 32) as u64, s),
            column_masks: buffer("chunk column masks", CHUNK_BATCH as u64 * TILE_COLUMNS * 8, wgpu::BufferUsages::STORAGE),
            chunk_meta: buffer("chunk meta", (MAX_CHUNKS * 32) as u64, s | wgpu::BufferUsages::COPY_SRC),
            mids: buffer("chunk mids", (MAX_CHUNKS * 64 * 16) as u64, wgpu::BufferUsages::STORAGE),
            mid_leaves: buffer("chunk mid leaves", (MAX_CHUNKS * 64 * 4) as u64, wgpu::BufferUsages::STORAGE),
            leaves: buffer("chunk leaves", LEAF_CAPACITY * 8, wgpu::BufferUsages::STORAGE),
            leaf_alloc: buffer("chunk leaf allocator", 8, s | wgpu::BufferUsages::COPY_SRC),
            column_materials: buffer("chunk column materials", CHUNK_BATCH as u64 * TILE_COLUMNS * 4, wgpu::BufferUsages::STORAGE),
            leaf_materials: buffer("chunk leaf materials", LEAF_CAPACITY, s),
            mid_materials: buffer("chunk mid materials", (MAX_CHUNKS * 64 * 4) as u64, wgpu::BufferUsages::STORAGE),
            params: buffer("chunk draw params", std::mem::size_of::<DrawParams>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
            palette,
            chunk_keys: buffer("chunk keys", (MAX_CHUNKS * 16) as u64, s),
            chunk_table: buffer("chunk table", (TABLE_SLOTS * 32) as u64, s),
            table_mask: 0,
            face_cells: planet.grid().cells() as u32,
            slot_instance: buffer("chunk slot instances", (MAX_CHUNKS * 4) as u64, s),
            instances: buffer("chunk instances", (MAX_CHUNKS * std::mem::size_of::<ChunkDraw>()) as u64, s),
            queries,
            counters: buffer("chunk counters", 16, s | wgpu::BufferUsages::COPY_SRC),
            chunks: Vec::new(),
            drawn: (0, 0),
        }
    }

    fn read<T: Pod + Default>(&self, source: &wgpu::Buffer, count: usize) -> Vec<T> {
        let size = (count * std::mem::size_of::<T>()) as u64;
        if size == 0 {
            return Vec::new();
        }
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size);
        self.queue.submit([encoder.finish()]);
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let out = bytemuck::cast_slice(&staging.slice(..).get_mapped_range().unwrap()).to_vec();
        out
    }

    fn gen_group(&self, brushes: &wgpu::Buffer, brush_refs: &wgpu::Buffer) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chunk generation"),
            layout: &self.gen_layout,
            entries: &[
                entry(1, &self.world),
                entry(16, &self.terrain),
                entry(2, &self.tile_jobs),
                entry(3, &self.tile_data),
                entry(4, &self.tile_ranges),
                entry(5, &self.chunk_jobs),
                entry(6, &self.column_masks),
                entry(7, brushes),
                entry(8, brush_refs),
                entry(9, &self.chunk_meta),
                entry(10, &self.mids),
                entry(11, &self.mid_leaves),
                entry(12, &self.leaves),
                entry(13, &self.leaf_alloc),
                entry(14, &self.column_materials),
                entry(15, &self.leaf_materials),
                entry(17, &self.mid_materials),
            ],
        })
    }

    /// Generate the chunks of the cut for a still `eye`: tiles from the
    /// coarsest level down, each refined while its cells would project
    /// larger than `lod_pixels` pixels (`pixel_angle` radians each) and
    /// within `max_distance` of the eye.
    pub fn build(&mut self, planet: &Planet, eye: DVec3, pixel_angle: f64, lod_pixels: f64, max_distance: f64) -> BuildStats {
        let started = std::time::Instant::now();
        let grid = *planet.grid();
        let mut stats = BuildStats::default();
        self.chunks.clear();
        self.queue.write_buffer(&self.leaf_alloc, 0, &[0u8; 8]);
        // Clear the leaf materials (written with atomic or).
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&self.leaf_materials, 0, None);
        self.queue.submit([encoder.finish()]);

        // Brushes kept analytic: the large ones, then the recent ones.
        let edits = planet.edits();
        let mut faces: Vec<FaceBrush> = Vec::new();
        for log in [&edits.large, &edits.recent] {
            for id in 0..log.len() as u32 {
                faces.extend(log.resolved(id).faces.iter().copied());
            }
        }
        let none = [FaceBrush::default()];
        let brushes = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk brushes"),
            contents: bytemuck::cast_slice(if faces.is_empty() { &none[..] } else { &faces }),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let top_level = grid.levels() - 1;
        let tiles_at = |level: u32| ((grid.cells() >> level) + CHUNK - 1) / CHUNK;
        let mut current: Vec<TileKey> = Vec::new();
        for &face in grid.faces() {
            let n = tiles_at(top_level);
            for cx in 0..n {
                for cy in 0..n {
                    let tile = TileKey { face, level: top_level, cx, cy };
                    let s = f64::from(1u32 << top_level);
                    let w = f64::from(CHUNK) * s;
                    let c = grid.position(face, [f64::from(cx) * w + w / 2.0, f64::from(cy) * w + w / 2.0, 0.0]);
                    let corner = grid.position(face, [f64::from(cx) * w, f64::from(cy) * w, 0.0]);
                    if (c - eye).length() - (corner - c).length() * 1.5 <= max_distance {
                        current.push(tile);
                    }
                }
            }
        }
        while !current.is_empty() {
            let mut next = Vec::new();
            for batch in current.chunks(TILE_BATCH) {
                stats.tiles += batch.len();
                let jobs: Vec<TileJob> = batch.iter().map(|t| TileJob { face: u32::from(t.face), level: t.level, cx: t.cx, cy: t.cy }).collect();
                self.queue.write_buffer(&self.tile_jobs, 0, bytemuck::cast_slice(&jobs));
                let empty: Vec<[i32; 4]> = vec![[i32::MAX, i32::MIN + 1, i32::MAX, i32::MIN + 1]; batch.len()];
                self.queue.write_buffer(&self.tile_ranges, 0, bytemuck::cast_slice(&empty));
                // Placeholder brush refs: rewritten per chunk batch below.
                let refs0 = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 4, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
                let group = self.gen_group(&brushes, &refs0);
                let mut encoder = self.device.create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("chunk tiles"), timestamp_writes: None });
                    pass.set_pipeline(&self.tile_columns);
                    pass.set_bind_group(0, &group, &[]);
                    pass.dispatch_workgroups(64, batch.len() as u32, 1);
                }
                self.queue.submit([encoder.finish()]);
                let ranges: Vec<[i32; 4]> = self.read(&self.tile_ranges, batch.len());

                // Refine or keep each tile; kept tiles' chunks are generated
                // while their columns are in the batch.
                let mut jobs: Vec<ChunkJob> = Vec::new();
                let mut refs: Vec<u32> = Vec::new();
                for (index, (tile, range)) in batch.iter().zip(&ranges).enumerate() {
                    // Phase 1 stand-in for visibility-driven residency (phase
                    // 2): the volume's whole extent (caves deep under the
                    // surface) only within a few chunks of the eye, the
                    // surface band elsewhere.
                    if range[0] > range[1] {
                        continue;
                    }
                    let near_eye = tile_distance(&grid, *tile, (range[0], range[1]), eye) < 4.0 * f64::from(CHUNK) * grid.level_size(tile.level);
                    let (mut lo, mut hi) = if near_eye { (range[0], range[1]) } else { (range[2], range[1]) };
                    let shift = tile.level + 1;
                    let hspan = |c: i32| (i64::from(c * CHUNK) << shift, (i64::from(c * CHUNK + CHUNK) << shift) - 1);
                    let (si, sj) = (hspan(tile.cx), hspan(tile.cy));
                    for fb in &faces {
                        if brush_meets(fb, tile.face, tile.level, si, sj, (i64::MIN / 4, i64::MAX / 4)) {
                            // Additions raise the stack, removals deepen it.
                            match fb.op() {
                                0 => lo = lo.min((fb.k_lo >> shift) - 1),
                                1 => hi = hi.max((fb.k_hi >> shift) + 1),
                                _ => {}
                            }
                        }
                    }
                    let distance = tile_distance(&grid, *tile, (lo, hi), eye);
                    if distance > max_distance {
                        continue;
                    }
                    if tile.level > 0 && grid.level_size(tile.level) / distance.max(1e-3) > pixel_angle * lod_pixels {
                        let n = tiles_at(tile.level - 1);
                        for (a, b) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                            let (cx, cy) = (tile.cx * 2 + a, tile.cy * 2 + b);
                            if cx < n && cy < n {
                                next.push(TileKey { face: tile.face, level: tile.level - 1, cx, cy });
                            }
                        }
                        continue;
                    }
                    stats.tiles_per_level[tile.level as usize] += 1;
                    stats.span_per_level[tile.level as usize] += (hi - lo).max(0) as u64;
                    for cz in lo.div_euclid(CHUNK)..=(hi - 1).div_euclid(CHUNK) {
                        if self.chunks.len() >= MAX_CHUNKS {
                            stats.overflow = true;
                            break;
                        }
                        let sk = hspan(cz);
                        let first = refs.len() as u32;
                        for (b, fb) in faces.iter().enumerate() {
                            if brush_meets(fb, tile.face, tile.level, si, sj, sk) {
                                refs.push(b as u32);
                            }
                        }
                        let slot = self.chunks.len() as u32;
                        jobs.push(ChunkJob { tile: index as u32, cz, slot, brushes: first, brush_count: refs.len() as u32 - first, _pad: [0; 3] });
                        self.chunks.push(Chunk { tile: *tile, cz, slot, frame: chunk_frame(&grid, *tile, cz), z: (0, 0) });
                        stats.per_level[tile.level as usize] += 1;
                    }
                }
                let refs_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("chunk brush refs"),
                    contents: bytemuck::cast_slice(if refs.is_empty() { &[0u32][..] } else { &refs }),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let group = self.gen_group(&brushes, &refs_buffer);
                for part in jobs.chunks(CHUNK_BATCH) {
                    self.queue.write_buffer(&self.chunk_jobs, 0, bytemuck::cast_slice(part));
                    let mut encoder = self.device.create_command_encoder(&Default::default());
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("chunk generate"), timestamp_writes: None });
                        pass.set_bind_group(0, &group, &[]);
                        pass.set_pipeline(&self.chunk_columns);
                        pass.dispatch_workgroups(64, part.len() as u32, 1);
                        pass.set_pipeline(&self.chunk_build);
                        pass.dispatch_workgroups(part.len() as u32, 1, 1);
                    }
                    self.queue.submit([encoder.finish()]);
                }
            }
            current = next;
        }

        let metas: Vec<ChunkMeta> = self.read(&self.chunk_meta, self.chunks.len());
        let alloc: Vec<[u32; 2]> = self.read(&self.leaf_alloc, 1);
        stats.leaves = u64::from(alloc[0][0]);
        stats.overflow |= alloc[0][1] != 0;
        let generated = self.chunks.len();
        // Keep the chunks with a surface: not empty, not solid throughout.
        self.chunks.retain(|c| {
            let m = &metas[c.slot as usize];
            m.status == 1 && m.root_occ != [0, 0] && m.root_full != [u32::MAX, u32::MAX]
        });
        for c in &mut self.chunks {
            let m = &metas[c.slot as usize];
            c.z = (m.z[0], m.z[1]);
        }
        stats.chunks = self.chunks.len();
        self.upload_table();
        stats.empty_chunks = generated - self.chunks.len();
        // Leaves with their material byte; per chunk its mid nodes (masks,
        // first leaf, material) and root.
        stats.bytes = stats.leaves * 9 + generated as u64 * (64 * 24 + 32);
        stats.seconds = started.elapsed().as_secs_f64();
        stats
    }

    /// Tile, chunk layer and occupied layers of the chunk in `slot`.
    pub fn chunk_at(&self, slot: u32) -> Option<(TileKey, i32, (i32, i32))> {
        self.chunks.iter().find(|c| c.slot == slot).map(|c| (c.tile, c.cz, c.z))
    }

    /// Chunks over base column (`face`, `i`, `j`): tile, chunk layer,
    /// occupied layers.
    pub fn chunks_over(&self, face: u8, i: i32, j: i32) -> Vec<(TileKey, i32, (i32, i32))> {
        self.chunks
            .iter()
            .filter(|c| c.tile.face == face && (i >> c.tile.level).div_euclid(CHUNK) == c.tile.cx && (j >> c.tile.level).div_euclid(CHUNK) == c.tile.cy)
            .map(|c| (c.tile, c.cz, c.z))
            .collect()
    }

    /// The chunk table (open addressing, keys by slot) of the kept chunks.
    fn upload_table(&mut self) {
        let key = |c: &Chunk| [i32::from(c.tile.face) | ((c.tile.level as i32) << 8), c.tile.cx, c.tile.cy, c.cz];
        let mut keys = vec![[0i32; 4]; MAX_CHUNKS];
        for c in &self.chunks {
            keys[c.slot as usize] = key(c);
        }
        let n = (2 * self.chunks.len()).next_power_of_two().clamp(64, TABLE_SLOTS);
        let mask = n as u32 - 1;
        let mut table = vec![[-1i32, 0, 0, 0]; 2 * n];
        let hash = |k: [i32; 4]| {
            let mut h = (k[0] as u32).wrapping_mul(0x9E37_79B1);
            h = (h ^ k[1] as u32).wrapping_mul(0x85EB_CA77);
            h = (h ^ k[2] as u32).wrapping_mul(0xC2B2_AE3D);
            h = (h ^ k[3] as u32).wrapping_mul(0x27D4_EB2F);
            h ^ (h >> 15)
        };
        for c in &self.chunks {
            let k = key(c);
            let mut i = hash(k) & mask;
            while table[2 * i as usize][0] != -1 {
                i = (i + 1) & mask;
            }
            table[2 * i as usize] = k;
            table[2 * i as usize + 1] = [c.slot as i32, 0, 0, 0];
        }
        self.table_mask = mask;
        self.queue.write_buffer(&self.chunk_keys, 0, bytemuck::cast_slice(&keys));
        self.queue.write_buffer(&self.chunk_table, 0, bytemuck::cast_slice(&table));
    }

    /// Chunks kept by the last [`Self::build`].
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// The targets of the last [`Self::draw`].
    pub fn targets(&self) -> Option<&Targets> {
        self.targets.as_ref()
    }

    /// Draw the chunks seen from `eye` (camera-relative `view_proj`) at
    /// `size` pixels into [`Self::targets`], timing it as draw number
    /// `frame` ([`Self::draw_ms`]).
    #[allow(clippy::too_many_arguments)]
    pub fn draw(&mut self, encoder: &mut wgpu::CommandEncoder, frame: u64, eye: DVec3, view_proj: Mat4, near: f32, sun: Vec3, size: [u32; 2]) {
        if self.targets.as_ref().is_none_or(|t| t.size != size) {
            self.targets = Some(Targets::new(&self.device, size));
        }
        // Frustum: a box is drawn unless all its corners lie outside one
        // clip plane.
        let mut outside = Vec::new();
        let mut inside = Vec::new();
        for c in &self.chunks {
            let origin = c.frame.origin - eye;
            let o = c.frame.inv * (-origin);
            let (z0, z1) = (f64::from(c.z.0), f64::from(c.z.1));
            let corner = |n: u32| {
                let local = DVec3::new(if n & 1 != 0 { 64.0 } else { 0.0 }, if n & 2 != 0 { 64.0 } else { 0.0 }, if n & 4 != 0 { z1 } else { z0 });
                view_proj * (origin + c.frame.m * local).as_vec3().extend(1.0)
            };
            let corners: Vec<glam::Vec4> = (0..8).map(corner).collect();
            let culled = corners.iter().all(|p| p.x < -p.w)
                || corners.iter().all(|p| p.x > p.w)
                || corners.iter().all(|p| p.y < -p.w)
                || corners.iter().all(|p| p.y > p.w)
                || corners.iter().all(|p| p.w <= 0.0);
            if culled {
                continue;
            }
            // Inside: the eye (with the near plane around it) in the box.
            let margin = DVec3::new(c.frame.inv.x_axis.length(), c.frame.inv.y_axis.length(), c.frame.inv.z_axis.length()) * f64::from(near) * 2.0;
            let held = o.x > -margin.x && o.x < 64.0 + margin.x && o.y > -margin.y && o.y < 64.0 + margin.y && o.z > z0 - margin.z && o.z < z1 + margin.z;
            let flags = u32::from(c.frame.m.determinant() < 0.0) | (u32::from(held) << 1) | (c.tile.level << 8);
            let m = c.frame.m.as_mat3();
            let inv = c.frame.inv.as_mat3();
            let draw = ChunkDraw {
                m: [m.x_axis.extend(origin.x as f32).to_array(), m.y_axis.extend(origin.y as f32).to_array(), m.z_axis.extend(origin.z as f32).to_array()],
                inv: [inv.row(0).extend(o.x as f32).to_array(), inv.row(1).extend(o.y as f32).to_array(), inv.row(2).extend(o.z as f32).to_array()],
                bounds: [z0 as f32, z1 as f32, f32::from_bits(c.slot), f32::from_bits(flags)],
            };
            let distance = (origin + c.frame.m * DVec3::new(32.0, 32.0, (z0 + z1) / 2.0)).length();
            if held { inside.push(draw) } else { outside.push((distance, draw)) }
        }
        // Boxes holding the eye first (their hits are the nearest), then
        // the others front to back.
        outside.sort_by(|a, b| a.0.total_cmp(&b.0));
        let n_inside = inside.len() as u32;
        let all: Vec<ChunkDraw> = inside.into_iter().chain(outside.into_iter().map(|(_, d)| d)).collect();
        let mut slot_instance = vec![u32::MAX; MAX_CHUNKS];
        for (n, d) in all.iter().enumerate() {
            slot_instance[d.bounds[2].to_bits() as usize] = n as u32;
        }
        self.queue.write_buffer(&self.slot_instance, 0, bytemuck::cast_slice(&slot_instance));
        self.drawn = (n_inside, all.len() as u32);
        if !all.is_empty() {
            self.queue.write_buffer(&self.instances, 0, bytemuck::cast_slice(&all));
        }
        let params = DrawParams {
            view_proj: view_proj.to_cols_array_2d(),
            inverse_view_proj: view_proj.inverse().to_cols_array_2d(),
            sun: sun.normalize().extend(near).to_array(),
            debug: [u32::from(std::env::var_os("HELIO_VOXEL_CHUNKS_LEVELS").is_some()), u32::from(std::env::var_os("HELIO_VOXEL_CHUNKS_COUNT").is_some()), 0, 0],
            table: [self.table_mask, self.face_cells, 0, 0],
        };
        self.queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chunk draw"),
            layout: &self.draw_layout,
            entries: &[entry(0, &self.params), entry(1, &self.instances), entry(2, &self.chunk_meta), entry(3, &self.mids), entry(4, &self.mid_leaves), entry(5, &self.leaves), entry(6, &self.leaf_materials), entry(7, &self.palette), entry(8, &self.mid_materials), entry(9, &self.counters), entry(10, &self.chunk_table), entry(11, &self.chunk_keys), entry(12, &self.slot_instance)],
        });
        let targets = self.targets.as_ref().unwrap();
        let ring = (frame % TIMING_RING) as u32;
        let (depth, seed, exits) = (view(&targets.depth), view(&targets.seed), view(&targets.exits));
        // Every ray starts unresolved (0).
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("chunk exits clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &exits,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let seed_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chunk seed exits"),
            layout: &self.seed_layout,
            entries: &[wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&exits) }],
        });
        let trace_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chunk trace"),
            layout: &self.trace_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&seed) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view(&targets.visibility)) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&view(&targets.colour)) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&view(&targets.distance)) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&exits) },
            ],
        });
        // Rounds of seed and trace: each resolves the rays that hit in
        // their box; the rest are seeded beyond it.
        let rounds = self.rounds;
        for round in 0..rounds {
            {
                // 1. Seed: each pixel's nearest box (beyond its last exit).
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("chunk seed"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &seed,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: f64::from(u32::MAX), g: 0.0, b: 0.0, a: 0.0 }), store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth,
                        depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: self.queries.as_ref().filter(|_| round == 0).map(|(set, _, _)| wgpu::RenderPassTimestampWrites { query_set: set, beginning_of_pass_write_index: Some(3 * ring), end_of_pass_write_index: Some(3 * ring + 1) }),
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_bind_group(0, &group, &[]);
                pass.set_bind_group(1, &seed_group, &[]);
                // Boxes holding the eye only in the first round.
                if self.drawn.0 > 0 && round == 0 {
                    pass.set_pipeline(&self.seed_inside);
                    pass.draw(0..36, 0..self.drawn.0);
                }
                if self.drawn.1 > self.drawn.0 {
                    pass.set_pipeline(&self.seed_outside);
                    pass.draw(0..36, self.drawn.0..self.drawn.1);
                }
            }
            // 2. Trace each pixel's ray through its seed chunk.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("chunk trace"),
                timestamp_writes: self.queries.as_ref().filter(|_| round + 1 == rounds).map(|(set, _, _)| wgpu::ComputePassTimestampWrites { query_set: set, beginning_of_pass_write_index: None, end_of_pass_write_index: Some(3 * ring + 2) }),
            });
            pass.set_pipeline(&self.trace);
            pass.set_bind_group(0, &group, &[]);
            pass.set_bind_group(1, &trace_group, &[]);
            pass.dispatch_workgroups(size[0].div_ceil(8), size[1].div_ceil(8), 1);
        }
        {
            // 3. The hits' depth.
            let resolve_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("chunk depth resolve"),
                layout: &self.resolve_layout,
                entries: &[wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&view(&targets.distance)) }],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("chunk depth resolve"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.resolve);
            pass.set_bind_group(0, &group, &[]);
            pass.set_bind_group(1, &resolve_group, &[]);
            pass.draw(0..3, 0..1);
        }
        if let Some((set, resolve, read)) = &self.queries {
            // Resolve offsets are 256-byte aligned.
            let at = u64::from(ring) * u64::from(wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT);
            encoder.resolve_query_set(set, 3 * ring..3 * ring + 3, resolve, at);
            encoder.copy_buffer_to_buffer(resolve, at, &read[ring as usize], 0, 24);
        }
    }

    /// Seeded pixels, tree steps and hits of the draws since the last call
    /// (with `HELIO_VOXEL_CHUNKS_COUNT`).
    pub fn take_counts(&self) -> [u32; 4] {
        let counts: Vec<[u32; 4]> = self.read(&self.counters, 1);
        self.queue.write_buffer(&self.counters, 0, &[0u8; 16]);
        counts[0]
    }

    /// Boxes drawn by the last [`Self::draw`]: from outside, from inside.
    pub fn drawn(&self) -> (u32, u32) {
        (self.drawn.1 - self.drawn.0, self.drawn.0)
    }

    /// GPU milliseconds of draw `frame`, submitted as `submission`: in
    /// all, and of the first seed (waits for it; read it before draw
    /// `frame + TIMING_RING` is recorded).
    pub fn draw_ms(&self, frame: u64, submission: wgpu::SubmissionIndex) -> Option<[f64; 2]> {
        let (_, _, ring) = self.queries.as_ref()?;
        let read = &ring[(frame % TIMING_RING) as usize];
        read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::Wait { submission_index: Some(submission), timeout: None }).ok()?;
        let t: [u64; 3] = bytemuck::cast_slice::<u8, u64>(&read.slice(..).get_mapped_range().ok()?).try_into().ok()?;
        read.unmap();
        let ms = |a: u64, b: u64| b.saturating_sub(a) as f64 * f64::from(self.queue.get_timestamp_period()) / 1.0e6;
        Some([ms(t[0], t[2]), ms(t[0], t[1])])
    }
}
