//! Chunk trees (`docs/chunk-trees.md`), phase 1: generation of chunk trees
//! from the terrain contract and primary visibility through rasterized
//! chunk boxes, measured against the span-column renderer.
//!
//! A tile is the 64x64 columns of one level under a stack of chunks; a
//! chunk is 64^3 cells of one level stored as a 64-tree. [`ChunkWorld::build`]
//! selects the cut of the tile quadtree whose cells project to at most
//! `lod_pixels` pixels and generates every chunk under it; [`ChunkWorld::draw`]
//! draws the chunks' boxes, and each fragment walks one chunk's tree.

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
    /// Direction to the sun, near plane.
    sun: [f32; 4],
    /// x: colour by level (`HELIO_VOXEL_CHUNKS_LEVELS`).
    debug: [u32; 4],
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
    outside: wgpu::RenderPipeline,
    inside: wgpu::RenderPipeline,
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
    instances: wgpu::Buffer,
    /// Timestamps of the last `TIMING_RING` draws: queries, resolve
    /// buffer, one readback buffer per draw.
    queries: Option<(wgpu::QuerySet, wgpu::Buffer, Vec<wgpu::Buffer>)>,
    /// Fragment, step and hit counts (`HELIO_VOXEL_CHUNKS_COUNT`).
    counters: wgpu::Buffer,
    chunks: Vec<Chunk>,
    drawn: (u32, u32),
}

pub const COLOUR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
pub const VISIBILITY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Uint;
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Distance from the eye to each pixel's hit.
pub const DISTANCE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

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

fn cleared(view: &wgpu::TextureView) -> wgpu::RenderPassColorAttachment<'_> {
    wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
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
        .replace("COUNT(2)", &count(2));
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
        let vf = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
        let draw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("chunk draw"),
            entries: &[uniform(0, vf), storage(1, true, vf), storage(2, true, vf), storage(3, true, vf), storage(4, true, vf), storage(5, true, vf), storage(6, true, vf), uniform(7, vf), storage(8, true, vf), storage(9, false, wgpu::ShaderStages::FRAGMENT)],
        });
        let draw_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&draw_layout)], immediate_size: 0 });
        let render = |fragment: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(fragment),
                layout: Some(&draw_pipeline_layout),
                vertex: wgpu::VertexState { module: &draw_module, entry_point: Some("chunk_vertex"), compilation_options: Default::default(), buffers: &[] },
                primitive: wgpu::PrimitiveState { cull_mode: Some(wgpu::Face::Back), front_face: wgpu::FrontFace::Ccw, ..Default::default() },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(if std::env::var_os("HELIO_VOXEL_CHUNKS_NO_DEPTH_TEST").is_some() { wgpu::CompareFunction::Always } else { wgpu::CompareFunction::Less }),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &draw_module,
                    entry_point: Some(fragment),
                    compilation_options: Default::default(),
                    targets: &[Some(VISIBILITY_FORMAT.into()), Some(COLOUR_FORMAT.into()), Some(DISTANCE_FORMAT.into())],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let outside = render("chunk_fragment");
        let inside = render("chunk_fragment_inside");

        let buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
        let s = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let world = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("chunk world"), contents: bytemuck::bytes_of(&WorldGpu::new(planet)), usage: wgpu::BufferUsages::UNIFORM });
        let terrain = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("chunk terrain"), contents: &terrain_bytes(&program), usage: wgpu::BufferUsages::UNIFORM });
        let appearance = planet.field().appearance();
        let palette: Vec<[f32; 4]> = appearance.materials.iter().map(|m| [m.colour[0].powf(2.2), m.colour[1].powf(2.2), m.colour[2].powf(2.2), 1.0]).collect();
        let palette = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("chunk palette"), contents: bytemuck::cast_slice(&palette), usage: wgpu::BufferUsages::UNIFORM });
        let queries = device.features().contains(wgpu::Features::TIMESTAMP_QUERY).then(|| {
            (
                device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("chunk draw"), ty: wgpu::QueryType::Timestamp, count: 2 * TIMING_RING as u32 }),
                buffer("chunk timestamps", u64::from(wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT) * TIMING_RING, wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC),
                (0..TIMING_RING).map(|_| buffer("chunk timestamps read", 16, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST)).collect(),
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
            outside,
            inside,
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
                    if std::env::var_os("HELIO_VOXEL_CHUNKS_DEEPER").is_some() {
                        lo -= CHUNK;
                    }
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

    /// Chunks kept by the last [`Self::build`].
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Draw the chunks seen from `eye` (camera-relative `view_proj`) into
    /// the visibility, colour and depth targets (cleared), timing the pass
    /// as draw number `frame` ([`Self::draw_ms`]).
    #[allow(clippy::too_many_arguments)]
    pub fn draw(&mut self, encoder: &mut wgpu::CommandEncoder, frame: u64, eye: DVec3, view_proj: Mat4, near: f32, sun: Vec3, visibility: &wgpu::TextureView, colour: &wgpu::TextureView, distance: &wgpu::TextureView, depth: &wgpu::TextureView) {
        // Frustum: a box is drawn unless all its corners lie outside one
        // clip plane.
        let mut outside = Vec::new();
        let mut inside = Vec::new();
        // Diagnostics: draw only levels in `HELIO_VOXEL_CHUNKS_LEVEL_RANGE=lo,hi`.
        let band: Vec<u32> = std::env::var("HELIO_VOXEL_CHUNKS_LEVEL_RANGE").ok().map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
        for c in &self.chunks {
            if band.len() == 2 && (c.tile.level < band[0] || c.tile.level > band[1]) {
                continue;
            }
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
            let mirrored = (c.frame.m.determinant() < 0.0) != held;
            let m = c.frame.m.as_mat3();
            let inv = c.frame.inv.as_mat3();
            let draw = ChunkDraw {
                m: [m.x_axis.extend(origin.x as f32).to_array(), m.y_axis.extend(origin.y as f32).to_array(), m.z_axis.extend(origin.z as f32).to_array()],
                inv: [inv.row(0).extend(o.x as f32).to_array(), inv.row(1).extend(o.y as f32).to_array(), inv.row(2).extend(o.z as f32).to_array()],
                bounds: [z0 as f32, z1 as f32, f32::from_bits(c.slot), f32::from_bits(u32::from(mirrored) | (c.tile.level << 8))],
            };
            let distance = (origin + c.frame.m * DVec3::new(32.0, 32.0, (z0 + z1) / 2.0)).length();
            if held { inside.push(draw) } else { outside.push((distance, draw)) }
        }
        // Boxes holding the eye first (their hits are the nearest), then
        // the others front to back.
        outside.sort_by(|a, b| a.0.total_cmp(&b.0));
        let n_inside = inside.len() as u32;
        let all: Vec<ChunkDraw> = inside.into_iter().chain(outside.into_iter().map(|(_, d)| d)).collect();
        self.drawn = (n_inside, all.len() as u32);
        if !all.is_empty() {
            self.queue.write_buffer(&self.instances, 0, bytemuck::cast_slice(&all));
        }
        let params = DrawParams {
            view_proj: view_proj.to_cols_array_2d(),
            sun: sun.normalize().extend(near).to_array(),
            debug: [u32::from(std::env::var_os("HELIO_VOXEL_CHUNKS_LEVELS").is_some()), u32::from(std::env::var_os("HELIO_VOXEL_CHUNKS_COUNT").is_some()), 0, 0],
        };
        self.queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chunk draw"),
            layout: &self.draw_layout,
            entries: &[entry(0, &self.params), entry(1, &self.instances), entry(2, &self.chunk_meta), entry(3, &self.mids), entry(4, &self.mid_leaves), entry(5, &self.leaves), entry(6, &self.leaf_materials), entry(7, &self.palette), entry(8, &self.mid_materials), entry(9, &self.counters)],
        });
        let ring = (frame % TIMING_RING) as u32;
        let timestamp_writes = self.queries.as_ref().map(|(set, _, _)| wgpu::RenderPassTimestampWrites { query_set: set, beginning_of_pass_write_index: Some(2 * ring), end_of_pass_write_index: Some(2 * ring + 1) });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("chunk primary"),
                color_attachments: &[Some(cleared(visibility)), Some(cleared(colour)), Some(cleared(distance))],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &group, &[]);
            if self.drawn.0 > 0 {
                pass.set_pipeline(&self.inside);
                pass.draw(0..36, 0..self.drawn.0);
            }
            if self.drawn.1 > self.drawn.0 {
                pass.set_pipeline(&self.outside);
                pass.draw(0..36, self.drawn.0..self.drawn.1);
            }
        }
        if let Some((set, resolve, read)) = &self.queries {
            // Resolve offsets are 256-byte aligned.
            let at = u64::from(ring) * u64::from(wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT);
            encoder.resolve_query_set(set, 2 * ring..2 * ring + 2, resolve, at);
            encoder.copy_buffer_to_buffer(resolve, at, &read[ring as usize], 0, 16);
        }
    }

    /// Fragments, tree steps and hits of the draws since the last call
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

    /// GPU milliseconds of draw `frame`, submitted as `submission` (waits
    /// for it; read it before draw `frame + TIMING_RING` is recorded).
    pub fn draw_ms(&self, frame: u64, submission: wgpu::SubmissionIndex) -> Option<f64> {
        let (_, _, ring) = self.queries.as_ref()?;
        let read = &ring[(frame % TIMING_RING) as usize];
        read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::Wait { submission_index: Some(submission), timeout: None }).ok()?;
        let t: [u64; 2] = bytemuck::cast_slice::<u8, u64>(&read.slice(..).get_mapped_range().ok()?).try_into().ok()?;
        read.unmap();
        Some(t[1].saturating_sub(t[0]) as f64 * f64::from(self.queue.get_timestamp_period()) / 1.0e6)
    }
}
