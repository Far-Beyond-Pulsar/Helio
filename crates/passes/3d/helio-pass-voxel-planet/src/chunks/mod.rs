//! Chunk trees (`docs/chunk-trees.md`): generation of chunk trees from the
//! terrain contract, residency for a moving eye and primary visibility.
//!
//! A tile is the 64x64 columns of one level under a stack of chunks; a
//! chunk is 64^3 cells of one level stored as a 64-tree. [`ChunkWorld::update`]
//! asks the [`residency::Planner`] which tiles to draw and generate, and
//! generates and evicts them on the GPU within a budget; [`ChunkWorld::draw`]
//! rasterizes the chunks' boxes to seed each pixel with the nearest box its
//! ray enters, then traces each pixel's ray through that chunk's tree in a
//! compute pass (shaders/chunk_draw.wgsl).

mod drawn;
pub mod pass;
pub mod residency;

use crate::edit_store::{ColumnKey, EditStore};
use crate::edits::{center_half, FaceBrush};
use crate::engine::{terrain_bytes, terrain_source, WorldGpu};
use crate::grid::Grid;
use crate::planet::{Edits, Planet};
use bytemuck::{Pod, Zeroable};
use glam::{DMat3, DVec3, Mat4, Vec3};
use rustc_hash::{FxHashMap, FxHashSet};
use wgpu::util::DeviceExt;

/// Cells along a chunk edge (and columns along a tile edge).
pub const CHUNK: i32 = 64;
const TILE_COLUMNS: u64 = 4096;
/// Tiles generated per dispatch (their columns are scratch until their
/// chunks are built).
const TILE_BATCH: usize = 512;
/// Chunks generated per dispatch (column masks are scratch).
const CHUNK_BATCH: usize = 2048;
/// Chunk slots.
pub const MAX_CHUNKS: usize = 131072;

/// `Cull` of chunk_draw.wgsl: two counts, then 65 bins' counts and cursors.
const CULL_BYTES: u64 = 4 * (2 + 2 * 65);

/// Metres over the highest possible solid cell where rays may end: far
/// above the f32 error of the ceiling test.
const CEILING_MARGIN: f64 = 16.0;
/// Leaves of every chunk together (8 bytes each).
const LEAF_CAPACITY: u64 = 48 << 20;
/// A brush buffer's content when there are none.
const NO_BRUSH: [FaceBrush; 1] = [FaceBrush { flags: 0, radius_half: 0, k_lo: 0, k_hi: 0, center: [0; 4], ball: [0; 4] }];

/// Leaf block size classes (16 << class leaves).
const LEAF_CLASSES: u32 = 9;

/// Entries of the free block stacks: room for every block of each class
/// (`stack_base` in chunk_generate.wgsl).
fn free_block_entries() -> u64 {
    (0..LEAF_CLASSES).map(|c| LEAF_CAPACITY / (16 << c)).sum()
}

/// Chunk table entries: the drawn chunks and the drawn cut's columns.
const TABLE_SLOTS: usize = 4 * MAX_CHUNKS;
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
    /// Refs of the large log (first), then the recent log's.
    large: u32,
    /// Its baked bricks' table in `baked_refs`, or NONE.
    baked: u32,
    _pad: u32,
}


/// `ChunkStatic` of chunk_draw.wgsl: a chunk's frame, written when it is
/// generated (its layers when they are read back).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ChunkStatic {
    m: [[f32; 4]; 3],
    inv: [[f32; 4]; 3],
    origin_hi: [f32; 4],
    origin_lo: [f32; 4],
    /// Lowest and one past the highest occupied layer, slot (bits), flags
    /// (bits; 1: mirror x; 8..16: level).
    bounds: [f32; 4],
}

/// A planet-centred position as a high and a low f32 part.
fn split(v: DVec3) -> ([f32; 4], [f32; 4]) {
    let hi = v.as_vec3();
    let lo = (v - hi.as_dvec3()).as_vec3();
    (hi.extend(0.0).to_array(), lo.extend(0.0).to_array())
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct DrawParams {
    view_proj: [[f32; 4]; 4],
    inverse_view_proj: [[f32; 4]; 4],
    prev_view_proj: [[f32; 4]; 4],
    eye_hi: [f32; 4],
    eye_lo: [f32; 4],
    /// Direction to the sun, near plane.
    sun: [f32; 4],
    /// x: colour by level (1, `HELIO_VOXEL_CHUNKS_LEVELS`) or by primary
    /// steps (2, `HELIO_VOXEL_CHUNKS_HEAT`, counting builds).
    debug: [u32; 4],
    /// Chunk table slots - 1, base cells per face edge, boxes holding the
    /// eye, frame.
    table: [u32; 4],
    /// The eye in noise-domain units: integer part, fraction.
    eye_domain: [i32; 4],
    eye_domain_frac: [f32; 4],
    /// Up at the eye (xyz) and the eye's distance from the planet's
    /// centre (w); a plane's up is +y.
    up: [f32; 4],
    /// x: the eye's squared radius less the ceiling's (a plane: its height
    /// over the ceiling), from f64.
    ceiling: [f32; 4],
    /// The planet's radius and voxel (m), its first exact level
    /// ([`exact_level`]), the eye's squared radius less the planet's (from
    /// f64).
    planet: [f32; 4],
}

/// `Materials` of chunk_draw.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ChunkMaterials {
    materials: [crate::engine::MaterialGpu; crate::engine::MATERIALS],
    detail: [f32; 4],
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
/// differences across the chunk, about its middle column at height `z`
/// (cells in the chunk: its columns' mean top, its ground). Across a chunk
/// the sphere curves `512 x cell / radius` of a cell from it (0.5 at the
/// coarsest level on Earth, where tiles span 600 km), and columns, which
/// meet at the planet's centre, lean `32 x cell / radius` a cell from it
/// per cell of height: about the ground neighbours agree to a twentieth of
/// a cell, about a 64-cell chunk's middle they parted by cells where the
/// ground lay far below it (docs/chunk-trees.md).
fn chunk_frame(grid: &Grid, tile: TileKey, cz: i32, z: f64) -> Frame {
    let s = f64::from(1u32 << tile.level);
    let half = f64::from(CHUNK / 2) * s;
    let centre = [f64::from(tile.cx * CHUNK) * s + half, f64::from(tile.cy * CHUNK) * s + half, (f64::from(cz * CHUNK) + z) * s];
    let at = |d: [f64; 3]| grid.position(tile.face, [centre[0] + d[0], centre[1] + d[1], centre[2] + d[2]]);
    let n = f64::from(CHUNK);
    let u = (at([half, 0.0, 0.0]) - at([-half, 0.0, 0.0])) / n;
    let v = (at([0.0, half, 0.0]) - at([0.0, -half, 0.0])) / n;
    let w = (at([0.0, 0.0, half]) - at([0.0, 0.0, -half])) / n;
    let m = DMat3::from_cols(u, v, w);
    Frame { origin: at([0.0; 3]) - m * DVec3::new(n / 2.0, n / 2.0, z), m, inv: m.inverse() }
}

/// The first level drawing walks in the grid's own cells (chunk_draw.wgsl
/// `exact_level`), not in chunks' affine frames: where a frame strays from
/// the sphere by a twentieth of a cell across a chunk (512 x cell /
/// radius: a height and a slope a cell's haze and light show), and
/// planet-centred f32 positions err by under a fiftieth of one. Level 13
/// on an Earth at 0.1 m voxels (820 m cells), level 2 on a 3 km world.
/// None on a plane.
fn exact_level(grid: &Grid) -> u32 {
    if grid.is_plane() {
        return u32::MAX;
    }
    (0..grid.levels())
        .find(|&level| {
            let cell = grid.level_size(level);
            512.0 * cell / grid.radius() >= 0.05 && cell >= 50.0 * grid.radius() * f64::from(f32::EPSILON)
        })
        .unwrap_or(grid.levels())
}

/// Cells an exact chunk's own cells reach past its frame's box (its raster
/// box grows by them): the grid's points over its occupied layers in its
/// frame, beyond the box, and half a cell.
fn reach_past(grid: &Grid, tile: TileKey, cz: i32, frame: &Frame, z: (f32, f32)) -> f32 {
    let s = f64::from(1u32 << tile.level);
    let mut excess = 0.0f64;
    for a in 0..=8 {
        for b in 0..=8 {
            for k in [z.0, z.1] {
                let local = DVec3::new(f64::from(a * 8), f64::from(b * 8), f64::from(k));
                let at = grid.position(tile.face, [(f64::from(tile.cx * CHUNK) + local.x) * s, (f64::from(tile.cy * CHUNK) + local.y) * s, (f64::from(cz * CHUNK) + local.z) * s]);
                let f = frame.inv * (at - frame.origin);
                excess = excess.max(-f.x).max(f.x - 64.0).max(-f.y).max(f.y - 64.0).max(f64::from(z.0) - f.z).max(f.z - f64::from(z.1));
            }
        }
    }
    (excess + 0.5) as f32
}

/// Chunk `cz` of `tile` in `slot` as drawing reads it, with occupied layers
/// `z` and its frame made about height `about` (cells in the chunk); at an
/// exact level (from `exact`) how far its cells reach past the frame's box
/// (m.x_axis.w).
fn chunk_static(grid: &Grid, tile: TileKey, cz: i32, slot: u32, z: (f32, f32), about: f32, exact: u32) -> ChunkStatic {
    let frame = chunk_frame(grid, tile, cz, f64::from(about));
    let (m, inv) = (frame.m.as_mat3(), frame.inv.as_mat3());
    let (origin_hi, origin_lo) = split(frame.origin);
    let flags = u32::from(frame.m.determinant() < 0.0) | (tile.level << 8);
    let grow = if tile.level >= exact { reach_past(grid, tile, cz, &frame, z) } else { 0.0 };
    ChunkStatic {
        m: [m.x_axis.extend(grow).to_array(), m.y_axis.extend(0.0).to_array(), m.z_axis.extend(0.0).to_array()],
        inv: [inv.row(0).extend(0.0).to_array(), inv.row(1).extend(0.0).to_array(), inv.row(2).extend(0.0).to_array()],
        origin_hi,
        origin_lo,
        bounds: [z.0, z.1, f32::from_bits(slot), f32::from_bits(flags)],
    }
}

/// An eye and, on each face over whose half-space it is, its continuous
/// index coordinates and the grid's axes there (metres per index step):
/// planning measures tiles near it in that linear frame (the inverse
/// mapping costs two arctangents a call; LOD needs no more than the frame's
/// accuracy), farther ones from their centres.
pub struct Eye {
    pub position: DVec3,
    faces: [Option<([f64; 3], [DVec3; 3])>; 6],
    /// Metres across the face the linear frame holds (a sphere curves
    /// d^2 / 2R from it: 1 % at R / 64; seen from orbit a face spans
    /// thousands of kilometres).
    reach: f64,
}

impl Eye {
    pub fn new(grid: &Grid, position: DVec3) -> Self {
        let mut faces = [None; 6];
        let cells = f64::from(grid.cells());
        for &face in grid.faces() {
            if let Some(c) = grid.face_coords(face, position) {
                let at = [c[0].clamp(0.0, cells), c[1].clamp(0.0, cells), c[2]];
                let p0 = grid.position(face, at);
                let axis = |a: usize| {
                    let mut q = at;
                    q[a] += 1.0;
                    grid.position(face, q) - p0
                };
                faces[face as usize] = Some((c, [axis(0), axis(1), axis(2)]));
            }
        }
        Self { position, faces, reach: if grid.is_plane() { f64::INFINITY } else { grid.radius() / 64.0 } }
    }

    /// From the eye to the nearest point of the tile's columns between
    /// layers `range` (level cells), in the eye's linear frame of the
    /// tile's face; `None` when the eye is not over that face or the tile
    /// lies beyond the frame's reach.
    fn offset(&self, tile: TileKey, range: (i32, i32)) -> Option<DVec3> {
        let (e, axes) = self.faces[tile.face as usize]?;
        let s = f64::from(1u32 << tile.level);
        let (i0, j0) = (f64::from(tile.cx * CHUNK) * s, f64::from(tile.cy * CHUNK) * s);
        let w = f64::from(CHUNK) * s;
        let d = [e[0].clamp(i0, i0 + w) - e[0], e[1].clamp(j0, j0 + w) - e[1], e[2].clamp(f64::from(range.0) * s, f64::from(range.1) * s) - e[2]];
        let across = axes[0] * d[0] + axes[1] * d[1];
        (across.length() <= self.reach).then(|| across + axes[2] * d[2])
    }

    /// From the eye towards the tile's columns between layers `range`:
    /// [`Self::offset`] near it, else to their centre.
    fn toward(&self, grid: &Grid, tile: TileKey, range: (i32, i32)) -> DVec3 {
        self.offset(tile, range).unwrap_or_else(|| {
            let s = f64::from(1u32 << tile.level);
            let middle = (f64::from(tile.cx * CHUNK) + f64::from(CHUNK) / 2.0) * s;
            let across = (f64::from(tile.cy * CHUNK) + f64::from(CHUNK) / 2.0) * s;
            grid.position(tile.face, [middle, across, f64::from(range.0 + range.1) * s / 2.0]) - self.position
        })
    }
}

/// Distance from `eye` to the tile's columns between layers `range` (level
/// cells): to the nearest point when the eye is over the tile's face and
/// near it ([`Eye::offset`]), else to its centre less its half-diagonal.
fn tile_distance(grid: &Grid, tile: TileKey, range: (i32, i32), eye: &Eye) -> f64 {
    if let Some(offset) = eye.offset(tile, range) {
        return offset.length();
    }
    let s = f64::from(1u32 << tile.level);
    let (i0, j0) = (f64::from(tile.cx * CHUNK) * s, f64::from(tile.cy * CHUNK) * s);
    let w = f64::from(CHUNK) * s;
    let (k0, k1) = (f64::from(range.0) * s, f64::from(range.1) * s);
    let c = grid.position(tile.face, [i0 + w / 2.0, j0 + w / 2.0, (k0 + k1) / 2.0]);
    let corner = grid.position(tile.face, [i0, j0, k0]);
    ((c - eye.position).length() - (corner - c).length()).max(0.0)
}

fn loaded(view: &wgpu::TextureView) -> Option<wgpu::RenderPassColorAttachment<'_>> {
    Some(wgpu::RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } })
}

/// The analytic brushes of `edits` resolved per face: the large log's,
/// then the recent log's; the count of large ones, and each recent brush's
/// face count.
fn analytic_faces(edits: &Edits) -> (Vec<FaceBrush>, usize, Vec<usize>) {
    let mut faces: Vec<FaceBrush> = (0..edits.large.len() as u32).flat_map(|id| edits.large.resolved(id).faces.iter().copied()).collect();
    let large = faces.len();
    let mut counts = Vec::new();
    for id in 0..edits.recent.len() as u32 {
        let resolved = &edits.recent.resolved(id).faces;
        faces.extend(resolved.iter().copied());
        counts.push(resolved.len());
    }
    (faces, large, counts)
}

/// Which edits a world shows: brushes applied, the history's hash, the
/// baked state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EditState {
    len: usize,
    hash: u64,
    sealed: (usize, u64),
}

impl EditState {
    fn of(edits: &Edits) -> Self {
        Self { len: edits.len(), hash: edits.hash(), sealed: edits.baked_state() }
    }
}

/// A frame's generation jobs and what they upload.
#[derive(Default)]
struct GenBatch {
    tile_jobs: Vec<TileJob>,
    chunk_jobs: Vec<ChunkJob>,
    refs: Vec<u32>,
    /// Baked brick tables (512 a job) and bricks (256 words each), each
    /// brick once.
    baked_refs: Vec<u32>,
    baked_cells: Vec<u32>,
    bricks: FxHashMap<usize, u32>,
}

impl GenBatch {
    /// The table of `store`'s bricks under chunk `cz` of `tile`, or NONE.
    fn baked_table(&mut self, store: &EditStore, tile: TileKey, cz: i32) -> u32 {
        let mut table = [u32::MAX; 512];
        let mut any = false;
        for by in 0..8 {
            for bx in 0..8 {
                let key = ColumnKey { face: tile.face, level: tile.level as u8, ci: tile.cx * 8 + bx, cj: tile.cy * 8 + by };
                let Some(column) = store.column(&key) else { continue };
                for (bk, brick) in column.iter() {
                    let bz = bk - cz * 8;
                    if !(0..8).contains(&bz) {
                        continue;
                    }
                    let cells = &mut self.baked_cells;
                    let at = *self.bricks.entry(std::sync::Arc::as_ptr(brick) as usize).or_insert_with(|| {
                        let n = (cells.len() / 256) as u32;
                        cells.extend(brick.cells.chunks(2).map(|p| u32::from(p[0].0) | (u32::from(p[1].0) << 16)));
                        n
                    });
                    table[((by * 8 + bx) * 8 + bz) as usize] = at;
                    any = true;
                }
            }
        }
        if !any {
            return u32::MAX;
        }
        let n = self.baked_refs.len() as u32;
        self.baked_refs.extend_from_slice(&table);
        n
    }
}

/// A chunk's key in the chunk table: face and level, tile, layer.
fn chunk_key(tile: TileKey, cz: i32) -> [i32; 4] {
    [i32::from(tile.face) | ((tile.level as i32) << 8), tile.cx, tile.cy, cz]
}

/// Layers [lo, hi) whose chunks `tile` generates (its `ranges` with its
/// edits, [`ChunkWorld::with_edits`]): the volume's whole extent where rays
/// reached under its surface band (`deep`: through cave mouths) or within a
/// few chunks of the eye, else from the surface band up.
fn layers_needed(grid: &Grid, tile: TileKey, ranges: residency::Ranges, eye: &Eye, deep: bool) -> (i32, i32) {
    let near_eye = tile_distance(grid, tile, ranges.volume, eye) < 4.0 * f64::from(CHUNK) * grid.level_size(tile.level);
    if deep || near_eye { ranges.volume } else { (ranges.surface.0, ranges.volume.1) }
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

/// A generated chunk.
struct Chunk {
    cz: i32,
    slot: u32,
    /// Generation it belongs to (matches its readback).
    serial: u64,
    /// Occupied layers, once read back.
    z: Option<(i32, i32)>,
}

/// A tile's generated chunks, their layers, and its ranges (as requested,
/// then measured).
struct TileChunks {
    chunks: Vec<Chunk>,
    layers: Layers,
    ranges: residency::Ranges,
}

/// A tile's chunk layers (bands, inclusive) and what lies where it has
/// none ([`ChunkWorld::layers`]).
#[derive(Clone, Debug, PartialEq)]
struct Layers {
    bands: Vec<(i32, i32)>,
    gap: drawn::Gap,
}

impl Layers {
    /// Whether these chunks hold every layer `need` asks for, with the same
    /// gap.
    fn holds(&self, need: &Layers) -> bool {
        self.gap == need.gap && need.bands.iter().all(|&(a, b)| (a..=b).all(|cz| self.bands.iter().any(|&(c, d)| (c..=d).contains(&cz))))
    }
}

/// What the traces did (counting builds).
#[derive(Clone, Copy, Debug)]
pub struct TraceCounts {
    /// Pixels traced, tree steps, hits, boxes crossed.
    pub seeded: u32,
    pub steps: u32,
    pub hits: u32,
    pub boxes: u32,
    /// Rays that ended without a hit, the boxes they crossed, and those
    /// that left the drawn cut.
    pub missed: u32,
    pub missed_boxes: u32,
    pub lost: u32,
}

/// A frame's GPU time (milliseconds).
#[derive(Clone, Copy, Debug)]
pub struct FrameTimings {
    pub frame: u64,
    /// The draw in all (seeds, traces), its first seed; then sun shadows.
    pub draw: f64,
    pub seed: f64,
    pub shadow: f64,
    /// The update's eviction and generation, and the tiles it generated.
    pub generate: f64,
    pub tiles: usize,
    /// Generation by kernel: eviction and tile columns, chunk columns,
    /// chunk builds.
    pub stages: [f64; 3],
}

/// Timestamps a frame writes: generation start, tile columns end, seed
/// start and end, trace end, chunk columns end, chunk builds end, shadows
/// end.
const QUERIES: u32 = 8;

/// `Reach`'s size: a count and 4095 keys.
const REACH_BYTES: u64 = 16 * 4096;

/// Picks a frame answers (more wait a frame).
const MAX_PICKS: usize = 8;

/// A frame's pick hits on their way back: per pick 256 bytes, its
/// visibility (8) then distance (at 16).
struct PickSlot {
    buffer: wgpu::Buffer,
    /// Request ids, the tool's picks, and the state as for `Reach`.
    requests: Vec<u64>,
    sink: Option<crate::engine::SharedPicks>,
    state: ReachState,
}

/// A copy of `Reach` on its way back.
struct ReachSlot {
    buffer: wgpu::Buffer,
    state: ReachState,
}

enum ReachState {
    Free,
    /// Copied in a frame not yet submitted.
    Copied,
    Mapping(std::sync::Arc<std::sync::atomic::AtomicBool>),
}

/// A frame's timestamps on their way back.
struct TimingSlot {
    buffer: wgpu::Buffer,
    /// Frame, tiles and chunks written, once resolved; mapping when
    /// `ready` is set.
    frame: Option<(u64, usize, usize)>,
    ready: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

/// One frame's generation results on their way back from the GPU.
struct Readback {
    buffer: wgpu::Buffer,
    ready: std::sync::Arc<std::sync::atomic::AtomicBool>,
    tiles: Vec<TileKey>,
    /// Chunk jobs in order: tile, layer, slot, serial.
    chunks: Vec<(TileKey, i32, u32, u64)>,
}

/// What [`ChunkWorld::update`] did.
#[derive(Clone, Copy, Debug, Default)]
pub struct UpdateStats {
    /// Tiles waiting to be generated.
    pub requests: usize,
    pub generated_tiles: usize,
    pub generated_chunks: usize,
    pub evicted_tiles: usize,
    /// Tiles regenerated because their measured layers left their chunks.
    pub regenerated_tiles: usize,
    pub drawn_tiles: usize,
    /// Drawn tiles fading in or out of a switch between levels.
    pub fading_tiles: usize,
    /// Tiles rays first reached under the chunks of (cave mouths): their
    /// whole volume generated.
    pub deep_tiles: usize,
    pub drawn_chunks: usize,
    pub resident_chunks: usize,
    /// Drawn tiles in view coarser than wanted, as a share of the drawn
    /// tiles in view.
    pub coarse_share: f64,
    /// [`residency::Plan::worst_px`] of the latest plan.
    pub worst_px: f64,
    /// Level switches in view a plan that arrived this frame made, and
    /// [`residency::Plan::switch_px`].
    pub switches: usize,
    pub switch_px: f64,
    /// Tiles this frame could generate.
    pub budget: usize,
    /// CPU milliseconds: results, plan, eviction, generation, drawn set,
    /// table.
    pub cpu_ms: [f64; 6],
    pub out_of_slots: bool,
    pub out_of_leaves: bool,
}

/// How [`ChunkWorld::update`] plans and how much it generates a frame.
#[derive(Clone, Copy, Debug)]
pub struct ResidencyConfig {
    pub plan: residency::PlanConfig,
    /// Tiles generated per frame at most.
    pub tiles_per_frame: usize,
    /// GPU milliseconds of generation a frame (measured per tile from the
    /// frames' timestamps).
    pub generate_ms: Option<f64>,
}

impl Default for ResidencyConfig {
    /// Cells of a pixel or less (a 45 degree view 1080 pixels tall), 400 km
    /// out, half a second of prefetch, two seconds of keep; up to 256 tiles
    /// and 3 ms of generation a frame.
    fn default() -> Self {
        Self {
            plan: residency::PlanConfig {
                pixel_angle: 2.0 * std::f64::consts::FRAC_PI_8.tan() / 1080.0,
                lod_pixels: 1.0,
                coarsen: 0.8,
                max_distance: 400_000.0,
                prefetch_seconds: 0.5,
                keep_frames: 120,
            },
            tiles_per_frame: 256,
            generate_ms: Some(3.0),
        }
    }
}

/// Chunk trees of one planet: resident chunks, their GPU storage and the
/// draw.
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
    /// The draw list built on the GPU (`cull_bin`, `cull_offsets`,
    /// `cull_scatter`).
    cull_layout: [wgpu::BindGroupLayout; 2],
    cull_bin: wgpu::ComputePipeline,
    cull_offsets: wgpu::ComputePipeline,
    cull_scatter: wgpu::ComputePipeline,
    trace: wgpu::ComputePipeline,
    shadow_sample_layout: wgpu::BindGroupLayout,
    shadow_sample: wgpu::ComputePipeline,
    /// `shadow_sample` and `sky_sample` for hits of exact levels.
    shadow_sample_far: wgpu::ComputePipeline,
    sky_sample_far: wgpu::ComputePipeline,
    sky_sample_layout: wgpu::BindGroupLayout,
    sky_sample: wgpu::ComputePipeline,
    shadow_layout: wgpu::BindGroupLayout,
    shadow: wgpu::ComputePipeline,
    shadow_fallback_layout: wgpu::BindGroupLayout,
    shadow_fallback: wgpu::ComputePipeline,
    resolve: wgpu::RenderPipeline,
    publish: wgpu::RenderPipeline,
    /// The last draw's bind group (its parameters and chunk storage).
    last_group: Option<wgpu::BindGroup>,
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
    materials: wgpu::Buffer,
    /// Chunk keys by slot, and the chunk table of the drawn chunks.
    chunk_keys: wgpu::Buffer,
    chunk_table: wgpu::Buffer,
    /// Base cells along a face edge.
    face_cells: u32,
    /// [`exact_level`].
    exact_level: u32,
    statics: wgpu::Buffer,
    draw_list: wgpu::Buffer,
    /// The drawn chunks' slots (uploaded when they change), the cull's
    /// counts (`Cull`), each drawn chunk's bin, and the seed's indirect draws.
    drawn_slots: wgpu::Buffer,
    cull_state: wgpu::Buffer,
    cull_bins: wgpu::Buffer,
    seed_args: wgpu::Buffer,
    /// GPU timestamps of the frames, read back without waiting.
    queries: Option<(wgpu::QuerySet, wgpu::Buffer)>,
    timing_slots: Vec<TimingSlot>,
    /// This frame's slot, when one is free.
    timing_now: Option<usize>,
    timings: Vec<FrameTimings>,
    /// Fragment, step and hit counts (`HELIO_VOXEL_CHUNKS_COUNT`).
    counters: wgpu::Buffer,
    /// Tiles rays reached under the chunks of (`Reach`), its readback
    /// slots, and the tiles whose whole volume is generated.
    reach: wgpu::Buffer,
    reach_slots: Vec<ReachSlot>,
    pick_slots: Vec<PickSlot>,
    /// Each slot's chunk key (`chunk_keys`, mirrored).
    slot_keys: Vec<[i32; 4]>,
    deep: FxHashSet<TileKey>,
    reached: Vec<TileKey>,
    evict_pipeline: wgpu::ComputePipeline,
    /// Chunk table changes scattered (`chunk_table.wgsl`).
    table_layout: wgpu::BindGroupLayout,
    table_pipeline: wgpu::ComputePipeline,
    free_blocks: wgpu::Buffer,
    job_results: wgpu::Buffer,
    /// Brushes kept analytic (resolved per face) and their buffer.
    /// Brushes kept analytic: `large_faces` of the large log, then the
    /// recent log's (`recent_counts` faces a brush).
    faces: Vec<FaceBrush>,
    large_faces: usize,
    recent_counts: Vec<usize>,
    brushes: wgpu::Buffer,
    /// The baked edits, and tiles that may hold their bricks.
    baked: EditStore,
    baked_tiles: FxHashSet<TileKey>,
    edit_state: EditState,
    /// Resident tiles to generate again: under brushes that came or went
    /// (first, nearest first), under baked edits that changed.
    urgent: FxHashSet<TileKey>,
    /// Urgent tiles' distances from the eye when first ranked (an edit
    /// settles in moments).
    urgent_distance: FxHashMap<TileKey, f64>,
    stale: FxHashSet<TileKey>,
    grid: Grid,
    /// Height over the datum above which nothing is solid (terrain,
    /// overhangs, edits), with a margin: rays rising past it end.
    ceiling: f64,
    /// The planet's height range in base layers (the roots' estimate).
    height_layers: (i32, i32),
    worker: Option<residency::Worker>,
    /// The latest finished plan, and events for the next.
    plan: residency::Plan,
    events: Vec<residency::Event>,
    tiles: FxHashMap<TileKey, TileChunks>,
    free_slots: Vec<u32>,
    serial: u64,
    readbacks: std::collections::VecDeque<Readback>,
    /// This frame's drawn chunks: tile and index in its chunks.
    drawn_set: drawn::DrawnSet,
    /// Tiles whose chunks changed since the drawn set last saw them.
    changed_tiles: FxHashSet<TileKey>,
    /// The drawn set's version whose slots `drawn_slots` holds.
    slots_version: Option<u64>,
    /// Results recorded by the last update, mapped once submitted.
    pending: Option<Readback>,
    last_requests: usize,
    /// What the latest plan was made for (none: no plan yet).
    planned_for: Option<residency::PlannedFor>,

    /// GPU cost of generation: measured (moving averages) from
    /// [`GenCost::PRIOR`].
    cost: GenCost,
    /// Tiles and chunks the last update generated.
    last_generated: (usize, usize),
}

/// What generation costs on the GPU: milliseconds of a tile's columns and
/// of a chunk's columns and tree, and chunks a tile.
#[derive(Clone, Copy, Debug)]
struct GenCost {
    tile: f64,
    chunk: f64,
    chunks_per_tile: f64,
}

impl GenCost {
    /// Before the first measured frame: an RTX 3060's (a 300 m/s cruise and
    /// a 50 m blast).
    const PRIOR: Self = Self { tile: 0.035, chunk: 0.007, chunks_per_tile: 1.7 };
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
    /// Sun visibility at each hit, and the sun's direction
    /// (Rgba16Float, Helio's `directional_visibility`), and its view.
    pub sun: wgpu::Texture,
    sun_view: wgpu::TextureView,
    /// The normal each hit is lit with (Rgba16Float).
    normal: wgpu::Texture,
    /// A sun sample a 2x2 block: flags, distance, slope (Rgba32Uint).
    sun_samples: wgpu::Texture,
    /// A sky visibility sample a 4x4 block: visibility, distance
    /// (Rg32Float).
    sky_samples: wgpu::Texture,
    /// Hits no sample serves: count, pixels; their indirect workgroups.
    fallbacks: wgpu::Buffer,
    fallback_groups: wgpu::Buffer,
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
        let sun = texture("chunk sun", wgpu::TextureFormat::Rgba16Float, s);
        Self {
            size,
            seed: texture("chunk seed", SEED_FORMAT, a),
            depth: texture("chunk depth", DEPTH_FORMAT, a),
            visibility: texture("chunk visibility", wgpu::TextureFormat::Rg32Uint, s),
            colour: texture("chunk colour", wgpu::TextureFormat::Rgba8Unorm, s),
            distance: texture("chunk distance", wgpu::TextureFormat::R32Float, s),
            exits: texture("chunk exits", wgpu::TextureFormat::R32Float, s | wgpu::TextureUsages::RENDER_ATTACHMENT),
            sun_view: view(&sun),
            sun,
            normal: texture("chunk normal", wgpu::TextureFormat::Rgba16Float, s),
            sun_samples: device.create_texture(&wgpu::TextureDescriptor {
                label: Some("chunk sun samples"),
                size: wgpu::Extent3d { width: size[0].div_ceil(2), height: size[1].div_ceil(2), depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Uint,
                usage: s,
                view_formats: &[],
            }),
            sky_samples: device.create_texture(&wgpu::TextureDescriptor {
                label: Some("chunk sky samples"),
                size: wgpu::Extent3d { width: size[0].div_ceil(4), height: size[1].div_ceil(4), depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rg32Float,
                usage: s,
                view_formats: &[],
            }),
            fallbacks: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk sun fallbacks"),
                size: 4 + 4 * u64::from(size[0]) * u64::from(size[1]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            fallback_groups: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk sun fallback workgroups"),
                size: 12,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
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

fn view_entry(binding: u32, v: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: wgpu::BindingResource::TextureView(v) }
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
/// The draw shader's walk templates (`// WALK {` .. `// } WALK`) twice:
/// with `walk` (`sun_walk`, `sky_visibility`), and with `walk_fine`
/// (`..._fine`: the kernels for hits finer than the exact levels).
fn walk_templates(source: &str) -> String {
    let mut out = String::with_capacity(source.len() * 2);
    let mut rest = source;
    while let Some(start) = rest.find("// WALK {\n") {
        let end = rest[start..].find("// } WALK\n").expect("closed walk template") + start;
        let body = &rest[start + "// WALK {\n".len()..end];
        out.push_str(&rest[..start]);
        out.push_str(body);
        out.push_str(
            &body
                .replace("walk(Walk(", "walk_fine(Walk(")
                .replace("fn sun_walk(", "fn sun_walk_fine(")
                .replace("fn sky_visibility(", "fn sky_visibility_fine("),
        );
        rest = &rest[end + "// } WALK\n".len()..];
    }
    out.push_str(rest);
    out
}

pub fn shader_sources(plane: bool, program: &crate::terrain::TerrainProgram, wide64: bool, counting: bool) -> [(&'static str, String); 2] {
    // Counters only in a counting build: side effects in a fragment shader
    // can turn early depth tests off.
    // Primary rays count in the trace (`COUNT`), in their walks when
    // `counted` (`WCOUNT`; shadow walks do not).
    let count = |n: u32| if counting { format!("atomicAdd(&counters[{n}], 1u);") } else { String::new() };
    let wcount = |n: u32| if counting { format!("if counted {{ atomicAdd(&counters[{n}], 1u); heat += u32({n} == 1); }}") } else { String::new() };
    // The noise library (materials' patches), then the draw.
    let draw = (format!("const DOMAIN_UNIT: f32 = {};\n", crate::grid::DOMAIN_UNIT) + include_str!("../../shaders/noise.wgsl") + &walk_templates(include_str!("../../shaders/chunk_draw.wgsl")))
        .replace("COUNTERS", if counting { "@group(0) @binding(9) var<storage, read_write> counters: array<atomic<u32>, 256>;" } else { "" })
        .replace("WCOUNT(1)", &wcount(1))
        .replace("WCOUNT(3)", &wcount(3))
        .replace("WCOUNT(8)", &wcount(8))
        .replace("WCOUNT(10)", &wcount(10))
        .replace("WCOUNT_BOXES", if counting { "if counted { atomicAdd(&counters[9], chunk); }" } else { "" })
        .replace("COUNT(0)", &count(0))
        .replace("COUNT(2)", &count(2))
        // Diagnostics (counting builds): the chunks one pixel's ray visits
        // (`HELIO_VOXEL_CHUNKS_DEBUG_PIXEL=x,y`), from counters[16].
        .replace(
            "TRACE_PIXEL(chunk, slot, cube_lo, real)",
            if counting {
                "if counted && all(pixel == params.debug.zw) && chunk < 48u { atomicStore(&counters[16u + chunk * 5u], slot); atomicStore(&counters[17u + chunk * 5u], bitcast<u32>(cube_lo.x)); atomicStore(&counters[18u + chunk * 5u], bitcast<u32>(cube_lo.y)); atomicStore(&counters[19u + chunk * 5u], bitcast<u32>(cube_lo.z)); atomicStore(&counters[20u + chunk * 5u], u32(real) + 2u * u32(key.x >> 8u)); atomicStore(&counters[4], chunk + 1u); }"
            } else {
                ""
            },
        )
        .replace("FADE_FRAMES", &drawn::FADE_FRAMES.to_string())
        .replace("PLANE_WORLD", if plane { "true" } else { "false" });
    [("chunk generation", terrain_source(plane, program, wide64) + include_str!("../../shaders/chunk_generate.wgsl")), ("chunk draw", draw)]
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
                storage(19, false, c),
                storage(20, false, c),
                storage(21, true, c),
                storage(22, true, c),
                storage(23, true, c),
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
        let evict_pipeline = compute("evict");
        let table_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("chunk table"), source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/chunk_table.wgsl").into()) });
        let table_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("chunk table"), entries: &[storage(0, true, c), storage(1, false, c)] });
        let table_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("chunk table"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&table_layout)], immediate_size: 0 })),
            module: &table_module,
            entry_point: Some("table_update"),
            compilation_options: Default::default(),
            cache: None,
        });

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
                storage(13, false, wgpu::ShaderStages::COMPUTE),
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
            entries: &[texture_entry(5, wgpu::TextureSampleType::Float { filterable: false }, wgpu::ShaderStages::FRAGMENT), storage(22, true, wgpu::ShaderStages::VERTEX)],
        });
        // The cull writes the draw list the draw layout reads: its own layouts.
        let cull_layout = [
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("chunk cull frames"),
                entries: &[uniform(0, c), storage(1, true, c), storage(2, true, c)],
            }),
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("chunk cull"),
                entries: &[storage(17, true, c), storage(18, false, c), storage(19, false, c), storage(20, false, c), storage(21, false, c)],
            }),
        ];
        let cull_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("chunk cull"), bind_group_layouts: &[Some(&cull_layout[0]), Some(&cull_layout[1])], immediate_size: 0 });
        let cull = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&cull_pipeline_layout),
                module: &draw_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let (cull_bin, cull_offsets, cull_scatter) = (cull("cull_bin"), cull("cull_offsets"), cull("cull_scatter"));
        let resolve_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("chunk depth resolve"),
            entries: &[
                texture_entry(4, wgpu::TextureSampleType::Float { filterable: false }, wgpu::ShaderStages::FRAGMENT),
                texture_entry(7, wgpu::TextureSampleType::Uint, wgpu::ShaderStages::FRAGMENT),
                texture_entry(14, wgpu::TextureSampleType::Float { filterable: false }, wgpu::ShaderStages::FRAGMENT),
            ],
        });
        let layout = |second: &wgpu::BindGroupLayout| device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&draw_layout), Some(second)], immediate_size: 0 });
        let seed_layout = layout(&seed_layout_1);
        let depth = |compare: wgpu::CompareFunction| {
            Some(wgpu::DepthStencilState { format: DEPTH_FORMAT, depth_write_enabled: Some(true), depth_compare: Some(compare), stencil: Default::default(), bias: Default::default() })
        };
        let seed = |vertex: &str, fragment: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(fragment),
                layout: Some(&seed_layout),
                vertex: wgpu::VertexState { module: &draw_module, entry_point: Some(vertex), compilation_options: Default::default(), buffers: &[] },
                primitive: wgpu::PrimitiveState { cull_mode: Some(wgpu::Face::Back), front_face: wgpu::FrontFace::Ccw, ..Default::default() },
                depth_stencil: depth(wgpu::CompareFunction::Less),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState { module: &draw_module, entry_point: Some(fragment), compilation_options: Default::default(), targets: &[Some(SEED_FORMAT.into())] }),
                multiview_mask: None,
                cache: None,
            })
        };
        let seed_outside = seed("seed_vertex", "seed_fragment");
        let seed_inside = seed("seed_vertex_inside", "seed_fragment_inside");
        let trace = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("chunk trace"),
            layout: Some(&layout(&trace_layout)),
            module: &draw_module,
            entry_point: Some("trace"),
            compilation_options: Default::default(),
            cache: None,
        });
        // Sun shadows: samples a 2x2 block, then every hit.
        let hits = [
            texture_entry(4, wgpu::TextureSampleType::Float { filterable: false }, wgpu::ShaderStages::COMPUTE),
            texture_entry(7, wgpu::TextureSampleType::Uint, wgpu::ShaderStages::COMPUTE),
        ];
        let shadow_sample_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("chunk sun samples"),
            entries: &[hits[0], hits[1], storage_texture(9, wgpu::TextureFormat::Rgba32Uint)],
        });
        let shadow_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("chunk shadow"),
            entries: &[
                hits[0],
                hits[1],
                storage_texture(2, wgpu::TextureFormat::Rgba8Unorm),
                storage_texture(8, wgpu::TextureFormat::Rgba16Float),
                texture_entry(10, wgpu::TextureSampleType::Uint, wgpu::ShaderStages::COMPUTE),
                storage(11, false, wgpu::ShaderStages::COMPUTE),
                storage(12, false, wgpu::ShaderStages::COMPUTE),
                storage_texture(13, wgpu::TextureFormat::Rgba16Float),
                texture_entry(16, wgpu::TextureSampleType::Float { filterable: false }, wgpu::ShaderStages::COMPUTE),
            ],
        });
        let sky_sample_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("chunk sky samples"),
            entries: &[hits[0], hits[1], storage_texture(15, wgpu::TextureFormat::Rg32Float)],
        });
        // The fallbacks' walks: their workgroups are the dispatch's.
        let shadow_fallback_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("chunk shadow fallbacks"),
            entries: &[hits[0], hits[1], storage_texture(2, wgpu::TextureFormat::Rgba8Unorm), storage_texture(8, wgpu::TextureFormat::Rgba16Float), storage(11, false, wgpu::ShaderStages::COMPUTE), storage_texture(13, wgpu::TextureFormat::Rgba16Float)],
        });
        let compute = |label: &str, layout: &wgpu::PipelineLayout, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                module: &draw_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let shadow_sample = compute("chunk sun samples", &layout(&shadow_sample_layout), "shadow_sample");
        let sky_sample = compute("chunk sky samples", &layout(&sky_sample_layout), "sky_sample");
        let shadow_sample_far = compute("chunk far sun samples", &layout(&shadow_sample_layout), "shadow_sample_far");
        let sky_sample_far = compute("chunk far sky samples", &layout(&sky_sample_layout), "sky_sample_far");
        let shadow = compute("chunk shadow", &layout(&shadow_layout), "shadow");
        let shadow_fallback = compute("chunk shadow fallbacks", &layout(&shadow_fallback_layout), "shadow_fallback");
        let publish = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("chunk gbuffer"),
            layout: Some(&layout(&resolve_layout)),
            vertex: wgpu::VertexState { module: &draw_module, entry_point: Some("resolve_vertex"), compilation_options: Default::default(), buffers: &[] },
            primitive: Default::default(),
            depth_stencil: depth(wgpu::CompareFunction::Less),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &draw_module,
                entry_point: Some("publish"),
                compilation_options: Default::default(),
                targets: &crate::engine::GBUFFER_FORMATS.map(|format| Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })),
            }),
            multiview_mask: None,
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
        // Brushes kept analytic: the large ones, then the recent ones.
        let edits = planet.edits();
        let (faces, large_faces, recent_counts) = analytic_faces(edits);
        let s = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let world = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("chunk world"), contents: bytemuck::bytes_of(&WorldGpu::new(planet)), usage: wgpu::BufferUsages::UNIFORM });
        let terrain = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("chunk terrain"), contents: &terrain_bytes(&program), usage: wgpu::BufferUsages::UNIFORM });
        let (materials, detail) = crate::engine::material_table(&planet.field().appearance());
        let materials = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("chunk materials"), contents: bytemuck::bytes_of(&ChunkMaterials { materials, detail }), usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST });
        let queries = device.features().contains(wgpu::Features::TIMESTAMP_QUERY).then(|| {
            (
                device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("chunk draw"), ty: wgpu::QueryType::Timestamp, count: QUERIES * TIMING_RING as u32 }),
                buffer("chunk timestamps", u64::from(wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT) * TIMING_RING, wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC),
            )
        });
        let timing_slots = (0..if queries.is_some() { TIMING_RING } else { 0 })
            .map(|_| TimingSlot { buffer: buffer("chunk timestamps read", u64::from(QUERIES) * 8, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST), frame: None, ready: None })
            .collect();
        let drawn_set = drawn::DrawnSet::new(MAX_CHUNKS, TABLE_SLOTS, planet.grid().levels() - 1);
        let chunk_table = buffer("chunk table", (TABLE_SLOTS * 32) as u64, s);
        queue.write_buffer(&chunk_table, 0, bytemuck::cast_slice(drawn_set.table()));
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
            cull_layout,
            cull_bin,
            cull_offsets,
            cull_scatter,
            trace,
            shadow_sample_layout,
            shadow_sample,
            shadow_sample_far,
            sky_sample_far,
            sky_sample_layout,
            sky_sample,
            shadow_layout,
            shadow,
            shadow_fallback_layout,
            shadow_fallback,
            resolve,
            publish,
            last_group: None,
            targets: None,
            rounds: std::env::var("HELIO_VOXEL_CHUNKS_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(1),
            world,
            terrain,
            tile_jobs: buffer("chunk tile jobs", (TILE_BATCH * 16) as u64, s),
            tile_data: buffer("chunk tile columns", TILE_BATCH as u64 * TILE_COLUMNS * 16, wgpu::BufferUsages::STORAGE),
            tile_ranges: buffer("chunk tile ranges", (TILE_BATCH * 16) as u64, s | wgpu::BufferUsages::COPY_SRC),
            chunk_jobs: buffer("chunk jobs", (CHUNK_BATCH * 32) as u64, s),
            column_masks: buffer("chunk column masks", CHUNK_BATCH as u64 * TILE_COLUMNS * 8, wgpu::BufferUsages::STORAGE),
            chunk_meta: buffer("chunk meta", (MAX_CHUNKS * 48) as u64, s | wgpu::BufferUsages::COPY_SRC),
            mids: buffer("chunk mids", (MAX_CHUNKS * 64 * 16) as u64, wgpu::BufferUsages::STORAGE),
            mid_leaves: buffer("chunk mid leaves", (MAX_CHUNKS * 64 * 4) as u64, wgpu::BufferUsages::STORAGE),
            leaves: buffer("chunk leaves", LEAF_CAPACITY * 8, wgpu::BufferUsages::STORAGE),
            leaf_alloc: buffer("chunk leaf allocator", 4 * (2 + LEAF_CLASSES as u64), s | wgpu::BufferUsages::COPY_SRC),
            column_materials: buffer("chunk column materials", CHUNK_BATCH as u64 * TILE_COLUMNS * 4, wgpu::BufferUsages::STORAGE),
            leaf_materials: buffer("chunk leaf materials", LEAF_CAPACITY * 2, s),
            mid_materials: buffer("chunk mid materials", (MAX_CHUNKS * 64 * 4) as u64, wgpu::BufferUsages::STORAGE),
            params: buffer("chunk draw params", std::mem::size_of::<DrawParams>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
            materials,
            chunk_keys: buffer("chunk keys", (MAX_CHUNKS * 16) as u64, s),
            chunk_table,

            face_cells: planet.grid().cells() as u32,
            exact_level: exact_level(planet.grid()),
            statics: buffer("chunk statics", (MAX_CHUNKS * std::mem::size_of::<ChunkStatic>()) as u64, s),
            draw_list: buffer("chunk draw list", (MAX_CHUNKS * 4) as u64, s),
            drawn_slots: buffer("chunk drawn slots", (MAX_CHUNKS * 4) as u64, s),
            cull_state: buffer("chunk cull", CULL_BYTES, s | wgpu::BufferUsages::COPY_SRC),
            cull_bins: buffer("chunk cull bins", (MAX_CHUNKS * 4) as u64, s),
            seed_args: buffer("chunk seed draws", 32, s | wgpu::BufferUsages::INDIRECT),
            queries,
            timing_slots,
            timing_now: None,
            timings: Vec::new(),
            counters: buffer("chunk counters", 1024, s | wgpu::BufferUsages::COPY_SRC),
            reach: buffer("chunk reach", REACH_BYTES, s | wgpu::BufferUsages::COPY_SRC),
            reach_slots: (0..TIMING_RING).map(|_| ReachSlot { buffer: buffer("chunk reach read", REACH_BYTES, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST), state: ReachState::Free }).collect(),
            deep: FxHashSet::default(),
            reached: Vec::new(),
            pick_slots: (0..3).map(|_| PickSlot { buffer: buffer("chunk picks", 256 * MAX_PICKS as u64, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST), requests: Vec::new(), sink: None, state: ReachState::Free }).collect(),
            slot_keys: vec![[-1, 0, 0, 0]; MAX_CHUNKS],
            evict_pipeline,
            table_layout,
            table_pipeline,
            free_blocks: buffer("chunk free blocks", free_block_entries() * 4, wgpu::BufferUsages::STORAGE),
            job_results: buffer("chunk job results", (CHUNK_BATCH * 16) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
            brushes: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("chunk brushes"),
                contents: bytemuck::cast_slice(if faces.is_empty() { &NO_BRUSH[..] } else { &faces }),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            faces,
            large_faces,
            recent_counts,
            baked: edits.baked.clone(),
            baked_tiles: edits.baked.bricks().map(|(k, _)| TileKey { face: k.face, level: u32::from(k.level), cx: k.bi >> 3, cy: k.bj >> 3 }).collect(),
            edit_state: EditState::of(edits),
            urgent: FxHashSet::default(),
            urgent_distance: FxHashMap::default(),
            stale: FxHashSet::default(),
            grid: *planet.grid(),
            ceiling: planet.outer_radius() - planet.grid().radius() + CEILING_MARGIN,
            height_layers: (
                (planet.min_terrain_height() / planet.grid().voxel_size()).floor() as i32 - 64,
                (planet.max_terrain_height() / planet.grid().voxel_size()).ceil() as i32 + 64,
            ),
            worker: None,
            plan: Default::default(),
            events: Vec::new(),
            tiles: Default::default(),
            free_slots: (0..MAX_CHUNKS as u32).rev().collect(),
            serial: 0,
            readbacks: std::collections::VecDeque::new(),
            drawn_set,
            changed_tiles: Default::default(),
            slots_version: None,
            pending: None,
            last_requests: 0,
            planned_for: None,

            cost: GenCost::PRIOR,
            last_generated: (0, 0),
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

    fn gen_group(&self, brush_refs: &wgpu::Buffer, evictions: &wgpu::Buffer, baked_refs: &wgpu::Buffer, baked_cells: &wgpu::Buffer) -> wgpu::BindGroup {
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
                entry(7, &self.brushes),
                entry(8, brush_refs),
                entry(9, &self.chunk_meta),
                entry(10, &self.mids),
                entry(11, &self.mid_leaves),
                entry(12, &self.leaves),
                entry(13, &self.leaf_alloc),
                entry(14, &self.column_materials),
                entry(15, &self.leaf_materials),
                entry(17, &self.mid_materials),
                entry(19, &self.job_results),
                entry(20, &self.free_blocks),
                entry(21, evictions),
                entry(22, baked_refs),
                entry(23, baked_cells),
            ],
        })
    }

    /// `tile`'s field `ranges` with the layers its edits change: the bands
    /// of the brushes that add or remove over it ([`Self::surface_bands`]),
    /// and its baked bricks.
    /// Residency measures a tile's distance by them (the floor of a deep dig
    /// is near an eye down there).
    fn with_edits(&self, tile: TileKey, ranges: residency::Ranges) -> residency::Ranges {
        let shift = tile.level + 1;
        let span = |c: i32| (i64::from(c * CHUNK) << shift, (i64::from(c * CHUNK + CHUNK) << shift) - 1);
        let (mut lo, mut hi) = (i32::MAX, i32::MIN);
        for fb in &self.faces {
            if fb.op() <= 1 && brush_meets(fb, tile.face, tile.level, span(tile.cx), span(tile.cy), (i64::MIN / 4, i64::MAX / 4)) {
                for band in self.surface_bands(fb, tile).into_iter().filter(|b| b.1 > b.0) {
                    lo = lo.min(band.0);
                    hi = hi.max(band.1);
                }
            }
        }
        if self.baked_tiles.contains(&tile) {
            for by in 0..8 {
                for bx in 0..8 {
                    let key = ColumnKey { face: tile.face, level: tile.level as u8, ci: tile.cx * 8 + bx, cj: tile.cy * 8 + by };
                    for (bk, _) in self.baked.column(&key).into_iter().flat_map(|c| c.iter()) {
                        lo = lo.min(bk * 8);
                        hi = hi.max(bk * 8 + 8);
                    }
                }
            }
        }
        residency::Ranges { edits: if lo > hi { residency::NO_EDITS } else { (lo, hi) }, ..ranges }
    }

    /// Layers [lo, hi) of `tile`'s level where brush `fb`'s surface can be
    /// over the tile, with a cell of margin (none: lo >= hi). A brush's
    /// heights; a ball's, the heights of its sphere's lower and upper caps
    /// over the tile (one band where the tile reaches the sphere's
    /// silhouette; the air inside holds nothing). Along a direction at
    /// angle t from the ball's centre (d from the planet's, radius r) the
    /// caps lie at d cos t -+ sqrt(r^2 - d^2 sin^2 t): the lower rises with
    /// t, the upper falls, so the tile's nearest and farthest angles bound
    /// them.
    fn surface_bands(&self, fb: &FaceBrush, tile: TileKey) -> [(i32, i32); 2] {
        let grid = &self.grid;
        let shift = tile.level + 1;
        let band = |k_lo: i32, k_hi: i32| ((k_lo.max(fb.k_lo) >> shift) - 1, (k_hi.min(fb.k_hi) >> shift) + 2);
        const NO_BAND: (i32, i32) = (0, 0);
        if fb.shape() != crate::edits::SHAPE_BALL || grid.is_plane() {
            return [band(fb.k_lo, fb.k_hi), NO_BAND];
        }
        let units = f64::from(grid.sphere_constants()[2]) / grid.radius();
        let centre = glam::IVec3::from_slice(&fb.ball[..3]).as_dvec3() / units;
        let (d, r) = (centre.length(), f64::from(fb.ball[3].unsigned_abs()) / units);
        // The tile's directions in the brushes' (domain) space: its middle,
        // and the angle to its farthest corner cell (its edges are great
        // circles) and two base cells more.
        let size = CHUNK << tile.level;
        let (i0, j0) = (tile.cx * size, tile.cy * size);
        let towards = |i: i32, j: i32| grid.domain_point(tile.face, i, j, 0).as_dvec3().normalize();
        let middle = towards(i0 + size / 2, j0 + size / 2);
        let spread = [(0, 0), (1, 0), (0, 1), (1, 1)]
            .iter()
            .map(|&(a, b)| middle.angle_between(towards(i0 + a * (size - 1), j0 + b * (size - 1))))
            .fold(0.0, f64::max)
            * 1.01
            + 2.0 * grid.voxel_size() / grid.radius();
        let t = if d > 0.0 { middle.angle_between(centre / d) } else { 0.0 };
        let (near, far) = ((t - spread).max(0.0), (t + spread).min(std::f64::consts::PI));
        // Half base cells of a radius, two cells of margin.
        let k = |radius: f64, margin: f64| (((radius - grid.radius()) / grid.voxel_size() + margin) * 2.0).clamp(-2.0e9, 2.0e9) as i32;
        let cap = |t: f64, sign: f64| d * t.cos() + sign * (r * r - (d * t.sin()).powi(2)).max(0.0).sqrt();
        if d <= r {
            // The planet's centre is inside: the upper cap alone.
            return [band(k(cap(far, 1.0), -2.0), k(cap(near, 1.0), 2.0)), NO_BAND];
        }
        let silhouette = (r / d).asin();
        if near >= silhouette {
            return [NO_BAND; 2];
        }
        if far >= silhouette {
            return [band(k(cap(near, -1.0), -2.0), k(cap(near, 1.0), 2.0)), NO_BAND];
        }
        [band(k(cap(near, -1.0), -2.0), k(cap(far, -1.0), 2.0)), band(k(cap(far, 1.0), -2.0), k(cap(near, 1.0), 2.0))]
    }

    /// The chunk layers `tile` generates and what lies where it has none.
    /// Its chunks hold every surface: the field's band
    /// ([`layers_needed`]), each brush's over it
    /// ([`Self::surface_bands`]) and its baked bricks', in bands two
    /// chunks apart or more. A region between bands, or under them, holds
    /// no surface, so it is all rock or all air across the tile: one cell
    /// of the planet tells which. Rock between bands is a solid gap (one;
    /// a second keeps its chunks), air under them a hollow; elsewhere
    /// missing layers are air between bands and rock under them.
    fn layers(&self, tile: TileKey, ranges: residency::Ranges, eye: &Eye, deep: bool) -> Layers {
        let grid = &self.grid;
        let shift = tile.level + 1;
        let span = |c: i32| (i64::from(c * CHUNK) << shift, (i64::from(c * CHUNK + CHUNK) << shift) - 1);
        let mut bands: Vec<(i32, i32)> = Vec::new();
        let (lo, hi) = layers_needed(grid, tile, ranges, eye, deep);
        if hi > lo {
            bands.push((lo, hi));
        }
        // The brushes that add or remove over the tile, in order.
        let mut shapes = Vec::new();
        for fb in &self.faces {
            if fb.op() <= 1 && brush_meets(fb, tile.face, tile.level, span(tile.cx), span(tile.cy), (i64::MIN / 4, i64::MAX / 4)) {
                let before = bands.len();
                bands.extend(self.surface_bands(fb, tile).into_iter().filter(|b| b.1 > b.0));
                if bands.len() > before {
                    shapes.push(fb);
                }
            }
        }
        if self.baked_tiles.contains(&tile) {
            for by in 0..8 {
                for bx in 0..8 {
                    let key = ColumnKey { face: tile.face, level: tile.level as u8, ci: tile.cx * 8 + bx, cj: tile.cy * 8 + by };
                    for (bk, _) in self.baked.column(&key).into_iter().flat_map(|c| c.iter()) {
                        bands.push((bk * 8, bk * 8 + 8));
                    }
                }
            }
        }
        // Without edits a tile is the field's band: rock under it, air over.
        let edited = bands.len() > usize::from(hi > lo);
        // Chunk layers, merged where fewer than two chunks apart.
        let mut layers: Vec<(i32, i32)> = bands.iter().filter(|b| b.1 > b.0).map(|&(a, b)| (a.div_euclid(CHUNK), (b - 1).div_euclid(CHUNK))).collect();
        layers.sort_unstable();
        let mut merged: Vec<(i32, i32)> = Vec::new();
        for (a, b) in layers {
            match merged.last_mut() {
                Some(last) if a <= last.1 + 2 => last.1 = last.1.max(b),
                _ => merged.push((a, b)),
            }
        }
        // Only regions clear of the field's volume (its caves may lie under
        // the surface band) hold no surface; a gap within it keeps its
        // chunks, and under the chunks stays rock reached through mouths.
        let volume = (ranges.volume.0.div_euclid(CHUNK), (ranges.volume.1 - 1).div_euclid(CHUNK));
        // What fills a region without surfaces: at a cell of it under the
        // tile's centre, the last brush that adds or removes there, else the
        // field (rock under its volume, air over it). Baked edits lie in
        // bands of their own.
        let solid = |z: i32| {
            let (i, j) = (tile.cx * CHUNK + CHUNK / 2, tile.cy * CHUNK + CHUNK / 2);
            let centre = [center_half(i, tile.level), center_half(j, tile.level), center_half(z, tile.level)];
            let mut point = None;
            shapes
                .iter()
                .rev()
                .find(|b| b.contains(centre, || *point.get_or_insert_with(|| grid.volume_point(tile.face, i, j, z, tile.level))))
                .map_or(z < ranges.volume.0, |b| b.op() == 1)
        };
        let clear = |a: i32, b: i32| b <= volume.0 || a > volume.1;
        let mut gap = drawn::Gap::default();
        if let Some(&(first, _)) = merged.first().filter(|_| edited) {
            gap.below_air = first <= volume.0 && !solid(first * CHUNK - CHUNK / 2);
            // Solid gaps: the widest is marked, the others keep chunks.
            let mut fill = Vec::new();
            let mut rock = Vec::new();
            for w in merged.windows(2) {
                let (a, b) = (w[0].1 + 1, w[1].0);
                if !clear(a, b) {
                    fill.push((a, b - 1));
                } else if solid((a + b) * CHUNK / 2) {
                    rock.push((a, b));
                }
            }
            if let Some(&widest) = rock.iter().max_by_key(|g| g.1 - g.0) {
                gap.solid = widest;
                fill.extend(rock.iter().filter(|&&g| g != widest).map(|&(a, b)| (a, b - 1)));
            }
            merged.extend(fill);
            merged.sort_unstable();
        }
        Layers { bands: merged, gap }
    }

    /// Jobs generating `key` (with `ranges`) into `batch`; its old chunks,
    /// if any, to `evictions`. False when the batch or the slots are full.
    #[allow(clippy::too_many_arguments)]
    fn queue_tile(&mut self, key: TileKey, ranges: residency::Ranges, eye_at: &Eye, batch: &mut GenBatch, readback: &mut Readback, evictions: &mut Vec<u32>, stats: &mut UpdateStats) -> bool {
        let grid = self.grid;
        let ranges = self.with_edits(key, ranges);
        let layers = self.layers(key, ranges, eye_at, self.deep.contains(&key));
        let count: usize = layers.bands.iter().map(|b| (b.1 - b.0 + 1) as usize).sum();
        if batch.tile_jobs.len() >= TILE_BATCH || batch.chunk_jobs.len() + count > CHUNK_BATCH {
            return false;
        }
        if self.free_slots.len() < count {
            stats.out_of_slots = true;
            return false;
        }
        let index = batch.tile_jobs.len() as u32;
        batch.tile_jobs.push(TileJob { face: u32::from(key.face), level: key.level, cx: key.cx, cy: key.cy });
        readback.tiles.push(key);
        let shift = key.level + 1;
        let span = |c: i32| (i64::from(c * CHUNK) << shift, (i64::from(c * CHUNK + CHUNK) << shift) - 1);
        let baked = self.baked_tiles.contains(&key);
        let mut chunks = Vec::new();
        for cz in layers.bands.iter().flat_map(|&(a, b)| a..=b) {
            let slot = self.free_slots.pop().unwrap();
            self.serial += 1;
            let first = batch.refs.len() as u32;
            let mut large = 0;
            for (b, fb) in self.faces.iter().enumerate() {
                if brush_meets(fb, key.face, key.level, span(key.cx), span(key.cy), span(cz)) {
                    batch.refs.push(b as u32);
                    large += u32::from(b < self.large_faces);
                }
            }
            let baked = if baked { batch.baked_table(&self.baked, key, cz) } else { u32::MAX };
            batch.chunk_jobs.push(ChunkJob { tile: index, cz, slot, brushes: first, brush_count: batch.refs.len() as u32 - first, large, baked, _pad: 0 });
            // Layers not read back yet: the whole chunk.
            let record = chunk_static(&grid, key, cz, slot, (0.0, 64.0), 32.0, self.exact_level);
            self.queue.write_buffer(&self.statics, u64::from(slot) * std::mem::size_of::<ChunkStatic>() as u64, bytemuck::bytes_of(&record));
            chunks.push(Chunk { cz, slot, serial: self.serial, z: None });
            self.queue.write_buffer(&self.chunk_keys, u64::from(slot) * 16, bytemuck::cast_slice(&chunk_key(key, cz)));
            self.slot_keys[slot as usize] = chunk_key(key, cz);
            readback.chunks.push((key, cz, slot, self.serial));
        }
        stats.generated_chunks += chunks.len();
        if let Some(old) = self.tiles.insert(key, TileChunks { chunks, layers, ranges }) {
            evictions.extend(old.chunks.iter().map(|c| c.slot));
        }
        self.changed_tiles.insert(key);
        self.events.push(residency::Event::Generated(key));
        stats.generated_tiles += 1;
        true
    }

    /// Follow `planet`'s edits (the recipe unchanged): resident tiles under
    /// brushes that came or went are generated again in the next update;
    /// under baked edits that changed (a seal's coarser levels), within the
    /// budget. False when its edits do not follow these (another history):
    /// build the world again.
    pub fn sync_edits(&mut self, planet: &Planet) -> bool {
        let edits = planet.edits();
        let state = EditState::of(edits);
        if state == self.edit_state {
            return true;
        }
        let Some(columns) = edits.baked_changes_since(self.edit_state.sealed) else { return false };
        let (faces, large_faces, recent_counts) = analytic_faces(edits);
        // Brushes in one list and not the other.
        let mut count: FxHashMap<[u32; 12], i32> = FxHashMap::default();
        for fb in &self.faces {
            *count.entry(bytemuck::cast(*fb)).or_default() -= 1;
        }
        for fb in &faces {
            *count.entry(bytemuck::cast(*fb)).or_default() += 1;
        }
        // Recent brushes sealed into the baked store left the list, not the
        // world (its coarser levels follow from the baked changes).
        let sealed = state.sealed.0.saturating_sub(self.edit_state.sealed.0);
        let sealed_faces: usize = self.recent_counts.iter().take(sealed).sum();
        for fb in &self.faces[self.large_faces..self.large_faces + sealed_faces] {
            if let Some(n) = count.get_mut(&bytemuck::cast::<FaceBrush, [u32; 12]>(*fb)) {
                if *n < 0 {
                    *n += 1;
                }
            }
        }
        for (words, n) in count {
            if n != 0 {
                self.mark_brush(&bytemuck::cast(words));
            }
        }
        for column in columns {
            let key = TileKey { face: column.face, level: u32::from(column.level), cx: column.ci >> 3, cy: column.cj >> 3 };
            self.baked_tiles.insert(key);
            if self.tiles.contains_key(&key) && !self.urgent.contains(&key) {
                self.stale.insert(key);
            }
        }
        self.brushes = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk brushes"),
            contents: bytemuck::cast_slice(if faces.is_empty() { &NO_BRUSH[..] } else { &faces }),
            usage: wgpu::BufferUsages::STORAGE,
        });
        self.faces = faces;
        self.large_faces = large_faces;
        self.recent_counts = recent_counts;
        self.baked = edits.baked.clone();
        self.edit_state = state;
        self.ceiling = planet.outer_radius() - planet.grid().radius() + CEILING_MARGIN;
        true
    }

    /// Resident tiles brush `fb` can change, at every level it applies at.
    fn mark_brush(&mut self, fb: &FaceBrush) {
        let (ci, cj, w) = (i64::from(fb.center[0]), i64::from(fb.center[1]), i64::from(fb.center[3]));
        for level in 0..self.grid.levels() {
            if !fb.active(level) {
                break;
            }
            // Half base cells to tiles of the level.
            let tile = |half: i64| (half.div_euclid(2) >> level).div_euclid(i64::from(CHUNK)) as i32;
            for cx in tile(ci - w)..=tile(ci + w) {
                for cy in tile(cj - w)..=tile(cj + w) {
                    let key = TileKey { face: fb.face(), level, cx, cy };
                    if self.tiles.contains_key(&key) {
                        self.stale.remove(&key);
                        self.urgent.insert(key);
                    }
                }
            }
        }
    }

    /// One frame of residency for an eye at `eye` moving at `velocity`,
    /// recorded into `encoder` (submit it before the next update): take
    /// back finished generation results, plan, evict, and generate the
    /// most wanted tiles within the budget. [`Self::draw`] then shows the
    /// plan's tiles.
    pub fn update(&mut self, encoder: &mut helio_core::CommandRecorder<'_>, eye: DVec3, velocity: DVec3, view: Option<residency::View>, frame: u64, config: &ResidencyConfig) -> UpdateStats {
        let grid = self.grid;
        let eye_at = Eye::new(&grid, eye);
        self.claim_timing(frame);
        let mut stats = UpdateStats::default();
        if self.worker.is_none() {
            self.worker = Some(residency::Worker::spawn(grid, self.height_layers, config.plan));
        }
        let mut evictions: Vec<u32> = Vec::new();

        let mut clock = std::time::Instant::now();
        let mut lap = |stats: &mut UpdateStats, i: usize| {
            stats.cpu_ms[i] = clock.elapsed().as_secs_f64() * 1e3;
            clock = std::time::Instant::now();
        };
        // 1. Finished generation results: measured layers and occupancy.
        let _ = self.device.poll(wgpu::PollType::Poll);
        while self.readbacks.front().is_some_and(|r| r.ready.load(std::sync::atomic::Ordering::Acquire)) {
            let r = self.readbacks.pop_front().unwrap();
            {
                let data = r.buffer.slice(..).get_mapped_range().unwrap();
                let words: &[i32] = bytemuck::cast_slice(&data);
                for (n, key) in r.tiles.iter().enumerate() {
                    let w = &words[n * 4..n * 4 + 4];
                    if w[0] > w[1] {
                        continue;
                    }
                    let ranges = self.with_edits(*key, residency::Ranges { volume: (w[0], w[1]), surface: (w[2], w[1]), edits: residency::NO_EDITS });
                    self.events.push(residency::Event::Measured(*key, ranges));
                    if let Some(t) = self.tiles.get_mut(key) {
                        t.ranges = ranges;
                    }
                    // Layers its chunks must hold beyond those generated
                    // (measured, or brushes since): generate the tile again
                    // in place, its chunks drawn until then.
                    let need = self.layers(*key, ranges, &eye_at, self.deep.contains(key));
                    if self.tiles.get(key).is_some_and(|t| !t.layers.holds(&need)) {
                        self.stale.remove(key);
                        self.urgent.insert(*key);
                        stats.regenerated_tiles += 1;
                    }
                }
                let base = r.tiles.len() * 4;
                for (n, &(key, _, _, serial)) in r.chunks.iter().enumerate() {
                    let w = &words[base + n * 4..base + n * 4 + 4];
                    let Some(tile) = self.tiles.get_mut(&key) else { continue };
                    let Some(i) = tile.chunks.iter().position(|c| c.serial == serial) else { continue };
                    self.changed_tiles.insert(key);
                    if w[3] & 8 != 0 {
                        stats.out_of_leaves = true;
                    }
                    if w[3] & (2 | 8) != 0 {
                        // Nothing to draw: empty, or not stored. (A chunk
                        // solid throughout stays: its faces are the
                        // surface where it lies on a chunk boundary.)
                        evictions.push(tile.chunks.swap_remove(i).slot);
                    } else {
                        let c = &mut tile.chunks[i];
                        c.z = Some((w[1], w[2]));
                        // Its frame about its columns' mean top (its ground),
                        // else its occupied layers.
                        let surface = (w[3] as u32) >> 16;
                        let about = if surface == 0xffff { (w[1] + w[2]) as f32 * 0.5 } else { surface as f32 / 4.0 };
                        let record = chunk_static(&self.grid, key, c.cz, c.slot, (w[1] as f32, w[2] as f32), about, self.exact_level);
                        self.queue.write_buffer(&self.statics, u64::from(c.slot) * std::mem::size_of::<ChunkStatic>() as u64, bytemuck::bytes_of(&record));
                    }
                }
            }
            r.buffer.unmap();
        }

        // Tiles rays reached under: their whole volume, generated again now.
        for key in std::mem::take(&mut self.reached) {
            if self.deep.insert(key) && self.tiles.contains_key(&key) {
                self.stale.remove(&key);
                self.urgent.insert(key);
                stats.deep_tiles += 1;
            }
        }
        lap(&mut stats, 0);
        // 2. The latest plan (made on the worker from an eye a frame or
        // two old) and the next; 3. evict what it no longer needs.
        let mut evicted = Vec::new();
        let (mut added, mut removed) = (Vec::new(), Vec::new());
        let worker = self.worker.as_mut().expect("spawned above");
        if let Some(planned) = worker.poll() {
            self.plan = planned.plan;
            stats.switches = self.plan.switches;
            stats.switch_px = self.plan.switch_px;
            evicted = planned.evicted;
            added = planned.added;
            removed = planned.removed;
            self.planned_for = Some(planned.made_for);
        }
        // Events the plan has not seen, or another eye or view (a camera
        // that moved since): its requests are not yet known.
        let unseen = !self.events.is_empty() || self.planned_for != Some(residency::PlannedFor { events_seen: worker.events_sent(), eye, view });
        worker.submit(eye, velocity, view, frame, &mut self.events);
        lap(&mut stats, 1);
        let plan = &self.plan;
        stats.requests = plan.waiting;
        stats.coarse_share = plan.coarse_share;
        stats.worst_px = plan.worst_px;
        for key in evicted {
            if let Some(old) = self.tiles.remove(&key) {
                evictions.extend(old.chunks.iter().map(|c| c.slot));
            }
            self.changed_tiles.insert(key);
            stats.evicted_tiles += 1;
        }
        lap(&mut stats, 2);

        // 4. Generate: tiles edits changed under (those brushes changed, all
        // now; baked edits, within the budget), then the most wanted.
        let mut batch = GenBatch::default();
        let mut readback = Readback {
            buffer: self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("chunk results"), size: 16, usage: wgpu::BufferUsages::MAP_READ, mapped_at_creation: false }),
            ready: Default::default(),
            tiles: Vec::new(),
            chunks: Vec::new(),
        };
        // The frame's generation: `tiles_per_frame` tiles at most and, with a
        // time budget, the measured cost of their columns and chunks within
        // it (eight tiles at least): a tall edited tile costs its chunks.
        let cost = config.generate_ms.map(|ms| (ms, self.cost));
        let most = config.tiles_per_frame.min(TILE_BATCH);
        stats.budget = cost.map_or(most, |(ms, c)| ((ms / (c.tile + c.chunk * c.chunks_per_tile)) as usize).clamp(8, most));
        let full = |batch: &GenBatch| {
            let tiles = batch.tile_jobs.len();
            tiles >= most || cost.is_some_and(|(ms, c)| tiles >= 8 && tiles as f64 * c.tile + batch.chunk_jobs.len() as f64 * c.chunk >= ms)
        };
        // Tiles edits changed: drawn ones first, nearest first, within the
        // frame's budget (a blast's farther and hidden tiles follow over the
        // next frames, drawn as they were until then). Then stale tiles.
        let mut edited: Vec<(TileKey, bool, f64)> = Vec::with_capacity(self.urgent.len());
        for k in self.urgent.drain() {
            let tiles = &self.tiles;
            let distance = *self.urgent_distance.entry(k).or_insert_with(|| tiles.get(&k).map_or(f64::INFINITY, |t| tile_distance(&grid, k, t.ranges.extent(), &eye_at)));
            edited.push((k, !self.drawn_set.holds(k), distance));
        }
        edited.sort_unstable_by(|a, b| a.1.cmp(&b.1).then(a.2.total_cmp(&b.2)));
        let edited: Vec<(TileKey, bool)> = edited.into_iter().map(|(k, ..)| (k, true)).chain(self.stale.iter().map(|k| (*k, false))).collect();
        for (key, urgent) in edited {
            if full(&batch) || batch.chunk_jobs.len() >= CHUNK_BATCH || stats.out_of_slots {
                if urgent {
                    self.urgent.insert(key);
                    continue;
                }
                break;
            }
            self.stale.remove(&key);
            let Some(ranges) = self.tiles.get(&key).map(|t| t.ranges) else { continue };
            if !self.queue_tile(key, ranges, &eye_at, &mut batch, &mut readback, &mut evictions, &mut stats) {
                if urgent {
                    self.urgent.insert(key);
                } else {
                    self.stale.insert(key);
                }
            }
        }
        let urgent = &self.urgent;
        self.urgent_distance.retain(|k, _| urgent.contains(k));
        let requests = std::mem::take(&mut self.plan.requests);
        for (key, ranges) in &requests {
            if full(&batch) {
                break;
            }
            // Generated since that plan.
            if self.tiles.contains_key(key) {
                continue;
            }
            if !self.queue_tile(*key, *ranges, &eye_at, &mut batch, &mut readback, &mut evictions, &mut stats) {
                break;
            }
        }
        self.plan.requests = requests;
        let GenBatch { tile_jobs, chunk_jobs, refs, baked_refs, baked_cells, .. } = batch;

        // GPU: blocks of evicted chunks back to their free lists, then the
        // tiles' columns, the chunks' columns and trees.
        let refs_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk brush refs"),
            contents: bytemuck::cast_slice(if refs.is_empty() { &[0u32][..] } else { &refs }),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let evictions_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk evictions"),
            contents: bytemuck::cast_slice(if evictions.is_empty() { &[u32::MAX][..] } else { &evictions }),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let baked_refs_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk baked refs"),
            contents: bytemuck::cast_slice(if baked_refs.is_empty() { &[u32::MAX][..] } else { &baked_refs }),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let baked_cells_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk baked cells"),
            contents: bytemuck::cast_slice(if baked_cells.is_empty() { &[0u32][..] } else { &baked_cells }),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let group = self.gen_group(&refs_buffer, &evictions_buffer, &baked_refs_buffer, &baked_cells_buffer);
        if !tile_jobs.is_empty() {
            self.queue.write_buffer(&self.tile_jobs, 0, bytemuck::cast_slice(&tile_jobs));
            let empty: Vec<[i32; 4]> = vec![[i32::MAX, i32::MIN + 1, i32::MAX, i32::MIN + 1]; tile_jobs.len()];
            self.queue.write_buffer(&self.tile_ranges, 0, bytemuck::cast_slice(&empty));
        }
        if !chunk_jobs.is_empty() {
            self.queue.write_buffer(&self.chunk_jobs, 0, bytemuck::cast_slice(&chunk_jobs));
        }
        // A pass per kernel, each timed (`QUERIES`).
        let stamp = |begin: Option<u32>, end: u32| {
            self.queries.as_ref().zip(self.timing_now).map(|((set, _), ring)| {
                let ring = ring as u32;
                wgpu::ComputePassTimestampWrites { query_set: set, beginning_of_pass_write_index: begin.map(|b| QUERIES * ring + b), end_of_pass_write_index: Some(QUERIES * ring + end) }
            })
        };
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("chunk tile columns"), timestamp_writes: stamp(Some(0), 1) });
            pass.set_bind_group(0, &group, &[]);
            if !evictions.is_empty() {
                pass.set_pipeline(&self.evict_pipeline);
                pass.dispatch_workgroups((evictions.len() as u32).div_ceil(64), 1, 1);
            }
            if !tile_jobs.is_empty() {
                pass.set_pipeline(&self.tile_columns);
                pass.dispatch_workgroups(64, tile_jobs.len() as u32, 1);
            }
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("chunk columns"), timestamp_writes: stamp(None, 5) });
            pass.set_bind_group(0, &group, &[]);
            if !chunk_jobs.is_empty() {
                pass.set_pipeline(&self.chunk_columns);
                pass.dispatch_workgroups(64, chunk_jobs.len() as u32, 1);
            }
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("chunk build"), timestamp_writes: stamp(None, 6) });
            pass.set_bind_group(0, &group, &[]);
            if !chunk_jobs.is_empty() {
                pass.set_pipeline(&self.chunk_build);
                pass.dispatch_workgroups(chunk_jobs.len() as u32, 1, 1);
            }
        }
        // Results back: tiles' layers, then each chunk's job result.
        if !tile_jobs.is_empty() {
            let tiles = (tile_jobs.len() * 16) as u64;
            let chunks = (chunk_jobs.len() * 16) as u64;
            readback.buffer = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("chunk results"), size: tiles + chunks, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            encoder.copy_buffer_to_buffer(&self.tile_ranges, 0, &readback.buffer, 0, tiles);
            if chunks > 0 {
                encoder.copy_buffer_to_buffer(&self.job_results, 0, &readback.buffer, tiles, chunks);
            }
            self.pending = Some(readback);
        }
        // Slots evicted now are reused from the next frame.
        self.free_slots.extend(evictions);

        lap(&mut stats, 3);
        // 5. This frame's drawn chunks and the table entries that changed.
        let tiles = &self.tiles;
        let chunks_of = |key: TileKey| -> drawn::TileDraw {
            tiles.get(&key).map_or_else(
                || Vec::new().into(),
                |tile| drawn::TileDraw {
                    chunks: tile
                        .chunks
                        .iter()
                        .map(|c| {
                            let z = c.z.map_or((0.0, 64.0), |(a, b)| (f64::from(a), f64::from(b)));
                            drawn::DrawnChunk { key: chunk_key(key, c.cz), slot: c.slot, z }
                        })
                        .collect(),
                    gap: tile.layers.gap,
                },
            )
        };
        self.drawn_set.update(&added, &removed, &self.changed_tiles, frame, chunks_of);
        self.changed_tiles.clear();
        lap(&mut stats, 4);
        // One upload of the changed entries, scattered on the GPU.
        let changes = self.drawn_set.take_changes();
        if !changes.is_empty() {
            let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("chunk table changes"), contents: bytemuck::cast_slice(&changes), usage: wgpu::BufferUsages::STORAGE });
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("chunk table"), layout: &self.table_layout, entries: &[entry(0, &buffer), entry(1, &self.chunk_table)] });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("chunk table"), timestamp_writes: None });
            pass.set_pipeline(&self.table_pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((changes.len() as u32 / 3).div_ceil(64), 1, 1);
        }
        lap(&mut stats, 5);
        stats.drawn_tiles = self.drawn_set.tile_count();
        stats.fading_tiles = self.drawn_set.fading();
        self.last_generated = (stats.generated_tiles, stats.generated_chunks);
        // Settled: a plan that saw every event wants nothing more.
        self.last_requests = self.plan.requests.iter().filter(|(k, _)| !self.tiles.contains_key(k)).count() + usize::from(unseen || !self.events.is_empty() || stats.generated_tiles > 0);
        stats.drawn_chunks = self.drawn_set.list.len();
        stats.resident_chunks = self.tiles.values().map(|t| t.chunks.len()).sum();
        stats
    }

    /// Start mapping the results the last update recorded (call after its
    /// submit).
    /// Copy the last draw's hits under `picks`' requests (at most
    /// `MAX_PICKS` a frame) for readback; [`Self::submitted`] answers them.
    pub fn copy_picks(&mut self, encoder: &mut helio_core::CommandRecorder<'_>, picks: &crate::engine::SharedPicks) {
        let Some(targets) = &self.targets else { return };
        let Some(slot) = self.pick_slots.iter_mut().find(|s| matches!(s.state, ReachState::Free)) else { return };
        let requests: Vec<crate::engine::PickRequest> = {
            let Ok(mut shared) = picks.lock() else { return };
            let n = shared.requests.len().min(MAX_PICKS);
            shared.requests.drain(..n).collect()
        };
        if requests.is_empty() {
            return;
        }
        let size = targets.size;
        for (n, request) in requests.iter().enumerate() {
            let x = ((request.uv[0].clamp(0.0, 1.0) * size[0] as f32) as u32).min(size[0].max(1) - 1);
            let y = ((request.uv[1].clamp(0.0, 1.0) * size[1] as f32) as u32).min(size[1].max(1) - 1);
            for (texture, at) in [(&targets.visibility, 0), (&targets.distance, 16)] {
                encoder.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d { x, y, z: 0 }, aspect: wgpu::TextureAspect::All },
                    wgpu::TexelCopyBufferInfo { buffer: &slot.buffer, layout: wgpu::TexelCopyBufferLayout { offset: 256 * n as u64 + at, bytes_per_row: None, rows_per_image: None } },
                    wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                );
            }
        }
        slot.requests = requests.iter().map(|r| r.id).collect();
        slot.sink = Some(picks.clone());
        slot.state = ReachState::Copied;
    }

    /// The hit of a pixel's visibility (slot, cell) and distance.
    fn pick_hit(&self, [slot, cell]: [u32; 2], t: f32) -> Option<crate::engine::PickHit> {
        let key = *self.slot_keys.get(slot as usize)?;
        if key[0] < 0 || !(t > 0.0) {
            return None;
        }
        let local = [cell & 63, (cell >> 6) & 63, (cell >> 12) & 63].map(|c| c as i32);
        let axis = (cell >> 18) & 3;
        // Visibility's side: 1 when the ray stepped down the axis.
        let down = (cell >> 20) & 1 == 1;
        Some(crate::engine::PickHit {
            distance: f64::from(t),
            cell: crate::grid::Cell::new((key[0] & 0xff) as u8, key[1] * CHUNK + local[0], key[2] * CHUNK + local[1], key[3] * CHUNK + local[2]),
            level: (key[0] >> 8) as u32,
            entered: Some(2 * axis + u32::from(!down)),
        })
    }

    pub fn submitted(&mut self) {
        // Picks: answer those mapped, map those just copied.
        for n in 0..self.pick_slots.len() {
            match &self.pick_slots[n].state {
                ReachState::Copied => {
                    let ready = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                    let flag = ready.clone();
                    self.pick_slots[n].buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                        if r.is_ok() {
                            flag.store(true, std::sync::atomic::Ordering::Release);
                        }
                    });
                    self.pick_slots[n].state = ReachState::Mapping(ready);
                }
                ReachState::Mapping(ready) if ready.load(std::sync::atomic::Ordering::Acquire) => {
                    let results: Vec<crate::engine::PickResult> = {
                        let data = self.pick_slots[n].buffer.slice(..).get_mapped_range().unwrap();
                        let words: &[u32] = bytemuck::cast_slice(&data);
                        self.pick_slots[n].requests.iter().enumerate().map(|(i, &id)| crate::engine::PickResult { id, hit: self.pick_hit([words[i * 64], words[i * 64 + 1]], f32::from_bits(words[i * 64 + 4])) }).collect()
                    };
                    let slot = &mut self.pick_slots[n];
                    slot.buffer.unmap();
                    if let Some(Ok(mut shared)) = slot.sink.take().as_ref().map(|s| s.lock()) {
                        shared.results.extend(results);
                    }
                    slot.requests.clear();
                    slot.state = ReachState::Free;
                }
                _ => {}
            }
        }
        // Tiles rays reached under: map the copies just submitted, read
        // those mapped.
        for slot in &mut self.reach_slots {
            match &slot.state {
                ReachState::Copied => {
                    let ready = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                    let flag = ready.clone();
                    slot.buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                        if r.is_ok() {
                            flag.store(true, std::sync::atomic::Ordering::Release);
                        }
                    });
                    slot.state = ReachState::Mapping(ready);
                }
                ReachState::Mapping(ready) if ready.load(std::sync::atomic::Ordering::Acquire) => {
                    {
                        let data = slot.buffer.slice(..).get_mapped_range().unwrap();
                        let words: &[[i32; 4]] = bytemuck::cast_slice(&data);
                        let count = (words[0][0] as u32 as usize).min(words.len() - 1);
                        for k in &words[1..=count] {
                            self.reached.push(TileKey { face: (k[0] & 0xff) as u8, level: (k[0] >> 8) as u32, cx: k[1], cy: k[2] });
                        }
                    }
                    slot.buffer.unmap();
                    slot.state = ReachState::Free;
                }
                _ => {}
            }
        }
        // Timestamps: map those just submitted, read those mapped.
        let period = f64::from(self.queue.get_timestamp_period()) / 1.0e6;
        for slot in &mut self.timing_slots {
            if let Some(ready) = &slot.ready {
                if !ready.load(std::sync::atomic::Ordering::Acquire) {
                    continue;
                }
                let (frame, tiles, chunks) = slot.frame.take().unwrap();
                {
                    let data = slot.buffer.slice(..).get_mapped_range().unwrap();
                    let t: &[u64] = bytemuck::cast_slice(&data);
                    let ms = |a: u64, b: u64| b.saturating_sub(a) as f64 * period;
                    let stages = [ms(t[0], t[1]), ms(t[1], t[5]), ms(t[5], t[6])];
                    let timing = FrameTimings { frame, draw: ms(t[2], t[4]), seed: ms(t[2], t[3]), shadow: ms(t[4], t[7]), generate: ms(t[0], t[6]), tiles, stages };
                    if tiles >= 8 {
                        let now = GenCost { tile: stages[0] / tiles as f64, chunk: (stages[1] + stages[2]) / chunks.max(1) as f64, chunks_per_tile: chunks as f64 / tiles as f64 };
                        let mix = |old: f64, new: f64| old * 0.9 + new * 0.1;
                        let c = self.cost;
                        self.cost = GenCost { tile: mix(c.tile, now.tile), chunk: mix(c.chunk, now.chunk), chunks_per_tile: mix(c.chunks_per_tile, now.chunks_per_tile) };
                    }
                    self.timings.push(timing);
                }
                slot.buffer.unmap();
                slot.ready = None;
            } else if slot.frame.is_some() {
                let ready = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                let flag = ready.clone();
                slot.buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                    if r.is_ok() {
                        flag.store(true, std::sync::atomic::Ordering::Release);
                    }
                });
                slot.ready = Some(ready);
            }
        }
        let _ = self.device.poll(wgpu::PollType::Poll);
        if let Some(r) = self.pending.take() {
            let ready = r.ready.clone();
            r.buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    ready.store(true, std::sync::atomic::Ordering::Release);
                }
            });
            self.readbacks.push_back(r);
        }
    }


    /// Tiles whose whole volume is generated because rays reached under
    /// their surface band.
    pub fn deep_tiles(&self) -> usize {
        self.deep.len()
    }

    /// Whether nothing waits to be generated or read back, and no switch
    /// between levels is fading (fades move with drawn frames).
    pub fn settled(&self) -> bool {
        self.pending.is_none() && self.readbacks.is_empty() && self.last_requests == 0 && self.urgent.is_empty() && self.stale.is_empty() && self.drawn_set.fading() == 0
    }

    /// Tile, chunk layer and occupied layers of the drawn chunk in `slot`.
    pub fn chunk_at(&self, slot: u32) -> Option<(TileKey, i32, Option<(i32, i32)>)> {
        self.tiles.iter().find_map(|(k, t)| t.chunks.iter().find(|c| c.slot == slot).map(|c| (*k, c.cz, c.z)))
    }

    /// Chunks over base column (`face`, `i`, `j`): tile, chunk layer,
    /// occupied layers, drawn.
    pub fn chunks_over(&self, face: u8, i: i32, j: i32) -> Vec<(TileKey, i32, Option<(i32, i32)>, bool)> {
        let mut out: Vec<_> = self
            .tiles
            .iter()
            .filter(|(k, _)| k.face == face && (i >> k.level).div_euclid(CHUNK) == k.cx && (j >> k.level).div_euclid(CHUNK) == k.cy)
            .flat_map(|(k, t)| t.chunks.iter().map(|c| (*k, c.cz, c.z, self.drawn_set.contains(c.slot))))
            .collect();
        out.sort_by_key(|c| (c.0.level, c.1));
        out
    }

    /// Chunks drawn by the last [`Self::update`].
    pub fn chunk_count(&self) -> usize {
        self.drawn_set.list.len()
    }

    /// Bytes of the resident trees: leaves with their two material bytes, and
    /// per resident chunk its mid nodes (masks, first leaf, material) and
    /// root (reads the leaf allocator back).
    pub fn resident_bytes(&self) -> u64 {
        let alloc: Vec<[i32; 2]> = self.read(&self.leaf_alloc, 1);
        let chunks: usize = self.tiles.values().map(|t| t.chunks.len()).sum();
        alloc[0][0].max(0) as u64 * 10 + chunks as u64 * (64 * 24 + 48)
    }

    /// Every pixel's hit in the last [`Self::draw`], row by row (waits for
    /// the GPU: tests and diagnostics).
    pub fn read_hits(&self) -> Vec<Option<crate::engine::PickHit>> {
        let Some(targets) = &self.targets else { return Vec::new() };
        let [w, h] = targets.size;
        let rows = |bytes: u32| (w * bytes).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let read = |texture: &wgpu::Texture, bytes: u32| -> Vec<u8> {
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: u64::from(rows(bytes) * h), usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let mut encoder = self.device.create_command_encoder(&Default::default());
            encoder.copy_texture_to_buffer(
                texture.as_image_copy(),
                wgpu::TexelCopyBufferInfo { buffer: &staging, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(rows(bytes)), rows_per_image: None } },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            self.queue.submit([encoder.finish()]);
            staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let out = staging.slice(..).get_mapped_range().unwrap().to_vec();
            out
        };
        let (visibility, distance) = (read(&targets.visibility, 8), read(&targets.distance, 4));
        let word = |data: &[u8], at: usize| u32::from_le_bytes(data[at..at + 4].try_into().unwrap());
        (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .map(|(x, y)| {
                let v = (y * rows(8) + x * 8) as usize;
                self.pick_hit([word(&visibility, v), word(&visibility, v + 4)], f32::from_bits(word(&distance, (y * rows(4) + x * 4) as usize)))
            })
            .collect()
    }

    /// The targets of the last [`Self::draw`].
    pub fn targets(&self) -> Option<&Targets> {
        self.targets.as_ref()
    }

    /// Draw the chunks seen from `eye` (camera-relative `view_proj`) at
    /// `size` pixels into [`Self::targets`], timing it as draw number
    /// `frame` ([`Self::draw_ms`]).
    #[allow(clippy::too_many_arguments)]
    pub fn draw(&mut self, encoder: &mut helio_core::CommandRecorder<'_>, frame: u64, eye: DVec3, view_proj: Mat4, prev_view_proj: Mat4, near: f32, sun: Vec3, size: [u32; 2]) {
        if self.targets.as_ref().is_none_or(|t| t.size != size) {
            self.targets = Some(Targets::new(&self.device, size));
        }
        if self.timing_now.is_none() {
            self.claim_timing(frame);
        }
        // The drawn chunks' slots, when they changed; the GPU culls them
        // (`cull_bin`) into the seed's draw list.
        let drawn = self.drawn_set.list.len() as u32;
        if self.slots_version != Some(self.drawn_set.version()) {
            let slots: Vec<u32> = self.drawn_set.list.iter().map(|c| c.slot).collect();
            if !slots.is_empty() {
                self.queue.write_buffer(&self.drawn_slots, 0, bytemuck::cast_slice(&slots));
            }
            self.slots_version = Some(self.drawn_set.version());
        }
        self.queue.write_buffer(&self.cull_state, 0, &[0u8; CULL_BYTES as usize]);
        let (eye_hi, eye_lo) = split(eye);
        let params = DrawParams {
            view_proj: view_proj.to_cols_array_2d(),
            inverse_view_proj: view_proj.inverse().to_cols_array_2d(),
            prev_view_proj: prev_view_proj.to_cols_array_2d(),
            eye_hi,
            eye_lo,
            sun: sun.normalize().extend(near).to_array(),
            debug: {
                let pixel: Vec<u32> = std::env::var("HELIO_VOXEL_CHUNKS_DEBUG_PIXEL").ok().map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
                [
                    if std::env::var_os("HELIO_VOXEL_CHUNKS_HEAT").is_some() { 2 } else { u32::from(std::env::var_os("HELIO_VOXEL_CHUNKS_LEVELS").is_some()) },
                    u32::from(std::env::var_os("HELIO_VOXEL_CHUNKS_COUNT").is_some()),
                    pixel.first().copied().unwrap_or(u32::MAX),
                    pixel.get(1).copied().unwrap_or(u32::MAX),
                ]
            },
            table: [self.drawn_set.mask(), self.face_cells, drawn, frame as u32],
            eye_domain: {
                let d = (eye / crate::grid::DOMAIN_UNIT).floor();
                [d.x as i32, d.y as i32, d.z as i32, 0]
            },
            eye_domain_frac: {
                let d = eye / crate::grid::DOMAIN_UNIT;
                (d - d.floor()).as_vec3().extend(0.0).to_array()
            },
            up: if self.grid.is_plane() { [0.0, 1.0, 0.0, 0.0] } else { eye.normalize_or_zero().as_vec3().extend(eye.length() as f32).to_array() },
            ceiling: {
                let c = self.grid.radius() + self.ceiling;
                [if self.grid.is_plane() { (eye.y - self.ceiling) as f32 } else { (eye.length_squared() - c * c) as f32 }, 0.0, 0.0, 0.0]
            },
            planet: [
                self.grid.radius() as f32,
                self.grid.voxel_size() as f32,
                self.exact_level.min(99) as f32,
                (eye.length_squared() - self.grid.radius() * self.grid.radius()) as f32,
            ],
        };
        self.queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        self.queue.write_buffer(&self.reach, 0, &[0u8; 16]);
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chunk draw"),
            layout: &self.draw_layout,
            entries: &[entry(0, &self.params), entry(1, &self.statics), entry(2, &self.chunk_meta), entry(3, &self.mids), entry(4, &self.mid_leaves), entry(5, &self.leaves), entry(6, &self.leaf_materials), entry(7, &self.materials), entry(8, &self.mid_materials), entry(9, &self.counters), entry(10, &self.chunk_table), entry(11, &self.chunk_keys), entry(12, &self.draw_list), entry(13, &self.reach)],
        });
        self.last_group = Some(group.clone());
        let targets = self.targets.as_ref().unwrap();
        let timing = self.queries.as_ref().zip(self.timing_now.filter(|&r| r == (frame % TIMING_RING) as usize)).map(|((set, resolve), ring)| (set, resolve, ring as u32));
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
            entries: &[wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&exits) }, entry(22, &self.cull_state)],
        });
        {
            // 0. The draw list: the drawn chunks in view, binned, ordered.
            let frames = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("chunk cull frames"),
                layout: &self.cull_layout[0],
                entries: &[entry(0, &self.params), entry(1, &self.statics), entry(2, &self.chunk_meta)],
            });
            let lists = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("chunk cull"),
                layout: &self.cull_layout[1],
                entries: &[entry(17, &self.drawn_slots), entry(18, &self.cull_state), entry(19, &self.cull_bins), entry(20, &self.draw_list), entry(21, &self.seed_args)],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("chunk cull"), timestamp_writes: None });
            pass.set_bind_group(0, &frames, &[]);
            pass.set_bind_group(1, &lists, &[]);
            pass.set_pipeline(&self.cull_bin);
            pass.dispatch_workgroups(drawn.div_ceil(64), 1, 1);
            pass.set_pipeline(&self.cull_offsets);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&self.cull_scatter);
            pass.dispatch_workgroups(drawn.div_ceil(64), 1, 1);
        }
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
                    timestamp_writes: timing.filter(|_| round == 0).map(|(set, _, ring)| wgpu::RenderPassTimestampWrites { query_set: set, beginning_of_pass_write_index: Some(QUERIES * ring + 2), end_of_pass_write_index: Some(QUERIES * ring + 3) }),
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_bind_group(0, &group, &[]);
                pass.set_bind_group(1, &seed_group, &[]);
                // Boxes holding the eye only in the first round.
                if round == 0 {
                    pass.set_pipeline(&self.seed_inside);
                    pass.draw_indirect(&self.seed_args, 0);
                }
                pass.set_pipeline(&self.seed_outside);
                pass.draw_indirect(&self.seed_args, 16);
            }
            // 2. Trace each pixel's ray through its seed chunk.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("chunk trace"),
                timestamp_writes: timing.filter(|_| round + 1 == rounds).map(|(set, _, ring)| wgpu::ComputePassTimestampWrites { query_set: set, beginning_of_pass_write_index: None, end_of_pass_write_index: Some(QUERIES * ring + 4) }),
            });
            pass.set_pipeline(&self.trace);
            pass.set_bind_group(0, &group, &[]);
            pass.set_bind_group(1, &trace_group, &[]);
            pass.dispatch_workgroups(size[0].div_ceil(8), size[1].div_ceil(8), 1);
        }
        {
            // 4. Sun shadows: samples, then every hit.
            let (distance, visibility, samples, colour, normal, sky) = (view(&targets.distance), view(&targets.visibility), view(&targets.sun_samples), view(&targets.colour), view(&targets.normal), view(&targets.sky_samples));
            let sample_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("chunk sun samples"),
                layout: &self.shadow_sample_layout,
                entries: &[view_entry(4, &distance), view_entry(7, &visibility), view_entry(9, &samples)],
            });
            let shadow_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("chunk shadow"),
                layout: &self.shadow_layout,
                entries: &[view_entry(4, &distance), view_entry(7, &visibility), view_entry(2, &colour), view_entry(8, &targets.sun_view), view_entry(10, &samples), entry(11, &targets.fallbacks), entry(12, &targets.fallback_groups), view_entry(13, &normal), view_entry(16, &sky)],
            });
            let sky_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("chunk sky samples"),
                layout: &self.sky_sample_layout,
                entries: &[view_entry(4, &distance), view_entry(7, &visibility), view_entry(15, &sky)],
            });
            let fallback_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("chunk shadow fallbacks"),
                layout: &self.shadow_fallback_layout,
                entries: &[view_entry(4, &distance), view_entry(7, &visibility), view_entry(2, &colour), view_entry(8, &targets.sun_view), entry(11, &targets.fallbacks), view_entry(13, &normal)],
            });
            // No fallbacks yet.
            self.queue.write_buffer(&targets.fallbacks, 0, bytemuck::cast_slice(&[0u32]));
            self.queue.write_buffer(&targets.fallback_groups, 0, bytemuck::cast_slice(&[0u32, 1, 1]));
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("chunk sun samples"), timestamp_writes: None });
                pass.set_bind_group(0, &group, &[]);
                pass.set_bind_group(1, &sample_group, &[]);
                for pipeline in [&self.shadow_sample, &self.shadow_sample_far] {
                    pass.set_pipeline(pipeline);
                    pass.dispatch_workgroups(size[0].div_ceil(2).div_ceil(8), size[1].div_ceil(2).div_ceil(8), 1);
                }
                pass.set_bind_group(1, &sky_group, &[]);
                for pipeline in [&self.sky_sample, &self.sky_sample_far] {
                    pass.set_pipeline(pipeline);
                    pass.dispatch_workgroups(size[0].div_ceil(4).div_ceil(8), size[1].div_ceil(4).div_ceil(8), 1);
                }
            }
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("chunk shadow"),
                timestamp_writes: timing.map(|(set, _, ring)| wgpu::ComputePassTimestampWrites { query_set: set, beginning_of_pass_write_index: None, end_of_pass_write_index: Some(QUERIES * ring + 7) }),
            });
            pass.set_pipeline(&self.shadow);
            pass.set_bind_group(0, &group, &[]);
            pass.set_bind_group(1, &shadow_group, &[]);
            pass.dispatch_workgroups(size[0].div_ceil(8), size[1].div_ceil(8), 1);
            pass.set_pipeline(&self.shadow_fallback);
            pass.set_bind_group(1, &fallback_group, &[]);
            pass.dispatch_workgroups_indirect(&targets.fallback_groups, 0);
        }
        {
            // 3. The hits' depth.
            let resolve_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("chunk depth resolve"),
                layout: &self.resolve_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&view(&targets.distance)) },
                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&view(&targets.visibility)) },
                    wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::TextureView(&view(&targets.normal)) },
                ],
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
        // The tiles reached under, back without waiting (a frame's copy
        // when a slot is free).
        if let Some(slot) = self.reach_slots.iter_mut().find(|s| matches!(s.state, ReachState::Free)) {
            encoder.copy_buffer_to_buffer(&self.reach, 0, &slot.buffer, 0, REACH_BYTES);
            slot.state = ReachState::Copied;
        }
        if let Some((set, resolve, ring)) = timing {
            // Resolve offsets are 256-byte aligned.
            let at = u64::from(ring) * u64::from(wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT);
            encoder.resolve_query_set(set, QUERIES * ring..QUERIES * ring + QUERIES, resolve, at);
            let slot = &mut self.timing_slots[ring as usize];
            encoder.copy_buffer_to_buffer(resolve, at, &slot.buffer, 0, u64::from(QUERIES) * 8);
            slot.frame = Some((frame, self.last_generated.0, self.last_generated.1));
            self.timing_now = None;
        }
    }

    /// The traces' counts since the last call (with
    /// `HELIO_VOXEL_CHUNKS_COUNT`).
    pub fn take_counts(&self) -> TraceCounts {
        let words: Vec<[u32; 4]> = self.read(&self.counters, 3);
        self.queue.write_buffer(&self.counters, 0, &[0u8; 1024]);
        TraceCounts { seeded: words[0][0], steps: words[0][1], hits: words[0][2], boxes: words[0][3], missed: words[2][0], missed_boxes: words[2][1], lost: words[2][2] }
    }

    /// The boxes the debug pixel's ray crossed (counting builds): the frame
    /// chunk's slot, the box's low corner in its cells, whether it is that
    /// chunk, its level; and its key and layer.
    pub fn pixel_trace(&self) -> Vec<(u32, [i32; 3], bool, u32, Option<(TileKey, i32)>)> {
        let words: Vec<[u32; 4]> = self.read(&self.counters, 64);
        let flat: Vec<u32> = words.iter().flatten().copied().collect();
        let n = flat[4] as usize;
        (0..n.min(48))
            .map(|c| {
                let w = &flat[16 + c * 5..21 + c * 5];
                let slot = w[0];
                let owner = self.tiles.iter().find_map(|(k, t)| t.chunks.iter().find(|ch| ch.slot == slot).map(|ch| (*k, ch.cz)));
                (slot, [w[1] as i32, w[2] as i32, w[3] as i32], w[4] & 1 != 0, w[4] >> 1, owner)
            })
            .collect()
    }

    /// Boxes drawn by the last [`Self::draw`]: from outside, from inside
    /// (reads the cull back: diagnostics).
    pub fn drawn(&self) -> (u32, u32) {
        let counts: Vec<u32> = self.read(&self.cull_state, 2);
        (counts[1] - counts[0], counts[0])
    }

    /// This frame's timestamp slot, when it is free again (its last
    /// frame's times were read).
    fn claim_timing(&mut self, frame: u64) {
        let ring = (frame % TIMING_RING) as usize;
        self.timing_now = self.timing_slots.get(ring).filter(|t| t.frame.is_none() && t.ready.is_none()).map(|_| ring);
    }

    /// Shade with `appearance` (the terrain's materials and detail).
    pub fn set_appearance(&mut self, appearance: &crate::engine::TerrainAppearance) {
        let (materials, detail) = crate::engine::material_table(appearance);
        self.queue.write_buffer(&self.materials, 0, bytemuck::bytes_of(&ChunkMaterials { materials, detail }));
    }

    /// The last draw's sun visibility (Helio's `directional_visibility`).
    pub fn sun_view(&self) -> Option<&wgpu::TextureView> {
        self.targets.as_ref().map(|t| &t.sun_view)
    }

    /// The last draw's hits into Helio's gbuffer (`engine::GBUFFER_FORMATS`)
    /// and depth (Depth32Float, tested against what is there).
    pub fn publish(&self, encoder: &mut helio_core::CommandRecorder<'_>, gbuffer: [&wgpu::TextureView; 8], depth: &wgpu::TextureView) {
        let Some(targets) = &self.targets else { return };
        let Some(group) = &self.last_group else { return };
        let resolve_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chunk gbuffer"),
            layout: &self.resolve_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&view(&targets.distance)) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&view(&targets.visibility)) },
                wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::TextureView(&view(&targets.normal)) },
            ],
        });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("chunk gbuffer"),
            color_attachments: &gbuffer.map(loaded),
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.publish);
        pass.set_bind_group(0, group, &[]);
        pass.set_bind_group(1, &resolve_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Frames whose GPU times came back since the last call.
    pub fn take_timings(&mut self) -> Vec<FrameTimings> {
        std::mem::take(&mut self.timings)
    }
}
