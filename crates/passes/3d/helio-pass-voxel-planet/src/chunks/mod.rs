//! Chunk trees in an octree (`docs/chunk-trees.md`, `docs/chunk-octree.md`):
//! generation of chunk trees from the terrain contract, residency for a
//! moving eye and primary visibility.
//!
//! A node is a chunk: 64^3 cells of one level, stored as a 64-tree; a
//! footprint is the 64x64 columns of one level under a stack of nodes.
//! [`ChunkWorld::update`] asks the [`residency::Planner`] which nodes to
//! draw and generate, and generates and evicts them on the GPU within a
//! budget; [`ChunkWorld::draw`] rasterizes the chunks' boxes to seed each
//! pixel with the nearest box its ray enters, then traces each pixel's ray
//! through the cut in a compute pass (shaders/chunk_draw.wgsl), and reports
//! what rays met so detail goes where it is seen.

pub mod classify;
mod drawn;
pub mod pass;
pub mod residency;

pub use classify::{Footprint, Kind, NodeKey};
use classify::{brush_meets, EditView, Ranges};
use crate::edit_store::{Brick, ColumnKey, EditStore};
use crate::edits::FaceBrush;
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
pub(crate) const TILE_BATCH: usize = 512;
/// Chunks generated per dispatch (column masks are scratch).
const CHUNK_BATCH: usize = 2048;
/// Chunk slots by default (`HELIO_VOXEL_CHUNKS_SLOTS`), within what the
/// device can bind.
pub const DEFAULT_CHUNKS: usize = 131072;
/// Chunk jobs under edits a batch (`cell_materials` slots: `CELL_WORDS`
/// words a column, 272 KiB each).
const DETAIL_JOBS: usize = 128;
/// Levels whose chunks under edits get per-voxel materials: cells of
/// 0.1-0.2 m at 0.1 m voxels, near the eye. Coarser chunks' leaves keep
/// their pairs (an edit's top materials vote in them): a dug wall's detail
/// pages at every level filled a game's pages after some 80 digs 8 m across.
const DETAIL_LEVELS: u32 = 2;
/// Kept chunks (`pack_main`, `restore_main` of chunk_generate.wgsl): words
/// a frame's packs may take, and packs a frame at most (more evicted
/// chunks are not kept: they generate again).
const PACK_WORDS: u64 = 1 << 19;
const MAX_PACKS: usize = 64;
/// Bytes a kept chunk counts besides its pack (its entry).
const KEPT_OVERHEAD: usize = 64;
/// Kept chunks' bytes (`KeptChunks::to_bytes`): magic, and the pack format
/// (bump with `pack_main`'s layout).
const KEPT_MAGIC: [u8; 4] = *b"HVPK";
const KEPT_FORMAT: u16 = 1;

/// A uniform brick's table entry flag (`UNIFORM_BRICK` of
/// chunk_generate.wgsl): its cells' one edit in the low 16 bits.
const UNIFORM_BRICK: u32 = 0x4000_0000;
/// Words a column of a `cell_materials` slot (`CELL_WORDS` of
/// chunk_generate.wgsl): 16 of cell bytes, the column's natural top.
const CELL_WORDS: u64 = 17;
/// Detail pages by default (`HELIO_VOXEL_DETAIL_PAGES`): 4 KiB each, one
/// for each leaf page holding a leaf under edits whose cells its pair would
/// draw wrong.
pub const DEFAULT_DETAIL_PAGES: usize = 16384;
/// Words a detail page (`DETAIL_WORDS` of chunk_generate.wgsl).
const DETAIL_WORDS: u64 = 1024;

/// `Cull` of chunk_draw.wgsl: two counts, then 65 bins' counts and cursors.
const CULL_BYTES: u64 = 4 * (2 + 2 * 65);

/// Metres over the highest possible solid cell where rays may end: far
/// above the f32 error of the ceiling test.
const CEILING_MARGIN: f64 = 16.0;
/// Leaves a chunk slot holds on average (8 bytes each), in pages of 64
/// (`chunk_generate.wgsl`), up to `LEAF_PAGES` a chunk.
const LEAVES_PER_SLOT: u64 = 384;
const LEAF_PAGE: u64 = 64;
const LEAF_PAGES: u64 = 64;
/// A brush buffer's content when there are none.
const NO_BRUSH: [FaceBrush; 1] = [FaceBrush { flags: 0, radius_half: 0, k_lo: 0, k_hi: 0, center: [0; 4], ball: [0; 4] }];

/// Chunk table entries a slot: the drawn nodes (Solid and Buried ones
/// without slots) and their ancestors.
const TABLE_PER_SLOT: usize = 8;
/// Draws whose timings can be pending at once.
pub const TIMING_RING: u64 = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TileJob {
    face: u32,
    level: u32,
    cx: i32,
    cy: i32,
    /// Its slot in the footprint pool.
    slot: u32,
    _pad: [u32; 3],
}

/// Footprints the pool holds by default: their columns, halos and material
/// slopes (75 KB each), within what the device binds.
const FOOTPRINT_POOL: usize = 2048;
/// Material slope words a footprint (`SLOPE_STRIDE` of chunk_generate.wgsl:
/// a level-3 footprint's 34x34 level-4 blocks).
const SLOPE_STRIDE: u64 = 1156;
/// Level-4 tops a footprint's slopes read, at most (`SLOPE_LATTICE`: 38x38).
const SLOPE_LATTICE: u64 = 1444;

/// What a [`ChunkWorld`] holds, each within what the device binds: chunk
/// slots (a chunk's 64-tree and leaves, about 7 KB each), detail pages
/// (4 KiB each), footprint pool slots (75 KB each, at least two batches')
/// cached baked bricks (1 KiB each, before the least recently used are
/// reused) and edited chunks kept on the CPU (bytes, least recently used
/// dropped). The default is a game's: 131072 chunks, a world and its device
/// about 2.9 GB of device memory (`HELIO_VOXEL_CHUNKS_SLOTS`,
/// `HELIO_VOXEL_DETAIL_PAGES`).
#[derive(Clone, Copy, Debug)]
pub struct Capacity {
    pub chunks: usize,
    pub detail_pages: usize,
    pub footprints: usize,
    /// Baked bricks kept on the GPU (1 KiB each; `BrickCache`).
    pub bricks: usize,
    /// Bytes of edited chunks kept on the CPU across eviction (`Kept`; 0:
    /// none kept, every reload generates).
    pub kept_bytes: usize,
}

impl Default for Capacity {
    fn default() -> Self {
        let var = |name: &str, default: usize| std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default);
        Self { chunks: var("HELIO_VOXEL_CHUNKS_SLOTS", DEFAULT_CHUNKS), detail_pages: var("HELIO_VOXEL_DETAIL_PAGES", DEFAULT_DETAIL_PAGES), footprints: FOOTPRINT_POOL, bricks: 65536, kept_bytes: 256 << 20 }
    }
}

/// Footprints whose columns (and lean halos) stay on the GPU across frames,
/// so a footprint's columns are computed once while it stays in use,
/// whichever of its nodes are generated when (they were 60-75 % of a
/// chunk's generation). They depend only on the terrain, never on edits.
/// Least recently used slots are reused (a clock), never one this frame's
/// batch uses.
struct FootprintPool {
    slots: FxHashMap<Footprint, u32>,
    keys: Vec<Option<Footprint>>,
    /// The frame each slot was last used in, and whether it was used since
    /// the clock's hand last passed.
    used: Vec<u64>,
    referenced: Vec<bool>,
    hand: usize,
}

impl FootprintPool {
    fn new(size: usize) -> Self {
        Self { slots: FxHashMap::default(), keys: vec![None; size], used: vec![u64::MAX; size], referenced: vec![false; size], hand: 0 }
    }

    /// `fp`'s slot at `frame`, and whether its columns are there already;
    /// none when every slot serves this frame.
    fn take(&mut self, fp: Footprint, frame: u64) -> Option<(u32, bool)> {
        if let Some(&slot) = self.slots.get(&fp) {
            self.used[slot as usize] = frame;
            self.referenced[slot as usize] = true;
            return Some((slot, true));
        }
        let n = self.keys.len();
        for _ in 0..2 * n {
            let h = self.hand;
            self.hand = (self.hand + 1) % n;
            if self.used[h] == frame {
                continue;
            }
            if std::mem::take(&mut self.referenced[h]) {
                continue;
            }
            if let Some(old) = self.keys[h].take() {
                self.slots.remove(&old);
            }
            self.keys[h] = Some(fp);
            self.slots.insert(fp, h as u32);
            self.used[h] = frame;
            self.referenced[h] = true;
            return Some((h as u32, false));
        }
        None
    }
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
    /// Its `cell_materials` slot when edits are over it (its leaves get
    /// per-voxel materials where their pairs would draw cells wrong), or
    /// NONE.
    detail: u32,
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
    /// x: the renderer's debug mode in the low byte (1 UV grid, 32 trace
    /// cost), the plain colour flag (bit 8), then environment-only voxel
    /// diagnostics: primary steps' heat (0x200, `HELIO_VOXEL_CHUNKS_HEAT`),
    /// colour by level (0x400, `HELIO_VOXEL_CHUNKS_LEVELS`; 0x800: the
    /// left half).
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
    /// x: the octree's coarsest level; y: nodes fading.
    cut: [u32; 4],
}

/// `Materials` of chunk_draw.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ChunkMaterials {
    materials: [crate::engine::MaterialGpu; crate::engine::MATERIALS],
    detail: [f32; 4],
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
fn chunk_frame(grid: &Grid, tile: Footprint, cz: i32, z: f64) -> Frame {
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
fn reach_past(grid: &Grid, tile: Footprint, cz: i32, frame: &Frame, z: (f32, f32)) -> f32 {
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
fn chunk_static(grid: &Grid, tile: Footprint, cz: i32, slot: u32, z: (f32, f32), about: f32, exact: u32) -> ChunkStatic {
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
    fn offset(&self, tile: Footprint, range: (i32, i32)) -> Option<DVec3> {
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
    fn toward(&self, grid: &Grid, tile: Footprint, range: (i32, i32)) -> DVec3 {
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
fn tile_distance(grid: &Grid, tile: Footprint, range: (i32, i32), eye: &Eye) -> f64 {
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
    /// Footprints whose columns the batch computes, then those it reads from
    /// the pool; a chunk job's `tile` is an index in the first, or with
    /// `CACHED` in the second (resolved at upload).
    computed: Vec<TileJob>,
    cached: Vec<TileJob>,
    chunk_jobs: Vec<ChunkJob>,
    refs: Vec<u32>,
    /// Baked brick tables (512 a job: `BrickCache` slots).
    baked_refs: Vec<u32>,
    /// `cell_materials` slots taken.
    details: u32,
    /// Restore jobs the batch carries too (their results follow its jobs').
    restored: usize,
}

/// `ChunkJob::tile` of a footprint the batch reads from the pool.
const CACHED: u32 = 1 << 31;

impl GenBatch {
    fn footprints(&self) -> usize {
        self.computed.len() + self.cached.len()
    }

    /// Whether `store` holds bricks under chunk `cz` of `tile`.
    fn bricks_under(store: &EditStore, tile: Footprint, cz: i32) -> bool {
        (0..64).any(|n| {
            let key = ColumnKey { face: tile.face, level: tile.level as u8, ci: tile.cx * 8 + n % 8, cj: tile.cy * 8 + n / 8 };
            store.column(&key).is_some_and(|column| column.iter().any(|(bk, _)| (0..8).contains(&(bk - cz * 8))))
        })
    }

    /// The table of `store`'s bricks under chunk `cz` of `tile` (their
    /// `cache` slots for `frame`), or NONE.
    fn baked_table(&mut self, cache: &mut BrickCache, frame: u64, store: &EditStore, tile: Footprint, cz: i32) -> u32 {
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
                    table[((by * 8 + bx) * 8 + bz) as usize] = cache.entry(brick, frame);
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

/// Baked bricks on the GPU across batches (`baked_cells`), 256 words a
/// slot: a brick is uploaded once while it stays in use, so regenerating
/// chunks under heavily edited ground costs their tables, not their bricks
/// again (a seal's regenerated chunks sent 5.5 MB of bricks a frame). Keyed
/// by the brick's address; the cache holds the brick, so the address names
/// it while cached. A brick of one edit (a large dig's inside) is its table
/// entry (`UNIFORM_BRICK`), no slot. Slots not used this frame are reused
/// least recently used first once `limit` are taken (the buffer grows to
/// them, and past them when one frame wants more).
struct BrickCache {
    entries: FxHashMap<usize, (u32, std::sync::Arc<Brick>, u64)>,
    free: Vec<u32>,
    /// Slots handed out; the buffer holds `buffer_slots`.
    slots: u32,
    limit: u32,
    buffer: wgpu::Buffer,
    buffer_slots: u32,
    /// Bricks given slots since the last upload.
    pending: Vec<(u32, std::sync::Arc<Brick>)>,
}

/// Bytes a cached brick takes.
const BRICK_BYTES: u64 = 1024;

impl BrickCache {
    fn new(device: &wgpu::Device, limit: u32) -> Self {
        let buffer_slots = 4096.min(limit.max(1));
        Self { entries: FxHashMap::default(), free: Vec::new(), slots: 0, limit, buffer: Self::buffer(device, buffer_slots), buffer_slots, pending: Vec::new() }
    }

    fn buffer(device: &wgpu::Device, slots: u32) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor { label: Some("chunk baked bricks"), size: u64::from(slots) * BRICK_BYTES, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false })
    }

    /// `brick`'s table entry for `frame`: its slot (uploaded at the next
    /// [`Self::upload`] when new), or its uniform edit.
    fn entry(&mut self, brick: &std::sync::Arc<Brick>, frame: u64) -> u32 {
        let key = std::sync::Arc::as_ptr(brick) as usize;
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.2 = frame;
            return entry.0;
        }
        let entry = if brick.cells.iter().all(|c| *c == brick.cells[0]) {
            UNIFORM_BRICK | u32::from(brick.cells[0].0)
        } else {
            let slot = self.free.pop().or_else(|| (self.slots >= self.limit).then(|| self.evict(frame)).flatten()).unwrap_or_else(|| {
                self.slots += 1;
                self.slots - 1
            });
            self.pending.push((slot, brick.clone()));
            slot
        };
        self.entries.insert(key, (entry, brick.clone(), frame));
        entry
    }

    /// The least recently used quarter of the entries not used this frame
    /// dropped; one of the slots they free.
    fn evict(&mut self, frame: u64) -> Option<u32> {
        let mut old: Vec<(u64, usize)> = self.entries.iter().filter(|(_, e)| e.2 < frame).map(|(k, e)| (e.2, *k)).collect();
        if old.is_empty() {
            return None;
        }
        old.sort_unstable();
        for &(_, key) in &old[..(old.len() / 4).max(1)] {
            if let Some((entry, ..)) = self.entries.remove(&key) {
                if entry & UNIFORM_BRICK == 0 {
                    self.free.push(entry);
                }
            }
        }
        self.free.pop()
    }

    /// The new bricks written, after growing the buffer to every slot
    /// handed out (its old contents copied first, then the new bricks
    /// copied over them in the same order).
    fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut helio_core::CommandRecorder<'_>) {
        let words = |brick: &Brick| brick.cells.chunks(2).map(|p| u32::from(p[0].0) | (u32::from(p[1].0) << 16)).collect::<Vec<u32>>();
        if self.slots <= self.buffer_slots {
            // Runs of consecutive slots in one write each.
            self.pending.sort_unstable_by_key(|(slot, _)| *slot);
            let mut at = 0;
            while at < self.pending.len() {
                let first = self.pending[at].0;
                let run = self.pending[at..].iter().enumerate().take_while(|(n, (slot, _))| *slot == first + *n as u32).count();
                let data: Vec<u32> = self.pending[at..at + run].iter().flat_map(|(_, brick)| words(brick)).collect();
                queue.write_buffer(&self.buffer, u64::from(first) * BRICK_BYTES, bytemuck::cast_slice(&data));
                at += run;
            }
            self.pending.clear();
            return;
        }
        let slots = self.slots.next_power_of_two();
        let buffer = Self::buffer(device, slots);
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &buffer, 0, u64::from(self.buffer_slots) * BRICK_BYTES);
        if !self.pending.is_empty() {
            let data: Vec<u32> = self.pending.iter().flat_map(|(_, brick)| words(brick)).collect();
            let staging = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("chunk baked bricks upload"), contents: bytemuck::cast_slice(&data), usage: wgpu::BufferUsages::COPY_SRC });
            for (n, (slot, _)) in self.pending.drain(..).enumerate() {
                encoder.copy_buffer_to_buffer(&staging, n as u64 * BRICK_BYTES, &buffer, u64::from(slot) * BRICK_BYTES, BRICK_BYTES);
            }
        }
        self.buffer = buffer;
        self.buffer_slots = slots;
    }
}

/// The edits of `planet` as classification and generation read them;
/// `bricks`, the index of its baked store when that is unchanged.
/// The analytic brushes `planet` shows and where they are (`EditView`):
/// the large layer's faces, then the recent brushes'. From an earlier view
/// (`from`), its large faces and their index are reused while the large
/// layer only extends them (seals append; an unseal or another history
/// starts again), so an edit costs the recent brushes, and a seal the
/// large layer's index.
fn edit_view(planet: &Planet, version: u64, from: Option<&EditView>, bricks: std::sync::Arc<classify::BrickIndex>) -> EditView {
    let edits = planet.edits();
    let large = &edits.large;
    let n = large.len();
    let prefix = |count: usize| if count == 0 { 0 } else { large.prefix_hash(count as u32 - 1) };
    let faces_of = |ids: std::ops::Range<usize>| ids.flat_map(|id| large.resolved(id as u32).faces.iter().copied()).collect::<Vec<FaceBrush>>();
    let (mut faces, index) = match from.filter(|v| v.large_brushes <= n && prefix(v.large_brushes) == v.large_prefix) {
        Some(v) if v.large_brushes == n => (v.faces[..v.large_faces].to_vec(), v.index.clone()),
        Some(v) => {
            let mut faces = v.faces[..v.large_faces].to_vec();
            faces.extend(faces_of(v.large_brushes..n));
            let index = std::sync::Arc::new(classify::BrushIndex::new(&faces, 0, true));
            (faces, index)
        }
        None => {
            let faces = faces_of(0..n);
            let index = std::sync::Arc::new(classify::BrushIndex::new(&faces, 0, true));
            (faces, index)
        }
    };
    let large_faces = faces.len();
    let mut recent_counts = Vec::new();
    for id in 0..edits.recent.len() as u32 {
        let resolved = &edits.recent.resolved(id).faces;
        faces.extend(resolved.iter().copied());
        recent_counts.push(resolved.len());
    }
    let recent_index = classify::BrushIndex::new(&faces[large_faces..], large_faces as u32, false);
    EditView {
        grid: *planet.grid(),
        faces,
        index,
        recent_index,
        large_faces,
        large_brushes: n,
        large_prefix: prefix(n),
        recent_counts,
        baked: edits.baked.clone(),
        bricks,
        version,
        volume_depth: (i64::from(planet.field().volume_bounds().0) / i64::from(planet.grid().layer_mm()) + 1) as i32,
    }
}

/// A brush buffer for `faces` with room for twice as many (at least 64):
/// edits write their tail into it (`ChunkWorld::write_brushes`).
fn brush_buffer(device: &wgpu::Device, queue: &wgpu::Queue, faces: &[FaceBrush]) -> (wgpu::Buffer, usize) {
    let capacity = (faces.len() * 2).max(64);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: Some("chunk brushes"), size: (capacity * std::mem::size_of::<FaceBrush>()) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    queue.write_buffer(&buffer, 0, bytemuck::cast_slice(if faces.is_empty() { &NO_BRUSH[..] } else { faces }));
    (buffer, capacity)
}

/// A batch's restore jobs (slot, pack offset, result index, flags), their
/// packs, and each restored node with its slot and serial.
#[derive(Default)]
struct Restores {
    jobs: Vec<[u32; 4]>,
    data: Vec<u32>,
    nodes: Vec<(NodeKey, u32, u64)>,
}

/// A generated node's chunk.
struct Resident {
    slot: u32,
    /// Generation it belongs to (matches its readback).
    serial: u64,
    /// Occupied layers, once read back.
    z: Option<(i32, i32)>,
    /// Whether it was ever drawn (diagnostics: nodes evicted unseen were
    /// generated for nothing).
    shown: bool,
    /// Whether edits are over it (its chunk is kept when evicted), and its
    /// last job result's flags.
    edited: bool,
    flags: i32,
}

/// An edited chunk kept across eviction: its packed tree (`pack_main`; none:
/// it is empty), its job result's flags, the frame it was last restored or
/// kept.
struct Kept {
    pack: std::sync::Arc<[u32]>,
    flags: i32,
    used: u64,
}

/// Kept chunks out of a world ([`ChunkWorld::kept_chunks`]) for the edits
/// that made them: what a save stores ([`Self::to_bytes`]) and a new world
/// for those edits restores ([`ChunkWorld::load_kept`]).
#[derive(Clone, Debug)]
pub struct KeptChunks {
    /// Of the generation shaders, terrain and world (`ChunkWorld`'s).
    fingerprint: u64,
    /// The history's length and hash.
    history: (u64, u64),
    /// Each node, its result flags and its pack, in node order.
    chunks: Vec<(NodeKey, i32, std::sync::Arc<[u32]>)>,
}

impl KeptChunks {
    /// Chunks held.
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// As bytes: a header (magic, format, fingerprint, the history's length
    /// and hash, the chunk count), then each chunk (node, result flags, its
    /// pack's length and words) deflated, then a checksum of it all.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&KEPT_MAGIC);
        out.extend_from_slice(&KEPT_FORMAT.to_le_bytes());
        for v in [self.fingerprint, self.history.0, self.history.1, self.chunks.len() as u64] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let mut body = Vec::new();
        for (key, flags, pack) in &self.chunks {
            body.push(key.face);
            body.push(key.level as u8);
            for v in [key.cx, key.cy, key.cz, *flags, pack.len() as i32] {
                body.extend_from_slice(&v.to_le_bytes());
            }
            body.extend_from_slice(bytemuck::cast_slice(pack));
        }
        out.extend_from_slice(&miniz_oxide::deflate::compress_to_vec(&body, 6));
        let sum = crate::record::fnv64(&out);
        out.extend_from_slice(&sum.to_le_bytes());
        out
    }

    /// From [`Self::to_bytes`]; refused when they are not kept chunks of
    /// this pack format, or damaged.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 46 || bytes[..4] != KEPT_MAGIC {
            return Err("not kept chunks".into());
        }
        let (body, sum) = bytes.split_at(bytes.len() - 8);
        if crate::record::fnv64(body) != u64::from_le_bytes(sum.try_into().unwrap()) {
            return Err("kept chunks: checksum".into());
        }
        let format = u16::from_le_bytes(body[4..6].try_into().unwrap());
        if format != KEPT_FORMAT {
            return Err(format!("kept chunks of pack format {format}"));
        }
        let u64_at = |n: usize| u64::from_le_bytes(body[6 + 8 * n..14 + 8 * n].try_into().unwrap());
        let (fingerprint, history, count) = (u64_at(0), (u64_at(1), u64_at(2)), u64_at(3));
        let body = miniz_oxide::inflate::decompress_to_vec(&body[38..]).map_err(|e| format!("kept chunks: {e:?}"))?;
        let mut at = 0;
        let mut take = |n: usize| -> Result<&[u8], String> {
            let out = body.get(at..at + n).ok_or("kept chunks: cut short")?;
            at += n;
            Ok(out)
        };
        let mut chunks = Vec::with_capacity(count.min(1 << 20) as usize);
        for _ in 0..count {
            let head = take(22)?;
            let i32_at = |n: usize| i32::from_le_bytes(head[2 + 4 * n..6 + 4 * n].try_into().unwrap());
            let key = NodeKey { face: head[0], level: u32::from(head[1]), cx: i32_at(0), cy: i32_at(1), cz: i32_at(2) };
            let (flags, len) = (i32_at(3), i32_at(4).max(0) as usize);
            let pack: std::sync::Arc<[u32]> = take(len * 4)?.chunks(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect();
            chunks.push((key, flags, pack));
        }
        if at != body.len() {
            return Err("kept chunks: trailing data".into());
        }
        Ok(Self { fingerprint, history, chunks })
    }
}

/// A frame's packs on their way back: each node's slot in the directory
/// and its result flags.
struct PackReadback {
    buffer: wgpu::Buffer,
    ready: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Each node, its result flags, and its serial when packed resident.
    nodes: Vec<(NodeKey, i32, Option<u64>)>,
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
    /// A lost ray's pixel, the last one (none: 0, 0 never written).
    pub lost_pixel: (u32, u32),
    /// Places found lost (any walk, shadows' too), by `lost_at`'s reason.
    pub lost_sites: [u32; 7],
}

/// A frame's GPU time (milliseconds).
#[derive(Clone, Copy, Debug)]
pub struct FrameTimings {
    pub frame: u64,
    /// The draw in all (seeds, traces), its first seed; then sun shadows.
    pub draw: f64,
    pub seed: f64,
    pub shadow: f64,
    /// The update's eviction and generation, and the tiles and chunks it
    /// generated.
    pub generate: f64,
    pub tiles: usize,
    pub chunks: usize,
    /// Generation by kernel: eviction and tile columns, chunk columns,
    /// chunk builds.
    pub stages: [f64; 3],
    /// The first stage's parts: eviction and the tiles' nodes, their
    /// columns, their halos.
    pub tile_stages: [f64; 3],
}

/// Timestamps a frame writes: generation start, tile halos end, seed start
/// and end, trace end, chunk columns end, chunk builds end, shadows end,
/// tile nodes end, tile columns end.
const QUERIES: u32 = 10;

/// `Buried`'s size: a count and 4095 keys.
const BURIED_BYTES: u64 = 16 * 4096;

/// Pixels a side of the block one ray report samples (`feedback`), and
/// the reports' buffer (a key a block, up to 4096x4096 pixels).
const FEEDBACK_BLOCK: u32 = 8;
/// Frames the reports take to sample every pixel of their blocks.
const FEEDBACK_FRAMES: u64 = (FEEDBACK_BLOCK * FEEDBACK_BLOCK) as u64;
const FEEDBACK_BYTES: u64 = 16 * (4096 / FEEDBACK_BLOCK as u64) * (4096 / FEEDBACK_BLOCK as u64);

/// Picks a frame answers (more wait a frame).
const MAX_PICKS: usize = 8;

/// A frame's pick hits on their way back: per pick 256 bytes, its
/// visibility (8) then distance (at 16).
struct PickSlot {
    buffer: wgpu::Buffer,
    /// Request ids, the tool's picks, and the state as for `Buried`.
    requests: Vec<u64>,
    sink: Option<crate::engine::SharedPicks>,
    state: ReachState,
}

/// A copy of a frame's ray reports (`Buried`, or the feedback's) on its
/// way back: its bytes.
struct ReachSlot {
    buffer: wgpu::Buffer,
    state: ReachState,
    bytes: u64,
}

/// Map the report copies just submitted; read (`read`) and free those
/// mapped.
fn take_reports(slots: &mut [ReachSlot], mut read: impl FnMut(&[u8])) {
    for slot in slots {
        match &slot.state {
            ReachState::Copied => {
                let ready = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                let flag = ready.clone();
                slot.buffer.slice(..slot.bytes).map_async(wgpu::MapMode::Read, move |r| {
                    if r.is_ok() {
                        flag.store(true, std::sync::atomic::Ordering::Release);
                    }
                });
                slot.state = ReachState::Mapping(ready);
            }
            ReachState::Mapping(ready) if ready.load(std::sync::atomic::Ordering::Acquire) => {
                read(&slot.buffer.slice(..slot.bytes).get_mapped_range().unwrap());
                slot.buffer.unmap();
                slot.state = ReachState::Free;
            }
            _ => {}
        }
    }
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
    footprints: Vec<Footprint>,
    /// Chunk jobs in order: node, slot, serial.
    chunks: Vec<(NodeKey, u32, u64)>,
}

/// What [`ChunkWorld::update`] did.
#[derive(Clone, Copy, Debug, Default)]
pub struct UpdateStats {
    /// Nodes waiting to be generated.
    pub requests: usize,
    /// Footprints whose columns were computed, footprints read from the
    /// pool, and chunks generated.
    pub generated_footprints: usize,
    pub cached_footprints: usize,
    pub generated_chunks: usize,
    pub evicted_chunks: usize,
    /// Evicted chunks that were never drawn, and generated chunks drawn for
    /// the first time (diagnostics: what generation was for).
    pub evicted_unseen: usize,
    pub shown_chunks: usize,
    /// Edited chunks the plan evicted left resident past the frame's packs
    /// (evicted, kept, by a later plan).
    pub evictions_deferred: usize,
    /// Drawn nodes (Solid and Buried ones too), those fading in or out of
    /// a switch between levels, and drawn chunks.
    pub drawn_nodes: usize,
    pub fading_nodes: usize,
    pub drawn_chunks: usize,
    pub resident_chunks: usize,
    /// Nodes rays met (ray reports this frame), and Buried nodes rays met
    /// (generated next).
    pub seen: usize,
    pub reached: usize,
    /// Drawn Mixed nodes in view coarser than wanted, as a share of them.
    pub coarse_share: f64,
    /// Frames since the eye the plan in use was made for, and the worker's
    /// milliseconds on it.
    pub plan_age: u64,
    pub plan_ms: f64,
    /// [`residency::Plan::worst_px`] of the latest plan, and its node;
    /// [`residency::Plan::worst_wanted`].
    pub worst_px: f64,
    pub worst_wanted: f64,
    pub worst: Option<residency::Worst>,
    /// Level switches in view a plan that arrived this frame made, and
    /// [`residency::Plan::switch_px`].
    pub switches: usize,
    pub switch_px: f64,
    /// Chunks this frame could generate.
    pub budget: usize,
    /// CPU milliseconds: results, plan, eviction, generation, drawn set,
    /// table.
    pub cpu_ms: [f64; 6],
    pub out_of_slots: bool,
    pub out_of_leaves: bool,
    /// A chunk kept its leaves' material pairs for want of detail pages.
    pub out_of_details: bool,
    /// Detail pages in use when last read back, of the capacity.
    pub detail_pages: (u64, u64),
    /// Edited chunks restored by copy this frame, those generated (from
    /// the field and the edits over them), and those kept.
    pub restored_chunks: usize,
    pub generated_edited: usize,
    pub kept_chunks: usize,
    pub kept_bytes: usize,
}

/// How [`ChunkWorld::update`] plans and how much it generates a frame.
#[derive(Clone, Copy, Debug)]
pub struct ResidencyConfig {
    pub plan: residency::PlanConfig,
    /// Footprints generated per frame at most.
    pub footprints_per_frame: usize,
    /// GPU milliseconds of generation a frame at 60 frames a second
    /// (against the measured cost of tiles and chunks); `ChunkPass` gives a
    /// slower frame proportionally more, up to twice.
    pub generate_ms: Option<f64>,
    /// Milliseconds a frame may take: `ChunkPass` lets generation grow into
    /// what the last measured frame left of them on the GPU and the CPU, by
    /// half a millisecond a frame at most, up to three times `generate_ms`
    /// (nodes wait only while the frame is full).
    pub frame_ms: Option<f64>,
}

impl Default for ResidencyConfig {
    /// Cells of a pixel or less where rays met them in the last 90 frames
    /// (two in view where none did, two out of view past a 20 degree guard;
    /// up to three where the eye's motion sweeps the image 32 pixels a
    /// frame; a 45 degree view 1080 pixels tall), 400 km out, half a second
    /// of prefetch, buried nodes generated where they hold the eye, chunks
    /// unvisited for 600 frames released (`HELIO_VOXEL_CHUNKS_RELEASE_FRAMES`;
    /// 0: only when slots run short); up to 256 footprints and 3 ms of
    /// generation a frame.
    fn default() -> Self {
        let release_frames = std::env::var("HELIO_VOXEL_CHUNKS_RELEASE_FRAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(600);
        Self {
            plan: residency::PlanConfig {
                pixel_angle: 2.0 * std::f64::consts::FRAC_PI_8.tan() / 1080.0,
                lod_pixels: 1.0,
                offscreen_pixels: 2.0,
                view_guard: 0.35,
                unseen_pixels: 2.0,
                seen_frames: 90,
                coarsen: 0.8,
                max_distance: 400_000.0,
                prefetch_seconds: 0.5,
                keep_frames: 120,
                release_frames,
                motion_flow: 16.0,
                motion_max: 3.0,
                near_cells: 2.0,
            },
            footprints_per_frame: 256,
            generate_ms: Some(3.0),
            frame_ms: Some(16.0),
        }
    }
}

/// Chunk trees of one planet: resident chunks, their GPU storage and the
/// draw.
pub struct ChunkWorld {
    device: wgpu::Device,
    queue: wgpu::Queue,
    gen_layout: wgpu::BindGroupLayout,
    tile_nodes: wgpu::ComputePipeline,
    tile_columns: wgpu::ComputePipeline,
    tile_halo: wgpu::ComputePipeline,
    tile_slope_nodes: wgpu::ComputePipeline,
    tile_slope_heights: wgpu::ComputePipeline,
    tile_slopes: wgpu::ComputePipeline,
    chunk_columns: wgpu::ComputePipeline,
    chunk_build: wgpu::ComputePipeline,
    draw_layout: wgpu::BindGroupLayout,
    trace_layout: wgpu::BindGroupLayout,
    seed_layout: wgpu::BindGroupLayout,
    resolve_layout: wgpu::BindGroupLayout,
    /// The UV reference image of debug mode 1 (trace and publish).
    uv_grid_view: wgpu::TextureView,
    uv_grid_sampler: wgpu::Sampler,
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
    // The batch's tiles' two-rate nodes (`engine::TILE_NODES`^2 a tile).
    tile_node_data: wgpu::Buffer,
    // Lean lattice heights past each tile's edges (`HALO_SIDE`^2 a tile).
    tile_halo_data: wgpu::Buffer,
    // Each footprint's material slopes (`SLOPE_STRIDE` a footprint), and
    // the batch's lattices of level-4 tops they come from.
    tile_slope_data: wgpu::Buffer,
    slope_lattice: wgpu::Buffer,
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
    // Each stored leaf's ground normal (two bytes, as `leaf_materials`).
    leaf_normals: wgpu::Buffer,
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
    /// Buried nodes rays met (`Buried`) and its readback slots.
    buried: wgpu::Buffer,
    buried_slots: Vec<ReachSlot>,
    /// Ray reports (`feedback`): a node a block of pixels' sampled hit
    /// wants, its readback slots, and the kernel.
    feedback: wgpu::Buffer,
    feedback_slots: Vec<ReachSlot>,
    feedback_layout: wgpu::BindGroupLayout,
    feedback_pipeline: wgpu::ComputePipeline,
    /// Whether this world polls the device for its readbacks: not on a
    /// host's shared device (`PrepareContext::owns_device`), where polling
    /// waits behind the host's submits and presents (wgpu maintains the
    /// device on every submit, which delivers the map callbacks).
    polls_device: bool,
    /// Whether draws write the plain colour target (a world drawn alone:
    /// tests, harnesses); a host's gbuffer takes the published surface.
    plain_colour: bool,
    /// The renderer-wide debug mode, for the voxel debug shading.
    debug_mode: u32,
    pick_slots: Vec<PickSlot>,
    /// Each slot's chunk key (`chunk_keys`, mirrored).
    slot_keys: Vec<[i32; 4]>,
    /// Nodes rays reported since the last update: met, and Buried ones met.
    seen: Vec<NodeKey>,
    reached: Vec<NodeKey>,
    /// Frames of ray reports read and not yet sent to the planner, and the
    /// planner's count of them when residency last changed: settled takes
    /// a plan that saw a full cycle of them since (`FEEDBACK_FRAMES`).
    reports_unsent: u64,
    quiet_from: u64,
    evict_pipeline: wgpu::ComputePipeline,
    /// Kept chunks: packing evicted edited chunks, restoring them.
    keep_layout: wgpu::BindGroupLayout,
    pack_pipeline: wgpu::ComputePipeline,
    restore_pipeline: wgpu::ComputePipeline,
    pack_data: wgpu::Buffer,
    pack_alloc: wgpu::Buffer,
    pack_readbacks: std::collections::VecDeque<PackReadback>,
    pack_pending: Option<PackReadback>,
    kept: FxHashMap<NodeKey, Kept>,
    /// Nodes packed this frame or the last, not edited since (their packs
    /// are kept when they come back).
    packed_since: FxHashSet<NodeKey>,
    kept_bytes: usize,
    kept_limit: usize,
    /// Resident edited chunks to pack without evicting them
    /// (`keep_resident`), and the fingerprint of what made packs
    /// (`kept_chunks`).
    to_pack: Vec<NodeKey>,
    fingerprint: u64,
    /// Chunk table changes scattered (`chunk_table.wgsl`).
    table_layout: wgpu::BindGroupLayout,
    table_pipeline: wgpu::ComputePipeline,
    /// The free leaf pages (a stack, `leaf_alloc` its count) and each
    /// slot's pages.
    free_pages: wgpu::Buffer,
    leaf_pages: wgpu::Buffer,
    job_results: wgpu::Buffer,
    /// The edits classification and generation read (brushes kept
    /// analytic, resolved per face, and the baked store), shared with the
    /// planner; the version it has not seen yet; the brushes' buffer.
    edits: std::sync::Arc<EditView>,
    edits_unsent: Option<std::sync::Arc<EditView>>,
    brushes: wgpu::Buffer,
    /// Faces `brushes` has room for.
    brush_capacity: usize,
    brick_cache: BrickCache,
    edit_state: EditState,
    /// Nodes to generate again: under brushes that came or went (first,
    /// coarsest first), drawn ones the frame they change; under baked
    /// edits that changed, as budget allows.
    urgent: FxHashSet<NodeKey>,
    /// Urgent nodes the planner hides until they are generated again
    /// (`residency::Event::Stale`).
    hidden: FxHashSet<NodeKey>,
    /// Urgent nodes' distances from the eye when first ranked (an edit
    /// settles in moments).
    urgent_distance: FxHashMap<NodeKey, f64>,
    stale: FxHashSet<NodeKey>,
    /// Chunk slots and leaves this world holds.
    max_chunks: usize,
    leaf_capacity: u64,
    /// Per-voxel materials (`chunk_generate.wgsl`): a batch's edited cell
    /// materials, each leaf page's detail page, the pages, their free
    /// stack and its count; pages this world holds and those free when last
    /// read back.
    cell_materials: wgpu::Buffer,
    detail_pages: wgpu::Buffer,
    detail_cells: wgpu::Buffer,
    detail_alloc: wgpu::Buffer,
    detail_free: wgpu::Buffer,
    /// Per slot: whether any of its leaf pages has a detail page.
    slot_details: wgpu::Buffer,
    detail_capacity: u64,
    detail_free_seen: u64,
    grid: Grid,
    /// Height over the datum above which nothing is solid (terrain,
    /// overhangs, edits), with a margin: rays rising past it end.
    ceiling: f64,
    /// The planet's height range in base layers (the roots' estimate).
    height_layers: (i32, i32),
    /// Free leaf pages when last read back.
    free_pages_seen: u64,
    worker: Option<residency::Worker>,
    /// The latest finished plan, and events for the next.
    plan: residency::Plan,
    events: Vec<residency::Event>,
    /// Footprints whose columns are on the GPU, and this update's frame.
    pool: FootprintPool,
    frame_now: u64,
    /// Generated nodes with a chunk, and their layers by footprint.
    nodes: FxHashMap<NodeKey, Resident>,
    footprint_nodes: FxHashMap<Footprint, Vec<i32>>,
    /// Slots of chunks dropped between updates (edits), freed by the next.
    dropped: Vec<u32>,
    free_slots: Vec<u32>,
    serial: u64,
    readbacks: std::collections::VecDeque<Readback>,
    /// This frame's drawn nodes.
    drawn_set: drawn::DrawnSet,
    /// Nodes whose chunks changed since the drawn set last saw them.
    changed_nodes: FxHashSet<NodeKey>,
    /// The drawn set's version whose slots `drawn_slots` holds.
    slots_version: Option<u64>,
    /// Results recorded by the last update, mapped once submitted.
    pending: Option<Readback>,
    last_requests: usize,
    /// What the latest plan was made for (none: no plan yet).
    planned_for: Option<residency::PlannedFor>,
    /// The frame the current plan was made for and the worker's
    /// milliseconds on it.
    plan_made: (u64, f64),

    /// GPU cost of generation: measured (moving averages) from
    /// [`GenCost::PRIOR`].
    cost: GenCost,
    /// Footprints and chunks the last update generated.
    last_generated: (usize, usize),
}

/// What generation costs on the GPU: milliseconds of a tile's columns and
/// of a chunk's columns and tree on average, chunks a tile, and the lines
/// through each frame's batch (a batch costs a fixed part, then each tile
/// and chunk: the averages of the small batches a flight makes hold mostly
/// that fixed part, and would budget a few times too few tiles).
#[derive(Clone, Copy, Debug)]
struct GenCost {
    tile: f64,
    chunk: f64,
    chunks_per_tile: f64,
    tiles: CostLine,
    chunks: CostLine,
}

impl GenCost {
    /// Before the first measured frame: an RTX 3060's (a 300 m/s cruise and
    /// a 50 m blast).
    const PRIOR: Self = Self { tile: 0.035, chunk: 0.007, chunks_per_tile: 1.7, tiles: CostLine::EMPTY, chunks: CostLine::EMPTY };

    /// A batch's fixed milliseconds, then a tile's and a chunk's: the lines'
    /// once their batches spread enough, else the averages.
    fn model(&self) -> (f64, f64, f64) {
        let (fixed_t, tile) = self.tiles.fit().unwrap_or((0.0, self.tile));
        let (fixed_c, chunk) = self.chunks.fit().unwrap_or((0.0, self.chunk));
        (fixed_t + fixed_c, tile, chunk)
    }
}

/// A least-squares line through (items, milliseconds) samples, recent ones
/// weighing more: what a batch costs, fixed and per item.
#[derive(Clone, Copy, Debug)]
struct CostLine {
    w: f64,
    x: f64,
    y: f64,
    xx: f64,
    xy: f64,
}

impl CostLine {
    const EMPTY: Self = Self { w: 0.0, x: 0.0, y: 0.0, xx: 0.0, xy: 0.0 };

    fn add(&mut self, x: f64, y: f64) {
        const KEEP: f64 = 0.98;
        *self = Self { w: self.w * KEEP + 1.0, x: self.x * KEEP + x, y: self.y * KEEP + y, xx: self.xx * KEEP + x * x, xy: self.xy * KEEP + x * y };
    }

    /// Fixed and per-item milliseconds, once enough batches of spread
    /// sizes were seen (a line through a size alone is anything).
    fn fit(&self) -> Option<(f64, f64)> {
        if self.w < 8.0 {
            return None;
        }
        let (mx, my) = (self.x / self.w, self.y / self.w);
        let var = self.xx / self.w - mx * mx;
        if var < (0.25 * mx).powi(2).max(4.0) {
            return None;
        }
        let per = (self.xy / self.w - mx * my) / var;
        let fixed = my - per * mx;
        // A fixed part under zero: the plain average, per item.
        if per <= 0.0 || fixed < 0.0 {
            return (mx > 0.0).then(|| (0.0, my / mx));
        }
        Some((fixed, per))
    }
}

/// The draw's targets: the seed (each pixel's nearest box) and the trace's
/// outputs.
pub struct Targets {
    pub size: [u32; 2],
    seed: wgpu::Texture,
    /// The seed's box entries (Depth32Float, reversed: `seed_box`).
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

fn sampler_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None }
}

/// The UV reference image (debug mode 1: faces shaded by their UVs), a
/// repeating, filtered texture.
fn create_uv_grid(device: &wgpu::Device, queue: &wgpu::Queue) -> (wgpu::TextureView, wgpu::Sampler) {
    const UV_GRID_BYTES: &[u8] = include_bytes!("../../../../../../assets/UV_Grid_Sm.jpg");
    let decoded = image::load_from_memory(UV_GRID_BYTES).expect("embedded UV reference image is valid").to_rgba8();
    let (width, height) = decoded.dimensions();
    let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("voxel UV reference image"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: &texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        decoded.as_raw(),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(width * 4), rows_per_image: Some(height) },
        size,
    );
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("voxel UV reference sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });
    (texture.create_view(&Default::default()), sampler)
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
    // Past a marker's line (checkouts may end lines with \r\n).
    let past_line = |text: &str, at: usize| text[at..].find('\n').map_or(text.len(), |n| at + n + 1);
    let mut out = String::with_capacity(source.len() * 2);
    let mut rest = source;
    while let Some(start) = rest.find("// WALK {") {
        let end = rest[start..].find("// } WALK").expect("closed walk template") + start;
        let body = &rest[past_line(rest, start)..end];
        out.push_str(&rest[..start]);
        out.push_str(body);
        out.push_str(
            &body
                .replace("walk(Walk(", "walk_fine(Walk(")
                .replace("fn sun_walk(", "fn sun_walk_fine(")
                .replace("fn sky_visibility(", "fn sky_visibility_fine("),
        );
        rest = &rest[past_line(rest, end)..];
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
    // The primary steps' heat (debug mode 32 and `HELIO_VOXEL_CHUNKS_HEAT`)
    // counts in every build.
    let wcount = |n: u32| {
        let heat = format!("if counted && (voxel_heat_debug() || debug_mode() == 32u) {{ heat += u32({n} == 1); }}");
        if counting { format!("if counted {{ atomicAdd(&counters[{n}], 1u); }} {heat}") } else { heat }
    };
    // The noise library (materials' patches), then the draw.
    let draw = (format!("const DOMAIN_UNIT: f32 = {};\n", crate::grid::DOMAIN_UNIT) + include_str!("../../shaders/noise.wgsl") + &walk_templates(include_str!("../../shaders/chunk_draw.wgsl")))
        .replace("COUNTERS", if counting { "@group(0) @binding(9) var<storage, read_write> counters: array<atomic<u32>, 256>;" } else { "" })
        .replace("WCOUNT(1)", &wcount(1))
        .replace("WCOUNT(3)", &wcount(3))
        .replace("WCOUNT(8)", &wcount(8))
        .replace("WCOUNT(10)", &wcount(10))
        // Lost places by reason (`lost_at`: counters 5, 6, 7, 11, 12, 13, 15).
        .replace("LOST_SITE", if counting { "atomicAdd(&counters[array<u32, 7>(5u, 6u, 7u, 11u, 12u, 13u, 15u)[site]], 1u);" } else { "" })
        // Where the last search past the top began: face | level << 8,
        // cell (counters[200..204]).
        .replace("LOST_WHERE", if counting { "atomicStore(&counters[200], u32(face | (level << 8u))); atomicStore(&counters[201], bitcast<u32>(cell.x)); atomicStore(&counters[202], bitcast<u32>(cell.y)); atomicStore(&counters[203], bitcast<u32>(cell.z));" } else { "" })
        // The last lost ray's pixel (counters[14]).
        .replace("LOST_PIXEL", if counting { "if counted { atomicStore(&counters[14], pixel.x | (pixel.y << 16u)); }" } else { "" })
        .replace("WCOUNT_BOXES", if counting { "if counted { atomicAdd(&counters[9], chunk); }" } else { "" })
        .replace("COUNT(0)", &count(0))
        .replace("COUNT(2)", &count(2))
        // Diagnostics (counting builds): the chunks one pixel's ray visits
        // (`HELIO_VOXEL_CHUNKS_DEBUG_PIXEL=x,y`), from counters[16].
        .replace(
            "TRACE_PIXEL(chunk, slot, cube_lo, real)",
            if counting {
                "if counted && all(pixel == params.debug.zw) && chunk < 24u { atomicStore(&counters[16u + chunk * 5u], slot); atomicStore(&counters[17u + chunk * 5u], bitcast<u32>(cube_lo.x)); atomicStore(&counters[18u + chunk * 5u], bitcast<u32>(cube_lo.y)); atomicStore(&counters[19u + chunk * 5u], bitcast<u32>(cube_lo.z)); atomicStore(&counters[20u + chunk * 5u], u32(real) + 2u * u32(key.x >> 8u)); atomicStore(&counters[4], chunk + 1u); }"
            } else {
                ""
            },
        )
        .replace("FADE_FRAMES", &drawn::FADE_FRAMES.to_string())
        .replace("PLANE_WORLD", if plane { "true" } else { "false" });
    [("chunk generation", terrain_source(plane, program, wide64, true) + &format!("const HALO_GROUPS: u32 = {}u;\n", crate::engine::HALO_GROUPS) + include_str!("../../shaders/chunk_generate.wgsl")), ("chunk draw", draw)]
}

impl ChunkWorld {
    /// A world of the default [`Capacity`].
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, planet: &Planet) -> Self {
        Self::with_capacity(device, queue, planet, Capacity::default())
    }

    pub fn with_capacity(device: &wgpu::Device, queue: &wgpu::Queue, planet: &Planet, capacity: Capacity) -> Self {
        let program = planet.field().program();
        let wide64 = device.features().contains(wgpu::Features::SHADER_INT64);
        let [(_, gen_source), (_, draw_source)] = shader_sources(planet.grid().is_plane(), &program, wide64, std::env::var_os("HELIO_VOXEL_CHUNKS_COUNT").is_some());
        // What generated packs depend on: the generation shaders, the
        // terrain's constants and the world (`kept_chunks`).
        let fingerprint = {
            let mut bytes = format!("chunk packs {KEPT_FORMAT} {DETAIL_LEVELS} {}", gen_source).into_bytes();
            bytes.extend_from_slice(&terrain_bytes(&program));
            bytes.extend_from_slice(bytemuck::bytes_of(&WorldGpu::new(planet)));
            crate::record::fnv64(&bytes)
        };
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
                storage(24, false, c),
                storage(25, false, c),
                storage(26, false, c),
                storage(27, false, c),
                storage(28, false, c),
                storage(29, false, c),
                storage(30, false, c),
                storage(31, false, c),
                storage(32, false, c),
                storage(33, false, c),
                storage(34, false, c),
                storage(35, false, c),
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
        let tile_nodes = compute("tile_nodes_main");
        let tile_columns = compute("tile_columns_main");
        let tile_halo = compute("tile_halo_main");
        let tile_slope_nodes = compute("tile_slope_nodes_main");
        let tile_slope_heights = compute("tile_slope_heights_main");
        let tile_slopes = compute("tile_slopes_main");
        let chunk_columns = compute("chunk_columns_main");
        let chunk_build = compute("chunk_build_main");
        let evict_pipeline = compute("evict");
        let keep_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("chunk keep"),
            entries: &[storage(0, true, c), storage(1, false, c), storage(2, false, c), storage(3, true, c), storage(4, true, c)],
        });
        let keep_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("chunk keep"), bind_group_layouts: &[Some(&gen_layout), Some(&keep_layout)], immediate_size: 0 });
        let keep = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(entry), layout: Some(&keep_pipeline_layout), module: &gen_module, entry_point: Some(entry), compilation_options: Default::default(), cache: None })
        };
        let (pack_pipeline, restore_pipeline) = (keep("pack_main"), keep("restore_main"));
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
                storage(14, true, all),
                storage(15, true, all),
                storage(16, true, all),
                storage(17, true, all),
                storage(18, true, all),
            ],
        });
        let trace_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("chunk trace"),
            entries: &[
                texture_entry(0, wgpu::TextureSampleType::Uint, wgpu::ShaderStages::COMPUTE),
                storage_texture(1, wgpu::TextureFormat::Rg32Uint),
                texture_entry(25, wgpu::TextureSampleType::Float { filterable: true }, wgpu::ShaderStages::COMPUTE),
                sampler_entry(26, wgpu::ShaderStages::COMPUTE),
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
        let (uv_grid_view, uv_grid_sampler) = create_uv_grid(device, queue);
        let resolve_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("chunk depth resolve"),
            entries: &[
                texture_entry(4, wgpu::TextureSampleType::Float { filterable: false }, wgpu::ShaderStages::FRAGMENT),
                texture_entry(7, wgpu::TextureSampleType::Uint, wgpu::ShaderStages::FRAGMENT),
                texture_entry(14, wgpu::TextureSampleType::Float { filterable: false }, wgpu::ShaderStages::FRAGMENT),
                texture_entry(25, wgpu::TextureSampleType::Float { filterable: true }, wgpu::ShaderStages::FRAGMENT),
                sampler_entry(26, wgpu::ShaderStages::FRAGMENT),
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
                // The nearest box wins (`seed_box`).
                depth_stencil: depth(helio_core::depth::NEARER),
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
        // Ray reports: the hits a block of pixels samples, as the nodes
        // they want.
        let feedback_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("chunk feedback"),
            entries: &[hits[0], hits[1], storage(24, false, wgpu::ShaderStages::COMPUTE)],
        });
        let feedback_pipeline = compute("chunk feedback", &layout(&feedback_layout), "feedback");
        let publish = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("chunk gbuffer"),
            layout: Some(&layout(&resolve_layout)),
            vertex: wgpu::VertexState { module: &draw_module, entry_point: Some("resolve_vertex"), compilation_options: Default::default(), buffers: &[] },
            primitive: Default::default(),
            depth_stencil: depth(helio_core::depth::NEARER),
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

        let buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
        // Brushes kept analytic: the large ones, then the recent ones.
        let edits = planet.edits();
        let view = std::sync::Arc::new(edit_view(planet, 0, None, std::sync::Arc::new(classify::BrickIndex::new(planet.grid(), &planet.edits().baked))));
        let (brushes, brush_capacity) = brush_buffer(device, queue, &view.faces);
        let s = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        // The capacity within what the device binds: the largest per-slot
        // buffer is its mids (1 KiB), and the leaves are 8 bytes each.
        let limits = device.limits();
        let binding = u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size);
        let max_chunks = capacity.chunks.min((binding / 1024) as usize).min((binding / (LEAVES_PER_SLOT * 8)) as usize).max(1024);
        let leaf_capacity = max_chunks as u64 * LEAVES_PER_SLOT;
        let detail_capacity = capacity.detail_pages.min((binding / (DETAIL_WORDS * 4)) as usize).max(64) as u64;
        let table_slots = (max_chunks * TABLE_PER_SLOT).next_power_of_two().min((binding / 32).next_power_of_two() as usize / 2);
        let pool_size = capacity.footprints.min((binding / (TILE_COLUMNS * 16)) as usize).max(2 * TILE_BATCH);
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
        let drawn_set = drawn::DrawnSet::new(max_chunks, table_slots, planet.grid().levels() - 1);
        let chunk_table = buffer("chunk table", (table_slots * 32) as u64, s);
        queue.write_buffer(&chunk_table, 0, bytemuck::cast_slice(drawn_set.table()));
        Self {
            device: device.clone(),
            queue: queue.clone(),
            gen_layout,
            tile_nodes,
            tile_columns,
            tile_halo,
            tile_slope_nodes,
            tile_slope_heights,
            tile_slopes,
            chunk_columns,
            chunk_build,
            draw_layout,
            trace_layout,
            seed_layout: seed_layout_1,
            resolve_layout,
            uv_grid_view,
            uv_grid_sampler,
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
            publish,
            last_group: None,
            targets: None,
            rounds: std::env::var("HELIO_VOXEL_CHUNKS_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(1),
            world,
            terrain,
            tile_jobs: buffer("chunk tile jobs", (TILE_BATCH * std::mem::size_of::<TileJob>()) as u64, s),
            tile_data: buffer("chunk footprint columns", pool_size as u64 * TILE_COLUMNS * 16, wgpu::BufferUsages::STORAGE),
            // The tiles' nodes, the scratch nodes, the tiles' slope lattice
            // rings, and the 8 a column's node words read past the last.
            tile_node_data: buffer(
                "chunk tile nodes",
                (2 * TILE_BATCH as u64 * u64::from(crate::engine::TILE_NODES.pow(2)) + u64::from(crate::engine::NODE_SCRATCH) + 8) * u64::from(crate::engine::node_words(&program).max(1)) * 4,
                wgpu::BufferUsages::STORAGE,
            ),
            tile_halo_data: buffer("chunk footprint halos", pool_size as u64 * 40 * 40 * 4, wgpu::BufferUsages::STORAGE),
            tile_slope_data: buffer("chunk footprint slopes", pool_size as u64 * SLOPE_STRIDE * 4, wgpu::BufferUsages::STORAGE),
            slope_lattice: buffer("chunk slope lattices", TILE_BATCH as u64 * SLOPE_LATTICE * 4, wgpu::BufferUsages::STORAGE),
            // Four words a tile, then the batch's tile count.
            tile_ranges: buffer("chunk tile ranges", (TILE_BATCH * 16 + 4) as u64, s | wgpu::BufferUsages::COPY_SRC),
            chunk_jobs: buffer("chunk jobs", (CHUNK_BATCH * 32) as u64, s),
            column_masks: buffer("chunk column masks", CHUNK_BATCH as u64 * TILE_COLUMNS * 8, wgpu::BufferUsages::STORAGE),
            chunk_meta: buffer("chunk meta", (max_chunks * 48) as u64, s | wgpu::BufferUsages::COPY_SRC),
            mids: buffer("chunk mids", (max_chunks * 64 * 16) as u64, wgpu::BufferUsages::STORAGE),
            mid_leaves: buffer("chunk mid leaves", (max_chunks * 64 * 4) as u64, wgpu::BufferUsages::STORAGE),
            leaves: buffer("chunk leaves", leaf_capacity * 8, wgpu::BufferUsages::STORAGE),
            // Every page free: the stack's count, no overflow.
            leaf_alloc: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("chunk leaf allocator"),
                contents: bytemuck::cast_slice(&[(leaf_capacity / LEAF_PAGE) as i32, 0]),
                usage: s | wgpu::BufferUsages::COPY_SRC,
            }),
            column_materials: buffer("chunk column materials", CHUNK_BATCH as u64 * TILE_COLUMNS * 4, wgpu::BufferUsages::STORAGE),
            leaf_materials: buffer("chunk leaf materials", leaf_capacity * 2, s),
            leaf_normals: buffer("chunk leaf normals", leaf_capacity * 2, s),
            mid_materials: buffer("chunk mid materials", (max_chunks * 64 * 4) as u64, wgpu::BufferUsages::STORAGE),
            params: buffer("chunk draw params", std::mem::size_of::<DrawParams>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
            materials,
            chunk_keys: buffer("chunk keys", (max_chunks * 16) as u64, s),
            chunk_table,

            face_cells: planet.grid().cells() as u32,
            exact_level: exact_level(planet.grid()),
            statics: buffer("chunk statics", (max_chunks * std::mem::size_of::<ChunkStatic>()) as u64, s),
            draw_list: buffer("chunk draw list", (max_chunks * 4) as u64, s),
            drawn_slots: buffer("chunk drawn slots", (max_chunks * 4) as u64, s),
            cull_state: buffer("chunk cull", CULL_BYTES, s | wgpu::BufferUsages::COPY_SRC),
            cull_bins: buffer("chunk cull bins", (max_chunks * 4) as u64, s),
            seed_args: buffer("chunk seed draws", 32, s | wgpu::BufferUsages::INDIRECT),
            queries,
            timing_slots,
            timing_now: None,
            timings: Vec::new(),
            counters: buffer("chunk counters", 1024, s | wgpu::BufferUsages::COPY_SRC),
            buried: buffer("chunk buried hits", BURIED_BYTES, s | wgpu::BufferUsages::COPY_SRC),
            polls_device: true,
            plain_colour: true,
            debug_mode: 0,
            buried_slots: (0..TIMING_RING).map(|_| ReachSlot { buffer: buffer("chunk buried hits read", BURIED_BYTES, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST), state: ReachState::Free, bytes: BURIED_BYTES }).collect(),
            feedback: buffer("chunk feedback", FEEDBACK_BYTES, s | wgpu::BufferUsages::COPY_SRC),
            feedback_slots: (0..TIMING_RING).map(|_| ReachSlot { buffer: buffer("chunk feedback read", FEEDBACK_BYTES, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST), state: ReachState::Free, bytes: 16 }).collect(),
            feedback_layout,
            feedback_pipeline,
            seen: Vec::new(),
            reports_unsent: 0,
            quiet_from: 0,
            reached: Vec::new(),
            pick_slots: (0..3).map(|_| PickSlot { buffer: buffer("chunk picks", 256 * MAX_PICKS as u64, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST), requests: Vec::new(), sink: None, state: ReachState::Free }).collect(),
            slot_keys: vec![[-1, 0, 0, 0]; max_chunks],
            evict_pipeline,
            keep_layout,
            pack_pipeline,
            restore_pipeline,
            pack_data: buffer("chunk packs", PACK_WORDS * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
            pack_alloc: buffer("chunk pack directory", (1 + 2 * MAX_PACKS as u64) * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST),
            pack_readbacks: Default::default(),
            pack_pending: None,
            kept: FxHashMap::default(),
            packed_since: FxHashSet::default(),
            kept_bytes: 0,
            kept_limit: capacity.kept_bytes,
            to_pack: Vec::new(),
            fingerprint,
            table_layout,
            table_pipeline,
            free_pages: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("chunk free pages"),
                contents: bytemuck::cast_slice(&(0..(leaf_capacity / LEAF_PAGE) as u32).collect::<Vec<u32>>()),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            leaf_pages: buffer("chunk leaf pages", max_chunks as u64 * LEAF_PAGES * 4, wgpu::BufferUsages::STORAGE),
            job_results: buffer("chunk job results", (CHUNK_BATCH * 16) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
            brushes,
            brush_capacity,
            brick_cache: BrickCache::new(device, capacity.bricks.min((binding / BRICK_BYTES) as usize) as u32),
            edits: view.clone(),
            edits_unsent: None,
            edit_state: EditState::of(edits),
            max_chunks,
            leaf_capacity,
            cell_materials: buffer("chunk cell materials", DETAIL_JOBS as u64 * TILE_COLUMNS * CELL_WORDS * 4, wgpu::BufferUsages::STORAGE),
            // No leaf page has a detail page yet.
            detail_pages: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("chunk detail pages"),
                contents: bytemuck::cast_slice(&vec![u32::MAX; (leaf_capacity / LEAF_PAGE) as usize]),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            detail_cells: buffer("chunk detail cells", detail_capacity * DETAIL_WORDS * 4, wgpu::BufferUsages::STORAGE),
            detail_alloc: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("chunk detail allocator"),
                contents: bytemuck::cast_slice(&[detail_capacity as i32, 0]),
                usage: s | wgpu::BufferUsages::COPY_SRC,
            }),
            detail_free: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("chunk detail free pages"),
                contents: bytemuck::cast_slice(&(0..detail_capacity as u32).collect::<Vec<u32>>()),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            slot_details: buffer("chunk slot details", max_chunks as u64 * 4, wgpu::BufferUsages::STORAGE),
            detail_capacity,
            detail_free_seen: detail_capacity,
            free_pages_seen: leaf_capacity / LEAF_PAGE,
            urgent: FxHashSet::default(),
            hidden: FxHashSet::default(),
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
            pool: FootprintPool::new(pool_size),
            frame_now: 0,
            nodes: Default::default(),
            footprint_nodes: Default::default(),
            dropped: Vec::new(),
            free_slots: (0..max_chunks as u32).rev().collect(),
            serial: 0,
            readbacks: std::collections::VecDeque::new(),
            drawn_set,
            changed_nodes: Default::default(),
            slots_version: None,
            pending: None,
            last_requests: 0,
            planned_for: None,
            plan_made: (0, 0.0),

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
                entry(25, &self.tile_node_data),
                entry(26, &self.tile_halo_data),
                entry(34, &self.tile_slope_data),
                entry(35, &self.slope_lattice),
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
                entry(27, &self.leaf_normals),
                entry(17, &self.mid_materials),
                entry(19, &self.job_results),
                entry(20, &self.free_pages),
                entry(21, evictions),
                entry(24, &self.leaf_pages),
                entry(22, baked_refs),
                entry(23, baked_cells),
                entry(28, &self.cell_materials),
                entry(29, &self.detail_pages),
                entry(30, &self.detail_cells),
                entry(31, &self.detail_alloc),
                entry(32, &self.detail_free),
                entry(33, &self.slot_details),
            ],
        })
    }

    /// A generated node's chunk is resident in `slot`.
    /// `kept` held for `key`, the least recently used dropped past the
    /// limit.
    fn keep(&mut self, key: NodeKey, kept: Kept) {
        self.kept_bytes += KEPT_OVERHEAD + kept.pack.len() * 4;
        if let Some(old) = self.kept.insert(key, kept) {
            self.kept_bytes -= KEPT_OVERHEAD + old.pack.len() * 4;
        }
        if self.kept_bytes > self.kept_limit {
            let mut by_use: Vec<(u64, NodeKey)> = self.kept.iter().map(|(k, v)| (v.used, *k)).collect();
            by_use.sort_unstable_by_key(|(used, _)| *used);
            for (_, key) in by_use {
                if self.kept_bytes <= self.kept_limit * 3 / 4 {
                    break;
                }
                self.drop_kept(key);
            }
        }
    }

    /// What was kept of `key` (and a pack of it on its way back) dropped:
    /// an edit is over it, or it was generated again.
    fn drop_kept(&mut self, key: NodeKey) {
        self.packed_since.remove(&key);
        if let Some(old) = self.kept.remove(&key) {
            self.kept_bytes -= KEPT_OVERHEAD + old.pack.len() * 4;
        }
    }

    /// `key` restored from what was kept of it into a new slot (a restore
    /// job of `restores`, its result after the batch's `generated` jobs):
    /// whether it was.
    fn restore(&mut self, key: NodeKey, restores: &mut Restores, generated: usize, evictions: &mut Vec<u32>, stats: &mut UpdateStats) -> bool {
        if generated + restores.jobs.len() >= CHUNK_BATCH || self.free_slots.is_empty() {
            return false;
        }
        let frame = self.frame_now;
        let Some(kept) = self.kept.get_mut(&key) else { return false };
        kept.used = frame;
        if kept.pack.is_empty() {
            // Empty: the plan learns it again, nothing is generated.
            self.events.push(residency::Event::Emptied(key));
            stats.restored_chunks += 1;
            return true;
        }
        let flags = kept.flags;
        let slot = self.free_slots.pop().unwrap();
        restores.jobs.push([slot, restores.data.len() as u32, 0, flags as u32]);
        restores.data.extend_from_slice(&kept.pack);
        self.serial += 1;
        restores.nodes.push((key, slot, self.serial));
        let record = chunk_static(&self.grid, key.footprint(), key.cz, slot, (0.0, 64.0), 32.0, self.exact_level);
        self.queue.write_buffer(&self.statics, u64::from(slot) * std::mem::size_of::<ChunkStatic>() as u64, bytemuck::bytes_of(&record));
        self.queue.write_buffer(&self.chunk_keys, u64::from(slot) * 16, bytemuck::cast_slice(&key.table_key()));
        self.slot_keys[slot as usize] = key.table_key();
        let shown = self.drawn_set.holds(key);
        if let Some(old) = self.insert_resident(key, Resident { slot, serial: self.serial, z: None, shown, edited: true, flags }) {
            evictions.push(old.slot);
        }
        stats.shown_chunks += usize::from(shown);
        stats.restored_chunks += 1;
        self.changed_nodes.insert(key);
        self.hidden.remove(&key);
        self.events.push(residency::Event::Generated(key));
        true
    }

    /// The bind group of pack jobs `packs`, or of restore jobs `restores`
    /// and their packs `data` (`keep_layout`).
    fn keep_group(&self, packs: &[[u32; 4]], restores: &[[u32; 4]], data: &[u32]) -> wgpu::BindGroup {
        let init = |label: &str, contents: &[u8]| self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage: wgpu::BufferUsages::STORAGE });
        let packs = init("chunk pack jobs", bytemuck::cast_slice(packs));
        let restores_buffer = init("chunk restore jobs", bytemuck::cast_slice(restores));
        let data = init("chunk restore packs", bytemuck::cast_slice(if data.is_empty() { &[0u32][..] } else { data }));
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chunk keep"),
            layout: &self.keep_layout,
            entries: &[entry(0, &packs), entry(1, &self.pack_data), entry(2, &self.pack_alloc), entry(3, &restores_buffer), entry(4, &data)],
        })
    }

    fn insert_resident(&mut self, key: NodeKey, resident: Resident) -> Option<Resident> {
        let old = self.nodes.insert(key, resident);
        if old.is_none() {
            self.footprint_nodes.entry(key.footprint()).or_default().push(key.cz);
        }
        old
    }

    fn remove_resident(&mut self, key: NodeKey) -> Option<Resident> {
        let old = self.nodes.remove(&key)?;
        if let Some(list) = self.footprint_nodes.get_mut(&key.footprint()) {
            list.retain(|&cz| cz != key.cz);
            if list.is_empty() {
                self.footprint_nodes.remove(&key.footprint());
            }
        }
        Some(old)
    }

    /// Jobs generating nodes `czs` of `fp` into `batch` (one footprint job,
    /// a chunk job each); their old chunks to `evictions`. How many fit.
    #[allow(clippy::too_many_arguments)]
    fn queue_footprint(&mut self, fp: Footprint, czs: &[i32], batch: &mut GenBatch, readback: &mut Readback, evictions: &mut Vec<u32>, stats: &mut UpdateStats) -> usize {
        if batch.footprints() >= TILE_BATCH || czs.is_empty() {
            return 0;
        }
        let room = CHUNK_BATCH.saturating_sub(batch.chunk_jobs.len() + batch.restored).min(czs.len());
        if self.free_slots.len() < room {
            stats.out_of_slots = true;
        }
        let n = room.min(self.free_slots.len());
        if n == 0 {
            return 0;
        }
        let grid = self.grid;
        let (i, j) = fp.span_half();
        let shift = fp.level + 1;
        let span = |c: i32| (i64::from(c * CHUNK) << shift, (i64::from(c * CHUNK + CHUNK) << shift) - 1);
        let edits = self.edits.clone();
        let baked = edits.bricks.footprints.contains(&fp);
        // Each chunk's brushes (from those over the footprint:
        // `BrushIndex`), and whether edits are over it (its leaves then get
        // per-voxel materials, a `cell_materials` slot of the batch's): the
        // chunks up to the first the batch has no slot for.
        let near = edits.brushes_near(fp.face, i, j, fp.level);
        let mut details = batch.details as usize;
        let mut chunks = Vec::with_capacity(n);
        for &cz in &czs[..n] {
            // Large brushes apply below their bake level (baked cells hold
            // them from it up: `planet::bake_level`).
            let refs: Vec<u32> = near
                .iter()
                .copied()
                .filter(|&b| {
                    let fb = &edits.faces[b as usize];
                    brush_meets(fb, fp.face, fp.level, i, j, span(cz)) && ((b as usize) >= edits.large_faces || fp.level < crate::planet::bake_level(fb))
                })
                .collect();
            let edited = fp.level < DETAIL_LEVELS && (!refs.is_empty() || (baked && GenBatch::bricks_under(&edits.baked, fp, cz)));
            if edited && details >= DETAIL_JOBS {
                break;
            }
            details += usize::from(edited);
            chunks.push((cz, refs, edited));
        }
        if chunks.is_empty() {
            return 0;
        }
        let Some((pool_slot, cached)) = self.pool.take(fp, self.frame_now) else { return 0 };
        let job = TileJob { face: u32::from(fp.face), level: fp.level, cx: fp.cx, cy: fp.cy, slot: pool_slot, _pad: [0; 3] };
        // A footprint already in this batch is in the pool by now (its
        // columns are computed before any chunk reads them).
        let index = match batch.computed.iter().position(|t| t.slot == pool_slot).map(|n| n as u32).or_else(|| batch.cached.iter().position(|t| t.slot == pool_slot).map(|n| n as u32 | CACHED)) {
            Some(index) => index,
            None if cached => {
                batch.cached.push(job);
                stats.cached_footprints += 1;
                (batch.cached.len() as u32 - 1) | CACHED
            }
            None => {
                batch.computed.push(job);
                readback.footprints.push(fp);
                stats.generated_footprints += 1;
                batch.computed.len() as u32 - 1
            }
        };
        let queued = chunks.len();
        for (cz, refs, edited) in chunks {
            let detail = if edited {
                batch.details += 1;
                batch.details - 1
            } else {
                u32::MAX
            };
            let key = fp.node(cz);
            let slot = self.free_slots.pop().unwrap();
            self.serial += 1;
            let first = batch.refs.len() as u32;
            let large = refs.iter().filter(|&&b| (b as usize) < edits.large_faces).count() as u32;
            batch.refs.extend_from_slice(&refs);
            let baked = if baked { batch.baked_table(&mut self.brick_cache, self.frame_now, &edits.baked, fp, cz) } else { u32::MAX };
            let under_edits = !refs.is_empty() || baked != u32::MAX;
            stats.generated_edited += usize::from(under_edits);
            batch.chunk_jobs.push(ChunkJob { tile: index, cz, slot, brushes: first, brush_count: batch.refs.len() as u32 - first, large, baked, detail });
            // Layers not read back yet: the whole chunk.
            let record = chunk_static(&grid, fp, cz, slot, (0.0, 64.0), 32.0, self.exact_level);
            self.queue.write_buffer(&self.statics, u64::from(slot) * std::mem::size_of::<ChunkStatic>() as u64, bytemuck::bytes_of(&record));
            self.queue.write_buffer(&self.chunk_keys, u64::from(slot) * 16, bytemuck::cast_slice(&key.table_key()));
            self.slot_keys[slot as usize] = key.table_key();
            readback.chunks.push((key, slot, self.serial));
            let shown = self.drawn_set.holds(key);
            // Generated again: what was kept of it no longer holds.
            self.drop_kept(key);
            if let Some(old) = self.insert_resident(key, Resident { slot, serial: self.serial, z: None, shown, edited: under_edits, flags: 0 }) {
                evictions.push(old.slot);
            }
            stats.shown_chunks += usize::from(shown);
            self.changed_nodes.insert(key);
            self.hidden.remove(&key);
            self.events.push(residency::Event::Generated(key));
        }
        stats.generated_chunks += queued;
        queued
    }

    /// Follow `planet`'s edits (the recipe unchanged): resident nodes under
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
        let columns: Vec<_> = columns.into_iter().collect();
        // Where the bricks are: unchanged without seals, extended by a
        // seal's columns (bricks are only added), built again after an
        // unseal.
        let bricks = if state.sealed == self.edit_state.sealed {
            self.edits.bricks.clone()
        } else if state.sealed.0 > self.edit_state.sealed.0 {
            std::sync::Arc::new(self.edits.bricks.extended(planet.grid(), &edits.baked, columns.iter().copied()))
        } else {
            std::sync::Arc::new(classify::BrickIndex::new(planet.grid(), &edits.baked))
        };
        let view = edit_view(planet, self.edits.version + 1, Some(&self.edits), bricks);
        // Brushes in one list and not the other: past the lists' common
        // start (the large layer's, at least, but for a seal's own).
        let same = self.edits.faces.iter().zip(&view.faces).take_while(|(a, b)| a == b).count();
        let mut count: FxHashMap<[u32; 12], i32> = FxHashMap::default();
        for fb in &self.edits.faces[same..] {
            *count.entry(bytemuck::cast(*fb)).or_default() -= 1;
        }
        for fb in &view.faces[same..] {
            *count.entry(bytemuck::cast(*fb)).or_default() += 1;
        }
        // Recent brushes sealed into the baked store left the list, not the
        // world (its coarser levels follow from the baked changes).
        let sealed = state.sealed.0.saturating_sub(self.edit_state.sealed.0);
        let sealed_faces: usize = self.edits.recent_counts.iter().take(sealed).sum();
        let large = self.edits.large_faces;
        for fb in &self.edits.faces[large..large + sealed_faces] {
            if let Some(n) = count.get_mut(&bytemuck::cast::<FaceBrush, [u32; 12]>(*fb)) {
                if *n < 0 {
                    *n += 1;
                }
            }
        }
        // Classify against the new edits (drawn uniform nodes a brush's
        // surface crosses).
        let old = std::mem::replace(&mut self.edits, std::sync::Arc::new(view));
        for (words, n) in count {
            if n != 0 {
                self.mark_brush(&bytemuck::cast(words));
            }
        }
        drop(old);
        // Baked changes (a seal's bricks) end what was kept under them.
        let changed: FxHashSet<Footprint> = columns.iter().map(|c| Footprint { face: c.face, level: u32::from(c.level), cx: c.ci >> 3, cy: c.cj >> 3 }).collect();
        if !changed.is_empty() {
            let kept: Vec<NodeKey> = self.kept.keys().chain(&self.packed_since).filter(|k| changed.contains(&k.footprint())).copied().collect();
            for key in kept {
                self.drop_kept(key);
            }
        }
        for column in columns {
            let fp = Footprint { face: column.face, level: u32::from(column.level), cx: column.ci >> 3, cy: column.cj >> 3 };
            for cz in self.footprint_nodes.get(&fp).cloned().unwrap_or_default() {
                let key = fp.node(cz);
                if !self.urgent.contains(&key) {
                    self.stale.insert(key);
                }
            }
        }
        self.write_brushes(same);
        self.edits_unsent = Some(self.edits.clone());
        self.edit_state = state;
        self.ceiling = planet.outer_radius() - planet.grid().radius() + CEILING_MARGIN;
        true
    }

    /// The brush buffer for `self.edits`: its faces from `from` on written
    /// in place, or a larger buffer when they outgrow it.
    fn write_brushes(&mut self, from: usize) {
        let faces = &self.edits.faces;
        if faces.len() > self.brush_capacity {
            (self.brushes, self.brush_capacity) = brush_buffer(&self.device, &self.queue, faces);
        } else if from < faces.len() {
            self.queue.write_buffer(&self.brushes, (from * std::mem::size_of::<FaceBrush>()) as u64, bytemuck::cast_slice(&faces[from..]));
        }
    }

    /// Nodes brush `fb` can change, at every level it applies at: resident
    /// ones in the cut are generated again (urgent), resident ones out of
    /// it are dropped (generated again if wanted), and drawn Solid or
    /// Buried ones its surface crosses are generated now, so a dig shows
    /// the frame it is made.
    fn mark_brush(&mut self, fb: &FaceBrush) {
        let (ci, cj, w) = (i64::from(fb.center[0]), i64::from(fb.center[1]), i64::from(fb.center[3]));
        // Half base cells to footprints and node layers of a level.
        let to = |half: i64, level: u32| (half.div_euclid(2) >> level).div_euclid(i64::from(CHUNK)) as i32;
        let span = |level: u32| (to(ci - w, level)..=to(ci + w, level), to(cj - w, level)..=to(cj + w, level), to(i64::from(fb.k_lo), level)..=to(i64::from(fb.k_hi), level));
        let levels = (0..self.grid.levels()).take_while(|&l| fb.active(l)).count() as u32;
        let inside = |k: &NodeKey| {
            let (x, y, z) = span(k.level);
            k.face == fb.face() && k.level < levels && x.contains(&k.cx) && y.contains(&k.cy) && z.contains(&k.cz)
        };
        let footprints: u64 = (0..levels)
            .map(|l| {
                let (x, y, _) = span(l);
                (x.end() - x.start() + 1) as u64 * (y.end() - y.start() + 1) as u64
            })
            .sum();
        // What was kept of nodes under it no longer holds.
        let kept: Vec<NodeKey> = self.kept.keys().chain(&self.packed_since).filter(|k| inside(k)).copied().collect();
        for key in kept {
            self.drop_kept(key);
        }
        let mut marked: Vec<NodeKey> = Vec::new();
        if footprints > self.footprint_nodes.len() as u64 {
            marked.extend(self.nodes.keys().filter(|k| inside(k)));
        } else {
            for level in 0..levels {
                let (xs, ys, zs) = span(level);
                for cx in xs {
                    for cy in ys.clone() {
                        let fp = Footprint { face: fb.face(), level, cx, cy };
                        for &cz in self.footprint_nodes.get(&fp).into_iter().flatten() {
                            if zs.contains(&cz) {
                                marked.push(fp.node(cz));
                            }
                        }
                    }
                }
            }
        }
        for key in marked {
            self.stale.remove(&key);
            if self.drawn_set.in_cut(key) {
                self.urgent.insert(key);
            } else if let Some(old) = self.remove_resident(key) {
                self.dropped.push(old.slot);
                self.events.push(residency::Event::Stale(key));
            }
        }
        // Drawn uniform nodes the brush's surface crosses become Mixed.
        let uniform: Vec<NodeKey> = self.drawn_set.uniform_nodes().filter(|k| inside(k)).collect();
        for key in uniform {
            let bands = self.edits.surface_bands(fb, key.footprint());
            let (lo, hi) = key.layers();
            if bands.iter().any(|&(a, b)| b > a && a < hi && b > lo) {
                self.urgent.insert(key);
            }
        }
    }

    /// One frame of residency for an eye at `eye` moving at `velocity`,
    /// recorded into `encoder` (submit it before the next update): take
    /// back finished generation results and ray reports, plan, evict, and
    /// generate the most wanted nodes within the budget. [`Self::draw`]
    /// then shows the plan's nodes.
    pub fn update(&mut self, encoder: &mut helio_core::CommandRecorder<'_>, eye: DVec3, velocity: DVec3, view: Option<residency::View>, frame: u64, config: &ResidencyConfig) -> UpdateStats {
        let grid = self.grid;
        let eye_at = Eye::new(&grid, eye);
        self.claim_timing(frame);
        self.frame_now = frame;
        let mut stats = UpdateStats::default();
        if self.worker.is_none() {
            self.worker = Some(residency::Worker::spawn(grid, self.height_layers, config.plan, self.edits.clone()));
            self.edits_unsent = None;
        }
        let mut evictions: Vec<u32> = std::mem::take(&mut self.dropped);

        let mut clock = std::time::Instant::now();
        let mut lap = |stats: &mut UpdateStats, i: usize| {
            stats.cpu_ms[i] = clock.elapsed().as_secs_f64() * 1e3;
            clock = std::time::Instant::now();
        };
        // 1. Finished generation results: measured footprints, occupancy.
        if self.polls_device {
            let _ = self.device.poll(wgpu::PollType::Poll);
        }
        while self.readbacks.front().is_some_and(|r| r.ready.load(std::sync::atomic::Ordering::Acquire)) {
            let r = self.readbacks.pop_front().unwrap();
            {
                let data = r.buffer.slice(..).get_mapped_range().unwrap();
                let words: &[i32] = bytemuck::cast_slice(&data);
                for (n, fp) in r.footprints.iter().enumerate() {
                    let w = &words[n * 4..n * 4 + 4];
                    if w[0] <= w[1] {
                        self.events.push(residency::Event::Measured(*fp, Ranges { volume: (w[0], w[1]), surface: (w[2], w[1]) }));
                    }
                }
                let base = r.footprints.len() * 4;
                for (n, &(key, _, serial)) in r.chunks.iter().enumerate() {
                    let w = &words[base + n * 4..base + n * 4 + 4];
                    if self.nodes.get(&key).is_none_or(|c| c.serial != serial) {
                        continue;
                    }
                    self.changed_nodes.insert(key);
                    if w[3] & 8 != 0 {
                        // Not stored: wanted again once leaves free up.
                        stats.out_of_leaves = true;
                        evictions.push(self.remove_resident(key).unwrap().slot);
                        self.events.push(residency::Event::Stale(key));
                    } else if w[3] & 2 != 0 {
                        // Nothing to draw. (A chunk solid throughout stays:
                        // its faces are the surface where it lies on a
                        // chunk boundary.)
                        let old = self.remove_resident(key).unwrap();
                        // An edited one is remembered empty (generating it
                        // again would read the edits over it again).
                        if old.edited && self.kept_limit > 0 {
                            self.keep(key, Kept { pack: Vec::new().into(), flags: w[3], used: frame });
                        }
                        evictions.push(old.slot);
                        self.events.push(residency::Event::Emptied(key));
                    } else {
                        // Out of detail pages: its leaves' pairs show.
                        stats.out_of_details |= w[3] & 16 != 0;
                        let c = self.nodes.get_mut(&key).unwrap();
                        c.z = Some((w[1], w[2]));
                        c.flags = w[3];
                        // Its frame about its columns' mean top (its ground),
                        // else its occupied layers.
                        let surface = (w[3] as u32) >> 16;
                        let about = if surface == 0xffff { (w[1] + w[2]) as f32 * 0.5 } else { surface as f32 / 4.0 };
                        let record = chunk_static(&self.grid, key.footprint(), key.cz, c.slot, (w[1] as f32, w[2] as f32), about, self.exact_level);
                        self.queue.write_buffer(&self.statics, u64::from(c.slot) * std::mem::size_of::<ChunkStatic>() as u64, bytemuck::bytes_of(&record));
                    }
                }
                let at = base + r.chunks.len() * 4;
                if let Some(&free) = words.get(at) {
                    self.free_pages_seen = free.max(0) as u64;
                }
                if let Some(&free) = words.get(at + 2) {
                    self.detail_free_seen = free.max(0) as u64;
                }
            }
            r.buffer.unmap();
        }
        while self.pack_readbacks.front().is_some_and(|r| r.ready.load(std::sync::atomic::Ordering::Acquire)) {
            let r = self.pack_readbacks.pop_front().unwrap();
            {
                let data = r.buffer.slice(..).get_mapped_range().unwrap();
                let words: &[u32] = bytemuck::cast_slice(&data);
                let data_at = 1 + 2 * MAX_PACKS;
                for (n, &(key, flags, serial)) in r.nodes.iter().enumerate() {
                    let (offset, len) = (words[1 + 2 * n], words[2 + 2 * n] as usize);
                    // Not packed (out of room), or edited since, or wanted
                    // again (evicted) or generated again (resident) since.
                    let current = match serial {
                        None => !self.nodes.contains_key(&key),
                        Some(serial) => self.nodes.get(&key).is_some_and(|c| c.serial == serial),
                    };
                    if !self.packed_since.remove(&key) || !current {
                        continue;
                    }
                    if offset == u32::MAX {
                        // Out of room: a resident one is packed again later.
                        if serial.is_some() {
                            self.to_pack.push(key);
                        }
                        continue;
                    }
                    let at = data_at + offset as usize;
                    self.keep(key, Kept { pack: words[at..at + len].into(), flags, used: frame });
                }
            }
            r.buffer.unmap();
        }
        // Ray reports: nodes met, and Buried nodes met (generate them).
        stats.seen = self.seen.len();
        stats.reached = self.reached.len();
        for key in std::mem::take(&mut self.seen) {
            self.events.push(residency::Event::Seen(key));
        }
        for key in std::mem::take(&mut self.reached) {
            self.events.push(residency::Event::Reached(key));
        }
        lap(&mut stats, 0);
        // 2. The latest plan (made on the worker from an eye a frame or two
        // old) and its evictions; then the next plan, with what slots
        // need freeing.
        let (mut added, mut removed) = (Vec::new(), Vec::new());
        let mut evicted = Vec::new();
        if let Some(planned) = self.worker.as_mut().expect("spawned above").poll() {
            self.plan = planned.plan;
            stats.switches = self.plan.switches;
            stats.switch_px = self.plan.switch_px;
            evicted = planned.evicted;
            added = planned.added;
            removed = planned.removed;
            self.planned_for = Some(planned.made_for);
            self.plan_made = (planned.frame, planned.ms);
        }
        stats.plan_age = frame.saturating_sub(self.plan_made.0);
        stats.plan_ms = self.plan_made.1;
        let mut packs: Vec<(NodeKey, u32, i32, Option<u64>)> = Vec::new();
        for key in evicted {
            // An edited chunk to keep past this frame's packs stays resident
            // (the planner told so: it is evicted again, packed, by a later
            // plan) rather than lost to generating again from the history.
            // An edited chunk, read back and unedited since, is kept.
            let keep = self.kept_limit > 0 && self.nodes.get(&key).is_some_and(|c| c.edited && c.z.is_some()) && !self.kept.contains_key(&key) && !self.urgent.contains(&key) && !self.stale.contains(&key);
            if keep && packs.len() >= MAX_PACKS {
                self.events.push(residency::Event::Generated(key));
                stats.evictions_deferred += 1;
                continue;
            }
            if let Some(old) = self.remove_resident(key) {
                if keep {
                    packs.push((key, old.slot, old.flags, None));
                    self.packed_since.insert(key);
                }
                evictions.push(old.slot);
                stats.evicted_unseen += usize::from(!old.shown);
                stats.evicted_chunks += 1;
            }
            self.hidden.remove(&key);
            self.urgent.remove(&key);
            self.stale.remove(&key);
            self.changed_nodes.insert(key);
        }
        // Resident edited chunks packed on demand (a save), as room allows.
        while packs.len() < MAX_PACKS {
            let Some(key) = self.to_pack.pop() else { break };
            let Some(c) = self.nodes.get(&key) else { continue };
            if c.edited && c.z.is_some() && !self.kept.contains_key(&key) && !self.packed_since.contains(&key) && !self.urgent.contains(&key) && !self.stale.contains(&key) {
                packs.push((key, c.slot, c.flags, Some(c.serial)));
                self.packed_since.insert(key);
            }
        }
        // Slots and leaf pages kept free for a frame's generation and
        // more: the least recently visited nodes go when either runs short.
        let reserve = (self.max_chunks / 8).max(2 * CHUNK_BATCH);
        let mut need = reserve.saturating_sub(self.free_slots.len() + evictions.len());
        let pages = self.leaf_capacity / LEAF_PAGE;
        if self.free_pages_seen < pages / 8 && !self.nodes.is_empty() {
            let per_node = ((pages - self.free_pages_seen) as f64 / self.nodes.len() as f64).max(1.0);
            need = need.max(((pages / 8 - self.free_pages_seen) as f64 / per_node).ceil() as usize);
        }
        let worker = self.worker.as_mut().expect("spawned above");
        let unseen = self.events.iter().any(residency::Event::structural) || self.planned_for.is_none_or(|p| p.events_seen != worker.events_sent() || p.eye != eye || p.view != view);
        worker.submit(eye, velocity, view, frame, &mut self.events, &mut self.reports_unsent, need, &mut self.edits_unsent);
        lap(&mut stats, 1);
        let plan = &self.plan;
        stats.requests = plan.waiting;
        stats.coarse_share = plan.coarse_share;
        stats.worst_px = plan.worst_px;
        stats.worst_wanted = plan.worst_wanted;
        stats.worst = plan.worst;
        lap(&mut stats, 2);

        // 3. Generate: nodes edits changed (those brushes changed, all now;
        // baked edits, within the budget), then the most wanted.
        let mut batch = GenBatch::default();
        let mut readback = Readback {
            buffer: self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("chunk results"), size: 16, usage: wgpu::BufferUsages::MAP_READ, mapped_at_creation: false }),
            ready: Default::default(),
            footprints: Vec::new(),
            chunks: Vec::new(),
        };
        // The frame's generation: `footprints_per_frame` footprints at most
        // and, with a time budget, the measured cost of their columns and
        // chunks within it (eight footprints at least).
        let cpt = self.cost.chunks_per_tile;
        let cost = config.generate_ms.map(|ms| (ms, self.cost.model()));
        let most = config.footprints_per_frame.min(TILE_BATCH);
        stats.budget = cost.map_or(most * 2, |(ms, (fixed, tile, chunk))| (((ms - fixed).max(0.0) / (tile / cpt.max(1.0) + chunk)) as usize).max(8));
        let full = |batch: &GenBatch| {
            let tiles = batch.computed.len();
            tiles >= most || batch.footprints() >= TILE_BATCH || batch.chunk_jobs.len() >= CHUNK_BATCH || cost.is_some_and(|(ms, (fixed, tile, chunk))| (tiles >= 8 || batch.chunk_jobs.len() >= 8) && fixed + tiles as f64 * tile + batch.chunk_jobs.len() as f64 * chunk >= ms)
        };
        // Nodes edits changed: coarsest first, then drawn ones, nearest
        // first, within the frame's budget. A blast shows at once in coarse
        // nodes; those left for later frames are hidden meanwhile (their
        // nearest generated ancestor drawn in their place) and refine as
        // they come. Then stale nodes.
        let mut edited: Vec<(NodeKey, bool, f64)> = Vec::with_capacity(self.urgent.len());
        for k in self.urgent.drain() {
            let distance = *self.urgent_distance.entry(k).or_insert_with(|| tile_distance(&grid, k.footprint(), k.layers(), &eye_at));
            edited.push((k, !self.drawn_set.holds(k), distance));
        }

        edited.sort_unstable_by(|a, b| b.0.level.cmp(&a.0.level).then(a.1.cmp(&b.1)).then(a.2.total_cmp(&b.2)));
        let edited: Vec<(NodeKey, bool)> = edited.into_iter().map(|(k, ..)| (k, true)).chain(self.stale.iter().map(|k| (*k, false))).collect();
        for (key, urgent) in edited {
            if full(&batch) || stats.out_of_slots {
                if urgent {
                    self.urgent.insert(key);
                    continue;
                }
                break;
            }
            self.stale.remove(&key);
            if self.queue_footprint(key.footprint(), &[key.cz], &mut batch, &mut readback, &mut evictions, &mut stats) == 0 {
                if urgent {
                    self.urgent.insert(key);
                } else {
                    self.stale.insert(key);
                }
            } else if urgent && self.drawn_set.kind(key).is_some_and(|k| k != Kind::Mixed) {
                // A drawn uniform node an edit made Mixed: its chunk is
                // drawn now, before the planner says so.
                added.push((key, Kind::Mixed));
                removed.push(key);
            }
        }
        for &key in &self.urgent {
            if self.hidden.insert(key) {
                self.events.push(residency::Event::Stale(key));
            }
        }
        let urgent = &self.urgent;
        self.urgent_distance.retain(|k, _| urgent.contains(k));
        // The plan's requests, a footprint's nodes together.
        let mut restores = Restores::default();
        let requests = std::mem::take(&mut self.plan.requests);
        let mut at = 0;
        while at < requests.len() && !full(&batch) {
            let fp = requests[at].footprint();
            let end = requests[at..].iter().position(|k| k.footprint() != fp).map_or(requests.len(), |n| at + n);
            // Generated since that plan.
            let mut czs: Vec<i32> = requests[at..end].iter().filter(|k| !self.nodes.contains_key(k)).map(|k| k.cz).collect();
            // Kept nodes come back by copy.
            czs.retain(|&cz| !self.restore(fp.node(cz), &mut restores, batch.chunk_jobs.len(), &mut evictions, &mut stats));
            batch.restored = restores.jobs.len();
            if !czs.is_empty() && self.queue_footprint(fp, &czs, &mut batch, &mut readback, &mut evictions, &mut stats) < czs.len() {
                break;
            }
            at = end;
        }
        self.plan.requests = requests;
        let GenBatch { computed, cached, mut chunk_jobs, refs, baked_refs, .. } = batch;
        // Restored chunks' results follow the generated ones'.
        let results_at = chunk_jobs.len() as u32;
        for (n, job) in restores.jobs.iter_mut().enumerate() {
            job[2] = results_at + n as u32;
        }
        readback.chunks.extend(restores.nodes.iter().copied());
        // The batch's footprints: those it computes first (the kernels run
        // over them), then those it reads from the pool.
        let computed_count = computed.len();
        for job in &mut chunk_jobs {
            if job.tile & CACHED != 0 {
                job.tile = computed_count as u32 + (job.tile & !CACHED);
            }
        }
        let tile_jobs: Vec<TileJob> = computed.into_iter().chain(cached).collect();

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
        self.brick_cache.upload(&self.device, &self.queue, encoder);
        let pack_group = (!packs.is_empty()).then(|| {
            let jobs: Vec<[u32; 4]> = packs.iter().map(|&(_, slot, ..)| [slot, 0, 0, 0]).collect();
            self.queue.write_buffer(&self.pack_alloc, 0, &vec![0u8; (1 + 2 * MAX_PACKS) * 4]);
            self.keep_group(&jobs, &[[0; 4]], &[0])
        });
        let restore_group = (!restores.jobs.is_empty()).then(|| self.keep_group(&[[0; 4]], &restores.jobs, &restores.data));
        let group = self.gen_group(&refs_buffer, &evictions_buffer, &baked_refs_buffer, &self.brick_cache.buffer);
        if !tile_jobs.is_empty() {
            self.queue.write_buffer(&self.tile_jobs, 0, bytemuck::cast_slice(&tile_jobs));
            let empty: Vec<[i32; 4]> = vec![[i32::MAX, i32::MIN + 1, i32::MAX, i32::MIN + 1]; computed_count.max(1)];
            self.queue.write_buffer(&self.tile_ranges, 0, bytemuck::cast_slice(&empty));
            self.queue.write_buffer(&self.tile_ranges, (TILE_BATCH * 16) as u64, bytemuck::bytes_of(&(computed_count as i32)));
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
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("chunk tile nodes"), timestamp_writes: stamp(Some(0), 8) });
            pass.set_bind_group(0, &group, &[]);
            if let Some(keep) = &pack_group {
                pass.set_pipeline(&self.pack_pipeline);
                pass.set_bind_group(1, keep, &[]);
                pass.dispatch_workgroups(packs.len() as u32, 1, 1);
            }
            if !evictions.is_empty() {
                pass.set_pipeline(&self.evict_pipeline);
                pass.dispatch_workgroups((evictions.len() as u32).div_ceil(64), 1, 1);
            }
            if computed_count > 0 {
                pass.set_pipeline(&self.tile_nodes);
                pass.dispatch_workgroups(computed_count as u32, 1, 1);
                pass.set_pipeline(&self.tile_slope_nodes);
                pass.dispatch_workgroups(computed_count as u32, 1, 1);
            }
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("chunk tile columns"), timestamp_writes: stamp(None, 9) });
            pass.set_bind_group(0, &group, &[]);
            if computed_count > 0 {
                pass.set_pipeline(&self.tile_columns);
                pass.dispatch_workgroups(64, computed_count as u32, 1);
                pass.set_pipeline(&self.tile_slope_heights);
                pass.dispatch_workgroups((SLOPE_LATTICE as u32).div_ceil(64), computed_count as u32, 1);
                pass.set_pipeline(&self.tile_slopes);
                pass.dispatch_workgroups((SLOPE_STRIDE as u32).div_ceil(64), computed_count as u32, 1);
            }
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("chunk tile halos"), timestamp_writes: stamp(None, 1) });
            pass.set_bind_group(0, &group, &[]);
            if computed_count > 0 {
                pass.set_pipeline(&self.tile_halo);
                pass.dispatch_workgroups((computed_count as u32).min(crate::engine::HALO_GROUPS), 1, 1);
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
            if let Some(keep) = &restore_group {
                pass.set_pipeline(&self.restore_pipeline);
                pass.set_bind_group(1, keep, &[]);
                pass.dispatch_workgroups(restores.jobs.len() as u32, 1, 1);
            }
        }
        // The packs back (mapped with the results).
        if !packs.is_empty() {
            let directory = (1 + 2 * MAX_PACKS as u64) * 4;
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("chunk packs read"), size: directory + PACK_WORDS * 4, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            encoder.copy_buffer_to_buffer(&self.pack_alloc, 0, &buffer, 0, directory);
            encoder.copy_buffer_to_buffer(&self.pack_data, 0, &buffer, directory, PACK_WORDS * 4);
            self.pack_pending = Some(PackReadback { buffer, ready: Default::default(), nodes: packs.iter().map(|&(key, _, flags, serial)| (key, flags, serial)).collect() });
        }
        // Results back: footprints' layers, each chunk's job result, the
        // free leaf pages.
        if !tile_jobs.is_empty() || !restores.jobs.is_empty() {
            let tiles = (computed_count * 16) as u64;
            let chunks = ((chunk_jobs.len() + restores.jobs.len()) * 16) as u64;
            readback.buffer = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("chunk results"), size: tiles + chunks + 16, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            if tiles > 0 {
                encoder.copy_buffer_to_buffer(&self.tile_ranges, 0, &readback.buffer, 0, tiles);
            }
            if chunks > 0 {
                encoder.copy_buffer_to_buffer(&self.job_results, 0, &readback.buffer, tiles, chunks);
            }
            encoder.copy_buffer_to_buffer(&self.leaf_alloc, 0, &readback.buffer, tiles + chunks, 8);
            encoder.copy_buffer_to_buffer(&self.detail_alloc, 0, &readback.buffer, tiles + chunks + 8, 8);
            self.pending = Some(readback);
        }
        // Slots evicted now are reused from the next frame.
        self.free_slots.extend(evictions);

        stats.kept_chunks = self.kept.len();
        stats.kept_bytes = self.kept_bytes;
        lap(&mut stats, 3);
        // 4. This frame's drawn nodes and the table entries that changed.
        let nodes = &self.nodes;
        let chunk_of = |key: NodeKey| nodes.get(&key).map(|c| drawn::NodeChunk { slot: c.slot, z: c.z.map_or((0.0, 64.0), |(a, b)| (f64::from(a), f64::from(b))) });
        self.drawn_set.update(&added, &removed, &self.changed_nodes, frame, chunk_of);
        for (key, _) in &added {
            if let Some(c) = self.nodes.get_mut(key) {
                stats.shown_chunks += usize::from(!c.shown);
                c.shown = true;
            }
        }
        self.changed_nodes.clear();
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
        stats.drawn_nodes = self.drawn_set.node_count();
        stats.fading_nodes = self.drawn_set.fading();
        self.last_generated = (stats.generated_footprints, stats.generated_chunks);
        // Settled: a plan that saw every event wants nothing more.
        self.last_requests = self.plan.requests.iter().filter(|k| !self.nodes.contains_key(k)).count() + usize::from(unseen || self.events.iter().any(residency::Event::structural) || stats.generated_chunks > 0);
        if self.last_requests > 0 {
            self.quiet_from = self.worker.as_ref().map_or(0, residency::Worker::reports_sent);
        }
        stats.drawn_chunks = self.drawn_set.list.len();
        stats.detail_pages = (self.detail_capacity - self.detail_free_seen.min(self.detail_capacity), self.detail_capacity);
        stats.resident_chunks = self.nodes.len();
        stats
    }

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
            material: (cell >> 21) & 31,
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
        // Ray reports: Buried nodes met, and the nodes the sampled hits
        // want (met, with their ancestors).
        let reached = &mut self.reached;
        take_reports(&mut self.buried_slots, |data| {
            let words: &[[i32; 4]] = bytemuck::cast_slice(data);
            let count = (words[0][0] as u32 as usize).min(words.len() - 1);
            reached.extend(words[1..=count].iter().map(|k| NodeKey::from_table_key(*k)));
        });
        let (seen, reports) = (&mut self.seen, &mut self.reports_unsent);
        take_reports(&mut self.feedback_slots, |data| {
            *reports += 1;
            let words: &[[i32; 4]] = bytemuck::cast_slice(data);
            let mut unique: FxHashSet<[i32; 4]> = FxHashSet::default();
            seen.extend(words.iter().filter(|k| k[0] >= 0 && unique.insert(**k)).map(|k| NodeKey::from_table_key(*k)));
        });
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
                    let tile_stages = [ms(t[0], t[8]), ms(t[8], t[9]), ms(t[9], t[1])];
                    let timing = FrameTimings { frame, draw: ms(t[2], t[4]), seed: ms(t[2], t[3]), shadow: ms(t[4], t[7]), generate: ms(t[0], t[6]), tiles, chunks, stages, tile_stages };
                    // Footprints computed and chunks generated are measured
                    // apart: with the footprint pool a frame may generate
                    // chunks of footprints it does not compute.
                    let mix = |old: f64, new: f64| old * 0.9 + new * 0.1;
                    if tiles > 0 {
                        self.cost.tiles.add(tiles as f64, stages[0]);
                    }
                    if chunks > 0 {
                        self.cost.chunks.add(chunks as f64, stages[1] + stages[2]);
                    }
                    if tiles >= 8 {
                        self.cost.tile = mix(self.cost.tile, stages[0] / tiles as f64);
                        self.cost.chunks_per_tile = mix(self.cost.chunks_per_tile, chunks as f64 / tiles as f64);
                    }
                    if chunks >= 8 {
                        self.cost.chunk = mix(self.cost.chunk, (stages[1] + stages[2]) / chunks as f64);
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
        if self.polls_device {
            let _ = self.device.poll(wgpu::PollType::Poll);
        }
        if let Some(r) = self.pending.take() {
            let ready = r.ready.clone();
            r.buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    ready.store(true, std::sync::atomic::Ordering::Release);
                }
            });
            self.readbacks.push_back(r);
        }
        if let Some(r) = self.pack_pending.take() {
            let ready = r.ready.clone();
            r.buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    ready.store(true, std::sync::atomic::Ordering::Release);
                }
            });
            self.pack_readbacks.push_back(r);
        }
    }

    /// Whether nothing waits to be generated or read back, no switch
    /// between levels is fading (fades move with drawn frames), and the
    /// rays have swept the view since residency last changed.
    pub fn settled(&self) -> bool {
        // Rays sample a pixel of each block a frame: a plan that has not
        // seen them all since residency last changed may still want more
        // where rays meet what the eye's distance does not ask for.
        let swept = self.targets.is_none() || self.planned_for.is_some_and(|p| p.reports_seen >= self.quiet_from + FEEDBACK_FRAMES);
        self.pending.is_none() && self.readbacks.is_empty() && self.last_requests == 0 && self.urgent.is_empty() && self.stale.is_empty() && self.drawn_set.fading() == 0 && swept
    }

    /// Node and occupied layers of the chunk in `slot`.
    pub fn chunk_at(&self, slot: u32) -> Option<(NodeKey, Option<(i32, i32)>)> {
        self.nodes.iter().find(|(_, c)| c.slot == slot).map(|(k, c)| (*k, c.z))
    }

    /// Resident chunks over base column (`face`, `i`, `j`): node, occupied
    /// layers, drawn.
    pub fn chunks_over(&self, face: u8, i: i32, j: i32) -> Vec<(NodeKey, Option<(i32, i32)>, bool)> {
        let mut out: Vec<_> = self
            .nodes
            .iter()
            .filter(|(k, _)| k.face == face && (i >> k.level).div_euclid(CHUNK) == k.cx && (j >> k.level).div_euclid(CHUNK) == k.cy)
            .map(|(k, c)| (*k, c.z, self.drawn_set.contains(c.slot)))
            .collect();
        out.sort_by_key(|c| (c.0.level, c.0.cz));
        out
    }

    /// `node`'s drawn state (diagnostics, `DrawnSet::describe`).
    pub fn describe_drawn(&self, node: NodeKey) -> (bool, Option<i32>, Option<(u64, bool, bool)>) {
        self.drawn_set.describe(node)
    }

    /// Whether to poll the device for readbacks (false on a host's
    /// shared device).
    pub fn set_polls_device(&mut self, polls: bool) {
        self.polls_device = polls;
    }

    /// Whether draws write the plain colour target ([`Targets::colour`]).
    pub fn set_plain_colour(&mut self, on: bool) {
        self.plain_colour = on;
    }

    /// The renderer-wide debug mode for this voxel world.
    pub fn set_debug_mode(&mut self, mode: u32) {
        self.debug_mode = mode & 0xff;
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
        let chunks = self.nodes.len();
        let leaves = (self.leaf_capacity / LEAF_PAGE).saturating_sub(alloc[0][0].max(0) as u64) * LEAF_PAGE;
        leaves * 10 + chunks as u64 * (64 * 24 + 48 + LEAF_PAGES * 4)
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
                    (if std::env::var_os("HELIO_VOXEL_CHUNKS_HEAT").is_some() {
                        0x200
                    } else {
                        std::env::var("HELIO_VOXEL_CHUNKS_LEVELS").map_or(0, |v| if v == "half" { 0xc00 } else { 0x400 })
                    }) | self.debug_mode | (u32::from(self.plain_colour) << 8),
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
            cut: [self.grid.levels() - 1, self.drawn_set.fading() as u32, 0, 0],
        };
        self.queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        self.queue.write_buffer(&self.buried, 0, &[0u8; 16]);
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chunk draw"),
            layout: &self.draw_layout,
            entries: &[entry(0, &self.params), entry(1, &self.statics), entry(2, &self.chunk_meta), entry(3, &self.mids), entry(4, &self.mid_leaves), entry(5, &self.leaves), entry(6, &self.leaf_materials), entry(7, &self.materials), entry(8, &self.mid_materials), entry(9, &self.counters), entry(10, &self.chunk_table), entry(11, &self.chunk_keys), entry(12, &self.draw_list), entry(13, &self.buried), entry(14, &self.leaf_pages), entry(15, &self.leaf_normals), entry(16, &self.detail_pages), entry(17, &self.detail_cells), entry(18, &self.slot_details)],
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
                view_entry(25, &self.uv_grid_view),
                wgpu::BindGroupEntry { binding: 26, resource: wgpu::BindingResource::Sampler(&self.uv_grid_sampler) },
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
                        depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(helio_core::depth::CLEAR), store: wgpu::StoreOp::Store }),
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
        // Ray reports, back without waiting (a frame's copy when a slot is
        // free): the node each block of pixels' sampled hit wants, and the
        // Buried nodes rays met.
        let blocks = (u64::from(size[0].div_ceil(FEEDBACK_BLOCK)) * u64::from(size[1].div_ceil(FEEDBACK_BLOCK))).min(FEEDBACK_BYTES / 16);
        if let Some(slot) = self.feedback_slots.iter_mut().find(|s| matches!(s.state, ReachState::Free)) {
            let (distance, visibility) = (view(&targets.distance), view(&targets.visibility));
            let feedback_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("chunk feedback"),
                layout: &self.feedback_layout,
                entries: &[view_entry(4, &distance), view_entry(7, &visibility), entry(24, &self.feedback)],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("chunk feedback"), timestamp_writes: None });
                pass.set_pipeline(&self.feedback_pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.set_bind_group(1, &feedback_group, &[]);
                pass.dispatch_workgroups(size[0].div_ceil(FEEDBACK_BLOCK).div_ceil(8), size[1].div_ceil(FEEDBACK_BLOCK).div_ceil(8), 1);
            }
            slot.bytes = blocks * 16;
            encoder.copy_buffer_to_buffer(&self.feedback, 0, &slot.buffer, 0, slot.bytes);
            slot.state = ReachState::Copied;
        }
        if let Some(slot) = self.buried_slots.iter_mut().find(|s| matches!(s.state, ReachState::Free)) {
            encoder.copy_buffer_to_buffer(&self.buried, 0, &slot.buffer, 0, BURIED_BYTES);
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
        let words: Vec<[u32; 4]> = self.read(&self.counters, 4);
        self.queue.write_buffer(&self.counters, 0, &[0u8; 1024]);
        TraceCounts { seeded: words[0][0], steps: words[0][1], hits: words[0][2], boxes: words[0][3], missed: words[2][0], missed_boxes: words[2][1], lost: words[2][2], lost_pixel: (words[3][2] & 0xffff, words[3][2] >> 16), lost_sites: [words[1][1], words[1][2], words[1][3], words[2][3], words[3][0], words[3][1], words[3][3]] }
    }

    /// The boxes the debug pixel's ray crossed (counting builds): the frame
    /// chunk's slot, the box's low corner in its cells, whether it is that
    /// chunk, its level; and its key and layer.
    pub fn pixel_trace(&self) -> Vec<(u32, [i32; 3], bool, u32, Option<NodeKey>)> {
        let words: Vec<[u32; 4]> = self.read(&self.counters, 64);
        let flat: Vec<u32> = words.iter().flatten().copied().collect();
        let n = flat[4] as usize;
        (0..n.min(24))
            .map(|c| {
                let w = &flat[16 + c * 5..21 + c * 5];
                let slot = w[0];
                let owner = self.nodes.iter().find(|(_, c)| c.slot == slot).map(|(k, _)| *k);
                (slot, [w[1] as i32, w[2] as i32, w[3] as i32], w[4] & 1 != 0, w[4] >> 1, owner)
            })
            .collect()
    }

    /// Where the last search past the top began (counting builds): face,
    /// level and cell. Read before `take_counts` (which clears it).
    pub fn lost_where(&self) -> (u8, u32, [i32; 3]) {
        let words: Vec<[u32; 4]> = self.read(&self.counters, 64);
        let w = words[50];
        ((w[0] & 0xff) as u8, w[0] >> 8, [w[1] as i32, w[2] as i32, w[3] as i32])
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
                view_entry(25, &self.uv_grid_view),
                wgpu::BindGroupEntry { binding: 26, resource: wgpu::BindingResource::Sampler(&self.uv_grid_sampler) },
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

    /// Pack every resident edited chunk not kept yet (over the next
    /// updates, as many a frame as packs fit), so [`Self::kept_chunks`]
    /// holds the whole world's edited chunks: before saving.
    pub fn keep_resident(&mut self) {
        let keys: Vec<NodeKey> = self.nodes.iter().filter(|(k, c)| c.edited && !self.kept.contains_key(k)).map(|(k, _)| *k).collect();
        self.to_pack = keys;
    }

    /// Edited chunks [`Self::keep_resident`] still has to pack or read back.
    pub fn keeping(&self) -> usize {
        self.to_pack.len() + self.packed_since.len()
    }

    /// The kept chunks, for `planet`'s edits as this world follows them
    /// (after [`Self::keep_resident`]: the whole world's edited chunks):
    /// what a save stores ([`KeptChunks::to_bytes`]). Shares the packs.
    pub fn kept_chunks(&self, planet: &Planet) -> KeptChunks {
        let edits = planet.edits();
        let mut chunks: Vec<_> = self.kept.iter().map(|(key, kept)| (*key, kept.flags, kept.pack.clone())).collect();
        chunks.sort_unstable_by_key(|(k, ..)| (k.face, k.level, k.cx, k.cy, k.cz));
        KeptChunks { fingerprint: self.fingerprint, history: (edits.len() as u64, edits.hash()), chunks }
    }

    /// `kept`'s chunks kept, restored by copy when wanted: how many. Before
    /// the world's first update; refused when they were made by other
    /// generation code, terrain or world, or for another history than
    /// `planet`'s.
    pub fn load_kept(&mut self, planet: &Planet, kept: &KeptChunks) -> Result<usize, String> {
        if kept.fingerprint != self.fingerprint {
            return Err("kept chunks of other generation code, terrain or world".into());
        }
        let edits = planet.edits();
        if kept.history != (edits.len() as u64, edits.hash()) {
            return Err("kept chunks of another history".into());
        }
        let frame = self.frame_now;
        for (key, flags, pack) in &kept.chunks {
            self.keep(*key, Kept { pack: pack.clone(), flags: *flags, used: frame });
        }
        Ok(kept.chunks.len())
    }

    /// Every resident node generated again (benchmarks: the same work
    /// each time), drawn as it is meanwhile.
    pub fn regenerate_all(&mut self) {
        let keys: Vec<NodeKey> = self.nodes.keys().copied().collect();
        self.stale.extend(keys);
    }
}

#[cfg(test)]
mod tests {
    use super::{walk_templates, CostLine, GenCost};

    /// Both walks come out of the templates whatever ends the lines (a
    /// checkout's \r\n lost the fine ones).
    /// A batch's fixed part apart from its tiles': batches of spread sizes
    /// find both; batches of one size find neither (no line).
    #[test]
    fn generation_cost_separates_a_batch_from_its_tiles() {
        let mut line = CostLine::EMPTY;
        for n in 0..200 {
            let tiles = 8 + (n * 37) % 120;
            line.add(tiles as f64, 0.2 + 0.01 * tiles as f64);
        }
        let (fixed, per) = line.fit().unwrap();
        assert!((fixed - 0.2).abs() < 1e-6 && (per - 0.01).abs() < 1e-8, "{fixed} {per}");
        let mut same = CostLine::EMPTY;
        for _ in 0..50 {
            same.add(14.0, 0.34);
        }
        assert!(same.fit().is_none());
        // Its budget: a 3 ms frame holds 280 such tiles, not the 75 their
        // small batches' average (0.04 ms) would allow.
        let cost = GenCost { tiles: line, chunks: CostLine::EMPTY, chunk: 0.0, ..GenCost::PRIOR };
        let (fixed, tile, _) = cost.model();
        assert_eq!(((3.0 - fixed) / tile).round(), 280.0);
    }

    /// Both shaders parse and validate as naga sees them, for spheres and
    /// planes, plain and counting builds (no GPU needed).
    #[test]
    fn chunk_shaders_validate() {
        let planet = crate::Planet::new(crate::PlanetRecipe::default()).unwrap();
        let program = planet.field().program();
        for plane in [false, true] {
            for counting in [false, true] {
                for (label, source) in super::shader_sources(plane, &program, false, counting) {
                    let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(&source)));
                    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap_or_else(|e| panic!("{label} (plane {plane}, counting {counting}): {}", e.emit_to_string(&source)));
                }
            }
        }
    }

    #[test]
    fn walk_templates_expand_with_any_line_ends() {
        for end in ["\n", "\r\n"] {
            let source = ["head", "// WALK {", "fn sun_walk() { walk(Walk(0)); }", "// } WALK", "tail", ""].join(end);
            let out = walk_templates(&source);
            assert!(out.contains("fn sun_walk() { walk(Walk(0)); }"), "{out:?}");
            assert!(out.contains("fn sun_walk_fine() { walk_fine(Walk(0)); }"), "{out:?}");
            assert!(!out.contains("WALK {") && out.starts_with("head") && out.trim_end().ends_with("tail"), "{out:?}");
        }
    }
}

