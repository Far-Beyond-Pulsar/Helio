// Chunk trees (docs/chunk-trees.md): generation.
//
// A tile is the 64x64 columns of one level under a stack of chunks; a chunk
// is 64^3 cells of one level, stored as a 64-tree: a root over 4^3 mid
// nodes of 16^3 cells, mid nodes over 4^3 leaves of 4^3 cells, leaves a
// 64-bit occupancy mask. Bit `x + 4 y + 16 z` of a node or leaf mask is its
// child or cell (x, y, z). Uniform children (all solid) are flagged in a
// second mask and store nothing.
//
// 1. `tile_columns`: each column's height, surface word and volume extent,
//    and the tile's range of layers that may hold a surface.
// 2. `chunk_columns`: each column's 64 cells in a chunk as a 64-bit mask
//    (bit z: layer `64 cz + z`), with volume terms and brushes, and the
//    material of its highest solid cell.
// 3. `chunk_build`: the chunk's 64-tree from its column masks, its leaves
//    in a block of a size class (16 << class leaves) taken from that
//    class's free blocks or the pool's end.
// 4. `evict`: evicted chunks' blocks back to their class's free blocks.

const TILE_COLUMNS: u32 = 4096u;
// Height of a tile column past the face's edge.
const OUTSIDE: i32 = -2147483647 - 1;

struct TileJob {
    face: u32,
    level: u32,
    cx: i32,
    cy: i32,
}

struct ChunkJob {
    tile: u32,  // index into tile_jobs
    cz: i32,
    slot: u32,  // chunk slot of the result
    brushes: u32, // first brush ref; count in `brush_count`
    brush_count: u32,
    // Refs of the large log (they come first), then the recent log's.
    large: u32,
    // Its baked bricks: 8 per brick column (`baked_refs`), or NONE.
    baked: u32,
    _pad: u32,
}

struct FaceBrush {
    flags: u32,
    radius_half: u32,
    k_lo: i32,
    k_hi: i32,
    center: vec4<i32>,
    ball: vec4<i32>,
}

struct ChunkMeta {
    root_occ: vec2<u32>,
    root_full: vec2<u32>,
    // Lowest and one past the highest occupied layer of the chunk (cells).
    z: vec2<i32>,
    // Leaves stored for the chunk (0: none) and their block (NONE: none).
    leaves: u32,
    block: u32,
}

const NONE: u32 = 0xffffffffu;
// Leaf block size classes: 16 << class leaves.
const CLASSES: u32 = 9u;

@group(0) @binding(1) var<uniform> world: World;
@group(0) @binding(16) var<uniform> terrain: TerrainConstants;
@group(0) @binding(2) var<storage, read> tile_jobs: array<TileJob>;
@group(0) @binding(3) var<storage, read_write> tile_columns: array<vec4<i32>>;
@group(0) @binding(4) var<storage, read_write> tile_ranges: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read> chunk_jobs: array<ChunkJob>;
@group(0) @binding(6) var<storage, read_write> column_masks: array<vec2<u32>>;
@group(0) @binding(7) var<storage, read> brushes: array<FaceBrush>;
@group(0) @binding(8) var<storage, read> brush_refs: array<u32>;
@group(0) @binding(9) var<storage, read_write> chunk_meta: array<ChunkMeta>;
@group(0) @binding(10) var<storage, read_write> mids: array<vec4<u32>>;
@group(0) @binding(11) var<storage, read_write> mid_leaves: array<u32>;
@group(0) @binding(12) var<storage, read_write> leaves: array<vec2<u32>>;
// The pool's end (leaves), an overflow flag, then each class's count of
// free blocks.
@group(0) @binding(13) var<storage, read_write> leaf_alloc: array<atomic<i32>>;
// Each class's free blocks (first leaf), from `stack_base`.
@group(0) @binding(20) var<storage, read_write> free_blocks: array<u32>;
// Per chunk job: slot, occupied layers [lo, hi), flags (1 built, 2 empty,
// 4 solid throughout, 8 out of leaf storage).
@group(0) @binding(19) var<storage, read_write> job_results: array<vec4<i32>>;
// Slots evicted this frame.
@group(0) @binding(21) var<storage, read> evictions: array<u32>;
// Baked edits (`edit_store`) under the chunk jobs: per job with any, a
// brick per brick column (8x8 of a chunk) and layer (8), NONE where none;
// a brick's 512 cell edits two a word.
@group(0) @binding(22) var<storage, read> baked_refs: array<u32>;
@group(0) @binding(23) var<storage, read> baked_cells: array<u32>;

fn class_of(n: u32) -> u32 {
    var c = 0u;
    while (16u << c) < n { c += 1u; }
    return c;
}

// First entry of class `c`'s free blocks: room for every block of each
// smaller class.
fn stack_base(c: u32) -> u32 {
    var b = 0u;
    for (var k = 0u; k < c; k++) { b += arrayLength(&leaves) / (16u << k); }
    return b;
}

// A block of `n` leaves (NONE: out of storage).
fn allocate(n: u32) -> u32 {
    let c = class_of(n);
    let i = atomicSub(&leaf_alloc[2u + c], 1);
    if i > 0 { return free_blocks[stack_base(c) + u32(i - 1)]; }
    atomicAdd(&leaf_alloc[2u + c], 1);
    let size = i32(16u << c);
    let b = atomicAdd(&leaf_alloc[0], size);
    if u32(b) + u32(size) > arrayLength(&leaves) {
        atomicStore(&leaf_alloc[1], 1);
        return NONE;
    }
    return u32(b);
}

@compute @workgroup_size(64)
fn evict(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&evictions) { return; }
    let slot = evictions[id.x];
    let m = chunk_meta[slot];
    if m.block == NONE || m.leaves == 0u { return; }
    let c = class_of(m.leaves);
    let i = atomicAdd(&leaf_alloc[2u + c], 1);
    free_blocks[stack_base(c) + u32(i)] = m.block;
    chunk_meta[slot].block = NONE;
    chunk_meta[slot].leaves = 0u;
}
// Scratch per chunk job: each column's materials (`chunk_columns_main`):
// its top cell's (bit 7: a single-voxel speck), the soil's and the rock's
// under it (bytes 0, 1, 2), and
// the lowest of the terrain's tops around it (its own and its neighbours')
// relative to the chunk, + 128 (byte 3): cells above it are a natural
// riser's, surface.
@group(0) @binding(14) var<storage, read_write> column_materials: array<u32>;
// Two material bytes per stored leaf (two leaves a word) and per mid node:
// the surface's, shown by its cells with air above, and the subsurface's
// (soil or rock by the leaf's depth), shown by the others.
@group(0) @binding(15) var<storage, read_write> leaf_materials: array<atomic<u32>>;
@group(0) @binding(17) var<storage, read_write> mid_materials: array<u32>;

var<workgroup> w_range: array<atomic<i32>, 4>;

// One workgroup per 8x8 columns of a tile: dispatch (64, tiles).
@compute @workgroup_size(64)
fn tile_columns_main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let job = tile_jobs[wg.y];
    let x = (wg.x & 7u) * 8u + (li & 7u);
    let y = (wg.x >> 3u) * 8u + (li >> 3u);
    let i = job.cx * 64 + i32(x);
    let j = job.cy * 64 + i32(y);
    if li == 0u {
        atomicStore(&w_range[0], 0x7fffffff);
        atomicStore(&w_range[1], -0x7fffffff);
        atomicStore(&w_range[2], 0x7fffffff);
        atomicStore(&w_range[3], -0x7fffffff);
    }
    workgroupBarrier();
    // Tiles at a face's edge reach past it: those columns are empty.
    let cells = world.grid.z >> job.level;
    if i < cells && j < cells {
        // Coarse levels keep the mean of the ridges they cannot resolve
        // (the display field), so they sit where finer levels' terrain sits.
        let field = generation_column(job.face, i, j, job.level, job.level >= 1u);
        let extent = terrain_extent(domain_point(job.face, i, j, job.level), job.level);
        let top = div_floor(field.x, world.grid.y) >> job.level;
        tile_columns[wg.y * TILE_COLUMNS + y * 64u + x] = vec4<i32>(field.x, field.y, extent.x, extent.y);
        // The cell under the top holds the surface; the volume may change
        // cells across its extent. Ranges: the volume's, the surface's.
        atomicMin(&w_range[0], top - 1 - extent.x);
        atomicMax(&w_range[1], top + extent.y);
        atomicMin(&w_range[2], top - 1);
        atomicMax(&w_range[3], top);
    } else {
        tile_columns[wg.y * TILE_COLUMNS + y * 64u + x] = vec4<i32>(OUTSIDE, 0, 0, 0);
    }
    workgroupBarrier();
    if li == 0u {
        atomicMin(&tile_ranges[wg.y * 4u], atomicLoad(&w_range[0]));
        atomicMax(&tile_ranges[wg.y * 4u + 1u], atomicLoad(&w_range[1]));
        atomicMin(&tile_ranges[wg.y * 4u + 2u], atomicLoad(&w_range[2]));
        atomicMax(&tile_ranges[wg.y * 4u + 3u], atomicLoad(&w_range[3]));
    }
}

fn set_bits(m: vec2<u32>, lo: i32, hi: i32, on: bool) -> vec2<u32> {
    // Bits [lo, hi) of a 64-bit mask set or cleared.
    let a = clamp(lo, 0, 64);
    let b = clamp(hi, 0, 64);
    if a >= b { return m; }
    var r = m;
    let lo_a = u32(min(a, 32));
    let lo_b = u32(min(b, 32));
    if lo_b > lo_a {
        let bits = select((1u << (lo_b - lo_a)) - 1u, 0xffffffffu, lo_b - lo_a == 32u) << lo_a;
        r.x = select(r.x & ~bits, r.x | bits, on);
    }
    let hi_a = u32(max(a, 32) - 32);
    let hi_b = u32(max(b, 32) - 32);
    if hi_b > hi_a {
        let bits = select((1u << (hi_b - hi_a)) - 1u, 0xffffffffu, hi_b - hi_a == 32u) << hi_a;
        r.y = select(r.y & ~bits, r.y | bits, on);
    }
    return r;
}

fn bit_at(m: vec2<u32>, b: u32) -> bool {
    return ((select(m.x, m.y, b >= 32u) >> (b & 31u)) & 1u) != 0u;
}

// Where a ball brush crosses the column over domain point `p` (half-cell
// centre `ch`): its first and last level cells, from the same solve as the
// span generator (`brush_crossings` in generate.wgsl).
fn ball_cells(b: FaceBrush, p: vec3<i32>, level: u32) -> vec3<i32> {
    let shift = level + 1u;
    let half = 1 << level;
    let h0 = (b.k_lo >> 1u) + (b.k_hi >> 1u);
    let q0 = volume_point_half(p, h0);
    var v = vec3<f32>(0.0, f32(world.scale.w) / 65536.0, 0.0);
    if !is_plane() { v = vec3<f32>(p) * (f32(world.scale.y) * exp2(-f32(world.scale.z) - 30.0)); }
    let a = dot(v, v);
    if a <= 0.0 { return vec3<i32>(0, -1, 0); }
    let d = vec3<f32>(q0 - b.ball.xyz);
    let r = f32(abs(b.ball.w));
    let bb = dot(d, v);
    let c = (length(d) - r) * (length(d) + r);
    let disc = bb * bb - a * c;
    if disc < 0.0 { return vec3<i32>(0, -1, 0); }
    let sq = sqrt(disc);
    let k0 = h0 >> shift;
    let rem = f32(h0 - (k0 << shift) - half);
    let cell = f32(1 << shift);
    return vec3<i32>(k0 + i32(floor((rem + (-bb - sq) / a) / cell + 0.5)), k0 + i32(floor((rem + (-bb + sq) / a) / cell + 0.5)), 1);
}

// The level cells brush `b` covers in the column over domain point `p`
// (half-cell centre `ch`): first, last, and whether any.
fn brush_cells(b: FaceBrush, p: vec3<i32>, ch: vec2<i32>, level: u32) -> vec3<i32> {
    if b.radius_half < (1u << level) { return vec3<i32>(0, -1, 0); }
    if ((b.flags >> 6u) & 3u) == 1u {
        if all(vec2<u32>(abs(ch - b.center.xy)) <= vec2<u32>(b.radius_half)) {
            let shift = level + 1u;
            let half = 1 << level;
            return vec3<i32>(-((half - b.k_lo) >> shift), (b.k_hi - half) >> shift, 1);
        }
        return vec3<i32>(0, -1, 0);
    }
    return ball_cells(b, p, level);
}

// Brush refs [first, end) of `job` applied in order to a column's cells
// from layer `k0`.
fn apply_brushes(mask: vec2<u32>, job: ChunkJob, first: u32, end: u32, p: vec3<i32>, ch: vec2<i32>, level: u32, k0: i32) -> vec2<u32> {
    var m = mask;
    for (var e = first; e < end; e++) {
        let b = brushes[brush_refs[job.brushes + e]];
        // Ops: 0 remove, 1 add; paint changes no cell (`edits::BrushOp`).
        let op = (b.flags >> 4u) & 3u;
        if op >= 2u { continue; }
        let cells = brush_cells(b, p, ch, level);
        if cells.z == 0 || cells.x > cells.y { continue; }
        m = set_bits(m, cells.x - k0, cells.y - k0 + 1, op == 1u);
    }
    return m;
}

// The baked edit of cell `z` of the job's column `li` in brick column
// `column` (`edit_store::CellEdit`: state in bits 0..2, 1 air, 2 solid,
// 3 paint; material in 8..16).
fn baked_edit(job: ChunkJob, column: u32, li: u32, z: u32) -> u32 {
    let b = baked_refs[job.baked + column * 8u + (z >> 3u)];
    if b == NONE { return 0u; }
    let n = (li & 7u) + 8u * ((li >> 3u) + 8u * (z & 7u));
    return (baked_cells[b * 256u + (n >> 1u)] >> ((n & 1u) * 16u)) & 0xffffu;
}

fn apply_baked(mask: vec2<u32>, job: ChunkJob, column: u32, li: u32) -> vec2<u32> {
    if job.baked == NONE { return mask; }
    var m = mask;
    for (var bz = 0u; bz < 8u; bz++) {
        if baked_refs[job.baked + column * 8u + bz] == NONE { continue; }
        for (var z = bz * 8u; z < bz * 8u + 8u; z++) {
            let state = baked_edit(job, column, li, z) & 3u;
            if state == 1u || state == 2u { m = set_bits(m, i32(z), i32(z) + 1, state == 2u); }
        }
    }
    return m;
}

// Solid cell `k`'s material after the edits over the terrain's
// `material`: the last brush that added or painted it, or its baked edit
// (material 0: the terrain's).
fn edited_material(material: u32, job: ChunkJob, column: u32, li: u32, p: vec3<i32>, ch: vec2<i32>, level: u32, k: i32, k0: i32) -> u32 {
    var m = material;
    for (var e = 0u; e <= job.brush_count; e++) {
        if e == job.large && job.baked != NONE {
            let edit = baked_edit(job, column, li, u32(k - k0));
            if (edit & 3u) >= 2u && (edit >> 8u) != 0u { m = edit >> 8u; }
        }
        if e == job.brush_count { break; }
        let b = brushes[brush_refs[job.brushes + e]];
        let brush_material = (b.flags >> 8u) & 0xffu;
        if ((b.flags >> 4u) & 3u) == 0u || brush_material == 0u { continue; }
        let cells = brush_cells(b, p, ch, level);
        if cells.z != 0 && k >= cells.x && k <= cells.y { m = brush_material; }
    }
    return m;
}

// One thread per column, one workgroup per 8x8 columns (a brick column):
// dispatch (64, chunks).
@compute @workgroup_size(64)
fn chunk_columns_main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let job = chunk_jobs[wg.y];
    let tile = tile_jobs[job.tile];
    let face = tile.face;
    let level = tile.level;
    let x = (wg.x & 7u) * 8u + (li & 7u);
    let y = (wg.x >> 3u) * 8u + (li >> 3u);
    let i = tile.cx * 64 + i32(x);
    let j = tile.cy * 64 + i32(y);
    let column = tile_columns[job.tile * TILE_COLUMNS + y * 64u + x];
    let at = wg.y * TILE_COLUMNS + y * 64u + x;
    if column.x == OUTSIDE {
        column_masks[at] = vec2<u32>(0u);
        column_materials[at] = M_AIR;
        return;
    }
    let height = column.x;
    let top = div_floor(height, world.grid.y) >> level;
    let k0 = job.cz * 64;
    var mask = set_bits(vec2<u32>(0u), 0, top - k0, true);
    let p = domain_point(face, i, j, level);
    if column.z > 0 || column.w > 0 {
        // Volume terms: the cells the program changes within its extent,
        // stepping over rock caves cannot reach (`terrain_clearance`).
        let k_lo = max(top - column.z, k0);
        let k_hi = min(top + column.w, k0 + 64);
        let band_lo = select(top, top - column.w, column.w > 0);
        let step = distance(vec3<f32>(volume_point(face, i, j, top + 1, level)), vec3<f32>(volume_point(face, i, j, top, level))) + 2.0;
        var scan = k_lo;
        loop {
            if scan >= k_hi { break; }
            if scan < band_lo {
                let clear = terrain_clearance(volume_point(face, i, j, scan, level), level, step);
                if clear > 0 {
                    scan = min(scan + clear + 1, band_lo);
                    continue;
                }
            }
            let solid = terrain_density(p, volume_point(face, i, j, scan, level), level, top, height, height, scan) > 0;
            mask = set_bits(mask, scan - k0, scan - k0 + 1, solid);
            scan += 1;
        }
    }
    // Edits in their order (`planet::Edits`): the large brushes, the baked
    // cells, the recent brushes.
    let ch = vec2<i32>(center_half(i, level), center_half(j, level));
    let natural = mask;
    mask = apply_brushes(mask, job, 0u, job.large, p, ch, level, k0);
    mask = apply_baked(mask, job, wg.x, li);
    mask = apply_brushes(mask, job, job.large, job.brush_count, p, ch, level, k0);
    column_masks[at] = mask;
    // Materials of the highest solid cell and under it.
    var material = M_AIR;
    let high = select(i32(firstLeadingBit(mask.x)), 32 + i32(firstLeadingBit(mask.y)), mask.y != 0u);
    if mask.x != 0u || mask.y != 0u {
        let k = k0 + high;
        let t = vec4<i32>(
            tile_top(job.tile, max(i32(x) - 1, 0), i32(y), level),
            tile_top(job.tile, min(i32(x) + 1, 63), i32(y), level),
            tile_top(job.tile, i32(x), max(i32(y) - 1, 0), level),
            tile_top(job.tile, i32(x), min(i32(y) + 1, 63), level));
        // The materials' slope (`Planet::material_slope`): the gradient over
        // 64 base cells (in the tile), so a terrace step is not a steep
        // slope (dirt along every contour); 8 per base layer a base cell.
        let n = max(32 >> level, 1);
        let xa = min(i32(x) + n, 63);
        let xb = max(i32(x) - n, 0);
        let ya = min(i32(y) + n, 63);
        let yb = max(i32(y) - n, 0);
        let gx = f32(tile_top(job.tile, xa, i32(y), level) - tile_top(job.tile, xb, i32(y), level)) / f32(max(xa - xb, 1));
        let gy = f32(tile_top(job.tile, i32(x), ya, level) - tile_top(job.tile, i32(x), yb, level)) / f32(max(ya - yb, 1));
        let slope = i32(round(max(abs(gx), abs(gy)) * 8.0));
        // The terrain's own top (where its volume put it, under the height
        // field's at caves and overhangs) is surface; a top edits moved
        // lies as deep as it is under the terrain's.
        let natural_high = select(i32(firstLeadingBit(natural.x)), 32 + i32(firstLeadingBit(natural.y)), natural.y != 0u);
        let moved = natural.x == 0u && natural.y == 0u || natural_high != high;
        let depth = select(0, max(top - 1 - k, 0) << level, moved);
        let ground = ground_material(p, u32(column.y), height, depth, slope, k << level);
        let terrain_material = ground & M_ID;
        let surface = edited_material(terrain_material, job, wg.x, li, p, ch, level, k, k0);
        let speck = select(0u, 0x80u, (ground & M_SPECK) != 0u && surface == terrain_material);
        // Under it: soil a cell below the terrain's top, rock below the
        // soil depth; an edited top's material throughout.
        var soil = surface;
        var rock = surface;
        if surface == terrain_material {
            soil = ground_material(p, u32(column.y), height, 1 << level, slope, (top - 2) << level) & M_ID;
            let deep = (terrain.header.z >> level) + 1;
            rock = ground_material(p, u32(column.y), height, deep << level, slope, (top - 1 - deep) << level) & M_ID;
        }
        let lowest = min(top, min(min(t.x, t.y), min(t.z, t.w)));
        material = (surface & 0x7fu) | speck | ((soil & 0xffu) << 8u) | ((rock & 0xffu) << 16u) | (u32(clamp(lowest - k0 + 128, 0, 255)) << 24u);
    }
    column_materials[at] = material;
}

fn tile_top(tile: u32, x: i32, y: i32, level: u32) -> i32 {
    let h = tile_columns[tile * TILE_COLUMNS + u32(y) * 64u + u32(x)].x;
    return div_floor(select(h, 0, h == OUTSIDE), world.grid.y) >> level;
}

// A leaf's two material bytes from its columns' materials
// (`chunk_columns_main`): the surface, the top material of a column whose
// top cell is in the leaf (`has_top`, `top`), else the subsurface's; the
// subsurface, by its first column at the leaf's centre (layer `lz` of a
// chunk of `level`): above the lowest terrain top around it (a natural
// riser) its surface, within the soil depth under it soil, else rock.
fn leaf_materials_of(column: u32, top: u32, has_top: u32, lz: u32, level: u32) -> u32 {
    let below = (i32(column >> 24u) - 128) - 1 - i32(lz * 4u + 2u);
    var sub = select((column >> 16u) & 0xffu, (column >> 8u) & 0xffu, (below << level) < terrain.header.z);
    if below < 0 { sub = column & 0x7fu; }
    return select(sub, top & 0x7fu, has_top != 0u) | (sub << 8u);
}

var<workgroup> w_occ: array<atomic<u32>, 128>;
var<workgroup> w_full: array<atomic<u32>, 128>;
var<workgroup> w_base: array<u32, 64>;
var<workgroup> w_z: array<atomic<i32>, 2>;
var<workgroup> w_total: u32;
var<workgroup> w_block: u32;
var<workgroup> w_mid_material: array<atomic<u32>, 64>;

fn rank64(m: vec2<u32>, b: u32) -> u32 {
    // Set bits of `m` below bit `b`.
    if b < 32u { return countOneBits(m.x & ((1u << b) - 1u)); }
    return countOneBits(m.x) + countOneBits(m.y & ((1u << (b - 32u)) - 1u));
}

// One workgroup per chunk: thread t builds the 16 leaves above leaf column
// (t & 15, t >> 4) of the chunk.
@compute @workgroup_size(256)
fn chunk_build_main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let job = chunk_jobs[wg.x];
    if t < 128u {
        atomicStore(&w_occ[t], 0u);
        atomicStore(&w_full[t], 0u);
    }
    if t < 64u { atomicStore(&w_mid_material[t], 0u); }
    if t == 0u {
        atomicStore(&w_z[0], 64);
        atomicStore(&w_z[1], 0);
    }
    workgroupBarrier();
    let lx = t & 15u;
    let ly = t >> 4u;
    // 16 leaves along z, built from this leaf column's 4x4 cell columns.
    var leaf: array<vec2<u32>, 16>;
    // Each leaf's materials: its first column with cells in it's, and for
    // its surface the most common top material of the columns whose top
    // cell is in it (a majority vote of all but single-voxel specks, which
    // a leaf of 4x4 columns would spread; `topped` where there are any).
    var leaf_material: array<u32, 16>;
    var leaf_top: array<u32, 16>;
    var votes: array<u32, 16>;
    var topped = 0u;
    var z_lo = 64;
    var z_hi = 0;
    for (var c = 0u; c < 16u; c++) {
        let cx = lx * 4u + (c & 3u);
        let cy = ly * 4u + (c >> 2u);
        let m = column_masks[wg.x * TILE_COLUMNS + cy * 64u + cx];
        if m.x == 0u && m.y == 0u { continue; }
        let column_material = column_materials[wg.x * TILE_COLUMNS + cy * 64u + cx];
        z_lo = min(z_lo, select(i32(firstTrailingBit(m.x)), 32 + i32(firstTrailingBit(m.y)), m.x == 0u));
        let high = select(i32(firstLeadingBit(m.x)), 32 + i32(firstLeadingBit(m.y)), m.y != 0u);
        z_hi = max(z_hi, 1 + high);
        let top_leaf = u32(high) >> 2u;
        topped |= 1u << top_leaf;
        if (column_material & 0x80u) != 0u {
            // A speck: the surface only while nothing else votes.
            if votes[top_leaf] == 0u { leaf_top[top_leaf] = column_material; }
        } else if votes[top_leaf] == 0u {
            leaf_top[top_leaf] = column_material;
            votes[top_leaf] = 1u;
        } else if (leaf_top[top_leaf] & 0x7fu) == (column_material & 0x7fu) {
            votes[top_leaf] += 1u;
        } else {
            votes[top_leaf] -= 1u;
        }
        for (var lz = 0u; lz < 16u; lz++) {
            let nib = (select(m.x, m.y, lz >= 8u) >> ((lz & 7u) * 4u)) & 15u;
            if nib != 0u && leaf_material[lz] == 0u { leaf_material[lz] = column_material; }
            leaf[lz].x |= ((nib & 1u) << c) | (((nib >> 1u) & 1u) << (c + 16u));
            leaf[lz].y |= (((nib >> 2u) & 1u) << c) | (((nib >> 3u) & 1u) << (c + 16u));
        }
    }
    atomicMin(&w_z[0], z_lo);
    atomicMax(&w_z[1], z_hi);
    for (var lz = 0u; lz < 16u; lz++) {
        let m = (lx >> 2u) + (ly >> 2u) * 4u + (lz >> 2u) * 16u;
        let l = (lx & 3u) + (ly & 3u) * 4u + (lz & 3u) * 16u;
        let word = m * 2u + (l >> 5u);
        let bit = 1u << (l & 31u);
        if leaf[lz].x != 0u || leaf[lz].y != 0u {
            atomicOr(&w_occ[word], bit);
            atomicMax(&w_mid_material[m], leaf_materials_of(leaf_material[lz], leaf_top[lz], (topped >> lz) & 1u, lz, tile_jobs[job.tile].level));
        }
        // A full leaf is implicit (its mid's material) unless a column tops
        // in it: then it is the surface and keeps its own materials.
        if leaf[lz].x == 0xffffffffu && leaf[lz].y == 0xffffffffu && ((topped >> lz) & 1u) == 0u { atomicOr(&w_full[word], bit); }
    }
    workgroupBarrier();
    if t == 0u {
        // Mixed leaves per mid node, packed in mid order.
        var total = 0u;
        for (var m = 0u; m < 64u; m++) {
            w_base[m] = total;
            let occ = vec2<u32>(atomicLoad(&w_occ[m * 2u]), atomicLoad(&w_occ[m * 2u + 1u]));
            let full = vec2<u32>(atomicLoad(&w_full[m * 2u]), atomicLoad(&w_full[m * 2u + 1u]));
            total += countOneBits(occ.x & ~full.x) + countOneBits(occ.y & ~full.y);
        }
        var base = NONE;
        if total > 0u { base = allocate(total); }
        let overflow = total > 0u && base == NONE;
        var root_occ = vec2<u32>(0u);
        var root_full = vec2<u32>(0u);
        for (var m = 0u; m < 64u; m++) {
            w_base[m] += base;
            let occ = vec2<u32>(atomicLoad(&w_occ[m * 2u]), atomicLoad(&w_occ[m * 2u + 1u]));
            let full = vec2<u32>(atomicLoad(&w_full[m * 2u]), atomicLoad(&w_full[m * 2u + 1u]));
            mids[job.slot * 64u + m] = vec4<u32>(occ, full);
            mid_leaves[job.slot * 64u + m] = w_base[m];
            mid_materials[job.slot * 64u + m] = atomicLoad(&w_mid_material[m]);
            let bit = 1u << (m & 31u);
            if occ.x != 0u || occ.y != 0u {
                if m < 32u { root_occ.x |= bit; } else { root_occ.y |= bit; }
            }
            if full.x == 0xffffffffu && full.y == 0xffffffffu {
                if m < 32u { root_full.x |= bit; } else { root_full.y |= bit; }
            }
        }
        w_total = select(total, 0u, overflow);
        w_block = base;
        let z = vec2<i32>(atomicLoad(&w_z[0]), atomicLoad(&w_z[1]));
        chunk_meta[job.slot] = ChunkMeta(root_occ, root_full, z, select(total, 0u, overflow), select(base, NONE, overflow));
        var flags = 1;
        if root_occ.x == 0u && root_occ.y == 0u { flags |= 2; }
        if root_full.x == 0xffffffffu && root_full.y == 0xffffffffu { flags |= 4; }
        if overflow { flags |= 8; }
        job_results[wg.x] = vec4<i32>(i32(job.slot), z.x, z.y, flags);
    }
    workgroupBarrier();
    let total = workgroupUniformLoad(&w_total);
    if total == 0u { return; }
    // The block's material bytes from earlier owners, cleared.
    let block = w_block;
    for (var w = block / 2u + t; w < (block + (16u << class_of(total))) / 2u; w += 256u) {
        atomicStore(&leaf_materials[w], 0u);
    }
    storageBarrier();
    for (var lz = 0u; lz < 16u; lz++) {
        let m = (lx >> 2u) + (ly >> 2u) * 4u + (lz >> 2u) * 16u;
        let l = (lx & 3u) + (ly & 3u) * 4u + (lz & 3u) * 16u;
        let occ = vec2<u32>(atomicLoad(&w_occ[m * 2u]), atomicLoad(&w_occ[m * 2u + 1u]));
        let full = vec2<u32>(atomicLoad(&w_full[m * 2u]), atomicLoad(&w_full[m * 2u + 1u]));
        let mixed = occ & ~full;
        if bit_at(mixed, l) {
            let index = w_base[m] + rank64(mixed, l);
            leaves[index] = leaf[lz];
            atomicOr(&leaf_materials[index >> 1u], leaf_materials_of(leaf_material[lz], leaf_top[lz], (topped >> lz) & 1u, lz, tile_jobs[job.tile].level) << ((index & 1u) * 16u));
        }
    }
}
