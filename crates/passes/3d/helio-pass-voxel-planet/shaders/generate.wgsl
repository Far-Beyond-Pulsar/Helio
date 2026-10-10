// GPU-driven column generation, run allocation and publication (span
// columns, docs/span-columns.md).
// Order per frame: evict -> generate -> count -> refill -> allocate -> fixup -> publish.

struct Job {
    key0: u32,
    key1: u32,
    record: u32,
    edits: u32,
    flags: u32,  // 1 = replaces a published column
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct JobOut {
    status: u32,   // 0 ok, 2 scratch full, 3 pool full, 4 skipped
    base: i32,     // natural tops' base (level cells)
    info: u32,     // record info bits: span count and flags
    n_mixed: u32,  // bricks copied to the pool (mixed bricks of BRICKS spans)
    scratch: u32,  // first scratch unit
    size_class: u32,
    run: u32,
    units: u32,    // header and span area units, before the bricks
    top: i32,      // first air layer above every solid cell (clipped above: the window top)
    pad2: u32,
    lo: i32,       // the window's bottom (level cells)
    summary: i32,  // bound of every solid cell, clipped or not (summary tops)
    n_eval: u32,   // bricks evaluated (in scratch after the header area)
    header: u32,   // header units
    pad0: u32,
    pad1: u32,
    mixed: array<u32, 8>,  // evaluated bricks copied to the pool
}

const CLASSES: u32 = 10u;
const PAGE_UNITS: u32 = 512u;
const A_TOP: u32 = 0u;
const A_FAILS: u32 = 10u;
const A_NEED: u32 = 16u;
const A_PAGES: u32 = 30u;
const A_SCRATCH: u32 = 31u;
// A column describes a window of 2 * WINDOW_CELLS level cells around the
// eye's layer at most; rays past a clipped side use the coarser level,
// whose window reaches twice as far, and the CPU regenerates the column
// when the eye moves a quarter window vertically. Any depth stays
// representable.
const WINDOW_CELLS: i32 = 1024;
// Scratch per job: the header area (at most six units), the evaluated
// bricks (at most the window's 256), then the span area.
const HEADER_UNITS_MAX: u32 = 5u;
const SPAN_UNITS_MAX: u32 = 16u;
// Candidate bricks are marked in a mask over the window (256 bricks); its
// runs become the intervals. At most MAX_INTERVALS remain after merging;
// with the gaps and the trimmed ends of brick spans a column has at most 31
// spans.
const MAX_INTERVALS: u32 = 7u;
// Readback status of a published column clipped below (5), above (6) or
// both (7), with its window's bottom (`residency::STATUS_CLIPPED`).
const STATUS_CLIPPED: u32 = 5u;

@group(0) @binding(7) var<storage, read> jobs: array<Job>;
@group(0) @binding(8) var<storage, read_write> job_out: array<JobOut>;
@group(0) @binding(9) var<storage, read_write> scratch: array<u32>;
@group(0) @binding(10) var<storage, read_write> alloc: array<atomic<i32>, 32>;
@group(0) @binding(11) var<storage, read_write> free_runs: array<u32>;
@group(0) @binding(12) var<storage, read_write> free_pages: array<u32>;
@group(0) @binding(13) var<storage, read> evictions: array<u32>;
// Failed jobs (key0, key1, status, 0), appended until the CPU copies them.
@group(0) @binding(17) var<storage, read_write> failures: array<u32>;
// Per pool page: free runs and size class. A page whose runs are all free
// returns to the free page stack under pool pressure (allocator_recycle).
struct PageMeta {
    free: atomic<u32>,
    size_class: u32,
}
@group(0) @binding(18) var<storage, read_write> page_meta: array<PageMeta>;

// What a column describes, decided once by lane 0: the merged candidate
// intervals (level cells, whole bricks), the window and its clipping, the
// bricks to evaluate and their scratch.
struct Plan {
    n: u32,
    clip: u32,
    w_lo: i32,
    w_hi: i32,
    // The planet's centre (whole bricks below it): a lane's radial line
    // ends there; below it the line runs out through the antipode.
    floor: i32,
    eval: u32,
    base: u32,
    summary: i32,
    iv: array<vec2<i32>, 8>,
}

fn class_offset(c: u32) -> u32 {
    let p = frame.counts.w;
    if c == 0u { return 0u; }
    return 2u * p - (p >> (c - 1u));
}

fn job_index(wg: vec3<u32>) -> u32 {
    return wg.x + wg.y * 32768u;
}

// Candidates: the window's bricks where some lane's occupancy can change,
// and the window's sides they pass.
var<workgroup> g_cand: array<atomic<u32>, 8>;
var<workgroup> g_cand_clip: atomic<u32>;
// Above every solid cell of the column (each lane's brush sweep): its
// summary top, and whether solid lies above the window at all.
var<workgroup> g_bound: atomic<i32>;
// The column's candidates, lowest and highest layer (stage 0 of the marking,
// `p_measure`): the window is chosen from them.
var<workgroup> g_range: array<atomic<i32>, 2>;
var<private> p_measure: bool;
var<private> p_range: vec2<i32>;
var<workgroup> g_plan: Plan;
// Per merged interval: lane summary bits (some solid 1, some air 2, not
// solid-below-air 4, tops not the natural ones 8, a top over a byte 16), the
// highest solid layer + 1 and each lane's top over the interval's start.
var<workgroup> g_iv_flags: array<atomic<u32>, 8>;
var<workgroup> g_iv_top: array<atomic<i32>, 8>;
var<workgroup> g_tops: array<atomic<u32>, 128>;
// Solid lanes of each gap between intervals (two words per gap).
var<workgroup> g_gap: array<atomic<u32>, 18>;
// The lowest natural top (level cells).
var<workgroup> g_natural: array<atomic<i32>, 1>;
var<workgroup> g_words: array<atomic<u32>, 32>;
var<workgroup> g_masks: array<atomic<u32>, 16>;
var<workgroup> g_any: array<atomic<u32>, 2>;
var<workgroup> g_topology_flags: u32;
var<workgroup> g_volume: atomic<u32>;
// The column's lane words (`lane_word` in common.wgsl).
var<workgroup> g_lane: array<u32, 64>;
// Surface words (`column_surface`), one byte per lane.
var<workgroup> g_surface: array<atomic<u32>, 16>;
// Per-brick brush culling: one chunk of the column's edit list at a time,
// kept brushes compacted in list order (ballot bits, then ranks).
var<workgroup> g_keep: array<atomic<u32>, 2>;
var<workgroup> g_list: array<FaceBrush, 64>;
// The job's edit block counts (`EditCounts`: large, recent, baked).
var<workgroup> g_edit_counts: vec3<u32>;
// Field heights of the column's lean lattice nodes (`terrain::lean_height`).
var<workgroup> g_lean: array<i32, 64>;
// g_volume bits: some lane's cells differ from the heightfield (generated
// volume); brushes that set materials; some lane may lean (overhangs: the
// lean lattice is evaluated).
const VOLUME_TERRAIN: u32 = 1u;
const VOLUME_MATERIALS: u32 = 2u;
const VOLUME_LEAN: u32 = 4u;

const NO_DENSITY: i32 = -2147483647 - 1;

// Relief of a generated surface (caves, overhangs): the zero crossing of
// the signed distances between the centres of the highest solid cell
// (density `solid`) and the air cell above it (`air`), in cells above the
// solid cell's centre. Unchanged lanes keep the heightfield's convention (the
// partial top cell is solid, cut at the exact height); changed lanes follow
// it too, or their neighbouring tops differed by up to half a cell: ledges
// all over overhang regions at levels with relief, gone at level 0.
fn volume_crossing(solid: i32, air: i32) -> f32 {
    return f32(solid) / f32(solid - air);
}

// Bilinear height of the lean lattice at `xy` (Q8 level cells from node
// (0, 0)), nodes 0..8 per axis (`terrain::lattice_bilinear`).
fn lean_bilinear(xy: vec2<i32>, spacing: i32) -> i32 {
    let step = spacing * 256;
    let n = clamp(vec2<i32>(div_floor(xy.x, step), div_floor(xy.y, step)), vec2<i32>(0), vec2<i32>(6));
    let f = clamp((xy - n * step) / spacing, vec2<i32>(0), vec2<i32>(256));
    let h00 = g_lean[n.x + n.y * 8];
    let h10 = g_lean[n.x + 1 + n.y * 8];
    let h01 = g_lean[n.x + (n.y + 1) * 8];
    let h11 = g_lean[n.x + 1 + (n.y + 1) * 8];
    let a = h00 + (((h10 - h00) * f.x) >> 8u);
    let b = h01 + (((h11 - h01) * f.x) >> 8u);
    return a + (((b - a) * f.y) >> 8u);
}

// Height (mm) of the heightfield displaced by the program's lean under cell
// `(i, j, k)` (`terrain::lean_height`); `node` is the lattice's first node.
fn lean_height(p: vec3<i32>, i: i32, j: i32, k: i32, level: u32, height: i32, spacing: i32, node: vec2<i32>) -> i32 {
    let o = terrain_lean_offset(p, i, j, k, level);
    if o.x == 0 && o.y == 0 { return height; }
    let xy = vec2<i32>(i, j) * 256 + 128 - node * spacing * 256;
    return lean_bilinear(xy + o, spacing) + height - lean_bilinear(xy, spacing);
}

// Density of generated cell `k` of column `(i, j)` (`terrain_density`), its
// surface leaning when `leaning` (lattice spacing and first node).
fn generated_density(p: vec3<i32>, face: u32, i: i32, j: i32, k: i32, level: u32, field_top: i32, height: i32, leaning: bool, spacing: i32, node: vec2<i32>) -> i32 {
    var lean_h = height;
    if leaning { lean_h = lean_height(p, i, j, k, level, height, spacing, node); }
    return terrain_density(p, volume_point(face, i, j, k, level), level, field_top, height, lean_h, k);
}

// Where a brush's surface crosses one lane (the column of cells over
// domain point `p`, half-cell centre `ch` across): the first and last
// level cells inside it, and how far either may be off (cells).
struct Crossing {
    lo: i32,
    hi: i32,
    margin: i32,
    hit: bool,
}

fn brush_crossings(b: FaceBrush, p: vec3<i32>, ch: vec2<i32>, level: u32) -> Crossing {
    var out = Crossing(0, 0, 1, false);
    let shift = level + 1u;
    let half = 1 << level;
    if ((b.flags >> 6u) & 3u) == 1u {
        // A box: the lanes inside its footprint, between its faces (cell
        // centres `(k << shift) + half` from k_lo to k_hi), exactly.
        if any(vec2<u32>(abs(ch - b.center.xy)) > vec2<u32>(b.radius_half)) { return out; }
        out.lo = -((half - b.k_lo) >> shift);
        out.hi = (b.k_hi - half) >> shift;
        out.hit = out.lo <= out.hi;
        return out;
    }
    // A ball in volume space. A lane's volume points are linear in the
    // layer (`volume_point_half`: the domain point scaled by a fixed ratio
    // per half layer, or the height itself on a plane), so the ball holds
    // one interval of it: solved in f32 around the lane's exact point at the
    // ball's height, along the exact slope (a slope measured between two
    // rounded points tens of cells apart was percents off, and moved a 3 km
    // crater's floor 45 cells), with the error of the f32 integers and
    // products (relative 2^-22) and of the volume points' rounding; near
    // tangency the root moves by the square root of the discriminant's
    // error. The interval only chooses the cells evaluated exactly.
    let h0 = (b.k_lo >> 1u) + (b.k_hi >> 1u);
    let q0 = volume_point_half(p, h0);
    var v = vec3<f32>(0.0, f32(world.scale.w) / 65536.0, 0.0);
    if !is_plane() { v = vec3<f32>(p) * (f32(world.scale.y) * exp2(-f32(world.scale.z) - 30.0)); }
    let a = dot(v, v);
    if a <= 0.0 { return out; }
    let d = vec3<f32>(q0 - b.ball.xyz);
    let r = f32(abs(b.ball.w));
    let dl = length(d);
    let bb = dot(d, v);
    let c = (dl - r) * (dl + r);
    let disc = bb * bb - a * c;
    let e = (dl + r) * 4.8e-7 + 4.0;
    let dd = a * 2.0 * (dl + r) * e;
    if disc < -dd { return out; }
    let sq = sqrt(max(disc, 0.0));
    let root_err = select(dd / (2.0 * sq), sqrt(dd), sq * sq < dd);
    let t_err = (root_err + e * sqrt(a)) / a;
    // Cells relative to the cell holding h0, so f32 never holds the
    // planet-scale height itself.
    let k0 = h0 >> shift;
    let rem = f32(h0 - (k0 << shift) - half);
    let cell = f32(1 << shift);
    out.lo = k0 + i32(floor((rem + (-bb - sq) / a) / cell + 0.5));
    out.hi = k0 + i32(floor((rem + (-bb + sq) / a) / cell + 0.5));
    out.margin = i32(ceil(t_err / cell)) + 2;
    out.hit = true;
    return out;
}

// The planet's centre at `level` (whole bricks): a lane's radial line ends
// there, below it the line runs out through the antipode.
fn column_floor(level: u32) -> i32 {
    if is_plane() { return NO_LAYER; }
    // Volume points reach zero at the planet's centre: (1 + ratio) = 0,
    // the ratio being h * scale.y >> scale.z in Q30 (half layers h).
    let core_half = -i32(round(exp2(30.0 + f32(world.scale.z)) / f32(world.scale.y)));
    return ((core_half >> (level + 1u)) >> 3u) << 3u;
}

// The column's window: `[w.x, w.y)` level cells, and its floor (`w.z`,
// `column_floor`). A column describes 2 * WINDOW_CELLS cells: all of its
// candidates `range` when they fit, wherever the eye is, so open ground, a
// dig or a crater wall never depend on the eye's height; a taller one (a
// crater kilometres deep at a fine level) describes the cells around the
// eye's layer within its candidates and follows the eye on the sides it
// clips. Windows around the eye's layer for every column left the ground
// of columns under a high eye clipped below: rays fell back level by level
// and every level regenerated each time the eye moved a quarter window (a
// descent from orbit cost 50 ms of primary rays). The coarsest level, the
// coverage every ray falls back to, runs from the centre up (a planet's
// radius is a few hundred of its cells), or around the datum on a plane.
fn column_window(level: u32, range: vec2<i32>) -> vec3<i32> {
    let floor = column_floor(level);
    let span = 2 * WINDOW_CELLS;
    var lo = (((frame.layer_i.x >> level) - WINDOW_CELLS) >> 3u) << 3u;
    if range.x < range.y {
        let a = (range.x >> 3u) << 3u;
        let b = ((range.y + 7) >> 3u) << 3u;
        if b - a <= span { lo = a; } else { lo = clamp(lo, a, b - span); }
    }
    if level + 1u >= u32(frame.layer_i.z) {
        lo = select(floor, -WINDOW_CELLS, is_plane());
    }
    lo = max(lo, floor);
    return vec3<i32>(lo, lo + span, floor);
}

// Mark the window's bricks a candidate interval `[iv.x, iv.y)` reaches;
// parts past the window clip its side, parts below the centre are no space.
// While measuring (`p_measure`), only extend the lane's candidate range.
fn mark_candidate(iv: vec2<i32>, w: vec3<i32>) {
    let lo = max(iv.x, w.z);
    if iv.y <= lo { return; }
    if p_measure {
        p_range = vec2<i32>(min(p_range.x, lo), max(p_range.y, iv.y));
        return;
    }
    if lo < w.x { atomicOr(&g_cand_clip, INFO_CLIP_BELOW); }
    if iv.y > w.y { atomicOr(&g_cand_clip, INFO_CLIP_ABOVE); }
    let a = (max(lo, w.x) - w.x) >> 3u;
    let b = (min(iv.y, w.y) - w.x + 7) >> 3u;
    for (var k = a; k < b; k++) {
        atomicOr(&g_cand[u32(k) >> 5u], 1u << (u32(k) & 31u));
    }
}

@compute @workgroup_size(64)
fn generate(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let index = job_index(wg);
    if index >= frame.counts.x { return; }
    let job = jobs[index];
    let face = (job.key0 >> 24u) & 7u;
    let level = job.key0 >> 27u;
    let ci = i32(job.key0 & 0xffffffu);
    let cj = bitcast<i32>(job.key1);
    let x = i32(li & 7u);
    let y = i32(li >> 3u);
    let i = ci * 8 + x;
    let j = cj * 8 + y;
    if li == 0u {
        let n = edit_counts(job.edits);
        // Baked cells may carve or fill; of the brushes, Paint keeps the
        // base geometry.
        var topology = n.baked != 0u;
        for (var e = 0u; e < n.large + n.recent && !topology; e++) {
            let b = edit_brush(job.edits, e);
            topology = b.radius_half >= (1u << level) && ((b.flags >> 4u) & 3u) < 2u;
        }
        g_topology_flags = select(0u, INFO_TOPOLOGY, topology);
        g_edit_counts = vec3<u32>(n.large, n.recent, n.baked);
        atomicStore(&g_keep[0], 0u);
        atomicStore(&g_keep[1], 0u);
        atomicStore(&g_cand_clip, 0u);
        atomicStore(&g_bound, -0x7fffffff);
        atomicStore(&g_range[0], 0x7fffffff);
        atomicStore(&g_range[1], -0x7fffffff);
        atomicStore(&g_natural[0], 0x7fffffff);
        atomicStore(&g_any[0], 0u);
        atomicStore(&g_any[1], 0u);
        atomicStore(&g_volume, 0u);
    }
    if li < 16u {
        atomicStore(&g_words[li], 0u);
        atomicStore(&g_words[li + 16u], 0u);
        atomicStore(&g_masks[li], 0u);
        atomicStore(&g_surface[li], 0u);
    }
    if li < 18u { atomicStore(&g_gap[li], 0u); }
    if li < 8u {
        atomicStore(&g_cand[li], 0u);
        atomicStore(&g_iv_flags[li], 0u);
        atomicStore(&g_iv_top[li], NO_LAYER);
    }
    atomicStore(&g_tops[li], 0u);
    atomicStore(&g_tops[li + 64u], 0u);
    // This replaces the existing initialization barrier; the edit list is
    // scanned once per workgroup, with no extra terrain query or barrier.
    let topology_flags = workgroupUniformLoad(&g_topology_flags);
    // An edit changes occupancy, not the display field of untouched lanes.
    // Continuous fractions require a heightfield, so topology brushes turn
    // them off.
    let display_base = (frame.hints.w & 8u) != 0u && level >= 1u;
    let relief = display_base && topology_flags == 0u;
    // Bit 16: the display height keeps unresolved ridges' mean (off only
    // for audits against the canonical field).
    let display = display_base && (frame.hints.w & 16u) != 0u;
    let lean = terrain_lean(level);
    // A leaning surface reads the field on a global lattice (nodes every
    // `spacing` cells, shared by neighbouring columns): each lane evaluates
    // one of the 8x8 nodes from `div_floor(8 ci - reach, spacing)`.
    let lean_node = vec2<i32>(div_floor(ci * 8 - lean.x, max(lean.y, 1)), div_floor(cj * 8 - lean.x, max(lean.y, 1)));
    let last = (world.grid.z >> level) - 1;
    let node_cell = vec2<i32>(clamp((lean_node.x + x) * lean.y, 0, last), clamp((lean_node.y + y) * lean.y, 0, last));
    // The lane's own column, then (when some lane may lean) its lattice
    // node: one interpreter call site (compilers inline every call).
    var column = vec2<i32>(0);
    var extent = vec2<i32>(0);
    var column_point = vec3<i32>(0);
    for (var phase = 0u; phase < 2u; phase++) {
        if phase == 0u || (atomicLoad(&g_volume) & VOLUME_LEAN) != 0u {
            let at = select(vec2<i32>(i, j), node_cell, phase == 1u);
            let field = generation_column(face, at.x, at.y, level, display);
            if phase == 1u {
                g_lean[li] = field.x;
            } else {
                column = field;
                column_point = domain_point(face, i, j, level);
                extent = terrain_extent(column_point, level);
                if lean.x != 0 && extent.y > 0 { atomicOr(&g_volume, VOLUME_LEAN); }
            }
        }
        workgroupBarrier();
    }
    let height = column.x;
    let base_top = div_floor(height, world.grid.y);
    let remainder = u32(base_top) & ((1u << level) - 1u);
    var top = base_top >> level;
    if relief && remainder != 0u { top += 1; }
    // Volumetric terrain (caves, overhangs): the program may change cells
    // within its extent around the heightfield top, evaluated in 3D. Only
    // the cells it changes make generated volume: a column whose cells all
    // keep the heightfield's kinds (most of a cave region's rock, ground too
    // flat to lean) stays a heightfield column. The scan finds the lane's
    // changed cells as runs; they and one more cell on each side are
    // evaluated (the generated surface's relief reads the densities around
    // it). Under the band the surface may lean through, only caves change
    // cells, and the scan steps over the rock they cannot reach
    // (`terrain_clearance`): every cell down to the cave depth (1,200 at 0.1
    // m) was evaluated, and a mountain's column cost 100 heightfield
    // columns. The band's runs form one, as the whole band once did.
    let field_top = base_top >> level;
    let leaning = lean.x != 0 && extent.y > 0;
    let k_lo = field_top - extent.x;
    let k_hi = field_top + extent.y;
    let band_lo = select(field_top, field_top - extent.y, extent.y > 0);
    let step = distance(vec3<f32>(volume_point(face, i, j, field_top + 1, level)), vec3<f32>(volume_point(face, i, j, field_top, level))) + 2.0;
    var runs: array<vec2<i32>, 8>;
    var n_runs = 0u;
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
        if (generated_density(column_point, face, i, j, scan, level, field_top, height, leaning, lean.y, lean_node) > 0) != (scan < field_top) {
            if n_runs > 0u && (scan <= runs[n_runs - 1u].y + 8 || (scan >= band_lo && runs[n_runs - 1u].y > band_lo)) {
                runs[n_runs - 1u].y = scan + 1;
            } else if n_runs < 8u {
                runs[n_runs] = vec2<i32>(scan, scan + 1);
                n_runs += 1u;
            } else {
                runs[7].y = scan + 1;
            }
        }
        scan += 1;
    }
    let changed = n_runs > 0u;
    // The highest evaluated cell.
    let eval_hi = select(-1, min(runs[max(n_runs, 1u) - 1u].y, k_hi - 1), changed);
    if changed { atomicOr(&g_volume, VOLUME_TERRAIN); }
    // The terrain's candidates: the cells around each lane's top whose kind
    // the heightfield sets (and the volume's runs, below).
    let t_lo = min(top, field_top) - 1;
    let t_hi = max(top, field_top) + 1;
    let ch = vec2<i32>(center_half(i, level), center_half(j, level));
    let edit_counts = workgroupUniformLoad(&g_edit_counts);
    let n_brushes = edit_counts.x + edit_counts.y;
    let baked_base = job.edits + 2u + n_brushes;
    if job.edits != 0u {
        for (var e = li; e < n_brushes; e += 64u) {
            let b = edit_brush(job.edits, e);
            // Shading reads brush materials only where some brush sets one.
            if b.radius_half >= (1u << level) && ((b.flags >> 4u) & 3u) != 0u { atomicOr(&g_volume, VOLUME_MATERIALS); }
        }
    }
    // The candidates are offered twice: stage 0 measures their range, from
    // which the window is chosen (`column_window`); stage 1 marks them in it.
    var window = vec3<i32>(0, 0, column_floor(level));
    var lane_bound = 0;
    p_range = vec2<i32>(0x7fffffff, -0x7fffffff);
    for (var stage = 0u; stage < 2u; stage++) {
        p_measure = stage == 0u;
        // The lane's highest solid layer + 1, from its ground (and the
        // volume's evaluated cells), raised by baked bricks and the brushes
        // below.
        lane_bound = select(top, max(top, eval_hi + 1), changed);
        mark_candidate(vec2<i32>(t_lo, t_hi), window);
        // The volume's evaluated cells and one more on each side.
        for (var r = 0u; r < n_runs; r++) {
            mark_candidate(vec2<i32>(max(runs[r].x - 1, k_lo) - 1, min(runs[r].y, k_hi - 1) + 2), window);
        }
        if job.edits != 0u {
            // Baked bricks: each a candidate.
            for (var e = li; e < edit_counts.z; e += 64u) {
                let bk = bitcast<i32>(edit_refs[baked_base + e * 2u]);
                mark_candidate(vec2<i32>(bk * 8, bk * 8 + 8), window);
                atomicMax(&g_bound, bk * 8 + 8);
                atomicOr(&g_volume, VOLUME_MATERIALS);
            }
        }
        // Brush surfaces: each lane runs the brushes over its own line in order,
        // as a union of solid intervals from its heightfield ground (a Remove
        // subtracts, an Add unites), and offers the ends of what remains. A
        // brush's surface buried in air another one dug out (hundreds of
        // overlapping strokes of a planet-scale brush) costs nothing; offering
        // every brush's crossings overflowed the candidates into one interval
        // from the deepest to the highest, evaluated cell by cell. Crossings are
        // exact to their margin, so a brush end landing within two margins of
        // the set's boundaries (abutting boxes, a thin sliver between two digs)
        // is offered as well. Baked bricks and the terrain's evaluated volume
        // are candidates of their own.
        if n_brushes != 0u {
            var solid: array<vec2<i32>, 8>;
            var count = 1u;
            solid[0] = vec2<i32>(-0x3fffffff, top);
            var margin = 1;
            var overflow = false;
            var lo_all = 0x7fffffff;
            var hi_all = -0x7fffffff;
            for (var e = 0u; e < n_brushes; e++) {
                let b = edit_brush(job.edits, e);
                let op = (b.flags >> 4u) & 3u;
                if b.radius_half < (1u << level) || op == 2u { continue; }
                let crossing = brush_crossings(b, column_point, ch, level);
                if !crossing.hit { continue; }
                margin = max(margin, crossing.margin);
                let range = vec2<i32>(crossing.lo, crossing.hi + 1);
                lo_all = min(lo_all, range.x);
                hi_all = max(hi_all, range.y);
                if overflow { continue; }
                for (var q = 0u; q < count; q++) {
                    let near = 2 * crossing.margin;
                    for (var end = 0u; end < 2u; end++) {
                        let x = select(range.x, range.y, end == 1u);
                        if abs(x - solid[q].x) <= near || abs(x - solid[q].y) <= near {
                            mark_candidate(vec2<i32>(x - near, x + near + 1), window);
                        }
                    }
                }
                var next: array<vec2<i32>, 8>;
                var m = 0u;
                if op == 0u {
                    for (var q = 0u; q < count; q++) {
                        let piece = solid[q];
                        if piece.y <= range.x || piece.x >= range.y {
                            if m < 8u { next[m] = piece; }
                            m += 1u;
                            continue;
                        }
                        if piece.x < range.x {
                            if m < 8u { next[m] = vec2<i32>(piece.x, range.x); }
                            m += 1u;
                        }
                        if piece.y > range.y {
                            if m < 8u { next[m] = vec2<i32>(range.y, piece.y); }
                            m += 1u;
                        }
                    }
                } else {
                    var merged = range;
                    for (var q = 0u; q < count; q++) {
                        let piece = solid[q];
                        if piece.y < merged.x || piece.x > merged.y {
                            if m < 8u { next[m] = piece; }
                            m += 1u;
                        } else {
                            merged = vec2<i32>(min(merged.x, piece.x), max(merged.y, piece.y));
                        }
                    }
                    if m < 8u { next[m] = merged; }
                    m += 1u;
                }
                if m > 8u {
                    overflow = true;
                    continue;
                }
                solid = next;
                count = m;
            }
            if overflow {
                mark_candidate(vec2<i32>(lo_all - margin, hi_all + margin + 1), window);
                lane_bound = max(lane_bound, hi_all + margin + 1);
            } else {
                lane_bound = -0x7fffffff;
                for (var q = 0u; q < count; q++) { lane_bound = max(lane_bound, solid[q].y + margin + 1); }
                if changed { lane_bound = max(lane_bound, eval_hi + 1); }
            }
            for (var q = 0u; q < count; q++) {
                if solid[q].x > -0x3fffffff { mark_candidate(vec2<i32>(solid[q].x - margin, solid[q].x + margin + 1), window); }
                mark_candidate(vec2<i32>(solid[q].y - margin, solid[q].y + margin + 1), window);
            }
        }
        if stage == 0u {
            atomicMin(&g_range[0], p_range.x);
            atomicMax(&g_range[1], p_range.y);
            workgroupBarrier();
            window = column_window(level, vec2<i32>(atomicLoad(&g_range[0]), atomicLoad(&g_range[1])));
        }
    }
    atomicMax(&g_bound, lane_bound);
    workgroupBarrier();
    // Lane 0 turns the candidate bricks into intervals (runs at most a brick
    // apart merged) and reserves the scratch.
    if li == 0u {
        var plan: Plan;
        var runs: array<vec2<i32>, 128>;
        var kept = 0u;
        var b = 0u;
        while b < 256u {
            if ((atomicLoad(&g_cand[b >> 5u]) >> (b & 31u)) & 1u) == 0u {
                b += 1u;
                continue;
            }
            var e = b;
            while e < 256u && ((atomicLoad(&g_cand[e >> 5u]) >> (e & 31u)) & 1u) != 0u { e += 1u; }
            let iv = vec2<i32>(window.x + i32(b) * 8, window.x + i32(e) * 8);
            if kept > 0u && iv.x <= runs[kept - 1u].y + 8 {
                runs[kept - 1u].y = iv.y;
            } else {
                runs[kept] = iv;
                kept += 1u;
            }
            b = e;
        }
        // At most MAX_INTERVALS: close the smallest gaps.
        while kept > MAX_INTERVALS {
            var best = 0u;
            for (var e = 1u; e + 1u < kept; e++) {
                if runs[e + 1u].x - runs[e].y < runs[best + 1u].x - runs[best].y { best = e; }
            }
            runs[best].y = runs[best + 1u].y;
            for (var e = best + 1u; e + 1u < kept; e++) { runs[e] = runs[e + 1u]; }
            kept -= 1u;
        }
        var clip = atomicLoad(&g_cand_clip);
        let floor = window.z;
        let w_lo = window.x;
        let w_hi = window.y;
        // Candidates above the window over air (a carved surface's old
        // ground, crossings inside dug space) clip nothing: only solid
        // there does. The summary top bounds the solid cells: a pit's
        // columns clipped at the old ground once kept its air from being
        // skipped, and rays crossed it column by column.
        let bound = atomicLoad(&g_bound);
        if bound <= w_hi { clip &= ~INFO_CLIP_ABOVE; }
        let summary = bound;
        var eval = 0u;
        for (var e = 0u; e < 8u; e++) {
            if e < kept {
                plan.iv[e] = runs[e];
                eval += u32(runs[e].y - runs[e].x) >> 3u;
            } else {
                plan.iv[e] = vec2<i32>(0);
            }
        }
        plan.n = kept;
        plan.floor = floor;
        plan.clip = clip;
        plan.w_lo = w_lo;
        plan.w_hi = w_hi;
        plan.eval = eval;
        plan.summary = summary;
        let need = i32(HEADER_UNITS_MAX + eval + SPAN_UNITS_MAX);
        let base = atomicAdd(&alloc[A_SCRATCH], need);
        if u32(base + need) * UNIT_WORDS > arrayLength(&scratch) {
            plan.base = NONE;
        } else {
            plan.base = u32(base);
        }
        g_plan = plan;
    }
    var plan = workgroupUniformLoad(&g_plan);
    if plan.base == NONE {
        if li == 0u { job_out[index].status = 2u; }
        return;
    }
    let bricks_base = plan.base + HEADER_UNITS_MAX;
    // Generated top of a lane: first air above its highest solid cell before
    // edits (overhang lips, cave openings), from the evaluated cells.
    var generated_top = NO_LAYER;
    // Caves or overhangs change this lane's top cell or the air above it:
    // its surface is no longer the relief surface.
    var surface_changed = false;
    // Densities at the generated surface (the highest solid cell inside the
    // volume and the air cell above it): the relief of a changed surface.
    var top_solid = NO_DENSITY;
    var top_air = NO_DENSITY;
    // The generated surface: the densities' zero crossing between the
    // highest solid cell's centre and the air above, level cells over the
    // bottom of that air cell (-0.5..0.5).
    var surface_crossing = 0.0;
    // The air cell of that crossing.
    var crossing_air = -0x7fffffff;
    // Lanes the overhangs fold take every evaluated cell, and their relief,
    // from the density.
    let dense_lane = leaning && changed;
    // The column's footprint in half cells, for culling brushes per brick.
    let half_cell = 1 << (level + 1u);
    let foot_lo = vec2<i32>(ci, cj) * 8 * half_cell;
    let foot_hi = foot_lo + 8 * half_cell;
    var gb = 0u;
    for (var iv = 0u; iv < plan.n; iv++) {
        let range = plan.iv[iv];
        // The lane over the interval: its highest solid layer + 1, whether
        // it has air, and whether it has solid above air.
        var lane_top = range.x;
        var lane_air = false;
        var lane_layered = false;
        for (var bk = range.x >> 3u; bk < (range.y >> 3u); bk++) {
            var kinds: array<u32, 8>;
            for (var z = 0u; z < 8u; z++) {
                let k = bk * 8 + i32(z);
                // Cells the volume leaves keep the heightfield's kinds (with
                // relief, its ceil top cell).
                var kind = terrain_kind(top, k);
                var evaluated = false;
                for (var r = 0u; r < n_runs; r++) {
                    evaluated = evaluated || (k >= max(runs[r].x - 1, k_lo) && k <= min(runs[r].y, k_hi - 1));
                }
                if evaluated {
                    // Overhangs fold the exact surface: their lanes take every
                    // evaluated cell from the density. Elsewhere (caves) a cell
                    // the volume leaves as the heightfield has it keeps its kind.
                    let density = generated_density(column_point, face, i, j, k, level, field_top, height, leaning, lean.y, lean_node);
                    let dense = select(0u, 1u, density > 0);
                    if dense != terrain_kind(field_top, k) {
                        kind = dense;
                        if k >= top - 1 { surface_changed = true; }
                    } else if dense_lane {
                        kind = dense;
                    }
                    if kind != 0u {
                        top_solid = density;
                        top_air = NO_DENSITY;
                    } else if top_solid != NO_DENSITY && top_air == NO_DENSITY {
                        top_air = density;
                        let t = select(0.5, volume_crossing(top_solid, density), top_solid > 0);
                        surface_crossing = t - 0.5;
                        crossing_air = k;
                        if relief && t > 0.5 {
                            // The surface rises into this cell: solid, cut there
                            // (its lane word's relief).
                            kind = 1u;
                        }
                    }
                }
                if kind != 0u { generated_top = max(generated_top, k + 1); }
                kinds[z] = kind;
            }
            // The edits in order: the large brushes, the baked cells, then the
            // recent brushes. Brushes go in chunks of 64: each lane culls one
            // against this brick's box, the kept ones are compacted in order and
            // every lane applies them to its eight cells. Bricks no brush
            // reaches cost one bounds test per brush, not per cell.
            let brick_lo = (bk * 8) * half_cell;
            let brick_hi = brick_lo + 8 * half_cell;
            for (var part = 0u; part < 2u; part++) {
                if part == 1u && edit_counts.z != 0u {
                    let n = EditCounts(edit_counts.x, edit_counts.y, edit_counts.z);
                    for (var z = 0u; z < 8u; z++) {
                        let cell = baked_cell(job.edits, n, i, j, bk * 8 + i32(z)) & 3u;
                        if cell == BAKED_AIR { kinds[z] = 0u; }
                        else if cell == BAKED_SOLID { kinds[z] = 1u; }
                    }
                }
                let first = select(0u, edit_counts.x, part == 1u);
                let end = select(edit_counts.x, n_brushes, part == 1u);
                for (var start = first; start < end; start += 64u) {
                    let e = start + li;
                    var keep = false;
                    var brush: FaceBrush;
                    if e < end {
                        brush = edit_brush(job.edits, e);
                        let extent_half = brush.center.w + half_cell;
                        keep = brush.radius_half >= (1u << level)
                            && brush.k_hi + half_cell >= brick_lo && brush.k_lo - half_cell < brick_hi
                            && brush.center.x + extent_half >= foot_lo.x && brush.center.x - extent_half < foot_hi.x
                            && brush.center.y + extent_half >= foot_lo.y && brush.center.y - extent_half < foot_hi.y;
                        if keep { atomicOr(&g_keep[li >> 5u], 1u << (li & 31u)); }
                    }
                    workgroupBarrier();
                    let m0 = atomicLoad(&g_keep[0]);
                    let m1 = atomicLoad(&g_keep[1]);
                    if keep {
                        var rank = countOneBits(m0 & ((1u << (li & 31u)) - 1u));
                        if li >= 32u { rank = countOneBits(m0) + countOneBits(m1 & ((1u << (li & 31u)) - 1u)); }
                        g_list[rank] = brush;
                    }
                    workgroupBarrier();
                    // Every lane has read the ballot; the next chunk starts after the
                    // closing barrier.
                    if li == 0u {
                        atomicStore(&g_keep[0], 0u);
                        atomicStore(&g_keep[1], 0u);
                    }
                    let kept = countOneBits(m0) + countOneBits(m1);
                    for (var z = 0u; z < 8u && kept != 0u; z++) {
                        let c = vec3<i32>(ch, center_half(bk * 8 + i32(z), level));
                        var q = vec3<i32>(0);
                        var q_ready = false;
                        for (var n = 0u; n < kept; n++) {
                            let bb = g_list[n];
                            if c.z < bb.k_lo || c.z > bb.k_hi { continue; }
                            if !q_ready && ((bb.flags >> 6u) & 3u) == 0u {
                                q = volume_point_half(column_point, c.z);
                                q_ready = true;
                            }
                            if !brush_contains(bb, c, q) { continue; }
                            let op = (bb.flags >> 4u) & 3u;
                            if op == 0u { kinds[z] = 0u; }
                            else if op == 1u { kinds[z] = 1u; }
                        }
                    }
                    workgroupBarrier();
                }
            }
            for (var z = 0u; z < 8u; z++) {
                let k = bk * 8 + i32(z);
                if kinds[z] != 0u {
                    let bit = li + z * 64u;
                    atomicOr(&g_words[bit >> 5u], 1u << (bit & 31u));
                    if lane_air { lane_layered = true; }
                    lane_top = k + 1;
                } else {
                    lane_air = true;
                }
            }
            workgroupBarrier();
            if li < 16u {
                let w = atomicExchange(&g_words[li], 0u);
                scratch[(bricks_base + gb) * UNIT_WORDS + li] = w;
                if w != 0u { atomicOr(&g_any[0], 1u); }
                if w != 0xffffffffu { atomicOr(&g_any[1], 1u); }
            }
            workgroupBarrier();
            if li == 0u {
                let some = atomicExchange(&g_any[0], 0u) != 0u;
                let holes = atomicExchange(&g_any[1], 0u) != 0u;
                if some && holes {
                    atomicOr(&g_masks[gb >> 5u], 1u << (gb & 31u));
                } else if some {
                    atomicOr(&g_masks[8u + (gb >> 5u)], 1u << (gb & 31u));
                }
            }
            workgroupBarrier();
            gb += 1u;
        }
        // The interval over all lanes.
        let natural_now = select(top, generated_top, changed && generated_top != NO_LAYER);
        var flags = 0u;
        if lane_top > range.x { flags |= 1u; }
        if lane_air { flags |= 2u; }
        if lane_layered { flags |= 4u; }
        if lane_top != clamp(natural_now, range.x, range.y) { flags |= 8u; }
        if lane_top - range.x > 255 { flags |= 16u; }
        atomicOr(&g_iv_flags[iv], flags);
        atomicMax(&g_iv_top[iv], lane_top);
        atomicOr(&g_tops[iv * 16u + (li >> 2u)], u32(clamp(lane_top - range.x, 0, 255)) << ((li & 3u) * 8u));
    }
    // Between intervals no lane changes state: each lane's kind at one cell
    // of a gap is its kind throughout it.
    for (var g = 0u; g <= plan.n; g++) {
        var k = plan.w_lo + WINDOW_CELLS;
        var empty = false;
        if plan.n != 0u {
            if g == 0u {
                k = plan.iv[0].x - 1;
                empty = plan.iv[0].x <= plan.floor || ((plan.clip & INFO_CLIP_BELOW) != 0u && plan.iv[0].x <= plan.w_lo);
            } else {
                k = plan.iv[g - 1u].y;
                empty = g == plan.n && (plan.clip & INFO_CLIP_ABOVE) != 0u && k >= plan.w_hi;
            }
        }
        if empty { continue; }
        var kind = latest_geometry(job.edits, level, vec3<i32>(ch, center_half(k, level)), column_point);
        if kind == NONE { kind = terrain_kind(top, k); }
        if kind == 1u { atomicOr(&g_gap[g * 2u + (li >> 5u)], 1u << (li & 31u)); }
    }
    // The natural surface: generated tops, whatever the edits did.
    let natural = select(top, generated_top, changed && generated_top != NO_LAYER);
    atomicMin(&g_natural[0], natural);
    workgroupBarrier();
    let volume_bits = atomicLoad(&g_volume);
    let volumetric = (volume_bits & VOLUME_TERRAIN) != 0u;
    // One cell under the lowest top.
    let natural_base = atomicLoad(&g_natural[0]) - 1;
    // The density describes the lane's surface only when its last zero
    // crossing is at the generated top (that air cell, or the one below when
    // the surface rose into it): an undercut below an untouched top crossed
    // lower, and its floor gave the top's relief and height (black specks
    // across overhang regions).
    let density_surface = (surface_changed || dense_lane) && top_air != NO_DENSITY
        && (crossing_air == generated_top || crossing_air + 1 == generated_top);
    // The exact surface's height over the natural top, Q16 level cells,
    // floored: the density's zero crossing, or the field's height (in
    // integers: whole base layers at levels up to 16). A relief column's top
    // cell is cut at its base layer (`relief_share`).
    var delta = 0;
    if density_surface {
        delta = i32(floor((f32(crossing_air - natural) + surface_crossing) * 65536.0));
    } else {
        let cell_mm = world.grid.y << level;
        let above = clamp(height - natural * cell_mm, -cell_mm, cell_mm);
        if level <= 16u {
            delta = div_floor(above << (16u - level), world.grid.y);
        } else {
            delta = div_floor(above, world.grid.y << (level - 16u));
        }
    }
    // The lane word (`lane_word` in common.wgsl).
    g_lane[li] = u32(clamp(natural - natural_base, 0, 0x7fff))
        | (u32(clamp(delta, -65536, 65535)) << 15u);
    atomicOr(&g_surface[li >> 2u], (u32(column.y) & 0xffu) << ((li & 3u) * 8u));
    workgroupBarrier();
    scratch[plan.base * UNIT_WORDS + li] = g_lane[li];
    if li < 16u { scratch[(plan.base + 4u) * UNIT_WORDS + li] = atomicLoad(&g_surface[li]); }
    if li == 0u {
        // The column's spans: gaps and intervals in order, uniform ones
        // merged, the solid below the first implicit and the air above the
        // top dropped.
        var kinds: array<u32, 32>;
        var starts: array<i32, 32>;
        var ends: array<i32, 32>;
        var refs: array<u32, 32>;
        // A BRICKS span's first evaluated brick.
        var bricks_from: array<u32, 32>;
        var count = 0u;
        var clip = plan.clip;
        let lo_bound = select(plan.floor, plan.w_lo, (clip & INFO_CLIP_BELOW) != 0u);
        let hi_bound = select(0x7fffffff, plan.w_hi, (clip & INFO_CLIP_ABOVE) != 0u);
        for (var g = 0u; g <= plan.n; g++) {
            var g_lo = lo_bound;
            if g > 0u { g_lo = plan.iv[g - 1u].y; }
            var g_hi = hi_bound;
            if g < plan.n { g_hi = plan.iv[g].x; }
            if g_lo < g_hi {
                let b0 = atomicLoad(&g_gap[g * 2u]);
                let b1 = atomicLoad(&g_gap[g * 2u + 1u]);
                var kind = SPAN_LANES;
                if (b0 | b1) == 0u { kind = SPAN_AIR; }
                if (b0 & b1) == 0xffffffffu { kind = SPAN_SOLID; }
                let same = count > 0u && kinds[count - 1u] == kind
                    && (kind != SPAN_LANES || (refs[count - 1u] >= 16u
                        && atomicLoad(&g_gap[(refs[count - 1u] - 16u) * 2u]) == b0
                        && atomicLoad(&g_gap[(refs[count - 1u] - 16u) * 2u + 1u]) == b1));
                if same {
                    ends[count - 1u] = g_hi;
                } else {
                    kinds[count] = kind;
                    starts[count] = g_lo;
                    ends[count] = g_hi;
                    refs[count] = 16u + g;
                    count += 1u;
                }
            }
            if g < plan.n {
                let f = atomicLoad(&g_iv_flags[g]);
                var kind = SPAN_BRICKS;
                if (f & 1u) == 0u { kind = SPAN_AIR; }
                else if (f & 2u) == 0u { kind = SPAN_SOLID; }
                else if (f & 12u) == 0u { kind = SPAN_NATURAL; }
                else if (f & 20u) == 0u { kind = SPAN_TOPS; }
                var range = plan.iv[g];
                var gb0 = 0u;
                for (var q = 0u; q < g; q++) { gb0 += u32(plan.iv[q].y - plan.iv[q].x) >> 3u; }
                if kind == SPAN_BRICKS {
                    // Its solid bottom bricks and air top bricks are spans of
                    // their own: rays cross them as one box, not brick by
                    // brick.
                    let nb = u32(range.y - range.x) >> 3u;
                    var lead = 0u;
                    while lead < nb && ((atomicLoad(&g_masks[8u + ((gb0 + lead) >> 5u)]) >> ((gb0 + lead) & 31u)) & 1u) != 0u { lead += 1u; }
                    var trail = 0u;
                    loop {
                        if lead + trail >= nb { break; }
                        let b = gb0 + nb - 1u - trail;
                        let occupied = ((atomicLoad(&g_masks[b >> 5u]) | atomicLoad(&g_masks[8u + (b >> 5u)])) >> (b & 31u)) & 1u;
                        if occupied != 0u { break; }
                        trail += 1u;
                    }
                    if lead > 0u {
                        if count > 0u && kinds[count - 1u] == SPAN_SOLID {
                            ends[count - 1u] = range.x + i32(lead) * 8;
                        } else {
                            kinds[count] = SPAN_SOLID;
                            starts[count] = range.x;
                            ends[count] = range.x + i32(lead) * 8;
                            refs[count] = g;
                            count += 1u;
                        }
                    }
                    kinds[count] = SPAN_BRICKS;
                    starts[count] = range.x + i32(lead) * 8;
                    ends[count] = range.y - i32(trail) * 8;
                    refs[count] = g;
                    bricks_from[count] = gb0 + lead;
                    count += 1u;
                    if trail > 0u {
                        kinds[count] = SPAN_AIR;
                        starts[count] = range.y - i32(trail) * 8;
                        ends[count] = range.y;
                        refs[count] = g;
                        count += 1u;
                    }
                } else if count > 0u && kinds[count - 1u] == kind && (kind == SPAN_AIR || kind == SPAN_SOLID || kind == SPAN_NATURAL) {
                    // Its top is the latest interval's.
                    ends[count - 1u] = range.y;
                    refs[count - 1u] = g;
                } else {
                    kinds[count] = kind;
                    starts[count] = range.x;
                    ends[count] = range.y;
                    refs[count] = g;
                    count += 1u;
                }
            }
        }
        // Air above everything (in theory always): a column whose last span
        // is not air yet reaches up forever describes only its window.
        if count > 0u && kinds[count - 1u] != SPAN_AIR && ends[count - 1u] == 0x7fffffff {
            ends[count - 1u] = plan.w_hi;
            clip |= INFO_CLIP_ABOVE;
        }
        var top_out = plan.w_hi;
        if (clip & INFO_CLIP_ABOVE) == 0u {
            while count > 0u && kinds[count - 1u] == SPAN_AIR { count -= 1u; }
            top_out = NO_LAYER >> level;
            if count > 0u {
                let e = count - 1u;
                top_out = ends[e];
                if kinds[e] != SPAN_SOLID && kinds[e] != SPAN_LANES { top_out = atomicLoad(&g_iv_top[refs[e]]); }
            } else if (clip & INFO_CLIP_BELOW) != 0u {
                top_out = plan.w_lo;
            }
        }
        // Solid below the first span is implicit.
        var first = 0u;
        while first < count && kinds[first] == SPAN_SOLID { first += 1u; }
        let n_spans = count - first;
        let heightfield = n_spans == 1u && kinds[first] == SPAN_NATURAL && clip == 0u && topology_flags == 0u;
        // Payload sizes, then the span area and the bricks after it.
        var words = 0u;
        if !heightfield {
            words = n_spans * 2u;
            for (var e = first; e < count; e++) {
                let nb = u32(ends[e] - starts[e]) >> 3u;
                switch kinds[e] {
                    case 2u: { words += 2u; }
                    case 3u: { words += 16u; }
                    case 5u: { words += 1u + 2u * ((nb + 31u) >> 5u); }
                    default: {}
                }
            }
        }
        let span_units = (words + UNIT_WORDS - 1u) / UNIT_WORDS;
        let table = (bricks_base + plan.eval) * UNIT_WORDS;
        var copied = array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
        var mixed_before = 0u;
        if !heightfield {
            var payload = n_spans * 2u;
            for (var e = first; e < count; e++) {
                let slot = e - first;
                scratch[table + slot * 2u] = bitcast<u32>(starts[e]);
                let kind = kinds[e];
                scratch[table + slot * 2u + 1u] = kind | select(0u, payload << 3u, kind == SPAN_LANES || kind == SPAN_TOPS || kind == SPAN_BRICKS);
                if kind == SPAN_LANES {
                    let g = refs[e] - 16u;
                    scratch[table + payload] = atomicLoad(&g_gap[g * 2u]);
                    scratch[table + payload + 1u] = atomicLoad(&g_gap[g * 2u + 1u]);
                    payload += 2u;
                } else if kind == SPAN_TOPS {
                    for (var w = 0u; w < 16u; w++) { scratch[table + payload + w] = atomicLoad(&g_tops[refs[e] * 16u + w]); }
                    payload += 16u;
                } else if kind == SPAN_BRICKS {
                    let gb0 = bricks_from[e];
                    let nb = u32(ends[e] - starts[e]) >> 3u;
                    let mask_words = (nb + 31u) >> 5u;
                    scratch[table + payload] = HEADER_UNITS_MAX + span_units + mixed_before;
                    for (var w = 0u; w < 2u * mask_words; w++) { scratch[table + payload + 1u + w] = 0u; }
                    for (var b = 0u; b < nb; b++) {
                        let g_bit = gb0 + b;
                        let is_mixed = ((atomicLoad(&g_masks[g_bit >> 5u]) >> (g_bit & 31u)) & 1u) != 0u;
                        let is_solid = ((atomicLoad(&g_masks[8u + (g_bit >> 5u)]) >> (g_bit & 31u)) & 1u) != 0u;
                        if is_mixed {
                            let at = table + payload + 1u + (b >> 5u);
                            scratch[at] = scratch[at] | (1u << (b & 31u));
                            copied[g_bit >> 5u] |= 1u << (g_bit & 31u);
                            mixed_before += 1u;
                        }
                        if is_solid {
                            let at = table + payload + 1u + mask_words + (b >> 5u);
                            scratch[at] = scratch[at] | (1u << (b & 31u));
                        }
                    }
                    payload += 1u + 2u * mask_words;
                }
            }
        }
        var out: JobOut;
        out.status = 0u;
        out.base = natural_base;
        out.info = select(n_spans, 0u, heightfield)
            | select(0u, INFO_RELIEF, relief)
            | select(0u, INFO_HEIGHTFIELD, heightfield) | topology_flags
            | select(0u, INFO_GENERATED, volumetric) | clip
            | select(0u, INFO_EDIT_MATERIALS, (volume_bits & VOLUME_MATERIALS) != 0u);
        out.n_mixed = mixed_before;
        out.scratch = plan.base;
        out.units = HEADER_UNITS_MAX + span_units;
        out.top = top_out;
        out.pad2 = 0u;
        out.lo = plan.w_lo;
        out.summary = max(plan.summary, top_out);
        out.n_eval = plan.eval;
        out.header = HEADER_UNITS_MAX;
        out.mixed = copied;
        job_out[index] = out;
    }
}

fn run_units(o: JobOut) -> u32 {
    return o.units + o.n_mixed;
}

fn class_of(units: u32) -> u32 {
    return 32u - countLeadingZeros(max(units, 1u) - 1u);
}

@compute @workgroup_size(64)
fn count(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= frame.counts.x || job_out[index].status != 0u { return; }
    let c = class_of(run_units(job_out[index]));
    job_out[index].size_class = c;
    atomicAdd(&alloc[A_NEED + c], 1);
}

@compute @workgroup_size(16)
fn refill(@builtin(local_invocation_index) c: u32) {
    if c >= CLASSES { return; }
    let need = atomicExchange(&alloc[A_NEED + c], 0);
    var top = atomicLoad(&alloc[A_TOP + c]);
    let size = 1u << c;
    let runs = PAGE_UNITS / size;
    let offset = class_offset(c);
    loop {
        if top >= need { break; }
        let page_index = atomicSub(&alloc[A_PAGES], 1) - 1;
        if page_index < 0 {
            atomicAdd(&alloc[A_PAGES], 1);
            break;
        }
        let page = free_pages[page_index];
        page_meta[page].size_class = c;
        atomicStore(&page_meta[page].free, runs);
        for (var r = 0u; r < runs; r++) {
            free_runs[offset + u32(top)] = page * PAGE_UNITS + r * size;
            top++;
        }
    }
    atomicStore(&alloc[A_TOP + c], top);
}

@compute @workgroup_size(64)
fn allocate(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= frame.counts.x || job_out[index].status != 0u { return; }
    let c = job_out[index].size_class;
    let slot = atomicSub(&alloc[A_TOP + c], 1) - 1;
    if slot < 0 {
        job_out[index].status = 3u;
        return;
    }
    let run = free_runs[class_offset(c) + u32(slot)];
    job_out[index].run = run;
    atomicSub(&page_meta[run / PAGE_UNITS].free, 1u);
}

@compute @workgroup_size(16)
fn fixup(@builtin(local_invocation_index) c: u32) {
    if c < CLASSES {
        atomicMax(&alloc[A_TOP + c], 0);
    }
    if c == 0u {
        atomicStore(&alloc[A_SCRATCH], 0);
    }
}

fn free_run(c: Column) {
    let cls = (c.info >> 18u) & 15u;
    let slot = atomicAdd(&alloc[A_TOP + cls], 1);
    free_runs[class_offset(cls) + u32(slot)] = c.run;
    atomicAdd(&page_meta[c.run / PAGE_UNITS].free, 1u);
}

@compute @workgroup_size(64)
fn evict(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame.counts.y { return; }
    let r = evictions[id.x];
    let c = records[r];
    if (c.info & INFO_VALID) != 0u {
        free_run(c);
        let ci = i32(c.key0 & 0xffffffu);
        let cj = bitcast<i32>(c.key1);
        for (var tier = 1u; tier <= 3u; tier++) {
            let bi = ci >> (2u * tier);
            let bj = cj >> (2u * tier);
            let slot = block_slot(c.key0 >> 27u, (c.key0 >> 24u) & 7u, tier, bi, bj) * 4u;
            if atomicLoad(&block_state[slot]) == bi && atomicLoad(&block_state[slot + 1u]) == bj {
                atomicSub(&block_state[slot + 3u], 1);
            }
        }
    }
    records[r].info = 0u;
    records[r].key0 = NONE;
}

// Table slot writes, stored as (slot, value) pairs after the eviction list.
@compute @workgroup_size(64)
fn patch_table(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame.extra.x { return; }
    let at = frame.counts.y + id.x * 2u;
    table[evictions[at]] = evictions[at + 1u];
}

// Summary block (re)initialisation: (slot, bi, bj) triples after the table
// patches. `bi = -1` releases a slot.
@compute @workgroup_size(64)
fn patch_blocks(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame.extra.z { return; }
    let at = frame.counts.y + frame.extra.x * 2u + id.x * 3u;
    let slot = evictions[at] * 4u;
    atomicStore(&block_state[slot], bitcast<i32>(evictions[at + 1u]));
    atomicStore(&block_state[slot + 1u], bitcast<i32>(evictions[at + 2u]));
    atomicStore(&block_state[slot + 2u], -0x3fffffff);
    atomicStore(&block_state[slot + 3u], 0);
    air_blocks[evictions[at]] = vec4<i32>(0);
}

// Bits [a, b) of the 32-bit word holding bits off..off+32 of a 64-bit mask.
fn mask_bits(a: i32, b: i32, off: i32) -> u32 {
    let lo = clamp(a - off, 0, 32);
    let hi = clamp(b - off, 0, 32);
    if hi <= lo { return 0u; }
    let width = u32(hi - lo);
    return select((1u << width) - 1u, 0xffffffffu, width == 32u) << u32(lo);
}

// Air bricks (8 layers from base + 8j, j < 64) wholly inside [lo, hi).
fn air_range(lo: i32, hi: i32, base: i32) -> vec2<u32> {
    let a = clamp((lo - base + 7) >> 3, 0, 64);
    let b = clamp((hi - base) >> 3, 0, 64);
    return vec2<u32>(mask_bits(a, b, 0), mask_bits(a, b, 32));
}

// A column's air bricks from base: air spans, air bricks of brick spans and
// the air above its top, inside the window it describes.
fn column_air(c: Column, base: i32) -> vec2<u32> {
    var m = vec2<u32>(0u);
    var known = -0x7fffffff;
    if (c.info & INFO_CLIP_BELOW) != 0u { known = c.lo; }
    if (c.info & INFO_CLIP_ABOVE) == 0u { m |= air_range(c.top, base + 512, base); }
    if (c.info & INFO_HEIGHTFIELD) != 0u { return m; }
    let table = span_table(c);
    let n = c.info & 31u;
    for (var e = 0u; e < n; e++) {
        let start = bitcast<i32>(pool[table + e * 2u]);
        let entry = pool[table + e * 2u + 1u];
        var end = c.top;
        if e + 1u < n { end = bitcast<i32>(pool[table + (e + 1u) * 2u]); }
        if end <= base || start >= base + 512 { continue; }
        let kind = entry & 7u;
        if kind == SPAN_AIR {
            m |= air_range(max(start, known), end, base);
        } else if kind == SPAN_BRICKS {
            let s = Span(kind, start, end, table + (entry >> 3u));
            let nb = u32(end - start) >> 3u;
            for (var b = 0u; b < nb; b++) {
                let k = start + i32(b) * 8;
                if k + 8 <= base || k >= base + 512 || k < known { continue; }
                if span_brick(c, s, b).x == 0u { m |= air_range(k, k + 8, base); }
            }
        }
    }
    return m;
}

var<workgroup> g_air: array<atomic<u32>, 2>;
var<workgroup> g_air_base: atomic<i32>;
var<workgroup> g_air_whole: atomic<u32>;

// After publication: the air entry of each job's tier-1 block from its 16
// columns (bricks air in all of them, over 512 layers ending just above
// the lowest natural top). A missing column leaves the block no air.
// Recomputed whenever a column of the block is generated (edits too).
@compute @workgroup_size(64)
fn air_blocks_build(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let index = job_index(wg);
    if index >= frame.counts.x { return; }
    let job = jobs[index];
    if job_out[index].status != 0u { return; }
    let level = job.key0 >> 27u;
    let face = (job.key0 >> 24u) & 7u;
    let bi = i32(job.key0 & 0xffffffu) >> 2u;
    let bj = bitcast<i32>(job.key1) >> 2u;
    if li == 0u {
        atomicStore(&g_air[0], 0xffffffffu);
        atomicStore(&g_air[1], 0xffffffffu);
        atomicStore(&g_air_base, 0x7fffffff);
        atomicStore(&g_air_whole, 1u);
    }
    workgroupBarrier();
    var col: Column;
    var have = false;
    if li < 16u {
        let c = vec2<i32>(bi * 4 + i32(li & 3u), bj * 4 + i32(li >> 2u));
        let record = find_column(column_key0(face, level, c.x), bitcast<u32>(c.y));
        if record != NONE {
            col = records[record];
            have = column_valid(col);
        }
        if have { atomicMin(&g_air_base, col.base); } else { atomicStore(&g_air_whole, 0u); }
    }
    workgroupBarrier();
    let base = ((atomicLoad(&g_air_base) - 504) >> 3u) << 3u;
    if have {
        let m = column_air(col, base);
        atomicAnd(&g_air[0], m.x);
        atomicAnd(&g_air[1], m.y);
    }
    workgroupBarrier();
    if li == 0u {
        var e = vec4<i32>(0);
        if atomicLoad(&g_air_whole) != 0u {
            e = vec4<i32>(base, bitcast<i32>(atomicLoad(&g_air[0])), bitcast<i32>(atomicLoad(&g_air[1])), air_key(bi, bj));
        }
        air_blocks[block_slot(level, face, 1u, bi, bj)] = e;
    }
}

@compute @workgroup_size(64)
fn publish(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let index = job_index(wg);
    if index >= frame.counts.x { return; }
    let job = jobs[index];
    let o = job_out[index];
    if o.status != 0u {
        // Report the failure (the CPU retries it); a new column stays
        // unpublished, a replaced column keeps its old data.
        if li == 0u {
            let at = u32(atomicAdd(&alloc[A_FAILS], 1)) * 4u;
            if at + 3u < arrayLength(&failures) {
                failures[at] = job.key0;
                failures[at + 1u] = job.key1;
                failures[at + 2u] = o.status;
            }
        }
        if li == 0u && (job.flags & 1u) == 0u {
            var c: Column;
            c.key0 = job.key0;
            c.key1 = job.key1;
            c.info = select(0u, INFO_OVERFLOW, o.status == 1u);
            records[job.record] = c;
        }
        return;
    }
    // The header, then the span area (after the evaluated bricks in
    // scratch), then the BRICKS spans' mixed bricks by rank.
    for (var w = li; w < o.header * UNIT_WORDS; w += 64u) {
        pool[o.run * UNIT_WORDS + w] = scratch[o.scratch * UNIT_WORDS + w];
    }
    let spans = o.units - o.header;
    for (var w = li; w < spans * UNIT_WORDS; w += 64u) {
        pool[(o.run + o.header) * UNIT_WORDS + w] = scratch[(o.scratch + HEADER_UNITS_MAX + o.n_eval) * UNIT_WORDS + w];
    }
    let total = select(o.n_eval * UNIT_WORDS, 0u, o.n_mixed == 0u);
    for (var w = li; w < total; w += 64u) {
        let b = w / UNIT_WORDS;
        let bit = b & 31u;
        let word = b >> 5u;
        if ((o.mixed[word] >> bit) & 1u) == 0u { continue; }
        var rank = countOneBits(o.mixed[word] & ((1u << bit) - 1u));
        for (var q = 0u; q < word; q++) { rank += countOneBits(o.mixed[q]); }
        pool[(o.run + o.units + rank) * UNIT_WORDS + (w % UNIT_WORDS)] =
            scratch[(o.scratch + HEADER_UNITS_MAX + b) * UNIT_WORDS + (w % UNIT_WORDS)];
    }
    if li == 0u {
        let previous = records[job.record];
        let was_published = (previous.info & INFO_VALID) != 0u && previous.key0 == job.key0 && previous.key1 == job.key1;
        if (job.flags & 1u) != 0u && (previous.info & INFO_VALID) != 0u {
            free_run(previous);
        }
        let level = job.key0 >> 27u;
        // A window clipped above bounds its summaries with every candidate.
        let clipped_above = (o.info & INFO_CLIP_ABOVE) != 0u;
        let top_cell = max(select(o.top, o.summary, clipped_above), -0x3fffffff >> level);
        let ci = i32(job.key0 & 0xffffffu);
        let cj = bitcast<i32>(job.key1);
        for (var tier = 1u; tier <= 3u; tier++) {
            let bi = ci >> (2u * tier);
            let bj = cj >> (2u * tier);
            let slot = block_slot(level, (job.key0 >> 24u) & 7u, tier, bi, bj) * 4u;
            if atomicLoad(&block_state[slot]) == bi && atomicLoad(&block_state[slot + 1u]) == bj {
                atomicMax(&block_state[slot + 2u], top_cell);
                if !was_published { atomicAdd(&block_state[slot + 3u], 1); }
            }
        }
        var c: Column;
        c.key0 = job.key0;
        c.key1 = job.key1;
        c.base = o.base;
        c.info = o.info | (o.size_class << 18u) | INFO_VALID;
        c.run = o.run;
        c.top = o.top;
        c.lo = o.lo;
        c.edits = job.edits;
        records[job.record] = c;
        atomicMax(&level_tops[level], top_cell << level);
        // Clipped columns are reported with their window's bottom and the
        // sides they clip, so the CPU regenerates them when the eye nears
        // a clipped side.
        let sides = select(0u, 1u, (o.info & INFO_CLIP_BELOW) != 0u) | select(0u, 2u, (o.info & INFO_CLIP_ABOVE) != 0u);
        if sides != 0u {
            let at = u32(atomicAdd(&alloc[A_FAILS], 1)) * 4u;
            if at + 3u < arrayLength(&failures) {
                failures[at] = job.key0;
                failures[at + 1u] = job.key1;
                failures[at + 2u] = STATUS_CLIPPED + sides - 1u;
                failures[at + 3u] = bitcast<u32>(o.lo);
            }
        }
    }
}

// Suffix maxima of the per-level occupied tops (entries 32.. hold the max of
// this level and every coarser one) for early sky exits.
@compute @workgroup_size(1)
fn level_suffix() {
    var best = -0x7fffffff;
    for (var level = 31i; level >= 0; level--) {
        best = max(best, atomicLoad(&level_tops[level]));
        atomicStore(&level_tops[32 + level], best);
    }
}
