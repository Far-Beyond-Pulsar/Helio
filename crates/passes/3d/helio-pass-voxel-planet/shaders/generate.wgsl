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
    centre: i32,   // clipped windows: the eye layer the window is centred on
    lo: i32,       // clipped below: the window bottom
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
const HEADER_UNITS_MAX: u32 = 6u;
const SPAN_UNITS_MAX: u32 = 12u;
// Candidate intervals gathered per column; more fall back to one interval
// over all of them. At most MAX_INTERVALS remain after merging, so a
// column has at most 15 spans.
const RAW_INTERVALS: u32 = 128u;
const MAX_INTERVALS: u32 = 7u;
// Readback status of a published clipped column (with its window centre).
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
    centre: i32,
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

// Candidate intervals: where some lane's occupancy can change.
var<workgroup> g_terrain: array<atomic<i32>, 2>;
var<workgroup> g_raw: array<vec2<i32>, 128>;
var<workgroup> g_raw_count: atomic<u32>;
var<workgroup> g_raw_bounds: array<atomic<i32>, 2>;
// A batch of eight brushes' crossings over the column's lanes: lowest and
// highest first cell inside, lowest and highest last cell inside, margin,
// any lane crossing.
var<workgroup> g_bnd: array<atomic<i32>, 48>;
var<workgroup> g_plan: Plan;
// Per merged interval: lane summary bits (some solid 1, some air 2, not
// solid-below-air 4, tops not the natural ones 8, a top over a byte 16), the
// highest solid layer + 1 and each lane's top over the interval's start.
var<workgroup> g_iv_flags: array<atomic<u32>, 8>;
var<workgroup> g_iv_top: array<atomic<i32>, 8>;
var<workgroup> g_tops: array<atomic<u32>, 128>;
// Solid lanes of each gap between intervals (two words per gap).
var<workgroup> g_gap: array<atomic<u32>, 18>;
// Natural tops: lowest, highest (level cells), highest base-cell top.
var<workgroup> g_natural: array<atomic<i32>, 3>;
var<workgroup> g_words: array<atomic<u32>, 32>;
var<workgroup> g_masks: array<atomic<u32>, 16>;
var<workgroup> g_any: array<atomic<u32>, 2>;
var<workgroup> g_fraction: array<atomic<u32>, 32>;
var<workgroup> g_topology_flags: u32;
var<workgroup> g_volume: atomic<u32>;
var<workgroup> g_surface: array<atomic<u32>, 16>;
// Surface offsets (`column_surface_offset`), one byte per lane.
var<workgroup> g_offset: array<atomic<u32>, 16>;
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

// Q16 fraction of a cell whose surface lies `h` cells above its bottom (0:
// the whole cell).
fn cell_fraction(h: f32) -> u32 {
    if h >= 1.0 { return 0u; }
    return clamp(u32(h * 65536.0), 1u, 65535u);
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

fn append_candidate(iv: vec2<i32>) {
    atomicMin(&g_raw_bounds[0], iv.x);
    atomicMax(&g_raw_bounds[1], iv.y);
    let n = atomicAdd(&g_raw_count, 1u);
    if n < RAW_INTERVALS { g_raw[n] = iv; }
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
        atomicStore(&g_terrain[0], 0x7fffffff);
        atomicStore(&g_terrain[1], -0x7fffffff);
        atomicStore(&g_raw_count, 0u);
        atomicStore(&g_raw_bounds[0], 0x7fffffff);
        atomicStore(&g_raw_bounds[1], -0x7fffffff);
        atomicStore(&g_natural[0], 0x7fffffff);
        atomicStore(&g_natural[1], -0x7fffffff);
        atomicStore(&g_natural[2], -0x7fffffff);
        atomicStore(&g_any[0], 0u);
        atomicStore(&g_any[1], 0u);
        atomicStore(&g_volume, 0u);
    }
    if li < 16u {
        atomicStore(&g_words[li], 0u);
        atomicStore(&g_words[li + 16u], 0u);
        atomicStore(&g_masks[li], 0u);
        atomicStore(&g_surface[li], 0u);
        atomicStore(&g_offset[li], 0u);
    }
    if li < 18u { atomicStore(&g_gap[li], 0u); }
    if li < 8u {
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
    var fraction = 0u;
    if relief && remainder != 0u {
        // Shifts avoid overflowing a u32 product at planetary coarse levels.
        if level <= 16u { fraction = remainder << (16u - level); }
        else { fraction = max(remainder >> (level - 16u), 1u); }
    }
    if relief && li < 32u { atomicStore(&g_fraction[li], 0u); }
    // Volumetric terrain (caves, overhangs): the program may change cells
    // within its extent around the heightfield top, evaluated in 3D. Only
    // the cells it changes make generated volume: a column whose cells all
    // keep the heightfield's kinds (most of a cave region's rock, ground too
    // flat to lean) stays a heightfield column. Pass 1 finds the lane's
    // changed cells; they and one more on each side are evaluated (the
    // generated surface's relief reads the densities around it).
    let field_top = base_top >> level;
    let leaning = lean.x != 0 && extent.y > 0;
    var changed_lo = 0x7fffffff;
    var changed_hi = -0x7fffffff;
    for (var k = field_top - extent.x; k < field_top + extent.y; k++) {
        if (generated_density(column_point, face, i, j, k, level, field_top, height, leaning, lean.y, lean_node) > 0) != (k < field_top) {
            changed_lo = min(changed_lo, k);
            changed_hi = max(changed_hi, k);
        }
    }
    let changed = changed_lo <= changed_hi;
    let eval_lo = select(0, max(changed_lo - 1, field_top - extent.x), changed);
    let eval_hi = select(-1, min(changed_hi + 1, field_top + extent.y - 1), changed);
    if changed { atomicOr(&g_volume, VOLUME_TERRAIN); }
    // The terrain's candidates: the cells around each lane's top whose kind
    // the heightfield sets, and the volume's evaluated cells.
    var t_lo = min(top, field_top) - 1;
    var t_hi = max(top, field_top) + 1;
    if changed {
        t_lo = min(t_lo, eval_lo - 1);
        t_hi = max(t_hi, eval_hi + 2);
    }
    atomicMin(&g_terrain[0], t_lo);
    atomicMax(&g_terrain[1], t_hi);
    let ch = vec2<i32>(center_half(i, level), center_half(j, level));
    let edit_counts = workgroupUniformLoad(&g_edit_counts);
    let n_brushes = edit_counts.x + edit_counts.y;
    if job.edits != 0u {
        // Baked bricks: each a candidate.
        let baked_base = job.edits + 2u + n_brushes;
        for (var e = li; e < edit_counts.z; e += 64u) {
            let bk = bitcast<i32>(edit_refs[baked_base + e * 2u]);
            append_candidate(vec2<i32>(bk * 8, bk * 8 + 8));
            atomicOr(&g_volume, VOLUME_MATERIALS);
        }
        for (var e = li; e < n_brushes; e += 64u) {
            let b = edit_brush(job.edits, e);
            // Shading reads brush materials only where some brush sets one.
            if b.radius_half >= (1u << level) && ((b.flags >> 4u) & 3u) != 0u { atomicOr(&g_volume, VOLUME_MATERIALS); }
        }
    }
    // Brush surfaces, eight brushes per pass: every lane solves where it
    // crosses each brush, and the column keeps, per brush, the range of the
    // crossings from below and from above. Between them no lane changes
    // state for that brush, however tall (a dig's carved air, a crater's
    // wall).
    for (var first = 0u; first < n_brushes; first += 8u) {
        if li < 48u {
            let f = li % 6u;
            let init = select(select(0, -0x7fffffff, f == 1u || f == 3u), 0x7fffffff, f == 0u || f == 2u);
            atomicStore(&g_bnd[li], init);
        }
        workgroupBarrier();
        for (var s = 0u; s < 8u; s++) {
            let e = first + s;
            if e >= n_brushes { break; }
            let b = edit_brush(job.edits, e);
            if b.radius_half < (1u << level) || ((b.flags >> 4u) & 3u) == 2u { continue; }
            let crossing = brush_crossings(b, column_point, ch, level);
            if crossing.hit {
                atomicMin(&g_bnd[s * 6u], crossing.lo);
                atomicMax(&g_bnd[s * 6u + 1u], crossing.lo);
                atomicMin(&g_bnd[s * 6u + 2u], crossing.hi);
                atomicMax(&g_bnd[s * 6u + 3u], crossing.hi);
                atomicMax(&g_bnd[s * 6u + 4u], crossing.margin);
                atomicStore(&g_bnd[s * 6u + 5u], 1);
            }
        }
        workgroupBarrier();
        if li < 8u && atomicLoad(&g_bnd[li * 6u + 5u]) != 0 {
            let m = atomicLoad(&g_bnd[li * 6u + 4u]);
            append_candidate(vec2<i32>(atomicLoad(&g_bnd[li * 6u]) - m, atomicLoad(&g_bnd[li * 6u + 1u]) + m + 1));
            append_candidate(vec2<i32>(atomicLoad(&g_bnd[li * 6u + 2u]) - m, atomicLoad(&g_bnd[li * 6u + 3u]) + m + 1));
        }
        workgroupBarrier();
    }
    workgroupBarrier();
    // Lane 0 merges the candidates into whole-brick intervals, clips them to
    // the window around the eye's layer and reserves the scratch.
    let centre = frame.layer_i.x >> level;
    if li == 0u {
        var plan: Plan;
        let raw = atomicLoad(&g_raw_count);
        var count = min(raw, RAW_INTERVALS);
        if raw > RAW_INTERVALS {
            g_raw[0] = vec2<i32>(atomicLoad(&g_raw_bounds[0]), atomicLoad(&g_raw_bounds[1]));
            count = 1u;
        }
        g_raw[count] = vec2<i32>(atomicLoad(&g_terrain[0]), atomicLoad(&g_terrain[1]));
        count += 1u;
        for (var e = 0u; e < count; e++) {
            let iv = g_raw[e];
            g_raw[e] = vec2<i32>((iv.x >> 3u) << 3u, ((iv.y + 7) >> 3u) << 3u);
        }
        // Insertion sort by start, then merge intervals that overlap or are
        // at most a brick apart.
        for (var e = 1u; e < count; e++) {
            let iv = g_raw[e];
            var m = e;
            while m > 0u && g_raw[m - 1u].x > iv.x {
                g_raw[m] = g_raw[m - 1u];
                m -= 1u;
            }
            g_raw[m] = iv;
        }
        var n = 0u;
        var summary = -0x7fffffff;
        for (var e = 0u; e < count; e++) {
            let iv = g_raw[e];
            summary = max(summary, iv.y);
            if n > 0u && iv.x <= g_raw[n - 1u].y + 8 {
                g_raw[n - 1u].y = max(g_raw[n - 1u].y, iv.y);
            } else {
                g_raw[n] = iv;
                n += 1u;
            }
        }
        // The window around the eye's layer. The coarsest level is the
        // coverage every ray falls back to: its columns describe their
        // whole radial line (a few dozen bricks from the core up).
        var w_lo = ((centre - WINDOW_CELLS) >> 3u) << 3u;
        var w_hi = w_lo + 2 * WINDOW_CELLS;
        if level + 1u >= u32(frame.layer_i.z) {
            w_lo = -(1 << 29);
            w_hi = 1 << 29;
        }
        // Volume points reach zero at the planet's centre: (1 + ratio) = 0,
        // the ratio being h * scale.y >> scale.z in Q30 (half layers h).
        var floor = NO_LAYER;
        if !is_plane() {
            let core_half = -i32(round(exp2(30.0 + f32(world.scale.z)) / f32(world.scale.y)));
            floor = ((core_half >> (level + 1u)) >> 3u) << 3u;
        }
        var clip = 0u;
        var kept = 0u;
        for (var e = 0u; e < n; e++) {
            var iv = g_raw[e];
            if iv.y <= floor { continue; }
            iv.x = max(iv.x, floor);
            if iv.y <= w_lo { clip |= INFO_CLIP_BELOW; continue; }
            if iv.x >= w_hi { clip |= INFO_CLIP_ABOVE; continue; }
            if iv.x < w_lo { iv.x = w_lo; clip |= INFO_CLIP_BELOW; }
            if iv.y > w_hi { iv.y = w_hi; clip |= INFO_CLIP_ABOVE; }
            g_raw[kept] = iv;
            kept += 1u;
        }
        // At most MAX_INTERVALS: close the smallest gaps.
        while kept > MAX_INTERVALS {
            var best = 0u;
            for (var e = 1u; e + 1u < kept; e++) {
                if g_raw[e + 1u].x - g_raw[e].y < g_raw[best + 1u].x - g_raw[best].y { best = e; }
            }
            g_raw[best].y = g_raw[best + 1u].y;
            for (var e = best + 1u; e + 1u < kept; e++) { g_raw[e] = g_raw[e + 1u]; }
            kept -= 1u;
        }
        var eval = 0u;
        for (var e = 0u; e < 8u; e++) {
            if e < kept {
                plan.iv[e] = g_raw[e];
                eval += u32(g_raw[e].y - g_raw[e].x) >> 3u;
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
        plan.centre = centre;
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
    var top_fraction = 0u;
    // Height (level cells) of the generated surface: the densities' zero
    // crossing between the highest solid cell's centre and the air above.
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
                if k >= eval_lo && k <= eval_hi {
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
                        top_fraction = 0u;
                    } else if top_solid != NO_DENSITY && top_air == NO_DENSITY {
                        top_air = density;
                        let t = select(0.5, volume_crossing(top_solid, density), top_solid > 0);
                        surface_crossing = f32(k) - 0.5 + t;
                        crossing_air = k;
                        if relief && t > 0.5 {
                            // The surface rises into this cell: solid, cut there.
                            kind = 1u;
                            top_fraction = cell_fraction(t - 0.5);
                        } else {
                            top_fraction = cell_fraction(0.5 + t);
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
        var k = centre;
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
    atomicMax(&g_natural[1], natural);
    atomicMax(&g_natural[2], base_top);
    workgroupBarrier();
    let volume_bits = atomicLoad(&g_volume);
    let volumetric = (volume_bits & VOLUME_TERRAIN) != 0u;
    // One cell under the lowest top: an inline top rounded up to its level
    // cell stays above the base in base cells.
    let natural_base = atomicLoad(&g_natural[0]) - 1;
    // When the column's authored tops fit 255 base cells, store them in the
    // tops' bytes instead of allocating two Q16 units.
    let inline_relief = relief && !volumetric && level <= 7u
        && atomicLoad(&g_natural[2]) - (natural_base << level) <= 255;
    let wide_relief = relief && !inline_relief;
    let tops_wide = !inline_relief && atomicLoad(&g_natural[1]) - natural_base > 255;
    let surface_words = world.sphere.w != 0u;
    let tops_units = select(1u, 2u, tops_wide);
    let surface_unit = tops_units + select(0u, 2u, wide_relief);
    let offset_unit = surface_unit + select(0u, 1u, surface_words);
    let header_units = offset_unit + 1u;
    if tops_wide {
        atomicOr(&g_words[li >> 1u], u32(natural - natural_base) << ((li & 1u) * 16u));
    } else {
        let stored = select(natural - natural_base, base_top - (natural_base << level), inline_relief);
        atomicOr(&g_words[li >> 2u], u32(clamp(stored, 0, 255)) << ((li & 3u) * 8u));
    }
    if surface_words {
        atomicOr(&g_surface[li >> 2u], (u32(column.y) & 0xffu) << ((li & 3u) * 8u));
    }
    // The density describes the lane's surface only when its last zero
    // crossing is at the generated top (that air cell, or the one below when
    // the surface rose into it): an undercut below an untouched top crossed
    // lower, and its floor gave the top's relief and height (black specks
    // across overhang regions).
    let density_surface = (surface_changed || dense_lane) && top_air != NO_DENSITY
        && (crossing_air == generated_top || crossing_air + 1 == generated_top);
    if wide_relief {
        let lane_fraction = select(fraction, top_fraction, volumetric && density_surface);
        atomicOr(&g_fraction[li >> 1u], lane_fraction << ((li & 1u) * 16u));
    }
    // Surface offset: the exact surface's height over the stored one, from -1
    // to 1 units of its precision in 128ths (`column_surface_offset`): base
    // cells over the relief's base-cell top or the level-0 top, level cells
    // over a whole-cell top. Only shading reads it (smooth normals and
    // material height at every level); occupancy keeps the voxels' grid.
    // A density surface takes it from its zero crossing over its generated
    // top (level cells, as a level-0 or whole-cell top is stored); a relief
    // fraction carries that precision itself.
    var offset = 128;
    if density_surface {
        if level == 0u || !relief {
            offset = 128 + i32(round((surface_crossing - f32(generated_top)) * 128.0));
        }
    } else {
        let base_units = relief || level == 0u;
        let stored = select(top << level, base_top, base_units);
        let unit = select(world.grid.y << level, world.grid.y, base_units);
        let above = clamp(height - stored * world.grid.y, -unit, unit);
        offset = 128 + i32(floor(f32(above) * 128.0 / f32(unit)));
    }
    atomicOr(&g_offset[li >> 2u], u32(clamp(offset, 0, 255)) << ((li & 3u) * 8u));
    workgroupBarrier();
    if li < 32u && (tops_wide || li < 16u) {
        scratch[plan.base * UNIT_WORDS + li] = atomicLoad(&g_words[li]);
    }
    if wide_relief && li < 32u {
        scratch[(plan.base + tops_units) * UNIT_WORDS + li] = atomicLoad(&g_fraction[li]);
    }
    if surface_words && li < 16u {
        scratch[(plan.base + surface_unit) * UNIT_WORDS + li] = atomicLoad(&g_surface[li]);
    }
    if li < 16u { scratch[(plan.base + offset_unit) * UNIT_WORDS + li] = atomicLoad(&g_offset[li]); }
    if li == 0u {
        // The column's spans: gaps and intervals in order, uniform ones
        // merged, the solid below the first implicit and the air above the
        // top dropped.
        var kinds: array<u32, 16>;
        var starts: array<i32, 16>;
        var ends: array<i32, 16>;
        var refs: array<u32, 16>;
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
                let range = plan.iv[g];
                if count > 0u && kinds[count - 1u] == kind && (kind == SPAN_AIR || kind == SPAN_SOLID || kind == SPAN_NATURAL) {
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
                    // The interval's bricks in scratch start after those of
                    // the intervals below it.
                    var gb0 = 0u;
                    for (var q = 0u; q < refs[e]; q++) { gb0 += u32(plan.iv[q].y - plan.iv[q].x) >> 3u; }
                    let nb = u32(ends[e] - starts[e]) >> 3u;
                    let mask_words = (nb + 31u) >> 5u;
                    scratch[table + payload] = header_units + span_units + mixed_before;
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
        out.info = select(n_spans, 0u, heightfield) | select(0u, INFO_TOPS_WIDE, tops_wide)
            | select(0u, INFO_RELIEF, relief) | select(0u, INFO_RELIEF_INLINE, inline_relief)
            | select(0u, INFO_HEIGHTFIELD, heightfield) | topology_flags
            | select(0u, INFO_GENERATED, volumetric) | clip
            | select(0u, INFO_EDIT_MATERIALS, (volume_bits & VOLUME_MATERIALS) != 0u);
        out.n_mixed = mixed_before;
        out.scratch = plan.base;
        out.units = header_units + span_units;
        out.top = top_out;
        out.centre = centre;
        out.lo = plan.w_lo;
        out.summary = max(plan.summary, top_out);
        out.n_eval = plan.eval;
        out.header = header_units;
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
        // Clipped columns are reported with their window centre, so the CPU
        // can regenerate them when the eye leaves the window.
        if (o.info & (INFO_CLIP_BELOW | INFO_CLIP_ABOVE)) != 0u {
            let at = u32(atomicAdd(&alloc[A_FAILS], 1)) * 4u;
            if at + 3u < arrayLength(&failures) {
                failures[at] = job.key0;
                failures[at + 1u] = job.key1;
                failures[at + 2u] = STATUS_CLIPPED;
                failures[at + 3u] = bitcast<u32>(o.centre);
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
