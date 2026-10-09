// GPU-driven column generation, run allocation and publication.
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
    k_lo: i32,
    n_band: u32,
    n_mixed: u32,
    scratch: u32,
    size_class: u32,
    run: u32,
    pad: u32,      // record info flags
    top: i32,      // first air above every occupied cell, clipped or not
    centre: i32,   // clipped bands: the eye layer the window is centred on
    mixed: array<u32, 8>,
    solid: array<u32, 8>,
}

const CLASSES: u32 = 10u;
const PAGE_UNITS: u32 = 512u;
const A_TOP: u32 = 0u;
const A_FAILS: u32 = 10u;
const A_NEED: u32 = 16u;
const A_PAGES: u32 = 30u;
const A_SCRATCH: u32 = 31u;
const MAX_BAND: u32 = 256u;
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

// A column's stored band: the highest top of its lanes, the brick range
// kept (after clipping to MAX_BAND), the clip flags and the lowest natural
// top of its lanes.
struct Band {
    hi_cell: i32,
    k_lo: i32,
    k_hi: i32,
    clip: u32,
    top_min: i32,
}

fn class_offset(c: u32) -> u32 {
    let p = frame.counts.w;
    if c == 0u { return 0u; }
    return 2u * p - (p >> (c - 1u));
}

fn job_index(wg: vec3<u32>) -> u32 {
    return wg.x + wg.y * 32768u;
}

var<workgroup> g_band: array<atomic<i32>, 2>;
// Lowest natural top of the column's lanes (whether tops fit counting down).
var<workgroup> g_top_min: atomic<i32>;
var<workgroup> g_words: array<atomic<u32>, 16>;
var<workgroup> g_masks: array<atomic<u32>, 16>;
var<workgroup> g_any: array<atomic<u32>, 2>;
var<workgroup> g_base: u32;
// The column's band, reduced from the atomic bounds once (`Band`).
var<workgroup> g_band_out: Band;
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
        atomicStore(&g_band[0], 0x7fffffff);
        atomicStore(&g_band[1], -0x7fffffff);
        atomicStore(&g_top_min, 0x7fffffff);
        atomicStore(&g_any[0], 0u);
        atomicStore(&g_any[1], 0u);
        atomicStore(&g_volume, 0u);
    }
    if li < 16u {
        atomicStore(&g_words[li], 0u);
        atomicStore(&g_masks[li], 0u);
    }
    if li < 16u {
        atomicStore(&g_surface[li], 0u);
        atomicStore(&g_offset[li], 0u);
    }
    // This replaces the existing initialization barrier; the edit list is
    // scanned once per workgroup, with no extra terrain query or barrier.
    let topology_flags = workgroupUniformLoad(&g_topology_flags);
    // An edit changes occupancy, not the display field of untouched lanes.
    // Mixed topology retains bitmap storage; continuous fractions require a
    // heightfield, so only their metadata is disabled for topology brushes.
    let display_base = (frame.hints.w & 8u) != 0u && level >= 1u;
    let requested_relief = display_base && topology_flags == 0u;
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
    if requested_relief && remainder != 0u { top += 1; }
    var fraction = 0u;
    if requested_relief && remainder != 0u {
        // Shifts avoid overflowing a u32 product at planetary coarse levels.
        if level <= 16u { fraction = remainder << (16u - level); }
        else { fraction = max(remainder >> (level - 16u), 1u); }
    }
    // The following band reduction barrier completes this clear before any
    // lane writes its packed fraction.
    if requested_relief && li < 32u { atomicStore(&g_fraction[li], 0u); }
    // Everything below the band is solid ground, everything above is air.
    // The band follows the terrain wherever it is: clamping its top to the
    // datum (a sea-level leftover) made every column below datum claim the
    // air up to height 0 as occupied (column tops, summary blocks and level
    // tops), so rays stepped cell by cell through it (5-20x primary cost in
    // lowland below datum) and each column stored the empty bricks.
    atomicMin(&g_band[0], top - 1);
    atomicMax(&g_band[1], top);
    atomicMin(&g_top_min, top);
    // Volumetric terrain (caves, overhangs): the program may change cells
    // within its extent around the heightfield top, evaluated in 3D. Only
    // the cells it changes make generated volume: a column whose cells all
    // keep the heightfield's kinds (most of a cave region's rock, ground too
    // flat to lean) stays a heightfield column, with its band, relief and
    // storage. Pass 1 finds the lane's changed cells; the band loop
    // evaluates them and one more on each side (the generated surface's
    // relief reads the densities around it).
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
    if changed {
        atomicMin(&g_band[0], eval_lo - 1);
        atomicMax(&g_band[1], eval_hi + 1);
        atomicOr(&g_volume, VOLUME_TERRAIN);
    }
    if job.edits != 0u {
        let n = edit_counts(job.edits);
        // Baked bricks may hold air below the terrain or solid above it.
        let base = job.edits + 2u + (n.large + n.recent) * 12u;
        for (var e = li; e < n.baked; e += 64u) {
            let bk = bitcast<i32>(edit_refs[base + e * 2u]);
            atomicMin(&g_band[0], bk * 8 - 1);
            atomicMax(&g_band[1], bk * 8 + 9);
            atomicOr(&g_volume, VOLUME_MATERIALS);
        }
        for (var e = li; e < n.large + n.recent; e += 64u) {
            let b = edit_brush(job.edits, e);
            if b.radius_half < (1u << level) { continue; }
            let shift = level + 1u;
            let lo = b.k_lo >> shift;
            let hi = (b.k_hi >> shift) + 1;
            let op = (b.flags >> 4u) & 3u;
            if op == 0u { atomicMin(&g_band[0], lo - 1); }
            if op == 1u { atomicMax(&g_band[1], hi + 1); }
            // Shading reads brush materials only where some brush sets one.
            if op != 0u { atomicOr(&g_volume, VOLUME_MATERIALS); }
        }
    }
    workgroupBarrier();
    // Lane 0 reduces the atomic bounds and broadcasts the band: the brick
    // loop below holds barriers, and FXC rejects barriers under control flow
    // that depends on per-lane atomic loads.
    let centre = frame.layer_i.x >> level;
    if li == 0u {
        let lo_cell = atomicLoad(&g_band[0]);
        let hi_cell = atomicLoad(&g_band[1]);
        var k_lo = lo_cell >> 3u;
        var k_hi = ((max(hi_cell, lo_cell + 1) - 1) >> 3u) + 1;
        // A band taller than MAX_BAND bricks (deep digs, deep caves, cliffs)
        // keeps the window of MAX_BAND bricks around the eye's layer at this
        // level: rays beyond a clipped side use coarser levels, whose windows
        // reach twice as far, and the CPU regenerates the column when the eye
        // moves a quarter window vertically. Any depth stays representable.
        var clip = 0u;
        if k_hi - k_lo > i32(MAX_BAND) {
            let lo = clamp((centre >> 3u) - i32(MAX_BAND / 2u), k_lo, k_hi - i32(MAX_BAND));
            if lo > k_lo { clip |= INFO_CLIP_BELOW; }
            if lo + i32(MAX_BAND) < k_hi { clip |= INFO_CLIP_ABOVE; }
            k_lo = lo;
            k_hi = lo + i32(MAX_BAND);
        }
        g_band_out = Band(hi_cell, k_lo, k_hi, clip, atomicLoad(&g_top_min));
    }
    let band = workgroupUniformLoad(&g_band_out);
    let hi_cell = band.hi_cell;
    let k_lo = band.k_lo;
    let k_hi = band.k_hi;
    let clip = band.clip;
    let n_band = u32(k_hi - k_lo);
    // An edited column whose natural tops no longer fit a byte above the band
    // base (a deep dig) counts them down from the band top when they fit
    // there (`INFO_TOPS_DOWN`).
    let tops_down = topology_flags != 0u && n_band >= 32u && (clip & INFO_CLIP_ABOVE) == 0u
        && k_hi * 8 - band.top_min <= 255;
    let volume_bits = atomicLoad(&g_volume);
    let volumetric = (volume_bits & VOLUME_TERRAIN) != 0u;
    // Generated caves and overhangs keep the relief of the natural surface
    // they leave intact (their tops count down from the band top, which a
    // band of at most 32 bricks always fits).
    let relief = requested_relief && n_band <= 32u && (volumetric || hi_cell - k_lo * 8 <= 255);
    let heightfield = topology_flags == 0u && !volumetric && n_band <= 32u && hi_cell - k_lo * 8 <= 255;
    if requested_relief && !relief { top = base_top >> level; }
    // When the whole column fits in 255 authored layers, store its exact
    // base-grid top in the existing byte instead of allocating two Q16 units.
    // The bound includes the ceil top, so every lane is guaranteed to fit.
    let inline_relief = relief && !volumetric && level <= 7u && hi_cell - k_lo * 8 <= i32(255u >> level);
    let wide_relief = relief && !inline_relief;
    let surface_words = world.sphere.w != 0u;
    let scratch_surface = select(1u, 3u, wide_relief);
    let scratch_offset = scratch_surface + select(0u, 1u, surface_words);
    let scratch_header = scratch_offset + 1u;
    if li == 0u {
        let need = i32(scratch_header + n_band);
        let base = atomicAdd(&alloc[A_SCRATCH], need);
        if u32(base + need) * UNIT_WORDS > arrayLength(&scratch) {
            g_base = NONE;
        } else {
            g_base = u32(base);
        }
    }
    let base = workgroupUniformLoad(&g_base);
    if base == NONE {
        if li == 0u { job_out[index].status = 2u; }
        return;
    }
    // A volumetric column's fractions follow its generation (below).
    if wide_relief && !volumetric { atomicOr(&g_fraction[li >> 1u], fraction << ((li & 1u) * 16u)); }
    // Fractions and byte-packed tops share the existing publication barrier.
    // Disabled metadata performs no fraction atomics or extra barriers.
    var stored_top = select(top - k_lo * 8, base_top - ((k_lo * 8) << level), inline_relief);
    if tops_down { stored_top = k_hi * 8 - top; }
    atomicOr(&g_words[li >> 2u], u32(clamp(stored_top, 0, 255)) << ((li & 3u) * 8u));
    if surface_words {
        let word = u32(column.y) & 0xffu;
        atomicOr(&g_surface[li >> 2u], word << ((li & 3u) * 8u));
    }
    workgroupBarrier();
    if surface_words && li < 16u {
        scratch[(base + scratch_surface) * UNIT_WORDS + li] = atomicLoad(&g_surface[li]);
    }
    if wide_relief && !volumetric && li < 32u {
        scratch[(base + 1u) * UNIT_WORDS + li] = atomicLoad(&g_fraction[li]);
    }
    if li < 16u {
        scratch[base * UNIT_WORDS + li] = atomicLoad(&g_words[li]);
        atomicStore(&g_words[li], 0u);
    }
    workgroupBarrier();
    // Generated top of a lane: first air above its highest solid cell before
    // edits (overhang lips, cave openings).
    var generated_top = k_lo * 8;
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
    let edit_counts = workgroupUniformLoad(&g_edit_counts);
    let ch = vec2<i32>(center_half(i, level), center_half(j, level));
    // The column's footprint in half cells, for culling brushes per brick.
    let half_cell = 1 << (level + 1u);
    let foot_lo = vec2<i32>(ci, cj) * 8 * half_cell;
    let foot_hi = foot_lo + 8 * half_cell;
    for (var b = 0u; b < n_band; b++) {
        var kinds: array<u32, 8>;
        for (var z = 0u; z < 8u; z++) {
            let k = (k_lo + i32(b)) * 8 + i32(z);
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
                    if requested_relief && t > 0.5 {
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
        let brick_lo = ((k_lo + i32(b)) * 8) * half_cell;
        let brick_hi = brick_lo + 8 * half_cell;
        for (var part = 0u; part < 2u; part++) {
            if part == 1u && edit_counts.z != 0u {
                let n = EditCounts(edit_counts.x, edit_counts.y, edit_counts.z);
                for (var z = 0u; z < 8u; z++) {
                    let cell = baked_cell(job.edits, n, i, j, (k_lo + i32(b)) * 8 + i32(z)) & 3u;
                    if cell == BAKED_AIR { kinds[z] = 0u; }
                    else if cell == BAKED_SOLID { kinds[z] = 1u; }
                }
            }
            let first = select(0u, edit_counts.x, part == 1u);
            let end = select(edit_counts.x, edit_counts.x + edit_counts.y, part == 1u);
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
                    let c = vec3<i32>(ch, center_half((k_lo + i32(b)) * 8 + i32(z), level));
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
            if kinds[z] != 0u {
                let bit = li + z * 64u;
                atomicOr(&g_words[bit >> 5u], 1u << (bit & 31u));
            }
        }
        workgroupBarrier();
        if li < 16u {
            let w = atomicExchange(&g_words[li], 0u);
            scratch[(base + scratch_header + b) * UNIT_WORDS + li] = w;
            if w != 0u { atomicOr(&g_any[0], 1u); }
            if w != 0xffffffffu { atomicOr(&g_any[1], 1u); }
        }
        workgroupBarrier();
        // The cleared words gather the generated tops after the last brick
        // (no extra barrier); they replace the header's heightfield tops.
        if volumetric && b + 1u == n_band {
            let down = u32(clamp(k_hi * 8 - generated_top, 0, 255));
            atomicOr(&g_words[li >> 2u], down << ((li & 3u) * 8u));
        }
        if li == 0u {
            let some = atomicExchange(&g_any[0], 0u) != 0u;
            let holes = atomicExchange(&g_any[1], 0u) != 0u;
            if some && holes {
                atomicOr(&g_masks[b >> 5u], 1u << (b & 31u));
            } else if some {
                atomicOr(&g_masks[8u + (b >> 5u)], 1u << (b & 31u));
            }
        }
        workgroupBarrier();
    }
    if volumetric && li < 16u { scratch[base * UNIT_WORDS + li] = atomicLoad(&g_words[li]); }
    // The density describes the lane's surface only when its last zero
    // crossing is at the generated top (that air cell, or the one below when
    // the surface rose into it): an undercut below an untouched top crossed
    // lower, and its floor gave the top's relief and height (black specks
    // across overhang regions).
    let density_surface = (surface_changed || dense_lane) && top_air != NO_DENSITY
        && (crossing_air == generated_top || crossing_air + 1 == generated_top);
    if wide_relief && volumetric {
        if density_surface { fraction = top_fraction; }
        atomicOr(&g_fraction[li >> 1u], fraction << ((li & 1u) * 16u));
    }
    // Surface offset: the exact surface's height over the stored one, from -1
    // to 1 units of its precision in 128ths (`column_surface_offset`): base
    // cells over the relief's base-cell top or the level-0 top, level cells
    // over a whole-cell top. Only shading reads it (smooth normals and
    // material height at every level); occupancy keeps the voxels' grid.
    // A density surface takes it from its zero crossing over its generated
    // top (level cells, as a level-0 or whole-cell top is stored); a relief
    // fraction carries that precision itself. A column with deep caves
    // outgrows relief: its density lanes kept whole-cell tops, a staircase
    // whose steps lit as dark dashes across distant cave regions.
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
    if wide_relief && volumetric && li < 32u {
        scratch[(base + 1u) * UNIT_WORDS + li] = atomicLoad(&g_fraction[li]);
    }
    if li < 16u { scratch[(base + scratch_offset) * UNIT_WORDS + li] = atomicLoad(&g_offset[li]); }
    if li == 0u {
        var out: JobOut;
        out.status = 0u;
        out.pad = select(0u, INFO_RELIEF, relief) | select(0u, INFO_RELIEF_INLINE, inline_relief)
            | select(0u, INFO_HEIGHTFIELD, heightfield) | topology_flags
            // Generated caves and overhangs: arbitrary occupancy under a
            // natural surface, tops counting down from the band top.
            | select(0u, INFO_GENERATED, volumetric) | select(0u, INFO_TOPS_DOWN, tops_down) | clip
            | select(0u, INFO_EDIT_MATERIALS, (volume_bits & VOLUME_MATERIALS) != 0u);
        out.top = hi_cell;
        out.centre = centre;
        out.k_lo = k_lo;
        out.n_band = n_band;
        out.scratch = base;
        var mixed = 0u;
        for (var w = 0u; w < 8u; w++) {
            out.mixed[w] = atomicLoad(&g_masks[w]);
            out.solid[w] = atomicLoad(&g_masks[8u + w]);
            mixed += countOneBits(out.mixed[w]);
        }
        // The scratch masks still certify exact summary tops. Natural
        // columns need no duplicate occupancy payload in the resident pool.
        out.n_mixed = select(mixed, 0u, heightfield);
        job_out[index] = out;
    }
}

fn run_units(o: JobOut) -> u32 {
    return select(1u, 2u, o.n_band > 32u)
        + select(0u, 2u, info_relief_wide(o.pad))
        + select(0u, 1u, world.sphere.w != 0u) + 1u + o.n_mixed;
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
    let ext = o.n_band > 32u;
    let wide_relief = info_relief_wide(o.pad);
    let surface_words = world.sphere.w != 0u;
    let scratch_surface = select(1u, 3u, wide_relief);
    let scratch_offset = scratch_surface + select(0u, 1u, surface_words);
    let scratch_header = scratch_offset + 1u;
    let plain_header = select(1u, 2u, ext);
    let pool_surface = plain_header + select(0u, 2u, wide_relief);
    let pool_offset = pool_surface + select(0u, 1u, surface_words);
    let header = pool_offset + 1u;
    if wide_relief && li < 32u {
        pool[(o.run + plain_header) * UNIT_WORDS + li] = scratch[(o.scratch + 1u) * UNIT_WORDS + li];
    }
    if surface_words && li < 16u {
        pool[(o.run + pool_surface) * UNIT_WORDS + li] = scratch[(o.scratch + scratch_surface) * UNIT_WORDS + li];
    }
    if li < 16u {
        pool[(o.run + pool_offset) * UNIT_WORDS + li] = scratch[(o.scratch + scratch_offset) * UNIT_WORDS + li];
    }
    if li < 16u {
        pool[o.run * UNIT_WORDS + li] = scratch[o.scratch * UNIT_WORDS + li];
        if ext {
            pool[(o.run + 1u) * UNIT_WORDS + li] = select(o.solid[li - 8u], o.mixed[li], li < 8u);
        }
    }
    let total = select(o.n_band * UNIT_WORDS, 0u, (o.pad & INFO_HEIGHTFIELD) != 0u);
    for (var w = li; w < total; w += 64u) {
        let b = w / UNIT_WORDS;
        let bit = b & 31u;
        let word = b >> 5u;
        if ((o.mixed[word] >> bit) & 1u) == 0u { continue; }
        var rank = countOneBits(o.mixed[word] & ((1u << bit) - 1u));
        for (var q = 0u; q < word; q++) { rank += countOneBits(o.mixed[q]); }
        pool[(o.run + header + rank) * UNIT_WORDS + (w % UNIT_WORDS)] =
            scratch[(o.scratch + scratch_header + b) * UNIT_WORDS + (w % UNIT_WORDS)];
    }
    if li == 0u {
        let previous = records[job.record];
        let was_published = (previous.info & INFO_VALID) != 0u && previous.key0 == job.key0 && previous.key1 == job.key1;
        if (job.flags & 1u) != 0u && (previous.info & INFO_VALID) != 0u {
            free_run(previous);
        }
        // Exact top: the highest occupied layer of the highest non-air band
        // brick (mixed bricks are scanned by z layer, two words each).
        let band_top = (o.k_lo + i32(o.n_band)) * 8;
        var exact = o.k_lo * 8;
        for (var b = i32(o.n_band) - 1; b >= 0; b--) {
            let word = u32(b) >> 5u;
            let bit = u32(b) & 31u;
            if ((o.solid[word] >> bit) & 1u) != 0u {
                exact = (o.k_lo + b + 1) * 8;
                break;
            }
            if ((o.mixed[word] >> bit) & 1u) != 0u {
                let base = (o.scratch + scratch_header + u32(b)) * UNIT_WORDS;
                var z = 7;
                while z > 0 && (scratch[base + 2u * u32(z)] | scratch[base + 2u * u32(z) + 1u]) == 0u { z -= 1; }
                exact = (o.k_lo + b) * 8 + z + 1;
                break;
            }
        }
        // A band clipped above bounds its summaries with the unclipped top.
        let clipped_above = (o.pad & INFO_CLIP_ABOVE) != 0u;
        let gap = select(u32(clamp(band_top - exact, 0, 7)), 0u, clipped_above);
        let top_cell = select(band_top - i32(gap), max(o.top, band_top), clipped_above);
        let ci = i32(job.key0 & 0xffffffu);
        let cj = bitcast<i32>(job.key1);
        for (var tier = 1u; tier <= 3u; tier++) {
            let bi = ci >> (2u * tier);
            let bj = cj >> (2u * tier);
            let slot = block_slot(job.key0 >> 27u, (job.key0 >> 24u) & 7u, tier, bi, bj) * 4u;
            if atomicLoad(&block_state[slot]) == bi && atomicLoad(&block_state[slot + 1u]) == bj {
                atomicMax(&block_state[slot + 2u], top_cell);
                if !was_published { atomicAdd(&block_state[slot + 3u], 1); }
            }
        }
        var c: Column;
        c.key0 = job.key0;
        c.key1 = job.key1;
        c.k_lo = o.k_lo;
        c.info = o.n_band | (o.size_class << 18u) | (gap << 22u) | select(0u, INFO_EXT, ext)
            | (o.pad & (INFO_RELIEF | INFO_RELIEF_INLINE | INFO_HEIGHTFIELD | INFO_TOPOLOGY | INFO_GENERATED | INFO_TOPS_DOWN | INFO_CLIP_BELOW | INFO_CLIP_ABOVE
                | INFO_EDIT_MATERIALS)) | INFO_VALID;
        c.run = o.run;
        c.mixed = o.mixed[0];
        c.solid = o.solid[0];
        c.edits = job.edits;
        records[job.record] = c;
        let level = job.key0 >> 27u;
        atomicMax(&level_tops[level], top_cell << level);
        // Clipped columns are reported with their window centre, so the CPU
        // can regenerate them when the eye leaves the window.
        if (o.pad & (INFO_CLIP_BELOW | INFO_CLIP_ABOVE)) != 0u {
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
