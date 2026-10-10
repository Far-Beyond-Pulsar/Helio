// Shared planet residency structures. `ACCESS` is replaced by `read` or
// `read_write` per pipeline module.

// Shade may low-pass procedural material detail for the current pixel.
// Other entry points and canonical material queries retain zero footprint.
var<private> material_footprint: f32 = 0.0;
// Natural surface coating is shade-only; canonical queries keep their IDs.
var<private> material_weathered_skin: bool = false;
// Radial height range of the material pixel, in metres; zero outside shade.
var<private> material_radial_span: f32 = 0.0;
// Display-only appearance a terrain program may report for the shaded cell
// (canonical IDs never change). A coverage between two materials (negative:
// none), a mix of four (negative first weight: none), and the material whose
// single-voxel flecks the filtered colour averages (air: none).
var<private> material_coverage: f32 = -1.0;
var<private> material_coverage_ids: vec2<u32> = vec2<u32>(0u);
var<private> material_mix: vec4<f32> = vec4<f32>(-1.0, 0.0, 0.0, 0.0);
var<private> material_mix_ids: vec4<u32> = vec4<u32>(0u);
var<private> material_fleck_base: u32 = 0u;

struct FaceGpu {
    m_a: vec4<f32>,   // α-family plane normal at the eye (xyz), distance to axis (w)
    q_a: vec4<f32>,   // in-plane radial direction (xyz), eye fraction (w)
    m_b: vec4<f32>,
    q_b: vec4<f32>,
    index: vec4<i32>, // eye base index a, b; valid; pad
}

struct Frame {
    faces: array<FaceGpu, 6>,
    eye: vec4<f32>,        // unit eye direction from the planet centre (xyz), |eye| (w)
    layer: vec4<f32>,      // eye layer fraction, voxel size, angular cell, outer radius - |eye|
    layer_i: vec4<i32>,    // eye base layer, cells per face, level count, eye face
    lod: vec4<f32>,        // level-0 distance, dither, sky bound cut height, max distance
    screen: vec4<f32>,     // width, height, frame, flags (2: sky bound)
    sun: vec4<f32>,        // direction to sun, enabled
    counts: vec4<u32>,     // jobs, evictions, table mask, pool units
    neighbours: array<vec4<u32>, 6>, // face across -a, +a, -b, +b
    extra: vec4<u32>,      // table patches, block region, block patches, live tier-1 blocks
    ring: array<vec4<f32>, 8>, // per level: sky bound block exclusion angle
    hints: vec4<u32>,
    materials: array<MaterialGpu, 32>,
    detail: vec4<f32>,      // patch contrast, pigment contrast, edge darkening
}

// A terrain material's appearance (`engine::MaterialAppearance`): shading
// knows materials only through this table.
struct MaterialGpu {
    colour: vec4<f32>,              // linear colour, roughness
    patches: array<vec4<f32>, 3>,   // linear dry/middle/lush patch colours; patches[0].w 1: varies
    links: vec4<u32>,               // lip, fleck, speck host (self: none), fleck share (Q16)
}

// A resident column: an 8x8 footprint of one level and its vertical
// content as spans (docs/span-columns.md). Everything below the first span
// is solid, everything from `top` up is air; a window clipped below or above
// does not describe the cells past it (rays there use a coarser level).
struct Column {
    key0: u32,   // column i | face << 24 | level << 27
    key1: u32,   // column j
    base: i32,   // natural tops' base (level cells)
    info: u32,   // spans 0..5 | wide tops 5 | clip below 9 | clip above 10 | edit materials 11 | generated 12 | class 18..22 | heightfield 25 | relief inline 26 | topology 27 | relief 28 | overflow 30 | valid 31
    run: u32,    // first pool unit
    top: i32,    // first layer above every solid cell (level cells); clipped above: the window's top
    lo: i32,     // clipped below: the lowest layer the column describes
    edits: u32,  // 1 + edit-ref list offset, or 0
}

// `edits::FaceBrush`: shape 0 is a ball in volume space, 1 a cube in index
// space; k_lo/k_hi bound the half-cell heights it touches.
struct FaceBrush {
    flags: u32,
    radius_half: u32,
    k_lo: i32,
    k_hi: i32,
    center: vec4<i32>,
    ball: vec4<i32>,
}

// World shape, substituted when the pipelines are built: 0 sphere, 1 plane.
const SHAPE: u32 = SHAPE_ID;
const PLANE_FACE: u32 = 2u;

fn is_plane() -> bool {
    return SHAPE == 1u;
}

const NONE: u32 = 0xffffffffu;
const TOMBSTONE: u32 = 0xfffffffeu;
const INFO_VALID: u32 = 0x80000000u;
const INFO_OVERFLOW: u32 = 0x40000000u;
const INFO_RELIEF: u32 = 0x10000000u;
// Effective Add/Remove lists: base-field gradients cannot describe cut faces.
const INFO_TOPOLOGY: u32 = 0x08000000u;
// Low-level authored tops share the natural tops' bytes with their
// fractional remainder (base cells over the column base).
const INFO_RELIEF_INLINE: u32 = 0x04000000u;
// The column is exactly solid below its natural tops: no span table.
const INFO_HEIGHTFIELD: u32 = 0x02000000u;
// The column's window was clipped below (above): cells under `lo` (from
// `top` up) are not described.
const INFO_CLIP_BELOW: u32 = 0x200u;
const INFO_CLIP_ABOVE: u32 = 0x400u;
// The column's edit list holds Add or Paint brushes: shading looks up brush
// materials only in such columns.
const INFO_EDIT_MATERIALS: u32 = 0x800u;
// Generated volume (caves, overhangs): its natural tops are the generated
// tops (first air above the highest generated solid cell); only
// INFO_TOPOLOGY (edit cuts) marks a cut.
const INFO_GENERATED: u32 = 0x1000u;
// Natural tops spread over more than a byte: 16 bits each.
const INFO_TOPS_WIDE: u32 = 0x20u;
const UNIT_WORDS: u32 = 16u;
const MAX_PROBES: u32 = 64u;

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> world: World;
@group(0) @binding(16) var<uniform> terrain: TerrainConstants;
@group(0) @binding(2) var<storage, ACCESS> table: array<u32>;
@group(0) @binding(3) var<storage, ACCESS> records: array<Column>;
@group(0) @binding(4) var<storage, ACCESS> pool: array<u32>;
// Baked brick slots (`edit_store::Brick`): 512 cells of 16 bits, 256 words.
@group(0) @binding(5) var<storage, read> baked: array<u32>;
@group(0) @binding(6) var<storage, read> edit_refs: array<u32>;
// The face brushes edit blocks reference (`residency` brush table).
@group(0) @binding(20) var<storage, read> brushes: array<FaceBrush>;
@group(0) @binding(14) var<storage, ACCESS> level_tops: array<LEVEL_TOP>;
// Direct-mapped summary blocks, 4 words per entry: [bi, bj, max occupied top
// (level cells), published columns].
@group(0) @binding(15) var<storage, ACCESS> block_state: array<BLOCK_ENTRY>;

fn column_key0(face: u32, level: u32, ci: i32) -> u32 {
    return (bitcast<u32>(ci) & 0xffffffu) | (face << 24u) | (level << 27u);
}

fn column_slot(key0: u32, key1: u32) -> u32 {
    return hash3(bitcast<i32>(key0), bitcast<i32>(key1), 0x2f6b1d3a, 0x9e3779b9u);
}

// Table edge log2 per tier: 7, 5, 3 (see residency::BLOCK_LOG2).
fn block_slot(level: u32, face: u32, tier: u32, bi: i32, bj: i32) -> u32 {
    var offset = (level * 6u + face) * frame.extra.y;
    if tier >= 2u { offset += 1u << 14u; }
    if tier >= 3u { offset += 1u << 10u; }
    let l = 9u - 2u * tier;
    let mask = (1 << l) - 1;
    return offset + (u32(bj & mask) << l) + u32(bi & mask);
}

fn find_column(key0: u32, key1: u32) -> u32 {
    let mask = frame.counts.z;
    var slot = column_slot(key0, key1) & mask;
    for (var probe = 0u; probe < MAX_PROBES; probe++) {
        let value = table[slot];
        if value == NONE { return NONE; }
        if value != TOMBSTONE {
            let c = records[value];
            if c.key0 == key0 && c.key1 == key1 { return value; }
        }
        slot = (slot + 1u) & mask;
    }
    return NONE;
}

fn column_valid(c: Column) -> bool {
    return (c.info & (INFO_VALID | INFO_OVERFLOW)) == INFO_VALID;
}

// Whether a valid column describes its level cell layer `k`.
fn column_knows(c: Column, k: i32) -> bool {
    if (c.info & INFO_CLIP_BELOW) != 0u && k < c.lo { return false; }
    if (c.info & INFO_CLIP_ABOVE) != 0u && k >= c.top { return false; }
    return true;
}

// Header units: the natural tops (one unit of bytes, two of 16 bits), wide
// relief fractions, the surface words when the program has them (one byte
// per cell), the surface offsets (one byte per cell); then the span table
// and its payloads, then the span bricks.
fn tops_units(c: Column) -> u32 {
    return select(1u, 2u, (c.info & INFO_TOPS_WIDE) != 0u);
}

fn surface_unit(c: Column) -> u32 {
    return tops_units(c) + select(0u, 2u, info_relief_wide(c.info));
}

fn offset_unit(c: Column) -> u32 {
    return surface_unit(c) + select(0u, 1u, world.sphere.w != 0u);
}

fn header_units(c: Column) -> u32 {
    return offset_unit(c) + 1u;
}

// Height of the exact surface of column cell (x, y) of `level` over its
// stored height, in level cells: the generator's surface below voxel
// precision, for smooth shading. Occupancy never reads it. The byte holds
// -1..1 units of the stored height's precision: a base cell over a relief or
// level-0 top, a level cell over a whole-cell top (edited columns keep no
// relief; base-cell units could not reach their surface, whose ground then
// sank below its neighbours': dark column outlines around every edit).
fn column_surface_offset(c: Column, x: u32, y: u32, level: u32) -> f32 {
    let cell = x + y * 8u;
    let word = pool[(c.run + offset_unit(c)) * UNIT_WORDS + (cell >> 2u)];
    let units = (f32((word >> ((cell & 3u) * 8u)) & 0xffu) - 128.0) / 128.0;
    let base_units = (c.info & INFO_RELIEF) != 0u || level == 0u;
    return select(units, units / f32(1u << level), base_units);
}

// Surface word of column cell (x, y) (`terrain_surface`); 0 without them.
fn column_surface(c: Column, x: u32, y: u32) -> u32 {
    if world.sphere.w == 0u { return 0u; }
    let cell = x + y * 8u;
    let word = pool[(c.run + surface_unit(c)) * UNIT_WORDS + (cell >> 2u)];
    return (word >> ((cell & 3u) * 8u)) & 0xffu;
}

fn brick_bit(unit: u32, x: u32, y: u32, z: u32) -> bool {
    let bit = x + y * 8u + z * 64u;
    return ((pool[unit * UNIT_WORDS + (bit >> 5u)] >> (bit & 31u)) & 1u) != 0u;
}

fn column_generated(c: Column) -> bool {
    return (c.info & INFO_GENERATED) != 0u;
}

// Relief fractions in two units after the tops: every relief column but
// an inline one.
fn info_relief_wide(info: u32) -> bool {
    return (info & INFO_RELIEF) != 0u && (info & INFO_RELIEF_INLINE) == 0u;
}

fn column_relief_inline(c: Column) -> bool {
    return (c.info & INFO_RELIEF) != 0u && (c.info & INFO_RELIEF_INLINE) != 0u;
}

// Natural surface top of column cell (x, y): first air layer above the
// generated ground (level cells), whatever edits did there. Material depth,
// relief and the smooth ground read it at any depth below it.
fn column_top(c: Column, x: u32, y: u32) -> i32 {
    let cell = x + y * 8u;
    if (c.info & INFO_TOPS_WIDE) != 0u {
        let word = pool[c.run * UNIT_WORDS + (cell >> 1u)];
        return c.base + i32((word >> ((cell & 1u) * 16u)) & 0xffffu);
    }
    let word = pool[c.run * UNIT_WORDS + (cell >> 2u)];
    let offset = (word >> ((cell & 3u) * 8u)) & 255u;
    if (c.info & INFO_RELIEF_INLINE) != 0u {
        let level = c.key0 >> 27u;
        return c.base + i32((offset + (1u << level) - 1u) >> level);
    }
    return c.base + i32(offset);
}

// Zero denotes a top exactly on the upper coarse-cell boundary. Other
// fractions reconstruct the authored base-layer top inside the last voxel.
fn column_relief_fraction(c: Column, x: u32, y: u32) -> u32 {
    if (c.info & INFO_RELIEF) == 0u { return 0u; }
    let cell = x + y * 8u;
    if column_relief_inline(c) {
        let level = c.key0 >> 27u;
        let word = pool[c.run * UNIT_WORDS + (cell >> 2u)];
        let offset = (word >> ((cell & 3u) * 8u)) & 255u;
        return (offset & ((1u << level) - 1u)) << (16u - level);
    }
    let word = pool[(c.run + tops_units(c)) * UNIT_WORDS + (cell >> 1u)];
    return (word >> ((cell & 1u) * 16u)) & 65535u;
}

// Span kinds. AIR, SOLID, LANES, TOPS and NATURAL are lane spans: each lane
// is solid below its own top inside the span (`span_lane_top`). BRICKS hold
// arbitrary occupancy.
const SPAN_AIR: u32 = 0u;
const SPAN_SOLID: u32 = 1u;
const SPAN_LANES: u32 = 2u;
const SPAN_TOPS: u32 = 3u;
const SPAN_NATURAL: u32 = 4u;
const SPAN_BRICKS: u32 = 5u;
const NO_LAYER: i32 = -0x7fffffff;

// A span `[start, end)` of a column: its kind and its payload's first word.
struct Span {
    kind: u32,
    start: i32,
    end: i32,
    payload: u32,
}

// The span table: (start, kind | payload word offset << 3) per span, from
// the first word after the header.
fn span_table(c: Column) -> u32 {
    return (c.run + header_units(c)) * UNIT_WORDS;
}

// The span holding layer `k` (below the column's top): below the first
// span everything is solid.
fn span_at(c: Column, k: i32) -> Span {
    var s = Span(SPAN_SOLID, NO_LAYER, c.top, 0u);
    if (c.info & INFO_HEIGHTFIELD) != 0u {
        s.kind = SPAN_NATURAL;
        return s;
    }
    let table = span_table(c);
    let n = c.info & 31u;
    for (var e = 0u; e < n; e++) {
        let start = bitcast<i32>(pool[table + e * 2u]);
        if k < start {
            s.end = start;
            return s;
        }
        let entry = pool[table + e * 2u + 1u];
        s = Span(entry & 7u, start, c.top, table + (entry >> 3u));
    }
    return s;
}

// Layer below which lane (x, y) of lane span `s` is solid: its start where
// the lane is air throughout, its end where solid throughout.
fn span_lane_top(c: Column, s: Span, x: u32, y: u32) -> i32 {
    let cell = x + y * 8u;
    switch s.kind {
        case 1u: { return s.end; }
        case 2u: {
            let bit = (pool[s.payload + (cell >> 5u)] >> (cell & 31u)) & 1u;
            return select(s.start, s.end, bit != 0u);
        }
        case 3u: {
            let word = pool[s.payload + (cell >> 2u)];
            return s.start + i32((word >> ((cell & 3u) * 8u)) & 255u);
        }
        case 4u: { return clamp(column_top(c, x, y), s.start, s.end); }
        default: { return s.start; }
    }
}

// Brick `b` (counted from the span's start) of a BRICKS span: 0 air, 1
// solid, 2 mixed with its pool unit. Payload: the run-relative unit of the
// span's first mixed brick, then mixed and solid bits per brick.
fn span_brick(c: Column, s: Span, b: u32) -> vec2<u32> {
    let words = ((u32(s.end - s.start) >> 3u) + 31u) >> 5u;
    let w = b >> 5u;
    let bit = b & 31u;
    let mixed = pool[s.payload + 1u + w];
    if ((mixed >> bit) & 1u) != 0u {
        var rank = countOneBits(mixed & ((1u << bit) - 1u));
        for (var q = 0u; q < w; q++) { rank += countOneBits(pool[s.payload + 1u + q]); }
        return vec2<u32>(2u, c.run + pool[s.payload] + rank);
    }
    let solid = ((pool[s.payload + 1u + words + w] >> bit) & 1u) != 0u;
    return vec2<u32>(select(0u, 1u, solid), 0u);
}

// Occupancy of level cell (x, y, k) of a valid column: 0 air, 1 solid, 2
// not described (beyond a clipped window).
fn column_cell(c: Column, x: u32, y: u32, k: i32) -> u32 {
    if !column_knows(c, k) { return 2u; }
    if k >= c.top { return 0u; }
    let s = span_at(c, k);
    if s.kind == SPAN_BRICKS {
        let state = span_brick(c, s, u32((k - s.start) >> 3u));
        if state.x == 2u { return select(0u, 1u, brick_bit(state.y, x, y, u32(k & 7))); }
        return state.x;
    }
    return select(0u, 1u, k < span_lane_top(c, s, x, y));
}

fn center_half(i: i32, level: u32) -> i32 {
    return (i << (level + 1u)) + (1 << level);
}

// Exact containment (`FaceBrush::contains`): cubes test the cell's half-cell
// centre `c`, balls its volume point `q`, with exact 64-bit squares.
fn brush_contains(b: FaceBrush, c: vec3<i32>, q: vec3<i32>) -> bool {
    if ((b.flags >> 6u) & 3u) == 1u {
        // A box: the radius across, its own range vertically.
        return all(vec2<u32>(abs(c.xy - b.center.xy)) <= vec2<u32>(b.radius_half)) && c.z >= b.k_lo && c.z <= b.k_hi;
    }
    let r = u32(abs(b.ball.w));
    let d = vec3<u32>(abs(q - b.ball.xyz));
    if any(d > vec3<u32>(r)) { return false; }
    let sum = add_wide(add_wide(mul_wide(d.x, d.x), mul_wide(d.y, d.y)), mul_wide(d.z, d.z));
    let rr = mul_wide(r, r);
    return sum.y < rr.y || (sum.y == rr.y && sum.x <= rr.x);
}

// A column's edit block (`residency::Residency::edit_list`; a column's or
// job's `edits` is 1 + its offset in `edit_refs`): the counts of its large
// and recent brushes and of its baked bricks, the brushes' slots in
// `brushes` (large then recent), then (brick height, baked slot) pairs.
// Cells read recent(baked(large(terrain))) (`planet::Edits`).
struct EditCounts {
    large: u32,
    recent: u32,
    baked: u32,
}

fn edit_counts(list: u32) -> EditCounts {
    if list == 0u { return EditCounts(0u, 0u, 0u); }
    return EditCounts(edit_refs[list - 1u], edit_refs[list], edit_refs[list + 1u]);
}

// Brush `e` of a block (large brushes first, then recent).
fn edit_brush(list: u32, e: u32) -> FaceBrush {
    return brushes[edit_refs[list + 2u + e]];
}

// `edit_store::CellEdit` states.
const BAKED_AIR: u32 = 1u;
const BAKED_SOLID: u32 = 2u;
const BAKED_PAINT: u32 = 3u;

// Baked edit (state in bits 0..2, material in 8..16; 0 unchanged) of level
// cell (i, j, k) of the block's column.
fn baked_cell(list: u32, n: EditCounts, i: i32, j: i32, k: i32) -> u32 {
    let base = list + 2u + n.large + n.recent;
    let bk = k >> 3u;
    for (var e = 0u; e < n.baked; e++) {
        if bitcast<i32>(edit_refs[base + e * 2u]) == bk {
            // A uniform brick: its edit, no slot (`residency::BAKED_UNIFORM`).
            let slot = edit_refs[base + e * 2u + 1u];
            if (slot & 0x80000000u) != 0u { return slot & 0xffffu; }
            let index = u32((i & 7) + 8 * ((j & 7) + 8 * (k & 7)));
            let word = baked[slot * 256u + (index >> 1u)];
            return (word >> ((index & 1u) * 16u)) & 0xffffu;
        }
    }
    return 0u;
}

// Whether brush `b` with an op in `ops` (bit per op) contains the level
// cell with half-cell centre `c` (volume point computed once into `q`).
fn edit_matches(b: FaceBrush, level: u32, c: vec3<i32>, p: vec3<i32>, ops: u32, q: ptr<function, vec3<i32>>, q_ready: ptr<function, bool>) -> bool {
    if b.radius_half < (1u << level) || ((ops >> ((b.flags >> 4u) & 3u)) & 1u) == 0u { return false; }
    if c.z < b.k_lo || c.z > b.k_hi { return false; }
    if !*q_ready && ((b.flags >> 6u) & 3u) == 0u {
        *q = volume_point_half(p, c.z);
        *q_ready = true;
    }
    return brush_contains(b, c, *q);
}

// Flags of the latest edit whose op is in `ops` that reached the level cell
// with half-cell centre `c` in the column with domain point `p`, or NONE:
// the recent brushes (latest first), then the cell's baked edit (as the
// flags of the brush that left it), then the large brushes.
fn latest_edit(list: u32, level: u32, c: vec3<i32>, p: vec3<i32>, ops: u32) -> u32 {
    if list == 0u { return NONE; }
    let n = edit_counts(list);
    var q = vec3<i32>(0);
    var q_ready = false;
    for (var e = n.large + n.recent; e > n.large; e--) {
        let b = edit_brush(list, e - 1u);
        if edit_matches(b, level, c, p, ops, &q, &q_ready) { return b.flags; }
    }
    let shift = level + 1u;
    let cell = baked_cell(list, n, c.x >> shift, c.y >> shift, c.z >> shift);
    switch cell & 3u {
        case 1u: { return select(NONE, 0u, (ops & OPS_REMOVE) != 0u); }
        case 2u: { return select(NONE, (1u << 4u) | (cell & 0xff00u), (ops & OPS_MATERIAL) != 0u); }
        case 3u: {
            if (ops & OPS_MATERIAL) != 0u { return (2u << 4u) | (cell & 0xff00u); }
        }
        default: {}
    }
    for (var e = n.large; e > 0u; e--) {
        let b = edit_brush(list, e - 1u);
        if edit_matches(b, level, c, p, ops, &q, &q_ready) { return b.flags; }
    }
    return NONE;
}

const OPS_REMOVE: u32 = 1u;
const OPS_MATERIAL: u32 = 6u;

// Brush material of a solid cell (0: the terrain's). The latest Add or
// Paint containing it decides: a later Remove containing it would have
// left air, and a Paint over air is always followed by an Add that filled
// it again.
fn edit_material(list: u32, level: u32, c: vec3<i32>, p: vec3<i32>) -> u32 {
    let flags = latest_edit(list, level, c, p, OPS_MATERIAL);
    if flags == NONE { return 0u; }
    return (flags >> 8u) & 255u;
}

// Kind (0 air, 1 solid) the edits leave level cell `c` (half-cell centre)
// of the column with domain point `p`, or NONE where none removes or adds
// there: the latest Remove or Add containing it decides (paint keeps it).
fn latest_geometry(list: u32, level: u32, c: vec3<i32>, p: vec3<i32>) -> u32 {
    if list == 0u { return NONE; }
    let n = edit_counts(list);
    var q = vec3<i32>(0);
    var q_ready = false;
    for (var e = n.large + n.recent; e > n.large; e--) {
        let b = edit_brush(list, e - 1u);
        if edit_matches(b, level, c, p, 3u, &q, &q_ready) { return (b.flags >> 4u) & 3u; }
    }
    let shift = level + 1u;
    let cell = baked_cell(list, n, c.x >> shift, c.y >> shift, c.z >> shift) & 3u;
    if cell == BAKED_AIR { return 0u; }
    if cell == BAKED_SOLID { return 1u; }
    for (var e = n.large; e > 0u; e--) {
        let b = edit_brush(list, e - 1u);
        if edit_matches(b, level, c, p, 3u, &q, &q_ready) { return (b.flags >> 4u) & 3u; }
    }
    return NONE;
}
