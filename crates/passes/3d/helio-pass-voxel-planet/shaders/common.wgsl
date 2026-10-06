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
    materials: array<MaterialGpu, 16>,
    detail: vec4<f32>,      // patch contrast, pigment contrast, edge darkening
}

// A terrain material's appearance (`engine::MaterialAppearance`): shading
// knows materials only through this table.
struct MaterialGpu {
    colour: vec4<f32>,              // linear colour, roughness
    patches: array<vec4<f32>, 3>,   // linear dry/middle/lush patch colours; patches[0].w 1: varies
    links: vec4<u32>,               // lip, fleck, speck host (self: none), fleck share (Q16)
}

struct Column {
    key0: u32,   // column i | face << 24 | level << 27
    key1: u32,   // column j
    k_lo: i32,   // lowest band brick layer (level bricks)
    info: u32,   // n_band 0..9 | n_mixed 9..18 | class 18..22 | top gap 22..25 | topology 27 | relief 28 | ext 29 | overflow 30 | valid 31
    run: u32,    // first pool unit
    mixed: u32,  // band bricks 0..32 that store an occupancy mask
    solid: u32,  // band bricks 0..32 that are completely occupied
    edits: u32,  // 1 + edit-ref list offset, or 0
}

struct FaceBrush {
    flags: u32,
    radius_half: u32,
    pad0: u32,
    pad1: u32,
    center: vec4<i32>,
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
const INFO_EXT: u32 = 0x20000000u;
const INFO_RELIEF: u32 = 0x10000000u;
// Effective Add/Remove lists: base-field gradients cannot describe cut faces.
const INFO_TOPOLOGY: u32 = 0x08000000u;
// Low-level authored tops can share the existing byte header with their
// fractional remainder. This changes storage only, not the traced surface.
const INFO_RELIEF_INLINE: u32 = 0x04000000u;
// Natural columns are exactly solid below their stored per-cell tops.
// Add/Remove columns retain arbitrary brick occupancy instead.
const INFO_HEIGHTFIELD: u32 = 0x02000000u;
const UNIT_WORDS: u32 = 16u;
const MAX_PROBES: u32 = 64u;

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> world: World;
@group(0) @binding(16) var<uniform> terrain: TerrainConstants;
@group(0) @binding(2) var<storage, ACCESS> table: array<u32>;
@group(0) @binding(3) var<storage, ACCESS> records: array<Column>;
@group(0) @binding(4) var<storage, ACCESS> pool: array<u32>;
@group(0) @binding(5) var<storage, read> brushes: array<FaceBrush>;
@group(0) @binding(6) var<storage, read> edit_refs: array<u32>;
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

fn band_count(c: Column) -> u32 { return c.info & 511u; }

// Relative terrain tops occupy one byte. A 32-brick band fits only when
// its highest occupied top is below the exact 256-cell upper boundary.
fn column_tops_fit(c: Column) -> bool {
    let count = band_count(c);
    return count < 32u || (count == 32u && ((c.info >> 22u) & 7u) != 0u);
}

// First empty layer above every occupied cell (level cells): the band top
// less the empty layers of its top brick.
fn column_top_cell(c: Column) -> i32 {
    return (c.k_lo + i32(band_count(c))) * 8 - i32((c.info >> 22u) & 7u);
}

// Header units: the tops (and extension masks), wide relief fractions, then
// the surface words when the program has them (one byte per cell).
fn header_units(c: Column) -> u32 {
    return surface_unit(c) + select(0u, 1u, world.sphere.w != 0u);
}

fn surface_unit(c: Column) -> u32 {
    return select(1u, 2u, (c.info & INFO_EXT) != 0u)
        + select(0u, 2u, (c.info & INFO_RELIEF) != 0u && (c.info & INFO_RELIEF_INLINE) == 0u);
}

// Surface word of column cell (x, y) (`terrain_surface`); 0 without them.
fn column_surface(c: Column, x: u32, y: u32) -> u32 {
    if world.sphere.w == 0u { return 0u; }
    let cell = x + y * 8u;
    let word = pool[(c.run + surface_unit(c)) * UNIT_WORDS + (cell >> 2u)];
    return (word >> ((cell & 3u) * 8u)) & 0xffu;
}

// Brick state of band brick `b`: 0 air, 1 solid, 2 mixed. Also returns the
// pool unit of a mixed brick.
fn brick_state(c: Column, b: u32) -> vec2<u32> {
    var mixed_bit = false;
    var solid_bit = false;
    var rank = 0u;
    if b < 32u {
        mixed_bit = ((c.mixed >> b) & 1u) != 0u;
        solid_bit = ((c.solid >> b) & 1u) != 0u;
        rank = countOneBits(c.mixed & ((1u << b) - 1u));
    } else {
        let ext = (c.run + 1u) * UNIT_WORDS;
        let w = b >> 5u;
        let bit = b & 31u;
        mixed_bit = ((pool[ext + w] >> bit) & 1u) != 0u;
        solid_bit = ((pool[ext + 8u + w] >> bit) & 1u) != 0u;
        for (var i = 0u; i < w; i++) { rank += countOneBits(pool[ext + i]); }
        rank += countOneBits(pool[ext + w] & ((1u << bit) - 1u));
    }
    if mixed_bit { return vec2<u32>(2u, c.run + header_units(c) + rank); }
    return vec2<u32>(select(0u, 1u, solid_bit), 0u);
}

fn brick_bit(unit: u32, x: u32, y: u32, z: u32) -> bool {
    let bit = x + y * 8u + z * 64u;
    return ((pool[unit * UNIT_WORDS + (bit >> 5u)] >> (bit & 31u)) & 1u) != 0u;
}

// With INFO_TOPOLOGY (never stored with relief): per-cell tops count down
// from the band top. Generated volumetric columns store their generated
// top (first air above the highest generated solid cell, caves and
// overhangs included) there; their bands are taller than a byte.
const INFO_TOPS_DOWN: u32 = 0x04000000u;

fn column_tops_down(c: Column) -> bool {
    return (c.info & (INFO_TOPOLOGY | INFO_TOPS_DOWN)) == (INFO_TOPOLOGY | INFO_TOPS_DOWN);
}

// Column-local surface top (first air layer above ground, level cells).
fn column_top(c: Column, x: u32, y: u32) -> i32 {
    let cell = x + y * 8u;
    let word = pool[c.run * UNIT_WORDS + (cell >> 2u)];
    let offset = (word >> ((cell & 3u) * 8u)) & 255u;
    if column_tops_down(c) {
        return (c.k_lo + i32(band_count(c))) * 8 - i32(offset);
    }
    if (c.info & INFO_RELIEF_INLINE) != 0u {
        let level = c.key0 >> 27u;
        return c.k_lo * 8 + i32((offset + (1u << level) - 1u) >> level);
    }
    return c.k_lo * 8 + i32(offset);
}

// Zero denotes a top exactly on the upper coarse-cell boundary. Other
// fractions reconstruct the authored base-layer top inside the last voxel.
fn column_relief_fraction(c: Column, x: u32, y: u32) -> u32 {
    if (c.info & INFO_RELIEF) == 0u || !column_tops_fit(c) { return 0u; }
    let cell = x + y * 8u;
    if (c.info & INFO_RELIEF_INLINE) != 0u {
        let level = c.key0 >> 27u;
        let word = pool[c.run * UNIT_WORDS + (cell >> 2u)];
        let offset = (word >> ((cell & 3u) * 8u)) & 255u;
        return (offset & ((1u << level) - 1u)) << (16u - level);
    }
    let offset = select(1u, 2u, (c.info & INFO_EXT) != 0u);
    let word = pool[(c.run + offset) * UNIT_WORDS + (cell >> 1u)];
    return (word >> ((cell & 1u) * 16u)) & 65535u;
}

fn center_half(i: i32, level: u32) -> i32 {
    return (i << (level + 1u)) + (1 << level);
}

fn brush_contains(b: FaceBrush, c: vec3<i32>) -> bool {
    let r = b.radius_half;
    let d = vec3<u32>(vec3<i32>(abs(c - b.center.xyz)));
    if any(d > vec3<u32>(r)) { return false; }
    if ((b.flags >> 6u) & 3u) == 1u { return true; }
    return d.x * d.x + d.y * d.y + d.z * d.z <= r * r;
}

// Applies an ordered edit list; returns (kind, material).
fn apply_edits(list: u32, level: u32, c: vec3<i32>, kind_in: u32) -> vec2<u32> {
    var kind = kind_in;
    var material = 0u;
    if list == 0u { return vec2<u32>(kind, material); }
    let count = edit_refs[list - 1u];
    for (var e = 0u; e < count; e++) {
        let b = brushes[edit_refs[list + e]];
        if b.radius_half < (1u << level) || !brush_contains(b, c) { continue; }
        let op = (b.flags >> 4u) & 3u;
        if op == 0u {
            kind = 0u;
            material = 0u;
        } else if op == 1u {
            kind = 1u;
            material = (b.flags >> 8u) & 255u;
        } else if kind == 1u {
            material = (b.flags >> 8u) & 255u;
        }
    }
    return vec2<u32>(kind, material);
}
