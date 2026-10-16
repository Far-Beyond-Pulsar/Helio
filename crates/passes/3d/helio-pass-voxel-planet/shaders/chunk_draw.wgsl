// Chunk trees (docs/chunk-trees.md): primary visibility.
//
// 1. Seed (raster): every visible chunk draws its box; a fragment writes
//    only the box's index. Boxes do not overlap, so along a ray their
//    entries are in the order of their hits: with the depth test on box
//    front faces, each pixel keeps the nearest box its ray enters, and
//    boxes behind it never run (the fragment shader writes no depth).
//    A ray that leaves its box without a hit is seeded again, from where
//    it left (`exits`), by the next round of seed and trace.
// 2. Trace (compute): each pixel's ray, in its seed box's affine frame
//    (cells of its level), walks the chunk's 64-tree: the root's 4^3 mid
//    nodes of 16^3 cells, their 4^3 leaves of 4^3 cells, a leaf's cells.
//    An empty child is crossed in one step to its exit; a full child is a
//    hit where the ray enters it. A ray leaving its chunk without a hit
//    continues by the drawn cut (`chunk_table`): the drawn tile over the
//    column it enters, its chunk there, or air: an empty layer, or the
//    column over the tile's chunks, as coarse as the cut's tops allow; past
//    a cube face's edge, on the face across it.
//    Output: visibility (chunk slot; cell, face), the hit's distance and a
//    plain colour.
// 3. Resolve (raster): the hits' exact depth, for meshes and later passes;
//    or, in the render graph, publish: the hits into Helio's gbuffer.
// 4. Sun shadows (compute): from each hit, a walk towards the sun.

struct DrawParams {
    view_proj: mat4x4<f32>,
    inverse_view_proj: mat4x4<f32>,
    // The last frame's view-projection, relative to this frame's eye.
    prev_view_proj: mat4x4<f32>,
    // The eye, planet-centred, as a high and a low f32 part.
    eye_hi: vec4<f32>,
    eye_lo: vec4<f32>,
    sun: vec4<f32>,  // direction to the sun (xyz), camera near plane (w)
    debug: vec4<u32>, // x: colour by level
    table: vec4<u32>, // x: chunk table slots - 1; y: base cells per face edge; z: drawn chunks; w: frame
    // The eye in noise-domain units (`grid::DOMAIN_UNIT`, planet-centred):
    // integer part, fraction.
    eye_domain: vec4<i32>,
    eye_domain_frac: vec4<f32>,
    // Up at the eye (xyz; a plane's +y) and the eye's radius (w).
    up: vec4<f32>,
    // x: the eye's squared radius less the ceiling's, over which nothing
    // is solid (a plane: the eye's height over it).
    ceiling: vec4<f32>,
    // The planet's radius and voxel (m), the first level walked in the
    // grid's own cells (`exact_level`), the eye's squared radius less the
    // planet's (from f64).
    planet: vec4<f32>,
}

// A chunk's frame, written when it is generated: M's columns and M^-1's
// rows (xyz), its origin planet-centred as a high and a low f32 part, and
// its occupied layers [lo, hi), slot and flags (bit 0: the frame mirrors
// x; bits 8..16: level).
struct ChunkStatic {
    m0: vec4<f32>,
    m1: vec4<f32>,
    m2: vec4<f32>,
    i0: vec4<f32>,
    i1: vec4<f32>,
    i2: vec4<f32>,
    origin_hi: vec4<f32>,
    origin_lo: vec4<f32>,
    box: vec4<f32>,
}

// A chunk as drawn: its frame relative to the eye (`chunk_frame`). Local
// cell coordinates c map to the world as O + M c; the eye is at local `o`.
struct ChunkDraw {
    m0: vec4<f32>, // M's columns (xyz) and O (w)
    m1: vec4<f32>,
    m2: vec4<f32>,
    i0: vec4<f32>, // M^-1's rows (xyz) and o (w)
    i1: vec4<f32>,
    i2: vec4<f32>,
    // As in `ChunkStatic`.
    box: vec4<f32>,
}

struct ChunkMeta {
    root_occ: vec2<u32>,
    root_full: vec2<u32>,
    z: vec2<i32>,
    leaves: u32,
    pages: u32,
    // A plane over every solid cell: z = top.z + top.x x + top.y y.
    top: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: DrawParams;
@group(0) @binding(1) var<storage, read> statics: array<ChunkStatic>;
@group(0) @binding(2) var<storage, read> chunk_meta: array<ChunkMeta>;
@group(0) @binding(3) var<storage, read> mids: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> mid_leaves: array<u32>;
@group(0) @binding(5) var<storage, read> leaves: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read> leaf_materials: array<u32>;
// Each slot's leaf pages (`LEAF_PAGES`, `chunk_generate.wgsl`): a chunk's
// leaves in mid order, 64 to a page.
@group(0) @binding(14) var<storage, read> leaf_pages: array<u32>;
// Each stored leaf's ground normal (`chunk_build_main`): two slope bytes
// (x | y << 8, by angle), 0 where none.
@group(0) @binding(15) var<storage, read> leaf_normals: array<u32>;
const LEAF_PAGES: u32 = 64u;

// Leaf `index` of chunk `slot` (in its mids' order) in the leaf storage.
fn leaf_at_page(slot: u32, index: u32) -> u32 {
    return (leaf_pages[slot * LEAF_PAGES + (index >> 6u)] << 6u) | (index & 63u);
}
// Shading of the terrain's materials (`engine::material_table`).
struct MaterialGpu {
    colour: vec4<f32>,            // linear colour, roughness
    patches: array<vec4<f32>, 3>, // linear dry/middle/lush patch colours; patches[0].w 1: varies
    links: vec4<u32>,             // lip, fleck, speck host, fleck share
}
struct Materials {
    m: array<MaterialGpu, 32>,
    detail: vec4<f32>, // patch contrast, pigment contrast, edge darkening, step softness
}
@group(0) @binding(7) var<uniform> materials: Materials;

fn palette(m: u32) -> vec4<f32> {
    return materials.m[min(m, 31u)].colour;
}
@group(0) @binding(8) var<storage, read> mid_materials: array<u32>;
// Open addressing (`drawn.rs`): chunks, key (face | level << 8, cx, cy, cz)
// then (slot, state, 0, 0); the drawn cut's columns, key (.., COLUMN) then
// (lo, hi, top, state): a drawn tile's layers [lo, hi), the top solid cell
// under the column (exclusive), and whether a pixel shows it (`choice`).
// An empty entry's key x is -1.
@group(0) @binding(10) var<storage, read> chunk_table: array<vec4<i32>>;
@group(0) @binding(11) var<storage, read> chunk_keys: array<vec4<i32>>;
// This frame's boxes: chunk slots, those holding the eye first.
@group(0) @binding(12) var<storage, read> draw_list: array<u32>;
// Drawn tiles rays reached under the chunks of (through a cave mouth):
// how many times, then the first ones' keys (face | level << 8, cx, cy),
// for their whole volume to be generated.
struct Reach {
    count: atomic<u32>,
    keys: array<vec4<i32>>,
}
@group(0) @binding(13) var<storage, read_write> reach: Reach;

// Chunk `slot`'s frame relative to the eye.
fn chunk_frame(slot: u32) -> ChunkDraw {
    let s = statics[slot];
    let origin = (s.origin_hi.xyz - params.eye_hi.xyz) + (s.origin_lo.xyz - params.eye_lo.xyz);
    let o = -vec3<f32>(dot(s.i0.xyz, origin), dot(s.i1.xyz, origin), dot(s.i2.xyz, origin));
    return ChunkDraw(vec4<f32>(s.m0.xyz, origin.x), vec4<f32>(s.m1.xyz, origin.y), vec4<f32>(s.m2.xyz, origin.z),
        vec4<f32>(s.i0.xyz, o.x), vec4<f32>(s.i1.xyz, o.y), vec4<f32>(s.i2.xyz, o.z), s.box);
}
COUNTERS
// Steps this invocation's primary walk took (counting builds; the step
// heatmap, debug mode 2).
var<private> heat: u32;

const NONE: u32 = 0xffffffffu;

fn key_hash(k: vec4<i32>) -> u32 {
    var h = bitcast<u32>(k.x) * 0x9E3779B1u;
    h = (h ^ bitcast<u32>(k.y)) * 0x85EBCA77u;
    h = (h ^ bitcast<u32>(k.z)) * 0xC2B2AE3Du;
    h = (h ^ bitcast<u32>(k.w)) * 0x27D4EB2Fu;
    return h ^ (h >> 15u);
}

// The entry of key `k`, or NONE.
fn find_entry(k: vec4<i32>) -> u32 {
    var i = key_hash(k) & params.table.x;
    for (var probe = 0u; probe < 64u; probe++) {
        let e = chunk_table[i * 2u];
        if e.x == -1 { return NONE; }
        if all(e == k) { return i; }
        i = (i + 1u) & params.table.x;
    }
    return NONE;
}

// ---- 0. Cull ----
// The drawn chunks in view, built into the seed's draw list on the GPU:
// those holding the eye first (drawn by their back faces), the others in
// buckets of distance, nearest first (early depth tests reject boxes
// behind nearer hits), and the two indirect draws.

struct Cull {
    // Boxes holding the eye, boxes in view.
    inside: atomic<u32>,
    total: atomic<u32>,
    // Per bin (0: holding the eye, then distance buckets): its boxes, then
    // the next free place in the list.
    bins: array<atomic<u32>, 65>,
    cursor: array<atomic<u32>, 65>,
}

// The drawn chunks' slots (`params.table.z` of them).
@group(1) @binding(17) var<storage, read> drawn_slots: array<u32>;
@group(1) @binding(18) var<storage, read_write> cull: Cull;
// Each drawn chunk's bin (NONE: out of view).
@group(1) @binding(19) var<storage, read_write> cull_bins: array<u32>;
@group(1) @binding(20) var<storage, read_write> draw_out: array<u32>;
// The seed's indirect draws: boxes holding the eye, then the others.
@group(1) @binding(21) var<storage, read_write> seed_args: array<u32, 8>;

// Bin of drawn chunk `id.x`: the frustum against its bounding sphere (the
// box under its tilted top), the eye (with the near plane around it) in
// its box, else its distance.
@compute @workgroup_size(64)
fn cull_bin(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.table.z { return; }
    let slot = drawn_slots[id.x];
    let c = chunk_frame(slot);
    let top = chunk_meta[slot].top;
    let grow = statics[slot].m0.w;
    let z0 = c.box.x - grow;
    let z1 = select(clamp(top.z + max(top.x, 0.0) * 64.0 + max(top.y, 0.0) * 64.0, c.box.y, 64.0), c.box.y + grow, grow > 0.0);
    let m = mat3x3<f32>(c.m0.xyz, c.m1.xyz, c.m2.xyz);
    let centre = vec3<f32>(c.m0.w, c.m1.w, c.m2.w) + m * vec3<f32>(32.0, 32.0, (z0 + z1) * 0.5);
    let half_z = (z1 - z0) * 0.5;
    let half = 32.0 + grow;
    let radius = max(length(m * vec3<f32>(half, half, half_z)), length(m * vec3<f32>(half, -half, half_z)));
    let vp = params.view_proj;
    let r0 = vec4<f32>(vp[0].x, vp[1].x, vp[2].x, vp[3].x);
    let r1 = vec4<f32>(vp[0].y, vp[1].y, vp[2].y, vp[3].y);
    let r3 = vec4<f32>(vp[0].w, vp[1].w, vp[2].w, vp[3].w);
    let planes = array<vec4<f32>, 5>(r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3);
    var bin = NONE;
    var seen = true;
    for (var i = 0u; i < 5u; i++) {
        let p = planes[i] / length(planes[i].xyz);
        if dot(p.xyz, centre) + p.w < -radius { seen = false; }
    }
    if seen {
        let distance = length(centre);
        bin = 1u + min(63u, u32(log2(max(distance, 1.0)) * 2.0));
        let near = params.sun.w;
        if distance < radius * 1.01 + near * 4.0 {
            let o = vec3<f32>(c.i0.w, c.i1.w, c.i2.w);
            let margin = vec3<f32>(length(c.i0.xyz), length(c.i1.xyz), length(c.i2.xyz)) * near * 2.0;
            if all(o > vec3<f32>(-grow - margin.x, -grow - margin.y, z0 - margin.z)) && all(o < vec3<f32>(64.0 + grow + margin.x, 64.0 + grow + margin.y, z1 + margin.z)) { bin = 0u; }
        }
        atomicAdd(&cull.bins[bin], 1u);
    }
    cull_bins[id.x] = bin;
}

// Each bin's place in the list, and the seed's draws.
@compute @workgroup_size(1)
fn cull_offsets() {
    var at = 0u;
    for (var b = 0u; b < 65u; b++) {
        let n = atomicLoad(&cull.bins[b]);
        atomicStore(&cull.cursor[b], at);
        at += n;
    }
    let inside = atomicLoad(&cull.bins[0]);
    atomicStore(&cull.inside, inside);
    atomicStore(&cull.total, at);
    seed_args = array<u32, 8>(36u, inside, 0u, 0u, 36u, at - inside, 0u, 0u);
}

@compute @workgroup_size(64)
fn cull_scatter(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.table.z { return; }
    let bin = cull_bins[id.x];
    if bin == NONE { return; }
    draw_out[atomicAdd(&cull.cursor[bin], 1u)] = drawn_slots[id.x];
}

// ---- 1. Seed ----

struct VOut {
    @builtin(position) position: vec4<f32>,
    // The box's chunk slot.
    @location(0) @interpolate(flat) instance: u32,
    @location(1) world: vec3<f32>,
}

// Per pixel: 0 before the first round, then where its ray left its last
// box without a hit, or -1 once resolved (hit or no box left).
@group(1) @binding(5) var exits: texture_2d<f32>;

// Box corners (bit 0 x, bit 1 y, bit 2 z), outward faces counter-clockwise.
const CORNERS = array<u32, 36>(
    0u, 4u, 6u, 0u, 6u, 2u,
    1u, 3u, 7u, 1u, 7u, 5u,
    0u, 1u, 5u, 0u, 5u, 4u,
    2u, 6u, 7u, 2u, 7u, 3u,
    0u, 2u, 3u, 0u, 3u, 1u,
    4u, 5u, 7u, 4u, 7u, 6u,
);

// The boxes holding the eye (`Cull::inside`), first in the draw list.
@group(1) @binding(22) var<storage, read> culled: array<u32>;

// Boxes the eye is outside of, after those holding it.
@vertex
fn seed_vertex(@builtin(vertex_index) v: u32, @builtin(instance_index) n: u32) -> VOut {
    return seed_box(draw_list[culled[0] + n], v, false);
}

// Boxes holding the eye.
@vertex
fn seed_vertex_inside(@builtin(vertex_index) v: u32, @builtin(instance_index) n: u32) -> VOut {
    return seed_box(draw_list[n], v, true);
}

// Corner `v` of chunk `slot`'s box (its back faces when `inside`).
fn seed_box(slot: u32, v: u32, inside: bool) -> VOut {
    let c = chunk_frame(slot);
    let corner = CORNERS[v];
    // An exact chunk's cells reach past its frame's box by `grow` cells:
    // its box holds them, flat over them.
    let grow = statics[slot].m0.w;
    var x = f32(corner & 1u) * (64.0 + 2.0 * grow) - grow;
    // Mirroring x maps the box onto itself and reverses its winding: a
    // left-handed frame, or a box drawn from inside (its back faces).
    let mirrored = (bitcast<u32>(c.box.w) & 1u) != 0u;
    if mirrored != inside { x = 64.0 - x; }
    // The box's top lies on its chunk's plane over the solid cells (never
    // below its bottom): raising a corner keeps the faces over the cells.
    let y = f32((corner >> 1u) & 1u) * (64.0 + 2.0 * grow) - grow;
    let top = chunk_meta[slot].top;
    let plane = select(max(c.box.x, top.z + top.x * x + top.y * y), c.box.y + grow, grow > 0.0);
    let local = vec3<f32>(x, y, select(c.box.x - grow, plane, (corner & 4u) != 0u));
    let world = vec3<f32>(c.m0.w, c.m1.w, c.m2.w) + c.m0.xyz * local.x + c.m1.xyz * local.y + c.m2.xyz * local.z;
    var out: VOut;
    // The camera's reversed depth, near / w (w: the view depth): precise to
    // a few parts in 10^8 of the distance anywhere, so boxes 100 m apart do
    // not tie 10 km away and seed a pixel with the farther box.
    out.position = params.view_proj * vec4<f32>(world, 1.0);
    out.instance = slot;
    out.world = world;
    return out;
}

// Boxes the eye is outside of: the box's front face is the depth. Later
// rounds keep only the boxes beyond where the pixel's ray left the last.
@fragment
fn seed_fragment(in: VOut) -> @location(0) u32 {
    let exit = textureLoad(exits, vec2<i32>(in.position.xy), 0).r;
    if exit < 0.0 || (exit > 0.0 && length(in.world) <= exit) { discard; }
    return in.instance;
}

struct SeedInside {
    @location(0) instance: u32,
    @builtin(frag_depth) depth: f32,
}

// Boxes holding the eye, drawn first by their back faces: entered at the
// eye, nearer than any other box (the reversed depth's nearest).
@fragment
fn seed_fragment_inside(in: VOut) -> SeedInside {
    return SeedInside(in.instance, 1.0);
}

// ---- 2. Trace ----

@group(1) @binding(0) var seed: texture_2d<u32>;
@group(1) @binding(1) var out_visibility: texture_storage_2d<rg32uint, write>;
@group(1) @binding(2) var out_colour: texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(3) var out_distance: texture_storage_2d<r32float, write>;
@group(1) @binding(6) var out_exits: texture_storage_2d<r32float, read_write>;

fn bit64(m: vec2<u32>, b: u32) -> bool {
    return ((select(m.x, m.y, b >= 32u) >> (b & 31u)) & 1u) != 0u;
}

fn rank64(m: vec2<u32>, b: u32) -> u32 {
    if b < 32u { return countOneBits(m.x & ((1u << b) - 1u)); }
    return countOneBits(m.x) + countOneBits(m.y & ((1u << (b - 32u)) - 1u));
}

struct Hit {
    t: f32,
    cell: vec3<i32>,
    axis: u32,
    hit: bool,
    // The hit cell's material pair (`shown_material` picks one).
    material: u32,
    // Whether the ray moved up the axis it entered by.
    positive: bool,
}

// Whether cell `c` of chunk `slot` is solid (outside the chunk: no).
fn solid_at(slot: u32, c: vec3<i32>) -> bool {
    if any(c < vec3<i32>(0)) || any(c > vec3<i32>(63)) { return false; }
    let root = chunk_meta[slot];
    let m = u32((c.x >> 4u) + ((c.y >> 4u) << 2u) + ((c.z >> 4u) << 4u));
    if !bit64(root.root_occ, m) { return false; }
    if bit64(root.root_full, m) { return true; }
    let node = mids[slot * 64u + m];
    let l = u32(((c.x >> 2u) & 3) + (((c.y >> 2u) & 3) << 2u) + (((c.z >> 2u) & 3) << 4u));
    if !bit64(node.xy, l) { return false; }
    if bit64(node.zw, l) { return true; }
    let index = mid_leaves[slot * 64u + m] + rank64(node.xy & ~node.zw, l);
    return bit64(leaves[leaf_at_page(slot, index)], u32((c.x & 3) + ((c.y & 3) << 2u) + ((c.z & 3) << 4u)));
}

// The heightfield's normal stored with cell `c`'s leaf (chunk `slot`,
// frame `f`), in the world: the exact slope where the leaf holds natural
// tops; zero where none (an edit, an overhang, a cell under the surface).
// Lit by it, cells of a pixel or less read as the ground they sample, not
// as the terraces their whole-cell tops make on a gentle slope.
fn leaf_ground(slot: u32, c: vec3<i32>, f: ChunkDraw) -> vec3<f32> {
    if any(c < vec3<i32>(0)) || any(c > vec3<i32>(63)) { return vec3<f32>(0.0); }
    let root = chunk_meta[slot];
    let m = u32((c.x >> 4u) + ((c.y >> 4u) << 2u) + ((c.z >> 4u) << 4u));
    if !bit64(root.root_occ, m) || bit64(root.root_full, m) { return vec3<f32>(0.0); }
    let node = mids[slot * 64u + m];
    let l = u32(((c.x >> 2u) & 3) + (((c.y >> 2u) & 3) << 2u) + (((c.z >> 2u) & 3) << 4u));
    if !bit64(node.xy, l) || bit64(node.zw, l) { return vec3<f32>(0.0); }
    let index = leaf_at_page(slot, mid_leaves[slot * 64u + m] + rank64(node.xy & ~node.zw, l));
    let e = (leaf_normals[index >> 1u] >> ((index & 1u) * 16u)) & 0xffffu;
    if e == 0u { return vec3<f32>(0.0); }
    let g = tan((vec2<f32>(f32(e & 0xffu), f32(e >> 8u)) - 128.0) * (3.14159265 / 252.0));
    let n = -g.x * f.i0.xyz - g.y * f.i1.xyz + f.i2.xyz;
    let length2 = dot(n, n);
    if !(length2 > 1e-12) { return vec3<f32>(0.0); }
    return n * inverseSqrt(length2);
}

// The material pair of the stored leaf holding cell `c` of chunk `slot`
// (NONE: the leaf is empty, full or past the chunk).
fn stored_leaf_pair(slot: u32, c: vec3<i32>) -> u32 {
    if any(c < vec3<i32>(0)) || any(c > vec3<i32>(63)) { return NONE; }
    let root = chunk_meta[slot];
    let m = u32((c.x >> 4u) + ((c.y >> 4u) << 2u) + ((c.z >> 4u) << 4u));
    if !bit64(root.root_occ, m) || bit64(root.root_full, m) { return NONE; }
    let node = mids[slot * 64u + m];
    let l = u32(((c.x >> 2u) & 3) + (((c.y >> 2u) & 3) << 2u) + (((c.z >> 2u) & 3) << 4u));
    if !bit64(node.xy, l) || bit64(node.zw, l) { return NONE; }
    let index = leaf_at_page(slot, mid_leaves[slot * 64u + m] + rank64(node.xy & ~node.zw, l));
    return (leaf_materials[index >> 1u] >> ((index & 1u) * 16u)) & 0xffffu;
}

// A hit cell's material from its leaf's (or mid's) two (`pair`: what a
// walk's hit carries): the surface's with air above it, the subsurface's
// under other cells. A ray entering by the top face (`from_above`) came
// through that air. Resolved once a ray has hit, not in the walk: its
// lookups in the chunk over (`solid_over`) inside the walk's loop cost the
// loop registers (+45% on a mountain's primary rays).
fn shown_material(pair: u32, slot: u32, c: vec3<i32>, from_above: bool) -> u32 {
    return select((pair >> 8u) & 0xffu, pair & 0xffu, from_above || !solid_over(slot, c));
}

// Whether the cell over cell `c` of chunk `slot` is solid: in the chunk;
// over its top layer, in the chunk over it (its tile's next layer), else
// rock in its tile's solid gap. (Counting a top layer's cells as under air
// drew the surface's material along every chunk layer a dug wall crosses.)
fn solid_over(slot: u32, c: vec3<i32>) -> bool {
    if c.z < 63 { return solid_at(slot, c + vec3<i32>(0, 0, 1)); }
    let key = chunk_keys[slot];
    let e = find_entry(vec4<i32>(key.xyz, key.w + 1));
    if e != NONE { return solid_at(bitcast<u32>(chunk_table[e * 2u + 1u].x), vec3<i32>(c.xy, 0)); }
    let g = gap_of(key.x, key.yz);
    return key.w + 1 >= g.x && key.w + 1 < g.y;
}

// The first solid cell of chunk `slot` along o + t d from t0 to t1, entering
// through face `axis`; cells [0, 64) across, layers z.
fn trace_chunk(slot: u32, o: vec3<f32>, d: vec3<f32>, t0: f32, t1: f32, axis0: u32, z: vec2<i32>, counted: bool) -> Hit {
    let root = chunk_meta[slot];
    let inv = 1.0 / select(d, vec3<f32>(1e-30), d == vec3<f32>(0.0));
    let o_inv = o * inv;
    let positive = d > vec3<f32>(0.0);
    var t = t0;
    var axis = axis0;
    var c = clamp(vec3<i32>(floor(o + d * t)), vec3<i32>(0, 0, z.x), vec3<i32>(63, 63, z.y - 1));
    // The mid node and leaf last read: steps inside them read nothing.
    var node_at = NONE;
    var node = vec4<u32>(0u);
    var first_leaf = 0u;
    var leaf_at = NONE;
    var stored = 0u;
    var leaf = vec2<u32>(0u);
    for (var n = 0u; n < 512u; n++) {
        WCOUNT(1)
        let m = u32((c.x >> 4u) + ((c.y >> 4u) << 2u) + ((c.z >> 4u) << 4u));
        var size = 16;
        if bit64(root.root_occ, m) {
            if bit64(root.root_full, m) { return Hit(t, c, axis, true, mid_materials[slot * 64u + m], positive[axis]); }
            if m != node_at {
                node = mids[slot * 64u + m];
                first_leaf = mid_leaves[slot * 64u + m];
                node_at = m;
            }
            let l = u32(((c.x >> 2u) & 3) + (((c.y >> 2u) & 3) << 2u) + (((c.z >> 2u) & 3) << 4u));
            size = 4;
            if bit64(node.xy, l) {
                if bit64(node.zw, l) { return Hit(t, c, axis, true, mid_materials[slot * 64u + m], positive[axis]); }
                let index = first_leaf + rank64(node.xy & ~node.zw, l);
                if index != leaf_at {
                    stored = leaf_at_page(slot, index);
                    leaf = leaves[stored];
                    leaf_at = index;
                }
                size = 1;
                if bit64(leaf, u32((c.x & 3) + ((c.y & 3) << 2u) + ((c.z & 3) << 4u))) {
                    return Hit(t, c, axis, true, leaf_materials[stored >> 1u] >> ((stored & 1u) * 16u), positive[axis]);
                }
            }
        }
        // Leave the empty cube of `size` cells holding c.
        let b = c & vec3<i32>(~(size - 1));
        let bound = vec3<f32>(b) + select(vec3<f32>(0.0), vec3<f32>(f32(size)), positive);
        let tt = fma(bound, inv, -o_inv);
        let tn = min(tt.x, min(tt.y, tt.z));
        if tn >= t1 { break; }
        axis = select(select(2u, 1u, tt.y <= tt.z), 0u, tt.x <= tt.y && tt.x <= tt.z);
        t = max(tn, t);
        // A cell steps to its neighbour; a cube lands where the ray leaves it.
        if size != 1 {
            c = clamp(vec3<i32>(floor(o + d * t)), b, b + vec3<i32>(size - 1));
        }
        let step = select(b - vec3<i32>(1), b + vec3<i32>(size), positive);
        if axis == 0u { c.x = step.x; } else if axis == 1u { c.y = step.y; } else { c.z = step.z; }
        if any(c < vec3<i32>(0, 0, z.x)) || any(c > vec3<i32>(63, 63, z.y - 1)) { break; }
    }
    return Hit(t1, c, axis, false, 0u, false);
}

// ---- Exact cells ----
// Chunks and air of level `params.planet.z` and coarser are walked in the
// grid's own cells, not in an affine frame: on a face, cells of constant i
// or j lie between planes through the planet's centre (equal angles), and
// layers between spheres about it, so a ray meets their sides exactly, at
// the root of a linear function of t (a plane) or a quadratic (a sphere).
// Across a chunk an affine frame strays from the sphere by 512 x cell /
// radius of a cell: half a cell at an Earth's coarsest level, whose tiles
// span 600 km (facets, cracks between tiles, haze by the facets' heights).
// Finer levels keep their frames: exact there to under a hundredth of a
// cell, and precise where planet-centred f32 positions are not.

const QUARTER_PI: f32 = 0.78539816;

// A ray planet-centred: origin p, unit direction d, dot(p, d) and
// |p|^2 - R^2 (the eye's from f64, moved to the origin).
struct ExactRay {
    p: vec3<f32>,
    d: vec3<f32>,
    b: f32,
    c: f32,
}

// The ray org + t dir (eye-relative).
fn exact_ray(org: vec3<f32>, dir: vec3<f32>) -> ExactRay {
    let e = params.up.xyz * params.up.w;
    return ExactRay(e + org, dir, dot(e, dir) + dot(org, dir), params.planet.w + 2.0 * dot(e, org) + dot(org, org));
}

// Whether boxes of `level` are walked in the grid's own cells.
fn exact_level(level: i32) -> bool {
    return !PLANE_WORLD && level >= i32(params.planet.z);
}

// The angle a cell of `level` spans across a face.
fn cell_angle(level: i32) -> f32 {
    return 1.5707964 / f32(params.table.y) * f32(1u << u32(level));
}

// A ray's first stretch in a box of cells from a time on: [t0, t1], and
// the sides it enters and leaves by (2 x axis, + 1 on the high side; 6:
// none, or out of the face's side of the centre).
struct Span {
    t0: f32,
    t1: f32,
    enter: u32,
    leave: u32,
}

// `s` where g0 + t g1 >= 0 (side `code`).
fn half_line(s: Span, g0: f32, g1: f32, code: u32) -> Span {
    var o = s;
    if g1 > 0.0 {
        let t = -g0 / g1;
        if t > o.t0 {
            o.t0 = t;
            o.enter = code;
        }
    } else if g1 < 0.0 {
        let t = -g0 / g1;
        if t < o.t1 {
            o.t1 = t;
            o.leave = code;
        }
    } else if g0 < 0.0 {
        o.t1 = o.t0 - 1.0;
    }
    return o;
}

// Where the ray meets the sphere of layer bound `k` of `level` (radius R
// + k cells): the roots of t^2 + 2 b t + c, (smaller, larger), or none
// (x > y: missed, or no sphere).
fn shell_roots(ray: ExactRay, level: i32, k: i32) -> vec2<f32> {
    let h = f32(k) * params.planet.y * f32(1u << u32(level));
    if params.planet.x + h <= 0.0 { return vec2<f32>(1.0, -1.0); }
    let c = ray.c - h * (2.0 * params.planet.x + h);
    let disc = ray.b * ray.b - c;
    if disc < 0.0 { return vec2<f32>(1.0, -1.0); }
    // Stable: q and c / q.
    let q = -ray.b - select(-1.0, 1.0, ray.b >= 0.0) * sqrt(disc);
    if q == 0.0 { return vec2<f32>(0.0); }
    return vec2<f32>(min(q, c / q), max(q, c / q));
}

// The ray's first stretch from `after` in cells [lo, hi) of `level` on
// `face` (the face's indices).
fn box_span(ray: ExactRay, face: i32, level: i32, lo: vec3<i32>, hi: vec3<i32>, after: f32) -> Span {
    var s = Span(after, 3.0e38, 6u, 6u);
    let n = vec3<f32>(FACE_N[face]);
    let np = dot(n, ray.p);
    let nd = dot(n, ray.d);
    // On the face's side of the centre (its cells' angles hold there).
    s = half_line(s, np, nd, 6u);
    let delta = cell_angle(level);
    for (var axis = 0; axis < 2; axis++) {
        let a = vec3<f32>(select(FACE_B[face], FACE_A[face], axis == 0));
        let ap = dot(a, ray.p);
        let ad = dot(a, ray.d);
        let tl = tan(f32(lo[axis]) * delta - QUARTER_PI);
        let th = tan(f32(hi[axis]) * delta - QUARTER_PI);
        // Index >= lo: ap - tl np >= 0; < hi: th np - ap > 0.
        s = half_line(s, ap - tl * np, ad - tl * nd, u32(2 * axis));
        s = half_line(s, th * np - ap, th * nd - ad, u32(2 * axis + 1));
    }
    // Under the high sphere.
    let outer = shell_roots(ray, level, hi.z);
    if outer.x > outer.y {
        s.t1 = s.t0 - 1.0;
        return s;
    }
    if outer.x > s.t0 {
        s.t0 = outer.x;
        s.enter = 5u;
    }
    if outer.y < s.t1 {
        s.t1 = outer.y;
        s.leave = 5u;
    }
    // Over the low one: the first stretch out of it.
    let inner = shell_roots(ray, level, lo.z);
    if inner.x <= inner.y && inner.y > s.t0 && inner.x < s.t1 {
        if inner.x > s.t0 {
            s.t1 = inner.x;
            s.leave = 4u;
        } else {
            s.t0 = inner.y;
            s.enter = 4u;
        }
    }
    return s;
}

// When and by which side a ray inside cells [lo, hi) of `level` on `face`
// at time `t` leaves them (`box_span`'s t1 and leave): only the sides
// ahead of it. Along a ray a face's index moves one way (u = a.p / n.p has
// a derivative of constant sign); its height falls until the ray's
// closest approach, then rises.
fn box_exit(ray: ExactRay, face: i32, level: i32, lo: vec3<i32>, hi: vec3<i32>, t: f32) -> Span {
    var s = Span(t, 3.0e38, 6u, 6u);
    let n = vec3<f32>(FACE_N[face]);
    let np = dot(n, ray.p);
    let nd = dot(n, ray.d);
    s = half_line(s, np, nd, 6u);
    let delta = cell_angle(level);
    for (var axis = 0; axis < 2; axis++) {
        let a = vec3<f32>(select(FACE_B[face], FACE_A[face], axis == 0));
        let ap = dot(a, ray.p);
        let ad = dot(a, ray.d);
        if ad * np - ap * nd > 0.0 {
            let th = tan(f32(hi[axis]) * delta - QUARTER_PI);
            s = half_line(s, th * np - ap, th * nd - ad, u32(2 * axis + 1));
        } else {
            let tl = tan(f32(lo[axis]) * delta - QUARTER_PI);
            s = half_line(s, ap - tl * np, ad - tl * nd, u32(2 * axis));
        }
    }
    if ray.b + t < 0.0 {
        let inner = shell_roots(ray, level, lo.z);
        if inner.x <= inner.y && inner.x > t && inner.x < s.t1 {
            s.t1 = inner.x;
            s.leave = 4u;
        }
    }
    let outer = shell_roots(ray, level, hi.z);
    if outer.x <= outer.y && outer.y < s.t1 {
        s.t1 = outer.y;
        s.leave = 5u;
    }
    return s;
}

// The ray's cell coordinates of `level` on `face` (continuous) at time t.
fn exact_cell(ray: ExactRay, face: i32, level: i32, t: f32) -> vec3<f32> {
    let q = ray.p + ray.d * t;
    let n = dot(vec3<f32>(FACE_N[face]), q);
    let delta = cell_angle(level);
    let i = (atan2(dot(vec3<f32>(FACE_A[face]), q), n) + QUARTER_PI) / delta;
    let j = (atan2(dot(vec3<f32>(FACE_B[face]), q), n) + QUARTER_PI) / delta;
    // Height over the datum from |q|^2 - R^2 (precise near it).
    let e = ray.c + t * (2.0 * ray.b + t);
    let k = e / (length(q) + params.planet.x) / (params.planet.y * f32(1u << u32(level)));
    return vec3<f32>(i, j, k);
}

// `trace_chunk` in the grid's own cells: chunk `slot`, cells `origin` +
// [0, 64) of `level` on `face`, from t0 to t1, entering through face
// `axis0` (up it: `positive0`).
fn trace_exact(slot: u32, ray: ExactRay, face: i32, level: i32, origin: vec3<i32>, t0: f32, t1: f32, axis0: u32, positive0: bool, z: vec2<i32>, counted: bool) -> Hit {
    let root = chunk_meta[slot];
    var t = t0;
    var axis = axis0;
    var positive = positive0;
    var c = clamp(vec3<i32>(floor(exact_cell(ray, face, level, t))) - origin, vec3<i32>(0, 0, z.x), vec3<i32>(63, 63, z.y - 1));
    var node_at = NONE;
    var node = vec4<u32>(0u);
    var first_leaf = 0u;
    var leaf_at = NONE;
    var stored = 0u;
    var leaf = vec2<u32>(0u);
    for (var n = 0u; n < 512u; n++) {
        WCOUNT(1)
        let m = u32((c.x >> 4u) + ((c.y >> 4u) << 2u) + ((c.z >> 4u) << 4u));
        var size = 16;
        if bit64(root.root_occ, m) {
            if bit64(root.root_full, m) { return Hit(t, c, axis, true, mid_materials[slot * 64u + m], positive); }
            if m != node_at {
                node = mids[slot * 64u + m];
                first_leaf = mid_leaves[slot * 64u + m];
                node_at = m;
            }
            let l = u32(((c.x >> 2u) & 3) + (((c.y >> 2u) & 3) << 2u) + (((c.z >> 2u) & 3) << 4u));
            size = 4;
            if bit64(node.xy, l) {
                if bit64(node.zw, l) { return Hit(t, c, axis, true, mid_materials[slot * 64u + m], positive); }
                let index = first_leaf + rank64(node.xy & ~node.zw, l);
                if index != leaf_at {
                    stored = leaf_at_page(slot, index);
                    leaf = leaves[stored];
                    leaf_at = index;
                }
                size = 1;
                if bit64(leaf, u32((c.x & 3) + ((c.y & 3) << 2u) + ((c.z & 3) << 4u))) {
                    return Hit(t, c, axis, true, leaf_materials[stored >> 1u] >> ((stored & 1u) * 16u), positive);
                }
            }
        }
        // Leave the empty cube of `size` cells holding c.
        let b = c & vec3<i32>(~(size - 1));
        let s = box_exit(ray, face, level, origin + b, origin + b + vec3<i32>(size), t);
        if s.t1 >= t1 || s.leave >= 6u { break; }
        axis = s.leave >> 1u;
        positive = (s.leave & 1u) == 1u;
        t = max(s.t1, t);
        if size != 1 {
            c = clamp(vec3<i32>(floor(exact_cell(ray, face, level, t))) - origin, b, b + vec3<i32>(size - 1));
        }
        let step = select(b - vec3<i32>(1), b + vec3<i32>(size), vec3<bool>(positive));
        if axis == 0u { c.x = step.x; } else if axis == 1u { c.y = step.y; } else { c.z = step.z; }
        if any(c < vec3<i32>(0, 0, z.x)) || any(c > vec3<i32>(63, 63, z.y - 1)) { break; }
    }
    return Hit(t1, c, axis, false, 0u, false);
}

// The frame shading reads at cell `cell` of chunk `slot`: the chunk's, or
// at an exact level the grid's own axes at the cell's centre (its
// mapping's derivatives there), eye-relative.
fn cell_frame(slot: u32, cell: vec3<i32>) -> ChunkDraw {
    let c = chunk_frame(slot);
    let key = chunk_keys[slot];
    let level = key.x >> 8u;
    if !exact_level(level) { return c; }
    let face = key.x & 7;
    let g = vec3<f32>(key.yzw * 64 + cell) + 0.5;
    let delta = cell_angle(level);
    let u = tan(g.x * delta - QUARTER_PI);
    let v = tan(g.y * delta - QUARTER_PI);
    let a = vec3<f32>(FACE_A[face]);
    let b = vec3<f32>(FACE_B[face]);
    let w = vec3<f32>(FACE_N[face]) + a * u + b * v;
    let len = length(w);
    let dir = w / len;
    let size = params.planet.y * f32(1u << u32(level));
    let radius = params.planet.x + g.z * size;
    let mx = (a - dir * dot(dir, a)) * (radius * (1.0 + u * u) * delta / len);
    let my = (b - dir * dot(dir, b)) * (radius * (1.0 + v * v) * delta / len);
    let mz = dir * size;
    let origin = dir * radius - params.up.xyz * params.up.w - (mx * (f32(cell.x) + 0.5) + my * (f32(cell.y) + 0.5) + mz * (f32(cell.z) + 0.5));
    // M^-1's rows: cross products over the determinant.
    let det = dot(mx, cross(my, mz));
    let i0 = cross(my, mz) / det;
    let i1 = cross(mz, mx) / det;
    let i2 = cross(mx, my) / det;
    return ChunkDraw(vec4<f32>(mx, origin.x), vec4<f32>(my, origin.y), vec4<f32>(mz, origin.z), vec4<f32>(i0, -dot(i0, origin)), vec4<f32>(i1, -dot(i1, origin)), vec4<f32>(i2, -dot(i2, origin)), c.box);
}

// The eye ray through pixel `p` (camera-relative, unit length).
fn pixel_ray(p: vec2<u32>, size: vec2<u32>) -> vec3<f32> {
    let ndc = vec2<f32>((f32(p.x) + 0.5) / f32(size.x) * 2.0 - 1.0, 1.0 - (f32(p.y) + 0.5) / f32(size.y) * 2.0);
    let far = params.inverse_view_proj * vec4<f32>(ndc, 0.5, 1.0);
    return normalize(far.xyz / far.w);
}

// What a pixel with dither `r` does at a tile of `state` (`drawn::choice`):
// 1 shows it, 0 looks finer (an ancestor, or the coarse side of a switch
// between levels this pixel does not show), 2 coarser (the fine side).
// A fading tile's state holds its first frame: the fade's progress q (of
// 1024) comes from this frame's.
fn choice(state: i32, r: i32) -> i32 {
    if state < 2 { return state; }
    let f = state - 2;
    let elapsed = (params.table.w - u32(f >> 2u)) & 0xfffffu;
    let q = i32(min(elapsed * 1024u / FADE_FRAMESu, 1024u));
    if (r < q) == ((f & 1) != 0) { return 1; }
    return select(2, 0, (f & 2) != 0);
}

// A pixel's dither in [0, 1024) for switches between levels: interleaved
// gradient noise moved each frame, so TSR blends the two levels.
fn fade_dither(p: vec2<u32>) -> i32 {
    let shift = 5.588238 * f32(params.table.w % 64u);
    let q = vec2<f32>(p) + shift;
    return i32(fract(52.9829189 * fract(0.06711056 * q.x + 0.00583715 * q.y)) * 1024.0);
}

// Cell `cell` of `level` (at `frac` within it) as a cell of `level + diff`:
// coarser for diff > 0, finer for diff < 0.
fn cell_at(cell: vec3<i32>, frac: vec3<f32>, diff: i32) -> vec3<i32> {
    if diff >= 0 { return cell >> vec3<u32>(u32(diff)); }
    let n = u32(-diff);
    return (cell << vec3<u32>(n)) + vec3<i32>(floor(frac * f32(1u << n)));
}

// What a ray meets at a cell: a drawn chunk (`slot`), or air to cross,
// cells [lo, hi) of `level` (LOST: off the drawn cut; BELOW: under a drawn
// tile's chunks, lo its tile and level, hi.x its face), in the frame of
// chunk `frame` when it is finer than the ray's (else NONE).
struct Place {
    slot: u32,
    level: i32,
    lo: vec3<i32>,
    hi: vec3<i32>,
    frame: u32,
}

const LOST: i32 = 99;
const BELOW: i32 = 98;
// Rock between a tile's chunks (`drawn::Gap`): a hit where the ray enters.
const ROCK: i32 = 97;
// A column entry's layer key (`drawn::COLUMN`).
const COLUMN: i32 = -2147483647 - 1;
// A gap entry's layer key (`drawn::GAP`): its solid chunk layers [x, y),
// z 1 when air (a hollow) lies under the tile's chunks.
const GAP: i32 = -2147483647;

// Tile `tile` of `tile_key`'s gap (zero: none).
fn gap_of(tile_key: i32, tile: vec2<i32>) -> vec4<i32> {
    let e = find_entry(vec4<i32>(tile_key, tile, GAP));
    if e == NONE { return vec4<i32>(0); }
    return chunk_table[e * 2u + 1u];
}

// Cube faces' bases: normal, a, b (`grid::FACE_BASIS`).
const FACE_N = array<vec3<i32>, 6>(vec3<i32>(1, 0, 0), vec3<i32>(-1, 0, 0), vec3<i32>(0, 1, 0), vec3<i32>(0, -1, 0), vec3<i32>(0, 0, 1), vec3<i32>(0, 0, -1));
const FACE_A = array<vec3<i32>, 6>(vec3<i32>(0, 0, -1), vec3<i32>(0, 0, 1), vec3<i32>(1, 0, 0), vec3<i32>(1, 0, 0), vec3<i32>(1, 0, 0), vec3<i32>(-1, 0, 0));
const FACE_B = array<vec3<i32>, 6>(vec3<i32>(0, 1, 0), vec3<i32>(0, 1, 0), vec3<i32>(0, 0, -1), vec3<i32>(0, 0, 1), vec3<i32>(0, 1, 0), vec3<i32>(0, 1, 0));

fn face_with_normal(n: vec3<i32>) -> i32 {
    if n.x != 0 { return select(1, 0, n.x > 0); }
    if n.y != 0 { return select(3, 2, n.y > 0); }
    return select(5, 4, n.z > 0);
}

// A cell past an edge of `face` (one tangential index outside [0, n), n
// cells to an edge at its level) as a cell of the face across it, and
// where in it. The faces are equal-angle: across the edge an index runs on
// mirrored, and along it cells meet exactly on the edge.
struct Across {
    face: i32,
    cell: vec3<i32>,
    frac: vec3<f32>,
    ok: bool,
}

fn across_edge(face: i32, cell: vec3<i32>, frac: vec3<f32>, n: i32) -> Across {
    let out_a = cell.x < 0 || cell.x >= n;
    let out_b = cell.y < 0 || cell.y >= n;
    // Inside, or past a corner.
    if out_a == out_b { return Across(face, cell, frac, false); }
    // The index across the edge and the one along it.
    let x = select(cell.y, cell.x, out_a);
    let fx = select(frac.y, frac.x, out_a);
    let y = select(cell.x, cell.y, out_a);
    let fy = select(frac.x, frac.y, out_a);
    let past = x >= n;
    let normal = FACE_N[face];
    let along = select(FACE_A[face], FACE_B[face], out_a);
    let next = face_with_normal(select(select(FACE_B[face], FACE_A[face], out_a) * -1, select(FACE_B[face], FACE_A[face], out_a), past));
    var out = Across(next, vec3<i32>(0, 0, cell.z), vec3<f32>(0.0, 0.0, frac.z), true);
    for (var t = 0; t < 2; t++) {
        let axis = select(FACE_B[next], FACE_A[next], t == 0);
        var c = 0;
        var f = 0.0;
        if all(axis == along) {
            c = y;
            f = fy;
        } else if all(axis == -along) {
            c = n - 1 - y;
            f = 1.0 - fy;
        } else if all(axis == normal) {
            c = select(n + x, 2 * n - 1 - x, past);
            f = select(fx, 1.0 - fx, past);
        } else {
            c = select(-1 - x, x - n, past);
            f = select(1.0 - fx, fx, past);
        }
        if t == 0 {
            out.cell.x = c;
            out.frac.x = f;
        } else {
            out.cell.y = c;
            out.frac.y = f;
        }
    }
    out.frac = clamp(out.frac, vec3<f32>(0.0), vec3<f32>(0.999));
    return out;
}

// No place for a ray at a cell: off the drawn cut, for reason `site` (counted
// by reason in counting builds, `TraceCounts::lost_sites`).
fn lost_at(site: u32) -> Place {
    LOST_SITE
    return Place(NONE, LOST, vec3<i32>(0), vec3<i32>(0), NONE);
}

// The place at cell `cell` of `level` (at `frac` within it) on `face`, for
// a pixel of dither `r`: the tile it shows over the column (this level's,
// else a coarser one where this level shows none, else a finer one under
// an ancestor's), then the chunk at its layer, an empty layer, or the air
// over the tile, grown to the coarsest column whose whole subtree lies
// under the cell. Air comes with a frame (`Place::frame`) when it is finer
// than the ray's, or on another face (`crossed`).
fn place_at(face: i32, level: i32, cell: vec3<i32>, frac: vec3<f32>, r: i32, crossed: bool) -> Place {
    // Mostly the next chunk of the same level.
    let same = find_entry(vec4<i32>(face | (level << 8), cell >> vec3<u32>(6u)));
    if same != NONE {
        let v = chunk_table[same * 2u + 1u];
        if choice(v.y, r) == 1 { return Place(bitcast<u32>(v.x), level, vec3<i32>(0), vec3<i32>(0), NONE); }
    }
    var diff = 0;
    var way = 0;
    var turned = false;
    var through_ancestors = true;
    var column = vec4<i32>(0);
    var shown = false;
    for (var n = 0u; n < 24u; n++) {
        let c = cell_at(cell, frac, diff);
        let e = find_entry(vec4<i32>(face | ((level + diff) << 8), c.xy >> vec2<u32>(6u), COLUMN));
        var go = 2;
        if e != NONE {
            column = chunk_table[e * 2u + 1u];
            go = choice(column.w, r);
            // Over an ancestor's whole subtree: air whichever tile shows
            // under it (no search down and climb back up). Not finer than
            // the ray: an ancestor has no chunk to cross its air in, and in
            // the ray's coarser cells its top would round.
            if column.w == 0 && diff >= 0 && c.z >= column.z {
                go = 1;
            }
        }
        if go == 1 {
            shown = true;
            break;
        }
        // Coarser; on the way up, past the ancestors of the hidden side of
        // a switch across levels.
        if go == 2 || (go == 0 && way > 0) {
            if way < 0 {
                // Down through undrawn ancestors only, to the fine side of a
                // switch that shows coarser: the tile shown is over where
                // the search began (a tile refined, then coarsened past).
                if turned || !through_ancestors { return lost_at(0u); }
                turned = true;
                diff = 0;
            }
            if level + diff >= 30 { LOST_WHERE return lost_at(1u); }
            way = 1;
            diff += 1;
        } else {
            // As deep as the tile shown (old fine tiles under an eye that
            // jumped up, a switch across many levels).
            if way > 0 || level + diff <= 0 { return lost_at(2u); }
            through_ancestors = through_ancestors && column.w == 0;
            way = -1;
            diff -= 1;
        }
    }
    if !shown { return lost_at(3u); }
    let c = cell_at(cell, frac, diff);
    let tile = c.xy >> vec2<u32>(6u);
    let layer = c.z >> 6u;
    let tile_key = face | ((level + diff) << 8);
    if c.z >= column.z {
        var top = column.z;
        var at = tile;
        var up = diff;
        for (var n = 0u; n < 24u && level + up < 30; n++) {
            let pc = cell_at(cell, frac, up + 1);
            let pe = find_entry(vec4<i32>(face | ((level + up + 1) << 8), pc.xy >> vec2<u32>(6u), COLUMN));
            if pe == NONE { break; }
            let ptop = chunk_table[pe * 2u + 1u].z;
            if pc.z < ptop { break; }
            up += 1;
            top = ptop;
            at = pc.xy >> vec2<u32>(6u);
        }
        // Finer than the ray's frame: crossed in the frame of the tile's top
        // chunk (in coarser cells the top would round; more than six levels
        // finer, a tile is under one of them: without a frame, lost).
        let f = frame_of(up < 0 || crossed, vec4<i32>(tile_key, tile, (column.z - 1) >> 6u));
        if (crossed || up < -6) && f == NONE && !exact_level(level + up) { return lost_at(4u); }
        return Place(NONE, level + up, vec3<i32>(at * 64, top), vec3<i32>(at * 64 + 64, top + (1 << 20)), f);
    }
    if layer < column.x {
        // Under the tile's chunks: air when they hang over a hollow, else
        // rock, reached through a cave mouth among them.
        if gap_of(tile_key, tile).z == 0 { return Place(NONE, BELOW, vec3<i32>(tile, level + diff), vec3<i32>(face, 0, 0), NONE); }
        let f = frame_of(diff < 0 || crossed, vec4<i32>(tile_key, tile, column.x));
        if (crossed || diff < -6) && f == NONE && !exact_level(level + diff) { return lost_at(5u); }
        return Place(NONE, level + diff, vec3<i32>(tile * 64, column.x * 64 - (1 << 20)), vec3<i32>(tile * 64 + 64, column.x * 64), f);
    }
    let s = find_entry(vec4<i32>(tile_key, tile, layer));
    if s != NONE { return Place(bitcast<u32>(chunk_table[s * 2u + 1u].x), level + diff, vec3<i32>(0), vec3<i32>(0), NONE); }
    // A layer among them without a chunk: rock in its solid gap, else air.
    let g = gap_of(tile_key, tile);
    if layer >= g.x && layer < g.y { return Place(NONE, ROCK, vec3<i32>(0), vec3<i32>(0), NONE); }
    let f = frame_of(diff < 0 || crossed, vec4<i32>(tile_key, tile, column.x));
    if (crossed || diff < -6) && f == NONE && !exact_level(level + diff) { return lost_at(6u); }
    return Place(NONE, level + diff, vec3<i32>(tile * 64, layer * 64), vec3<i32>(tile * 64 + 64, layer * 64 + 64), f);
}

// Chunk `k`'s slot when `wanted` and drawn, else NONE.
fn frame_of(wanted: bool, k: vec4<i32>) -> u32 {
    if !wanted { return NONE; }
    let e = find_entry(k);
    if e == NONE { return NONE; }
    return bitcast<u32>(chunk_table[e * 2u + 1u].x);
}

// Box corner `hi` in the cells of chunk `key`, kept on its face: a coarse
// tile can reach past the face's edge (its columns there are empty), and a
// ray crossing them would skip what lies across the edge.
fn clip_to_face(hi: vec3<i32>, key: vec4<i32>) -> vec3<i32> {
    let edge = vec2<i32>(i32(params.table.y >> u32(key.x >> 8u))) - key.yz * 64;
    return vec3<i32>(min(hi.xy, edge), hi.z);
}

// Cell corner `a` of level `level + diff` in the cells of chunk `key` (of
// `level`): exact on tile and layer bounds for diff >= -6, else rounded up
// (`up`) or down; far coarse corners are clamped.
fn to_frame(a: vec3<i32>, diff: i32, key: vec4<i32>, up: bool) -> vec3<i32> {
    let origin = key.yzw * 64;
    if diff < 0 {
        let n = u32(-diff);
        return ((a + select(0, (1 << n) - 1, up)) >> vec3<u32>(n)) - origin;
    }
    let n = vec3<u32>(u32(diff));
    let base = origin >> n;
    let limit = 1 << u32(max(28 - diff, 0));
    return (clamp(a - base, vec3<i32>(-limit), vec3<i32>(limit)) << n) - (origin - (base << n));
}

// A walk through the drawn cut: the chunk whose frame it is in (its last)
// and the box it crosses in that frame's cells, the chunk itself (`real`)
// or air.
struct Walk {
    slot: u32,
    lo: vec3<i32>,
    hi: vec3<i32>,
    real: bool,
}

// Where a walk ended: a hit (chunk, cell, the face it entered by and its
// side, material), or how far it went when it left the cut or ran out of
// boxes.
struct WalkEnd {
    hit: bool,
    t: f32,
    slot: u32,
    cell: vec3<i32>,
    axis: u32,
    sign: u32,
    // The hit's material pair (`Hit::material`).
    material: u32,
}

// The walk of ray org + t dir (org eye-relative, dir unit) from t_start to
// t_end, for
// a pixel of dither `r`, over at most `boxes` boxes (`counted`: in the
// counting build's counters, as `pixel`). A box of an exact level (a chunk
// or air) is crossed in the grid's own cells, a finer one in its frame.
fn walk(start: Walk, org: vec3<f32>, dir: vec3<f32>, t_start: f32, t_end: f32, r: i32, boxes: u32, counted: bool, pixel: vec2<u32>) -> WalkEnd {
    // Over the ceiling and rising, a ray meets nothing more (the squared
    // radius q0 + 2 b t + t^2 only grows once b + t > 0; a plane's height
    // q0 + b t once b > 0): it ends there, t -1.
    var q0 = params.ceiling.x + 2.0 * params.up.w * dot(params.up.xyz, org) + dot(org, org);
    var b = params.up.w * dot(params.up.xyz, dir) + dot(org, dir);
    if PLANE_WORLD {
        q0 = params.ceiling.x + org.y;
        b = dir.y;
    }
    var frame = chunk_frame(start.slot);
    var key = chunk_keys[start.slot];
    var cube_lo = start.lo;
    var cube_hi = start.hi;
    var real = start.real;
    var t_prev = t_start;
    // The box's chunk (in air, the last), and packed as in the visibility
    // buffer the cell of it the ray last left and the side it crossed into
    // the box by: what it meets under a tile's chunks or in rock is shown
    // there.
    var slot = start.slot;
    var last = (2u << 18u) | (1u << 20u);
    // A box of an exact level: its face and level in key.x, its cells (the
    // face's) in cube_lo and cube_hi. Such a walk starts where the ray
    // enters its seed box (an exact chunk's raster box reaches past its
    // cells, over its neighbours'): the drawn cut there.
    var exact = exact_level(key.x >> 8u);
    var place = exact;
    if exact {
        let o = vec3<f32>(frame.i0.w, frame.i1.w, frame.i2.w) + vec3<f32>(dot(frame.i0.xyz, org), dot(frame.i1.xyz, org), dot(frame.i2.xyz, org));
        let d = vec3<f32>(dot(frame.i0.xyz, dir), dot(frame.i1.xyz, dir), dot(frame.i2.xyz, dir));
        let inv = 1.0 / select(d, vec3<f32>(1e-30), d == vec3<f32>(0.0));
        let grow = statics[start.slot].m0.w;
        let ta = (vec3<f32>(-grow, -grow, frame.box.x - grow) - o) * inv;
        let tb = (vec3<f32>(64.0 + grow, 64.0 + grow, frame.box.y + grow) - o) * inv;
        let near = min(ta, tb);
        t_prev = max(max(near.x, max(near.y, near.z)), t_prev);
        cube_lo += key.yzw * 64;
        cube_hi += key.yzw * 64;
    }
    var chunk = 0u;
    for (; chunk < boxes; chunk++) {
        let over = select(b + t_prev > 0.0 && q0 + t_prev * (2.0 * b + t_prev) > 0.0, b > 0.0 && q0 + b * t_prev > 0.0, PLANE_WORLD);
        if over { return WalkEnd(false, -1.0, NONE, vec3<i32>(0), 0u, 0u, 0u); }
        WCOUNT(3)
        // Where the ray goes next: the face, level, cell and where in it.
        var face = key.x & 7;
        let level = key.x >> 8u;
        var cell = vec3<i32>(0);
        var frac = vec3<f32>(0.0);
        if exact {
            let ray = exact_ray(org, dir);
            // A coarse tile can reach past its face's edge: its columns
            // there are empty.
            let hi = vec3<i32>(min(cube_hi.xy, vec2<i32>(i32(params.table.y >> u32(level)))), cube_hi.z);
            if place {
                place = false;
                let q = exact_cell(ray, face, level, t_prev);
                cell = vec3<i32>(floor(q));
                frac = clamp(fract(q), vec3<f32>(0.0), vec3<f32>(0.999));
                let c = clamp(cell - cube_lo, vec3<i32>(0), vec3<i32>(63));
                last = u32(c.x) | (u32(c.y) << 6u) | (u32(c.z) << 12u) | (2u << 18u) | (1u << 20u);
            } else {
                if real {
                    // The walk over the chunk's occupied layers, entering
                    // by their side, else by the one it crossed into the
                    // chunk by.
                    let z = vec2<i32>(statics[slot].box.xy);
                    let occupied = box_span(ray, face, level, vec3<i32>(cube_lo.xy, cube_lo.z + z.x), vec3<i32>(hi.xy, cube_lo.z + z.y), t_prev);
                    let t1 = min(occupied.t1, t_end);
                    if t1 > occupied.t0 {
                        let entered = occupied.enter < 6u;
                        let axis0 = select((last >> 18u) & 3u, occupied.enter >> 1u, entered);
                        let positive0 = select(((last >> 20u) & 1u) == 0u, (occupied.enter & 1u) == 0u, entered);
                        let h = trace_exact(slot, ray, face, level, cube_lo, occupied.t0, t1, axis0, positive0, z, counted);
                        if h.hit {
                            return WalkEnd(true, h.t, slot, h.cell, h.axis, select(1u, 0u, h.positive), h.material);
                        }
                    }
                }
                let span = box_span(ray, face, level, cube_lo, hi, t_prev);
                if span.leave >= 6u { break; }
                let t_leave = max(span.t1, t_prev);
                t_prev = t_leave;
                if t_prev >= t_end { break; }
                // The cell just past the side it leaves by.
                let axis = span.leave >> 1u;
                let positive = (span.leave & 1u) == 1u;
                let q = exact_cell(ray, face, level, t_leave);
                cell = clamp(vec3<i32>(floor(q)), cube_lo, hi - 1);
                var c = vec3<i32>(vec3<u32>(last, last >> 6u, last >> 12u) & vec3<u32>(63u));
                if real { c = clamp(cell - cube_lo, vec3<i32>(0), vec3<i32>(63)); }
                last = u32(c.x) | (u32(c.y) << 6u) | (u32(c.z) << 12u) | (axis << 18u) | (select(1u, 0u, positive) << 20u);
                frac = clamp(fract(q), vec3<f32>(0.0), vec3<f32>(0.999));
                let side = select(cube_lo - 1, hi, vec3<bool>(positive));
                let entry = select(0.999, 0.0, positive);
                if axis == 0u {
                    cell.x = side.x;
                    frac.x = entry;
                } else if axis == 1u {
                    cell.y = side.y;
                    frac.y = entry;
                } else {
                    cell.z = side.z;
                    frac.z = entry;
                }
            }
        } else {
            slot = bitcast<u32>(frame.box.z);
            TRACE_PIXEL(chunk, slot, cube_lo, real)
            // The ray in the frame chunk's cells.
            let o = vec3<f32>(frame.i0.w, frame.i1.w, frame.i2.w) + vec3<f32>(dot(frame.i0.xyz, org), dot(frame.i1.xyz, org), dot(frame.i2.xyz, org));
            let d = vec3<f32>(dot(frame.i0.xyz, dir), dot(frame.i1.xyz, dir), dot(frame.i2.xyz, dir));
            let inv = 1.0 / select(d, vec3<f32>(1e-30), d == vec3<f32>(0.0));
            if real { cube_hi = clip_to_face(vec3<i32>(64), key); }
            let ca = (vec3<f32>(cube_lo) - o) * inv;
            let cb = (vec3<f32>(cube_hi) - o) * inv;
            let cube_far = max(ca, cb);
            let t_leave = min(cube_far.x, min(cube_far.y, cube_far.z));
            if real {
                // The walk over the chunk's occupied layers.
                let ta = (vec3<f32>(0.0, 0.0, frame.box.x) - o) * inv;
                let tb = (vec3<f32>(f32(cube_hi.x), f32(cube_hi.y), frame.box.y) - o) * inv;
                let lo = min(ta, tb);
                let hi = max(ta, tb);
                var t0 = max(max(lo.x, max(lo.y, lo.z)), t_prev);
                var t1 = min(min(hi.x, min(hi.y, hi.z)), t_end);
                // Only under the chunk's plane over its solid cells.
                let top = chunk_meta[slot].top;
                let s0 = o.z - top.x * o.x - top.y * o.y - top.z;
                let ds = d.z - top.x * d.x - top.y * d.y;
                if ds > 0.0 { t1 = min(t1, -s0 / ds); } else if ds < 0.0 { t0 = max(t0, -s0 / ds); } else if s0 > 0.0 { t1 = t0; }
                if t1 > t0 {
                    let axis0 = select(select(2u, 1u, lo.y >= lo.z), 0u, lo.x >= lo.y && lo.x >= lo.z);
                    let h = trace_chunk(slot, o, d, t0, t1, axis0, vec2<i32>(i32(frame.box.x), i32(frame.box.y)), counted);
                    if h.hit {
                        return WalkEnd(true, h.t, slot, h.cell, h.axis, select(1u, 0u, h.positive), h.material);
                    }
                }
            }
            // The cell just past the face the ray leaves the box by, in the
            // frame chunk's cells (exact integers far from the origin).
            t_prev = max(t_leave, t_prev);
            if t_prev >= t_end { break; }
            let q = o + d * t_leave;
            let axis = select(select(2u, 1u, cube_far.y <= cube_far.z), 0u, cube_far.x <= cube_far.y && cube_far.x <= cube_far.z);
            var local = clamp(vec3<i32>(floor(q)), cube_lo, cube_hi - 1);
            let c = clamp(local, vec3<i32>(0), vec3<i32>(63));
            last = u32(c.x) | (u32(c.y) << 6u) | (u32(c.z) << 12u) | (axis << 18u) | (select(1u, 0u, d[axis] > 0.0) << 20u);
            let step = select(cube_lo - 1, cube_hi, d > vec3<f32>(0.0));
            if axis == 0u { local.x = step.x; } else if axis == 1u { local.y = step.y; } else { local.z = step.z; }
            cell = key.yzw * 64 + local;
            // Where in the next cell the ray enters: the side it crossed on
            // the exit axis.
            frac = clamp(fract(q), vec3<f32>(0.0), vec3<f32>(0.999));
            let entry = select(0.999, 0.0, d[axis] > 0.0);
            if axis == 0u { frac.x = entry; } else if axis == 1u { frac.y = entry; } else { frac.z = entry; }
        }
        // Past a face's edge: the face across it (a plane's edge ends it).
        let n = i32(params.table.y >> u32(level));
        let crossed = any(cell.xy < vec2<i32>(0)) || any(cell.xy >= vec2<i32>(n));
        if crossed {
            if PLANE_WORLD { break; }
            let across = across_edge(face, cell, frac, n);
            if !across.ok { break; }
            face = across.face;
            cell = across.cell;
            frac = across.frac;
        }
        // Air crossed in a frame needs one on the face it crossed to.
        let next = place_at(face, level, cell, frac, r, crossed && !exact);
        if next.slot != NONE {
            slot = next.slot;
            key = chunk_keys[next.slot];
            real = true;
            exact = exact_level(key.x >> 8u);
            if exact {
                cube_lo = key.yzw * 64;
                cube_hi = cube_lo + vec3<i32>(64);
            } else {
                frame = chunk_frame(next.slot);
                cube_lo = vec3<i32>(0);
                cube_hi = vec3<i32>(64);
            }
        } else {
            if next.level == LOST {
                WCOUNT(10)
                LOST_PIXEL
                break;
            }
            if next.level == BELOW || next.level == ROCK {
                // Under a tile's chunks, which hold only its surface band:
                // solid where the ray leaves this box, until the tile's
                // whole volume (its caves) is generated. Rock between its
                // chunks is solid as it is.
                if next.level == BELOW {
                    let n = atomicAdd(&reach.count, 1u);
                    if n < arrayLength(&reach.keys) { reach.keys[n] = vec4<i32>(next.hi.x | (next.lo.z << 8u), next.lo.xy, 0); }
                }
                return WalkEnd(true, t_prev, slot, vec3<i32>(vec3<u32>(last, last >> 6u, last >> 12u) & vec3<u32>(63u)), (last >> 18u) & 3u, (last >> 20u) & 1u, 0u);
            }
            real = false;
            // Air of an exact level, or without a frame where the ray's is
            // not one (exact, or on another face): in the grid's own cells.
            if (exact_level(next.level) || (next.frame == NONE && (exact || crossed))) {
                exact = true;
                key = vec4<i32>(face | (next.level << 8u), 0, 0, 0);
                cube_lo = next.lo;
                cube_hi = next.hi;
            } else {
                // Crossed in this frame or the finer one given.
                exact = false;
                if next.frame != NONE {
                    frame = chunk_frame(next.frame);
                    key = chunk_keys[next.frame];
                }
                cube_lo = to_frame(next.lo, next.level - (key.x >> 8u), key, true);
                cube_hi = clip_to_face(to_frame(next.hi, next.level - (key.x >> 8u), key, false), key);
            }
        }
    }
    WCOUNT(8)
    WCOUNT_BOXES
    return WalkEnd(false, t_prev, NONE, vec3<i32>(0), 0u, 0u, 0u);
}

@compute @workgroup_size(8, 8)
fn trace(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(seed);
    if id.x >= size.x || id.y >= size.y { return; }
    let p = vec2<i32>(id.xy);
    let exit = textureLoad(out_exits, p).r;
    if exit < 0.0 { return; }
    if exit == 0.0 {
        textureStore(out_visibility, p, vec4<u32>(0xffffffffu, 0u, 0u, 0u));
        textureStore(out_distance, p, vec4<f32>(0.0));
        if plain_colour() { textureStore(out_colour, p, vec4<f32>(0.0)); }
    }
    var box = textureLoad(seed, p, 0).x;
    // No box left: the sky.
    if box == NONE {
        textureStore(out_exits, p, vec4<f32>(-1.0));
        return;
    }
    COUNT(0)
    let dir = pixel_ray(id.xy, size);
    var t_prev = max(exit, params.sun.w);
    // The chunk whose frame the ray is in (its last), and the box it is
    // crossing in that frame's cells: the chunk itself, or air.
    var frame = chunk_frame(box);
    var key = chunk_keys[bitcast<u32>(frame.box.z)];
    var cube_lo = vec3<i32>(0);
    var cube_hi = vec3<i32>(64);
    var real = true;
    let r = fade_dither(id.xy);
    // Both levels of a switch are drawn: a box of the one this pixel does
    // not show starts the walk where the ray enters it.
    let own = find_entry(key);
    if own != NONE && choice(chunk_table[own * 2u + 1u].y, r) != 1 && !exact_level(key.x >> 8u) {
        let o = vec3<f32>(frame.i0.w, frame.i1.w, frame.i2.w);
        let d = vec3<f32>(dot(frame.i0.xyz, dir), dot(frame.i1.xyz, dir), dot(frame.i2.xyz, dir));
        let inv = 1.0 / select(d, vec3<f32>(1e-30), d == vec3<f32>(0.0));
        let ta = (vec3<f32>(0.0, 0.0, frame.box.x) - o) * inv;
        let tb = (vec3<f32>(64.0, 64.0, frame.box.y) - o) * inv;
        let lo = min(ta, tb);
        t_prev = max(max(lo.x, max(lo.y, lo.z)), t_prev);
        let q = o + d * t_prev;
        let local = clamp(vec3<i32>(floor(q)), vec3<i32>(0, 0, i32(frame.box.x)), vec3<i32>(63, 63, i32(frame.box.y) - 1));
        let next = place_at(key.x & 7, key.x >> 8u, key.yzw * 64 + local, clamp(fract(q), vec3<f32>(0.0), vec3<f32>(0.999)), r, false);
        if next.slot != NONE {
            frame = chunk_frame(next.slot);
            key = chunk_keys[next.slot];
        } else if next.level == LOST || next.level == BELOW || next.level == ROCK {
            textureStore(out_exits, p, vec4<f32>(t_prev + 1e-4));
            return;
        } else {
            if next.frame != NONE {
                frame = chunk_frame(next.frame);
                key = chunk_keys[next.frame];
            }
            cube_lo = to_frame(next.lo, next.level - (key.x >> 8u), key, true);
            cube_hi = clip_to_face(to_frame(next.hi, next.level - (key.x >> 8u), key, false), key);
            real = false;
        }
    }
    let end = walk(Walk(bitcast<u32>(frame.box.z), cube_lo, cube_hi, real), vec3<f32>(0.0), dir, t_prev, 1e30, r, 128u, true, id.xy);
    if !end.hit {
        // Unresolved: the next round seeds the boxes beyond (none past
        // the ceiling: the sky).
        textureStore(out_exits, p, vec4<f32>(select(end.t + 1e-4, -1.0, end.t < 0.0)));
        return;
    }
    COUNT(2)
    textureStore(out_exits, p, vec4<f32>(-1.0));
    let material = shown_material(end.material, end.slot, end.cell, end.axis == 2u && end.sign == 1u);
    // The material uses five bits (the voxel palette has 32 entries); six
    // spare high bits retain a quantized primary-walk cost for publish().
    let packed_cost = min(heat / 3u, 63u);
    let packed_cell = u32(end.cell.x) | (u32(end.cell.y) << 6u) | (u32(end.cell.z) << 12u)
        | (end.axis << 18u) | (end.sign << 20u) | (min(material, 31u) << 21u)
        | (packed_cost << 26u);
    textureStore(out_visibility, p, vec4<u32>(end.slot, packed_cell, 0u, 0u));
    textureStore(out_distance, p, vec4<f32>(end.t, 0.0, 0.0, 0.0));
    if !plain_colour() { return; }
    let c = cell_frame(end.slot, end.cell);
    let axis_world = select(select(c.m2.xyz, c.m1.xyz, end.axis == 1u), c.m0.xyz, end.axis == 0u);
    let normal = normalize(axis_world) * select(-1.0, 1.0, end.sign == 1u);
    let light = 0.25 + 0.75 * max(dot(normal, params.sun.xyz), 0.0);
    var colour = palette(material).rgb;
    if debug_mode() == 2u || debug_mode() == 32u {
        // Steps: red to 64, yellow to 128, white to 192.
        let h = f32(heat) / 64.0;
        colour = clamp(vec3<f32>(h, h - 1.0, h - 2.0), vec3<f32>(0.0), vec3<f32>(1.0)) / light;
    } else if debug_mode() == 1u {
        colour = level_colour((bitcast<u32>(c.box.w) >> 8u) & 31u);
    }
    textureStore(out_colour, p, vec4<f32>(colour * light, 1.0));
}

// ---- 3. Publish ----

@group(1) @binding(4) var hit_distance: texture_2d<f32>;

@vertex
fn resolve_vertex(@builtin(vertex_index) v: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((v << 1u) & 2u), f32(v & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

@group(1) @binding(7) var hit_visibility: texture_2d<u32>;
@group(1) @binding(14) var hit_normal: texture_2d<f32>;

// ---- Appearance ----

// Visibility of the voxel mosaic (each cell's own hue and brightness) for a
// cell `projected` pixels wide: kept down to about half a pixel; TSR
// averages what is smaller.
fn mosaic_weight(projected: f32) -> f32 {
    return smoothstep(0.35, 0.7, projected);
}

// How far a cell of chunk key `key` spanning `projected` pixels is coarser
// than its cut wants (0 to 1): the cut keeps every level but the finest at
// a pixel a cell or less, so a larger one is a tile still refining (an eye
// dropped from orbit, a slow frame's generation). It reads as the ground it
// stands for, not as the box it is: lit by the slope, without its own
// pigment, edges or corner shading. Finest cells keep their voxel look at
// any size.
fn under_refined(key: vec4<i32>, projected: f32) -> f32 {
    if (key.x >> 8u) == 0 { return 0.0; }
    return smoothstep(1.5, 4.0, projected);
}

// A pixel's width (m) at distance `t` along pixel `q`'s ray.
fn pixel_width(q: vec2<u32>, size: vec2<u32>, t: f32) -> f32 {
    return t * length(pixel_ray(q + vec2<u32>(1u, 0u), size) - pixel_ray(q, size));
}

// Material `m`'s colour varied by its patches (409 m and 25.6 m), clumps
// (6.4 m and 1.6 m, fading below a few pixels) and voxels (`blade`, by
// `voxel`), at domain point `p` for a pixel `pixel` m wide.
fn material_albedo(m: u32, p: vec3<i32>, blade_hash: u32, pixel: f32, voxel: f32) -> vec3<f32> {
    let material = materials.m[min(m, 31u)];
    if material.patches[0].w <= 0.0 { return material.colour.rgb; }
    let broad = f32(noise(p, 15u, 0x3c6ef372u)) / f32(NOISE_ONE);
    let patches = f32(noise(p, 11u, 0xa54ff53au)) / f32(NOISE_ONE) * clamp((12.8 - pixel) / 6.4, 0.0, 1.0);
    var clumps = 0.0;
    if pixel < 3.2 {
        clumps += 0.35 * f32(noise(p, 9u, 0x510e527fu)) / f32(NOISE_ONE) * clamp((3.2 - pixel) / 1.6, 0.0, 1.0);
    }
    if pixel < 0.8 {
        clumps += 0.3 * f32(noise(p, 7u, 0x9b05688cu)) / f32(NOISE_ONE) * clamp((0.8 - pixel) / 0.4, 0.0, 1.0);
    }
    let blade = f32(blade_hash & 255u) / 255.0 - 0.5;
    let t = clamp(0.52 + materials.detail.x * (0.7 * broad + 0.16 * patches + clumps) + materials.detail.y * blade * voxel, 0.0, 1.0);
    return select(mix(material.patches[1].rgb, material.patches[2].rgb, t * 2.0 - 1.0), mix(material.patches[0].rgb, material.patches[1].rgb, t * 2.0), t < 0.5);
}

fn corner_ao(side1: bool, side2: bool, corner: bool) -> f32 {
    if side1 && side2 { return 0.0; }
    return 3.0 - f32(u32(side1) + u32(side2) + u32(corner));
}

// Whether cell `c` (in chunk `slot`'s cells, possibly past them: then in
// the drawn chunk of its level there) is solid.
fn solid_near(slot: u32, key: vec4<i32>, c: vec3<i32>) -> bool {
    if all(c >= vec3<i32>(0)) && all(c <= vec3<i32>(63)) { return solid_at(slot, c); }
    let at = key.yzw * 64 + c;
    let e = find_entry(vec4<i32>(key.x, at >> vec3<u32>(6u)));
    if e == NONE { return false; }
    return solid_at(bitcast<u32>(chunk_table[e * 2u + 1u].x), at & vec3<i32>(63));
}

struct Look {
    albedo: vec3<f32>,
    ao: f32,
}

// The colour and occlusion of hit cell `cell` of chunk `slot` (its face:
// `axis`, `sign` 1 the positive side) of `material`, at eye-relative
// `position`, for a pixel `pixel_m` wide: the material's patches and the
// cell's pigment, corner occlusion from the cells around the air in front
// of the face, and its edges darkened while it spans several pixels. Each
// fades as the cell shrinks towards a pixel (TSR averages the rest).
fn surface_look(slot: u32, cell: vec3<i32>, axis: u32, sign: u32, hit_material: u32, position: vec3<f32>, pixel_m: f32, pixel: vec2<u32>) -> Look {
    let c = cell_frame(slot, cell);
    let key = chunk_keys[slot];
    let projected = length(c.m0.xyz) / max(pixel_m, 1e-6);
    // Material boundaries filtered where cells are a pixel or less: a
    // leaf's one surface material (its 4x4 columns' vote) would show as
    // blocks of four pixels. A top face takes the surface material at a
    // point jittered up to a leaf's width each frame, of the stored leaf
    // there (TSR averages the frames into a filtered boundary).
    var material = hit_material;
    let filtered = 1.0 - smoothstep(1.0, 3.0, projected);
    if filtered > 0.0 && axis == 2u && sign == 1u {
        let r = vec2<f32>(f32(fade_dither(pixel)), f32(fade_dither(pixel + vec2<u32>(37u, 11u)))) / 1024.0;
        let local = vec3<f32>(c.i0.w, c.i1.w, c.i2.w) + vec3<f32>(dot(c.i0.xyz, position), dot(c.i1.xyz, position), dot(c.i2.xyz, position));
        let at = vec2<i32>(floor(local.xy + (r - 0.5) * 4.0 * filtered));
        // A top cell there (solid under air): its leaf holds a surface.
        let there = vec3<i32>(at, cell.z);
        if any((at >> vec2<u32>(2u)) != (cell.xy >> vec2<u32>(2u))) && solid_at(slot, there) && !solid_at(slot, there + vec3<i32>(0, 0, 1)) {
            let pair = stored_leaf_pair(slot, there);
            if pair != NONE && (pair & 0xffu) != 0u { material = min(pair & 0xffu, 31u); }
        }
    }
    let smooth_ground = 1.0 - under_refined(key, projected);
    let mosaic = mosaic_weight(projected) * smooth_ground;
    // The cell centre's noise-domain point: the same pigment from every
    // face of it.
    let centre = vec3<f32>(c.m0.w, c.m1.w, c.m2.w) + mat3x3<f32>(c.m0.xyz, c.m1.xyz, c.m2.xyz) * (vec3<f32>(cell) + 0.5);
    let p = params.eye_domain.xyz + vec3<i32>(floor(params.eye_domain_frac.xyz + centre / DOMAIN_UNIT));
    let hv = hash3(p.x, p.y, p.z, 0x68bc21ebu);
    let pigment = 1.0 + materials.detail.y * (mix(0.5, f32(hv & 255u) / 255.0, mosaic) - 0.5);
    let albedo = material_albedo(material, p, hash3(p.x, p.y, p.z, 0x1b873593u), pixel_m, mosaic) * pigment;
    // Occlusion: around the air cell in front of the face, where voxels
    // (not cells: the same at every level, as `hit_face`'s slope) are a
    // few pixels wide; under that the sky samples hold the occlusion and
    // the eight lookups would only add sub-pixel noise.
    let voxel_px = params.planet.y / max(pixel_m, 1e-6);
    let ao_weight = smoothstep(1.0, 3.0, voxel_px) * smoothstep(0.75, 1.25, projected) * smooth_ground;
    var ao = 1.0;
    if ao_weight > 0.0 {
        let out_dir = select(-1, 1, sign == 1u);
        let along = vec3<i32>(select(0, out_dir, axis == 0u), select(0, out_dir, axis == 1u), select(0, out_dir, axis == 2u));
        let du = vec3<i32>(select(1, 0, axis == 0u), select(0, 1, axis == 0u), 0);
        let dv = vec3<i32>(0, select(0, 1, axis == 2u), select(1, 0, axis == 2u));
        let f = cell + along;
        let s0 = solid_near(slot, key, f - du);
        let s1 = solid_near(slot, key, f + du);
        let s2 = solid_near(slot, key, f - dv);
        let s3 = solid_near(slot, key, f + dv);
        let c00 = corner_ao(s0, s2, solid_near(slot, key, f - du - dv));
        let c10 = corner_ao(s1, s2, solid_near(slot, key, f + du - dv));
        let c01 = corner_ao(s0, s3, solid_near(slot, key, f - du + dv));
        let c11 = corner_ao(s1, s3, solid_near(slot, key, f + du + dv));
        // Where in the face the hit is.
        let local = vec3<f32>(c.i0.w, c.i1.w, c.i2.w) + vec3<f32>(dot(c.i0.xyz, position), dot(c.i1.xyz, position), dot(c.i2.xyz, position));
        let in_cell = clamp(local - vec3<f32>(cell), vec3<f32>(0.0), vec3<f32>(1.0));
        let uv = vec2<f32>(dot(in_cell, vec3<f32>(du)), dot(in_cell, vec3<f32>(dv)));
        let a = mix(mix(c00, c10, uv.x), mix(c01, c11, uv.x), uv.y) / 3.0;
        ao = mix(1.0, mix(0.42, 1.0, a), ao_weight);
        // Crisp voxel edges while a cell spans several pixels.
        let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
        let fade = clamp((projected - 3.0) / 6.0, 0.0, 1.0);
        ao *= 1.0 - materials.detail.z * fade * ao_weight * (1.0 - smoothstep(0.0, 0.12, edge));
    }
    return Look(albedo, ao);
}

// The debug mode (`params.debug.x` & 0xff): 1 levels, 2 the steps' heat,
// 3 levels on the left half, 32 the editor's voxel trace-cost view.
fn debug_mode() -> u32 {
    return params.debug.x & 0xffu;
}

// Whether the walks write the plain colour target (bit 8 of
// `params.debug.x`: a world drawn alone; a host's gbuffer takes the
// published surface instead).
fn plain_colour() -> bool {
    return (params.debug.x & 0x100u) != 0u;
}

// A level's colour (`HELIO_VOXEL_CHUNKS_LEVELS`, debug modes 1 and 3).
fn level_colour(level: u32) -> vec3<f32> {
    let colours = array<vec3<f32>, 8>(vec3<f32>(0.9, 0.2, 0.2), vec3<f32>(0.9, 0.6, 0.1), vec3<f32>(0.8, 0.9, 0.1), vec3<f32>(0.2, 0.8, 0.2),
        vec3<f32>(0.1, 0.8, 0.8), vec3<f32>(0.2, 0.3, 0.9), vec3<f32>(0.6, 0.2, 0.9), vec3<f32>(0.9, 0.2, 0.7));
    return colours[level & 7u];
}

// Helio's gbuffer (`engine::GBUFFER_FORMATS`).
struct GOut {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) orm: vec4<f32>,
    @location(3) emissive: vec4<f32>,
    @location(4) lightmap_uv: vec2<f32>,
    @location(5) sss: vec4<f32>,
    @location(6) extra: vec4<f32>,
    @location(7) velocity: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn publish(@builtin(position) pixel: vec4<f32>) -> GOut {
    let p = vec2<i32>(pixel.xy);
    let v = textureLoad(hit_visibility, p, 0).xy;
    if v.x == NONE { discard; }
    let t = textureLoad(hit_distance, p, 0).r;
    let size = textureDimensions(hit_distance);
    let position = pixel_ray(vec2<u32>(pixel.xy), size) * t;
    let clip = params.view_proj * vec4<f32>(position, 1.0);
    let c = chunk_frame(v.x);
    let axis = (v.y >> 18u) & 3u;
    let sign = (v.y >> 20u) & 1u;
    let material = (v.y >> 21u) & 31u;
    let shading = textureLoad(hit_normal, p, 0);
    let normal = shading.xyz;
    let cell = vec3<i32>(vec3<u32>(v.y, v.y >> 6u, v.y >> 12u) & vec3<u32>(63u));
    // A pixel's width at the hit (m).
    let pixel_m = pixel_width(vec2<u32>(pixel.xy), size, t);
    let look = surface_look(v.x, cell, axis, sign, material, position, pixel_m, vec2<u32>(pixel.xy));
    let f0 = 0.03;
    var out: GOut;
    out.albedo = vec4<f32>(look.albedo, 1.0);
    if debug_mode() == 32u {
        let h = f32((v.y >> 26u) & 63u) * 3.0 / 64.0;
        let cost = clamp(vec3<f32>(h, h - 1.0, h - 2.0), vec3<f32>(0.0), vec3<f32>(1.0));
        out.albedo = vec4<f32>(cost, 1.0);
    }
    // Level colours (`HELIO_VOXEL_CHUNKS_LEVELS`; `=half`: the left half,
    // beside the shaded right).
    if debug_mode() == 1u || (debug_mode() == 3u && p.x < i32(size.x / 2u)) { out.albedo = vec4<f32>(level_colour(u32(chunk_keys[v.x].x >> 8u) & 31u), 1.0); }
    out.normal = vec4<f32>(normal, f0);
    out.orm = vec4<f32>(look.ao * shading.w, clamp(palette(material).w, 0.04, 1.0), 0.0, f0);
    out.emissive = vec4<f32>(0.0, 0.0, 0.0, f0);
    // Terrain tag consumed by deferred lighting (skips screen-space AO).
    out.lightmap_uv = vec2<f32>(-1.0, -2.0);
    out.sss = vec4<f32>(0.0);
    out.extra = vec4<f32>(0.0);
    let previous = params.prev_view_proj * vec4<f32>(position, 1.0);
    var velocity = vec2<f32>(0.0);
    if previous.w > 0.0 {
        let ndc = previous.xy / previous.w;
        velocity = pixel.xy - (vec2<f32>(ndc.x, -ndc.y) * 0.5 + 0.5) * vec2<f32>(size);
    }
    out.velocity = vec4<f32>(velocity, 0.0, 0.0);
    out.depth = clip.z / clip.w;
    return out;
}

// ---- 4. Sun shadows ----
// A walk towards the sun from a quarter of the hits (one a 2x2 block, the
// pixel moving each frame: TSR gathers them all), a hundredth of a cell
// off their face; every hit then takes the sample of its block, or of a
// neighbouring one, that walked within four of its cells; the others
// (silhouettes) are listed and walk in a pass of their own, so a few of
// them do not hold whole warps. The samples also measure the slope's
// normal (smooth: it serves the block).

// Helio's directional visibility: x lit (0 or 1), yzw the sun it is for.
@group(1) @binding(8) var out_sun: texture_storage_2d<rgba16float, write>;
// The normal each hit is lit with (`Face::shading`) and its sky visibility,
// for the gbuffer.
@group(1) @binding(13) var out_normal: texture_storage_2d<rgba16float, write>;
// Samples: lit (bit 0), walked (it faced the sun, bit 1), a hit (bit 2);
// its distance's bits; its slope's normal (`pack_unit`, 0: none).
@group(1) @binding(9) var out_sun_samples: texture_storage_2d<rgba32uint, write>;
@group(1) @binding(10) var sun_samples: texture_2d<u32>;
// Hits no sample serves: their count and pixels (x | y << 16), and the
// workgroups of their indirect dispatch (x, 1, 1).
struct Fallbacks {
    count: atomic<u32>,
    pixels: array<u32>,
}
@group(1) @binding(11) var<storage, read_write> fallbacks: Fallbacks;
@group(1) @binding(12) var<storage, read_write> fallback_groups: array<atomic<u32>, 3>;

// A hit's face: its unit normal, the length of a cell across it; the
// normal it is lit with, and how far (cells) its sun ray ignores what it
// meets (natural steps: `shading`); the slope's normal there (zero: none).
struct Face {
    normal: vec3<f32>,
    cell: f32,
    shading: vec3<f32>,
    skip: f32,
    ground: vec3<f32>,
}

// The top of the column `d` cells from `cell` (chunk `slot`'s cells) within
// four layers of `cell`'s, and whether there is one.
fn column_top_near(slot: u32, key: vec4<i32>, cell: vec3<i32>, d: vec2<i32>) -> vec2<f32> {
    var c = cell + vec3<i32>(d, 0);
    if solid_near(slot, key, c + vec3<i32>(0, 0, 1)) {
        for (var n = 0; n < 4; n++) {
            c.z += 1;
            if !solid_near(slot, key, c + vec3<i32>(0, 0, 1)) { return vec2<f32>(f32(c.z), 1.0); }
        }
        return vec2<f32>(0.0);
    }
    for (var n = 0; n < 5; n++) {
        if solid_near(slot, key, c) { return vec2<f32>(f32(c.z), 1.0); }
        c.z -= 1;
    }
    return vec2<f32>(0.0);
}

// The ground's normal at a natural step `cell` (chunk `slot`, frame `c`):
// the slope of the column tops two cells away each way, in the world;
// zero where they are not found.
fn slope_normal(slot: u32, key: vec4<i32>, cell: vec3<i32>, c: ChunkDraw) -> vec3<f32> {
    let x0 = column_top_near(slot, key, cell, vec2<i32>(-2, 0));
    let x1 = column_top_near(slot, key, cell, vec2<i32>(2, 0));
    let y0 = column_top_near(slot, key, cell, vec2<i32>(0, -2));
    let y1 = column_top_near(slot, key, cell, vec2<i32>(0, 2));
    let top = f32(cell.z);
    // One-sided where a side has no top near.
    let gx = select(select((x1.x - top) / 2.0, (top - x0.x) / 2.0, x0.y > 0.0), (x1.x - x0.x) / 4.0, x0.y > 0.0 && x1.y > 0.0);
    let gy = select(select((y1.x - top) / 2.0, (top - y0.x) / 2.0, y0.y > 0.0), (y1.x - y0.x) / 4.0, y0.y > 0.0 && y1.y > 0.0);
    if x0.y + x1.y == 0.0 || y0.y + y1.y == 0.0 { return vec3<f32>(0.0); }
    // A cell-space normal to the world: by M's inverse transposed.
    let n = -gx * c.i0.xyz - gy * c.i1.xyz + c.i2.xyz;
    // Zero (not NaN) where it cannot be normalised.
    let l = dot(n, n);
    if !(l > 1e-12) { return vec3<f32>(0.0); }
    return n * inverseSqrt(l);
}

// The ground's normal at pixel `q` (hit at distance `t`) from the hits two
// pixels each way: on each axis the side nearer in distance (not across a
// silhouette), facing the eye; zero where an axis has no hit near. Over
// cells of a pixel or less it spans more than a cell: the slope, not the
// steps.
fn screen_ground(q: vec2<u32>, size: vec2<u32>, t: f32) -> vec3<f32> {
    let p = pixel_ray(q, size) * t;
    var d = array<vec3<f32>, 2>(vec3<f32>(0.0), vec3<f32>(0.0));
    for (var axis = 0u; axis < 2u; axis++) {
        var best = 1e30;
        for (var side = -1; side <= 1; side += 2) {
            var at = vec2<i32>(q);
            at[axis] += 2 * side;
            if any(at < vec2<i32>(0)) || any(at >= vec2<i32>(size)) { continue; }
            let tn = textureLoad(hit_distance, at, 0).r;
            if tn <= 0.0 || abs(tn - t) >= best { continue; }
            best = abs(tn - t);
            d[axis] = (pixel_ray(vec2<u32>(at), size) * tn - p) * f32(side);
        }
        if best > 0.25 * t { return vec3<f32>(0.0); }
    }
    let n = cross(d[0], d[1]);
    let l = dot(n, n);
    if !(l > 1e-20) { return vec3<f32>(0.0); }
    return n * inverseSqrt(l) * select(1.0, -1.0, dot(n, p) > 0.0);
}

// Hit `v` (pixels `pixel_m` wide there). Voxels a few pixels wide are lit
// by their own faces (a slope reads as its steps, each shadowing the next);
// a natural step (air above it) is lit partly with the slope's normal,
// `Materials::detail.w` (step softness), its sun ray passing its own steps.
// Voxels of a pixel or less are lit by the slope's normal, at every level:
// the weight follows the voxel's size on screen, not the cell's, so it is
// the same on both sides of a switch between levels, and a coarse level's
// steps (each a cell tall, their shadows long under a low sun) never show
// as stripes. A cell larger than its cut wants (`under_refined`) is lit by
// the slope's normal too. The slope's normal is `ground` when given (a
// nearby sample's), else measured here.
fn hit_face(v: vec2<u32>, ground_hint: vec3<f32>, q: vec2<u32>, size: vec2<u32>, t: f32) -> Face {
    let pixel_m = pixel_width(q, size, t);
    let cell = vec3<i32>(vec3<u32>(v.y, v.y >> 6u, v.y >> 12u) & vec3<u32>(63u));
    let c = cell_frame(v.x, cell);
    let key = chunk_keys[v.x];
    let axis = (v.y >> 18u) & 3u;
    let sign = (v.y >> 20u) & 1u;
    let axis_world = select(select(c.m2.xyz, c.m1.xyz, axis == 1u), c.m0.xyz, axis == 0u);
    let normal = normalize(axis_world) * select(-1.0, 1.0, sign == 1u);
    let cell_m = length(axis_world);
    var f = Face(normal, cell_m, normal, 0.0, vec3<f32>(0.0));
    let small = 1.0 - smoothstep(1.0, 3.0, params.planet.y / max(pixel_m, 1e-6));
    let coarse = max(under_refined(key, length(c.m0.xyz) / max(pixel_m, 1e-6)), small);
    let step = !solid_near(v.x, key, cell + vec3<i32>(0, 0, 1));
    if !step && coarse == 0.0 { return f; }
    let w = max(select(0.0, materials.detail.w, step), coarse);
    // The heightfield's exact slope stored with the leaf; else, cells of a
    // pixel or less: the slope the hits around make (a few loads); larger
    // ones measure their column tops (`slope_normal`).
    var ground = ground_hint;
    if dot(ground, ground) == 0.0 { ground = leaf_ground(v.x, cell, c); }
    if dot(ground, ground) == 0.0 && length(c.m0.xyz) <= 1.5 * pixel_m { ground = screen_ground(q, size, t); }
    if dot(ground, ground) == 0.0 { ground = slope_normal(v.x, key, cell, c); }
    // A unit normal (written so NaN fails it), else the face's own.
    if w <= 0.0 || !(abs(dot(ground, ground) - 1.0) < 0.01) { return f; }
    let shading = mix(normal, ground, w);
    let l = dot(shading, shading);
    if !(l > 1e-6) { return f; }
    f.ground = ground;
    f.shading = shading * inverseSqrt(l);
    f.skip = 2.0 * w;
    return f;
}

// A unit vector in two snorm halves (octahedral) and back.
fn pack_unit(n: vec3<f32>) -> u32 {
    var p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    if n.z < 0.0 { p = (1.0 - abs(p.yx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), p >= vec2<f32>(0.0)); }
    return pack2x16snorm(p);
}

fn unpack_unit(u: u32) -> vec3<f32> {
    let p = unpack2x16snorm(u);
    var n = vec3<f32>(p, 1.0 - abs(p.x) - abs(p.y));
    if n.z < 0.0 { n = vec3<f32>((1.0 - abs(n.yx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), n.xy >= vec2<f32>(0.0)), n.z); }
    return normalize(n);
}


// Sky rays a sample casts a frame (one sample a 4x4 block), and how far
// they reach (cells).
const SKY_RAYS: u32 = 4u;
const SKY_RANGE_CELLS: f32 = 48.0;

// `walk` crossing every box in a frame, as at levels finer than the exact
// ones: the walk of the kernels for hits there (sun and sky samples),
// whose rays an exact box's code would cost registers. Coarse air on
// their way strays from the grid by up to 512 x cell / radius of its cells.
fn walk_fine(start: Walk, org: vec3<f32>, dir: vec3<f32>, t_start: f32, t_end: f32, r: i32, boxes: u32, counted: bool, pixel: vec2<u32>) -> WalkEnd {
    // Over the ceiling and rising, a ray meets nothing more (the squared
    // radius q0 + 2 b t + t^2 only grows once b + t > 0; a plane's height
    // q0 + b t once b > 0): it ends there, t -1.
    var q0 = params.ceiling.x + 2.0 * params.up.w * dot(params.up.xyz, org) + dot(org, org);
    var b = params.up.w * dot(params.up.xyz, dir) + dot(org, dir);
    if PLANE_WORLD {
        q0 = params.ceiling.x + org.y;
        b = dir.y;
    }
    var frame = chunk_frame(start.slot);
    var key = chunk_keys[start.slot];
    var cube_lo = start.lo;
    var cube_hi = start.hi;
    var real = start.real;
    var t_prev = t_start;
    var chunk = 0u;
    for (; chunk < boxes; chunk++) {
        let over = select(b + t_prev > 0.0 && q0 + t_prev * (2.0 * b + t_prev) > 0.0, b > 0.0 && q0 + b * t_prev > 0.0, PLANE_WORLD);
        if over { return WalkEnd(false, -1.0, NONE, vec3<i32>(0), 0u, 0u, 0u); }
        WCOUNT(3)
        let slot = bitcast<u32>(frame.box.z);
        TRACE_PIXEL(chunk, slot, cube_lo, real)
        // The ray in the frame chunk's cells.
        let o = vec3<f32>(frame.i0.w, frame.i1.w, frame.i2.w) + vec3<f32>(dot(frame.i0.xyz, org), dot(frame.i1.xyz, org), dot(frame.i2.xyz, org));
        let d = vec3<f32>(dot(frame.i0.xyz, dir), dot(frame.i1.xyz, dir), dot(frame.i2.xyz, dir));
        let inv = 1.0 / select(d, vec3<f32>(1e-30), d == vec3<f32>(0.0));
        if real { cube_hi = clip_to_face(vec3<i32>(64), key); }
        let ca = (vec3<f32>(cube_lo) - o) * inv;
        let cb = (vec3<f32>(cube_hi) - o) * inv;
        let cube_far = max(ca, cb);
        let t_leave = min(cube_far.x, min(cube_far.y, cube_far.z));
        if real {
            // The walk over the chunk's occupied layers.
            let ta = (vec3<f32>(0.0, 0.0, frame.box.x) - o) * inv;
            let tb = (vec3<f32>(f32(cube_hi.x), f32(cube_hi.y), frame.box.y) - o) * inv;
            let lo = min(ta, tb);
            let hi = max(ta, tb);
            var t0 = max(max(lo.x, max(lo.y, lo.z)), t_prev);
            var t1 = min(min(hi.x, min(hi.y, hi.z)), t_end);
            // Only under the chunk's plane over its solid cells.
            let top = chunk_meta[slot].top;
            let s0 = o.z - top.x * o.x - top.y * o.y - top.z;
            let ds = d.z - top.x * d.x - top.y * d.y;
            if ds > 0.0 { t1 = min(t1, -s0 / ds); } else if ds < 0.0 { t0 = max(t0, -s0 / ds); } else if s0 > 0.0 { t1 = t0; }
            if t1 > t0 {
                let axis0 = select(select(2u, 1u, lo.y >= lo.z), 0u, lo.x >= lo.y && lo.x >= lo.z);
                let h = trace_chunk(slot, o, d, t0, t1, axis0, vec2<i32>(i32(frame.box.x), i32(frame.box.y)), counted);
                if h.hit {
                    return WalkEnd(true, h.t, slot, h.cell, h.axis, select(1u, 0u, h.positive), h.material);
                }
            }
        }
        // The cell just past the face the ray leaves the box by, in the
        // frame chunk's cells (exact integers far from the origin).
        t_prev = max(t_leave, t_prev);
        if t_prev >= t_end { break; }
        let level = key.x >> 8u;
        let q = o + d * t_leave;
        let axis = select(select(2u, 1u, cube_far.y <= cube_far.z), 0u, cube_far.x <= cube_far.y && cube_far.x <= cube_far.z);
        var local = clamp(vec3<i32>(floor(q)), cube_lo, cube_hi - 1);
        let step = select(cube_lo - 1, cube_hi, d > vec3<f32>(0.0));
        if axis == 0u { local.x = step.x; } else if axis == 1u { local.y = step.y; } else { local.z = step.z; }
        var cell = key.yzw * 64 + local;
        // Where in the next cell the ray enters: the side it crossed on
        // the exit axis.
        var frac = clamp(fract(q), vec3<f32>(0.0), vec3<f32>(0.999));
        let entry = select(0.999, 0.0, d[axis] > 0.0);
        if axis == 0u { frac.x = entry; } else if axis == 1u { frac.y = entry; } else { frac.z = entry; }
        // Past a face's edge: the face across it (a plane's edge ends it).
        var face = key.x & 7;
        let n = i32(params.table.y >> u32(level));
        let crossed = any(cell.xy < vec2<i32>(0)) || any(cell.xy >= vec2<i32>(n));
        if crossed {
            if PLANE_WORLD { break; }
            let across = across_edge(face, cell, frac, n);
            if !across.ok { break; }
            face = across.face;
            cell = across.cell;
            frac = across.frac;
        }
        let next = place_at(face, level, cell, frac, r, crossed);
        if next.slot != NONE {
            frame = chunk_frame(next.slot);
            key = chunk_keys[next.slot];
            cube_lo = vec3<i32>(0);
            cube_hi = vec3<i32>(64);
            real = true;
        } else {
            if next.level == LOST {
                WCOUNT(10)
                LOST_PIXEL
                break;
            }
            if next.level == BELOW || next.level == ROCK {
                // Under a tile's chunks, which hold only its surface band:
                // solid where the ray leaves this box, until the tile's
                // whole volume (its caves) is generated. Rock between its
                // chunks is solid as it is.
                if next.level == BELOW {
                    let n = atomicAdd(&reach.count, 1u);
                    if n < arrayLength(&reach.keys) { reach.keys[n] = vec4<i32>(next.hi.x | (next.lo.z << 8u), next.lo.xy, 0); }
                }
                return WalkEnd(true, t_leave, slot, clamp(local, vec3<i32>(0), vec3<i32>(63)), axis, select(1u, 0u, d[axis] > 0.0), 0u);
            }
            // Air, crossed in this frame or the finer one given.
            if next.frame != NONE {
                frame = chunk_frame(next.frame);
                key = chunk_keys[next.frame];
            }
            cube_lo = to_frame(next.lo, next.level - (key.x >> 8u), key, true);
            cube_hi = clip_to_face(to_frame(next.hi, next.level - (key.x >> 8u), key, false), key);
            real = false;
        }
    }
    WCOUNT(8)
    WCOUNT_BOXES
    return WalkEnd(false, t_prev, NONE, vec3<i32>(0), 0u, 0u, 0u);
}


// Where a ray from hit face `f` along `dir` starts: past the cell steps
// a slope-lit face stands among (`Face::skip`), until it has risen a cell
// off the slope (12 cells at most: a grazing ray).
fn skip_along(f: Face, dir: vec3<f32>) -> f32 {
    if f.skip == 0.0 { return 0.0; }
    let rise = dot(dir, f.ground);
    return f.skip * f.cell * clamp(0.75 / max(rise, 1e-3), 1.0, 12.0);
}

// WALK {
// Whether the hit at pixel `q` (distance `t`, face `f`) is in the sun.
fn sun_walk(q: vec2<u32>, size: vec2<u32>, slot: u32, t: f32, f: Face) -> f32 {
    let org = pixel_ray(q, size) * t + f.normal * (f.cell * 0.01);
    let end = walk(Walk(slot, vec3<i32>(0), vec3<i32>(64), true), org, params.sun.xyz, skip_along(f, params.sun.xyz), 1e30, fade_dither(q), 96u, false, q);
    return select(1.0, 0.0, end.hit);
}

// The share of the sky the hit at pixel `q` (distance `t`, face `f`, chunk
// `slot`) sees: cosine-weighted rays about the planet's up, turned each
// frame (TSR gathers them); an occluder near the range end shades less (no
// edge at the range). A ceiling looks out sideways.
fn sky_visibility(q: vec2<u32>, size: vec2<u32>, slot: u32, t: f32, f: Face) -> f32 {
    let at = pixel_ray(q, size) * t;
    let up = select(normalize(params.up.xyz * params.up.w + at), vec3<f32>(0.0, 1.0, 0.0), PLANE_WORLD);
    let ceiling = dot(f.normal, up) < -0.25;
    let t1 = normalize(select(cross(up, vec3<f32>(0.0, 0.0, 1.0)), cross(up, vec3<f32>(1.0, 0.0, 0.0)), abs(up.z) > 0.9));
    let t2 = cross(up, t1);
    let range = SKY_RANGE_CELLS * f.cell;
    let org = pixel_ray(q, size) * t + f.normal * (f.cell * 0.01);
    let turn = f32(params.table.w % 64u) * 0.618034;
    var open = 0.0;
    var total = 0.0;
    for (var i = 0u; i < SKY_RAYS; i++) {
        let u = fract((f32(i) + 0.5) / f32(SKY_RAYS) + turn);
        let phi = (f32(i) + turn * 7.0) * 2.39996323;
        let around = t1 * cos(phi) + t2 * sin(phi);
        var dir = normalize(around * sqrt(u) + up * sqrt(1.0 - u));
        if ceiling { dir = normalize(around - up * 0.02); }
        else if dot(dir, f.normal) <= 0.0 { continue; }
        total += 1.0;
        let end = walk(Walk(slot, vec3<i32>(0), vec3<i32>(64), true), org, dir, skip_along(f, dir), range, fade_dither(q), 24u, false, q);
        open += select(1.0, smoothstep(0.5, 1.0, end.t / range), end.hit && end.t < range);
    }
    return select(1.0, open / total, total > 0.0);
}
// } WALK

// Sky visibility samples, one a 4x4 block: visibility, the sample's
// distance (0: no hit).
@group(1) @binding(15) var out_sky_samples: texture_storage_2d<rg32float, write>;
@group(1) @binding(16) var sky_samples: texture_2d<f32>;

// The pixel 4x4 block `b` samples this frame (16 frames visit them all).
fn sky_pixel(b: vec2<u32>, size: vec2<u32>) -> vec2<u32> {
    let turn = (params.table.w * 7u) & 15u;
    return min(b * 4u + vec2<u32>(turn & 3u, turn >> 2u), size - 1u);
}

// One thread a 4x4 block: dispatch over a quarter of each side.
@compute @workgroup_size(8, 8)
fn sky_sample(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(hit_distance);
    let quarter = (size + 3u) / 4u;
    if id.x >= quarter.x || id.y >= quarter.y { return; }
    let q = sky_pixel(id.xy, size);
    let v = textureLoad(hit_visibility, vec2<i32>(q), 0).xy;
    var sample = vec2<f32>(1.0, 0.0);
    if v.x != NONE {
        if exact_level(chunk_keys[v.x].x >> 8u) { return; }
        let t = textureLoad(hit_distance, vec2<i32>(q), 0).r;
        sample = vec2<f32>(sky_visibility_fine(q, size, v.x, t, hit_face(v, vec3<f32>(0.0), q, size, t)), t);
    }
    textureStore(out_sky_samples, vec2<i32>(id.xy), vec4<f32>(sample, 0.0, 0.0));
}

// `sky_sample` for the hits of exact levels (after it).
@compute @workgroup_size(8, 8)
fn sky_sample_far(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(hit_distance);
    let quarter = (size + 3u) / 4u;
    if id.x >= quarter.x || id.y >= quarter.y { return; }
    let q = sky_pixel(id.xy, size);
    let v = textureLoad(hit_visibility, vec2<i32>(q), 0).xy;
    if v.x == NONE || !exact_level(chunk_keys[v.x].x >> 8u) { return; }
    var sample = vec2<f32>(1.0, 0.0);
    {
        let t = textureLoad(hit_distance, vec2<i32>(q), 0).r;
        sample = vec2<f32>(sky_visibility(q, size, v.x, t, hit_face(v, vec3<f32>(0.0), q, size, t)), t);
    }
    textureStore(out_sky_samples, vec2<i32>(id.xy), vec4<f32>(sample, 0.0, 0.0));
}

// A hit's sky visibility (at `at_hit`, pixel `p`): its 4x4 block's sample
// within `reach`, else the nearest such neighbour's, else its block's.
fn sky_at(p: vec2<i32>, at_hit: vec3<f32>, reach: f32, size: vec2<u32>) -> f32 {
    let quarter = vec2<i32>((size + 3u) / 4u);
    let block = p / 4;
    let toward = select(vec2<i32>(-1), vec2<i32>(1), (p & vec2<i32>(3)) >= vec2<i32>(2));
    let blocks = array<vec2<i32>, 4>(block, block + vec2<i32>(toward.x, 0), block + vec2<i32>(0, toward.y), block + toward);
    for (var b = 0u; b < 4u; b++) {
        let at = blocks[b];
        if any(at < vec2<i32>(0)) || any(at >= quarter) { continue; }
        let s = textureLoad(sky_samples, at, 0).xy;
        if s.y > 0.0 && distance(pixel_ray(sky_pixel(vec2<u32>(at), size), size) * s.y, at_hit) <= reach * 2.0 { return s.x; }
    }
    return textureLoad(sky_samples, block, 0).x;
}

// The pixel block `b` samples this frame.
fn sample_pixel(b: vec2<u32>, size: vec2<u32>) -> vec2<u32> {
    let turn = params.table.w & 3u;
    return min(b * 2u + vec2<u32>(turn & 1u, (turn >> 1u) ^ (turn & 1u)), size - 1u);
}

// One thread a 2x2 block: dispatch over half the size.
@compute @workgroup_size(8, 8)
fn shadow_sample(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(hit_distance);
    let half = (size + 1u) / 2u;
    if id.x >= half.x || id.y >= half.y { return; }
    let q = sample_pixel(id.xy, size);
    let v = textureLoad(hit_visibility, vec2<i32>(q), 0).xy;
    var sample = vec4<u32>(0u);
    if v.x != NONE {
        if exact_level(chunk_keys[v.x].x >> 8u) { return; }
        let t = textureLoad(hit_distance, vec2<i32>(q), 0).r;
        let f = hit_face(v, vec3<f32>(0.0), q, size, t);
        sample = vec4<u32>(4u, bitcast<u32>(t), select(0u, pack_unit(f.ground), dot(f.ground, f.ground) > 0.0), 0u);
        if dot(f.shading, params.sun.xyz) > 0.0 { sample.x |= 2u | u32(sun_walk_fine(q, size, v.x, t, f)); }
    }
    textureStore(out_sun_samples, vec2<i32>(id.xy), sample);
}

// `shadow_sample` for the hits of exact levels (after it).
@compute @workgroup_size(8, 8)
fn shadow_sample_far(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(hit_distance);
    let half = (size + 1u) / 2u;
    if id.x >= half.x || id.y >= half.y { return; }
    let q = sample_pixel(id.xy, size);
    let v = textureLoad(hit_visibility, vec2<i32>(q), 0).xy;
    if v.x == NONE || !exact_level(chunk_keys[v.x].x >> 8u) { return; }
    var sample = vec4<u32>(0u);
    {
        let t = textureLoad(hit_distance, vec2<i32>(q), 0).r;
        let f = hit_face(v, vec3<f32>(0.0), q, size, t);
        sample = vec4<u32>(4u, bitcast<u32>(t), select(0u, pack_unit(f.ground), dot(f.ground, f.ground) > 0.0), 0u);
        if dot(f.shading, params.sun.xyz) > 0.0 { sample.x |= 2u | u32(sun_walk(q, size, v.x, t, f)); }
    }
    textureStore(out_sun_samples, vec2<i32>(id.xy), sample);
}

@compute @workgroup_size(8, 8)
fn shadow(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(hit_distance);
    if id.x >= size.x || id.y >= size.y { return; }
    let p = vec2<i32>(id.xy);
    let sun = params.sun.xyz;
    let v = textureLoad(hit_visibility, p, 0).xy;
    if v.x == NONE {
        textureStore(out_sun, p, vec4<f32>(1.0, sun));
        return;
    }
    let t = textureLoad(hit_distance, p, 0).r;
    // Its block's sample, then the nearer neighbours' (towards the pixel's
    // side of its block): the first within four of its cells (or eight
    // pixels' widths: cells under a pixel, a grazing slope) lends its
    // slope's normal, the first of those that walked its sunlight.
    let at_hit = pixel_ray(id.xy, size) * t;
    let reach = max(4.0 * length(chunk_frame(v.x).m0.xyz), 8.0 * pixel_width(id.xy, size, t));
    let half = vec2<i32>((size + 1u) / 2u);
    let block = p / 2;
    let toward = vec2<i32>(p & vec2<i32>(1)) * 2 - 1;
    let blocks = array<vec2<i32>, 4>(block, block + vec2<i32>(toward.x, 0), block + vec2<i32>(0, toward.y), block + toward);
    var ground = vec3<f32>(0.0);
    var sampled = 0u;
    for (var b = 0u; b < 4u; b++) {
        let at = blocks[b];
        if any(at < vec2<i32>(0)) || any(at >= half) { continue; }
        let s = textureLoad(sun_samples, at, 0);
        if (s.x & 4u) == 0u || distance(pixel_ray(sample_pixel(vec2<u32>(at), size), size) * bitcast<f32>(s.y), at_hit) > reach { continue; }
        if dot(ground, ground) == 0.0 && s.z != 0u { ground = unpack_unit(s.z); }
        if sampled == 0u && (s.x & 2u) != 0u { sampled = 2u | (s.x & 1u); }
    }
    let f = hit_face(v, ground, id.xy, size, t);
    textureStore(out_normal, p, vec4<f32>(f.shading, sky_at(p, at_hit, reach, size)));
    var lit = 0.0;
    if dot(f.shading, sun) > 0.0 {
        let found = sampled != 0u;
        lit = f32(sampled & 1u);
        if !found {
            let n = atomicAdd(&fallbacks.count, 1u);
            fallbacks.pixels[n] = id.x | (id.y << 16u);
            atomicMax(&fallback_groups[0], n / 64u + 1u);
            return;
        }
    }
    store_sun(p, v, f, lit);
}

fn store_sun(p: vec2<i32>, v: vec2<u32>, f: Face, lit: f32) {
    let sun = params.sun.xyz;
    textureStore(out_sun, p, vec4<f32>(lit, sun));
    if plain_colour() && debug_mode() == 0u {
        let material = (v.y >> 21u) & 31u;
        textureStore(out_colour, p, vec4<f32>(palette(material).rgb * (0.25 + 0.75 * max(dot(f.shading, sun), 0.0) * lit), 1.0));
    }
}

// The listed hits' own walks: dispatched indirectly, 64 a workgroup.
@compute @workgroup_size(64)
fn shadow_fallback(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= atomicLoad(&fallbacks.count) { return; }
    let packed = fallbacks.pixels[id.x];
    let q = vec2<u32>(packed & 0xffffu, packed >> 16u);
    let size = textureDimensions(hit_distance);
    let v = textureLoad(hit_visibility, vec2<i32>(q), 0).xy;
    let t = textureLoad(hit_distance, vec2<i32>(q), 0).r;
    let f = hit_face(v, vec3<f32>(0.0), q, size, t);
    store_sun(vec2<i32>(q), v, f, sun_walk(q, size, v.x, t, f));
}
