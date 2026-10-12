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
    block: u32,
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
// heatmap, `params.debug.x == 2`).
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
    let z0 = c.box.x;
    let z1 = clamp(top.z + max(top.x, 0.0) * 64.0 + max(top.y, 0.0) * 64.0, c.box.y, 64.0);
    let m = mat3x3<f32>(c.m0.xyz, c.m1.xyz, c.m2.xyz);
    let centre = vec3<f32>(c.m0.w, c.m1.w, c.m2.w) + m * vec3<f32>(32.0, 32.0, (z0 + z1) * 0.5);
    let half_z = (z1 - z0) * 0.5;
    let radius = max(length(m * vec3<f32>(32.0, 32.0, half_z)), length(m * vec3<f32>(32.0, -32.0, half_z)));
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
            if all(o > vec3<f32>(-margin.x, -margin.y, z0 - margin.z)) && all(o < vec3<f32>(64.0 + margin.x, 64.0 + margin.y, z1 + margin.z)) { bin = 0u; }
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
    var x = f32(corner & 1u) * 64.0;
    // Mirroring x maps the box onto itself and reverses its winding: a
    // left-handed frame, or a box drawn from inside (its back faces).
    let mirrored = (bitcast<u32>(c.box.w) & 1u) != 0u;
    if mirrored != inside { x = 64.0 - x; }
    // The box's top lies on its chunk's plane over the solid cells (never
    // below its bottom): raising a corner keeps the faces over the cells.
    let y = f32((corner >> 1u) & 1u) * 64.0;
    let top = chunk_meta[slot].top;
    let local = vec3<f32>(x, y, select(c.box.x, max(c.box.x, top.z + top.x * x + top.y * y), (corner & 4u) != 0u));
    let world = vec3<f32>(c.m0.w, c.m1.w, c.m2.w) + c.m0.xyz * local.x + c.m1.xyz * local.y + c.m2.xyz * local.z;
    var out: VOut;
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
// eye, nearer than any other box.
@fragment
fn seed_fragment_inside(in: VOut) -> SeedInside {
    return SeedInside(in.instance, 0.0);
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
    material: u32,
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
    return bit64(leaves[index], u32((c.x & 3) + ((c.y & 3) << 2u) + ((c.z & 3) << 4u)));
}

// A hit cell's material from its leaf's (or mid's) two: the surface's with
// air above it, the subsurface's under other cells. A ray entering by the
// top face (`from_above`) came through that air.
fn shown_material(pair: u32, slot: u32, c: vec3<i32>, from_above: bool) -> u32 {
    return select((pair >> 8u) & 0xffu, pair & 0xffu, from_above || !solid_at(slot, c + vec3<i32>(0, 0, 1)));
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
    var leaf = vec2<u32>(0u);
    for (var n = 0u; n < 512u; n++) {
        WCOUNT(1)
        let m = u32((c.x >> 4u) + ((c.y >> 4u) << 2u) + ((c.z >> 4u) << 4u));
        var size = 16;
        if bit64(root.root_occ, m) {
            if bit64(root.root_full, m) { return Hit(t, c, axis, true, shown_material(mid_materials[slot * 64u + m], slot, c, axis == 2u && !positive.z)); }
            if m != node_at {
                node = mids[slot * 64u + m];
                first_leaf = mid_leaves[slot * 64u + m];
                node_at = m;
            }
            let l = u32(((c.x >> 2u) & 3) + (((c.y >> 2u) & 3) << 2u) + (((c.z >> 2u) & 3) << 4u));
            size = 4;
            if bit64(node.xy, l) {
                if bit64(node.zw, l) { return Hit(t, c, axis, true, shown_material(mid_materials[slot * 64u + m], slot, c, axis == 2u && !positive.z)); }
                let index = first_leaf + rank64(node.xy & ~node.zw, l);
                if index != leaf_at {
                    leaf = leaves[index];
                    leaf_at = index;
                }
                size = 1;
                if bit64(leaf, u32((c.x & 3) + ((c.y & 3) << 2u) + ((c.z & 3) << 4u))) {
                    return Hit(t, c, axis, true, shown_material(leaf_materials[index >> 1u] >> ((index & 1u) * 16u), slot, c, axis == 2u && !positive.z));
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
    return Hit(t1, c, axis, false, 0u);
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
// A column entry's layer key (`drawn::COLUMN`).
const COLUMN: i32 = -2147483647 - 1;

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
    let lost = Place(NONE, LOST, vec3<i32>(0), vec3<i32>(0), NONE);
    var diff = 0;
    var way = 0;
    var column = vec4<i32>(0);
    var shown = false;
    for (var n = 0u; n < 16u; n++) {
        let c = cell_at(cell, frac, diff);
        let e = find_entry(vec4<i32>(face | ((level + diff) << 8), c.xy >> vec2<u32>(6u), COLUMN));
        var go = 2;
        if e != NONE {
            column = chunk_table[e * 2u + 1u];
            go = choice(column.w, r);
        }
        if go == 1 {
            shown = true;
            break;
        }
        // Coarser; on the way up, past the ancestors of the hidden side of
        // a switch across levels (a finer search found a hole: lost).
        if go == 2 || (go == 0 && way > 0) {
            if way < 0 || level + diff >= 30 { return lost; }
            way = 1;
            diff += 1;
        } else {
            if way > 0 || diff <= -6 || level + diff <= 0 { return lost; }
            way = -1;
            diff -= 1;
        }
    }
    if !shown { return lost; }
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
        // chunk (in coarser cells the top would round).
        let f = frame_of(up < 0 || crossed, vec4<i32>(tile_key, tile, (column.z - 1) >> 6u));
        if crossed && f == NONE { return lost; }
        return Place(NONE, level + up, vec3<i32>(at * 64, top), vec3<i32>(at * 64 + 64, top + (1 << 20)), f);
    }
    // Under the tile's chunks: reached through a cave mouth among them.
    if layer < column.x { return Place(NONE, BELOW, vec3<i32>(tile, level + diff), vec3<i32>(face, 0, 0), NONE); }
    let s = find_entry(vec4<i32>(tile_key, tile, layer));
    if s != NONE { return Place(bitcast<u32>(chunk_table[s * 2u + 1u].x), level + diff, vec3<i32>(0), vec3<i32>(0), NONE); }
    // An empty layer among them.
    let f = frame_of(diff < 0 || crossed, vec4<i32>(tile_key, tile, column.x));
    if crossed && f == NONE { return lost; }
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
    material: u32,
}

// The walk of ray org + t dir (org eye-relative, dir unit) from t_start to
// t_end, for
// a pixel of dither `r`, over at most `boxes` boxes (`counted`: in the
// counting build's counters, as `pixel`).
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
                    return WalkEnd(true, h.t, slot, h.cell, h.axis, select(1u, 0u, d[h.axis] > 0.0), h.material);
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
                break;
            }
            if next.level == BELOW {
                // Under a tile's chunks, which hold only its surface band:
                // solid where the ray leaves this box, until the tile's
                // whole volume (its caves) is generated.
                let n = atomicAdd(&reach.count, 1u);
                if n < arrayLength(&reach.keys) { reach.keys[n] = vec4<i32>(next.hi.x | (next.lo.z << 8u), next.lo.xy, 0); }
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
        textureStore(out_colour, p, vec4<f32>(0.0));
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
    if own != NONE && choice(chunk_table[own * 2u + 1u].y, r) != 1 {
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
        } else if next.level == LOST || next.level == BELOW {
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
    textureStore(out_visibility, p, vec4<u32>(end.slot, u32(end.cell.x) | (u32(end.cell.y) << 6u) | (u32(end.cell.z) << 12u) | (end.axis << 18u) | (end.sign << 20u) | (min(end.material, 255u) << 21u), 0u, 0u));
    textureStore(out_distance, p, vec4<f32>(end.t, 0.0, 0.0, 0.0));
    let c = chunk_frame(end.slot);
    let axis_world = select(select(c.m2.xyz, c.m1.xyz, end.axis == 1u), c.m0.xyz, end.axis == 0u);
    let normal = normalize(axis_world) * select(-1.0, 1.0, end.sign == 1u);
    let light = 0.25 + 0.75 * max(dot(normal, params.sun.xyz), 0.0);
    var colour = palette(end.material).rgb;
    if params.debug.x == 2u {
        // Steps: red to 64, yellow to 128, white to 192.
        let h = f32(heat) / 64.0;
        colour = clamp(vec3<f32>(h, h - 1.0, h - 2.0), vec3<f32>(0.0), vec3<f32>(1.0)) / light;
    } else if params.debug.x == 1u {
        // Level colours.
        let level = (bitcast<u32>(c.box.w) >> 8u) & 31u;
        let colours = array<vec3<f32>, 8>(vec3<f32>(0.9, 0.2, 0.2), vec3<f32>(0.9, 0.6, 0.1), vec3<f32>(0.8, 0.9, 0.1), vec3<f32>(0.2, 0.8, 0.2),
            vec3<f32>(0.1, 0.8, 0.8), vec3<f32>(0.2, 0.3, 0.9), vec3<f32>(0.6, 0.2, 0.9), vec3<f32>(0.9, 0.2, 0.7));
        colour = colours[level & 7u];
    }
    textureStore(out_colour, p, vec4<f32>(colour * light, 1.0));
}

// ---- 3. Resolve ----

@group(1) @binding(4) var hit_distance: texture_2d<f32>;

@vertex
fn resolve_vertex(@builtin(vertex_index) v: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((v << 1u) & 2u), f32(v & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn resolve_depth(@builtin(position) pixel: vec4<f32>) -> @builtin(frag_depth) f32 {
    let t = textureLoad(hit_distance, vec2<i32>(pixel.xy), 0).r;
    if t <= 0.0 { discard; }
    let size = textureDimensions(hit_distance);
    let clip = params.view_proj * vec4<f32>(pixel_ray(vec2<u32>(pixel.xy), size) * t, 1.0);
    return clip.z / clip.w;
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
fn surface_look(slot: u32, cell: vec3<i32>, axis: u32, sign: u32, material: u32, position: vec3<f32>, pixel_m: f32) -> Look {
    let c = chunk_frame(slot);
    let key = chunk_keys[slot];
    let projected = length(c.m0.xyz) / max(pixel_m, 1e-6);
    let mosaic = mosaic_weight(projected);
    // The cell centre's noise-domain point: the same pigment from every
    // face of it.
    let centre = vec3<f32>(c.m0.w, c.m1.w, c.m2.w) + mat3x3<f32>(c.m0.xyz, c.m1.xyz, c.m2.xyz) * (vec3<f32>(cell) + 0.5);
    let p = params.eye_domain.xyz + vec3<i32>(floor(params.eye_domain_frac.xyz + centre / DOMAIN_UNIT));
    let hv = hash3(p.x, p.y, p.z, 0x68bc21ebu);
    let pigment = 1.0 + materials.detail.y * (mix(0.5, f32(hv & 255u) / 255.0, mosaic) - 0.5);
    let albedo = material_albedo(material, p, hash3(p.x, p.y, p.z, 0x1b873593u), pixel_m, mosaic) * pigment;
    // Occlusion: around the air cell in front of the face.
    let ao_weight = smoothstep(0.75, 1.25, projected);
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
    let material = min((v.y >> 21u) & 0xffu, 31u);
    let shading = textureLoad(hit_normal, p, 0);
    let normal = shading.xyz;
    let cell = vec3<i32>(vec3<u32>(v.y, v.y >> 6u, v.y >> 12u) & vec3<u32>(63u));
    // A pixel's width at the hit (m).
    let pixel_m = t * length(pixel_ray(vec2<u32>(pixel.xy) + vec2<u32>(1u, 0u), size) - pixel_ray(vec2<u32>(pixel.xy), size));
    let look = surface_look(v.x, cell, axis, sign, material, position, pixel_m);
    let f0 = 0.03;
    var out: GOut;
    out.albedo = vec4<f32>(look.albedo, 1.0);
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
    return normalize(n);
}

// Hit `v`. Voxels are lit by their own faces (a
// slope reads as its steps, each shadowing the next); a natural step (air
// above it) is lit partly with the slope's normal, `Materials::detail.w`
// (step softness), its sun ray passing its own steps. Never by the voxel's
// size on screen: cells span a pixel to half one across every level of
// the cut, so a size-dependent light bands by level.
// The slope's normal is `ground` when given (a nearby sample's), else
// measured here.
fn hit_face(v: vec2<u32>, ground_hint: vec3<f32>) -> Face {
    let c = chunk_frame(v.x);
    let key = chunk_keys[v.x];
    let axis = (v.y >> 18u) & 3u;
    let sign = (v.y >> 20u) & 1u;
    let axis_world = select(select(c.m2.xyz, c.m1.xyz, axis == 1u), c.m0.xyz, axis == 0u);
    let normal = normalize(axis_world) * select(-1.0, 1.0, sign == 1u);
    let cell_m = length(axis_world);
    var f = Face(normal, cell_m, normal, 0.0, vec3<f32>(0.0));
    let cell = vec3<i32>(vec3<u32>(v.y, v.y >> 6u, v.y >> 12u) & vec3<u32>(63u));
    if solid_near(v.x, key, cell + vec3<i32>(0, 0, 1)) { return f; }
    let w = materials.detail.w;
    var ground = ground_hint;
    if dot(ground, ground) == 0.0 { ground = slope_normal(v.x, key, cell, c); }
    if w <= 0.0 || dot(ground, ground) == 0.0 { return f; }
    f.ground = ground;
    f.shading = normalize(mix(normal, ground, w));
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

// Whether the hit at pixel `q` (distance `t`, face `f`) is in the sun.
fn sun_walk(q: vec2<u32>, size: vec2<u32>, slot: u32, t: f32, f: Face) -> f32 {
    let org = pixel_ray(q, size) * t + f.normal * (f.cell * 0.01);
    let end = walk(Walk(slot, vec3<i32>(0), vec3<i32>(64), true), org, params.sun.xyz, f.skip * f.cell, 1e30, fade_dither(q), 96u, false, q);
    return select(1.0, 0.0, end.hit);
}

// Sky rays a sample casts a frame (one sample a 4x4 block), and how far
// they reach (cells).
const SKY_RAYS: u32 = 4u;
const SKY_RANGE_CELLS: f32 = 48.0;

// The share of the sky the hit at pixel `q` (distance `t`, face `f`, chunk
// `slot`) sees: cosine-weighted rays about the planet's up, turned each
// frame (TSR gathers them); an occluder near the range end shades less (no
// edge at the range). A ceiling looks out sideways.
fn sky_visibility(q: vec2<u32>, size: vec2<u32>, slot: u32, t: f32, f: Face) -> f32 {
    let up = normalize(chunk_frame(slot).m2.xyz);
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
        let end = walk(Walk(slot, vec3<i32>(0), vec3<i32>(64), true), org, dir, f.skip * f.cell, range, fade_dither(q), 24u, false, q);
        open += select(1.0, smoothstep(0.5, 1.0, end.t / range), end.hit && end.t < range);
    }
    return select(1.0, open / total, total > 0.0);
}

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
        let t = textureLoad(hit_distance, vec2<i32>(q), 0).r;
        sample = vec2<f32>(sky_visibility(q, size, v.x, t, hit_face(v, vec3<f32>(0.0))), t);
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
        let t = textureLoad(hit_distance, vec2<i32>(q), 0).r;
        let f = hit_face(v, vec3<f32>(0.0));
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
    // side of its block): the first within four of its cells lends its
    // slope's normal, the first of those that walked its sunlight.
    let at_hit = pixel_ray(id.xy, size) * t;
    let reach = 4.0 * length(chunk_frame(v.x).m0.xyz);
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
    let f = hit_face(v, ground);
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
    if params.debug.x == 0u {
        let material = min((v.y >> 21u) & 0xffu, 31u);
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
    let f = hit_face(v, vec3<f32>(0.0));
    store_sun(vec2<i32>(q), v, f, sun_walk(q, size, v.x, t, f));
}
