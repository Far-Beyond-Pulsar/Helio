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
//    continues in the chunk beyond the face it left by (`chunk_table`: the
//    same level, else the coarser or finer one there). Output: visibility
//    (chunk slot; cell, face), the hit's distance and a plain colour.
// 3. Resolve (raster): the hits' exact depth, for meshes and later passes;
//    or, in the render graph, publish: the hits into Helio's gbuffer.

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
    table: vec4<u32>, // x: chunk table slots - 1; y: base cells per face edge; z: boxes drawn from inside
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
}

@group(0) @binding(0) var<uniform> params: DrawParams;
@group(0) @binding(1) var<storage, read> statics: array<ChunkStatic>;
@group(0) @binding(2) var<storage, read> chunk_meta: array<ChunkMeta>;
@group(0) @binding(3) var<storage, read> mids: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> mid_leaves: array<u32>;
@group(0) @binding(5) var<storage, read> leaves: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read> leaf_materials: array<u32>;
@group(0) @binding(7) var<uniform> palette: array<vec4<f32>, 32>;
@group(0) @binding(8) var<storage, read> mid_materials: array<u32>;
// Open addressing: key (face | level << 8, cx, cy, cz) then (slot, 0, 0, 0);
// an empty entry's key x is -1.
@group(0) @binding(10) var<storage, read> chunk_table: array<vec4<i32>>;
@group(0) @binding(11) var<storage, read> chunk_keys: array<vec4<i32>>;
// This frame's boxes: chunk slots, those holding the eye first.
@group(0) @binding(12) var<storage, read> draw_list: array<u32>;

// Chunk `slot`'s frame relative to the eye.
fn chunk_frame(slot: u32) -> ChunkDraw {
    let s = statics[slot];
    let origin = (s.origin_hi.xyz - params.eye_hi.xyz) + (s.origin_lo.xyz - params.eye_lo.xyz);
    let o = -vec3<f32>(dot(s.i0.xyz, origin), dot(s.i1.xyz, origin), dot(s.i2.xyz, origin));
    return ChunkDraw(vec4<f32>(s.m0.xyz, origin.x), vec4<f32>(s.m1.xyz, origin.y), vec4<f32>(s.m2.xyz, origin.z),
        vec4<f32>(s.i0.xyz, o.x), vec4<f32>(s.i1.xyz, o.y), vec4<f32>(s.i2.xyz, o.z), s.box);
}
COUNTERS

const NONE: u32 = 0xffffffffu;

fn key_hash(k: vec4<i32>) -> u32 {
    var h = bitcast<u32>(k.x) * 0x9E3779B1u;
    h = (h ^ bitcast<u32>(k.y)) * 0x85EBCA77u;
    h = (h ^ bitcast<u32>(k.z)) * 0xC2B2AE3Du;
    h = (h ^ bitcast<u32>(k.w)) * 0x27D4EB2Fu;
    return h ^ (h >> 15u);
}

// The slot of chunk `k`, or NONE.
fn find_chunk(k: vec4<i32>) -> u32 {
    var i = key_hash(k) & params.table.x;
    for (var probe = 0u; probe < 64u; probe++) {
        let e = chunk_table[i * 2u];
        if e.x == -1 { return NONE; }
        if all(e == k) { return bitcast<u32>(chunk_table[i * 2u + 1u].x); }
        i = (i + 1u) & params.table.x;
    }
    return NONE;
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

@vertex
fn seed_vertex(@builtin(vertex_index) v: u32, @builtin(instance_index) n: u32) -> VOut {
    let slot = draw_list[n];
    let c = chunk_frame(slot);
    let corner = CORNERS[v];
    var x = f32(corner & 1u) * 64.0;
    // Mirroring x maps the box onto itself and reverses its winding: a
    // left-handed frame, or a box drawn from inside (its back faces).
    let mirrored = (bitcast<u32>(c.box.w) & 1u) != 0u;
    if mirrored != (n < params.table.z) { x = 64.0 - x; }
    let local = vec3<f32>(x, f32((corner >> 1u) & 1u) * 64.0, select(c.box.x, c.box.y, (corner & 4u) != 0u));
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

// The first solid cell of chunk `slot` along o + t d from t0 to t1, entering
// through face `axis`; cells [0, 64) across, layers z.
fn trace_chunk(slot: u32, o: vec3<f32>, d: vec3<f32>, t0: f32, t1: f32, axis0: u32, z: vec2<i32>) -> Hit {
    let root = chunk_meta[slot];
    let inv = 1.0 / select(d, vec3<f32>(1e-30), d == vec3<f32>(0.0));
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
        COUNT(1)
        let m = u32((c.x >> 4u) + ((c.y >> 4u) << 2u) + ((c.z >> 4u) << 4u));
        var size = 16;
        if bit64(root.root_occ, m) {
            if bit64(root.root_full, m) { return Hit(t, c, axis, true, mid_materials[slot * 64u + m]); }
            if m != node_at {
                node = mids[slot * 64u + m];
                first_leaf = mid_leaves[slot * 64u + m];
                node_at = m;
            }
            let l = u32(((c.x >> 2u) & 3) + (((c.y >> 2u) & 3) << 2u) + (((c.z >> 2u) & 3) << 4u));
            size = 4;
            if bit64(node.xy, l) {
                if bit64(node.zw, l) { return Hit(t, c, axis, true, mid_materials[slot * 64u + m]); }
                let index = first_leaf + rank64(node.xy & ~node.zw, l);
                if index != leaf_at {
                    leaf = leaves[index];
                    leaf_at = index;
                }
                size = 1;
                if bit64(leaf, u32((c.x & 3) + ((c.y & 3) << 2u) + ((c.z & 3) << 4u))) {
                    return Hit(t, c, axis, true, (leaf_materials[index >> 2u] >> ((index & 3u) * 8u)) & 0xffu);
                }
            }
        }
        // Leave the empty cube of `size` cells holding c.
        let b = c & vec3<i32>(~(size - 1));
        let bound = vec3<f32>(b) + select(vec3<f32>(0.0), vec3<f32>(f32(size)), positive);
        let tt = (bound - o) * inv;
        let tn = min(tt.x, min(tt.y, tt.z));
        if tn >= t1 { break; }
        axis = select(select(2u, 1u, tt.y <= tt.z), 0u, tt.x <= tt.y && tt.x <= tt.z);
        t = max(tn, t);
        var next = clamp(vec3<i32>(floor(o + d * t)), b, b + vec3<i32>(size - 1));
        let step = select(b - vec3<i32>(1), b + vec3<i32>(size), positive);
        if axis == 0u { next.x = step.x; } else if axis == 1u { next.y = step.y; } else { next.z = step.z; }
        c = next;
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

// The drawn chunk holding cell `cell` of `level` (at `frac` within it)
// next to chunk key `k` (same face): that level's, else a coarser one,
// else a finer one. Neighbouring drawn tiles can be levels apart (a parent
// stays drawn until all its children are generated).
fn neighbour(k: vec4<i32>, level: i32, cell: vec3<i32>, frac: vec3<f32>) -> u32 {
    let face = k.x & 7;
    var slot = find_chunk(vec4<i32>(k.x, cell >> vec3<u32>(6u)));
    for (var up = 1; up <= 6 && slot == NONE && level + up <= 30; up++) {
        slot = find_chunk(vec4<i32>(face | ((level + up) << 8), cell >> vec3<u32>(u32(6 + up))));
    }
    for (var down = 1; down <= 3 && slot == NONE && level - down >= 0; down++) {
        let sub = vec3<i32>(floor(frac * f32(1 << u32(down))));
        let finer = (cell << vec3<u32>(u32(down))) + sub;
        slot = find_chunk(vec4<i32>(face | ((level - down) << 8), finer >> vec3<u32>(6u)));
    }
    return slot;
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
    // The chunk whose frame the ray is in, and the chunk-sized cube it is
    // crossing in that frame's cells: the chunk itself, or space next to
    // it where no chunk is (air), crossed as an empty chunk of its level.
    var frame = chunk_frame(box);
    var key = chunk_keys[bitcast<u32>(frame.box.z)];
    // Empty space is crossed in cubes of a chunk of level `level + shift`
    // (absolute index `cube` at that level), coarser after every few.
    var cube = vec3<i32>(0);
    var shift = 0u;
    var real = true;
    var empty_steps = 0u;
    // Grazing rays cross many chunks of a level before their hit.
    for (var chunk = 0u; chunk < 128u; chunk++) {
        COUNT(3)
        let slot = bitcast<u32>(frame.box.z);
        TRACE_PIXEL(chunk, slot, cube, real)
        let o = vec3<f32>(frame.i0.w, frame.i1.w, frame.i2.w);
        let d = vec3<f32>(dot(frame.i0.xyz, dir), dot(frame.i1.xyz, dir), dot(frame.i2.xyz, dir));
        let inv = 1.0 / select(d, vec3<f32>(1e-30), d == vec3<f32>(0.0));
        // The cube in the frame chunk's cells.
        let cube_lo = select(cube * (64 << shift) - key.yzw * 64, vec3<i32>(0), real);
        let cube_size = f32(64 << shift);
        let ca = (vec3<f32>(cube_lo) - o) * inv;
        let cb = (vec3<f32>(cube_lo) + cube_size - o) * inv;
        let cube_hi = max(ca, cb);
        let t_leave = min(cube_hi.x, min(cube_hi.y, cube_hi.z));
        if real {
            // The walk over the chunk's occupied layers.
            let ta = (vec3<f32>(0.0, 0.0, frame.box.x) - o) * inv;
            let tb = (vec3<f32>(64.0, 64.0, frame.box.y) - o) * inv;
            let lo = min(ta, tb);
            let hi = max(ta, tb);
            let t0 = max(max(lo.x, max(lo.y, lo.z)), t_prev);
            let t1 = min(hi.x, min(hi.y, hi.z));
            if t1 > t0 {
                let axis0 = select(select(2u, 1u, lo.y >= lo.z), 0u, lo.x >= lo.y && lo.x >= lo.z);
                let h = trace_chunk(slot, o, d, t0, t1, axis0, vec2<i32>(i32(frame.box.x), i32(frame.box.y)));
                if h.hit {
                    COUNT(2)
                    textureStore(out_exits, p, vec4<f32>(-1.0));
                    let sign = select(1u, 0u, d[h.axis] > 0.0);
                    textureStore(out_visibility, p, vec4<u32>(slot, u32(h.cell.x) | (u32(h.cell.y) << 6u) | (u32(h.cell.z) << 12u) | (h.axis << 18u) | (sign << 20u) | (min(h.material, 255u) << 21u), 0u, 0u));
                    textureStore(out_distance, p, vec4<f32>(h.t, 0.0, 0.0, 0.0));
                    let axis_world = select(select(frame.m2.xyz, frame.m1.xyz, h.axis == 1u), frame.m0.xyz, h.axis == 0u);
                    let normal = normalize(axis_world) * select(-1.0, 1.0, sign == 1u);
                    let light = 0.25 + 0.75 * max(dot(normal, params.sun.xyz), 0.0);
                    var colour = palette[min(h.material, 31u)].rgb;
                    if params.debug.x != 0u {
                        // Level colours.
                        let level = (bitcast<u32>(frame.box.w) >> 8u) & 31u;
                        let colours = array<vec3<f32>, 8>(vec3<f32>(0.9, 0.2, 0.2), vec3<f32>(0.9, 0.6, 0.1), vec3<f32>(0.8, 0.9, 0.1), vec3<f32>(0.2, 0.8, 0.2),
                            vec3<f32>(0.1, 0.8, 0.8), vec3<f32>(0.2, 0.3, 0.9), vec3<f32>(0.6, 0.2, 0.9), vec3<f32>(0.9, 0.2, 0.7));
                        colour = colours[level & 7u];
                    }
                    textureStore(out_colour, p, vec4<f32>(colour * light, 1.0));
                    return;
                }
            }
        }
        // The cell just past the face the ray leaves the cube by, in the
        // frame chunk's cells (exact integers far from the origin).
        t_prev = max(t_leave, t_prev);
        let level = key.x >> 8u;
        let q = o + d * t_leave;
        let axis = select(select(2u, 1u, cube_hi.y <= cube_hi.z), 0u, cube_hi.x <= cube_hi.y && cube_hi.x <= cube_hi.z);
        let span = 64 << shift;
        var local = clamp(vec3<i32>(floor(q)), cube_lo, cube_lo + span - 1);
        let step = select(cube_lo - 1, cube_lo + span, d > vec3<f32>(0.0));
        if axis == 0u { local.x = step.x; } else if axis == 1u { local.y = step.y; } else { local.z = step.z; }
        let cell = key.yzw * 64 + local;
        // Faces: tiles across an edge belong to the next face (the next
        // round seeds them).
        let tiles = (i32(params.table.y >> u32(level)) + 63) / 64;
        let tile = cell.xy >> vec2<u32>(6u);
        if any(tile < vec2<i32>(0)) || any(tile >= vec2<i32>(max(tiles, 1))) { break; }
        // Where in the next cell the ray enters: the side it crossed on
        // the exit axis.
        var frac = clamp(fract(q), vec3<f32>(0.0), vec3<f32>(0.999));
        let entry = select(0.999, 0.0, d[axis] > 0.0);
        if axis == 0u { frac.x = entry; } else if axis == 1u { frac.y = entry; } else { frac.z = entry; }
        let next = neighbour(key, level, cell, frac);
        if next != NONE {
            frame = chunk_frame(next);
            key = chunk_keys[bitcast<u32>(frame.box.z)];
            shift = 0u;
            real = true;
            empty_steps = 0u;
        } else {
            // No chunk there: empty space (air over lower ground), crossed
            // in this frame in cubes that double every four.
            empty_steps += 1u;
            if empty_steps > 40u { break; }
            if empty_steps % 4u == 0u && level + i32(shift) < 30 { shift += 1u; }
            cube = cell >> vec3<u32>(6u + shift);
            real = false;
        }
    }
    // Unresolved: the next round seeds the boxes beyond.
    textureStore(out_exits, p, vec4<f32>(t_prev + 1e-4));
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
    let axis_world = select(select(c.m2.xyz, c.m1.xyz, axis == 1u), c.m0.xyz, axis == 0u);
    let normal = normalize(axis_world) * select(-1.0, 1.0, sign == 1u);
    let f0 = 0.03;
    var out: GOut;
    out.albedo = vec4<f32>(palette[material].rgb, 1.0);
    out.normal = vec4<f32>(normal, f0);
    out.orm = vec4<f32>(1.0, clamp(palette[material].w, 0.04, 1.0), 0.0, f0);
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
