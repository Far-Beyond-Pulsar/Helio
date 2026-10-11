// Chunk trees (docs/chunk-trees.md): primary visibility.
//
// Each visible chunk draws its box. The fragment shader takes the eye ray
// into the chunk's affine frame (cells of its level) and walks the chunk's
// 64-tree: the root's 4^3 mid nodes of 16^3 cells, their 4^3 leaves of 4^3
// cells, a leaf's cells. An empty child is crossed in one step to its exit;
// a full child is a hit where the ray enters it. Output: the visibility
// buffer (chunk slot; cell, face) and the hit's distance.
//
// Depth: boxes do not overlap, so along a ray their entries are in the
// order of their hits, and a box's front face is a correct depth for the
// nearest box with a hit. Writing it (not the hit's depth) keeps early
// depth tests on every GPU: a box behind another's hit never runs. The
// hit's exact depth comes from the distance target.

struct DrawParams {
    view_proj: mat4x4<f32>,
    sun: vec4<f32>,  // direction to the sun (xyz), camera near plane (w)
    debug: vec4<u32>, // x: colour by level
}

// A chunk as drawn: its frame relative to the eye. Local cell coordinates
// c map to the world as O + M c; the eye is at local `o`.
struct ChunkDraw {
    m0: vec4<f32>, // M's columns (xyz) and O (w)
    m1: vec4<f32>,
    m2: vec4<f32>,
    i0: vec4<f32>, // M^-1's rows (xyz) and o (w)
    i1: vec4<f32>,
    i2: vec4<f32>,
    // Lowest and one past the highest occupied layer, chunk slot, flags
    // (bit 0: the frame mirrors x; bits 8..16: level).
    box: vec4<f32>,
}

struct ChunkMeta {
    root_occ: vec2<u32>,
    root_full: vec2<u32>,
    z: vec2<i32>,
    leaves: u32,
    status: u32,
}

@group(0) @binding(0) var<uniform> params: DrawParams;
@group(0) @binding(1) var<storage, read> instances: array<ChunkDraw>;
@group(0) @binding(2) var<storage, read> chunk_meta: array<ChunkMeta>;
@group(0) @binding(3) var<storage, read> mids: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> mid_leaves: array<u32>;
@group(0) @binding(5) var<storage, read> leaves: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read> leaf_materials: array<u32>;
@group(0) @binding(8) var<storage, read> mid_materials: array<u32>;
COUNTERS
@group(0) @binding(7) var<uniform> palette: array<vec4<f32>, 32>;

struct VOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) @interpolate(flat) instance: u32,
}

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
fn chunk_vertex(@builtin(vertex_index) v: u32, @builtin(instance_index) n: u32) -> VOut {
    let c = instances[n];
    let corner = CORNERS[v];
    var x = f32(corner & 1u) * 64.0;
    // Mirroring x maps the box onto itself and reverses its winding: a
    // left-handed frame, or a box drawn from inside (its back faces).
    if (bitcast<u32>(c.box.w) & 1u) != 0u { x = 64.0 - x; }
    let local = vec3<f32>(x, f32((corner >> 1u) & 1u) * 64.0, select(c.box.x, c.box.y, (corner & 4u) != 0u));
    let world = vec3<f32>(c.m0.w, c.m1.w, c.m2.w) + c.m0.xyz * local.x + c.m1.xyz * local.y + c.m2.xyz * local.z;
    var out: VOut;
    out.position = params.view_proj * vec4<f32>(world, 1.0);
    out.world = world;
    out.instance = n;
    return out;
}

struct Shaded {
    visibility: vec2<u32>,
    colour: vec4<f32>,
    distance: f32,
}

struct FOut {
    @location(0) visibility: vec2<u32>,
    @location(1) colour: vec4<f32>,
    @location(2) distance: f32,
}

struct FOutInside {
    @location(0) visibility: vec2<u32>,
    @location(1) colour: vec4<f32>,
    @location(2) distance: f32,
    @builtin(frag_depth) depth: f32,
}

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
// through face `axis`.
fn trace_chunk(slot: u32, o: vec3<f32>, d: vec3<f32>, t0: f32, t1: f32, axis0: u32, z: vec2<i32>) -> Hit {
    let root = chunk_meta[slot];
    let inv = 1.0 / select(d, vec3<f32>(1e-30), d == vec3<f32>(0.0));
    let positive = d > vec3<f32>(0.0);
    var t = t0;
    var axis = axis0;
    var c = clamp(vec3<i32>(floor(o + d * t)), vec3<i32>(0, 0, z.x), vec3<i32>(63, 63, z.y - 1));
    for (var n = 0u; n < 512u; n++) {
        COUNT(1)
        let m = u32((c.x >> 4u) + ((c.y >> 4u) << 2u) + ((c.z >> 4u) << 4u));
        var size = 16;
        if bit64(root.root_occ, m) {
            if bit64(root.root_full, m) { return Hit(t, c, axis, true, mid_materials[slot * 64u + m]); }
            let node = mids[slot * 64u + m];
            let l = u32(((c.x >> 2u) & 3) + (((c.y >> 2u) & 3) << 2u) + (((c.z >> 2u) & 3) << 4u));
            size = 4;
            if bit64(node.xy, l) {
                if bit64(node.zw, l) { return Hit(t, c, axis, true, mid_materials[slot * 64u + m]); }
                let index = mid_leaves[slot * 64u + m] + rank64(node.xy & ~node.zw, l);
                let leaf = leaves[index];
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

fn shade_chunk(in: VOut) -> Shaded {
    COUNT(0)
    let c = instances[in.instance];
    let slot = bitcast<u32>(c.box.z);
    let dir = normalize(in.world);
    let o = vec3<f32>(c.i0.w, c.i1.w, c.i2.w);
    let d = vec3<f32>(dot(c.i0.xyz, dir), dot(c.i1.xyz, dir), dot(c.i2.xyz, dir));
    let inv = 1.0 / select(d, vec3<f32>(1e-30), d == vec3<f32>(0.0));
    let ta = (vec3<f32>(0.0, 0.0, c.box.x) - o) * inv;
    let tb = (vec3<f32>(64.0, 64.0, c.box.y) - o) * inv;
    let lo = min(ta, tb);
    let hi = max(ta, tb);
    let t_enter = max(lo.x, max(lo.y, lo.z));
    let t_exit = min(hi.x, min(hi.y, hi.z));
    let t0 = max(t_enter, params.sun.w);
    if t_exit <= t0 { discard; }
    let axis0 = select(select(2u, 1u, lo.y >= lo.z), 0u, lo.x >= lo.y && lo.x >= lo.z);
    let z = vec2<i32>(i32(c.box.x), i32(c.box.y));
    let h = trace_chunk(slot, o, d, t0, t_exit, axis0, z);
    if !h.hit { discard; }
    COUNT(2)
    var out: Shaded;
    out.distance = h.t;
    let sign = select(1u, 0u, d[h.axis] > 0.0);
    out.visibility = vec2<u32>(slot, u32(h.cell.x) | (u32(h.cell.y) << 6u) | (u32(h.cell.z) << 12u) | (h.axis << 18u) | (sign << 20u));
    let axis_world = select(select(c.m2.xyz, c.m1.xyz, h.axis == 1u), c.m0.xyz, h.axis == 0u);
    let normal = normalize(axis_world) * select(-1.0, 1.0, sign == 1u);
    let material = h.material;
    let light = 0.25 + 0.75 * max(dot(normal, params.sun.xyz), 0.0);
    out.colour = vec4<f32>(palette[min(material, 31u)].rgb * light, 1.0);
    if params.debug.x != 0u {
        // Level colours.
        let level = (bitcast<u32>(c.box.w) >> 8u) & 31u;
        let colours = array<vec3<f32>, 8>(vec3<f32>(0.9, 0.2, 0.2), vec3<f32>(0.9, 0.6, 0.1), vec3<f32>(0.8, 0.9, 0.1), vec3<f32>(0.2, 0.8, 0.2),
            vec3<f32>(0.1, 0.8, 0.8), vec3<f32>(0.2, 0.3, 0.9), vec3<f32>(0.6, 0.2, 0.9), vec3<f32>(0.9, 0.2, 0.7));
        out.colour = vec4<f32>(colours[level & 7u] * light, 1.0);
    }
    return out;
}

// Boxes the eye is outside of: the box's front face is the depth.
@fragment
fn chunk_fragment(in: VOut) -> FOut {
    let s = shade_chunk(in);
    return FOut(s.visibility, s.colour, s.distance);
}

// Boxes holding the eye draw their back faces, first: a hit there is
// nearer than any other box.
@fragment
fn chunk_fragment_inside(in: VOut) -> FOutInside {
    let s = shade_chunk(in);
    return FOutInside(s.visibility, s.colour, s.distance, 0.0);
}
