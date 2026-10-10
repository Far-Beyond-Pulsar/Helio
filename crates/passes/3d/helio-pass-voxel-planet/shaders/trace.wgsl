// Exact traversal of the canonical cube-sphere grid.
//
// A ray is `eye + e + t l` (eye-relative offset `e`, unit direction `l`).
// Plane crossings are solved in each face's eye-centred angle frame and
// sphere crossings with a stable quadratic, so f32 stays precise at
// planetary distance. Level L is chosen from the ray distance: level cells
// project to about one pixel at the start of their range.

struct Hit {
    t: f32,
    i: i32,
    j: i32,
    k: i32,
    info: u32,    // status 0..2 | face 2..5 | level 5..10 | normal 10..13 | loading 13
    record: u32,
    u: f32,
    v: f32,
}

const ST_MISS: u32 = 0u;
const ST_HIT: u32 = 1u;
const ST_EXHAUSTED: u32 = 2u;
const ST_LOADING: u32 = 3u;
const MAX_STEPS: u32 = 2048u;

struct Ray {
    e: vec3<f32>,
    l: vec3<f32>,
    b: f32,    // (eye + e) · l
    eo: f32,   // eye_dir · e
    ee: f32,   // |e|^2
    ol: f32,   // eye_dir · l
    el: f32,   // e · l
}

fn make_ray(e: vec3<f32>, l: vec3<f32>) -> Ray {
    var r: Ray;
    r.e = e;
    r.l = l;
    r.eo = dot(frame.eye.xyz, e);
    r.ee = dot(e, e);
    r.ol = dot(frame.eye.xyz, l);
    r.el = dot(e, l);
    r.b = frame.eye.w * r.ol + r.el;
    return r;
}

struct FaceRay {
    face: u32,
    rho: vec2<f32>,  // distance of the ray origin from each plane axis, along q
    em: vec2<f32>,   // origin offset along each eye plane normal
    lm: vec2<f32>,
    lq: vec2<f32>,
    idx: vec2<i32>,
    frac: vec2<f32>,
    dir: vec2<i32>,  // crossing direction along each index axis (-1, 0, 1)
}

fn face_ray(face: u32, r: Ray) -> FaceRay {
    let f = frame.faces[face];
    var o: FaceRay;
    o.face = face;
    if is_plane() {
        // Parallel cell planes with normals m: coordinates are linear in t.
        o.em = vec2<f32>(dot(r.e, f.m_a.xyz), dot(r.e, f.m_b.xyz));
        o.lm = vec2<f32>(dot(r.l, f.m_a.xyz), dot(r.l, f.m_b.xyz));
        o.idx = f.index.xy;
        o.frac = vec2<f32>(f.q_a.w, f.q_b.w);
        o.dir = vec2<i32>(sign(o.lm));
        return o;
    }
    o.rho = vec2<f32>(f.m_a.w + dot(r.e, f.q_a.xyz), f.m_b.w + dot(r.e, f.q_b.xyz));
    o.em = vec2<f32>(dot(r.e, f.m_a.xyz), dot(r.e, f.m_b.xyz));
    o.lm = vec2<f32>(dot(r.l, f.m_a.xyz), dot(r.l, f.m_b.xyz));
    o.lq = vec2<f32>(dot(r.l, f.q_a.xyz), dot(r.l, f.q_b.xyz));
    o.idx = f.index.xy;
    o.frac = vec2<f32>(f.q_a.w, f.q_b.w);
    // Angular velocity about each plane axis has a constant sign on a line.
    let w = o.rho * o.lm - o.em * o.lq;
    o.dir = vec2<i32>(sign(w));
    return o;
}

fn sincos_small(x: f32) -> vec2<f32> {
    if abs(x) < 0.3 {
        let x2 = x * x;
        // Reciprocal constants: a division compiles to a full divide (the
        // hottest lines of the traversal's profile).
        let s = x * (1.0 - x2 * (1.0 / 6.0) * (1.0 - x2 * (1.0 / 20.0) * (1.0 - x2 * (1.0 / 42.0) * (1.0 - x2 * (1.0 / 72.0)))));
        let c = 1.0 - x2 * 0.5 * (1.0 - x2 * (1.0 / 12.0) * (1.0 - x2 * (1.0 / 30.0) * (1.0 - x2 * (1.0 / 56.0))));
        return vec2<f32>(s, c);
    }
    return vec2<f32>(sin(x), cos(x));
}

fn atan_small(y: f32, x: f32) -> f32 {
    if x > 0.0 && abs(y) < 0.25 * x {
        let r = y / x;
        let r2 = r * r;
        return r * (1.0 - r2 * (1.0 / 3.0 - r2 * (1.0 / 5.0 - r2 * (1.0 / 7.0 - r2 * (1.0 / 9.0)))));
    }
    return atan2(y, x);
}

// Ray parameter where it crosses the plane at base index `plane`.
// Components are chosen with `select` so no vector is indexed dynamically.
fn plane_t(fr: FaceRay, axis: u32, plane: i32) -> f32 {
    let b = axis == 1u;
    if select(fr.dir.x, fr.dir.y, b) == 0 { return 3.0e38; }
    if is_plane() {
        let lm = select(fr.lm.x, fr.lm.y, b);
        if lm * f32(select(fr.dir.x, fr.dir.y, b)) <= 0.0 { return 3.0e38; }
        let d = (f32(plane - select(fr.idx.x, fr.idx.y, b)) - select(fr.frac.x, fr.frac.y, b)) * frame.layer.z;
        return (d - select(fr.em.x, fr.em.y, b)) / lm;
    }
    let delta = (f32(plane - select(fr.idx.x, fr.idx.y, b)) - select(fr.frac.x, fr.frac.y, b)) * frame.layer.z;
    let sc = sincos_small(delta);
    let num = select(fr.rho.x, fr.rho.y, b) * sc.x - select(fr.em.x, fr.em.y, b) * sc.y;
    let den = select(fr.lm.x, fr.lm.y, b) * sc.y - select(fr.lq.x, fr.lq.y, b) * sc.x;
    if den * f32(select(fr.dir.x, fr.dir.y, b)) <= 0.0 { return 3.0e38; }
    return num / den;
}

// Continuous base index (relative to the eye's index) at distance t.
fn face_coord(fr: FaceRay, axis: u32, t: f32) -> f32 {
    let b = axis == 1u;
    if is_plane() {
        return select(fr.frac.x, fr.frac.y, b) + (select(fr.em.x, fr.em.y, b) + t * select(fr.lm.x, fr.lm.y, b)) / frame.layer.z;
    }
    let y = select(fr.em.x, fr.em.y, b) + t * select(fr.lm.x, fr.lm.y, b);
    let x = select(fr.rho.x, fr.rho.y, b) + t * select(fr.lq.x, fr.lq.y, b);
    return select(fr.frac.x, fr.frac.y, b) + atan_small(y, x) / frame.layer.z;
}

// Height of the ray point above the eye's radius.
fn height_rel(r: Ray, t: f32) -> f32 {
    if is_plane() { return r.eo + t * r.ol; }
    let rho = frame.eye.w;
    let ow = r.eo + t * r.ol;
    let ww = r.ee + 2.0 * t * r.el + t * t;
    let num = 2.0 * rho * ow + ww;
    let len = sqrt(max(rho * rho + num, 0.0));
    return num / (len + rho);
}

fn layer_coord(r: Ray, t: f32) -> f32 {
    return frame.layer.x + height_rel(r, t) / frame.layer.y;
}

fn radial_c(r: Ray, layer: i32) -> f32 {
    let dh = (f32(layer - frame.layer_i.x) - frame.layer.x) * frame.layer.y;
    return 2.0 * frame.eye.w * (dh - r.eo) + (dh * dh - r.ee);
}

// Next crossing of the shell [lower, upper) after t: (t, direction).
fn radial_exit(r: Ray, lower: i32, upper: i32, t: f32) -> vec2<f32> {
    if is_plane() {
        // Horizontal layer planes.
        if r.ol < 0.0 { return vec2<f32>(max((layer_height(lower) - r.eo) / r.ol, t), -1.0); }
        if r.ol > 0.0 { return vec2<f32>(max((layer_height(upper) - r.eo) / r.ol, t), 1.0); }
        return vec2<f32>(3.0e38, 1.0);
    }
    let b = r.b;
    if t < -b {
        let c = radial_c(r, lower);
        let d = b * b + c;
        if d >= 0.0 {
            let big = -b + sqrt(d);
            let small = -c / big;
            if small >= t - abs(t) * 1e-6 { return vec2<f32>(max(small, t), -1.0); }
        }
    }
    let c = radial_c(r, upper);
    let root = sqrt(max(b * b + c, 0.0));
    var exit = -b + root;
    if b > 0.0 { exit = c / (b + root); }
    return vec2<f32>(max(exit, t), 1.0);
}

fn level_for(t: f32) -> u32 {
    let ratio = t / frame.lod.x;
    if ratio <= 1.0 { return 0u; }
    return min(u32(floor(log2(ratio))) + 1u, u32(frame.layer_i.z) - 1u);
}

fn cells_at(level: u32) -> i32 {
    return frame.layer_i.y >> level;
}

// Face containing the ray point at t (approximate direction, refined by
// the caller's index checks).
fn face_at(r: Ray, t: f32) -> u32 {
    if is_plane() { return PLANE_FACE; }
    let p = frame.eye.xyz + (r.e + t * r.l) / frame.eye.w;
    let a = abs(p);
    if a.x >= a.y && a.x >= a.z { return select(1u, 0u, p.x >= 0.0); }
    if a.y >= a.z { return select(3u, 2u, p.y >= 0.0); }
    return select(5u, 4u, p.z >= 0.0);
}

struct Cursor {
    face: u32,
    level: u32,
    i: i32,
    j: i32,
    k: i32,
}

// Locate the level cell containing the ray point at t.
fn locate(r: Ray, fr: FaceRay, t: f32, level: u32) -> Cursor {
    work_locates += 1u;
    var c: Cursor;
    c.face = fr.face;
    c.level = level;
    let n = frame.layer_i.y;
    let bi = clamp(fr.idx.x + i32(floor(face_coord(fr, 0u, t))), 0, n - 1);
    let bj = clamp(fr.idx.y + i32(floor(face_coord(fr, 1u, t))), 0, n - 1);
    let bk = frame.layer_i.x + i32(floor(layer_coord(r, t)));
    c.i = bi >> level;
    c.j = bj >> level;
    c.k = bk >> level;
    return c;
}

fn normal_code(axis: u32, step: i32) -> u32 {
    // Normal of the entered face points back against the step.
    return axis * 2u + select(0u, 1u, step > 0);
}

// Per-pixel work counters (diagnostics): loop steps, column lookups, block
// skips, relocations, summed over every trace of the invocation (a primary
// ray the sky bound cut traces again).
var<private> work_steps: u32;
var<private> work_lookups: u32;
var<private> work_skips: u32;
var<private> work_locates: u32;

fn make_hit(status: u32, t: f32, c: Cursor, normal: u32, record: u32) -> Hit {
    var h: Hit;
    h.u = bitcast<f32>(min(work_steps, 65535u) | (min(work_lookups, 65535u) << 16u));
    h.v = bitcast<f32>(min(work_skips, 65535u) | (min(work_locates, 65535u) << 16u));
    h.t = t;
    h.i = c.i;
    h.j = c.j;
    h.k = c.k;
    h.info = status | (c.face << 2u) | (c.level << 5u) | (normal << 10u);
    h.record = record;
    return h;
}

fn level_top_value(i: u32) -> i32 {
    return level_tops[i];
}

// Height above the eye radius of base layer `layer`.
fn layer_height(layer: i32) -> f32 {
    return (f32(layer - frame.layer_i.x) - frame.layer.x) * frame.layer.y;
}

// The ray moves away from the ground (outwards on a planet, up on a plane).
fn rising(r: Ray, t: f32) -> bool {
    if is_plane() { return r.ol > 0.0; }
    return r.b + t > 0.0;
}

// Ray interval inside a plane's footprint (index box [0, n) on both axes):
// (enter, exit), enter > exit when it misses.
fn plane_footprint(fr: FaceRay) -> vec2<f32> {
    let n = f32(frame.layer_i.y);
    var span = vec2<f32>(0.0, 3.0e38);
    for (var axis = 0u; axis < 2u; axis++) {
        let b = axis == 1u;
        let x0 = f32(select(fr.idx.x, fr.idx.y, b)) + select(fr.frac.x, fr.frac.y, b) + select(fr.em.x, fr.em.y, b) / frame.layer.z;
        let v = select(fr.lm.x, fr.lm.y, b) / frame.layer.z;
        if v == 0.0 {
            if x0 < 0.0 || x0 >= n { return vec2<f32>(1.0, 0.0); }
            continue;
        }
        let ta = (0.0 - x0) / v;
        let tb = (n - x0) / v;
        span = vec2<f32>(max(span.x, min(ta, tb)), min(span.y, max(ta, tb)));
    }
    return span;
}

// Highest occupied layer (base cells) of any resident column at `level` or
// coarser. Stale values are only ever too high, which is conservative.
fn sky_layer(level: u32) -> i32 {
    return level_top_value(32u + level);
}

// Continue on the neighbouring face when a level index leaves the face.
fn cross_face(r: Ray, cur_in: Cursor, t: f32) -> Cursor {
    let n_l = cells_at(cur_in.level);
    var edge = 0u;
    if cur_in.i >= n_l { edge = 1u; } else if cur_in.j < 0 { edge = 2u; } else if cur_in.j >= n_l { edge = 3u; }
    let next = frame.neighbours[cur_in.face][edge];
    var c = locate(r, face_ray(next, r), t, cur_in.level);
    c.k = cur_in.k;
    return c;
}

// Largest complete summary block (64/16/4 columns) containing level column
// (ci, cj) whose maximum top lies at or below layer k: (tier, block i,
// block j, top), tier 0 if none. All three loads issue before any test. A
// complete block is fully resident and empty above its top.
fn summary_block(level: u32, face: u32, ci: i32, cj: i32, k: i32) -> vec4<i32> {
    let b3 = vec2<i32>(ci >> 6u, cj >> 6u);
    let b2 = vec2<i32>(ci >> 4u, cj >> 4u);
    let b1 = vec2<i32>(ci >> 2u, cj >> 2u);
    let e3 = block_state[block_slot(level, face, 3u, b3.x, b3.y)];
    let e2 = block_state[block_slot(level, face, 2u, b2.x, b2.y)];
    let e1 = block_state[block_slot(level, face, 1u, b1.x, b1.y)];
    if all(e3.xy == b3) && e3.w == 4096 && k >= e3.z { return vec4<i32>(3, b3, e3.z); }
    if all(e2.xy == b2) && e2.w == 256 && k >= e2.z { return vec4<i32>(2, b2, e2.z); }
    if all(e1.xy == b1) && e1.w == 16 && k >= e1.z { return vec4<i32>(1, b1, e1.z); }
    return vec4<i32>(0);
}

// The run of air layers around layer k shared by the 4x4 columns of
// (ci, cj)'s tier-1 block (`air_blocks`): [x, y), or y <= x when k is not
// in one.
fn air_run(level: u32, face: u32, ci: i32, cj: i32, k: i32) -> vec2<i32> {
    let bi = ci >> 2u;
    let bj = cj >> 2u;
    let e = air_blocks[block_slot(level, face, 1u, bi, bj)];
    let j = (k - e.x) >> 3u;
    if e.w != air_key(bi, bj) || j < 0 || j >= 64 { return vec2<i32>(0); }
    let lo = bitcast<u32>(e.y);
    let hi = bitcast<u32>(e.z);
    let bit = select((lo >> u32(j)) & 1u, (hi >> u32(j - 32)) & 1u, j >= 32);
    if bit == 0u { return vec2<i32>(0); }
    // Up to the first non-air brick at or above j.
    let nlo = ~lo;
    let nhi = ~hi;
    var up = 64;
    if j < 32 {
        let t = nlo & (0xffffffffu << u32(j));
        if t != 0u { up = i32(firstTrailingBit(t)); } else if nhi != 0u { up = 32 + i32(firstTrailingBit(nhi)); }
    } else {
        let t = nhi & (0xffffffffu << u32(j - 32));
        if t != 0u { up = 32 + i32(firstTrailingBit(t)); }
    }
    // Down to just above the last non-air brick below j.
    var down = 0;
    if j > 32 {
        let t = nhi & ((1u << u32(j - 32)) - 1u);
        if t != 0u { down = 32 + i32(firstLeadingBit(t)) + 1; } else if nlo != 0u { down = i32(firstLeadingBit(nlo)) + 1; }
    } else {
        let t = nlo & select((1u << u32(j)) - 1u, 0xffffffffu, j == 32);
        if t != 0u { down = i32(firstLeadingBit(t)) + 1; }
    }
    return vec2<i32>(e.x + down * 8, e.x + up * 8);
}

// Whether traversal may use a level column, from its tier-1 summary block
// and without a hash lookup: 0 no, 1 yes, 2 look it up. While a level
// streams in, only complete 4x4-column blocks are used (partial ones fall
// back to the coarser level), so rays keep large empty-space skips instead
// of crawling column by column through patchy data. The top level always
// looks up, so coverage never has holes.
fn column_hint(level: u32, face: u32, ci: i32, cj: i32) -> u32 {
    if frame.hints.x == 0u { return 2u; }
    let b = vec2<i32>(ci >> 2u, cj >> 2u);
    let e = block_state[block_slot(level, face, 1u, b.x, b.y)];
    if e.w == 16 && all(e.xy == b) { return 1u; }
    // A finite face may end inside a tier-1 block. Only its in-world
    // columns can be published; interior blocks still require all 16.
    let cols = cells_at(level) >> 3u;
    if (cols & 3) != 0 && all(b >= vec2<i32>(0)) && all(e.xy == b) {
        let extent = clamp(vec2<i32>(cols) - b * 4, vec2<i32>(0), vec2<i32>(4));
        let expected = extent.x * extent.y;
        if expected > 0 && e.w == expected { return 1u; }
    }
    if level + 1u >= u32(frame.layer_i.z) { return 2u; }
    return 0u;
}

// Changing resolution is not a ray/solid boundary. A different level can
// already be solid at the cursor, although the ray has only visited air at
// its current level. Keep walking the current field in that overlap instead
// of publishing an interior hit with the last (unrelated) face normal.
// Unedited coarse columns keep an authored-height remainder in their
// header. Occupancy and all summaries enclose it with a ceil-height top.
fn relief_height(c: Column, i: i32, j: i32, level: u32, fraction: u32) -> f32 {
    let top = column_top(c, u32(i & 7), u32(j & 7));
    var correction = 0.0;
    if fraction != 0u { correction = (1.0 - f32(fraction) / 65536.0) * f32(1u << level) * frame.layer.y; }
    return layer_height(top << level) - correction;
}

// First inward crossing of an arbitrary radial surface; no crossing means
// the conservative top voxel was only a broad-phase overlap (including
// grazing silhouette rays). The angular DDA then moves to the next cell.
fn relief_enter(r: Ray, height: f32, t: f32) -> f32 {
    if is_plane() {
        if r.ol >= 0.0 { return 3.0e38; }
        let enter = (height - r.eo) / r.ol;
        if enter < t - abs(t) * 1e-6 { return 3.0e38; }
        return max(enter, t);
    }
    if t >= -r.b { return 3.0e38; }
    let c = 2.0 * frame.eye.w * (height - r.eo) + (height * height - r.ee);
    let discriminant = r.b * r.b + c;
    if discriminant < 0.0 { return 3.0e38; }
    let enter = -c / (-r.b + sqrt(discriminant));
    if enter < t - abs(t) * 1e-6 { return 3.0e38; }
    return max(enter, t);
}

fn level_contains_solid(c: Cursor, record: u32, r: Ray, t: f32) -> bool {
    let col = records[record];
    let x = u32(c.i & 7);
    let y = u32(c.j & 7);
    if !column_knows(col, c.k) { return false; }
    if (col.info & INFO_RELIEF) != 0u {
        let fraction = column_relief_fraction(col, x, y);
        // The relief surface cuts the top cell; below it, occupancy (caves).
        if fraction != 0u && c.k >= column_top(col, x, y) - 1 {
            return height_rel(r, t) <= relief_height(col, c.i, c.j, c.level, fraction);
        }
    }
    return column_cell(col, x, y, c.k) == 1u;
}

// Walk the ray from t_start to t_end. Level selection uses
// `(t + lod_offset) * lod_scale` (dither for primary rays, eye distance for
// secondary rays).
//
// Every outer step classifies the cursor into the largest provably empty
// box in index space (a complete 64/16/4-column summary block or a column
// above its band, or an air brick) and exits it with one generic boundary
// computation. This keeps the SIMD lanes of a warp on the same code path.
// Only mixed bricks run an exact inner cell DDA. Entering a new column, the
// directly addressed summary blocks are tested before the column record is
// looked up: a skip needs no hash lookup at all.
fn trace(r: Ray, t_start: f32, t_end: f32, lod_offset: f32, lod_scale: f32, dither: f32) -> Hit {
    var t = t_start;
    let outer = frame.layer.w;
    if is_plane() {
        // Enter the terrain slab from above and the plane's footprint.
        if height_rel(r, t) > outer {
            if r.ol >= 0.0 { return make_hit(ST_MISS, 0.0, Cursor(), 0u, NONE); }
            t = max(t, (outer - r.eo) / r.ol);
        }
        let span = plane_footprint(face_ray(PLANE_FACE, r));
        if span.x > span.y || span.y < t { return make_hit(ST_MISS, 0.0, Cursor(), 0u, NONE); }
        t = max(t, span.x * (1.0 + 1e-6));
    } else if height_rel(r, t) > outer {
        let c = 2.0 * frame.eye.w * (outer - r.eo) + (outer * outer - r.ee);
        let d = r.b * r.b + c;
        if r.b >= 0.0 || d < 0.0 { return make_hit(ST_MISS, 0.0, Cursor(), 0u, NONE); }
        let big = -r.b + sqrt(d);
        t = max(t, -c / big);
    }
    var fr = face_ray(face_at(r, t), r);
    var cur = locate(r, fr, t, level_for((t + lod_offset) * lod_scale));
    var record = NONE;
    var col: Column;
    var loaded = vec4<i32>(-1);
    var normal = 6u;
    let n_base = frame.layer_i.y - 1;
    var last_t = -1.0;
    var stalls = 0u;
    for (var step = 0u; step < MAX_STEPS; step++) {
        work_steps += 1u;
        if t > t_end { return make_hit(ST_MISS, t, cur, normal, NONE); }
        // Progress guard: near-tangent boundaries can round to zero advance
        // and alternate between two cells. Nudge forward and re-locate.
        if t <= last_t * (1.0 + 1e-7) {
            stalls += 1u;
            if stalls > 3u {
                t = t * (1.0 + 2e-6) + frame.layer.y * 1e-3;
                if t > t_end { return make_hit(ST_MISS, t_end, cur, normal, NONE); }
                cur = locate(r, fr, t, cur.level);
                stalls = 0u;
            }
        } else {
            stalls = 0u;
        }
        last_t = t;
        let n_l = cells_at(cur.level);
        if cur.i < 0 || cur.j < 0 || cur.i >= n_l || cur.j >= n_l {
            // A plane ends at its edges.
            if is_plane() { return make_hit(ST_MISS, t, cur, normal, NONE); }
            cur = cross_face(r, cur, t);
            fr = face_ray(cur.face, r);
            continue;
        }
        let key = vec4<i32>(cur.i >> 3u, cur.j >> 3u, i32(cur.face), i32(cur.level));
        var skip = vec4<i32>(0);
        var summary_checked = false;
        // An accepted level transition already probed this exact column.
        // Reuse that record only within this step, after the normal hint gate.
        var transition_record = NONE;
        if any(key != loaded) {
            // Column-coherent stochastic LOD transition: each column switches
            // level at its own distance within the dither band. The threshold
            // is fixed per column (not per frame), so a moving camera sees
            // each column change level once instead of flickering between
            // two levels every frame while temporal history is rejected.
            // Neighbouring rays in one column agree, so warps stay coherent.
            let hd = f32(hash3(key.x, key.y, key.z | (key.w << 3u), 0x2545f491u) & 1023u) * (1.0 / 1023.0);
            let want = level_for((t + lod_offset) * lod_scale * (1.0 + dither * (hd - 0.5)));
            if want > cur.level {
                let d = want - cur.level;
                var coarser = cur;
                coarser.i >>= d;
                coarser.j >>= d;
                coarser.k >>= d;
                coarser.level = want;
                let found = find_column(column_key0(cur.face, want, coarser.i >> 3u), bitcast<u32>(coarser.j >> 3u));
                if found == NONE || !column_valid(records[found]) || !level_contains_solid(coarser, found, r, t) {
                    cur = coarser;
                    if found != NONE && column_valid(records[found]) { transition_record = found; }
                }
            } else if want < cur.level {
                let finer = locate(r, fr, t, want);
                let hint = column_hint(want, finer.face, finer.i >> 3u, finer.j >> 3u);
                if hint != 0u {
                    let found = find_column(column_key0(finer.face, want, finer.i >> 3u), bitcast<u32>(finer.j >> 3u));
                    if found != NONE && column_valid(records[found]) && column_knows(records[found], finer.k)
                        && !level_contains_solid(finer, found, r, t) {
                        cur = finer;
                        transition_record = found;
                    }
                }
            }
            skip = summary_block(cur.level, cur.face, cur.i >> 3u, cur.j >> 3u, cur.k);
            summary_checked = true;
        }
        if skip.x == 0 && any(vec4<i32>(cur.i >> 3u, cur.j >> 3u, i32(cur.face), i32(cur.level)) != loaded) {
            loop {
                // Unusable columns skip the hash probe.
                if column_hint(cur.level, cur.face, cur.i >> 3u, cur.j >> 3u) != 0u {
                    work_lookups += 1u;
                    record = transition_record;
                    if record == NONE {
                        record = find_column(column_key0(cur.face, cur.level, cur.i >> 3u), bitcast<u32>(cur.j >> 3u));
                    }
                    if record != NONE {
                        col = records[record];
                        if column_valid(col) && column_knows(col, cur.k) { break; }
                    }
                }
                if cur.level + 1u >= u32(frame.layer_i.z) {
                    return make_hit(ST_LOADING, t, cur, normal, NONE);
                }
                cur.i >>= 1u;
                cur.j >>= 1u;
                cur.k >>= 1u;
                cur.level += 1u;
                transition_record = NONE;
                // Fallback changed the key and radial cursor of the query.
                summary_checked = false;
            }
            loaded = vec4<i32>(cur.i >> 3u, cur.j >> 3u, i32(cur.face), i32(cur.level));
        }
        let lv = cur.level;
        let ci = cur.i >> 3u;
        let cj = cur.j >> 3u;
        // Empty box around the cursor: [i0, i1) x [j0, j1) x [k0, k1).
        var i0 = ci * 8;
        var j0 = cj * 8;
        var span = 8;
        var k0 = 0;
        var k1 = 0x3fffffff >> lv;
        var above = skip.x > 0;
        var air_wide = false;
        if !above && !column_knows(col, cur.k) {
            // Beyond a clipped window (the cursor moved vertically inside the
            // column): continue at the coarser level, whose window is larger.
            if cur.level + 1u >= u32(frame.layer_i.z) {
                return make_hit(ST_LOADING, t, cur, normal, NONE);
            }
            cur.i >>= 1u;
            cur.j >>= 1u;
            cur.k >>= 1u;
            cur.level += 1u;
            loaded = vec4<i32>(-1);
            continue;
        }
        if !above {
            k0 = col.top;
            above = cur.k >= k0;
            if above && !summary_checked {
                skip = summary_block(lv, cur.face, ci, cj, cur.k);
            }
        }
        if above {
            // Above every occupied cell here: sky exit (later columns may
            // dither to a finer level than this one), then the summary block.
            if rising(r, t) && height_rel(r, t) > layer_height(sky_layer(min(lv, level_for((t + lod_offset) * lod_scale * (1.0 - 0.5 * dither))))) {
                return make_hit(ST_MISS, t, cur, normal, NONE);
            }
            if skip.x > 0 {
                span = 8 << (2u * u32(skip.x));
                i0 = skip.y * span;
                j0 = skip.z * span;
                k0 = skip.w;
                work_skips += 1u;
            }
        } else {
            let x = u32(cur.i & 7);
            let y = u32(cur.j & 7);
            var fraction = 0u;
            if (col.info & INFO_RELIEF) != 0u {
                fraction = column_relief_fraction(col, x, y);
                // A zero remainder has the same occupied layers as whole
                // cells, so it needs no arbitrary-radius solve. Below the top
                // cell the spans decide (generated caves keep their
                // occupancy under a relief surface).
                if fraction != 0u && cur.k < column_top(col, x, y) - 1 { fraction = 0u; }
            }
            if fraction != 0u {
                // Preserve the actual radial top of this angular cell.
                let surface = relief_height(col, cur.i, cur.j, lv, fraction);
                if height_rel(r, t) <= surface { return make_hit(ST_HIT, t, cur, normal, record); }
                let ta = select(plane_t(fr, 0u, select(cur.i, cur.i + 1, fr.dir.x > 0) << lv), 3.0e38, fr.dir.x == 0);
                let tb = select(plane_t(fr, 1u, select(cur.j, cur.j + 1, fr.dir.y > 0) << lv), 3.0e38, fr.dir.y == 0);
                let enter = relief_enter(r, surface, t);
                if enter <= min(ta, tb) && enter < 3.0e38 {
                    if enter > t_end { return make_hit(ST_MISS, t_end, cur, normal, NONE); }
                    // The solve can cross several radial cells. Publish this
                    // angular cell's enclosing solid layer, not the cursor
                    // from before the solve.
                    var entered = cur;
                    entered.k = column_top(col, x, y) - 1;
                    return make_hit(ST_HIT, enter, entered, normal_code(2u, -1), record);
                }
                let next = max(min(ta, tb), t);
                if next >= 3.0e38 { return make_hit(ST_MISS, t, cur, normal, NONE); }
                t = next;
                cur.k = (frame.layer_i.x + i32(floor(layer_coord(r, t)))) >> lv;
                if ta <= tb { cur.i += fr.dir.x; normal = normal_code(0u, fr.dir.x); }
                else { cur.j += fr.dir.y; normal = normal_code(1u, fr.dir.y); }
                continue;
            }
            let s = span_at(col, cur.k);
            if s.kind == SPAN_AIR {
                // Air across the whole footprint (dug out, a cave's hall): one
                // box, however tall.
                k0 = s.start;
                k1 = s.end;
                air_wide = true;
            } else if s.kind == SPAN_BRICKS {
                let b = u32((cur.k - s.start) >> 3u);
                let state = span_brick(col, s, b);
                if state.x == 1u {
                    return make_hit(ST_HIT, t, cur, normal, record);
                }
                k0 = s.start + i32(b) * 8;
                k1 = k0 + 8;
                air_wide = state.x == 0u;
                if state.x == 2u {
                    // Exact cell DDA inside the mixed brick.
                    var ta = select(plane_t(fr, 0u, select(cur.i, cur.i + 1, fr.dir.x > 0) << lv), 3.0e38, fr.dir.x == 0);
                    var tb = select(plane_t(fr, 1u, select(cur.j, cur.j + 1, fr.dir.y > 0) << lv), 3.0e38, fr.dir.y == 0);
                    var tr = radial_exit(r, cur.k << lv, (cur.k + 1) << lv, t);
                    loop {
                        // The inner DDA can cross the requested endpoint before
                        // returning to the outer traversal range check.
                        if t > t_end { return make_hit(ST_MISS, t_end, cur, normal, NONE); }
                        if brick_bit(state.y, u32(cur.i & 7), u32(cur.j & 7), u32(cur.k & 7)) {
                            return make_hit(ST_HIT, t, cur, normal, record);
                        }
                        let t_next = max(min(ta, min(tb, tr.x)), t);
                        if t_next >= 3.0e38 { return make_hit(ST_MISS, t, cur, normal, NONE); }
                        t = t_next;
                        var angular_crossing = false;
                        if ta <= tb && ta <= tr.x {
                            angular_crossing = true;
                            cur.i += fr.dir.x;
                            normal = normal_code(0u, fr.dir.x);
                            ta = plane_t(fr, 0u, select(cur.i, cur.i + 1, fr.dir.x > 0) << lv);
                        } else if tb <= tr.x {
                            angular_crossing = true;
                            cur.j += fr.dir.y;
                            normal = normal_code(1u, fr.dir.y);
                            tb = plane_t(fr, 1u, select(cur.j, cur.j + 1, fr.dir.y > 0) << lv);
                        } else {
                            let dk = i32(tr.y);
                            cur.k += dk;
                            normal = normal_code(2u, dk);
                            tr = radial_exit(r, cur.k << lv, (cur.k + 1) << lv, t);
                        }
                        if (cur.i >> 3u) != ci || (cur.j >> 3u) != cj || cur.k < k0 || cur.k >= k1 { break; }
                        // A relief cell's top is cut by its surface: reclassify
                        // it before taking its enclosing cell as solid.
                        if angular_crossing && (col.info & INFO_RELIEF) != 0u
                            && column_relief_fraction(col, u32(cur.i & 7), u32(cur.j & 7)) != 0u { break; }
                    }
                    continue;
                }
            } else {
                // A lane span: this lane is solid below its own top inside the
                // span, air above it. The next event is leaving the lane,
                // leaving the span, or descending onto the lane's top: a
                // column of cells crossed in one solve, however tall (the air
                // beside a pit's wall, over a dug floor).
                let lane_top = span_lane_top(col, s, x, y);
                if cur.k < lane_top { return make_hit(ST_HIT, t, cur, normal, record); }
                let ta = select(plane_t(fr, 0u, select(cur.i, cur.i + 1, fr.dir.x > 0) << lv), 3.0e38, fr.dir.x == 0);
                let tb = select(plane_t(fr, 1u, select(cur.j, cur.j + 1, fr.dir.y > 0) << lv), 3.0e38, fr.dir.y == 0);
                let tr = radial_exit(r, lane_top << lv, s.end << lv, t);
                let t_next = max(min(ta, min(tb, tr.x)), t);
                if t_next >= 3.0e38 { return make_hit(ST_MISS, t, cur, normal, NONE); }
                if t_next > t_end { return make_hit(ST_MISS, t_end, cur, normal, NONE); }
                t = t_next;
                if ta <= tb && ta <= tr.x {
                    cur.i += fr.dir.x;
                    normal = normal_code(0u, fr.dir.x);
                    cur.k = (frame.layer_i.x + i32(floor(layer_coord(r, t)))) >> lv;
                } else if tb <= tr.x {
                    cur.j += fr.dir.y;
                    normal = normal_code(1u, fr.dir.y);
                    cur.k = (frame.layer_i.x + i32(floor(layer_coord(r, t)))) >> lv;
                } else if tr.y < 0.0 {
                    // Down through the lane's top (or the span's start).
                    cur.k = lane_top - 1;
                    normal = normal_code(2u, -1);
                } else {
                    cur.k = s.end;
                    normal = normal_code(2u, 1);
                }
                continue;
            }
        }
        // In air under the surface, an eye ray that is not mostly vertical
        // crosses the air its 4x4-column block shares (tunnels, digs) in one
        // step: 21% fewer steps along a tunnel. Sun and sky rays (a level
        // offset) start at a surface and leave it within a few cells, where
        // the extra lookup cost more than it saved (skylight +12%).
        if air_wide && lod_offset == 0.0 && abs(r.ol) < 0.8 {
            let run = air_run(lv, cur.face, ci, cj, cur.k);
            if run.y > run.x {
                span = 32;
                i0 = (ci >> 2u) * 32;
                j0 = (cj >> 2u) * 32;
                k0 = run.x;
                k1 = run.y;
                work_skips += 1u;
            }
        }
        // Exit the empty box with one boundary evaluation per family.
        let xa = select(plane_t(fr, 0u, select(i0, i0 + span, fr.dir.x > 0) << lv), 3.0e38, fr.dir.x == 0);
        let xb = select(plane_t(fr, 1u, select(j0, j0 + span, fr.dir.y > 0) << lv), 3.0e38, fr.dir.y == 0);
        let xr = radial_exit(r, k0 << lv, k1 << lv, t);
        let t_next = max(min(xa, min(xb, xr.x)), t);
        if t_next >= 3.0e38 { return make_hit(ST_MISS, t, cur, normal, NONE); }
        t = t_next;
        let li = clamp(fr.idx.x + i32(floor(face_coord(fr, 0u, t))), 0, n_base) >> lv;
        let lj = clamp(fr.idx.y + i32(floor(face_coord(fr, 1u, t))), 0, n_base) >> lv;
        let lk = (frame.layer_i.x + i32(floor(layer_coord(r, t)))) >> lv;
        cur.i = clamp(li, i0, i0 + span - 1);
        cur.j = clamp(lj, j0, j0 + span - 1);
        cur.k = clamp(lk, k0, k1 - 1);
        if xa <= xb && xa <= xr.x {
            cur.i = select(i0 - 1, i0 + span, fr.dir.x > 0);
            normal = normal_code(0u, fr.dir.x);
        } else if xb <= xr.x {
            cur.j = select(j0 - 1, j0 + span, fr.dir.y > 0);
            normal = normal_code(1u, fr.dir.y);
        } else {
            let dk = i32(xr.y);
            cur.k = select(k0 - 1, k1, dk > 0);
            normal = normal_code(2u, dk);
        }
    }
    return make_hit(ST_EXHAUSTED, t, cur, normal, NONE);
}
