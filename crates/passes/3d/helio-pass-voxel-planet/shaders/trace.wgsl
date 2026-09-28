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
        let s = x * (1.0 - x2 / 6.0 * (1.0 - x2 / 20.0 * (1.0 - x2 / 42.0 * (1.0 - x2 / 72.0))));
        let c = 1.0 - x2 / 2.0 * (1.0 - x2 / 12.0 * (1.0 - x2 / 30.0 * (1.0 - x2 / 56.0)));
        return vec2<f32>(s, c);
    }
    return vec2<f32>(sin(x), cos(x));
}

fn atan_small(y: f32, x: f32) -> f32 {
    if x > 0.0 && abs(y) < 0.25 * x {
        let r = y / x;
        let r2 = r * r;
        return r * (1.0 - r2 * (1.0 / 3.0 - r2 * (1.0 / 5.0 - r2 * (1.0 / 7.0 - r2 / 9.0))));
    }
    return atan2(y, x);
}

// Ray parameter where it crosses the plane at base index `plane`.
// Components are chosen with `select` so no vector is indexed dynamically.
fn plane_t(fr: FaceRay, axis: u32, plane: i32) -> f32 {
    let b = axis == 1u;
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
    let y = select(fr.em.x, fr.em.y, b) + t * select(fr.lm.x, fr.lm.y, b);
    let x = select(fr.rho.x, fr.rho.y, b) + t * select(fr.lq.x, fr.lq.y, b);
    return select(fr.frac.x, fr.frac.y, b) + atan_small(y, x) / frame.layer.z;
}

// Height of the ray point above the eye's radius.
fn height_rel(r: Ray, t: f32) -> f32 {
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

// Per-ray work counters (diagnostics): loop steps, column lookups, block skips,
// relocations.
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

// Walk the ray from t_start to t_end. Level selection uses
// `(t + lod_offset) * lod_scale` (dither for primary rays, eye distance for
// secondary rays).
//
// Every outer step classifies the cursor into the largest provably empty
// box in index space (a complete 64/16/4-column summary block or a column
// above its band, or an air brick) and exits it with one generic boundary
// computation. This keeps the SIMD lanes of a warp on the same code path.
// Only mixed bricks run an exact inner cell DDA.
fn trace(r: Ray, t_start: f32, t_end: f32, lod_offset: f32, lod_scale: f32, dither: f32) -> Hit {
    var t = t_start;
    let outer = frame.layer.w;
    if height_rel(r, t) > outer {
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
    work_steps = 0u;
    work_lookups = 0u;
    work_skips = 0u;
    work_locates = 0u;
    var last_t = -1.0;
    var stalls = 0u;
    for (var step = 0u; step < MAX_STEPS; step++) {
        work_steps = step;
        if t > t_end { return make_hit(ST_MISS, t, cur, normal, NONE); }
        // Progress guard: near-tangent boundaries can round to zero advance
        // and alternate between two cells. Nudge forward and re-locate.
        if t <= last_t * (1.0 + 1e-7) {
            stalls += 1u;
            if stalls > 3u {
                t = t * (1.0 + 2e-6) + frame.layer.y * 1e-3;
                cur = locate(r, fr, t, cur.level);
                stalls = 0u;
            }
        } else {
            stalls = 0u;
        }
        last_t = t;
        let n_l = cells_at(cur.level);
        if cur.i < 0 || cur.j < 0 || cur.i >= n_l || cur.j >= n_l {
            cur = cross_face(r, cur, t);
            fr = face_ray(cur.face, r);
            continue;
        }
        let key = vec4<i32>(cur.i >> 3u, cur.j >> 3u, i32(cur.face), i32(cur.level));
        if any(key != loaded) {
            // Column-coherent stochastic LOD transition (TAA resolves it);
            // neighbouring rays in one column agree, so warps stay coherent.
            let hd = f32(hash3(key.x, key.y, key.z | (key.w << 3u), u32(frame.screen.z)) & 1023u) / 1023.0;
            let want = level_for((t + lod_offset) * lod_scale * (1.0 + dither * (hd - 0.5)));
            if want > cur.level {
                let d = want - cur.level;
                cur.i >>= d;
                cur.j >>= d;
                cur.k >>= d;
                cur.level = want;
            } else if want < cur.level {
                let finer = locate(r, fr, t, want);
                let found = find_column(column_key0(finer.face, want, finer.i >> 3u), bitcast<u32>(finer.j >> 3u));
                if found != NONE && column_valid(records[found]) {
                    cur = finer;
                }
            }
            loop {
                work_lookups += 1u;
                record = find_column(column_key0(cur.face, cur.level, cur.i >> 3u), bitcast<u32>(cur.j >> 3u));
                if record != NONE {
                    col = records[record];
                    if column_valid(col) { break; }
                }
                if cur.level + 1u >= u32(frame.layer_i.z) {
                    return make_hit(ST_LOADING, t, cur, normal, NONE);
                }
                cur.i >>= 1u;
                cur.j >>= 1u;
                cur.k >>= 1u;
                cur.level += 1u;
            }
            loaded = vec4<i32>(cur.i >> 3u, cur.j >> 3u, i32(cur.face), i32(cur.level));
        }
        let lv = cur.level;
        let ci = cur.i >> 3u;
        let cj = cur.j >> 3u;
        let top_cell = column_top_cell(col);
        let bottom_cell = col.k_lo * 8;
        if cur.k < bottom_cell {
            return make_hit(ST_HIT, t, cur, normal, record);
        }
        // Empty box around the cursor: [i0, i1) x [j0, j1) x [k0, k1).
        var i0 = ci * 8;
        var j0 = cj * 8;
        var span = 8;
        var k0 = top_cell;
        var k1 = 0x3fffffff >> lv;
        if cur.k >= top_cell {
            // Above the band: sky exit, then the largest complete summary
            // block whose maximum is below the cursor layer.
            // Later columns may dither to a finer level than this one.
            if r.b + t > 0.0 && height_rel(r, t) > layer_height(sky_layer(min(lv, level_for((t + lod_offset) * lod_scale * (1.0 - 0.5 * dither))))) {
                return make_hit(ST_MISS, t, cur, normal, NONE);
            }
            for (var tier = 3u; tier >= 1u; tier--) {
                let bi = ci >> (2u * tier);
                let bj = cj >> (2u * tier);
                let slot = block_slot(lv, cur.face, tier, bi, bj) * 4u;
                if block_state[slot] == bi && block_state[slot + 1u] == bj
                    && block_state[slot + 3u] == (1 << (4u * tier)) && cur.k >= block_state[slot + 2u] {
                    span = 8 << (2u * tier);
                    i0 = bi * span;
                    j0 = bj * span;
                    k0 = block_state[slot + 2u];
                    work_skips += 1u;
                    break;
                }
            }
        } else {
            let kb = (cur.k >> 3u) - col.k_lo;
            let s = brick_state(col, u32(kb));
            if s.x == 1u {
                return make_hit(ST_HIT, t, cur, normal, record);
            }
            k0 = (cur.k >> 3u) << 3u;
            k1 = k0 + 8;
            if s.x == 2u {
                // Exact cell DDA inside the mixed brick.
                var ta = select(plane_t(fr, 0u, select(cur.i, cur.i + 1, fr.dir.x > 0) << lv), 3.0e38, fr.dir.x == 0);
                var tb = select(plane_t(fr, 1u, select(cur.j, cur.j + 1, fr.dir.y > 0) << lv), 3.0e38, fr.dir.y == 0);
                var tr = radial_exit(r, cur.k << lv, (cur.k + 1) << lv, t);
                loop {
                    if brick_bit(s.y, u32(cur.i & 7), u32(cur.j & 7), u32(cur.k & 7)) {
                        return make_hit(ST_HIT, t, cur, normal, record);
                    }
                    let t_next = max(min(ta, min(tb, tr.x)), t);
                    if t_next >= 3.0e38 { return make_hit(ST_MISS, t, cur, normal, NONE); }
                    t = t_next;
                    if ta <= tb && ta <= tr.x {
                        cur.i += fr.dir.x;
                        normal = normal_code(0u, fr.dir.x);
                        ta = plane_t(fr, 0u, select(cur.i, cur.i + 1, fr.dir.x > 0) << lv);
                    } else if tb <= tr.x {
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
                }
                continue;
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
