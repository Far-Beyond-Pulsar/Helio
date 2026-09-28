// Conservative beam optimization (after Laine & Karras 2010, "Efficient
// Sparse Voxel Octrees"): trace one cone per 4x4 pixel tile and return a
// distance before which the whole cone is inside verified empty boxes.
// Primary rays of the tile start there, so they skip the empty space that
// every tile ray shares.
//
// Rigour: the cone's cross-section disc (radius w = t tan a) must stay inside
// the current empty box. At each box entry the disc must clear every face
// except the one it came through; the box exit is shrunk by w; crossing into
// the neighbour box requires that neighbour to be empty on the next step.
// Both LOD levels a dithered primary ray may select at a point are checked.

const BEAM: u32 = 4u;
const BEAM_STEPS: u32 = 192u;
const BIG: i32 = 0x3fffffff;

@group(0) @binding(16) var<storage, read_write> beams: array<f32>;

struct BeamBox {
    ok: bool,
    level: u32,
    i0: i32,
    j0: i32,
    span: i32,
    k0: i32,
    k1: i32,
}

fn beam_fail() -> BeamBox {
    var b: BeamBox;
    b.ok = false;
    return b;
}

// Largest empty box at `level_in` (or the coarser level a primary ray would
// fall back to) containing base cell (bi, bj, bk).
fn beam_box(face: u32, level_in: u32, bi: i32, bj: i32, bk: i32) -> BeamBox {
    var level = level_in;
    var col: Column;
    loop {
        let record = find_column(column_key0(face, level, (bi >> level) >> 3u), bitcast<u32>((bj >> level) >> 3u));
        if record != NONE {
            col = records[record];
            if column_valid(col) { break; }
        }
        if level + 1u >= u32(frame.layer_i.z) { return beam_fail(); }
        level += 1u;
    }
    let i = bi >> level;
    let j = bj >> level;
    let k = bk >> level;
    let ci = i >> 3u;
    let cj = j >> 3u;
    let top = column_top_cell(col);
    if k < col.k_lo * 8 { return beam_fail(); }
    var b: BeamBox;
    b.ok = true;
    b.level = level;
    b.i0 = ci * 8;
    b.j0 = cj * 8;
    b.span = 8;
    if k >= top {
        b.k0 = top;
        b.k1 = BIG >> level;
        for (var tier = 3u; tier >= 1u; tier--) {
            let tbi = ci >> (2u * tier);
            let tbj = cj >> (2u * tier);
            let slot = block_slot(level, face, tier, tbi, tbj) * 4u;
            if block_state[slot] == tbi && block_state[slot + 1u] == tbj
                && block_state[slot + 3u] == (1 << (4u * tier)) && k >= block_state[slot + 2u] {
                b.span = 8 << (2u * tier);
                b.i0 = tbi * b.span;
                b.j0 = tbj * b.span;
                b.k0 = block_state[slot + 2u];
                break;
            }
        }
        return b;
    }
    let s = brick_state(col, u32((k >> 3u) - col.k_lo));
    if s.x != 0u { return beam_fail(); }
    b.k0 = (k >> 3u) << 3u;
    b.k1 = b.k0 + 8;
    return b;
}

// Plane crossing at base index `plane + offset` (integer part kept exact).
fn plane_tf(fr: FaceRay, axis: u32, plane: i32, offset: f32) -> f32 {
    let b = axis == 1u;
    let delta = (f32(plane - select(fr.idx.x, fr.idx.y, b)) + offset - select(fr.frac.x, fr.frac.y, b)) * frame.layer.z;
    let sc = sincos_small(delta);
    let num = select(fr.rho.x, fr.rho.y, b) * sc.x - select(fr.em.x, fr.em.y, b) * sc.y;
    let den = select(fr.lm.x, fr.lm.y, b) * sc.y - select(fr.lq.x, fr.lq.y, b) * sc.x;
    if den * f32(select(fr.dir.x, fr.dir.y, b)) <= 0.0 { return 3.0e38; }
    return num / den;
}

// Crossing of the sphere at a fractional base layer, entering from above
// (descending) or leaving through it (ascending).
fn sphere_tf(r: Ray, layer: i32, offset: f32, t: f32, descending: bool) -> f32 {
    let dh = (f32(layer - frame.layer_i.x) + offset - frame.layer.x) * frame.layer.y;
    let c = 2.0 * frame.eye.w * (dh - r.eo) + (dh * dh - r.ee);
    let d = r.b * r.b + c;
    if descending {
        if t >= -r.b || d < 0.0 { return 3.0e38; }
        return -c / (-r.b + sqrt(d));
    }
    let root = sqrt(max(d, 0.0));
    return select(-r.b + root, c / (r.b + root), r.b > 0.0);
}

fn beam_trace(r: Ray, tan_a: f32, lo_scale: f32, hi_scale: f32, sky: SkyRay) -> f32 {
    var t = 0.0;
    let outer = frame.layer.w;
    if height_rel(r, t) > outer {
        let c = 2.0 * frame.eye.w * (outer - r.eo) + (outer * outer - r.ee);
        let d = r.b * r.b + c;
        if r.b >= 0.0 || d < 0.0 { return frame.lod.w; }
        // The cone widens past the centre ray: back off by its radius.
        t = max(0.0, -c / (-r.b + sqrt(d)) * (1.0 - 2.0 * tan_a) - 1.0);
    }
    var fr = face_ray(face_at(r, t), r);
    var entry = 3u;
    var touch = t;
    let n = frame.layer_i.y;
    for (var step = 0u; step < BEAM_STEPS; step++) {
        // Coordinates relative to the eye's base indices keep f32 precise.
        let ri = face_coord(fr, 0u, t);
        let rj = face_coord(fr, 1u, t);
        let rk = layer_coord(r, t);
        let bi = fr.idx.x + i32(floor(ri));
        let bj = fr.idx.y + i32(floor(rj));
        let bk = frame.layer_i.x + i32(floor(rk));
        if bi < 0 || bj < 0 || bi >= n || bj >= n { return touch; }
        let w = t * tan_a + 0.001;
        // Every ray of the cone must be rising: its elevation differs from
        // the centre ray's by at most the cone half angle.
        if r.b + t > frame.eye.w * tan_a * 1.01 + 0.01 {
            var bound = layer_height(sky_layer(level_for(t * lo_scale)));
            if sky.sector > -2 && height_rel(r, t) - w > frame.lod.z {
                // Disc points are at most w (arc w / radius) nearer the eye.
                bound = min(bound, layer_height(horizon_layer(sky, eye_phi(sky, t) - w / (frame.eye.w * 0.99))));
            }
            if height_rel(r, t) - w > bound {
                return frame.lod.w;
            }
        }
        // Boxes for both LOD levels a dithered primary ray may use here,
        // intersected in the finer level's cells.
        var a = beam_box(fr.face, level_for(t * lo_scale), bi, bj, bk);
        if !a.ok { return touch; }
        var c = a;
        let l_hi = level_for(t * hi_scale);
        if l_hi != a.level {
            c = beam_box(fr.face, l_hi, bi, bj, bk);
            if !c.ok { return touch; }
        }
        if c.level < a.level {
            let swap = a;
            a = c;
            c = swap;
        }
        let lv = a.level;
        let d = c.level - lv;
        let i0 = max(a.i0, c.i0 << d);
        let j0 = max(a.j0, c.j0 << d);
        let i1 = min(a.i0 + a.span, (c.i0 + c.span) << d);
        let j1 = min(a.j0 + a.span, (c.j0 + c.span) << d);
        let k0 = max(a.k0, c.k0 << d);
        var k1 = a.k1;
        if c.k1 < (BIG >> c.level) { k1 = min(k1, c.k1 << d); }
        let scale = f32(1 << lv);
        // Disc radius in level cells (tangential cells are >= 0.75 nominal).
        let wh = w / (frame.layer.y * scale * 0.75);
        let wv = w / (frame.layer.y * scale);
        let open_top = k1 >= (BIG >> lv);
        let descending_now = t < -r.b;
        // Clearance (level cells) to every face, except the face this box was
        // entered through (the region behind it was verified already).
        let lo_i = (ri - f32((i0 << lv) - fr.idx.x)) / scale;
        let hi_i = (f32((i1 << lv) - fr.idx.x) - ri) / scale;
        let lo_j = (rj - f32((j0 << lv) - fr.idx.y)) / scale;
        let hi_j = (f32((j1 << lv) - fr.idx.y) - rj) / scale;
        let lo_k = (rk - f32((k0 << lv) - frame.layer_i.x)) / scale;
        let hi_k = select((f32((k1 << lv) - frame.layer_i.x) - rk) / scale, 1.0e30, open_top);
        let behind_i = select(hi_i, lo_i, fr.dir.x > 0);
        let ahead_i = select(lo_i, hi_i, fr.dir.x > 0);
        let behind_j = select(hi_j, lo_j, fr.dir.y > 0);
        let ahead_j = select(lo_j, hi_j, fr.dir.y > 0);
        let behind_k = select(lo_k, hi_k, descending_now);
        let ahead_k = select(hi_k, lo_k, descending_now);
        if (fr.dir.x == 0 && min(lo_i, hi_i) < wh) || ahead_i < wh || (entry != 0u && behind_i < wh)
            || (fr.dir.y == 0 && min(lo_j, hi_j) < wh) || ahead_j < wh || (entry != 1u && behind_j < wh)
            || ahead_k < wv || (entry != 2u && behind_k < wv) {
            return touch;
        }
        // Point exit of the box.
        let pa = select(i0, i1, fr.dir.x > 0) << lv;
        let pb = select(j0, j1, fr.dir.y > 0) << lv;
        let ta = select(plane_tf(fr, 0u, pa, 0.0), 3.0e38, fr.dir.x == 0);
        let tb = select(plane_tf(fr, 1u, pb, 0.0), 3.0e38, fr.dir.y == 0);
        let descending = descending_now;
        var tr = 3.0e38;
        if descending {
            tr = sphere_tf(r, k0 << lv, 0.0, t, true);
        } else if !open_top {
            tr = sphere_tf(r, k1 << lv, 0.0, t, false);
        }
        let t_exit = max(min(ta, min(tb, tr)), t);
        if t_exit >= 3.0e38 { return frame.lod.w; }
        // Shrunk exits with the disc radius at the (later) point exit.
        let we = t_exit * tan_a + 0.001;
        let weh = we / (frame.layer.y * scale * 0.75);
        let wev = we / (frame.layer.y * scale);
        let sa = select(plane_tf(fr, 0u, pa, -f32(fr.dir.x) * weh * scale), 3.0e38, fr.dir.x == 0);
        let sb = select(plane_tf(fr, 1u, pb, -f32(fr.dir.y) * weh * scale), 3.0e38, fr.dir.y == 0);
        var sr = 3.0e38;
        if descending {
            sr = sphere_tf(r, k0 << lv, wev * scale, t, true);
        } else if !open_top {
            sr = sphere_tf(r, k1 << lv, -wev * scale, t, false);
        }
        let t_touch = max(min(sa, min(sb, sr)), t);
        touch = max(touch, t_touch);
        // The disc must reach the face the point leaves through first.
        var axis = 2u;
        if ta <= tb && ta <= tr { axis = 0u; } else if tb <= tr { axis = 1u; }
        let first = select(select(sr, sb, axis == 1u), sa, axis == 0u);
        if first > t_touch + t_touch * 1e-6 { return touch; }
        t = t_exit + t_exit * 1e-6;
        entry = axis;
        let next_face = face_at(r, t);
        if next_face != fr.face {
            fr = face_ray(next_face, r);
            entry = 3u;
        }
    }
    return touch;
}

@compute @workgroup_size(8, 8)
fn beam(@builtin(global_invocation_id) id: vec3<u32>) {
    let tiles = (vec2<u32>(frame.screen.xy) + BEAM - 1u) / BEAM;
    if any(id.xy >= tiles) { return; }
    let base = vec2<f32>(id.xy * BEAM);
    let centre = pixel_ray(base + f32(BEAM) * 0.5);
    var cos_min = 1.0;
    for (var c = 0u; c < 4u; c++) {
        let corner = base + vec2<f32>(select(0.5, f32(BEAM) - 0.5, (c & 1u) != 0u), select(0.5, f32(BEAM) - 0.5, (c & 2u) != 0u));
        cos_min = min(cos_min, dot(centre, pixel_ray(corner)));
    }
    let tan_a = sqrt(max(1.0 - cos_min * cos_min, 0.0)) / max(cos_min, 1e-4) * 1.02 + 1e-5;
    let d = frame.lod.y;
    let r = make_ray(camera.position_near.xyz, centre);
    beams[id.x + id.y * tiles.x] = beam_trace(r, tan_a, (1.0 - 0.5 * d) * 0.97, 1.0 + 0.5 * d, eye_sky(centre, tan_a));
}
