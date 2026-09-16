// Hi-Z screen-space ray marching — the traversal itself, shared.
//
// Prepended to any shader whose source contains `//!use helio_hiz`.
// See helio_core::shader.
//
// This exists because SSR and water both need to march the same pyramid, and
// the water pass had grown its own linear march with a hit test that compared
// raw NDC depths against a fixed tolerance — in a space where nearly the whole
// visible range sits above 0.9, which accepted the first on-screen sample as a
// hit. Two implementations of a subtle algorithm is one too many.
//
// The pyramid MUST be min-reduced (`hiz_min`). The shared `hiz` resource is
// max-reduced for occlusion culling — conservative farthest — and marching that
// makes rays tunnel straight through geometry.
//
// NDC depth is *linear in screen space* (projective z/w), so a ray is a straight
// line in (uv, depth01) space. Nothing in the loop needs perspective correction
// or linearization, which is what makes the cell-skipping valid.
//
// The texture is a parameter rather than a binding: group and binding stay the
// caller's business, same principle as the rest of the prelude.

/// Result of a Hi-Z march.
struct HelioHizHit {
    /// Hit position as (uv.xy, depth01). Only meaningful when `hit` is true.
    pos: vec3<f32>,
    /// False when the ray left the screen, exceeded the depth range, or ran out
    /// of iterations. Callers must treat a miss as "no reflection", never as a
    /// hit at the last position.
    hit: bool,
}

fn helio_hiz_level_size(hiz: texture_2d<f32>, level: i32) -> vec2<f32> {
    return vec2<f32>(max(vec2<u32>(1u), textureDimensions(hiz) >> vec2<u32>(u32(level))));
}

fn helio_hiz_cell(uv: vec2<f32>, size: vec2<f32>) -> vec2<f32> {
    return floor(uv * size);
}

fn helio_hiz_at_depth(o: vec3<f32>, d: vec3<f32>, z: f32) -> vec3<f32> {
    return o + d * ((z - o.z) / d.z);
}

fn helio_hiz_exit_cell(
    o: vec3<f32>,
    d: vec3<f32>,
    cell: vec2<f32>,
    size: vec2<f32>,
    cross_step: vec2<f32>,
    cross_offset: vec2<f32>,
) -> vec3<f32> {
    let boundary = (cell + cross_step) / size + cross_offset;
    let delta = (boundary - o.xy) / d.xy;
    return o + d * min(delta.x, delta.y);
}

/// March a ray through the min-depth pyramid.
///
/// `p0` and `p1` are the ray's start and end in (uv, depth01) — project them
/// with the caller's own view/projection before calling. `start_level` is the
/// mip to begin at (2 is a good default: coarse enough to skip empty space,
/// fine enough not to overshoot), `max_level` caps how far up the traversal may
/// climb, and `max_iter` bounds the loop.
fn helio_hiz_march(
    hiz: texture_2d<f32>,
    p0: vec3<f32>,
    p1: vec3<f32>,
    start_level: i32,
    max_level: i32,
    max_iter: u32,
) -> HelioHizHit {
    var out: HelioHizHit;
    out.pos = p0;
    out.hit = false;

    var d = p1 - p0;
    // A ray with no screen-space extent has no cells to walk.
    if abs(d.x) < 1e-7 && abs(d.y) < 1e-7 {
        return out;
    }
    d.x = select(d.x, 1e-7, abs(d.x) < 1e-7);
    d.y = select(d.y, 1e-7, abs(d.y) < 1e-7);

    let cross_step = vec2<f32>(select(0.0, 1.0, d.x >= 0.0), select(0.0, 1.0, d.y >= 0.0));
    let cross_offset = (cross_step * 2.0 - 1.0) * 1e-6;

    let ceiling = min(max_level, i32(textureNumLevels(hiz)) - 1);
    var level = min(start_level, ceiling);

    var ray = p0;
    {
        let size = helio_hiz_level_size(hiz, level);
        ray = helio_hiz_exit_cell(p0, d, helio_hiz_cell(p0.xy, size), size, cross_step, cross_offset);
    }

    var iter = 0u;
    var hit = false;

    while level >= 0 && iter < max_iter {
        iter += 1u;
        if any(ray.xy < vec2<f32>(0.0)) || any(ray.xy > vec2<f32>(1.0)) { break; }
        if ray.z > 1.0 { break; }

        let size = helio_hiz_level_size(hiz, level);
        let old_cell = helio_hiz_cell(ray.xy, size);
        let tile_min = textureLoad(hiz, vec2<i32>(old_cell), level).r;

        var next = ray;
        let in_front = ray.z < tile_min;
        if in_front && d.z > 0.0 {
            next = helio_hiz_at_depth(p0, d, tile_min);
        }

        let new_cell = helio_hiz_cell(next.xy, size);
        let skip_tile = in_front && d.z <= 0.0;

        if skip_tile || any(new_cell != old_cell) {
            next = helio_hiz_exit_cell(ray, d, old_cell, size, cross_step, cross_offset);
            level = min(ceiling, level + 1);
        } else {
            level -= 1;
            if level < 0 { hit = true; }
        }
        ray = next;
    }

    if hit && all(ray.xy >= vec2<f32>(0.0)) && all(ray.xy <= vec2<f32>(1.0)) {
        out.pos = ray;
        out.hit = true;
    }
    return out;
}
