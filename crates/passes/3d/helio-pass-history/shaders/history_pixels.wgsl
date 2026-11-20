//!use helio_prelude
//!use frame_history
// Where a pixel was (`helio_frame_api::history::PIXELS`) and this frame's
// guides for the next one, one thread a pixel at the internal resolution.
//
// For each pixel: its view depth, normal and roughness go to this frame's
// guides; its previous position is the velocity's where the surface moves by
// itself (bit 0 of the velocity's w), the reprojection's elsewhere; each of
// the 2x2 texels of the previous guides around that position holds the same
// surface when
//
// - its view depth is the one the surface had there, within what the
//   surface's own slope moves depth by over the tap's offset. The pixel's
//   neighbours, reprojected too, map a step of one current pixel along each
//   axis to a displacement on the previous raster and a change of previous
//   depth (`Axis`); the tap's offset, solved in those two steps, times the
//   larger of each axis's two one-pixel changes bounds what a plane's depth
//   does there (depth along a plane is monotone and convex in screen
//   position, so the change over part of a pixel on either side is at most
//   that part of the larger side's; one side alone counts twice). The
//   neighbours are the same surface's
//   where there are any (a neighbour whose normal turns by more than CREASE
//   is another surface meeting this one, a wall at the ground, whose depth
//   says nothing of this surface's slope), and the larger side is held to
//   twice the smaller (`axis_of`: a step, a silhouette before a parallel
//   surface, is not slope); plus the precision floor below; and
// - its normal faces the same side (n . n' > 0): depth alone cannot tell the
//   two sides of a surface thinner than the tolerance apart (a leaf, a thin
//   wall), whose normals oppose; a stricter test would reject a normal-mapped
//   surface's own detail, which differs between texels of one surface.
//
// A surface that moves by itself keeps its view depth only approximately
// (the velocity carries screen motion alone): it may change it by
// MOVING_DEPTH of itself a frame.

@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var normal_tex: texture_2d<f32>;
@group(0) @binding(2) var orm_tex: texture_2d<f32>;
@group(0) @binding(3) var velocity_tex: texture_2d<f32>;
@group(0) @binding(4) var<storage, read> cameras: array<Camera, 2>;
@group(0) @binding(5) var<uniform> history: HistoryReprojection;
@group(0) @binding(6) var previous_depth: texture_2d<f32>;
@group(0) @binding(7) var previous_normal: texture_2d<f32>;
@group(0) @binding(8) var out_depth: texture_storage_2d<r32float, write>;
@group(0) @binding(9) var out_normal: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var out_pixels: texture_storage_2d<rgba16float, write>;
// x: the previous guides hold this view's previous frame, y: the velocity is
// bound.
@group(0) @binding(11) var<uniform> info: vec4<u32>;

// The precision floor of the depth test, relative to view depth: an order
// above the f32 rounding of the depth buffer and of the frame-space position
// a drawer projects (2^-24 each, within the frame origin's cell) and of the
// reprojection's own f32 evaluation; 2^-12 is 0.24 mm at 1 m, 2.4 cm at 100 m.
const RELATIVE_DEPTH: f32 = 0.000244140625;
// Absolute floor: 2^-24 of the 4 km a frame-space position spans at most
// near the eye (the origin's 1024 m cell and the eye's offset in it), x4.
const ABSOLUTE_DEPTH: f32 = 0.001;
// Neighbours whose normals differ by more than 60 degrees are another
// surface for the slope: a crease (a wall on the ground is 90); a
// normal-mapped surface's own detail stays within it.
const CREASE: f32 = 0.5;
// A surface that moves by itself changes its view depth by at most this
// share a frame before its history is rejected: 20 m/s at 10 m and 60 Hz.
const MOVING_DEPTH: f32 = 0.03125;

// The workgroup's 8x8 pixels and a ring of one, each texel once: its
// previous position (pixels) and previous view depth (as a static surface),
// its view depth (0: sky or past the image) and its normal. Every pixel's
// four neighbours read from here rather than each loading and reprojecting
// them again.
const TILE: i32 = 10;
var<workgroup> tile_previous: array<vec4<f32>, 100>;
var<workgroup> tile_normal: array<vec3<f32>, 100>;

fn tile_at(q: vec2<i32>, origin: vec2<i32>) -> u32 {
    let l = q - origin;
    return u32(l.y * TILE + l.x);
}

// One axis of the local map from the current pixel grid to the previous
// frame: the previous-raster displacement of one current pixel's step and
// the bound on the previous depth's change over it.
struct Axis {
    step: vec2<f32>,
    change: f32,
}

// Neighbour `q` as seen from the pixel (previous position `at`, previous
// depth `expected`): its previous displacement and depth change, w 1 when it
// counts (a surface, and with `same` this surface by its normal). A surface
// that moves by itself has no reprojection of its neighbours: its current
// depth change and no displacement (`axis_of` then takes the step as one
// previous pixel).
fn neighbour(q: vec2<i32>, origin: vec2<i32>, at: vec2<f32>, expected: f32, moves: bool, n: vec3<f32>, same: bool) -> vec4<f32> {
    let i = tile_at(q, origin);
    let r = tile_previous[i];
    if r.w <= 0.0 || (same && dot(n, tile_normal[i]) < CREASE) {
        return vec4<f32>(0.0);
    }
    if moves {
        return vec4<f32>(0.0, 0.0, abs(r.w - expected), 1.0);
    }
    return vec4<f32>(r.xy - at, abs(r.z - expected), 1.0);
}

// The axis `axis` of the pixel `p` (`Axis`): the same surface's neighbours
// where there are any, else any. One side alone stands for both at twice its
// change: along a plane, d = k / u in screen position u from its horizon,
// and the slope at the pixel exceeds the chord's to a neighbour one step
// away by a factor of 1 + step / u, under 2 unless the horizon is within the
// step.
fn axis_of(p: vec2<i32>, axis: vec2<i32>, origin: vec2<i32>, at: vec2<f32>, expected: f32, moves: bool, n: vec3<f32>) -> Axis {
    var minus = neighbour(p - axis, origin, at, expected, moves, n, true);
    var plus = neighbour(p + axis, origin, at, expected, moves, n, true);
    if minus.w + plus.w == 0.0 {
        minus = neighbour(p - axis, origin, at, expected, moves, n, false);
        plus = neighbour(p + axis, origin, at, expected, moves, n, false);
    }
    var a: Axis;
    if minus.w + plus.w == 0.0 {
        a.step = vec2<f32>(axis);
        return a;
    }
    if minus.w == 0.0 {
        minus = vec4<f32>(-plus.xy, 2.0 * plus.z, plus.w);
        plus.z *= 2.0;
    }
    if plus.w == 0.0 {
        plus = vec4<f32>(-minus.xy, 2.0 * minus.z, minus.w);
        minus.z *= 2.0;
    }
    a.step = 0.5 * (plus.xy - minus.xy);
    if moves {
        a.step = vec2<f32>(axis);
    }
    a.change = min(max(minus.z, plus.z), 2.0 * min(minus.z, plus.z));
    return a;
}

@compute @workgroup_size(8, 8)
fn history_pixels(@builtin(global_invocation_id) id: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>,
                  @builtin(local_invocation_index) thread: u32) {
    let size = vec2<i32>(textureDimensions(depth_tex));
    let origin = vec2<i32>(group.xy) * 8 - 1;
    for (var i = thread; i < 100u; i += 64u) {
        let q = origin + vec2<i32>(i32(i % 10u), i32(i / 10u));
        var r = vec4<f32>(0.0);
        var m = vec3<f32>(0.0);
        if all(q >= vec2<i32>(0)) && all(q < size) {
            let d = textureLoad(depth_tex, q, 0);
            if !helio_is_sky(d) {
                let h = history_reproject(history, (vec2<f32>(q) + 0.5) / vec2<f32>(size), d);
                r = vec4<f32>(h.xy * vec2<f32>(size), h.w, helio_view_depth(cameras[0].proj, d));
                m = helio_gbuffer_normal(textureLoad(normal_tex, q, 0).xyz);
            }
        }
        tile_previous[i] = r;
        tile_normal[i] = m;
    }
    workgroupBarrier();
    let p = vec2<i32>(id.xy);
    if any(p >= size) {
        return;
    }
    let depth = textureLoad(depth_tex, p, 0);
    let sky = helio_is_sky(depth);
    let vd = select(helio_view_depth(cameras[0].proj, depth), 0.0, sky);
    var n = vec3<f32>(0.0);
    var roughness = 0.0;
    if !sky {
        n = helio_gbuffer_normal(textureLoad(normal_tex, p, 0).xyz);
        roughness = textureLoad(orm_tex, p, 0).g;
    }
    textureStore(out_depth, p, vec4<f32>(vd, 0.0, 0.0, 0.0));
    textureStore(out_normal, p, vec4<f32>(n, roughness));

    let centre = vec2<f32>(p) + 0.5;
    let uv = centre / vec2<f32>(size);
    let r = history_reproject(history, uv, depth);
    var previous = r.xy * vec2<f32>(size);
    var expected = r.w;
    var flags = 0u;
    var in_front = r.z > 0.0;
    if info.y != 0u && !sky {
        let m = textureLoad(velocity_tex, p, 0);
        // A non-finite vector (a writer's division by a zero w) is none.
        let finite = all((bitcast<vec2<u32>>(m.xy) & vec2<u32>(0x7f800000u)) != vec2<u32>(0x7f800000u));
        if finite && (u32(m.w) & 1u) != 0u {
            previous = centre - m.xy;
            expected = vd;
            flags |= HISTORY_PIXEL_MOVES;
            in_front = true;
        }
    }
    var taps = 0u;
    if info.x != 0u && history.info.x != 0u && in_front {
        flags |= HISTORY_PIXEL_HAS_PREVIOUS;
        var floor_tolerance = 0.0;
        var ax: Axis;
        var ay: Axis;
        // The previous raster in steps of the current pixels (the axes'
        // inverse); none where the map is degenerate: whole previous pixels.
        var inverse = mat2x2<f32>(1.0, 0.0, 0.0, 1.0);
        if !sky {
            let moves = (flags & HISTORY_PIXEL_MOVES) != 0u;
            ax = axis_of(p, vec2<i32>(1, 0), origin, previous, expected, moves, n);
            ay = axis_of(p, vec2<i32>(0, 1), origin, previous, expected, moves, n);
            let j = mat2x2<f32>(ax.step, ay.step);
            let det = determinant(j);
            if abs(det) > 1e-6 {
                inverse = mat2x2<f32>(j[1][1], -j[0][1], -j[1][0], j[0][0]) * (1.0 / det);
            }
            floor_tolerance = ABSOLUTE_DEPTH + RELATIVE_DEPTH * expected;
            if moves {
                floor_tolerance += MOVING_DEPTH * vd;
            }
        }
        let base = vec2<i32>(floor(previous - 0.5));
        for (var i = 0u; i < 4u; i++) {
            let t = base + vec2<i32>(i32(i & 1u), i32(i >> 1u));
            if any(t < vec2<i32>(0)) || any(t >= size) {
                continue;
            }
            let d = textureLoad(previous_depth, t, 0).r;
            // The tap's offset from the position in current pixel steps.
            let o = inverse * (vec2<f32>(t) + 0.5 - previous);
            let tolerance = floor_tolerance + abs(o.x) * ax.change + abs(o.y) * ay.change;
            var same = false;
            if sky {
                // Sky continues where the previous frame saw sky.
                same = d <= 0.0;
            } else if d > 0.0 && abs(d - expected) <= tolerance {
                same = dot(n, textureLoad(previous_normal, t, 0).xyz) > 0.0;
            }
            taps |= select(0u, 1u << i, same);
        }
    }
    textureStore(out_pixels, p, vec4<f32>(previous - centre, f32(taps), f32(flags)));
}
