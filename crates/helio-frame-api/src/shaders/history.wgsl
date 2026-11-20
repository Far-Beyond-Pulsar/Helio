// A pixel's history (`helio_frame_api::history::SNIPPET`, `//!use
// frame_history`; needs the prelude): the view's reprojection and the decode
// of "where a pixel was" (`helio_frame_api::history::PIXELS`). Positions are
// on the rasters as drawn, each frame with its own jitter.

// `helio_frame_api::history::HistoryReprojection`: applied to this frame's
// clip position c = (ndc, raw depth, 1), the previous frame's homogeneous
// clip position of the point is clip_to_previous_clip * c, the point's w is
// dot(clip_to_world_w, c), the previous clip position of this frame's eye is
// eye_to_previous_clip, and the previous view's z times w is
// dot(clip_to_previous_view_z, c). Composed on the CPU in f64.
struct HistoryReprojection {
    // x: 1 when the view has a previous frame, y: frames it has rendered.
    info: vec4<u32>,
    clip_to_previous_clip: mat4x4<f32>,
    clip_to_world_w: vec4<f32>,
    eye_to_previous_clip: vec4<f32>,
    clip_to_previous_view_z: vec4<f32>,
}

// Where a point of this frame's raster (`raster_uv`, jittered, y down; raw
// `depth`) was on the previous frame's raster: xy its uv there (jittered),
// z positive where it was in front of the previous camera, w its previous
// view depth (0 for sky). Sky reprojects as a direction: the homogeneous
// point minus w times the eye, exact for any w, the w = 0 of depth 0
// (infinity) included.
fn history_reproject(h: HistoryReprojection, raster_uv: vec2<f32>, depth: f32) -> vec4<f32> {
    let c = vec4<f32>(helio_uv_to_ndc(raster_uv), depth, 1.0);
    let w = dot(h.clip_to_world_w, c);
    let sky = helio_is_sky(depth);
    let previous = h.clip_to_previous_clip * c - select(0.0, w, sky) * h.eye_to_previous_clip;
    let uv = helio_ndc_to_uv(previous.xy / previous.w);
    // A point's clip w is the homogeneous one over w (same sign as times w).
    let front = select(previous.w * w, previous.w, sky);
    return vec4<f32>(uv, front, select(-dot(h.clip_to_previous_view_z, c) / w, 0.0, sky));
}

// `helio_frame_api::history::pixel_flags`.
const HISTORY_PIXEL_MOVES: u32 = 1u;
const HISTORY_PIXEL_HAS_PREVIOUS: u32 = 2u;

struct HistoryPixel {
    // The surface's position on the previous frame's raster, in pixels
    // (texel centres at + 0.5).
    previous: vec2<f32>,
    // Bit i: the texel floor(previous - 0.5) + (i & 1, i >> 1) held the
    // same surface.
    taps: u32,
    // HISTORY_PIXEL_*.
    flags: u32,
}

// The `PIXELS` texel `texel` of pixel `pixel`, decoded.
fn history_pixel(texel: vec4<f32>, pixel: vec2<u32>) -> HistoryPixel {
    return HistoryPixel(vec2<f32>(pixel) + 0.5 + texel.xy, u32(texel.z + 0.5), u32(texel.w + 0.5));
}

// The bilinear footprint of a history read: the first texel of the 2x2 and
// each tap's weight, the bilinear weight where the tap held the same surface
// and 0 elsewhere. Not renormalised: `history_taps_weight` is how much of
// the footprint is the same surface, and a reader divides by it (or keeps
// less history where it is small).
struct HistoryTaps {
    base: vec2<i32>,
    weights: vec4<f32>,
}

fn history_taps(p: HistoryPixel) -> HistoryTaps {
    let at = p.previous - 0.5;
    let base = floor(at);
    let f = at - base;
    let bilinear = vec4<f32>((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
    let valid = vec4<f32>(vec4<u32>(p.taps, p.taps >> 1u, p.taps >> 2u, p.taps >> 3u) & vec4<u32>(1u));
    return HistoryTaps(vec2<i32>(base), bilinear * valid);
}

fn history_taps_weight(t: HistoryTaps) -> f32 {
    return dot(t.weights, vec4<f32>(1.0));
}

// Tap i's texel.
fn history_tap(t: HistoryTaps, i: u32) -> vec2<i32> {
    return t.base + vec2<i32>(i32(i & 1u), i32(i >> 1u));
}
