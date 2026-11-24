//!use helio_prelude
//!use frame_history
//!use frame_surface_stops
//!use frame_denoise_irradiance
// The denoiser's stages for an irradiance signal (lib.rs documents them and
// where each constant comes from). Light is spread where the information is,
// per traced sample; only what needs a pixel's own geometry runs per pixel:
//
// 1. `temporal` (a pixel): this frame's samples at the pixel (the traced
//    pixels of the nearest 2x2 blocks, a tent of the block over their
//    distance times the surface and occluder stops) folded into the pixel's history,
//    read where the frame's history says the pixel was (the bilinear taps
//    that held the same surface). The history's length falls by the
//    producer's confidence and the share of the footprint that held the
//    surface, and is capped.
// 2. `records` (a block, for blocks of 4 and 16): the block's record, the
//    temporal result at its representative pixel (its centre). A signal
//    traced at every pixel has its pixels as records.
// 3. `spatial`, twice (a block): an a-trous filter of the linear B-spline's
//    3x3 over the records, at steps of s and 2s blocks, s set by the record's
//    history length (4 / sqrt(length): wide where history is short, one
//    block once it holds 16 frames), stopped by geometry alone (distance
//    against the irradiance's own scale, the footprint's pooled hit
//    distance; normal; plane; the frame's depth between the two).
// 4. `reconstruct` (a pixel): the filtered records of the pixel's nearest
//    2x2 blocks (tent times geometry, at the pixel's own normal), combined
//    with the pixel's own temporal result by the samples each rests on. Its
//    result is the signal's output and the history the next frame reads:
//    the spatial stage is recurrent (its work accumulates over frames, as
//    ReBLUR's post-blur history does), while the luminance's moments stay
//    the raw samples' (the variance never sees the filter).
//
// The mean is kept: no weight that decides how much a sample counts depends
// on the light it carries (only on geometry, a hit distance pooled over
// the footprint, and sample counts), so every stage is a normalised linear
// filter whose expectation is its input's; chroma is averaged weighted by luminance (a ratio's mean
// weighted by the luminance it scales is the ratio of the means).
//
// Values are the linear irradiance a + b . n (`denoise_irradiance_linear`),
// its chroma as ratios to luminance, 1 / hit distance; a pixel's state also
// a history length and the first two moments of the luminance at the
// pixel's normal (f32: the variance is their difference, which half floats
// cannot resolve). Layouts (the history's two first are the outputs):
//   A Rgba16Float (a, Co / Y, Cg / Y, standard error; -1: no light yet)
//   B Rgba16Float (b, 1 / hit distance)
//   M Rgba32Float (length, mean, mean square)    pixels only; the temporal
//                                                result's is the next
//                                                frame's history's
//   N Rg32Float (length, samples)                 records only: the history
//                                                length that sets the step,
//                                                and how many samples' worth
//                                                the filtered record holds

struct DenoiseParams {
    // The view's inverse projection (view space from clip; no frame-space
    // translation, so positions keep their precision far from the origin).
    inv_proj: mat4x4<f32>,
    // The view's rotation (frame space to view space; no translation): the
    // guides' normals are frame space, their positions view space.
    view_rotation: mat4x4<f32>,
    // The projection's P[2][2], P[2][3], P[3][2], P[3][3]: raw depth from
    // view depth.
    proj_zw: vec4<f32>,
    size: vec2<u32>,
    traced: vec2<u32>,
    side: u32,
    frame: u32,
    // Bit 0: the signal has a confidence; bit 1: the frame's history is bound.
    flags: u32,
    max_history: f32,
    // A pixel's width a metre of view depth.
    pixel_scale: f32,
    _pad: vec3<u32>,
}

// The frame's guides at a pixel (`helio_frame_api::history::DEPTH`, `NORMAL`).
@group(0) @binding(0) var guide_depth: texture_2d<f32>;
@group(0) @binding(1) var guide_normal: texture_2d<f32>;
@group(0) @binding(2) var history_pixels: texture_2d<f32>;
@group(0) @binding(3) var<uniform> params: DenoiseParams;

// temporal
@group(1) @binding(0) var signal0: texture_2d<f32>;
@group(1) @binding(1) var signal1: texture_2d<f32>;
@group(1) @binding(2) var hist_a_in: texture_2d<f32>;
@group(1) @binding(3) var hist_b_in: texture_2d<f32>;
@group(1) @binding(4) var hist_m_in: texture_2d<f32>;
@group(1) @binding(5) var temp_a_out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(6) var temp_b_out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(7) var temp_m_out: texture_storage_2d<rgba32float, write>;
// records
@group(1) @binding(8) var rec_temp_a: texture_2d<f32>;
@group(1) @binding(9) var rec_temp_b: texture_2d<f32>;
@group(1) @binding(10) var rec_temp_m: texture_2d<f32>;
@group(1) @binding(11) var rec_a_out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(12) var rec_b_out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(13) var rec_n_out: texture_storage_2d<rg32float, write>;
// spatial (the length is the x of `in_n`: a record's N or a pixel's M; the
// samples a record holds its y after the first iteration, its length before)
@group(1) @binding(14) var in_a: texture_2d<f32>;
@group(1) @binding(15) var in_b: texture_2d<f32>;
@group(1) @binding(16) var in_n: texture_2d<f32>;
@group(1) @binding(17) var out_a: texture_storage_2d<rgba16float, write>;
@group(1) @binding(18) var out_b: texture_storage_2d<rgba16float, write>;
@group(1) @binding(19) var out_n: texture_storage_2d<rg32float, write>;
// x: the iteration.
@group(1) @binding(20) var<uniform> iteration: vec4<u32>;
// reconstruct
@group(1) @binding(21) var own_a: texture_2d<f32>;
@group(1) @binding(22) var own_b: texture_2d<f32>;
@group(1) @binding(23) var own_m: texture_2d<f32>;
@group(1) @binding(24) var rec_a: texture_2d<f32>;
@group(1) @binding(25) var rec_b: texture_2d<f32>;
@group(1) @binding(26) var hist_a_out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(27) var hist_b_out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(28) var rec_n: texture_2d<f32>;

// An irradiance record is valid over a radius of its harmonic mean hit
// distance (Ward 1988: the error of using it at distance d is d / R), never
// below the diagonal of the pixel that holds it.
const RADIUS_MIN_PIXELS: f32 = 1.41421356;
// The spatial stage's step: a record's estimate rests on `length` frames;
// the filter's footprint of (2 step + 1)^2 records adds about step^2 of
// them, so a step of 4 / sqrt(length) keeps about 16 frames' worth behind
// every estimate (one block from 16 frames on, four at one).
const SPATIAL_FRAMES: f32 = 16.0;
// Chroma ratios of real colours: Co / Y and Cg / Y stay within +-2 (pure
// red: Co / Y = 2).
const CHROMA_LIMIT: f32 = 2.0;
// A hit distance a half float cannot hold is no hit.
const NO_HIT: f32 = 65504.0;

struct Guide {
    // View-space position (metres), frame-space normal (what the
    // irradiance's b is dotted with), view-space normal (what positions
    // are), view depth.
    x: vec3<f32>,
    n: vec3<f32>,
    nv: vec3<f32>,
    depth: f32,
    sky: bool,
}

// The view-space point at pixel `q` (clamped into the image) of view depth
// `vd` (positive).
fn view_point(q: vec2<i32>, vd: f32) -> vec3<f32> {
    // The raw depth of view depth vd (`helio_view_depth` inverted), then
    // the view-space point through the inverse projection.
    let z = params.proj_zw;
    let d = (z.z - vd * z.x) / (z.w - vd * z.y);
    let uv = (vec2<f32>(q) + 0.5) / vec2<f32>(params.size);
    let v = params.inv_proj * vec4<f32>(helio_uv_to_ndc(uv), d, 1.0);
    return v.xyz / v.w;
}

fn guide(p: vec2<i32>) -> Guide {
    let size = vec2<i32>(params.size);
    let q = clamp(p, vec2<i32>(0), size - 1);
    let vd = textureLoad(guide_depth, q, 0).r;
    var g: Guide;
    g.sky = vd <= 0.0;
    if g.sky {
        return g;
    }
    g.x = view_point(q, vd);
    g.depth = vd;
    g.n = normalize(textureLoad(guide_normal, q, 0).xyz);
    g.nv = (params.view_rotation * vec4<f32>(g.n, 0.0)).xyz;
    return g;
}

// The representative pixel of block `b` (its centre, inside the image).
fn representative(b: vec2<i32>) -> vec2<i32> {
    let side = i32(params.side);
    return min(b * side + side / 2, vec2<i32>(params.size) - 1);
}

fn footprint(g: Guide) -> f32 {
    return params.pixel_scale * g.depth;
}

// The radius over which the record at `g` (its 1 / hit distance) holds.
fn record_radius(g: Guide, inverse_hit: f32) -> f32 {
    return max(1.0 / max(inverse_hit, 1e-12), RADIUS_MIN_PIXELS * footprint(g));
}

// Whether `q` lies on `p`'s surface: the frame's surface stops, in view
// space.
fn surface_weight(p: Guide, q: Guide) -> f32 {
    return surface_stops_weight(p.x, p.nv, q.x, q.nv, footprint(p));
}

// How well the irradiance of a record at `q`, valid over `radius`, serves `p`.
fn geometry_weight(p: Guide, q: Guide, radius: f32) -> f32 {
    return saturate(1.0 - distance(p.x, q.x) / radius) * surface_weight(p, q);
}

// The view-space point at pixel `q` and a pixel's footprint there (0: the
// sky), as the surface stops' walk reads the frame's depth
// (`surface_stops_seen`).
fn surface_stops_point(q: vec2<i32>) -> vec4<f32> {
    let vd = textureLoad(guide_depth, q, 0).r;
    if vd <= 0.0 { return vec4<f32>(0.0); }
    return vec4<f32>(view_point(q, vd), params.pixel_scale * vd);
}

// Whether the frame's depth shows nothing between pixels `a` and `b` (their
// guides `ga`, `gb`) that stands out of both receivers' surfaces
// (`surface_stops_seen`).
fn seen(a: vec2<i32>, b: vec2<i32>, ga: Guide, gb: Guide) -> bool {
    return surface_stops_seen(a, b, ga.x, ga.nv, gb.x, gb.nv);
}

fn inverse_hit(hit: f32) -> f32 {
    let finite = (bitcast<u32>(hit) & 0x7f800000u) != 0x7f800000u;
    return select(0.0, 1.0 / max(hit, 1e-6), finite && hit < NO_HIT);
}

fn finite4(v: vec4<f32>) -> vec4<f32> {
    let exponent = bitcast<vec4<u32>>(v) & vec4<u32>(0x7f800000u);
    return select(v, vec4<f32>(0.0), exponent == vec4<u32>(0x7f800000u));
}

// A hash of `v` (PCG).
fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

// `v` as a half float, rounded up with probability `u` of its distance from
// the half below: the stored value's mean is `v` whatever rounding the
// hardware's conversion uses (it receives a value a half holds exactly).
// What the next frame reads back is blended by 1 / length: a conversion that
// truncates (several GPUs' storage writes) loses half a step a frame, which
// the running mean amplifies by its length (1% at 32 frames).
fn half_unbiased(v: f32, u: f32) -> f32 {
    let a = min(abs(v), 65504.0);
    var lo: f32;
    var hi: f32;
    if a < 6.103515625e-5 {
        // Subnormal halves: steps of 2^-24.
        lo = floor(a * 16777216.0) / 16777216.0;
        hi = lo + 1.0 / 16777216.0;
    } else {
        let bits = bitcast<u32>(a) & 0xffffe000u;
        lo = bitcast<f32>(bits);
        hi = bitcast<f32>(bits + 0x2000u);
    }
    let r = select(lo, hi, u * (hi - lo) < a - lo);
    return select(r, -r, v < 0.0);
}

// A texel's four channels rounded so, from one hash: a byte of it a channel
// (a rounding probability resolved to 1/256 of a step: a bias under 2^-19
// of the value).
fn half4(v: vec4<f32>, p: vec2<i32>, stream: u32) -> vec4<f32> {
    let h = pcg((u32(p.y) * params.size.x + u32(p.x)) ^ pcg(params.frame * 4u + stream));
    let u = (vec4<f32>(vec4<u32>(h, h >> 8u, h >> 16u, h >> 24u) & vec4<u32>(255u)) + 0.5) / 256.0;
    return vec4<f32>(half_unbiased(v.x, u.x), half_unbiased(v.y, u.y), half_unbiased(v.z, u.z),
        half_unbiased(v.w, u.w));
}

// The luminance of linear irradiance (a, b) at normal `n`, as a weight
// (never below 0).
fn luma_at(a: f32, b: vec3<f32>, n: vec3<f32>) -> f32 {
    return max(a + dot(b, n), 0.0);
}

// Chroma ratios from chroma summed times luminance and that luminance's sum
// (none where no light: its colour does not matter).
fn chroma_ratio(chroma: vec2<f32>, luma: f32) -> vec2<f32> {
    return select(vec2<f32>(0.0), chroma / luma, luma > 0.0);
}

// A pixel's state while it is blended: (a, Co, Cg, 1 / hit) with the chroma
// times the luminance at the pixel's normal (so blending by any weight
// averages the ratios by luminance), (b), (mean, mean square).
struct State {
    a: vec4<f32>,
    b: vec3<f32>,
    m: vec2<f32>,
}

fn state_scaled(s: State, w: f32) -> State {
    return State(s.a * w, s.b * w, s.m * w);
}

fn state_add(x: State, y: State) -> State {
    return State(x.a + y.a, x.b + y.b, x.m + y.m);
}

fn load_history(t: vec2<i32>, n: vec3<f32>) -> State {
    let a = finite4(textureLoad(hist_a_in, t, 0));
    let b = finite4(textureLoad(hist_b_in, t, 0));
    let y = luma_at(a.x, b.xyz, n);
    return State(vec4<f32>(a.x, a.yz * y, b.w), b.xyz, textureLoad(hist_m_in, t, 0).yz);
}

// The standard error of a mean of `length` samples of moments `m`; -1 for
// no light.
fn standard_error(m: vec2<f32>, length: f32) -> f32 {
    return select(-1.0, sqrt(max(m.y - m.x * m.x, 0.0) / max(length, 1.0)), length > 0.0);
}

fn store_temp(p: vec2<i32>, s: State, length: f32, n: vec3<f32>) {
    let chroma = chroma_ratio(s.a.yz, luma_at(s.a.x, s.b, n));
    textureStore(temp_a_out, p, half4(vec4<f32>(s.a.x, chroma, standard_error(s.m, length)), p, 0u));
    textureStore(temp_b_out, p, half4(vec4<f32>(s.b, s.a.w), p, 1u));
    textureStore(temp_m_out, p, vec4<f32>(length, s.m, 0.0));
}

@compute @workgroup_size(8, 8)
fn temporal(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = vec2<i32>(id.xy);
    if any(id.xy >= params.size) {
        return;
    }
    var empty: State;
    let g = guide(p);
    if g.sky {
        store_temp(p, empty, 0.0, vec3<f32>(0.0));
        return;
    }
    // This frame's samples at the pixel.
    let side = params.side;
    let fside = f32(side);
    let first = vec2<i32>(floor((vec2<f32>(p) + 0.5) / fside - 0.5));
    var fresh: State;
    var weight = 0.0;
    var confidence = 0.0;
    for (var i = 0u; i < 4u; i++) {
        // Blocks past the image's edge hold no sample (their share of the
        // tent is lost: an edge pixel's sample counts a little less).
        let block = first + vec2<i32>(i32(i & 1u), i32(i >> 1u));
        if any(block < vec2<i32>(0)) || any(block >= vec2<i32>(params.traced)) {
            continue;
        }
        let t = vec2<i32>(denoise_traced_pixel(vec2<u32>(block), params.frame, side, params.size));
        let tent = saturate(1.0 - abs(vec2<f32>(p - t)) / fside);
        let w_space = tent.x * tent.y;
        if w_space <= 0.0 {
            continue;
        }
        let gt = guide(t);
        if gt.sky {
            continue;
        }
        let s0 = finite4(textureLoad(signal0, block, 0));
        let s1 = finite4(textureLoad(signal1, block, 0));
        // Within the block's tent by its surface alone: a sample's own ray
        // length goes with the light it found (a near source is short), so
        // a radius from it would weigh the sample by its light; contact
        // detail finer than a block is the producer's (a gather's near
        // field), and the records' radius stops the spatial stage.
        let w = w_space * surface_weight(g, gt);
        if w <= 0.0 || !seen(p, t, g, gt) {
            continue;
        }
        let l = denoise_irradiance_linear(s0, s1, gt.n);
        let y = l.x + dot(l.yzw, g.n);
        // Chroma as ratios to the sample's own luminance: exact for each
        // sample of one colour whatever normal it was traced at; carried
        // times the luminance it scales.
        let chroma = clamp(s0.yz / max(s0.x, 1e-12), vec2<f32>(-CHROMA_LIMIT), vec2<f32>(CHROMA_LIMIT));
        fresh = state_add(fresh, State(vec4<f32>(l.x, chroma * max(y, 0.0), inverse_hit(s0.w)) * w, l.yzw * w, vec2<f32>(y, y * y) * w));
        confidence += w * select(1.0, saturate(s1.w), (params.flags & 1u) != 0u);
        weight += w;
    }
    // The history where the pixel was.
    var history: State;
    var length = 0.0;
    if (params.flags & 2u) != 0u {
        let hp = history_pixel(textureLoad(history_pixels, p, 0), id.xy);
        let taps = history_taps(hp);
        let footprint = history_taps_weight(taps);
        if footprint > 1e-3 {
            var held = 0.0;
            for (var i = 0u; i < 4u; i++) {
                let w = taps.weights[i];
                if w > 0.0 {
                    let t = history_tap(taps, i);
                    history = state_add(history, state_scaled(load_history(t, g.n), w / footprint));
                    held += textureLoad(hist_m_in, t, 0).x * w / footprint;
                }
            }
            // A footprint the surface held in part is a weaker history.
            length = held * footprint;
        }
    }
    if weight > 0.0 {
        fresh = state_scaled(fresh, 1.0 / weight);
        length *= confidence / weight;
    }
    let total = min(length + weight, params.max_history);
    if total <= 0.0 {
        store_temp(p, empty, 0.0, g.n);
        return;
    }
    let alpha = select(0.0, weight / total, weight > 0.0);
    store_temp(p, State(mix(history.a, fresh.a, alpha), mix(history.b, fresh.b, alpha), mix(history.m, fresh.m, alpha)), total, g.n);
}

@compute @workgroup_size(8, 8)
fn records(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= params.traced) {
        return;
    }
    let b = vec2<i32>(id.xy);
    let r = representative(b);
    textureStore(rec_a_out, b, textureLoad(rec_temp_a, r, 0));
    textureStore(rec_b_out, b, textureLoad(rec_temp_b, r, 0));
    let length = textureLoad(rec_temp_m, r, 0).x;
    textureStore(rec_n_out, b, vec4<f32>(length, length, 0.0, 0.0));
}

// The linear B-spline's weight at tap `k` from the centre: 1/2, 1/4.
fn b1(k: i32) -> f32 {
    return select(0.25, 0.5, k == 0);
}

@compute @workgroup_size(8, 8)
fn spatial(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= params.traced) {
        return;
    }
    let b = vec2<i32>(id.xy);
    let rb = representative(b);
    let g = guide(rb);
    let length = textureLoad(in_n, b, 0).x;
    if g.sky {
        textureStore(out_a, b, vec4<f32>(0.0, 0.0, 0.0, -1.0));
        textureStore(out_b, b, vec4<f32>(0.0));
        textureStore(out_n, b, vec4<f32>(length, 0.0, 0.0, 0.0));
        return;
    }
    let s = clamp(round(sqrt(SPATIAL_FRAMES / max(length, 1.0))), 1.0, 4.0);
    let step = i32(s) << iteration.x;
    // One radius for every tap: the harmonic mean hit distance of the
    // footprint's records, each counted alike (a record's own ray lengths
    // go with its light, so a radius of its own would weigh it by its
    // light; pooled over the footprint, one record moves it by a ninth).
    var pooled = 0.0;
    var counted = 0.0;
    for (var j = 0; j < 9; j++) {
        let q = b + vec2<i32>(j % 3 - 1, j / 3 - 1) * step;
        if any(q < vec2<i32>(0)) || any(q >= vec2<i32>(params.traced)) || textureLoad(in_a, q, 0).w < 0.0 {
            continue;
        }
        pooled += textureLoad(in_b, q, 0).w;
        counted += 1.0;
    }
    let radius = record_radius(g, pooled / max(counted, 1.0));
    var sum_a = vec4<f32>(0.0);
    var sum_b = vec4<f32>(0.0);
    var sum_luma = 0.0;
    var sum_variance = 0.0;
    var sum_spread = 0.0;
    var total = 0.0;
    for (var j = 0; j < 9; j++) {
        let o = vec2<i32>(j % 3 - 1, j / 3 - 1);
        let q = b + o * step;
        if any(q < vec2<i32>(0)) || any(q >= vec2<i32>(params.traced)) {
            continue;
        }
        let fa = textureLoad(in_a, q, 0);
        if fa.w < 0.0 {
            continue;
        }
        let fb = textureLoad(in_b, q, 0);
        var w = b1(abs(o.x)) * b1(abs(o.y));
        if any(o != vec2<i32>(0)) {
            let rq = representative(q);
            let gq = guide(rq);
            if gq.sky {
                continue;
            }
            // The neighbour's record held over the footprint's radius, and
            // seen from ours.
            w *= geometry_weight(g, gq, radius);
            if w <= 0.0 || !seen(rb, rq, g, gq) {
                continue;
            }
        }
        // The samples behind the neighbour's estimate: its history's length
        // before the first iteration, what that iteration left after.
        let nq = select(textureLoad(in_n, q, 0).x, textureLoad(in_n, q, 0).y, iteration.x > 0u);
        let y = luma_at(fa.x, fb.xyz, g.n);
        sum_a += w * vec4<f32>(fa.x, fa.yz * y, 0.0);
        sum_luma += w * y;
        sum_b += w * fb;
        sum_variance += w * w * fa.w * fa.w;
        sum_spread += w * w / max(nq, 1e-6);
        total += w;
    }
    if total <= 0.0 {
        textureStore(out_a, b, vec4<f32>(0.0, 0.0, 0.0, -1.0));
        textureStore(out_b, b, vec4<f32>(0.0));
        textureStore(out_n, b, vec4<f32>(length, 0.0, 0.0, 0.0));
        return;
    }
    let inv = 1.0 / total;
    // Rounded without bias: the records reach the history (`reconstruct`).
    let stream = 4u + 2u * iteration.x;
    let chroma = chroma_ratio(sum_a.yz, sum_luma);
    textureStore(out_a, b, half4(vec4<f32>(sum_a.x * inv, chroma, sqrt(sum_variance) * inv), b, stream));
    textureStore(out_b, b, half4(sum_b * inv, b, stream + 1u));
    // A weighted mean of independent estimates of n_q samples each holds
    // (sum w)^2 / sum (w^2 / n_q) samples' worth.
    textureStore(out_n, b, vec4<f32>(length, total * total / sum_spread, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn reconstruct(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = vec2<i32>(id.xy);
    if any(id.xy >= params.size) {
        return;
    }
    // The moments are the temporal result's: its M is the next frame's
    // history M (the two swap, nothing is copied).
    let m = textureLoad(own_m, p, 0);
    let g = guide(p);
    if g.sky {
        textureStore(hist_a_out, p, vec4<f32>(0.0, 0.0, 0.0, -1.0));
        textureStore(hist_b_out, p, vec4<f32>(0.0));
        return;
    }
    let oa = textureLoad(own_a, p, 0);
    let ob = textureLoad(own_b, p, 0);
    // The filtered records of the nearest 2x2 blocks (a wider 4x4 where
    // none of them holds this surface: a feature narrower than a block).
    let side = f32(params.side);
    let first = vec2<i32>(floor((vec2<f32>(p) + 0.5) / side - 0.5));
    var sa = vec4<f32>(0.0);
    var sb = vec4<f32>(0.0);
    var luma = 0.0;
    var error = 0.0;
    var spread = 0.0;
    var total = 0.0;
    for (var ring = 0; ring < 2 && total <= 0.0; ring++) {
        let span = 2 + 2 * ring;
        for (var j = 0; j < span * span; j++) {
            let o = vec2<i32>(j % span, j / span) - ring;
            if ring == 1 && all(o >= vec2<i32>(0)) && all(o <= vec2<i32>(1)) {
                continue;
            }
            let q = first + o;
            if any(q < vec2<i32>(0)) || any(q >= vec2<i32>(params.traced)) {
                continue;
            }
            // A tent of the block where it reaches (the 2x2), the geometry
            // alone past it.
            let r = representative(q);
            let tent = saturate(1.0 - abs(vec2<f32>(p - r)) / side);
            var w = select(tent.x * tent.y, 1.0, ring == 1);
            if w <= 0.0 {
                continue;
            }
            let ra = textureLoad(rec_a, q, 0);
            if ra.w < 0.0 {
                continue;
            }
            let rb = textureLoad(rec_b, q, 0);
            let gq = guide(r);
            if gq.sky {
                continue;
            }
            w *= geometry_weight(g, gq, record_radius(gq, rb.w));
            if w <= 0.0 || !seen(p, r, g, gq) {
                continue;
            }
            let y = luma_at(ra.x, rb.xyz, g.n);
            sa += w * vec4<f32>(ra.x, ra.yz * y, 0.0);
            luma += w * y;
            sb += w * rb;
            error += w * ra.w;
            spread += w * w / max(textureLoad(rec_n, q, 0).y, 1e-6);
            total += w;
        }
    }
    var a = oa;
    var b = ob;
    if total > 0.0 {
        let e_s = error / total;
        let chroma_s = chroma_ratio(sa.yz, luma);
        sa /= total;
        sb /= total;
        if oa.w < 0.0 {
            a = vec4<f32>(sa.x, chroma_s, e_s);
            b = sb;
        } else {
            // By the samples each rests on: the pixel's own its length, the
            // records' what the spatial stage left them, as a weighted mean
            // of them holds, times the share of the pixel's tent their
            // geometry lets them serve (1 on an open surface; toward 0 at a
            // contact, where the pixel keeps its own). Neither depends on
            // the light, so the mix keeps the mean.
            let n_s = min(total, 1.0) * total * total / spread;
            let t = n_s / (n_s + max(m.x, 1e-6));
            let y_p = luma_at(oa.x, ob.xyz, g.n);
            let y_s = luma_at(sa.x, sb.xyz, g.n);
            let wy_p = (1.0 - t) * y_p;
            let wy_s = t * y_s;
            let chroma = select(oa.yz, (wy_p * oa.yz + wy_s * chroma_s) / (wy_p + wy_s), wy_p + wy_s > 0.0);
            let e = sqrt((1.0 - t) * (1.0 - t) * oa.w * oa.w + t * t * e_s * e_s);
            a = vec4<f32>(mix(oa.x, sa.x, t), chroma, e);
            b = mix(ob, sb, t);
        }
    }
    textureStore(hist_a_out, p, half4(a, p, 2u));
    textureStore(hist_b_out, p, half4(b, p, 3u));
}
