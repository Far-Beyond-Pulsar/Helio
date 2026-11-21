//!use helio_prelude
//!use frame_history
//!use frame_denoise_irradiance
// The denoiser's stages for an irradiance signal (lib.rs documents them and
// where each constant comes from). Light is spread where the information is,
// per traced sample; only what needs a pixel's own geometry runs per pixel:
//
// 1. `temporal` (a pixel): this frame's samples at the pixel (the traced
//    pixels of the nearest 2x2 blocks, a tent of the block over their
//    distance times the geometry weight) folded into the pixel's history,
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
//    block once it holds 16 frames), stopped by the luminance's standard
//    error (variance guided) and by geometry (distance against the
//    irradiance's own scale, the hit distance; normal; plane). A record
//    already within the half floats' rounding is left as it is.
// 4. `reconstruct` (a pixel): the filtered records of the pixel's nearest
//    2x2 blocks (tent times geometry, at the pixel's own normal), combined
//    with the pixel's own temporal result by inverse variance, the records'
//    variance taken with the bias their difference from the pixel shows. Its
//    result is the signal's output and the history the next frame reads:
//    the spatial stage is recurrent (its work accumulates over frames, as
//    ReBLUR's post-blur history does), while the luminance's moments stay
//    the raw samples' (the variance never sees the filter).
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
//   N R32Float (length)                           records only

struct DenoiseParams {
    // The view's inverse projection (view space from clip; no frame-space
    // translation, so positions keep their precision far from the origin).
    inv_proj: mat4x4<f32>,
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
@group(1) @binding(13) var rec_n_out: texture_storage_2d<r32float, write>;
// spatial (the length is the x of `in_n`: a record's N or a pixel's M)
@group(1) @binding(14) var in_a: texture_2d<f32>;
@group(1) @binding(15) var in_b: texture_2d<f32>;
@group(1) @binding(16) var in_n: texture_2d<f32>;
@group(1) @binding(17) var out_a: texture_storage_2d<rgba16float, write>;
@group(1) @binding(18) var out_b: texture_storage_2d<rgba16float, write>;
@group(1) @binding(19) var out_n: texture_storage_2d<r32float, write>;
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

// The normal stop: a sample's irradiance holds the hemisphere about the
// normal it was traced at; seen from a normal tilted by t, a share
// (1 - cos t) / 2 of the cosine-weighted hemisphere was never sampled. The
// weight falls to 0 where that share reaches 1/8 (t = 41 degrees).
const NORMAL_SHARE: f32 = 0.125;
// The plane stop: the chord between two points of a surface whose normals
// differ by the normal stop's 41 degrees leaves the tangent plane by
// tan(41 / 2 degrees) = 0.378 of its length; a step (a silhouette, a ledge)
// by a whole length or more.
const PLANE_SLOPE: f32 = 0.378;
// An irradiance record is valid over a radius of its harmonic mean hit
// distance (Ward 1988: the error of using it at distance d is d / R), never
// below the diagonal of the pixel that holds it.
const RADIUS_MIN_PIXELS: f32 = 1.41421356;
// The luminance stop: two estimates of one irradiance differ by noise of
// standard deviation sqrt(sp^2 + sq^2); a difference of LUMA_SIGMAS of it is
// weighted 1/e (3: 99.7% of noise differences lie within it)...
const LUMA_SIGMAS: f32 = 3.0;
// ...and a difference within the half floats' rounding (2^-10 of the value)
// is no difference.
const LUMA_FLOOR: f32 = 0.0009765625;
// The spatial stage's step: a record's estimate rests on `length` frames;
// the filter's footprint of (2 step + 1)^2 records adds about step^2 of
// them, so a step of 4 / sqrt(length) keeps about 16 frames' worth behind
// every estimate (one block from 16 frames on, four at one).
const SPATIAL_FRAMES: f32 = 16.0;
// History shorter than this many frames has no variance of its own yet:
// the moments of fewer than four samples are dominated by their own noise
// (SVGF's threshold).
const FIX_HISTORY: f32 = 4.0;
// Chroma ratios of real colours: Co / Y and Cg / Y stay within +-2 (pure
// red: Co / Y = 2).
const CHROMA_LIMIT: f32 = 2.0;
// A hit distance a half float cannot hold is no hit.
const NO_HIT: f32 = 65504.0;

struct Guide {
    // View-space position (metres), frame-space normal, view depth.
    x: vec3<f32>,
    n: vec3<f32>,
    depth: f32,
    sky: bool,
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
    // The raw depth of view depth vd (`helio_view_depth` inverted), then
    // the view-space point through the inverse projection.
    let z = params.proj_zw;
    let d = (z.z - vd * z.x) / (z.w - vd * z.y);
    let uv = (vec2<f32>(q) + 0.5) / vec2<f32>(size);
    let v = params.inv_proj * vec4<f32>(helio_uv_to_ndc(uv), d, 1.0);
    g.x = v.xyz / v.w;
    g.depth = vd;
    g.n = normalize(textureLoad(guide_normal, q, 0).xyz);
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

// How well the irradiance of a record at `q`, valid over `radius`, serves `p`.
fn geometry_weight(p: Guide, q: Guide, radius: f32) -> f32 {
    let d = distance(p.x, q.x);
    let w_distance = saturate(1.0 - d / radius);
    let w_normal = saturate(1.0 - (1.0 - dot(p.n, q.n)) / (2.0 * NORMAL_SHARE));
    let plane = abs(dot(p.n, q.x - p.x));
    let w_plane = saturate(1.0 - plane / (PLANE_SLOPE * d + footprint(p)));
    return w_distance * w_normal * w_plane;
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

// A pixel's state while it is blended: (a, Co / Y, Cg / Y, 1 / hit), (b),
// (mean, mean square), length.
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

fn load_history(t: vec2<i32>) -> State {
    let a = finite4(textureLoad(hist_a_in, t, 0));
    let b = finite4(textureLoad(hist_b_in, t, 0));
    return State(vec4<f32>(a.xyz, b.w), b.xyz, textureLoad(hist_m_in, t, 0).yz);
}

// The standard error of a mean of `length` samples of moments `m`; -1 for
// no light.
fn standard_error(m: vec2<f32>, length: f32) -> f32 {
    return select(-1.0, sqrt(max(m.y - m.x * m.x, 0.0) / max(length, 1.0)), length > 0.0);
}

fn store_temp(p: vec2<i32>, s: State, length: f32) {
    textureStore(temp_a_out, p, half4(vec4<f32>(s.a.xyz, standard_error(s.m, length)), p, 0u));
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
        store_temp(p, empty, 0.0);
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
        let inv = inverse_hit(s0.w);
        let w = w_space * geometry_weight(g, gt, record_radius(gt, inv));
        if w <= 0.0 {
            continue;
        }
        let l = denoise_irradiance_linear(s0, s1, gt.n);
        let y = l.x + dot(l.yzw, g.n);
        // Chroma as ratios to the sample's own luminance: exact for each
        // sample of one colour whatever normal it was traced at.
        let chroma = clamp(s0.yz / max(s0.x, 1e-12), vec2<f32>(-CHROMA_LIMIT), vec2<f32>(CHROMA_LIMIT));
        fresh = state_add(fresh, State(vec4<f32>(l.x, chroma, inv) * w, l.yzw * w, vec2<f32>(y, y * y) * w));
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
                    history = state_add(history, state_scaled(load_history(t), w / footprint));
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
        store_temp(p, empty, 0.0);
        return;
    }
    let alpha = select(0.0, weight / total, weight > 0.0);
    store_temp(p, State(mix(history.a, fresh.a, alpha), mix(history.b, fresh.b, alpha), mix(history.m, fresh.m, alpha)), total);
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
    textureStore(rec_n_out, b, vec4<f32>(textureLoad(rec_temp_m, r, 0).x, 0.0, 0.0, 0.0));
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
    let g = guide(representative(b));
    let ca = textureLoad(in_a, b, 0);
    let cb = textureLoad(in_b, b, 0);
    let length = textureLoad(in_n, b, 0).x;
    textureStore(out_n, b, vec4<f32>(length, 0.0, 0.0, 0.0));
    if g.sky {
        textureStore(out_a, b, vec4<f32>(0.0, 0.0, 0.0, -1.0));
        textureStore(out_b, b, vec4<f32>(0.0));
        return;
    }
    let valid = ca.w >= 0.0;
    let y_centre = ca.x + dot(cb.xyz, g.n);
    // Converged: within the half floats' rounding, filtering adds nothing.
    if valid && LUMA_SIGMAS * ca.w <= LUMA_FLOOR * abs(y_centre) {
        textureStore(out_a, b, ca);
        textureStore(out_b, b, cb);
        return;
    }
    let radius = record_radius(g, cb.w);
    let s = clamp(round(sqrt(SPATIAL_FRAMES / max(length, 1.0))), 1.0, 4.0);
    let step = i32(s) << iteration.x;
    var sum_a = vec4<f32>(0.0);
    var sum_b = vec4<f32>(0.0);
    var sum_variance = 0.0;
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
            let gq = guide(representative(q));
            if gq.sky {
                continue;
            }
            // The neighbour's record held at its own radius and ours.
            w *= geometry_weight(g, gq, min(radius, record_radius(gq, fb.w)));
            if valid {
                let y = fa.x + dot(fb.xyz, g.n);
                let spread = LUMA_SIGMAS * sqrt(ca.w * ca.w + fa.w * fa.w)
                    + LUMA_FLOOR * max(abs(y), abs(y_centre)) + 1e-12;
                w *= exp(-abs(y - y_centre) / spread);
            }
        }
        if w <= 0.0 {
            continue;
        }
        sum_a += w * vec4<f32>(fa.xyz, 0.0);
        sum_b += w * fb;
        sum_variance += w * w * fa.w * fa.w;
        total += w;
    }
    if total <= 0.0 {
        textureStore(out_a, b, vec4<f32>(0.0, 0.0, 0.0, -1.0));
        textureStore(out_b, b, vec4<f32>(0.0));
        return;
    }
    let inv = 1.0 / total;
    // Rounded without bias: the records reach the history (`reconstruct`).
    let stream = 4u + 2u * iteration.x;
    textureStore(out_a, b, half4(vec4<f32>(sum_a.xyz * inv, sqrt(sum_variance) * inv), b, stream));
    textureStore(out_b, b, half4(sum_b * inv, b, stream + 1u));
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
    var error = 0.0;
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
            if w <= 0.0 {
                continue;
            }
            sa += w * vec4<f32>(ra.xyz, 0.0);
            sb += w * rb;
            error += w * ra.w;
            total += w;
        }
    }
    var a = oa;
    var b = ob;
    if total > 0.0 {
        sa /= total;
        sb /= total;
        let e_s = error / total;
        if oa.w < 0.0 {
            a = vec4<f32>(sa.xyz, e_s);
            b = sb;
        } else {
            // Inverse variance; the records' variance holds the bias their
            // difference from the pixel shows beyond both noises (the
            // pixel's own detail the records cannot hold).
            let d = (sa.x + dot(sb.xyz, g.n)) - (oa.x + dot(ob.xyz, g.n));
            let e_p2 = oa.w * oa.w;
            let bias2 = max(d * d - e_p2 - e_s * e_s, 0.0);
            let w_s = 1.0 / (e_s * e_s + bias2 + 1e-20);
            let w_p = 1.0 / (e_p2 + 1e-20);
            // Below FIX_HISTORY frames the moments do not yet measure the
            // pixel's variance (SVGF estimates it spatially there): its own
            // estimate counts by its length.
            let t = max(w_s / (w_s + w_p), 1.0 - m.x / FIX_HISTORY);
            a = vec4<f32>(mix(oa.xyz, sa.xyz, t), sqrt(1.0 / (w_s + w_p)));
            b = mix(ob, sb, t);
        }
    }
    textureStore(hist_a_out, p, half4(a, p, 2u));
    textureStore(hist_b_out, p, half4(b, p, 3u));
}
