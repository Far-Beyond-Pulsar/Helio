//!use helio_prelude
//!use frame_history
//!use frame_denoise_irradiance
// The denoiser's stages for an irradiance signal (lib.rs documents them and
// where each constant comes from), one thread a pixel at the internal
// resolution:
//
// 0. `guides`: each pixel's raw depth and normal (octahedral) into one
//    texel, so every later tap reads its geometry with one load.
// 1. `temporal`: this frame's samples at the pixel (the traced pixels of the
//    nearest 2x2 blocks, a tent of the block over their distance times the
//    geometry weight) folded into the pixel's history, read where the frame's
//    history says the pixel was (the bilinear taps that held the same
//    surface). The history's length falls by the producer's confidence and
//    the share of the footprint that held the surface, and is capped.
// 2. `fix`: where history is short, the neighbours' accumulated values,
//    weighted by their history length and the geometry, over a radius that
//    shrinks as history grows. Its result is the history the next frame reads.
// 3. `spatial`, four times at steps of 1, 2, 4 and 8 blocks: an a-trous
//    filter of the linear B-spline's 3x3 (the same reach as three of the
//    cubic B-spline's 5x5, at 36 taps instead of 75) stopped by the
//    luminance's standard error (variance guided), by geometry (distance
//    against the irradiance's own scale, the hit distance; normal; plane). A
//    pixel whose estimate is already within the half floats' rounding is not
//    filtered.
//
// Values are the linear irradiance a + b . n (`denoise_irradiance_linear`),
// its chroma as ratios to luminance, 1 / hit distance; history also a length
// and the first two moments of the luminance at the pixel's normal (f32:
// the variance is their difference, which half floats cannot resolve).

struct DenoiseParams {
    // The view's inverse projection (view space from clip; no frame-space
    // translation, so positions keep their precision far from the origin).
    inv_proj: mat4x4<f32>,
    size: vec2<u32>,
    traced: vec2<u32>,
    side: u32,
    frame: u32,
    // Bit 0: the signal has a confidence; bit 1: the frame's history is bound.
    flags: u32,
    max_history: f32,
    fix_history: f32,
    // A pixel's width a metre of view depth.
    pixel_scale: f32,
    _pad: vec2<u32>,
}

@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var normal_tex: texture_2d<f32>;
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
@group(1) @binding(7) var temp_m_out: texture_storage_2d<rg32float, write>;
// fix
@group(1) @binding(8) var temp_a_in: texture_2d<f32>;
@group(1) @binding(9) var temp_b_in: texture_2d<f32>;
@group(1) @binding(10) var temp_m_in: texture_2d<f32>;
@group(1) @binding(11) var hist_a_out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(12) var hist_b_out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(13) var hist_m_out: texture_storage_2d<rg32float, write>;
// spatial
@group(1) @binding(14) var in_a: texture_2d<f32>;
@group(1) @binding(15) var in_b: texture_2d<f32>;
@group(1) @binding(16) var in_m: texture_2d<f32>;
@group(1) @binding(17) var out_a: texture_storage_2d<rgba16float, write>;
@group(1) @binding(18) var out_b: texture_storage_2d<rgba16float, write>;
// x: the step in pixels, y: the iteration (0 reads history).
@group(1) @binding(19) var<uniform> iteration: vec4<u32>;
// guides: written, then read by every other stage.
@group(1) @binding(20) var guides_out: texture_storage_2d<rgba32float, write>;
@group(1) @binding(21) var guides_in: texture_2d<f32>;

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

// A unit vector's octahedral encoding.
fn oct_encode(n: vec3<f32>) -> vec2<f32> {
    let p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    let folded = (1.0 - abs(p.yx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), p >= vec2<f32>(0.0));
    return select(p, folded, n.z < 0.0);
}

fn oct_decode(e: vec2<f32>) -> vec3<f32> {
    var n = vec3<f32>(e, 1.0 - abs(e.x) - abs(e.y));
    let t = saturate(-n.z);
    n = vec3<f32>(n.xy + select(vec2<f32>(t), vec2<f32>(-t), n.xy >= vec2<f32>(0.0)), n.z);
    return normalize(n);
}

@compute @workgroup_size(8, 8)
fn guides(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= params.size) {
        return;
    }
    let p = vec2<i32>(id.xy);
    let d = textureLoad(depth_tex, p, 0);
    var e = vec2<f32>(0.0);
    if !helio_is_sky(d) {
        e = oct_encode(helio_gbuffer_normal(textureLoad(normal_tex, p, 0).xyz));
    }
    textureStore(guides_out, p, vec4<f32>(d, e, 0.0));
}

fn guide(p: vec2<i32>) -> Guide {
    let size = vec2<i32>(params.size);
    let q = clamp(p, vec2<i32>(0), size - 1);
    let t = textureLoad(guides_in, q, 0);
    var g: Guide;
    g.sky = helio_is_sky(t.x);
    if g.sky {
        return g;
    }
    let uv = (vec2<f32>(q) + 0.5) / vec2<f32>(size);
    let v = params.inv_proj * vec4<f32>(helio_uv_to_ndc(uv), t.x, 1.0);
    g.x = v.xyz / v.w;
    g.depth = -g.x.z;
    g.n = oct_decode(t.yz);
    return g;
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

// History: a = (a, Co / Y, Cg / Y, 1 / hit), b = (b, length), m = second
// the luminance's first two moments (at the pixel's normal).
struct State {
    a: vec4<f32>,
    b: vec4<f32>,
    m: vec2<f32>,
}

fn state_scaled(s: State, w: f32) -> State {
    return State(s.a * w, s.b * w, s.m * w);
}

fn state_add(x: State, y: State) -> State {
    return State(x.a + y.a, x.b + y.b, x.m + y.m);
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

// `v` as a half float, rounded up with the probability of its distance
// from the half below: the stored value's mean is `v` whatever rounding the
// hardware's conversion uses (it receives a value a half holds exactly).
// History is stored and read back every frame and blended by 1 / length: a
// conversion that truncates (several GPUs' storage writes) loses half a
// step a frame, which the running mean amplifies by its length (1% at 32
// frames).
fn half_unbiased(v: f32, seed: u32) -> f32 {
    let a = min(abs(v), 65504.0);
    let u = f32(pcg(seed) >> 8u) * (1.0 / 16777216.0);
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

fn half4(v: vec4<f32>, p: vec2<i32>, stream: u32) -> vec4<f32> {
    let seed = pcg((u32(p.y) * params.size.x + u32(p.x)) ^ pcg(params.frame * 8u + stream));
    return vec4<f32>(half_unbiased(v.x, seed), half_unbiased(v.y, seed + 1u),
        half_unbiased(v.z, seed + 2u), half_unbiased(v.w, seed + 3u));
}

fn load_history(t: vec2<i32>) -> State {
    return State(finite4(textureLoad(hist_a_in, t, 0)), finite4(textureLoad(hist_b_in, t, 0)),
        textureLoad(hist_m_in, t, 0).xy);
}

fn store_temp(p: vec2<i32>, s: State) {
    textureStore(temp_a_out, p, half4(s.a, p, 0u));
    textureStore(temp_b_out, p, half4(s.b, p, 1u));
    textureStore(temp_m_out, p, vec4<f32>(s.m, 0.0, 0.0));
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
        store_temp(p, empty);
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
        fresh = state_add(fresh, State(vec4<f32>(l.x, chroma, inv) * w, vec4<f32>(l.yzw, 0.0) * w, vec2<f32>(y, y * y) * w));
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
            for (var i = 0u; i < 4u; i++) {
                let w = taps.weights[i];
                if w > 0.0 {
                    history = state_add(history, state_scaled(load_history(history_tap(taps, i)), w / footprint));
                }
            }
            // A footprint the surface held in part is a weaker history.
            length = history.b.w * footprint;
        }
    }
    if weight > 0.0 {
        fresh = state_scaled(fresh, 1.0 / weight);
        length *= confidence / weight;
    }
    let total = min(length + weight, params.max_history);
    if total <= 0.0 {
        store_temp(p, empty);
        return;
    }
    let alpha = select(0.0, weight / total, weight > 0.0);
    var out = State(mix(history.a, fresh.a, alpha), mix(history.b, fresh.b, alpha), mix(history.m, fresh.m, alpha));
    out.b.w = total;
    store_temp(p, out);
}

fn load_temp(t: vec2<i32>) -> State {
    let q = clamp(t, vec2<i32>(0), vec2<i32>(params.size) - 1);
    return State(textureLoad(temp_a_in, q, 0), textureLoad(temp_b_in, q, 0), textureLoad(temp_m_in, q, 0).xy);
}

@compute @workgroup_size(8, 8)
fn fix(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = vec2<i32>(id.xy);
    if any(id.xy >= params.size) {
        return;
    }
    var own = load_temp(p);
    let length = own.b.w;
    let g = guide(p);
    if !g.sky && length < params.fix_history {
        // A 5x5 of taps spanning +-2 strides: two blocks each way where the
        // pixel has no history, one block once it holds half of it.
        let stride = max(1, i32(ceil(f32(params.side) * (1.0 - 0.5 * length / params.fix_history))));
        var sum = state_scaled(own, length);
        var total = length;
        for (var j = 0; j < 25; j++) {
            let o = vec2<i32>(j % 5 - 2, j / 5 - 2);
            if all(o == vec2<i32>(0)) {
                continue;
            }
            let q = p + o * stride;
            if any(q < vec2<i32>(0)) || any(q >= vec2<i32>(params.size)) {
                continue;
            }
            let s = load_temp(q);
            if s.b.w <= 0.0 {
                continue;
            }
            let gq = guide(q);
            if gq.sky {
                continue;
            }
            let w = s.b.w * geometry_weight(g, gq, record_radius(gq, s.a.w));
            sum = state_add(sum, state_scaled(s, w));
            total += w;
        }
        if total > 0.0 {
            own = state_scaled(sum, 1.0 / total);
            // Its length is its own: the neighbours' samples are not this
            // pixel's, and the next frames' replace them as fast as before.
            own.b.w = length;
        }
    }
    textureStore(hist_a_out, p, half4(own.a, p, 2u));
    textureStore(hist_b_out, p, half4(own.b, p, 3u));
    textureStore(hist_m_out, p, vec4<f32>(own.m, 0.0, 0.0));
}

// A pixel as the spatial stage filters it: the linear irradiance, chroma
// ratios, the standard error of its luminance at `n`, 1 / hit distance.
// `valid` false where it holds no light yet.
struct Filtered {
    a: f32,
    b: vec3<f32>,
    chroma: vec2<f32>,
    error: f32,
    inverse_hit: f32,
    valid: bool,
}

fn load_filtered(t: vec2<i32>, n: vec3<f32>) -> Filtered {
    let q = clamp(t, vec2<i32>(0), vec2<i32>(params.size) - 1);
    let a = textureLoad(in_a, q, 0);
    let b = textureLoad(in_b, q, 0);
    var f: Filtered;
    f.a = a.x;
    f.b = b.xyz;
    f.chroma = a.yz;
    if iteration.y == 0u {
        // History: its moments to an error.
        let length = b.w;
        f.valid = length > 0.0;
        let y = max(a.x + dot(b.xyz, n), 0.0);
        let m = textureLoad(in_m, q, 0).xy;
        let variance = max(m.y - m.x * m.x, 0.0);
        f.error = sqrt(variance / max(length, 1.0));
        f.inverse_hit = a.w;
    } else {
        f.valid = a.w >= 0.0;
        f.error = a.w;
        f.inverse_hit = b.w;
    }
    return f;
}

// The linear B-spline's weight at tap `k` from the centre: 1/2, 1/4.
fn b1(k: i32) -> f32 {
    return select(0.25, 0.5, k == 0);
}

@compute @workgroup_size(8, 8)
fn spatial(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = vec2<i32>(id.xy);
    if any(id.xy >= params.size) {
        return;
    }
    let g = guide(p);
    if g.sky {
        textureStore(out_a, p, vec4<f32>(0.0, 0.0, 0.0, -1.0));
        textureStore(out_b, p, vec4<f32>(0.0));
        return;
    }
    let centre = load_filtered(p, g.n);
    let y_centre = centre.a + dot(centre.b, g.n);
    // Converged: within the half floats' rounding, filtering adds nothing.
    if centre.valid && LUMA_SIGMAS * centre.error <= LUMA_FLOOR * abs(y_centre) {
        let stream = 4u + 2u * iteration.y;
        textureStore(out_a, p, half4(vec4<f32>(centre.a, centre.chroma, centre.error), p, stream));
        textureStore(out_b, p, half4(vec4<f32>(centre.b, centre.inverse_hit), p, stream + 1u));
        return;
    }
    let radius = record_radius(g, centre.inverse_hit);
    let step = i32(iteration.x);
    var sum_a = 0.0;
    var sum_b = vec3<f32>(0.0);
    var sum_chroma = vec2<f32>(0.0);
    var sum_inverse_hit = 0.0;
    var sum_variance = 0.0;
    var total = 0.0;
    for (var j = 0; j < 9; j++) {
        let o = vec2<i32>(j % 3 - 1, j / 3 - 1);
        let q = p + o * step;
        if any(q < vec2<i32>(0)) || any(q >= vec2<i32>(params.size)) {
            continue;
        }
        let gq = guide(q);
        if gq.sky {
            continue;
        }
        let f = load_filtered(q, gq.n);
        if !f.valid {
            continue;
        }
        var w = b1(abs(o.x)) * b1(abs(o.y));
        if any(o != vec2<i32>(0)) {
            // The neighbour's record held at its own radius and ours.
            w *= geometry_weight(g, gq, min(radius, record_radius(gq, f.inverse_hit)));
            if centre.valid {
                let y = f.a + dot(f.b, g.n);
                let spread = LUMA_SIGMAS * sqrt(centre.error * centre.error + f.error * f.error)
                    + LUMA_FLOOR * max(abs(y), abs(y_centre)) + 1e-12;
                w *= exp(-abs(y - y_centre) / spread);
            }
        }
        if w <= 0.0 {
            continue;
        }
        sum_a += w * f.a;
        sum_b += w * f.b;
        sum_chroma += w * f.chroma;
        sum_inverse_hit += w * f.inverse_hit;
        sum_variance += w * w * f.error * f.error;
        total += w;
    }
    if total <= 0.0 {
        textureStore(out_a, p, vec4<f32>(0.0, 0.0, 0.0, -1.0));
        textureStore(out_b, p, vec4<f32>(0.0));
        return;
    }
    let inv = 1.0 / total;
    let stream = 4u + 2u * iteration.y;
    textureStore(out_a, p, half4(vec4<f32>(sum_a * inv, sum_chroma * inv, sqrt(sum_variance) * inv), p, stream));
    textureStore(out_b, p, half4(vec4<f32>(sum_b * inv, sum_inverse_hit * inv), p, stream + 1u));
}
