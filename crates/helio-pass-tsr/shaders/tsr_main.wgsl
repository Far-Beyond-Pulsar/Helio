// Temporal Super-Resolution (TSR) — compute-free fullscreen resolve
//
// Algorithm:
//   1. Depth-based reprojection → history UV
//   2. Pixel classification (LARGE_MOTION, DISOCCLUSION, SPECULAR_SHIMMER, EDGE)
//   3. Neighbourhood sampling in YCoCg space (5×5 tap for Quality/Native, 3×3 otherwise)
//   4. AABB clamping of the history sample
//   5. Adaptive temporal blend driven by classification
//   6. Contrast-Adaptive Sharpening (CAS) on the upsampled output
//
// References:
//   UE5 TSR — https://docs.unrealengine.com/en-US/temporal-super-resolution/
//   FSR 2.x — https://gpuopen.com/fidelityfx-super-resolution-2
//   Playdead temporal — https://github.com/playdeadgames/temporal (MIT)
//   AMD CAS — https://gpuopen.com/fidelityfx-cas

// ── Classification bit flags ──────────────────────────────────────────────────
const CLASS_NONE:             u32 = 0u;
const CLASS_LARGE_MOTION:     u32 = 1u;   // |velocity| > threshold → shorter accumulation
const CLASS_DISOCCLUSION:     u32 = 2u;   // depth mismatch → discard history
const CLASS_SPECULAR_SHIMMER: u32 = 4u;   // luminance variance > threshold → smooth
const CLASS_EDGE:             u32 = 8u;   // depth discontinuity → aggressive clamping

// ── Constants ─────────────────────────────────────────────────────────────────
const C_POS_INFTY:              f32 = 1.0e32;
const C_NEG_INFTY:              f32 = -1.0e32;
const MIN_HISTORY_BLEND_RATE:   f32 = 0.04;
const MAX_HISTORY_BLEND_RATE:   f32 = 1.0;
const LARGE_MOTION_THRESHOLD:   f32 = 0.01;  // UV-space velocity magnitude
const DISOCCLUSION_THRESHOLD:   f32 = 0.05;  // normalised depth difference
const SHIMMER_VAR_THRESHOLD:    f32 = 0.03;
const EDGE_DEPTH_THRESHOLD:     f32 = 0.02;
const CAS_SHARPNESS:            f32 = 0.45;

// ── Bindings ──────────────────────────────────────────────────────────────────

@group(0) @binding(0) var current_frame:  texture_2d<f32>;  // pre-AA at internal res
@group(0) @binding(1) var history_frame:  texture_2d<f32>;  // previous TSR output at output res
@group(0) @binding(2) var depth_tex:      texture_depth_2d; // depth at internal res
@group(0) @binding(3) var linear_sampler: sampler;
@group(0) @binding(4) var point_sampler:  sampler;

struct CameraUniforms {
    view:           mat4x4<f32>,
    proj:           mat4x4<f32>,
    view_proj:      mat4x4<f32>,
    inv_view_proj:  mat4x4<f32>,
    position_near:  vec4<f32>,
    forward_far:    vec4<f32>,
    jitter_frame:   vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}
@group(0) @binding(5) var<uniform> camera: CameraUniforms;

struct TsrUniform {
    jitter_offset:  vec2<f32>, // sub-pixel jitter in [-0.5, 0.5)
    reactivity:     f32,       // 0 = full history, 1 = no history
    reset:          u32,       // 1 on first frame / after reset_history()
    time_delta:     f32,       // seconds since last frame
    tap_radius:     u32,       // 1 = 3×3, 2 = 5×5
    _pad:           vec2<f32>,
}
@group(0) @binding(6) var<uniform> tsr: TsrUniform;

// ── Vertex passthrough ────────────────────────────────────────────────────────

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    var out: VertexOutput;
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

// ── Colour space helpers ──────────────────────────────────────────────────────

fn rgb_to_ycocg(rgb: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(rgb, vec3<f32>( 0.25,  0.5,  0.25)),
        dot(rgb, vec3<f32>( 0.5,   0.0, -0.5 )),
        dot(rgb, vec3<f32>(-0.25,  0.5, -0.25)),
    );
}

fn ycocg_to_rgb(ycocg: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        ycocg.x + ycocg.y - ycocg.z,
        ycocg.x            + ycocg.z,
        ycocg.x - ycocg.y - ycocg.z,
    );
}

// ── Reversible Reinhard tonemapper ────────────────────────────────────────────

fn max3(v: vec3<f32>) -> f32 { return max(v.r, max(v.g, v.b)); }
fn tonemap(c: vec3<f32>)         -> vec3<f32> { return c / (max3(c) + 1.0); }
fn reverse_tonemap(c: vec3<f32>) -> vec3<f32> { return c / (1.0 - max3(c) + 1.0e-8); }

// ── Catmull-Rom history sampling ──────────────────────────────────────────────

fn sample_catmull_rom(tex: texture_2d<f32>, samp: sampler, uv: vec2<f32>) -> vec3<f32> {
    let dims = vec2<f32>(textureDimensions(tex));
    let sp   = uv * dims;
    let tc   = floor(sp - 0.5) + 0.5;
    let f    = sp - tc;

    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);

    let w12 = w1 + w2;
    let o12 = w2 / w12;

    let ts   = 1.0 / dims;
    let uv0  = (tc - 1.0) * ts;
    let uv12 = (tc + o12) * ts;
    let uv3  = (tc + 2.0) * ts;

    var r = vec3<f32>(0.0);
    r += textureSampleLevel(tex, samp, vec2<f32>(uv0.x,  uv0.y),  0.0).rgb * w0.x  * w0.y;
    r += textureSampleLevel(tex, samp, vec2<f32>(uv12.x, uv0.y),  0.0).rgb * w12.x * w0.y;
    r += textureSampleLevel(tex, samp, vec2<f32>(uv3.x,  uv0.y),  0.0).rgb * w3.x  * w0.y;
    r += textureSampleLevel(tex, samp, vec2<f32>(uv0.x,  uv12.y), 0.0).rgb * w0.x  * w12.y;
    r += textureSampleLevel(tex, samp, vec2<f32>(uv12.x, uv12.y), 0.0).rgb * w12.x * w12.y;
    r += textureSampleLevel(tex, samp, vec2<f32>(uv3.x,  uv12.y), 0.0).rgb * w3.x  * w12.y;
    r += textureSampleLevel(tex, samp, vec2<f32>(uv0.x,  uv3.y),  0.0).rgb * w0.x  * w3.y;
    r += textureSampleLevel(tex, samp, vec2<f32>(uv12.x, uv3.y),  0.0).rgb * w12.x * w3.y;
    r += textureSampleLevel(tex, samp, vec2<f32>(uv3.x,  uv3.y),  0.0).rgb * w3.x  * w3.y;
    return max(r, vec3<f32>(0.0));
}

// ── Neighbourhood statistics ──────────────────────────────────────────────────

struct Neighbourhood {
    aabb_min: vec3<f32>,
    aabb_max: vec3<f32>,
    avg:      vec3<f32>,
    variance: f32,      // luminance variance
    depth_range: vec2<f32>, // min, max depth in neighbourhood
}

// Gather neighbourhood statistics in tonemapped YCoCg space.
// tap_radius: 1 = 3×3 (9 samples), 2 = 5×5 (25 samples).
fn gather_neighbourhood(
    tex: texture_2d<f32>,
    depth: texture_depth_2d,
    uv: vec2<f32>,
    texel: vec2<f32>,
    tap_radius: i32,
) -> Neighbourhood {
    var aabb_min = vec3<f32>(C_POS_INFTY);
    var aabb_max = vec3<f32>(C_NEG_INFTY);
    var w_sum   = 0.0;
    var l1      = vec3<f32>(0.0);
    var l2      = vec3<f32>(0.0);
    var d_min   = C_POS_INFTY;
    var d_max   = C_NEG_INFTY;

    for (var y = -tap_radius; y <= tap_radius; y++) {
        for (var x = -tap_radius; x <= tap_radius; x++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            let s  = textureSampleLevel(tex, point_sampler, uv + offset, 0.0).rgb;
            let q  = rgb_to_ycocg(tonemap(s));
            // Distance-based weight (centre-heavy)
            let dist = abs(f32(x)) + abs(f32(y));
            let w  = exp(-0.5 * dist);

            aabb_min = min(aabb_min, q);
            aabb_max = max(aabb_max, q);
            w_sum   += w;
            l1      += w * q;
            l2      += w * q * q;

            let d = textureSample(depth, point_sampler, uv + offset);
            d_min = min(d_min, d);
            d_max = max(d_max, d);
        }
    }

    l1 /= w_sum;
    l2 /= w_sum;

    let variance_vec = max(l2 - l1 * l1, vec3<f32>(0.0));
    let luma_variance = variance_vec.x; // Y channel variance = luminance variance

    var n: Neighbourhood;
    n.aabb_min   = aabb_min;
    n.aabb_max   = aabb_max;
    n.avg        = l1;
    n.variance   = luma_variance;
    n.depth_range = vec2<f32>(d_min, d_max);
    return n;
}

// ── Pixel classification ──────────────────────────────────────────────────────

// Classify this pixel and return a bitmask of CLASS_* flags.
fn classify_pixel(
    velocity:      vec2<f32>, // screen-space velocity (UV per frame)
    depth:         f32,
    history_depth: f32,
    n:             Neighbourhood,
) -> u32 {
    var flags = CLASS_NONE;

    // Large motion: sub-pixel accumulation breaks down
    if length(velocity) > LARGE_MOTION_THRESHOLD {
        flags |= CLASS_LARGE_MOTION;
    }

    // Disocclusion: reprojected history depth mismatches current depth
    let depth_diff = abs(depth - history_depth) / (depth + 1.0e-5);
    if depth_diff > DISOCCLUSION_THRESHOLD {
        flags |= CLASS_DISOCCLUSION;
    }

    // Specular shimmer: high luminance variance in neighbourhood
    if n.variance > SHIMMER_VAR_THRESHOLD {
        flags |= CLASS_SPECULAR_SHIMMER;
    }

    // Edge: large depth range in neighbourhood
    if (n.depth_range.y - n.depth_range.x) > EDGE_DEPTH_THRESHOLD {
        flags |= CLASS_EDGE;
    }

    return flags;
}

// ── Adaptive blend factor ─────────────────────────────────────────────────────

// Maps classification flags to a [MIN, MAX] blend factor.
// blend_factor = how much weight the CURRENT frame gets (1 = no history).
fn compute_blend_factor(
    flags:      u32,
    reactivity: f32,
    time_delta: f32,
    history_ycocg: vec3<f32>,
    n: Neighbourhood,
) -> f32 {
    var base = MIN_HISTORY_BLEND_RATE;

    // Forced fast-blend cases
    if (flags & CLASS_DISOCCLUSION) != 0u {
        base = 0.5;
    } else if (flags & CLASS_LARGE_MOTION) != 0u {
        base = 0.25;
    }

    // Clamp history luminance to AABB; if already inside, keep history
    let clamped = clamp(history_ycocg, n.aabb_min, n.aabb_max);
    let dist    = length(history_ycocg - clamped);
    // Extra push toward current when history is far outside the AABB
    let aabb_push = clamp(dist * 8.0, 0.0, 0.5);
    base += aabb_push;

    // Shimmer: temporal smoothing (reduce blend so history accumulates)
    if (flags & CLASS_SPECULAR_SHIMMER) != 0u {
        base = max(base - 0.05, MIN_HISTORY_BLEND_RATE);
    }

    // Reactivity override (camera cut / scene change via set_reactivity)
    base = mix(base, MAX_HISTORY_BLEND_RATE, reactivity);

    // Frame-rate independent adaptation: more blend if long time has passed
    let time_factor = 1.0 - exp(-time_delta * 4.0);
    base = mix(base, 0.5, time_factor * 0.5);

    return clamp(base, MIN_HISTORY_BLEND_RATE, MAX_HISTORY_BLEND_RATE);
}

// ── Contrast-Adaptive Sharpening (CAS) ───────────────────────────────────────

// Lightweight CAS pass on the resolved colour.
// Locally modulated: less sharpening in high-variance (noisy) regions.
fn apply_cas(rgb: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    // Use output-res texel size from current_frame dimensions scaled to output dims
    let texel = 1.0 / vec2<f32>(textureDimensions(current_frame));

    // 5-tap cross kernel
    let c = rgb;
    let n = textureSampleLevel(current_frame, point_sampler, uv + vec2<f32>(0.0, -texel.y), 0.0).rgb;
    let s = textureSampleLevel(current_frame, point_sampler, uv + vec2<f32>(0.0,  texel.y), 0.0).rgb;
    let e = textureSampleLevel(current_frame, point_sampler, uv + vec2<f32>( texel.x, 0.0), 0.0).rgb;
    let w = textureSampleLevel(current_frame, point_sampler, uv + vec2<f32>(-texel.x, 0.0), 0.0).rgb;

    let luma    = vec3<f32>(0.2126, 0.7152, 0.0722);
    let lc = dot(c, luma);
    let ln = dot(n, luma); let ls = dot(s, luma);
    let le = dot(e, luma); let lw = dot(w, luma);

    let contrast = max(max(max(max(lc, ln), ls), le), lw)
                 - min(min(min(min(lc, ln), ls), le), lw);

    // CAS formula: sharpen flat areas, leave edges alone
    let blur     = (n + s + e + w) * 0.25;
    let strength = CAS_SHARPNESS * saturate(1.0 - 2.0 * contrast);
    return clamp(c + (c - blur) * strength, vec3<f32>(0.0), vec3<f32>(1.0));
}

// ── Main fragment shader ──────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let in_dims  = vec2<f32>(textureDimensions(current_frame));
    let out_dims = vec2<f32>(textureDimensions(history_frame));
    let in_texel = 1.0 / in_dims;

    // ── Jitter correction ─────────────────────────────────────────────────────
    let jitter_uv = tsr.jitter_offset * vec2<f32>(1.0, -1.0) / in_dims;
    let cur_uv    = in.uv + jitter_uv;

    // ── Current frame sample (jitter-corrected) ───────────────────────────────
    let current_rgb = textureSampleLevel(current_frame, linear_sampler, cur_uv, 0.0).rgb;

    // ── RESET path ────────────────────────────────────────────────────────────
    if tsr.reset != 0u {
        let sharpened = apply_cas(current_rgb, cur_uv);
        return vec4<f32>(sharpened, 1.0);
    }

    // ── Depth-based reprojection → history UV ─────────────────────────────────
    let depth_val = textureSample(depth_tex, point_sampler, in.uv);
    let ndc_xy    = vec2<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    let clip      = vec4<f32>(ndc_xy, depth_val, 1.0);
    let world_h   = camera.inv_view_proj * clip;
    let world_pos = world_h.xyz / world_h.w;
    let prev_clip = camera.prev_view_proj * vec4<f32>(world_pos, 1.0);
    let prev_ndc  = prev_clip.xy / prev_clip.w;
    let history_uv = vec2<f32>((prev_ndc.x + 1.0) * 0.5, (1.0 - prev_ndc.y) * 0.5);

    // If reprojected UV is out of screen, use current frame only
    if any(history_uv < vec2<f32>(0.0)) || any(history_uv > vec2<f32>(1.0)) {
        let sharpened = apply_cas(current_rgb, cur_uv);
        return vec4<f32>(sharpened, 1.0);
    }

    // ── History sample (Catmull-Rom for quality) ───────────────────────────────
    let history_rgb = sample_catmull_rom(history_frame, linear_sampler, history_uv);

    // ── Reprojected history depth ─────────────────────────────────────────────
    // We use the current depth value for the disocclusion comparison. Sampling
    // at history_uv would cause false disocclusions at depth edges because the
    // jitter delta between frames shifts history_uv away from in.uv, even for
    // a perfectly static scene. That spurious CLASS_DISOCCLUSION forces blend
    // to 0.5 and prevents temporal accumulation (visible as shimmer/shake).
    // A proper implementation would keep a separate history depth buffer.
    let history_depth_approx = depth_val;

    // ── Neighbourhood statistics ───────────────────────────────────────────────
    let tap_radius = i32(tsr.tap_radius);
    let n = gather_neighbourhood(current_frame, depth_tex, cur_uv, in_texel, tap_radius);

    // ── Screen-space velocity (UV-space) ──────────────────────────────────────
    let velocity = history_uv - in.uv;

    // ── Pixel classification ──────────────────────────────────────────────────
    let flags = classify_pixel(velocity, depth_val, history_depth_approx, n);

    // ── Tonemap for stable accumulation ───────────────────────────────────────
    let current_tm  = rgb_to_ycocg(tonemap(current_rgb));
    let history_tm  = rgb_to_ycocg(tonemap(history_rgb));

    // ── AABB clamp (YCoCg) → prevents ghosting ─────────────────────────────────
    var aabb_min = n.aabb_min;
    var aabb_max = n.aabb_max;

    // Widen AABB slightly for specular shimmer (avoid over-clamping moving highlights)
    if (flags & CLASS_SPECULAR_SHIMMER) != 0u {
        let widen = vec3<f32>(0.03);
        aabb_min -= widen;
        aabb_max += widen;
    }

    // Narrow AABB on edges to reduce bleeding across depth discontinuities
    if (flags & CLASS_EDGE) != 0u {
        let avg = (aabb_min + aabb_max) * 0.5;
        let half_extent = (aabb_max - aabb_min) * 0.35;
        aabb_min = avg - half_extent;
        aabb_max = avg + half_extent;
    }

    let clamped_history = clamp(history_tm, aabb_min, aabb_max);

    // ── Adaptive blend ─────────────────────────────────────────────────────────
    let blend = compute_blend_factor(flags, tsr.reactivity, tsr.time_delta, history_tm, n);

    // ── Blend ─────────────────────────────────────────────────────────────────
    let blended_ycocg = mix(clamped_history, current_tm, blend);
    let blended_rgb   = ycocg_to_rgb(blended_ycocg);
    let result_linear = reverse_tonemap(clamp(blended_rgb, vec3<f32>(0.0), vec3<f32>(1.0)));

    // ── CAS sharpening ────────────────────────────────────────────────────────
    // Only sharpen in areas with enough variance to have recoverable detail.
    let cas_weight = clamp((n.variance - 0.002) * 10.0, 0.0, 1.0);
    let cas_result = apply_cas(result_linear, cur_uv);
    let output     = mix(result_linear, cas_result, cas_weight);

    return vec4<f32>(output, 1.0);
}
