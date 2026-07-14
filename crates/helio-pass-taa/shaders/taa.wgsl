// TAA (Temporal Anti-Aliasing) — TSR-style temporal resolve
//
// Algorithm based on the shadertoy TSR demo:
//   - Weighted 3×3 YCoCg neighbourhood min/max clamp (Playdead-style)
//   - Variance-driven adaptive blend rate
//   - Sub-pixel offset weight for jitter-aware accumulation
//   - Low-discrepancy R1/R2 (plastic ratio) jitter sequence
//   - Catmull-Rom history sampling
//   - Depth-based reprojection for motion vectors
//
// References:
//   https://www.shadertoy.com/view/ (TSR demo)
//   https://www.elopezr.com/temporal-aa-and-the-quest-for-the-holy-trail
//   https://github.com/playdeadgames/temporal (MIT)

const MIN_HISTORY_BLEND_RATE: f32 = 0.015;
const C_POS_INFTY: f32 = 1.0e32;
const C_NEG_INFTY: f32 = -1.0e32;
const C_MIN_STD: f32 = 1.0 / 16.0;
const C_MIN_VAR: f32 = C_MIN_STD * C_MIN_STD;

@group(0) @binding(0) var current_frame: texture_2d<f32>;
@group(0) @binding(1) var history_frame: texture_2d<f32>;

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
@group(0) @binding(2) var<uniform> camera: CameraUniforms;
@group(0) @binding(3) var depth_tex: texture_depth_2d;
@group(0) @binding(4) var linear_sampler: sampler;
@group(0) @binding(5) var point_sampler: sampler;

struct TaaUniform {
    upscale_factor: f32,
    reset: u32,
    time_delta: f32,
    _pad: f32,
}
@group(0) @binding(6) var<uniform> taa: TaaUniform;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

// ── YCoCg colour-space helpers ────────────────────────────────────────────────

fn rgb_to_ycocg(rgb: vec3<f32>) -> vec3<f32> {
    let y = dot(rgb, vec3<f32>(0.25, 0.5, 0.25));
    let co = dot(rgb, vec3<f32>(0.5, 0.0, -0.5));
    let cg = dot(rgb, vec3<f32>(-0.25, 0.5, -0.25));
    return vec3<f32>(y, co, cg);
}

fn ycocg_to_rgb(ycocg: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        ycocg.x + ycocg.y - ycocg.z,
        ycocg.x + ycocg.z,
        ycocg.x - ycocg.y - ycocg.z,
    );
}

// ── Catmull-Rom texture sampling ──────────────────────────────────────────────

fn sample_catmull_rom(tex: texture_2d<f32>, samp: sampler, uv: vec2<f32>) -> vec3<f32> {
    let dimensions = vec2<f32>(textureDimensions(tex));
    let sample_pos = uv * dimensions;
    let tex_pos = floor(sample_pos - 0.5) + 0.5;
    let f = sample_pos - tex_pos;

    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);

    let w12 = w1 + w2;
    let offset12 = w2 / w12;

    let texel_size = 1.0 / dimensions;
    let uv0 = (tex_pos - 1.0) * texel_size;
    let uv12 = (tex_pos + offset12) * texel_size;
    let uv3 = (tex_pos + 2.0) * texel_size;

    var result = vec3<f32>(0.0);
    result += textureSampleLevel(tex, samp, vec2<f32>(uv0.x, uv0.y), 0.0).rgb * w0.x * w0.y;
    result += textureSampleLevel(tex, samp, vec2<f32>(uv12.x, uv0.y), 0.0).rgb * w12.x * w0.y;
    result += textureSampleLevel(tex, samp, vec2<f32>(uv3.x, uv0.y), 0.0).rgb * w3.x * w0.y;

    result += textureSampleLevel(tex, samp, vec2<f32>(uv0.x, uv12.y), 0.0).rgb * w0.x * w12.y;
    result += textureSampleLevel(tex, samp, vec2<f32>(uv12.x, uv12.y), 0.0).rgb * w12.x * w12.y;
    result += textureSampleLevel(tex, samp, vec2<f32>(uv3.x, uv12.y), 0.0).rgb * w3.x * w12.y;

    result += textureSampleLevel(tex, samp, vec2<f32>(uv0.x, uv3.y), 0.0).rgb * w0.x * w3.y;
    result += textureSampleLevel(tex, samp, vec2<f32>(uv12.x, uv3.y), 0.0).rgb * w12.x * w3.y;
    result += textureSampleLevel(tex, samp, vec2<f32>(uv3.x, uv3.y), 0.0).rgb * w3.x * w3.y;

    return max(result, vec3<f32>(0.0));
}

// ── YCoCg neighbourhood statistics ───────────────────────────────────────────

struct ColorRange {
    min: vec3<f32>,
    max: vec3<f32>,
    avg: vec3<f32>,
    dev: vec3<f32>,
}

// Statistics are computed in tonemapped YCoCg space so they match the space
// used during blending. Including alpha would corrupt the stats because the
// history texture stores a large confidence value (1/blend_rate) in alpha.
fn sample_range(tex: texture_2d<f32>, uv: vec2<f32>, step: vec2<f32>) -> ColorRange {
    var min_color = vec3<f32>(C_POS_INFTY);
    var max_color = vec3<f32>(C_NEG_INFTY);
    var total_weight = 0.0;
    var l1 = vec3<f32>(0.0);
    var l2 = vec3<f32>(0.0);

    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let s = textureSampleLevel(tex, point_sampler, uv + vec2<f32>(f32(x), f32(y)) * step, 0.0);
            let q = rgb_to_ycocg(tonemap(s.rgb));

            min_color = min(min_color, q);
            max_color = max(max_color, q);

            let w = 2.0 / f32(1 + abs(x) + abs(y));
            total_weight += w;
            l1 += w * q;
            l2 += w * q * q;
        }
    }

    l1 /= total_weight;
    l2 /= total_weight;

    var result: ColorRange;
    result.min = min_color;
    result.max = max_color;
    result.avg = l1;
    result.dev = sqrt(C_MIN_VAR + l2 - l1 * l1);
    return result;
}

fn clamp_to_range(color: vec3<f32>, range: ColorRange) -> vec3<f32> {
    let ycocg = rgb_to_ycocg(color);
    let clamped = clamp(ycocg, range.min, range.max);
    return ycocg_to_rgb(clamped);
}

fn variance_range_to_range(lhs: ColorRange, rhs: ColorRange) -> f32 {
    let inv_std = 1.0 / ((1.0 / lhs.dev) + (1.0 / rhs.dev));
    let diff = lhs.avg - rhs.avg;
    let variance = (C_MIN_VAR + diff * diff) / (inv_std * inv_std);
    return length(variance / 4.0);
}

// ── Reversible tonemapper (Reinhard) ─────────────────────────────────────────

fn max3(v: vec3<f32>) -> f32 { return max(v.r, max(v.g, v.b)); }
fn tonemap(c: vec3<f32>) -> vec3<f32> { return c * (1.0 / (max3(c) + 1.0)); }
fn reverse_tonemap(c: vec3<f32>) -> vec3<f32> { return c * (1.0 / (1.0 - max3(c) + 1.0e-8)); }

// ── Main resolve ──────────────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let in_dims  = vec2<f32>(textureDimensions(current_frame));
    let in_texel = 1.0 / in_dims;
    let out_dims = vec2<f32>(textureDimensions(history_frame));
    let out_texel = 1.0 / out_dims;

    // ── Jitter correction ───────────────────────────────────────────────────
    // This is the exact NDC translation already applied to camera.proj. NDC
    // spans two UV units and viewport Y points down, hence the 0.5 and Y flip.
    let jitter_uv = camera.jitter_frame.xy * vec2<f32>(0.5, -0.5);
    let jitter_offset_pixels = camera.jitter_frame.xy * in_dims * 0.5;
    let cur_uv    = in.uv + jitter_uv;

    // ── Current frame sample ────────────────────────────────────────────────
    let original_color = textureSample(current_frame, point_sampler, cur_uv);

    // ── RESET (first frame) ─────────────────────────────────────────────────
    if taa.reset != 0u {
        return vec4<f32>(original_color.rgb, 1.0 / MIN_HISTORY_BLEND_RATE);
    }

    // ── Depth-based reprojection → history UV ───────────────────────────────
    // Color, depth, and clip coordinates must identify the same jittered pixel.
    let depth_val  = textureSample(depth_tex, point_sampler, cur_uv);
    let ndc_xy     = vec2<f32>(cur_uv.x * 2.0 - 1.0, 1.0 - cur_uv.y * 2.0);
    let clip       = vec4<f32>(ndc_xy, depth_val, 1.0);
    let world_h    = camera.inv_view_proj * clip;
    let world_pos  = world_h.xyz / world_h.w;
    let prev_clip  = camera.prev_view_proj * vec4<f32>(world_pos, 1.0);
    let prev_ndc   = prev_clip.xy / prev_clip.w;
    let history_uv = vec2<f32>((prev_ndc.x + 1.0) * 0.5, (1.0 - prev_ndc.y) * 0.5);

    if any(history_uv < vec2<f32>(0.0)) || any(history_uv > vec2<f32>(1.0)) {
        return vec4<f32>(original_color.rgb, 1.0);
    }

    // ── History sample ──────────────────────────────────────────────────────
    let history_rgb = sample_catmull_rom(history_frame, linear_sampler, history_uv);

    // ── Tonemap for stable accumulation ─────────────────────────────────────
    let current_color = tonemap(original_color.rgb);
    let history_color = tonemap(history_rgb);

    // ── Weighted 3×3 YCoCg neighbourhood analysis ──────────────────────────
    let next_range = sample_range(current_frame, cur_uv, in_texel);
    let prev_range = sample_range(history_frame, history_uv, out_texel);

    // Clamp history to current frame's YCoCg [min, max]  → prevents ghosting
    let clamped = clamp_to_range(history_color, next_range);

    // Blend history toward current based on distance to the clamped value.
    // If history is inside current's AABB, ratio → 0 (keep history).
    // If history is outside current's AABB, ratio → 1 (favour current).
    let prev_dist = distance(history_color, clamped);
    let next_dist = distance(current_color, clamped);
    let blend_toward_current = prev_dist / (next_dist + prev_dist + 1.0e-6);
    var blended = mix(history_color, current_color, blend_toward_current);

    // ── Sub-pixel jitter offset weight ──────────────────────────────────────
    // Pixels sampled near the sub-pixel centre are more reliable.
    let jitter_len_sq = dot(jitter_offset_pixels, jitter_offset_pixels);
    let offset_w = exp(-4.0 * (1.0 - blend_toward_current) * jitter_len_sq);
    let w = offset_w * taa.upscale_factor * taa.upscale_factor;

    // ── Variance-driven blend rate ──────────────────────────────────────────
    // When current and history neighbourhoods have similar statistics, var is
    // small → rc is small → history is preserved (temporal stability).
    // When they differ (disocclusion, fast motion), var is large → rc is large
    // → current frame dominates (no ghosting).
    let variance = variance_range_to_range(next_range, prev_range);
    let rc = 1.0 - exp(-16.0 * max(taa.time_delta, 1.0 / 60.0) * variance * w);
    let blend_rate = clamp(rc, MIN_HISTORY_BLEND_RATE, 1.0);

    // ── Final blend & output ────────────────────────────────────────────────
    let result_rgb = mix(blended, current_color, blend_rate);
    let result = reverse_tonemap(result_rgb);

    // Alpha encodes effective blend confidence (inverse of blend_rate).
    // Higher confidence = slower accumulation = more temporal stability.
    // External passes can read this for debug visualisation.
    let confidence = 1.0 / blend_rate;

    return vec4<f32>(result, confidence);
}
