// TAA (Temporal Anti-Aliasing) shader
//
// References:
//   https://www.elopezr.com/temporal-aa-and-the-quest-for-the-holy-trail
//   http://behindthepixels.io/assets/files/TemporalAA.pdf
//   Playdead clip_towards_aabb_center: https://github.com/playdeadgames/temporal (MIT)
//   Bevy TAA: https://github.com/bevyengine/bevy (MIT/Apache-2.0)

// How much of the current frame to blend in.
// Lower = more temporal smoothing, more ghosting risk on fast motion.
const DEFAULT_HISTORY_BLEND_RATE: f32 = 0.1;   // used when history is uncertain
const MIN_HISTORY_BLEND_RATE:     f32 = 0.015; // used when history is very confident

@group(0) @binding(0) var current_frame: texture_2d<f32>;
@group(0) @binding(1) var history_frame: texture_2d<f32>;
@group(0) @binding(2) var velocity_tex: texture_2d<f32>;
@group(0) @binding(3) var depth_tex: texture_depth_2d;
@group(0) @binding(4) var linear_sampler: sampler;
@group(0) @binding(5) var point_sampler: sampler;

struct TaaUniform {
    feedback_min:  f32,          // unused — kept for layout compat
    feedback_max:  f32,          // unused — kept for layout compat
    jitter_offset: vec2<f32>,    // Halton-0.5 offset for this frame
    reset:         u32,          // 1 on the very first frame
    _pad:          u32,
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

fn rgb_to_ycocg(rgb: vec3<f32>) -> vec3<f32> {
    let y = dot(rgb, vec3<f32>(0.25, 0.5, 0.25));
    let co = dot(rgb, vec3<f32>(0.5, 0.0, -0.5));
    let cg = dot(rgb, vec3<f32>(-0.25, 0.5, -0.25));
    return vec3<f32>(y, co, cg);
}

fn ycocg_to_rgb(ycocg: vec3<f32>) -> vec3<f32> {
    let y = ycocg.x;
    let co = ycocg.y;
    let cg = ycocg.z;
    return vec3<f32>(
        y + co - cg,
        y + cg,
        y - co - cg
    );
}

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
    
    // Use textureSampleLevel (explicit LOD=0) so this function may be called from
    // non-uniform control flow (history_uv depends on velocity which is non-uniform).
    var result = vec3<f32>(0.0);
    result = result + textureSampleLevel(tex, samp, vec2<f32>(uv0.x, uv0.y), 0.0).rgb * w0.x * w0.y;
    result = result + textureSampleLevel(tex, samp, vec2<f32>(uv12.x, uv0.y), 0.0).rgb * w12.x * w0.y;
    result = result + textureSampleLevel(tex, samp, vec2<f32>(uv3.x, uv0.y), 0.0).rgb * w3.x * w0.y;
    
    result = result + textureSampleLevel(tex, samp, vec2<f32>(uv0.x, uv12.y), 0.0).rgb * w0.x * w12.y;
    result = result + textureSampleLevel(tex, samp, vec2<f32>(uv12.x, uv12.y), 0.0).rgb * w12.x * w12.y;
    result = result + textureSampleLevel(tex, samp, vec2<f32>(uv3.x, uv12.y), 0.0).rgb * w3.x * w12.y;
    
    result = result + textureSampleLevel(tex, samp, vec2<f32>(uv0.x, uv3.y), 0.0).rgb * w0.x * w3.y;
    result = result + textureSampleLevel(tex, samp, vec2<f32>(uv12.x, uv3.y), 0.0).rgb * w12.x * w3.y;
    result = result + textureSampleLevel(tex, samp, vec2<f32>(uv3.x, uv3.y), 0.0).rgb * w3.x * w3.y;
    
    return max(result, vec3<f32>(0.0));
}

// Clip history_color towards the AABB centre rather than clamping to the AABB surface.
// From Playdead's temporal reprojection (MIT licence):
//   https://github.com/playdeadgames/temporal
// This preserves more valid history than plain clamp while still preventing ghosting.
fn clip_towards_aabb_center(
    history_color: vec3<f32>,
    current_color: vec3<f32>,
    aabb_min: vec3<f32>,
    aabb_max: vec3<f32>,
) -> vec3<f32> {
    let p_clip = 0.5 * (aabb_max + aabb_min);
    let e_clip = 0.5 * (aabb_max - aabb_min) + 1e-7;
    let v_clip = history_color - p_clip;
    let v_unit = v_clip / e_clip;
    let a_unit = abs(v_unit);
    let ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));
    if ma_unit > 1.0 {
        return p_clip + (v_clip / ma_unit);
    }
    return history_color;
}

// Reversible tonemapper — keeps HDR values from dominating temporal accumulation.
// (Reinhard per-channel max; GPU Open optimised version.)
fn rcp(x: f32) -> f32 { return 1.0 / x; }
fn max3(v: vec3<f32>) -> f32 { return max(v.r, max(v.g, v.b)); }
fn tonemap(c: vec3<f32>)         -> vec3<f32> { return c * rcp(max3(c) + 1.0); }
fn reverse_tonemap(c: vec3<f32>) -> vec3<f32> { return c * rcp(1.0 - max3(c)); }

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let in_dims  = vec2<f32>(textureDimensions(current_frame));
    let in_texel = 1.0 / in_dims;

    // ── Jitter correction ───────────────────────────────────────────────────
    // The projection was shifted by NDC (raw-0.5)*2/W, which equals UV (raw-0.5)/W.
    // Output pixel in.uv represents the UNJITTERED world-point at in.uv.
    // Because jitter shifts all geometry by +jitter_uv, the world-point at in.uv
    // landed at in.uv + jitter_uv in the rendered current_frame.
    // So we add jitter_uv when reading current_frame, and do NOT subtract it from
    // history_uv (since history already stores accumulated unjittered results).
    let jitter_uv = taa.jitter_offset * vec2<f32>(1.0, -1.0) / in_dims;
    let cur_uv    = in.uv + jitter_uv;

    // ── Current frame sample ────────────────────────────────────────────────
    let original_color = textureSample(current_frame, point_sampler, cur_uv);
    let current_color  = tonemap(original_color.rgb);

    // ── RESET (first frame) ─────────────────────────────────────────────────
    // Write current directly and prime history with high confidence so
    // accumulation starts at MIN_HISTORY_BLEND_RATE from frame 2 onwards.
    if taa.reset != 0u {
        return vec4<f32>(original_color.rgb, 1.0 / MIN_HISTORY_BLEND_RATE);
    }

    // ── Motion / history UV ─────────────────────────────────────────────────
    // history_uv = where this world-point was last frame (no jitter, history
    // is stored in unjittered output space).
    let velocity   = textureSample(velocity_tex, point_sampler, in.uv).xy;
    let history_uv = in.uv - velocity;

    if any(history_uv < vec2<f32>(0.0)) || any(history_uv > vec2<f32>(1.0)) {
        return vec4<f32>(original_color.rgb, 1.0);
    }

    // ── History ─────────────────────────────────────────────────────────────
    // Read confidence from history alpha (point sampler — no filtering needed).
    let raw_confidence = textureSampleLevel(history_frame, point_sampler, history_uv, 0.0).a;

    // Catmull-Rom reduces blur compared to bilinear.
    // History was written as original (non-tonemapped) colors; tonemap before blending.
    let history_color = tonemap(sample_catmull_rom(history_frame, linear_sampler, history_uv));

    // ── 3×3 neighbourhood AABB in YCoCg ────────────────────────────────────
    // Sample around the jitter-corrected UV so the box bounds the correct world-point.
    var m1 = vec3<f32>(0.0);
    var m2 = vec3<f32>(0.0);
    for (var x = -1; x <= 1; x = x + 1) {
        for (var y = -1; y <= 1; y = y + 1) {
            let s = rgb_to_ycocg(tonemap(
                textureSample(current_frame, point_sampler,
                    cur_uv + vec2<f32>(f32(x), f32(y)) * in_texel).rgb));
            m1 += s;
            m2 += s * s;
        }
    }
    let mean    = m1 / 9.0;
    let std_dev = sqrt(max(m2 / 9.0 - mean * mean, vec3<f32>(0.0)));

    // Clip history towards AABB centre (Playdead method).
    let clipped_history = ycocg_to_rgb(clip_towards_aabb_center(
        rgb_to_ycocg(history_color),
        rgb_to_ycocg(current_color),
        mean - std_dev,
        mean + std_dev,
    ));

    // ── Confidence-based blend rate ─────────────────────────────────────────
    // Static pixels accumulate confidence → blend rate approaches MIN (0.015).
    // Moving pixels reset confidence to 1 → blend rate = DEFAULT (0.1).
    let pixel_motion = abs(velocity) * in_dims;
    var new_confidence: f32;
    if pixel_motion.x < 0.01 && pixel_motion.y < 0.01 {
        new_confidence = raw_confidence + 10.0;
    } else {
        new_confidence = 1.0;
    }
    let blend_rate = clamp(1.0 / new_confidence, MIN_HISTORY_BLEND_RATE, DEFAULT_HISTORY_BLEND_RATE);

    // ── Blend and output ────────────────────────────────────────────────────
    // Result is kept in original (non-tonemapped) space for history storage.
    // Alpha carries the confidence counter for next frame.
    let result = reverse_tonemap(mix(clipped_history, current_color, blend_rate));
    return vec4<f32>(result, new_confidence);
}
