// TAA (Temporal Anti-Aliasing) shader
//
// Implements the Bevy-inspired algorithm adapted for camera-matrix reprojection.
// No velocity buffer is required — motion vectors are derived from depth + camera matrices.
//
// References:
//   https://www.elopezr.com/temporal-aa-and-the-quest-for-the-holy-trail
//   http://behindthepixels.io/assets/files/TemporalAA.pdf
//   https://advances.realtimerendering.com/s2014/index.html#_HIGH-QUALITY_TEMPORAL_SUPERSAMPLING
//   Playdead clip_towards_aabb_center: https://github.com/playdeadgames/temporal (MIT)
//   Bevy TAA: https://github.com/bevyengine/bevy (MIT/Apache-2.0)

// Blend rates for history accumulation.
// Lower current_color_factor = more history = more temporal smoothing.
const DEFAULT_HISTORY_BLEND_RATE: f32 = 0.1;   // used when history is uncertain (motion)
const MIN_HISTORY_BLEND_RATE:     f32 = 0.015; // used when history is very confident (static)

// Bindings
// ────────────────────────────────────────────────────────────────────────────
// NOTE: history_frame is stored in Rgba16Float with:
//   RGB = tonemapped resolved color
//   A   = confidence counter (unbounded float, grows by 10/frame when static)
//
// Using Rgba16Float for history is REQUIRED.  If an 8-bit surface format were
// used, confidence would be clamped to 1.0 every frame and the adaptive blend
// rate would never drop below DEFAULT_HISTORY_BLEND_RATE, defeating the
// noise-reduction purpose of temporal accumulation entirely.
// Similarly, HDR specular highlights (luminance > 1) would be clipped in 8-bit
// history, corrupting the temporal average for reflective surfaces.
@group(0) @binding(0) var current_frame: texture_2d<f32>;
@group(0) @binding(1) var history_frame: texture_2d<f32>;
@group(0) @binding(2) var depth_tex: texture_depth_2d;
@group(0) @binding(3) var linear_sampler: sampler;
@group(0) @binding(4) var point_sampler: sampler;

struct Camera {
    view:          mat4x4<f32>,
    proj:          mat4x4<f32>,
    view_proj:     mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    position_near: vec4<f32>,
    forward_far:   vec4<f32>,
    jitter_frame:  vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}

@group(0) @binding(5) var<uniform> camera: Camera;

struct TaaUniform {
    jitter_offset: vec2<f32>,
    reset:         u32,
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

// ── Tonemap (reversible, GPU Open optimised) ──────────────────────────────────
// Applied before AABB math and history interpolation, reversed in the blit output.
// https://gpuopen.com/learn/optimized-reversible-tonemapper-for-resolve
fn rcp(x: f32) -> f32 { return 1.0 / x; }
fn max3(v: vec3<f32>) -> f32 { return max(v.r, max(v.g, v.b)); }
fn tonemap(c: vec3<f32>)         -> vec3<f32> { return c * rcp(max3(c) + 1.0); }
fn reverse_tonemap(c: vec3<f32>) -> vec3<f32> { return c * rcp(1.0 - max3(c)); }

// ── YCoCg color space (Playdead, MIT-licensed) ────────────────────────────────
fn RGB_to_YCoCg(rgb: vec3<f32>) -> vec3<f32> {
    let y  = (rgb.r / 4.0) + (rgb.g / 2.0) + (rgb.b / 4.0);
    let co = (rgb.r / 2.0) - (rgb.b / 2.0);
    let cg = (-rgb.r / 4.0) + (rgb.g / 2.0) - (rgb.b / 4.0);
    return vec3(y, co, cg);
}

fn YCoCg_to_RGB(ycocg: vec3<f32>) -> vec3<f32> {
    let r = ycocg.x + ycocg.y - ycocg.z;
    let g = ycocg.x + ycocg.z;
    let b = ycocg.x - ycocg.y - ycocg.z;
    return saturate(vec3(r, g, b));
}

// Clip history towards the AABB center (Playdead method, MIT licensed).
// Preserves more valid history than plain clamp while still preventing ghosting.
fn clip_towards_aabb_center(
    history_color: vec3<f32>,
    current_color: vec3<f32>,
    aabb_min: vec3<f32>,
    aabb_max: vec3<f32>,
) -> vec3<f32> {
    let p_clip = 0.5 * (aabb_max + aabb_min);
    let e_clip = 0.5 * (aabb_max - aabb_min) + 0.00000001;
    let v_clip = history_color - p_clip;
    let v_unit = v_clip / e_clip;
    let a_unit = abs(v_unit);
    let ma_unit = max3(a_unit);
    if ma_unit > 1.0 {
        return p_clip + (v_clip / ma_unit);
    } else {
        return history_color;
    }
}

// ── Reprojection via camera matrices (no velocity buffer required) ────────────
// Reconstructs the world position using the JITTERED inv_view_proj (matches the
// jittered depth buffer), then reprojects to the UNJITTERED previous frame using
// prev_view_proj.  For static geometry this gives an exact motion vector; for
// dynamic geometry the vector will be approximate (camera motion only, no skinning),
// but the AABB clip below limits ghosting.
fn compute_motion_vector(uv: vec2<f32>, depth: f32) -> vec2<f32> {
    // UV (0,0) = top-left → NDC X=-1, Y=+1 in WebGPU clip space.
    let ndc     = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world_h = camera.inv_view_proj * ndc;
    let world   = world_h.xyz / world_h.w;

    let prev_h = camera.prev_view_proj * vec4<f32>(world, 1.0);
    if prev_h.w == 0.0 { return vec2<f32>(0.0); }
    let prev_ndc = prev_h.xyz / prev_h.w;
    let prev_uv  = vec2<f32>(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);
    return uv - prev_uv;
}

// History stores tonemapped RGB — read directly (no extra tonemap needed).
fn sample_history(u: f32, v: f32) -> vec3<f32> {
    return textureSampleLevel(history_frame, linear_sampler, vec2(u, v), 0.0).rgb;
}

// Sample current frame, tonemap, then convert to YCoCg for variance clipping.
fn sample_current_ycocg(uv: vec2<f32>) -> vec3<f32> {
    return RGB_to_YCoCg(tonemap(
        textureSampleLevel(current_frame, point_sampler, uv, 0.0).rgb));
}

// ── Fragment shader ────────────────────────────────────────────────────────────
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texture_size = vec2<f32>(textureDimensions(current_frame));
    let texel_size   = 1.0 / texture_size;

    // Fetch the current sample (raw HDR).
    let original_color = textureSampleLevel(current_frame, point_sampler, in.uv, 0.0);
    var current_color  = tonemap(original_color.rgb);

    // On reset: prime history with tonemapped current + high confidence so
    // the blend rate starts at MIN_HISTORY_BLEND_RATE from frame 2.
    if taa.reset != 0u {
        return vec4<f32>(current_color, 1.0 / MIN_HISTORY_BLEND_RATE);
    }

    // ── Closest foreground depth from 5 samples ────────────────────────────
    // Picks the depth of the foreground geometry near this pixel.
    // Prevents background depth bleeding across silhouette edges, fixing the
    // most common cause of temporal flickering on thin objects.
    // https://advances.realtimerendering.com/s2014/index.html#_HIGH-QUALITY_TEMPORAL_SUPERSAMPLING, slide 27
    let offset = texel_size * 2.0;
    let d_uv_tl = in.uv + vec2(-offset.x,  offset.y);
    let d_uv_tr = in.uv + vec2( offset.x,  offset.y);
    let d_uv_bl = in.uv + vec2(-offset.x, -offset.y);
    let d_uv_br = in.uv + vec2( offset.x, -offset.y);

    var closest_uv    = in.uv;
    var closest_depth = textureSampleLevel(depth_tex, point_sampler, in.uv, 0);
    let d_tl = textureSampleLevel(depth_tex, point_sampler, d_uv_tl, 0);
    let d_tr = textureSampleLevel(depth_tex, point_sampler, d_uv_tr, 0);
    let d_bl = textureSampleLevel(depth_tex, point_sampler, d_uv_bl, 0);
    let d_br = textureSampleLevel(depth_tex, point_sampler, d_uv_br, 0);

    // Reversed-Z: larger depth value = closer to camera.
    if d_tl > closest_depth { closest_uv = d_uv_tl; closest_depth = d_tl; }
    if d_tr > closest_depth { closest_uv = d_uv_tr; closest_depth = d_tr; }
    if d_bl > closest_depth { closest_uv = d_uv_bl; closest_depth = d_bl; }
    if d_br > closest_depth { closest_uv = d_uv_br; }

    // Motion vector at the foreground sample (includes camera + static geometry motion).
    let closest_motion_vector = compute_motion_vector(closest_uv, closest_depth);
    let history_uv = in.uv - closest_motion_vector;

    // ── 5-sample Catmull-Rom history read ──────────────────────────────────
    // Reduces blurriness compared to bilinear; skips corners per CoD TAA.
    // https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1
    // https://www.activision.com/cdn/research/Dynamic_Temporal_Antialiasing_and_Upsampling_in_Call_of_Duty_v4.pdf#page=68
    let sample_pos   = history_uv * texture_size;
    let texel_center = floor(sample_pos - 0.5) + 0.5;
    let f = sample_pos - texel_center;

    let w0  = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1  = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2  = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3  = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;

    let t0  = (texel_center - 1.0) * texel_size;
    let t3  = (texel_center + 2.0) * texel_size;
    let t12 = (texel_center + (w2 / w12)) * texel_size;

    var history_color  = sample_history(t12.x, t0.y)  * w12.x * w0.y;
    history_color     += sample_history(t0.x,  t12.y) * w0.x  * w12.y;
    history_color     += sample_history(t12.x, t12.y) * w12.x * w12.y;
    history_color     += sample_history(t3.x,  t12.y) * w3.x  * w12.y;
    history_color     += sample_history(t12.x, t3.y)  * w12.x * w3.y;

    // ── 3×3 YCoCg variance clipping ───────────────────────────────────────
    // Constrains history to the color neighborhood of the current frame,
    // eliminating ghosting while preserving valid temporal accumulation.
    // https://advances.realtimerendering.com/s2014/index.html#_HIGH-QUALITY_TEMPORAL_SUPERSAMPLING, slide 33
    // https://developer.download.nvidia.com/gameworks/events/GDC2016/msalvi_temporal_supersampling.pdf
    let s_tl = sample_current_ycocg(in.uv + vec2(-texel_size.x,  texel_size.y));
    let s_tm = sample_current_ycocg(in.uv + vec2( 0.0,           texel_size.y));
    let s_tr = sample_current_ycocg(in.uv + vec2( texel_size.x,  texel_size.y));
    let s_ml = sample_current_ycocg(in.uv + vec2(-texel_size.x,  0.0));
    let s_mm = RGB_to_YCoCg(current_color);
    let s_mr = sample_current_ycocg(in.uv + vec2( texel_size.x,  0.0));
    let s_bl = sample_current_ycocg(in.uv + vec2(-texel_size.x, -texel_size.y));
    let s_bm = sample_current_ycocg(in.uv + vec2( 0.0,          -texel_size.y));
    let s_br = sample_current_ycocg(in.uv + vec2( texel_size.x, -texel_size.y));

    let moment_1 = s_tl + s_tm + s_tr + s_ml + s_mm + s_mr + s_bl + s_bm + s_br;
    let moment_2 = (s_tl*s_tl) + (s_tm*s_tm) + (s_tr*s_tr)
                 + (s_ml*s_ml) + (s_mm*s_mm) + (s_mr*s_mr)
                 + (s_bl*s_bl) + (s_bm*s_bm) + (s_br*s_br);
    let mean          = moment_1 / 9.0;
    let variance      = (moment_2 / 9.0) - (mean * mean);
    let std_deviation = sqrt(max(variance, vec3(0.0)));

    history_color = RGB_to_YCoCg(history_color);
    history_color = clip_towards_aabb_center(history_color, s_mm, mean - std_deviation, mean + std_deviation);
    history_color = YCoCg_to_RGB(history_color);

    // ── Confidence-based blend rate ────────────────────────────────────────
    // Confidence is stored in history alpha (Rgba16Float — can exceed 1.0).
    // Static pixels: +10 per frame → confidence grows → blend rate approaches MIN.
    // Moving pixels: reset to 1 → blend rate = DEFAULT.
    // This is the Bevy/Hoppe approach: https://hhoppe.com/supersample.pdf, section 4.1
    //
    // IMPORTANT: This requires Rgba16Float history.  8-bit formats would clamp
    // confidence to 1.0, locking the blend rate at DEFAULT forever.
    var history_confidence = textureSampleLevel(history_frame, point_sampler, in.uv, 0.0).a;
    let pixel_motion = abs(closest_motion_vector) * texture_size;
    if pixel_motion.x < 0.01 && pixel_motion.y < 0.01 {
        history_confidence += 10.0;
    } else {
        history_confidence = 1.0;
    }

    var current_color_factor = clamp(1.0 / history_confidence, MIN_HISTORY_BLEND_RATE, DEFAULT_HISTORY_BLEND_RATE);

    // Reject history that reprojects outside the screen.
    if any(saturate(history_uv) != history_uv) {
        current_color_factor = 1.0;
        history_confidence = 1.0;
    }

    current_color = mix(history_color, current_color, current_color_factor);

    // Output: tonemapped color in RGB, confidence in alpha.
    // History texture is Rgba16Float so confidence values > 1.0 are preserved.
    // The blit pass sharpens these values and writes them directly to the display
    // surface — soft Reinhard tone mapping is already baked in, preserving specular
    // detail that would otherwise be hard-clipped by an 8-bit surface format.
    return vec4<f32>(current_color, history_confidence);
}
