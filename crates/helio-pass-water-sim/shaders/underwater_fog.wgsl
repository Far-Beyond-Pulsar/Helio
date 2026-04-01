// Fullscreen underwater post-process effect.
//
// Step 1: reads water_output (the fully composited scene) via scene_tex.
// Step 2: applies lens distortion, chromatic aberration, colour tint and
//         edge vignette to produce the final underwater look.
// Step 3: output is written to a scratch texture with alpha = 1 (full replace).
// Step 4: Rust side blits scratch → water_output.
//
// When the camera is above the water surface the scene is passed through
// unchanged (no cost above water except one texture fetch per pixel).
//
// Bindings
//   0  camera    uniform   Camera
//   1  volumes   storage   array<WaterVolume>
//   2  scene_tex texture2d — water_output bound as source
//   3  scene_samp sampler  — linear clamp

struct Camera {
    view:           mat4x4f,
    proj:           mat4x4f,
    view_proj:      mat4x4f,
    inv_view_proj:  mat4x4f,
    position_near:  vec4f,  // xyz = camera world position
    forward_far:    vec4f,
    jitter_frame:   vec4f,  // w = frame index (used as time)
    prev_view_proj: mat4x4f,
}

struct WaterVolume {
    bounds_min:            vec4f,
    bounds_max:            vec4f,  // w = surface_height
    wave_params:           vec4f,
    wave_direction:        vec4f,
    water_color:           vec4f,  // xyz = tint colour
    extinction:            vec4f,
    reflection_refraction: vec4f,
    caustics_params:       vec4f,
    fog_params:            vec4f,
    sim_params:            vec4f,
    shadow_params:         vec4f,
    sun_direction:         vec4f,
    ssr_params:            vec4f,
    pad1: vec4f, pad2: vec4f, pad3: vec4f,
}

@group(0) @binding(0) var<uniform>       camera:    Camera;
@group(0) @binding(1) var<storage, read> volumes:   array<WaterVolume>;
@group(0) @binding(2) var scene_tex:                texture_2d<f32>;
@group(0) @binding(3) var scene_samp:               sampler;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    var out: VertexOutput;
    out.position = vec4f(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2f(x, y);
    return out;
}

// ---------------------------------------------------------------------------
// Noise helpers for lens distortion
// ---------------------------------------------------------------------------

fn hash2(p: vec2f) -> vec2f {
    let k = vec2f(127.1, 311.7);
    let s = sin(dot(p, k)) * 43758.5453;
    return fract(vec2f(s, s + 0.618));
}

fn smooth_noise(p: vec2f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);  // smoothstep
    let a = dot(hash2(i          ), f - vec2f(0.0, 0.0));
    let b = dot(hash2(i + vec2f(1.0, 0.0)), f - vec2f(1.0, 0.0));
    let c = dot(hash2(i + vec2f(0.0, 1.0)), f - vec2f(0.0, 1.0));
    let d = dot(hash2(i + vec2f(1.0, 1.0)), f - vec2f(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y) * 0.5 + 0.5;
}

// Returns a 2-D warp offset for underwater lens wobble.
// uv  — screen UV in [0, 1]
// t   — time in seconds
fn water_distortion(uv: vec2f, t: f32) -> vec2f {
    let scale = 3.0;
    let p0 = uv * scale + vec2f(t * 0.11, t * 0.07);
    let p1 = uv * scale + vec2f(-t * 0.09, t * 0.13);
    let nx = smooth_noise(p0) - 0.5;
    let ny = smooth_noise(p1) - 0.5;
    return vec2f(nx, ny);
}

// ---------------------------------------------------------------------------
// Fragment shader
// ---------------------------------------------------------------------------

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    // ---- Find the volume the camera is actually inside -------------------
    // Must be within XYZ bounds and below the surface (bounds_max.w).
    let cam_pos = camera.position_near.xyz;
    var vol_idx: i32 = -1;
    for (var i = 0u; i < arrayLength(&volumes); i++) {
        let v         = volumes[i];
        let surface_h = v.bounds_max.w;
        // Y: below surface, above floor
        if cam_pos.y >= surface_h || cam_pos.y < v.bounds_min.y { continue; }
        // X bounds
        if cam_pos.x < v.bounds_min.x || cam_pos.x > v.bounds_max.x { continue; }
        // Z bounds
        if cam_pos.z < v.bounds_min.z || cam_pos.z > v.bounds_max.z { continue; }
        vol_idx = i32(i);
        break;
    }

    // Camera is not inside any water volume — pass scene through unchanged.
    if vol_idx < 0 {
        return textureSample(scene_tex, scene_samp, in.uv);
    }

    let vol       = volumes[u32(vol_idx)];
    let surface_h = vol.bounds_max.w;

    // ---- Depth below surface ---------------------------------------------
    let cam_depth = surface_h - cam_pos.y;

    // ---- Animated lens distortion ----------------------------------------
    // Time comes from the frame counter stored in jitter_frame.w.
    let t            = camera.jitter_frame.w * 0.016;
    // Two octaves for richer shape distortion — fast small ripples over
    // slower large swells, both scrolling at different angles.
    let dist0        = water_distortion(in.uv, t);
    let dist1        = water_distortion(in.uv * 2.1 + vec2f(0.37, 0.71), t * 0.6);
    let dist_raw     = dist0 * 0.7 + dist1 * 0.3;
    // 1% base so distortion is visible at the surface; ramps up to 8% deep.
    let dist_strength = clamp(cam_depth * 0.030, 0.010, 0.080);
    let warp_uv      = in.uv + dist_raw * dist_strength;
    // Keep UVs on-screen so we never sample outside the texture.
    let safe_uv      = clamp(warp_uv, vec2f(0.001), vec2f(0.999));

    // ---- Chromatic aberration -------------------------------------------
    // R and B channels are offset radially from the screen centre;
    // G uses the distorted UV directly.
    let ca_strength = clamp(cam_depth * 0.008, 0.001, 0.018);
    let centre = vec2f(0.5);
    let to_edge = safe_uv - centre;
    let r_uv = clamp(centre + to_edge * (1.0 + ca_strength),       vec2f(0.001), vec2f(0.999));
    let b_uv = clamp(centre + to_edge * (1.0 - ca_strength * 1.4), vec2f(0.001), vec2f(0.999));

    let r_col = textureSample(scene_tex, scene_samp, r_uv).r;
    let g_col = textureSample(scene_tex, scene_samp, safe_uv).g;
    let b_col = textureSample(scene_tex, scene_samp, b_uv).b;
    let scene_color = vec3f(r_col, g_col, b_col);

    // ---- Colour tint matching the water volume ---------------------------
    // Use vol.water_color with a blue-green fallback.
    let water_tint = max(vol.water_color.xyz, vec3f(0.01, 0.10, 0.28));

    // Beer-Lambert-style absorption: more tint the deeper the camera is.
    let tint_strength = clamp(0.40 + cam_depth * 0.055, 0.40, 0.72);
    let tinted = mix(scene_color, scene_color * water_tint, tint_strength);

    // ---- Edge vignette ---------------------------------------------------
    let d        = in.uv - 0.5;
    let vignette = 1.0 - dot(d, d) * 3.2;
    let vig_str  = clamp(cam_depth * 0.10, 0.0, 0.65);
    let vig_mult = mix(1.0, max(vignette, 0.0), vig_str);

    let out_color = tinted * vig_mult;

    // Full opaque output — scratch texture is blitted back by the Rust side.
    return vec4f(out_color, 1.0);
}
