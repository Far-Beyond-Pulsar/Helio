//! Final post-process pass.
//!
//! Reads the HDR Rgba16Float scene-colour buffer and outputs LDR to the swap-chain.
//! Applies (in order):
//!   1. Auto-exposure  – scales by `exposure` uniform
//!   2. ACES filmic    – Hill/Narkowicz approximation
//!   3. Color grading  – lift/gamma/gain + saturation + contrast
//!   4. Vignette       – subtle radial darkening
//!   5. Film grain     – animated Gaussian noise
//!   6. Chromatic aberration – barrel-distort R/B channels

struct PostProcessUniforms {
    exposure:          f32,
    // Grading
    saturation:        f32,
    contrast:          f32,
    // Vignette
    vignette_strength: f32,
    vignette_radius:   f32,
    // Grain
    grain_strength:    f32,
    // Chromatic aberration
    ca_strength:       f32,
    // Frame counter for animated grain
    frame:             u32,
}

@group(0) @binding(0) var hdr_texture:  texture_2d<f32>;
@group(0) @binding(1) var hdr_sampler:  sampler;
@group(0) @binding(2) var <uniform> pp: PostProcessUniforms;

// ── Fullscreen triangle ───────────────────────────────────────────────────────

struct VSOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)      uv:       vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0,-1.0),
    );
    var out: VSOut;
    out.clip_pos = vec4<f32>(pos[vi], 0.0, 1.0);
    out.uv       = uvs[vi];
    return out;
}

// ── Tonemapping ───────────────────────────────────────────────────────────────

fn aces_filmic(x: vec3<f32>) -> vec3<f32> {
    // Hill modified ACES (same coefficients as UE5 default)
    let a = 2.51;  let b = 0.03;
    let c = 2.43;  let d = 0.59;  let e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

// ── Color grading ─────────────────────────────────────────────────────────────

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn apply_grading(color: vec3<f32>) -> vec3<f32> {
    // Saturation: mix toward luminance
    let lum = luminance(color);
    var c   = mix(vec3<f32>(lum), color, pp.saturation);
    // Contrast: S-curve, pivot at 0.5
    let mid = 0.5;
    c = clamp(mid + (c - mid) * pp.contrast, vec3<f32>(0.0), vec3<f32>(1.0));
    return c;
}

// ── Vignette ──────────────────────────────────────────────────────────────────

fn vignette(uv: vec2<f32>) -> f32 {
    let d = length(uv * 2.0 - 1.0);
    return 1.0 - smoothstep(pp.vignette_radius, pp.vignette_radius + 0.4, d) * pp.vignette_strength;
}

// ── Film grain ────────────────────────────────────────────────────────────────
// Simple hash-based high-frequency noise

fn hash(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(127.1, 311.7));
    q += dot(q, q.yx + 19.19);
    return fract(q.x * q.y);
}

fn film_grain(uv: vec2<f32>, frame: u32) -> f32 {
    // Jitter UV per frame so grain is animated
    let jitter = vec2<f32>(f32(frame % 101u) * 0.01234, f32(frame % 73u) * 0.01876);
    let noise  = hash(uv * 1024.0 + jitter);
    return (noise - 0.5) * pp.grain_strength;
}

// ── Chromatic aberration ──────────────────────────────────────────────────────

fn sample_ca(uv: vec2<f32>) -> vec3<f32> {
    // R channel displaced outward, B channel inward along radial direction
    let center = uv - 0.5;
    let r = textureSample(hdr_texture, hdr_sampler, 0.5 + center * (1.0 + pp.ca_strength)).r;
    let g = textureSample(hdr_texture, hdr_sampler, uv).g;
    let b = textureSample(hdr_texture, hdr_sampler, 0.5 + center * (1.0 - pp.ca_strength)).b;
    return vec3<f32>(r, g, b);
}

// ── Fragment ──────────────────────────────────────────────────────────────────

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let uv = in.uv;

    // 1. Sample HDR scene colour (with chromatic aberration on r/b channels)
    var hdr = sample_ca(uv);

    // 2. Auto-exposure
    hdr *= pp.exposure;

    // 3. ACES filmic tonemapping
    var ldr = aces_filmic(hdr);

    // 4. Color grading (post-tonemap, in gamma-ish space)
    ldr = apply_grading(ldr);

    // 5. Vignette
    ldr *= vignette(uv);

    // 6. Film grain (additive, tiny amplitude)
    ldr += film_grain(uv, pp.frame);

    return vec4<f32>(clamp(ldr, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
