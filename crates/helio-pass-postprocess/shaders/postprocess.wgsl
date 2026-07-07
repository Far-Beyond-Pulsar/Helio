// ── Helio Post-Processing Pipeline ─────────────────────────────────────────────
//
// Two bind groups:
//   @group(0) — main: uniforms, samplers, hdr/depth inputs, bloom sampled, avg_lum
//   @group(1) — bloom compute: per-dispatch src (sampled) + dst (storage write)
//
// Entry points:
//   cs_exposure           — compute: luminance histogram → average log-luminance
//   cs_bloom_down_extract — compute: extract brights from HDR → bloom mip 0
//   cs_bloom_down         — compute: 2x downsample from bloom_src → bloom_dst
//   vs_fullscreen         — vertex: fullscreen triangle
//   fs_uber               — fragment: tonemap + color grade + vignette + CA + grain
//
// Effect order (uber pass):
//   1. Exposure scale
//   2. Bloom composite (sum of all 5 mip levels)
//   3. Color grading (saturation, contrast, gamma, gain, offset)
//   4. White balance
//   5. Tonemapping
//   6. Vignette
//   7. Chromatic aberration
//   8. Film grain

// ── Constants ───────────────────────────────────────────────────────────────────

const PI: f32 = 3.14159265359;
const WG: u32 = 256u;
const BLOOM_MIPS: u32 = 5u;

// ── Uniforms ───────────────────────────────────────────────────────────────────

struct GpuPostProcessUniforms {
    // Exposure (16 bytes)
    exposure_mode:          u32,
    exposure_compensation:  f32,
    exposure_min:           f32,
    exposure_max:           f32,

    // Bloom (32 bytes)
    bloom_intensity:        f32,
    bloom_threshold:        f32,
    bloom_knee:             f32,
    bloom_radius:           f32,
    bloom_tint:             vec3<f32>,
    bloom_enabled:          u32,

    // Color grading (48 bytes)
    color_saturation:       vec3<f32>,
    _pad4:                  f32,
    color_contrast:         vec3<f32>,
    _pad5:                  f32,
    color_gamma:            vec3<f32>,
    _pad6:                  f32,
    color_gain:             vec3<f32>,
    _pad7:                  f32,
    color_offset:           vec3<f32>,
    _pad8:                  f32,

    // White balance (16 bytes)
    white_temp:             f32,
    white_tint:             f32,
    white_balance_enabled:  u32,
    _pad9:                  f32,

    // Tonemap (16 bytes)
    tonemap_operator:       u32,
    tonemap_exposure:       f32,
    tonemap_white_point:    f32,
    _pad10:                 f32,

    // Vignette (32 bytes)
    vignette_intensity:     f32,
    vignette_smoothness:    f32,
    vignette_roundness:     f32,
    _pad_vignette:          f32,
    vignette_color:         vec3<f32>,
    vignette_enabled:       u32,

    // Chromatic aberration (16 bytes)
    ca_intensity:           f32,
    ca_start_offset:        f32,
    ca_enabled:             u32,
    _pad11:                 f32,

    // Film grain (16 bytes)
    grain_intensity:        f32,
    grain_response:         f32,
    grain_size:             f32,
    grain_enabled:          u32,

    // Depth of Field (32 bytes)
    dof_focal_distance:     f32,
    dof_focal_region:       f32,
    dof_near_transition:    f32,
    dof_far_transition:     f32,
    dof_scale:              f32,
    dof_max_bokeh_size:     f32,
    dof_enabled:            u32,
    _pad12:                 f32,

    // Motion blur (16 bytes)
    motion_blur_amount:     f32,
    motion_blur_max:        f32,
    motion_blur_enabled:    u32,
    _pad13:                 f32,

    // Blend weights (32 bytes)
    blend_weight_bloom:        f32,
    blend_weight_dof:          f32,
    blend_weight_motion_blur:  f32,
    blend_weight_vignette:     f32,
    blend_weight_ca:           f32,
    blend_weight_grain:        f32,
    blend_weight_exposure:     f32,
    _pad14:                    f32,
}

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

// ── Group 0: main bindings (shared across all entry points) ────────────────────

@group(0) @binding(0)  var<uniform>            postprocess:  GpuPostProcessUniforms;
@group(0) @binding(1)  var<uniform>            camera:       CameraUniforms;
@group(0) @binding(2)  var                     hdr_input:    texture_2d<f32>;
@group(0) @binding(3)  var                     depth_input:  texture_depth_2d;
@group(0) @binding(4)  var                     linear_samp:  sampler;
@group(0) @binding(5)  var                     point_samp:   sampler;
// Bloom mips as sampled textures (read by fs_uber to composite bloom)
@group(0) @binding(6)  var                     bloom_0:      texture_2d<f32>;
@group(0) @binding(7)  var                     bloom_1:      texture_2d<f32>;
@group(0) @binding(8)  var                     bloom_2:      texture_2d<f32>;
@group(0) @binding(9)  var                     bloom_3:      texture_2d<f32>;
@group(0) @binding(10) var                     bloom_4:      texture_2d<f32>;
// Auto-exposure luminance average
@group(0) @binding(11) var<storage, read_write> avg_luminance: array<f32>;
// Custom effect infrastructure: noise texture + params buffer
@group(0) @binding(12) var                     noise_tex:    texture_2d<f32>;
@group(0) @binding(13) var                     noise_samp:   sampler;
@group(0) @binding(14) var<storage, read>      pp_custom:    array<vec4<f32>>;

// ── Group 1: per-dispatch bloom compute src/dst ────────────────────────────────
// Bound only during bloom compute passes. src unused by cs_bloom_down_extract.

@group(1) @binding(0) var bloom_src: texture_2d<f32>;
@group(1) @binding(1) var bloom_dst: texture_storage_2d<rgba16float, write>;

// ── Fullscreen vertex ──────────────────────────────────────────────────────────

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VOut {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    var out: VOut;
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv  = vec2<f32>(x, y);
    return out;
}

// ── Luminance / Exposure ───────────────────────────────────────────────────────

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// ── cs_exposure: histogram-based auto exposure ─────────────────────────────────

var<workgroup> wg_sum:   array<f32, 256>;
var<workgroup> wg_count: array<u32, 256>;

@compute @workgroup_size(16, 16)
fn cs_exposure(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let dims = textureDimensions(hdr_input);
    let w = dims.x;
    let h = dims.y;

    let stride = 4u;
    var sum_log: f32 = 0.0;
    var count: u32 = 0u;

    for (var y = gid.y * stride; y < h; y += stride * 16u) {
        for (var x = gid.x * stride; x < w; x += stride * 16u) {
            let col = textureLoad(hdr_input, vec2<i32>(i32(x), i32(y)), 0).rgb;
            let l = max(luminance(col), 0.0001);
            sum_log += log2(l);
            count++;
        }
    }

    let lidx = lid.y * 16u + lid.x;
    wg_sum[lidx] = sum_log;
    wg_count[lidx] = count;
    workgroupBarrier();

    var reduce_active = 128u;
    loop {
        if reduce_active == 0u { break; }
        if lidx < reduce_active {
            wg_sum[lidx] += wg_sum[lidx + reduce_active];
            wg_count[lidx] += wg_count[lidx + reduce_active];
        }
        workgroupBarrier();
        reduce_active >>= 1u;
    }

    if lidx == 0u && wg_count[0] > 0u {
        let avg_log = wg_sum[0] / f32(wg_count[0]);
        avg_luminance[0] = avg_log;
    }
}

// ── cs_bloom_down_extract: extract brights from HDR → mip 0 ───────────────────
// Reads from hdr_input (group 0), writes to bloom_dst (group 1).
// Dispatched once for mip 0 with dimensions = hdr / 2.

@compute @workgroup_size(8, 8)
fn cs_bloom_down_extract(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_dims = textureDimensions(bloom_dst);
    let ix = i32(gid.x);
    let iy = i32(gid.y);
    if ix >= i32(dst_dims.x) || iy >= i32(dst_dims.y) { return; }

    let hdr_dims = textureDimensions(hdr_input);
    let hw = i32(hdr_dims.x);
    let hh = i32(hdr_dims.y);

    var color = vec3<f32>(0.0);
    for (var dy = 0i; dy < 2; dy++) {
        for (var dx = 0i; dx < 2; dx++) {
            let sx = ix * 2 + dx;
            let sy = iy * 2 + dy;
            if sx < hw && sy < hh {
                color += textureLoad(hdr_input, vec2<i32>(sx, sy), 0).rgb;
            }
        }
    }
    color *= 0.25;

    let l = luminance(color);
    let knee = postprocess.bloom_knee;
    let thresh = postprocess.bloom_threshold;
    var excess: f32;
    if l <= thresh - knee {
        excess = 0.0;
    } else if l >= thresh {
        excess = l - thresh;
    } else {
        let t = (l - (thresh - knee)) / knee;
        excess = t * t * knee * 0.25;
    }
    var brights = color * (excess / max(l, 0.0001));
    brights *= postprocess.bloom_intensity * postprocess.blend_weight_bloom;
    textureStore(bloom_dst, vec2<i32>(ix, iy), vec4<f32>(brights * postprocess.bloom_tint, 0.0));
}

// ── cs_bloom_down: 2x downsample bloom_src → bloom_dst ────────────────────────
// Dispatched for mips 1-4, each with a separate per-mip bind group for group 1.

@compute @workgroup_size(8, 8)
fn cs_bloom_down(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_dims = textureDimensions(bloom_dst);
    let ix = i32(gid.x);
    let iy = i32(gid.y);
    if ix >= i32(dst_dims.x) || iy >= i32(dst_dims.y) { return; }

    let src_dims = textureDimensions(bloom_src);
    let sw = i32(src_dims.x);
    let sh = i32(src_dims.y);

    var color = vec3<f32>(0.0);
    for (var dy = 0i; dy < 2; dy++) {
        for (var dx = 0i; dx < 2; dx++) {
            let sx = ix * 2 + dx;
            let sy = iy * 2 + dy;
            if sx < sw && sy < sh {
                color += textureLoad(bloom_src, vec2<i32>(sx, sy), 0).rgb;
            }
        }
    }
    textureStore(bloom_dst, vec2<i32>(ix, iy), vec4<f32>(color * 0.25, 0.0));
}

// ── Tonemapping operators ──────────────────────────────────────────────────────

fn tonemap_aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

fn tonemap_filmic(x: vec3<f32>) -> vec3<f32> {
    let a = vec3<f32>(0.15); let b = vec3<f32>(0.50);
    let c = vec3<f32>(0.10); let d = vec3<f32>(0.20);
    let e = vec3<f32>(0.02); let f = vec3<f32>(0.30);
    return saturate(((x * (a * x + c * b) + d * e)) / (x * (a * x + b) + d * f) - e / f);
}

fn tonemap_reinhard(x: vec3<f32>) -> vec3<f32> {
    return x / (1.0 + x);
}

fn uncharted2_curve(v: vec3<f32>) -> vec3<f32> {
    let A = 0.15; let B = 0.50; let C = 0.10; let D = 0.20;
    let E = 0.02; let F = 0.30;
    return ((v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F)) - E / F;
}

fn tonemap_uncharted2(x: vec3<f32>) -> vec3<f32> {
    let W = 11.2;
    let white_scale = 1.0 / uncharted2_curve(vec3<f32>(W));
    return saturate(uncharted2_curve(x) * white_scale);
}

fn lottes_curve(v: vec3<f32>, a: f32, b: f32, c: f32, d: f32) -> vec3<f32> {
    return ((v * (a * v + b)) / (v * (a - 1.0) * v + (b + 1.0))) * c + d;
}

fn tonemap_lottes(x: vec3<f32>) -> vec3<f32> {
    let a = 1.6; let d = 0.977;
    let mid_in = 0.18;
    let mid_out = 0.267;
    let b = (-d * mid_in + (a - 1.0) * mid_out) / ((a - 1.0) * d * mid_in + mid_out);
    let c = (a * d * mid_in + (a - 1.0) * b * mid_out) / ((a - 1.0) * d * mid_in + mid_out);
    return saturate(lottes_curve(x, a, b, c, d));
}

fn apply_tonemap(color: vec3<f32>) -> vec3<f32> {
    let op = postprocess.tonemap_operator;
    var c = color * postprocess.tonemap_exposure;
    c = c / postprocess.tonemap_white_point;
    if op == 0u { return tonemap_aces(c); }
    if op == 1u { return tonemap_filmic(c); }
    if op == 2u { return tonemap_reinhard(c); }
    if op == 3u { return tonemap_uncharted2(c); }
    if op == 4u { return tonemap_lottes(c); }
    return tonemap_aces(c);
}

// ── Color grading ──────────────────────────────────────────────────────────────

fn color_grade(color: vec3<f32>) -> vec3<f32> {
    var c = color;
    c = c * postprocess.color_gain + postprocess.color_offset;
    c = pow(max(c, vec3<f32>(0.0)), postprocess.color_gamma);
    c = c * postprocess.color_contrast;
    c = c * postprocess.color_saturation;
    return c;
}

// ── White balance ──────────────────────────────────────────────────────────────

fn white_balance(color: vec3<f32>) -> vec3<f32> {
    if postprocess.white_balance_enabled == 0u { return color; }
    let temp = postprocess.white_temp * 0.0001;
    let r = 1.0 / max(temp, 0.001);
    let g = 1.0;
    let b = temp;
    let tint = postprocess.white_tint;
    return color * vec3<f32>(r * (1.0 - tint), g, b * (1.0 + tint));
}

// ── Vignette ───────────────────────────────────────────────────────────────────

fn apply_vignette(color: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    if postprocess.vignette_enabled == 0u { return color; }
    let center = uv - 0.5;
    let dist = length(center * vec2<f32>(1.0 / max(postprocess.vignette_roundness, 0.001), 1.0));
    let vignette = 1.0 - saturate(dist * postprocess.vignette_smoothness) * postprocess.vignette_intensity;
    return mix(postprocess.vignette_color, color, vignette);
}

// ── Chromatic aberration ───────────────────────────────────────────────────────

fn apply_ca(color: vec3<f32>, uv: vec2<f32>, dims: vec2<f32>) -> vec3<f32> {
    if postprocess.ca_enabled == 0u { return color; }
    let center = uv - 0.5;
    let dist = length(center);
    let offset = max(dist - postprocess.ca_start_offset, 0.0) * postprocess.ca_intensity;
    let dir = normalize(center);
    let r_uv = uv + dir * offset * (1.0 / dims);
    let b_uv = uv - dir * offset * (1.0 / dims);
    let r = textureSampleLevel(hdr_input, linear_samp, r_uv, 0.0).r;
    let g = color.g;
    let b = textureSampleLevel(hdr_input, linear_samp, b_uv, 0.0).b;
    return vec3<f32>(r, g, b);
}

// ── Film grain ─────────────────────────────────────────────────────────────────

fn hash(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

fn apply_grain(color: vec3<f32>, uv: vec2<f32>, frame: f32) -> vec3<f32> {
    if postprocess.grain_enabled == 0u { return color; }
    let gsize = max(postprocess.grain_size, 0.01);
    let g_uv = uv * vec2<f32>(textureDimensions(hdr_input)) / gsize;
    let grain = hash(g_uv + frame * 0.001) * 2.0 - 1.0;
    let l = luminance(color);
    let amount = postprocess.grain_intensity * pow(1.0 - l, postprocess.grain_response);
    return color + grain * amount;
}

// ── Depth of Field (Gaussian approximation) ────────────────────────────────────

fn dof_coc(depth: f32) -> f32 {
    let linear_depth = -camera.proj[3][2] / (depth * 2.0 - 1.0 + camera.proj[2][2]);
    let focal_dist = postprocess.dof_focal_distance;
    let focal_region = postprocess.dof_focal_region;
    let near_blur = max(focal_dist - focal_region - linear_depth, 0.0) / max(postprocess.dof_near_transition, 0.001);
    let far_blur = max(linear_depth - (focal_dist + focal_region), 0.0) / max(postprocess.dof_far_transition, 0.001);
    let coc = max(near_blur, far_blur) * postprocess.dof_scale;
    return clamp(coc, 0.0, postprocess.dof_max_bokeh_size);
}

fn apply_dof(color: vec3<f32>, uv: vec2<f32>, depth: f32, dims: vec2<f32>) -> vec3<f32> {
    if postprocess.dof_enabled == 0u { return color; }
    let coc = dof_coc(depth) * postprocess.blend_weight_dof;
    if coc < 0.5 { return color; }
    let radius = clamp(coc, 1.0, postprocess.dof_max_bokeh_size);
    let taps = 7u;
    let step = radius / f32(taps);
    var blurred = vec3<f32>(0.0);
    var total = 0.0;
    for (var dy = -(i32(taps) / 2); dy <= i32(taps) / 2; dy++) {
        for (var dx = -(i32(taps) / 2); dx <= i32(taps) / 2; dx++) {
            let offset = vec2<f32>(f32(dx), f32(dy)) * step * (1.0 / dims);
            let tap = textureSampleLevel(hdr_input, linear_samp, uv + offset, 0.0).rgb;
            let w = exp(-f32(dx * dx + dy * dy) / (2.0 * radius * 0.5));
            blurred += tap * w;
            total += w;
        }
    }
    if total > 0.0 { blurred /= total; }
    return mix(color, blurred, clamp(coc / postprocess.dof_max_bokeh_size, 0.0, 1.0));
}

// ── Motion blur ────────────────────────────────────────────────────────────────

fn apply_motion_blur(color: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    if postprocess.motion_blur_enabled == 0u { return color; }
    var blurred = color;
    let samples = 8u;
    let amount = postprocess.motion_blur_amount * postprocess.blend_weight_motion_blur;
    if amount <= 0.0 { return color; }
    let velocity = vec2<f32>(amount, 0.0);
    let max_len = postprocess.motion_blur_max / f32(textureDimensions(hdr_input).x);
    for (var i = 1u; i < samples; i++) {
        let t = f32(i) / f32(samples);
        let sample_uv = uv - velocity * t * max_len;
        blurred += textureSampleLevel(hdr_input, linear_samp, sample_uv, 0.0).rgb;
    }
    return blurred / f32(samples + 1u);
}

// ── fs_uber ────────────────────────────────────────────────────────────────────

@fragment
fn fs_uber(in: VOut) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(hdr_input));
    let uv = in.uv;

    var color = textureSampleLevel(hdr_input, linear_samp, uv, 0.0).rgb;

    // User-defined effects
    color = user_effects(color, uv, dims);

    return vec4<f32>(color, 1.0);
}
