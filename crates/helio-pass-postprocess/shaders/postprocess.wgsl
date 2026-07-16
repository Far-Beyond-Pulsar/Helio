// ── Helio Post-Processing Pipeline ─────────────────────────────────────────────
//
// Bind groups:
//   @group(0) — main: uniforms, samplers, hdr/depth inputs, bloom sampled, avg_lum,
//               noise, custom params, volume data, blend output
//   @group(1) — bloom compute: per-dispatch src (sampled) + dst (storage write)
//
// Entry points:
//   cs_exposure              — compute: luminance histogram → avg log-luminance
//   cs_volume_blend          — compute: blend active post-process volumes → output
//   cs_bloom_down_extract    — compute: extract brights from HDR → bloom mip 0
//   cs_bloom_down            — compute: 2x downsample from bloom_src → bloom_dst
//   vs_fullscreen            — vertex: fullscreen triangle
//   fs_uber                  — fragment: effects chain (see INJECTION_POINT markers)
//
// Effect order (uber pass):
//   INJECTION_POINT_0  — user effects (pre-blend)
//   1. Exposure scale
//   2. Bloom composite
//   3. Color grading
//   4. White balance
//   5. Tonemapping
//   INJECTION_POINT_1  — user effects (post-tonemap)
//   6. Vignette
//   7. Chromatic aberration
//   8. Film grain
//   INJECTION_POINT_2  — user effects (post-grain)
//   9. Depth of Field
//   10. Motion blur
//   INJECTION_POINT_3  — user effects (final)

// ── Constants ───────────────────────────────────────────────────────────────────

const PI: f32 = 3.14159265359;
const WG_BLOOM: u32 = 8u;
const WG_EXPOSURE_X: u32 = 16u;
const WG_EXPOSURE_Y: u32 = 16u;
const MAX_PP_VOLUMES: u32 = 256u;

// ── GpuPostProcessUniforms ─────────────────────────────────────────────────────
// Matches CPU-side layout in libhelio/src/postprocess.rs

struct GpuPostProcessUniforms {
    exposure_mode:          u32,
    exposure_compensation:  f32,
    exposure_min:           f32,
    exposure_max:           f32,
    bloom_intensity:        f32,
    bloom_threshold:        f32,
    bloom_knee:             f32,
    bloom_radius:           f32,
    bloom_tint:             vec3<f32>,
    bloom_enabled:          u32,
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
    white_temp:             f32,
    white_tint:             f32,
    white_balance_enabled:  u32,
    _pad9:                  f32,
    tonemap_operator:       u32,
    tonemap_exposure:       f32,
    tonemap_white_point:    f32,
    _pad10:                 f32,
    vignette_intensity:     f32,
    vignette_smoothness:    f32,
    vignette_roundness:     f32,
    _pad_vignette:          f32,
    vignette_color:         vec3<f32>,
    vignette_enabled:       u32,
    ca_intensity:           f32,
    ca_start_offset:        f32,
    ca_enabled:             u32,
    _pad11:                 f32,
    grain_intensity:        f32,
    grain_response:         f32,
    grain_size:             f32,
    grain_enabled:          u32,
    dof_focal_distance:     f32,
    dof_focal_region:       f32,
    dof_near_transition:    f32,
    dof_far_transition:     f32,
    dof_scale:              f32,
    dof_max_bokeh_size:     f32,
    dof_enabled:            u32,
    _pad12:                 f32,
    motion_blur_amount:     f32,
    motion_blur_max:        f32,
    motion_blur_enabled:    u32,
    _pad13:                 f32,
    blend_weight_bloom:        f32,
    blend_weight_dof:          f32,
    blend_weight_motion_blur:  f32,
    blend_weight_vignette:     f32,
    blend_weight_ca:           f32,
    blend_weight_grain:        f32,
    blend_weight_exposure:     f32,
    _pad14:                    f32,
    // ── Volumetric fog (64 bytes, offsets 304..368) ──
    // fog_color lands at 336 and fog_emissive at 352 — both multiples of 16, which
    // is what lets this match #[repr(C)] on the CPU. vec3<f32> aligns to 16 in WGSL
    // but to 4 in Rust, so reordering these fields silently desyncs the two sides.
    fog_enabled:               u32,   // 304
    fog_mode:                  u32,   // 308
    fog_density:               f32,   // 312
    fog_height_falloff:        f32,   // 316
    fog_start_distance:        f32,   // 320
    fog_max_distance:          f32,   // 324
    fog_height:                f32,   // 328
    fog_scattering_anisotropy: f32,   // 332
    fog_color:                 vec3<f32>, // 336
    _pad_fog_color:            f32,   // 348
    fog_emissive:              vec3<f32>, // 352
    _pad_fog_emissive:         f32,   // 364 → struct ends at 368
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

// ── GpuPostProcessVolume (matches CPU-side layout) ─────────────────────────────

struct GpuPostProcessVolume {
    bounds_min:     vec4<f32>,   // 0
    bounds_max:     vec4<f32>,   // 16
    priority:       f32,         // 32
    blend_radius:   f32,         // 36
    blend_weight:   f32,         // 40
    unbound:        u32,         // 44
    // vec4, not vec2: `settings` contains vec3s so it aligns to 16 and lands at
    // 64 regardless. Spelling the pad out to 64 keeps this struct honest about
    // where settings actually starts — the CPU side has to pad to match, and a
    // vec2 here silently hid an 8-byte hole that #[repr(C)] did not reproduce.
    _pad_vol:       vec4<f32>,   // 48..64
    settings:       GpuPostProcessUniforms,  // 64
}

// ── Group 0: main bindings ─────────────────────────────────────────────────────

@group(0) @binding(0)  var<uniform>            postprocess:  GpuPostProcessUniforms;
@group(0) @binding(1)  var<uniform>            camera:       CameraUniforms;
@group(0) @binding(2)  var                     hdr_input:    texture_2d<f32>;
@group(0) @binding(3)  var                     depth_input:  texture_depth_2d;
@group(0) @binding(4)  var                     linear_samp:  sampler;
@group(0) @binding(5)  var                     point_samp:   sampler;
@group(0) @binding(6)  var                     bloom_0:      texture_2d<f32>;
@group(0) @binding(7)  var                     bloom_1:      texture_2d<f32>;
@group(0) @binding(8)  var                     bloom_2:      texture_2d<f32>;
@group(0) @binding(9)  var                     bloom_3:      texture_2d<f32>;
@group(0) @binding(10) var                     bloom_4:      texture_2d<f32>;
@group(0) @binding(11) var<storage, read_write> avg_luminance: array<f32>;
@group(0) @binding(12) var                     noise_tex:    texture_2d<f32>;
@group(0) @binding(13) var                     noise_samp:   sampler;
@group(0) @binding(14) var<storage, read>      pp_custom:    array<vec4<f32>>;
@group(0) @binding(15) var<storage, read>      pp_volumes:   array<GpuPostProcessVolume>;
@group(0) @binding(16) var<storage, read_write> blend_output: GpuPostProcessUniforms;
// Volumetric fog accumulation at quarter internal resolution (fs_uber only).
// rgb = in-scattered radiance, a = transmittance. Bound to a 1x1 (0,0,0,1)
// fallback when no fog pass is in the graph, which composites to a no-op.
@group(0) @binding(17) var                     fog_input:    texture_2d<f32>;

// ── Group 1: per-dispatch bloom compute src/dst ────────────────────────────────

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

// ── Luminance ──────────────────────────────────────────────────────────────────

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// ── cs_volume_blend: GPU post-process volume blending ─────────────────────────
// Single workgroup (1 thread) that reads all active volumes and blends them
// with camera defaults, writing the result to blend_output.

var<workgroup> vol_weights: array<f32, 256>;
var<workgroup> vol_indices: array<u32, 256>;

fn lerpf(a: f32, b: f32, t: f32) -> f32 { return a + (b - a) * t; }
fn lerp3v(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> { return a + (b - a) * t; }

fn blend_settings(base: GpuPostProcessUniforms, vol: GpuPostProcessUniforms, t: f32) -> GpuPostProcessUniforms {
    var r: GpuPostProcessUniforms;
    r.exposure_mode          = select(base.exposure_mode, vol.exposure_mode, t > 0.5);
    r.exposure_compensation  = lerpf(base.exposure_compensation, vol.exposure_compensation, t);
    r.exposure_min           = lerpf(base.exposure_min, vol.exposure_min, t);
    r.exposure_max           = lerpf(base.exposure_max, vol.exposure_max, t);
    r.bloom_intensity        = lerpf(base.bloom_intensity, vol.bloom_intensity, t);
    r.bloom_threshold        = lerpf(base.bloom_threshold, vol.bloom_threshold, t);
    r.bloom_knee             = lerpf(base.bloom_knee, vol.bloom_knee, t);
    r.bloom_radius           = lerpf(base.bloom_radius, vol.bloom_radius, t);
    r.bloom_tint             = lerp3v(base.bloom_tint, vol.bloom_tint, t);
    r.bloom_enabled          = select(base.bloom_enabled, vol.bloom_enabled, t > 0.5);
    r.color_saturation       = lerp3v(base.color_saturation, vol.color_saturation, t);
    r.color_contrast         = lerp3v(base.color_contrast, vol.color_contrast, t);
    r.color_gamma            = lerp3v(base.color_gamma, vol.color_gamma, t);
    r.color_gain             = lerp3v(base.color_gain, vol.color_gain, t);
    r.color_offset           = lerp3v(base.color_offset, vol.color_offset, t);
    r.white_temp             = lerpf(base.white_temp, vol.white_temp, t);
    r.white_tint             = lerpf(base.white_tint, vol.white_tint, t);
    r.white_balance_enabled  = select(base.white_balance_enabled, vol.white_balance_enabled, t > 0.5);
    r.tonemap_operator       = select(base.tonemap_operator, vol.tonemap_operator, t > 0.5);
    r.tonemap_exposure       = lerpf(base.tonemap_exposure, vol.tonemap_exposure, t);
    r.tonemap_white_point    = lerpf(base.tonemap_white_point, vol.tonemap_white_point, t);
    r.vignette_intensity     = lerpf(base.vignette_intensity, vol.vignette_intensity, t);
    r.vignette_smoothness    = lerpf(base.vignette_smoothness, vol.vignette_smoothness, t);
    r.vignette_roundness     = lerpf(base.vignette_roundness, vol.vignette_roundness, t);
    r.vignette_color         = lerp3v(base.vignette_color, vol.vignette_color, t);
    r.vignette_enabled       = select(base.vignette_enabled, vol.vignette_enabled, t > 0.5);
    r.ca_intensity           = lerpf(base.ca_intensity, vol.ca_intensity, t);
    r.ca_start_offset        = lerpf(base.ca_start_offset, vol.ca_start_offset, t);
    r.ca_enabled             = select(base.ca_enabled, vol.ca_enabled, t > 0.5);
    r.grain_intensity        = lerpf(base.grain_intensity, vol.grain_intensity, t);
    r.grain_response         = lerpf(base.grain_response, vol.grain_response, t);
    r.grain_size             = lerpf(base.grain_size, vol.grain_size, t);
    r.grain_enabled          = select(base.grain_enabled, vol.grain_enabled, t > 0.5);
    r.dof_focal_distance     = lerpf(base.dof_focal_distance, vol.dof_focal_distance, t);
    r.dof_focal_region       = lerpf(base.dof_focal_region, vol.dof_focal_region, t);
    r.dof_near_transition    = lerpf(base.dof_near_transition, vol.dof_near_transition, t);
    r.dof_far_transition     = lerpf(base.dof_far_transition, vol.dof_far_transition, t);
    r.dof_scale              = lerpf(base.dof_scale, vol.dof_scale, t);
    r.dof_max_bokeh_size     = lerpf(base.dof_max_bokeh_size, vol.dof_max_bokeh_size, t);
    r.dof_enabled            = select(base.dof_enabled, vol.dof_enabled, t > 0.5);
    r.motion_blur_amount     = lerpf(base.motion_blur_amount, vol.motion_blur_amount, t);
    r.motion_blur_max        = lerpf(base.motion_blur_max, vol.motion_blur_max, t);
    r.motion_blur_enabled    = select(base.motion_blur_enabled, vol.motion_blur_enabled, t > 0.5);
    r.blend_weight_bloom        = lerpf(base.blend_weight_bloom, vol.blend_weight_bloom, t);
    r.blend_weight_dof          = lerpf(base.blend_weight_dof, vol.blend_weight_dof, t);
    r.blend_weight_motion_blur  = lerpf(base.blend_weight_motion_blur, vol.blend_weight_motion_blur, t);
    r.blend_weight_vignette     = lerpf(base.blend_weight_vignette, vol.blend_weight_vignette, t);
    r.blend_weight_ca           = lerpf(base.blend_weight_ca, vol.blend_weight_ca, t);
    r.blend_weight_grain        = lerpf(base.blend_weight_grain, vol.blend_weight_grain, t);
    r.blend_weight_exposure     = lerpf(base.blend_weight_exposure, vol.blend_weight_exposure, t);
    // Fog. Every field must be assigned: `r` is declared uninitialized, so a field
    // left unwritten here is indeterminate — and the caller copies this whole struct
    // over the post-process uniform buffer, so a miss would corrupt fog config for
    // every frame in which any post-process volume is active.
    r.fog_enabled               = select(base.fog_enabled, vol.fog_enabled, t > 0.5);
    r.fog_mode                  = select(base.fog_mode, vol.fog_mode, t > 0.5);
    r.fog_density               = lerpf(base.fog_density, vol.fog_density, t);
    r.fog_height_falloff        = lerpf(base.fog_height_falloff, vol.fog_height_falloff, t);
    r.fog_start_distance        = lerpf(base.fog_start_distance, vol.fog_start_distance, t);
    r.fog_max_distance          = lerpf(base.fog_max_distance, vol.fog_max_distance, t);
    r.fog_height                = lerpf(base.fog_height, vol.fog_height, t);
    r.fog_scattering_anisotropy = lerpf(base.fog_scattering_anisotropy, vol.fog_scattering_anisotropy, t);
    r.fog_color                 = lerp3v(base.fog_color, vol.fog_color, t);
    r.fog_emissive              = lerp3v(base.fog_emissive, vol.fog_emissive, t);
    // Padding fields are implicitly copied via the field-by-field assignment above.
    // The struct is fully written by this function; uninitialized fields get default values.
    r._pad4 = 0.0; r._pad5 = 0.0; r._pad6 = 0.0; r._pad7 = 0.0; r._pad8 = 0.0;
    r._pad9 = 0.0; r._pad10 = 0.0; r._pad_vignette = 0.0; r._pad11 = 0.0;
    r._pad12 = 0.0; r._pad13 = 0.0; r._pad14 = 0.0;
    r._pad_fog_color = 0.0; r._pad_fog_emissive = 0.0;
    return r;
}

@compute @workgroup_size(1, 1, 1)
fn cs_volume_blend(@builtin(local_invocation_index) lid: u32) {
    let cam_pos = camera.position_near.xyz;
    var vol_count: u32 = 0u;

    // Phase 1: evaluate all active volumes, store weight + index
    for (var i = 0u; i < MAX_PP_VOLUMES; i++) {
        let v = pp_volumes[i];
        if v.blend_weight <= 0.0 { continue; }

        if v.unbound == 0u {
            // Bounded volume — camera must be inside AABB
            let inside = cam_pos.x >= v.bounds_min.x && cam_pos.x <= v.bounds_max.x
                      && cam_pos.y >= v.bounds_min.y && cam_pos.y <= v.bounds_max.y
                      && cam_pos.z >= v.bounds_min.z && cam_pos.z <= v.bounds_max.z;
            if !inside { continue; }

            let clamped = vec3<f32>(
                clamp(cam_pos.x, v.bounds_min.x, v.bounds_max.x),
                clamp(cam_pos.y, v.bounds_min.y, v.bounds_max.y),
                clamp(cam_pos.z, v.bounds_min.z, v.bounds_max.z),
            );
            let dx = cam_pos.x - clamped.x;
            let dy = cam_pos.y - clamped.y;
            let dz = cam_pos.z - clamped.z;
            let dist = sqrt(dx * dx + dy * dy + dz * dz);
            let blend_dist = max(v.blend_radius, 0.001);
            let inside_weight = 1.0 - clamp(dist / blend_dist, 0.0, 1.0);
            if inside_weight <= 0.0 { continue; }

            vol_weights[vol_count] = inside_weight * v.blend_weight;
        } else {
            // Unbound volume — always applies at full blend_weight
            vol_weights[vol_count] = v.blend_weight;
        }
        vol_indices[vol_count] = i;
        vol_count++;
    }

    if vol_count == 0u {
        blend_output = postprocess;
        return;
    }

    // Phase 2: sort active volumes by priority descending (insertion sort)
    for (var si = 0u; si < vol_count; si++) {
        for (var sj = si + 1u; sj < vol_count; sj++) {
            let pri_i = pp_volumes[vol_indices[si]].priority;
            let pri_j = pp_volumes[vol_indices[sj]].priority;
            if (pri_j > pri_i) {
                let tmp_w = vol_weights[si];
                let tmp_i = vol_indices[si];
                vol_weights[si] = vol_weights[sj];
                vol_indices[si] = vol_indices[sj];
                vol_weights[sj] = tmp_w;
                vol_indices[sj] = tmp_i;
            }
        }
    }

    // Phase 3: hierarchical blend from camera defaults toward each volume
    var result: GpuPostProcessUniforms = postprocess;
    var total_weight = 1.0;

    for (var vi = 0u; vi < vol_count; vi++) {
        let w = vol_weights[vi];
        let t = clamp(w / (total_weight + w), 0.0, 1.0);
        result = blend_settings(result, pp_volumes[vol_indices[vi]].settings, t);
        total_weight += w;
    }

    blend_output = result;
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
    if op == 5u { return color; } // None — skip
    var c = color * postprocess.tonemap_exposure;
    c = c / postprocess.tonemap_white_point;
    if op == 0u { return tonemap_aces(c); }
    if op == 1u { return tonemap_filmic(c); }
    if op == 2u { return tonemap_reinhard(c); }
    if op == 3u { return tonemap_uncharted2(c); }
    if op == 4u { return tonemap_lottes(c); }
    return c; // fallback: pass through
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

fn apply_grain(color: vec3<f32>, uv: vec2<f32>, dims: vec2<f32>) -> vec3<f32> {
    if postprocess.grain_enabled == 0u { return color; }
    let gsize = max(postprocess.grain_size, 0.01);
    let g_uv = uv * dims / gsize;
    let grain = hash(g_uv) * 2.0 - 1.0;
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

    // 0. Volumetric fog composite.
    //
    // Before exposure and tonemapping, not after: in-scattering is scene-linear
    // radiance like anything else in hdr_input, so it has to go through the same
    // exposure and tonemap curve. Composited after the tonemapper it would sit in
    // display-referred space — unresponsive to exposure, washed out, and invisible
    // to bloom, so bright shafts could never bloom.
    //
    // fog.rgb is already premultiplied by the transmittance in front of it, so this
    // is a straight over: attenuate the scene, add what scattered in.
    if postprocess.fog_enabled != 0u {
        let fog = textureSampleLevel(fog_input, linear_samp, uv, 0.0);
        color = color * fog.a + fog.rgb;
    }

    //%P0

    // 1. Exposure
    color *= exp2(postprocess.exposure_compensation);

    // 2. Bloom composite
    if postprocess.bloom_enabled != 0u && postprocess.bloom_intensity > 0.0 {
        var bloom = vec3<f32>(0.0);
        bloom += textureSampleLevel(bloom_0, linear_samp, uv, 0.0).rgb;
        bloom += textureSampleLevel(bloom_1, linear_samp, uv, 0.0).rgb;
        bloom += textureSampleLevel(bloom_2, linear_samp, uv, 0.0).rgb;
        bloom += textureSampleLevel(bloom_3, linear_samp, uv, 0.0).rgb;
        bloom += textureSampleLevel(bloom_4, linear_samp, uv, 0.0).rgb;
        color += bloom;
    }

    // 3. Color grading
    color = color_grade(color);

    // 4. White balance
    color = white_balance(color);

    // 5. Tonemapping
    color = apply_tonemap(color);

    //%P1

    // 6. Vignette
    color = apply_vignette(color, uv);

    // 7. Chromatic aberration
    color = apply_ca(color, uv, dims);

    // 8. Film grain
    color = apply_grain(color, uv, dims);

    //%P2

    // 9. Depth of Field
    let raw_depth = textureLoad(depth_input, vec2<i32>(i32(uv.x * dims.x), i32(uv.y * dims.y)), 0);
    color = apply_dof(color, uv, raw_depth, dims);

    // 10. Motion blur
    color = apply_motion_blur(color, uv);

    //%P3

    return vec4<f32>(color, 1.0);
}
