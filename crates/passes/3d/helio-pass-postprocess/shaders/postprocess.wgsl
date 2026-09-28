//!use helio_prelude
// ── Helio Post-Processing Pipeline ─────────────────────────────────────────────
//
// Opts into the prelude for the froxel depth<->slice mapping, which the fog
// composite must perform *identically* to the fog pass that fills the grid. The
// prelude's `Camera` is unused here — this file keeps its own `CameraUniforms`.
//
// Bind groups:
//   @group(0) — main: uniforms, samplers, hdr/depth inputs, bloom sampled, avg_lum,
//               noise, custom params, volume data, blend output
//   @group(1) — bloom compute: per-dispatch src (sampled) + dst (storage write)
//   @group(2) — bloom combine: mips 1-4 (sampled) + mip-0-sized sum (storage write)
//
// Entry points:
//   cs_exposure/reduce       — compute: sampled log-luminance reduction
//   cs_volume_blend          — compute: blend active post-process volumes → output
//   cs_bloom_down_extract    — compute: extract brights from HDR → bloom mip 0
//   cs_bloom_down            — compute: 2x downsample from bloom_src → bloom_dst
//   cs_bloom_combine         — compute: sum mips 1-4 at mip-0 resolution
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
// Kept equal to `PostProcessVolumeComponent`'s SceneDB packed-layout
// auto-register capacity (`pulsar_scenedb::gpu::world_mirror::
// DEFAULT_AUTO_REGISTER_CAPACITY`) -- see helio-pass-postprocess's own
// `MAX_PP_VOLUMES` doc for why the two must never drift apart.
const MAX_PP_VOLUMES: u32 = 64u;

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
    exposure_speed_up:                  f32,
    color_contrast:         vec3<f32>,
    exposure_speed_down:                  f32,
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
    dof_aperture_shape:     f32,
    dof_aperture_rotation:  f32,
    dof_near_transition:    f32,
    dof_far_transition:     f32,
    dof_max_bokeh_size:     f32,
    dof_sensor_diagonal:    f32,
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
    _pad_fog_emissive:         f32,   // 364
    // ── HDR Output (16 bytes) ──
    hdr_output_mode:           u32,   // 368
    hdr_max_nits:              f32,   // 372
    hdr_ui_brightness:         f32,   // 376
    _pad_hdr_end:              f32,   // 380
    // ── Advanced Color Grading (80 bytes) ──
    lift_color:                vec3<f32>, // 384
    _pad_lift:                 f32,   // 396
    gamma_color:               vec3<f32>, // 400
    _pad_gamma:                f32,   // 412
    gain_color:                vec3<f32>, // 416
    _pad_gain:                 f32,   // 428
    shadows_max:               f32,   // 432
    highlights_min:            f32,   // 436
    shadow_highlight_balance:  f32,   // 440
    hue_shift:                 f32,   // 444
    lut_generation:            u32,   // 448
    lut_intensity:             f32,   // 452
    lut_platform:              u32,   // 456
    _pad_grading_end:          f32,   // 460
    lens_enabled: u32, // 464
    lens_quality: u32, // 468
    lens_profile: u32, // 472
    lens_ghost_count: u32, // 476
    lens_intensity: f32, // 480
    lens_threshold: f32, // 484
    lens_soft_knee: f32, // 488
    lens_ghost_intensity: f32, // 492
    lens_halo_intensity: f32, // 496
    lens_glare_intensity: f32, // 500
    lens_streak_intensity: f32, // 504
    lens_dispersion: f32, // 508
    lens_aperture_f_number: f32, // 512
    lens_focal_length_mm: f32, // 516
    lens_sensor_width_mm: f32, // 520
    lens_vignette: f32, // 524
    lens_starburst_intensity: f32, // 528
    lens_starburst_length: f32,    // 532
    lens_aperture_blades: u32,     // 536
    lens_aperture_rotation: f32,   // 540
    lens_coating_strength: f32,    // 544
    lens_ghost_rim: f32,           // 548
    lens_dirt_intensity: f32,      // 552
    lens_light_sources: u32,       // 556
    lens_light_intensity: f32,     // 560
    lens_field_margin: f32,        // 564
    lens_response_time: f32,       // 568
    _pad_lens_ext0: f32,           // 572
    _pad_lens_ext1: vec4<f32>,     // 576 → struct ends at 592
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
    override_mask:  vec4<u32>,   // 48..64
    settings:       GpuPostProcessUniforms,  // 64
}

struct CameraPostProcessComponent {
    view_id: u32,
    enabled: u32,
    _pad: vec2<u32>,
    settings: GpuPostProcessUniforms,
}
@group(0) @binding(20) var<storage, read> camera_postprocess: array<CameraPostProcessComponent>;

// ── Group 0: main bindings ─────────────────────────────────────────────────────

@group(0) @binding(0)  var<uniform>            postprocess:  GpuPostProcessUniforms;
@group(0) @binding(1)  var<storage, read> cameras: array<CameraUniforms, 2>;
@group(0) @binding(2)  var                     hdr_input:    texture_2d<f32>;
@group(0) @binding(3)  var                     depth_input:  texture_depth_2d;
@group(0) @binding(4)  var                     linear_samp:  sampler;
@group(0) @binding(5)  var                     point_samp:   sampler;
@group(0) @binding(6)  var                     bloom_0:      texture_2d<f32>;
@group(0) @binding(7)  var                     bloom_1:      texture_2d<f32>;
@group(0) @binding(8)  var                     bloom_2:      texture_2d<f32>;
@group(0) @binding(9)  var                     bloom_3:      texture_2d<f32>;
@group(0) @binding(10) var                     bloom_4:      texture_2d<f32>;
// Mips 1-4 already reconstructed and summed at mip-0 resolution
// (cs_bloom_combine), so fs_uber upsamples two textures instead of five.
// bloom_1..bloom_4 stay bound for user effects.
@group(0) @binding(21) var                     bloom_coarse: texture_2d<f32>;
// [0] this frame's mean log2 luminance (cs_exposure_reduce)
// [1] adapted mean log2 luminance (cs_exposure_adapt; read by fs_uber)
// [2] frame delta seconds, [3] 1 when [1] holds valid history (CPU-written)
@group(0) @binding(11) var<storage, read_write> avg_luminance: array<f32>;
@group(0) @binding(15) var<storage, read_write> exposure_partials: array<vec2<f32>>;
@group(0) @binding(12) var                     noise_tex:    texture_2d<f32>;
@group(0) @binding(13) var                     noise_samp:   sampler;
@group(0) @binding(14) var<storage, read>      pp_custom:    array<vec4<f32>>;
@group(0) @binding(15) var<storage, read>      pp_volumes:   array<GpuPostProcessVolume>;
@group(0) @binding(16) var<storage, read_write> blend_output: GpuPostProcessUniforms;
// Scene-linear lens response (fs_uber only), published by LensFlarePass at
// reduced resolution from the same pre-exposure image. Participating media are
// already composited into hdr_input by FogCompositePass. Bound to a 1x1 black
// fallback when no lens pass is in the graph.
@group(0) @binding(17) var                     lens_input:   texture_2d<f32>;
@group(0) @binding(18) var                     velocity_tex: texture_2d<f32>;
@group(0) @binding(19) var                     lut_tex:      texture_3d<f32>;

// ── Group 1: per-dispatch bloom compute src/dst ────────────────────────────────

@group(1) @binding(0) var bloom_src: texture_2d<f32>;
@group(1) @binding(1) var bloom_dst: texture_storage_2d<rgba16float, write>;

// ── Group 2: bloom combine (mips 1-4 → mip-0-sized sum) ────────────────────────

@group(2) @binding(0) var bloom_combine_1: texture_2d<f32>;
@group(2) @binding(1) var bloom_combine_2: texture_2d<f32>;
@group(2) @binding(2) var bloom_combine_3: texture_2d<f32>;
@group(2) @binding(3) var bloom_combine_4: texture_2d<f32>;
@group(2) @binding(4) var bloom_combine_dst: texture_storage_2d<rgba16float, write>;

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

// The four mask words use the indices exported by PostProcessProperty.
fn overridden(mask: vec4<u32>, property: u32) -> bool {
    return (mask[property / 32u] & (1u << (property % 32u))) != 0u;
}
fn blend_settings(base: GpuPostProcessUniforms, vol: GpuPostProcessUniforms, t: f32, mask: vec4<u32>) -> GpuPostProcessUniforms {
    var r = base;
    if overridden(mask, 0u) { r.exposure_mode = select(base.exposure_mode, vol.exposure_mode, t >= 0.5); }
    if overridden(mask, 1u) { r.exposure_compensation = base.exposure_compensation + (vol.exposure_compensation - base.exposure_compensation) * t; }
    if overridden(mask, 2u) { r.exposure_min = base.exposure_min + (vol.exposure_min - base.exposure_min) * t; }
    if overridden(mask, 3u) { r.exposure_max = base.exposure_max + (vol.exposure_max - base.exposure_max) * t; }
    if overridden(mask, 4u) { r.bloom_intensity = base.bloom_intensity + (vol.bloom_intensity - base.bloom_intensity) * t; }
    if overridden(mask, 5u) { r.bloom_threshold = base.bloom_threshold + (vol.bloom_threshold - base.bloom_threshold) * t; }
    if overridden(mask, 6u) { r.bloom_knee = base.bloom_knee + (vol.bloom_knee - base.bloom_knee) * t; }
    if overridden(mask, 7u) { r.bloom_radius = base.bloom_radius + (vol.bloom_radius - base.bloom_radius) * t; }
    if overridden(mask, 8u) { r.bloom_tint = base.bloom_tint + (vol.bloom_tint - base.bloom_tint) * t; }
    if overridden(mask, 9u) { r.bloom_enabled = select(base.bloom_enabled, vol.bloom_enabled, t >= 0.5); }
    if overridden(mask, 10u) { r.color_saturation = base.color_saturation + (vol.color_saturation - base.color_saturation) * t; }
    if overridden(mask, 11u) { r.exposure_speed_up = base.exposure_speed_up + (vol.exposure_speed_up - base.exposure_speed_up) * t; }
    if overridden(mask, 12u) { r.color_contrast = base.color_contrast + (vol.color_contrast - base.color_contrast) * t; }
    if overridden(mask, 13u) { r.exposure_speed_down = base.exposure_speed_down + (vol.exposure_speed_down - base.exposure_speed_down) * t; }
    if overridden(mask, 14u) { r.color_gamma = base.color_gamma + (vol.color_gamma - base.color_gamma) * t; }
    if overridden(mask, 15u) { r.color_gain = base.color_gain + (vol.color_gain - base.color_gain) * t; }
    if overridden(mask, 16u) { r.color_offset = base.color_offset + (vol.color_offset - base.color_offset) * t; }
    if overridden(mask, 17u) { r.white_temp = base.white_temp + (vol.white_temp - base.white_temp) * t; }
    if overridden(mask, 18u) { r.white_tint = base.white_tint + (vol.white_tint - base.white_tint) * t; }
    if overridden(mask, 19u) { r.white_balance_enabled = select(base.white_balance_enabled, vol.white_balance_enabled, t >= 0.5); }
    if overridden(mask, 20u) { r.tonemap_operator = select(base.tonemap_operator, vol.tonemap_operator, t >= 0.5); }
    if overridden(mask, 21u) { r.tonemap_exposure = base.tonemap_exposure + (vol.tonemap_exposure - base.tonemap_exposure) * t; }
    if overridden(mask, 22u) { r.tonemap_white_point = base.tonemap_white_point + (vol.tonemap_white_point - base.tonemap_white_point) * t; }
    if overridden(mask, 23u) { r.vignette_intensity = base.vignette_intensity + (vol.vignette_intensity - base.vignette_intensity) * t; }
    if overridden(mask, 24u) { r.vignette_smoothness = base.vignette_smoothness + (vol.vignette_smoothness - base.vignette_smoothness) * t; }
    if overridden(mask, 25u) { r.vignette_roundness = base.vignette_roundness + (vol.vignette_roundness - base.vignette_roundness) * t; }
    if overridden(mask, 26u) { r.vignette_color = base.vignette_color + (vol.vignette_color - base.vignette_color) * t; }
    if overridden(mask, 27u) { r.vignette_enabled = select(base.vignette_enabled, vol.vignette_enabled, t >= 0.5); }
    if overridden(mask, 28u) { r.ca_intensity = base.ca_intensity + (vol.ca_intensity - base.ca_intensity) * t; }
    if overridden(mask, 29u) { r.ca_start_offset = base.ca_start_offset + (vol.ca_start_offset - base.ca_start_offset) * t; }
    if overridden(mask, 30u) { r.ca_enabled = select(base.ca_enabled, vol.ca_enabled, t >= 0.5); }
    if overridden(mask, 31u) { r.grain_intensity = base.grain_intensity + (vol.grain_intensity - base.grain_intensity) * t; }
    if overridden(mask, 32u) { r.grain_response = base.grain_response + (vol.grain_response - base.grain_response) * t; }
    if overridden(mask, 33u) { r.grain_size = base.grain_size + (vol.grain_size - base.grain_size) * t; }
    if overridden(mask, 34u) { r.grain_enabled = select(base.grain_enabled, vol.grain_enabled, t >= 0.5); }
    if overridden(mask, 35u) { r.dof_focal_distance = base.dof_focal_distance + (vol.dof_focal_distance - base.dof_focal_distance) * t; }
    if overridden(mask, 36u) { r.dof_focal_region = base.dof_focal_region + (vol.dof_focal_region - base.dof_focal_region) * t; }
    if overridden(mask, 37u) { r.dof_aperture_shape = select(base.dof_aperture_shape, vol.dof_aperture_shape, t >= 0.5); }
    if overridden(mask, 38u) { r.dof_aperture_rotation = base.dof_aperture_rotation + (vol.dof_aperture_rotation - base.dof_aperture_rotation) * t; }
    if overridden(mask, 39u) { r.dof_near_transition = base.dof_near_transition + (vol.dof_near_transition - base.dof_near_transition) * t; }
    if overridden(mask, 40u) { r.dof_far_transition = base.dof_far_transition + (vol.dof_far_transition - base.dof_far_transition) * t; }
    if overridden(mask, 41u) { r.dof_max_bokeh_size = base.dof_max_bokeh_size + (vol.dof_max_bokeh_size - base.dof_max_bokeh_size) * t; }
    if overridden(mask, 42u) { r.dof_sensor_diagonal = base.dof_sensor_diagonal + (vol.dof_sensor_diagonal - base.dof_sensor_diagonal) * t; }
    if overridden(mask, 43u) { r.motion_blur_amount = base.motion_blur_amount + (vol.motion_blur_amount - base.motion_blur_amount) * t; }
    if overridden(mask, 44u) { r.motion_blur_max = base.motion_blur_max + (vol.motion_blur_max - base.motion_blur_max) * t; }
    if overridden(mask, 45u) { r.motion_blur_enabled = select(base.motion_blur_enabled, vol.motion_blur_enabled, t >= 0.5); }
    if overridden(mask, 46u) { r.blend_weight_bloom = base.blend_weight_bloom + (vol.blend_weight_bloom - base.blend_weight_bloom) * t; }
    if overridden(mask, 47u) { r.blend_weight_dof = base.blend_weight_dof + (vol.blend_weight_dof - base.blend_weight_dof) * t; }
    if overridden(mask, 48u) { r.blend_weight_motion_blur = base.blend_weight_motion_blur + (vol.blend_weight_motion_blur - base.blend_weight_motion_blur) * t; }
    if overridden(mask, 49u) { r.blend_weight_vignette = base.blend_weight_vignette + (vol.blend_weight_vignette - base.blend_weight_vignette) * t; }
    if overridden(mask, 50u) { r.blend_weight_ca = base.blend_weight_ca + (vol.blend_weight_ca - base.blend_weight_ca) * t; }
    if overridden(mask, 51u) { r.blend_weight_grain = base.blend_weight_grain + (vol.blend_weight_grain - base.blend_weight_grain) * t; }
    if overridden(mask, 52u) { r.blend_weight_exposure = base.blend_weight_exposure + (vol.blend_weight_exposure - base.blend_weight_exposure) * t; }
    if overridden(mask, 53u) { r.fog_enabled = select(base.fog_enabled, vol.fog_enabled, t >= 0.5); }
    if overridden(mask, 54u) { r.fog_mode = select(base.fog_mode, vol.fog_mode, t >= 0.5); }
    if overridden(mask, 55u) { r.fog_density = base.fog_density + (vol.fog_density - base.fog_density) * t; }
    if overridden(mask, 56u) { r.fog_height_falloff = base.fog_height_falloff + (vol.fog_height_falloff - base.fog_height_falloff) * t; }
    if overridden(mask, 57u) { r.fog_start_distance = base.fog_start_distance + (vol.fog_start_distance - base.fog_start_distance) * t; }
    if overridden(mask, 58u) { r.fog_max_distance = base.fog_max_distance + (vol.fog_max_distance - base.fog_max_distance) * t; }
    if overridden(mask, 59u) { r.fog_height = base.fog_height + (vol.fog_height - base.fog_height) * t; }
    if overridden(mask, 60u) { r.fog_scattering_anisotropy = base.fog_scattering_anisotropy + (vol.fog_scattering_anisotropy - base.fog_scattering_anisotropy) * t; }
    if overridden(mask, 61u) { r.fog_color = base.fog_color + (vol.fog_color - base.fog_color) * t; }
    if overridden(mask, 62u) { r.fog_emissive = base.fog_emissive + (vol.fog_emissive - base.fog_emissive) * t; }
    if overridden(mask, 63u) { r.hdr_output_mode = select(base.hdr_output_mode, vol.hdr_output_mode, t >= 0.5); }
    if overridden(mask, 64u) { r.hdr_max_nits = base.hdr_max_nits + (vol.hdr_max_nits - base.hdr_max_nits) * t; }
    if overridden(mask, 65u) { r.hdr_ui_brightness = base.hdr_ui_brightness + (vol.hdr_ui_brightness - base.hdr_ui_brightness) * t; }
    if overridden(mask, 66u) { r.lift_color = base.lift_color + (vol.lift_color - base.lift_color) * t; }
    if overridden(mask, 67u) { r.gamma_color = base.gamma_color + (vol.gamma_color - base.gamma_color) * t; }
    if overridden(mask, 68u) { r.gain_color = base.gain_color + (vol.gain_color - base.gain_color) * t; }
    if overridden(mask, 69u) { r.shadows_max = base.shadows_max + (vol.shadows_max - base.shadows_max) * t; }
    if overridden(mask, 70u) { r.highlights_min = base.highlights_min + (vol.highlights_min - base.highlights_min) * t; }
    if overridden(mask, 71u) { r.shadow_highlight_balance = base.shadow_highlight_balance + (vol.shadow_highlight_balance - base.shadow_highlight_balance) * t; }
    if overridden(mask, 72u) { r.hue_shift = base.hue_shift + (vol.hue_shift - base.hue_shift) * t; }
    if overridden(mask, 73u) { r.lut_generation = select(base.lut_generation, vol.lut_generation, t >= 0.5); }
    if overridden(mask, 74u) { r.lut_intensity = base.lut_intensity + (vol.lut_intensity - base.lut_intensity) * t; }
    if overridden(mask, 75u) { r.lut_platform = select(base.lut_platform, vol.lut_platform, t >= 0.5); }
    if overridden(mask, 76u) { r.lens_enabled = select(base.lens_enabled, vol.lens_enabled, t >= 0.5); }
    if overridden(mask, 77u) { r.lens_quality = select(base.lens_quality, vol.lens_quality, t >= 0.5); }
    if overridden(mask, 78u) { r.lens_profile = select(base.lens_profile, vol.lens_profile, t >= 0.5); }
    if overridden(mask, 79u) { r.lens_ghost_count = select(base.lens_ghost_count, vol.lens_ghost_count, t >= 0.5); }
    if overridden(mask, 80u) { r.lens_intensity = base.lens_intensity + (vol.lens_intensity - base.lens_intensity) * t; }
    if overridden(mask, 81u) { r.lens_threshold = base.lens_threshold + (vol.lens_threshold - base.lens_threshold) * t; }
    if overridden(mask, 82u) { r.lens_soft_knee = base.lens_soft_knee + (vol.lens_soft_knee - base.lens_soft_knee) * t; }
    if overridden(mask, 83u) { r.lens_ghost_intensity = base.lens_ghost_intensity + (vol.lens_ghost_intensity - base.lens_ghost_intensity) * t; }
    if overridden(mask, 84u) { r.lens_halo_intensity = base.lens_halo_intensity + (vol.lens_halo_intensity - base.lens_halo_intensity) * t; }
    if overridden(mask, 85u) { r.lens_glare_intensity = base.lens_glare_intensity + (vol.lens_glare_intensity - base.lens_glare_intensity) * t; }
    if overridden(mask, 86u) { r.lens_streak_intensity = base.lens_streak_intensity + (vol.lens_streak_intensity - base.lens_streak_intensity) * t; }
    if overridden(mask, 87u) { r.lens_dispersion = base.lens_dispersion + (vol.lens_dispersion - base.lens_dispersion) * t; }
    if overridden(mask, 88u) { r.lens_aperture_f_number = base.lens_aperture_f_number + (vol.lens_aperture_f_number - base.lens_aperture_f_number) * t; }
    if overridden(mask, 89u) { r.lens_focal_length_mm = base.lens_focal_length_mm + (vol.lens_focal_length_mm - base.lens_focal_length_mm) * t; }
    if overridden(mask, 90u) { r.lens_sensor_width_mm = base.lens_sensor_width_mm + (vol.lens_sensor_width_mm - base.lens_sensor_width_mm) * t; }
    if overridden(mask, 91u) { r.lens_vignette = base.lens_vignette + (vol.lens_vignette - base.lens_vignette) * t; }
    if overridden(mask, 92u) { r.lens_starburst_intensity = base.lens_starburst_intensity + (vol.lens_starburst_intensity - base.lens_starburst_intensity) * t; }
    if overridden(mask, 93u) { r.lens_starburst_length = base.lens_starburst_length + (vol.lens_starburst_length - base.lens_starburst_length) * t; }
    if overridden(mask, 94u) { r.lens_aperture_blades = select(base.lens_aperture_blades, vol.lens_aperture_blades, t >= 0.5); }
    if overridden(mask, 95u) { r.lens_aperture_rotation = base.lens_aperture_rotation + (vol.lens_aperture_rotation - base.lens_aperture_rotation) * t; }
    if overridden(mask, 96u) { r.lens_coating_strength = base.lens_coating_strength + (vol.lens_coating_strength - base.lens_coating_strength) * t; }
    if overridden(mask, 97u) { r.lens_ghost_rim = base.lens_ghost_rim + (vol.lens_ghost_rim - base.lens_ghost_rim) * t; }
    if overridden(mask, 98u) { r.lens_dirt_intensity = base.lens_dirt_intensity + (vol.lens_dirt_intensity - base.lens_dirt_intensity) * t; }
    if overridden(mask, 99u) { r.lens_light_sources = select(base.lens_light_sources, vol.lens_light_sources, t >= 0.5); }
    if overridden(mask, 100u) { r.lens_light_intensity = base.lens_light_intensity + (vol.lens_light_intensity - base.lens_light_intensity) * t; }
    if overridden(mask, 101u) { r.lens_field_margin = base.lens_field_margin + (vol.lens_field_margin - base.lens_field_margin) * t; }
    if overridden(mask, 102u) { r.lens_response_time = base.lens_response_time + (vol.lens_response_time - base.lens_response_time) * t; }
    return r;
}
fn volume_weight(pos: vec3<f32>, v: GpuPostProcessVolume) -> f32 {
    let weight = clamp(v.blend_weight, 0.0, 1.0);
    if v.unbound != 0u { return weight; }
    if !all(pos >= v.bounds_min.xyz) || !all(pos <= v.bounds_max.xyz) { return 0.0; }
    let d = min(pos - v.bounds_min.xyz, v.bounds_max.xyz - pos);
    let boundary = min(d.x, min(d.y, d.z));
    if v.blend_radius > 0.0 { return weight * clamp(boundary / v.blend_radius, 0.0, 1.0); }
    return weight;
}

// SceneDB rows are indexed by entity, so these arrays span the whole entity
// range and are mostly empty. 256 threads compact the live rows reading only
// their small header fields; one thread then blends just the active volumes.
const RESOLVE_THREADS: u32 = 256u;
/// Active volumes blended per frame; further rows are ignored (lowest rows win).
const MAX_ACTIVE_VOLUMES: u32 = 256u;
var<workgroup> active_rows: array<u32, 256>;
var<workgroup> active_count: atomic<u32>;
var<workgroup> camera_row: atomic<u32>;

fn finite_f32(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}

@compute @workgroup_size(256)
fn cs_volume_blend(@builtin(local_invocation_index) lid: u32) {
    if lid == 0u {
        atomicStore(&active_count, 0u);
        atomicStore(&camera_row, 0xffffffffu);
    }
    workgroupBarrier();
    let view_id = bitcast<u32>(cameras[0].jitter_frame.w);
    let camera_rows = arrayLength(&camera_postprocess);
    for (var i = lid; i < camera_rows; i += RESOLVE_THREADS) {
        if camera_postprocess[i].enabled != 0u && camera_postprocess[i].view_id == view_id {
            atomicMin(&camera_row, i);
        }
    }
    let volume_rows = arrayLength(&pp_volumes);
    for (var i = lid; i < volume_rows; i += RESOLVE_THREADS) {
        let w = pp_volumes[i].blend_weight;
        if w > 0.0 && finite_f32(w) && finite_f32(pp_volumes[i].priority) {
            let slot = atomicAdd(&active_count, 1u);
            if slot < MAX_ACTIVE_VOLUMES { active_rows[slot] = i; }
        }
    }
    workgroupBarrier();
    if lid != 0u { return; }

    var baseline = postprocess;
    let camera = atomicLoad(&camera_row);
    if camera != 0xffffffffu { baseline = camera_postprocess[camera].settings; }
    let count = min(atomicLoad(&active_count), MAX_ACTIVE_VOLUMES);
    // Stable ascending (priority, row): insertion sort of the compact list.
    for (var i = 1u; i < count; i++) {
        let row = active_rows[i];
        let priority = pp_volumes[row].priority;
        var j = i;
        loop {
            if j == 0u { break; }
            let other = active_rows[j - 1u];
            let other_priority = pp_volumes[other].priority;
            if other_priority < priority || (other_priority == priority && other < row) { break; }
            active_rows[j] = other;
            j--;
        }
        active_rows[j] = row;
    }

    var result = baseline;
    var medium = baseline;
    var bounded_range = 0.0;
    var bounded_fog = false;
    let position = cameras[0].position_near.xyz;
    for (var n = 0u; n < count; n++) {
        let v = pp_volumes[active_rows[n]];
        let weight = volume_weight(position, v);
        if weight > 0.0 { result = blend_settings(result, v.settings, weight, v.override_mask); }
        if v.unbound != 0u {
            medium = blend_settings(medium, v.settings, clamp(v.blend_weight, 0.0, 1.0), v.override_mask);
        } else if overridden(v.override_mask, 53u) && v.settings.fog_enabled != 0u && v.settings.fog_density > 0.0 {
            // Bounded media are evaluated in world space by the fog pass, not
            // at the camera; they only extend the integration range here.
            bounded_fog = true;
            bounded_range = max(bounded_range, v.settings.fog_max_distance);
        }
    }
    result.fog_enabled = medium.fog_enabled;
    result.fog_mode = medium.fog_mode;
    result.fog_height_falloff = medium.fog_height_falloff;
    result.fog_start_distance = medium.fog_start_distance;
    result.fog_height = medium.fog_height;
    result.fog_scattering_anisotropy = medium.fog_scattering_anisotropy;
    result.fog_color = medium.fog_color;
    result.fog_emissive = medium.fog_emissive;
    result.fog_density = select(0.0, medium.fog_density, medium.fog_enabled != 0u);
    var range = select(0.0, medium.fog_max_distance, medium.fog_enabled != 0u);
    if bounded_fog {
        result.fog_enabled = 1u;
        range = max(range, bounded_range);
    }
    result.fog_max_distance = max(range, 1.0);
    blend_output = result;
}

// ── cs_exposure: unique workgroup partials, then one final reduction ──────────

var<workgroup> wg_sum:   array<f32, 256>;
var<workgroup> wg_count: array<u32, 256>;

@compute @workgroup_size(16, 16)
fn cs_exposure(@builtin(global_invocation_id) gid: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>,
               @builtin(workgroup_id) group: vec3<u32>) {
    let dims = textureDimensions(hdr_input);
    let stride = 4u;
    var sum_log: f32 = 0.0;
    var count: u32 = 0u;
    let pixel = gid.xy * stride;
    if all(pixel < dims) {
        let col = textureLoad(hdr_input, vec2<i32>(pixel), 0).rgb;
        sum_log = log2(max(luminance(col), 0.0001));
        count = 1u;
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

    if lidx == 0u {
        let groups_x = (dims.x + stride * 16u - 1u) / (stride * 16u);
        exposure_partials[group.y * groups_x + group.x] = vec2<f32>(wg_sum[0], f32(wg_count[0]));
    }
}

@compute @workgroup_size(256)
fn cs_exposure_reduce(@builtin(local_invocation_index) lid: u32) {
    let dims = textureDimensions(hdr_input);
    let groups_x = (dims.x + 63u) / 64u;
    let groups_y = (dims.y + 63u) / 64u;
    let partial_count = groups_x * groups_y;
    var sum_log = 0.0;
    var count = 0u;
    for (var index = lid; index < partial_count; index += 256u) {
        let partial = exposure_partials[index];
        sum_log += partial.x;
        count += u32(partial.y);
    }
    wg_sum[lid] = sum_log;
    wg_count[lid] = count;
    workgroupBarrier();
    var reduce_active = 128u;
    loop {
        if reduce_active == 0u { break; }
        if lid < reduce_active {
            wg_sum[lid] += wg_sum[lid + reduce_active];
            wg_count[lid] += wg_count[lid + reduce_active];
        }
        workgroupBarrier();
        reduce_active >>= 1u;
    }
    if lid == 0u && wg_count[0] > 0u {
        avg_luminance[0] = wg_sum[0] / f32(wg_count[0]);
    }
}

// ── cs_exposure_adapt: exponential eye adaptation in log2 luminance ──────────
//
// Frame-rate independent: the blend toward the metered value uses
// 1 - exp(-dt / tau), with tau = exposure_speed_up while the scene brightens
// and exposure_speed_down while it darkens. First valid frame snaps.

@compute @workgroup_size(1)
fn cs_exposure_adapt() {
    let measured = avg_luminance[0];
    if avg_luminance[3] < 0.5 || !(abs(avg_luminance[1]) < 64.0) {
        avg_luminance[1] = measured;
        return;
    }
    let current = avg_luminance[1];
    let tau = select(postprocess.exposure_speed_down, postprocess.exposure_speed_up, measured > current);
    let dt = clamp(avg_luminance[2], 0.0, 1.0);
    let t = select(1.0 - exp(-dt / tau), 1.0, !(tau > 1e-4));
    avg_luminance[1] = current + (measured - current) * t;
}

// Scene-linear exposure multiplier, shared by the image, bloom and lens so a
// single exposure is applied once. Auto mode targets middle grey (0.18) from
// the adapted log2 mean, limited to [exposure_min, exposure_max] EV.
fn exposure_scale() -> f32 {
    var ev = postprocess.exposure_compensation;
    if postprocess.exposure_mode == 1u {
        let auto_ev = clamp(log2(0.18) - avg_luminance[1],
            postprocess.exposure_min, postprocess.exposure_max);
        ev += auto_ev * clamp(postprocess.blend_weight_exposure, 0.0, 1.0);
    }
    return exp2(ev);
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
    if any(gid.xy >= dst_dims) { return; }
    // 13-tap downsample (Jimenez, "Next Generation Post Processing in Call of
    // Duty", SIGGRAPH 2014): overlapping 4x4 box footprints weighted toward the
    // centre. A plain 2x2 box keeps the source texel grid, so very bright
    // sources bloom into visible squares at the coarse mips.
    let texel = 1.0 / vec2<f32>(textureDimensions(bloom_src));
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dst_dims);
    let a = textureSampleLevel(bloom_src, linear_samp, uv + texel * vec2(-2.0, -2.0), 0.0).rgb;
    let b = textureSampleLevel(bloom_src, linear_samp, uv + texel * vec2( 0.0, -2.0), 0.0).rgb;
    let c = textureSampleLevel(bloom_src, linear_samp, uv + texel * vec2( 2.0, -2.0), 0.0).rgb;
    let d = textureSampleLevel(bloom_src, linear_samp, uv + texel * vec2(-1.0, -1.0), 0.0).rgb;
    let e = textureSampleLevel(bloom_src, linear_samp, uv + texel * vec2( 1.0, -1.0), 0.0).rgb;
    let f = textureSampleLevel(bloom_src, linear_samp, uv + texel * vec2(-2.0,  0.0), 0.0).rgb;
    let g = textureSampleLevel(bloom_src, linear_samp, uv, 0.0).rgb;
    let h = textureSampleLevel(bloom_src, linear_samp, uv + texel * vec2( 2.0,  0.0), 0.0).rgb;
    let i = textureSampleLevel(bloom_src, linear_samp, uv + texel * vec2(-1.0,  1.0), 0.0).rgb;
    let j = textureSampleLevel(bloom_src, linear_samp, uv + texel * vec2( 1.0,  1.0), 0.0).rgb;
    let k = textureSampleLevel(bloom_src, linear_samp, uv + texel * vec2(-2.0,  2.0), 0.0).rgb;
    let l = textureSampleLevel(bloom_src, linear_samp, uv + texel * vec2( 0.0,  2.0), 0.0).rgb;
    let m = textureSampleLevel(bloom_src, linear_samp, uv + texel * vec2( 2.0,  2.0), 0.0).rgb;
    let color = g * 0.125 + (a + c + k + m) * 0.03125 + (b + f + h + l) * 0.0625 + (d + e + i + j) * 0.125;
    textureStore(bloom_dst, vec2<i32>(gid.xy), vec4<f32>(color, 0.0));
}

// ── cs_bloom_combine: mips 1-4 → one mip-0-sized texture ──────────────────────
// Evaluates the B-spline reconstruction of each coarse mip at mip-0 texel
// centres and sums them. fs_uber then B-spline upsamples this sum once instead
// of upsampling each coarse mip at output resolution (16 taps per output pixel
// → 4 taps per output pixel plus 16 per mip-0 texel). The coarse mips are at
// least 2x coarser than mip 0, so the extra reconstruction step barely changes
// their already-wide glow.
@compute @workgroup_size(8, 8, 1)
fn cs_bloom_combine(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_dims = textureDimensions(bloom_combine_dst);
    if gid.x >= dst_dims.x || gid.y >= dst_dims.y { return; }
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dst_dims);
    let sum = sample_bspline(bloom_combine_1, uv)
            + sample_bspline(bloom_combine_2, uv)
            + sample_bspline(bloom_combine_3, uv)
            + sample_bspline(bloom_combine_4, uv);
    textureStore(bloom_combine_dst, vec2<i32>(gid.xy), vec4<f32>(sum, 0.0));
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

fn apply_lift_gamma_gain(c: vec3<f32>) -> vec3<f32> {
    // Lift/Gamma/Gain colour wheels
    // shadows = c * (1 - lift) + lift  (lift shifts shadows)
    // midtones = pow(c, gamma)
    // highlights = c * gain
    var result = c;
    let lift  = postprocess.lift_color;
    let gamma = postprocess.gamma_color;
    let gain  = postprocess.gain_color;
    result = result * (vec3<f32>(1.0) - lift) + lift;
    result = pow(max(result, vec3<f32>(0.0)), gamma + vec3<f32>(1.0));
    result = result * gain;
    return result;
}

fn hue_shift_rgb(c: vec3<f32>, shift_deg: f32) -> vec3<f32> {
    if abs(shift_deg) < 0.001 { return c; }
    let angle = shift_deg * 3.14159265 / 180.0;
    let cos_a = cos(angle);
    let sin_a = sin(angle);
    // RGB hue rotation matrix
    let m = mat3x3<f32>(
        vec3<f32>(0.213, 0.213 - 0.213 * cos_a + 0.144 * sin_a, 0.213 - 0.213 * cos_a - 0.756 * sin_a),
        vec3<f32>(0.715, 0.715 - 0.715 * cos_a - 0.283 * sin_a, 0.715 - 0.715 * cos_a + 0.416 * sin_a),
        vec3<f32>(0.072, 0.072 - 0.072 * cos_a + 0.860 * sin_a, 0.072 - 0.072 * cos_a - 0.461 * sin_a),
    );
    return m * c;
}

fn sample_lut(c: vec3<f32>) -> vec3<f32> {
    let uv = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    return textureSampleLevel(lut_tex, linear_samp, uv, 0.0).rgb;
}

fn apply_lut_grade(color: vec3<f32>) -> vec3<f32> {
    // Pre-LUT: hue shift
    var c = hue_shift_rgb(color, postprocess.hue_shift);
    // Sample LUT
    let graded = sample_lut(c);
    // Blend by intensity
    c = mix(c, graded, postprocess.lut_intensity);
    // Post-LUT: lift/gamma/gain
    c = apply_lift_gamma_gain(c);
    return c;
}

fn color_grade(color: vec3<f32>) -> vec3<f32> {
    if postprocess.lut_platform > 0u {
        return apply_lut_grade(color);
    }
    // Simple path (backward compatibility)
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

// ── DOF mode constants (match CPU-side dof_aperture_shape encoding) ──
//   dof_aperture_shape < 0 → disabled
//   dof_aperture_shape == 0 → DOF_MODE_GAUSSIAN (circular fallback)
//   dof_aperture_shape > 0 → DOF_MODE_BOKEH with floor(shape) blades

fn dof_coc(depth: f32) -> f32 {
    let linear_depth = -cameras[0].proj[3][2] / (depth * 2.0 - 1.0 + cameras[0].proj[2][2]);
    let focal_dist = postprocess.dof_focal_distance;
    let focal_region = postprocess.dof_focal_region;
    let near_blur = max(focal_dist - focal_region - linear_depth, 0.0) / max(postprocess.dof_near_transition, 0.001);
    let far_blur = max(linear_depth - (focal_dist + focal_region), 0.0) / max(postprocess.dof_far_transition, 0.001);
    // Thin-lens CoC: sensor_diagonal / focal_dist gives the physical blur circle
    // scaled to screen pixels via max_bokeh_size.
    let coc = max(near_blur, far_blur) * postprocess.dof_sensor_diagonal * 0.02;
    return clamp(coc, 0.0, postprocess.dof_max_bokeh_size);
}

fn apply_dof_gaussian(color: vec3<f32>, uv: vec2<f32>, depth: f32, dims: vec2<f32>) -> vec3<f32> {
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

fn apply_dof(color: vec3<f32>, uv: vec2<f32>, depth: f32, dims: vec2<f32>) -> vec3<f32> {
    let shape = postprocess.dof_aperture_shape;
    if shape < 0.0 { return color; }
    // Gaussian fallback (shape == 0) runs inline; bokeh mode is handled by
    // the separate DofPass when it is present in the graph.
    return apply_dof_gaussian(color, uv, depth, dims);
}

// ── Motion blur ────────────────────────────────────────────────────────────────

fn apply_motion_blur(color: vec3<f32>, uv: vec2<f32>, dims: vec2<f32>) -> vec3<f32> {
    if postprocess.motion_blur_enabled == 0u { return color; }

    let velocity = textureLoad(velocity_tex, vec2<i32>(i32(uv.x * dims.x), i32(uv.y * dims.y)), 0).rg;
    let vel_len = length(velocity);
    if vel_len < 0.5 { return color; }

    let max_len = postprocess.motion_blur_max;
    let clamped_vel = normalize(velocity) * min(vel_len, max_len);
    let samples = min(i32(vel_len / 2.0 + 2.0), 16);
    let step = clamped_vel / f32(samples) / dims;

    var blurred = vec3<f32>(0.0);
    for (var i = 0; i < samples; i++) {
        let t = f32(i) / f32(samples);
        let sample_uv = uv - step * f32(i);
        blurred += textureSampleLevel(hdr_input, linear_samp, sample_uv, 0.0).rgb;
    }
    return blurred / f32(samples + 1);
}

// Cubic B-spline reconstruction of a reduced-resolution image in 4 bilinear
// taps (each tap resolves a 2x2 weighted footprint). Bilinear magnification
// of bloom mips and the quarter-res lens response shows each source texel
// as a hard block around very bright sources; the B-spline is C2-smooth and
// radially near-isotropic, so glows stay round at any upscale factor.
fn sample_bspline(tex: texture_2d<f32>, uv: vec2<f32>) -> vec3<f32> {
    let size = vec2<f32>(textureDimensions(tex));
    let p = uv * size - 0.5;
    let base = floor(p);
    let f = p - base;
    let f2 = f * f;
    let f3 = f2 * f;
    let w0 = (1.0 - f) * (1.0 - f) * (1.0 - f) / 6.0;
    let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    let w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
    let w3 = f3 / 6.0;
    let s0 = w0 + w1;
    let s1 = w2 + w3;
    let t0 = (base - 1.0 + w1 / s0 + 0.5) / size;
    let t1 = (base + 1.0 + w3 / s1 + 0.5) / size;
    return textureSampleLevel(tex, linear_samp, vec2(t0.x, t0.y), 0.0).rgb * s0.x * s0.y
         + textureSampleLevel(tex, linear_samp, vec2(t1.x, t0.y), 0.0).rgb * s1.x * s0.y
         + textureSampleLevel(tex, linear_samp, vec2(t0.x, t1.y), 0.0).rgb * s0.x * s1.y
         + textureSampleLevel(tex, linear_samp, vec2(t1.x, t1.y), 0.0).rgb * s1.x * s1.y;
}

// ── fs_uber ────────────────────────────────────────────────────────────────────

@fragment
fn fs_uber(in: VOut) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(hdr_input));
    let uv = in.uv;

    // hdr_input is scene-linear and already contains participating media:
    // FogCompositePass applied transmittance and in-scattering at each
    // surface's depth before transparency and AA, so metering, bloom and lens
    // extraction below all see scattered light.
    var color = textureSampleLevel(hdr_input, linear_samp, uv, 0.0).rgb;

    //%P0

    // 1. Exposure — one scene-linear multiplier for image, bloom and lens.
    let exposure = exposure_scale();
    color *= exposure;

    // 2. Bloom composite. Bloom was extracted from pre-exposure radiance, so it
    // takes the same exposure as the image it came from.
    if postprocess.bloom_enabled != 0u && postprocess.bloom_intensity > 0.0 {
        var bloom = vec3<f32>(0.0);
        bloom += sample_bspline(bloom_0, uv);
        bloom += sample_bspline(bloom_coarse, uv);
        // Mean over the chain: every mip carries the same extracted energy, so
        // bloom_intensity is the fraction of it scattered, not five times that.
        color += bloom * (exposure / 5.0);
    }

    // 2b. Lens response: optical ghosts/halo/glare/streaks are light the lens
    // scattered, in the same scene-linear units, so they are exposed like the
    // image and tone mapped once with it. B-spline upsampled from quarter
    // resolution so compact glare around small sources stays round.
    if postprocess.lens_enabled != 0u {
        color += max(sample_bspline(lens_input, uv), vec3<f32>(0.0)) * exposure;
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

    // 9. Depth of Field (Gaussian fallback when no DofPass is in the graph).
    // The depth fetch is only needed when DOF is on (shape >= 0).
    if !(postprocess.dof_aperture_shape < 0.0) {
        let raw_depth = textureLoad(depth_input, vec2<i32>(i32(uv.x * dims.x), i32(uv.y * dims.y)), 0);
        color = apply_dof(color, uv, raw_depth, dims);
    }

    // 10. Motion blur
    color = apply_motion_blur(color, uv, dims);

    //%P3

    // 11. HDR display encoding
    //
    // Scene values are in arbitrary linear units. The uniform fields
    // hdr_max_nits and hdr_ui_brightness map them to cd/m²:
    //   scene value = hdr_ui_brightness  →  hdr_max_nits cd/m²
    //   scene value = 1.0                →  hdr_max_nits / hdr_ui_brightness cd/m²
    if postprocess.hdr_output_mode == 1u {
        // HDR10: PQ ST 2084 per-channel + BT.2020 gamut
        const PQ_M1: f32 = 0.1593017578125;
        const PQ_M2: f32 = 78.84375;
        const PQ_C1: f32 = 0.8359375;
        const PQ_C2: f32 = 18.8515625;
        const PQ_C3: f32 = 18.6875;
        const REC709_TO_BT2020: mat3x3<f32> = mat3x3<f32>(
            vec3<f32>(0.6274, 0.0691, 0.0164),
            vec3<f32>(0.3293, 0.9355, 0.1370),
            vec3<f32>(0.0433, -0.0046, 0.8466),
        );
        // Map scene units → absolute linear cd/m²
        let scene_to_nits = postprocess.hdr_max_nits / max(postprocess.hdr_ui_brightness, 0.001);
        color = color * (scene_to_nits / 10000.0);  // normalise to [0, 1] where 1 = 10000 nits
        // BT.2020 primaries (applied to linear scene values)
        color = REC709_TO_BT2020 * color;
        // PQ ST 2084 per-channel: linear light → non-linear code values
        let Y = pow(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(PQ_M1));
        color = pow((PQ_C1 + PQ_C2 * Y) / (vec3<f32>(1.0) + PQ_C3 * Y), vec3<f32>(PQ_M2));
    } else if postprocess.hdr_output_mode == 2u {
        // scRGB: linear float, 1.0 = 80 cd/m²
        let scene_to_nits = postprocess.hdr_max_nits / max(postprocess.hdr_ui_brightness, 0.001);
        color = color * (scene_to_nits / 80.0);
        color = clamp(color, vec3<f32>(0.0), vec3<f32>(65504.0)); // f16 max
    }
    // LDR (mode 0) and Passthrough (mode 3): pass through as-is

    return vec4<f32>(color, 1.0);
}
