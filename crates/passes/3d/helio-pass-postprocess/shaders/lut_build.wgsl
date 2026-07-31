// Must match the CPU-side GpuPostProcessUniforms layout (464 bytes).
// Only the fields used by the LUT builder are declared; the rest are
// accounted for as padding to keep offsets correct.
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
    // Color grading (48 bytes) — old simple path, not used by LUT builder
    _simple_sat:            vec3<f32>,
    _pad_a:                 f32,
    _simple_con:            vec3<f32>,
    _pad_b:                 f32,
    _simple_gam:            vec3<f32>,
    _pad_c:                 f32,
    _simple_gain:           vec3<f32>,
    _pad_d:                 f32,
    _simple_off:            vec3<f32>,
    _pad_e:                 f32,
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
    fog_enabled:               u32,
    fog_mode:                  u32,
    fog_density:               f32,
    fog_height_falloff:        f32,
    fog_start_distance:        f32,
    fog_max_distance:          f32,
    fog_height:                f32,
    fog_scattering_anisotropy: f32,
    fog_color:                 vec3<f32>,
    _pad_fog_color:            f32,
    fog_emissive:              vec3<f32>,
    _pad_fog_emissive:         f32,
    hdr_output_mode:           u32,
    hdr_max_nits:              f32,
    hdr_ui_brightness:         f32,
    _pad_hdr_end:              f32,
    // ── Advanced Color Grading (80 bytes) — used by LUT builder ──
    lift_color:                vec3<f32>,
    _pad_lift:                 f32,
    gamma_color:               vec3<f32>,
    _pad_gamma:                f32,
    gain_color:                vec3<f32>,
    _pad_gain:                 f32,
    shadows_max:               f32,
    highlights_min:            f32,
    shadow_highlight_balance:  f32,
    hue_shift:                 f32,
    lut_generation:            u32,
    lut_intensity:             f32,
    lut_platform:              u32,
    _pad_grading_end:          f32,
}

@group(0) @binding(0) var<uniform> postprocess: GpuPostProcessUniforms;
@group(0) @binding(1) var lut_output: texture_storage_3d<rgba16float, write>;

override lut_size: u32 = 16u;

fn hue_shift_rgb(c: vec3<f32>, shift_deg: f32) -> vec3<f32> {
    if abs(shift_deg) < 0.001 { return c; }
    let angle = shift_deg * 3.14159265 / 180.0;
    let cos_a = cos(angle);
    let sin_a = sin(angle);
    let m = mat3x3<f32>(
        vec3<f32>(0.213, 0.213 - 0.213 * cos_a + 0.144 * sin_a, 0.213 - 0.213 * cos_a - 0.756 * sin_a),
        vec3<f32>(0.715, 0.715 - 0.715 * cos_a - 0.283 * sin_a, 0.715 - 0.715 * cos_a + 0.416 * sin_a),
        vec3<f32>(0.072, 0.072 - 0.072 * cos_a + 0.860 * sin_a, 0.072 - 0.072 * cos_a - 0.461 * sin_a),
    );
    return m * c;
}

fn apply_lift_gamma_gain(c: vec3<f32>) -> vec3<f32> {
    let lift  = postprocess.lift_color;
    let gamma = postprocess.gamma_color;
    let gain  = postprocess.gain_color;
    var result = c * (vec3<f32>(1.0) - lift) + lift;
    result = pow(max(result, vec3<f32>(0.0)), gamma + vec3<f32>(1.0));
    result = result * gain;
    return result;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if any(gid >= vec3<u32>(lut_size)) { return; }
    let coord = (vec3<f32>(gid) + 0.5) / f32(lut_size);
    var graded = hue_shift_rgb(coord, postprocess.hue_shift);
    graded = apply_lift_gamma_gain(graded);
    textureStore(lut_output, vec3<i32>(gid), vec4<f32>(graded, 1.0));
}
