// The post settings the tone map reads (`helio_pass_postprocess::POST_SETTINGS`,
// `GpuPostProcessUniforms` on the CPU), shared with their publishers.


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
    // Local exposure (offset 192; the bytes of the removed chromatic aberration).
    local_exposure_highlight_contrast: f32,
    local_exposure_shadow_contrast:    f32,
    local_exposure_detail_strength:    f32,
    _pad_local_exposure:               f32,
    grain_intensity:        f32,
    grain_response:         f32,
    grain_size:             f32,
    grain_enabled:          u32,
    motion_blur_amount:     f32,
    motion_blur_max:        f32,
    motion_blur_enabled:    u32,
    _pad13:                 f32,
    blend_weight_bloom:        f32,
    blend_weight_motion_blur:  f32,
    blend_weight_vignette:     f32,
    _pad_bw_ca:                f32,
    blend_weight_grain:        f32,
    blend_weight_exposure:     f32,
    _pad14:                    f32,
    _pad15:                    f32,
    // ── HDR Output (16 bytes) ──
    hdr_output_mode:           u32,   // 272
    hdr_max_nits:              f32,   // 276
    hdr_ui_brightness:         f32,   // 280
    _pad_hdr_end:              f32,   // 284
    // ── Advanced Color Grading (80 bytes) ──
    lift_color:                vec3<f32>, // 288
    _pad_lift:                 f32,   // 300
    gamma_color:               vec3<f32>, // 304
    _pad_gamma:                f32,   // 316
    gain_color:                vec3<f32>, // 320
    _pad_gain:                 f32,   // 332
    shadows_max:               f32,   // 336
    highlights_min:            f32,   // 340
    shadow_highlight_balance:  f32,   // 344
    hue_shift:                 f32,   // 348
    lut_generation:            u32,   // 352
    lut_intensity:             f32,   // 356
    lut_platform:              u32,   // 360
    _pad_grading_end:          f32,   // 364
}

