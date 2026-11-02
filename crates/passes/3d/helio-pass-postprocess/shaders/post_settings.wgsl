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
    // ── Volumetric fog (64 bytes, offsets 272..336) ──
    // fog_color lands at 304 and fog_emissive at 320 — both multiples of 16, which
    // is what lets this match #[repr(C)] on the CPU. vec3<f32> aligns to 16 in WGSL
    // but to 4 in Rust, so reordering these fields silently desyncs the two sides.
    fog_enabled:               u32,   // 272
    fog_mode:                  u32,   // 276
    fog_density:               f32,   // 280
    fog_height_falloff:        f32,   // 284
    fog_start_distance:        f32,   // 288
    fog_max_distance:          f32,   // 292
    fog_height:                f32,   // 296
    fog_scattering_anisotropy: f32,   // 300
    fog_color:                 vec3<f32>, // 304
    _pad_fog_color:            f32,   // 316
    fog_emissive:              vec3<f32>, // 320
    _pad_fog_emissive:         f32,   // 332
    // ── HDR Output (16 bytes) ──
    hdr_output_mode:           u32,   // 336
    hdr_max_nits:              f32,   // 340
    hdr_ui_brightness:         f32,   // 344
    _pad_hdr_end:              f32,   // 348
    // ── Advanced Color Grading (80 bytes) ──
    lift_color:                vec3<f32>, // 352
    _pad_lift:                 f32,   // 364
    gamma_color:               vec3<f32>, // 368
    _pad_gamma:                f32,   // 380
    gain_color:                vec3<f32>, // 384
    _pad_gain:                 f32,   // 396
    shadows_max:               f32,   // 400
    highlights_min:            f32,   // 404
    shadow_highlight_balance:  f32,   // 408
    hue_shift:                 f32,   // 412
    lut_generation:            u32,   // 416
    lut_intensity:             f32,   // 420
    lut_platform:              u32,   // 424
    _pad_grading_end:          f32,   // 428
    lens_enabled: u32,             // 432
    lens_quality: u32,             // 436
    lens_profile: u32,             // 440
    lens_ghost_count: u32,         // 444
    lens_intensity: f32,           // 448
    lens_threshold: f32,           // 452
    lens_soft_knee: f32,           // 456
    lens_ghost_intensity: f32,     // 460
    lens_halo_intensity: f32,      // 464
    lens_glare_intensity: f32,     // 468
    lens_streak_intensity: f32,    // 472
    lens_dispersion: f32,          // 476
    _reserved_lens_vignette: f32,  // 480
    lens_starburst_intensity: f32, // 484
    lens_starburst_length: f32,    // 488
    lens_coating_strength: f32,    // 492
    lens_ghost_rim: f32,           // 496
    lens_dirt_intensity: f32,      // 500
    lens_light_sources: u32,       // 504
    lens_light_intensity: f32,     // 508
    lens_field_margin: f32,        // 512
    lens_response_time: f32,       // 516
    _pad_lens0: f32,               // 520
    _pad_lens1: f32,               // 524 → struct ends at 528
}

