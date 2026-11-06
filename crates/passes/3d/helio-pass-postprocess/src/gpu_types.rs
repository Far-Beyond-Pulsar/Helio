use bytemuck::{Pod, Zeroable};

// ── Tonemap operators ──────────────────────────────────────────────────────────

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TonemapOperator {
    None = 5, // skip tonemapping entirely (clips highlights)
    Aces = 0,
    Filmic = 1,
    Reinhard = 2,
    Uncharted2 = 3,
    Lottes = 4,
    /// AgX (Sobotka): a log encoding in a slightly desaturated working space
    /// and a sigmoid. Bright saturated colours roll toward white instead of
    /// skewing in hue (sunlit grass stays green, not lime).
    Agx = 6,
    /// AgX with its punchy look (Blender's): contrast and saturation in its
    /// display encoding, before it leaves its working space; a deep sky and
    /// rich foliage without AgX's base wash. Colours darker than one EV under
    /// middle grey keep the hue and saturation they have there (darker, not
    /// purer), so shade does not clip to a pure primary.
    AgxPunchy = 7,
}

// ── Exposure mode ──────────────────────────────────────────────────────────────

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExposureMode {
    Manual = 0,
    Auto = 1,
}

// ── HDR output mode ────────────────────────────────────────────────────────────

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HdrOutputMode {
    /// Tonemap → sRGB (current behaviour)
    Ldr = 0,
    /// Tonemap → PQ ST 2084 → BT.2020 → 10-bit
    Hdr10 = 1,
    /// Linear float output (scRGB, Windows HDR)
    ScRgb = 2,
    /// Raw HDR float, no tonemap (for external grading or recording)
    Passthrough = 3,
}

// ── GpuPostProcessUniforms ─────────────────────────────────────────────────────
//
// Flat uniform struct uploaded to GPU each frame. All fields are driven by the
// CPU-side PostProcessBlender which evaluates active volumes + camera defaults.

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuPostProcessUniforms {
    // ── Exposure (16 bytes) ──
    pub exposure_mode: u32,         // 0 = Manual, 1 = Auto (histogram-based)
    pub exposure_compensation: f32, // EV offset applied after metering
    pub exposure_min: f32,          // min EV for auto exposure
    pub exposure_max: f32,          // max EV for auto exposure

    // ── Bloom (8 x 4 = 32 bytes) ──
    pub bloom_intensity: f32,
    pub bloom_threshold: f32,
    pub bloom_knee: f32,   // soft knee around threshold
    pub bloom_radius: f32, // scatter size (1.0 = default)
    pub bloom_tint: [f32; 3],
    pub bloom_enabled: u32,

    // ── Color Grading (12 x 4 = 48 bytes) ──
    pub color_saturation: [f32; 3],
    pub exposure_speed_up: f32,
    pub color_contrast: [f32; 3],
    pub exposure_speed_down: f32,
    pub color_gamma: [f32; 3],
    pub pad_col_gam: f32,
    pub color_gain: [f32; 3],
    pub pad_col_gai: f32,
    pub color_offset: [f32; 3],
    pub pad_col_off: f32,

    // ── White balance (16 bytes) ──
    pub white_temp: f32, // correlated colour temperature (K)
    pub white_tint: f32, // green/magenta offset
    pub white_balance_enabled: u32,
    pub pad_wb: f32,

    // ── Tonemap (16 bytes) ──
    pub tonemap_operator: u32, // TonemapOperator discriminant
    pub tonemap_exposure: f32, // scene-linear exposure multiplier
    pub tonemap_white_point: f32,
    pub pad_tm: f32,

    // ── Vignette (32 bytes) ──
    // _pad_vignette aligns vignette_color to a 16-byte boundary, matching
    // WGSL's vec3<f32> alignment requirement (vec3 align = 16).
    pub vignette_intensity: f32,
    pub vignette_smoothness: f32,
    pub vignette_roundness: f32,
    pub _pad_vignette: f32,
    pub vignette_color: [f32; 3],
    pub vignette_enabled: u32,

    // ── Local exposure (16 bytes, offset 192) ──
    // Edge-aware local tone mapping (`postprocess.wgsl`, `cs_local_exposure_*`):
    // the mean EV of each pixel's neighbourhood of similar brightness, from a
    // bilateral grid, is moved toward middle grey by these contrast scales.
    // 1 leaves that side unchanged (all three at 1 turn the effect off).
    // These bytes held the removed post-process chromatic aberration, so
    // every later offset, the SceneDB row split and the flare binding stay.
    pub local_exposure_highlight_contrast: f32, // 192 — regions above middle grey, 0..1
    pub local_exposure_shadow_contrast: f32,    // 196 — regions below middle grey, 0..1
    pub local_exposure_detail_strength: f32,    // 200 — contrast within a region, 1 = kept
    pub pad_local_exposure: f32,                // 204

    // ── Film Grain (16 bytes) ──
    pub grain_intensity: f32,
    pub grain_response: f32, // curve exponent
    pub grain_size: f32,
    pub grain_enabled: u32,

    // ── Motion Blur (16 bytes) ──
    pub motion_blur_amount: f32,
    pub motion_blur_max: f32,
    pub motion_blur_enabled: u32,
    pub pad_mb: f32,

    // ── Per-effect blend weights (8 x 4 = 32 bytes) ──
    pub blend_weight_bloom: f32,
    pub blend_weight_motion_blur: f32,
    pub blend_weight_vignette: f32,
    pub pad_bw_ca: f32,
    pub blend_weight_grain: f32,
    pub blend_weight_exposure: f32,
    pub pad_bw0: f32,
    pub pad_bw1: f32,

    // ── HDR Output (16 bytes) ──
    pub hdr_output_mode: u32,   // 272
    pub hdr_max_nits: f32,      // 276
    pub hdr_ui_brightness: f32, // 280
    pub pad_hdr_end: f32,       // 284

    // ── Advanced Color Grading (48 bytes) ──
    pub lift_color: [f32; 3],          // 288 — shadow tint
    pub pad_lift: f32,                 // 300
    pub gamma_color: [f32; 3],         // 304 — midtone tint
    pub pad_gamma: f32,                // 316
    pub gain_color: [f32; 3],          // 320 — highlight tint
    pub pad_gain: f32,                 // 332
    pub shadows_max: f32,              // 336 — luminance threshold for shadow region
    pub highlights_min: f32,           // 340 — luminance threshold for highlight region
    pub shadow_highlight_balance: f32, // 344 — 0-1 blend between shadow and highlight
    pub hue_shift: f32,                // 348 — global hue rotation (degrees)
    pub lut_generation: u32,           // 352 — incremented when LUT needs rebuilding
    pub lut_intensity: f32,            // 356 — blend 0-1 between graded and ungraded
    pub lut_platform: u32,             // 360 — 0=none, 1=16x16x16, 2=32x32x32
    pub pad_grading_end: f32,          // 364

    // Lens flare: appended 96-byte block (24 words); shared with the optics
    // pass. Camera optics (f-number, focal length, sensor, iris) are not here:
    // they belong to the camera's `LensRow` (the lens plugin's `helio-pass-lens`).
    pub lens_enabled: u32,          // 368
    pub lens_quality: u32,          // 372
    pub lens_profile: u32,          // 376
    pub lens_ghost_count: u32,      // 380
    pub lens_intensity: f32,        // 384
    pub lens_threshold: f32,        // 388
    pub lens_soft_knee: f32,        // 392
    pub lens_ghost_intensity: f32,  // 396
    pub lens_halo_intensity: f32,   // 400
    pub lens_glare_intensity: f32,  // 404
    pub lens_streak_intensity: f32, // 408
    pub lens_dispersion: f32,       // 412
    /// Reserved: natural cos^4 falloff is the lens plugin's (`helio-pass-lens`).
    pub reserved_lens_vignette: f32, // 416
    pub lens_starburst_intensity: f32, // 420
    pub lens_starburst_length: f32, // 424
    pub lens_coating_strength: f32, // 428
    pub lens_ghost_rim: f32,        // 432
    pub lens_dirt_intensity: f32,   // 436
    pub lens_light_sources: u32,    // 440
    pub lens_light_intensity: f32,  // 444
    pub lens_field_margin: f32,     // 448
    /// Temporal response time constant in seconds (0 = instantaneous).
    pub lens_response_time: f32, // 452
    pub pad_lens: [f32; 2],         // 456..464
}

// Total: 464 bytes (29 uniform slots); the lens block starts at 368. (The fog
// block that sat at 272 left: fog is the fog plugin's media, not a post setting.)
//
// This struct is mirrored by hand in helio-pass-postprocess/shaders/postprocess.wgsl
// and is embedded in GpuPostProcessVolume, which cs_volume_blend reads as a storage
// array. A field added here without updating that mirror misreads the buffer silently.
const _: () = assert!(std::mem::size_of::<GpuPostProcessUniforms>() == 464);
const _: () = assert!(std::mem::size_of::<GpuPostProcessUniforms>() % 16 == 0);

// ── Defaults ───────────────────────────────────────────────────────────────────

/// Default local exposure: bright regions keep most of their contrast, shade
/// is lifted toward middle grey (a region 4 EV under it by 2 EV), and
/// texture within a region keeps its contrast.
pub const LOCAL_EXPOSURE_HIGHLIGHT_CONTRAST: f32 = 0.85;
pub const LOCAL_EXPOSURE_SHADOW_CONTRAST: f32 = 0.5;
pub const LOCAL_EXPOSURE_DETAIL_STRENGTH: f32 = 1.0;

impl Default for GpuPostProcessUniforms {
    fn default() -> Self {
        Self {
            exposure_mode: ExposureMode::Manual as u32,
            exposure_compensation: 0.0,
            exposure_min: -4.0,
            exposure_max: 4.0,

            bloom_intensity: 0.3,
            bloom_threshold: 1.0,
            bloom_knee: 0.1,
            bloom_radius: 1.0,
            bloom_tint: [1.0, 1.0, 1.0],
            bloom_enabled: 0,

            color_saturation: [1.0, 1.0, 1.0],
            exposure_speed_up: 0.5,
            color_contrast: [1.0, 1.0, 1.0],
            exposure_speed_down: 1.0,
            color_gamma: [1.0, 1.0, 1.0],
            pad_col_gam: 0.0,
            color_gain: [1.0, 1.0, 1.0],
            pad_col_gai: 0.0,
            color_offset: [0.0, 0.0, 0.0],
            pad_col_off: 0.0,

            white_temp: 6500.0,
            white_tint: 0.0,
            white_balance_enabled: 0,
            pad_wb: 0.0,

            tonemap_operator: TonemapOperator::AgxPunchy as u32,
            tonemap_exposure: 1.0,
            tonemap_white_point: 1.0,
            pad_tm: 0.0,

            vignette_intensity: 0.0,
            vignette_smoothness: 0.5,
            vignette_roundness: 0.5,
            _pad_vignette: 0.0,
            vignette_color: [0.0, 0.0, 0.0],
            vignette_enabled: 0,

            local_exposure_highlight_contrast: LOCAL_EXPOSURE_HIGHLIGHT_CONTRAST,
            local_exposure_shadow_contrast: LOCAL_EXPOSURE_SHADOW_CONTRAST,
            local_exposure_detail_strength: LOCAL_EXPOSURE_DETAIL_STRENGTH,
            pad_local_exposure: 0.0,

            grain_intensity: 0.0,
            grain_response: 1.0,
            grain_size: 1.0,
            grain_enabled: 0,

            motion_blur_amount: 0.0,
            motion_blur_max: 64.0,
            motion_blur_enabled: 0,
            pad_mb: 0.0,

            blend_weight_bloom: 1.0,
            blend_weight_motion_blur: 1.0,
            blend_weight_vignette: 1.0,
            pad_bw_ca: 0.0,
            blend_weight_grain: 1.0,
            blend_weight_exposure: 1.0,
            pad_bw0: 0.0,
            pad_bw1: 0.0,

            hdr_output_mode: HdrOutputMode::Ldr as u32,
            hdr_max_nits: 1000.0,
            hdr_ui_brightness: 200.0,
            pad_hdr_end: 0.0,


            lift_color: [0.0; 3],
            pad_lift: 0.0,
            gamma_color: [0.0; 3],
            pad_gamma: 0.0,
            gain_color: [1.0; 3],
            pad_gain: 0.0,
            shadows_max: 0.3,
            highlights_min: 0.7,
            shadow_highlight_balance: 0.5,
            hue_shift: 0.0,
            lut_generation: 0,
            lut_intensity: 1.0,
            lut_platform: 0,
            pad_grading_end: 0.0,
            lens_enabled: 0,
            lens_quality: 0,
            lens_profile: 0,
            lens_ghost_count: 4,
            lens_intensity: 0.3,
            lens_threshold: 2.0,
            lens_soft_knee: 1.0,
            lens_ghost_intensity: 1.0,
            lens_halo_intensity: 0.5,
            lens_glare_intensity: 0.3,
            lens_streak_intensity: 0.5,
            lens_dispersion: 0.01,
            reserved_lens_vignette: 0.0,
            lens_starburst_intensity: 0.3,
            lens_starburst_length: 0.25,
            lens_coating_strength: 0.8,
            lens_ghost_rim: 0.5,
            lens_dirt_intensity: 0.0,
            lens_light_sources: 1,
            lens_light_intensity: 1.0,
            lens_field_margin: 0.35,
            lens_response_time: 0.06,
            pad_lens: [0.0; 2],
        }
    }
}

// ── PostProcessSettings (CPU-side, full parameter set) ─────────────────────────
//
// Intended for use in Camera defaults, PostProcessVolume descriptors,
// and as the blending unit for the CPU blender.

#[derive(Clone, Debug)]
pub struct PostProcessSettings {
    pub lens_flare: LensFlareSettings,
    // Exposure
    pub exposure_mode: ExposureMode,
    pub exposure_compensation: f32,
    pub exposure_min: f32,
    pub exposure_max: f32,
    pub exposure_speed_up: f32,   // seconds to bright-adapt
    pub exposure_speed_down: f32, // seconds to dark-adapt

    // Bloom
    pub bloom_intensity: f32,
    pub bloom_threshold: f32,
    pub bloom_knee: f32,
    pub bloom_radius: f32,
    pub bloom_tint: [f32; 3],
    pub bloom_enabled: bool,

    // Color Grading
    pub color_saturation: [f32; 3],
    pub color_contrast: [f32; 3],
    pub color_gamma: [f32; 3],
    pub color_gain: [f32; 3],
    pub color_offset: [f32; 3],

    // White Balance
    pub white_temp: f32,
    pub white_tint: f32,
    pub white_balance_enabled: bool,

    // Tonemap
    pub tonemap_operator: TonemapOperator,
    pub tonemap_exposure: f32,
    pub tonemap_white_point: f32,

    // Vignette
    pub vignette_intensity: f32,
    pub vignette_smoothness: f32,
    pub vignette_roundness: f32,
    pub vignette_color: [f32; 3],
    pub vignette_enabled: bool,

    // Local exposure: shade and bright regions moved toward middle grey,
    // each by its own contrast scale (1 = unchanged, 0 = flattened to grey),
    // with edges kept (no halos at silhouettes). Detail strength scales the
    // contrast within a region (1 = kept, above 1 sharpened).
    pub local_exposure_highlight_contrast: f32,
    pub local_exposure_shadow_contrast: f32,
    pub local_exposure_detail_strength: f32,

    // Film Grain
    pub grain_intensity: f32,
    pub grain_response: f32,
    pub grain_size: f32,
    pub grain_enabled: bool,

    // Motion Blur
    pub motion_blur_amount: f32,
    pub motion_blur_max: f32,
    pub motion_blur_enabled: bool,

    // Per-effect blend weights (for transitions)
    pub blend_weight_bloom: f32,
    pub blend_weight_motion_blur: f32,
    pub blend_weight_vignette: f32,
    pub blend_weight_grain: f32,
    pub blend_weight_exposure: f32,

    // HDR Output
    pub hdr_output_mode: HdrOutputMode,
    pub hdr_max_nits: f32,
    pub hdr_ui_brightness: f32,

    // Advanced Color Grading
    pub lift_color: [f32; 3],
    pub gamma_color: [f32; 3],
    pub gain_color: [f32; 3],
    pub shadows_max: f32,
    pub highlights_min: f32,
    pub shadow_highlight_balance: f32,
    pub hue_shift: f32,
    pub lut_generation: u32,
    pub lut_intensity: f32,
    pub lut_platform: u32,
}

impl PostProcessSettings {
    /// Pack CPU settings into GPU uniform struct.
    pub fn to_gpu(&self) -> GpuPostProcessUniforms {
        GpuPostProcessUniforms {
            exposure_mode: self.exposure_mode as u32,
            exposure_compensation: self.exposure_compensation,
            exposure_min: self.exposure_min,
            exposure_max: self.exposure_max,

            bloom_intensity: self.bloom_intensity,
            bloom_threshold: self.bloom_threshold,
            bloom_knee: self.bloom_knee,
            bloom_radius: self.bloom_radius,
            bloom_tint: self.bloom_tint,
            bloom_enabled: self.bloom_enabled as u32,

            color_saturation: self.color_saturation,
            exposure_speed_up: self.exposure_speed_up,
            color_contrast: self.color_contrast,
            exposure_speed_down: self.exposure_speed_down,
            color_gamma: self.color_gamma,
            pad_col_gam: 0.0,
            color_gain: self.color_gain,
            pad_col_gai: 0.0,
            color_offset: self.color_offset,
            pad_col_off: 0.0,

            white_temp: self.white_temp,
            white_tint: self.white_tint,
            white_balance_enabled: self.white_balance_enabled as u32,
            pad_wb: 0.0,

            tonemap_operator: self.tonemap_operator as u32,
            tonemap_exposure: self.tonemap_exposure,
            tonemap_white_point: self.tonemap_white_point,
            pad_tm: 0.0,

            vignette_intensity: self.vignette_intensity,
            vignette_smoothness: self.vignette_smoothness,
            vignette_roundness: self.vignette_roundness,
            _pad_vignette: 0.0,
            vignette_color: self.vignette_color,
            vignette_enabled: self.vignette_enabled as u32,

            local_exposure_highlight_contrast: self.local_exposure_highlight_contrast,
            local_exposure_shadow_contrast: self.local_exposure_shadow_contrast,
            local_exposure_detail_strength: self.local_exposure_detail_strength,
            pad_local_exposure: 0.0,

            grain_intensity: self.grain_intensity,
            grain_response: self.grain_response,
            grain_size: self.grain_size,
            grain_enabled: self.grain_enabled as u32,

            motion_blur_amount: self.motion_blur_amount,
            motion_blur_max: self.motion_blur_max,
            motion_blur_enabled: self.motion_blur_enabled as u32,
            pad_mb: 0.0,

            blend_weight_bloom: self.blend_weight_bloom,
            blend_weight_motion_blur: self.blend_weight_motion_blur,
            blend_weight_vignette: self.blend_weight_vignette,
            pad_bw_ca: 0.0,
            blend_weight_grain: self.blend_weight_grain,
            blend_weight_exposure: self.blend_weight_exposure,
            pad_bw0: 0.0,
            pad_bw1: 0.0,

            hdr_output_mode: self.hdr_output_mode as u32,
            hdr_max_nits: self.hdr_max_nits,
            hdr_ui_brightness: self.hdr_ui_brightness,
            pad_hdr_end: 0.0,

            lift_color: self.lift_color,
            pad_lift: 0.0,
            gamma_color: self.gamma_color,
            pad_gamma: 0.0,
            gain_color: self.gain_color,
            pad_gain: 0.0,
            shadows_max: self.shadows_max,
            highlights_min: self.highlights_min,
            shadow_highlight_balance: self.shadow_highlight_balance,
            hue_shift: self.hue_shift,
            lut_generation: self.lut_generation,
            lut_intensity: self.lut_intensity,
            lut_platform: self.lut_platform,
            pad_grading_end: 0.0,
            lens_enabled: self.lens_flare.enabled as u32,
            lens_quality: self.lens_flare.quality,
            lens_profile: self.lens_flare.profile,
            lens_ghost_count: self.lens_flare.ghost_count,
            lens_intensity: self.lens_flare.intensity,
            lens_threshold: self.lens_flare.threshold,
            lens_soft_knee: self.lens_flare.soft_knee,
            lens_ghost_intensity: self.lens_flare.ghost_intensity,
            lens_halo_intensity: self.lens_flare.halo_intensity,
            lens_glare_intensity: self.lens_flare.glare_intensity,
            lens_streak_intensity: self.lens_flare.streak_intensity,
            lens_dispersion: self.lens_flare.dispersion,
            reserved_lens_vignette: 0.0,
            lens_starburst_intensity: self.lens_flare.starburst_intensity,
            lens_starburst_length: self.lens_flare.starburst_length,
            lens_coating_strength: self.lens_flare.coating_strength,
            lens_ghost_rim: self.lens_flare.ghost_rim,
            lens_dirt_intensity: self.lens_flare.dirt_intensity,
            lens_light_sources: self.lens_flare.light_sources as u32,
            lens_light_intensity: self.lens_flare.light_intensity,
            lens_field_margin: self.lens_flare.field_margin,
            lens_response_time: self.lens_flare.response_time,
            pad_lens: [0.0; 2],
        }
    }
}

impl Default for PostProcessSettings {
    fn default() -> Self {
        Self {
            lens_flare: LensFlareSettings::default(),
            exposure_mode: ExposureMode::Manual,
            exposure_compensation: 0.0,
            exposure_min: -4.0,
            exposure_max: 4.0,
            exposure_speed_up: 0.5,
            exposure_speed_down: 1.0,

            bloom_intensity: 0.3,
            bloom_threshold: 1.0,
            bloom_knee: 0.1,
            bloom_radius: 1.0,
            bloom_tint: [1.0, 1.0, 1.0],
            bloom_enabled: false,

            color_saturation: [1.0, 1.0, 1.0],
            color_contrast: [1.0, 1.0, 1.0],
            color_gamma: [1.0, 1.0, 1.0],
            color_gain: [1.0, 1.0, 1.0],
            color_offset: [0.0, 0.0, 0.0],

            white_temp: 6500.0,
            white_tint: 0.0,
            white_balance_enabled: false,

            // AgX punchy: every host and scene without a look of its own
            // renders the engine's calibrated one, never clipped HDR.
            tonemap_operator: TonemapOperator::AgxPunchy,
            tonemap_exposure: 1.0,
            tonemap_white_point: 1.0,

            vignette_intensity: 0.0,
            vignette_smoothness: 0.5,
            vignette_roundness: 0.5,
            vignette_color: [0.0, 0.0, 0.0],
            vignette_enabled: false,

            local_exposure_highlight_contrast: LOCAL_EXPOSURE_HIGHLIGHT_CONTRAST,
            local_exposure_shadow_contrast: LOCAL_EXPOSURE_SHADOW_CONTRAST,
            local_exposure_detail_strength: LOCAL_EXPOSURE_DETAIL_STRENGTH,

            grain_intensity: 0.0,
            grain_response: 1.0,
            grain_size: 1.0,
            grain_enabled: false,

            motion_blur_amount: 0.0,
            motion_blur_max: 64.0,
            motion_blur_enabled: false,

            hdr_output_mode: HdrOutputMode::Ldr,
            hdr_max_nits: 1000.0,
            hdr_ui_brightness: 200.0,

            blend_weight_bloom: 1.0,
            blend_weight_motion_blur: 1.0,
            blend_weight_vignette: 1.0,
            blend_weight_grain: 1.0,
            blend_weight_exposure: 1.0,

            lift_color: [0.0; 3],
            gamma_color: [0.0; 3],
            gain_color: [1.0; 3],
            shadows_max: 0.3,
            highlights_min: 0.7,
            shadow_highlight_balance: 0.5,
            hue_shift: 0.0,
            lut_generation: 0,
            lut_intensity: 1.0,
            lut_platform: 0,
        }
    }
}

// ── GpuPostProcessVolume ───────────────────────────────────────────────────────
//
// Per-volume GPU struct read by the cs_volume_blend compute shader.
// The `unbound` flag marks volumes that apply regardless of camera position.

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuPostProcessVolume {
    pub bounds_min: [f32; 4],
    pub bounds_max: [f32; 4],
    pub priority: f32,
    pub blend_radius: f32,
    pub blend_weight: f32, // 0-1, global volume opacity
    pub unbound: u32,      // 0 = bounded, 1 = applies everywhere
    // 16 bytes, not 8. `settings` contains vec3s, so WGSL gives it 16-byte
    // alignment and places it at offset 64 — while #[repr(C)] would happily put
    // it at 56 after an 8-byte pad. That 8-byte skew made the GPU read every
    // volume's settings shifted (and stride 424 against WGSL's 432, so each
    // successive volume drifted further). Padding to 64 here makes both sides
    // agree; the asserts below keep them that way.
    pub override_mask: [u32; 4],
    pub settings: GpuPostProcessUniforms,
}

// WGSL places `settings` at 64 because GpuPostProcessUniforms aligns to 16.
const _: () = assert!(std::mem::offset_of!(GpuPostProcessVolume, settings) == 64);
// Storage-buffer array stride must match WGSL's, which rounds to the 16-byte alignment.
// 64 (header) + 464 (settings) = 528.
const _: () = assert!(std::mem::size_of::<GpuPostProcessVolume>() == 528);
const _: () = assert!(std::mem::size_of::<GpuPostProcessVolume>() % 16 == 0);

// ── PostProcessVolume descriptor (CPU-side) ────────────────────────────────────

#[derive(Clone, Debug)]
pub struct PostProcessVolumeDescriptor {
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub priority: f32,
    pub blend_radius: f32,
    pub blend_weight: f32,
    pub unbound: bool, // infinite volume (camera always inside)
    pub override_mask: [u32; 4],
    pub settings: PostProcessSettings,
}

impl Default for PostProcessVolumeDescriptor {
    fn default() -> Self {
        Self {
            bounds_min: [-1000.0, -1000.0, -1000.0],
            bounds_max: [1000.0, 1000.0, 1000.0],
            priority: 0.0,
            blend_radius: 200.0,
            blend_weight: 1.0,
            unbound: false,
            override_mask: PostProcessProperty::ALL,
            settings: PostProcessSettings::default(),
        }
    }
}

impl PostProcessVolumeDescriptor {
    pub fn to_gpu(&self) -> GpuPostProcessVolume {
        GpuPostProcessVolume {
            bounds_min: [
                self.bounds_min[0],
                self.bounds_min[1],
                self.bounds_min[2],
                0.0,
            ],
            bounds_max: [
                self.bounds_max[0],
                self.bounds_max[1],
                self.bounds_max[2],
                0.0,
            ],
            priority: self.priority,
            blend_radius: self.blend_radius,
            blend_weight: self.blend_weight,
            unbound: self.unbound as u32,
            override_mask: self.override_mask,
            settings: self.settings.to_gpu(),
        }
    }
}

/// Shared lens authoring settings. Quality: 0 cheap, 1 high; profile: 0 spherical, 1 anamorphic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LensFlareSettings {
    pub enabled: bool,
    pub quality: u32,
    pub profile: u32,
    pub ghost_count: u32,
    pub intensity: f32,
    pub threshold: f32,
    pub soft_knee: f32,
    pub ghost_intensity: f32,
    pub halo_intensity: f32,
    pub glare_intensity: f32,
    pub streak_intensity: f32,
    pub dispersion: f32,
    /// Diffraction spikes from the iris blades. Gain and length (fraction of width).
    pub starburst_intensity: f32,
    pub starburst_length: f32,
    /// 0 = neutral grey ghosts; 1 = full anti-reflection coating tints.
    pub coating_strength: f32,
    /// Brighter, chromatically fringed ghost edges (0 = flat disks).
    pub ghost_rim: f32,
    /// Lens dirt modulating light scattered off the front element.
    pub dirt_intensity: f32,
    /// Analytic lens sources from scene lights: flares persist and fade
    /// physically when a light leaves the frame, with shadow-map visibility.
    pub light_sources: bool,
    pub light_intensity: f32,
    /// How far past the frame edge (fraction of the half-frame) light still
    /// reaches the front element before the lens barrel blocks it.
    pub field_margin: f32,
    /// Seconds for the lens response to settle after a change (occlusion,
    /// entering the frame, enabling). Fades instead of popping; 0 = instant.
    pub response_time: f32,
}
impl Default for LensFlareSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            quality: 0,
            profile: 0,
            ghost_count: 4,
            intensity: 0.3,
            // Full soft knee: a smooth ramp from 0 to 2x threshold, so sources
            // never switch at a cutoff, while dim extended areas (a lit floor)
            // do not form recognisable ghost images. 0/0 is fully linear.
            threshold: 2.0,
            soft_knee: 1.0,
            ghost_intensity: 1.0,
            halo_intensity: 0.5,
            glare_intensity: 0.3,
            streak_intensity: 0.5,
            dispersion: 0.01,
            starburst_intensity: 0.3,
            starburst_length: 0.25,
            coating_strength: 0.8,
            ghost_rim: 0.5,
            dirt_intensity: 0.0,
            light_sources: true,
            light_intensity: 1.0,
            field_margin: 0.35,
            response_time: 0.06,
        }
    }
}

impl GpuPostProcessUniforms {
    pub const LENS_BLOCK_OFFSET: u64 = 368;
    pub const LENS_BLOCK_SIZE: u64 = 96;
}
const _: () = assert!(std::mem::offset_of!(GpuPostProcessUniforms, lens_enabled) == 368);

/// Override bit indices, in non-padding, non-reserved GPU declaration order, with later
/// additions appended (saved masks keep their meaning). Vector colors use one bit.
/// ALL is the default for authored volumes; a zero mask overrides nothing.
pub struct PostProcessProperty;
#[allow(non_upper_case_globals)]
impl PostProcessProperty {
    pub const ALL: [u32; 4] = [u32::MAX; 4];
    pub const NONE: [u32; 4] = [0; 4];
    pub const EXPOSURE_MODE: usize = 0;
    pub const EXPOSURE_COMPENSATION: usize = 1;
    pub const EXPOSURE_MIN: usize = 2;
    pub const EXPOSURE_MAX: usize = 3;
    pub const BLOOM_INTENSITY: usize = 4;
    pub const BLOOM_THRESHOLD: usize = 5;
    pub const BLOOM_KNEE: usize = 6;
    pub const BLOOM_RADIUS: usize = 7;
    pub const BLOOM_TINT: usize = 8;
    pub const BLOOM_ENABLED: usize = 9;
    pub const COLOR_SATURATION: usize = 10;
    pub const EXPOSURE_SPEED_UP: usize = 11;
    pub const COLOR_CONTRAST: usize = 12;
    pub const EXPOSURE_SPEED_DOWN: usize = 13;
    pub const COLOR_GAMMA: usize = 14;
    pub const COLOR_GAIN: usize = 15;
    pub const COLOR_OFFSET: usize = 16;
    pub const WHITE_TEMP: usize = 17;
    pub const WHITE_TINT: usize = 18;
    pub const WHITE_BALANCE_ENABLED: usize = 19;
    pub const TONEMAP_OPERATOR: usize = 20;
    pub const TONEMAP_EXPOSURE: usize = 21;
    pub const TONEMAP_WHITE_POINT: usize = 22;
    pub const VIGNETTE_INTENSITY: usize = 23;
    pub const VIGNETTE_SMOOTHNESS: usize = 24;
    pub const VIGNETTE_ROUNDNESS: usize = 25;
    pub const VIGNETTE_COLOR: usize = 26;
    pub const VIGNETTE_ENABLED: usize = 27;
    pub const GRAIN_INTENSITY: usize = 28;
    pub const GRAIN_RESPONSE: usize = 29;
    pub const GRAIN_SIZE: usize = 30;
    pub const GRAIN_ENABLED: usize = 31;
    pub const MOTION_BLUR_AMOUNT: usize = 32;
    pub const MOTION_BLUR_MAX: usize = 33;
    pub const MOTION_BLUR_ENABLED: usize = 34;
    pub const BLEND_WEIGHT_BLOOM: usize = 35;
    pub const BLEND_WEIGHT_MOTION_BLUR: usize = 36;
    pub const BLEND_WEIGHT_VIGNETTE: usize = 37;
    pub const BLEND_WEIGHT_GRAIN: usize = 38;
    pub const BLEND_WEIGHT_EXPOSURE: usize = 39;
    // 40..=49 were the fog block's (retired: fog is the fog plugin's media);
    // the numbers are not reused, so saved masks keep their meaning.
    pub const RETIRED: std::ops::RangeInclusive<usize> = 40..=49;
    pub const HDR_OUTPUT_MODE: usize = 50;
    pub const HDR_MAX_NITS: usize = 51;
    pub const HDR_UI_BRIGHTNESS: usize = 52;
    pub const LIFT_COLOR: usize = 53;
    pub const GAMMA_COLOR: usize = 54;
    pub const GAIN_COLOR: usize = 55;
    pub const SHADOWS_MAX: usize = 56;
    pub const HIGHLIGHTS_MIN: usize = 57;
    pub const SHADOW_HIGHLIGHT_BALANCE: usize = 58;
    pub const HUE_SHIFT: usize = 59;
    pub const LUT_GENERATION: usize = 60;
    pub const LUT_INTENSITY: usize = 61;
    pub const LUT_PLATFORM: usize = 62;
    pub const LENS_ENABLED: usize = 63;
    pub const LENS_QUALITY: usize = 64;
    pub const LENS_PROFILE: usize = 65;
    pub const LENS_GHOST_COUNT: usize = 66;
    pub const LENS_INTENSITY: usize = 67;
    pub const LENS_THRESHOLD: usize = 68;
    pub const LENS_SOFT_KNEE: usize = 69;
    pub const LENS_GHOST_INTENSITY: usize = 70;
    pub const LENS_HALO_INTENSITY: usize = 71;
    pub const LENS_GLARE_INTENSITY: usize = 72;
    pub const LENS_STREAK_INTENSITY: usize = 73;
    pub const LENS_DISPERSION: usize = 74;
    pub const LENS_STARBURST_INTENSITY: usize = 75;
    pub const LENS_STARBURST_LENGTH: usize = 76;
    pub const LENS_COATING_STRENGTH: usize = 77;
    pub const LENS_GHOST_RIM: usize = 78;
    pub const LENS_DIRT_INTENSITY: usize = 79;
    pub const LENS_LIGHT_SOURCES: usize = 80;
    pub const LENS_LIGHT_INTENSITY: usize = 81;
    pub const LENS_FIELD_MARGIN: usize = 82;
    pub const LENS_RESPONSE_TIME: usize = 83;
    pub const LOCAL_EXPOSURE_HIGHLIGHT_CONTRAST: usize = 84;
    pub const LOCAL_EXPOSURE_SHADOW_CONTRAST: usize = 85;
    pub const LOCAL_EXPOSURE_DETAIL_STRENGTH: usize = 86;
    /// One past the last property index.
    pub const COUNT: usize = 87;
    pub fn set(mask: &mut [u32; 4], property: usize, enabled: bool) {
        assert!(property < Self::COUNT);
        let bit = 1 << (property % 32);
        if enabled {
            mask[property / 32] |= bit;
        } else {
            mask[property / 32] &= !bit;
        }
    }
    pub fn contains(mask: &[u32; 4], property: usize) -> bool {
        property < Self::COUNT && mask[property / 32] & (1 << (property % 32)) != 0
    }
}

/// CPU reference for the production GPU resolver. Equal priorities retain input row order.
pub struct PostProcessBlender;
impl PostProcessBlender {
    pub fn blend(
        camera_pos: [f32; 3],
        volumes: &[GpuPostProcessVolume],
        camera_settings: &PostProcessSettings,
    ) -> GpuPostProcessUniforms {
        Self::blend_gpu(camera_pos, volumes, camera_settings.to_gpu())
    }

    pub fn blend_gpu(
        camera_pos: [f32; 3],
        volumes: &[GpuPostProcessVolume],
        baseline: GpuPostProcessUniforms,
    ) -> GpuPostProcessUniforms {
        let mut ordered: Vec<_> = volumes
            .iter()
            .filter(|v| {
                v.priority.is_finite() && v.blend_weight.is_finite() && v.blend_weight > 0.0
            })
            .collect();
        ordered.sort_by(|a, b| a.priority.partial_cmp(&b.priority).unwrap());
        let mut result = baseline;
        for v in ordered {
            let weight = volume_weight(camera_pos, v);
            if weight > 0.0 {
                result = blend_properties(result, &v.settings, weight, &v.override_mask);
            }
        }
        result
    }
}

pub fn volume_weight(pos: [f32; 3], v: &GpuPostProcessVolume) -> f32 {
    if !v.blend_weight.is_finite() || v.blend_weight <= 0.0 {
        return 0.0;
    }
    let weight = v.blend_weight.clamp(0.0, 1.0);
    if v.unbound != 0 {
        return weight;
    }
    let mut boundary = f32::INFINITY;
    for axis in 0..3 {
        if !(pos[axis] >= v.bounds_min[axis] && pos[axis] <= v.bounds_max[axis]) {
            return 0.0;
        }
        boundary = boundary
            .min(pos[axis] - v.bounds_min[axis])
            .min(v.bounds_max[axis] - pos[axis]);
    }
    if v.blend_radius > 0.0 {
        weight * (boundary / v.blend_radius).clamp(0.0, 1.0)
    } else {
        weight
    }
}

fn blend_properties(
    mut a: GpuPostProcessUniforms,
    b: &GpuPostProcessUniforms,
    t: f32,
    mask: &[u32; 4],
) -> GpuPostProcessUniforms {
    if PostProcessProperty::contains(mask, 0) {
        if t >= 0.5 {
            a.exposure_mode = b.exposure_mode;
        }
    }
    if PostProcessProperty::contains(mask, 1) {
        a.exposure_compensation += (b.exposure_compensation - a.exposure_compensation) * t;
    }
    if PostProcessProperty::contains(mask, 2) {
        a.exposure_min += (b.exposure_min - a.exposure_min) * t;
    }
    if PostProcessProperty::contains(mask, 3) {
        a.exposure_max += (b.exposure_max - a.exposure_max) * t;
    }
    if PostProcessProperty::contains(mask, 4) {
        a.bloom_intensity += (b.bloom_intensity - a.bloom_intensity) * t;
    }
    if PostProcessProperty::contains(mask, 5) {
        a.bloom_threshold += (b.bloom_threshold - a.bloom_threshold) * t;
    }
    if PostProcessProperty::contains(mask, 6) {
        a.bloom_knee += (b.bloom_knee - a.bloom_knee) * t;
    }
    if PostProcessProperty::contains(mask, 7) {
        a.bloom_radius += (b.bloom_radius - a.bloom_radius) * t;
    }
    if PostProcessProperty::contains(mask, 8) {
        for c in 0..3 {
            a.bloom_tint[c] += (b.bloom_tint[c] - a.bloom_tint[c]) * t;
        }
    }
    if PostProcessProperty::contains(mask, 9) {
        if t >= 0.5 {
            a.bloom_enabled = b.bloom_enabled;
        }
    }
    if PostProcessProperty::contains(mask, 10) {
        for c in 0..3 {
            a.color_saturation[c] += (b.color_saturation[c] - a.color_saturation[c]) * t;
        }
    }
    if PostProcessProperty::contains(mask, 11) {
        a.exposure_speed_up += (b.exposure_speed_up - a.exposure_speed_up) * t;
    }
    if PostProcessProperty::contains(mask, 12) {
        for c in 0..3 {
            a.color_contrast[c] += (b.color_contrast[c] - a.color_contrast[c]) * t;
        }
    }
    if PostProcessProperty::contains(mask, 13) {
        a.exposure_speed_down += (b.exposure_speed_down - a.exposure_speed_down) * t;
    }
    if PostProcessProperty::contains(mask, 14) {
        for c in 0..3 {
            a.color_gamma[c] += (b.color_gamma[c] - a.color_gamma[c]) * t;
        }
    }
    if PostProcessProperty::contains(mask, 15) {
        for c in 0..3 {
            a.color_gain[c] += (b.color_gain[c] - a.color_gain[c]) * t;
        }
    }
    if PostProcessProperty::contains(mask, 16) {
        for c in 0..3 {
            a.color_offset[c] += (b.color_offset[c] - a.color_offset[c]) * t;
        }
    }
    if PostProcessProperty::contains(mask, 17) {
        a.white_temp += (b.white_temp - a.white_temp) * t;
    }
    if PostProcessProperty::contains(mask, 18) {
        a.white_tint += (b.white_tint - a.white_tint) * t;
    }
    if PostProcessProperty::contains(mask, 19) {
        if t >= 0.5 {
            a.white_balance_enabled = b.white_balance_enabled;
        }
    }
    if PostProcessProperty::contains(mask, 20) {
        if t >= 0.5 {
            a.tonemap_operator = b.tonemap_operator;
        }
    }
    if PostProcessProperty::contains(mask, 21) {
        a.tonemap_exposure += (b.tonemap_exposure - a.tonemap_exposure) * t;
    }
    if PostProcessProperty::contains(mask, 22) {
        a.tonemap_white_point += (b.tonemap_white_point - a.tonemap_white_point) * t;
    }
    if PostProcessProperty::contains(mask, 23) {
        a.vignette_intensity += (b.vignette_intensity - a.vignette_intensity) * t;
    }
    if PostProcessProperty::contains(mask, 24) {
        a.vignette_smoothness += (b.vignette_smoothness - a.vignette_smoothness) * t;
    }
    if PostProcessProperty::contains(mask, 25) {
        a.vignette_roundness += (b.vignette_roundness - a.vignette_roundness) * t;
    }
    if PostProcessProperty::contains(mask, 26) {
        for c in 0..3 {
            a.vignette_color[c] += (b.vignette_color[c] - a.vignette_color[c]) * t;
        }
    }
    if PostProcessProperty::contains(mask, 27) {
        if t >= 0.5 {
            a.vignette_enabled = b.vignette_enabled;
        }
    }
    if PostProcessProperty::contains(mask, 28) {
        a.grain_intensity += (b.grain_intensity - a.grain_intensity) * t;
    }
    if PostProcessProperty::contains(mask, 29) {
        a.grain_response += (b.grain_response - a.grain_response) * t;
    }
    if PostProcessProperty::contains(mask, 30) {
        a.grain_size += (b.grain_size - a.grain_size) * t;
    }
    if PostProcessProperty::contains(mask, 31) {
        if t >= 0.5 {
            a.grain_enabled = b.grain_enabled;
        }
    }
    if PostProcessProperty::contains(mask, 32) {
        a.motion_blur_amount += (b.motion_blur_amount - a.motion_blur_amount) * t;
    }
    if PostProcessProperty::contains(mask, 33) {
        a.motion_blur_max += (b.motion_blur_max - a.motion_blur_max) * t;
    }
    if PostProcessProperty::contains(mask, 34) {
        if t >= 0.5 {
            a.motion_blur_enabled = b.motion_blur_enabled;
        }
    }
    if PostProcessProperty::contains(mask, 35) {
        a.blend_weight_bloom += (b.blend_weight_bloom - a.blend_weight_bloom) * t;
    }
    if PostProcessProperty::contains(mask, 36) {
        a.blend_weight_motion_blur += (b.blend_weight_motion_blur - a.blend_weight_motion_blur) * t;
    }
    if PostProcessProperty::contains(mask, 37) {
        a.blend_weight_vignette += (b.blend_weight_vignette - a.blend_weight_vignette) * t;
    }
    if PostProcessProperty::contains(mask, 38) {
        a.blend_weight_grain += (b.blend_weight_grain - a.blend_weight_grain) * t;
    }
    if PostProcessProperty::contains(mask, 39) {
        a.blend_weight_exposure += (b.blend_weight_exposure - a.blend_weight_exposure) * t;
    }
    if PostProcessProperty::contains(mask, 50) {
        if t >= 0.5 {
            a.hdr_output_mode = b.hdr_output_mode;
        }
    }
    if PostProcessProperty::contains(mask, 51) {
        a.hdr_max_nits += (b.hdr_max_nits - a.hdr_max_nits) * t;
    }
    if PostProcessProperty::contains(mask, 52) {
        a.hdr_ui_brightness += (b.hdr_ui_brightness - a.hdr_ui_brightness) * t;
    }
    if PostProcessProperty::contains(mask, 53) {
        for c in 0..3 {
            a.lift_color[c] += (b.lift_color[c] - a.lift_color[c]) * t;
        }
    }
    if PostProcessProperty::contains(mask, 54) {
        for c in 0..3 {
            a.gamma_color[c] += (b.gamma_color[c] - a.gamma_color[c]) * t;
        }
    }
    if PostProcessProperty::contains(mask, 55) {
        for c in 0..3 {
            a.gain_color[c] += (b.gain_color[c] - a.gain_color[c]) * t;
        }
    }
    if PostProcessProperty::contains(mask, 56) {
        a.shadows_max += (b.shadows_max - a.shadows_max) * t;
    }
    if PostProcessProperty::contains(mask, 57) {
        a.highlights_min += (b.highlights_min - a.highlights_min) * t;
    }
    if PostProcessProperty::contains(mask, 58) {
        a.shadow_highlight_balance += (b.shadow_highlight_balance - a.shadow_highlight_balance) * t;
    }
    if PostProcessProperty::contains(mask, 59) {
        a.hue_shift += (b.hue_shift - a.hue_shift) * t;
    }
    if PostProcessProperty::contains(mask, 60) {
        if t >= 0.5 {
            a.lut_generation = b.lut_generation;
        }
    }
    if PostProcessProperty::contains(mask, 61) {
        a.lut_intensity += (b.lut_intensity - a.lut_intensity) * t;
    }
    if PostProcessProperty::contains(mask, 62) {
        if t >= 0.5 {
            a.lut_platform = b.lut_platform;
        }
    }
    if PostProcessProperty::contains(mask, 63) {
        if t >= 0.5 {
            a.lens_enabled = b.lens_enabled;
        }
    }
    if PostProcessProperty::contains(mask, 64) {
        if t >= 0.5 {
            a.lens_quality = b.lens_quality;
        }
    }
    if PostProcessProperty::contains(mask, 65) {
        if t >= 0.5 {
            a.lens_profile = b.lens_profile;
        }
    }
    if PostProcessProperty::contains(mask, 66) {
        if t >= 0.5 {
            a.lens_ghost_count = b.lens_ghost_count;
        }
    }
    if PostProcessProperty::contains(mask, 67) {
        a.lens_intensity += (b.lens_intensity - a.lens_intensity) * t;
    }
    if PostProcessProperty::contains(mask, 68) {
        a.lens_threshold += (b.lens_threshold - a.lens_threshold) * t;
    }
    if PostProcessProperty::contains(mask, 69) {
        a.lens_soft_knee += (b.lens_soft_knee - a.lens_soft_knee) * t;
    }
    if PostProcessProperty::contains(mask, 70) {
        a.lens_ghost_intensity += (b.lens_ghost_intensity - a.lens_ghost_intensity) * t;
    }
    if PostProcessProperty::contains(mask, 71) {
        a.lens_halo_intensity += (b.lens_halo_intensity - a.lens_halo_intensity) * t;
    }
    if PostProcessProperty::contains(mask, 72) {
        a.lens_glare_intensity += (b.lens_glare_intensity - a.lens_glare_intensity) * t;
    }
    if PostProcessProperty::contains(mask, 73) {
        a.lens_streak_intensity += (b.lens_streak_intensity - a.lens_streak_intensity) * t;
    }
    if PostProcessProperty::contains(mask, 74) {
        a.lens_dispersion += (b.lens_dispersion - a.lens_dispersion) * t;
    }
    if PostProcessProperty::contains(mask, 75) {
        a.lens_starburst_intensity += (b.lens_starburst_intensity - a.lens_starburst_intensity) * t;
    }
    if PostProcessProperty::contains(mask, 76) {
        a.lens_starburst_length += (b.lens_starburst_length - a.lens_starburst_length) * t;
    }
    if PostProcessProperty::contains(mask, 77) {
        a.lens_coating_strength += (b.lens_coating_strength - a.lens_coating_strength) * t;
    }
    if PostProcessProperty::contains(mask, 78) {
        a.lens_ghost_rim += (b.lens_ghost_rim - a.lens_ghost_rim) * t;
    }
    if PostProcessProperty::contains(mask, 79) {
        a.lens_dirt_intensity += (b.lens_dirt_intensity - a.lens_dirt_intensity) * t;
    }
    if PostProcessProperty::contains(mask, 80) {
        if t >= 0.5 {
            a.lens_light_sources = b.lens_light_sources;
        }
    }
    if PostProcessProperty::contains(mask, 81) {
        a.lens_light_intensity += (b.lens_light_intensity - a.lens_light_intensity) * t;
    }
    if PostProcessProperty::contains(mask, 82) {
        a.lens_field_margin += (b.lens_field_margin - a.lens_field_margin) * t;
    }
    if PostProcessProperty::contains(mask, 83) {
        a.lens_response_time += (b.lens_response_time - a.lens_response_time) * t;
    }
    if PostProcessProperty::contains(mask, 84) {
        a.local_exposure_highlight_contrast +=
            (b.local_exposure_highlight_contrast - a.local_exposure_highlight_contrast) * t;
    }
    if PostProcessProperty::contains(mask, 85) {
        a.local_exposure_shadow_contrast +=
            (b.local_exposure_shadow_contrast - a.local_exposure_shadow_contrast) * t;
    }
    if PostProcessProperty::contains(mask, 86) {
        a.local_exposure_detail_strength +=
            (b.local_exposure_detail_strength - a.local_exposure_detail_strength) * t;
    }
    a
}
