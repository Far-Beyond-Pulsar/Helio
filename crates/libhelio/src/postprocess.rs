use bytemuck::{Pod, Zeroable};

// ── Tonemap operators ──────────────────────────────────────────────────────────

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TonemapOperator {
    None = 5,       // skip tonemapping entirely (default)
    Aces = 0,
    Filmic = 1,
    Reinhard = 2,
    Uncharted2 = 3,
    Lottes = 4,
}

// ── Exposure mode ──────────────────────────────────────────────────────────────

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExposureMode {
    Manual = 0,
    Auto = 1,
}

// ── GpuPostProcessUniforms ─────────────────────────────────────────────────────
//
// Flat uniform struct uploaded to GPU each frame. All fields are driven by the
// CPU-side PostProcessBlender which evaluates active volumes + camera defaults.

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuPostProcessUniforms {
    // ── Exposure (16 bytes) ──
    pub exposure_mode: u32,           // 0 = Manual, 1 = Auto (histogram-based)
    pub exposure_compensation: f32,   // EV offset applied after metering
    pub exposure_min: f32,            // min EV for auto exposure
    pub exposure_max: f32,            // max EV for auto exposure

    // ── Bloom (8 x 4 = 32 bytes) ──
    pub bloom_intensity: f32,
    pub bloom_threshold: f32,
    pub bloom_knee: f32,              // soft knee around threshold
    pub bloom_radius: f32,            // scatter size (1.0 = default)
    pub bloom_tint: [f32; 3],
    pub bloom_enabled: u32,

    // ── Color Grading (12 x 4 = 48 bytes) ──
    pub color_saturation: [f32; 3],
    pub pad_col_sat: f32,
    pub color_contrast: [f32; 3],
    pub pad_col_con: f32,
    pub color_gamma: [f32; 3],
    pub pad_col_gam: f32,
    pub color_gain: [f32; 3],
    pub pad_col_gai: f32,
    pub color_offset: [f32; 3],
    pub pad_col_off: f32,

    // ── White balance (16 bytes) ──
    pub white_temp: f32,              // correlated colour temperature (K)
    pub white_tint: f32,              // green/magenta offset
    pub white_balance_enabled: u32,
    pub pad_wb: f32,

    // ── Tonemap (16 bytes) ──
    pub tonemap_operator: u32,        // TonemapOperator discriminant
    pub tonemap_exposure: f32,        // scene-linear exposure multiplier
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

    // ── Chromatic Aberration (16 bytes) ──
    pub ca_intensity: f32,            // 0 = disabled
    pub ca_start_offset: f32,         // radial distance where CA begins (0 = center)
    pub ca_enabled: u32,
    pub pad_ca: f32,

    // ── Film Grain (16 bytes) ──
    pub grain_intensity: f32,
    pub grain_response: f32,          // curve exponent
    pub grain_size: f32,
    pub grain_enabled: u32,

    // ── Depth of Field (32 bytes) ──
    pub dof_focal_distance: f32,
    pub dof_focal_region: f32,
    pub dof_near_transition: f32,
    pub dof_far_transition: f32,
    pub dof_scale: f32,
    pub dof_max_bokeh_size: f32,
    pub dof_enabled: u32,
    pub pad_dof: f32,

    // ── Motion Blur (16 bytes) ──
    pub motion_blur_amount: f32,
    pub motion_blur_max: f32,
    pub motion_blur_enabled: u32,
    pub pad_mb: f32,

    // ── Per-effect blend weights (8 x 4 = 32 bytes) ──
    pub blend_weight_bloom: f32,
    pub blend_weight_dof: f32,
    pub blend_weight_motion_blur: f32,
    pub blend_weight_vignette: f32,
    pub blend_weight_ca: f32,
    pub blend_weight_grain: f32,
    pub blend_weight_exposure: f32,
    pub pad_bw: f32,
}

// Total: 16 + 32 + 80 + 16 + 16 + 32 + 16 + 16 + 32 + 16 + 32 = 304 bytes
// WGSL uniform buffer rule: must be multiple of 16 → 304 / 16 = 19 slots. ✓

// ── Defaults ───────────────────────────────────────────────────────────────────

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
            pad_col_sat: 0.0,
            color_contrast: [1.0, 1.0, 1.0],
            pad_col_con: 0.0,
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

            tonemap_operator: TonemapOperator::None as u32,
            tonemap_exposure: 1.0,
            tonemap_white_point: 1.0,
            pad_tm: 0.0,

            vignette_intensity: 0.0,
            vignette_smoothness: 0.5,
            vignette_roundness: 0.5,
            _pad_vignette: 0.0,
            vignette_color: [0.0, 0.0, 0.0],
            vignette_enabled: 0,

            ca_intensity: 0.0,
            ca_start_offset: 0.0,
            ca_enabled: 0,
            pad_ca: 0.0,

            grain_intensity: 0.0,
            grain_response: 1.0,
            grain_size: 1.0,
            grain_enabled: 0,

            dof_focal_distance: 100.0,
            dof_focal_region: 50.0,
            dof_near_transition: 100.0,
            dof_far_transition: 100.0,
            dof_scale: 1.0,
            dof_max_bokeh_size: 10.0,
            dof_enabled: 0,
            pad_dof: 0.0,

            motion_blur_amount: 0.0,
            motion_blur_max: 64.0,
            motion_blur_enabled: 0,
            pad_mb: 0.0,

            blend_weight_bloom: 1.0,
            blend_weight_dof: 1.0,
            blend_weight_motion_blur: 1.0,
            blend_weight_vignette: 1.0,
            blend_weight_ca: 1.0,
            blend_weight_grain: 1.0,
            blend_weight_exposure: 1.0,
            pad_bw: 0.0,
        }
    }
}

// ── PostProcessSettings (CPU-side, full parameter set) ─────────────────────────
//
// Intended for use in Camera defaults, PostProcessVolume descriptors,
// and as the blending unit for the CPU blender.

#[derive(Clone, Debug)]
pub struct PostProcessSettings {
    // Exposure
    pub exposure_mode: ExposureMode,
    pub exposure_compensation: f32,
    pub exposure_min: f32,
    pub exposure_max: f32,
    pub exposure_speed_up: f32,     // seconds to bright-adapt
    pub exposure_speed_down: f32,   // seconds to dark-adapt

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

    // Chromatic Aberration
    pub ca_intensity: f32,
    pub ca_start_offset: f32,
    pub ca_enabled: bool,

    // Film Grain
    pub grain_intensity: f32,
    pub grain_response: f32,
    pub grain_size: f32,
    pub grain_enabled: bool,

    // Depth of Field
    pub dof_focal_distance: f32,
    pub dof_focal_region: f32,
    pub dof_near_transition: f32,
    pub dof_far_transition: f32,
    pub dof_scale: f32,
    pub dof_max_bokeh_size: f32,
    pub dof_aperture_blades: u32,
    pub dof_enabled: bool,

    // Motion Blur
    pub motion_blur_amount: f32,
    pub motion_blur_max: f32,
    pub motion_blur_enabled: bool,

    // Per-effect blend weights (for transitions)
    pub blend_weight_bloom: f32,
    pub blend_weight_dof: f32,
    pub blend_weight_motion_blur: f32,
    pub blend_weight_vignette: f32,
    pub blend_weight_ca: f32,
    pub blend_weight_grain: f32,
    pub blend_weight_exposure: f32,
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
            pad_col_sat: 0.0,
            color_contrast: self.color_contrast,
            pad_col_con: 0.0,
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

            ca_intensity: self.ca_intensity,
            ca_start_offset: self.ca_start_offset,
            ca_enabled: self.ca_enabled as u32,
            pad_ca: 0.0,

            grain_intensity: self.grain_intensity,
            grain_response: self.grain_response,
            grain_size: self.grain_size,
            grain_enabled: self.grain_enabled as u32,

            dof_focal_distance: self.dof_focal_distance,
            dof_focal_region: self.dof_focal_region,
            dof_near_transition: self.dof_near_transition,
            dof_far_transition: self.dof_far_transition,
            dof_scale: self.dof_scale,
            dof_max_bokeh_size: self.dof_max_bokeh_size,
            dof_enabled: self.dof_enabled as u32,
            pad_dof: 0.0,

            motion_blur_amount: self.motion_blur_amount,
            motion_blur_max: self.motion_blur_max,
            motion_blur_enabled: self.motion_blur_enabled as u32,
            pad_mb: 0.0,

            blend_weight_bloom: self.blend_weight_bloom,
            blend_weight_dof: self.blend_weight_dof,
            blend_weight_motion_blur: self.blend_weight_motion_blur,
            blend_weight_vignette: self.blend_weight_vignette,
            blend_weight_ca: self.blend_weight_ca,
            blend_weight_grain: self.blend_weight_grain,
            blend_weight_exposure: self.blend_weight_exposure,
            pad_bw: 0.0,
        }
    }
}

impl Default for PostProcessSettings {
    fn default() -> Self {
        Self {
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

            tonemap_operator: TonemapOperator::None,
            tonemap_exposure: 1.0,
            tonemap_white_point: 1.0,

            vignette_intensity: 0.0,
            vignette_smoothness: 0.5,
            vignette_roundness: 0.5,
            vignette_color: [0.0, 0.0, 0.0],
            vignette_enabled: false,

            ca_intensity: 0.0,
            ca_start_offset: 0.0,
            ca_enabled: false,

            grain_intensity: 0.0,
            grain_response: 1.0,
            grain_size: 1.0,
            grain_enabled: false,

            dof_focal_distance: 100.0,
            dof_focal_region: 50.0,
            dof_near_transition: 100.0,
            dof_far_transition: 100.0,
            dof_scale: 1.0,
            dof_max_bokeh_size: 10.0,
            dof_aperture_blades: 5,
            dof_enabled: false,

            motion_blur_amount: 0.0,
            motion_blur_max: 64.0,
            motion_blur_enabled: false,

            blend_weight_bloom: 1.0,
            blend_weight_dof: 1.0,
            blend_weight_motion_blur: 1.0,
            blend_weight_vignette: 1.0,
            blend_weight_ca: 1.0,
            blend_weight_grain: 1.0,
            blend_weight_exposure: 1.0,
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
    pub blend_weight: f32,         // 0-1, global volume opacity
    pub unbound: u32,              // 0 = bounded, 1 = applies everywhere
    pub _pad: [f32; 2],
    pub settings: GpuPostProcessUniforms,
}

// ── PostProcessVolume descriptor (CPU-side) ────────────────────────────────────

#[derive(Clone, Debug)]
pub struct PostProcessVolumeDescriptor {
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub priority: f32,
    pub blend_radius: f32,
    pub blend_weight: f32,
    pub unbound: bool,               // infinite volume (camera always inside)
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
            settings: PostProcessSettings::default(),
        }
    }
}

impl PostProcessVolumeDescriptor {
    pub fn to_gpu(&self) -> GpuPostProcessVolume {
        GpuPostProcessVolume {
            bounds_min: [self.bounds_min[0], self.bounds_min[1], self.bounds_min[2], 0.0],
            bounds_max: [self.bounds_max[0], self.bounds_max[1], self.bounds_max[2], 0.0],
            priority: self.priority,
            blend_radius: self.blend_radius,
            blend_weight: self.blend_weight,
            unbound: self.unbound as u32,
            _pad: [0.0; 2],
            settings: self.settings.to_gpu(),
        }
    }
}

// ── PostProcessBlender ─────────────────────────────────────────────────────────
//
// CPU-side blender that evaluates active volumes and produces final uniforms.

pub struct PostProcessBlender;

impl PostProcessBlender {
    /// Blend camera settings with all active volumes at `camera_pos`.
    ///
    /// Returns the final GPU uniforms ready for upload.
    pub fn blend(
        camera_pos: [f32; 3],
        volumes: &[GpuPostProcessVolume],
        camera_settings: &PostProcessSettings,
    ) -> GpuPostProcessUniforms {
        if volumes.is_empty() {
            return camera_settings.to_gpu();
        }

        // Collect weighted contributions
        let mut active: Vec<(f32, &GpuPostProcessVolume)> = Vec::with_capacity(volumes.len());

        for v in volumes {
            if v.blend_weight <= 0.0 {
                continue;
            }

            let weight = if v.priority >= 1e10 {
                // Unbound / infinite priority volume
                v.blend_weight
            } else {
                let pos = [
                    camera_pos[0].clamp(v.bounds_min[0], v.bounds_max[0]),
                    camera_pos[1].clamp(v.bounds_min[1], v.bounds_max[1]),
                    camera_pos[2].clamp(v.bounds_min[2], v.bounds_max[2]),
                ];
                let dx = camera_pos[0] - pos[0];
                let dy = camera_pos[1] - pos[1];
                let dz = camera_pos[2] - pos[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                let blend_dist = v.blend_radius.max(0.001);
                let inside_weight = 1.0 - (dist / blend_dist).clamp(0.0, 1.0);
                if inside_weight <= 0.0 {
                    continue;
                }
                inside_weight * v.blend_weight
            };

            active.push((weight, v));
        }

        if active.is_empty() {
            return camera_settings.to_gpu();
        }

        // Sort by priority descending
        active.sort_by(|a, b| b.1.priority.partial_cmp(&a.1.priority).unwrap());

        // Blend: higher-priority volumes override lower-priority ones.
        // We accumulate with a cumulative weight that gives priority to
        // higher-priority volumes for overlapping regions.
        let mut result = camera_settings.clone();
        let mut total_weight = 1.0_f32; // camera settings have implicit weight 1.0

        for (weight, volume) in &active {
            let vol_settings = unpack_settings(&volume.settings);
            let w = *weight;
            // Blend: lerp from current result toward volume settings
            let t = (w / (total_weight + w)).clamp(0.0, 1.0);
            result = Self::lerp_settings(&result, &vol_settings, t);
            total_weight += w;
        }

        result.to_gpu()
    }

    /// Linearly interpolate between two PostProcessSettings.
    fn lerp_settings(a: &PostProcessSettings, b: &PostProcessSettings, t: f32) -> PostProcessSettings {
        PostProcessSettings {
            exposure_mode: if t > 0.5 { b.exposure_mode } else { a.exposure_mode },
            exposure_compensation: lerp(a.exposure_compensation, b.exposure_compensation, t),
            exposure_min: lerp(a.exposure_min, b.exposure_min, t),
            exposure_max: lerp(a.exposure_max, b.exposure_max, t),
            exposure_speed_up: lerp(a.exposure_speed_up, b.exposure_speed_up, t),
            exposure_speed_down: lerp(a.exposure_speed_down, b.exposure_speed_down, t),

            bloom_intensity: lerp(a.bloom_intensity, b.bloom_intensity, t),
            bloom_threshold: lerp(a.bloom_threshold, b.bloom_threshold, t),
            bloom_knee: lerp(a.bloom_knee, b.bloom_knee, t),
            bloom_radius: lerp(a.bloom_radius, b.bloom_radius, t),
            bloom_tint: lerp3(a.bloom_tint, b.bloom_tint, t),
            bloom_enabled: if t > 0.5 { b.bloom_enabled } else { a.bloom_enabled },

            color_saturation: lerp3(a.color_saturation, b.color_saturation, t),
            color_contrast: lerp3(a.color_contrast, b.color_contrast, t),
            color_gamma: lerp3(a.color_gamma, b.color_gamma, t),
            color_gain: lerp3(a.color_gain, b.color_gain, t),
            color_offset: lerp3(a.color_offset, b.color_offset, t),

            white_temp: lerp(a.white_temp, b.white_temp, t),
            white_tint: lerp(a.white_tint, b.white_tint, t),
            white_balance_enabled: if t > 0.5 { b.white_balance_enabled } else { a.white_balance_enabled },

            tonemap_operator: if t > 0.5 { b.tonemap_operator } else { a.tonemap_operator },
            tonemap_exposure: lerp(a.tonemap_exposure, b.tonemap_exposure, t),
            tonemap_white_point: lerp(a.tonemap_white_point, b.tonemap_white_point, t),

            vignette_intensity: lerp(a.vignette_intensity, b.vignette_intensity, t),
            vignette_smoothness: lerp(a.vignette_smoothness, b.vignette_smoothness, t),
            vignette_roundness: lerp(a.vignette_roundness, b.vignette_roundness, t),
            vignette_color: lerp3(a.vignette_color, b.vignette_color, t),
            vignette_enabled: if t > 0.5 { b.vignette_enabled } else { a.vignette_enabled },

            ca_intensity: lerp(a.ca_intensity, b.ca_intensity, t),
            ca_start_offset: lerp(a.ca_start_offset, b.ca_start_offset, t),
            ca_enabled: if t > 0.5 { b.ca_enabled } else { a.ca_enabled },

            grain_intensity: lerp(a.grain_intensity, b.grain_intensity, t),
            grain_response: lerp(a.grain_response, b.grain_response, t),
            grain_size: lerp(a.grain_size, b.grain_size, t),
            grain_enabled: if t > 0.5 { b.grain_enabled } else { a.grain_enabled },

            dof_focal_distance: lerp(a.dof_focal_distance, b.dof_focal_distance, t),
            dof_focal_region: lerp(a.dof_focal_region, b.dof_focal_region, t),
            dof_near_transition: lerp(a.dof_near_transition, b.dof_near_transition, t),
            dof_far_transition: lerp(a.dof_far_transition, b.dof_far_transition, t),
            dof_scale: lerp(a.dof_scale, b.dof_scale, t),
            dof_max_bokeh_size: lerp(a.dof_max_bokeh_size, b.dof_max_bokeh_size, t),
            dof_aperture_blades: if t > 0.5 { b.dof_aperture_blades } else { a.dof_aperture_blades },
            dof_enabled: if t > 0.5 { b.dof_enabled } else { a.dof_enabled },

            motion_blur_amount: lerp(a.motion_blur_amount, b.motion_blur_amount, t),
            motion_blur_max: lerp(a.motion_blur_max, b.motion_blur_max, t),
            motion_blur_enabled: if t > 0.5 { b.motion_blur_enabled } else { a.motion_blur_enabled },

            blend_weight_bloom: lerp(a.blend_weight_bloom, b.blend_weight_bloom, t),
            blend_weight_dof: lerp(a.blend_weight_dof, b.blend_weight_dof, t),
            blend_weight_motion_blur: lerp(a.blend_weight_motion_blur, b.blend_weight_motion_blur, t),
            blend_weight_vignette: lerp(a.blend_weight_vignette, b.blend_weight_vignette, t),
            blend_weight_ca: lerp(a.blend_weight_ca, b.blend_weight_ca, t),
            blend_weight_grain: lerp(a.blend_weight_grain, b.blend_weight_grain, t),
            blend_weight_exposure: lerp(a.blend_weight_exposure, b.blend_weight_exposure, t),
        }
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 { a + (b - a) * t }
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)]
}

fn unpack_settings(gpu: &GpuPostProcessUniforms) -> PostProcessSettings {
    PostProcessSettings {
        exposure_mode: if gpu.exposure_mode == 0 { ExposureMode::Manual } else { ExposureMode::Auto },
        exposure_compensation: gpu.exposure_compensation,
        exposure_min: gpu.exposure_min,
        exposure_max: gpu.exposure_max,
        exposure_speed_up: 0.5,
        exposure_speed_down: 1.0,

        bloom_intensity: gpu.bloom_intensity,
        bloom_threshold: gpu.bloom_threshold,
        bloom_knee: gpu.bloom_knee,
        bloom_radius: gpu.bloom_radius,
        bloom_tint: gpu.bloom_tint,
        bloom_enabled: gpu.bloom_enabled != 0,

        color_saturation: gpu.color_saturation,
        color_contrast: gpu.color_contrast,
        color_gamma: gpu.color_gamma,
        color_gain: gpu.color_gain,
        color_offset: gpu.color_offset,

        white_temp: gpu.white_temp,
        white_tint: gpu.white_tint,
        white_balance_enabled: gpu.white_balance_enabled != 0,

        tonemap_operator: match gpu.tonemap_operator {
            0 => TonemapOperator::Aces,
            1 => TonemapOperator::Filmic,
            2 => TonemapOperator::Reinhard,
            3 => TonemapOperator::Uncharted2,
            4 => TonemapOperator::Lottes,
            _ => TonemapOperator::None,
        },
        tonemap_exposure: gpu.tonemap_exposure,
        tonemap_white_point: gpu.tonemap_white_point,

        vignette_intensity: gpu.vignette_intensity,
        vignette_smoothness: gpu.vignette_smoothness,
        vignette_roundness: gpu.vignette_roundness,
        vignette_color: gpu.vignette_color,
        vignette_enabled: gpu.vignette_enabled != 0,

        ca_intensity: gpu.ca_intensity,
        ca_start_offset: gpu.ca_start_offset,
        ca_enabled: gpu.ca_enabled != 0,

        grain_intensity: gpu.grain_intensity,
        grain_response: gpu.grain_response,
        grain_size: gpu.grain_size,
        grain_enabled: gpu.grain_enabled != 0,

        dof_focal_distance: gpu.dof_focal_distance,
        dof_focal_region: gpu.dof_focal_region,
        dof_near_transition: gpu.dof_near_transition,
        dof_far_transition: gpu.dof_far_transition,
        dof_scale: gpu.dof_scale,
        dof_max_bokeh_size: gpu.dof_max_bokeh_size,
        dof_aperture_blades: 5,
        dof_enabled: gpu.dof_enabled != 0,

        motion_blur_amount: gpu.motion_blur_amount,
        motion_blur_max: gpu.motion_blur_max,
        motion_blur_enabled: gpu.motion_blur_enabled != 0,

        blend_weight_bloom: gpu.blend_weight_bloom,
        blend_weight_dof: gpu.blend_weight_dof,
        blend_weight_motion_blur: gpu.blend_weight_motion_blur,
        blend_weight_vignette: gpu.blend_weight_vignette,
        blend_weight_ca: gpu.blend_weight_ca,
        blend_weight_grain: gpu.blend_weight_grain,
        blend_weight_exposure: gpu.blend_weight_exposure,
    }
}
