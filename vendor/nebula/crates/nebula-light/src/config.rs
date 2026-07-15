use serde::{Deserialize, Serialize};
use nebula_core::traits::BakeInput;

/// Controls how the lightmap baker runs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LightmapConfig {
    // ── Resolution ──────────────────────────────────────────────────────────
    /// Lightmap atlas texture side (power of 2; clamped to [64, 8192]).
    pub resolution: u32,

    // ── Quality ─────────────────────────────────────────────────────────────
    /// Path-tracing samples per texel for GI (set 0 to disable GI).
    pub samples_per_texel: u32,
    /// Maximum number of indirect-lighting bounces (0 = direct only).
    pub bounce_count: u32,
    /// Maximum ray distance in world units.
    pub max_ray_distance: f32,

    // ── Denoising ────────────────────────────────────────────────────────────
    /// Apply a simple 3×3 Gaussian spatial filter after baking.
    pub denoise: bool,

    // ── HDR output ───────────────────────────────────────────────────────────
    /// Store full 32-bit float (RGBA32F).  `false` = RGBA16F (half float).
    pub hdr_output: bool,

    // ── Area light approximation ─────────────────────────────────────────────
    /// Number of sample points used to approximate area lights.
    pub area_light_samples: u32,

    // ── Debug ────────────────────────────────────────────────────────────────
    /// Write an intermediate texel-normal pass to a separate texture (slow).
    pub debug_normals: bool,
}

impl Default for LightmapConfig {
    fn default() -> Self {
        Self {
            resolution:          1024,
            samples_per_texel:   64,
            bounce_count:        2,
            max_ray_distance:    1000.0,
            denoise:             true,
            hdr_output:          true,
            area_light_samples:  16,
            debug_normals:       false,
        }
    }
}

impl LightmapConfig {
    pub fn fast() -> Self {
        Self { resolution: 512, samples_per_texel: 8, bounce_count: 1, denoise: false, ..Default::default() }
    }

    pub fn ultra() -> Self {
        Self { resolution: 4096, samples_per_texel: 512, bounce_count: 4, ..Default::default() }
    }
}

impl BakeInput for LightmapConfig {}
