use serde::{Deserialize, Serialize};
use nebula_core::traits::BakeInput;

/// Configuration for reflection / irradiance probe baking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeConfig {
    /// Cubemap face resolution (must be power-of-two).
    pub face_resolution: u32,
    /// Number of specular mip levels to generate (IBL pre-filter chain).
    pub specular_mip_levels: u32,
    /// Spherical-harmonic order used for diffuse irradiance (1–3).
    ///   Order 1 → 4 coefficients, Order 2 → 9, Order 3 → 16.
    pub sh_order: u32,
    /// Number of samples per probe face (Monte Carlo).
    pub samples_per_face: u32,
    /// Exposure multiplier applied before encoding.
    pub exposure: f32,
    /// Whether to encode the cubemap as RGBE (6 bytes/pixel) instead of RGBA32F.
    pub use_rgbe: bool,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            face_resolution: 256,
            specular_mip_levels: 8,
            sh_order: 3,
            samples_per_face: 1024,
            exposure: 1.0,
            use_rgbe: false,
        }
    }
}

impl ProbeConfig {
    /// Low-quality fast preview preset.
    pub fn fast() -> Self {
        Self { face_resolution: 64, specular_mip_levels: 4, samples_per_face: 128, ..Default::default() }
    }

    /// Production quality preset.
    pub fn ultra() -> Self {
        Self { face_resolution: 512, specular_mip_levels: 10, samples_per_face: 8192, ..Default::default() }
    }
}

impl BakeInput for ProbeConfig {}
