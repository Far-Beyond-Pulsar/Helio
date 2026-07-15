use serde::{Deserialize, Serialize};
use nebula_core::traits::BakeInput;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AoConfig {
    /// Lightmap-space resolution (shared with the lightmap baker if used together).
    pub resolution:      u32,
    /// Number of hemisphere rays per texel.
    pub ray_count:       u32,
    /// Maximum occlusion check distance.
    pub max_distance:    f32,
    /// Increase to push rays away from biased self-intersection (world units).
    pub bias:            f32,
    /// Apply a spatial denoising filter after baking.
    pub denoise:         bool,
}

impl Default for AoConfig {
    fn default() -> Self {
        Self {
            resolution:   1024,
            ray_count:    128,
            max_distance: 10.0,
            bias:         0.001,
            denoise:      true,
        }
    }
}

impl AoConfig {
    pub fn fast() -> Self {
        Self { resolution: 512, ray_count: 16, denoise: false, ..Default::default() }
    }
    pub fn ultra() -> Self {
        Self { resolution: 4096, ray_count: 512, ..Default::default() }
    }
}

impl BakeInput for AoConfig {}
