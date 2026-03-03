//! Bloom post-processing feature

use super::{FeatureContext, PrepareContext};
use crate::features::{Feature, ShaderDefine};
use crate::Result;
use std::collections::HashMap;

/// Bloom post-processing feature
///
/// Adds a glow effect to pixels that exceed a luminance threshold.
/// Uses WGSL `override` constants so the compiler can optimize away
/// the bloom code path when this feature is disabled.
pub struct BloomFeature {
    enabled: bool,
    /// Luminance threshold above which bloom is applied (default: 1.0)
    pub threshold: f32,
    /// Bloom glow strength (default: 0.3, range: 0.0–2.0)
    pub intensity: f32,
}

impl BloomFeature {
    pub fn new() -> Self {
        Self {
            enabled: true,
            threshold: 1.0,
            intensity: 0.3,
        }
    }

    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity.clamp(0.0, 2.0);
        self
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold.max(0.0);
        self
    }
}

impl Default for BloomFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl Feature for BloomFeature {
    fn name(&self) -> &str {
        "bloom"
    }

    fn register(&mut self, _ctx: &mut FeatureContext) -> Result<()> {
        log::info!(
            "Bloom feature registered (intensity={:.2}, threshold={:.2})",
            self.intensity,
            self.threshold
        );
        Ok(())
    }

    fn prepare(&mut self, _ctx: &PrepareContext) -> Result<()> {
        Ok(())
    }

    fn shader_defines(&self) -> HashMap<String, ShaderDefine> {
        let mut defines = HashMap::new();
        defines.insert("ENABLE_BLOOM".into(), ShaderDefine::Bool(self.enabled));
        defines.insert("BLOOM_INTENSITY".into(), ShaderDefine::F32(if self.enabled { self.intensity } else { 0.3 }));
        defines.insert("BLOOM_THRESHOLD".into(), ShaderDefine::F32(if self.enabled { self.threshold } else { 1.0 }));
        defines
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}
