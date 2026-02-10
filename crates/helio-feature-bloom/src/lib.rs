use helio_features::{Feature, ShaderInjection, ShaderInjectionPoint};

/// Efficient bloom effect for overlit objects.
/// Adds glow to pixels with luminance > 1.0 through shader injection.
pub struct Bloom {
    enabled: bool,
    intensity: f32,
}

impl Bloom {
    /// Create a new bloom effect with default intensity (0.3).
    pub fn new() -> Self {
        Self {
            enabled: true,
            intensity: 0.3,
        }
    }

    /// Set bloom intensity (default: 0.3, range: 0.0-1.0).
    /// Higher values create stronger glow on overbright pixels.
    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity.clamp(0.0, 1.0);
        self
    }

    /// Set the bloom intensity at runtime.
    pub fn set_intensity(&mut self, intensity: f32) {
        self.intensity = intensity.clamp(0.0, 1.0);
    }

    /// Get the current bloom intensity.
    pub fn intensity(&self) -> f32 {
        self.intensity
    }
}

impl Default for Bloom {
    fn default() -> Self {
        Self::new()
    }
}

impl Feature for Bloom {
    fn name(&self) -> &str {
        "bloom"
    }

    fn init(&mut self, _context: &helio_features::FeatureContext) {
        // No initialization needed - shader-only feature
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn shader_injections(&self) -> Vec<ShaderInjection> {
        vec![
            // Add bloom function to fragment preamble
            ShaderInjection::with_priority(
                ShaderInjectionPoint::FragmentPreamble,
                include_str!("../shaders/bloom.wgsl"),
                15,
            ),
            // Apply bloom after all other lighting/color calculations
            ShaderInjection::with_priority(
                ShaderInjectionPoint::FragmentColorCalculation,
                "    final_color = apply_bloom(final_color);",
                20,
            ),
        ]
    }
}
