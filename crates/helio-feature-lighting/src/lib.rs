use helio_features::{Feature, FeatureContext, ShaderInjection, ShaderInjectionPoint};

/// Basic lighting feature with diffuse and ambient lighting.
///
/// Provides simple directional lighting calculations in the fragment shader.
/// The light direction is hardcoded for simplicity.
pub struct BasicLighting {
    enabled: bool,
}

impl BasicLighting {
    /// Create a new basic lighting feature.
    pub fn new() -> Self {
        Self {
            enabled: true,
        }
    }
}

impl Default for BasicLighting {
    fn default() -> Self {
        Self::new()
    }
}

impl Feature for BasicLighting {
    fn name(&self) -> &str {
        "basic_lighting"
    }

    fn init(&mut self, _context: &FeatureContext) {
        log::debug!("Basic lighting feature initialized");
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn shader_injections(&self) -> Vec<ShaderInjection> {
        vec![
            // Inject lighting functions
            ShaderInjection::new(
                ShaderInjectionPoint::FragmentPreamble,
                include_str!("../shaders/lighting_functions.wgsl"),
            ),
            // Apply lighting in color calculation
            ShaderInjection::new(
                ShaderInjectionPoint::FragmentColorCalculation,
                "    final_color = apply_basic_lighting(normalize(input.world_normal), final_color);",
            ),
        ]
    }
    
    fn cleanup(&mut self, _context: &FeatureContext) {
        // No GPU resources to clean up
    }
}
