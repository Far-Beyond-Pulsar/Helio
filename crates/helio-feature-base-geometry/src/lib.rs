use helio_features::{Feature, FeatureContext, ShaderInjection};

/// Base geometry feature providing fundamental geometry rendering.
///
/// This feature provides the base shader template that other features
/// inject into. It handles vertex transformation, normal calculation,
/// and basic output setup.
pub struct BaseGeometry {
    enabled: bool,
    shader_template: &'static str,
}

impl BaseGeometry {
    /// Create a new base geometry feature.
    pub fn new() -> Self {
        Self {
            enabled: true,
            shader_template: include_str!("../shaders/base_geometry.wgsl"),
        }
    }

    /// Get the shader template for use with the feature renderer.
    ///
    /// This template contains injection markers where other features
    /// can insert their shader code.
    pub fn shader_template(&self) -> &str {
        self.shader_template
    }
}

impl Default for BaseGeometry {
    fn default() -> Self {
        Self::new()
    }
}

impl Feature for BaseGeometry {
    fn name(&self) -> &str {
        "base_geometry"
    }

    fn init(&mut self, _context: &FeatureContext) {
        log::debug!("Base geometry feature initialized");
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn shader_injections(&self) -> Vec<ShaderInjection> {
        // Base geometry provides the template, not injections
        Vec::new()
    }
    
    fn cleanup(&mut self, _context: &FeatureContext) {
        // No GPU resources to clean up
    }
}
