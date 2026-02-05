use helio_features::{Feature, FeatureContext, ShaderInjection};

pub struct BaseGeometry {
    enabled: bool,
    shader_template: String,
}

impl BaseGeometry {
    pub fn new() -> Self {
        let shader_template = include_str!("../shaders/base_geometry.wgsl").to_string();

        Self {
            enabled: true,
            shader_template,
        }
    }

    pub fn shader_template(&self) -> &str {
        &self.shader_template
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
        // Base geometry doesn't need initialization
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn shader_injections(&self) -> Vec<ShaderInjection> {
        // Base geometry provides the template, not injections
        // Other features will inject into this template
        Vec::new()
    }
}
