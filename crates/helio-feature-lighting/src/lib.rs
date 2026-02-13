use helio_features::{Feature, FeatureContext, ShaderInjection, ShaderInjectionPoint};

/// Basic lighting feature with diffuse and ambient lighting.
///
/// Provides simple directional lighting calculations in the fragment shader.
/// The light direction is hardcoded for simplicity.
///
/// # Material Data Integration
/// This feature can query material properties from the materials feature
/// for PBR calculations. See `prepare_frame()` for an example of accessing
/// exported material data.
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

    fn prepare_frame(&mut self, _context: &FeatureContext) {
        // Example: In a full implementation, you would query material data here:
        //
        // if let Some(data) = registry.get_exported_data("basic_materials", "properties") {
        //     if let Some(props) = data.downcast_arc::<MaterialProperties>() {
        //         // Use props.metallic, props.roughness for PBR calculations
        //         // Update uniforms or shader parameters based on material properties
        //         log::debug!("Material roughness: {}", props.roughness);
        //         log::debug!("Material metallic: {}", props.metallic);
        //     }
        // }
        //
        // Note: To access the registry, you would need to pass it to prepare_frame()
        // or store a reference to it. This is left as an exercise for integration.
    }
    
    fn cleanup(&mut self, _context: &FeatureContext) {
        // No GPU resources to clean up
    }
}
