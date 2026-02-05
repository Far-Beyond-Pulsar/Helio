use helio_features::{Feature, FeatureContext, ShaderInjection, ShaderInjectionPoint};

pub struct BasicLighting {
    enabled: bool,
    lighting_functions: String,
}

impl BasicLighting {
    pub fn new() -> Self {
        let lighting_functions = include_str!("../shaders/lighting_functions.wgsl").to_string();

        Self {
            enabled: true,
            lighting_functions,
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
        // Basic lighting doesn't need GPU resources
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn shader_injections(&self) -> Vec<ShaderInjection> {
        vec![
            // Inject lighting functions before fragment shader
            ShaderInjection {
                point: ShaderInjectionPoint::FragmentPreamble,
                code: self.lighting_functions.clone(),
                priority: 0,
            },
            // Inject lighting calculation in fragment shader
            ShaderInjection {
                point: ShaderInjectionPoint::FragmentColorCalculation,
                code: "    final_color = apply_basic_lighting(normalize(input.world_normal), final_color);".to_string(),
                priority: 0,
            },
        ]
    }
}
