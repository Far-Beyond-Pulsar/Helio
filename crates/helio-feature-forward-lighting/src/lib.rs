use helio_features::ShaderFeature;

pub struct ForwardLighting {
    shader_code: String,
}

impl ForwardLighting {
    pub fn new() -> Self {
        let shader_code = std::fs::read_to_string("shaders/features/forward_lighting/blinn_phong.wgsl")
            .expect("Failed to load forward lighting shader");
        
        Self { shader_code }
    }
}

impl ShaderFeature for ForwardLighting {
    fn shader_code(&self) -> &str {
        &self.shader_code
    }
    
    fn name(&self) -> &str {
        "forward_lighting"
    }
}

impl Default for ForwardLighting {
    fn default() -> Self {
        Self::new()
    }
}
