use blade_graphics as gpu;
use std::sync::Arc;

/// Context shared between all features
pub struct FeatureContext {
    pub gpu_context: Arc<gpu::Context>,
    pub surface_format: gpu::TextureFormat,
    pub width: u32,
    pub height: u32,
}

/// Feature that injects shader code into the main rendering shader
/// Examples: forward lighting, fog, color grading
pub trait ShaderFeature {
    /// Get the WGSL code to inject into the shader
    fn shader_code(&self) -> &str;
    
    /// Update per-frame data (e.g., light positions, params)
    fn update(&mut self, _delta_time: f32) {}
    
    /// Feature name for debugging
    fn name(&self) -> &str;
}

/// Feature that runs as a separate render or compute pass
/// Examples: ray-traced GI, shadow mapping, SSAO, post-processing
pub trait RenderPassFeature {
    /// Initialize the feature (create pipelines, buffers, textures)
    fn init(context: &FeatureContext) -> Self where Self: Sized;
    
    /// Prepare resources before rendering (e.g., build acceleration structures)
    fn prepare(&mut self, context: &FeatureContext);
    
    /// Execute the render pass
    fn execute(&self, encoder: &mut gpu::CommandEncoder, context: &FeatureContext);
    
    /// Get the output texture (if any) for compositing with other passes
    fn output_texture(&self) -> Option<&gpu::Texture> {
        None
    }
    
    /// Feature name for debugging
    fn name(&self) -> &str;
    
    /// Whether this feature is currently enabled
    fn is_enabled(&self) -> bool {
        true
    }
    
    /// Toggle the feature on/off
    fn set_enabled(&mut self, _enabled: bool) {}
}

/// Registry for managing all features
pub struct FeatureRegistry {
    shader_features: Vec<Box<dyn ShaderFeature>>,
    pass_features: Vec<Box<dyn RenderPassFeature>>,
}

impl FeatureRegistry {
    pub fn new() -> Self {
        Self {
            shader_features: Vec::new(),
            pass_features: Vec::new(),
        }
    }
    
    pub fn register_shader_feature(&mut self, feature: Box<dyn ShaderFeature>) {
        self.shader_features.push(feature);
    }
    
    pub fn register_pass_feature(&mut self, feature: Box<dyn RenderPassFeature>) {
        self.pass_features.push(feature);
    }
    
    /// Get all shader code from registered shader features
    pub fn collect_shader_code(&self) -> String {
        self.shader_features
            .iter()
            .map(|f| f.shader_code())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
    
    /// Execute all enabled render pass features
    pub fn execute_passes(&self, encoder: &mut gpu::CommandEncoder, context: &FeatureContext) {
        for feature in &self.pass_features {
            if feature.is_enabled() {
                feature.execute(encoder, context);
            }
        }
    }
    
    /// Prepare all features (called once per frame)
    pub fn prepare_all(&mut self, context: &FeatureContext, delta_time: f32) {
        for feature in &mut self.shader_features {
            feature.update(delta_time);
        }
        
        for feature in &mut self.pass_features {
            if feature.is_enabled() {
                feature.prepare(context);
            }
        }
    }
}

impl Default for FeatureRegistry {
    fn default() -> Self {
        Self::new()
    }
}
