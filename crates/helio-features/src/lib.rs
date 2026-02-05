use blade_graphics as gpu;
use std::sync::Arc;

/// Defines where shader code should be injected in the rendering pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderInjectionPoint {
    /// Injected at the start of the vertex shader (before main)
    VertexPreamble,
    /// Injected at the start of vertex main function
    VertexMain,
    /// Injected at the end of vertex main function (after position calculation)
    VertexPostProcess,
    /// Injected at the start of the fragment shader (before main)
    FragmentPreamble,
    /// Injected at the start of fragment main function
    FragmentMain,
    /// Injected for final color calculation in fragment shader
    FragmentColorCalculation,
    /// Injected at the end of fragment main (for post-processing)
    FragmentPostProcess,
}

/// Shader code that a feature contributes at a specific injection point
#[derive(Clone)]
pub struct ShaderInjection {
    pub point: ShaderInjectionPoint,
    pub code: String,
    pub priority: i32, // Higher priority = injected later
}

/// Context provided to features during rendering
pub struct FeatureContext {
    pub gpu: Arc<gpu::Context>,
    pub surface_size: (u32, u32),
    pub frame_index: u64,
    pub delta_time: f32,
}

/// Core trait that all rendering features must implement
pub trait Feature: Send + Sync {
    /// Unique name for this feature
    fn name(&self) -> &str;

    /// Initialize the feature with GPU context
    fn init(&mut self, context: &FeatureContext);

    /// Check if this feature is currently enabled
    fn is_enabled(&self) -> bool {
        true
    }

    /// Enable or disable this feature at runtime
    fn set_enabled(&mut self, enabled: bool);

    /// Get shader code injections provided by this feature
    fn shader_injections(&self) -> Vec<ShaderInjection> {
        Vec::new()
    }

    /// Optional: Execute a compute or additional render pass before main rendering
    fn pre_render_pass(&mut self, encoder: &mut gpu::CommandEncoder, context: &FeatureContext) {
        let _ = (encoder, context);
    }

    /// Optional: Execute a compute or additional render pass after main rendering
    fn post_render_pass(&mut self, encoder: &mut gpu::CommandEncoder, context: &FeatureContext) {
        let _ = (encoder, context);
    }

    /// Optional: Prepare per-frame data before rendering
    fn prepare_frame(&mut self, context: &FeatureContext) {
        let _ = context;
    }

    /// Optional: Bind feature-specific resources during rendering
    /// Pass can be either RenderCommandEncoder or PipelineEncoder
    fn bind_resources(&self, _context: &FeatureContext) {
        // Features can override this to bind resources
        // Binding will be done through the pass parameter in bind_all_resources
    }
}

/// Registry that manages all rendering features and composing them into shaders
pub struct FeatureRegistry {
    features: Vec<Box<dyn Feature>>,
}

impl FeatureRegistry {
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
        }
    }

    /// Add a feature to the registry
    pub fn register<F: Feature + 'static>(&mut self, feature: F) {
        self.features.push(Box::new(feature));
    }

    /// Initialize all features
    pub fn init_all(&mut self, context: &FeatureContext) {
        for feature in &mut self.features {
            feature.init(context);
        }
    }

    /// Compose shader code from all enabled features
    pub fn compose_shader(&self, base_shader: &str) -> String {
        use std::collections::HashMap;

        // Collect all injections from enabled features
        let mut injections: HashMap<ShaderInjectionPoint, Vec<ShaderInjection>> = HashMap::new();

        for feature in &self.features {
            if feature.is_enabled() {
                for injection in feature.shader_injections() {
                    injections
                        .entry(injection.point)
                        .or_insert_with(Vec::new)
                        .push(injection);
                }
            }
        }

        // Sort injections by priority
        for injections_at_point in injections.values_mut() {
            injections_at_point.sort_by_key(|inj| inj.priority);
        }

        // Replace injection points in base shader
        let mut result = base_shader.to_string();

        for (point, injections_at_point) in injections {
            let marker = format!("// INJECT_{:?}", point).to_uppercase();
            let code = injections_at_point
                .iter()
                .map(|inj| inj.code.as_str())
                .collect::<Vec<_>>()
                .join("\n");

            result = result.replace(&marker, &code);
        }

        result
    }

    /// Prepare all features for the current frame
    pub fn prepare_frame(&mut self, context: &FeatureContext) {
        for feature in &mut self.features {
            if feature.is_enabled() {
                feature.prepare_frame(context);
            }
        }
    }

    /// Execute pre-render passes for all enabled features
    pub fn execute_pre_passes(&mut self, encoder: &mut gpu::CommandEncoder, context: &FeatureContext) {
        for feature in &mut self.features {
            if feature.is_enabled() {
                feature.pre_render_pass(encoder, context);
            }
        }
    }

    /// Execute post-render passes for all enabled features
    pub fn execute_post_passes(&mut self, encoder: &mut gpu::CommandEncoder, context: &FeatureContext) {
        for feature in &mut self.features {
            if feature.is_enabled() {
                feature.post_render_pass(encoder, context);
            }
        }
    }

    /// Prepare all features for binding
    pub fn prepare_binding(&self, context: &FeatureContext) {
        for feature in &self.features {
            if feature.is_enabled() {
                feature.bind_resources(context);
            }
        }
    }

    /// Get all features
    pub fn features(&self) -> &[Box<dyn Feature>] {
        &self.features
    }

    /// Get mutable reference to all features
    pub fn features_mut(&mut self) -> &mut [Box<dyn Feature>] {
        &mut self.features
    }

    /// Get a feature by name
    pub fn get_feature(&self, name: &str) -> Option<&dyn Feature> {
        self.features
            .iter()
            .find(|f| f.name() == name)
            .map(|f| f.as_ref())
    }

    /// Get a mutable reference to a feature by name
    pub fn get_feature_mut(&mut self, name: &str) -> Option<&mut Box<dyn Feature>> {
        self.features
            .iter_mut()
            .find(|f| f.name() == name)
    }

    /// Toggle a feature by name
    pub fn toggle_feature(&mut self, name: &str) -> bool {
        if let Some(feature) = self.get_feature_mut(name) {
            let new_state = !feature.is_enabled();
            feature.set_enabled(new_state);
            true
        } else {
            false
        }
    }
}

impl Default for FeatureRegistry {
    fn default() -> Self {
        Self::new()
    }
}
