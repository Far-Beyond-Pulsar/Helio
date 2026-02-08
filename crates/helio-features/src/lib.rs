use blade_graphics as gpu;
use std::borrow::Cow;
use std::sync::Arc;

/// Defines where shader code should be injected in the rendering pipeline.
///
/// Injection points are processed in order during shader composition. Each feature
/// can contribute code at multiple injection points, and multiple features can
/// contribute to the same injection point (ordered by priority).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderInjectionPoint {
    /// Injected before the vertex shader's main function.
    ///
    /// Use for: Global variable declarations, helper functions, struct definitions.
    VertexPreamble,
    
    /// Injected at the beginning of the vertex shader's main function.
    ///
    /// Use for: Early vertex processing, input modifications.
    VertexMain,
    
    /// Injected at the end of the vertex shader's main function, after position calculation.
    ///
    /// Use for: Final vertex output modifications, passing data to fragment shader.
    VertexPostProcess,
    
    /// Injected before the fragment shader's main function.
    ///
    /// Use for: Global declarations, texture bindings, helper functions.
    FragmentPreamble,
    
    /// Injected at the beginning of the fragment shader's main function.
    ///
    /// Use for: Early fragment processing, discard operations.
    FragmentMain,
    
    /// Injected during color calculation in the fragment shader.
    ///
    /// Use for: Lighting calculations, material application, texture sampling.
    /// This is where most rendering features modify the final color.
    FragmentColorCalculation,
    
    /// Injected at the end of the fragment shader, after all color calculations.
    ///
    /// Use for: Post-processing effects, tone mapping, color grading.
    FragmentPostProcess,
}

/// Shader code contributed by a feature at a specific injection point.
///
/// Features return a collection of these from `shader_injections()`. The shader
/// composer will insert them into the base shader template at the appropriate markers.
///
/// # Priority
/// When multiple features inject at the same point, priority determines order:
/// - Lower values execute first (e.g., -10 for base material setup)
/// - Higher values execute later (e.g., 10 for final post-processing)
/// - Default is 0 for most features
#[derive(Clone)]
pub struct ShaderInjection {
    /// Where in the pipeline to inject this code
    pub point: ShaderInjectionPoint,
    
    /// The WGSL code to inject (uses Cow to avoid clones when possible)
    pub code: Cow<'static, str>,
    
    /// Execution priority - lower values run first, higher values run last
    pub priority: i32,
}

impl ShaderInjection {
    /// Create a new shader injection with default priority (0).
    ///
    /// # Example
    /// ```ignore
    /// ShaderInjection::new(
    ///     ShaderInjectionPoint::FragmentPreamble,
    ///     include_str!("shaders/functions.wgsl")
    /// )
    /// ```
    pub fn new(point: ShaderInjectionPoint, code: impl Into<Cow<'static, str>>) -> Self {
        Self {
            point,
            code: code.into(),
            priority: 0,
        }
    }
    
    /// Create a new shader injection with custom priority.
    pub fn with_priority(
        point: ShaderInjectionPoint,
        code: impl Into<Cow<'static, str>>,
        priority: i32,
    ) -> Self {
        Self {
            point,
            code: code.into(),
            priority,
        }
    }
}

/// Context provided to features during initialization and rendering.
///
/// This provides features with access to the GPU context and frame state.
/// Features should store any resources they create rather than creating
/// them every frame.
pub struct FeatureContext {
    /// The GPU context for creating resources
    pub gpu: Arc<gpu::Context>,
    
    /// Current surface/framebuffer dimensions
    pub surface_size: (u32, u32),
    
    /// Current frame number (useful for animation/time-based effects)
    pub frame_index: u64,
    
    /// Time since last frame in seconds
    pub delta_time: f32,
    
    /// The depth buffer format used by the renderer
    pub depth_format: gpu::TextureFormat,
    
    /// The color target format used by the renderer
    pub color_format: gpu::TextureFormat,
}

impl FeatureContext {
    /// Create a new feature context with the given parameters.
    pub fn new(
        gpu: Arc<gpu::Context>,
        surface_size: (u32, u32),
        depth_format: gpu::TextureFormat,
        color_format: gpu::TextureFormat,
    ) -> Self {
        Self {
            gpu,
            surface_size,
            frame_index: 0,
            delta_time: 0.0,
            depth_format,
            color_format,
        }
    }
    
    /// Update the context for a new frame.
    pub fn update_frame(&mut self, frame_index: u64, delta_time: f32) {
        self.frame_index = frame_index;
        self.delta_time = delta_time;
    }
    
    /// Update the surface size (call on window resize).
    pub fn update_surface_size(&mut self, width: u32, height: u32) {
        self.surface_size = (width, height);
    }
}

/// Core trait that all rendering features must implement.
///
/// # Feature Lifecycle
/// 1. **Creation**: Feature is created with `new()` or builder pattern
/// 2. **Registration**: Added to `FeatureRegistry` via `register()`
/// 3. **Initialization**: `init()` is called once - create GPU resources here
/// 4. **Per-Frame**:
///    - `prepare_frame()` - Update per-frame data (uniforms, etc.)
///    - `pre_render_pass()` - Optional pre-pass (shadows, etc.)
///    - Main render pass happens (feature binds resources via shader data layouts)
///    - `post_render_pass()` - Optional post-processing
/// 5. **Cleanup**: `cleanup()` - Release GPU resources
///
/// # Thread Safety
/// Features must be `Send + Sync` to support multi-threaded rendering.
///
/// # Example
/// ```ignore
/// pub struct MyFeature {
///     enabled: bool,
///     my_texture: Option<gpu::Texture>,
/// }
///
/// impl Feature for MyFeature {
///     fn name(&self) -> &str { "my_feature" }
///     
///     fn init(&mut self, context: &FeatureContext) {
///         // Create GPU resources once
///         self.my_texture = Some(context.gpu.create_texture(...));
///     }
///     
///     fn shader_injections(&self) -> Vec<ShaderInjection> {
///         vec![
///             ShaderInjection::new(
///                 ShaderInjectionPoint::FragmentPreamble,
///                 include_str!("../shaders/my_feature.wgsl")
///             )
///         ]
///     }
///     
///     fn cleanup(&mut self, context: &FeatureContext) {
///         if let Some(tex) = self.my_texture.take() {
///             context.gpu.destroy_texture(tex);
///         }
///     }
/// }
/// ```
pub trait Feature: Send + Sync {
    /// Returns a unique identifier for this feature.
    ///
    /// Used for feature lookup and debugging. Should be lowercase snake_case.
    fn name(&self) -> &str;

    /// Initialize the feature and create GPU resources.
    ///
    /// Called once when the feature is registered and the renderer is created.
    /// This is where you should create textures, buffers, pipelines, etc.
    ///
    /// # Important
    /// - Store created resources as `Option<Resource>` in your feature struct
    /// - Don't create resources in `new()`, do it here with the GPU context
    /// - This is only called once, not per-frame
    fn init(&mut self, context: &FeatureContext);

    /// Check if this feature is currently enabled.
    ///
    /// Disabled features are skipped during shader composition and rendering.
    /// Default implementation returns `true`.
    fn is_enabled(&self) -> bool {
        true
    }

    /// Enable or disable this feature at runtime.
    ///
    /// # Note
    /// Changing enabled state requires rebuilding the render pipeline via
    /// `FeatureRenderer::rebuild_pipeline()` to recompose shaders.
    fn set_enabled(&mut self, enabled: bool);

    /// Returns shader code injections for this feature.
    ///
    /// Only called for enabled features. The injections are sorted by priority
    /// and inserted into the base shader template at the corresponding markers.
    ///
    /// # Example
    /// ```ignore
    /// vec![
    ///     ShaderInjection::new(
    ///         ShaderInjectionPoint::FragmentPreamble,
    ///         include_str!("../shaders/functions.wgsl")
    ///     ),
    ///     ShaderInjection::with_priority(
    ///         ShaderInjectionPoint::FragmentColorCalculation,
    ///         "    final_color = my_function(final_color);",
    ///         10  // High priority - runs late
    ///     ),
    /// ]
    /// ```
    fn shader_injections(&self) -> Vec<ShaderInjection> {
        Vec::new()
    }

    /// Execute a custom render or compute pass before the main render pass.
    ///
    /// Called once per frame for enabled features. Useful for:
    /// - Shadow map generation
    /// - Pre-compute effects (e.g., particle simulation)
    /// - Generating textures used in main pass
    ///
    /// # Arguments
    /// - `encoder`: Command encoder to record GPU commands
    /// - `context`: Current frame context with timing and size info
    fn pre_render_pass(&mut self, encoder: &mut gpu::CommandEncoder, context: &FeatureContext) {
        let _ = (encoder, context);
    }

    /// Execute a custom render or compute pass after the main render pass.
    ///
    /// Called once per frame for enabled features. Useful for:
    /// - Post-processing effects (bloom, tone mapping)
    /// - Compute-based analysis of render results
    /// - Debug visualization
    fn post_render_pass(&mut self, encoder: &mut gpu::CommandEncoder, context: &FeatureContext) {
        let _ = (encoder, context);
    }

    /// Prepare per-frame data before rendering.
    ///
    /// Called once per frame for enabled features, before any rendering.
    /// Use this to update uniforms, animation state, etc.
    ///
    /// # Example
    /// ```ignore
    /// fn prepare_frame(&mut self, context: &FeatureContext) {
    ///     self.time += context.delta_time;
    ///     // Update uniform buffer with new time value
    /// }
    /// ```
    fn prepare_frame(&mut self, context: &FeatureContext) {
        let _ = context;
    }

    /// Clean up GPU resources when the feature is destroyed.
    ///
    /// Called when the renderer is dropped or the feature is removed.
    /// Make sure to destroy all GPU resources (textures, buffers, pipelines).
    ///
    /// # Example
    /// ```ignore
    /// fn cleanup(&mut self, context: &FeatureContext) {
    ///     if let Some(texture) = self.texture.take() {
    ///         context.gpu.destroy_texture(texture);
    ///     }
    ///     if let Some(buffer) = self.buffer.take() {
    ///         context.gpu.destroy_buffer(buffer);
    ///     }
    /// }
    /// ```
    fn cleanup(&mut self, context: &FeatureContext) {
        let _ = context;
    }

    /// Render geometry for shadow map or other depth-only passes.
    ///
    /// Called by shadow-casting features to render scene geometry into shadow maps.
    /// Only override this if your feature needs to render shadows.
    ///
    /// # Arguments
    /// - `encoder`: Command encoder for recording render commands
    /// - `context`: Current frame context
    /// - `meshes`: Scene geometry to render
    /// - `light_view_proj`: Light's view-projection matrix
    fn render_shadow_pass(
        &mut self,
        encoder: &mut gpu::CommandEncoder,
        context: &FeatureContext,
        meshes: &[crate::MeshData],
        light_view_proj: [[f32; 4]; 4],
    ) {
        let _ = (encoder, context, meshes, light_view_proj);
    }
}

/// Mesh data passed to shadow rendering.
///
/// Represents geometry that can be rendered into shadow maps or other
/// depth-only passes. Contains all data needed to draw indexed geometry.
pub struct MeshData {
    /// Model transformation matrix (world space)
    pub transform: [[f32; 4]; 4],
    
    /// Vertex buffer containing packed vertex data
    pub vertex_buffer: gpu::BufferPiece,
    
    /// Index buffer for indexed drawing
    pub index_buffer: gpu::BufferPiece,
    
    /// Number of indices to draw
    pub index_count: u32,
}

/// Error type for feature system operations.
#[derive(Debug, Clone)]
pub enum FeatureError {
    /// Feature with given name not found
    FeatureNotFound(String),
    
    /// Shader composition failed
    ShaderCompositionFailed(String),
    
    /// Invalid injection point marker in base shader
    InvalidInjectionPoint(String),
}

impl std::fmt::Display for FeatureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FeatureError::FeatureNotFound(name) => 
                write!(f, "Feature '{}' not found in registry", name),
            FeatureError::ShaderCompositionFailed(reason) => 
                write!(f, "Shader composition failed: {}", reason),
            FeatureError::InvalidInjectionPoint(point) => 
                write!(f, "Invalid injection point: {}", point),
        }
    }
}

impl std::error::Error for FeatureError {}

/// Registry that manages all rendering features and composes them into shaders.
///
/// The registry is the central manager for all features in your rendering pipeline.
/// It handles feature registration, initialization, shader composition, and
/// coordinating feature lifecycle.
///
/// # Example
/// ```ignore
/// let mut registry = FeatureRegistry::new();
/// registry.register(BaseGeometry::new());
/// registry.register(BasicLighting::new());
/// registry.register(BasicMaterials::new());
///
/// // Or use builder pattern:
/// let registry = FeatureRegistry::builder()
///     .with_feature(BaseGeometry::new())
///     .with_feature(BasicLighting::new())
///     .with_feature(BasicMaterials::new())
///     .build();
/// ```
pub struct FeatureRegistry {
    features: Vec<Box<dyn Feature>>,
    /// Enable debug output of composed shaders
    debug_output: bool,
}

impl FeatureRegistry {
    /// Create a new empty feature registry.
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
            debug_output: false,
        }
    }
    
    /// Create a builder for fluent feature registration.
    ///
    /// # Example
    /// ```ignore
    /// let registry = FeatureRegistry::builder()
    ///     .with_feature(BaseGeometry::new())
    ///     .with_feature(BasicLighting::new())
    ///     .debug_output(true)
    ///     .build();
    /// ```
    pub fn builder() -> FeatureRegistryBuilder {
        FeatureRegistryBuilder::new()
    }
    
    /// Enable or disable debug output of composed shaders.
    ///
    /// When enabled, composed shaders are written to `composed_shader_debug.wgsl`
    /// in the working directory for inspection.
    pub fn set_debug_output(&mut self, enabled: bool) {
        self.debug_output = enabled;
    }

    /// Add a feature to the registry.
    ///
    /// Features are processed in registration order during shader composition
    /// (though priority values can override this within injection points).
    ///
    /// # Example
    /// ```ignore
    /// registry.register(BasicLighting::new());
    /// ```
    pub fn register<F: Feature + 'static>(&mut self, feature: F) {
        self.features.push(Box::new(feature));
    }

    /// Initialize all features with the given context.
    ///
    /// Called automatically by `FeatureRenderer::new()`. Only call manually
    /// if you're managing features outside of `FeatureRenderer`.
    pub fn init_all(&mut self, context: &FeatureContext) {
        for feature in &mut self.features {
            log::debug!("Initializing feature: {}", feature.name());
            feature.init(context);
        }
    }
    
    /// Clean up all features and release GPU resources.
    ///
    /// Called automatically when `FeatureRenderer` is dropped. Call manually
    /// if you need to explicitly clean up resources.
    pub fn cleanup_all(&mut self, context: &FeatureContext) {
        for feature in &mut self.features {
            log::debug!("Cleaning up feature: {}", feature.name());
            feature.cleanup(context);
        }
    }

    /// Compose shader code from all enabled features.
    ///
    /// Takes a base shader template with injection markers and replaces them
    /// with code from enabled features. Markers have the format:
    /// `// INJECT_<INJECTIONPOINT>` (e.g., `// INJECT_FRAGMENTPREAMBLE`)
    ///
    /// # Returns
    /// The fully composed shader source code as a String.
    ///
    /// # Validation
    /// Logs warnings if injection markers are unused (no features inject there).
    pub fn compose_shader(&self, base_shader: &str) -> String {
        use std::collections::HashMap;

        // Collect all injections from enabled features
        let mut injections: HashMap<ShaderInjectionPoint, Vec<ShaderInjection>> = HashMap::new();
        let mut used_points = std::collections::HashSet::new();

        for feature in &self.features {
            if feature.is_enabled() {
                log::debug!("Composing shader with feature: {}", feature.name());
                for injection in feature.shader_injections() {
                    used_points.insert(injection.point);
                    injections
                        .entry(injection.point)
                        .or_insert_with(Vec::new)
                        .push(injection);
                }
            }
        }

        // Sort injections by priority at each point
        for injections_at_point in injections.values_mut() {
            injections_at_point.sort_by_key(|inj| inj.priority);
        }

        // Replace injection points in base shader
        let mut result = base_shader.to_string();
        
        // Check for unused injection markers
        for point in [
            ShaderInjectionPoint::VertexPreamble,
            ShaderInjectionPoint::VertexMain,
            ShaderInjectionPoint::VertexPostProcess,
            ShaderInjectionPoint::FragmentPreamble,
            ShaderInjectionPoint::FragmentMain,
            ShaderInjectionPoint::FragmentColorCalculation,
            ShaderInjectionPoint::FragmentPostProcess,
        ] {
            let marker = format!("// INJECT_{:?}", point).to_uppercase();
            if result.contains(&marker) && !used_points.contains(&point) {
                log::debug!("Injection point {:?} present but unused", point);
            }
        }

        for (point, injections_at_point) in injections {
            let marker = format!("// INJECT_{:?}", point).to_uppercase();
            
            if !result.contains(&marker) {
                log::warn!(
                    "Injection point {:?} has code but marker '{}' not found in base shader",
                    point, marker
                );
                continue;
            }
            
            let code = injections_at_point
                .iter()
                .map(|inj| inj.code.as_ref())
                .collect::<Vec<_>>()
                .join("\n");

            result = result.replace(&marker, &code);
        }
        
        // Write debug output if enabled
        if self.debug_output {
            if let Err(e) = std::fs::write("composed_shader_debug.wgsl", &result) {
                log::warn!("Failed to write debug shader output: {}", e);
            } else {
                log::info!("Wrote composed shader to composed_shader_debug.wgsl");
            }
        }

        result
    }

    /// Prepare all features for the current frame.
    ///
    /// Calls `prepare_frame()` on all enabled features.
    pub fn prepare_frame(&mut self, context: &FeatureContext) {
        for feature in &mut self.features {
            if feature.is_enabled() {
                feature.prepare_frame(context);
            }
        }
    }

    /// Execute pre-render passes for all enabled features.
    ///
    /// Called before the main render pass. Features can render shadow maps,
    /// run compute shaders, or perform other preparatory work.
    pub fn execute_pre_passes(&mut self, encoder: &mut gpu::CommandEncoder, context: &FeatureContext) {
        for feature in &mut self.features {
            if feature.is_enabled() {
                feature.pre_render_pass(encoder, context);
            }
        }
    }

    /// Execute post-render passes for all enabled features.
    ///
    /// Called after the main render pass. Features can apply post-processing
    /// effects, run analysis, or perform cleanup work.
    pub fn execute_post_passes(&mut self, encoder: &mut gpu::CommandEncoder, context: &FeatureContext) {
        for feature in &mut self.features {
            if feature.is_enabled() {
                feature.post_render_pass(encoder, context);
            }
        }
    }

    /// Execute shadow passes for all enabled features that support shadow rendering.
    ///
    /// # Arguments
    /// - `encoder`: Command encoder for recording commands
    /// - `context`: Current frame context
    /// - `meshes`: Scene geometry to render into shadow maps
    /// - `light_view_proj`: Light's view-projection matrix
    pub fn execute_shadow_passes(
        &mut self,
        encoder: &mut gpu::CommandEncoder,
        context: &FeatureContext,
        meshes: &[MeshData],
        light_view_proj: [[f32; 4]; 4],
    ) {
        for feature in &mut self.features {
            if feature.is_enabled() {
                feature.render_shadow_pass(encoder, context, meshes, light_view_proj);
            }
        }
    }

    /// Get an immutable reference to all features.
    pub fn features(&self) -> &[Box<dyn Feature>] {
        &self.features
    }

    /// Get a mutable reference to all features.
    pub fn features_mut(&mut self) -> &mut [Box<dyn Feature>] {
        &mut self.features
    }

    /// Get a feature by name.
    ///
    /// # Example
    /// ```ignore
    /// if let Some(lighting) = registry.get_feature("basic_lighting") {
    ///     println!("Lighting is enabled: {}", lighting.is_enabled());
    /// }
    /// ```
    pub fn get_feature(&self, name: &str) -> Option<&dyn Feature> {
        self.features
            .iter()
            .find(|f| f.name() == name)
            .map(|f| f.as_ref())
    }

    /// Get a mutable reference to a feature by name.
    pub fn get_feature_mut(&mut self, name: &str) -> Option<&mut Box<dyn Feature>> {
        self.features
            .iter_mut()
            .find(|f| f.name() == name)
    }

    /// Toggle a feature's enabled state by name.
    ///
    /// # Returns
    /// `Ok(new_state)` if the feature was found and toggled,
    /// `Err(FeatureError)` if the feature doesn't exist.
    ///
    /// # Note
    /// After toggling, you must call `FeatureRenderer::rebuild_pipeline()`
    /// to recompose shaders with the new feature state.
    ///
    /// # Example
    /// ```ignore
    /// if let Ok(new_state) = registry.toggle_feature("basic_lighting") {
    ///     println!("Lighting is now: {}", if new_state { "enabled" } else { "disabled" });
    ///     renderer.rebuild_pipeline();
    /// }
    /// ```
    pub fn toggle_feature(&mut self, name: &str) -> Result<bool, FeatureError> {
        if let Some(feature) = self.get_feature_mut(name) {
            let new_state = !feature.is_enabled();
            feature.set_enabled(new_state);
            Ok(new_state)
        } else {
            Err(FeatureError::FeatureNotFound(name.to_string()))
        }
    }
    
    /// Enable a feature by name.
    ///
    /// # Returns
    /// `Ok(())` if successful, `Err(FeatureError)` if feature not found.
    pub fn enable_feature(&mut self, name: &str) -> Result<(), FeatureError> {
        if let Some(feature) = self.get_feature_mut(name) {
            feature.set_enabled(true);
            Ok(())
        } else {
            Err(FeatureError::FeatureNotFound(name.to_string()))
        }
    }
    
    /// Disable a feature by name.
    ///
    /// # Returns
    /// `Ok(())` if successful, `Err(FeatureError)` if feature not found.
    pub fn disable_feature(&mut self, name: &str) -> Result<(), FeatureError> {
        if let Some(feature) = self.get_feature_mut(name) {
            feature.set_enabled(false);
            Ok(())
        } else {
            Err(FeatureError::FeatureNotFound(name.to_string()))
        }
    }
    
    /// Get the count of enabled features.
    pub fn enabled_count(&self) -> usize {
        self.features.iter().filter(|f| f.is_enabled()).count()
    }
    
    /// Get the total count of registered features.
    pub fn total_count(&self) -> usize {
        self.features.len()
    }
    
    /// List all feature names.
    pub fn feature_names(&self) -> Vec<&str> {
        self.features.iter().map(|f| f.name()).collect()
    }
}

impl Default for FeatureRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for fluent FeatureRegistry construction.
///
/// # Example
/// ```ignore
/// let registry = FeatureRegistry::builder()
///     .with_feature(BaseGeometry::new())
///     .with_feature(BasicLighting::new())
///     .with_feature(BasicMaterials::new())
///     .debug_output(true)
///     .build();
/// ```
pub struct FeatureRegistryBuilder {
    features: Vec<Box<dyn Feature>>,
    debug_output: bool,
}

impl FeatureRegistryBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
            debug_output: false,
        }
    }
    
    /// Add a feature to the registry.
    pub fn with_feature<F: Feature + 'static>(mut self, feature: F) -> Self {
        self.features.push(Box::new(feature));
        self
    }
    
    /// Enable debug output of composed shaders.
    pub fn debug_output(mut self, enabled: bool) -> Self {
        self.debug_output = enabled;
        self
    }
    
    /// Build the FeatureRegistry.
    pub fn build(self) -> FeatureRegistry {
        FeatureRegistry {
            features: self.features,
            debug_output: self.debug_output,
        }
    }
}

impl Default for FeatureRegistryBuilder {
    fn default() -> Self {
        Self::new()
    }
}
