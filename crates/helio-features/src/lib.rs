use blade_graphics as gpu;
use std::any::Any;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

/// Helper trait enabling safe downcasting of `dyn Feature` to a concrete type.
///
/// Implemented automatically for all `'static` types, so any `Feature` implementor
/// that owns its data (no lifetime parameters) gets this for free.
///
/// # Example
/// ```ignore
/// if let Some(shadows) = feature.as_any().downcast_ref::<ProceduralShadows>() {
///     let data = shadows.get_shadow_map_data();
/// }
/// ```
pub trait AsAny: 'static {
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

impl<T: 'static> AsAny for T {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

// ============================================================================
// Feature Data Export System
// ============================================================================

/// Marker trait for CPU-side data that can be exported between features.
///
/// Any struct that implements this trait can be wrapped in `ExportedData::CpuData`
/// and shared between features with zero-copy semantics via `Arc`.
///
/// # Requirements
/// - `Any`: Enables runtime type checking and downcasting
/// - `Send + Sync`: Thread-safe sharing across features
///
/// # Example
/// ```ignore
/// #[derive(Clone)]
/// pub struct MaterialProperties {
///     pub base_color: [f32; 4],
///     pub metallic: f32,
///     pub roughness: f32,
/// }
///
/// impl FeatureData for MaterialProperties {}
/// ```
pub trait FeatureData: Any + Send + Sync {
    /// Optional: Return a human-readable description of this data.
    fn description(&self) -> &str {
        "Feature data"
    }
}

/// Represents data exported by a feature, supporting both CPU and GPU resources.
///
/// Uses zero-copy semantics:
/// - CPU data is shared via `Arc<T>` - multiple features can hold references
/// - GPU resources are handles/views - cheap to copy, no actual data duplication
///
/// # Zero-Copy Guarantee
/// - `CpuData`: Uses `Arc` for shared ownership without cloning the actual data
/// - `GpuBuffer`/`GpuTexture`: Handles are lightweight references to GPU memory
///
/// # Example
/// ```ignore
/// // Export material properties (CPU data shared via Arc)
/// let props = Arc::new(MaterialProperties { ... });
/// ExportedData::CpuData(props)
///
/// // Export GPU buffer handle (cheap to copy)
/// ExportedData::GpuBuffer(material_buffer_handle)
/// ```
#[derive(Clone)]
pub enum ExportedData {
    /// CPU-side metadata shared via Arc (zero-copy reference counting).
    ///
    /// Multiple features can hold references to the same data without duplication.
    /// To access: `downcast_arc::<T>()`
    CpuData(Arc<dyn FeatureData>),
    
    /// GPU buffer handle (cheap to copy, references GPU memory).
    ///
    /// Represents a reference to a GPU buffer. The handle itself is small,
    /// the actual buffer lives in GPU memory.
    GpuBuffer(gpu::BufferPiece),
    
    /// GPU texture view (cheap to copy, references GPU texture).
    ///
    /// Represents a view into GPU texture memory. The view is lightweight,
    /// the actual texture data lives in GPU memory.
    GpuTexture(gpu::TextureView),
}

impl ExportedData {
    /// Downcast CPU data to a specific type and get a cloned Arc.
    ///
    /// Returns `Some(Arc<T>)` if this is `CpuData` containing type `T`, `None` otherwise.
    /// This is zero-copy - only the Arc is cloned (incrementing ref count), not the data.
    ///
    /// # Example
    /// ```ignore
    /// if let Some(props) = data.downcast_arc::<MaterialProperties>() {
    ///     println!("Metallic: {}", props.metallic);
    /// }
    /// ```
    pub fn downcast_arc<T: FeatureData + 'static>(&self) -> Option<Arc<T>> {
        match self {
            ExportedData::CpuData(arc) => {
                // Clone the Arc (cheap), then try downcast
                let arc_any = Arc::clone(arc) as Arc<dyn Any + Send + Sync>;
                arc_any.downcast::<T>().ok()
            }
            _ => None,
        }
    }
    
    /// Get GPU buffer handle if this is buffer data.
    pub fn as_gpu_buffer(&self) -> Option<&gpu::BufferPiece> {
        match self {
            ExportedData::GpuBuffer(buffer) => Some(buffer),
            _ => None,
        }
    }
    
    /// Get GPU texture view if this is texture data.
    pub fn as_gpu_texture(&self) -> Option<&gpu::TextureView> {
        match self {
            ExportedData::GpuTexture(view) => Some(view),
            _ => None,
        }
    }
}

// Helper to enable Arc downcasting for FeatureData
impl dyn FeatureData {
    /// Downcast self to Any for type checking.
    pub fn as_any(&self) -> &dyn Any {
        self as &dyn Any
    }
}

/// Optional trait for features that export data to other features.
///
/// Features that produce data useful to other features should implement this trait.
/// All data is shared with zero-copy semantics via Arc or GPU handles.
///
/// **NOTE**: You typically don't need to implement this trait directly. Instead,
/// implement the `export_data()` method on your Feature implementation (it has a
/// default empty implementation).
///
/// # When to Implement
/// Implement `export_data()` if your feature produces:
/// - Material properties for lighting calculations
/// - Light data for shadow rendering
/// - Texture maps for post-processing
/// - Any shared configuration or state
///
/// # Example
/// ```ignore
/// impl Feature for MaterialFeature {
///     // ... other Feature methods ...
///     
///     fn export_data(&self) -> HashMap<String, ExportedData> {
///         let mut exports = HashMap::new();
///         
///         // Export CPU metadata (zero-copy via Arc)
///         let props = Arc::new(MaterialProperties {
///             base_color: self.base_color,
///             metallic: self.metallic,
///             roughness: self.roughness,
///         });
///         exports.insert("properties".to_string(), ExportedData::CpuData(props));
///         
///         // Export GPU buffer handle (cheap handle copy)
///         if let Some(buffer) = &self.material_buffer {
///             exports.insert("buffer".to_string(), ExportedData::GpuBuffer(*buffer));
///         }
///         
///         exports
///     }
/// }
/// ```
pub trait DataExporter {
    /// Export all data provided by this feature.
    ///
    /// Returns a map of export names to data. Export names should be descriptive
    /// and unique within the feature (e.g., "properties", "buffer", "shadow_map").
    ///
    /// All data uses zero-copy semantics:
    /// - CPU data wrapped in `Arc` for shared ownership
    /// - GPU resources are handles/views (cheap to copy)
    ///
    /// This method is called lazily when another feature queries for data,
    /// so it should be fast - just wrap existing data in Arc/return handles.
    fn export_data(&self) -> HashMap<String, ExportedData>;
}

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
pub trait Feature: Send + Sync + AsAny {
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
    fn render_shadow_pass(
        &mut self,
        encoder: &mut gpu::CommandEncoder,
        context: &FeatureContext,
        meshes: &[crate::MeshData],
    ) {
        let _ = (encoder, context, meshes);
    }

    /// Get the shadow map texture view for binding in shaders.
    ///
    /// Features that implement shadow mapping should return their shadow map texture view.
    /// The renderer will bind this at group 2 for use in the main render pass.
    ///
    /// # Returns
    /// `Some(texture_view)` if this feature provides a shadow map, `None` otherwise.
    fn get_shadow_map_view(&self) -> Option<gpu::TextureView> {
        None
    }

    /// Export data that other features can consume (optional).
    ///
    /// Features that produce shareable data should override this method.
    /// All data uses zero-copy semantics via Arc or GPU handles.
    ///
    /// # Zero-Copy Guarantee
    /// - CPU data: Wrap in `Arc::new(data)` - shared ownership, no cloning
    /// - GPU resources: Return handles/views - cheap to copy
    ///
    /// # Example
    /// ```ignore
    /// fn export_data(&self) -> HashMap<String, ExportedData> {
    ///     let mut exports = HashMap::new();
    ///     
    ///     // CPU data (zero-copy via Arc)
    ///     let props = Arc::new(MaterialProperties {
    ///         metallic: self.metallic,
    ///         roughness: self.roughness,
    ///     });
    ///     exports.insert("properties".to_string(), ExportedData::CpuData(props));
    ///     
    ///     // GPU handle (cheap copy)
    ///     if let Some(buf) = &self.buffer {
    ///         exports.insert("buffer".to_string(), ExportedData::GpuBuffer(*buf));
    ///     }
    ///     
    ///     exports
    /// }
    /// ```
    fn export_data(&self) -> HashMap<String, ExportedData> {
        HashMap::new()
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
    /// Custom path for debug shader output (None = "composed_shader_debug.wgsl")
    debug_output_path: Option<String>,
    /// HashMap for O(1) feature name lookup
    feature_indices: std::collections::HashMap<String, usize>,
}

impl FeatureRegistry {
    /// Create a new empty feature registry.
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
            debug_output: false,
            debug_output_path: None,
            feature_indices: std::collections::HashMap::new(),
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
        let name = feature.name().to_string();
        let index = self.features.len();
        self.features.push(Box::new(feature));
        self.feature_indices.insert(name, index);
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

        for feature in &self.features {
            if feature.is_enabled() {
                log::debug!("Composing shader with feature: {}", feature.name());
                for injection in feature.shader_injections() {
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

        // Pre-compute injection markers once
        let markers: HashMap<ShaderInjectionPoint, String> = [
            ShaderInjectionPoint::VertexPreamble,
            ShaderInjectionPoint::VertexMain,
            ShaderInjectionPoint::VertexPostProcess,
            ShaderInjectionPoint::FragmentPreamble,
            ShaderInjectionPoint::FragmentMain,
            ShaderInjectionPoint::FragmentColorCalculation,
            ShaderInjectionPoint::FragmentPostProcess,
        ]
        .iter()
        .map(|&point| {
            let marker = format!("// INJECT_{:?}", point).to_uppercase();
            (point, marker)
        })
        .collect();

        // Single-pass shader composition using efficient string building
        let mut result = String::with_capacity(base_shader.len() + injections.len() * 256);
        let mut current_pos = 0;
        
        // Find all marker positions in base shader
        let mut marker_positions: Vec<(usize, ShaderInjectionPoint, &str)> = Vec::new();
        for (point, marker) in &markers {
            if let Some(pos) = base_shader[current_pos..].find(marker.as_str()) {
                marker_positions.push((pos + current_pos, *point, marker.as_str()));
            }
        }
        
        // Sort by position for sequential replacement
        marker_positions.sort_by_key(|(pos, _, _)| *pos);
        
        // Build result string in single pass
        for (pos, point, marker) in marker_positions {
            // Add text up to marker
            result.push_str(&base_shader[current_pos..pos]);
            
            // Add injected code if exists
            if let Some(injections_at_point) = injections.get(&point) {
                for inj in injections_at_point {
                    result.push_str(inj.code.as_ref());
                    result.push('\n');
                }
            } else {
                // Keep marker if no injections for this point
                result.push_str(marker);
            }
            
            current_pos = pos + marker.len();
        }
        
        // Add remaining shader text
        result.push_str(&base_shader[current_pos..]);
        
        // Log warnings for injections without markers
        for (point, _) in &injections {
            if let Some(marker) = markers.get(point) {
                if !base_shader.contains(marker.as_str()) {
                    log::warn!(
                        "Injection point {:?} has code but marker '{}' not found in base shader",
                        point, marker
                    );
                }
            }
        }
        
        // Write debug output if enabled
        if self.debug_output {
            let path = self.debug_output_path.as_deref().unwrap_or("composed_shader_debug.wgsl");
            if let Err(e) = std::fs::write(path, &result) {
                log::warn!("Failed to write debug shader output to '{}': {}", path, e);
            } else {
                log::info!("Wrote composed shader to '{}'", path);
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
    pub fn execute_shadow_passes(
        &mut self,
        encoder: &mut gpu::CommandEncoder,
        context: &FeatureContext,
        meshes: &[MeshData],
    ) {
        for feature in &mut self.features {
            if feature.is_enabled() {
                feature.render_shadow_pass(encoder, context, meshes);
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
        self.feature_indices
            .get(name)
            .and_then(|&index| self.features.get(index))
            .map(|f| f.as_ref())
    }

    /// Get a mutable reference to a feature by name.
    pub fn get_feature_mut(&mut self, name: &str) -> Option<&mut Box<dyn Feature>> {
        if let Some(&index) = self.feature_indices.get(name) {
            self.features.get_mut(index)
        } else {
            None
        }
    }

    /// Get a feature downcast to a concrete type.
    ///
    /// Returns `None` if the feature is not found or is not of type `T`.
    ///
    /// # Example
    /// ```ignore
    /// if let Some(shadows) = registry.get_feature_as::<ProceduralShadows>("procedural_shadows") {
    ///     println!("Shadow map size: {}", shadows.shadow_map_size());
    /// }
    /// ```
    pub fn get_feature_as<T: 'static>(&self, name: &str) -> Option<&T> {
        self.get_feature(name)?.as_any().downcast_ref::<T>()
    }

    /// Get a mutable reference to a feature downcast to a concrete type.
    ///
    /// Returns `None` if the feature is not found or is not of type `T`.
    ///
    /// # Example
    /// ```ignore
    /// if let Some(shadows) = registry.get_feature_as_mut::<ProceduralShadows>("procedural_shadows") {
    ///     shadows.clear_lights();
    ///     shadows.add_light(my_light).unwrap();
    /// }
    /// ```
    pub fn get_feature_as_mut<T: 'static>(&mut self, name: &str) -> Option<&mut T> {
        let idx = *self.feature_indices.get(name)?;
        self.features.get_mut(idx)?.as_any_mut().downcast_mut::<T>()
    }

    /// Query exported data from a feature.
    ///
    /// This is the primary way for features to access data from other features
    /// with zero-copy semantics. The feature must implement the `export_data()`
    /// method and the specific export name must exist.
    ///
    /// # Arguments
    /// * `feature_name` - Name of the feature to query (e.g., "materials")
    /// * `export_name` - Name of the specific data export (e.g., "properties", "buffer")
    ///
    /// # Returns
    /// `Some(ExportedData)` if the feature exists and has an export with the given name.
    /// `None` otherwise.
    ///
    /// # Zero-Copy Guarantee
    /// - CPU data is returned as `Arc<T>` - no cloning of the actual data
    /// - GPU handles are cheap to copy - no GPU memory duplication
    ///
    /// # Example
    /// ```ignore
    /// // In lighting feature, query material properties
    /// if let Some(data) = registry.get_exported_data("materials", "properties") {
    ///     if let Some(props) = data.downcast_arc::<MaterialProperties>() {
    ///         // Use props.metallic, props.roughness for PBR calculations
    ///         self.apply_pbr_lighting(&props);
    ///     }
    /// }
    ///
    /// // Query GPU buffer handle
    /// if let Some(data) = registry.get_exported_data("materials", "buffer") {
    ///     if let Some(buffer) = data.as_gpu_buffer() {
    ///         // Bind buffer to shader
    ///     }
    /// }
    /// ```
    pub fn get_exported_data(&self, feature_name: &str, export_name: &str) -> Option<ExportedData> {
        let feature = self.get_feature(feature_name)?;
        feature.export_data().get(export_name).cloned()
    }

    /// Get all exported data from a feature.
    ///
    /// Returns all data exports from a feature.
    ///
    /// # Example
    /// ```ignore
    /// if let Some(exports) = registry.get_all_exported_data("materials") {
    ///     for (name, data) in exports {
    ///         println!("Export: {}", name);
    ///     }
    /// }
    /// ```
    pub fn get_all_exported_data(&self, feature_name: &str) -> Option<HashMap<String, ExportedData>> {
        let feature = self.get_feature(feature_name)?;
        Some(feature.export_data())
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
    debug_output_path: Option<String>,
}

impl FeatureRegistryBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
            debug_output: false,
            debug_output_path: None,
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
    
    /// Set custom path for debug shader output.
    /// 
    /// # Example
    /// ```ignore
    /// let registry = FeatureRegistry::builder()
    ///     .debug_output(true)
    ///     .debug_output_path("output/my_shader.wgsl")
    ///     .build();
    /// ```
    pub fn debug_output_path(mut self, path: impl Into<String>) -> Self {
        self.debug_output_path = Some(path.into());
        self
    }
    
    /// Build the FeatureRegistry.
    pub fn build(self) -> FeatureRegistry {
        let mut feature_indices = std::collections::HashMap::new();
        for (i, feature) in self.features.iter().enumerate() {
            feature_indices.insert(feature.name().to_string(), i);
        }
        
        FeatureRegistry {
            features: self.features,
            debug_output: self.debug_output,
            debug_output_path: self.debug_output_path,
            feature_indices,
        }
    }
}

impl Default for FeatureRegistryBuilder {
    fn default() -> Self {
        Self::new()
    }
}
