use std::borrow::Cow;
use std::sync::Arc;

pub trait AsAny: 'static {
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

impl<T: 'static> AsAny for T {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderInjectionPoint {
    VertexPreamble,
    VertexMain,
    VertexPostProcess,
    FragmentPreamble,
    FragmentMain,
    FragmentColorCalculation,
    FragmentPostProcess,
}

#[derive(Clone)]
pub struct ShaderInjection {
    pub point: ShaderInjectionPoint,
    pub code: Cow<'static, str>,
    pub priority: i32,
}

impl ShaderInjection {
    pub fn new(point: ShaderInjectionPoint, code: impl Into<Cow<'static, str>>) -> Self {
        Self { point, code: code.into(), priority: 0 }
    }
    pub fn with_priority(point: ShaderInjectionPoint, code: impl Into<Cow<'static, str>>, priority: i32) -> Self {
        Self { point, code: code.into(), priority }
    }
}

pub struct FeatureContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub surface_size: (u32, u32),
    pub frame_index: u64,
    pub delta_time: f32,
    pub depth_format: wgpu::TextureFormat,
    pub color_format: wgpu::TextureFormat,
    pub camera_position: [f32; 3],
}

impl FeatureContext {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        surface_size: (u32, u32),
        depth_format: wgpu::TextureFormat,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        Self { device, queue, surface_size, frame_index: 0, delta_time: 0.0, depth_format, color_format, camera_position: [0.0; 3] }
    }
    pub fn update_frame(&mut self, frame_index: u64, delta_time: f32) {
        self.frame_index = frame_index;
        self.delta_time = delta_time;
    }
    pub fn update_surface_size(&mut self, width: u32, height: u32) {
        self.surface_size = (width, height);
    }
}

pub struct MeshData {
    pub transform: [[f32; 4]; 4],
    pub vertex_buffer: Arc<wgpu::Buffer>,
    pub index_buffer: Arc<wgpu::Buffer>,
    pub index_count: u32,
}

pub trait Feature: Send + Sync + AsAny {
    fn name(&self) -> &str;
    fn init(&mut self, context: &FeatureContext);
    fn is_enabled(&self) -> bool { true }
    fn set_enabled(&mut self, enabled: bool);
    fn shader_injections(&self) -> Vec<ShaderInjection> { Vec::new() }

    fn pre_render_pass(&mut self, encoder: &mut wgpu::CommandEncoder, context: &FeatureContext) {
        let _ = (encoder, context);
    }
    fn post_render_pass(&mut self, encoder: &mut wgpu::CommandEncoder, context: &FeatureContext) {
        let _ = (encoder, context);
    }
    fn prepare_frame(&mut self, context: &FeatureContext) { let _ = context; }
    fn cleanup(&mut self, context: &FeatureContext) { let _ = context; }

    fn render_shadow_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        context: &FeatureContext,
        meshes: &[MeshData],
    ) {
        let _ = (encoder, context, meshes);
    }

    /// Returns (group_index, BindGroupLayout) for this feature\'s main-pass bindings.
    /// Called once during pipeline creation (after init).
    fn main_pass_bind_group_layout(&self, device: &wgpu::Device) -> Option<(u32, wgpu::BindGroupLayout)> {
        let _ = device;
        None
    }

    /// Returns (group_index, &BindGroup) to set in the main render pass each frame.
    fn main_pass_bind_group(&self) -> Option<(u32, &wgpu::BindGroup)> { None }
}

#[derive(Debug, Clone)]
pub enum FeatureError {
    FeatureNotFound(String),
    ShaderCompositionFailed(String),
}

impl std::fmt::Display for FeatureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FeatureError::FeatureNotFound(n) => write!(f, "Feature {} not found", n),
            FeatureError::ShaderCompositionFailed(r) => write!(f, "Shader composition failed: {}", r),
        }
    }
}
impl std::error::Error for FeatureError {}

pub struct FeatureRegistry {
    features: Vec<Box<dyn Feature>>,
    debug_output: bool,
    feature_indices: std::collections::HashMap<String, usize>,
}

impl FeatureRegistry {
    pub fn new() -> Self {
        Self { features: Vec::new(), debug_output: false, feature_indices: std::collections::HashMap::new() }
    }
    pub fn builder() -> FeatureRegistryBuilder { FeatureRegistryBuilder::new() }
    pub fn set_debug_output(&mut self, enabled: bool) { self.debug_output = enabled; }

    pub fn register<F: Feature + 'static>(&mut self, feature: F) {
        let name = feature.name().to_string();
        let index = self.features.len();
        self.features.push(Box::new(feature));
        self.feature_indices.insert(name, index);
    }

    pub fn init_all(&mut self, context: &FeatureContext) {
        for feature in &mut self.features {
            feature.init(context);
        }
    }

    pub fn cleanup_all(&mut self, context: &FeatureContext) {
        for feature in &mut self.features {
            feature.cleanup(context);
        }
    }

    pub fn compose_shader(&self, base_shader: &str) -> String {
        use std::collections::HashMap;
        let mut injections: HashMap<ShaderInjectionPoint, Vec<ShaderInjection>> = HashMap::new();
        for feature in &self.features {
            if feature.is_enabled() {
                for injection in feature.shader_injections() {
                    injections.entry(injection.point).or_insert_with(Vec::new).push(injection);
                }
            }
        }
        for v in injections.values_mut() { v.sort_by_key(|i| i.priority); }

        let markers: HashMap<ShaderInjectionPoint, String> = [
            ShaderInjectionPoint::VertexPreamble,
            ShaderInjectionPoint::VertexMain,
            ShaderInjectionPoint::VertexPostProcess,
            ShaderInjectionPoint::FragmentPreamble,
            ShaderInjectionPoint::FragmentMain,
            ShaderInjectionPoint::FragmentColorCalculation,
            ShaderInjectionPoint::FragmentPostProcess,
        ].iter().map(|&p| {
            let m = format!("// INJECT_{:?}", p).to_uppercase();
            (p, m)
        }).collect();

        let mut result = String::with_capacity(base_shader.len() + 4096);
        let mut current_pos = 0;
        let mut marker_positions: Vec<(usize, ShaderInjectionPoint, String)> = Vec::new();
        for (point, marker) in &markers {
            if let Some(pos) = base_shader.find(marker.as_str()) {
                marker_positions.push((pos, *point, marker.clone()));
            }
        }
        marker_positions.sort_by_key(|(pos, _, _)| *pos);

        for (pos, point, marker) in &marker_positions {
            result.push_str(&base_shader[current_pos..*pos]);
            if let Some(injs) = injections.get(point) {
                for inj in injs { result.push_str(&inj.code); result.push('\n'); }
            } else {
                result.push_str(marker);
            }
            current_pos = pos + marker.len();
        }
        result.push_str(&base_shader[current_pos..]);

        if self.debug_output {
            if let Err(e) = std::fs::write("composed_shader_debug.wgsl", &result) {
                log::warn!("Failed to write debug shader: {}", e);
            }
        }
        result
    }

    pub fn prepare_frame(&mut self, context: &FeatureContext) {
        for feature in &mut self.features {
            if feature.is_enabled() { feature.prepare_frame(context); }
        }
    }

    pub fn execute_pre_passes(&mut self, encoder: &mut wgpu::CommandEncoder, context: &FeatureContext) {
        for feature in &mut self.features {
            if feature.is_enabled() { feature.pre_render_pass(encoder, context); }
        }
    }

    pub fn execute_post_passes(&mut self, encoder: &mut wgpu::CommandEncoder, context: &FeatureContext) {
        for feature in &mut self.features {
            if feature.is_enabled() { feature.post_render_pass(encoder, context); }
        }
    }

    pub fn execute_shadow_passes(&mut self, encoder: &mut wgpu::CommandEncoder, context: &FeatureContext, meshes: &[MeshData]) {
        for feature in &mut self.features {
            if feature.is_enabled() { feature.render_shadow_pass(encoder, context, meshes); }
        }
    }

    pub fn features(&self) -> &[Box<dyn Feature>] { &self.features }
    pub fn features_mut(&mut self) -> &mut [Box<dyn Feature>] { &mut self.features }

    pub fn get_feature(&self, name: &str) -> Option<&dyn Feature> {
        self.feature_indices.get(name).and_then(|&i| self.features.get(i)).map(|f| f.as_ref())
    }

    pub fn get_feature_mut(&mut self, name: &str) -> Option<&mut Box<dyn Feature>> {
        if let Some(&index) = self.feature_indices.get(name) {
            self.features.get_mut(index)
        } else { None }
    }

    pub fn get_feature_as<T: 'static>(&self, name: &str) -> Option<&T> {
        self.get_feature(name)?.as_any().downcast_ref::<T>()
    }

    pub fn get_feature_as_mut<T: 'static>(&mut self, name: &str) -> Option<&mut T> {
        let idx = *self.feature_indices.get(name)?;
        self.features.get_mut(idx)?.as_any_mut().downcast_mut::<T>()
    }

    pub fn toggle_feature(&mut self, name: &str) -> Result<bool, FeatureError> {
        if let Some(feature) = self.get_feature_mut(name) {
            let new_state = !feature.is_enabled();
            feature.set_enabled(new_state);
            Ok(new_state)
        } else { Err(FeatureError::FeatureNotFound(name.to_string())) }
    }

    pub fn enable_feature(&mut self, name: &str) -> Result<(), FeatureError> {
        if let Some(f) = self.get_feature_mut(name) { f.set_enabled(true); Ok(()) }
        else { Err(FeatureError::FeatureNotFound(name.to_string())) }
    }

    pub fn disable_feature(&mut self, name: &str) -> Result<(), FeatureError> {
        if let Some(f) = self.get_feature_mut(name) { f.set_enabled(false); Ok(()) }
        else { Err(FeatureError::FeatureNotFound(name.to_string())) }
    }

    pub fn enabled_count(&self) -> usize { self.features.iter().filter(|f| f.is_enabled()).count() }
    pub fn total_count(&self) -> usize { self.features.len() }
    pub fn feature_names(&self) -> Vec<&str> { self.features.iter().map(|f| f.name()).collect() }
}

impl Default for FeatureRegistry { fn default() -> Self { Self::new() } }

pub struct FeatureRegistryBuilder {
    features: Vec<Box<dyn Feature>>,
    debug_output: bool,
}

impl FeatureRegistryBuilder {
    pub fn new() -> Self { Self { features: Vec::new(), debug_output: false } }
    pub fn with_feature<F: Feature + 'static>(mut self, feature: F) -> Self {
        self.features.push(Box::new(feature)); self
    }
    pub fn debug_output(mut self, enabled: bool) -> Self { self.debug_output = enabled; self }
    pub fn build(self) -> FeatureRegistry {
        let mut feature_indices = std::collections::HashMap::new();
        for (i, f) in self.features.iter().enumerate() {
            feature_indices.insert(f.name().to_string(), i);
        }
        FeatureRegistry { features: self.features, debug_output: self.debug_output, feature_indices }
    }
}

impl Default for FeatureRegistryBuilder { fn default() -> Self { Self::new() } }
