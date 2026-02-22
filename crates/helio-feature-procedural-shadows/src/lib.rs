use blade_graphics as gpu;
use helio_core::TextureManager;
use helio_features::{Feature, FeatureContext, ShaderInjection, ShaderInjectionPoint};
use std::sync::Arc;

/// Number of cascade levels for radiance cascades
pub const NUM_CASCADES: usize = 5;

/// Light source in the scene
#[derive(Debug, Clone, Copy)]
pub struct Light {
    pub position: glam::Vec3,
    pub color: glam::Vec3,
    pub intensity: f32,
    pub radius: f32,
    pub light_type: LightType,
}

/// GPU representation of a light
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLight {
    pub position: [f32; 3],
    pub radius: f32,
    pub color: [f32; 3],
    pub intensity: f32,
}

impl From<&Light> for GpuLight {
    fn from(light: &Light) -> Self {
        Self {
            position: light.position.to_array(),
            radius: light.radius,
            color: light.color.to_array(),
            intensity: light.intensity,
        }
    }
}

/// Legacy light configuration (kept for backward compatibility)
#[derive(Debug, Clone, Copy)]
pub struct LightConfig {
    pub sun_direction: glam::Vec3,
    pub sun_color: glam::Vec3,
    pub sun_intensity: f32,
    pub sky_color: glam::Vec3,
    pub ambient_intensity: f32,
}

impl Default for LightConfig {
    fn default() -> Self {
        Self {
            sun_direction: glam::Vec3::new(-0.5, -1.0, -0.3).normalize(),
            sun_color: glam::Vec3::new(1.0, 0.9, 0.65),
            sun_intensity: 2.5,
            sky_color: glam::Vec3::new(0.7, 0.8, 1.0),
            ambient_intensity: 0.1,
        }
    }
}

/// Light type enum
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LightType {
    Point,
    Directional,
    Spot { inner_angle: f32, outer_angle: f32 },
    Rect { width: f32, height: f32 },
}

/// GPU representation of light configuration
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLightConfig {
    pub sun_direction: [f32; 3],
    pub _pad0: f32,
    pub sun_color: [f32; 3],
    pub sun_intensity: f32,
    pub sky_color: [f32; 3],
    pub ambient_intensity: f32,
}

impl From<&LightConfig> for GpuLightConfig {
    fn from(config: &LightConfig) -> Self {
        Self {
            sun_direction: config.sun_direction.to_array(),
            _pad0: 0.0,
            sun_color: config.sun_color.to_array(),
            sun_intensity: config.sun_intensity,
            sky_color: config.sky_color.to_array(),
            ambient_intensity: config.ambient_intensity,
        }
    }
}

/// GPU uniforms for scene state
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneUniforms {
    pub time: f32,
    pub cascade_index: u32,
    pub _pad0: f32,
    pub _pad1: f32,
}

/// Camera uniforms for ray tracing
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuCamera {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub position: [f32; 3],
    pub _pad: f32,
}

/// Shader data for compute pass
#[derive(blade_macros::ShaderData)]
struct RadianceComputeData {
    light_config: GpuLightConfig,
    scene: SceneUniforms,
    camera: GpuCamera,
    lights: [GpuLight; 16],
    num_lights: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    cascade_texture: gpu::TextureView,
    prev_cascade_texture: gpu::TextureView,
    prev_cascade_sampler: gpu::Sampler,
    scene_depth: gpu::TextureView,
    scene_color: gpu::TextureView,
    linear_sampler: gpu::Sampler,
}

/// Shader data for main render pass
#[derive(blade_macros::ShaderData)]
pub struct RadianceLookupData {
    pub lighting: GpuLightConfig,
    pub shadow_maps: gpu::TextureView,
    pub shadow_sampler: gpu::Sampler,
}

// Re-export for compatibility with render crate
pub type ShadowMapData = RadianceLookupData;

/// Radiance Cascades GI feature
///
/// Implements full global illumination using radiance cascades - a technique that
/// traces rays from surface-based probes at multiple scales to compute multi-bounce
/// indirect lighting and soft shadows.
///
/// # Algorithm
/// - **Surface Probes**: Probes are placed on geometry surfaces (not volumetric)
/// - **Cascades**: Multiple levels with different probe spacing (2, 4, 8, 16, 32)
/// - **Merging**: Cascades are merged using visibility-weighted interpolation
/// - **Ray Tracing**: Each probe shoots rays in hemispherical directions
///
/// # Performance
/// Cascade texture size: 1024×2048 RGBA16F per cascade level
/// Total memory: ~40 MB for 5 cascade levels
pub struct RadianceCascades {
    enabled: bool,

    // Cascade textures (one per level)
    cascade_textures: Vec<Option<gpu::Texture>>,
    cascade_views: Vec<Option<gpu::TextureView>>,
    cascade_sampler: Option<gpu::Sampler>,

    // Scene data for ray tracing
    prev_depth_view: Option<gpu::TextureView>,
    prev_color_view: Option<gpu::TextureView>,
    linear_sampler: Option<gpu::Sampler>,
    camera_data: GpuCamera,

    // Light sources
    lights: Vec<Light>,

    // Configuration (legacy)
    light_config: LightConfig,
    current_time: f32,

    // GPU resources
    context: Option<Arc<gpu::Context>>,
    compute_pipeline: Option<gpu::ComputePipeline>,

    texture_manager: Option<Arc<TextureManager>>,
}

impl RadianceCascades {
    /// Create a new radiance cascades GI feature
    pub fn new() -> Self {
        Self {
            enabled: true,
            cascade_textures: vec![None; NUM_CASCADES],
            cascade_views: vec![None; NUM_CASCADES],
            cascade_sampler: None,
            prev_depth_view: None,
            prev_color_view: None,
            linear_sampler: None,
            camera_data: GpuCamera {
                view_proj: [[0.0; 4]; 4],
                inv_view_proj: [[0.0; 4]; 4],
                position: [0.0; 3],
                _pad: 0.0,
            },
            lights: Vec::new(),
            light_config: LightConfig::default(),
            current_time: 0.0,
            context: None,
            compute_pipeline: None,
            texture_manager: None,
        }
    }

    /// Add a light source to the scene
    pub fn add_light(&mut self, light: Light) -> Result<(), String> {
        if self.lights.len() >= 16 {
            return Err("Maximum of 16 lights supported".to_string());
        }
        self.lights.push(light);
        Ok(())
    }

    /// Clear all lights
    pub fn clear_lights(&mut self) {
        self.lights.clear();
    }

    /// Get all lights
    pub fn get_lights(&self) -> &[Light] {
        &self.lights
    }

    /// Set camera data for ray tracing
    pub fn set_camera(&mut self, view_proj: glam::Mat4, position: glam::Vec3) {
        self.camera_data.view_proj = view_proj.to_cols_array_2d();
        self.camera_data.inv_view_proj = view_proj.inverse().to_cols_array_2d();
        self.camera_data.position = position.to_array();
    }

    /// Set previous frame buffers for scene ray tracing
    pub fn set_scene_buffers(&mut self, depth_view: gpu::TextureView, color_view: gpu::TextureView) {
        self.prev_depth_view = Some(depth_view);
        self.prev_color_view = Some(color_view);
    }

    /// Set the light configuration
    pub fn with_light(mut self, light: LightConfig) -> Self {
        self.light_config = light;
        self
    }

    /// Update light configuration at runtime
    pub fn set_light(&mut self, light: LightConfig) {
        self.light_config = light;
    }

    /// Get current light configuration
    pub fn light(&self) -> &LightConfig {
        &self.light_config
    }

    /// Set the texture manager
    pub fn set_texture_manager(&mut self, manager: Arc<TextureManager>) {
        self.texture_manager = Some(manager);
    }

    /// Update the current time (for animated scenes)
    pub fn update_time(&mut self, time: f32) {
        self.current_time = time;
    }

    /// Get the radiance lookup data for binding in shaders
    pub fn get_radiance_lookup_data(&self) -> Option<RadianceLookupData> {
        match (self.cascade_views[0], self.cascade_sampler) {
            (Some(view), Some(sampler)) => {
                Some(RadianceLookupData {
                    lighting: GpuLightConfig::from(&self.light_config),
                    shadow_maps: view,
                    shadow_sampler: sampler,
                })
            }
            _ => None,
        }
    }

    /// Compute radiance cascades
    fn compute_cascades(
        &mut self,
        encoder: &mut gpu::CommandEncoder,
        _context: &FeatureContext,
    ) {
        let compute_pipeline = match &self.compute_pipeline {
            Some(pipeline) => pipeline,
            None => {
                log::warn!("Compute pipeline not initialized");
                return;
            }
        };

        let cascade_sampler = match self.cascade_sampler {
            Some(s) => s,
            None => {
                log::warn!("Cascade sampler not initialized");
                return;
            }
        };

        let linear_sampler = match self.linear_sampler {
            Some(s) => s,
            None => {
                log::warn!("Linear sampler not initialized");
                return;
            }
        };

        // Use previous frame buffers or first cascade as placeholder
        let depth_view = self.prev_depth_view.unwrap_or_else(|| self.cascade_views[0].unwrap());
        let color_view = self.prev_color_view.unwrap_or_else(|| self.cascade_views[0].unwrap());

        // Compute cascades from coarsest to finest
        for cascade_idx in (0..NUM_CASCADES).rev() {
            let cascade_view = match self.cascade_views[cascade_idx] {
                Some(view) => view,
                None => continue,
            };

            // For the finest cascade, use the previous frame's data
            // For coarser cascades, use the next finer cascade as input
            let prev_view = if cascade_idx == NUM_CASCADES - 1 {
                cascade_view
            } else {
                match self.cascade_views[cascade_idx + 1] {
                    Some(view) => view,
                    None => cascade_view,
                }
            };

            let scene_uniforms = SceneUniforms {
                time: self.current_time,
                cascade_index: cascade_idx as u32,
                _pad0: 0.0,
                _pad1: 0.0,
            };

            // Convert lights to GPU format
            let mut gpu_lights = [GpuLight {
                position: [0.0; 3],
                radius: 0.0,
                color: [0.0; 3],
                intensity: 0.0,
            }; 16];
            for (i, light) in self.lights.iter().take(16).enumerate() {
                gpu_lights[i] = GpuLight::from(light);
            }

            let compute_data = RadianceComputeData {
                light_config: GpuLightConfig::from(&self.light_config),
                scene: scene_uniforms,
                camera: self.camera_data,
                lights: gpu_lights,
                num_lights: self.lights.len().min(16) as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
                cascade_texture: cascade_view,
                prev_cascade_texture: prev_view,
                prev_cascade_sampler: cascade_sampler,
                scene_depth: depth_view,
                scene_color: color_view,
                linear_sampler,
            };

            let mut compute_pass = encoder.compute(&format!("radiance_cascade_{}", cascade_idx));
            let mut rc = compute_pass.with(compute_pipeline);
            rc.bind(0, &compute_data);

            // Dispatch compute shader
            // Texture is 1024x2048, workgroup size is 8x8
            let dispatch_x = (1024 + 7) / 8;
            let dispatch_y = (2048 + 7) / 8;
            rc.dispatch([dispatch_x, dispatch_y, 1]);

            drop(rc);
            drop(compute_pass);
        }
    }
}

impl Default for RadianceCascades {
    fn default() -> Self {
        Self::new()
    }
}

impl Feature for RadianceCascades {
    fn name(&self) -> &str {
        "procedural_shadows"
    }

    fn init(&mut self, context: &FeatureContext) {
        log::info!(
            "Initializing Radiance Cascades GI with {} cascade levels",
            NUM_CASCADES
        );

        self.context = Some(context.gpu.clone());

        // Create cascade textures (1024x2048 RGBA16Float)
        for i in 0..NUM_CASCADES {
            let cascade_texture = context.gpu.create_texture(gpu::TextureDesc {
                name: &format!("radiance_cascade_{}", i),
                format: gpu::TextureFormat::Rgba16Float,
                size: gpu::Extent {
                    width: 1024,
                    height: 2048,
                    depth: 1,
                },
                dimension: gpu::TextureDimension::D2,
                array_layer_count: 1,
                mip_level_count: 1,
                usage: gpu::TextureUsage::RESOURCE | gpu::TextureUsage::STORAGE,
                sample_count: 1,
                external: None,
            });

            let cascade_view = context.gpu.create_texture_view(
                cascade_texture,
                gpu::TextureViewDesc {
                    name: &format!("radiance_cascade_{}_view", i),
                    format: gpu::TextureFormat::Rgba16Float,
                    dimension: gpu::ViewDimension::D2,
                    subresources: &Default::default(),
                },
            );

            self.cascade_textures[i] = Some(cascade_texture);
            self.cascade_views[i] = Some(cascade_view);
        }

        // Create sampler for cascade lookup
        let cascade_sampler = context.gpu.create_sampler(gpu::SamplerDesc {
            name: "cascade_sampler",
            address_modes: [gpu::AddressMode::ClampToEdge; 3],
            mag_filter: gpu::FilterMode::Linear,
            min_filter: gpu::FilterMode::Linear,
            mipmap_filter: gpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: Some(100.0),
            compare: None,
            border_color: None,
            anisotropy_clamp: 1,
        });

        self.cascade_sampler = Some(cascade_sampler);

        // Create linear sampler for scene textures
        let linear_sampler = context.gpu.create_sampler(gpu::SamplerDesc {
            name: "linear_sampler",
            address_modes: [gpu::AddressMode::ClampToEdge; 3],
            mag_filter: gpu::FilterMode::Linear,
            min_filter: gpu::FilterMode::Linear,
            mipmap_filter: gpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: Some(100.0),
            compare: None,
            border_color: None,
            anisotropy_clamp: 1,
        });

        self.linear_sampler = Some(linear_sampler);

        // Create compute pipeline for ray tracing
        let compute_shader_source = include_str!("../shaders/radiance_cascade_trace.wgsl");
        let compute_shader = context.gpu.create_shader(gpu::ShaderDesc {
            source: compute_shader_source,
        });

        let compute_layout = <RadianceComputeData as gpu::ShaderData>::layout();

        let compute_pipeline = context.gpu.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "radiance_cascade_compute",
            data_layouts: &[&compute_layout],
            compute: compute_shader.at("main"),
        });

        self.compute_pipeline = Some(compute_pipeline);

        log::info!("Radiance Cascades GI initialized successfully");
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn shader_injections(&self) -> Vec<ShaderInjection> {
        vec![
            // Radiance lookup functions
            ShaderInjection::with_priority(
                ShaderInjectionPoint::FragmentPreamble,
                include_str!("../shaders/radiance_lookup.wgsl"),
                5,
            ),
            // Capture raw material albedo before other lighting
            ShaderInjection::with_priority(
                ShaderInjectionPoint::FragmentMain,
                "    let radiance_albedo = final_color;",
                -5,
            ),
            // Apply radiance cascade GI
            ShaderInjection::with_priority(
                ShaderInjectionPoint::FragmentColorCalculation,
                "    final_color = apply_radiance_cascade(radiance_albedo, input.world_position, input.world_normal);",
                10,
            ),
        ]
    }

    fn pre_render_pass(
        &mut self,
        encoder: &mut gpu::CommandEncoder,
        context: &FeatureContext,
    ) {
        if !self.enabled {
            return;
        }

        // Compute radiance cascades
        self.compute_cascades(encoder, context);
    }

    fn render_shadow_pass(
        &mut self,
        _encoder: &mut gpu::CommandEncoder,
        _context: &FeatureContext,
        _meshes: &[helio_features::MeshData],
    ) {
        // Not used for radiance cascades (we use compute shader instead)
    }

    fn cleanup(&mut self, context: &FeatureContext) {
        log::debug!("Cleaning up radiance cascades");

        if let Some(sampler) = self.cascade_sampler.take() {
            context.gpu.destroy_sampler(sampler);
        }

        if let Some(sampler) = self.linear_sampler.take() {
            context.gpu.destroy_sampler(sampler);
        }

        for view in &mut self.cascade_views {
            if let Some(v) = view.take() {
                context.gpu.destroy_texture_view(v);
            }
        }

        for texture in &mut self.cascade_textures {
            if let Some(t) = texture.take() {
                context.gpu.destroy_texture(t);
            }
        }

        self.compute_pipeline = None;
    }

    fn get_shadow_map_view(&self) -> Option<gpu::TextureView> {
        // Return the finest cascade view for compatibility
        self.cascade_views[0]
    }
}

// Re-export for compatibility with existing code
pub type ProceduralShadows = RadianceCascades;

// For backward compatibility - configure sun from directional light
impl RadianceCascades {
    pub fn with_directional_light(mut self, direction: glam::Vec3) -> Self {
        self.light_config.sun_direction = direction.normalize();
        self
    }

    pub fn with_ambient(mut self, ambient: f32) -> Self {
        self.light_config.ambient_intensity = ambient;
        self
    }

    pub fn set_ambient(&mut self, ambient: f32) {
        self.light_config.ambient_intensity = ambient;
    }

    pub fn ambient(&self) -> f32 {
        self.light_config.ambient_intensity
    }

    pub fn with_light_legacy(self, _config: LightConfig) -> Self {
        self
    }

    pub fn set_light_config(&mut self, _config: LightConfig) {}

    pub fn light_config(&self) -> Option<&LightConfig> {
        Some(&self.light_config)
    }

    /// Get shadow map data for compatibility with render crate
    pub fn get_shadow_map_data(&self) -> Option<ShadowMapData> {
        self.get_radiance_lookup_data()
    }

    /// Get spotlight icon texture (not used in radiance cascades, returns None)
    pub fn spotlight_icon_texture(&self) -> Option<TextureId> {
        None
    }

    /// Generate light billboards for debug visualization (empty for radiance cascades)
    pub fn generate_light_billboards(&self) -> Vec<(glam::Vec3, f32)> {
        Vec::new()
    }
}

// Re-export TextureId for compatibility
pub use helio_core::TextureId;
