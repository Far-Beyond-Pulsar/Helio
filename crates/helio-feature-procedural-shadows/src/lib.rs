use blade_graphics as gpu;
use helio_features::{Feature, FeatureContext, ShaderInjection, ShaderInjectionPoint};
use std::sync::Arc;

/// Maximum number of overlapping lights that can affect a single fragment
const MAX_OVERLAPPING_LIGHTS: usize = 8;

/// Light type for shadow mapping
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LightType {
    /// Directional light (sun) - parallel rays, orthographic projection
    /// Only one directional light can exist at a time
    Directional,
    /// Point light - omnidirectional, perspective projection (6 faces for cubemap)
    Point { radius: f32 },
    /// Spot light - cone of light, perspective projection
    Spot { inner_angle: f32, outer_angle: f32, radius: f32 },
    /// Rectangular area light - soft shadows, orthographic projection
    Rect { width: f32, height: f32, radius: f32 },
}

impl LightType {
    /// Get the attenuation radius for this light type
    pub fn attenuation_radius(&self) -> Option<f32> {
        match self {
            LightType::Directional => None,
            LightType::Point { radius } => Some(*radius),
            LightType::Spot { radius, .. } => Some(*radius),
            LightType::Rect { radius, .. } => Some(*radius),
        }
    }

    /// Get the light type as a u32 for shader use
    fn as_u32(&self) -> u32 {
        match self {
            LightType::Directional => 0,
            LightType::Point { .. } => 1,
            LightType::Spot { .. } => 2,
            LightType::Rect { .. } => 3,
        }
    }
}

/// Configuration for a light source
#[derive(Debug, Clone, Copy)]
pub struct LightConfig {
    pub light_type: LightType,
    pub position: glam::Vec3,
    pub direction: glam::Vec3,
    pub intensity: f32,
    pub color: glam::Vec3,
}

impl Default for LightConfig {
    fn default() -> Self {
        Self {
            light_type: LightType::Directional,
            position: glam::Vec3::new(10.0, 15.0, 10.0),
            direction: glam::Vec3::new(0.5, -1.0, 0.3).normalize(),
            intensity: 1.0,
            color: glam::Vec3::ONE,
        }
    }
}

/// GPU representation of a light source
/// Aligned for WGSL uniform buffer (std140 layout)
#[repr(C, packed(2))]
#[derive(Clone, Copy)]
pub struct GpuLight {
    pub light_type: u32,
    pub intensity: f32,
    pub radius: f32,
    pub _padding1: f32,
    
    pub position: [f32; 3],
    pub inner_angle: f32,
    
    pub direction: [f32; 3],
    pub outer_angle: f32,
    
    pub color: [f32; 3],
    pub width: f32,
    
    pub light_view_proj: [[f32; 4]; 4],
    
    pub height: f32,
    pub _padding2: [f32; 3],
    
    // Array stride padding: 18 bytes to reach 162 total
    pub _padding3: [u8; 18],
}

unsafe impl bytemuck::Pod for GpuLight {}
unsafe impl bytemuck::Zeroable for GpuLight {}

impl Default for GpuLight {
    fn default() -> Self {
        Self {
            light_type: 0,
            intensity: 0.0,
            radius: 0.0,
            _padding1: 0.0,
            position: [0.0; 3],
            inner_angle: 0.0,
            direction: [0.0; 3],
            outer_angle: 0.0,
            color: [0.0; 3],
            width: 0.0,
            light_view_proj: [[0.0; 4]; 4],
            height: 0.0,
            _padding2: [0.0; 3],
            _padding3: [0; 18],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowUniforms {
    pub light_count: u32,
    pub _padding: [u32; 3],
    pub lights: [GpuLight; MAX_OVERLAPPING_LIGHTS],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ObjectUniforms {
    pub model: [[f32; 4]; 4],
}

#[derive(blade_macros::ShaderData)]
struct ShadowData {
    shadow_uniforms: ShadowUniforms,
}

#[derive(blade_macros::ShaderData)]
struct ObjectData {
    object_uniforms: ObjectUniforms,
}

/// Shadow map texture and sampler for binding in shaders
#[derive(blade_macros::ShaderData)]
pub struct ShadowMapData {
    pub shadow_map: gpu::TextureView,
    pub shadow_sampler: gpu::Sampler,
    pub shadow_uniforms: ShadowUniforms,
}

/// Procedural shadow mapping feature.
///
/// Renders scene geometry into shadow maps from multiple light perspectives,
/// then samples them during main rendering to produce shadows.
///
/// Supports up to 8 overlapping lights per fragment:
/// - 1 directional light (sun) maximum
/// - Unlimited point, spot, and rect lights (only closest 8 affect each pixel)
///
/// # Performance
/// Shadow map size significantly impacts performance. Use 1024 or 2048
/// for most applications. Higher resolutions provide sharper shadows at
/// the cost of memory and rendering time.
pub struct ProceduralShadows {
    enabled: bool,
    shadow_maps: Vec<gpu::Texture>,
    shadow_map_views: Vec<gpu::TextureView>,
    shadow_sampler: Option<gpu::Sampler>,
    shadow_map_size: u32,
    lights: Vec<LightConfig>,
    context: Option<Arc<gpu::Context>>,
    shadow_pipeline: Option<gpu::RenderPipeline>,
}

impl ProceduralShadows {
    /// Create a new procedural shadows feature with default 2048x2048 shadow maps.
    pub fn new() -> Self {
        Self {
            enabled: true,
            shadow_maps: Vec::new(),
            shadow_map_views: Vec::new(),
            shadow_sampler: None,
            shadow_map_size: 2048,
            lights: vec![LightConfig::default()],
            context: None,
            shadow_pipeline: None,
        }
    }

    /// Set the shadow map resolution (must be called before init).
    ///
    /// # Example
    /// ```ignore
    /// let shadows = ProceduralShadows::new().with_size(1024);
    /// ```
    pub fn with_size(mut self, size: u32) -> Self {
        self.shadow_map_size = size;
        self
    }

    /// Add a light source. Can call multiple times to add multiple lights.
    /// Note: Only one directional light is allowed. Adding a second will replace the first.
    pub fn add_light(&mut self, config: LightConfig) -> Result<(), &'static str> {
        // Check if adding a directional light
        if matches!(config.light_type, LightType::Directional) {
            // Check if we already have a directional light
            if let Some(idx) = self.lights.iter().position(|l| matches!(l.light_type, LightType::Directional)) {
                log::warn!("Replacing existing directional light");
                self.lights[idx] = config;
                return Ok(());
            }
        }
        
        self.lights.push(config);
        Ok(())
    }

    /// Remove all lights.
    pub fn clear_lights(&mut self) {
        self.lights.clear();
    }

    /// Get all light configurations.
    pub fn lights(&self) -> &[LightConfig] {
        &self.lights
    }

    /// Configure the light source (for single light setup).
    pub fn with_light(mut self, config: LightConfig) -> Self {
        self.lights = vec![config];
        self
    }

    /// Create a directional light (sun).
    pub fn with_directional_light(mut self, direction: glam::Vec3, color: glam::Vec3) -> Self {
        self.lights = vec![LightConfig {
            light_type: LightType::Directional,
            direction: direction.normalize(),
            color,
            ..Default::default()
        }];
        self
    }

    /// Create a point light.
    pub fn with_point_light(mut self, position: glam::Vec3, color: glam::Vec3, radius: f32) -> Self {
        self.lights = vec![LightConfig {
            light_type: LightType::Point { radius },
            position,
            color,
            ..Default::default()
        }];
        self
    }

    /// Create a spotlight.
    pub fn with_spot_light(mut self, position: glam::Vec3, direction: glam::Vec3, color: glam::Vec3, inner_angle: f32, outer_angle: f32, radius: f32) -> Self {
        self.lights = vec![LightConfig {
            light_type: LightType::Spot { inner_angle, outer_angle, radius },
            position,
            direction: direction.normalize(),
            color,
            ..Default::default()
        }];
        self
    }

    /// Create a rectangular area light.
    pub fn with_rect_light(mut self, position: glam::Vec3, direction: glam::Vec3, color: glam::Vec3, width: f32, height: f32, radius: f32) -> Self {
        self.lights = vec![LightConfig {
            light_type: LightType::Rect { width, height, radius },
            position,
            direction: direction.normalize(),
            color,
            ..Default::default()
        }];
        self
    }

    /// Get the light's view-projection matrix for shadow rendering.
    pub fn get_light_view_proj(&self, light: &LightConfig) -> glam::Mat4 {
        match light.light_type {
            LightType::Directional => {
                // Directional light: position light far away along direction
                let light_pos = -light.direction * 20.0;
                let view = glam::Mat4::look_at_rh(
                    light_pos,
                    glam::Vec3::ZERO,
                    glam::Vec3::Y,
                );

                // Orthographic projection for parallel light rays
                let projection = glam::Mat4::orthographic_rh(
                    -8.0, 8.0,  // left, right
                    -8.0, 8.0,  // bottom, top
                    0.1, 40.0,  // near, far
                );

                projection * view
            }
            LightType::Point { .. } => {
                // Point light: use perspective projection
                // For full point light shadows, we'd need a cubemap (6 faces)
                // For now, use single face pointing down
                let view = glam::Mat4::look_at_rh(
                    light.position,
                    light.position + light.direction,
                    glam::Vec3::Y,
                );

                // Perspective projection with 90 degree FOV for point light
                let projection = glam::Mat4::perspective_rh(
                    90.0_f32.to_radians(),
                    1.0,  // square aspect ratio
                    0.1,
                    50.0,
                );

                projection * view
            }
            LightType::Spot { inner_angle: _, outer_angle, .. } => {
                // Spotlight: perspective projection with cone angle
                let view = glam::Mat4::look_at_rh(
                    light.position,
                    light.position + light.direction,
                    glam::Vec3::Y,
                );

                // Perspective projection matching spotlight cone
                let fov = outer_angle * 2.0;  // outer_angle is half-angle
                let projection = glam::Mat4::perspective_rh(
                    fov,
                    1.0,  // square aspect ratio
                    0.1,
                    50.0,
                );

                projection * view
            }
            LightType::Rect { width, height, .. } => {
                // Rectangular area light: orthographic projection
                let view = glam::Mat4::look_at_rh(
                    light.position,
                    light.position + light.direction,
                    glam::Vec3::Y,
                );

                // Orthographic projection matching rect dimensions
                let half_width = width / 2.0;
                let half_height = height / 2.0;
                let projection = glam::Mat4::orthographic_rh(
                    -half_width, half_width,
                    -half_height, half_height,
                    0.1, 40.0,
                );

                projection * view
            }
        }
    }

    /// Convert a light config to GPU representation
    fn light_to_gpu(&self, light: &LightConfig) -> GpuLight {
        let view_proj = self.get_light_view_proj(light);
        
        let (inner_angle, outer_angle) = match light.light_type {
            LightType::Spot { inner_angle, outer_angle, .. } => (inner_angle, outer_angle),
            _ => (0.0, 0.0),
        };
        
        let (width, height) = match light.light_type {
            LightType::Rect { width, height, .. } => (width, height),
            _ => (0.0, 0.0),
        };
        
        let radius = light.light_type.attenuation_radius().unwrap_or(0.0);
        
        GpuLight {
            light_type: light.light_type.as_u32(),
            intensity: light.intensity,
            radius,
            _padding1: 0.0,
            position: light.position.to_array(),
            inner_angle,
            direction: light.direction.to_array(),
            outer_angle,
            color: light.color.to_array(),
            width,
            light_view_proj: view_proj.to_cols_array_2d(),
            height,
            _padding2: [0.0; 3],
            _padding3: [0; 18],
        }
    }

    /// Get the shadow map data for binding in the main render pass.
    ///
    /// Returns shader data containing shadow map texture views and comparison sampler.
    /// This should be bound at group 2 during the main render pass.
    pub fn get_shadow_map_data(&self) -> Option<ShadowMapData> {
        if self.shadow_map_views.is_empty() || self.shadow_sampler.is_none() {
            return None;
        }
        
        // Create shadow uniforms for all active lights
        let shadow_uniforms = self.create_shadow_uniforms();
        
        // For now, return first shadow map
        // TODO: Support texture array for multiple shadow maps
        Some(ShadowMapData {
            shadow_map: self.shadow_map_views[0],
            shadow_sampler: self.shadow_sampler.unwrap(),
            shadow_uniforms,
        })
    }

    /// Create shadow uniforms for all lights
    pub fn create_shadow_uniforms(&self) -> ShadowUniforms {
        let mut gpu_lights = [GpuLight::default(); MAX_OVERLAPPING_LIGHTS];
        
        // Convert active lights to GPU format
        for (i, light) in self.lights.iter().take(MAX_OVERLAPPING_LIGHTS).enumerate() {
            gpu_lights[i] = self.light_to_gpu(light);
        }
        
        ShadowUniforms {
            light_count: self.lights.len().min(MAX_OVERLAPPING_LIGHTS) as u32,
            _padding: [0; 3],
            lights: gpu_lights,
        }
    }
}

impl Default for ProceduralShadows {
    fn default() -> Self {
        Self::new()
    }
}

impl Feature for ProceduralShadows {
    fn name(&self) -> &str {
        "procedural_shadows"
    }

    fn init(&mut self, context: &FeatureContext) {
        log::info!(
            "Initializing procedural shadows with {}x{} shadow maps for {} lights",
            self.shadow_map_size,
            self.shadow_map_size,
            self.lights.len()
        );
        
        self.context = Some(context.gpu.clone());

        // Create shadow map for each light
        for (i, _light) in self.lights.iter().enumerate() {
            let shadow_map = context.gpu.create_texture(gpu::TextureDesc {
                name: &format!("shadow_map_{}", i),
                format: gpu::TextureFormat::Depth32Float,
                size: gpu::Extent {
                    width: self.shadow_map_size,
                    height: self.shadow_map_size,
                    depth: 1,
                },
                dimension: gpu::TextureDimension::D2,
                array_layer_count: 1,
                mip_level_count: 1,
                usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
                sample_count: 1,
                external: None,
            });

            let shadow_map_view = context.gpu.create_texture_view(
                shadow_map,
                gpu::TextureViewDesc {
                    name: &format!("shadow_map_view_{}", i),
                    format: gpu::TextureFormat::Depth32Float,
                    dimension: gpu::ViewDimension::D2,
                    subresources: &Default::default(),
                },
            );

            self.shadow_maps.push(shadow_map);
            self.shadow_map_views.push(shadow_map_view);
        }

        // Create comparison sampler for shadow map sampling
        let shadow_sampler = context.gpu.create_sampler(gpu::SamplerDesc {
            name: "shadow_sampler",
            address_modes: [gpu::AddressMode::ClampToEdge; 3],
            mag_filter: gpu::FilterMode::Linear,
            min_filter: gpu::FilterMode::Linear,
            mipmap_filter: gpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: Some(100.0),
            compare: Some(gpu::CompareFunction::LessEqual),
            border_color: None,
            anisotropy_clamp: 1,
        });

        self.shadow_sampler = Some(shadow_sampler);

        // Create shadow rendering pipeline (depth-only pass)
        let shadow_shader_source = include_str!("../shaders/shadow_pass.wgsl");
        let shadow_shader = context.gpu.create_shader(gpu::ShaderDesc {
            source: shadow_shader_source,
        });

        let shadow_layout = <ShadowData as gpu::ShaderData>::layout();
        let object_layout = <ObjectData as gpu::ShaderData>::layout();

        let shadow_pipeline = context.gpu.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "shadow_pass",
            data_layouts: &[&shadow_layout, &object_layout],
            vertex: shadow_shader.at("vs_main"),
            vertex_fetches: &[gpu::VertexFetchState {
                layout: &<helio_core::PackedVertex as gpu::Vertex>::layout(),
                instanced: false,
            }],
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleList,
                front_face: gpu::FrontFace::Ccw,
                // Use front-face culling for shadow pass to reduce self-shadowing
                cull_mode: Some(gpu::Face::Front),
                ..Default::default()
            },
            depth_stencil: Some(gpu::DepthStencilState {
                format: gpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: gpu::CompareFunction::Less,
                stencil: Default::default(),
                // Hardware depth bias to prevent shadow acne
                bias: gpu::DepthBiasState {
                    constant: 2,      // Constant bias
                    slope_scale: 2.0, // Slope-scaled bias
                    clamp: 0.0,
                },
            }),
            fragment: None, // Depth-only pass
            color_targets: &[],
            multisample_state: gpu::MultisampleState::default(),
        });

        self.shadow_pipeline = Some(shadow_pipeline);
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn shader_injections(&self) -> Vec<ShaderInjection> {
        vec![
            // Shadow sampling functions
            ShaderInjection::with_priority(
                ShaderInjectionPoint::FragmentPreamble,
                include_str!("../shaders/shadow_functions.wgsl"),
                5,
            ),
            // Apply shadows after lighting (higher priority runs later)
            ShaderInjection::with_priority(
                ShaderInjectionPoint::FragmentColorCalculation,
                "    final_color = apply_shadow(final_color, input.world_position, input.world_normal);",
                10,
            ),
        ]
    }

    fn render_shadow_pass(
        &mut self,
        encoder: &mut gpu::CommandEncoder,
        _context: &FeatureContext,
        meshes: &[helio_features::MeshData],
        _light_view_proj: [[f32; 4]; 4],
    ) {
        let shadow_pipeline = match &self.shadow_pipeline {
            Some(pipeline) => pipeline,
            None => {
                log::warn!("Shadow pipeline not initialized");
                return;
            }
        };

        // Render shadow pass for each light
        for (light_idx, light) in self.lights.iter().enumerate() {
            if light_idx >= self.shadow_map_views.len() {
                log::warn!("Not enough shadow map views for light {}", light_idx);
                break;
            }

            let shadow_view = self.shadow_map_views[light_idx];

            let mut pass = encoder.render(
                &format!("shadow_depth_pass_{}", light_idx),
                gpu::RenderTargetSet {
                    colors: &[],
                    depth_stencil: Some(gpu::RenderTarget {
                        view: shadow_view,
                        init_op: gpu::InitOp::Clear(gpu::TextureColor::White),
                        finish_op: gpu::FinishOp::Store,
                    }),
                },
            );

            let shadow_data = ShadowData {
                shadow_uniforms: ShadowUniforms {
                    light_count: 1,  // Only one light per pass
                    _padding: [0; 3],
                    lights: {
                        let mut lights = [GpuLight::default(); MAX_OVERLAPPING_LIGHTS];
                        lights[0] = self.light_to_gpu(light);
                        lights
                    },
                },
            };

            let mut rc = pass.with(shadow_pipeline);
            rc.bind(0, &shadow_data);

            for mesh in meshes {
                let object_data = ObjectData {
                    object_uniforms: ObjectUniforms {
                        model: mesh.transform,
                    },
                };
                rc.bind(1, &object_data);
                rc.bind_vertex(0, mesh.vertex_buffer);
                rc.draw_indexed(mesh.index_buffer, gpu::IndexType::U32, mesh.index_count, 0, 0, 1);
            }
        }
    }
    
    fn cleanup(&mut self, context: &FeatureContext) {
        log::debug!("Cleaning up procedural shadows");

        if let Some(sampler) = self.shadow_sampler.take() {
            context.gpu.destroy_sampler(sampler);
        }

        for view in self.shadow_map_views.drain(..) {
            context.gpu.destroy_texture_view(view);
        }

        for texture in self.shadow_maps.drain(..) {
            context.gpu.destroy_texture(texture);
        }

        // Pipelines are automatically cleaned up by blade
        self.shadow_pipeline = None;
    }

    fn get_shadow_map_view(&self) -> Option<gpu::TextureView> {
        self.shadow_map_views.first().copied()
    }
}
