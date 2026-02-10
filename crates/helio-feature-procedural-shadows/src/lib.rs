use blade_graphics as gpu;
use helio_features::{Feature, FeatureContext, ShaderInjection, ShaderInjectionPoint};
use std::sync::Arc;

/// Light type for shadow mapping
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LightType {
    /// Directional light (sun) - parallel rays, orthographic projection
    Directional,
    /// Point light - omnidirectional, perspective projection (6 faces for cubemap)
    Point,
    /// Spot light - cone of light, perspective projection
    Spot { inner_angle: f32, outer_angle: f32 },
    /// Rectangular area light - soft shadows, orthographic projection
    Rect { width: f32, height: f32 },
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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowUniforms {
    pub light_view_proj: [[f32; 4]; 4],
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
}

/// Procedural shadow mapping feature.
///
/// Renders scene geometry into a shadow map from the light's perspective,
/// then samples it during main rendering to produce shadows.
///
/// # Performance
/// Shadow map size significantly impacts performance. Use 1024 or 2048
/// for most applications. Higher resolutions provide sharper shadows at
/// the cost of memory and rendering time.
pub struct ProceduralShadows {
    enabled: bool,
    shadow_map: Option<gpu::Texture>,
    shadow_map_view: Option<gpu::TextureView>,
    shadow_sampler: Option<gpu::Sampler>,
    shadow_map_size: u32,
    light_config: LightConfig,
    context: Option<Arc<gpu::Context>>,
    shadow_pipeline: Option<gpu::RenderPipeline>,
}

impl ProceduralShadows {
    /// Create a new procedural shadows feature with default 2048x2048 shadow map.
    pub fn new() -> Self {
        Self {
            enabled: true,
            shadow_map: None,
            shadow_map_view: None,
            shadow_sampler: None,
            shadow_map_size: 2048,
            light_config: LightConfig::default(),
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

    /// Configure the light source.
    pub fn with_light(mut self, config: LightConfig) -> Self {
        self.light_config = config;
        self
    }

    /// Create a directional light (sun).
    pub fn with_directional_light(mut self, direction: glam::Vec3) -> Self {
        self.light_config.light_type = LightType::Directional;
        self.light_config.direction = direction.normalize();
        self
    }

    /// Create a point light.
    pub fn with_point_light(mut self, position: glam::Vec3) -> Self {
        self.light_config.light_type = LightType::Point;
        self.light_config.position = position;
        self
    }

    /// Create a spotlight.
    pub fn with_spot_light(mut self, position: glam::Vec3, direction: glam::Vec3, inner_angle: f32, outer_angle: f32) -> Self {
        self.light_config.light_type = LightType::Spot { inner_angle, outer_angle };
        self.light_config.position = position;
        self.light_config.direction = direction.normalize();
        self
    }

    /// Create a rectangular area light.
    pub fn with_rect_light(mut self, position: glam::Vec3, direction: glam::Vec3, width: f32, height: f32) -> Self {
        self.light_config.light_type = LightType::Rect { width, height };
        self.light_config.position = position;
        self.light_config.direction = direction.normalize();
        self
    }

    /// Update the light configuration at runtime.
    pub fn set_light_config(&mut self, config: LightConfig) {
        self.light_config = config;
    }

    /// Get the current light configuration.
    pub fn light_config(&self) -> &LightConfig {
        &self.light_config
    }

    /// Get the light's view-projection matrix for shadow rendering.
    pub fn get_light_view_proj(&self) -> glam::Mat4 {
        match self.light_config.light_type {
            LightType::Directional => {
                // Directional light: position light far away along direction
                let light_pos = -self.light_config.direction * 20.0;
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
            LightType::Point => {
                // Point light: use perspective projection
                // For full point light shadows, we'd need a cubemap (6 faces)
                // For now, use single face pointing down
                let view = glam::Mat4::look_at_rh(
                    self.light_config.position,
                    self.light_config.position + self.light_config.direction,
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
            LightType::Spot { inner_angle: _, outer_angle } => {
                // Spotlight: perspective projection with cone angle
                let view = glam::Mat4::look_at_rh(
                    self.light_config.position,
                    self.light_config.position + self.light_config.direction,
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
            LightType::Rect { width, height } => {
                // Rectangular area light: orthographic projection
                let view = glam::Mat4::look_at_rh(
                    self.light_config.position,
                    self.light_config.position + self.light_config.direction,
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

    /// Get the shadow map data for binding in the main render pass.
    ///
    /// Returns shader data containing the shadow map texture view and comparison sampler.
    /// This should be bound at group 2 during the main render pass.
    pub fn get_shadow_map_data(&self) -> Option<ShadowMapData> {
        match (self.shadow_map_view, self.shadow_sampler) {
            (Some(view), Some(sampler)) => Some(ShadowMapData {
                shadow_map: view,
                shadow_sampler: sampler,
            }),
            _ => None,
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
            "Initializing procedural shadows with {}x{} shadow map",
            self.shadow_map_size,
            self.shadow_map_size
        );
        
        self.context = Some(context.gpu.clone());

        // Create shadow map texture
        let shadow_map = context.gpu.create_texture(gpu::TextureDesc {
            name: "shadow_map",
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
                name: "shadow_map_view",
                format: gpu::TextureFormat::Depth32Float,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );

        self.shadow_map = Some(shadow_map);
        self.shadow_map_view = Some(shadow_map_view);

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
        light_view_proj: [[f32; 4]; 4],
    ) {
        let shadow_pipeline = match &self.shadow_pipeline {
            Some(pipeline) => pipeline,
            None => {
                log::warn!("Shadow pipeline not initialized");
                return;
            }
        };

        let shadow_view = match self.shadow_map_view {
            Some(view) => view,
            None => {
                log::warn!("Shadow map view not initialized");
                return;
            }
        };

        let mut pass = encoder.render(
            "shadow_depth_pass",
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
            shadow_uniforms: ShadowUniforms { light_view_proj },
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
    
    fn cleanup(&mut self, context: &FeatureContext) {
        log::debug!("Cleaning up procedural shadows");

        if let Some(sampler) = self.shadow_sampler.take() {
            context.gpu.destroy_sampler(sampler);
        }

        if let Some(view) = self.shadow_map_view.take() {
            context.gpu.destroy_texture_view(view);
        }

        if let Some(texture) = self.shadow_map.take() {
            context.gpu.destroy_texture(texture);
        }

        // Pipelines are automatically cleaned up by blade
        self.shadow_pipeline = None;
    }

    fn get_shadow_map_view(&self) -> Option<gpu::TextureView> {
        self.shadow_map_view
    }
}
