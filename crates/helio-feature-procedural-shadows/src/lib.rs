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
    /// Attenuation radius - distance where light influence reaches zero
    /// Only applies to Point, Spot, and Rect lights (not Directional)
    pub attenuation_radius: f32,
    /// Attenuation falloff exponent - controls how quickly light fades
    /// Typical values: 1.0 (linear), 2.0 (quadratic/physically accurate), 4.0 (sharp falloff)
    pub attenuation_falloff: f32,
}

impl Default for LightConfig {
    fn default() -> Self {
        Self {
            light_type: LightType::Directional,
            position: glam::Vec3::new(10.0, 15.0, 10.0),
            direction: glam::Vec3::new(0.5, -1.0, 0.3).normalize(),
            intensity: 1.0,
            color: glam::Vec3::ONE,
            attenuation_radius: 10.0,
            attenuation_falloff: 2.0,
        }
    }
}

/// Maximum number of shadow-casting lights that can overlap
pub const MAX_SHADOW_LIGHTS: usize = 8;

/// GPU representation of a single light for shaders
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLight {
    /// Light view-projection matrix for shadow mapping
    pub view_proj: [[f32; 4]; 4],
    /// Light position in world space (w component is light type)
    /// type: 0.0 = Directional, 1.0 = Point, 2.0 = Spot, 3.0 = Rect
    pub position_and_type: [f32; 4],
    /// Light direction (normalized) and attenuation radius
    pub direction_and_radius: [f32; 4],
    /// Light color (RGB) and intensity
    pub color_and_intensity: [f32; 4],
    /// Spot light angles (inner, outer) and attenuation falloff, plus shadow map layer index
    pub params: [f32; 4],
}

impl GpuLight {
    fn from_config(config: &LightConfig, shadow_layer: u32) -> Self {
        let light_type = match config.light_type {
            LightType::Directional => 0.0,
            LightType::Point => 1.0,
            LightType::Spot { .. } => 2.0,
            LightType::Rect { .. } => 3.0,
        };

        let (inner_angle, outer_angle) = match config.light_type {
            LightType::Spot { inner_angle, outer_angle } => (inner_angle, outer_angle),
            _ => (0.0, 0.0),
        };

        Self {
            view_proj: [[0.0; 4]; 4], // Will be filled by caller
            position_and_type: [
                config.position.x,
                config.position.y,
                config.position.z,
                light_type,
            ],
            direction_and_radius: [
                config.direction.x,
                config.direction.y,
                config.direction.z,
                config.attenuation_radius,
            ],
            color_and_intensity: [
                config.color.x,
                config.color.y,
                config.color.z,
                config.intensity,
            ],
            params: [
                inner_angle,
                outer_angle,
                config.attenuation_falloff,
                shadow_layer as f32,
            ],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowUniforms {
    pub light_view_proj: [[f32; 4]; 4],
}

/// Uniforms containing all light data for the scene
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightingUniforms {
    /// Number of active lights (stored in .x, rest unused)
    pub light_count: [f32; 4],
    /// Array of all lights
    pub lights: [GpuLight; MAX_SHADOW_LIGHTS],
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
    pub shadow_maps: gpu::TextureView,  // Texture array with MAX_SHADOW_LIGHTS layers
    pub shadow_sampler: gpu::Sampler,
    pub lighting: LightingUniforms,
}

/// Procedural shadow mapping feature.
///
/// Renders scene geometry into shadow maps from multiple light perspectives,
/// then samples them during main rendering to produce shadows. Supports up to
/// 8 overlapping shadow-casting lights plus one directional skylight.
///
/// # Light Types
/// - **Directional**: Sun/skylight - parallel rays, no attenuation
/// - **Point**: Omnidirectional light with radius-based attenuation
/// - **Spot**: Cone of light with angle and radius-based attenuation
/// - **Rect**: Rectangular area light with radius-based attenuation
///
/// # Performance
/// Shadow map size significantly impacts performance. Use 1024 or 2048
/// for most applications. With 8 lights, memory usage = size² × 8 × 4 bytes.
pub struct ProceduralShadows {
    enabled: bool,
    shadow_map: Option<gpu::Texture>,
    shadow_map_view: Option<gpu::TextureView>,
    shadow_sampler: Option<gpu::Sampler>,
    shadow_map_size: u32,
    lights: Vec<LightConfig>,
    context: Option<Arc<gpu::Context>>,
    shadow_pipeline: Option<gpu::RenderPipeline>,
}

impl ProceduralShadows {
    /// Create a new procedural shadows feature with default 1024x1024 shadow map.
    ///
    /// Each light slot uses 6 texture array layers (one per cube face), so total memory
    /// is `size² × MAX_SHADOW_LIGHTS × 6 × 4 bytes`. At 1024×1024 with 8 lights: ~200 MB.
    pub fn new() -> Self {
        Self {
            enabled: true,
            shadow_map: None,
            shadow_map_view: None,
            shadow_sampler: None,
            shadow_map_size: 1024,
            lights: Vec::new(),
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

    /// Add a light source to the scene.
    ///
    /// # Panics
    /// Panics if more than MAX_SHADOW_LIGHTS (8) non-directional lights are added.
    pub fn add_light(&mut self, config: LightConfig) {
        if config.light_type != LightType::Directional {
            let non_directional_count = self.lights.iter()
                .filter(|l| l.light_type != LightType::Directional)
                .count();
            
            if non_directional_count >= MAX_SHADOW_LIGHTS {
                panic!("Maximum of {} shadow-casting lights exceeded", MAX_SHADOW_LIGHTS);
            }
        }
        self.lights.push(config);
    }

    /// Clear all lights from the scene.
    pub fn clear_lights(&mut self) {
        self.lights.clear();
    }

    /// Get all current lights.
    pub fn lights(&self) -> &[LightConfig] {
        &self.lights
    }

    /// Get mutable access to all lights.
    pub fn lights_mut(&mut self) -> &mut Vec<LightConfig> {
        &mut self.lights
    }

    /// Configure the light source (legacy single-light API).
    pub fn with_light(mut self, config: LightConfig) -> Self {
        self.lights.clear();
        self.lights.push(config);
        self
    }

    /// Create a directional light (sun).
    pub fn with_directional_light(mut self, direction: glam::Vec3) -> Self {
        let mut config = LightConfig::default();
        config.light_type = LightType::Directional;
        config.direction = direction.normalize();
        self.lights.clear();
        self.lights.push(config);
        self
    }

    /// Create a point light.
    pub fn with_point_light(mut self, position: glam::Vec3, radius: f32) -> Self {
        let mut config = LightConfig::default();
        config.light_type = LightType::Point;
        config.position = position;
        config.attenuation_radius = radius;
        self.lights.clear();
        self.lights.push(config);
        self
    }

    /// Create a spotlight.
    pub fn with_spot_light(mut self, position: glam::Vec3, direction: glam::Vec3, inner_angle: f32, outer_angle: f32, radius: f32) -> Self {
        let mut config = LightConfig::default();
        config.light_type = LightType::Spot { inner_angle, outer_angle };
        config.position = position;
        config.direction = direction.normalize();
        config.attenuation_radius = radius;
        self.lights.clear();
        self.lights.push(config);
        self
    }

    /// Create a rectangular area light.
    pub fn with_rect_light(mut self, position: glam::Vec3, direction: glam::Vec3, width: f32, height: f32, radius: f32) -> Self {
        let mut config = LightConfig::default();
        config.light_type = LightType::Rect { width, height };
        config.position = position;
        config.direction = direction.normalize();
        config.attenuation_radius = radius;
        self.lights.clear();
        self.lights.push(config);
        self
    }

    /// Update the light configuration at runtime (legacy single-light API).
    pub fn set_light_config(&mut self, config: LightConfig) {
        self.lights.clear();
        self.lights.push(config);
    }

    /// Get the current light configuration (legacy single-light API).
    pub fn light_config(&self) -> Option<&LightConfig> {
        self.lights.first()
    }

    /// Cube face directions for point light shadow rendering: (forward, up).
    ///
    /// Matches the face ordering used by the WGSL `select_cube_face` / `get_cube_face_view_proj`
    /// functions. Face indices: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z.
    const CUBE_FACE_DIRS: [(glam::Vec3, glam::Vec3); 6] = [
        (glam::Vec3::X,     glam::Vec3::NEG_Y),  // +X
        (glam::Vec3::NEG_X, glam::Vec3::NEG_Y),  // -X
        (glam::Vec3::Y,     glam::Vec3::Z),       // +Y
        (glam::Vec3::NEG_Y, glam::Vec3::NEG_Z),  // -Y
        (glam::Vec3::Z,     glam::Vec3::NEG_Y),  // +Z
        (glam::Vec3::NEG_Z, glam::Vec3::NEG_Y),  // -Z
    ];

    /// Returns view-projection matrices for rendering into the shadow map.
    ///
    /// For point lights this yields 6 matrices (one per cube face). All other
    /// light types yield exactly 1. The returned matrices must be rendered into
    /// consecutive shadow map layers starting at `light_index * 6`.
    fn get_shadow_render_matrices(config: &LightConfig) -> Vec<glam::Mat4> {
        match config.light_type {
            LightType::Directional => {
                let light_pos = -config.direction * 20.0;
                let view = glam::Mat4::look_at_rh(light_pos, glam::Vec3::ZERO, glam::Vec3::Y);
                let projection = glam::Mat4::orthographic_rh(-8.0, 8.0, -8.0, 8.0, 0.1, 40.0);
                vec![projection * view]
            }
            LightType::Point => {
                // Proper cubemap: 6 faces with 90° FOV each for full omnidirectional coverage.
                let projection = glam::Mat4::perspective_rh(
                    90.0_f32.to_radians(),
                    1.0,
                    0.1,
                    config.attenuation_radius,
                );
                Self::CUBE_FACE_DIRS.iter().map(|(forward, up)| {
                    let view = glam::Mat4::look_at_rh(
                        config.position,
                        config.position + *forward,
                        *up,
                    );
                    projection * view
                }).collect()
            }
            LightType::Spot { inner_angle: _, outer_angle } => {
                let view = glam::Mat4::look_at_rh(
                    config.position,
                    config.position + config.direction,
                    glam::Vec3::Y,
                );
                let fov = outer_angle * 2.0;
                let projection = glam::Mat4::perspective_rh(fov, 1.0, 0.1, config.attenuation_radius);
                vec![projection * view]
            }
            LightType::Rect { width, height } => {
                let view = glam::Mat4::look_at_rh(
                    config.position,
                    config.position + config.direction,
                    glam::Vec3::Y,
                );
                let (hw, hh) = (width / 2.0, height / 2.0);
                let projection = glam::Mat4::orthographic_rh(-hw, hw, -hh, hh, 0.1, config.attenuation_radius);
                vec![projection * view]
            }
        }
    }

    /// Returns the view-projection matrix stored in GpuLight for shader use.
    ///
    /// For point lights the shader reconstructs per-face matrices dynamically, so this
    /// stores the +X face matrix as a representative value. For all other light types
    /// this is the single shadow map matrix used during sampling.
    fn get_light_view_proj(config: &LightConfig) -> glam::Mat4 {
        match config.light_type {
            LightType::Directional => {
                let light_pos = -config.direction * 20.0;
                let view = glam::Mat4::look_at_rh(light_pos, glam::Vec3::ZERO, glam::Vec3::Y);
                let projection = glam::Mat4::orthographic_rh(-8.0, 8.0, -8.0, 8.0, 0.1, 40.0);
                projection * view
            }
            LightType::Point => {
                // Shader reconstructs all 6 face matrices dynamically from the light position.
                // Store face 0 (+X) here so the field is not left as zeros.
                let (forward, up) = Self::CUBE_FACE_DIRS[0];
                let view = glam::Mat4::look_at_rh(
                    config.position,
                    config.position + forward,
                    up,
                );
                let projection = glam::Mat4::perspective_rh(
                    90.0_f32.to_radians(),
                    1.0,
                    0.1,
                    config.attenuation_radius,
                );
                projection * view
            }
            LightType::Spot { inner_angle: _, outer_angle } => {
                let view = glam::Mat4::look_at_rh(
                    config.position,
                    config.position + config.direction,
                    glam::Vec3::Y,
                );
                let fov = outer_angle * 2.0;
                let projection = glam::Mat4::perspective_rh(fov, 1.0, 0.1, config.attenuation_radius);
                projection * view
            }
            LightType::Rect { width, height } => {
                let view = glam::Mat4::look_at_rh(
                    config.position,
                    config.position + config.direction,
                    glam::Vec3::Y,
                );
                let (hw, hh) = (width / 2.0, height / 2.0);
                let projection = glam::Mat4::orthographic_rh(-hw, hw, -hh, hh, 0.1, config.attenuation_radius);
                projection * view
            }
        }
    }

    /// Get the shadow map data for binding in the main render pass.
    ///
    /// Returns shader data containing the shadow map texture array, comparison sampler,
    /// and all light data. This should be bound at group 2 during the main render pass.
    pub fn get_shadow_map_data(&self) -> Option<ShadowMapData> {
        match (self.shadow_map_view, self.shadow_sampler) {
            (Some(view), Some(sampler)) => {
                // Build lighting uniforms
                let mut gpu_lights = [GpuLight {
                    view_proj: [[0.0; 4]; 4],
                    position_and_type: [0.0; 4],
                    direction_and_radius: [0.0; 4],
                    color_and_intensity: [0.0; 4],
                    params: [0.0; 4],
                }; MAX_SHADOW_LIGHTS];

                let mut light_count = 0;
                for (i, config) in self.lights.iter().enumerate() {
                    if i >= MAX_SHADOW_LIGHTS {
                        break;
                    }

                    // Each light slot occupies 6 consecutive layers (one per cube face).
                    // Non-point lights only use the first of those 6 layers.
                    let base_layer = (i * 6) as u32;
                    let view_proj = Self::get_light_view_proj(config);
                    let mut gpu_light = GpuLight::from_config(config, base_layer);
                    gpu_light.view_proj = view_proj.to_cols_array_2d();
                    gpu_lights[i] = gpu_light;
                    light_count += 1;
                }

                let lighting = LightingUniforms {
                    light_count: [light_count as f32, 0.0, 0.0, 0.0],
                    lights: gpu_lights,
                };

                Some(ShadowMapData {
                    shadow_maps: view,
                    shadow_sampler: sampler,
                    lighting,
                })
            }
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
            "Initializing procedural shadows with {}x{} shadow map array ({} layers, 6 per light)",
            self.shadow_map_size,
            self.shadow_map_size,
            MAX_SHADOW_LIGHTS * 6
        );
        
        self.context = Some(context.gpu.clone());

        // Create shadow map texture array: 6 layers per light (one per cube face).
        // This supports both point lights (all 6 faces used) and other light types
        // (only face 0 used, but all 6 layers are allocated for consistent indexing).
        let shadow_map = context.gpu.create_texture(gpu::TextureDesc {
            name: "shadow_map_array",
            format: gpu::TextureFormat::Depth32Float,
            size: gpu::Extent {
                width: self.shadow_map_size,
                height: self.shadow_map_size,
                depth: 1,
            },
            dimension: gpu::TextureDimension::D2,
            array_layer_count: (MAX_SHADOW_LIGHTS * 6) as u32,
            mip_level_count: 1,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
            sample_count: 1,
            external: None,
        });

        let shadow_map_view = context.gpu.create_texture_view(
            shadow_map,
            gpu::TextureViewDesc {
                name: "shadow_map_array_view",
                format: gpu::TextureFormat::Depth32Float,
                dimension: gpu::ViewDimension::D2Array,
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
            // Shadow sampling + ACES/gamma helpers
            ShaderInjection::with_priority(
                ShaderInjectionPoint::FragmentPreamble,
                include_str!("../shaders/shadow_functions.wgsl"),
                5,
            ),
            // Snapshot raw material albedo BEFORE BasicLighting (priority 0) runs.
            // apply_shadow must receive unlit albedo — passing the pre-lit value
            // causes double-multiplication and a washed-out result.
            ShaderInjection::with_priority(
                ShaderInjectionPoint::FragmentMain,
                "    let shadow_albedo = final_color;",
                -5,
            ),
            // Recompute full lighting from albedo, overwriting BasicLighting's output.
            ShaderInjection::with_priority(
                ShaderInjectionPoint::FragmentColorCalculation,
                "    final_color = apply_shadow(shadow_albedo, input.world_position, input.world_normal);",
                10,
            ),
        ]
    }

    fn render_shadow_pass(
        &mut self,
        encoder: &mut gpu::CommandEncoder,
        context: &FeatureContext,
        meshes: &[helio_features::MeshData],
    ) {
        let shadow_pipeline = match &self.shadow_pipeline {
            Some(pipeline) => pipeline,
            None => {
                log::warn!("Shadow pipeline not initialized");
                return;
            }
        };

        let shadow_texture = match self.shadow_map {
            Some(texture) => texture,
            None => {
                log::warn!("Shadow map not initialized");
                return;
            }
        };

        // Render each light's shadow map. Each light slot occupies 6 consecutive layers.
        // Point lights fill all 6 faces; other light types only render into face 0.
        for (light_index, light_config) in self.lights.iter().enumerate() {
            if light_index >= MAX_SHADOW_LIGHTS {
                break;
            }

            let base_layer = light_index * 6;
            let face_matrices = Self::get_shadow_render_matrices(light_config);

            for (face_idx, face_view_proj) in face_matrices.iter().enumerate() {
                let layer = base_layer + face_idx;

                let layer_view = context.gpu.create_texture_view(
                    shadow_texture,
                    gpu::TextureViewDesc {
                        name: &format!("shadow_map_light{}_face{}", light_index, face_idx),
                        format: gpu::TextureFormat::Depth32Float,
                        dimension: gpu::ViewDimension::D2,
                        subresources: &gpu::TextureSubresources {
                            base_mip_level: 0,
                            mip_level_count: std::num::NonZero::new(1),
                            base_array_layer: layer as u32,
                            array_layer_count: std::num::NonZero::new(1),
                        },
                    },
                );

                let mut pass = encoder.render(
                    &format!("shadow_pass_light{}_face{}", light_index, face_idx),
                    gpu::RenderTargetSet {
                        colors: &[],
                        depth_stencil: Some(gpu::RenderTarget {
                            view: layer_view,
                            init_op: gpu::InitOp::Clear(gpu::TextureColor::White),
                            finish_op: gpu::FinishOp::Store,
                        }),
                    },
                );

                let shadow_data = ShadowData {
                    shadow_uniforms: ShadowUniforms {
                        light_view_proj: face_view_proj.to_cols_array_2d(),
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
                    rc.draw_indexed(
                        mesh.index_buffer,
                        gpu::IndexType::U32,
                        mesh.index_count,
                        0,
                        0,
                        1,
                    );
                }

                drop(pass);
                context.gpu.destroy_texture_view(layer_view);
            }
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
