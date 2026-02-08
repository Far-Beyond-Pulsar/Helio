use blade_graphics as gpu;
use bytemuck::{Pod, Zeroable};
use helio_features::{FeatureContext, FeatureRegistry};
use std::sync::Arc;

/// Uniforms for camera transformation.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CameraUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub position: [f32; 3],
    pub _pad: f32,
}

/// Uniforms for object transformation.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TransformUniforms {
    pub model: [[f32; 4]; 4],
}

#[derive(blade_macros::ShaderData)]
struct SceneData {
    camera: CameraUniforms,
}

#[derive(blade_macros::ShaderData)]
struct ObjectData {
    transform: TransformUniforms,
}

pub struct Renderer {
    pipeline: gpu::RenderPipeline,
    depth_texture: gpu::Texture,
    depth_view: gpu::TextureView,
    context: Arc<gpu::Context>,
}

/// Feature-based renderer with modular shader composition.
///
/// This renderer uses the feature system to compose shaders dynamically
/// from registered features. It manages the render pipeline, depth buffers,
/// and coordinates feature lifecycle.
///
/// # Example
/// ```ignore
/// let registry = FeatureRegistry::builder()
///     .with_feature(BaseGeometry::new())
///     .with_feature(BasicLighting::new())
///     .build();
///
/// let renderer = FeatureRenderer::new(
///     context,
///     surface_format,
///     width,
///     height,
///     registry,
///     &base_shader_template,
/// );
///
/// // Each frame:
/// renderer.render(&mut encoder, target_view, camera, &meshes, delta_time);
/// ```
pub struct FeatureRenderer {
    pipeline: gpu::RenderPipeline,
    depth_texture: gpu::Texture,
    depth_view: gpu::TextureView,
    context: Arc<gpu::Context>,
    registry: FeatureRegistry,
    frame_index: u64,
    base_shader: String,
    surface_format: gpu::TextureFormat,
}

impl Renderer {
    pub fn new(
        context: Arc<gpu::Context>,
        surface_format: gpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let scene_layout = <SceneData as gpu::ShaderData>::layout();
        let object_layout = <ObjectData as gpu::ShaderData>::layout();
        
        let shader_source = std::fs::read_to_string("shaders/main.wgsl")
            .expect("Failed to read shader");
        let shader = context.create_shader(gpu::ShaderDesc {
            source: &shader_source,
        });
        
        let depth_texture = context.create_texture(gpu::TextureDesc {
            name: "depth",
            format: gpu::TextureFormat::Depth32Float,
            size: gpu::Extent { width, height, depth: 1 },
            dimension: gpu::TextureDimension::D2,
            array_layer_count: 1,
            mip_level_count: 1,
            usage: gpu::TextureUsage::TARGET,
            sample_count: 1,
            external: None,
        });
        
        let depth_view = context.create_texture_view(
            depth_texture,
            gpu::TextureViewDesc {
                name: "depth_view",
                format: gpu::TextureFormat::Depth32Float,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );
        
        let pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "main",
            data_layouts: &[&scene_layout, &object_layout],
            vertex: shader.at("vs_main"),
            vertex_fetches: &[gpu::VertexFetchState {
                layout: &<helio_core::PackedVertex as gpu::Vertex>::layout(),
                instanced: false,
            }],
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleList,
                front_face: gpu::FrontFace::Ccw,
                cull_mode: Some(gpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(gpu::DepthStencilState {
                format: gpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: gpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            fragment: Some(shader.at("fs_main")),
            color_targets: &[gpu::ColorTargetState {
                format: surface_format,
                blend: None,
                write_mask: gpu::ColorWrites::default(),
            }],
            multisample_state: gpu::MultisampleState::default(),
        });
        
        Self {
            pipeline,
            depth_texture,
            depth_view,
            context,
        }
    }
    
    pub fn render(
        &self,
        command_encoder: &mut gpu::CommandEncoder,
        target_view: gpu::TextureView,
        camera: CameraUniforms,
        meshes: &[(TransformUniforms, gpu::BufferPiece, gpu::BufferPiece, u32)],
    ) {
        let mut pass = command_encoder.render(
            "main",
            gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: target_view,
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                    finish_op: gpu::FinishOp::Store,
                }],
                depth_stencil: Some(gpu::RenderTarget {
                    view: self.depth_view,
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::White),
                    finish_op: gpu::FinishOp::Store,
                }),
            },
        );
        {
            let scene_data = SceneData { camera };
            let mut rc = pass.with(&self.pipeline);
            rc.bind(0, &scene_data);

            for (transform, vertex_buf, index_buf, index_count) in meshes {
                let object_data = ObjectData { transform: *transform };
                rc.bind(1, &object_data);
                rc.bind_vertex(0, *vertex_buf);
                rc.draw_indexed(*index_buf, gpu::IndexType::U32, *index_count, 0, 0, 1);
            }
        }
    }
    
    pub fn resize(&mut self, width: u32, height: u32) {
        self.context.destroy_texture_view(self.depth_view);
        self.context.destroy_texture(self.depth_texture);

        self.depth_texture = self.context.create_texture(gpu::TextureDesc {
            name: "depth",
            format: gpu::TextureFormat::Depth32Float,
            size: gpu::Extent { width, height, depth: 1 },
            dimension: gpu::TextureDimension::D2,
            array_layer_count: 1,
            mip_level_count: 1,
            usage: gpu::TextureUsage::TARGET,
            sample_count: 1,
            external: None,
        });

        self.depth_view = self.context.create_texture_view(
            self.depth_texture,
            gpu::TextureViewDesc {
                name: "depth_view",
                format: gpu::TextureFormat::Depth32Float,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );
    }
}

impl FeatureRenderer {
    /// Create a new feature-based renderer.
    ///
    /// # Arguments
    /// - `context`: GPU context for creating resources
    /// - `surface_format`: Color target format (usually from the surface)
    /// - `width`: Initial render target width
    /// - `height`: Initial render target height
    /// - `mut registry`: Feature registry with registered features
    /// - `base_shader`: Base shader template with injection markers
    ///
    /// # Example
    /// ```ignore
    /// let renderer = FeatureRenderer::new(
    ///     gpu_context,
    ///     gpu::TextureFormat::Bgra8UnormSrgb,
    ///     1920,
    ///     1080,
    ///     registry,
    ///     include_str!("shaders/base.wgsl"),
    /// );
    /// ```
    pub fn new(
        context: Arc<gpu::Context>,
        surface_format: gpu::TextureFormat,
        width: u32,
        height: u32,
        mut registry: FeatureRegistry,
        base_shader: &str,
    ) -> Self {
        let depth_format = gpu::TextureFormat::Depth32Float;

        let feature_context = FeatureContext::new(
            context.clone(),
            (width, height),
            depth_format,
            surface_format,
        );

        // Initialize all features
        registry.init_all(&feature_context);

        // Compose shader from features
        let composed_shader = registry.compose_shader(base_shader);
        log::debug!("Composed shader from {} features", registry.enabled_count());

        let shader = context.create_shader(gpu::ShaderDesc {
            source: &composed_shader,
        });

        let scene_layout = <SceneData as gpu::ShaderData>::layout();
        let object_layout = <ObjectData as gpu::ShaderData>::layout();
        let data_layouts = vec![&scene_layout, &object_layout];

        let depth_texture = context.create_texture(gpu::TextureDesc {
            name: "depth",
            format: depth_format,
            size: gpu::Extent { width, height, depth: 1 },
            dimension: gpu::TextureDimension::D2,
            array_layer_count: 1,
            mip_level_count: 1,
            usage: gpu::TextureUsage::TARGET,
            sample_count: 1,
            external: None,
        });

        let depth_view = context.create_texture_view(
            depth_texture,
            gpu::TextureViewDesc {
                name: "depth_view",
                format: depth_format,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );

        let pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "feature_main",
            data_layouts: &data_layouts,
            vertex: shader.at("vs_main"),
            vertex_fetches: &[gpu::VertexFetchState {
                layout: &<helio_core::PackedVertex as gpu::Vertex>::layout(),
                instanced: false,
            }],
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleList,
                front_face: gpu::FrontFace::Ccw,
                cull_mode: Some(gpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(gpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: gpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            fragment: Some(shader.at("fs_main")),
            color_targets: &[gpu::ColorTargetState {
                format: surface_format,
                blend: None,
                write_mask: gpu::ColorWrites::default(),
            }],
            multisample_state: gpu::MultisampleState::default(),
        });

        Self {
            pipeline,
            depth_texture,
            depth_view,
            context,
            registry,
            frame_index: 0,
            base_shader: base_shader.to_string(),
            surface_format,
        }
    }

    /// Rebuild the render pipeline with current feature state.
    ///
    /// Call this after enabling/disabling features to recompose the shader
    /// and recreate the pipeline.
    ///
    /// # Example
    /// ```ignore
    /// renderer.registry_mut().toggle_feature("basic_lighting")?;
    /// renderer.rebuild_pipeline();
    /// ```
    pub fn rebuild_pipeline(&mut self) {
        log::info!("Rebuilding pipeline with {} enabled features", 
                  self.registry.enabled_count());

        let composed_shader = self.registry.compose_shader(&self.base_shader);

        let shader = self.context.create_shader(gpu::ShaderDesc {
            source: &composed_shader,
        });

        let scene_layout = <SceneData as gpu::ShaderData>::layout();
        let object_layout = <ObjectData as gpu::ShaderData>::layout();
        let data_layouts = vec![&scene_layout, &object_layout];

        self.pipeline = self.context.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "feature_main",
            data_layouts: &data_layouts,
            vertex: shader.at("vs_main"),
            vertex_fetches: &[gpu::VertexFetchState {
                layout: &<helio_core::PackedVertex as gpu::Vertex>::layout(),
                instanced: false,
            }],
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleList,
                front_face: gpu::FrontFace::Ccw,
                cull_mode: Some(gpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(gpu::DepthStencilState {
                format: gpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: gpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            fragment: Some(shader.at("fs_main")),
            color_targets: &[gpu::ColorTargetState {
                format: self.surface_format,
                blend: None,
                write_mask: gpu::ColorWrites::default(),
            }],
            multisample_state: gpu::MultisampleState::default(),
        });
    }

    /// Render a frame with the feature-based pipeline.
    ///
    /// Coordinates the full rendering process:
    /// 1. Prepares all features for the frame
    /// 2. Executes shadow passes (if any features implement them)
    /// 3. Executes pre-render passes
    /// 4. Main render pass
    /// 5. Executes post-render passes
    ///
    /// # Arguments
    /// - `command_encoder`: GPU command encoder
    /// - `target_view`: Render target (usually from surface)
    /// - `camera`: Camera transformation uniforms
    /// - `meshes`: Scene geometry to render
    /// - `delta_time`: Time since last frame in seconds
    ///
    /// # Example
    /// ```ignore
    /// renderer.render(
    ///     &mut encoder,
    ///     surface_view,
    ///     camera_uniforms,
    ///     &[(transform, vbuf, ibuf, index_count)],
    ///     0.016, // ~60 FPS
    /// );
    /// ```
    pub fn render(
        &mut self,
        command_encoder: &mut gpu::CommandEncoder,
        target_view: gpu::TextureView,
        camera: CameraUniforms,
        meshes: &[(TransformUniforms, gpu::BufferPiece, gpu::BufferPiece, u32)],
        delta_time: f32,
    ) {
        use helio_features::MeshData;

        let mut context = FeatureContext::new(
            self.context.clone(),
            (0, 0), // Size will be updated if needed
            gpu::TextureFormat::Depth32Float,
            self.surface_format,
        );
        context.update_frame(self.frame_index, delta_time);

        // Prepare all features for this frame
        self.registry.prepare_frame(&context);

        // Convert meshes to MeshData for shadow pass
        let mesh_data: Vec<MeshData> = meshes
            .iter()
            .map(|(transform, vertex_buf, index_buf, index_count)| MeshData {
                transform: transform.model,
                vertex_buffer: *vertex_buf,
                index_buffer: *index_buf,
                index_count: *index_count,
            })
            .collect();

        // Calculate light view projection for shadow mapping
        // TODO: This should come from a lighting feature, not be hardcoded
        let light_direction = glam::Vec3::new(0.5, -1.0, 0.3).normalize();
        let light_pos = -light_direction * 20.0;
        let light_view = glam::Mat4::look_at_rh(light_pos, glam::Vec3::ZERO, glam::Vec3::Y);
        let light_projection = glam::Mat4::orthographic_rh(-8.0, 8.0, -8.0, 8.0, 0.1, 40.0);
        let light_view_proj = (light_projection * light_view).to_cols_array_2d();

        // Execute shadow passes before main render
        self.registry.execute_shadow_passes(command_encoder, &context, &mesh_data, light_view_proj);

        // Execute pre-render passes
        self.registry.execute_pre_passes(command_encoder, &context);

        // Main render pass
        {
            let mut pass = command_encoder.render(
                "feature_main",
                gpu::RenderTargetSet {
                    colors: &[gpu::RenderTarget {
                        view: target_view,
                        init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                        finish_op: gpu::FinishOp::Store,
                    }],
                    depth_stencil: Some(gpu::RenderTarget {
                        view: self.depth_view,
                        init_op: gpu::InitOp::Clear(gpu::TextureColor::White),
                        finish_op: gpu::FinishOp::Store,
                    }),
                },
            );
            
            let scene_data = SceneData { camera };
            let mut rc = pass.with(&self.pipeline);
            rc.bind(0, &scene_data);

            for (transform, vertex_buf, index_buf, index_count) in meshes {
                let object_data = ObjectData { transform: *transform };
                rc.bind(1, &object_data);
                rc.bind_vertex(0, *vertex_buf);
                rc.draw_indexed(*index_buf, gpu::IndexType::U32, *index_count, 0, 0, 1);
            }
        }

        // Execute post-render passes
        self.registry.execute_post_passes(command_encoder, &context);
        
        self.frame_index += 1;
    }

    /// Resize the renderer's render targets.
    ///
    /// Call this when the window/surface is resized. Recreates depth buffer
    /// with new dimensions.
    pub fn resize(&mut self, width: u32, height: u32) {
        log::debug!("Resizing renderer to {}x{}", width, height);
        
        self.context.destroy_texture_view(self.depth_view);
        self.context.destroy_texture(self.depth_texture);

        self.depth_texture = self.context.create_texture(gpu::TextureDesc {
            name: "depth",
            format: gpu::TextureFormat::Depth32Float,
            size: gpu::Extent { width, height, depth: 1 },
            dimension: gpu::TextureDimension::D2,
            array_layer_count: 1,
            mip_level_count: 1,
            usage: gpu::TextureUsage::TARGET,
            sample_count: 1,
            external: None,
        });

        self.depth_view = self.context.create_texture_view(
            self.depth_texture,
            gpu::TextureViewDesc {
                name: "depth_view",
                format: gpu::TextureFormat::Depth32Float,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );
    }

    /// Get an immutable reference to the feature registry.
    ///
    /// Use this to query feature state or inspect registered features.
    pub fn registry(&self) -> &FeatureRegistry {
        &self.registry
    }

    /// Get a mutable reference to the feature registry.
    ///
    /// Use this to modify features, toggle them, or change their state.
    /// Remember to call `rebuild_pipeline()` after enabling/disabling features.
    pub fn registry_mut(&mut self) -> &mut FeatureRegistry {
        &mut self.registry
    }
    
    /// Get the current frame index.
    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }
    
    /// Get the GPU context.
    pub fn context(&self) -> &Arc<gpu::Context> {
        &self.context
    }
}

impl Drop for FeatureRenderer {
    fn drop(&mut self) {
        log::debug!("Cleaning up FeatureRenderer");
        
        let context = FeatureContext::new(
            self.context.clone(),
            (0, 0),
            gpu::TextureFormat::Depth32Float,
            self.surface_format,
        );
        
        self.registry.cleanup_all(&context);
    }
}
