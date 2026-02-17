use blade_graphics as gpu;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use helio_core::MeshBuffer;
use helio_features::{FeatureContext, FeatureRegistry};
use std::sync::Arc;

/// Uniforms for camera transformation.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CameraUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub position: [f32; 3],
    /// Elapsed time in seconds, packed into the vec3 padding slot.
    /// Accessible in WGSL as `camera.time` for cloud animation and day/night cycles.
    pub time: f32,
}

impl CameraUniforms {
    /// Build camera uniforms from a pre-composed view-projection matrix, world position,
    /// and elapsed time in seconds (used for cloud/sky animation).
    pub fn new(view_proj: Mat4, position: Vec3, time: f32) -> Self {
        Self {
            view_proj: view_proj.to_cols_array_2d(),
            position: position.to_array(),
            time,
        }
    }
}

/// Uniforms for object transformation.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TransformUniforms {
    pub model: [[f32; 4]; 4],
}

impl TransformUniforms {
    /// Build transform uniforms from a model matrix.
    pub fn from_matrix(mat: Mat4) -> Self {
        Self { model: mat.to_cols_array_2d() }
    }
}

/// Simple FPS-style camera controller.
///
/// Handles the view-matrix math without coupling to any windowing library.
/// The calling code is responsible for tracking which keys are pressed and
/// calling [`update_movement`] / [`handle_mouse_delta`] each frame.
///
/// # Example
/// ```ignore
/// let mut camera = FpsCamera::new(Vec3::new(0.0, 5.0, 15.0));
///
/// // Each frame (inside event/render loop):
/// let fwd = keys.w as i8 - keys.s as i8;
/// let rgt = keys.d as i8 - keys.a as i8;
/// let up  = keys.space as i8 - keys.shift as i8;
/// camera.update_movement(fwd as f32, rgt as f32, up as f32, delta_time);
///
/// let camera_uniforms = camera.build_camera_uniforms(60.0, aspect_ratio);
/// ```
pub struct FpsCamera {
    pub position: Vec3,
    /// Horizontal angle in radians (default points down -Z).
    pub yaw: f32,
    /// Vertical angle in radians, clamped to ±89°.
    pub pitch: f32,
    /// World-space units per second.
    pub move_speed: f32,
    /// Mouse sensitivity scalar (degrees per pixel scale).
    pub look_speed: f32,
}

impl FpsCamera {
    /// Create a camera at `position` looking down -Z.
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            yaw: -90.0_f32.to_radians(),
            pitch: 0.0,
            move_speed: 5.0,
            look_speed: 0.1,
        }
    }

    /// Unit forward vector in world space.
    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize()
    }

    /// Unit right vector (perpendicular to forward, on the XZ plane).
    pub fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize()
    }

    /// Move the camera. Positive `forward`/`right`/`up` = forward/right/up.
    ///
    /// Pass −1, 0, or 1 (from key state) scaled by any analogue magnitude.
    pub fn update_movement(&mut self, forward: f32, right: f32, up: f32, dt: f32) {
        let fwd = self.forward();
        let rgt = self.right();
        let speed = self.move_speed * dt;
        self.position += fwd * forward * speed;
        self.position += rgt * right * speed;
        self.position.y += up * speed;
    }

    /// Apply a mouse delta (pixels). Positive dx = look right, positive dy = look down.
    pub fn handle_mouse_delta(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * self.look_speed * 0.01;
        self.pitch -= dy * self.look_speed * 0.01;
        self.pitch = self.pitch.clamp(-89.0_f32.to_radians(), 89.0_f32.to_radians());
    }

    /// Right-handed view matrix.
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.position + self.forward(), Vec3::Y)
    }

    /// Build ready-to-use [`CameraUniforms`] for this camera.
    ///
    /// `fov_deg` is the vertical field of view in degrees.
    /// `time` is elapsed seconds since start, used for cloud and sky animation in shaders.
    pub fn build_camera_uniforms(&self, fov_deg: f32, aspect: f32, time: f32) -> CameraUniforms {
        let proj = Mat4::perspective_rh(fov_deg.to_radians(), aspect, 0.1, 10000.0);
        CameraUniforms::new(proj * self.view_matrix(), self.position, time)
    }
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
    ) -> Result<Self, String> {
        let scene_layout = <SceneData as gpu::ShaderData>::layout();
        let object_layout = <ObjectData as gpu::ShaderData>::layout();
        
        let shader_source = std::fs::read_to_string("shaders/main.wgsl")
            .map_err(|e| format!("Failed to read shader file 'shaders/main.wgsl': {}", e))?;
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
                cull_mode: None, // Disable culling for sky sphere to work
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
        
        Ok(Self {
            pipeline,
            depth_texture,
            depth_view,
            context,
        })
    }
    
    pub fn render(
        &self,
        command_encoder: &mut gpu::CommandEncoder,
        target_view: gpu::TextureView,
        camera: CameraUniforms,
        meshes: &[(TransformUniforms, &MeshBuffer)],
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

            for (transform, mesh) in meshes {
                let object_data = ObjectData { transform: *transform };
                rc.bind(1, &object_data);
                rc.bind_vertex(0, mesh.vertex_piece());
                rc.draw_indexed(mesh.index_piece(), gpu::IndexType::U32, mesh.index_count, 0, 0, 1);
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
    ) -> Result<Self, String> {
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
        
        // DEBUG: Write composed shader to file for inspection
        if let Err(e) = std::fs::write("D:\\Documents\\GitHub\\genesis\\Pulsar-Native\\composed_shader_debug.wgsl", &composed_shader) {
            log::warn!("Failed to write debug shader: {}", e);
        } else {
            log::info!("✅ Composed shader written to composed_shader_debug.wgsl");
        }

        let shader = context.create_shader(gpu::ShaderDesc {
            source: &composed_shader,
        });

        let scene_layout = <SceneData as gpu::ShaderData>::layout();
        let object_layout = <ObjectData as gpu::ShaderData>::layout();

        // Check if any feature provides a shadow map and add its layout
        #[cfg(feature = "shadows")]
        let shadow_layout_storage;
        let mut data_layouts = vec![&scene_layout, &object_layout];

        // Check if the procedural shadows feature is enabled
        #[cfg(feature = "shadows")]
        {
            let has_shadows = registry.features().iter().any(|f| {
                f.is_enabled() && f.name() == "procedural_shadows"
            });

            log::info!("Checking for shadow feature: has_shadows={}", has_shadows);

            if has_shadows {
                use helio_feature_procedural_shadows::ShadowMapData;
                shadow_layout_storage = <ShadowMapData as gpu::ShaderData>::layout();
                data_layouts.push(&shadow_layout_storage);
                log::info!("Added shadow map layout to pipeline. Total layouts: {}", data_layouts.len());
            } else {
                log::warn!("Shadows feature is compiled but no shadow feature is enabled");
            }
        }

        log::info!("Creating pipeline with {} data layouts", data_layouts.len());
        let data_layouts = &data_layouts[..];

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

        Ok(Self {
            pipeline,
            depth_texture,
            depth_view,
            context,
            registry,
            frame_index: 0,
            base_shader: base_shader.to_string(),
            surface_format,
        })
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
        
        // DEBUG: Write composed shader to file for inspection
        if let Err(e) = std::fs::write("D:\\Documents\\GitHub\\genesis\\Pulsar-Native\\composed_shader_debug.wgsl", &composed_shader) {
            log::warn!("Failed to write debug shader: {}", e);
        } else {
            log::info!("✅ Composed shader written to composed_shader_debug.wgsl");
        }

        let shader = self.context.create_shader(gpu::ShaderDesc {
            source: &composed_shader,
        });

        let scene_layout = <SceneData as gpu::ShaderData>::layout();
        let object_layout = <ObjectData as gpu::ShaderData>::layout();

        // Check if any feature provides a shadow map and add its layout
        #[cfg(feature = "shadows")]
        let shadow_layout_storage;
        let mut data_layouts = vec![&scene_layout, &object_layout];

        // Check if the procedural shadows feature is enabled
        #[cfg(feature = "shadows")]
        {
            let has_shadows = self.registry.features().iter().any(|f| {
                f.is_enabled() && f.name() == "procedural_shadows"
            });

            if has_shadows {
                use helio_feature_procedural_shadows::ShadowMapData;
                shadow_layout_storage = <ShadowMapData as gpu::ShaderData>::layout();
                data_layouts.push(&shadow_layout_storage);
                log::info!("Added shadow map layout to rebuilt pipeline");
            }
        }

        let data_layouts = &data_layouts[..];

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
        meshes: &[(TransformUniforms, &MeshBuffer)],
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
            .map(|(transform, mesh)| MeshData {
                transform: transform.model,
                vertex_buffer: mesh.vertex_piece(),
                index_buffer: mesh.index_piece(),
                index_count: mesh.index_count,
            })
            .collect();

        // Execute shadow passes before main render.
        // Each shadow feature computes its own light matrices internally.
        self.registry.execute_shadow_passes(command_encoder, &context, &mesh_data);

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

            // Bind shadow map if available
            #[cfg(feature = "shadows")]
            {
                use helio_feature_procedural_shadows::ProceduralShadows;
                use helio_features::Feature;
                for feature in self.registry.features() {
                    if feature.is_enabled() {
                        // Deref through Box to dyn Feature so as_any() dispatches
                        // through the Feature vtable (not Box<dyn Feature>'s blanket impl).
                        let feature_obj: &dyn Feature = &**feature;
                        if let Some(shadows) = feature_obj.as_any().downcast_ref::<ProceduralShadows>() {
                            if let Some(shadow_data) = shadows.get_shadow_map_data() {
                                rc.bind(2, &shadow_data);
                                break;
                            }
                        }
                    }
                }
            }

            for (transform, mesh) in meshes {
                let object_data = ObjectData { transform: *transform };
                rc.bind(1, &object_data);
                rc.bind_vertex(0, mesh.vertex_piece());
                rc.draw_indexed(mesh.index_piece(), gpu::IndexType::U32, mesh.index_count, 0, 0, 1);
            }
        }

        // Execute post-render passes
        self.registry.execute_post_passes(command_encoder, &context);

        // Billboard overlay pass: collect light billboard data from ProceduralShadows,
        // then render via BillboardFeature using the same target and depth views.
        #[cfg(feature = "billboards")]
        {
            use helio_feature_billboards::{BillboardData, BillboardFeature};

            // Collect billboard positions/sizes/textures from any shadow feature
            let mut billboard_data: Vec<BillboardData> = Vec::new();

            #[cfg(feature = "shadows")]
            {
                use helio_feature_procedural_shadows::ProceduralShadows;
                for feature in self.registry.features() {
                    if feature.is_enabled() {
                        let feature_obj: &dyn helio_features::Feature = &**feature;
                        if let Some(shadows) = feature_obj.as_any().downcast_ref::<ProceduralShadows>() {
                            if let Some(texture_id) = shadows.spotlight_icon_texture() {
                                for (pos, size) in shadows.generate_light_billboards() {
                                    billboard_data.push(BillboardData::new(
                                        [pos.x, pos.y, pos.z],
                                        [size, size],
                                        texture_id,
                                    ));
                                }
                            }
                        }
                    }
                }
            }

            if !billboard_data.is_empty() {
                if let Some(bf_box) = self.registry.get_feature_mut("billboards") {
                    let feature_obj: &mut dyn helio_features::Feature = &mut **bf_box;
                    if let Some(bf) = feature_obj.as_any_mut().downcast_mut::<BillboardFeature>() {
                        bf.render_billboard_overlay(
                            command_encoder,
                            target_view,
                            self.depth_view,
                            camera.view_proj,
                            camera.position,
                            &billboard_data,
                        );
                    }
                }
            }
        }

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

    /// Toggle a feature and immediately rebuild the pipeline.
    ///
    /// Convenience wrapper around `registry_mut().toggle_feature()` + `rebuild_pipeline()`.
    /// Returns `Ok(new_enabled_state)` or `Err` if the feature name is not found.
    ///
    /// # Example
    /// ```ignore
    /// if let Ok(enabled) = renderer.toggle_and_rebuild("basic_lighting") {
    ///     println!("Lighting: {}", if enabled { "ON" } else { "OFF" });
    /// }
    /// ```
    pub fn toggle_and_rebuild(&mut self, name: &str) -> Result<bool, helio_features::FeatureError> {
        let state = self.registry.toggle_feature(name)?;
        self.rebuild_pipeline();
        Ok(state)
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
