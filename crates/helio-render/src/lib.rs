use blade_graphics as gpu;
use bytemuck::{Pod, Zeroable};
use helio_features::FeatureRegistry;
use std::sync::Arc;
use std::time::Instant;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CameraUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub position: [f32; 3],
    pub _pad: f32,
}

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

pub struct FeatureRenderer {
    pipeline: gpu::RenderPipeline,
    depth_texture: gpu::Texture,
    depth_view: gpu::TextureView,
    context: Arc<gpu::Context>,
    registry: FeatureRegistry,
    frame_index: u64,
    base_shader: String,
    surface_format: gpu::TextureFormat,
    shadow_pipeline: Option<gpu::RenderPipeline>,
    fps_last_time: Instant,
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
    pub fn new(
        context: Arc<gpu::Context>,
        surface_format: gpu::TextureFormat,
        width: u32,
        height: u32,
        mut registry: FeatureRegistry,
        base_shader: &str,
    ) -> Self {
        use helio_features::FeatureContext;

        let feature_context = FeatureContext {
            gpu: context.clone(),
            surface_size: (width, height),
            frame_index: 0,
            delta_time: 0.0,
        };

        registry.init_all(&feature_context);

        let composed_shader = registry.compose_shader(base_shader);

        log::debug!("Composed shader:\n{}", composed_shader);

        let shader = context.create_shader(gpu::ShaderDesc {
            source: &composed_shader,
        });

        let scene_layout = <SceneData as gpu::ShaderData>::layout();
        let object_layout = <ObjectData as gpu::ShaderData>::layout();

        let data_layouts = vec![&scene_layout, &object_layout];

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
            shadow_pipeline: None,
            fps_last_time: Instant::now(),
        }
    }

    pub fn rebuild_pipeline(&mut self) {
        log::debug!("Rebuilding pipeline with updated features");

        let composed_shader = self.registry.compose_shader(&self.base_shader);
        log::debug!("Recomposed shader:\n{}", composed_shader);

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

    pub fn render(
        &mut self,
        command_encoder: &mut gpu::CommandEncoder,
        target_view: gpu::TextureView,
        camera: CameraUniforms,
        meshes: &[(TransformUniforms, gpu::BufferPiece, gpu::BufferPiece, u32)],
        delta_time: f32,
    ) {
        use helio_features::{FeatureContext, MeshData};

        let context = FeatureContext {
            gpu: self.context.clone(),
            surface_size: (0, 0),
            frame_index: self.frame_index,
            delta_time,
        };

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
        // Using same direction as BasicLighting feature
        let light_direction = glam::Vec3::new(0.5, -1.0, 0.3).normalize();
        let light_pos = -light_direction * 20.0;
        let light_view = glam::Mat4::look_at_rh(light_pos, glam::Vec3::ZERO, glam::Vec3::Y);
        let light_projection = glam::Mat4::orthographic_rh(-8.0, 8.0, -8.0, 8.0, 0.1, 40.0);
        let light_view_proj = (light_projection * light_view).to_cols_array_2d();

        // Execute shadow passes before main render
        self.registry.execute_shadow_passes(command_encoder, &context, &mesh_data, light_view_proj);

        self.registry.execute_pre_passes(command_encoder, &context);

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

        self.registry.execute_post_passes(command_encoder, &context);
        self.frame_index += 1;
        
        if self.frame_index % 1000 == 0 {
            let now = Instant::now();
            let elapsed = now.duration_since(self.fps_last_time);
            let fps = 1000.0 / elapsed.as_secs_f64();
            println!("FPS: {:.2}", fps);
            self.fps_last_time = now;
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

    pub fn registry(&self) -> &FeatureRegistry {
        &self.registry
    }

    pub fn registry_mut(&mut self) -> &mut FeatureRegistry {
        &mut self.registry
    }
}
