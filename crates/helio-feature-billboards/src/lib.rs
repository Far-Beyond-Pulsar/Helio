use blade_graphics as gpu;
use bytemuck::{Pod, Zeroable};
use helio_core::{BillboardVertex, TextureId, TextureManager};
use helio_features::{Feature, FeatureContext, ShaderInjection};
use std::ptr;
use std::sync::Arc;

/// Blending mode for billboard rendering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    /// No blending, opaque rendering
    Opaque,
    /// Alpha blending for transparency
    Transparent,
}

/// Single billboard instance in the scene
#[derive(Clone)]
pub struct BillboardData {
    /// World space position
    pub position: [f32; 3],
    /// Scale (width, height) in world units
    pub scale: [f32; 2],
    /// Texture to display
    pub texture: TextureId,
    /// Blending mode
    pub blend_mode: BlendMode,
}

impl BillboardData {
    pub fn new(position: [f32; 3], scale: [f32; 2], texture: TextureId) -> Self {
        Self {
            position,
            scale,
            texture,
            blend_mode: BlendMode::Transparent,
        }
    }

    pub fn with_blend_mode(mut self, mode: BlendMode) -> Self {
        self.blend_mode = mode;
        self
    }
}

// --- Shader data structs matching billboard.wgsl ---

/// Camera uniforms matching the WGSL CameraUniforms struct:
///   view_proj: mat4x4<f32>  (64 bytes, offset 0)
///   position:  vec3<f32>    (12 bytes, offset 64, padded to 80)
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BillboardCameraUniforms {
    view_proj: [[f32; 4]; 4],
    position: [f32; 3],
    _pad: f32,
}

#[derive(blade_macros::ShaderData)]
struct BillboardCameraData {
    camera: BillboardCameraUniforms,
}

/// Billboard instance uniforms matching the WGSL BillboardInstance struct:
///   world_position: vec3<f32>  (align 16, offset 0, size 12)
///   scale:          vec2<f32>  (align  8, offset 16, size 8)
///   struct size rounds up to 32 bytes
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BillboardInstanceUniforms {
    world_position: [f32; 3],
    _pad1: f32,
    scale: [f32; 2],
    _pad2: [f32; 2],
}

#[derive(blade_macros::ShaderData)]
struct BillboardInstanceData {
    billboard: BillboardInstanceUniforms,
}

/// Texture + sampler bound at group 2
#[derive(blade_macros::ShaderData)]
struct BillboardTextureData {
    tex: gpu::TextureView,
    tex_sampler: gpu::Sampler,
}

/// Billboard rendering feature
///
/// This feature adds support for camera-facing billboards with PNG textures.
/// Billboards can be used for standalone scene objects or editor gizmos (lights, etc.)
pub struct BillboardFeature {
    enabled: bool,
    texture_manager: Option<Arc<TextureManager>>,
    // GPU resources (created in init)
    pipeline: Option<gpu::RenderPipeline>,
    quad_vertex_buffer: Option<gpu::Buffer>,
    quad_index_buffer: Option<gpu::Buffer>,
    // Cached surface format for pipeline creation
    surface_format: Option<gpu::TextureFormat>,
}

impl BillboardFeature {
    pub fn new() -> Self {
        Self {
            enabled: true,
            texture_manager: None,
            pipeline: None,
            quad_vertex_buffer: None,
            quad_index_buffer: None,
            surface_format: None,
        }
    }

    /// Set the texture manager for loading billboard textures
    pub fn set_texture_manager(&mut self, manager: Arc<TextureManager>) {
        self.texture_manager = Some(manager);
    }

    /// Get a reference to the texture manager
    pub fn texture_manager(&self) -> Option<&Arc<TextureManager>> {
        self.texture_manager.as_ref()
    }

    /// Render billboard overlays after the main pass.
    ///
    /// Call this after the main render pass with the same target view.
    /// Uses `InitOp::Load` to preserve already-rendered geometry.
    pub fn render_billboard_overlay(
        &self,
        encoder: &mut gpu::CommandEncoder,
        target_view: gpu::TextureView,
        depth_view: gpu::TextureView,
        view_proj: [[f32; 4]; 4],
        camera_pos: [f32; 3],
        billboards: &[BillboardData],
    ) {
        if billboards.is_empty() {
            return;
        }

        let pipeline = match &self.pipeline {
            Some(p) => p,
            None => {
                log::warn!("Billboard pipeline not initialized, skipping overlay");
                return;
            }
        };
        let quad_vbuf = match self.quad_vertex_buffer {
            Some(b) => b,
            None => {
                log::warn!("Billboard quad vertex buffer not initialized");
                return;
            }
        };
        let quad_ibuf = match self.quad_index_buffer {
            Some(b) => b,
            None => {
                log::warn!("Billboard quad index buffer not initialized");
                return;
            }
        };
        let texture_manager = match &self.texture_manager {
            Some(tm) => tm,
            None => {
                log::warn!("No texture manager set for billboards");
                return;
            }
        };

        let camera_data = BillboardCameraData {
            camera: BillboardCameraUniforms {
                view_proj,
                position: camera_pos,
                _pad: 0.0,
            },
        };

        let mut pass = encoder.render(
            "billboard_overlay",
            gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: target_view,
                    init_op: gpu::InitOp::Load,
                    finish_op: gpu::FinishOp::Store,
                }],
                depth_stencil: Some(gpu::RenderTarget {
                    view: depth_view,
                    init_op: gpu::InitOp::Load,
                    finish_op: gpu::FinishOp::Store,
                }),
            },
        );

        let mut rc = pass.with(pipeline);
        rc.bind(0, &camera_data);

        for billboard in billboards {
            let gpu_texture = match texture_manager.get(billboard.texture) {
                Some(t) => t,
                None => {
                    log::warn!("Billboard references unknown texture {:?}", billboard.texture);
                    continue;
                }
            };

            let instance_data = BillboardInstanceData {
                billboard: BillboardInstanceUniforms {
                    world_position: billboard.position,
                    _pad1: 0.0,
                    scale: billboard.scale,
                    _pad2: [0.0, 0.0],
                },
            };
            rc.bind(1, &instance_data);

            let texture_data = BillboardTextureData {
                tex: gpu_texture.view,
                tex_sampler: gpu_texture.sampler,
            };
            rc.bind(2, &texture_data);

            rc.bind_vertex(0, quad_vbuf.into());
            rc.draw_indexed(quad_ibuf.into(), gpu::IndexType::U32, 6, 0, 0, 1);
        }

        drop(pass);
    }
}

impl Feature for BillboardFeature {
    fn name(&self) -> &str {
        "billboards"
    }

    fn init(&mut self, context: &FeatureContext) {
        log::info!("Initializing billboard feature");

        // Create the quad mesh
        let quad = helio_core::create_billboard_quad(1.0);

        let quad_vbuf = context.gpu.create_buffer(gpu::BufferDesc {
            name: "billboard_quad_vertices",
            size: (quad.vertices.len() * std::mem::size_of::<BillboardVertex>()) as u64,
            memory: gpu::Memory::Shared,
        });
        unsafe {
            ptr::copy_nonoverlapping(
                quad.vertices.as_ptr(),
                quad_vbuf.data() as *mut BillboardVertex,
                quad.vertices.len(),
            );
        }
        context.gpu.sync_buffer(quad_vbuf);

        let quad_ibuf = context.gpu.create_buffer(gpu::BufferDesc {
            name: "billboard_quad_indices",
            size: (quad.indices.len() * std::mem::size_of::<u32>()) as u64,
            memory: gpu::Memory::Shared,
        });
        unsafe {
            ptr::copy_nonoverlapping(
                quad.indices.as_ptr(),
                quad_ibuf.data() as *mut u32,
                quad.indices.len(),
            );
        }
        context.gpu.sync_buffer(quad_ibuf);

        self.quad_vertex_buffer = Some(quad_vbuf);
        self.quad_index_buffer = Some(quad_ibuf);
        self.surface_format = Some(context.color_format);

        // Create billboard render pipeline
        let camera_layout = <BillboardCameraData as gpu::ShaderData>::layout();
        let instance_layout = <BillboardInstanceData as gpu::ShaderData>::layout();
        let texture_layout = <BillboardTextureData as gpu::ShaderData>::layout();

        let shader_source = include_str!("../shaders/billboard.wgsl");
        let shader = context.gpu.create_shader(gpu::ShaderDesc {
            source: shader_source,
        });

        let pipeline = context.gpu.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "billboard",
            data_layouts: &[&camera_layout, &instance_layout, &texture_layout],
            vertex: shader.at("vs_main"),
            vertex_fetches: &[gpu::VertexFetchState {
                layout: &<BillboardVertex as gpu::Vertex>::layout(),
                instanced: false,
            }],
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleList,
                front_face: gpu::FrontFace::Ccw,
                cull_mode: None, // Billboards are always visible from both sides
                ..Default::default()
            },
            depth_stencil: Some(gpu::DepthStencilState {
                format: context.depth_format,
                depth_write_enabled: false, // Don't write depth for transparent overlays
                depth_compare: gpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            fragment: Some(shader.at("fs_main")),
            color_targets: &[gpu::ColorTargetState {
                format: context.color_format,
                blend: Some(gpu::BlendState::ALPHA_BLENDING),
                write_mask: gpu::ColorWrites::default(),
            }],
            multisample_state: gpu::MultisampleState::default(),
        });

        self.pipeline = Some(pipeline);
        log::info!("Billboard pipeline created");
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn shader_injections(&self) -> Vec<ShaderInjection> {
        // Billboards use their own separate pipeline, no main shader injection
        Vec::new()
    }

    fn cleanup(&mut self, context: &FeatureContext) {
        if let Some(mut pipeline) = self.pipeline.take() {
            context.gpu.destroy_render_pipeline(&mut pipeline);
        }
        if let Some(buf) = self.quad_vertex_buffer.take() {
            context.gpu.destroy_buffer(buf);
        }
        if let Some(buf) = self.quad_index_buffer.take() {
            context.gpu.destroy_buffer(buf);
        }
    }
}

impl Default for BillboardFeature {
    fn default() -> Self {
        Self::new()
    }
}
