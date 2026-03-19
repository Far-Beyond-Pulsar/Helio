use std::num::NonZeroU32;
use std::sync::Arc;

use arrayvec::ArrayVec;
use bytemuck::{Pod, Zeroable};
use helio_v3::{RenderGraph, RenderPass, Result as HelioResult};

use crate::mesh::MeshBuffers;
use crate::handles::{LightId, MaterialId, MeshId, ObjectId};
use crate::material::MAX_TEXTURES;
use crate::mesh::MeshUpload;
use crate::scene::{Camera, ObjectDescriptor, Result as SceneResult, Scene};

pub fn required_wgpu_features(adapter_features: wgpu::Features) -> wgpu::Features {
    let required = wgpu::Features::TEXTURE_BINDING_ARRAY
        | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
    let optional = wgpu::Features::MULTI_DRAW_INDIRECT;
    required | (adapter_features & optional)
}

#[derive(Debug, Clone, Copy)]
pub struct RendererConfig {
    pub width: u32,
    pub height: u32,
    pub surface_format: wgpu::TextureFormat,
}

impl RendererConfig {
    pub fn new(width: u32, height: u32, surface_format: wgpu::TextureFormat) -> Self {
        Self {
            width,
            height,
            surface_format,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniforms {
    ambient_color: [f32; 4],
    clear_color: [f32; 4],
    light_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub struct Renderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    graph: Option<RenderGraph>,
    scene: Scene,
    pipeline: wgpu::RenderPipeline,
    scene_bind_group_layout: wgpu::BindGroupLayout,
    frame_uniforms: wgpu::Buffer,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    use_multi_draw_indirect: bool,
    ambient_color: [f32; 3],
    ambient_intensity: f32,
    clear_color: [f32; 4],
}

impl Renderer {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        config: RendererConfig,
    ) -> Self {
        let texture_array_count = NonZeroU32::new(MAX_TEXTURES as u32).expect("non-zero texture table size");
        let mut scene = Scene::new(device.clone(), queue.clone());
        scene.set_render_size(config.width, config.height);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Helio Forward Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("forward.wgsl").into()),
        });

        let scene_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Helio Forward Scene BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: Some(texture_array_count),
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: Some(texture_array_count),
                    },
                ],
            });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Helio Forward Pipeline Layout"),
            bind_group_layouts: &[&scene_bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Helio Forward Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 40,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32,
                            offset: 12,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 16,
                            shader_location: 2,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 24,
                            shader_location: 3,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32,
                            offset: 32,
                            shader_location: 4,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32,
                            offset: 36,
                            shader_location: 5,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let frame_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Helio Frame Uniforms"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (depth_texture, depth_view) = create_depth_resources(&device, config.width, config.height);
        Self {
            device: device.clone(),
            queue,
            graph: None,
            scene,
            pipeline,
            scene_bind_group_layout,
            frame_uniforms,
            depth_texture,
            depth_view,
            use_multi_draw_indirect: device
                .features()
                .contains(wgpu::Features::MULTI_DRAW_INDIRECT),
            ambient_color: [0.05, 0.05, 0.08],
            ambient_intensity: 1.0,
            clear_color: [0.02, 0.02, 0.03, 1.0],
        }
    }

    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    pub fn scene_mut(&mut self) -> &mut Scene {
        &mut self.scene
    }

    pub fn mesh_buffers(&self) -> MeshBuffers<'_> {
        self.scene.mesh_buffers()
    }

    pub fn add_pass(&mut self, pass: Box<dyn RenderPass>) {
        self.graph
            .get_or_insert_with(|| RenderGraph::new(&self.device, &self.queue))
            .add_pass(pass);
    }

    pub fn set_render_size(&mut self, width: u32, height: u32) {
        self.scene.set_render_size(width, height);
        let (depth_texture, depth_view) = create_depth_resources(&self.device, width, height);
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
    }

    pub fn set_clear_color(&mut self, color: [f32; 4]) {
        self.clear_color = color;
    }

    pub fn set_ambient(&mut self, color: [f32; 3], intensity: f32) {
        self.ambient_color = color;
        self.ambient_intensity = intensity;
    }

    pub fn insert_mesh(&mut self, mesh: MeshUpload) -> MeshId {
        self.scene.insert_mesh(mesh)
    }

    pub fn insert_material(&mut self, material: crate::GpuMaterial) -> MaterialId {
        self.scene.insert_material(material)
    }

    pub fn update_material(
        &mut self,
        id: MaterialId,
        material: crate::GpuMaterial,
    ) -> SceneResult<()> {
        self.scene.update_material(id, material)
    }

    pub fn insert_light(&mut self, light: crate::GpuLight) -> LightId {
        self.scene.insert_light(light)
    }

    pub fn update_light(&mut self, id: LightId, light: crate::GpuLight) -> SceneResult<()> {
        self.scene.update_light(id, light)
    }

    pub fn insert_object(&mut self, desc: ObjectDescriptor) -> SceneResult<ObjectId> {
        self.scene.insert_object(desc)
    }

    pub fn update_object_transform(
        &mut self,
        id: ObjectId,
        transform: glam::Mat4,
    ) -> SceneResult<()> {
        self.scene.update_object_transform(id, transform)
    }

    pub fn render(
        &mut self,
        camera: &Camera,
        target: &wgpu::TextureView,
    ) -> HelioResult<()> {
        self.scene.update_camera(*camera);
        self.scene.flush();
        let frame_uniforms = FrameUniforms {
            ambient_color: [
                self.ambient_color[0],
                self.ambient_color[1],
                self.ambient_color[2],
                self.ambient_intensity,
            ],
            clear_color: self.clear_color,
            light_count: self.scene.gpu_scene().lights.len() as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        self.scene
            .gpu_scene()
            .queue
            .write_buffer(&self.frame_uniforms, 0, bytemuck::bytes_of(&frame_uniforms));

        let resources = self.scene.gpu_scene().resources();
        let mut texture_views = ArrayVec::<&wgpu::TextureView, MAX_TEXTURES>::new();
        let mut samplers = ArrayVec::<&wgpu::Sampler, MAX_TEXTURES>::new();
        for slot in 0..MAX_TEXTURES {
            texture_views.push(self.scene.texture_view_for_slot(slot));
            samplers.push(self.scene.texture_sampler_for_slot(slot));
        }
        let scene_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Helio Forward Scene BG"),
            layout: &self.scene_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.camera.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.frame_uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.instances.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.materials.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: resources.lights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.scene.material_texture_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureViewArray(texture_views.as_slice()),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::SamplerArray(samplers.as_slice()),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Helio Forward Encoder"),
            });
        let mesh_buffers = self.scene.mesh_buffers();
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Helio Forward Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: self.clear_color[0] as f64,
                            g: self.clear_color[1] as f64,
                            b: self.clear_color[2] as f64,
                            a: self.clear_color[3] as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &scene_bind_group, &[]);
            pass.set_vertex_buffer(0, mesh_buffers.vertices.slice(..));
            pass.set_index_buffer(mesh_buffers.indices.slice(..), wgpu::IndexFormat::Uint32);
            if resources.draw_count > 0 {
                if self.use_multi_draw_indirect {
                    pass.multi_draw_indexed_indirect(resources.indirect, 0, resources.draw_count);
                } else {
                    let stride = std::mem::size_of::<crate::DrawIndexedIndirectArgs>() as u64;
                    for draw_index in 0..resources.draw_count {
                        pass.draw_indexed_indirect(resources.indirect, draw_index as u64 * stride);
                    }
                }
            }
        }
        self.scene.gpu_scene().queue.submit([encoder.finish()]);
        self.scene.advance_frame();
        Ok(())
    }
}

fn create_depth_resources(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Helio Depth Texture"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}
