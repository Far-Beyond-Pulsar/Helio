use std::num::NonZeroU32;

use bytemuck::{Pod, Zeroable};
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

use crate::material::MAX_TEXTURES;

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

pub struct ForwardPass {
    pipeline: wgpu::RenderPipeline,
    scene_bind_group_layout: wgpu::BindGroupLayout,
    frame_uniforms: wgpu::Buffer,
    scene_bind_group: Option<wgpu::BindGroup>,
    scene_bind_group_version: Option<u64>,
    use_multi_draw_indirect: bool,
}

impl ForwardPass {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let texture_array_count =
            NonZeroU32::new(MAX_TEXTURES as u32).expect("non-zero texture table size");
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
                    format: surface_format,
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

        Self {
            pipeline,
            scene_bind_group_layout,
            frame_uniforms,
            scene_bind_group: None,
            scene_bind_group_version: None,
            use_multi_draw_indirect: device
                .features()
                .contains(wgpu::Features::MULTI_DRAW_INDIRECT),
        }
    }

    fn ensure_scene_bind_group(&mut self, ctx: &PassContext<'_>) {
        let main_scene = ctx
            .frame
            .main_scene
            .as_ref()
            .expect("Helio ForwardPass requires main_scene frame resources");
        let needs_rebuild = self.scene_bind_group_version != Some(main_scene.material_textures.version)
            || self.scene_bind_group.is_none();
        if needs_rebuild {
            self.scene_bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Helio Forward Scene BG"),
                layout: &self.scene_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ctx.scene.camera.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.frame_uniforms.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: ctx.scene.instances.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: ctx.scene.materials.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: ctx.scene.lights.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: main_scene
                            .material_textures
                            .material_textures
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureViewArray(
                            main_scene.material_textures.texture_views,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::SamplerArray(
                            main_scene.material_textures.samplers,
                        ),
                    },
                ],
            }));
            self.scene_bind_group_version = Some(main_scene.material_textures.version);
        }
    }
}

impl RenderPass for ForwardPass {
    fn name(&self) -> &'static str {
        "Forward"
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let main_scene = ctx
            .frame_resources
            .main_scene
            .as_ref()
            .expect("Helio ForwardPass requires main_scene frame resources");
        let frame_uniforms = FrameUniforms {
            ambient_color: [
                main_scene.ambient_color[0],
                main_scene.ambient_color[1],
                main_scene.ambient_color[2],
                main_scene.ambient_intensity,
            ],
            clear_color: main_scene.clear_color,
            light_count: ctx.scene.lights.len() as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        ctx.write_buffer(&self.frame_uniforms, 0, bytemuck::bytes_of(&frame_uniforms));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let main_scene = ctx
            .frame
            .main_scene
            .as_ref()
            .expect("Helio ForwardPass requires main_scene frame resources");
        self.ensure_scene_bind_group(ctx);
        let scene_bind_group = self
            .scene_bind_group
            .as_ref()
            .expect("forward pass scene bind group");
        let pipeline = &self.pipeline;
        let use_multi_draw_indirect = self.use_multi_draw_indirect;
        let clear_color = main_scene.clear_color;
        let draw_count = ctx.scene.draw_count;
        let indirect = ctx.scene.indirect;
        let color_attachments = [Some(wgpu::RenderPassColorAttachment {
            view: ctx.target,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: clear_color[0] as f64,
                    g: clear_color[1] as f64,
                    b: clear_color[2] as f64,
                    a: clear_color[3] as f64,
                }),
                store: wgpu::StoreOp::Store,
            },
        })];
        let render_pass = wgpu::RenderPassDescriptor {
            label: Some("Helio Forward Pass"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: ctx.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        };
        let mut pass = ctx.begin_render_pass(&render_pass);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, scene_bind_group, &[]);
        pass.set_vertex_buffer(0, main_scene.mesh_buffers.vertices.slice(..));
        pass.set_index_buffer(
            main_scene.mesh_buffers.indices.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        if draw_count > 0 {
            if use_multi_draw_indirect {
                pass.multi_draw_indexed_indirect(indirect, 0, draw_count);
            } else {
                let stride = std::mem::size_of::<crate::DrawIndexedIndirectArgs>() as u64;
                for draw_index in 0..draw_count {
                    pass.draw_indexed_indirect(indirect, draw_index as u64 * stride);
                }
            }
        }
        Ok(())
    }
}
