use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use wgpu::util::DeviceExt;

// ── Vertex layout ─────────────────────────────────────────────────────────────
// 36 bytes: position(12) | normal(12) | color(12)

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CubeVertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
}

fn v(position: [f32; 3], normal: [f32; 3], color: [f32; 3]) -> CubeVertex {
    CubeVertex {
        position,
        normal,
        color,
    }
}

// Six faces, four vertices each (24 total), wound counter-clockwise when viewed
// from outside. Each face gets a distinct saturated color so orientation is
// immediately obvious.
fn cube_vertices() -> [CubeVertex; 24] {
    let r = [1.0_f32, 0.25, 0.25]; // +X  red
    let c = [0.25_f32, 1.0, 1.0]; // -X  cyan
    let g = [0.25_f32, 1.0, 0.25]; // +Y  green
    let m = [1.0_f32, 0.25, 1.0]; // -Y  magenta
    let b = [0.3_f32, 0.5, 1.0]; // +Z  blue
    let y = [1.0_f32, 1.0, 0.25]; // -Z  yellow

    [
        // +X face
        v([0.5, -0.5, -0.5], [1., 0., 0.], r),
        v([0.5, 0.5, -0.5], [1., 0., 0.], r),
        v([0.5, 0.5, 0.5], [1., 0., 0.], r),
        v([0.5, -0.5, 0.5], [1., 0., 0.], r),
        // -X face
        v([-0.5, -0.5, 0.5], [-1., 0., 0.], c),
        v([-0.5, 0.5, 0.5], [-1., 0., 0.], c),
        v([-0.5, 0.5, -0.5], [-1., 0., 0.], c),
        v([-0.5, -0.5, -0.5], [-1., 0., 0.], c),
        // +Y face
        v([-0.5, 0.5, -0.5], [0., 1., 0.], g),
        v([-0.5, 0.5, 0.5], [0., 1., 0.], g),
        v([0.5, 0.5, 0.5], [0., 1., 0.], g),
        v([0.5, 0.5, -0.5], [0., 1., 0.], g),
        // -Y face
        v([-0.5, -0.5, 0.5], [0., -1., 0.], m),
        v([-0.5, -0.5, -0.5], [0., -1., 0.], m),
        v([0.5, -0.5, -0.5], [0., -1., 0.], m),
        v([0.5, -0.5, 0.5], [0., -1., 0.], m),
        // +Z face
        v([-0.5, -0.5, 0.5], [0., 0., 1.], b),
        v([0.5, -0.5, 0.5], [0., 0., 1.], b),
        v([0.5, 0.5, 0.5], [0., 0., 1.], b),
        v([-0.5, 0.5, 0.5], [0., 0., 1.], b),
        // -Z face
        v([0.5, -0.5, -0.5], [0., 0., -1.], y),
        v([-0.5, -0.5, -0.5], [0., 0., -1.], y),
        v([-0.5, 0.5, -0.5], [0., 0., -1.], y),
        v([0.5, 0.5, -0.5], [0., 0., -1.], y),
    ]
}

// 6 faces × 2 triangles × 3 vertices = 36 indices
fn cube_indices() -> [u16; 36] {
    let mut idx = [0u16; 36];
    for face in 0..6u16 {
        let b = face * 4;
        let o = (face * 6) as usize;
        idx[o] = b;
        idx[o + 1] = b + 1;
        idx[o + 2] = b + 2;
        idx[o + 3] = b;
        idx[o + 4] = b + 2;
        idx[o + 5] = b + 3;
    }
    idx
}

// ── Pass struct ───────────────────────────────────────────────────────────────

pub struct SimpleCubePass {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,
    bind_group_key: Option<usize>,
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    #[allow(dead_code)]
    surface_format: wgpu::TextureFormat,
}

impl SimpleCubePass {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SimpleCube Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/simple_cube.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SimpleCube BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SimpleCube Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SimpleCube Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 36,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 12,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 24,
                            shader_location: 2,
                        },
                    ],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let vertices = cube_vertices();
        let indices = cube_indices();

        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SimpleCube VB"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SimpleCube IB"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            pipeline,
            bgl,
            bind_group: None,
            bind_group_key: None,
            vertex_buf,
            index_buf,
            surface_format,
        }
    }
}

impl RenderPass for SimpleCubePass {
    fn name(&self) -> &'static str {
        "SimpleCube"
    }

    fn prepare(&mut self, _ctx: &PrepareContext) -> HelioResult<()> {
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // Rebuild camera bind group when the camera buffer pointer changes.
        let camera_ptr = ctx.scene.camera as *const _ as usize;
        if self.bind_group_key != Some(camera_ptr) {
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SimpleCube BG"),
                layout: &self.bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ctx.scene.camera.as_entire_binding(),
                }],
            }));
            self.bind_group_key = Some(camera_ptr);
        }

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SimpleCube"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.01,
                        g: 0.01,
                        b: 0.02,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
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
            multiview_mask: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..36, 0, 0..1);
        Ok(())
    }
}

