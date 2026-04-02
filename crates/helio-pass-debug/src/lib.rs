//! Debug draw pass — line segments and points for dev visualization.
//!
//! CPU uploads line vertex data via `update_lines()`. GPU renders with a
//! single draw call. Completely no-op when no debug geometry is queued.
//! O(1) CPU execution — single `draw(0..vertex_count, 0..1)`.

use bytemuck::{Pod, Zeroable};
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

const MAX_DEBUG_VERTS: u32 = 65536;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DebugCameraUniform {
    pub view_proj: [[f32; 4]; 4],
}

/// A single debug vertex — position + colour.
///
/// Matches the vertex input layout in `debug_draw.wgsl`:
///   location(0) position: vec3<f32>
///   location(1) color:    vec4<f32>
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DebugVertex {
    pub position: [f32; 3],
    pub _pad: f32, // aligns color to 16-byte boundary
    pub color: [f32; 4],
}

pub struct DebugPass {
    pipeline_depth: wgpu::RenderPipeline,
    pipeline_no_depth: wgpu::RenderPipeline,
    #[allow(dead_code)]
    bgl: wgpu::BindGroupLayout,
    camera_buf: wgpu::Buffer,
    bind_group: Option<wgpu::BindGroup>,
    bind_group_key: Option<usize>,
    vertex_buf: wgpu::Buffer,
    pub vertex_count: u32,
    depth_test_enabled: bool,
}

impl DebugPass {
    /// Create the debug pass.
    ///
    /// - `camera_buf`    — camera uniform (must match `Camera` struct in debug_draw.wgsl)
    /// - `target_format` — colour attachment format
    /// - `depth_test`    — whether to reject fragments behind scene depth
    pub fn new(
        device: &wgpu::Device,
        camera_buf: &wgpu::Buffer,
        target_format: wgpu::TextureFormat,
        depth_test: bool,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Debug Draw Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/debug_draw.wgsl").into()),
        });

        // Group 0: camera uniform (binding 0)
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Debug Draw BGL"),
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
            label: Some("Debug Draw PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        // Vertex buffer: position (vec3 → Float32x3) + pad + color (vec4 → Float32x4)
        // Stride 32 bytes: [pos.x, pos.y, pos.z, _pad, col.r, col.g, col.b, col.a]
        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Vertex Buffer"),
            size: (MAX_DEBUG_VERTS as usize * std::mem::size_of::<DebugVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline_depth = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Debug Draw Pipeline Depth"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<DebugVertex>() as u64, // 32
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 16,
                            shader_location: 1,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let pipeline_no_depth = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Debug Draw Pipeline NoDepth"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<DebugVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 16,
                            shader_location: 1,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline_depth,
            pipeline_no_depth,
            bgl,
            camera_buf: camera_buf.clone(),
            bind_group: None,
            bind_group_key: None,
            vertex_buf,
            vertex_count: 0,
            depth_test_enabled: depth_test,
        }
    }

    /// Upload line vertices. Each pair of vertices forms one line segment.
    ///
    /// Call this from the game loop before `execute()`. O(n) upload but O(1) GPU draw.
    pub fn update_lines(&mut self, queue: &wgpu::Queue, verts: &[DebugVertex]) {
        let count = verts.len().min(MAX_DEBUG_VERTS as usize);
        if count > 0 {
            helio_v3::upload::write_buffer(
                queue,
                &self.vertex_buf,
                0,
                bytemuck::cast_slice(&verts[..count]),
            );
        }
        self.vertex_count = count as u32;
    }

    /// Clear all queued debug geometry (no-ops the next execute).
    pub fn clear(&mut self) {
        self.vertex_count = 0;
    }
}

impl DebugPass {
    /// Execute applying this debug draw pass to a specific LHS render target.
    pub fn execute_on_target(
        &mut self,
        ctx: &mut PassContext,
        target: &wgpu::TextureView,
    ) -> HelioResult<()> {
        // O(1): single draw call — completely skipped when nothing is queued.
        if self.vertex_count == 0 {
            return Ok(());
        }

        // Always sample from the internal depth buffer (ctx.depth) so debug lines
        // are occluded by scene geometry, not by post-upscale sky proxies.
        let camera_key = &self.camera_buf as *const _ as usize;
        if self.bind_group_key != Some(camera_key) {
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Debug Draw BG"),
                layout: &self.bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buf.as_entire_binding(),
                }],
            }));
            self.bind_group_key = Some(camera_key);
        }

        let depth_attachment = if self.depth_test_enabled {
            let depth_view = if let Some(frd) = ctx.resources.full_res_depth {
                frd
            } else {
                ctx.depth
            };
            Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            })
        } else {
            None
        };

        let color_attachment = wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        };
        let color_attachments = [Some(color_attachment)];
        let desc = wgpu::RenderPassDescriptor {
            label: Some("DebugDraw"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: depth_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };

        let mut pass = ctx.encoder.begin_render_pass(&desc);
        if self.depth_test_enabled {
            pass.set_pipeline(&self.pipeline_depth);
        } else {
            pass.set_pipeline(&self.pipeline_no_depth);
        }
        pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        pass.draw(0..self.vertex_count, 0..1); // O(1) — single draw call
        Ok(())
    }
}

impl RenderPass for DebugPass {
    fn name(&self) -> &'static str {
        "DebugDraw"
    }

    fn prepare(&mut self, _ctx: &PrepareContext) -> HelioResult<()> {
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        self.execute_on_target(ctx, ctx.target)
    }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

