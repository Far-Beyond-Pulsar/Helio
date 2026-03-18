//! Depth prepass — writes depth buffer before main geometry pass.
//!
//! O(1) CPU: single `multi_draw_indexed_indirect` call regardless of scene size.
//!
//! # Vertex / Index Buffers
//!
//! This pass owns **no** mesh data.  The caller (render graph) must bind the
//! shared mesh vertex buffer (slot 0) and index buffer **before** this pass
//! executes, or the GPU draw will read from undefined memory.

use helio_v3::{RenderPass, PassContext, PrepareContext, Result as HelioResult};

pub struct DepthPrepassPass {
    pipeline: wgpu::RenderPipeline,
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

impl DepthPrepassPass {
    /// Create the depth-prepass pipeline.
    ///
    /// * `camera_buf`    – scene camera uniform buffer (view_proj + position)
    /// * `instances_buf` – per-instance transform storage buffer
    /// * `depth_format`  – format of the depth attachment (e.g. `Depth32Float`)
    pub fn new(
        device: &wgpu::Device,
        camera_buf: &wgpu::Buffer,
        instances_buf: &wgpu::Buffer,
        depth_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DepthPrepass Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/depth_prepass.wgsl").into(),
            ),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("DepthPrepass BGL"),
                entries: &[
                    // binding 0: camera uniform (VERTEX)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 1: per-instance transforms (VERTEX, read-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DepthPrepass BG"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: instances_buf.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DepthPrepass PL"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Vertex layout matches the shared mesh vertex buffer (stride = 32 bytes).
        //   offset  0 — position       (Float32x3, location 0)
        //   offset 12 — bitangent_sign (Float32,   location 1) — skipped here
        //   offset 16 — tex_coords     (Float32x2, location 2)
        //   offset 24 — normal         (Uint32,    location 3) — skipped here
        //   offset 28 — tangent        (Uint32,    location 4) — skipped here
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DepthPrepass Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 32,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 16,
                            shader_location: 2,
                        },
                    ],
                }],
            },
            // Depth-only: no fragment stage, no color outputs.
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            bind_group,
        }
    }
}

impl RenderPass for DepthPrepassPass {
    fn name(&self) -> &'static str {
        "DepthPrepass"
    }

    fn prepare(&mut self, _ctx: &PrepareContext) -> HelioResult<()> {
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // O(1): single multi_draw_indexed_indirect — no CPU loop over draw calls.
        let draw_count = ctx.scene.draw_count;
        if draw_count == 0 {
            return Ok(());
        }

        // Extract before the mutable encoder borrow.
        let indirect = ctx.scene.indirect;

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("DepthPrepass"),
            // Depth-only pass: zero color attachments.
            color_attachments: &[],
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
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        // TODO: Caller (render graph) must call set_vertex_buffer(0, mesh_vb, ..)
        //       and set_index_buffer(mesh_ib, IndexFormat::Uint32) before this pass.
        pass.multi_draw_indexed_indirect(indirect, 0, draw_count);
        Ok(())
    }
}
