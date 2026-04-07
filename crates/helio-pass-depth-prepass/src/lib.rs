//! Depth prepass — writes depth buffer before main geometry pass.
//!
//! O(1) CPU: single `multi_draw_indexed_indirect` call regardless of scene size.
//!
//! # Vertex / Index Buffers
//!
//! This pass owns **no** mesh data.  The caller (render graph) must bind the
//! shared mesh vertex buffer (slot 0) and index buffer **before** this pass
//! executes, or the GPU draw will read from undefined memory.

use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

pub struct DepthPrepassPass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,
    bind_group_key: Option<(usize, usize)>,
}

impl DepthPrepassPass {
    /// Create the depth-prepass pipeline.
    ///
    /// * `depth_format` – format of the depth attachment (e.g. `Depth32Float`)
    pub fn new(device: &wgpu::Device, depth_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DepthPrepass Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/depth_prepass.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DepthPrepass PL"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // Vertex layout matches the shared mesh vertex buffer (stride = 40 bytes).
        //   offset  0 — position       (Float32x3, location 0)
        //   offset 12 — bitangent_sign (Float32,   location 1) — skipped here
        //   offset 16 — tex_coords0   (Float32x2, location 2)
        //   offset 24 — tex_coords1   (Float32x2) — skipped here
        //   offset 32 — normal        (Uint32,    location 3) — skipped here
        //   offset 36 — tangent       (Uint32,    location 4) — skipped here
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DepthPrepass Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 40, // PackedVertex: pos(12)+bitan(4)+uv0(8)+uv1(8)+normal(4)+tangent(4)
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
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            bind_group: None,
            bind_group_key: None,
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
        let main_scene = ctx.resources.main_scene.as_ref().ok_or_else(|| {
            helio_v3::Error::InvalidPassConfig(
                "DepthPrepass requires main_scene mesh buffers".to_string(),
            )
        })?;

        // Extract before the mutable encoder borrow.
        let camera_ptr = ctx.scene.camera as *const _ as usize;
        let instances_ptr = ctx.scene.instances as *const _ as usize;
        let key = (camera_ptr, instances_ptr);
        if self.bind_group_key != Some(key) {
            log::debug!("DepthPrepass: rebuilding bind group (buffer pointers changed)");
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("DepthPrepass BG"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ctx.scene.camera.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: ctx.scene.instances.as_entire_binding(),
                    },
                ],
            }));
            self.bind_group_key = Some(key);
        }
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
            multiview_mask: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        pass.set_vertex_buffer(0, main_scene.mesh_buffers.vertices.slice(..));
        pass.set_index_buffer(
            main_scene.mesh_buffers.indices.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        #[cfg(not(target_arch = "wasm32"))]
        pass.multi_draw_indexed_indirect(indirect, 0, draw_count);
        #[cfg(target_arch = "wasm32")]
        for i in 0..draw_count {
            pass.draw_indexed_indirect(indirect, i as u64 * 20);
        }
        Ok(())
    }
}
