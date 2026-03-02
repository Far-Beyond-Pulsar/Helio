//! Billboard render pass - instanced alpha-blended camera-facing quads

use crate::graph::{RenderPass, PassContext, PassResourceBuilder, ResourceHandle};
use crate::{Result, Error};
use std::sync::{Arc, atomic::Ordering};

/// Renders camera-facing billboard quads using instanced drawing
pub struct BillboardPass {
    vertex_buffer: Arc<wgpu::Buffer>,
    index_buffer: Arc<wgpu::Buffer>,
    instance_buffer: Arc<wgpu::Buffer>,
    instance_count: Arc<std::sync::atomic::AtomicU32>,
    surface_format: wgpu::TextureFormat,
    /// Bind group 1: sprite texture + sampler (may be a white fallback)
    sprite_bind_group: Arc<wgpu::BindGroup>,
    sprite_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    pipeline: Option<Arc<wgpu::RenderPipeline>>,
}

impl BillboardPass {
    pub fn new(
        vertex_buffer: Arc<wgpu::Buffer>,
        index_buffer: Arc<wgpu::Buffer>,
        instance_buffer: Arc<wgpu::Buffer>,
        instance_count: Arc<std::sync::atomic::AtomicU32>,
        surface_format: wgpu::TextureFormat,
        sprite_bind_group: Arc<wgpu::BindGroup>,
        sprite_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    ) -> Self {
        Self {
            vertex_buffer,
            index_buffer,
            instance_buffer,
            instance_count,
            surface_format,
            sprite_bind_group,
            sprite_bind_group_layout,
            pipeline: None,
        }
    }

    fn build_pipeline(
        &mut self,
        device: &wgpu::Device,
        global_layout: &wgpu::BindGroupLayout,
        target_format: wgpu::TextureFormat,
    ) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Billboard Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/passes/billboard.wgsl").into(),
            ),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Billboard Pipeline Layout"),
            bind_group_layouts: &[global_layout, &self.sprite_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Billboard Pipeline"),
            layout: Some(&layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[
                    // Slot 0: quad vertex (position + uv, per-vertex)
                    wgpu::VertexBufferLayout {
                        array_stride: 16,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 8,
                                shader_location: 1,
                            },
                        ],
                    },
                    // Slot 1: per-instance data
                    wgpu::VertexBufferLayout {
                        array_stride: 48, // GpuBillboardInstance: 12 × f32 = 48 bytes
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            // position (vec3) + pad (f32)
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 0,
                                shader_location: 2,
                            },
                            // scale (vec2) + screen_scale (u32) + pad (u32)
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 16,
                                shader_location: 3,
                            },
                            // color (vec4)
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 32,
                                shader_location: 4,
                            },
                        ],
                    },
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,  // Billboards render without depth buffer
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        self.pipeline = Some(Arc::new(pipeline));
    }
}

impl RenderPass for BillboardPass {
    fn name(&self) -> &str {
        "billboards"
    }

    fn declare_resources(&self, builder: &mut PassResourceBuilder) {
        // Reads color target written by GeometryPass → enforces geometry-before-billboard order
        builder.read(ResourceHandle::named("color_target"));
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let count = self.instance_count.load(Ordering::Relaxed);
        if count == 0 {
            return Ok(());
        }

        // Lazy pipeline build on first use
        if self.pipeline.is_none() {
            let format = self.surface_format;
            self.build_pipeline(
                ctx.resources.device(),
                &ctx.resources.bind_group_layouts.global,
                format,
            );
        }

        let pipeline = self.pipeline.as_ref().ok_or_else(|| {
            Error::Pipeline("BillboardPass pipeline build failed".into())
        })?;

        // Extract refs before the mutable encoder borrow from the render pass
        let target = ctx.target;
        let global_bg = ctx.global_bind_group;

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Billboard Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, global_bg, &[]);
        pass.set_bind_group(1, &self.sprite_bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..6, 0, 0..count);

        Ok(())
    }
}
