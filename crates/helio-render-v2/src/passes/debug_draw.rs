use std::sync::{Arc, Mutex};

use crate::debug_draw::DebugDrawBatch;
use crate::graph::{PassContext, PassResourceBuilder, RenderPass, ResourceHandle};
use crate::Result;

pub struct DebugDrawPass {
    pipeline: Arc<wgpu::RenderPipeline>,
    batch: Arc<Mutex<Option<DebugDrawBatch>>>,
}

impl DebugDrawPass {
    pub(crate) fn new(
        device: &wgpu::Device,
        global_layout: &wgpu::BindGroupLayout,
        target_format: wgpu::TextureFormat,
        batch: Arc<Mutex<Option<DebugDrawBatch>>>,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Debug Draw Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/passes/debug_draw.wgsl").into(),
            ),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Debug Draw Pipeline Layout"),
            bind_group_layouts: &[Some(global_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Debug Draw Pipeline"),
            layout: Some(&layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<crate::debug_draw::DebugDrawVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 12,
                            shader_location: 1,
                        },
                    ],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
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
        });

        Self {
            pipeline: Arc::new(pipeline),
            batch,
        }
    }
}

impl RenderPass for DebugDrawPass {
    fn name(&self) -> &str {
        "debug_draw"
    }

    fn declare_resources(&self, builder: &mut PassResourceBuilder) {
        builder.read(ResourceHandle::named("color_target"));
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let batch = self.batch.lock().unwrap().clone();
        let Some(batch) = batch else {
            return Ok(());
        };

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Debug Draw Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: ctx.depth_view,
                depth_ops: None,
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, ctx.global_bind_group, &[]);
        pass.set_vertex_buffer(0, batch.vertex_buffer.slice(..));
        pass.set_index_buffer(batch.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..batch.index_count, 0, 0..1);

        Ok(())
    }
}
