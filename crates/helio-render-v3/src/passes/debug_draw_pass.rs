/// Debug draw pass — renders line primitives from the DebugDrawBatch.
/// Single `draw` call. Buffer overwritten each frame via queue.write_buffer.
use std::sync::Arc;
use crate::{
    graph::pass::{RenderPass, PassContext},
    debug_draw::DebugVertex,
};

pub struct DebugDrawPass {
    pipeline:    Arc<wgpu::RenderPipeline>,
    vertex_buf:  wgpu::Buffer,
    bind_group:  wgpu::BindGroup,
    max_verts:   u32,
}

impl DebugDrawPass {
    pub fn new(
        device:         &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_buffer:  &wgpu::Buffer,
        max_vertices:   u32,
    ) -> Self {
        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("debug_vertices"),
            size:               max_vertices as u64 * std::mem::size_of::<DebugVertex>() as u64,
            usage:              wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("debug_draw_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("debug_draw_bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("debug_draw_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/debug_draw.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("debug_draw_layout"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0,
        });

        let vert_attrs = [
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0,  shader_location: 0 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32,   offset: 12, shader_location: 1 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 2 },
        ];

        let pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("debug_draw"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader, entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<DebugVertex>() as u64,
                    step_mode:    wgpu::VertexStepMode::Vertex,
                    attributes:   &vert_attrs,
                }],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format:              wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare:       Some(wgpu::CompareFunction::Always), // always-on debug overlay
                stencil:             Default::default(), bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader, entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format:     surface_format,
                    blend:      Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None, cache: None,
        }));

        DebugDrawPass { pipeline, vertex_buf, bind_group, max_verts: max_vertices }
    }
}

impl RenderPass for DebugDrawPass {
    fn execute(&mut self, ctx: &mut PassContext) {
        let batch = match ctx.debug_batch { Some(b) => b, None => return };
        if batch.is_empty() { return; }

        let count = batch.vertex_count().min(self.max_verts);
        ctx.queue.write_buffer(&self.vertex_buf, 0, bytemuck::cast_slice(&batch.vertices[..count as usize]));

        let mut rpass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("debug_draw"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &ctx.frame_tex.pre_aa_view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &ctx.frame_tex.depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        rpass.draw(0..count, 0..1);
    }
}
