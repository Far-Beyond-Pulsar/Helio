/// Billboard pass — camera-facing quads for particles/sprites.
///
/// Sprites are uploaded as a flat array of `BillboardInstance` structs.
/// All instances are drawn in a single `draw` call (one vertex buffer, no index buffer).
/// The vertex shader expands each instance to 2 triangles (6 vertices) using gl_VertexIndex.
use std::sync::Arc;
use crate::graph::pass::{RenderPass, PassContext};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BillboardInstance {
    pub position: [f32; 3],
    pub size:     f32,
    pub color:    [f32; 4],
    pub uv_rect:  [f32; 4],  // x, y, w, h in atlas UV space
}

#[derive(Clone, Debug)]
pub struct BillboardConfig {
    pub max_instances: u32,
}

pub struct BillboardPass {
    pipeline:       Arc<wgpu::RenderPipeline>,
    instance_buf:   wgpu::Buffer,
    bind_group:     wgpu::BindGroup,
    max_instances:  u32,
    last_count:     u32,
    staging:        Vec<BillboardInstance>,
}

impl BillboardPass {
    pub fn new(
        device:         &wgpu::Device,
        config:         BillboardConfig,
        surface_format: wgpu::TextureFormat,
        camera_buffer:  &wgpu::Buffer,
        atlas_view:     &wgpu::TextureView,
        linear_sampler: &wgpu::Sampler,
    ) -> Self {
        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("billboard_instances"),
            size:               config.max_instances as u64 * std::mem::size_of::<BillboardInstance>() as u64,
            usage:              wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("billboard_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("billboard_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(atlas_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(linear_sampler) },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("billboard_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/billboard.wgsl").into()),
        });
        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("billboard_layout"),
            bind_group_layouts:   &[Some(&bgl)],
            immediate_size:       0,
        });

        let instance_attrs: Vec<wgpu::VertexAttribute> = vec![
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0,  shader_location: 0 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32,   offset: 12, shader_location: 1 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 2 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 3 },
        ];
        let buf_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<BillboardInstance>() as u64,
            step_mode:    wgpu::VertexStepMode::Instance,
            attributes:   &instance_attrs,
        };

        let pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("billboard"),
            layout: Some(&pl_layout),
            vertex: wgpu::VertexState {
                module: &shader, entry_point: Some("vs_main"),
                buffers: &[buf_layout], compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(), bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader, entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend:  Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None, cache: None,
        }));

        BillboardPass { pipeline, instance_buf, bind_group, max_instances: config.max_instances, last_count: 0, staging: Vec::new() }
    }

    pub fn set_instances(&mut self, instances: Vec<BillboardInstance>) {
        self.staging = instances;
    }
}

impl RenderPass for BillboardPass {
    fn execute(&mut self, ctx: &mut PassContext) {
        if self.staging.is_empty() { return; }

        let count = self.staging.len().min(self.max_instances as usize) as u32;
        ctx.queue.write_buffer(&self.instance_buf, 0, bytemuck::cast_slice(&self.staging[..count as usize]));
        self.last_count = count;
        self.staging.clear();

        let mut rpass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("billboard"),
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
        rpass.set_vertex_buffer(0, self.instance_buf.slice(..));
        rpass.draw(0..6, 0..self.last_count);
    }
}
