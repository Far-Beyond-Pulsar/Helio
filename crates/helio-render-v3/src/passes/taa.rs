/// TAA (Temporal Anti-Aliasing) pass — ping-pong history, velocity reprojection.
/// Resolves pre_aa (current) + history → surface.
use std::sync::Arc;
use crate::graph::pass::{RenderPass, PassContext};

pub struct TaaPass {
    pipeline:      Arc<wgpu::RenderPipeline>,
    bind_group_a:  wgpu::BindGroup,  // uses history_a as accumulator
    bind_group_b:  wgpu::BindGroup,  // uses history_b as accumulator
    frame_parity:  bool,
}

impl TaaPass {
    pub fn new(
        device:          &wgpu::Device,
        surface_format:  wgpu::TextureFormat,
        pre_aa_view:     &wgpu::TextureView,
        history_a_view:  &wgpu::TextureView,
        history_b_view:  &wgpu::TextureView,
        velocity_view:   &wgpu::TextureView,
        linear_sampler:  &wgpu::Sampler,
    ) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("taa_bgl"),
            entries: &[
                tex_entry(0), tex_entry(1), tex_entry(2),
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });

        let make_bg = |current_history: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("taa_bg"), layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(pre_aa_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(current_history) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(velocity_view) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(linear_sampler) },
                ],
            })
        };

        let bind_group_a = make_bg(history_a_view);
        let bind_group_b = make_bg(history_b_view);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("taa_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/taa.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("taa_layout"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0,
        });
        let pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("taa"),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), buffers: &[], compilation_options: Default::default() },
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader, entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState { format: surface_format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None, cache: None,
        }));

        TaaPass { pipeline, bind_group_a, bind_group_b, frame_parity: false }
    }
}

impl RenderPass for TaaPass {
    fn execute(&mut self, ctx: &mut PassContext) {
        let bg = if self.frame_parity { &self.bind_group_a } else { &self.bind_group_b };
        self.frame_parity = !self.frame_parity;

        let mut rpass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("taa"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.surface_view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                depth_slice: None,
            })],
            ..Default::default()
        });
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, bg, &[]);
        rpass.draw(0..3, 0..1);
    }
}

fn tex_entry(b: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None }
}
