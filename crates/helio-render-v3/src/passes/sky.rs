/// Sky rendering pass — fullscreen triangle sampling the sky LUT.
/// Renders into pre_aa before any geometry passes.
use std::sync::Arc;
use crate::graph::pass::{RenderPass, PassContext};

pub struct SkyPass {
    pipeline:   Arc<wgpu::RenderPipeline>,
    bind_group: wgpu::BindGroup,
}

impl SkyPass {
    pub fn new(
        device:         &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_buffer:  &wgpu::Buffer,
        globals_buffer: &wgpu::Buffer,
        sky_lut_view:   &wgpu::TextureView,
        linear_sampler: &wgpu::Sampler,
    ) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sky_bgl"),
            entries: &[
                // 0: camera
                bgl_entry_uniform(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                // 1: globals
                bgl_entry_uniform(1, wgpu::ShaderStages::VERTEX_FRAGMENT),
                // 2: sky lut
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type:    wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled:   false,
                    }, count: None,
                },
                // 3: sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sky_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: globals_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(sky_lut_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(linear_sampler) },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("sky_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sky.wgsl").into()),
        });

        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("sky_pl_layout"),
            bind_group_layouts:   &[Some(&bgl)],
            immediate_size:       0,
        });

        let pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("sky_pipeline"),
            layout: Some(&pl_layout),
            vertex: wgpu::VertexState {
                module:      &shader,
                entry_point: Some("vs_main"),
                buffers:     &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology:          wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face:        wgpu::FrontFace::Ccw,
                cull_mode:         None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format:                wgpu::TextureFormat::Depth32Float,
                depth_write_enabled:   Some(false),
                depth_compare:         Some(wgpu::CompareFunction::LessEqual),
                stencil:               wgpu::StencilState::default(),
                bias:                  wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module:      &shader,
                entry_point: Some("fs_main"),
                targets:     &[Some(wgpu::ColorTargetState {
                    format:      surface_format,
                    blend:       None,
                    write_mask:  wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        }));

        SkyPass { pipeline, bind_group }
    }
}

impl RenderPass for SkyPass {
    fn execute(&mut self, ctx: &mut PassContext) {
        if ctx.sky_atmosphere.is_none() { return; }

        let mut rpass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("sky"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view:           &ctx.frame_tex.pre_aa_view,
                resolve_target: None,
                ops:            wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice:    None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &ctx.frame_tex.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.draw(0..3, 0..1);
    }
}

fn bgl_entry_uniform(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        }, count: None,
    }
}
