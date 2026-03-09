//! Screen-Space Reflections pass.
//!
//! View-space ray march from reflection vector with exponential step growth.
//! Binary search refinement for accurate hit positions.
//! Confidence fades by roughness, edge distance, and ray length.
//!
//! Result is composited additively onto the HDR intermediate buffer.

use std::sync::Arc;
use wgpu::util::DeviceExt;
use crate::Result;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SsrUniforms {
    pub view:       [[f32; 4]; 4],
    pub proj:       [[f32; 4]; 4],
    pub proj_inv:   [[f32; 4]; 4],
    pub max_steps:  u32,
    pub max_dist:   f32,
    pub thickness:  f32,
    pub _pad:       f32,
}

impl Default for SsrUniforms {
    fn default() -> Self {
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        Self { view: identity, proj: identity, proj_inv: identity, max_steps: 64, max_dist: 50.0, thickness: 0.5, _pad: 0.0 }
    }
}

pub struct SsrPass {
    pipeline:           Arc<wgpu::RenderPipeline>,
    bind_group_layout:  wgpu::BindGroupLayout,
    sampler:            wgpu::Sampler,
    pub uniform_buffer: wgpu::Buffer,
    bind_group:         Option<wgpu::BindGroup>,
}

impl SsrPass {
    pub fn new(device: &wgpu::Device) -> Self {
        // Group 0: camera uniform (re-uses global bind group slot 0)
        // Group 1: gbuf_normal, gbuf_orm, gbuf_depth, scene_color, samp, SsrUniforms
        // — We use a single combined layout here for simplicity.
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSR BGL"),
            entries: &[
                // 0: gbuf_normal
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    }, count: None,
                },
                // 1: gbuf_orm
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    }, count: None,
                },
                // 2: gbuf_depth (non-filterable, depth sampled with textureLoad)
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    }, count: None,
                },
                // 3: scene_color (HDR buffer)
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    }, count: None,
                },
                // 4: linear sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // 5: SsrUniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 5, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }, count: None,
                },
            ],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:      Some("SSR Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("SSR Uniform Buffer"),
            contents: bytemuck::bytes_of(&SsrUniforms::default()),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("SSR Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl as &wgpu::BindGroupLayout)],
            immediate_size:     0,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("SSR Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/passes/ssr.wgsl").into()
            ),
        });

        // Additive blend: SSR result composites onto the HDR buffer.
        let additive = Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation:  wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        });

        let pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("SSR Pipeline"),
            layout: Some(&pipeline_layout),
            cache:  None,
            vertex: wgpu::VertexState {
                module: &shader, entry_point: Some("vs_main"),
                buffers: &[], compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader, entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format:     wgpu::TextureFormat::Rgba16Float,
                    blend:      additive,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive:      wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil:  None,
            multisample:    wgpu::MultisampleState::default(),
            multiview_mask: None,
        }));

        Self { pipeline, bind_group_layout: bgl, sampler, uniform_buffer, bind_group: None }
    }

    /// Build / rebuild the bind group.  `hdr_view` is used both as `scene_color` input
    /// and as the render attachment output — this is fine because the render pass reads
    /// via the sampler (before the fragment stage overwrites via the attachment).
    /// Call after construction and on every resize.
    pub fn create_bind_group(
        &mut self,
        device:         &wgpu::Device,
        gbuf_normal:    &wgpu::TextureView,
        gbuf_orm:       &wgpu::TextureView,
        depth_view:     &wgpu::TextureView,
        hdr_view:       &wgpu::TextureView,
    ) {
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("SSR Bind Group"),
            layout:  &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(gbuf_normal) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(gbuf_orm) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(hdr_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 5, resource: self.uniform_buffer.as_entire_binding() },
            ],
        }));
    }

    pub fn execute(
        &self,
        encoder:  &mut wgpu::CommandEncoder,
        hdr_view: &wgpu::TextureView,
    ) -> Result<()> {
        let bg = match &self.bind_group { Some(b) => b, None => return Ok(()) };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSR Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: hdr_view, resolve_target: None, depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes:         None,
            occlusion_query_set:      None,
            multiview_mask:           None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}
