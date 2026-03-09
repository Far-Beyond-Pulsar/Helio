//! God rays (crepuscular rays) pass.
//!
//! Radial blur from the sun's screen-space position.  Only sky pixels
//! (depth ≥ 0.9999) contribute — scene geometry occludes the rays.
//! Result is additively blended onto the HDR intermediate buffer.

use std::sync::Arc;
use wgpu::util::DeviceExt;
use crate::Result;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GodRayUniforms {
    /// Sun position in [0,1]×[0,1] screen space (or outside for offscreen sun).
    pub sun_screen_pos: [f32; 2],
    pub density:        f32,
    pub decay:          f32,
    pub weight:         f32,
    pub exposure:       f32,
    pub num_samples:    u32,
    pub _pad:           u32,
}

impl Default for GodRayUniforms {
    fn default() -> Self {
        Self {
            sun_screen_pos: [0.5, 0.3],
            density:        0.97,
            decay:          0.95,
            weight:         0.4,
            exposure:       0.1,
            num_samples:    64,
            _pad:           0,
        }
    }
}

pub struct GodRaysPass {
    pipeline:           Arc<wgpu::RenderPipeline>,
    bind_group_layout:  wgpu::BindGroupLayout,
    sampler:            wgpu::Sampler,
    pub uniform_buffer: wgpu::Buffer,
    bind_group:         Option<wgpu::BindGroup>,
}

impl GodRaysPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GodRays BGL"),
            entries: &[
                // 0: scene_color (HDR buffer)
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    }, count: None,
                },
                // 1: scene_depth
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    }, count: None,
                },
                // 2: linear sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // 3: GodRayUniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }, count: None,
                },
            ],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:      Some("GodRays Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("GodRay Uniform Buffer"),
            contents: bytemuck::bytes_of(&GodRayUniforms::default()),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("GodRays Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl as &wgpu::BindGroupLayout)],
            immediate_size:     0,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("GodRays Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/passes/god_rays.wgsl").into()
            ),
        });

        let additive = Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation:  wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        });

        let pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("GodRays Pipeline"),
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

    /// Build / rebuild bind group. Call after construction and on resize.
    pub fn create_bind_group(
        &mut self,
        device:     &wgpu::Device,
        hdr_view:   &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
    ) {
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("GodRays Bind Group"),
            layout:  &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(hdr_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: self.uniform_buffer.as_entire_binding() },
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
            label: Some("GodRays Pass"),
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
