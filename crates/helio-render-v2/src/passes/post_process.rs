//! Final post-processing pass.
//!
//! Reads from the HDR Rgba16Float intermediate buffer and writes LDR output to
//! either the pre-AA texture (if AA is enabled) or directly to the swapchain.
//!
//! Applies in order:
//!   1. ACES filmic tonemap + exposure
//!   2. Saturation + contrast colour grading
//!   3. Radial vignette
//!   4. Animated film grain
//!   5. Chromatic aberration (barrel-split R/B channels)

use std::sync::Arc;
use crate::Result;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PostProcessUniforms {
    pub exposure:          f32,
    pub saturation:        f32,
    pub contrast:          f32,
    pub vignette_strength: f32,
    pub vignette_radius:   f32,
    pub grain_strength:    f32,
    pub ca_strength:       f32,
    pub frame:             u32,
}

impl Default for PostProcessUniforms {
    fn default() -> Self {
        Self {
            exposure:          1.0,
            saturation:        1.05,
            contrast:          1.05,
            vignette_strength: 0.35,
            vignette_radius:   0.75,
            grain_strength:    0.025,
            ca_strength:       0.0015,
            frame:             0,
        }
    }
}

pub struct PostProcessPass {
    pipeline:           Arc<wgpu::RenderPipeline>,
    bind_group_layout:  wgpu::BindGroupLayout,
    sampler:            wgpu::Sampler,
    pub uniform_buffer: wgpu::Buffer,
    /// Cached bind group.  Rebuilt when the HDR source view changes (resize).
    bind_group:         Option<wgpu::BindGroup>,
}

impl PostProcessPass {
    /// Create the pass.  `output_format` must match the render attachment the
    /// final pass writes to (either `surface_format` or `pre_aa_texture` format).
    pub fn new(device: &wgpu::Device, output_format: wgpu::TextureFormat) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PostProcess BGL"),
            entries: &[
                // binding 0: HDR source texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 1: linear sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 2: PostProcessUniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
            ],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:      Some("PostProcess Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let uniform_data = PostProcessUniforms::default();
        let uniform_buffer = {
            use wgpu::util::DeviceExt;
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some("PostProcess Uniform Buffer"),
                contents: bytemuck::bytes_of(&uniform_data),
                usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("PostProcess Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout as &wgpu::BindGroupLayout)],
            immediate_size:     0,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("PostProcess Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/passes/post_process.wgsl").into()
            ),
        });

        let pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("PostProcess Pipeline"),
            layout: Some(&pipeline_layout),
            cache:  None,
            vertex: wgpu::VertexState {
                module:               &shader,
                entry_point:          Some("vs_main"),
                buffers:              &[],
                compilation_options:  Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:              &shader,
                entry_point:         Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format:     output_format,
                    blend:      Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample:   wgpu::MultisampleState::default(),
            multiview_mask: None,
        }));

        Self { pipeline, bind_group_layout, sampler, uniform_buffer, bind_group: None }
    }

    /// Build / rebuild the bind group pointing at `hdr_view`.
    /// Call this once after construction and whenever the texture is resized.
    pub fn create_bind_group(&mut self, device: &wgpu::Device, hdr_view: &wgpu::TextureView) {
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("PostProcess Bind Group"),
            layout:  &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: wgpu::BindingResource::TextureView(hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding:  2,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        }));
    }

    /// Encode the post-process fullscreen pass.
    /// Writes to `output_view` (pre-AA texture or swapchain).
    pub fn execute(
        &self,
        encoder:     &mut wgpu::CommandEncoder,
        output_view: &wgpu::TextureView,
    ) -> Result<()> {
        let bg = match &self.bind_group {
            Some(bg) => bg,
            None => return Ok(()),   // not yet initialised (first frame)
        };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("PostProcess Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view:           output_view,
                resolve_target: None,
                depth_slice:    None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
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
