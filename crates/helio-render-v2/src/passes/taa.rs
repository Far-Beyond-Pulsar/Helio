//! TAA (Temporal Anti-Aliasing) pass

use std::sync::Arc;
use crate::graph::{RenderPass, PassContext, PassResourceBuilder, ResourceHandle};
use crate::Result;

/// TAA configuration
pub struct TaaConfig {
    pub feedback_min: f32,
    pub feedback_max: f32,
}

impl Default for TaaConfig {
    fn default() -> Self {
        Self {
            feedback_min: 0.88,
            feedback_max: 0.97,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TaaUniform {
    feedback_min: f32,
    feedback_max: f32,
    jitter_offset: [f32; 2],
}

pub struct TaaPass {
    pipeline: Arc<wgpu::RenderPipeline>,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform_buffer: wgpu::Buffer,
    config: TaaConfig,
    
    // History texture (swapped each frame)
    history_texture: Option<wgpu::Texture>,
    history_view: Option<wgpu::TextureView>,
    
    jitter_index: u32,
}

impl TaaPass {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        config: TaaConfig,
    ) -> Self {
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TAA Uniform Buffer"),
            size: std::mem::size_of::<TaaUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TAA Bind Group Layout"),
            entries: &[
                // Current frame
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
                // History frame
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Velocity texture
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Depth texture
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("TAA Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("TAA Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TAA Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/passes/taa.wgsl").into()),
        });

        let pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("TAA Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        }));

        Self {
            pipeline,
            bind_group_layout,
            sampler,
            uniform_buffer,
            config,
            history_texture: None,
            history_view: None,
            jitter_index: 0,
        }
    }

    /// Get the current jitter offset for camera projection
    pub fn get_jitter_offset(&self) -> [f32; 2] {
        // Halton sequence for jitter pattern
        Self::halton_jitter(self.jitter_index)
    }

    /// Halton sequence jitter pattern (8 samples)
    fn halton_jitter(index: u32) -> [f32; 2] {
        let patterns = [
            [0.0, 0.0],
            [0.5, 0.333],
            [0.25, 0.666],
            [0.75, 0.111],
            [0.125, 0.444],
            [0.625, 0.777],
            [0.375, 0.222],
            [0.875, 0.555],
        ];
        let pattern = patterns[(index % 8) as usize];
        [(pattern[0] - 0.5) * 2.0, (pattern[1] - 0.5) * 2.0]
    }

    pub fn advance_jitter(&mut self) {
        self.jitter_index = (self.jitter_index + 1) % 8;
    }

    pub fn create_history_texture(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("TAA History Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        
        let view = texture.create_view(&Default::default());
        self.history_texture = Some(texture);
        self.history_view = Some(view);
    }

    pub fn update_uniforms(&self, queue: &wgpu::Queue) {
        let uniform = TaaUniform {
            feedback_min: self.config.feedback_min,
            feedback_max: self.config.feedback_max,
            jitter_offset: self.get_jitter_offset(),
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
    }
}

impl RenderPass for TaaPass {
    fn name(&self) -> &str { "taa" }

    fn declare_resources(&self, builder: &mut PassResourceBuilder) {
        builder.read(ResourceHandle::named("pre_aa_color"));
        builder.read(ResourceHandle::named("velocity"));
        builder.read(ResourceHandle::named("depth"));
        builder.write(ResourceHandle::named("color_target"));
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let color_attach = Some(wgpu::RenderPassColorAttachment {
            view: ctx.target,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        });

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("TAA Pass"),
            color_attachments: &[color_attach],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(&self.pipeline);
        // Note: would need to set bind group here
        pass.draw(0..3, 0..1);

        Ok(())
    }
}
