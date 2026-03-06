//! Screen-Space Ambient Occlusion (SSAO) pass.
//!
//! Renders AO into a single-channel texture that is later sampled
//! by the deferred lighting pass.

use std::sync::{Arc, Mutex};
use crate::graph::{RenderPass, PassContext, PassResourceBuilder, ResourceHandle};
use crate::Result;

/// SSAO configuration
pub struct SsaoConfig {
    pub radius: f32,
    pub bias: f32,
    pub power: f32,
    pub samples: u32,
}

impl Default for SsaoConfig {
    fn default() -> Self {
        Self {
            radius: 0.5,
            bias: 0.025,
            power: 2.0,
            samples: 16,
        }
    }
}

/// SSAO uniform data
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SsaoUniform {
    radius: f32,
    bias: f32,
    power: f32,
    samples: u32,
    noise_scale: [f32; 2],
    _pad: [f32; 2],
}

pub struct SsaoPass {
    pipeline: Arc<wgpu::RenderPipeline>,
    bind_group: Arc<Mutex<Arc<wgpu::BindGroup>>>,
    uniform_buffer: wgpu::Buffer,
    config: SsaoConfig,
}

impl SsaoPass {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gbuffer_layout: &wgpu::BindGroupLayout,
        global_layout: &wgpu::BindGroupLayout,
        _gbuffer_bind_group: Arc<Mutex<Arc<wgpu::BindGroup>>>, 
        _width: u32,
        _height: u32,
        config: SsaoConfig,
    ) -> Self {
        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SSAO Uniform Buffer"),
            size: std::mem::size_of::<SsaoUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Generate sample kernel
        let samples = Self::generate_sample_kernel(config.samples as usize);
        let sample_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SSAO Sample Buffer"),
            size: (samples.len() * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&sample_buffer, 0, bytemuck::cast_slice(&samples));

        // Generate noise texture (4x4 random rotation vectors)
        let noise_texture = Self::create_noise_texture(device, queue);
        let noise_view = noise_texture.create_view(&Default::default());
        let noise_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SSAO Noise Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            ..Default::default()
        });

        // Create SSAO-specific bind group layout
        let ssao_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSAO Bind Group Layout"),
            entries: &[
                // Uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Sample kernel
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Noise texture
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
                // Noise sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        let ssao_bind_group = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SSAO Bind Group"),
            layout: &ssao_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: sample_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&noise_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&noise_sampler) },
            ],
        }));

        // Create pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SSAO Pipeline Layout"),
            bind_group_layouts: &[
                Some(global_layout),      // Group 0: camera, globals
                Some(gbuffer_layout),     // Group 1: G-buffer textures + depth
                Some(&ssao_layout),       // Group 2: SSAO data
            ],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SSAO Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/passes/ssao.wgsl").into()),
        });

        let pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SSAO Pipeline"),
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
                    format: wgpu::TextureFormat::R8Unorm,
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
            bind_group: Arc::new(Mutex::new(ssao_bind_group)),
            uniform_buffer,
            config,
        }
    }

    /// Generate hemisphere sample kernel
    fn generate_sample_kernel(sample_count: usize) -> Vec<[f32; 4]> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut samples = Vec::with_capacity(sample_count);

        for i in 0..sample_count {
            let mut sample = [
                rng.gen::<f32>() * 2.0 - 1.0,
                rng.gen::<f32>() * 2.0 - 1.0,
                rng.gen::<f32>(),
                0.0,
            ];

            // Normalize
            let len = (sample[0] * sample[0] + sample[1] * sample[1] + sample[2] * sample[2]).sqrt();
            sample[0] /= len;
            sample[1] /= len;
            sample[2] /= len;

            // Scale to distribute more samples closer to origin
            let scale = i as f32 / sample_count as f32;
            let scale = 0.1 + scale * scale * 0.9;
            sample[0] *= scale;
            sample[1] *= scale;
            sample[2] *= scale;

            samples.push(sample);
        }

        samples
    }

    /// Create 4x4 noise texture with random rotation vectors
    fn create_noise_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Texture {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut noise_data = Vec::with_capacity(4 * 4 * 4); // 4x4 RGBA

        for _ in 0..16 {
            // Random rotation around Z axis
            let x = rng.gen::<f32>() * 2.0 - 1.0;
            let y = rng.gen::<f32>() * 2.0 - 1.0;
            let len = (x * x + y * y).sqrt();
            noise_data.push(((x / len * 0.5 + 0.5) * 255.0) as u8);
            noise_data.push(((y / len * 0.5 + 0.5) * 255.0) as u8);
            noise_data.push(128); // z = 0
            noise_data.push(255); // unused
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSAO Noise Texture"),
            size: wgpu::Extent3d { width: 4, height: 4, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &noise_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(16),
                rows_per_image: Some(4),
            },
            wgpu::Extent3d { width: 4, height: 4, depth_or_array_layers: 1 },
        );

        texture
    }

    pub fn update_uniforms(&self, queue: &wgpu::Queue, width: u32, height: u32) {
        let uniform = SsaoUniform {
            radius: self.config.radius,
            bias: self.config.bias,
            power: self.config.power,
            samples: self.config.samples,
            noise_scale: [width as f32 / 4.0, height as f32 / 4.0],
            _pad: [0.0; 2],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
    }
}

impl RenderPass for SsaoPass {
    fn name(&self) -> &str { "ssao" }

    fn declare_resources(&self, builder: &mut PassResourceBuilder) {
        builder.read(ResourceHandle::named("gbuffer"));
        builder.write(ResourceHandle::named("ssao_texture"));
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let bind_group = self.bind_group.lock().unwrap();

        let color_attach = Some(wgpu::RenderPassColorAttachment {
            view: ctx.target,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        });

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSAO Pass"),
            color_attachments: &[color_attach],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, ctx.global_bind_group, &[]);
        pass.set_bind_group(1, ctx.lighting_bind_group, &[]); // G-buffer
        pass.set_bind_group(2, Some(&**bind_group), &[]);
        pass.draw(0..3, 0..1); // Fullscreen triangle

        Ok(())
    }
}
