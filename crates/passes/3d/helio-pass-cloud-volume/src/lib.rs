//! Authored cloud-volume simulation and fullscreen raymarch.
//!
//! The two WGSL modules are included from the aligned Cloud Engine reference
//! tree. This crate owns the persistent volume textures and does not provide a
//! CPU fallback.

use bytemuck::{Pod, Zeroable};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

pub const VOLUME_SIZE: wgpu::Extent3d = wgpu::Extent3d {
    width: 96,
    height: 48,
    depth_or_array_layers: 96,
};
pub const SIM_PARAMS_SIZE: u64 = 112;
pub const RENDER_PARAMS_SIZE: u64 = 272;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Exact layout of the authored simulation block (7 vec4 values / 112 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SimParams {
    pub values: [[f32; 4]; 7],
}

/// Exact layout of the authored render block (17 vec4 values / 272 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct RenderParams {
    pub values: [[f32; 4]; 17],
}

pub const SIM_SHADER: &str = include_str!("shaders/simulate.wgsl");
pub const RENDER_SHADER: &str = include_str!("shaders/render.wgsl");

pub struct CloudVolumePass {
    sim_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    sim_bgl: wgpu::BindGroupLayout,
    render_bgl: wgpu::BindGroupLayout,
    sim_params: wgpu::Buffer,
    render_params: wgpu::Buffer,
    sampler: wgpu::Sampler,
    volumes: [wgpu::Texture; 2],
    volume_views: [wgpu::TextureView; 2],
    sim_groups: [wgpu::BindGroup; 2],
    render_groups: [wgpu::BindGroup; 2],
    ping: usize,
    target_format: wgpu::TextureFormat,
}

fn volume(device: &wgpu::Device, label: &str) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: VOLUME_SIZE,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D3),
        ..Default::default()
    });
    (texture, view)
}

impl CloudVolumePass {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let sim_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cloud Volume Simulation"),
            source: wgpu::ShaderSource::Wgsl(SIM_SHADER.into()),
        });
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cloud Volume Raymarch"),
            source: wgpu::ShaderSource::Wgsl(RENDER_SHADER.into()),
        });
        let sim_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cloud Volume Simulation BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(SIM_PARAMS_SIZE),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
            ],
        });
        let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cloud Volume Render BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(RENDER_PARAMS_SIZE),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let sim_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cloud Simulation Params (112 bytes)"),
            size: SIM_PARAMS_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let render_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cloud Render Params (272 bytes)"),
            size: RENDER_PARAMS_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Cloud Volume Repeat Clamp Repeat Linear"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let (a, av) = volume(device, "Cloud Volume A");
        let (b, bv) = volume(device, "Cloud Volume B");
        let volumes = [a, b];
        let volume_views = [av, bv];
        let sim_groups = [0, 1].map(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Cloud Simulation Ping-Pong Bind Group"),
                layout: &sim_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: sim_params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&volume_views[i]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&volume_views[1 - i]),
                    },
                ],
            })
        });
        let render_groups = [0, 1].map(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Cloud Raymarch Bind Group"),
                layout: &render_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: render_params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&volume_views[i]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            })
        });
        let sim_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Cloud Simulation Pipeline Layout"),
            bind_group_layouts: &[Some(&sim_bgl)],
            immediate_size: 0,
        });
        let sim_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Cloud Simulation 4x4x4"),
            layout: Some(&sim_layout),
            module: &sim_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let render_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Cloud Raymarch Pipeline Layout"),
            bind_group_layouts: &[Some(&render_bgl)],
            immediate_size: 0,
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Cloud Fullscreen Triangle Raymarch"),
            layout: Some(&render_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            sim_pipeline,
            render_pipeline,
            sim_bgl,
            render_bgl,
            sim_params,
            render_params,
            sampler,
            volumes,
            volume_views,
            sim_groups,
            render_groups,
            ping: 0,
            target_format,
        }
    }

    pub fn write_sim_params(&self, queue: &wgpu::Queue, params: &SimParams) {
        queue.write_buffer(&self.sim_params, 0, bytemuck::bytes_of(params));
    }
    pub fn write_render_params(&self, queue: &wgpu::Queue, params: &RenderParams) {
        queue.write_buffer(&self.render_params, 0, bytemuck::bytes_of(params));
    }
    pub fn volume_view(&self) -> &wgpu::TextureView {
        &self.volume_views[self.ping]
    }
    pub fn dispatch(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Cloud Volume Simulation"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.sim_pipeline);
        pass.set_bind_group(0, &self.sim_groups[self.ping], &[]);
        pass.dispatch_workgroups(24, 12, 24);
        drop(pass);
        self.ping = 1 - self.ping;
    }
    pub fn render(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.render_pipeline);
        pass.set_bind_group(0, &self.render_groups[self.ping], &[]);
        pass.draw(0..3, 0..1);
    }
}

impl RenderPass for CloudVolumePass {
    fn name(&self) -> &'static str {
        "CloudVolumePass"
    }
    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        // The pass owns no target view. Returning None lets execute create a
        // short-lived descriptor whose attachment array has a sound lifetime.
        None
    }
    fn prepare(&mut self, _ctx: &PrepareContext) -> HelioResult<()> {
        Ok(())
    }
    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let encoder = unsafe { &mut *ctx.compute_encoder_ptr };
        self.dispatch(encoder);
        if let Some(ptr) = ctx.active_render_pass_ptr() {
            unsafe {
                self.render(&mut *ptr);
            }
        } else {
            let attachments = [Some(wgpu::RenderPassColorAttachment {
                view: ctx.target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })];
            let mut pass = unsafe {
                (&mut *ctx.encoder_ptr).begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Cloud Volume Raymarch"),
                    color_attachments: &attachments,
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
            };
            self.render(&mut pass);
        }
        Ok(())
    }
}
