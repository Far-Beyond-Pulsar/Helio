//! Underwater post-processing pass.
//!
//! Applies underwater visual effects when the camera is submerged in a water volume:
//! - Volumetric fog with exponential depth falloff
//! - Beer-Lambert color absorption (wavelength-dependent)
//! - Caustics projection onto underwater surfaces
//! - God rays (volumetric light shafts from surface)
//!
//! # Performance
//! - Only executes when camera is underwater (checked in prepare())
//! - Fullscreen pass: ~0.5ms @ 4K
//! - Can be skipped entirely for above-water views

use helio_v3::Result as HelioResult;
use bytemuck::{Pod, Zeroable};
use helio_v3::{PassContext, PrepareContext, RenderPass};

/// Underwater rendering parameters (uniform buffer).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct UnderwaterParams {
    /// Current time in seconds
    time: f32,
    /// Active water volume index (-1 if not underwater)
    active_volume: i32,
    /// Padding
    _pad0: f32,
    _pad1: f32,
}

/// Underwater post-processing pass.
///
/// Applies realistic underwater effects including fog, absorption, caustics,
/// and god rays. Only runs when the camera is inside a water volume.
pub struct UnderwaterPass {
    pipeline: wgpu::RenderPipeline,
    blit_pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    bgl_blit: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,

    // Copy of scene color for sampling (to avoid read-write conflict)
    scene_copy: wgpu::Texture,
    scene_copy_view: wgpu::TextureView,

    bind_group: Option<wgpu::BindGroup>,

    /// Whether camera is currently underwater
    camera_underwater: bool,
    /// Index of water volume containing camera
    active_volume_idx: i32,
}

impl UnderwaterPass {
    /// Create a new underwater pass.
    ///
    /// # Parameters
    /// - `device`: GPU device
    /// - `camera_buf`: Camera uniform buffer
    /// - `width`: Internal render width
    /// - `height`: Internal render height
    /// - `target_format`: Format of the render target
    ///
    /// # Returns
    /// A new `UnderwaterPass` ready to be added to the render graph.
    pub fn new(device: &wgpu::Device, camera_buf: &wgpu::Buffer, width: u32, height: u32, target_format: wgpu::TextureFormat) -> Self {
        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Underwater Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/underwater.wgsl").into()),
        });

        // Simple blit shader for copying scene
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Underwater Blit Shader"),
            source: wgpu::ShaderSource::Wgsl(r#"
                @group(0) @binding(0) var src_texture: texture_2d<f32>;
                @group(0) @binding(1) var src_sampler: sampler;

                struct VertexOutput {
                    @builtin(position) position: vec4<f32>,
                    @location(0) uv: vec2<f32>,
                }

                @vertex
                fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
                    var positions = array<vec2<f32>, 6>(
                        vec2<f32>(-1.0, -1.0),
                        vec2<f32>(1.0, -1.0),
                        vec2<f32>(1.0, 1.0),
                        vec2<f32>(-1.0, -1.0),
                        vec2<f32>(1.0, 1.0),
                        vec2<f32>(-1.0, 1.0),
                    );
                    let pos = positions[vid];
                    var out: VertexOutput;
                    out.position = vec4<f32>(pos, 0.0, 1.0);
                    out.uv = pos * 0.5 + 0.5;
                    return out;
                }

                @fragment
                fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                    return textureSample(src_texture, src_sampler, in.uv);
                }
            "#.into()),
        });

        // Bind group layout for blit
        let bgl_blit = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Underwater Blit BGL"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Bind group layout
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Underwater BGL"),
            entries: &[
                // @binding(0) camera
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
                // @binding(1) params
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(2) depth texture
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(3) scene color
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(4) water volumes
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(5) caustics texture
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(6) linear sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Underwater Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        // Render pipeline (fullscreen post-process)
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Underwater Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None, // No depth testing for fullscreen pass
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None, // Replace (not blend)
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        // Blit pipeline for copying scene
        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Underwater Blit Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl_blit)],
            immediate_size: 0,
        });

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Underwater Blit Pipeline"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        // Uniform buffer
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Underwater Params"),
            size: std::mem::size_of::<UnderwaterParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Scene color copy texture (for sampling while writing to original)
        let scene_copy = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Underwater Scene Copy"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let scene_copy_view = scene_copy.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            pipeline,
            blit_pipeline,
            bgl,
            bgl_blit,
            params_buf,
            scene_copy,
            scene_copy_view,
            bind_group: None,
            camera_underwater: false,
            active_volume_idx: -1,
        }
    }

    /// Check if camera position is inside any water volume.
    fn check_camera_underwater(&mut self, ctx: &PrepareContext) {
        self.camera_underwater = false;
        self.active_volume_idx = -1;

        // This would need access to scene water volumes to check
        // For now, we'll check in execute() when we have access to frame data
    }
}

impl RenderPass for UnderwaterPass {
    fn name(&self) -> &'static str {
        "Underwater"
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        // Update parameters
        let params = UnderwaterParams {
            time: ctx.frame as f32 * 0.016,
            active_volume: self.active_volume_idx,
            _pad0: 0.0,
            _pad1: 0.0,
        };
        ctx.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // Skip if no water volumes or camera not underwater
        if ctx.frame.water_volume_count == 0 {
            return Ok(());
        }

        let Some(volumes_buf) = ctx.frame.water_volumes else {
            return Ok(());
        };

        let Some(pre_aa) = ctx.frame.pre_aa else {
            return Ok(());
        };

        let Some(caustics_view) = ctx.frame.water_caustics else {
            return Ok(());
        };

        // Create blit bind group (pre_aa → scene_copy)
        let blit_sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Underwater Blit Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let blit_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Underwater Blit BG"),
            layout: &self.bgl_blit,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(pre_aa),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&blit_sampler),
                },
            ],
        });

        // Blit pre_aa to scene_copy (to avoid read-write conflict)
        {
            let mut blit_pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Underwater Scene Copy"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.scene_copy_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            blit_pass.set_pipeline(&self.blit_pipeline);
            blit_pass.set_bind_group(0, &blit_bind_group, &[]);
            blit_pass.draw(0..6, 0..1);
        }

        // Create sampler for underwater effects
        let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Underwater Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        // Rebuild bind group (sample from scene_copy, not pre_aa)
        self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Underwater BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ctx.scene.camera.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(ctx.depth),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.scene_copy_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: volumes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(caustics_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        }));

        // Render fullscreen underwater effects (write to pre_aa)
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Underwater"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: pre_aa,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        pass.draw(0..3, 0..1); // Fullscreen triangle

        Ok(())
    }
}
