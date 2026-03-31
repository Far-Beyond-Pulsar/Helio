//! Water surface rendering pass.
//!
//! Renders water surfaces with realistic waves, reflections, refraction, foam,
//! and depth fade. Uses Gerstner waves for physically-based wave displacement
//! and screen-space techniques for high-quality reflections and refraction.
//!
//! # Features
//! - Multi-octave Gerstner waves with animated displacement
//! - Screen-space reflections (SSR) with variable step count
//! - Screen-space refraction with chromatic aberration support
//! - Fresnel-based reflection/refraction blending
//! - Foam generation based on wave steepness
//! - Smooth depth fade at water edges
//! - Integration with caustics and scene lighting
//!
//! # Performance
//! - Renders fullscreen quads for water volumes
//! - Early depth testing against scene geometry
//! - Configurable SSR quality (step count)
//! - Estimated cost: ~1.5ms @ 4K with 8 SSR steps

use bytemuck::{Pod, Zeroable};
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

/// Water surface rendering parameters (uniform buffer).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WaterParams {
    /// Current time in seconds
    time: f32,
    /// SSR maximum steps
    ssr_steps: u32,
    /// SSR step size in world units
    ssr_step_size: f32,
    /// Debug flags
    debug_flags: u32,
}

/// Water surface rendering pass.
///
/// Renders water surfaces with advanced visual effects including waves,
/// reflections, refraction, and foam.
pub struct WaterSurfacePass {
    pipeline: wgpu::RenderPipeline,
    bgl_0: wgpu::BindGroupLayout, // Camera, params
    bgl_1: wgpu::BindGroupLayout, // GBuffer, depth, scene color
    bgl_2: wgpu::BindGroupLayout, // Water volumes, caustics

    params_buf: wgpu::Buffer,

    bind_group_0: Option<wgpu::BindGroup>,
    bind_group_1: Option<wgpu::BindGroup>,
    bind_group_2: Option<wgpu::BindGroup>,

    // Cache keys for bind group rebuilding
    camera_key: Option<usize>,
    gbuffer_key: Option<usize>,
    volumes_key: Option<usize>,
}

impl WaterSurfacePass {
    /// Create a new water surface pass.
    ///
    /// # Parameters
    /// - `device`: GPU device
    /// - `camera_buf`: Camera uniform buffer
    /// - `width`: Render target width
    /// - `height`: Render target height
    ///
    /// # Returns
    /// A new `WaterSurfacePass` ready to be added to the render graph.
    pub fn new(
        device: &wgpu::Device,
        camera_buf: &wgpu::Buffer,
        width: u32,
        height: u32,
    ) -> Self {
        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water Surface Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/water_surface.wgsl").into()),
        });

        // Bind group layout 0: Camera + params
        let bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Water Surface BGL 0"),
            entries: &[
                // @binding(0) camera
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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
            ],
        });

        // Bind group layout 1: GBuffer + depth + scene color
        let bgl_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Water Surface BGL 1"),
            entries: &[
                // @binding(0) depth texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(1) gbuffer normal
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
                // @binding(2) scene color (pre_aa)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(3) linear sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Bind group layout 2: Water volumes + caustics
        let bgl_2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Water Surface BGL 2"),
            entries: &[
                // @binding(0) water volumes
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(1) caustics texture
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
                // @binding(2) caustics sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Water Surface Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl_0), Some(&bgl_1), Some(&bgl_2)],
            immediate_size: 0,
        });

        // Render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Water Surface Pipeline"),
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
                cull_mode: None, // Render both sides
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false), // Don't write depth (transparent)
                depth_compare: Some(wgpu::CompareFunction::LessEqual), // Test against scene depth
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
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
                    format: wgpu::TextureFormat::Rgba16Float, // HDR format (pre_aa)
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        // Uniform buffer for parameters
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Water Surface Params"),
            size: std::mem::size_of::<WaterParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bgl_0,
            bgl_1,
            bgl_2,
            params_buf,
            bind_group_0: None,
            bind_group_1: None,
            bind_group_2: None,
            camera_key: None,
            gbuffer_key: None,
            volumes_key: None,
        }
    }
}

impl RenderPass for WaterSurfacePass {
    fn name(&self) -> &'static str {
        "WaterSurface"
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        // Update parameters
        let params = WaterParams {
            time: ctx.frame as f32 * 0.016, // 60 FPS
            ssr_steps: 8,                    // SSR quality
            ssr_step_size: 0.5,              // World units per step
            debug_flags: 0,
        };
        ctx.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // Skip if no water volumes
        if ctx.frame.water_volume_count == 0 {
            return Ok(());
        }

        let Some(volumes_buf) = ctx.frame.water_volumes else {
            return Ok(());
        };

        let Some(pre_aa) = ctx.frame.pre_aa else {
            return Ok(());
        };

        let Some(gbuffer) = &ctx.frame.gbuffer else {
            return Ok(());
        };

        let Some(caustics_view) = ctx.frame.water_caustics else {
            return Ok(());
        };

        // Rebuild bind group 0 if camera changed
        let camera_ptr = ctx.scene.camera as *const _ as usize;
        if self.camera_key != Some(camera_ptr) {
            self.bind_group_0 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Water Surface BG 0"),
                layout: &self.bgl_0,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ctx.scene.camera.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.params_buf.as_entire_binding(),
                    },
                ],
            }));
            self.camera_key = Some(camera_ptr);
        }

        // Rebuild bind group 1 if gbuffer changed
        let gbuffer_ptr = gbuffer.albedo as *const _ as usize;
        if self.gbuffer_key != Some(gbuffer_ptr) {
            // Create linear sampler
            let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Water Linear Sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            });

            self.bind_group_1 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Water Surface BG 1"),
                layout: &self.bgl_1,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(ctx.depth),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(gbuffer.normal),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(pre_aa),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            }));
            self.gbuffer_key = Some(gbuffer_ptr);
        }

        // Rebuild bind group 2 if volumes changed
        let volumes_ptr = volumes_buf as *const _ as usize;
        if self.volumes_key != Some(volumes_ptr) {
            // Create caustics sampler
            let caustics_sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Caustics Sampler"),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                address_mode_w: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            });

            self.bind_group_2 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Water Surface BG 2"),
                layout: &self.bgl_2,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: volumes_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(caustics_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&caustics_sampler),
                    },
                ],
            }));
            self.volumes_key = Some(volumes_ptr);
        }

        // Store volume count before creating render pass
        let volume_count = ctx.frame.water_volume_count;

        // Render water surfaces
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("WaterSurface"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: pre_aa,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load, // Blend with existing scene
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: ctx.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load, // Test against scene depth
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, self.bind_group_0.as_ref().unwrap(), &[]);
        pass.set_bind_group(1, self.bind_group_1.as_ref().unwrap(), &[]);
        pass.set_bind_group(2, self.bind_group_2.as_ref().unwrap(), &[]);

        // Draw fullscreen quad for each water volume (instanced)
        // 6 vertices per quad, instanced by volume count
        pass.draw(0..6, 0..volume_count);

        Ok(())
    }
}
