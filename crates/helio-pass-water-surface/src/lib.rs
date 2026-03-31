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
    blit_pipeline: wgpu::RenderPipeline,
    bgl_0: wgpu::BindGroupLayout, // Camera, params
    bgl_1: wgpu::BindGroupLayout, // GBuffer, depth, scene color
    bgl_2: wgpu::BindGroupLayout, // Water volumes, caustics
    bgl_blit: wgpu::BindGroupLayout, // Blit texture + sampler

    params_buf: wgpu::Buffer,

    // Copy of scene color for sampling (to avoid read-write conflict)
    scene_copy: wgpu::Texture,
    scene_copy_view: wgpu::TextureView,

    bind_group_0: Option<wgpu::BindGroup>,
    bind_group_1: Option<wgpu::BindGroup>,
    bind_group_2: Option<wgpu::BindGroup>,

    // Cache keys for bind group rebuilding
    camera_key: Option<usize>,
    gbuffer_key: Option<usize>,
    volumes_key: Option<usize>,

    width: u32,
    height: u32,
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
        target_format: wgpu::TextureFormat,
    ) -> Self {
        // Load shaders
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water Surface Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/water_surface.wgsl").into()),
        });

        // Simple blit shader for copying scene
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water Blit Shader"),
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

        // Bind group layout for blit
        let bgl_blit = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Water Blit BGL"),
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
            depth_stencil: None, // Manual depth testing in shader
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

        // Blit pipeline for copying scene
        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Water Blit Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl_blit)],
            immediate_size: 0,
        });

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Water Blit Pipeline"),
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

        // Uniform buffer for parameters
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Water Surface Params"),
            size: std::mem::size_of::<WaterParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Scene color copy texture (for sampling while writing to original)
        let scene_copy = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Water Scene Copy"),
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
            bgl_0,
            bgl_1,
            bgl_2,
            bgl_blit,
            params_buf,
            scene_copy,
            scene_copy_view,
            bind_group_0: None,
            bind_group_1: None,
            bind_group_2: None,
            camera_key: None,
            gbuffer_key: None,
            volumes_key: None,
            width,
            height,
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

        // Create blit bind group every frame (pre_aa view changes)
        let blit_sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Water Blit Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let blit_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Water Blit BG"),
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
                label: Some("Water Scene Copy"),
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
                        resource: wgpu::BindingResource::TextureView(&self.scene_copy_view),
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
        // Note: No depth attachment because we sample from depth texture in shader
        // Manual depth testing is done in fragment shader (lines 239-247 in water_surface.wgsl)
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
            depth_stencil_attachment: None, // Can't use depth attachment while sampling depth texture
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, self.bind_group_0.as_ref().unwrap(), &[]);
        pass.set_bind_group(1, self.bind_group_1.as_ref().unwrap(), &[]);
        pass.set_bind_group(2, self.bind_group_2.as_ref().unwrap(), &[]);

        // Draw full cube for each water volume (instanced)
        // 36 vertices per cube (6 faces × 6 vertices per quad), instanced by volume count
        pass.draw(0..36, 0..volume_count);

        Ok(())
    }
}
