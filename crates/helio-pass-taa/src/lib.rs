//! Temporal Anti-Aliasing (TAA) pass.
//!
//! Blends the current frame with a history buffer using velocity-based reprojection
//! and YCoCg variance-clipped neighbourhood clamping.
//!
//! ## O(1) guarantee
//! `execute()` records exactly one fullscreen `draw(0..3, 0..1)` for the TAA
//! resolve, one `copy_texture_to_texture` to update history, and one fullscreen
//! `draw(0..3, 0..1)` blit that writes the resolved image to `ctx.target`.
//! All three are constant-time GPU operations.
//!
//! ## Jitter
//! A Halton(2, 3) sequence of 16 entries is indexed by `frame_num % 16`.
//! The jitter offset is uploaded to the GPU uniform every frame in `prepare()`.
//!
//! ## History ping-pong
//! The pass owns two textures: `output_texture` (render target each frame) and
//! `history_texture` (sampled as temporal history).  After each TAA resolve the
//! output is GPU-copied into history so the next frame sees the updated accumulation.
//!
//! ## Lazy bind group
//! The TAA bind group is rebuilt lazily when `frame.pre_aa` or `ctx.depth`
//! pointer changes (i.e. on resize). No views are required at construction time.

use bytemuck::{Pod, Zeroable};
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

const HALTON_JITTER: [[f32; 2]; 16] = [
    [0.500000, 0.333333],
    [0.250000, 0.666667],
    [0.750000, 0.111111],
    [0.125000, 0.444444],
    [0.625000, 0.777778],
    [0.375000, 0.222222],
    [0.875000, 0.555556],
    [0.062500, 0.888889],
    [0.562500, 0.037037],
    [0.312500, 0.370370],
    [0.812500, 0.703704],
    [0.187500, 0.148148],
    [0.687500, 0.481481],
    [0.437500, 0.814815],
    [0.937500, 0.259259],
    [0.031250, 0.592593],
];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TaaUniform {
    feedback_min: f32,
    feedback_max: f32,
    jitter: [f32; 2],
}

/// Minimal blit shader: samples a texture and outputs it unchanged.
const BLIT_WGSL: &str = "
@group(0) @binding(0) var blit_tex:     texture_2d<f32>;
@group(0) @binding(1) var blit_sampler: sampler;

struct VertexOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }

@vertex fn vs_blit(@builtin(vertex_index) vi: u32) -> VertexOut {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    return VertexOut(vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0), vec2<f32>(x, y));
}

@fragment fn fs_blit(in: VertexOut) -> @location(0) vec4<f32> {
    return textureSample(blit_tex, blit_sampler, in.uv);
}
";

pub struct TaaPass {
    pipeline: wgpu::RenderPipeline,
    blit_pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    blit_bgl: wgpu::BindGroupLayout,
    /// Lazy TAA bind group (pre_aa + history + velocity_fallback + depth + samplers + uniform).
    bind_group: Option<wgpu::BindGroup>,
    /// (pre_aa_ptr, depth_ptr)
    bind_group_key: Option<(usize, usize)>,
    /// Static blit bind group: output_view + linear_sampler.
    blit_bind_group: wgpu::BindGroup,
    taa_uniform_buf: wgpu::Buffer,
    pub history_texture: wgpu::Texture,
    pub history_view: wgpu::TextureView,
    pub output_texture: wgpu::Texture,
    pub output_view: wgpu::TextureView,
    linear_sampler: wgpu::Sampler,
    point_sampler: wgpu::Sampler,
    velocity_fallback_texture: wgpu::Texture,
    velocity_fallback_view: wgpu::TextureView,
}

impl TaaPass {
    /// Create a new TAA pass. No texture views needed at construction time.
    pub fn new(device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TAA Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/taa.wgsl").into()),
        });
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TAA Blit Shader"),
            source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
        });

        let taa_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TAA Uniform"),
            size: std::mem::size_of::<TaaUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("TAA Linear Sampler"),
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let point_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("TAA Point Sampler"),
            min_filter: wgpu::FilterMode::Nearest,
            mag_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let tex_desc = |label: &'static str, extra: wgpu::TextureUsages| wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT | extra,
            view_formats: &[],
        };

        let history_texture = device.create_texture(&tex_desc("TAA History", wgpu::TextureUsages::COPY_DST));
        let history_view = history_texture.create_view(&Default::default());
        let output_texture = device.create_texture(&tex_desc("TAA Output", wgpu::TextureUsages::COPY_SRC));
        let output_view = output_texture.create_view(&Default::default());

        let velocity_fallback_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("TAA Velocity Fallback"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let velocity_fallback_view = velocity_fallback_texture.create_view(&Default::default());

        // ── TAA BGL ────────────────────────────────────────────────────────────
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TAA BGL"),
            entries: &[
                tex_entry(0, wgpu::TextureSampleType::Float { filterable: true }),
                tex_entry(1, wgpu::TextureSampleType::Float { filterable: true }),
                tex_entry(2, wgpu::TextureSampleType::Float { filterable: false }),
                tex_entry(3, wgpu::TextureSampleType::Depth),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
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

        // ── Blit BGL ───────────────────────────────────────────────────────────
        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TAA Blit BGL"),
            entries: &[
                tex_entry(0, wgpu::TextureSampleType::Float { filterable: true }),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TAA Blit BG"),
            layout: &blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&linear_sampler),
                },
            ],
        });

        // ── TAA pipeline ───────────────────────────────────────────────────────
        let taa_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("TAA PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("TAA Pipeline"),
            layout: Some(&taa_pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // ── Blit pipeline ──────────────────────────────────────────────────────
        let blit_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("TAA Blit PL"),
            bind_group_layouts: &[Some(&blit_bgl)],
            immediate_size: 0,
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("TAA Blit Pipeline"),
            layout: Some(&blit_pl),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_blit"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_blit"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            blit_pipeline,
            bgl,
            blit_bgl,
            bind_group: None,
            bind_group_key: None,
            blit_bind_group,
            taa_uniform_buf,
            history_texture,
            history_view,
            output_texture,
            output_view,
            linear_sampler,
            point_sampler,
            velocity_fallback_texture,
            velocity_fallback_view,
        }
    }
}

fn tex_entry(binding: u32, sample_type: wgpu::TextureSampleType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

impl RenderPass for TaaPass {
    fn name(&self) -> &'static str { "TAA" }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let jitter_idx = (ctx.frame % 16) as usize;
        let raw = HALTON_JITTER[jitter_idx];
        let uniforms = TaaUniform {
            feedback_min: 0.88,
            feedback_max: 0.97,
            jitter: [raw[0] - 0.5, raw[1] - 0.5],
        };
        ctx.queue.write_buffer(&self.taa_uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // ── 1. Lazy bind group ────────────────────────────────────────────────
        let pre_aa_view = ctx.frame.pre_aa.ok_or_else(|| {
            helio_v3::Error::InvalidPassConfig(
                "TaaPass requires frame.pre_aa (published by DeferredLightPass)".to_string(),
            )
        })?;
        let key = (pre_aa_view as *const _ as usize, ctx.depth as *const _ as usize);
        if self.bind_group_key != Some(key) {
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("TAA BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(pre_aa_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.history_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.velocity_fallback_view) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(ctx.depth) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.linear_sampler) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&self.point_sampler) },
                    wgpu::BindGroupEntry { binding: 6, resource: self.taa_uniform_buf.as_entire_binding() },
                ],
            }));
            self.bind_group_key = Some(key);
        }

        // ── 2. TAA resolve → output_view ─────────────────────────────────────
        {
            let attachments = [Some(wgpu::RenderPassColorAttachment {
                view: &self.output_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })];
            let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("TAA Resolve"),
                color_attachments: &attachments,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
            pass.draw(0..3, 0..1);
        }

        // ── 3. Copy output → history ──────────────────────────────────────────
        ctx.encoder.copy_texture_to_texture(
            self.output_texture.as_image_copy(),
            self.history_texture.as_image_copy(),
            wgpu::Extent3d { width: ctx.width, height: ctx.height, depth_or_array_layers: 1 },
        );

        // ── 4. Blit output_view → ctx.target ─────────────────────────────────
        {
            let attachments = [Some(wgpu::RenderPassColorAttachment {
                view: ctx.target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })];
            let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("TAA Blit"),
                color_attachments: &attachments,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.blit_pipeline);
            pass.set_bind_group(0, &self.blit_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        Ok(())
    }
}
