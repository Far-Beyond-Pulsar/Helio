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
    feedback_min: f32,   // unused — kept for struct-layout compatibility
    feedback_max: f32,   // unused — kept for struct-layout compatibility
    jitter: [f32; 2],
    reset: u32,          // 1 on the very first frame so RESET path runs
    _pad: u32,           // pad to match WGSL struct alignment (vec2 → align-8 → 24 bytes)
}

/// Post-TAA sharpening blit.
///
/// Unreal TSR and Unity HDRP both apply a spatial sharpen **on the output only**,
/// never on the history buffer.  Sharpening the history would amplify ringing
/// artefacts over multiple frames; keeping history unsharpened preserves temporal
/// stability while the sharpened output recovers fine mesh / material detail lost
/// by the temporal low-pass filter.
///
/// Algorithm: contrast-adaptive unsharp mask (5-tap cross kernel).
///   blur        = (N + S + E + W) / 4
///   edge        = center - blur
///   sharpened   = center + edge * strength * (1 - 2 * local_contrast)
///
/// The `(1 - 2*contrast)` term reduces sharpening on already-sharp edges and
/// boosts it on smooth regions that lost detail — the same idea as AMD CAS.
const BLIT_WGSL: &str = "
@group(0) @binding(0) var blit_tex:     texture_2d<f32>;
@group(0) @binding(1) var blit_sampler: sampler;

// Sharpening strength: 0 = disabled, 0.4 = default (matches UE4 TAA sharpening).
// Increasing this recovers more texture/mesh detail at cost of potential ringing.
const SHARPEN_STRENGTH: f32 = 0.4;

struct VertexOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }

@vertex fn vs_blit(@builtin(vertex_index) vi: u32) -> VertexOut {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    return VertexOut(vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0), vec2<f32>(x, y));
}

@fragment fn fs_blit(in: VertexOut) -> @location(0) vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(blit_tex));
    let c  = textureSampleLevel(blit_tex, blit_sampler, in.uv, 0.0).rgb;
    let n  = textureSampleLevel(blit_tex, blit_sampler, in.uv + vec2<f32>( 0.0, -texel.y), 0.0).rgb;
    let s  = textureSampleLevel(blit_tex, blit_sampler, in.uv + vec2<f32>( 0.0,  texel.y), 0.0).rgb;
    let e  = textureSampleLevel(blit_tex, blit_sampler, in.uv + vec2<f32>( texel.x,  0.0), 0.0).rgb;
    let w  = textureSampleLevel(blit_tex, blit_sampler, in.uv + vec2<f32>(-texel.x,  0.0), 0.0).rgb;

    // Local luminance contrast — reduce sharpening on already-sharp edges
    let luma = vec3<f32>(0.2126, 0.7152, 0.0722);
    let lc   = dot(c, luma);
    let ln   = dot(n, luma); let ls = dot(s, luma);
    let le   = dot(e, luma); let lw = dot(w, luma);
    let contrast = max(max(max(max(lc, ln), ls), le), lw)
                 - min(min(min(min(lc, ln), ls), le), lw);

    // Contrast-adaptive unsharp mask
    let blur     = (n + s + e + w) * 0.25;
    let strength = SHARPEN_STRENGTH * saturate(1.0 - 2.0 * contrast);
    let result   = clamp(c + (c - blur) * strength, vec3<f32>(0.0), vec3<f32>(1.0));

    return vec4<f32>(result, 1.0);
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
    /// Set to true on construction; cleared after the first prepare() so the
    /// shader's RESET path runs exactly once to prime the history texture.
    first_frame: bool,
    /// Format of output / history textures (needed to recreate them on resize).
    output_format: wgpu::TextureFormat,
}

impl TaaPass {
    /// Create a new TAA pass. No texture views needed at construction time.
    ///
    /// - `internal_width / internal_height` — geometry (pre-AA) render resolution.
    ///   When equal to `output_*` this is standard TAA; when smaller this is temporal
    ///   upscaling (the shader bilinearly upsamples the input to output resolution).
    /// - `output_width / output_height` — native display resolution; the history
    ///   and output textures, and the final blit, all run at this size.
    pub fn new(
        device: &wgpu::Device,
        _internal_width: u32,
        _internal_height: u32,
        output_width: u32,
        output_height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
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

        // history and output textures are always at OUTPUT (display) resolution so
        // temporal accumulation is gathered at full quality even when rendering at
        // a lower internal resolution.
        let tex_desc = |label: &'static str, extra: wgpu::TextureUsages| wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: output_width, height: output_height, depth_or_array_layers: 1 },
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
            first_frame: true,
            output_format: format,
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

    fn on_resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let fmt = self.output_format;
        let tex_desc = |label: &'static str, extra: wgpu::TextureUsages| wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: fmt,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT | extra,
            view_formats: &[],
        };

        self.history_texture = device.create_texture(&tex_desc("TAA History", wgpu::TextureUsages::COPY_DST));
        self.history_view = self.history_texture.create_view(&Default::default());
        self.output_texture = device.create_texture(&tex_desc("TAA Output", wgpu::TextureUsages::COPY_SRC));
        self.output_view = self.output_texture.create_view(&Default::default());

        // blit_bind_group references output_view — must be rebuilt.
        self.blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TAA Blit BG"),
            layout: &self.blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
            ],
        });

        // TAA bind group references history_view — invalidate so it is rebuilt in execute().
        self.bind_group = None;
        self.bind_group_key = None;
        // Reset history so the old (stale) history texture is not accumulated.
        self.first_frame = true;
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let jitter_idx = (ctx.frame_num % 16) as usize;
        let raw = HALTON_JITTER[jitter_idx];
        // Consume the first_frame flag so RESET runs on exactly one frame.
        let reset = if self.first_frame { self.first_frame = false; 1u32 } else { 0u32 };
        let uniforms = TaaUniform {
            feedback_min: 0.88,  // unused but kept for layout
            feedback_max: 0.97,  // unused but kept for layout
            jitter: [raw[0] - 0.5, raw[1] - 0.5],
            reset,
            _pad: 0,
        };
        ctx.queue.write_buffer(&self.taa_uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // ── 1. Lazy bind group ────────────────────────────────────────────────
        let pre_aa_view = ctx.resources.pre_aa.ok_or_else(|| {
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
