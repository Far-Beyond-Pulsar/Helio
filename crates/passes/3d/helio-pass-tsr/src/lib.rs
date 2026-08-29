//! Temporal Super-Resolution (TSR) pass.
//!
//! A dedicated temporal upscaling pass that replaces the simple bilinear upscale
//! in `TaaPass`. Given the pre-AA HDR frame at internal resolution and a history
//! buffer at output resolution it produces a sharp, temporally stable image at
//! the full display resolution.
//!
//! ## Quality presets
//!
//! [`TsrQuality`] controls the recommended `render_scale`, neighbourhood tap count,
//! and temporal accumulation parameters.  Select a preset via
//! [`RendererConfig::with_tsr_quality`] when building the render graph.
//!
//! ## Reactivity API
//!
//! [`TsrPass::reset_history`] discards the temporal history (call on camera cuts,
//! level loads, teleports).  [`TsrPass::set_reactivity`] biases the per-pixel
//! blend factor toward the current frame.
//!
//! ## O(1) guarantee
//!
//! `execute()` records exactly:
//! 1. One fullscreen draw (TSR resolve) into `output_texture`.
//! 2. One `copy_texture_to_texture` (history ping-pong).
//! 3. One fullscreen draw (passthrough blit) into `ctx.target`.
//!
//! All three are constant-time GPU operations regardless of scene complexity.

use bytemuck::{Pod, Zeroable};
use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

// ── Passthrough blit shader ───────────────────────────────────────────────────

/// Simple two-tap passthrough blit that copies `blit_tex` to the render target.
/// No sharpening — all processing happens in the main TSR resolve step.
const BLIT_WGSL: &str = "
@group(0) @binding(0) var blit_tex:     texture_2d<f32>;
@group(0) @binding(1) var blit_sampler: sampler;

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
}

@vertex fn vs_blit(@builtin(vertex_index) vi: u32) -> VertexOut {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    return VertexOut(vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0), vec2<f32>(x, y));
}

@fragment fn fs_blit(in: VertexOut) -> @location(0) vec4<f32> {
    return textureSampleLevel(blit_tex, blit_sampler, in.uv, 0.0);
}
";

// ── R1/R2 low-discrepancy jitter ──────────────────────────────────────────────

/// R1/R2 low-discrepancy jitter — same sub-pixel sequence as `TaaPass` so
/// coverage is coherent between the two passes.
fn r1_r2_jitter(frame: u64) -> [f32; 2] {
    const INV_R1: f64 = 0.7548776662466927;
    const INV_R2: f64 = 0.5698402905980539;
    const PHASE: f64 = 0.5;
    let fx = frame as f64 * INV_R1 + PHASE;
    let fy = frame as f64 * INV_R2 + PHASE;
    [(fx.fract() - 0.5) as f32, (fy.fract() - 0.5) as f32]
}

// ── Quality presets ───────────────────────────────────────────────────────────

/// Temporal Super-Resolution quality presets.
///
/// Each preset specifies:
/// - The recommended `render_scale` (internal / display resolution ratio).
/// - The neighbourhood tap radius used in the TSR shader.
///
/// Pass a preset to [`TsrPass::new`] and set `RendererConfig::render_scale` to
/// `preset.render_scale()` for the recommended operating point.
///
/// ## Example
/// ```rust,ignore
/// let quality = TsrQuality::Quality;
/// let config  = RendererConfig::new(1920, 1080, surface_format)
///     .with_render_scale(quality.render_scale())
///     .with_tsr_quality(quality);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TsrQuality {
    /// 0.50× internal resolution, 3×3 (radius 1) neighbourhood.
    ///
    /// Maximum performance uplift; some temporal shimmer on thin geometry
    /// at very fast camera speeds.
    Performance,

    /// 0.67× internal resolution, 3×3 (radius 1) neighbourhood.
    ///
    /// Good balance of performance and quality; suitable for mid-range GPUs.
    Balanced,

    /// 0.75× internal resolution, 5×5 (radius 2) neighbourhood.
    ///
    /// High quality; matches UE5 TSR "Quality" preset. **Default.**
    #[default]
    Quality,

    /// 1.0× internal resolution, 5×5 (radius 2) neighbourhood.
    ///
    /// No resolution upscaling — TSR still runs for temporal AA.
    Native,
}

impl TsrQuality {
    /// Recommended `render_scale` (internal / display) for this preset.
    pub fn render_scale(self) -> f32 {
        match self {
            Self::Performance => 0.50,
            Self::Balanced => 0.67,
            Self::Quality => 0.75,
            Self::Native => 1.00,
        }
    }

    /// Neighbourhood tap radius sent to the TSR shader.
    ///
    /// `1` → 3×3 (9 taps), `2` → 5×5 (25 taps).
    pub fn tap_radius(self) -> u32 {
        match self {
            Self::Performance | Self::Balanced => 1,
            Self::Quality | Self::Native => 2,
        }
    }
}

// ── GPU uniform ───────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TsrUniform {
    jitter_offset: [f32; 2], // sub-pixel jitter in [-0.5, 0.5)
    reactivity: f32,         // extra blend toward current (0 = full history)
    reset: u32,              // 1 on first frame / after reset_history()
    time_delta: f32,         // seconds since last frame
    tap_radius: u32,         // 1 = 3×3, 2 = 5×5
    _pad: [f32; 2],
}

// ── Pass ──────────────────────────────────────────────────────────────────────

/// Temporal Super-Resolution pass.
///
/// Placed **in place of** `TaaPass` in the render graph when TSR is enabled.
/// Reads `"pre_aa"` from [`FrameResources`](libhelio::FrameResources) and
/// writes the upsampled, temporally accumulated image to `ctx.target`.
pub struct TsrPass {
    // ── Main TSR pipeline (resolve) ───────────────────────────────────────────
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,
    bind_group_key: Option<(usize, usize)>,
    uniform_buf: wgpu::Buffer,

    // ── Blit pipeline (output_texture → ctx.target) ───────────────────────────
    blit_pipeline: wgpu::RenderPipeline,
    blit_bgl: wgpu::BindGroupLayout,
    blit_bind_group: wgpu::BindGroup,

    // ── Temporal history (ping-pong) ──────────────────────────────────────────
    /// Previous frame's TSR output at display resolution.
    pub history_texture: wgpu::Texture,
    pub history_view: wgpu::TextureView,
    /// Current frame's TSR output (rendered to, then copied → history).
    pub output_texture: wgpu::Texture,
    pub output_view: wgpu::TextureView,

    // ── Samplers ──────────────────────────────────────────────────────────────
    linear_sampler: wgpu::Sampler,
    point_sampler: wgpu::Sampler,

    // ── Dimensions ────────────────────────────────────────────────────────────
    internal_width: u32,
    internal_height: u32,
    output_width: u32,
    output_height: u32,

    // ── Reactivity state ──────────────────────────────────────────────────────
    /// `true` until the first frame (or after `reset_history()`).
    first_frame: bool,
    /// Blend bias toward current frame (`0` = full history, `1` = no history).
    reactivity: f32,

    quality: TsrQuality,
}

impl TsrPass {
    /// Create a new TSR pass.
    ///
    /// - `internal_*` — pre-AA (geometry) render resolution.
    /// - `output_*`   — display resolution; history/output textures live here.
    /// - `format`     — swapchain / surface format for the final blit.
    /// - `quality`    — neighbourhood tap count and accumulation parameters.
    pub fn new(
        device: &wgpu::Device,
        internal_width: u32,
        internal_height: u32,
        output_width: u32,
        output_height: u32,
        format: wgpu::TextureFormat,
        quality: TsrQuality,
    ) -> Self {
        let tsr_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TSR Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tsr_main.wgsl").into()),
        });
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TSR Blit Shader"),
            source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TSR Uniform"),
            size: std::mem::size_of::<TsrUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("TSR Linear Sampler"),
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let point_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("TSR Point Sampler"),
            min_filter: wgpu::FilterMode::Nearest,
            mag_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let output_width = output_width.max(1);
        let output_height = output_height.max(1);

        let (history_texture, history_view, output_texture, output_view) =
            Self::create_textures(device, output_width, output_height);

        // ── TSR BGL ───────────────────────────────────────────────────────────
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TSR BGL"),
            entries: &[
                tex_entry(0, wgpu::TextureSampleType::Float { filterable: true }), // current_frame
                tex_entry(1, wgpu::TextureSampleType::Float { filterable: true }), // history_frame
                tex_entry(2, wgpu::TextureSampleType::Depth),                      // depth_tex
                sampler_entry(3, wgpu::SamplerBindingType::Filtering),             // linear_sampler
                sampler_entry(4, wgpu::SamplerBindingType::NonFiltering),          // point_sampler
                camera_storage_entry(5),                                           // camera
                uniform_entry(6),                                                  // tsr
            ],
        });

        // ── Blit BGL ──────────────────────────────────────────────────────────
        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TSR Blit BGL"),
            entries: &[
                tex_entry(0, wgpu::TextureSampleType::Float { filterable: true }),
                sampler_entry(1, wgpu::SamplerBindingType::Filtering),
            ],
        });

        let blit_bind_group = make_blit_bg(device, &blit_bgl, &output_view, &linear_sampler);

        // ── TSR pipeline ──────────────────────────────────────────────────────
        let tsr_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("TSR PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("TSR Pipeline"),
            layout: Some(&tsr_pl),
            vertex: wgpu::VertexState {
                module: &tsr_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &tsr_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    // Rgba16Float so the full HDR range of the resolved image is preserved.
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // ── Blit pipeline ─────────────────────────────────────────────────────
        let blit_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("TSR Blit PL"),
            bind_group_layouts: &[Some(&blit_bgl)],
            immediate_size: 0,
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("TSR Blit Pipeline"),
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
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            bgl,
            bind_group: None,
            bind_group_key: None,
            uniform_buf,
            blit_pipeline,
            blit_bgl,
            blit_bind_group,
            history_texture,
            history_view,
            output_texture,
            output_view,
            linear_sampler,
            point_sampler,
            internal_width,
            internal_height,
            output_width,
            output_height,
            first_frame: true,
            reactivity: 0.0,
            quality,
        }
    }

    // ── Public reactivity API ─────────────────────────────────────────────────

    /// Discard all temporal history.
    ///
    /// Call this on camera cuts, level loads, teleports, or any sudden scene
    /// change where stale history would cause visible ghosting.
    pub fn reset_history(&mut self) {
        self.first_frame = true;
    }

    /// Set a per-frame reactivity bias.
    ///
    /// `factor` in `[0.0, 1.0]`:
    /// - `0.0` — full history (maximum stability, no ghosting bias). **Default.**
    /// - `1.0` — current frame only (maximum reactivity, every frame is fresh).
    ///
    /// Expose this via post-process volumes so different regions of the level
    /// can have different TSR responsiveness.
    pub fn set_reactivity(&mut self, factor: f32) {
        self.reactivity = factor.clamp(0.0, 1.0);
    }

    /// The quality preset this pass was constructed with.
    pub fn quality(&self) -> TsrQuality {
        self.quality
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn create_textures(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (
        wgpu::Texture,
        wgpu::TextureView,
        wgpu::Texture,
        wgpu::TextureView,
    ) {
        let make = |label: &'static str, extra: wgpu::TextureUsages| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT
                    | extra,
                view_formats: &[],
            })
        };
        let history = make("TSR History", wgpu::TextureUsages::COPY_DST);
        let hv = history.create_view(&Default::default());
        let output = make("TSR Output", wgpu::TextureUsages::COPY_SRC);
        let ov = output.create_view(&Default::default());
        (history, hv, output, ov)
    }
}

// ── BGL entry helpers ─────────────────────────────────────────────────────────

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

fn sampler_entry(binding: u32, ty: wgpu::SamplerBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(ty),
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

// The engine-wide camera buffer (`GpuCameraBuffer`, label "Camera Storage") is a
// storage buffer sized for 2 cameras (mono/stereo), matching every other pass's
// `var<storage, read> cameras: array<CameraUniforms, 2>`. Bindings that reference
// it must use this, not `uniform_entry`.
fn camera_storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn make_blit_bg(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("TSR Blit BG"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

// ── RenderPass impl ───────────────────────────────────────────────────────────

impl RenderPass for TsrPass {
    fn name(&self) -> &'static str {
        "TSR"
    }

    fn requires_camera_jitter(&self) -> bool {
        true
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read("pre_aa");
    }

    fn on_resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.output_width = width.max(1);
        self.output_height = height.max(1);

        let (ht, hv, ot, ov) = Self::create_textures(device, self.output_width, self.output_height);
        self.history_texture = ht;
        self.history_view = hv;
        self.output_texture = ot;
        self.output_view = ov;

        // Rebuild blit bind group (references output_view).
        self.blit_bind_group = make_blit_bg(
            device,
            &self.blit_bgl,
            &self.output_view,
            &self.linear_sampler,
        );

        // Invalidate main bind group (references history_view via binding 1).
        self.bind_group = None;
        self.bind_group_key = None;

        // History is stale after resize — prime the next frame.
        self.first_frame = true;
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let jitter = r1_r2_jitter(ctx.frame_num);
        let reset = if self.first_frame {
            self.first_frame = false;
            1u32
        } else {
            0u32
        };

        let u = TsrUniform {
            jitter_offset: jitter,
            reactivity: self.reactivity,
            reset,
            time_delta: ctx.delta_time.max(0.0),
            tap_radius: self.quality.tap_radius(),
            _pad: [0.0; 2],
        };

        ctx.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // ── 1. Lazy bind group ─────────────────────────────────────────────────
        let pre_aa_view = ctx.resources.pre_aa.read("TSR").ok_or_else(|| {
            helio_core::Error::InvalidPassConfig(
                "TsrPass requires frame.pre_aa (published by DeferredLightPass)".into(),
            )
        })?;

        let key = (
            pre_aa_view as *const _ as usize,
            ctx.depth as *const _ as usize,
        );
        if self.bind_group_key != Some(key) {
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("TSR BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(pre_aa_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.history_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(ctx.depth),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&self.point_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: ctx.scene.camera.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.uniform_buf.as_entire_binding(),
                    },
                ],
            }));
            self.bind_group_key = Some(key);
        }

        // ── 2. TSR resolve → output_view ──────────────────────────────────────
        {
            let attachments = [Some(wgpu::RenderPassColorAttachment {
                view: &self.output_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })];
            let mut pass =
                unsafe { &mut *ctx.encoder_ptr }.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("TSR Resolve"),
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

        // ── 3. Copy output → history ───────────────────────────────────────────
        unsafe { &mut *ctx.encoder_ptr }.copy_texture_to_texture(
            self.output_texture.as_image_copy(),
            self.history_texture.as_image_copy(),
            wgpu::Extent3d {
                width: self.output_width,
                height: self.output_height,
                depth_or_array_layers: 1,
            },
        );

        // ── 4. Blit output_view → ctx.target ──────────────────────────────────
        {
            let attachments = [Some(wgpu::RenderPassColorAttachment {
                view: ctx.target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })];
            let mut pass =
                unsafe { &mut *ctx.encoder_ptr }.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("TSR Blit"),
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
