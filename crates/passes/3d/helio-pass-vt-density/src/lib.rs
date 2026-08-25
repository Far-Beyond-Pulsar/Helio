//! Texel-streaming demand feedback (Helio#238 §1) — the frame's measurement
//! half: a quarter-res density target every texel-producing fragment stage
//! writes wanted-mips into, plus the end-of-frame compaction that reduces it
//! to one packed maximum per bindless slot.
//!
//! # Clear → write → compact ordering (this pair's whole reason)
//!
//! 1. **Clear** ([`VtFeedbackClearPass`]) — start of frame, before any
//!    geometry. Zero-fills the quarter-res target with a compute dispatch
//!    rather than a render-pass load-op clear, because the target is never an
//!    attachment: five different passes' fragment stages write it as storage.
//! 2. **Write** — NOT here. The writes happen inside `//!use helio_vt`
//!    (`vt_feedback_write`), beside each existing `textureSample` site in the
//!    gbuffer / forward-lit / transparent / VG / portal shaders, and
//!    analytically (flagged approximate) in the decal compute. Pixels are the
//!    only ground truth for perceptual demand; this crate only provides where
//!    they are recorded.
//! 3. **Compact** ([`VtDensityCompactPass`]) — after the LAST texel-producing
//!    pass of the frame. In the default deferred graph that position is right
//!    after `TransparentPass` (transparent runs post-lighting — see the
//!    registration comment in helio-default-graphs for why "right after the
//!    gbuffer" would silently drop transparent/VG/portal demand). Reduces all
//!    cells to per-slot atomicMax maxima in [`VtDensityCompactPass::feedback_buffer`].
//!
//! The renderer then copies `feedback_buffer` into its own staging buffer and
//! maps it fence-style (the same consume-previous-then-enqueue pattern as the
//! cull-stats readback) — readback ownership deliberately stays OUT of these
//! passes so they record GPU work only and remain trivially stateless between
//! frames.

use helio_core::graph::{ResourceBuilder, ResourceSize};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

const WORKGROUP: u32 = 8;
/// Output buffer words: 256 per-slot maxima + touched counter + padding to a
/// tidy copy size. Layout documented on the WGSL side too.
pub const FEEDBACK_BUFFER_WORDS: u64 = 264;
/// Slot count the feedback arrays cover — the bindless table's width
/// (mirrors libhelio::VT_SLOT_COUNT; pinned by tests).
pub const FEEDBACK_SLOT_COUNT: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FeedbackDims {
    dims: [u32; 2],
    _pad: [u32; 2],
}

fn quarter_dims(width: u32, height: u32) -> (u32, u32) {
    (width.div_ceil(4).max(1), height.div_ceil(4).max(1))
}

fn dispatch_groups(qw: u32, qh: u32) -> (u32, u32) {
    (qw.div_ceil(WORKGROUP).max(1), qh.div_ceil(WORKGROUP).max(1))
}

// ── Shared shader plumbing ───────────────────────────────────────────────────

/// Group-0 layout for the CLEAR entry point: dims uniform @0 + write-only
/// view @1. ONE LAYOUT PER ENTRY POINT — the two share a WGSL module but read
/// different binding slots (`cs_compact` reads @2/@3), and wgpu requires a
/// bind group to fill its layout COMPLETELY, so a single 4-entry layout would
/// make each pass's 2-entry bind group fail `create_bind_group` validation.
fn clear_group0_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("VtFeedback BGL0 Clear"),
        entries: &[
            uniform_entry(0),
            storage_texture_entry(1, wgpu::StorageTextureAccess::WriteOnly),
        ],
    })
}

/// Group-0 layout for the COMPACT entry point: dims uniform @2 + read-only
/// view @3 (the output buffer rides group 1). See [`clear_group0_bgl`].
fn compact_group0_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("VtFeedback BGL0 Compact"),
        entries: &[
            uniform_entry(2),
            storage_texture_entry(3, wgpu::StorageTextureAccess::ReadOnly),
        ],
    })
}

fn group1_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("VtFeedback BGL1"),
        entries: &[storage_buffer_rw_entry(0)],
    })
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_texture_entry(binding: u32, access: wgpu::StorageTextureAccess) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access,
            format: wgpu::TextureFormat::R32Uint,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn storage_buffer_rw_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

// ── Clear ────────────────────────────────────────────────────────────────────

/// Start-of-frame zero-fill of the quarter-res density target. See crate docs
/// for why this exists and why it is a compute pass.
pub struct VtFeedbackClearPass {
    pipeline: wgpu::ComputePipeline,
    bgl0: wgpu::BindGroupLayout,
    dims_buf: wgpu::Buffer,
    bind_group: Option<wgpu::BindGroup>,
    width: u32,
    height: u32,
}

impl VtFeedbackClearPass {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let module = helio_core::shader::module(
            device,
            "vt_feedback_clear",
            include_str!("../shaders/vt_feedback.wgsl"),
        );
        let bgl0 = clear_group0_bgl(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("VtFeedbackClear PL"),
            immediate_size: 0,
            bind_group_layouts: &[Some(&bgl0)],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("VtFeedbackClear Pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("cs_clear"),
            compilation_options: Default::default(),
            cache: None,
        });
        let dims_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VtFeedbackClear Dims"),
            size: std::mem::size_of::<FeedbackDims>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            bgl0,
            dims_buf,
            bind_group: None,
            width: width.max(1),
            height: height.max(1),
        }
    }
}

impl RenderPass for VtFeedbackClearPass {
    fn name(&self) -> &'static str {
        "VtFeedbackClear"
    }

    fn reads(&self) -> &'static [&'static str] {
        &[]
    }

    fn writes(&self) -> &'static [&'static str] {
        &["vt_density"]
    }

    /// Declares the target this pair shares with the compaction pass and the
    /// five raster writers. Quarter-res of the INTERNAL resolution: cells
    /// must track gbuffer pixels (`ScaledInternal`, not `Scaled`). R32Uint
    /// because feedback values are packed `(slot << 8) | (mip+1)` words;
    /// STORAGE_BINDING because the texture lives as storage everywhere, and
    /// COPY_SRC for potential debug readback of raw cells.
    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.write_color_raw(
            "vt_density",
            wgpu::TextureFormat::R32Uint,
            ResourceSize::ScaledInternal { divisor: 4 },
        );
        // STORAGE_BINDING: written from five fragment stages, read by the
        // compaction compute. TEXTURE_BINDING: the debug heatmap samples the
        // finished cells. COPY_SRC: raw-cell readback tooling.
        builder.with_extra_usage(
            wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        );
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        if ctx.resize || self.bind_group.is_none() {
            self.width = ctx.width.max(1);
            self.height = ctx.height.max(1);
            let (qw, qh) = quarter_dims(self.width, self.height);
            let dims = FeedbackDims { dims: [qw, qh], _pad: [0; 2] };
            ctx.write_buffer(&self.dims_buf, 0, bytemuck::bytes_of(&dims));
        }
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if self.bind_group.is_none() {
            let view = ctx
                .resource_pool
                .get_view("vt_density")
                .expect("vt_density must be declared as a graph resource");
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("VtFeedbackClear BG"),
                layout: &self.bgl0,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.dims_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                ],
            }));
        }
        let (qw, qh) = quarter_dims(self.width, self.height);
        let (gx, gy) = dispatch_groups(qw, qh);
        record_dispatch(
            ctx,
            "VtFeedbackClear",
            &self.pipeline,
            &[self.bind_group.as_ref().unwrap()],
            gx,
            gy,
        )
    }

    fn on_resize(&mut self, _device: &wgpu::Device, width: u32, height: u32) {
        // Graph re-allocates the texture; drop the lazy bind group so it
        // rebuilds against the new view next frame (HiZ discipline).
        self.width = width.max(1);
        self.height = height.max(1);
        self.bind_group = None;
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a helio_core::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }
}

// ── Compact ──────────────────────────────────────────────────────────────────

/// End-of-frame reduction of the density target to per-slot maxima. Owns the
/// persistent output buffer the Renderer reads back through its own staging.
pub struct VtDensityCompactPass {
    pipeline: wgpu::ComputePipeline,
    bgl0: wgpu::BindGroupLayout,
    bgl1: wgpu::BindGroupLayout,
    dims_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    bg0: Option<wgpu::BindGroup>,
    bg1: Option<wgpu::BindGroup>,
    width: u32,
    height: u32,
}

impl VtDensityCompactPass {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let module = helio_core::shader::module(
            device,
            "vt_density_compact",
            include_str!("../shaders/vt_feedback.wgsl"),
        );
        let bgl0 = compact_group0_bgl(device);
        let bgl1 = group1_bgl(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("VtDensityCompact PL"),
            immediate_size: 0,
            bind_group_layouts: &[Some(&bgl0), Some(&bgl1)],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("VtDensityCompact Pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("cs_compact"),
            compilation_options: Default::default(),
            cache: None,
        });
        let dims_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VtDensityCompact Dims"),
            size: std::mem::size_of::<FeedbackDims>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VtDensityCompact Output"),
            size: FEEDBACK_BUFFER_WORDS * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            bgl0,
            bgl1,
            dims_buf,
            out_buf,
            bg0: None,
            bg1: None,
            width: width.max(1),
            height: height.max(1),
        }
    }

    /// Per-slot packed maxima + touched counter, fully rewritten every frame.
    ///
    /// Words `[0..256)` hold `libhelio::pack_feedback(slot, max_wanted)`
    /// (zero = slot untouched this frame); word `[256]` counts feedback
    /// stores observed. Copied out by the RENDERER after graph execution —
    /// never mapped or polled here.
    pub fn feedback_buffer(&self) -> &wgpu::Buffer {
        &self.out_buf
    }

    /// Size of one frame's copy out of [`Self::feedback_buffer`] (the whole
    /// word array; small enough that slicing is not worth a second contract).
    pub fn feedback_copy_bytes(&self) -> u64 {
        FEEDBACK_BUFFER_WORDS * 4
    }
}

impl RenderPass for VtDensityCompactPass {
    fn name(&self) -> &'static str {
        "VtDensityCompact"
    }

    fn reads(&self) -> &'static [&'static str] {
        &["vt_density"]
    }

    fn writes(&self) -> &'static [&'static str] {
        &["vt_feedback"]
    }

    /// The graph tracks `vt_feedback` for dependency ordering; the buffer is
    /// allocated here (buffer declarations are bookkeeping-only today).
    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.write_buffer("vt_feedback");
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        if ctx.resize || self.bg0.is_none() || self.bg1.is_none() {
            self.width = ctx.width.max(1);
            self.height = ctx.height.max(1);
            let (qw, qh) = quarter_dims(self.width, self.height);
            let dims = FeedbackDims { dims: [qw, qh], _pad: [0; 2] };
            ctx.write_buffer(&self.dims_buf, 0, bytemuck::bytes_of(&dims));
        }
        // Zero the output EVERY frame BEFORE the atomics land. queue writes
        // execute at enqueue time, ordered ahead of everything submitted
        // after — the same discipline Hi-Z globals use.
        ctx.write_buffer(&self.out_buf, 0, &vec![0u8; (FEEDBACK_BUFFER_WORDS * 4) as usize]);
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if self.bg0.is_none() || self.bg1.is_none() {
            let view = ctx
                .resource_pool
                .get_view("vt_density")
                .expect("vt_density must be declared as a graph resource");
            self.bg0 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("VtDensityCompact BG0"),
                layout: &self.bgl0,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.dims_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                ],
            }));
            self.bg1 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("VtDensityCompact BG1"),
                layout: &self.bgl1,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.out_buf.as_entire_binding(),
                }],
            }));
        }
        let (qw, qh) = quarter_dims(self.width, self.height);
        let (gx, gy) = dispatch_groups(qw, qh);
        record_dispatch(
            ctx,
            "VtDensityCompact",
            &self.pipeline,
            &[self.bg0.as_ref().unwrap(), self.bg1.as_ref().unwrap()],
            gx,
            gy,
        )
    }

    fn on_resize(&mut self, _device: &wgpu::Device, width: u32, height: u32) {
        self.width = width.max(1);
        self.height = height.max(1);
        self.bg0 = None;
        self.bg1 = None;
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a helio_core::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }
}

// ── Debug viewmodes ──────────────────────────────────────────────────────────

/// Heatmap mode for [`VtHeatmapPass`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VtHeatmapMode {
    /// Pass renders nothing (default).
    Off,
    /// Demand heat: green→yellow by wanted mip.
    Density,
    /// Page-miss flash: red where demand exceeds the published floor.
    MissFlash,
}

impl VtHeatmapMode {
    fn as_u32(self) -> u32 {
        match self {
            VtHeatmapMode::Off => 0,
            VtHeatmapMode::Density => 0,
            VtHeatmapMode::MissFlash => 1,
        }
    }

    fn active(self) -> bool {
        !matches!(self, VtHeatmapMode::Off)
    }
}

/// Fullscreen overlay visualizing the quarter-res feedback target. Reads
/// `vt_density` (as a plain texture — compaction has already consumed it) and
/// the VT meta rows; draws over `ctx.target`. Toggled per-frame by the host:
///
/// ```ignore
/// renderer.find_pass_mut::<VtHeatmapPass>()
///     .map(|p| p.set_mode(VtHeatmapMode::MissFlash));
/// ```
pub struct VtHeatmapPass {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    dims_buf: wgpu::Buffer,
    vt_binder: helio_core::shader::vt_binder::VtGroupBinder,
    bg: Option<wgpu::BindGroup>,
    bg_key: helio_core::shader::vt_binder::VtGroupKey,
    mode: VtHeatmapMode,
}

impl VtHeatmapPass {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let module = helio_core::shader::resolve(include_str!("../shaders/vt_heatmap.wgsl"))
            .to_string();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VT Heatmap"),
            source: wgpu::ShaderSource::Wgsl(module.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VtHeatmap BGL"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        // Group 2 layout comes straight from the shared binder (meta rows).
        let vt_binder = helio_core::shader::vt_binder::VtGroupBinder::new(device);
        let vt_layout = vt_binder.layout().clone();
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("VtHeatmap PL"),
            bind_group_layouts: &[Some(&bgl), Some(&vt_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("VtHeatmap Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let dims_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VtHeatmap Uniforms"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            bgl,
            dims_buf,
            vt_binder,
            bg: None,
            bg_key: helio_core::shader::vt_binder::VtGroupKey::default(),
            mode: VtHeatmapMode::Off,
        }
    }

    /// Switches the visualization mode (host-facing toggle).
    pub fn set_mode(&mut self, mode: VtHeatmapMode) {
        self.mode = mode;
    }

    pub fn mode(&self) -> VtHeatmapMode {
        self.mode
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HeatmapUniforms {
    dims_mode: [u32; 4],
}

impl RenderPass for VtHeatmapPass {
    fn name(&self) -> &'static str {
        "VtHeatmap"
    }

    fn reads(&self) -> &'static [&'static str] {
        &["vt_density", "main_scene"]
    }

    fn writes(&self) -> &'static [&'static str] {
        &[]
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let (qw, qh) = quarter_dims(ctx.width.max(1), ctx.height.max(1));
        let uniforms = HeatmapUniforms {
            dims_mode: [qw, qh, self.mode.as_u32(), 0],
        };
        ctx.write_buffer(&self.dims_buf, 0, bytemuck::bytes_of(&uniforms));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if !self.mode.active() {
            return Ok(());
        }
        let main_scene = ctx.resources.main_scene.read(self.name());
        let meta_buf = main_scene.and_then(|ms| ms.vt_bindings.get().map(|v| v.vt_meta_buffer));
        let density_view = ctx.resource_pool.get_view("vt_density");
        let key = helio_core::shader::vt_binder::VtGroupKey {
            meta_ptr: meta_buf.map(|b| b as *const _ as usize).unwrap_or(0),
            density_ptr: density_view.map(|v| v as *const _ as usize).unwrap_or(0),
            version: 0,
        };
        if self.bg_key != key || self.bg.is_none() {
            self.bg = Some(self.vt_binder.bind_group(ctx.device, meta_buf, density_view));
            self.bg_key = key;
        }
        let pass = unsafe { &mut *ctx.active_render_pass_ptr().expect("VtHeatmap must fuse into a render pass") };
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.make_bg0(ctx.device, density_view), &[]);
        pass.set_bind_group(1, self.bg.as_ref().unwrap(), &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }
}

impl VtHeatmapPass {
    fn make_bg0(&self, device: &wgpu::Device, density: Option<&wgpu::TextureView>) -> wgpu::BindGroup {
        // Fallback 1×1 zero-cell texture for graphs without the feedback pair.
        let fallback = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("VtHeatmap Fallback Cells"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default());
        let view = density.unwrap_or(&fallback);
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("VtHeatmap BG0"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.dims_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(view),
                },
            ],
        })
    }
}


// ── Shared dispatch recording ────────────────────────────────────────────────

/// Records the dispatch through whichever encoder the executor provided —
/// compute passes must ride the graph's active compute pass when fused.
fn record_dispatch(
    ctx: &mut PassContext<'_>,
    label: &'static str,
    pipeline: &wgpu::ComputePipeline,
    bind_groups: &[&wgpu::BindGroup],
    gx: u32,
    gy: u32,
) -> HelioResult<()> {
    let _ = label; // profiling scopes land with begin/end_gpu_pass lifetime fixes
    if let Some(encoder_ptr) = ctx.active_compute_pass_ptr() {
        let pass = unsafe { &mut *encoder_ptr };
        pass.set_pipeline(pipeline);
        for (idx, bg) in bind_groups.iter().enumerate() {
            pass.set_bind_group(idx as u32, *bg, &[]);
        }
        pass.dispatch_workgroups(gx, gy, 1);
    } else {
        let descriptor = wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        };
        let mut cpass = ctx.begin_compute_pass(&descriptor);
        cpass.set_pipeline(pipeline);
        for (idx, bg) in bind_groups.iter().enumerate() {
            cpass.set_bind_group(idx as u32, *bg, &[]);
        }
        cpass.dispatch_workgroups(gx, gy, 1);
    }
    Ok(())
}
