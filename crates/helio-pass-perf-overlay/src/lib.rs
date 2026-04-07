//! GPU-based performance overlay pass for Helio renderer.
//!
//! Provides real-time visualization of rendering performance metrics:
//! - **Pass-to-pass overdraw**: Tracks when passes overwrite each other's pixels
//! - **Shader complexity**: Heatmap based on GBuffer ORM values
//! - **Tile light count**: Forward+ culling visualization
//! - **Pass output inspection**: Debug viewer for render targets
//!
//! Zero cost when disabled. Works universally with all passes without shader modifications.

use std::sync::{Arc, Mutex};
use bytemuck::{Pod, Zeroable};
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

pub const TILE_SIZE: u32 = 16;

// ─────────────────────────────────────────────────────────────────────────────
// Visualization Modes
// ─────────────────────────────────────────────────────────────────────────────

/// Performance overlay visualization modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum PerfOverlayMode {
    /// Disabled: zero GPU cost.
    #[default]
    Disabled = 0,
    /// Pass-to-pass overdraw tracking (warm = high pass overlap).
    PassOverdraw = 1,
    /// Shader complexity based on GBuffer ORM heuristic.
    ShaderComplexity = 2,
    /// Tile light count from forward+ culling (warm = many lights).
    TileLightCount = 3,
    /// Pass output inspector (debug viewer for render targets).
    PassOutput = 4,
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU-side uniforms
// ─────────────────────────────────────────────────────────────────────────────

/// Depth comparison compute shader parameters.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ColorCompareParams {
    screen_width: u32,
    screen_height: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Tile aggregation compute shader parameters.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AggregateParams {
    num_tiles_x: u32,
    num_tiles_y: u32,
    num_tiles: u32,
    screen_width: u32,
    screen_height: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Tile metrics (aggregated per 16×16 tile).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TileMetrics {
    pass_overdraw_max: u32,  // Max pass overwrites in tile
    light_count: u32,        // From LightCullPass
    complexity_avg: u32,     // GBuffer ORM heuristic
    _pad: u32,
}

/// Visualization shader parameters.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VisualizeParams {
    mode: u32,              // PerfOverlayMode as u32
    num_tiles_x: u32,
    num_tiles_y: u32,
    internal_width: u32,    // Buffer dimensions (internal resolution)
    internal_height: u32,
    display_width: u32,     // Target dimensions (display resolution)
    display_height: u32,
    opacity: f32,           // Blend factor (0.0 = invisible, 1.0 = full overlay)
    heatmap_scale: f32,     // Max value for normalization
    _pad0: u32,
    _pad1: u32,
}

struct PerfOverlayRuntime {
    frame_num: u64,
    snapshot_valid: bool,
}

pub struct PerfOverlayShared {
    // Internal (render) resolution - buffers are sized to this
    internal_width: u32,
    internal_height: u32,
    // Display (output) resolution - rendering target size
    display_width: u32,
    display_height: u32,
    num_tiles_x: u32,
    num_tiles_y: u32,

    color_snapshot_prev: wgpu::Texture,
    color_snapshot_prev_view: wgpu::TextureView,
    pass_overdraw_buf: wgpu::Buffer,

    color_compare_pipeline: wgpu::ComputePipeline,
    color_compare_bgl: wgpu::BindGroupLayout,
    color_compare_params_buf: wgpu::Buffer,

    blit_pipeline: wgpu::ComputePipeline,
    blit_bgl: wgpu::BindGroupLayout,

    mode: Mutex<PerfOverlayMode>,
    opacity: Mutex<f32>,
    runtime: Mutex<PerfOverlayRuntime>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass struct
// ─────────────────────────────────────────────────────────────────────────────

pub struct PerfOverlayPass {
    shared: Arc<Mutex<PerfOverlayShared>>,
    aggregate_pipeline: wgpu::ComputePipeline,
    aggregate_bgl: wgpu::BindGroupLayout,
    aggregate_params_buf: wgpu::Buffer,
    aggregate_bind_group: Option<wgpu::BindGroup>,
    tile_metrics_buf: wgpu::Buffer,
    visualize_pipeline: wgpu::RenderPipeline,
    visualize_bgl: wgpu::BindGroupLayout,
    visualize_params_buf: wgpu::Buffer,
    visualize_bind_group: Option<wgpu::BindGroup>,
    bind_group_key: Option<(usize, usize, usize)>,
}

/// Analyzer pass that runs after render passes and compares the current depth
/// buffer against the previous snapshot to count pass overwrites.
pub struct PerfOverlayAnalyzerPass {
    shared: Arc<Mutex<PerfOverlayShared>>,
}

impl PerfOverlayShared {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Arc<Mutex<Self>> {
        let num_tiles_x = width.div_ceil(TILE_SIZE);
        let num_tiles_y = height.div_ceil(TILE_SIZE);

        let color_snapshot_prev = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("PerfOverlay Color Snapshot"),
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
                | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let color_snapshot_prev_view =
            color_snapshot_prev.create_view(&wgpu::TextureViewDescriptor::default());

        let pixel_count = (width as u64) * (height as u64);
        let pass_overdraw_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PerfOverlay Pass Overdraw Counters"),
            size: pixel_count * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let color_compare_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PerfOverlay Color Compare Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/analyze_color_overdraw.wgsl").into(),
            ),
        });

        let color_compare_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("PerfOverlay Color Compare BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let color_compare_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("PerfOverlay Color Compare PL"),
                bind_group_layouts: &[Some(&color_compare_bgl)],
                immediate_size: 0,
            });

        let color_compare_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("PerfOverlay Color Compare Pipeline"),
                layout: Some(&color_compare_pipeline_layout),
                module: &color_compare_shader,
                entry_point: Some("analyze_color_overdraw"),
                compilation_options: Default::default(),
                cache: None,
            });

        let color_compare_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PerfOverlay Color Compare Params"),
            size: std::mem::size_of::<ColorCompareParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Blit shader for copying color textures
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PerfOverlay Blit Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/blit_color.wgsl").into(),
            ),
        });

        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PerfOverlay Blit BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PerfOverlay Blit PL"),
            bind_group_layouts: &[Some(&blit_bgl)],
            immediate_size: 0,
        });

        let blit_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PerfOverlay Blit Pipeline"),
            layout: Some(&blit_pipeline_layout),
            module: &blit_shader,
            entry_point: Some("blit_color"),
            compilation_options: Default::default(),
            cache: None,
        });

        Arc::new(Mutex::new(Self {
            internal_width: width,
            internal_height: height,
            display_width: width, // Initially same as internal
            display_height: height,
            num_tiles_x,
            num_tiles_y,
            color_snapshot_prev,
            color_snapshot_prev_view,
            pass_overdraw_buf,
            color_compare_pipeline,
            color_compare_bgl,
            color_compare_params_buf,
            blit_pipeline,
            blit_bgl,
            mode: Mutex::new(PerfOverlayMode::Disabled),
            opacity: Mutex::new(0.6),
            runtime: Mutex::new(PerfOverlayRuntime {
                frame_num: 0,
                snapshot_valid: false,
            }),
        }))
    }

    pub fn on_resize(&mut self, _device: &wgpu::Device, width: u32, height: u32) {
        // Update display resolution (rendering target size)
        // Buffers remain at internal resolution to match pre_aa buffer
        if width == self.display_width && height == self.display_height {
            return;
        }

        self.display_width = width;
        self.display_height = height;
        // Note: internal buffers are NOT resized here - they stay at internal resolution
        // to match the pre_aa color buffer they're tracking
    }

    pub fn get_mode(&self) -> PerfOverlayMode {
        *self.mode.lock().unwrap()
    }

    pub fn set_mode(&self, mode: PerfOverlayMode) {
        *self.mode.lock().unwrap() = mode;
    }

    pub fn get_opacity(&self) -> f32 {
        *self.opacity.lock().unwrap()
    }

    pub fn set_opacity(&self, opacity: f32) {
        *self.opacity.lock().unwrap() = opacity.clamp(0.0, 1.0);
    }
}

impl PerfOverlayPass {
    pub fn new(
        device: &wgpu::Device,
        shared: Arc<Mutex<PerfOverlayShared>>,
        target_format: wgpu::TextureFormat,
    ) -> Self {
        let shared_guard = shared.lock().unwrap();
        let num_tiles = shared_guard
            .num_tiles_x
            .checked_mul(shared_guard.num_tiles_y)
            .expect("tile grid overflow: viewport dimensions too large");
        drop(shared_guard);

        let aggregate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PerfOverlay Aggregate Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/aggregate_tiles.wgsl").into(),
            ),
        });

        let aggregate_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PerfOverlay Aggregate BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let aggregate_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("PerfOverlay Aggregate PL"),
                bind_group_layouts: &[Some(&aggregate_bgl)],
                immediate_size: 0,
            });

        let aggregate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PerfOverlay Aggregate Pipeline"),
            layout: Some(&aggregate_pipeline_layout),
            module: &aggregate_shader,
            entry_point: Some("cs_aggregate_tiles"),
            compilation_options: Default::default(),
            cache: None,
        });

        let aggregate_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PerfOverlay Aggregate Params"),
            size: std::mem::size_of::<AggregateParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tile_metrics_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PerfOverlay Tile Metrics"),
            size: (num_tiles as u64 * std::mem::size_of::<TileMetrics>() as u64).max(4),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let visualize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PerfOverlay Visualize Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/visualize.wgsl").into()),
        });

        let visualize_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PerfOverlay Visualize BGL"),
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
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let visualize_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("PerfOverlay Visualize PL"),
                bind_group_layouts: &[Some(&visualize_bgl)],
                immediate_size: 0,
            });

        let visualize_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PerfOverlay Visualize Pipeline"),
            layout: Some(&visualize_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &visualize_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &visualize_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
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

        let visualize_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PerfOverlay Visualize Params"),
            size: std::mem::size_of::<VisualizeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            shared,
            aggregate_pipeline,
            aggregate_bgl,
            aggregate_params_buf,
            aggregate_bind_group: None,
            tile_metrics_buf,
            visualize_pipeline,
            visualize_bgl,
            visualize_params_buf,
            visualize_bind_group: None,
            bind_group_key: None,
        }
    }

    pub fn set_mode(&mut self, mode: PerfOverlayMode) {
        self.shared.lock().unwrap().set_mode(mode);
    }

    pub fn set_opacity(&mut self, opacity: f32) {
        self.shared.lock().unwrap().set_opacity(opacity);
    }
}

impl PerfOverlayAnalyzerPass {
    pub fn new(shared: Arc<Mutex<PerfOverlayShared>>) -> Self {
        Self { shared }
    }
}

impl RenderPass for PerfOverlayPass {
    fn name(&self) -> &'static str {
        "PerfOverlay"
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let shared = self.shared.lock().unwrap();
        let aggregate_params = AggregateParams {
            num_tiles_x: shared.num_tiles_x,
            num_tiles_y: shared.num_tiles_y,
            num_tiles: shared.num_tiles_x * shared.num_tiles_y,
            screen_width: shared.internal_width,
            screen_height: shared.internal_height,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        ctx.write_buffer(
            &self.aggregate_params_buf,
            0,
            bytemuck::bytes_of(&aggregate_params),
        );

        let visualize_params = VisualizeParams {
            mode: shared.mode.lock().unwrap().clone() as u32,
            num_tiles_x: shared.num_tiles_x,
            num_tiles_y: shared.num_tiles_y,
            internal_width: shared.internal_width,
            internal_height: shared.internal_height,
            display_width: shared.display_width,
            display_height: shared.display_height,
            opacity: *shared.opacity.lock().unwrap(),
            heatmap_scale: 5.0,
            _pad0: 0,
            _pad1: 0,
        };
        ctx.write_buffer(
            &self.visualize_params_buf,
            0,
            bytemuck::bytes_of(&visualize_params),
        );

        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let shared = self.shared.lock().unwrap();
        if *shared.mode.lock().unwrap() == PerfOverlayMode::Disabled {
            return Ok(());
        }

        if let (Some(gbuffer), Some(tile_light_counts)) = (ctx.resources.gbuffer, ctx.resources.tile_light_counts) {
            let gbuffer_orm_ptr = gbuffer.orm as *const _ as usize;
            let tile_light_counts_ptr = tile_light_counts as *const _ as usize;
            let key = (gbuffer_orm_ptr, tile_light_counts_ptr, 0);

            if self.bind_group_key != Some(key) {
                self.aggregate_bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("PerfOverlay Aggregate BG"),
                    layout: &self.aggregate_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.aggregate_params_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: shared.pass_overdraw_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(gbuffer.orm),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: tile_light_counts.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.tile_metrics_buf.as_entire_binding(),
                        },
                    ],
                }));
                self.bind_group_key = Some(key);
            }

            let mut pass = ctx.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PerfOverlay Aggregate"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.aggregate_pipeline);
            pass.set_bind_group(0, self.aggregate_bind_group.as_ref().unwrap(), &[]);
            let num_tiles = shared.num_tiles_x * shared.num_tiles_y;
            pass.dispatch_workgroups(num_tiles.div_ceil(256), 1, 1);
        }

        if let Some(pre_aa) = ctx.resources.pre_aa {
            let pre_aa_ptr = pre_aa as *const _ as usize;
            let key = (0, 0, pre_aa_ptr);

            if self.bind_group_key != Some(key) || self.visualize_bind_group.is_none() {
                let scene_sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("PerfOverlay Scene Sampler"),
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    ..Default::default()
                });

                self.visualize_bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("PerfOverlay Visualize BG"),
                    layout: &self.visualize_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.visualize_params_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: shared.pass_overdraw_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.tile_metrics_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(pre_aa),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::Sampler(&scene_sampler),
                        },
                    ],
                }));
            }

            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: ctx.target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })];
            let render_pass_desc = wgpu::RenderPassDescriptor {
                label: Some("PerfOverlay Visualize"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            };

            let mut pass = ctx.begin_render_pass(&render_pass_desc);
            pass.set_pipeline(&self.visualize_pipeline);
            pass.set_bind_group(0, self.visualize_bind_group.as_ref().unwrap(), &[]);
            pass.draw(0..3, 0..1);
        }

        Ok(())
    }

    fn on_resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let mut shared = self.shared.lock().unwrap();
        shared.on_resize(device, width, height);

        let num_tiles = shared.num_tiles_x * shared.num_tiles_y;
        self.tile_metrics_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PerfOverlay Tile Metrics"),
            size: (num_tiles as u64 * std::mem::size_of::<TileMetrics>() as u64).max(4),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        self.aggregate_bind_group = None;
        self.visualize_bind_group = None;
        self.bind_group_key = None;
    }
}

impl RenderPass for PerfOverlayAnalyzerPass {
    fn name(&self) -> &'static str {
        "PerfOverlay Color Analyzer"
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let shared = self.shared.lock().unwrap();
        let color_compare_params = ColorCompareParams {
            screen_width: shared.internal_width,
            screen_height: shared.internal_height,
            _pad0: 0,
            _pad1: 0,
        };
        ctx.write_buffer(
            &shared.color_compare_params_buf,
            0,
            bytemuck::bytes_of(&color_compare_params),
        );
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let shared = self.shared.lock().unwrap();
        if *shared.mode.lock().unwrap() != PerfOverlayMode::PassOverdraw {
            return Ok(());
        }

        // Get the color render target (pre-AA buffer)
        let color_texture = if let Some(pre_aa) = ctx.resources.pre_aa {
            pre_aa
        } else {
            // If no pre_aa, use the main target (though this is less ideal)
            ctx.target
        };

        if shared.runtime.lock().unwrap().frame_num != ctx.frame_num {
            ctx.encoder.clear_buffer(&shared.pass_overdraw_buf, 0, None);
            let mut runtime = shared.runtime.lock().unwrap();
            runtime.frame_num = ctx.frame_num;
            runtime.snapshot_valid = false;
        }

        let mut runtime = shared.runtime.lock().unwrap();
        if runtime.snapshot_valid {
            let color_compare_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PerfOverlay Color Compare BG"),
                layout: &shared.color_compare_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: shared.color_compare_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&shared.color_snapshot_prev_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(color_texture),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: shared.pass_overdraw_buf.as_entire_binding(),
                    },
                ],
            });

            let mut pass = ctx.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PerfOverlay Color Compare"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&shared.color_compare_pipeline);
            pass.set_bind_group(0, &color_compare_bg, &[]);
            let dispatch_x = shared.internal_width.div_ceil(16);
            let dispatch_y = shared.internal_height.div_ceil(16);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        } else {
            runtime.snapshot_valid = true;
        }
        drop(runtime);

        // Copy current color to snapshot for next pass comparison using blit shader
        let blit_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PerfOverlay Blit BG"),
            layout: &shared.blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(color_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&shared.color_snapshot_prev_view),
                },
            ],
        });

        let mut pass = ctx.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("PerfOverlay Blit Color"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&shared.blit_pipeline);
        pass.set_bind_group(0, &blit_bg, &[]);
        let dispatch_x = shared.internal_width.div_ceil(16);
        let dispatch_y = shared.internal_height.div_ceil(16);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);

        Ok(())
    }

    fn on_resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.shared.lock().unwrap().on_resize(device, width, height);
    }
}
