//! GPU-based performance overlay pass for Helio renderer.
//!
//! Provides real-time visualization of rendering performance metrics:
//! - **Pass-to-pass overdraw**: Tracks when passes overwrite each other's pixels
//! - **Shader complexity**: Heatmap based on GBuffer ORM values
//! - **Tile light count**: Forward+ culling visualization
//! - **Pass output inspection**: Debug viewer for render targets
//!
//! Zero cost when disabled. Works universally with all passes without shader modifications.

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
struct DepthCompareParams {
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
    opacity: f32,           // Blend factor (0.0 = invisible, 1.0 = full overlay)
    heatmap_scale: f32,     // Max value for normalization
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass struct
// ─────────────────────────────────────────────────────────────────────────────

/// Performance overlay pass.
///
/// Tracks pass-to-pass overdraw by comparing depth snapshots and visualizes
/// performance metrics as GPU-rendered heatmaps. Zero cost when disabled.
pub struct PerfOverlayPass {
    // ── Depth snapshot tracking ──────────────────────────────────────────────
    /// Previous frame's depth snapshot for comparison.
    depth_snapshot_prev: wgpu::Texture,
    depth_snapshot_prev_view: wgpu::TextureView,
    /// Per-pixel pass overdraw counter (u32 per pixel).
    pass_overdraw_buf: wgpu::Buffer,

    // ── Depth comparison compute ─────────────────────────────────────────────
    depth_compare_pipeline: wgpu::ComputePipeline,
    depth_compare_bgl: wgpu::BindGroupLayout,
    depth_compare_params_buf: wgpu::Buffer,
    depth_compare_bind_group: Option<wgpu::BindGroup>,

    // ── Tile aggregation (compute) ───────────────────────────────────────────
    aggregate_pipeline: wgpu::ComputePipeline,
    aggregate_bgl: wgpu::BindGroupLayout,
    aggregate_params_buf: wgpu::Buffer,
    aggregate_bind_group: Option<wgpu::BindGroup>,
    /// Tile metrics buffer (TileMetrics per 16×16 tile).
    tile_metrics_buf: wgpu::Buffer,

    // ── Visualization (fullscreen quad) ──────────────────────────────────────
    visualize_pipeline: wgpu::RenderPipeline,
    visualize_bgl: wgpu::BindGroupLayout,
    visualize_params_buf: wgpu::Buffer,
    visualize_bind_group: Option<wgpu::BindGroup>,

    // ── Cached state ─────────────────────────────────────────────────────────
    bind_group_key: Option<(usize, usize, usize)>, // (gbuffer_orm_ptr, tile_light_counts_ptr, pre_aa_ptr)
    num_tiles_x: u32,
    num_tiles_y: u32,
    width: u32,
    height: u32,
    first_frame: bool, // Skip comparison on first frame

    // ── Runtime control ──────────────────────────────────────────────────────
    mode: PerfOverlayMode,
    opacity: f32,
}

impl PerfOverlayPass {
    /// Creates a new performance overlay pass.
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        target_format: wgpu::TextureFormat,
    ) -> Self {
        let num_tiles_x = width.div_ceil(TILE_SIZE);
        let num_tiles_y = height.div_ceil(TILE_SIZE);
        let num_tiles = num_tiles_x
            .checked_mul(num_tiles_y)
            .expect("tile grid overflow: viewport dimensions too large");

        // ── Depth snapshot texture ───────────────────────────────────────────
        let depth_snapshot_prev = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("PerfOverlay Depth Snapshot"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let depth_snapshot_prev_view =
            depth_snapshot_prev.create_view(&wgpu::TextureViewDescriptor::default());

        // ── Pass overdraw counter buffer ─────────────────────────────────────
        let pixel_count = (width as u64) * (height as u64);
        let pass_overdraw_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PerfOverlay Pass Overdraw Counters"),
            size: pixel_count * 4, // u32 per pixel
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Depth comparison compute pipeline ────────────────────────────────
        let depth_compare_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PerfOverlay Depth Compare Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/analyze_pass_overdraw.wgsl").into(),
            ),
        });

        let depth_compare_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("PerfOverlay Depth Compare BGL"),
                entries: &[
                    // 0: params (uniform)
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
                    // 1: depth_prev (texture_depth_2d)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // 2: depth_current (texture_depth_2d)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // 3: pass_overdraw (storage, read_write)
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

        let depth_compare_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("PerfOverlay Depth Compare PL"),
                bind_group_layouts: &[Some(&depth_compare_bgl)],
                immediate_size: 0,
            });

        let depth_compare_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("PerfOverlay Depth Compare Pipeline"),
                layout: Some(&depth_compare_pipeline_layout),
                module: &depth_compare_shader,
                entry_point: Some("analyze_pass_overdraw"),
                compilation_options: Default::default(),
                cache: None,
            });

        let depth_compare_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PerfOverlay Depth Compare Params"),
            size: std::mem::size_of::<DepthCompareParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Tile aggregation compute pipeline ────────────────────────────────
        let aggregate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PerfOverlay Aggregate Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/aggregate_tiles.wgsl").into(),
            ),
        });

        let aggregate_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PerfOverlay Aggregate BGL"),
            entries: &[
                // 0: params (uniform)
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
                // 1: pass_overdraw_counters (storage, read)
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
                // 2: gbuffer_orm (texture_2d<f32>)
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
                // 3: tile_light_counts (storage, read)
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
                // 4: tile_metrics (storage, read_write)
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

        // ── Visualization fullscreen quad pipeline ───────────────────────────
        let visualize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PerfOverlay Visualize Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/visualize.wgsl").into()),
        });

        let visualize_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PerfOverlay Visualize BGL"),
            entries: &[
                // 0: params (uniform)
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
                // 1: tile_metrics (storage, read)
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
                // 2: scene_color (texture_2d<f32>)
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
                // 3: scene_sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
            depth_snapshot_prev,
            depth_snapshot_prev_view,
            pass_overdraw_buf,
            depth_compare_pipeline,
            depth_compare_bgl,
            depth_compare_params_buf,
            depth_compare_bind_group: None,
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
            num_tiles_x,
            num_tiles_y,
            width,
            height,
            first_frame: true,
            mode: PerfOverlayMode::Disabled,
            opacity: 0.6,
        }
    }

    /// Sets the visualization mode.
    pub fn set_mode(&mut self, mode: PerfOverlayMode) {
        self.mode = mode;
    }

    /// Sets the overlay opacity (0.0 = invisible, 1.0 = full overlay).
    pub fn set_opacity(&mut self, opacity: f32) {
        self.opacity = opacity.clamp(0.0, 1.0);
    }
}

impl RenderPass for PerfOverlayPass {
    fn name(&self) -> &'static str {
        "PerfOverlay"
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        // Upload depth compare params
        let depth_compare_params = DepthCompareParams {
            screen_width: self.width,
            screen_height: self.height,
            _pad0: 0,
            _pad1: 0,
        };
        ctx.write_buffer(
            &self.depth_compare_params_buf,
            0,
            bytemuck::bytes_of(&depth_compare_params),
        );

        // Upload aggregate params
        let aggregate_params = AggregateParams {
            num_tiles_x: self.num_tiles_x,
            num_tiles_y: self.num_tiles_y,
            num_tiles: self.num_tiles_x * self.num_tiles_y,
            screen_width: self.width,
            screen_height: self.height,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        ctx.write_buffer(
            &self.aggregate_params_buf,
            0,
            bytemuck::bytes_of(&aggregate_params),
        );

        // Upload visualize params
        let visualize_params = VisualizeParams {
            mode: self.mode as u32,
            num_tiles_x: self.num_tiles_x,
            num_tiles_y: self.num_tiles_y,
            opacity: self.opacity,
            heatmap_scale: 5.0, // Max expected pass overlaps
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        ctx.write_buffer(
            &self.visualize_params_buf,
            0,
            bytemuck::bytes_of(&visualize_params),
        );

        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // Zero-cost early exit when disabled
        if self.mode == PerfOverlayMode::Disabled {
            return Ok(());
        }

        // Get required resources
        let depth_texture = if let Some(full_res) = ctx.resources.full_res_depth_texture {
            full_res
        } else if let Some(depth_texture) = ctx.resources.depth_texture {
            depth_texture
        } else {
            return Ok(()); // No depth buffer available
        };

        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // ── Step 1: Clear overdraw counters ─────────────────────────────────────
        ctx.encoder.clear_buffer(&self.pass_overdraw_buf, 0, None);

        // ── Step 2: Depth comparison (skip on first frame) ──────────────────────
        if !self.first_frame {
            // Create bind group for depth comparison
            let depth_compare_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PerfOverlay Depth Compare BG"),
                layout: &self.depth_compare_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.depth_compare_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.depth_snapshot_prev_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.pass_overdraw_buf.as_entire_binding(),
                    },
                ],
            });

            let mut pass = ctx.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PerfOverlay Depth Compare"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.depth_compare_pipeline);
            pass.set_bind_group(0, &depth_compare_bg, &[]);

            let dispatch_x = self.width.div_ceil(16);
            let dispatch_y = self.height.div_ceil(16);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        // ── Step 3: Aggregate tiles ──────────────────────────────────────────────
        // Create/update bind group if resources changed
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
                            resource: self.pass_overdraw_buf.as_entire_binding(),
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

            let num_tiles = self.num_tiles_x * self.num_tiles_y;
            pass.dispatch_workgroups(num_tiles.div_ceil(256), 1, 1);
        }

        // ── Step 4: Visualize heatmap ─────────────────────────────────────────────
        if let Some(pre_aa) = ctx.resources.pre_aa {
            let pre_aa_ptr = pre_aa as *const _ as usize;
            let key = (0, 0, pre_aa_ptr);

            if self.bind_group_key != Some(key) || self.visualize_bind_group.is_none() {
                // Create a simple linear sampler for scene texture
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
                            resource: self.tile_metrics_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(pre_aa),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
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
            pass.draw(0..3, 0..1); // Fullscreen triangle
        }

        // ── Step 5: Copy current depth snapshot for next frame ────────────────
        ctx.encoder.copy_texture_to_texture(
            depth_texture.as_image_copy(),
            self.depth_snapshot_prev.as_image_copy(),
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        self.first_frame = false;
        Ok(())
    }

    fn on_resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == self.width && height == self.height {
            return;
        }

        self.width = width;
        self.height = height;
        self.num_tiles_x = width.div_ceil(TILE_SIZE);
        self.num_tiles_y = height.div_ceil(TILE_SIZE);

        // Recreate depth snapshot texture
        self.depth_snapshot_prev = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("PerfOverlay Depth Snapshot"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.depth_snapshot_prev_view =
            self.depth_snapshot_prev
                .create_view(&wgpu::TextureViewDescriptor::default());

        // Recreate pass overdraw buffer
        let pixel_count = (width as u64) * (height as u64);
        self.pass_overdraw_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PerfOverlay Pass Overdraw Counters"),
            size: pixel_count * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Recreate tile metrics buffer
        let num_tiles = self.num_tiles_x * self.num_tiles_y;
        self.tile_metrics_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PerfOverlay Tile Metrics"),
            size: (num_tiles as u64 * std::mem::size_of::<TileMetrics>() as u64).max(4),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Invalidate bind groups
        self.depth_compare_bind_group = None;
        self.aggregate_bind_group = None;
        self.visualize_bind_group = None;
        self.bind_group_key = None;
        self.first_frame = true;
    }
}
