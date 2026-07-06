use crate::{
    AggregateParams, ColorCompareParams, ComputeCostParams, MaterialProfileParams,
    MaterialTimingEntry, PerfOverlayMode, PerfOverlayRuntime, TileMetrics, VisualizeParams,
    TILE_SIZE,
};
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use std::sync::{Arc, Mutex};

// ── Shared GPU state (created once, shared between passes) ──────────────────────

pub struct PerfOverlayShared {
    pub(crate) internal_width: u32,
    pub(crate) internal_height: u32,
    pub(crate) display_width: u32,
    pub(crate) display_height: u32,
    pub(crate) num_tiles_x: u32,
    pub(crate) num_tiles_y: u32,

    pub(crate) color_snapshot_prev: wgpu::Texture,
    pub(crate) color_snapshot_prev_view: wgpu::TextureView,
    pub(crate) pass_overdraw_buf: wgpu::Buffer,
    pub(crate) shader_cost_buf: wgpu::Buffer,
    pub(crate) material_timing_buf: wgpu::Buffer,

    pub(crate) color_compare_pipeline: wgpu::ComputePipeline,
    pub(crate) color_compare_bgl: wgpu::BindGroupLayout,
    pub(crate) color_compare_params_buf: wgpu::Buffer,

    pub(crate) blit_pipeline: wgpu::ComputePipeline,
    pub(crate) blit_bgl: wgpu::BindGroupLayout,

    pub(crate) cost_compute_pipeline: wgpu::ComputePipeline,
    pub(crate) cost_compute_bgl: wgpu::BindGroupLayout,
    pub(crate) cost_compute_params_buf: wgpu::Buffer,

    pub(crate) material_profiler: Option<MaterialProfiler>,

    pub(crate) mode: Mutex<PerfOverlayMode>,
    pub(crate) runtime: Mutex<PerfOverlayRuntime>,
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
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
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

        let shader_cost_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PerfOverlay Shader Cost"),
            size: pixel_count * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let material_timing_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PerfOverlay Material Timing Data"),
            size: 128 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cost_compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PerfOverlay Cost Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/compute_shader_cost.wgsl").into(),
            ),
        });

        let cost_compute_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("PerfOverlay Cost Compute BGL"),
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let cost_compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("PerfOverlay Cost Compute PL"),
                bind_group_layouts: &[Some(&cost_compute_bgl)],
                immediate_size: 0,
            });

        let cost_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("PerfOverlay Cost Compute Pipeline"),
                layout: Some(&cost_compute_pipeline_layout),
                module: &cost_compute_shader,
                entry_point: Some("cs_compute_shader_cost"),
                compilation_options: Default::default(),
                cache: None,
            });

        let cost_compute_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PerfOverlay Cost Compute Params"),
            size: std::mem::size_of::<ComputeCostParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Arc::new(Mutex::new(Self {
            internal_width: width,
            internal_height: height,
            display_width: width,
            display_height: height,
            num_tiles_x,
            num_tiles_y,
            color_snapshot_prev,
            color_snapshot_prev_view,
            pass_overdraw_buf,
            shader_cost_buf,
            material_timing_buf,
            color_compare_pipeline,
            color_compare_bgl,
            color_compare_params_buf,
            blit_pipeline,
            blit_bgl,
            cost_compute_pipeline,
            cost_compute_bgl,
            cost_compute_params_buf,
            material_profiler: None,
            mode: Mutex::new(PerfOverlayMode::Disabled),
            runtime: Mutex::new(PerfOverlayRuntime {
                frame_num: 0,
                snapshot_valid: false,
            }),
        }))
    }

    pub fn on_resize(&mut self, _device: &wgpu::Device, width: u32, height: u32) {
        if width == self.display_width && height == self.display_height {
            return;
        }
        self.display_width = width;
        self.display_height = height;
    }

    pub fn get_mode(&self) -> PerfOverlayMode {
        *self.mode.lock().unwrap()
    }

    pub fn set_mode(&self, mode: PerfOverlayMode) {
        *self.mode.lock().unwrap() = mode;
    }

    pub fn set_opacity(&self, _opacity: f32) {
    }

    pub fn init_material_profiler(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.material_profiler.is_none() {
            self.material_profiler = Some(MaterialProfiler::new(device, queue));
        }
    }

    pub fn is_profiling_complete(&self) -> bool {
        self.material_profiler
            .as_ref()
            .map(|p| p.is_complete())
            .unwrap_or(false)
    }

    pub fn get_material_timings(&self) -> &[MaterialTimingEntry] {
        self.material_profiler
            .as_ref()
            .map(|p| p.get_timing_table())
            .unwrap_or(&[])
    }
}

// ── PerfOverlayPass ─────────────────────────────────────────────────────────────

pub struct PerfOverlayPass {
    pub(crate) shared: Arc<Mutex<PerfOverlayShared>>,
    pub(crate) aggregate_pipeline: wgpu::ComputePipeline,
    pub(crate) aggregate_bgl: wgpu::BindGroupLayout,
    pub(crate) aggregate_params_buf: wgpu::Buffer,
    pub(crate) aggregate_bind_group: Option<wgpu::BindGroup>,
    pub(crate) tile_metrics_buf: wgpu::Buffer,
    pub(crate) visualize_pipeline: wgpu::RenderPipeline,
    pub(crate) visualize_bgl: wgpu::BindGroupLayout,
    pub(crate) visualize_params_buf: wgpu::Buffer,
    pub(crate) visualize_bind_group: Option<wgpu::BindGroup>,
    pub(crate) aggregate_bind_group_key: Option<(usize, usize)>,
    pub(crate) visualize_bind_group_key: Option<usize>,
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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

        let aggregate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
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
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
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
            aggregate_bind_group_key: None,
            visualize_bind_group_key: None,
        }
    }

    pub fn set_mode(&mut self, mode: PerfOverlayMode) {
        self.shared.lock().unwrap().set_mode(mode);
    }

    pub fn set_opacity(&mut self, _opacity: f32) {
    }
}

impl RenderPass for PerfOverlayPass {
    fn name(&self) -> &'static str {
        "PerfOverlay"
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let shared = self.shared.lock().unwrap();
        let mode = *shared.mode.lock().unwrap();
        if mode == PerfOverlayMode::Disabled {
            return Ok(());
        }

        let num_tiles_x = shared.num_tiles_x;
        let num_tiles_y = shared.num_tiles_y;
        let internal_width = shared.internal_width;
        let internal_height = shared.internal_height;
        let display_width = shared.display_width;
        let display_height = shared.display_height;

        let aggregate_params = AggregateParams {
            num_tiles_x,
            num_tiles_y,
            num_tiles: num_tiles_x * num_tiles_y,
            screen_width: internal_width,
            screen_height: internal_height,
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
            mode: mode as u32,
            num_tiles_x,
            num_tiles_y,
            internal_width,
            internal_height,
            display_width,
            display_height,
            heatmap_scale: 5.0,
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

    fn render_pass_descriptor<'a>(
        &'a self,
        target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        let color_attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>] = Box::leak(
            Box::new([Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })]),
        );
        Some(wgpu::RenderPassDescriptor {
            label: Some("PerfOverlay Visualize"),
            color_attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let shared = self.shared.lock().unwrap();
        let mode = *shared.mode.lock().unwrap();
        if mode == PerfOverlayMode::Disabled {
            return Ok(());
        }

        if let (Some(gbuffer), Some(tile_light_counts)) = (
            ctx.resources.gbuffer.get(),
            ctx.resources.tile_light_counts.get(),
        ) {
            let gbuffer_orm_ptr = gbuffer.orm as *const _ as usize;
            let tile_light_counts_ptr = tile_light_counts as *const _ as usize;
            let key = (gbuffer_orm_ptr, tile_light_counts_ptr);

            if self.aggregate_bind_group_key != Some(key) {
                self.aggregate_bind_group = Some(
                    ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                    }),
                );
                self.aggregate_bind_group_key = Some(key);
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

        if let (Some(_pre_aa), Some(gbuffer)) =
            (ctx.resources.pre_aa.get(), ctx.resources.gbuffer.get())
        {
            let gbuffer_orm_ptr = gbuffer.orm as *const _ as usize;
            let key = gbuffer_orm_ptr;

            if self.visualize_bind_group_key != Some(key)
                || self.visualize_bind_group.is_none()
            {
                self.visualize_bind_group = Some(
                    ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                                resource: wgpu::BindingResource::TextureView(gbuffer.orm),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: shared.shader_cost_buf.as_entire_binding(),
                            },
                        ],
                    }),
                );
                self.visualize_bind_group_key = Some(key);
            }

            let rp = unsafe { &mut *ctx.active_render_pass_ptr().unwrap() };
            rp.set_pipeline(&self.visualize_pipeline);
            rp.set_bind_group(0, self.visualize_bind_group.as_ref().unwrap(), &[]);
            rp.draw(0..3, 0..1);
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
        self.aggregate_bind_group_key = None;
        self.visualize_bind_group_key = None;
    }
}

// ── Material Profiler ───────────────────────────────────────────────────────────

const SAMPLES_PER_CONFIG: usize = 3;

pub struct MaterialProfiler {
    pub(crate) profile_pipeline: wgpu::RenderPipeline,
    pub(crate) profile_bgl: wgpu::BindGroupLayout,
    pub(crate) profile_params_bufs: Vec<wgpu::Buffer>,
    pub(crate) test_texture: wgpu::Texture,
    pub(crate) test_texture_view: wgpu::TextureView,
    pub(crate) query_set: wgpu::QuerySet,
    pub(crate) query_buffer: wgpu::Buffer,
    pub(crate) resolve_buffer: wgpu::Buffer,
    pub(crate) timing_samples: Vec<Vec<u64>>,
    pub(crate) timing_table: Vec<MaterialTimingEntry>,
    pub(crate) profiling_complete: bool,
    pub(crate) timings_uploaded: bool,
    pub(crate) current_config_index: usize,
    pub(crate) current_sample_index: usize,
    pub(crate) timestamp_period: f32,
}

impl MaterialProfiler {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let test_size = 32;
        let test_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Material Profile Test Texture"),
            size: wgpu::Extent3d {
                width: test_size,
                height: test_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let test_texture_view = test_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Material Profile QuerySet"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });

        let query_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Material Profile Query Buffer"),
            size: 2 * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Material Profile Resolve Buffer"),
            size: 2 * 8,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let profile_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Material Profile Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/profile_material.wgsl").into(),
            ),
        });

        let profile_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Material Profile BGL"),
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
            ],
        });

        let profile_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Material Profile PL"),
                bind_group_layouts: &[Some(&profile_bgl)],
                immediate_size: 0,
            });

        let profile_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Material Profile Pipeline"),
            layout: Some(&profile_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &profile_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &profile_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let configs = Self::get_material_configs();
        let mut profile_params_bufs = Vec::with_capacity(configs.len());
        for (roughness, metallic, num_lights) in &configs {
            let params = MaterialProfileParams {
                roughness: *roughness,
                metallic: *metallic,
                num_lights: *num_lights,
                num_shadow_lights: 0,
            };
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Material Profile Params"),
                size: std::mem::size_of::<MaterialProfileParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            {
                let mut view = buf.slice(..).get_mapped_range_mut();
                view.copy_from_slice(bytemuck::bytes_of(&params));
            }
            buf.unmap();
            profile_params_bufs.push(buf);
        }

        let timestamp_period = queue.get_timestamp_period();
        let timing_samples = vec![Vec::with_capacity(SAMPLES_PER_CONFIG); configs.len()];

        Self {
            profile_pipeline,
            profile_bgl,
            profile_params_bufs,
            test_texture,
            test_texture_view,
            query_set,
            query_buffer,
            resolve_buffer,
            timing_samples,
            timing_table: Vec::new(),
            profiling_complete: false,
            timings_uploaded: false,
            current_config_index: 0,
            current_sample_index: 0,
            timestamp_period,
        }
    }

    pub fn is_complete(&self) -> bool {
        self.profiling_complete
    }

    fn get_material_configs() -> Vec<(f32, f32, u32)> {
        let mut configs = Vec::new();
        for &roughness in &[0.1, 0.4, 0.7, 1.0] {
            for &metallic in &[0.0, 0.5, 1.0] {
                for &num_lights in &[4, 16, 32] {
                    configs.push((roughness, metallic, num_lights));
                }
            }
        }
        configs
    }

    pub fn profile_next(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        light_buf: &wgpu::Buffer,
    ) -> bool {
        if self.profiling_complete {
            return true;
        }

        let configs = Self::get_material_configs();
        if self.current_config_index >= configs.len() {
            self.profiling_complete = true;
            return true;
        }

        let params_buf = &self.profile_params_bufs[self.current_config_index];

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Material Profile BG"),
            layout: &self.profile_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: light_buf.as_entire_binding(),
                },
            ],
        });

        encoder.write_timestamp(&self.query_set, 0);

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Material Profile Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.test_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            pass.set_pipeline(&self.profile_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        encoder.write_timestamp(&self.query_set, 1);

        encoder.resolve_query_set(&self.query_set, 0..2, &self.query_buffer, 0);

        encoder.copy_buffer_to_buffer(&self.query_buffer, 0, &self.resolve_buffer, 0, 2 * 8);

        self.current_sample_index += 1;
        if self.current_sample_index >= SAMPLES_PER_CONFIG {
            self.current_sample_index = 0;
            self.current_config_index += 1;
        }

        false
    }

    pub fn read_current_sample_blocking(&mut self, device: &wgpu::Device, owns_device: bool) {
        if !owns_device {
            return;
        }
        let buffer_slice = self.resolve_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely());

        let data = buffer_slice.get_mapped_range();
        let timestamps: &[u64] = bytemuck::cast_slice(&data);

        let start_ts = timestamps[0];
        let end_ts = timestamps[1];
        let gpu_ticks = end_ts.saturating_sub(start_ts);

        let config_idx = if self.current_sample_index == 0 {
            self.current_config_index.saturating_sub(1)
        } else {
            self.current_config_index
        };

        if config_idx < self.timing_samples.len() {
            self.timing_samples[config_idx].push(gpu_ticks);
        }

        drop(data);
        self.resolve_buffer.unmap();
    }

    pub fn compute_final_timings(&mut self) {
        let configs = Self::get_material_configs();
        self.timing_table.clear();

        for (i, (roughness, metallic, num_lights)) in configs.iter().enumerate() {
            let samples = &self.timing_samples[i];
            if samples.is_empty() {
                continue;
            }

            let avg_ticks = samples.iter().sum::<u64>() / samples.len() as u64;
            let gpu_time_ns = (avg_ticks as f32 * self.timestamp_period) as u64;

            self.timing_table.push(MaterialTimingEntry {
                roughness: *roughness,
                metallic: *metallic,
                num_lights: *num_lights,
                gpu_time_ns,
            });
        }
    }

    pub fn get_timing_table(&self) -> &[MaterialTimingEntry] {
        &self.timing_table
    }
}
