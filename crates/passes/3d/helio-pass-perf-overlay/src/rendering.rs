use crate::{
    AggregateParams, ColorCompareParams, PerfOverlayMode, PerfOverlayRuntime, TileMetrics,
    VisualizeParams, TILE_SIZE,
};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use std::sync::{Arc, Mutex};

const VISUALIZE_SHADER: &str = include_str!("../shaders/visualize.wgsl");

/// The overlay's composed shaders as compiled, for shader validation without
/// a device: each file's path and its source with its snippets resolved.
pub fn validation_sources() -> Vec<(String, String)> {
    vec![(
        concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/visualize.wgsl").to_string(),
        helio_core::shader::resolve_with(
            VISUALIZE_SHADER,
            &[helio_frame_api::diagnostics::HEATMAP_SNIPPET],
        )
        .into_owned(),
    )]
}

// ── Shared GPU state (created once, shared between passes) ──────────────────────

pub struct PerfOverlayShared {
    pub(crate) internal_width: u32,
    pub(crate) internal_height: u32,
    pub(crate) display_width: u32,
    pub(crate) display_height: u32,
    pub(crate) num_tiles_x: u32,
    pub(crate) num_tiles_y: u32,

    /// Whether the per-pixel targets below are full size. They are 1-texel
    /// placeholders until a mode asks for them (the passes are graph members
    /// only then), and go back to placeholders when the overlay is off.
    pub(crate) targets_allocated: bool,
    /// Advances whenever the per-pixel targets are replaced, so passes drop
    /// bind groups built over the old ones.
    pub(crate) targets_generation: u64,
    pub(crate) color_snapshot_prev: wgpu::Texture,
    pub(crate) color_snapshot_prev_view: wgpu::TextureView,
    pub(crate) pass_overdraw_buf: wgpu::Buffer,

    pub(crate) color_compare_pipeline: wgpu::ComputePipeline,
    pub(crate) color_compare_bgl: wgpu::BindGroupLayout,
    pub(crate) color_compare_params_buf: wgpu::Buffer,

    pub(crate) blit_pipeline: wgpu::ComputePipeline,
    pub(crate) blit_bgl: wgpu::BindGroupLayout,

    /// What a kind offered the overlay to show (`helio_frame_api::diagnostics`).
    pub(crate) tile_counter: Option<helio_frame_api::diagnostics::TileCounter>,

    pub(crate) mode: Mutex<PerfOverlayMode>,
    pub(crate) runtime: Mutex<PerfOverlayRuntime>,
}

/// The overlay's per-pixel targets: the previous colour snapshot and the
/// overdraw counters, `width` x `height`.
struct PixelTargets {
    color_snapshot_prev: wgpu::Texture,
    color_snapshot_prev_view: wgpu::TextureView,
    pass_overdraw_buf: wgpu::Buffer,
}

impl PixelTargets {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let (width, height) = (width.max(1), height.max(1));
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
        Self {
            color_snapshot_prev,
            color_snapshot_prev_view,
            pass_overdraw_buf: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("PerfOverlay Pass Overdraw Counters"),
                size: pixel_count * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        }
    }
}

impl PerfOverlayShared {
    /// `tile_counter`: what the graph's kinds offered
    /// (`helio_frame_api::diagnostics`). A counter whose tiles are not the
    /// overlay's [`TILE_SIZE`] is not read.
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        tile_counter: Option<helio_frame_api::diagnostics::TileCounter>,
    ) -> Arc<Mutex<Self>> {
        let tile_counter = tile_counter.filter(|counter| counter.tile_px == TILE_SIZE);
        let num_tiles_x = width.div_ceil(TILE_SIZE);
        let num_tiles_y = height.div_ceil(TILE_SIZE);

        let PixelTargets {
            color_snapshot_prev,
            color_snapshot_prev_view,
            pass_overdraw_buf,
        } = PixelTargets::new(device, 1, 1);

        let color_compare_shader = helio_core::shader::module(
            device,
            "PerfOverlay Color Compare Shader",
            helio_core::include_wgsl!("../shaders/analyze_color_overdraw.wgsl"),
        );

        let color_compare_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let color_compare_pipeline = helio_core::shader::create_compute_pipeline(
            &device,
            wgpu::ComputePipelineDescriptor {
                label: Some("PerfOverlay Color Compare Pipeline"),
                layout: Some(&color_compare_pipeline_layout),
                module: &color_compare_shader,
                entry_point: Some("analyze_color_overdraw"),
                compilation_options: Default::default(),
                cache: None,
            },
        );

        let color_compare_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PerfOverlay Color Compare Params"),
            size: std::mem::size_of::<ColorCompareParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let blit_shader = helio_core::shader::module(
            device,
            "PerfOverlay Blit Shader",
            helio_core::include_wgsl!("../shaders/blit_color.wgsl"),
        );

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

        let blit_pipeline = helio_core::shader::create_compute_pipeline(
            &device,
            wgpu::ComputePipelineDescriptor {
                label: Some("PerfOverlay Blit Pipeline"),
                layout: Some(&blit_pipeline_layout),
                module: &blit_shader,
                entry_point: Some("blit_color"),
                compilation_options: Default::default(),
                cache: None,
            },
        );

        Arc::new(Mutex::new(Self {
            internal_width: width,
            internal_height: height,
            display_width: width,
            display_height: height,
            num_tiles_x,
            num_tiles_y,
            targets_allocated: false,
            targets_generation: 0,
            color_snapshot_prev,
            color_snapshot_prev_view,
            pass_overdraw_buf,
            color_compare_pipeline,
            color_compare_bgl,
            color_compare_params_buf,
            blit_pipeline,
            blit_bgl,
            tile_counter,
            mode: Mutex::new(PerfOverlayMode::Disabled),
            runtime: Mutex::new(PerfOverlayRuntime {
                frame_num: 0,
                snapshot_valid: false,
            }),
        }))
    }

    pub fn on_resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == self.internal_width && height == self.internal_height {
            return;
        }
        self.internal_width = width;
        self.internal_height = height;
        self.display_width = width;
        self.display_height = height;
        self.num_tiles_x = width.div_ceil(TILE_SIZE);
        self.num_tiles_y = height.div_ceil(TILE_SIZE);
        if self.targets_allocated {
            self.set_targets(device, true);
        }
    }

    /// Full-size per-pixel targets (`true`) or placeholders.
    fn set_targets(&mut self, device: &wgpu::Device, allocated: bool) {
        let targets = if allocated {
            PixelTargets::new(device, self.internal_width, self.internal_height)
        } else {
            PixelTargets::new(device, 1, 1)
        };
        self.color_snapshot_prev = targets.color_snapshot_prev;
        self.color_snapshot_prev_view = targets.color_snapshot_prev_view;
        self.pass_overdraw_buf = targets.pass_overdraw_buf;
        self.targets_allocated = allocated;
        self.targets_generation += 1;
        self.runtime.lock().unwrap().snapshot_valid = false;
    }

    /// Gives the per-pixel targets their full size; called by a pass that
    /// has joined the graph, before it uses them.
    pub(crate) fn ensure_targets(&mut self, device: &wgpu::Device) {
        if !self.targets_allocated {
            self.set_targets(device, true);
        }
    }

    /// Back to placeholders once no mode is on (a pass leaving the graph).
    pub(crate) fn release_targets(&mut self, device: &wgpu::Device) {
        if self.targets_allocated && !self.shows() {
            self.set_targets(device, false);
        }
    }

    /// Whether the overlay draws the current mode. `ShaderComplexity` is the
    /// view of the kind that owns shading cost: with that kind absent there
    /// is nothing to show, and the overlay does not stand in for it.
    pub fn shows(&self) -> bool {
        !matches!(
            self.get_mode(),
            PerfOverlayMode::Disabled | PerfOverlayMode::ShaderComplexity
        )
    }

    pub fn get_mode(&self) -> PerfOverlayMode {
        *self.mode.lock().unwrap()
    }

    pub fn set_mode(&self, mode: PerfOverlayMode) {
        *self.mode.lock().unwrap() = mode;
    }

    pub fn set_opacity(&self, _opacity: f32) {}
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
    /// (tile counts, shared targets generation).
    pub(crate) aggregate_bind_group_key: Option<(usize, u64)>,
    /// Shared targets generation.
    pub(crate) visualize_bind_group_key: Option<u64>,
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

        let aggregate_shader = helio_core::shader::module(
            device,
            "PerfOverlay Aggregate Shader",
            helio_core::include_wgsl!("../shaders/aggregate_tiles.wgsl"),
        );

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
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
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

        let aggregate_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("PerfOverlay Aggregate PL"),
                bind_group_layouts: &[Some(&aggregate_bgl)],
                immediate_size: 0,
            });

        let aggregate_pipeline = helio_core::shader::create_compute_pipeline(
            &device,
            wgpu::ComputePipelineDescriptor {
                label: Some("PerfOverlay Aggregate Pipeline"),
                layout: Some(&aggregate_pipeline_layout),
                module: &aggregate_shader,
                entry_point: Some("cs_aggregate_tiles"),
                compilation_options: Default::default(),
                cache: None,
            },
        );

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

        let visualize_shader = helio_core::shader::module_with(
            device,
            "PerfOverlay Visualize Shader",
            helio_core::include_wgsl!("../shaders/visualize.wgsl"),
            &[helio_frame_api::diagnostics::HEATMAP_SNIPPET],
        );

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
            ],
        });

        let visualize_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("PerfOverlay Visualize PL"),
                bind_group_layouts: &[Some(&visualize_bgl)],
                immediate_size: 0,
            });

        let visualize_pipeline = helio_core::shader::create_render_pipeline(
            &device,
            wgpu::RenderPipelineDescriptor {
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
            },
        );

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

    pub fn set_opacity(&mut self, _opacity: f32) {}
}

impl RenderPass for PerfOverlayPass {
    fn name(&self) -> &'static str {
        "PerfOverlay"
    }

    /// A member while an overlay mode it draws is on.
    fn is_member(&mut self, _scene: &mut helio_core::SceneContent<'_>) -> bool {
        self.shared.lock().unwrap().shows()
    }

    fn on_leave(&mut self, device: &wgpu::Device) {
        self.shared.lock().unwrap().release_targets(device);
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        self.shared.lock().unwrap().ensure_targets(ctx.device);
        let shared = self.shared.lock().unwrap();
        let mode = *shared.mode.lock().unwrap();
        if !shared.shows() {
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

    fn render_pass_descriptor_with_storage<'a>(
        &'a self,
        target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a helio_core::ResourceRegistry<'a>,
        storage: &'a mut helio_core::RenderFrameStorage,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        let color_attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>] = storage
            .retain_boxed_slice(Box::new([Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })]));
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
        if !shared.shows() {
            return Ok(());
        }

        if let Some(tile_light_counts) = shared.tile_counter.as_ref().and_then(|counter| {
            ctx.registry
                .get::<&wgpu::Buffer>(helio_core::ResourceKey::new(counter.resource))
        }) {
            let tile_light_counts_ptr = tile_light_counts as *const _ as usize;
            let key = (tile_light_counts_ptr, shared.targets_generation);

            if self.aggregate_bind_group_key != Some(key) {
                self.aggregate_bind_group =
                    Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                                resource: tile_light_counts.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: self.tile_metrics_buf.as_entire_binding(),
                            },
                        ],
                    }));
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

        if ctx
            .registry
            .read_texture_view(
                helio_core::ResourceKey::new(helio_frame_api::color::SCENE_COLOR),
                "PerfOverlay",
            )
            .is_some()
        {
            let key = shared.targets_generation;

            if self.visualize_bind_group_key != Some(key) || self.visualize_bind_group.is_none() {
                self.visualize_bind_group =
                    Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                        ],
                    }));
                self.visualize_bind_group_key = Some(key);
            }

            let mut rp = ctx.render_cmds().unwrap();
            rp.set_pipeline(&self.visualize_pipeline);
            rp.set_bind_group(0, self.visualize_bind_group.as_ref().unwrap(), &[]);
            rp.draw(0..3, 0..1);
        }

        Ok(())
    }

    fn on_resize(&mut self, device: &wgpu::Device, size: helio_core::RenderSize) {
        let [width, height] = size.internal;
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
