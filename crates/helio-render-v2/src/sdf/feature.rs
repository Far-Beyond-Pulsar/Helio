//! SdfFeature — implements the Feature trait for SDF rendering
//!
//! Creates GPU resources (3D volume texture, edit buffer, compute + render pipelines)
//! in `register()` and uploads per-frame data in `prepare()`.
//!
//! Supports multiple rendering modes via `SdfMode`:
//! - **Dense**: Evaluates all edits at every voxel of a 128^3 grid (Phase 1+2)
//! - **Sparse**: Only evaluates active bricks near the SDF surface (Phase 3)
//! - **ClipMap**: Multiple nested LOD levels centered on camera (Phase 4)

use crate::features::{Feature, FeatureContext, PrepareContext, ShaderDefine};
use crate::Result;
use super::edit_list::{SdfEditList, SdfEdit, GpuSdfEdit, MAX_EDITS};
use super::uniforms::SdfGridParams;
use super::clip_map::{SdfClipMap, DEFAULT_CLIP_LEVELS};
use super::passes::evaluate_dense::SdfEvaluateDensePass;
use super::passes::evaluate_sparse::SdfEvaluateSparsePass;
use super::passes::clip_update::SdfClipUpdatePass;
use super::passes::ray_march::SdfRayMarchPass;
use super::brick::{BrickMap, DEFAULT_BRICK_SIZE};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU32, Ordering};

/// Default grid resolution (128^3)
const DEFAULT_GRID_DIM: u32 = 128;

/// Default atlas bricks per axis (16 → 16^3 = 4096 max bricks)
const DEFAULT_ATLAS_BRICKS_PER_AXIS: u32 = 16;

/// SDF rendering mode
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SdfMode {
    /// Dense 3D grid — evaluates every voxel (Phase 1+2)
    Dense,
    /// Sparse brick map — only evaluates active bricks (Phase 3)
    Sparse,
    /// Geometry clip maps — multi-resolution LOD centered on camera (Phase 4)
    ClipMap,
}

/// SDF rendering feature
///
/// # Usage
/// ```ignore
/// let sdf = SdfFeature::new()
///     .with_mode(SdfMode::Sparse)
///     .with_grid_dim(128)
///     .with_volume_bounds([-10.0, -10.0, -10.0], [10.0, 10.0, 10.0]);
/// ```
pub struct SdfFeature {
    enabled: bool,
    mode: SdfMode,
    debug_mode: bool,
    grid_dim: u32,
    volume_min: [f32; 3],
    volume_max: [f32; 3],

    // CPU-side state
    edit_list: SdfEditList,
    last_uploaded_gen: u64,

    // GPU resources shared between modes
    edit_buffer: Option<Arc<wgpu::Buffer>>,
    params_buffer: Option<Arc<wgpu::Buffer>>,

    // Dense mode resources
    volume_texture: Option<Arc<wgpu::Texture>>,
    volume_view: Option<Arc<wgpu::TextureView>>,
    sampler: Option<Arc<wgpu::Sampler>>,
    eval_pipeline: Option<Arc<wgpu::ComputePipeline>>,
    eval_bind_group: Option<Arc<wgpu::BindGroup>>,
    eval_bg_layout: Option<wgpu::BindGroupLayout>,
    march_pipeline: Option<Arc<wgpu::RenderPipeline>>,
    march_bind_group: Option<Arc<wgpu::BindGroup>>,

    // Sparse mode resources
    brick_map: Option<BrickMap>,
    sparse_active_count: Option<Arc<AtomicU32>>,
    sparse_eval_pipeline: Option<Arc<wgpu::ComputePipeline>>,
    sparse_eval_bind_group: Option<Arc<wgpu::BindGroup>>,
    sparse_eval_bg_layout: Option<wgpu::BindGroupLayout>,
    sparse_march_pipeline: Option<Arc<wgpu::RenderPipeline>>,
    sparse_march_bind_group: Option<Arc<wgpu::BindGroup>>,

    // ClipMap mode resources
    clip_map: Option<SdfClipMap>,
    clip_active_counts: Option<Arc<Mutex<Vec<u32>>>>,
    clip_march_pipeline: Option<Arc<wgpu::RenderPipeline>>,
    clip_march_bind_group: Option<Arc<wgpu::BindGroup>>,
    all_brick_indices_buffer: Option<Arc<wgpu::Buffer>>,
}

impl SdfFeature {
    pub fn new() -> Self {
        Self {
            enabled: true,
            mode: SdfMode::Dense,
            debug_mode: false,
            grid_dim: DEFAULT_GRID_DIM,
            volume_min: [-10.0, -10.0, -10.0],
            volume_max: [10.0, 10.0, 10.0],
            edit_list: SdfEditList::new(),
            last_uploaded_gen: u64::MAX,
            edit_buffer: None,
            params_buffer: None,
            volume_texture: None,
            volume_view: None,
            sampler: None,
            eval_pipeline: None,
            eval_bind_group: None,
            eval_bg_layout: None,
            march_pipeline: None,
            march_bind_group: None,
            brick_map: None,
            sparse_active_count: None,
            sparse_eval_pipeline: None,
            sparse_eval_bind_group: None,
            sparse_eval_bg_layout: None,
            sparse_march_pipeline: None,
            sparse_march_bind_group: None,
            clip_map: None,
            clip_active_counts: None,
            clip_march_pipeline: None,
            clip_march_bind_group: None,
            all_brick_indices_buffer: None,
        }
    }

    pub fn with_mode(mut self, mode: SdfMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_grid_dim(mut self, dim: u32) -> Self {
        self.grid_dim = dim;
        self
    }

    pub fn with_volume_bounds(mut self, min: [f32; 3], max: [f32; 3]) -> Self {
        self.volume_min = min;
        self.volume_max = max;
        self
    }

    /// Toggle debug visualization mode.
    pub fn toggle_debug(&mut self) {
        self.debug_mode = !self.debug_mode;
        // Force re-upload so debug_flags makes it to the GPU
        self.last_uploaded_gen = u64::MAX;
        log::info!("SDF debug mode: {}", if self.debug_mode { "ON" } else { "OFF" });
    }

    /// Add an SDF edit to the scene.
    pub fn add_edit(&mut self, edit: SdfEdit) {
        self.edit_list.add(edit);
    }

    /// Remove an SDF edit by index.
    pub fn remove_edit(&mut self, index: usize) {
        self.edit_list.remove(index);
    }

    /// Replace an SDF edit at the given index.
    pub fn set_edit(&mut self, index: usize, edit: SdfEdit) {
        self.edit_list.set(index, edit);
    }

    /// Clear all SDF edits.
    pub fn clear_edits(&mut self) {
        self.edit_list.clear();
    }

    /// Access the edit list.
    pub fn edit_list(&self) -> &SdfEditList {
        &self.edit_list
    }

    /// Mutable access to the edit list.
    pub fn edit_list_mut(&mut self) -> &mut SdfEditList {
        &mut self.edit_list
    }

    // ── Dense mode registration ──────────────────────────────────────────────

    fn register_dense(&mut self, ctx: &mut FeatureContext) -> Result<()> {
        let device = ctx.device;
        let dim = self.grid_dim;

        // 3D Volume Texture
        let volume_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SDF Volume"),
            size: wgpu::Extent3d {
                width: dim,
                height: dim,
                depth_or_array_layers: dim,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }));
        let volume_view = Arc::new(volume_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        }));

        // Sampler (trilinear, clamp to edge)
        let sampler = Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SDF Volume Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        }));

        // Compute Pipeline (SDF evaluation)
        let eval_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Evaluate Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/sdf/sdf_evaluate_dense.wgsl").into(),
            ),
        });
        let eval_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("SDF Evaluate Pipeline"),
                layout: None,
                module: &eval_shader,
                entry_point: Some("cs_evaluate"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));
        let eval_bg_layout = eval_pipeline.get_bind_group_layout(0);

        let params_buffer = self.params_buffer.as_ref().unwrap();
        let edit_buffer = self.edit_buffer.as_ref().unwrap();

        let eval_bind_group = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SDF Evaluate Bind Group"),
            layout: &eval_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: edit_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&volume_view),
                },
            ],
        }));

        // Render Pipeline (SDF ray march)
        let march_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Ray March Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/sdf/sdf_ray_march.wgsl").into(),
            ),
        });

        let march_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SDF Ray March BG Layout"),
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
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let global_layout = &ctx.resources.bind_group_layouts.global;
        let march_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SDF Ray March Pipeline Layout"),
            bind_group_layouts: &[
                Some(global_layout.as_ref()),
                Some(&march_bg_layout),
            ],
            immediate_size: 0,
        });

        let march_pipeline = Arc::new(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("SDF Ray March Pipeline"),
                layout: Some(&march_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &march_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &march_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: ctx.surface_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        ));

        let march_bind_group = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SDF Ray March Bind Group"),
            layout: &march_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&volume_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        }));

        // Register passes
        ctx.graph.add_pass(SdfEvaluateDensePass::new(
            eval_pipeline.clone(),
            eval_bind_group.clone(),
            dim,
        ));
        ctx.graph.add_pass(SdfRayMarchPass::new(
            march_pipeline.clone(),
            march_bind_group.clone(),
        ));

        // Store resources
        self.volume_texture = Some(volume_texture);
        self.volume_view = Some(volume_view);
        self.sampler = Some(sampler);
        self.eval_pipeline = Some(eval_pipeline);
        self.eval_bind_group = Some(eval_bind_group);
        self.eval_bg_layout = Some(eval_bg_layout);
        self.march_pipeline = Some(march_pipeline);
        self.march_bind_group = Some(march_bind_group);

        Ok(())
    }

    // ── Sparse mode registration ─────────────────────────────────────────────

    fn register_sparse(&mut self, ctx: &mut FeatureContext) -> Result<()> {
        let device = ctx.device;
        let dim = self.grid_dim;
        let params_buffer = self.params_buffer.as_ref().unwrap();
        let edit_buffer = self.edit_buffer.as_ref().unwrap();

        // Create BrickMap and its GPU resources
        let mut brick_map = BrickMap::new(dim, DEFAULT_BRICK_SIZE, DEFAULT_ATLAS_BRICKS_PER_AXIS);
        brick_map.create_gpu_resources(device);

        let atlas_view = brick_map.atlas_view.as_ref().unwrap().clone();
        let brick_index_buffer = brick_map.brick_index_buffer.as_ref().unwrap().clone();
        let active_bricks_buffer = brick_map.active_bricks_buffer.as_ref().unwrap().clone();

        // Shared active count between feature and pass
        let active_count = Arc::new(AtomicU32::new(0));

        // ── Compute Pipeline (sparse evaluation) ─────────────────────────────
        let eval_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Evaluate Sparse Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/sdf/sdf_evaluate_sparse.wgsl").into(),
            ),
        });
        let eval_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("SDF Evaluate Sparse Pipeline"),
                layout: None,
                module: &eval_shader,
                entry_point: Some("cs_evaluate_sparse"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));
        let eval_bg_layout = eval_pipeline.get_bind_group_layout(0);

        let eval_bind_group = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SDF Evaluate Sparse Bind Group"),
            layout: &eval_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: edit_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: active_bricks_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: brick_index_buffer.as_entire_binding(),
                },
            ],
        }));

        // ── Render Pipeline (sparse ray march) ──────────────────────────────
        let march_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Ray March Sparse Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/sdf/sdf_ray_march_sparse.wgsl").into(),
            ),
        });

        // Sparse ray march bind group layout (group 1):
        //   b0: uniform (params)
        //   b1: texture_3d (atlas, non-filterable — we use textureLoad)
        //   b2: storage buffer read-only (brick_index)
        let march_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SDF Ray March Sparse BG Layout"),
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
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
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

        let global_layout = &ctx.resources.bind_group_layouts.global;
        let march_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SDF Ray March Sparse Pipeline Layout"),
            bind_group_layouts: &[
                Some(global_layout.as_ref()),
                Some(&march_bg_layout),
            ],
            immediate_size: 0,
        });

        let march_pipeline = Arc::new(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("SDF Ray March Sparse Pipeline"),
                layout: Some(&march_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &march_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &march_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: ctx.surface_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        ));

        let march_bind_group = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SDF Ray March Sparse Bind Group"),
            layout: &march_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: brick_index_buffer.as_entire_binding(),
                },
            ],
        }));

        // Register passes
        ctx.graph.add_pass(SdfEvaluateSparsePass::new(
            eval_pipeline.clone(),
            eval_bind_group.clone(),
            active_count.clone(),
        ));
        ctx.graph.add_pass(SdfRayMarchPass::new(
            march_pipeline.clone(),
            march_bind_group.clone(),
        ));

        // Store resources
        self.brick_map = Some(brick_map);
        self.sparse_active_count = Some(active_count);
        self.sparse_eval_pipeline = Some(eval_pipeline);
        self.sparse_eval_bind_group = Some(eval_bind_group);
        self.sparse_eval_bg_layout = Some(eval_bg_layout);
        self.sparse_march_pipeline = Some(march_pipeline);
        self.sparse_march_bind_group = Some(march_bind_group);

        Ok(())
    }

    // ── ClipMap mode registration ────────────────────────────────────────────

    fn register_clipmap(&mut self, ctx: &mut FeatureContext) -> Result<()> {
        let device = ctx.device;
        let dim = self.grid_dim;
        let edit_buffer = self.edit_buffer.as_ref().unwrap();

        // Compute base voxel size from volume bounds and grid dim
        let range_x = self.volume_max[0] - self.volume_min[0];
        let base_voxel_size = range_x / dim as f32;

        // Create clip map with per-level BrickMaps
        let mut clip_map = SdfClipMap::new(dim, base_voxel_size, DEFAULT_CLIP_LEVELS);
        clip_map.create_gpu_resources(device);

        // Shared active counts between feature and clip update pass
        let clip_active_counts = Arc::new(Mutex::new(vec![0u32; DEFAULT_CLIP_LEVELS as usize]));

        // ── Compute Pipeline (reuse sparse evaluate shader) ──────────────────
        let eval_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Evaluate Sparse Shader (ClipMap)"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/sdf/sdf_evaluate_sparse.wgsl").into(),
            ),
        });
        let eval_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("SDF Evaluate Sparse Pipeline (ClipMap)"),
                layout: None,
                module: &eval_shader,
                entry_point: Some("cs_evaluate_sparse"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));
        let eval_bg_layout = eval_pipeline.get_bind_group_layout(0);

        // Create per-level compute bind groups
        let mut level_bind_groups = Vec::with_capacity(DEFAULT_CLIP_LEVELS as usize);
        for level in clip_map.levels() {
            let atlas_view = level.brick_map.atlas_view.as_ref().unwrap().clone();
            let brick_index_buffer = level.brick_map.brick_index_buffer.as_ref().unwrap().clone();
            let active_bricks_buffer = level.brick_map.active_bricks_buffer.as_ref().unwrap().clone();
            let params_buffer = level.params_buffer.as_ref().unwrap().clone();

            let bg = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("SDF Clip Level {} Evaluate BG", level.level_index)),
                layout: &eval_bg_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: edit_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&atlas_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: active_bricks_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: brick_index_buffer.as_entire_binding(),
                    },
                ],
            }));
            level_bind_groups.push(bg);
        }

        // ── All brick indices buffer (concatenated for ray march shader) ─────
        let brick_grid_dim = dim / DEFAULT_BRICK_SIZE as u32;
        let entries_per_level = (brick_grid_dim * brick_grid_dim * brick_grid_dim) as usize;
        let total_entries = entries_per_level * DEFAULT_CLIP_LEVELS as usize;
        let all_brick_indices_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Clip All Brick Indices"),
            size: (total_entries * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // ── Render Pipeline (clip map ray march) ─────────────────────────────
        let march_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Ray March ClipMap Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/sdf/sdf_ray_march_clipmap.wgsl").into(),
            ),
        });

        // ClipMap ray march bind group layout (group 1):
        //   b0: uniform (SdfClipMapParams)
        //   b1: texture_3d (atlas_0)
        //   b2: texture_3d (atlas_1)
        //   b3: texture_3d (atlas_2)
        //   b4: texture_3d (atlas_3)
        //   b5: storage read (all_brick_indices)
        let march_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SDF Ray March ClipMap BG Layout"),
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
                // atlas_0
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // atlas_1
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // atlas_2
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // atlas_3
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // all_brick_indices
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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

        let global_layout = &ctx.resources.bind_group_layouts.global;
        let march_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SDF Ray March ClipMap Pipeline Layout"),
            bind_group_layouts: &[
                Some(global_layout.as_ref()),
                Some(&march_bg_layout),
            ],
            immediate_size: 0,
        });

        let march_pipeline = Arc::new(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("SDF Ray March ClipMap Pipeline"),
                layout: Some(&march_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &march_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &march_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: ctx.surface_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        ));

        // Create ray march bind group with per-level atlas views
        let clip_params_buffer = clip_map.clip_params_buffer.as_ref().unwrap().clone();
        let atlas_views: Vec<_> = clip_map.levels().iter()
            .map(|l| l.brick_map.atlas_view.as_ref().unwrap().clone())
            .collect();

        let march_bind_group = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SDF Ray March ClipMap Bind Group"),
            layout: &march_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: clip_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&atlas_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&atlas_views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&atlas_views[2]),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&atlas_views[3]),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: all_brick_indices_buffer.as_entire_binding(),
                },
            ],
        }));

        // Register passes
        ctx.graph.add_pass(SdfClipUpdatePass::new(
            eval_pipeline.clone(),
            level_bind_groups,
            clip_active_counts.clone(),
        ));
        ctx.graph.add_pass(SdfRayMarchPass::new(
            march_pipeline.clone(),
            march_bind_group.clone(),
        ));

        // Store resources
        self.clip_map = Some(clip_map);
        self.clip_active_counts = Some(clip_active_counts);
        self.clip_march_pipeline = Some(march_pipeline);
        self.clip_march_bind_group = Some(march_bind_group);
        self.all_brick_indices_buffer = Some(all_brick_indices_buffer);

        Ok(())
    }
}

impl Feature for SdfFeature {
    fn name(&self) -> &str { "sdf" }

    fn register(&mut self, ctx: &mut FeatureContext) -> Result<()> {
        let device = ctx.device;

        log::info!("SDF Feature: registering with {}^3 grid, mode={:?}", self.grid_dim, self.mode);

        // ── Shared resources (both modes need these) ─────────────────────────
        let edit_buffer_size = (MAX_EDITS * std::mem::size_of::<GpuSdfEdit>()) as u64;
        let edit_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Edit Buffer"),
            size: edit_buffer_size.max(64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let params_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Params Uniform"),
            size: std::mem::size_of::<SdfGridParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.edit_buffer = Some(edit_buffer);
        self.params_buffer = Some(params_buffer);

        // ── Mode-specific registration ───────────────────────────────────────
        match self.mode {
            SdfMode::Dense => self.register_dense(ctx)?,
            SdfMode::Sparse => self.register_sparse(ctx)?,
            SdfMode::ClipMap => self.register_clipmap(ctx)?,
        }

        log::info!("SDF Feature: registered successfully ({:?} mode)", self.mode);
        Ok(())
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> Result<()> {
        if !self.enabled { return Ok(()); }

        let gen = self.edit_list.generation();
        let needs_upload = gen != self.last_uploaded_gen;

        if needs_upload {
            // Upload edit list to GPU
            let gpu_edits = self.edit_list.flush_gpu_data();
            if !gpu_edits.is_empty() {
                ctx.queue.write_buffer(
                    self.edit_buffer.as_ref().unwrap(),
                    0,
                    bytemuck::cast_slice(&gpu_edits),
                );
            }

            match self.mode {
                SdfMode::Dense => {
                    let params = SdfGridParams::new(
                        self.volume_min,
                        self.volume_max,
                        self.grid_dim,
                        self.edit_list.len() as u32,
                    ).with_debug(self.debug_mode);
                    ctx.queue.write_buffer(
                        self.params_buffer.as_ref().unwrap(),
                        0,
                        bytemuck::bytes_of(&params),
                    );
                }
                SdfMode::Sparse => {
                    let brick_map = self.brick_map.as_mut().unwrap();

                    // CPU brick classification
                    brick_map.classify(
                        self.edit_list.edits(),
                        self.volume_min,
                        self.volume_max,
                    );

                    // Upload brick_index and active_bricks to GPU
                    brick_map.upload(ctx.queue);

                    // Update shared active count for the pass
                    let count = brick_map.active_count();
                    self.sparse_active_count.as_ref().unwrap().store(count, Ordering::Relaxed);

                    // Upload extended SdfGridParams with brick fields
                    let params = SdfGridParams::new_sparse(
                        self.volume_min,
                        self.volume_max,
                        self.grid_dim,
                        self.edit_list.len() as u32,
                        brick_map.brick_size(),
                        count,
                        brick_map.atlas_bricks_per_axis(),
                    ).with_debug(self.debug_mode);
                    ctx.queue.write_buffer(
                        self.params_buffer.as_ref().unwrap(),
                        0,
                        bytemuck::bytes_of(&params),
                    );

                    log::debug!(
                        "SDF Sparse: {} active bricks out of {} total",
                        count,
                        brick_map.brick_grid_dim().pow(3)
                    );
                }
                SdfMode::ClipMap => {
                    // ClipMap edit upload is handled below in the per-frame section
                }
            }

            self.last_uploaded_gen = gen;
        }

        // ClipMap needs per-frame center tracking (not just when edits change)
        if self.mode == SdfMode::ClipMap {
            let clip_map = self.clip_map.as_mut().unwrap();

            // Center clip map on volume midpoint (not camera) so edits stay in the
            // finest level regardless of camera position. Camera-following will be
            // re-enabled when terrain streaming (Phase 6) is implemented.
            let volume_center = glam::Vec3::new(
                (self.volume_min[0] + self.volume_max[0]) * 0.5,
                (self.volume_min[1] + self.volume_max[1]) * 0.5,
                (self.volume_min[2] + self.volume_max[2]) * 0.5,
            );
            let mut dirty_mask = clip_map.update_center(volume_center);

            // If edits changed this frame, force all levels dirty
            if needs_upload {
                dirty_mask = (1 << clip_map.level_count()) - 1;
            }

            if dirty_mask != 0 {
                // Classify active bricks on dirty levels
                clip_map.classify_dirty_levels(dirty_mask, self.edit_list.edits());

                // Upload per-level brick data + params (with correct edit count)
                let edit_count = self.edit_list.len() as u32;
                let counts = clip_map.upload_dirty_levels(dirty_mask, ctx.queue, edit_count);

                // Update shared active counts for the clip update pass
                {
                    let mut shared = self.clip_active_counts.as_ref().unwrap().lock().unwrap();
                    *shared = counts;
                }

                // Upload clip map params uniform
                let mut clip_params = clip_map.build_clip_params();
                clip_params.debug_flags = if self.debug_mode { 1 } else { 0 };
                ctx.queue.write_buffer(
                    clip_map.clip_params_buffer.as_ref().unwrap(),
                    0,
                    bytemuck::bytes_of(&clip_params),
                );

                // Upload concatenated brick indices
                let all_indices = clip_map.build_all_brick_indices();
                ctx.queue.write_buffer(
                    self.all_brick_indices_buffer.as_ref().unwrap(),
                    0,
                    bytemuck::cast_slice(&all_indices),
                );

                log::debug!(
                    "SDF ClipMap: dirty_mask={:#06b}, active counts={:?}",
                    dirty_mask,
                    self.clip_active_counts.as_ref().unwrap().lock().unwrap()
                );
            }
        }

        Ok(())
    }

    fn shader_defines(&self) -> HashMap<String, ShaderDefine> {
        let mut defines = HashMap::new();
        defines.insert("ENABLE_SDF".into(), ShaderDefine::Bool(self.enabled));
        defines
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn cleanup(&mut self, _device: &wgpu::Device) {
        // Shared
        self.edit_buffer = None;
        self.params_buffer = None;
        // Dense
        self.volume_texture = None;
        self.volume_view = None;
        self.sampler = None;
        self.eval_pipeline = None;
        self.eval_bind_group = None;
        self.eval_bg_layout = None;
        self.march_pipeline = None;
        self.march_bind_group = None;
        // Sparse
        self.brick_map = None;
        self.sparse_active_count = None;
        self.sparse_eval_pipeline = None;
        self.sparse_eval_bind_group = None;
        self.sparse_eval_bg_layout = None;
        self.sparse_march_pipeline = None;
        self.sparse_march_bind_group = None;
        // ClipMap
        self.clip_map = None;
        self.clip_active_counts = None;
        self.clip_march_pipeline = None;
        self.clip_march_bind_group = None;
        self.all_brick_indices_buffer = None;
    }
}
