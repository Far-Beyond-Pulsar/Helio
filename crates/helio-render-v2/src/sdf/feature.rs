//! SdfFeature — implements the Feature trait for SDF rendering
//!
//! Creates GPU resources (3D volume texture, edit buffer, compute + render pipelines)
//! in `register()` and uploads per-frame data in `prepare()`.
//!
//! Uses geometry clip maps with multiple nested LOD levels centered on camera.

use crate::features::{Feature, FeatureContext, PrepareContext, ShaderDefine};
use crate::Result;
use super::edit_list::{SdfEditList, SdfEdit, GpuSdfEdit, BooleanOp};
use super::edit_bvh::EditBvh;
use super::primitives::SdfShapeType;
use super::uniforms::SdfGridParams;
use super::terrain::{TerrainConfig, GpuTerrainParams};
use super::clip_map::{SdfClipMap, DEFAULT_CLIP_LEVELS};
use super::passes::clip_update::SdfClipUpdatePass;
use super::passes::ray_march::SdfRayMarchPass;
use super::brick::{BrickMap, DEFAULT_BRICK_SIZE};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Default grid resolution (128^3)
const DEFAULT_GRID_DIM: u32 = 128;

/// Result of a CPU-side surface pick (ray march hit).
#[derive(Clone, Debug)]
pub struct PickResult {
    /// World-space hit position on the SDF surface.
    pub position: glam::Vec3,
    /// Estimated surface normal at the hit point.
    pub normal: glam::Vec3,
    /// Distance from ray origin to the hit point.
    pub distance: f32,
}

/// SDF rendering feature
///
/// # Usage
/// ```ignore
/// let sdf = SdfFeature::new()
///     .with_grid_dim(128)
///     .with_volume_bounds([-10.0, -10.0, -10.0], [10.0, 10.0, 10.0]);
/// ```
pub struct SdfFeature {
    enabled: bool,
    debug_mode: bool,
    grid_dim: u32,
    volume_min: [f32; 3],
    volume_max: [f32; 3],

    // CPU-side state
    edit_list: SdfEditList,
    last_uploaded_gen: u64,

    // GPU resources
    edit_buffer: Option<Arc<wgpu::Buffer>>,
    params_buffer: Option<Arc<wgpu::Buffer>>,

    // ClipMap mode resources
    clip_map: Option<SdfClipMap>,
    clip_active_counts: Option<Arc<Mutex<Vec<u32>>>>,
    clip_march_pipeline: Option<Arc<wgpu::RenderPipeline>>,
    clip_march_bind_group: Option<Arc<wgpu::BindGroup>>,
    all_brick_indices_buffer: Option<Arc<wgpu::Buffer>>,

    // Terrain
    terrain_config: Option<TerrainConfig>,
    terrain_params_buffer: Option<Arc<wgpu::Buffer>>,

    // Incremental classification
    /// When set, the last change was a single edit add — use incremental classify.
    incremental_edit: Option<SdfEdit>,

    // BVH for O(log n) edit culling
    edit_bvh: EditBvh,
}

impl SdfFeature {
    pub fn new() -> Self {
        Self {
            enabled: true,
            debug_mode: false,
            grid_dim: DEFAULT_GRID_DIM,
            volume_min: [-10.0, -10.0, -10.0],
            volume_max: [10.0, 10.0, 10.0],
            edit_list: SdfEditList::new(),
            last_uploaded_gen: u64::MAX,
            edit_buffer: None,
            params_buffer: None,
            clip_map: None,
            clip_active_counts: None,
            clip_march_pipeline: None,
            clip_march_bind_group: None,
            all_brick_indices_buffer: None,
            terrain_config: None,
            terrain_params_buffer: None,
            incremental_edit: None,
            edit_bvh: EditBvh::new(),
        }
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

    /// Enable procedural terrain generation with the given configuration.
    pub fn with_terrain(mut self, config: TerrainConfig) -> Self {
        self.terrain_config = Some(config);
        self
    }

    /// Set or replace the terrain configuration at runtime.
    pub fn set_terrain(&mut self, config: Option<TerrainConfig>) {
        self.terrain_config = config;
        // Force re-upload so terrain params reach the GPU
        self.last_uploaded_gen = u64::MAX;
    }

    /// Access the current terrain configuration.
    pub fn terrain_config(&self) -> Option<&TerrainConfig> {
        self.terrain_config.as_ref()
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
        self.incremental_edit = Some(edit.clone());
        self.edit_list.add(edit);
    }

    /// Remove an SDF edit by index.
    pub fn remove_edit(&mut self, index: usize) {
        self.incremental_edit = None; // can't use incremental for removals
        self.edit_list.remove(index);
    }

    /// Replace an SDF edit at the given index.
    pub fn set_edit(&mut self, index: usize, edit: SdfEdit) {
        self.incremental_edit = None; // can't use incremental for replacements
        self.edit_list.set(index, edit);
    }

    /// Clear all SDF edits.
    pub fn clear_edits(&mut self) {
        self.incremental_edit = None;
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

    // ── CPU-side picking ─────────────────────────────────────────────────────

    /// CPU-side sphere-trace along a ray to find the nearest surface hit.
    /// Returns `Some(PickResult)` if a surface is hit within `max_dist`.
    ///
    /// The pick evaluates the full SDF analytically on the CPU:
    /// terrain (if enabled) + all edits with boolean operations.
    pub fn pick_surface(&self, ray_origin: glam::Vec3, ray_dir: glam::Vec3, max_dist: f32) -> Option<PickResult> {
        let edits = self.edit_list.edits();
        let terrain = self.terrain_config.as_ref();
        let hit_threshold = 0.02;
        let max_steps = 256;

        // Pre-compute inverse transforms once (avoids 256*N matrix inversions)
        let inv_transforms: Vec<glam::Mat4> = edits.iter()
            .map(|e| e.transform.inverse())
            .collect();

        let mut t = 0.0f32;
        for _ in 0..max_steps {
            let p = ray_origin + ray_dir * t;
            let d = Self::cpu_evaluate_sdf(p, edits, &inv_transforms, terrain);
            if d.abs() < hit_threshold {
                let normal = Self::cpu_estimate_normal(p, edits, &inv_transforms, terrain);
                return Some(PickResult {
                    position: p,
                    normal,
                    distance: t,
                });
            }
            t += d.max(0.01);
            if t > max_dist {
                break;
            }
        }
        None
    }

    /// Evaluate the full SDF at a world position on the CPU.
    fn cpu_evaluate_sdf(
        pos: glam::Vec3,
        edits: &[SdfEdit],
        inv_transforms: &[glam::Mat4],
        terrain: Option<&TerrainConfig>,
    ) -> f32 {
        let mut dist = match terrain {
            Some(cfg) => super::noise::terrain_sdf(pos, cfg),
            None => 1e10,
        };

        for (edit, inv) in edits.iter().zip(inv_transforms.iter()) {
            let local_pos = (*inv * glam::Vec4::new(pos.x, pos.y, pos.z, 1.0)).truncate();
            let d = Self::cpu_evaluate_shape(local_pos, edit);
            dist = Self::cpu_apply_boolean(dist, d, edit.op, edit.blend_radius);
        }

        dist
    }

    /// Evaluate a single SDF primitive in local space.
    fn cpu_evaluate_shape(p: glam::Vec3, edit: &SdfEdit) -> f32 {
        match edit.shape {
            SdfShapeType::Sphere => {
                p.length() - edit.params.param0
            }
            SdfShapeType::Cube => {
                let half = glam::Vec3::new(edit.params.param0, edit.params.param1, edit.params.param2);
                let d = p.abs() - half;
                d.max(glam::Vec3::ZERO).length() + d.x.max(d.y.max(d.z)).min(0.0)
            }
            SdfShapeType::Capsule => {
                let radius = edit.params.param0;
                let half_h = edit.params.param1;
                let mut q = p;
                q.y -= q.y.clamp(-half_h, half_h);
                q.length() - radius
            }
            SdfShapeType::Torus => {
                let major_r = edit.params.param0;
                let minor_r = edit.params.param1;
                let q = glam::Vec2::new(glam::Vec2::new(p.x, p.z).length() - major_r, p.y);
                q.length() - minor_r
            }
            SdfShapeType::Cylinder => {
                let radius = edit.params.param0;
                let half_h = edit.params.param1;
                let d = glam::Vec2::new(glam::Vec2::new(p.x, p.z).length(), p.y).abs()
                    - glam::Vec2::new(radius, half_h);
                d.x.max(d.y).min(0.0) + d.max(glam::Vec2::ZERO).length()
            }
        }
    }

    /// Apply a boolean operation between accumulated distance and a shape distance.
    fn cpu_apply_boolean(d1: f32, d2: f32, op: BooleanOp, k: f32) -> f32 {
        let use_blend = k > 0.001;
        match op {
            BooleanOp::Union => {
                if use_blend {
                    let h = (0.5 + 0.5 * (d2 - d1) / k).clamp(0.0, 1.0);
                    d1 * h + d2 * (1.0 - h) - k * h * (1.0 - h)
                } else {
                    d1.min(d2)
                }
            }
            BooleanOp::Subtraction => {
                if use_blend {
                    let h = (0.5 - 0.5 * (d2 + d1) / k).clamp(0.0, 1.0);
                    d1 * (1.0 - h) + (-d2) * h + k * h * (1.0 - h)
                } else {
                    d1.max(-d2)
                }
            }
            BooleanOp::Intersection => {
                if use_blend {
                    let h = (0.5 - 0.5 * (d2 - d1) / k).clamp(0.0, 1.0);
                    d1 * h + d2 * (1.0 - h) + k * h * (1.0 - h)
                } else {
                    d1.max(d2)
                }
            }
        }
    }

    /// Estimate the SDF normal at a point via central differences.
    fn cpu_estimate_normal(
        p: glam::Vec3,
        edits: &[SdfEdit],
        inv_transforms: &[glam::Mat4],
        terrain: Option<&TerrainConfig>,
    ) -> glam::Vec3 {
        let eps = 0.01;
        let dx = Self::cpu_evaluate_sdf(p + glam::Vec3::X * eps, edits, inv_transforms, terrain)
               - Self::cpu_evaluate_sdf(p - glam::Vec3::X * eps, edits, inv_transforms, terrain);
        let dy = Self::cpu_evaluate_sdf(p + glam::Vec3::Y * eps, edits, inv_transforms, terrain)
               - Self::cpu_evaluate_sdf(p - glam::Vec3::Y * eps, edits, inv_transforms, terrain);
        let dz = Self::cpu_evaluate_sdf(p + glam::Vec3::Z * eps, edits, inv_transforms, terrain)
               - Self::cpu_evaluate_sdf(p - glam::Vec3::Z * eps, edits, inv_transforms, terrain);
        glam::Vec3::new(dx, dy, dz).normalize_or_zero()
    }

    // ── ClipMap registration ────────────────────────────────────────────────

    fn register_clipmap(&mut self, ctx: &mut FeatureContext) -> Result<()> {
        let device = ctx.device;
        let dim = self.grid_dim;
        let edit_buffer = self.edit_buffer.as_ref().unwrap();
        let terrain_buffer = self.terrain_params_buffer.as_ref().unwrap();

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
            let atlas_buffer = level.brick_map.atlas_buffer.as_ref().unwrap().clone();
            let brick_index_buffer = level.brick_map.brick_index_buffer.as_ref().unwrap().clone();
            let active_bricks_buffer = level.brick_map.active_bricks_buffer.as_ref().unwrap().clone();
            let edit_list_offsets_buffer = level.brick_map.edit_list_offsets_buffer.as_ref().unwrap().clone();
            let edit_list_data_buffer = level.brick_map.edit_list_data_buffer.as_ref().unwrap().clone();
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
                        resource: atlas_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: active_bricks_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: brick_index_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: terrain_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: edit_list_offsets_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: edit_list_data_buffer.as_entire_binding(),
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
        //   b1..b(level_count): storage read (atlas buffer per level)
        //   b(level_count+1): storage read (all_brick_indices)
        let level_count = DEFAULT_CLIP_LEVELS as usize;
        let mut layout_entries = vec![
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
        ];
        for i in 0..level_count {
            layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: (1 + i) as u32,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: (1 + level_count) as u32,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });

        let march_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SDF Ray March ClipMap BG Layout"),
            entries: &layout_entries,
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

        // Create ray march bind group with per-level atlas buffers
        let clip_params_buffer = clip_map.clip_params_buffer.as_ref().unwrap().clone();
        let atlas_buffers: Vec<_> = clip_map.levels().iter()
            .map(|l| l.brick_map.atlas_buffer.as_ref().unwrap().clone())
            .collect();

        let mut bg_entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: clip_params_buffer.as_entire_binding(),
            },
        ];
        for (i, buf) in atlas_buffers.iter().enumerate() {
            bg_entries.push(wgpu::BindGroupEntry {
                binding: (1 + i) as u32,
                resource: buf.as_entire_binding(),
            });
        }
        bg_entries.push(wgpu::BindGroupEntry {
            binding: (1 + atlas_buffers.len()) as u32,
            resource: all_brick_indices_buffer.as_entire_binding(),
        });

        let march_bind_group = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SDF Ray March ClipMap Bind Group"),
            layout: &march_bg_layout,
            entries: &bg_entries,
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

        log::info!("SDF Feature: registering with {}^3 grid (ClipMap mode)", self.grid_dim);

        // ── Shared resources ────────────────────────────────────────────────
        // Edit buffer: starts with room for 1024 edits, can be reallocated if needed.
        let initial_edit_capacity = 1024;
        let edit_buffer_size = (initial_edit_capacity * std::mem::size_of::<GpuSdfEdit>()) as u64;
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

        // Terrain params buffer (always created; uploaded as disabled if no terrain config)
        let terrain_params_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Terrain Params"),
            size: std::mem::size_of::<GpuTerrainParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.terrain_params_buffer = Some(terrain_params_buffer);

        // ── ClipMap registration ─────────────────────────────────────────────
        self.register_clipmap(ctx)?;

        log::info!("SDF Feature: registered successfully");
        Ok(())
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> Result<()> {
        if !self.enabled { return Ok(()); }

        let gen = self.edit_list.generation();
        let needs_upload = gen != self.last_uploaded_gen;

        if needs_upload {
            // Rebuild BVH from current edits for O(log n) spatial queries
            let edits = self.edit_list.edits();
            let bounds: Vec<(glam::Vec3, f32)> = edits
                .iter()
                .map(|edit| BrickMap::edit_bounding_sphere(edit))
                .collect();
            self.edit_bvh.rebuild(&bounds);

            // Upload edit list to GPU
            let gpu_edits = self.edit_list.flush_gpu_data();
            if !gpu_edits.is_empty() {
                ctx.queue.write_buffer(
                    self.edit_buffer.as_ref().unwrap(),
                    0,
                    bytemuck::cast_slice(&gpu_edits),
                );
            }

            // Upload terrain params
            let terrain_gpu = match &self.terrain_config {
                Some(cfg) => cfg.build_gpu_params(),
                None => GpuTerrainParams::disabled(),
            };
            ctx.queue.write_buffer(
                self.terrain_params_buffer.as_ref().unwrap(),
                0,
                bytemuck::bytes_of(&terrain_gpu),
            );

            // ClipMap edit handling continues below in the per-frame section

            self.last_uploaded_gen = gen;
        }

        // Per-frame center tracking for clip map streaming
        {
            // Split borrows: mut clip_map + immutable refs to other fields
            let SdfFeature {
                ref mut clip_map,
                ref edit_bvh,
                ref terrain_config,
                ref edit_list,
                ref volume_min,
                ref volume_max,
                ref debug_mode,
                ref mut incremental_edit,
                ref clip_active_counts,
                ref all_brick_indices_buffer,
                ..
            } = *self;
            let clip_map = clip_map.as_mut().unwrap();
            let bvh_opt = Some(edit_bvh as &EditBvh);

            // When terrain is enabled, center on camera so terrain streams as
            // the player moves. Otherwise center on volume midpoint to keep
            // static edits in the finest level.
            let center = if terrain_config.is_some() {
                ctx.camera.position
            } else {
                glam::Vec3::new(
                    (volume_min[0] + volume_max[0]) * 0.5,
                    (volume_min[1] + volume_max[1]) * 0.5,
                    (volume_min[2] + volume_max[2]) * 0.5,
                )
            };
            let camera_dirty_mask = clip_map.update_center(center);
            let edit_count = edit_list.len() as u32;

            // ── Path A: Camera moved → toroidal scroll (edge-only update) ──
            if camera_dirty_mask != 0 {
                if needs_upload {
                    // Both camera AND edits changed: full toroidal reclassify
                    let all_mask = (1 << clip_map.level_count()) - 1;
                    clip_map.classify_toroidal_levels(all_mask, edit_list.edits(), terrain_config.as_ref(), bvh_opt);
                } else {
                    // Camera moved, edits unchanged: scroll (L-shaped edge update)
                    clip_map.scroll_toroidal_levels(camera_dirty_mask, edit_list.edits(), terrain_config.as_ref(), bvh_opt);
                }

                let counts = clip_map.upload_dirty_toroidal(ctx.queue, edit_count);
                {
                    let mut shared = clip_active_counts.as_ref().unwrap().lock().unwrap();
                    *shared = counts;
                }

                // Upload clip map params and brick indices (always needed for ray march)
                let mut clip_params = clip_map.build_clip_params();
                clip_params.debug_flags = if *debug_mode { 1 } else { 0 };
                ctx.queue.write_buffer(
                    clip_map.clip_params_buffer.as_ref().unwrap(),
                    0,
                    bytemuck::bytes_of(&clip_params),
                );
                let update_mask = if needs_upload {
                    (1u32 << clip_map.level_count()) - 1 // edit + scroll: all levels
                } else {
                    camera_dirty_mask // scroll only: just scrolled levels
                };
                clip_map.update_cached_indices(update_mask);
                ctx.queue.write_buffer(
                    all_brick_indices_buffer.as_ref().unwrap(),
                    0,
                    bytemuck::cast_slice(clip_map.cached_all_indices()),
                );

                log::debug!(
                    "SDF ClipMap (scroll): camera_dirty={:#06b}, dirty counts={:?}",
                    camera_dirty_mask,
                    clip_active_counts.as_ref().unwrap().lock().unwrap()
                );
            }
            // ── Path B: Edits changed, camera static → edit-only dirty ──
            else if needs_upload {
                if let Some(ref inc_edit) = *incremental_edit {
                    // Single edit added: mark only overlapping bricks dirty
                    clip_map.mark_edit_dirty_levels(inc_edit, edit_list.edits(), terrain_config.as_ref(), bvh_opt);
                } else {
                    // Bulk change (undo/redo/clear): full toroidal reclassify
                    let all_mask = (1 << clip_map.level_count()) - 1;
                    clip_map.classify_toroidal_levels(all_mask, edit_list.edits(), terrain_config.as_ref(), bvh_opt);
                }

                let counts = clip_map.upload_dirty_toroidal(ctx.queue, edit_count);
                {
                    let mut shared = clip_active_counts.as_ref().unwrap().lock().unwrap();
                    *shared = counts;
                }

                let mut clip_params = clip_map.build_clip_params();
                clip_params.debug_flags = if *debug_mode { 1 } else { 0 };
                ctx.queue.write_buffer(
                    clip_map.clip_params_buffer.as_ref().unwrap(),
                    0,
                    bytemuck::bytes_of(&clip_params),
                );
                let all_levels_mask = (1u32 << clip_map.level_count()) - 1;
                clip_map.update_cached_indices(all_levels_mask);
                ctx.queue.write_buffer(
                    all_brick_indices_buffer.as_ref().unwrap(),
                    0,
                    bytemuck::cast_slice(clip_map.cached_all_indices()),
                );

                log::debug!(
                    "SDF ClipMap (edit): dirty counts={:?}",
                    clip_active_counts.as_ref().unwrap().lock().unwrap()
                );
            }
            // ── Path C: Nothing changed → no GPU work ──
            else {
                // Reset dispatch counts so the compute pass doesn't re-evaluate
                // stale dirty bricks from the previous frame.
                let mut shared = clip_active_counts.as_ref().unwrap().lock().unwrap();
                for c in shared.iter_mut() { *c = 0; }
            }

            // Consume incremental_edit now that ClipMap has used it
            *incremental_edit = None;
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
        self.edit_buffer = None;
        self.params_buffer = None;
        self.clip_map = None;
        self.clip_active_counts = None;
        self.clip_march_pipeline = None;
        self.clip_march_bind_group = None;
        self.all_brick_indices_buffer = None;
        self.terrain_params_buffer = None;
    }
}
