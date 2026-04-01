//! Helio render pass: sparse SDF terrain with clip-map LOD, brick atlas, and
//! fullscreen ray march.
//!
//! The pass owns a [`SdfClipMap`] (multi-level sparse brick maps) centered on
//! the camera. Each frame `prepare()` handles edit uploads, BVH updates, and
//! toroidal scroll/classify, while `execute()` dispatches one compute workgroup
//! per dirty brick per level, then draws a fullscreen triangle that sphere-traces
//! through the cached SDF volume.

pub mod brick;
pub mod clip_map;
pub mod edit_bvh;
pub mod edit_list;
pub mod noise;
pub mod primitives;
pub mod terrain;
pub mod uniforms;

pub use clip_map::SdfClipMap;
pub use edit_list::{BooleanOp, GpuSdfEdit, SdfEdit, SdfEditList};
pub use primitives::{SdfShapeParams, SdfShapeType};
pub use terrain::{GpuTerrainParams, TerrainConfig, TerrainStyle};
pub use uniforms::SdfGridParams;

use brick::{BrickMap, DEFAULT_BRICK_SIZE};
use clip_map::DEFAULT_CLIP_LEVELS;
use edit_bvh::EditBvh;

use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

/// Default grid resolution per clip level.
const DEFAULT_GRID_DIM: u32 = 128;

/// Maximum number of edits before GPU buffer grows.
const INITIAL_EDIT_CAPACITY: usize = 1024;

// The camera uniform buffer is NOT owned by this pass.
// We bind the engine's camera buffer (`ctx.scene.camera`) directly,
// which has the layout defined by `libhelio::camera::GpuCameraUniforms`.

// ─────────────────────────────────────────────────────────────────────────────
// SDF pass
// ─────────────────────────────────────────────────────────────────────────────

/// Self-contained SDF render pass implementing `helio_v3::RenderPass`.
///
/// # Pipelines
///
/// * **Compute** (`eval_pipeline`): dispatches one workgroup per dirty brick per
///   clip level — evaluates the edit list + terrain noise, writes u8 distances
///   into the atlas.
/// * **Render** (`march_pipeline`): fullscreen triangle that sphere-traces through
///   the clip map levels, writing color + depth.
pub struct SdfPass {
    // ── GPU pipelines ──────────────────────────────────────────────────────
    eval_pipeline: wgpu::ComputePipeline,
    eval_bgl: wgpu::BindGroupLayout,
    march_pipeline: wgpu::RenderPipeline,
    march_bgl: wgpu::BindGroupLayout,

    // ── Buffers ────────────────────────────────────────────────────────────
    edit_buffer: wgpu::Buffer,
    terrain_params_buffer: wgpu::Buffer,
    all_brick_indices_buffer: wgpu::Buffer,

    // ── Per-level compute bind groups ──────────────────────────────────────
    level_bind_groups: Vec<wgpu::BindGroup>,
    /// Ray march bind group — rebuilt when the engine's camera buffer changes.
    march_bind_group: Option<wgpu::BindGroup>,
    /// Pointer-based key to detect engine camera buffer reallocation.
    march_bg_camera_key: usize,

    // ── CPU-side state ─────────────────────────────────────────────────────
    clip_map: SdfClipMap,
    edit_list: SdfEditList,
    edit_bvh: EditBvh,
    terrain_config: Option<TerrainConfig>,
    last_uploaded_gen: u64,
    incremental_edit: Option<SdfEdit>,
    level_dirty_counts: Vec<u32>,
    debug_mode: bool,
    enabled: bool,
    /// If true, loads existing color/depth (for deferred pipeline integration).
    /// If false, clears to sky-blue (for standalone SDF-only rendering).
    preserve_framebuffer: bool,

    // ── Volume settings ────────────────────────────────────────────────────
    #[allow(dead_code)]
    grid_dim: u32,
    volume_min: [f32; 3],
    volume_max: [f32; 3],
    #[allow(dead_code)]
    surface_format: wgpu::TextureFormat,
}

impl SdfPass {
    /// Creates a new SDF pass, constructing all GPU resources and pipelines.
    ///
    /// `surface_format` should match the render target colour format (e.g.,
    /// `Bgra8UnormSrgb`).
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        terrain: Option<TerrainConfig>,
    ) -> Self {
        Self::with_grid(
            device,
            surface_format,
            DEFAULT_GRID_DIM,
            [-50.0; 3],
            [50.0; 3],
            terrain,
        )
    }

    pub fn with_grid(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        grid_dim: u32,
        volume_min: [f32; 3],
        volume_max: [f32; 3],
        terrain: Option<TerrainConfig>,
    ) -> Self {
        let range_x = volume_max[0] - volume_min[0];
        let base_voxel_size = range_x / grid_dim as f32;

        // ── Clip map ──────────────────────────────────────────────────────────
        let mut clip_map = SdfClipMap::new(grid_dim, base_voxel_size, DEFAULT_CLIP_LEVELS);
        clip_map.create_gpu_resources(device);

        // ── Shared buffers ────────────────────────────────────────────────────
        let edit_buffer_size = (INITIAL_EDIT_CAPACITY * std::mem::size_of::<GpuSdfEdit>()) as u64;
        let edit_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Edit Buffer"),
            size: edit_buffer_size.max(64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let terrain_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Terrain Params"),
            size: std::mem::size_of::<GpuTerrainParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgd = grid_dim / DEFAULT_BRICK_SIZE;
        let entries_per_level = (bgd * bgd * bgd) as usize;
        let total_entries = entries_per_level * DEFAULT_CLIP_LEVELS as usize;
        let all_brick_indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Clip All Brick Indices"),
            size: (total_entries * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Compute pipeline (evaluate sparse) ───────────────────────────────
        let eval_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Evaluate"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sdf_evaluate.wgsl").into()),
        });
        let eval_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SDF Evaluate Pipeline"),
            layout: None, // auto layout from shader
            module: &eval_shader,
            entry_point: Some("cs_evaluate_sparse"),
            compilation_options: Default::default(),
            cache: None,
        });
        let eval_bgl = eval_pipeline.get_bind_group_layout(0);

        // Per-level compute bind groups
        let level_bind_groups = Self::build_level_bind_groups(
            device,
            &eval_bgl,
            &clip_map,
            &edit_buffer,
            &terrain_params_buffer,
        );

        // ── Render pipeline (ray march) ──────────────────────────────────────
        let march_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Ray March"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sdf_ray_march.wgsl").into()),
        });

        let march_bgl = Self::build_march_bgl(device);

        let march_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("SDF Ray March PL"),
                bind_group_layouts: &[Some(&march_bgl)],
                immediate_size: 0,
            });

        let march_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                    format: surface_format,
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
        });

        Self {
            eval_pipeline,
            eval_bgl,
            march_pipeline,
            march_bgl,
            edit_buffer,
            terrain_params_buffer,
            all_brick_indices_buffer,
            level_bind_groups,
            march_bind_group: None,
            march_bg_camera_key: 0,
            clip_map,
            edit_list: SdfEditList::new(),
            edit_bvh: EditBvh::new(),
            terrain_config: terrain,
            last_uploaded_gen: u64::MAX,
            incremental_edit: None,
            level_dirty_counts: vec![0; DEFAULT_CLIP_LEVELS as usize],
            debug_mode: false,
            enabled: true,
            preserve_framebuffer: false, // Default: clear framebuffer (standalone mode)
            grid_dim,
            volume_min,
            volume_max,
            surface_format,
        }
    }

    // ── Public API ─────────────────────────────────────────────────────────

    pub fn add_edit(&mut self, edit: SdfEdit) {
        self.incremental_edit = Some(edit.clone());
        self.edit_list.add(edit);
    }

    /// Removes an edit. Note: This triggers full reclassification because
    /// edit indices shift and the spatial dirty region is hard to track incrementally.
    /// For better performance, prefer using `add_edit` for additive changes.
    pub fn remove_edit(&mut self, index: usize) {
        // TODO: Implement incremental removal tracking for better performance.
        // This would require tracking the removed edit's bounds separately.
        self.incremental_edit = None;
        self.edit_list.remove(index);
    }

    /// Updates an existing edit. If the edit's transform changed, this triggers
    /// full reclassification. Use `prepare_edit_move` + `commit_edit_move` for
    /// incremental move support (not yet implemented).
    pub fn set_edit(&mut self, index: usize, edit: SdfEdit) {
        // TODO: Implement incremental edit modification tracking.
        // This would require marking dirty in both old and new positions.
        self.incremental_edit = None;
        self.edit_list.set(index, edit);
    }

    pub fn clear_edits(&mut self) {
        self.incremental_edit = None;
        self.edit_list.clear();
    }

    pub fn edit_list(&self) -> &SdfEditList {
        &self.edit_list
    }
    pub fn edit_list_mut(&mut self) -> &mut SdfEditList {
        &mut self.edit_list
    }

    pub fn set_terrain(&mut self, config: Option<TerrainConfig>) {
        self.terrain_config = config;
        self.last_uploaded_gen = u64::MAX; // force re-upload
    }

    pub fn terrain_config(&self) -> Option<&TerrainConfig> {
        self.terrain_config.as_ref()
    }

    pub fn toggle_debug(&mut self) {
        self.debug_mode = !self.debug_mode;
        self.last_uploaded_gen = u64::MAX;
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Set whether to preserve framebuffer contents (for deferred pipeline)
    /// or clear to sky-blue (for standalone rendering).
    pub fn set_preserve_framebuffer(&mut self, preserve: bool) {
        self.preserve_framebuffer = preserve;
    }
    pub fn preserve_framebuffer(&self) -> bool {
        self.preserve_framebuffer
    }

    // ── CPU-side picking ─────────────────────────────────────────────────

    /// CPU-side sphere-trace for surface picking.
    pub fn pick_surface(
        &self,
        ray_origin: glam::Vec3,
        ray_dir: glam::Vec3,
        max_dist: f32,
    ) -> Option<PickResult> {
        let edits = self.edit_list.edits();
        let terrain = self.terrain_config.as_ref();
        let inv_transforms: Vec<glam::Mat4> = edits.iter().map(|e| e.transform.inverse()).collect();

        let mut t = 0.0f32;
        for _ in 0..256 {
            let p = ray_origin + ray_dir * t;
            let d = cpu_evaluate_sdf(p, edits, &inv_transforms, terrain);
            if d.abs() < 0.02 {
                let n = cpu_estimate_normal(p, edits, &inv_transforms, terrain);
                return Some(PickResult {
                    position: p,
                    normal: n,
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

    // ── Internal helpers ───────────────────────────────────────────────────

    fn build_level_bind_groups(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        clip_map: &SdfClipMap,
        edit_buffer: &wgpu::Buffer,
        terrain_buffer: &wgpu::Buffer,
    ) -> Vec<wgpu::BindGroup> {
        clip_map
            .levels()
            .iter()
            .map(|level| {
                let bm = &level.brick_map;
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("SDF Eval BG L{}", level.level_index)),
                    layout: bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: level.params_buffer.as_ref().unwrap().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: edit_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: bm.atlas_buffer.as_ref().unwrap().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: bm
                                .active_bricks_buffer
                                .as_ref()
                                .unwrap()
                                .as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: bm.brick_index_buffer.as_ref().unwrap().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: terrain_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: bm
                                .edit_list_offsets_buffer
                                .as_ref()
                                .unwrap()
                                .as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: bm
                                .edit_list_data_buffer
                                .as_ref()
                                .unwrap()
                                .as_entire_binding(),
                        },
                    ],
                })
            })
            .collect()
    }

    fn build_march_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let level_count = DEFAULT_CLIP_LEVELS as usize;
        // b0: camera uniform
        // b1: clip params uniform
        // b2..b(2+level_count-1): atlas storage per level
        // b(2+level_count): all_brick_indices storage
        let mut entries = vec![
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];
        for i in 0..level_count {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: (2 + i) as u32,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: (2 + level_count) as u32,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });

        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SDF Ray March BGL"),
            entries: &entries,
        })
    }

    fn build_march_bind_group(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        clip_map: &SdfClipMap,
        all_brick_indices: &wgpu::Buffer,
        camera_buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let level_count = DEFAULT_CLIP_LEVELS as usize;
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: clip_map
                    .clip_params_buffer
                    .as_ref()
                    .unwrap()
                    .as_entire_binding(),
            },
        ];
        for (i, level) in clip_map.levels().iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: (2 + i) as u32,
                resource: level
                    .brick_map
                    .atlas_buffer
                    .as_ref()
                    .unwrap()
                    .as_entire_binding(),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: (2 + level_count) as u32,
            resource: all_brick_indices.as_entire_binding(),
        });

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SDF Ray March BG"),
            layout: bgl,
            entries: &entries,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RenderPass implementation
// ─────────────────────────────────────────────────────────────────────────────

impl RenderPass for SdfPass {
    fn name(&self) -> &'static str {
        "SDF"
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        if !self.enabled {
            return Ok(());
        }

        let gen = self.edit_list.generation();
        let needs_upload = gen != self.last_uploaded_gen;

        // ── Upload edits + terrain when dirty ────────────────────────────────
        if needs_upload {
            let edits = self.edit_list.edits();
            let bounds: Vec<(glam::Vec3, f32)> =
                edits.iter().map(BrickMap::edit_bounding_sphere).collect();

            // BVH: incremental insert for single edit, full rebuild otherwise
            if self.incremental_edit.is_some() {
                let idx = edits.len() - 1;
                let (center, radius) = bounds[idx];
                let aabb = edit_bvh::Aabb::from_center_radius(center, radius);
                self.edit_bvh.insert(idx, aabb);
            } else {
                self.edit_bvh.rebuild(&bounds);
            }

            // Upload edit data to GPU (grow buffer when needed)
            let gpu_edits = self.edit_list.flush_gpu_data();
            if !gpu_edits.is_empty() {
                let required = (gpu_edits.len() * std::mem::size_of::<GpuSdfEdit>()) as u64;
                if required > self.edit_buffer.size() {
                    let new_size = (required * 2).max(64);
                    log::info!(
                        "SDF edit buffer grown: {} -> {} bytes",
                        self.edit_buffer.size(),
                        new_size
                    );
                    self.edit_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("SDF Edit Buffer"),
                        size: new_size,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    // Rebuild compute bind groups that reference the edit buffer
                    self.level_bind_groups = Self::build_level_bind_groups(
                        ctx.device,
                        &self.eval_bgl,
                        &self.clip_map,
                        &self.edit_buffer,
                        &self.terrain_params_buffer,
                    );
                    // Force march BG rebuild next execute()
                    self.march_bind_group = None;
                    self.march_bg_camera_key = 0;
                }
                ctx.queue
                    .write_buffer(&self.edit_buffer, 0, bytemuck::cast_slice(&gpu_edits));
            }

            // Upload terrain params
            let terrain_gpu = match &self.terrain_config {
                Some(cfg) => cfg.build_gpu_params(),
                None => GpuTerrainParams::disabled(),
            };
            ctx.queue.write_buffer(
                &self.terrain_params_buffer,
                0,
                bytemuck::bytes_of(&terrain_gpu),
            );

            self.last_uploaded_gen = gen;
        }

        // ── Camera position for clip map centering ─────────────────────────
        // The engine uploads real camera matrices to scene.camera.buffer();
        // we bind that buffer directly in execute().
        let cam_pos = ctx.scene.camera.position();

        // ── Per-frame clip map streaming ───────────────────────────────────
        let center = if self.terrain_config.is_some() {
            glam::Vec3::from(cam_pos)
        } else {
            glam::Vec3::new(
                (self.volume_min[0] + self.volume_max[0]) * 0.5,
                (self.volume_min[1] + self.volume_max[1]) * 0.5,
                (self.volume_min[2] + self.volume_max[2]) * 0.5,
            )
        };

        let camera_dirty_mask = self.clip_map.update_center(center);
        let edit_count = self.edit_list.len() as u32;
        let bvh_opt = Some(&self.edit_bvh as &EditBvh);

        if camera_dirty_mask != 0 {
            if needs_upload {
                let all_mask = (1u32 << self.clip_map.level_count()) - 1;
                self.clip_map.classify_toroidal_levels(
                    all_mask,
                    self.edit_list.edits(),
                    self.terrain_config.as_ref(),
                    bvh_opt,
                );
            } else {
                self.clip_map.scroll_toroidal_levels(
                    camera_dirty_mask,
                    self.edit_list.edits(),
                    self.terrain_config.as_ref(),
                    bvh_opt,
                );
            }

            self.level_dirty_counts = self.clip_map.upload_dirty_toroidal(ctx.queue, edit_count);
            log::info!(
                "SDF cam_dirty={:#x}: dirty_counts={:?} center={:?}",
                camera_dirty_mask,
                self.level_dirty_counts,
                center,
            );

            let mut clip_params = self.clip_map.build_clip_params();
            clip_params.debug_flags = if self.debug_mode { 1 } else { 0 };
            ctx.queue.write_buffer(
                self.clip_map.clip_params_buffer.as_ref().unwrap(),
                0,
                bytemuck::bytes_of(&clip_params),
            );

            let update_mask = if needs_upload {
                (1u32 << self.clip_map.level_count()) - 1
            } else {
                camera_dirty_mask
            };
            self.clip_map.update_cached_indices(update_mask);
            ctx.queue.write_buffer(
                &self.all_brick_indices_buffer,
                0,
                bytemuck::cast_slice(self.clip_map.cached_all_indices()),
            );
        } else if needs_upload {
            if let Some(ref inc_edit) = self.incremental_edit {
                self.clip_map.mark_edit_dirty_levels(
                    inc_edit,
                    self.edit_list.edits(),
                    self.terrain_config.as_ref(),
                    bvh_opt,
                );
            } else {
                let all_mask = (1u32 << self.clip_map.level_count()) - 1;
                self.clip_map.classify_toroidal_levels(
                    all_mask,
                    self.edit_list.edits(),
                    self.terrain_config.as_ref(),
                    bvh_opt,
                );
            }

            self.level_dirty_counts = self.clip_map.upload_dirty_toroidal(ctx.queue, edit_count);
            log::info!("SDF edit_dirty: dirty_counts={:?}", self.level_dirty_counts,);

            let mut clip_params = self.clip_map.build_clip_params();
            clip_params.debug_flags = if self.debug_mode { 1 } else { 0 };
            ctx.queue.write_buffer(
                self.clip_map.clip_params_buffer.as_ref().unwrap(),
                0,
                bytemuck::bytes_of(&clip_params),
            );
            let all_levels_mask = (1u32 << self.clip_map.level_count()) - 1;
            self.clip_map.update_cached_indices(all_levels_mask);
            ctx.queue.write_buffer(
                &self.all_brick_indices_buffer,
                0,
                bytemuck::cast_slice(self.clip_map.cached_all_indices()),
            );
        } else {
            // Nothing changed — clear dirty counts
            for c in self.level_dirty_counts.iter_mut() {
                *c = 0;
            }
        }

        // Consume incremental edit
        self.incremental_edit = None;

        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if !self.enabled {
            return Ok(());
        }

        // ── Compute pass: evaluate dirty bricks across all clip levels ──────
        let any_dirty = self.level_dirty_counts.iter().any(|c| *c > 0);
        if any_dirty {
            let mut cpass = ctx.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SDF Evaluate"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.eval_pipeline);

            for (i, bg) in self.level_bind_groups.iter().enumerate() {
                let count = self.level_dirty_counts.get(i).copied().unwrap_or(0);
                if count > 0 {
                    cpass.set_bind_group(0, bg, &[]);
                    cpass.dispatch_workgroups(count, 1, 1);
                }
            }
        }

        // ── Rebuild march bind group if engine camera buffer changed ────────
        let camera_ptr = ctx.scene.camera as *const _ as usize;
        if self.march_bg_camera_key != camera_ptr || self.march_bind_group.is_none() {
            self.march_bind_group = Some(Self::build_march_bind_group(
                ctx.device,
                &self.march_bgl,
                &self.clip_map,
                &self.all_brick_indices_buffer,
                ctx.scene.camera,
            ));
            self.march_bg_camera_key = camera_ptr;
        }

        // ── Render pass: fullscreen ray march ───────────────────────────────
        {
            let depth_view = ctx.frame.full_res_depth.unwrap_or(ctx.depth);
            // Choose load op based on preserve_framebuffer setting:
            // - Clear: standalone SDF-only rendering (clears to sky-blue)
            // - Load: deferred pipeline integration (preserves existing color/depth)
            let color_load_op = if self.preserve_framebuffer {
                wgpu::LoadOp::Load
            } else {
                wgpu::LoadOp::Clear(wgpu::Color {
                    r: 0.53,
                    g: 0.72,
                    b: 0.90,
                    a: 1.0,
                }) // sky-blue
            };
            let depth_load_op = if self.preserve_framebuffer {
                wgpu::LoadOp::Load
            } else {
                wgpu::LoadOp::Clear(1.0)
            };

            let color_attachment = wgpu::RenderPassColorAttachment {
                view: ctx.target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: color_load_op,
                    store: wgpu::StoreOp::Store,
                },
            };
            let desc = wgpu::RenderPassDescriptor {
                label: Some("SDF Ray March"),
                color_attachments: &[Some(color_attachment)],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: depth_load_op,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            };

            let mut rpass = ctx.begin_render_pass(&desc);
            rpass.set_pipeline(&self.march_pipeline);
            rpass.set_bind_group(0, self.march_bind_group.as_ref().unwrap(), &[]);
            rpass.draw(0..3, 0..1); // fullscreen triangle
        }

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU SDF helpers (for picking)
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a CPU-side surface pick.
#[derive(Clone, Debug)]
pub struct PickResult {
    pub position: glam::Vec3,
    pub normal: glam::Vec3,
    pub distance: f32,
}

fn cpu_evaluate_sdf(
    pos: glam::Vec3,
    edits: &[SdfEdit],
    inv_transforms: &[glam::Mat4],
    terrain: Option<&TerrainConfig>,
) -> f32 {
    let mut dist = match terrain {
        Some(cfg) => noise::terrain_sdf(pos, cfg),
        None => 1e10,
    };

    for (edit, inv) in edits.iter().zip(inv_transforms.iter()) {
        let local_pos = (*inv * glam::Vec4::new(pos.x, pos.y, pos.z, 1.0)).truncate();
        let d = cpu_evaluate_shape(local_pos, edit);
        dist = cpu_apply_boolean(dist, d, edit.op, edit.blend_radius);
    }

    dist
}

fn cpu_evaluate_shape(p: glam::Vec3, edit: &SdfEdit) -> f32 {
    match edit.shape {
        SdfShapeType::Sphere => p.length() - edit.params.param0,
        SdfShapeType::Cube => {
            let half = glam::Vec3::new(edit.params.param0, edit.params.param1, edit.params.param2);
            let d = p.abs() - half;
            d.max(glam::Vec3::ZERO).length() + d.x.max(d.y.max(d.z)).min(0.0)
        }
        SdfShapeType::Capsule => {
            let r = edit.params.param0;
            let hh = edit.params.param1;
            let mut q = p;
            q.y -= q.y.clamp(-hh, hh);
            q.length() - r
        }
        SdfShapeType::Torus => {
            let maj = edit.params.param0;
            let min = edit.params.param1;
            let q = glam::Vec2::new(glam::Vec2::new(p.x, p.z).length() - maj, p.y);
            q.length() - min
        }
        SdfShapeType::Cylinder => {
            let r = edit.params.param0;
            let hh = edit.params.param1;
            let d = glam::Vec2::new(glam::Vec2::new(p.x, p.z).length(), p.y).abs()
                - glam::Vec2::new(r, hh);
            d.x.max(d.y).min(0.0) + d.max(glam::Vec2::ZERO).length()
        }
    }
}

fn cpu_apply_boolean(d1: f32, d2: f32, op: BooleanOp, k: f32) -> f32 {
    let blend = k > 0.001;
    match op {
        BooleanOp::Union => {
            if blend {
                let h = (0.5 + 0.5 * (d2 - d1) / k).clamp(0.0, 1.0);
                d1 * h + d2 * (1.0 - h) - k * h * (1.0 - h)
            } else {
                d1.min(d2)
            }
        }
        BooleanOp::Subtraction => {
            if blend {
                let h = (0.5 - 0.5 * (d2 + d1) / k).clamp(0.0, 1.0);
                d1 * (1.0 - h) + (-d2) * h + k * h * (1.0 - h)
            } else {
                d1.max(-d2)
            }
        }
        BooleanOp::Intersection => {
            if blend {
                let h = (0.5 - 0.5 * (d2 - d1) / k).clamp(0.0, 1.0);
                d1 * h + d2 * (1.0 - h) + k * h * (1.0 - h)
            } else {
                d1.max(d2)
            }
        }
    }
}

fn cpu_estimate_normal(
    p: glam::Vec3,
    edits: &[SdfEdit],
    inv_transforms: &[glam::Mat4],
    terrain: Option<&TerrainConfig>,
) -> glam::Vec3 {
    let eps = 0.01;
    let dx = cpu_evaluate_sdf(p + glam::Vec3::X * eps, edits, inv_transforms, terrain)
        - cpu_evaluate_sdf(p - glam::Vec3::X * eps, edits, inv_transforms, terrain);
    let dy = cpu_evaluate_sdf(p + glam::Vec3::Y * eps, edits, inv_transforms, terrain)
        - cpu_evaluate_sdf(p - glam::Vec3::Y * eps, edits, inv_transforms, terrain);
    let dz = cpu_evaluate_sdf(p + glam::Vec3::Z * eps, edits, inv_transforms, terrain)
        - cpu_evaluate_sdf(p - glam::Vec3::Z * eps, edits, inv_transforms, terrain);
    glam::Vec3::new(dx, dy, dz).normalize_or_zero()
}
