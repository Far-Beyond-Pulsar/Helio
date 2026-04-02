//! GPU-native SDF render pass.
//!
//! Three compute passes run entirely on the GPU, followed by a fullscreen
//! ray-march render pass.  CPU cost per frame at steady state is O(1):
//! a single 96-byte write to reset the indirect-dispatch counters.
//!
//! # Pass order (per frame)
//! 1. **cs_scroll**    (1 WG × 8 threads) — compare camera position to stored
//!    snap origins; set per-level `dirty_flags` when the camera has moved or
//!    the edit generation changed.
//! 2. **cs_classify**  (fixed 512 WGs)   — GPU BVH traversal per brick × level;
//!    builds per-brick edit lists; atomically fills the indirect-dispatch buffer
//!    and dirty-brick list.
//! 3. **cs_evaluate**  (indirect per level) — evaluates dirty bricks and writes
//!    quantised SDF distances into the per-level atlas.
//! 4. **Ray march**    (fullscreen triangle) — sphere-traces through the clip-map.
//!
//! Edit changes O(n_edits) — BVH rebuild + edit upload.
//! Camera movement O(1)   — GPU detects scroll and classifies only dirty bricks.

pub mod edit_list;
pub mod gpu_bvh;
pub mod noise;
pub mod primitives;
pub mod terrain;
pub mod uniforms;

pub use edit_list::{BooleanOp, GpuSdfEdit, SdfEdit, SdfEditList};
pub use primitives::{SdfShapeParams, SdfShapeType};
pub use terrain::{GpuTerrainParams, TerrainConfig, TerrainStyle};
pub use uniforms::SdfGridParams;

use gpu_bvh::build_flat_bvh;
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

// ── Constants ──────────────────────────────────────────────────────────────────

/// Default grid resolution (voxels per axis per clip level).
const DEFAULT_GRID_DIM: u32 = 128;

/// Number of clip-map levels (finest → coarsest).
const DEFAULT_CLIP_LEVELS: u32 = 8;

/// Voxels per brick axis.  Atlas mapping: atlas_id = brick_flat.
const DEFAULT_BRICK_SIZE: u32 = 8;

/// Initial GPU edit-buffer capacity.
const INITIAL_EDIT_CAPACITY: usize = 1024;

/// Initial GPU BVH-buffer capacity (nodes).
const INITIAL_BVH_CAPACITY: usize = 2048;

/// Dirty-brick list stride per level — MUST match `MAX_BRICKS_PER_LEVEL` in WGSL.
const MAX_BRICKS_PER_LEVEL: u32 = 4096;

/// Per-brick edit list stride (1 count + 64 indices) — MUST match WGSL.
#[allow(dead_code)]
const EDIT_LIST_STRIDE: u32 = 65;

// ── GPU structs (CPU-side mirrors, must match WGSL exactly) ────────────────────

/// Static clip configuration uploaded once (and whenever edits/terrain change).
/// Matches `ClipConfig` in `sdf_scroll.wgsl` and `sdf_classify.wgsl` (96 bytes).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuClipConfig {
    level_count:           u32,
    grid_dim:              u32,
    brick_size:            u32,
    brick_grid_dim:        u32,      //  = grid_dim / brick_size
    bricks_per_level:      u32,      //  = brick_grid_dim^3
    atlas_bricks_per_axis: u32,      //  = brick_grid_dim  (direct mapping)
    base_voxel_size:       f32,
    edit_count:            u32,
    bvh_node_count:        u32,
    terrain_enabled:       u32,
    terrain_y_min:         f32,
    terrain_y_max:         f32,
    _pad0:                 u32,
    _pad1:                 u32,
    _pad2:                 u32,
    _pad3:                 u32,
    /// Voxel sizes for levels 0-3.
    voxel_sizes_lo:        [f32; 4],
    /// Voxel sizes for levels 4-7.
    voxel_sizes_hi:        [f32; 4],
}

const _: () = assert!(
    std::mem::size_of::<GpuClipConfig>() == 96,
    "GpuClipConfig must be 96 bytes",
);

/// Persistent GPU scroll state (read-write by the scroll shader).
/// Matches `ScrollState` in both scroll and classify shaders (144 bytes).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuScrollState {
    /// Snapped brick-coord origins per level (xyz + w=0 padding).
    snap_origins:  [[i32; 4]; 8],    //  128 bytes
    /// Monotonically-increasing edit generation written by the CPU.
    edit_gen:      u32,
    /// Last edit generation seen by the GPU (GPU writes this).
    prev_edit_gen: u32,
    _pad0:         u32,
    _pad1:         u32,
}

const _: () = assert!(
    std::mem::size_of::<GpuScrollState>() == 144,
    "GpuScrollState must be 144 bytes",
);

// ── SDF pass ──────────────────────────────────────────────────────────────────

/// CPU-side picking result.
#[derive(Clone, Debug)]
pub struct PickResult {
    pub position: glam::Vec3,
    pub normal: glam::Vec3,
    pub distance: f32,
}

/// Fully GPU-native SDF render pass.
pub struct SdfPass {
    // ── Pipelines ──────────────────────────────────────────────────────────
    scroll_pipeline:  wgpu::ComputePipeline,
    classify_pipeline: wgpu::ComputePipeline,
    eval_pipeline:    wgpu::ComputePipeline,
    march_pipeline:   wgpu::RenderPipeline,
    march_bgl:        wgpu::BindGroupLayout,

    // ── Global GPU buffers ─────────────────────────────────────────────────
    /// Packed GpuSdfEdit array — grows when edit count exceeds capacity.
    edit_buffer:                  wgpu::Buffer,
    /// GpuTerrainParams uniform (64 bytes).
    terrain_params_buffer:        wgpu::Buffer,
    /// Flat GPU BVH nodes — grows when needed.
    bvh_nodes_buffer:             wgpu::Buffer,
    /// GpuScrollState (144 bytes, GPU rw).
    scroll_state_buffer:          wgpu::Buffer,
    /// Per-level dirty flags written by scroll, read by classify (32 bytes).
    dirty_flags_buffer:           wgpu::Buffer,
    /// GpuClipConfig uniform (96 bytes).
    clip_config_buffer:           wgpu::Buffer,
    /// Per-brick FNV-1a hash for change detection — `level_count * bricks_per_level` u32s.
    per_brick_hashes_buffer:      wgpu::Buffer,
    /// Per-brick edit lists written by classify — `level_count * bricks_per_level * EDIT_LIST_STRIDE` u32s.
    per_brick_edit_lists_buffer:  wgpu::Buffer,
    /// Atlas slot index per (level, brick) — `level_count * bricks_per_level` u32s.
    all_brick_indices_buffer:     wgpu::Buffer,
    /// GPU-built dirty brick list — `level_count * MAX_BRICKS_PER_LEVEL` u32s.
    dirty_bricks_buffer:          wgpu::Buffer,
    /// Indirect dispatch counter + args — `level_count * 3` u32s (96 bytes for 8 levels).
    eval_indirect_buffer:         wgpu::Buffer,
    /// Pre-initialised copy source for eval_indirect reset — [0,1,1]×level_count, never mutated.
    eval_indirect_template_buffer: wgpu::Buffer,

    // ── Per-level GPU buffers ──────────────────────────────────────────────
    /// Packed u8 SDF atlas per level (stored as u32 words).
    atlas_buffers:                Vec<wgpu::Buffer>,
    /// SdfGridParams uniform per level (96 bytes each).
    level_params_buffers:         Vec<wgpu::Buffer>,

    // ── Bind groups ────────────────────────────────────────────────────────
    scroll_bg:           Option<wgpu::BindGroup>,
    scroll_bg_camera_key: usize,
    classify_bg:         wgpu::BindGroup,
    eval_bgs:            Vec<wgpu::BindGroup>,
    march_bg:            Option<wgpu::BindGroup>,
    march_bg_camera_key: usize,

    // ── Minimal CPU state ──────────────────────────────────────────────────
    edit_list:        SdfEditList,
    terrain_config:   Option<TerrainConfig>,
    last_gen:         u64,
    /// Written to `scroll_state.edit_gen` whenever the edit list changes.
    edit_generation:  u32,
    /// Set when edit_buffer or bvh_nodes_buffer is reallocated.
    bindings_dirty:   bool,
    debug_mode:       bool,
    enabled:          bool,
    /// If true, loads existing color/depth (deferred pipeline integration).
    preserve_framebuffer: bool,

    // ── Static geometry constants ──────────────────────────────────────────
    level_count:       u32,
    bricks_per_level:  u32,
    brick_grid_dim:    u32,
    brick_size:        u32,
    grid_dim:          u32,
    base_voxel_size:   f32,
    /// Atlas bytes per brick = `(brick_size+1)^3`.
    #[allow(dead_code)]
    padded_brick_voxels: u32,
    volume_min:        [f32; 3],
    volume_max:        [f32; 3],
    #[allow(dead_code)]
    surface_format:    wgpu::TextureFormat,
}

impl SdfPass {
    // ── Constructors ────────────────────────────────────────────────────────

    /// Creates the pass with a ±50 unit world volume, 128-voxel grid, 8 levels.
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        terrain: Option<TerrainConfig>,
    ) -> Self {
        Self::with_grid(device, surface_format, DEFAULT_GRID_DIM, [-50.0; 3], [50.0; 3], terrain)
    }

    /// Creates the pass with an explicit world volume and grid resolution.
    pub fn with_grid(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        grid_dim: u32,
        volume_min: [f32; 3],
        volume_max: [f32; 3],
        terrain: Option<TerrainConfig>,
    ) -> Self {
        let level_count   = DEFAULT_CLIP_LEVELS;
        let brick_size    = DEFAULT_BRICK_SIZE;
        let brick_grid_dim = grid_dim / brick_size;
        let bricks_per_level = brick_grid_dim * brick_grid_dim * brick_grid_dim;

        let range = volume_max[0] - volume_min[0];
        let base_voxel_size = range / grid_dim as f32;
        let padded_brick_voxels = (brick_size + 1) * (brick_size + 1) * (brick_size + 1);

        // ── Atlas buffers (one per level) ─────────────────────────────────
        // Atlas is a flat array of u32 words; each word holds 4 packed u8 SDF values.
        // Direct mapping: atlas_id = brick_flat, so atlas size = bricks_per_level * padded voxels.
        let atlas_word_count = (bricks_per_level * padded_brick_voxels + 3) / 4;
        let atlas_buffers: Vec<wgpu::Buffer> = (0..level_count as usize)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("SDF Atlas L{i}")),
                    size: (atlas_word_count * 4) as u64,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            })
            .collect();

        // ── SdfGridParams buffers (per level) ─────────────────────────────
        let level_params_buffers: Vec<wgpu::Buffer> = (0..level_count as usize)
            .map(|i| {
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("SDF Level Params L{i}")),
                    size: std::mem::size_of::<SdfGridParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                buf
            })
            .collect();

        // ── Edit buffer ───────────────────────────────────────────────────
        let edit_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Edit Buffer"),
            size: (INITIAL_EDIT_CAPACITY * std::mem::size_of::<GpuSdfEdit>()).max(64) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Terrain params ────────────────────────────────────────────────
        let terrain_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Terrain Params"),
            size: std::mem::size_of::<GpuTerrainParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── BVH nodes buffer ──────────────────────────────────────────────
        let bvh_nodes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF BVH Nodes"),
            size: (INITIAL_BVH_CAPACITY * std::mem::size_of::<gpu_bvh::GpuBvhNode>()).max(64) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Scroll state ──────────────────────────────────────────────────
        let scroll_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Scroll State"),
            size: std::mem::size_of::<GpuScrollState>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Dirty flags ───────────────────────────────────────────────────
        let dirty_flags_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Dirty Flags"),
            size: (level_count * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Clip config (for scroll + classify) ───────────────────────────
        let clip_config_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Clip Config"),
            size: std::mem::size_of::<GpuClipConfig>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Per-brick hashes ──────────────────────────────────────────────
        let per_brick_hashes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Per-Brick Hashes"),
            size: (level_count * bricks_per_level * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Per-brick edit lists ──────────────────────────────────────────
        let per_brick_edit_lists_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Per-Brick Edit Lists"),
            size: (level_count * bricks_per_level * EDIT_LIST_STRIDE * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── All-brick indices ─────────────────────────────────────────────
        let all_brick_indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF All Brick Indices"),
            size: (level_count * bricks_per_level * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Dirty bricks ──────────────────────────────────────────────────
        let dirty_bricks_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Dirty Bricks"),
            size: (level_count * MAX_BRICKS_PER_LEVEL * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Eval indirect ─────────────────────────────────────────────────
        // Layout: [x_dirty_count, y=1, z=1] × level_count.
        // `x` is written by classify via atomicAdd and used as the indirect dispatch x.
        // The template buffer holds [0,1,1]×level_count and never changes; each frame
        // execute() copies it to eval_indirect_buffer via GPU DMA — zero CPU work.
        let indirect_byte_size = (level_count * 3 * 4) as u64;
        let eval_indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Eval Indirect"),
            size: indirect_byte_size,
            usage: wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let eval_indirect_template_buffer = {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("SDF Eval Indirect Template"),
                size: indirect_byte_size,
                usage: wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            });
            let template_data: Vec<u32> = (0..level_count).flat_map(|_| [0u32, 1, 1]).collect();
            buf.slice(..).get_mapped_range_mut().copy_from_slice(bytemuck::cast_slice(&template_data));
            buf.unmap();
            buf
        };

        // ── Compute pipelines ─────────────────────────────────────────────
        let scroll_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Scroll"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/sdf_scroll.wgsl").into(),
            ),
        });
        let scroll_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SDF Scroll Pipeline"),
            layout: None,
            module: &scroll_shader,
            entry_point: Some("cs_scroll"),
            compilation_options: Default::default(),
            cache: None,
        });

        let classify_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Classify"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/sdf_classify.wgsl").into(),
            ),
        });
        let classify_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SDF Classify Pipeline"),
            layout: None,
            module: &classify_shader,
            entry_point: Some("cs_classify"),
            compilation_options: Default::default(),
            cache: None,
        });

        let eval_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Evaluate"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/sdf_evaluate.wgsl").into(),
            ),
        });
        let eval_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SDF Evaluate Pipeline"),
            layout: None,
            module: &eval_shader,
            entry_point: Some("cs_evaluate_sparse"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ── Classify bind group ───────────────────────────────────────────
        let classify_bgl = classify_pipeline.get_bind_group_layout(0);
        let classify_bg = Self::build_classify_bg_impl(
            device,
            &classify_bgl,
            &clip_config_buffer,
            &scroll_state_buffer,
            &dirty_flags_buffer,
            &bvh_nodes_buffer,
            &per_brick_hashes_buffer,
            &per_brick_edit_lists_buffer,
            &all_brick_indices_buffer,
            &dirty_bricks_buffer,
            &eval_indirect_buffer,
        );

        // ── Evaluate bind groups (one per level) ──────────────────────────
        let eval_bgl = eval_pipeline.get_bind_group_layout(0);
        let eval_bgs = Self::build_eval_bgs_impl(
            device,
            &eval_bgl,
            &level_params_buffers,
            &edit_buffer,
            &atlas_buffers,
            &dirty_bricks_buffer,
            &per_brick_edit_lists_buffer,
            &terrain_params_buffer,
        );

        // ── Render pipeline (ray march) ───────────────────────────────────
        let march_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Ray March"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/sdf_ray_march.wgsl").into(),
            ),
        });

        let march_bgl = Self::build_march_bgl(device, level_count as usize);
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
            scroll_pipeline,
            classify_pipeline,
            eval_pipeline,
            march_pipeline,
            march_bgl,
            edit_buffer,
            terrain_params_buffer,
            bvh_nodes_buffer,
            scroll_state_buffer,
            dirty_flags_buffer,
            clip_config_buffer,
            per_brick_hashes_buffer,
            per_brick_edit_lists_buffer,
            all_brick_indices_buffer,
            dirty_bricks_buffer,
            eval_indirect_buffer,
            eval_indirect_template_buffer,
            atlas_buffers,
            level_params_buffers,
            scroll_bg: None,
            scroll_bg_camera_key: 0,
            classify_bg,
            eval_bgs,
            march_bg: None,
            march_bg_camera_key: 0,
            edit_list: SdfEditList::new(),
            terrain_config: terrain,
            last_gen: u64::MAX, // force initial upload
            edit_generation: 0,
            bindings_dirty: false,
            debug_mode: false,
            enabled: true,
            preserve_framebuffer: false,
            level_count,
            bricks_per_level,
            brick_grid_dim,
            brick_size,
            grid_dim,
            base_voxel_size,
            padded_brick_voxels,
            volume_min,
            volume_max,
            surface_format,
        }
    }

    // ── Public API ──────────────────────────────────────────────────────────

    pub fn add_edit(&mut self, edit: SdfEdit) {
        self.edit_list.add(edit);
    }

    pub fn remove_edit(&mut self, index: usize) {
        self.edit_list.remove(index);
    }

    pub fn set_edit(&mut self, index: usize, edit: SdfEdit) {
        self.edit_list.set(index, edit);
    }

    pub fn clear_edits(&mut self) {
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
        self.last_gen = u64::MAX; // force re-upload
    }

    pub fn terrain_config(&self) -> Option<&TerrainConfig> {
        self.terrain_config.as_ref()
    }

    pub fn toggle_debug(&mut self) {
        self.debug_mode = !self.debug_mode;
        self.last_gen = u64::MAX; // force params re-upload
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn set_preserve_framebuffer(&mut self, preserve: bool) {
        self.preserve_framebuffer = preserve;
    }

    pub fn preserve_framebuffer(&self) -> bool {
        self.preserve_framebuffer
    }

    // ── CPU-side surface picking ────────────────────────────────────────────

    /// CPU sphere-trace for surface picking. Not performance-critical.
    pub fn pick_surface(
        &self,
        ray_origin: glam::Vec3,
        ray_dir: glam::Vec3,
        max_dist: f32,
    ) -> Option<PickResult> {
        let edits = self.edit_list.edits();
        let terrain = self.terrain_config.as_ref();
        let inv_transforms: Vec<glam::Mat4> =
            edits.iter().map(|e| e.transform.inverse()).collect();

        let mut t = 0.0f32;
        for _ in 0..256 {
            let p = ray_origin + ray_dir * t;
            let d = cpu_evaluate_sdf(p, edits, &inv_transforms, terrain);
            if d.abs() < 0.02 {
                let n = cpu_estimate_normal(p, edits, &inv_transforms, terrain);
                return Some(PickResult { position: p, normal: n, distance: t });
            }
            t += d.max(0.01);
            if t > max_dist { break; }
        }
        None
    }

    // ── Internal helpers ────────────────────────────────────────────────────

    /// Conservatively maps a single edit to a world-space `(center, radius)` sphere.
    fn sdf_edit_bounds(edit: &SdfEdit) -> (glam::Vec3, f32) {
        use SdfShapeType::*;
        let local_radius = match edit.shape {
            Sphere   => edit.params.param0,
            Cube     => glam::Vec3::new(edit.params.param0, edit.params.param1, edit.params.param2).length(),
            Capsule  => edit.params.param0 + edit.params.param1,
            Torus    => edit.params.param0 + edit.params.param1,
            Cylinder => (edit.params.param0 * edit.params.param0
                         + edit.params.param1 * edit.params.param1).sqrt(),
        };
        // World-space center via forward transform of origin.
        let center = edit.transform.transform_point3(glam::Vec3::ZERO);
        // Conservative radius: scale sphere by the longest column of the transform.
        let col0_len = edit.transform.col(0).truncate().length();
        let col1_len = edit.transform.col(1).truncate().length();
        let col2_len = edit.transform.col(2).truncate().length();
        let max_scale = col0_len.max(col1_len).max(col2_len);
        (center, local_radius * max_scale + edit.blend_radius)
    }

    /// Level voxel size for level `i`.
    fn voxel_size_for_level(&self, level: u32) -> f32 {
        self.base_voxel_size * (1u32 << level) as f32
    }

    /// Build the static clip config for scroll + classify shaders.
    fn build_clip_config(
        &self,
        edit_count: u32,
        bvh_node_count: u32,
        terrain: Option<&TerrainConfig>,
    ) -> GpuClipConfig {
        let mut voxel_sizes_lo = [0.0f32; 4];
        let mut voxel_sizes_hi = [0.0f32; 4];
        for i in 0..4usize {
            voxel_sizes_lo[i] = self.voxel_size_for_level(i as u32);
            voxel_sizes_hi[i] = self.voxel_size_for_level((4 + i) as u32);
        }

        let (terrain_enabled, terrain_y_min, terrain_y_max) = match terrain {
            Some(cfg) => (1u32, cfg.height - cfg.amplitude * 3.0, cfg.height + cfg.amplitude * 2.0),
            None => (0u32, -1e10, 1e10),
        };

        GpuClipConfig {
            level_count: self.level_count,
            grid_dim: self.grid_dim,
            brick_size: self.brick_size,
            brick_grid_dim: self.brick_grid_dim,
            bricks_per_level: self.bricks_per_level,
            atlas_bricks_per_axis: self.brick_grid_dim,
            base_voxel_size: self.base_voxel_size,
            edit_count,
            bvh_node_count,
            terrain_enabled,
            terrain_y_min,
            terrain_y_max,
            _pad0: 0, _pad1: 0, _pad2: 0, _pad3: 0,
            voxel_sizes_lo,
            voxel_sizes_hi,
        }
    }

    /// Build per-level SdfGridParams.
    fn build_level_params(&self, level: u32, edit_count: u32) -> SdfGridParams {
        let vs = self.voxel_size_for_level(level);
        let max_march_dist = self.grid_dim as f32 * vs * 2.0;
        SdfGridParams {
            volume_min:            self.volume_min,
            _pad0:                 0.0,
            volume_max:            self.volume_max,
            _pad1:                 0.0,
            grid_dim:              self.grid_dim,
            edit_count,
            voxel_size:            vs,
            max_march_dist,
            brick_size:            self.brick_size,
            brick_grid_dim:        self.brick_grid_dim,
            level_idx:             level,
            atlas_bricks_per_axis: self.brick_grid_dim,
            grid_origin:           [0.0; 3],
            debug_flags:           if self.debug_mode { 1 } else { 0 },
            bricks_per_level:      self.bricks_per_level,
            _pad2: 0, _pad3: 0, _pad4: 0,
        }
    }

    // ── Bind group builders ─────────────────────────────────────────────────

    fn build_scroll_bg(
        device: &wgpu::Device,
        scroll_bgl: &wgpu::BindGroupLayout,
        camera_buf: &wgpu::Buffer,
        clip_config_buf: &wgpu::Buffer,
        scroll_state_buf: &wgpu::Buffer,
        dirty_flags_buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SDF Scroll BG"),
            layout: scroll_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: clip_config_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: scroll_state_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: dirty_flags_buf.as_entire_binding() },
            ],
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn build_classify_bg_impl(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        clip_config_buf:            &wgpu::Buffer,
        scroll_state_buf:           &wgpu::Buffer,
        dirty_flags_buf:            &wgpu::Buffer,
        bvh_nodes_buf:              &wgpu::Buffer,
        per_brick_hashes_buf:       &wgpu::Buffer,
        per_brick_edit_lists_buf:   &wgpu::Buffer,
        all_brick_indices_buf:      &wgpu::Buffer,
        dirty_bricks_buf:           &wgpu::Buffer,
        eval_indirect_buf:          &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SDF Classify BG"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: clip_config_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: scroll_state_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dirty_flags_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: bvh_nodes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: per_brick_hashes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: per_brick_edit_lists_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: all_brick_indices_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: dirty_bricks_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: eval_indirect_buf.as_entire_binding() },
            ],
        })
    }

    fn build_eval_bgs_impl(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        level_params_bufs: &[wgpu::Buffer],
        edit_buf:           &wgpu::Buffer,
        atlas_bufs:         &[wgpu::Buffer],
        dirty_bricks_buf:   &wgpu::Buffer,
        per_brick_edit_lists_buf: &wgpu::Buffer,
        terrain_params_buf: &wgpu::Buffer,
    ) -> Vec<wgpu::BindGroup> {
        level_params_bufs.iter().enumerate().zip(atlas_bufs.iter())
            .map(|((i, params_buf), atlas_buf)| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("SDF Eval BG L{i}")),
                    layout: bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: edit_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: atlas_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: dirty_bricks_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: per_brick_edit_lists_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: terrain_params_buf.as_entire_binding() },
                    ],
                })
            })
            .collect()
    }

    fn build_march_bgl(device: &wgpu::Device, level_count: usize) -> wgpu::BindGroupLayout {
        let mut entries = vec![
            // b0: camera uniform (vertex + frag)
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
            // b1: GpuClipConfig uniform (frag) — static, only changes with edits
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
            // b2: GpuScrollState storage (frag, read-only) — GPU-written each frame by scroll pass
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
        ];
        // b3..b(3+level_count-1): per-level atlas storage (frag, read-only)
        for i in 0..level_count {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: (3 + i) as u32,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        // b(3+level_count): all_brick_indices storage (frag, read-only)
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: (3 + level_count) as u32,
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

    fn build_march_bg(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        camera_buf: &wgpu::Buffer,
        clip_config_buf: &wgpu::Buffer,
        scroll_state_buf: &wgpu::Buffer,
        atlas_bufs: &[wgpu::Buffer],
        all_brick_indices_buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let level_count = atlas_bufs.len();
        let mut entries = vec![
            wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: clip_config_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: scroll_state_buf.as_entire_binding() },
        ];
        for (i, atlas) in atlas_bufs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: (3 + i) as u32,
                resource: atlas.as_entire_binding(),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: (3 + level_count) as u32,
            resource: all_brick_indices_buf.as_entire_binding(),
        });
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SDF Ray March BG"),
            layout: bgl,
            entries: &entries,
        })
    }
}

// ── RenderPass implementation ─────────────────────────────────────────────────

impl RenderPass for SdfPass {
    fn name(&self) -> &'static str {
        "SDF"
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        if !self.enabled {
            return Ok(());
        }

        let gen = self.edit_list.generation();
        let needs_upload = gen != self.last_gen;

        // ── Upload edits + BVH + terrain on change ────────────────────────
        if needs_upload {
            // flush_gpu_data() must come before edits() to avoid a simultaneous
            // mutable + immutable borrow of self.edit_list.
            let gpu_edits = self.edit_list.flush_gpu_data();
            let edit_count = gpu_edits.len() as u32;

            // Grow edit_buffer if needed.
            let required = (gpu_edits.len() * std::mem::size_of::<GpuSdfEdit>()).max(64) as u64;
            if required > self.edit_buffer.size() {
                let new_size = (required * 2).max(64);
                self.edit_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("SDF Edit Buffer"),
                    size: new_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.bindings_dirty = true;
            }
            if !gpu_edits.is_empty() {
                ctx.queue.write_buffer(
                    &self.edit_buffer,
                    0,
                    bytemuck::cast_slice(&gpu_edits),
                );
            }

            // Build + upload flat GPU BVH (O(n_edits), not O(n_bricks)).
            let bounds: Vec<(glam::Vec3, f32)> =
                self.edit_list.edits().iter().map(Self::sdf_edit_bounds).collect();
            let bvh = build_flat_bvh(&bounds);
            let bvh_node_count = bvh.len() as u32;

            let bvh_bytes = bytemuck::cast_slice::<_, u8>(&bvh);
            let bvh_required = bvh_bytes.len().max(64) as u64;
            if bvh_required > self.bvh_nodes_buffer.size() {
                let new_size = (bvh_required * 2).max(64);
                self.bvh_nodes_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("SDF BVH Nodes"),
                    size: new_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.bindings_dirty = true;
            }
            ctx.queue.write_buffer(&self.bvh_nodes_buffer, 0, bvh_bytes);

            // Upload terrain params.
            let terrain_gpu = self.terrain_config.as_ref()
                .map(|c| c.build_gpu_params())
                .unwrap_or_else(GpuTerrainParams::disabled);
            ctx.queue.write_buffer(
                &self.terrain_params_buffer,
                0,
                bytemuck::bytes_of(&terrain_gpu),
            );

            // Upload clip config with updated counts.
            let clip_cfg = self.build_clip_config(
                edit_count,
                bvh_node_count,
                self.terrain_config.as_ref(),
            );
            ctx.queue.write_buffer(
                &self.clip_config_buffer,
                0,
                bytemuck::bytes_of(&clip_cfg),
            );

            // Upload per-level SdfGridParams with updated edit_count + debug_flags.
            for level in 0..self.level_count {
                let params = self.build_level_params(level, edit_count);
                ctx.queue.write_buffer(
                    &self.level_params_buffers[level as usize],
                    0,
                    bytemuck::bytes_of(&params),
                );
            }

            // Bump edit_gen so the scroll shader forces a full re-classify.
            self.edit_generation = self.edit_generation.wrapping_add(1);
            // edit_gen is at byte offset 128: after snap_origins = [[i32;4];8] = 128 bytes.
            let edit_gen_offset: u64 = 128;
            ctx.queue.write_buffer(
                &self.scroll_state_buffer,
                edit_gen_offset,
                bytemuck::bytes_of(&self.edit_generation),
            );

            self.last_gen = gen;
        }

        // ── Rebuild classify/eval bind groups if buffers were reallocated ──
        if self.bindings_dirty {
            let classify_bgl = self.classify_pipeline.get_bind_group_layout(0);
            self.classify_bg = Self::build_classify_bg_impl(
                ctx.device,
                &classify_bgl,
                &self.clip_config_buffer,
                &self.scroll_state_buffer,
                &self.dirty_flags_buffer,
                &self.bvh_nodes_buffer,
                &self.per_brick_hashes_buffer,
                &self.per_brick_edit_lists_buffer,
                &self.all_brick_indices_buffer,
                &self.dirty_bricks_buffer,
                &self.eval_indirect_buffer,
            );
            let eval_bgl = self.eval_pipeline.get_bind_group_layout(0);
            self.eval_bgs = Self::build_eval_bgs_impl(
                ctx.device,
                &eval_bgl,
                &self.level_params_buffers,
                &self.edit_buffer,
                &self.atlas_buffers,
                &self.dirty_bricks_buffer,
                &self.per_brick_edit_lists_buffer,
                &self.terrain_params_buffer,
            );
            // Force rebuild of march BG (edit_buffer/atlas changed).
            self.march_bg = None;
            self.march_bg_camera_key = 0;
            self.bindings_dirty = false;
        }

        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if !self.enabled {
            return Ok(());
        }

        // ── Rebuild camera-dependent bind groups if needed ─────────────────
        let camera_ptr = ctx.scene.camera as *const _ as usize;
        if self.scroll_bg_camera_key != camera_ptr || self.scroll_bg.is_none() {
            let scroll_bgl = self.scroll_pipeline.get_bind_group_layout(0);
            self.scroll_bg = Some(Self::build_scroll_bg(
                ctx.device,
                &scroll_bgl,
                ctx.scene.camera,
                &self.clip_config_buffer,
                &self.scroll_state_buffer,
                &self.dirty_flags_buffer,
            ));
            self.scroll_bg_camera_key = camera_ptr;
        }
        if self.march_bg_camera_key != camera_ptr || self.march_bg.is_none() {
            self.march_bg = Some(Self::build_march_bg(
                ctx.device,
                &self.march_bgl,
                ctx.scene.camera,
                &self.clip_config_buffer,
                &self.scroll_state_buffer,
                &self.atlas_buffers,
                &self.all_brick_indices_buffer,
            ));
            self.march_bg_camera_key = camera_ptr;
        }

        // ── GPU reset (zero CPU work — pure encoder commands) ─────────────
        // Restore indirect dispatch args to [0,1,1]×level_count via GPU DMA.
        // classify will atomically increment element x (dirty count); y,z stay 1.
        ctx.encoder.copy_buffer_to_buffer(
            &self.eval_indirect_template_buffer,
            0,
            &self.eval_indirect_buffer,
            0,
            self.level_count as u64 * 3 * 4,
        );
        ctx.encoder.clear_buffer(&self.dirty_flags_buffer, 0, None);
        ctx.encoder.clear_buffer(&self.dirty_bricks_buffer, 0, None);

        // ── Pass 1: Scroll detection ───────────────────────────────────────
        // One workgroup of 8 threads checks each level's snap origin.
        {
            let mut cpass = ctx.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SDF Scroll"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.scroll_pipeline);
            cpass.set_bind_group(0, self.scroll_bg.as_ref().unwrap(), &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        // ── Pass 2: Classify (GPU BVH traversal per brick × level) ────────
        // Fixed dispatch: 64 WGs × level_count = 512 WGs for default config.
        // Each WG handles 64 bricks (grid_dim/brick_size = 16; 4096/64 = 64 WGs).
        {
            let wgs_x = self.bricks_per_level / 64; // 4096 / 64 = 64
            let mut cpass = ctx.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SDF Classify"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.classify_pipeline);
            cpass.set_bind_group(0, &self.classify_bg, &[]);
            cpass.dispatch_workgroups(wgs_x, self.level_count, 1);
        }

        // ── Pass 3: Evaluate (indirect per level) ─────────────────────────
        // Each level dispatches x WGs where x = number of dirty bricks for that level.
        // x was written atomically by the classify pass.
        {
            let mut cpass = ctx.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SDF Evaluate"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.eval_pipeline);
            for (level, eval_bg) in self.eval_bgs.iter().enumerate() {
                cpass.set_bind_group(0, eval_bg, &[]);
                // Each level's indirect args are at byte offset `level * 3 * 4`.
                cpass.dispatch_workgroups_indirect(
                    &self.eval_indirect_buffer,
                    (level * 3 * 4) as u64,
                );
            }
        }

        // ── Pass 4: Fullscreen ray march ───────────────────────────────────
        {
            let depth_view = ctx.resources.full_res_depth.unwrap_or(ctx.depth);
            let color_load_op = if self.preserve_framebuffer {
                wgpu::LoadOp::Load
            } else {
                wgpu::LoadOp::Clear(wgpu::Color { r: 0.53, g: 0.72, b: 0.90, a: 1.0 })
            };
            let depth_load_op = if self.preserve_framebuffer {
                wgpu::LoadOp::Load
            } else {
                wgpu::LoadOp::Clear(1.0)
            };

            let desc = wgpu::RenderPassDescriptor {
                label: Some("SDF Ray March"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: ctx.target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations { load: color_load_op, store: wgpu::StoreOp::Store },
                })],
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
            rpass.set_bind_group(0, self.march_bg.as_ref().unwrap(), &[]);
            rpass.draw(0..3, 0..1);
        }

        Ok(())
    }
}

// ── CPU SDF helpers (for pick_surface only) ───────────────────────────────────

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
                d1 * (1.0 - h) + d2 * h + k * h * (1.0 - h)
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
    let dx = glam::Vec3::new(eps, 0.0, 0.0);
    let dy = glam::Vec3::new(0.0, eps, 0.0);
    let dz = glam::Vec3::new(0.0, 0.0, eps);
    let nx = cpu_evaluate_sdf(p + dx, edits, inv_transforms, terrain)
           - cpu_evaluate_sdf(p - dx, edits, inv_transforms, terrain);
    let ny = cpu_evaluate_sdf(p + dy, edits, inv_transforms, terrain)
           - cpu_evaluate_sdf(p - dy, edits, inv_transforms, terrain);
    let nz = cpu_evaluate_sdf(p + dz, edits, inv_transforms, terrain)
           - cpu_evaluate_sdf(p - dz, edits, inv_transforms, terrain);
    glam::Vec3::new(nx, ny, nz).normalize_or_zero()
}
