pub mod edit_list;
pub mod gpu_bvh;
pub mod noise;
pub mod primitives;
pub mod rendering;
pub mod terrain;
pub mod uniforms;

pub use edit_list::{BooleanOp, GpuSdfEdit, SdfEdit, SdfEditList};
pub use primitives::{SdfShapeParams, SdfShapeType};
pub use terrain::{GpuTerrainParams, TerrainConfig, TerrainStyle};
pub use uniforms::SdfGridParams;

// ═══════════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════════

pub(crate) const INITIAL_EDIT_CAPACITY: usize = 1024;
pub(crate) const INITIAL_BVH_CAPACITY: usize = 2048;
pub(crate) const MAX_BRICKS_PER_LEVEL: u32 = 4096;

// ═══════════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
pub struct PickResult {
    pub position: glam::Vec3,
    pub normal: glam::Vec3,
    pub distance: f32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// SdfPass — fully GPU-native SDF render pass
// ═══════════════════════════════════════════════════════════════════════════════

pub struct SdfPass {
    pub(crate) scroll_pipeline:   wgpu::ComputePipeline,
    pub(crate) classify_pipeline: wgpu::ComputePipeline,
    pub(crate) eval_pipeline:     wgpu::ComputePipeline,
    pub(crate) march_pipeline:    wgpu::RenderPipeline,
    pub(crate) march_bgl:         wgpu::BindGroupLayout,
    pub(crate) edit_buffer:                   wgpu::Buffer,
    pub(crate) terrain_params_buffer:         wgpu::Buffer,
    pub(crate) bvh_nodes_buffer:              wgpu::Buffer,
    pub(crate) scroll_state_buffer:           wgpu::Buffer,
    pub(crate) dirty_flags_buffer:            wgpu::Buffer,
    pub(crate) clip_config_buffer:            wgpu::Buffer,
    pub(crate) per_brick_hashes_buffer:       wgpu::Buffer,
    pub(crate) per_brick_edit_lists_buffer:   wgpu::Buffer,
    pub(crate) all_brick_indices_buffer:      wgpu::Buffer,
    pub(crate) dirty_bricks_buffer:           wgpu::Buffer,
    pub(crate) eval_indirect_buffer:          wgpu::Buffer,
    pub(crate) eval_indirect_template_buffer: wgpu::Buffer,
    pub(crate) atlas_buffers:                 Vec<wgpu::Buffer>,
    pub(crate) level_params_buffers:          Vec<wgpu::Buffer>,
    pub(crate) scroll_bg:           Option<wgpu::BindGroup>,
    pub(crate) scroll_bg_camera_key: usize,
    pub(crate) classify_bg:         wgpu::BindGroup,
    pub(crate) eval_bgs:            Vec<wgpu::BindGroup>,
    pub(crate) march_bg:            Option<wgpu::BindGroup>,
    pub(crate) march_bg_camera_key: usize,
    pub(crate) edit_list:        SdfEditList,
    pub(crate) terrain_config:   Option<TerrainConfig>,
    pub(crate) last_gen:         u64,
    pub(crate) edit_generation:  u32,
    pub(crate) bindings_dirty:   bool,
    pub(crate) debug_mode:       bool,
    pub(crate) enabled:          bool,
    pub(crate) preserve_framebuffer: bool,
    pub(crate) level_count:       u32,
    pub(crate) bricks_per_level:  u32,
    pub(crate) brick_grid_dim:    u32,
    pub(crate) brick_size:        u32,
    pub(crate) grid_dim:          u32,
    pub(crate) base_voxel_size:   f32,
    pub(crate) padded_brick_voxels: u32,
    pub(crate) volume_min:        [f32; 3],
    pub(crate) volume_max:        [f32; 3],
    pub(crate) surface_format:    wgpu::TextureFormat,
    pub(crate) cached_snap_origins: [[i32; 3]; 8],
    pub(crate) gpu_passes_clean:    bool,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════════

impl SdfPass {
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
        self.last_gen = u64::MAX;
    }

    pub fn terrain_config(&self) -> Option<&TerrainConfig> {
        self.terrain_config.as_ref()
    }

    pub fn toggle_debug(&mut self) {
        self.debug_mode = !self.debug_mode;
        self.last_gen = u64::MAX;
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

    pub(crate) fn sdf_edit_bounds(edit: &SdfEdit) -> (glam::Vec3, f32) {
        use SdfShapeType::*;
        let local_radius = match edit.shape {
            Sphere   => edit.params.param0,
            Cube     => glam::Vec3::new(edit.params.param0, edit.params.param1, edit.params.param2).length(),
            Capsule  => edit.params.param0 + edit.params.param1,
            Torus    => edit.params.param0 + edit.params.param1,
            Cylinder => (edit.params.param0 * edit.params.param0
                         + edit.params.param1 * edit.params.param1).sqrt(),
        };
        let center = edit.transform.transform_point3(glam::Vec3::ZERO);
        let col0_len = edit.transform.col(0).truncate().length();
        let col1_len = edit.transform.col(1).truncate().length();
        let col2_len = edit.transform.col(2).truncate().length();
        let max_scale = col0_len.max(col1_len).max(col2_len);
        (center, local_radius * max_scale + edit.blend_radius)
    }

    pub(crate) fn voxel_size_for_level(&self, level: u32) -> f32 {
        self.base_voxel_size * (1u32 << level) as f32
    }

    pub(crate) fn build_clip_config(
        &self,
        edit_count: u32,
        bvh_node_count: u32,
        terrain: Option<&TerrainConfig>,
    ) -> crate::rendering::GpuClipConfig {
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
        crate::rendering::GpuClipConfig {
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

    pub(crate) fn build_level_params(&self, level: u32, edit_count: u32) -> SdfGridParams {
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
}

// ═══════════════════════════════════════════════════════════════════════════════
// CPU SDF helpers (for pick_surface only)
// ═══════════════════════════════════════════════════════════════════════════════

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
