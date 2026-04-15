//! GPU-driven per-face shadow dirty detection.
//!
//! Runs as a compute pass immediately after `ShadowMatrixPass`.  For each movable
//! shadow-caster draw call, it compares the object's current world-space position
//! with the stored previous-frame position.  If the object moved, it sphere-tests
//! the object's bounding sphere against every active shadow-face frustum (planes
//! extracted from the VP matrix via Gribb-Hartmann).  Any intersecting face is
//! marked dirty in a GPU buffer that `ShadowPass` reads directly via
//! `multi_draw_indexed_indirect_count` — no CPU readback and no O(N·M) CPU loop.
//!
//! # Architecture
//!
//! ```text
//! ShadowMatrixPass  ─writes─►  shadow_mats (VP per face)
//!        ↓
//! ShadowDirtyPass   ─reads──►  instances, movable_draws, prev_positions, shadow_mats
//!                   ─writes─►  face_dirty[256]     (0/1, is this face dirty?)
//!                              face_geom_count[256] (0 or movable_draw_count)
//!        ↓
//! ShadowPass        ─reads──►  face_dirty (as clear-draw indirect count)
//!                              face_geom_count (as geometry indirect count)
//! ```
//!
//! # Granularity
//!
//! The dirty check is **per shadow face**, not per caster.  A spinning object on the
//! +X side of a point light does NOT re-render the -X, ±Y, ±Z cube faces.
//!
//! # Topology changes
//!
//! When `shadow_movable_draw_count` changes between frames (objects added/removed),
//! the pass sets `force_dirty_all = 1` in its uniform buffer, causing the shader to
//! dirty every active face and update all prev_positions to the current frame.
//! Subsequent frames return to normal per-object dirty detection.

use bytemuck::{Pod, Zeroable};
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use std::sync::Arc;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Maximum shadow atlas faces.  Must match `MAX_FACES` in the WGSL shader and
/// `MAX_SHADOW_FACES` in `helio-pass-shadow`.
const MAX_SHADOW_FACES: usize = 256;

const WORKGROUP_SIZE: u32 = 64;

// ── Uniforms ──────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShadowDirtyUniforms {
    movable_draw_count: u32,
    face_count: u32,
    /// 1 on the frame when `movable_draw_count` changes — forces all faces dirty.
    force_dirty_all: u32,
    _pad: u32,
}

// ── Pass struct ───────────────────────────────────────────────────────────────

pub struct ShadowDirtyPass {
    pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    bgl: wgpu::BindGroupLayout,

    /// Uniform buffer holding per-frame parameters.
    uniform_buf: wgpu::Buffer,

    /// Previous-frame world-space XYZ positions of each movable draw call's object.
    /// Layout: `array<vec4f>` indexed by draw-call index (NOT instance index).
    /// Sized to `MAX_SHADOW_FACES * 16` bytes; only the first `movable_draw_count`
    /// entries are valid.
    prev_positions_buf: wgpu::Buffer,

    /// Per-face dirty flag: 0 = clean, 1 = dirty (atomic u32 array, 256 entries).
    /// Shared with `ShadowPass` — published via `Arc` so the shadow pass can bind it.
    pub face_dirty_buf: Arc<wgpu::Buffer>,

    /// Per-face geometry draw count (non-atomic u32 array, 256 entries).
    /// ShadowPass uses this as the `count_buffer` argument to
    /// `multi_draw_indexed_indirect_count` for movable geometry draws.
    pub face_geom_count_buf: Arc<wgpu::Buffer>,

    /// Bind group (lazy; rebuilt whenever the `instances` or `shadow_mats` buffer
    /// pointer changes due to `GrowableBuffer` reallocation).
    bind_group: Option<wgpu::BindGroup>,
    bind_group_key: Option<(usize, usize, usize)>,

    /// `movable_draw_count` seen last frame; used to detect topology changes.
    last_movable_draw_count: u32,
}

impl ShadowDirtyPass {
    /// Allocate all GPU resources.  Pass the shared buffers to `ShadowPass::new()`.
    pub fn new(device: &wgpu::Device) -> Self {
        // ── Shader ────────────────────────────────────────────────────────────
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ShadowDirty Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/shadow_dirty.wgsl").into(),
            ),
        });

        // ── Bind Group Layout ─────────────────────────────────────────────────
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ShadowDirty BGL"),
            entries: &[
                // 0: instances (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: movable_draws (read-only storage)
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
                // 2: prev_positions (read-write storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: shadow_mats (read-only storage)
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
                // 4: face_dirty (read-write storage, atomic)
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
                // 5: face_geom_count (read-write storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 6: uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // ── Pipeline ──────────────────────────────────────────────────────────
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ShadowDirty PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ShadowDirty Pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ── Buffers ───────────────────────────────────────────────────────────

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ShadowDirty/Uniforms"),
            size: std::mem::size_of::<ShadowDirtyUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // prev_positions: one vec4f per movable draw slot.
        // MAX_SHADOW_FACES (256) is a safe upper bound — scenes rarely have
        // more than a few dozen movable shadow casters.
        let prev_positions_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ShadowDirty/PrevPositions"),
            size: (MAX_SHADOW_FACES * 16) as u64,  // 256 × vec4f
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // face_dirty: one atomic<u32> per shadow face.  Zeroed by the shader each frame.
        let face_dirty_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ShadowDirty/FaceDirty"),
            size: (MAX_SHADOW_FACES * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        }));

        // face_geom_count: one u32 per shadow face.  Written by this shader, read by ShadowPass.
        let face_geom_count_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ShadowDirty/FaceGeomCount"),
            size: (MAX_SHADOW_FACES * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        }));

        Self {
            pipeline,
            bgl,
            uniform_buf,
            prev_positions_buf,
            face_dirty_buf,
            face_geom_count_buf,
            bind_group: None,
            bind_group_key: None,
            last_movable_draw_count: u32::MAX, // force force_dirty_all on first frame
        }
    }
}

// ── RenderPass impl ───────────────────────────────────────────────────────────

impl RenderPass for ShadowDirtyPass {
    fn name(&self) -> &'static str {
        "ShadowDirty"
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let movable_draw_count = ctx.scene.shadow_movable_draw_count;
        let face_count = (ctx.scene.shadow_matrices.len() as u32).min(MAX_SHADOW_FACES as u32);

        // Detect topology changes (objects added/removed from movable set).
        let force_dirty_all = if movable_draw_count != self.last_movable_draw_count {
            self.last_movable_draw_count = movable_draw_count;
            1u32
        } else {
            0u32
        };

        let u = ShadowDirtyUniforms {
            movable_draw_count,
            face_count,
            force_dirty_all,
            _pad: 0,
        };
        ctx.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let movable_draw_count = ctx.scene.shadow_movable_draw_count;
        let face_count = ctx.scene.shadow_count;

        if face_count == 0 {
            return Ok(());
        }

        // ── Lazy bind group rebuild on GrowableBuffer reallocation ─────────────
        let inst_ptr     = ctx.scene.instances as *const _ as usize;
        let mov_ptr      = ctx.scene.shadow_movable_indirect as *const _ as usize;
        let sm_ptr       = ctx.scene.shadow_matrices as *const _ as usize;
        let key = (inst_ptr, mov_ptr, sm_ptr);

        if self.bind_group_key != Some(key) {
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ShadowDirty BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ctx.scene.instances.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: ctx.scene.shadow_movable_indirect.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.prev_positions_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: ctx.scene.shadow_matrices.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.face_dirty_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.face_geom_count_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.uniform_buf.as_entire_binding(),
                    },
                ],
            }));
            self.bind_group_key = Some(key);
        }

        let bg = self.bind_group.as_ref().unwrap();

        // Dispatch enough threads to cover all movable draw calls.
        // Thread 0 also zeroes face_dirty / face_geom_count, so dispatch at least 1.
        let thread_count = movable_draw_count.max(1);
        let workgroups = thread_count.div_ceil(WORKGROUP_SIZE);

        let mut pass = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ShadowDirty"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
        Ok(())
    }
}
