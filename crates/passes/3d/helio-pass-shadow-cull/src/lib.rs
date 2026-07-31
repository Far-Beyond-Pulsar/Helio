//! GPU per-face shadow frustum culling pass.
//!
//! Runs AFTER ShadowDirtyPass and BEFORE ShadowPass.  For each movable draw call,
//! tests the instance's bounding sphere against each dirty shadow face's frustum
//! (planes extracted from the VP matrix via Gribb-Hartmann).  Visible draws are
//! atomically appended to a per-face compacted indirect list.
//!
//! # Buffers produced
//!
//! | Buffer              | Format                                          |
//! |---------------------|-------------------------------------------------|
//! | `face_indirect_buf` | `MAX_FACES × MAX_DRAWS_PER_FACE × 20` bytes     |
//! | `face_counts_buf`   | `MAX_FACES` × `u32` — per-face visible draw cnt |
//!
//! ShadowPass reads these buffers to issue per-face indirect draws, replacing
//! the global `shadow_movable_indirect` + `face_geom_count_buf` path.
//!
//! # Integration
//!
//! ```ignore
//! let cull_pass = ShadowCullPass::new(device, face_dirty_buf);
//! let face_indirect_buf = Arc::clone(&cull_pass.face_indirect_buf);
//! let face_counts_buf = Arc::clone(&cull_pass.face_counts_buf);
//! graph.add_pass(Box::new(cull_pass));
//!
//! ShadowPass::new(device, face_dirty_buf, face_geom_count_buf, atlas_size)
//!     .with_culled_buffers(face_indirect_buf, face_counts_buf);
//! ```

use bytemuck::{Pod, Zeroable};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use std::sync::Arc;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Maximum shadow atlas faces.  Must match `MAX_FACES` in the WGSL shader.
const MAX_FACES: usize = 256;

/// Maximum draws per face after culling.  Must match the shader.
const MAX_DRAWS_PER_FACE: u32 = 4096;

const WORKGROUP_SIZE: u32 = 64;

// ── Uniforms ──────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CullUniforms {
    instance_count:    u32,
    max_draws_per_face: u32,
    _pad0:             u32,
    _pad1:             u32,
}

// ── Pass struct ───────────────────────────────────────────────────────────────

pub struct ShadowCullPass {
    pipeline: wgpu::ComputePipeline,
    bgl:      wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,

    /// Per-face compacted indirect draw commands.
    /// Layout: for face N, draws are at offsets `[N * MAX_DRAWS_PER_FACE, (N+1) * MAX_DRAWS_PER_FACE)`.
    pub face_indirect_buf: Arc<wgpu::Buffer>,

    /// Per-face atomic draw counts — read by `multi_draw_indexed_indirect_count`.
    pub face_counts_buf: Arc<wgpu::Buffer>,

    /// Shared face-dirty buffer from ShadowDirtyPass (read-only here).
    face_dirty_buf: Arc<wgpu::Buffer>,

    /// Lazy bind group, rebuilt when scene buffer pointers change.
    bind_group:     Option<wgpu::BindGroup>,
    bind_group_key: Option<(usize, usize, usize, usize)>,
}

impl ShadowCullPass {
    /// Allocate all GPU resources.
    ///
    /// `face_dirty_buf` is shared with `ShadowDirtyPass` (and `ShadowPass`) —
    /// this pass only *reads* it to skip clean faces.
    pub fn new(
        device: &wgpu::Device,
        face_dirty_buf: Arc<wgpu::Buffer>,
    ) -> Self {
        // ── Shader ────────────────────────────────────────────────────────────
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ShadowCull Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/shadow_cull.wgsl").into(),
            ),
        });

        // ── Uniform buffer ────────────────────────────────────────────────────
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("ShadowCull/Uniforms"),
            size:               std::mem::size_of::<CullUniforms>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Output buffers ────────────────────────────────────────────────────
        // Per-face indirect commands: MAX_FACES faces × MAX_DRAWS_PER_FACE × 20 bytes each
        let face_indirect_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ShadowCull/FaceIndirect"),
            size: (MAX_FACES as u64) * (MAX_DRAWS_PER_FACE as u64) * 20u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        }));

        // Per-face atomic counters: one u32 per face
        let face_counts_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ShadowCull/FaceCounts"),
            size: (MAX_FACES as u64) * 4u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // ── Bind Group Layout ─────────────────────────────────────────────────
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ShadowCull BGL"),
            entries: &[
                // 0: CullUniforms (uniform)
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
                // 1: shadow_matrices (storage read)
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
                // 2: instances (storage read)
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
                // 3: src_indirect (storage read) — shadow_movable_indirect
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
                // 4: dst_indirect (storage read_write) — per-face output
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
                // 5: face_counts (storage read_write, atomic)
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
                // 6: face_dirty (storage read)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
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

        // ── Pipeline ──────────────────────────────────────────────────────────
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("ShadowCull PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size:     0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:               Some("ShadowCull Pipeline"),
            layout:              Some(&pl),
            module:              &shader,
            entry_point:         Some("main"),
            compilation_options: Default::default(),
            cache:               None,
        });

        Self {
            pipeline,
            bgl,
            uniform_buf,
            face_indirect_buf,
            face_counts_buf,
            face_dirty_buf,
            bind_group:     None,
            bind_group_key: None,
        }
    }
}

impl RenderPass for ShadowCullPass {
    fn name(&self) -> &'static str {
        "ShadowCull"
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let u = CullUniforms {
            instance_count:    ctx.scene.shadow_movable_draw_count,
            max_draws_per_face: MAX_DRAWS_PER_FACE,
            _pad0:             0,
            _pad1:             0,
        };
        ctx.queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let movable_count = ctx.scene.shadow_movable_draw_count;
        let face_count    = ctx.scene.shadow_count;

        if face_count == 0 || movable_count == 0 {
            return Ok(());
        }

        // ── Reset face counters to zero ───────────────────────────────────────
        unsafe { &mut *ctx.encoder_ptr }.clear_buffer(
            &self.face_counts_buf,
            0,
            Some((MAX_FACES as u64) * 4u64),
        );

        // ── Lazy bind-group rebuild on GrowableBuffer reallocation ────────────
        let sm_ptr  = ctx.scene.shadow_matrices       as *const _ as usize;
        let inst_ptr = ctx.scene.instances             as *const _ as usize;
        let src_ptr  = ctx.scene.shadow_movable_indirect as *const _ as usize;
        let fd_ptr   = &*self.face_dirty_buf           as *const _ as usize;
        let key = (sm_ptr, inst_ptr, src_ptr, fd_ptr);

        if self.bind_group_key != Some(key) {
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ShadowCull BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: ctx.scene.shadow_matrices.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: ctx.scene.instances.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: ctx.scene.shadow_movable_indirect.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.face_indirect_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.face_counts_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.face_dirty_buf.as_entire_binding(),
                    },
                ],
            }));
            self.bind_group_key = Some(key);
        }

        let bg = self.bind_group.as_ref().unwrap();

        let wg = movable_count.div_ceil(WORKGROUP_SIZE);
        let mut pass = unsafe { &mut *ctx.encoder_ptr }.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label:            Some("ShadowCull"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(wg, 1, 1);
        Ok(())
    }
}
