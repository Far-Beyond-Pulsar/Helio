//! GPU invalidation of cached shadow faces. Hashes all instances in each draw,
//! including transforms and coordinate spaces, and tests both old and new bounds.
//! Dirty bits remain set until the bounded shadow scheduler renders their tile.

use bytemuck::{Pod, Zeroable};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use std::sync::Arc;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Maximum shadow atlas faces.  Must match `MAX_FACES` in the WGSL shader and
/// `MAX_SHADOW_FACES` in `helio-pass-shadow`.
const MAX_SHADOW_FACES: usize = helio_pass_shadow_matrix::MAX_SHADOW_FACES as usize;

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

    /// Per-face dirty flag: 0 = clean, 1 = dirty (atomic u32 array, MAX_SHADOW_FACES entries).
    /// Shared with `ShadowPass` — published via `Arc` so the shadow pass can bind it.
    pub face_dirty_buf: Arc<wgpu::Buffer>,

    /// Per-face geometry draw count (atomic u32 array, MAX_SHADOW_FACES entries).
    /// ShadowPass uses this as the `count_buffer` argument to
    /// `multi_draw_indexed_indirect_count` for movable geometry draws.
    pub face_geom_count_buf: Arc<wgpu::Buffer>,

    /// Per-caster flags written by ShadowMatrixPass when a light matrix changes.
    light_dirty_buf: Arc<wgpu::Buffer>,

    /// Bind group (lazy; rebuilt whenever the `instances` or `shadow_mats` buffer
    /// pointer changes due to `GrowableBuffer` reallocation).
    bind_group: Option<wgpu::BindGroup>,
    bind_group_key: Option<[wgpu::Buffer; 5]>,

    /// `movable_draw_count` seen last frame; used to detect topology changes.
    last_movable_draw_count: u32,
}

impl ShadowDirtyPass {
    /// Allocate all GPU resources.  Pass the shared buffers to `ShadowPass::new()`.
    pub fn new(device: &wgpu::Device, light_dirty_buf: Arc<wgpu::Buffer>) -> Self {
        // ── Shader ────────────────────────────────────────────────────────────
        let shader = helio_core::shader::module(device, "ShadowDirty Shader", helio_core::include_wgsl!("../shaders/shadow_dirty.wgsl"));

        // ── Bind Group Layout ─────────────────────────────────────────────────
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ShadowDirty BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
                // 7: per-caster light dirty flags from ShadowMatrixPass
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
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
        // MAX_SHADOW_FACES is a safe upper bound — scenes rarely have
        // more than a few dozen movable shadow casters.
        let prev_positions_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ShadowDirty/PrevPositions"),
            size: (MAX_SHADOW_FACES * 32) as u64, // 256 × vec4f
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // face_dirty: one atomic<u32> per shadow face. Cleared by the command
        // encoder before the compute dispatch, which provides ordering across
        // every workgroup (a shader workgroup barrier cannot do that).
        let face_dirty_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ShadowDirty/FaceDirty"),
            size: (MAX_SHADOW_FACES * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // face_geom_count: one u32 per shadow face.  Written by this shader, read by ShadowPass.
        let face_geom_count_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ShadowDirty/FaceGeomCount"),
            size: (MAX_SHADOW_FACES * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        Self {
            pipeline,
            bgl,
            uniform_buf,
            prev_positions_buf,
            face_dirty_buf,
            face_geom_count_buf,
            light_dirty_buf,
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

    fn declare_resources(&self, builder: &mut helio_core::graph::ResourceBuilder) {
        builder.read("object_batch");
        builder.read("shadow_matrices");
        builder.write_buffer("shadow_dirty");
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a helio_core::ResourceRegistry<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let movable_draw_count = ctx
            .registry
            .get::<helio_pass_gbuffer::ObjectBatchFrameData<'_>>(helio_core::ResourceKey::new(
                "object_batch",
            ))
            .map(|b| b.shadow_movable_draw_count)
            .unwrap_or(0);
        let face_count = ctx
            .registry
            .get::<helio_pass_shadow_matrix::ShadowMatricesFrameData<'_>>(
                helio_core::resource_keys::shadow_matrices(),
            )
            .map(|s| s.shadow_count)
            .unwrap_or(0)
            .min(MAX_SHADOW_FACES as u32);

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
        let Some(batch) = ctx
            .registry
            .get::<helio_pass_gbuffer::ObjectBatchFrameData<'_>>(helio_core::ResourceKey::new(
                "object_batch",
            ))
        else {
            return Ok(());
        };
        let movable_draw_count = batch.shadow_movable_draw_count;
        let Some(coords) = ctx
            .registry
            .get::<helio_pass_gbuffer::CoordinateSpacesFrameData<'_>>(
                helio_core::resource_keys::coordinate_spaces(),
            )
        else {
            return Ok(());
        };
        let required = u64::from(movable_draw_count.max(1)) * 32;
        if required > self.prev_positions_buf.size() {
            self.prev_positions_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Shadow draw history"),
                size: required.next_power_of_two(),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.bind_group_key = None;
        }
        let Some(shadow_data) = ctx
            .registry
            .get::<helio_pass_shadow_matrix::ShadowMatricesFrameData<'_>>(
                helio_core::resource_keys::shadow_matrices(),
            )
        else {
            return Ok(());
        };
        let face_count = shadow_data.shadow_count;

        if face_count == 0 {
            return Ok(());
        }

        // ── Lazy bind group rebuild on GrowableBuffer reallocation ─────────────
        let key = [
            batch.instances.clone(),
            batch.shadow_movable_indirect.clone(),
            shadow_data
                .desired_matrices
                .unwrap_or(shadow_data.shadow_matrices)
                .clone(),
            (*self.light_dirty_buf).clone(),
            coords.coordinate_spaces.clone(),
        ];

        if self.bind_group_key.as_ref() != Some(&key) {
            self.bind_group = Some(
                ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ShadowDirty BG"),
                    layout: &self.bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: coords.coordinate_spaces.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: batch.instances.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: batch.shadow_movable_indirect.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.prev_positions_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: shadow_data
                                .desired_matrices
                                .unwrap_or(shadow_data.shadow_matrices)
                                .as_entire_binding(),
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
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: self.light_dirty_buf.as_entire_binding(),
                        },
                    ],
                }),
            );
            self.bind_group_key = Some(key);
        }

        let bg = self.bind_group.as_ref().unwrap();

        // Reset the complete output arrays before dispatch. Doing this as
        // encoder commands avoids the cross-workgroup race that occurs when
        // invocation zero clears storage while other workgroups write it.
        let mut cmds = ctx.graphics_cmds();
        // Pending dirty bits survive until ShadowPass services their tile.

        // Dispatch enough threads to cover all movable draw calls.
        // Dispatch at least one thread so topology changes with an empty
        // movable set still pass through the force-dirty path.
        let thread_count = movable_draw_count.max(1);
        let workgroups = thread_count.div_ceil(WORKGROUP_SIZE);

        let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ShadowDirty"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
        Ok(())
    }
}
