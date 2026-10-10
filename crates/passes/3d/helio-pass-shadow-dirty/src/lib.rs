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
    face_count: u32,
    _pad: [u32; 3],
}

// ── Pass struct ───────────────────────────────────────────────────────────────

pub struct ShadowDirtyPass {
    pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    bgl: wgpu::BindGroupLayout,

    /// Uniform buffer holding per-frame parameters.
    uniform_buf: wgpu::Buffer,

    /// Previous-frame bounds and hash of each movable draw call.
    /// Layout: `array<Previous>` indexed by draw-call index (NOT instance index),
    /// sized for every draw the batch can produce; only the first live-count
    /// entries are valid.
    prev_positions_buf: wgpu::Buffer,

    /// The movable caster count the last dispatch saw (one `u32`, GPU-only):
    /// a change dirties every face.
    last_movable_count_buf: wgpu::Buffer,

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
    bind_group_key: Option<[wgpu::Buffer; 6]>,
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
                // 9: Object Batch's GPU counts (movable caster count at word 2)
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 10: the movable count the last dispatch saw
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
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

        // Starts at u32::MAX, which no live count equals: the first dispatch
        // dirties every face.
        let last_movable_count_buf =
            wgpu::util::DeviceExt::create_buffer_init(device, &wgpu::util::BufferInitDescriptor {
                label: Some("ShadowDirty/LastMovableCount"),
                contents: &u32::MAX.to_le_bytes(),
                usage: wgpu::BufferUsages::STORAGE,
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
            last_movable_count_buf,
            face_dirty_buf,
            face_geom_count_buf,
            light_dirty_buf,
            bind_group: None,
            bind_group_key: None,
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
        let face_count = ctx
            .registry
            .get::<helio_pass_shadow_matrix::ShadowMatricesFrameData<'_>>(
                helio_core::resource_keys::shadow_matrices(),
            )
            .map(|s| s.shadow_count)
            .unwrap_or(0)
            .min(MAX_SHADOW_FACES as u32);

        // The movable caster count, and whether it changed, are GPU-only:
        // see `last_movable_count_buf`.
        let u = ShadowDirtyUniforms {
            face_count,
            _pad: [0; 3],
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
        // Every movable draw the batch can produce this frame; the live count
        // is read on the GPU.
        let draw_capacity = batch.group_capacity;
        let Some(coords) = ctx
            .registry
            .get::<helio_pass_gbuffer::CoordinateSpacesFrameData<'_>>(
                helio_core::resource_keys::coordinate_spaces(),
            )
        else {
            return Ok(());
        };
        let required = u64::from(draw_capacity.max(1)) * 32;
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
            batch.draw_counts_gpu.clone(),
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
                        wgpu::BindGroupEntry {
                            binding: 9,
                            resource: batch.draw_counts_gpu.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 10,
                            resource: self.last_movable_count_buf.as_entire_binding(),
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

        // Dispatch enough threads to cover every movable draw the batch can
        // produce; threads past this frame's live count return at once.
        // At least one thread, so a change to an empty movable set still
        // passes through the force-dirty path.
        let thread_count = draw_capacity.max(1);
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
