//! The M3-b T5 cull compute pass: the first GPU work that consumes
//! SceneDB's published scene data and emits indirect draw commands (design
//! S4's β term list: `i < count` guard -> S3.1 generation validation ->
//! mesh_index bounds check -> S11 |M3x3| world AABB -> S12 near-clip bypass
//! -> frustum -> S14.2 bounded-atomic command-slot allocation).
//!
//! [`CullPass`] owns the compute pipeline plus the ONE bind-group layout it
//! adds (group(2) -- [`crate::wgsl::CULL_WGSL`]'s module doc has the full
//! storage-budget arithmetic). It binds `@group(0)` from an existing
//! [`crate::SceneDbBinding`] (the cull-read set); it NEVER binds `@group(1)`
//! (see `CULL_WGSL`'s module doc for how, not just "by convention").
//!
//! [`CullOutputBuffers`] owns the per-view output allocation (the combined
//! atomics-header + `CullRecord` array `CULL_WGSL` calls `CullOutput`) --
//! capacity-managed like [`pulsar_scenedb::gpu::ViewTokenBuffers`] (grow, or
//! caller-sized up front; this pass does not resize it automatically, since
//! unlike the per-view token pair this buffer's natural size is a driver
//! policy decision -- S14.2's "CPU clamps on readback" already handles
//! under-provisioning without silent corruption).

use pulsar_scenedb::gpu::ViewTokenBuffers;

use crate::wgsl::{CULL_WGSL, SCENE_BINDINGS_WGSL};

const WORKGROUP_SIZE: u32 = 64;

/// Standalone indirect-draw wire format (spec S14.1) -- 20 bytes, Rust twin
/// of [`CULL_WGSL`]'s `DrawCommand` struct. Helio-owned derived data (C0):
/// SceneDB never defines or touches this type, so it has NO twin in
/// `pulsar_scenedb` -- this crate's own `tests/binding_layout.rs` Test 3
/// rows are the only reflection coverage it needs.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DrawCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}
const _: () = assert!(std::mem::size_of::<DrawCommand>() == 20);
// SAFETY: `#[repr(C)]`, `Copy`, every field is scalar POD, no padding (5x
// 4-byte fields, all 4-aligned) -- matches the const-asserted 20-byte size.
unsafe impl bytemuck::Zeroable for DrawCommand {}
unsafe impl bytemuck::Pod for DrawCommand {}

/// One combined per-command-slot output record -- Rust twin of
/// [`CULL_WGSL`]'s `CullRecord`. First 5 fields are field-for-field
/// identical to [`DrawCommand`] (same order, same offsets 0/4/8/12/16);
/// `row` is the design's row-valued `visible_instance_ids[slot]` (S3/R11);
/// `flags` bit 0 ([`NEAR_CLIP_FLAG`]) carries the S12 near-clip bypass.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CullRecord {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
    pub row: u32,
    pub flags: u32,
    pub reserved: u32,
}
const _: () = assert!(std::mem::size_of::<CullRecord>() == 32);
unsafe impl bytemuck::Zeroable for CullRecord {}
unsafe impl bytemuck::Pod for CullRecord {}

/// [`CullRecord::flags`] bit 0 (spec S12): set when the near-plane W<=0
/// bypass fired for this instance -- downstream passes must treat it as
/// covering an indeterminate screen-space area rather than trusting a
/// garbage projected rect (spec S12's exact wording).
pub const NEAR_CLIP_FLAG: u32 = 1;

/// Rust twin of [`CULL_WGSL`]'s `CullUniforms` -- the cull pass's per-
/// dispatch constant inputs: the view-proj matrix (S12's corner-projection
/// W<=0 test), 6 world-space frustum planes in the `n.p+d>=0`-inside
/// convention (`spatial::Frustum`'s own convention, S4's frustum term), and
/// the three scalar bounds the shader's guards read (`count` for the
/// `i < count` dispatch guard, `mesh_count` for the mesh_index bounds
/// check, `capacity` for the S14.2 slot-allocation ceiling).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CullUniforms {
    pub view_proj: [f32; 16],
    pub planes: [[f32; 4]; 6],
    pub count: u32,
    pub mesh_count: u32,
    pub capacity: u32,
    pub reserved: u32,
}
const _: () = assert!(std::mem::size_of::<CullUniforms>() == 176);
unsafe impl bytemuck::Zeroable for CullUniforms {}
unsafe impl bytemuck::Pod for CullUniforms {}

/// Byte layout of the combined `CullOutput` buffer (`CULL_WGSL`'s module
/// doc): a fixed 16-byte atomics header (`visible_count`/`stale_drops`/
/// `oob_drops`/`frustum_drops`, each `u32`) followed by `capacity` many
/// 32-byte [`CullRecord`]s.
pub struct CullOutputBuffers {
    buffer: wgpu::Buffer,
    capacity: u32,
}

impl CullOutputBuffers {
    pub const HEADER_BYTES: u64 = 16;
    pub const RECORD_BYTES: u64 = std::mem::size_of::<CullRecord>() as u64;

    /// Allocates a zero-initialized (WebGPU-guaranteed, per spec -- wgpu
    /// conforms) `CullOutput` buffer sized for `capacity` records. `label`
    /// names the underlying `wgpu::Buffer` for GPU-debugger identification.
    #[must_use]
    pub fn new(device: &wgpu::Device, label: &str, capacity: u32) -> Self {
        let size = Self::HEADER_BYTES + capacity as u64 * Self::RECORD_BYTES;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self { buffer, capacity }
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn byte_size(&self) -> u64 {
        Self::HEADER_BYTES + self.capacity as u64 * Self::RECORD_BYTES
    }

    /// Zero the 4 atomic counters (offsets 0..16). The shader only ever
    /// increments them -- nothing else resets them between dispatches, so a
    /// per-frame driver reusing this buffer across frames MUST call this
    /// before each [`CullPass::record`]. A freshly [`Self::new`]-allocated
    /// buffer does not need it (WebGPU zero-init), but calling it anyway is
    /// harmless and matches what a real per-frame driver does.
    pub fn clear_counters(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.buffer, 0, &[0u8; Self::HEADER_BYTES as usize]);
    }
}

/// The cull compute pass itself: one pipeline, one bind-group layout
/// (group(2) -- [`CULL_WGSL`]'s module doc has the full storage-budget
/// arithmetic), one uniform buffer. Stateless across dispatches beyond that
/// (no cached bind group -- callers build a fresh group(2) bind group per
/// [`Self::build_output_bind_group`] call whenever the underlying buffers
/// change, mirroring `SceneDbBinding::new`'s "rebuilt, never mutated in
/// place" idiom).
pub struct CullPass {
    pipeline: wgpu::ComputePipeline,
    output_layout: wgpu::BindGroupLayout,
    uniforms_buf: wgpu::Buffer,
}

impl CullPass {
    /// `cull_layout` is [`crate::SceneDbBinding::cull_layout`] (group(0) --
    /// the caller already built a `SceneDbBinding`; this pass borrows only
    /// its cull-read layout, never `draw_layout`, matching `CULL_WGSL`'s "no
    /// group(1)" design).
    #[must_use]
    pub fn new(device: &wgpu::Device, cull_layout: &wgpu::BindGroupLayout) -> Self {
        let shader_src = format!("{SCENE_BINDINGS_WGSL}\n{CULL_WGSL}");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("helio-scenedb-cull-shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let storage_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let output_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("helio-scenedb-cull-output-layout"),
            entries: &[
                storage_entry(0, true),  // cull_tokens
                storage_entry(1, true),  // cull_expected_gens
                storage_entry(2, false), // cull_output (counters + records)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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

        let uniforms_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("helio-scenedb-cull-uniforms"),
            size: std::mem::size_of::<CullUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // group(1) genuinely absent -- see CULL_WGSL's module doc for why
        // this is sound (wgpu/naga validate against the entry point's
        // reachable bindings, and `cull_main` never touches group(1)).
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("helio-scenedb-cull-pipeline-layout"),
            bind_group_layouts: &[Some(cull_layout), None, Some(&output_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("helio-scenedb-cull-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cull_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self { pipeline, output_layout, uniforms_buf }
    }

    pub fn write_uniforms(&self, queue: &wgpu::Queue, uniforms: &CullUniforms) {
        queue.write_buffer(&self.uniforms_buf, 0, bytemuck::bytes_of(uniforms));
    }

    /// Builds the group(2) bind group over `view`'s token pair and
    /// `output`'s combined counters+records buffer, plus this pass's own
    /// uniform buffer. Rebuild whenever `view`/`output`'s underlying
    /// `wgpu::Buffer`s change (e.g. `ViewTokenBuffers::upload` grew the
    /// buffers) -- no caching here, matching `SceneDbBinding::new`'s idiom.
    #[must_use]
    pub fn build_output_bind_group(
        &self,
        device: &wgpu::Device,
        view: &ViewTokenBuffers,
        output: &CullOutputBuffers,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("helio-scenedb-cull-output-bind-group"),
            layout: &self.output_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: view.tokens_buffer().as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: view.expected_gens_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry { binding: 2, resource: output.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.uniforms_buf.as_entire_binding() },
            ],
        })
    }

    /// Dispatches `cull_main` over `dispatch_count` threads (one per
    /// harvested token -- callers pass [`ViewTokenBuffers::count`]).
    /// `cull_bind_group` is [`crate::SceneDbBinding::cull_bind_group`]
    /// (group(0)); `output_bind_group` is
    /// [`Self::build_output_bind_group`]'s result (group(2)). group(1) is
    /// NEVER bound -- see [`CULL_WGSL`]'s module doc.
    ///
    /// No-op (does not even open a compute pass) when `dispatch_count == 0`
    /// -- there is nothing to cull and nothing to clear (the shader itself
    /// never runs, so it cannot touch the counters either; callers that
    /// need a guaranteed-zeroed counter buffer for an empty view call
    /// [`CullOutputBuffers::clear_counters`] themselves).
    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        cull_bind_group: &wgpu::BindGroup,
        output_bind_group: &wgpu::BindGroup,
        dispatch_count: u32,
    ) {
        if dispatch_count == 0 {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("helio-scenedb-cull-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, cull_bind_group, &[]);
        pass.set_bind_group(2, output_bind_group, &[]);
        pass.dispatch_workgroups(dispatch_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    }
}
