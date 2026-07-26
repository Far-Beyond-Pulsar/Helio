//! The M3-b T9-followup repack pass: promotes the compute-shader repack
//! `benches/pass_timing.rs` prototyped bench-local (T9 review, strategy
//! (b')'s worked example) into a real library capability. This is the
//! bridge `DrawExecutor::record_multi_indirect` (`draw.rs`, the T9-
//! recommended default for M3-γ/M4) needs but cannot provide itself:
//! `multi_draw_indexed_indirect` requires a TIGHTLY PACKED 20-byte
//! `DrawIndexedIndirectArgs` array, but [`crate::cull::CullOutputBuffers`]'s
//! `CullRecord` stride is 32 bytes (`cull.rs`'s module doc has the
//! group(2) storage-budget reason those extra 12 bytes exist). One GPU-side
//! compute dispatch bridges the gap: read each `CullRecord`'s first 20
//! bytes, write them densely to a separate buffer. No CPU readback, no CPU
//! stall.
//!
//! [`crate::wgsl::REPACK_WGSL`]'s module doc has the full 32->20 byte field
//! contract (which `CullRecord` field maps to which packed-args field, and
//! why `first_instance` specifically MUST survive unchanged -- it is the
//! §14.1 command-slot bindless key `DRAW_WGSL`'s `vs_main` resolves
//! `draw_cull_output.records[iid].row` through). Read that doc before
//! touching either side of this contract.
//!
//! A real dense (non-compacting) cull-shader variant that writes the packed
//! format directly and never needs a repack pass at all is M3-γ/M4 scope,
//! not this task's.

use crate::cull::CullOutputBuffers;
use crate::wgsl::REPACK_WGSL;

/// Rust twin of [`REPACK_WGSL`]'s `RepackUniforms` -- the repack pass's
/// per-dispatch constant input: just the dispatch bound (`capacity`, the
/// same value the cull pass's own `CullUniforms::capacity` names -- how
/// many slots exist in the source `CullOutputBuffers`/destination packed
/// buffer, NOT the cull pass's measured `visible_count`; repacking every
/// slot, visible or not, is what lets strategy (b')'s "conservative
/// max-count" steady-state pre-fill -- `instance_count == 0` on
/// never-touched tail slots -- flow through the repack unchanged, matching
/// (b)'s own no-readback design).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RepackUniforms {
    pub capacity: u32,
    pub reserved: [u32; 3],
}
const _: () = assert!(std::mem::size_of::<RepackUniforms>() == 16);
// SAFETY: `#[repr(C)]`, `Copy`, scalar/array-of-scalar fields only, no
// padding (4x 4-byte fields, all 4-aligned).
unsafe impl bytemuck::Zeroable for RepackUniforms {}
unsafe impl bytemuck::Pod for RepackUniforms {}

/// The repack pass itself: one compute pipeline, one `@group(0)` bind-group
/// layout (`input`/`packed`/`u` -- [`REPACK_WGSL`]'s module doc has the
/// binding-layout rationale), one uniform buffer. Stateless across
/// dispatches beyond that, matching [`crate::cull::CullPass`]'s and
/// [`crate::draw::DrawExecutor`]'s "rebuilt, never mutated in place" idiom.
pub struct RepackPass {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,
}

impl RepackPass {
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("helio-scenedb-repack-shader"),
            source: wgpu::ShaderSource::Wgsl(REPACK_WGSL.into()),
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
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("helio-scenedb-repack-layout"),
            entries: &[
                storage_entry(0, true),  // input (CullOutputBuffers, whole-buffer bind -- see REPACK_WGSL doc)
                storage_entry(1, false), // packed (destination, tightly-packed DrawIndexedIndirectArgs)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("helio-scenedb-repack-pipeline-layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("helio-scenedb-repack-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("repack_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("helio-scenedb-repack-uniforms"),
            size: std::mem::size_of::<RepackUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { pipeline, layout, uniform_buf }
    }

    pub fn write_capacity(&self, queue: &wgpu::Queue, capacity: u32) {
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&RepackUniforms { capacity, reserved: [0; 3] }));
    }

    /// Builds the `@group(0)` bind group over `cull_output`'s combined
    /// counters+records buffer (read-only source, whole-buffer bind -- see
    /// [`REPACK_WGSL`]'s doc) and `packed_buf` (the write target -- callers
    /// pass a buffer sized by [`packed_indirect_buffer`]). Rebuild whenever
    /// either underlying `wgpu::Buffer` changes, mirroring `CullPass::
    /// build_output_bind_group`'s idiom.
    #[must_use]
    pub fn build_bind_group(
        &self,
        device: &wgpu::Device,
        cull_output: &CullOutputBuffers,
        packed_buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("helio-scenedb-repack-bind-group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: cull_output.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: packed_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.uniform_buf.as_entire_binding() },
            ],
        })
    }

    /// Dispatches `repack_main` over `capacity` threads (one per record
    /// slot -- callers pass the SAME `capacity` [`Self::write_capacity`]
    /// last wrote, matching [`crate::cull::CullPass::record`]'s
    /// dispatch-count-mirrors-uniform idiom). No-op (does not even open a
    /// compute pass) when `capacity == 0`.
    pub fn record(&self, encoder: &mut wgpu::CommandEncoder, bind_group: &wgpu::BindGroup, capacity: u32) {
        if capacity == 0 {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("helio-scenedb-repack-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(capacity.div_ceil(64), 1, 1);
    }
}

/// Byte stride of one packed record -- identical to
/// [`crate::cull::DrawCommand`]'s size (the packed buffer's WGSL-side
/// `PackedArgs` struct and `DrawCommand` describe the same 20-byte
/// `DrawIndexedIndirectArgs` shape; this reuses that existing const-asserted
/// Rust type as the single source of truth rather than a second literal
/// `20` living here too).
pub const PACKED_RECORD_BYTES: u64 = std::mem::size_of::<crate::cull::DrawCommand>() as u64;

/// Allocates a tightly-packed (20 B/record) `DrawIndexedIndirectArgs`-shaped
/// buffer -- [`RepackPass`]'s output, and `multi_draw_indexed_indirect`'s
/// (`DrawExecutor::record_multi_indirect`'s) required input shape. `usage`
/// carries `STORAGE` (the repack pass's compute-shader write target),
/// `INDIRECT` (wgpu's validation requirement for a buffer
/// `multi_draw_indexed_indirect` reads its args from), and `COPY_SRC`
/// (mirrors [`crate::cull::CullOutputBuffers::new`]'s own usage set --
/// tests/spot-checks need to read the repacked bytes back to verify the
/// 32->20 byte contract, `tests/draw_multi_indirect_equivalence.rs`'s job).
#[must_use]
pub fn packed_indirect_buffer(device: &wgpu::Device, capacity: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("helio-scenedb-repacked-indirect"),
        size: capacity.max(1) as u64 * PACKED_RECORD_BYTES,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}
