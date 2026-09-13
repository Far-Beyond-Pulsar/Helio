//! GPU-driven object sort/group/batch pipeline.
//!
//! See `shaders/object_batch.wgsl`'s module doc for the full 8-stage
//! pipeline design. This crate is the Rust-side pipeline construction,
//! buffer growth management, dispatch sequencing, and the small async
//! CPU readback of the (bounded, GPU-computed) material-class range tables
//! and shadow-partition draw counts every `multi_draw_indexed_indirect`
//! call site downstream still needs to know on the CPU.
//!
//! # What this pass owns vs. what it doesn't
//!
//! This pass owns every buffer it writes (`instances`/`aabbs`/`draw_calls`/
//! the three range tables/the two shadow-partition indirect lists) --
//! nothing here mutates `GpuScene`'s own fields directly. Wiring these
//! buffers into the render graph in place of `Scene::rebuild_instance_
//! buffers`'s CPU-populated `GpuScene` fields (the same "point at an
//! externally-owned buffer" seam `Scene::rebind_transform_buffer` already
//! established for SceneDB's `Transform` buffer) is a separate integration
//! step, tracked apart from this crate.

use bytemuck::{Pod, Zeroable};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use helio_pass_gbuffer::StaticObjectComponent;
use pulsar_scenedb::gpu::BufferKey;

mod readback;

const WG: u32 = 256;
const SORT_BITS: usize = 32;
/// Below this many rows, scratch buffers still allocate at this floor --
/// avoids reallocating on every single insert/remove around a tiny live
/// count, matching `GrowableBuffer`'s own "small initial capacity" idiom
/// used everywhere else in this codebase.
const MIN_SCRATCH_CAPACITY: u32 = 256;

// ── GPU-mirrored uniform/struct shapes ──────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BatchUniforms {
    capacity: u32,
    max_objects: u32,
    max_groups: u32,
    max_ranges: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniformGpu {
    count: u32,
    num_blocks: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SortUniformsGpu {
    bit: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RangeBlockUniformGpu {
    num_blocks: u32,
}

/// Matches `shaders/object_batch.wgsl`'s `GpuDrawCallOut` -- 20 bytes, same
/// shape as `libhelio::GpuDrawCall` (this crate doesn't depend on `libhelio`
/// for it since it never needs the typed version, only the byte layout for
/// buffer sizing).
const DRAW_CALL_BYTES: u64 = 20;
/// Matches `shaders/object_batch.wgsl`'s `GpuInstanceDataOut` -- 208 bytes,
/// see `libhelio::GpuInstanceData`'s own doc for the exact field breakdown.
const INSTANCE_BYTES: u64 = 208;
/// Matches `shaders/object_batch.wgsl`'s `GpuInstanceAabbOut` -- 16 bytes.
const AABB_BYTES: u64 = 16;
/// Matches `shaders/object_batch.wgsl`'s `GpuRangeOut` -- 20 bytes.
const RANGE_BYTES: u64 = 20;
/// Matches `shaders/object_batch.wgsl`'s `DrawIndexedIndirectArgsOut` -- 20
/// bytes, same shape as `libhelio::DrawIndexedIndirectArgs`.
const INDIRECT_ARGS_BYTES: u64 = 20;

fn create_storage_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// Same as [`create_storage_buffer`] but also usable as a `<uniform>`
/// binding (the `FrameUniform`/`dispatch_args`-style "write as storage in
/// one kernel, read as uniform in another" dual-bind trick already used by
/// `helio-pass-sprite-cull`/`helio-pass-corona`) and as an indirect-dispatch
/// source.
fn create_dual_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::UNIFORM
            | wgpu::BufferUsages::INDIRECT
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn bgl_entry_storage(binding: u32, visibility: wgpu::ShaderStages, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_entry_uniform(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bg_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn make_compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    bgl: &wgpu::BindGroupLayout,
    module: &wgpu::ShaderModule,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(bgl)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// One CPU-visible material-class-batched draw range, mirroring the CPU
/// reference's `(material_class, graph_hash, start, count)` tuple exactly
/// -- `start`/`count` index the `draw_calls` array, not `instances`.
pub type RangeTuple = (u32, u64, u32, u32);

struct BindGroupLayouts {
    gather: wgpu::BindGroupLayout,
    prepare: wgpu::BindGroupLayout,
    histogram: wgpu::BindGroupLayout,
    scan: wgpu::BindGroupLayout,
    scatter: wgpu::BindGroupLayout,
    final_gather: wgpu::BindGroupLayout,
    group_local_scan: wgpu::BindGroupLayout,
    group_block_scan: wgpu::BindGroupLayout,
    group_write: wgpu::BindGroupLayout,
    group_write_sentinel: wgpu::BindGroupLayout,
    prepare_groups: wgpu::BindGroupLayout,
    build_draw_calls: wgpu::BindGroupLayout,
    range_local_scan: wgpu::BindGroupLayout,
    range_block_scan: wgpu::BindGroupLayout,
    range_write: wgpu::BindGroupLayout,
    shadow_partition: wgpu::BindGroupLayout,
}

struct Pipelines {
    gather: wgpu::ComputePipeline,
    prepare: wgpu::ComputePipeline,
    histogram: wgpu::ComputePipeline,
    scan: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    final_gather: wgpu::ComputePipeline,
    group_local_scan: wgpu::ComputePipeline,
    group_block_scan: wgpu::ComputePipeline,
    group_write: wgpu::ComputePipeline,
    group_write_sentinel: wgpu::ComputePipeline,
    prepare_groups: wgpu::ComputePipeline,
    build_draw_calls: wgpu::ComputePipeline,
    range_local_scan: wgpu::ComputePipeline,
    range_block_scan: wgpu::ComputePipeline,
    range_write: wgpu::ComputePipeline,
    shadow_partition: wgpu::ComputePipeline,
}

/// Every scratch buffer this pipeline owns, sized to `capacity` rows (or
/// `capacity.div_ceil(WG)` blocks) -- see `ensure_capacity`.
struct ScratchBuffers {
    gather_count: wgpu::Buffer,
    frame_uniform: wgpu::Buffer,
    dispatch_args: wgpu::Buffer,
    keys_a: wgpu::Buffer,
    keys_b: wgpu::Buffer,
    indices_a: wgpu::Buffer,
    indices_b: wgpu::Buffer,
    block_hist: wgpu::Buffer,
    instances_out: wgpu::Buffer,
    aabbs_out: wgpu::Buffer,
    local_group_rank: wgpu::Buffer,
    block_group_totals: wgpu::Buffer,
    group_count: wgpu::Buffer,
    group_starts: wgpu::Buffer,
    dispatch_args_groups: wgpu::Buffer,
    group_material_class: wgpu::Buffer,
    group_graph_hash_lo: wgpu::Buffer,
    group_graph_hash_hi: wgpu::Buffer,
    group_shading: wgpu::Buffer,
    local_range_rank: wgpu::Buffer,
    block_range_totals: wgpu::Buffer,
    range_count: wgpu::Buffer,
    opaque_ranges: wgpu::Buffer,
    transparent_ranges: wgpu::Buffer,
    forward_ranges: wgpu::Buffer,
    range_bucket_counts: wgpu::Buffer,
    shadow_static_indirect: wgpu::Buffer,
    shadow_movable_indirect: wgpu::Buffer,
    shadow_counts: wgpu::Buffer,
}

impl ScratchBuffers {
    fn new(device: &wgpu::Device, capacity: u32) -> Self {
        let n = capacity as u64;
        let blocks = capacity.div_ceil(WG) as u64;
        Self {
            gather_count: create_storage_buffer(device, "ObjBatch GatherCount", 16),
            frame_uniform: create_dual_buffer(device, "ObjBatch FrameUniform", 8),
            dispatch_args: create_dual_buffer(device, "ObjBatch DispatchArgs", 12),
            keys_a: create_storage_buffer(device, "ObjBatch KeysA", n * 4),
            keys_b: create_storage_buffer(device, "ObjBatch KeysB", n * 4),
            indices_a: create_storage_buffer(device, "ObjBatch IndicesA", n * 4),
            indices_b: create_storage_buffer(device, "ObjBatch IndicesB", n * 4),
            block_hist: create_storage_buffer(device, "ObjBatch BlockHist", blocks * 2 * 4),
            instances_out: create_storage_buffer(device, "ObjBatch Instances", n * INSTANCE_BYTES),
            aabbs_out: create_storage_buffer(device, "ObjBatch Aabbs", n * AABB_BYTES),
            local_group_rank: create_storage_buffer(device, "ObjBatch LocalGroupRank", n * 4),
            block_group_totals: create_storage_buffer(device, "ObjBatch BlockGroupTotals", blocks * 4),
            group_count: create_storage_buffer(device, "ObjBatch GroupCount", 16),
            group_starts: create_storage_buffer(device, "ObjBatch GroupStarts", (n + 1) * 4),
            dispatch_args_groups: create_dual_buffer(device, "ObjBatch DispatchArgsGroups", 12),
            group_material_class: create_storage_buffer(device, "ObjBatch GroupClass", n * 4),
            group_graph_hash_lo: create_storage_buffer(device, "ObjBatch GroupHashLo", n * 4),
            group_graph_hash_hi: create_storage_buffer(device, "ObjBatch GroupHashHi", n * 4),
            group_shading: create_storage_buffer(device, "ObjBatch GroupShading", n * 4),
            local_range_rank: create_storage_buffer(device, "ObjBatch LocalRangeRank", n * 4),
            block_range_totals: create_storage_buffer(device, "ObjBatch BlockRangeTotals", blocks * 4),
            range_count: create_storage_buffer(device, "ObjBatch RangeCount", 16),
            opaque_ranges: create_storage_buffer(device, "ObjBatch OpaqueRanges", n * RANGE_BYTES),
            transparent_ranges: create_storage_buffer(device, "ObjBatch TransparentRanges", n * RANGE_BYTES),
            forward_ranges: create_storage_buffer(device, "ObjBatch ForwardRanges", n * RANGE_BYTES),
            range_bucket_counts: create_storage_buffer(device, "ObjBatch RangeBucketCounts", 16),
            shadow_static_indirect: create_storage_buffer(device, "ObjBatch ShadowStaticIndirect", n * INDIRECT_ARGS_BYTES),
            shadow_movable_indirect: create_storage_buffer(device, "ObjBatch ShadowMovableIndirect", n * INDIRECT_ARGS_BYTES),
            shadow_counts: create_storage_buffer(device, "ObjBatch ShadowCounts", 16),
        }
    }
}

pub struct ObjectBatchPass {
    bgls: BindGroupLayouts,
    pipelines: Pipelines,
    sort_pass_uniforms: [wgpu::Buffer; SORT_BITS],
    batch_uniform: wgpu::Buffer,
    draw_calls_out: wgpu::Buffer,
    /// Same per-group data as `draw_calls_out`, reordered to `wgpu`'s
    /// hardware indirect-draw ABI -- see `object_batch.wgsl`'s
    /// `DrawIndexedIndirectArgsOut` doc for why this needs its own buffer.
    indirect_out: wgpu::Buffer,
    scratch: ScratchBuffers,
    scratch_capacity: u32,

    gather_bg: Option<wgpu::BindGroup>,
    prepare_bg: Option<wgpu::BindGroup>,
    histogram_bgs: Option<[wgpu::BindGroup; SORT_BITS]>,
    scan_bg: Option<wgpu::BindGroup>,
    scatter_bgs: Option<[wgpu::BindGroup; SORT_BITS]>,
    final_gather_bg: Option<wgpu::BindGroup>,
    group_local_scan_bg: Option<wgpu::BindGroup>,
    group_block_scan_bg: Option<wgpu::BindGroup>,
    group_write_bg: Option<wgpu::BindGroup>,
    group_write_sentinel_bg: Option<wgpu::BindGroup>,
    prepare_groups_bg: Option<wgpu::BindGroup>,
    build_draw_calls_bg: Option<wgpu::BindGroup>,
    range_local_scan_bg: Option<wgpu::BindGroup>,
    range_block_scan_bg: Option<wgpu::BindGroup>,
    range_write_bg: Option<wgpu::BindGroup>,
    shadow_partition_bg: Option<wgpu::BindGroup>,
    /// `(scratch_capacity, static_objects buffer identity, materials buffer
    /// identity)` -- any change forces every bind group above to rebuild
    /// (cheap; not a hot-path cost next to the compute work itself).
    bind_group_key: Option<(u32, Option<u64>, usize)>,

    readback: readback::RangeReadback,

    /// Fallback bound in `static_objects`'/`materials`' place before either
    /// has ever been resolved (mirrors every other SceneDB-direct pass's
    /// "bind *some* valid buffer so bind-group creation can't fail" fallback
    /// -- `capacity`/dispatch counts are all 0 whenever this is bound, so it
    /// is never actually dereferenced).
    fallback_buf: wgpu::Buffer,
}

impl ObjectBatchPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ObjectBatch Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/object_batch.wgsl").into()),
        });

        let cs = wgpu::ShaderStages::COMPUTE;
        let bgls = BindGroupLayouts {
            gather: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ObjBatch Gather BGL"),
                entries: &[
                    bgl_entry_uniform(0, cs),
                    bgl_entry_storage(1, cs, true),
                    bgl_entry_storage(2, cs, false),
                    bgl_entry_storage(3, cs, false),
                    bgl_entry_storage(4, cs, false),
                ],
            }),
            prepare: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ObjBatch Prepare BGL"),
                entries: &[
                    bgl_entry_storage(0, cs, true),
                    bgl_entry_storage(1, cs, false),
                    bgl_entry_storage(2, cs, false),
                ],
            }),
            histogram: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ObjBatch Histogram BGL"),
                entries: &[
                    bgl_entry_uniform(0, cs),
                    bgl_entry_uniform(1, cs),
                    bgl_entry_storage(2, cs, true),
                    bgl_entry_storage(3, cs, false),
                ],
            }),
            scan: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ObjBatch Scan BGL"),
                entries: &[bgl_entry_uniform(0, cs), bgl_entry_storage(1, cs, false)],
            }),
            scatter: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ObjBatch Scatter BGL"),
                entries: &[
                    bgl_entry_uniform(0, cs),
                    bgl_entry_uniform(1, cs),
                    bgl_entry_storage(2, cs, true),
                    bgl_entry_storage(3, cs, true),
                    bgl_entry_storage(4, cs, false),
                    bgl_entry_storage(5, cs, false),
                    bgl_entry_storage(6, cs, true),
                ],
            }),
            final_gather: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ObjBatch FinalGather BGL"),
                entries: &[
                    bgl_entry_uniform(0, cs),
                    bgl_entry_storage(1, cs, true),
                    bgl_entry_storage(2, cs, true),
                    bgl_entry_storage(3, cs, false),
                    bgl_entry_storage(4, cs, false),
                ],
            }),
            group_local_scan: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ObjBatch GroupLocalScan BGL"),
                entries: &[
                    bgl_entry_uniform(0, cs),
                    bgl_entry_storage(1, cs, true),
                    bgl_entry_storage(2, cs, true),
                    bgl_entry_storage(3, cs, false),
                    bgl_entry_storage(4, cs, false),
                ],
            }),
            group_block_scan: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ObjBatch GroupBlockScan BGL"),
                entries: &[
                    bgl_entry_uniform(0, cs),
                    bgl_entry_storage(1, cs, false),
                    bgl_entry_storage(2, cs, false),
                ],
            }),
            group_write: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ObjBatch GroupWrite BGL"),
                entries: &[
                    bgl_entry_uniform(0, cs),
                    bgl_entry_storage(1, cs, true),
                    bgl_entry_storage(2, cs, true),
                    bgl_entry_storage(3, cs, true),
                    bgl_entry_storage(4, cs, true),
                    bgl_entry_storage(5, cs, false),
                ],
            }),
            group_write_sentinel: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ObjBatch GroupWriteSentinel BGL"),
                entries: &[
                    bgl_entry_storage(0, cs, true),
                    bgl_entry_storage(1, cs, false),
                    bgl_entry_storage(2, cs, true),
                ],
            }),
            prepare_groups: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ObjBatch PrepareGroups BGL"),
                entries: &[bgl_entry_storage(0, cs, true), bgl_entry_storage(1, cs, false)],
            }),
            build_draw_calls: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ObjBatch BuildDrawCalls BGL"),
                entries: &[
                    bgl_entry_storage(0, cs, true),
                    bgl_entry_storage(1, cs, true),
                    bgl_entry_storage(2, cs, true),
                    bgl_entry_storage(3, cs, true),
                    bgl_entry_storage(4, cs, false),
                    bgl_entry_storage(5, cs, false),
                    bgl_entry_storage(6, cs, false),
                    bgl_entry_storage(7, cs, false),
                    bgl_entry_storage(8, cs, false),
                    bgl_entry_storage(9, cs, true),
                    bgl_entry_storage(10, cs, false),
                ],
            }),
            range_local_scan: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ObjBatch RangeLocalScan BGL"),
                entries: &[
                    bgl_entry_storage(0, cs, true),
                    bgl_entry_storage(1, cs, true),
                    bgl_entry_storage(2, cs, true),
                    bgl_entry_storage(3, cs, true),
                    bgl_entry_storage(4, cs, false),
                    bgl_entry_storage(5, cs, false),
                ],
            }),
            range_block_scan: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ObjBatch RangeBlockScan BGL"),
                entries: &[
                    bgl_entry_uniform(0, cs),
                    bgl_entry_storage(1, cs, false),
                    bgl_entry_storage(2, cs, false),
                ],
            }),
            range_write: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ObjBatch RangeWrite BGL"),
                entries: &[
                    bgl_entry_storage(0, cs, true),
                    bgl_entry_storage(1, cs, true),
                    bgl_entry_storage(2, cs, true),
                    bgl_entry_storage(3, cs, true),
                    bgl_entry_storage(4, cs, true),
                    bgl_entry_storage(5, cs, true),
                    bgl_entry_storage(6, cs, true),
                    bgl_entry_storage(7, cs, false),
                    bgl_entry_storage(8, cs, false),
                    bgl_entry_storage(9, cs, false),
                    bgl_entry_storage(10, cs, false),
                ],
            }),
            shadow_partition: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ObjBatch ShadowPartition BGL"),
                entries: &[
                    bgl_entry_uniform(0, cs),
                    bgl_entry_storage(1, cs, true),
                    bgl_entry_storage(2, cs, true),
                    bgl_entry_storage(3, cs, false),
                    bgl_entry_storage(4, cs, false),
                    bgl_entry_storage(5, cs, false),
                ],
            }),
        };

        let pipelines = Pipelines {
            gather: make_compute_pipeline(device, "ObjBatch Gather", &bgls.gather, &shader, "cs_gather"),
            prepare: make_compute_pipeline(device, "ObjBatch Prepare", &bgls.prepare, &shader, "cs_prepare"),
            histogram: make_compute_pipeline(device, "ObjBatch Histogram", &bgls.histogram, &shader, "cs_histogram"),
            scan: make_compute_pipeline(device, "ObjBatch Scan", &bgls.scan, &shader, "cs_scan"),
            scatter: make_compute_pipeline(device, "ObjBatch Scatter", &bgls.scatter, &shader, "cs_scatter"),
            final_gather: make_compute_pipeline(device, "ObjBatch FinalGather", &bgls.final_gather, &shader, "cs_final_gather"),
            group_local_scan: make_compute_pipeline(device, "ObjBatch GroupLocalScan", &bgls.group_local_scan, &shader, "cs_group_local_scan"),
            group_block_scan: make_compute_pipeline(device, "ObjBatch GroupBlockScan", &bgls.group_block_scan, &shader, "cs_group_block_scan"),
            group_write: make_compute_pipeline(device, "ObjBatch GroupWrite", &bgls.group_write, &shader, "cs_group_write"),
            group_write_sentinel: make_compute_pipeline(device, "ObjBatch GroupWriteSentinel", &bgls.group_write_sentinel, &shader, "cs_group_write_sentinel"),
            prepare_groups: make_compute_pipeline(device, "ObjBatch PrepareGroups", &bgls.prepare_groups, &shader, "cs_prepare_groups"),
            build_draw_calls: make_compute_pipeline(device, "ObjBatch BuildDrawCalls", &bgls.build_draw_calls, &shader, "cs_build_draw_calls"),
            range_local_scan: make_compute_pipeline(device, "ObjBatch RangeLocalScan", &bgls.range_local_scan, &shader, "cs_range_local_scan"),
            range_block_scan: make_compute_pipeline(device, "ObjBatch RangeBlockScan", &bgls.range_block_scan, &shader, "cs_range_block_scan"),
            range_write: make_compute_pipeline(device, "ObjBatch RangeWrite", &bgls.range_write, &shader, "cs_range_write"),
            shadow_partition: make_compute_pipeline(device, "ObjBatch ShadowPartition", &bgls.shadow_partition, &shader, "cs_shadow_partition"),
        };

        let sort_pass_uniforms: [wgpu::Buffer; SORT_BITS] = std::array::from_fn(|_| {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ObjBatch SortBit"),
                size: std::mem::size_of::<SortUniformsGpu>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            buf
        });

        let batch_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ObjBatch BatchUniforms"),
            size: std::mem::size_of::<BatchUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scratch_capacity = MIN_SCRATCH_CAPACITY;
        let scratch = ScratchBuffers::new(device, scratch_capacity);
        let draw_calls_out = create_storage_buffer(device, "ObjBatch DrawCalls", scratch_capacity as u64 * DRAW_CALL_BYTES);
        let indirect_out = create_storage_buffer(device, "ObjBatch Indirect", scratch_capacity as u64 * INDIRECT_ARGS_BYTES);

        let fallback_buf = create_dual_buffer(device, "ObjBatch Fallback", 16);

        Self {
            bgls,
            pipelines,
            sort_pass_uniforms,
            batch_uniform,
            draw_calls_out,
            indirect_out,
            scratch,
            scratch_capacity,
            gather_bg: None,
            prepare_bg: None,
            histogram_bgs: None,
            scan_bg: None,
            scatter_bgs: None,
            final_gather_bg: None,
            group_local_scan_bg: None,
            group_block_scan_bg: None,
            group_write_bg: None,
            group_write_sentinel_bg: None,
            prepare_groups_bg: None,
            build_draw_calls_bg: None,
            range_local_scan_bg: None,
            range_block_scan_bg: None,
            range_write_bg: None,
            shadow_partition_bg: None,
            bind_group_key: None,
            readback: readback::RangeReadback::new(),
            fallback_buf,
        }
    }

    /// Grows every scratch buffer to at least `needed` rows, next-power-of-
    /// two rounded (with `MIN_SCRATCH_CAPACITY` as a floor) -- same "grow on
    /// demand, no data preserved across growth" policy `corona`'s
    /// `upload_sort_steps` and every `GrowableBuffer` in this codebase
    /// already use. Returns `true` if it actually reallocated (the caller
    /// must then rebuild every bind group referencing these buffers).
    fn ensure_capacity(&mut self, device: &wgpu::Device, needed: u32) -> bool {
        if needed <= self.scratch_capacity {
            return false;
        }
        let new_capacity = needed.next_power_of_two().max(MIN_SCRATCH_CAPACITY);
        self.scratch = ScratchBuffers::new(device, new_capacity);
        self.draw_calls_out = create_storage_buffer(device, "ObjBatch DrawCalls", new_capacity as u64 * DRAW_CALL_BYTES);
        self.indirect_out = create_storage_buffer(device, "ObjBatch Indirect", new_capacity as u64 * INDIRECT_ARGS_BYTES);
        self.scratch_capacity = new_capacity;
        true
    }

    fn rebuild_bind_groups(&mut self, device: &wgpu::Device, static_objects: &wgpu::Buffer, materials: &wgpu::Buffer) {
        let s = &self.scratch;
        let cs = wgpu::ShaderStages::COMPUTE;
        let _ = cs;

        self.gather_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ObjBatch Gather BG"),
            layout: &self.bgls.gather,
            entries: &[
                bg_entry(0, &self.batch_uniform),
                bg_entry(1, static_objects),
                bg_entry(2, &s.keys_a),
                bg_entry(3, &s.indices_a),
                bg_entry(4, &s.gather_count),
            ],
        }));

        self.prepare_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ObjBatch Prepare BG"),
            layout: &self.bgls.prepare,
            entries: &[
                bg_entry(0, &s.gather_count),
                bg_entry(1, &s.frame_uniform),
                bg_entry(2, &s.dispatch_args),
            ],
        }));

        // Radix sort: 32 bind groups per kernel, ping-ponging src/dst each
        // bit -- identical structure to `helio-pass-sprite-cull`'s own
        // `hist_bind_groups`/`scatter_bind_groups` construction.
        self.histogram_bgs = Some(std::array::from_fn(|i| {
            let src_keys = if i % 2 == 0 { &s.keys_a } else { &s.keys_b };
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ObjBatch Histogram BG"),
                layout: &self.bgls.histogram,
                entries: &[
                    bg_entry(0, &self.sort_pass_uniforms[i]),
                    bg_entry(1, &s.frame_uniform),
                    bg_entry(2, src_keys),
                    bg_entry(3, &s.block_hist),
                ],
            })
        }));

        self.scan_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ObjBatch Scan BG"),
            layout: &self.bgls.scan,
            entries: &[bg_entry(0, &s.frame_uniform), bg_entry(1, &s.block_hist)],
        }));

        self.scatter_bgs = Some(std::array::from_fn(|i| {
            let (src_keys, src_indices, dst_keys, dst_indices) = if i % 2 == 0 {
                (&s.keys_a, &s.indices_a, &s.keys_b, &s.indices_b)
            } else {
                (&s.keys_b, &s.indices_b, &s.keys_a, &s.indices_a)
            };
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ObjBatch Scatter BG"),
                layout: &self.bgls.scatter,
                entries: &[
                    bg_entry(0, &self.sort_pass_uniforms[i]),
                    bg_entry(1, &s.frame_uniform),
                    bg_entry(2, src_keys),
                    bg_entry(3, src_indices),
                    bg_entry(4, dst_keys),
                    bg_entry(5, dst_indices),
                    bg_entry(6, &s.block_hist),
                ],
            })
        }));

        // 32 (even) passes -- sorted result ends back in A, same as
        // `helio-pass-sprite-cull`.
        self.final_gather_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ObjBatch FinalGather BG"),
            layout: &self.bgls.final_gather,
            entries: &[
                bg_entry(0, &s.frame_uniform),
                bg_entry(1, &s.indices_a),
                bg_entry(2, static_objects),
                bg_entry(3, &s.instances_out),
                bg_entry(4, &s.aabbs_out),
            ],
        }));

        self.group_local_scan_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ObjBatch GroupLocalScan BG"),
            layout: &self.bgls.group_local_scan,
            entries: &[
                bg_entry(0, &s.frame_uniform),
                bg_entry(1, &s.indices_a),
                bg_entry(2, static_objects),
                bg_entry(3, &s.local_group_rank),
                bg_entry(4, &s.block_group_totals),
            ],
        }));

        self.group_block_scan_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ObjBatch GroupBlockScan BG"),
            layout: &self.bgls.group_block_scan,
            entries: &[
                bg_entry(0, &s.frame_uniform),
                bg_entry(1, &s.block_group_totals),
                bg_entry(2, &s.group_count),
            ],
        }));

        self.group_write_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ObjBatch GroupWrite BG"),
            layout: &self.bgls.group_write,
            entries: &[
                bg_entry(0, &s.frame_uniform),
                bg_entry(1, &s.indices_a),
                bg_entry(2, static_objects),
                bg_entry(3, &s.local_group_rank),
                bg_entry(4, &s.block_group_totals),
                bg_entry(5, &s.group_starts),
            ],
        }));

        self.group_write_sentinel_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ObjBatch GroupWriteSentinel BG"),
            layout: &self.bgls.group_write_sentinel,
            entries: &[
                bg_entry(0, &s.frame_uniform),
                bg_entry(1, &s.group_starts),
                bg_entry(2, &s.group_count),
            ],
        }));

        self.prepare_groups_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ObjBatch PrepareGroups BG"),
            layout: &self.bgls.prepare_groups,
            entries: &[bg_entry(0, &s.group_count), bg_entry(1, &s.dispatch_args_groups)],
        }));

        self.build_draw_calls_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ObjBatch BuildDrawCalls BG"),
            layout: &self.bgls.build_draw_calls,
            entries: &[
                bg_entry(0, &s.group_count),
                bg_entry(1, &s.group_starts),
                bg_entry(2, &s.indices_a),
                bg_entry(3, static_objects),
                bg_entry(4, &self.draw_calls_out),
                bg_entry(5, &s.group_material_class),
                bg_entry(6, &s.group_graph_hash_lo),
                bg_entry(7, &s.group_graph_hash_hi),
                bg_entry(8, &s.group_shading),
                bg_entry(9, materials),
                bg_entry(10, &self.indirect_out),
            ],
        }));

        self.range_local_scan_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ObjBatch RangeLocalScan BG"),
            layout: &self.bgls.range_local_scan,
            entries: &[
                bg_entry(0, &s.group_count),
                bg_entry(1, &s.group_material_class),
                bg_entry(2, &s.group_graph_hash_lo),
                bg_entry(3, &s.group_graph_hash_hi),
                bg_entry(4, &s.local_range_rank),
                bg_entry(5, &s.block_range_totals),
            ],
        }));

        self.range_block_scan_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ObjBatch RangeBlockScan BG"),
            layout: &self.bgls.range_block_scan,
            entries: &[
                bg_entry(0, &s.dispatch_args_groups),
                bg_entry(1, &s.block_range_totals),
                bg_entry(2, &s.range_count),
            ],
        }));

        self.range_write_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ObjBatch RangeWrite BG"),
            layout: &self.bgls.range_write,
            entries: &[
                bg_entry(0, &s.group_count),
                bg_entry(1, &s.group_material_class),
                bg_entry(2, &s.group_graph_hash_lo),
                bg_entry(3, &s.group_graph_hash_hi),
                bg_entry(4, &s.group_shading),
                bg_entry(5, &s.local_range_rank),
                bg_entry(6, &s.block_range_totals),
                bg_entry(7, &s.opaque_ranges),
                bg_entry(8, &s.transparent_ranges),
                bg_entry(9, &s.forward_ranges),
                bg_entry(10, &s.range_bucket_counts),
            ],
        }));

        self.shadow_partition_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ObjBatch ShadowPartition BG"),
            layout: &self.bgls.shadow_partition,
            entries: &[
                bg_entry(0, &s.frame_uniform),
                bg_entry(1, &s.indices_a),
                bg_entry(2, static_objects),
                bg_entry(3, &s.shadow_static_indirect),
                bg_entry(4, &s.shadow_movable_indirect),
                bg_entry(5, &s.shadow_counts),
            ],
        }));
    }

    /// Records the full 8-stage dispatch sequence. Shared by the graph-
    /// integrated `execute()` and [`Self::run_once_for_testing`] -- same
    /// "test records identically to the real path" discipline `helio-pass-
    /// sprite-cull`'s `record`/`run_once_for_testing` split uses.
    fn record(&self, encoder: &mut wgpu::CommandEncoder, capacity: u32) {
        encoder.clear_buffer(&self.scratch.gather_count, 0, None);
        encoder.clear_buffer(&self.scratch.group_count, 0, None);
        encoder.clear_buffer(&self.scratch.range_count, 0, None);
        encoder.clear_buffer(&self.scratch.range_bucket_counts, 0, None);
        encoder.clear_buffer(&self.scratch.shadow_counts, 0, None);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ObjBatch Gather"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.gather);
            pass.set_bind_group(0, self.gather_bg.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(capacity.div_ceil(WG).max(1), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ObjBatch Prepare"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.prepare);
            pass.set_bind_group(0, self.prepare_bg.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        let hist_bgs = self.histogram_bgs.as_ref().unwrap();
        let scatter_bgs = self.scatter_bgs.as_ref().unwrap();
        for i in 0..SORT_BITS {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ObjBatch Histogram"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.histogram);
                pass.set_bind_group(0, &hist_bgs[i], &[]);
                pass.dispatch_workgroups_indirect(&self.scratch.dispatch_args, 0);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ObjBatch Scan"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.scan);
                pass.set_bind_group(0, self.scan_bg.as_ref().unwrap(), &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ObjBatch Scatter"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.scatter);
                pass.set_bind_group(0, &scatter_bgs[i], &[]);
                pass.dispatch_workgroups_indirect(&self.scratch.dispatch_args, 0);
            }
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ObjBatch FinalGather"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.final_gather);
            pass.set_bind_group(0, self.final_gather_bg.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups_indirect(&self.scratch.dispatch_args, 0);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ObjBatch GroupLocalScan"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.group_local_scan);
            pass.set_bind_group(0, self.group_local_scan_bg.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups_indirect(&self.scratch.dispatch_args, 0);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ObjBatch GroupBlockScan"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.group_block_scan);
            pass.set_bind_group(0, self.group_block_scan_bg.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ObjBatch GroupWrite"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.group_write);
            pass.set_bind_group(0, self.group_write_bg.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups_indirect(&self.scratch.dispatch_args, 0);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ObjBatch GroupWriteSentinel"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.group_write_sentinel);
            pass.set_bind_group(0, self.group_write_sentinel_bg.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ObjBatch PrepareGroups"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.prepare_groups);
            pass.set_bind_group(0, self.prepare_groups_bg.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ObjBatch BuildDrawCalls"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.build_draw_calls);
            pass.set_bind_group(0, self.build_draw_calls_bg.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups_indirect(&self.scratch.dispatch_args_groups, 0);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ObjBatch RangeLocalScan"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.range_local_scan);
            pass.set_bind_group(0, self.range_local_scan_bg.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups_indirect(&self.scratch.dispatch_args_groups, 0);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ObjBatch RangeBlockScan"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.range_block_scan);
            pass.set_bind_group(0, self.range_block_scan_bg.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ObjBatch RangeWrite"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.range_write);
            pass.set_bind_group(0, self.range_write_bg.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups_indirect(&self.scratch.dispatch_args_groups, 0);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ObjBatch ShadowPartition"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.shadow_partition);
            pass.set_bind_group(0, self.shadow_partition_bg.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups_indirect(&self.scratch.dispatch_args, 0);
        }
    }

    /// GPU-produced, sorted-order instance buffer -- same shape as
    /// `libhelio::GpuInstanceData`.
    pub fn instances_buffer(&self) -> &wgpu::Buffer {
        &self.scratch.instances_out
    }
    pub fn aabbs_buffer(&self) -> &wgpu::Buffer {
        &self.scratch.aabbs_out
    }
    pub fn draw_calls_buffer(&self) -> &wgpu::Buffer {
        &self.draw_calls_out
    }
    /// `wgpu`-ABI indirect draw args, one per draw-call group, same order
    /// as [`Self::draw_calls_buffer`] -- what `multi_draw_indexed_indirect`
    /// actually reads.
    pub fn indirect_buffer(&self) -> &wgpu::Buffer {
        &self.indirect_out
    }
    pub fn shadow_static_indirect_buffer(&self) -> &wgpu::Buffer {
        &self.scratch.shadow_static_indirect
    }
    pub fn shadow_movable_indirect_buffer(&self) -> &wgpu::Buffer {
        &self.scratch.shadow_movable_indirect
    }
    /// Raw buffers for direct, synchronous verification (integration tests
    /// only -- production code should read [`Self::opaque_ranges`]/
    /// [`Self::transparent_ranges`]/[`Self::forward_ranges`]/[`Self::
    /// counts`] instead, which are the async-readback-backed, non-stalling
    /// versions of the same data).
    /// Per-group shading classification bits and material_class -- used by
    /// `tests/gpu_object_batch_validation.rs`'s `OBJ_BATCH_DEBUG`-gated
    /// diagnostics when a correctness assertion needs more detail than the
    /// production getters expose.
    #[doc(hidden)]
    pub fn debug_group_shading_buffer(&self) -> &wgpu::Buffer {
        &self.scratch.group_shading
    }
    #[doc(hidden)]
    pub fn debug_group_material_class_buffer(&self) -> &wgpu::Buffer {
        &self.scratch.group_material_class
    }
    pub fn group_count_buffer(&self) -> &wgpu::Buffer {
        &self.scratch.group_count
    }
    pub fn range_bucket_counts_buffer(&self) -> &wgpu::Buffer {
        &self.scratch.range_bucket_counts
    }
    pub fn shadow_counts_buffer(&self) -> &wgpu::Buffer {
        &self.scratch.shadow_counts
    }
    pub fn opaque_ranges_buffer(&self) -> &wgpu::Buffer {
        &self.scratch.opaque_ranges
    }
    pub fn transparent_ranges_buffer(&self) -> &wgpu::Buffer {
        &self.scratch.transparent_ranges
    }
    pub fn forward_ranges_buffer(&self) -> &wgpu::Buffer {
        &self.scratch.forward_ranges
    }

    /// Last-completed-frame's opaque/transparent/forward range tables --
    /// one frame of latency behind the GPU work that produced them (see
    /// `readback` module doc). Empty until the first successful readback.
    pub fn opaque_ranges(&self) -> &[RangeTuple] {
        self.readback.opaque()
    }
    pub fn transparent_ranges(&self) -> &[RangeTuple] {
        self.readback.transparent()
    }
    pub fn forward_ranges(&self) -> &[RangeTuple] {
        self.readback.forward()
    }
    /// Last-completed-frame's live draw-call-group count and shadow static/
    /// movable draw counts -- `(draw_count, shadow_static, shadow_movable)`.
    pub fn counts(&self) -> (u32, u32, u32) {
        self.readback.counts()
    }
    /// Live instance count -- same value as [`Self::instances_buffer`]'s
    /// valid prefix length.
    pub fn instance_count(&self) -> u32 {
        self.readback.instance_count()
    }
    /// Static-shadow-atlas cache invalidation signal -- bumps whenever the
    /// static (non-`INSTANCE_FLAG_MOVABLE`) object set's size last changed,
    /// same role `GpuScene::static_objects_generation` used to serve.
    pub fn shadow_static_generation(&self) -> u64 {
        self.readback.shadow_static_generation()
    }

    /// Runs the full pipeline once, outside a `RenderGraph`, blocking until
    /// the GPU is done -- for integration tests that need to read back
    /// results without standing up a full graph + `PassContext`. Mirrors
    /// `helio-pass-sprite-cull::SpriteCullPass::run_once_for_testing`.
    pub fn run_once_for_testing(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        static_objects: &wgpu::Buffer,
        materials: &wgpu::Buffer,
        capacity: u32,
    ) {
        self.ensure_capacity(device, capacity);
        self.rebuild_bind_groups(device, static_objects, materials);
        queue.write_buffer(
            &self.batch_uniform,
            0,
            bytemuck::bytes_of(&BatchUniforms {
                capacity,
                max_objects: self.scratch_capacity,
                max_groups: self.scratch_capacity,
                max_ranges: self.scratch_capacity,
            }),
        );
        for (i, buf) in self.sort_pass_uniforms.iter().enumerate() {
            queue.write_buffer(buf, 0, bytemuck::bytes_of(&SortUniformsGpu { bit: i as u32 }));
        }
        // `RangeBlockUniform` reuses `dispatch_args_groups[0]` directly (the
        // dual-bind trick) -- nothing to write here.

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ObjBatch Test Encoder"),
        });
        self.record(&mut encoder, capacity);
        queue.submit([encoder.finish()]);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
    }
}

impl RenderPass for ObjectBatchPass {
    fn name(&self) -> &'static str {
        "ObjectBatch"
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None // Compute-only pass -- no render pass.
    }

    fn writes(&self) -> &'static [&'static str] {
        &["object_batch"]
    }

    fn declare_resources(&self, builder: &mut helio_core::graph::ResourceBuilder) {
        builder.write_buffer("object_batch");
    }

    fn publish<'a>(&'a self, frame: &mut libhelio::FrameResources<'a>) {
        let (draw_count, shadow_static_draw_count, shadow_movable_draw_count) = self.counts();
        frame.object_batch.write(
            libhelio::ObjectBatchFrameData {
                instances: &self.scratch.instances_out,
                aabbs: &self.scratch.aabbs_out,
                draw_calls: &self.draw_calls_out,
                indirect: &self.indirect_out,
                draw_count,
                instance_count: self.instance_count(),
                opaque_ranges: self.opaque_ranges(),
                transparent_ranges: self.transparent_ranges(),
                forward_ranges: self.forward_ranges(),
                shadow_static_indirect: &self.scratch.shadow_static_indirect,
                shadow_static_draw_count,
                shadow_movable_indirect: &self.scratch.shadow_movable_indirect,
                shadow_movable_draw_count,
                shadow_static_generation: self.shadow_static_generation(),
            },
            "ObjectBatch",
        );
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let static_objects_handle = ctx.scene_buffers.get(BufferKey::of("static_objects"));
        let capacity = static_objects_handle
            .as_ref()
            .map(|h| (h.buffer.size() / std::mem::size_of::<StaticObjectComponent>() as u64) as u32)
            .unwrap_or(0);

        // Growth (a `&mut self` borrow) must fully finish before we take any
        // reference into `self` (the fallback buffer) below.
        let grew = self.ensure_capacity(ctx.device, capacity.max(1));

        // Cloned (a cheap resource-handle clone, not a data copy -- same as
        // `BufferHandle::buffer`'s own "owned clone" doc) rather than
        // borrowed, so it doesn't tie up `self` across the `&mut self`
        // calls below. Change-detection uses the handle's own `epoch`
        // (bumped on every reallocation) rather than a pointer into this
        // freshly-cloned local, which would have a different stack address
        // every single frame and defeat the cache entirely.
        let static_objects_epoch = static_objects_handle.as_ref().map(|h| h.epoch);
        let static_objects_buf: wgpu::Buffer = static_objects_handle
            .as_ref()
            .map(|h| h.buffer.clone())
            .unwrap_or_else(|| self.fallback_buf.clone());
        let materials_buf = ctx
            .frame_resources
            .materials
            .get()
            .map(|m| m.materials)
            .unwrap_or(&static_objects_buf);
        let key = (
            self.scratch_capacity,
            static_objects_epoch,
            materials_buf as *const _ as usize,
        );
        if grew || self.bind_group_key != Some(key) {
            self.rebuild_bind_groups(ctx.device, &static_objects_buf, materials_buf);
            self.bind_group_key = Some(key);
        }

        ctx.write_buffer(
            &self.batch_uniform,
            0,
            bytemuck::bytes_of(&BatchUniforms {
                capacity,
                max_objects: self.scratch_capacity,
                max_groups: self.scratch_capacity,
                max_ranges: self.scratch_capacity,
            }),
        );
        for (i, buf) in self.sort_pass_uniforms.iter().enumerate() {
            ctx.write_buffer(buf, 0, bytemuck::bytes_of(&SortUniformsGpu { bit: i as u32 }));
        }

        self.readback.poll_and_kick_off(ctx.device, ctx.queue, &self.scratch);
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let capacity = {
            let handle = ctx.scene_buffers.get(BufferKey::of("static_objects"));
            handle
                .map(|h| (h.buffer.size() / std::mem::size_of::<StaticObjectComponent>() as u64) as u32)
                .unwrap_or(0)
        };
        self.record(unsafe { &mut *ctx.encoder_ptr }, capacity);
        Ok(())
    }
}

// Compile-time layout sanity: keeps `object_batch.wgsl`'s `StaticObjectRow`
// and this crate's assumed size in lock-step with the real
// `StaticObjectComponent` -- a drift here fails `cargo test`, not a draw.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_object_component_size_matches_wgsl_mirror() {
        // 4×u32 (16) + transform(64) + prev_transform(64) + normal_mat(48)
        // + bounds(16) + 7×u32/i32/u32 (28) = 236 bytes -- see `object_batch
        // .wgsl`'s `StaticObjectRow` doc.
        assert_eq!(std::mem::size_of::<StaticObjectComponent>(), 236);
    }

    #[test]
    fn output_struct_sizes_match_wgsl() {
        assert_eq!(INSTANCE_BYTES, 208);
        assert_eq!(AABB_BYTES, 16);
        assert_eq!(DRAW_CALL_BYTES, 20);
        assert_eq!(RANGE_BYTES, 20);
        assert_eq!(INDIRECT_ARGS_BYTES, 20);
    }
}
