//! GPU-driven static-object batch: the sorted instance/draw-call/range/
//! shadow-partition data every geometry-drawing pass needs, all derived
//! fresh each frame from SceneDB's [`crate::StaticObjectComponent`] rows with
//! zero per-frame CPU iteration.
//!
//! Produced by `helio-pass-object-batch`'s `ObjectBatchPass`, consumed by
//! every pass that used to read the equivalent fields directly off the old
//! central scene storage (`helio-pass-gbuffer`, `helio-pass-occlusion-cull`,
//! `helio-pass-indirect-dispatch`, `helio-pass-shadow` and its
//! `-cull`/`-dirty` siblings, `helio-pass-transparent`, `helio-pass-
//! forward-lit`, `helio-pass-depth-prepass`, `helio-pass-portal-cull`/
//! `-instances`).
//!
//! Defined here rather than in `helio-pass-object-batch` itself because that
//! crate already depends on `helio-pass-gbuffer` for [`crate::
//! StaticObjectComponent`]'s schema; putting the frame-data type in the
//! opposite direction would be a dependency cycle. `helio-pass-gbuffer`
//! already plays this "shared contract" role for other cross-pass shapes
//! (`helio-pass-transparent`, `helio-pass-object-batch`, and `helio-pass-
//! portal-instances` all already depend on it), so this is consistent with
//! the existing shape rather than a new pattern.
//!
//! `opaque_ranges`/`transparent_ranges`/`forward_ranges` are a small,
//! bounded, ASYNC (one-frame-latency) CPU readback -- see `helio-pass-
//! object-batch`'s `readback` module doc for why that's the correct
//! tradeoff for exactly this one piece of the pipeline's output (PSO
//! selection needs `(start, count)` as plain `u32`s on the CPU before
//! `multi_draw_indexed_indirect` can be recorded; nothing else here is a
//! CPU readback of any kind).
#[derive(Clone, Copy)]
pub struct ObjectBatchFrameData<'a> {
    /// Sorted-order instance data (`helio_pass_object_batch::GpuInstanceData` layout).
    pub instances: &'a wgpu::Buffer,
    /// Sorted-order bounding spheres, same order as `instances`.
    pub aabbs: &'a wgpu::Buffer,
    /// One entry per draw-call group (`helio_pass_object_batch::GpuDrawCall`
    /// layout) -- used by culling passes that need
    /// `index_count`/`first_index`/`vertex_offset` directly, not just the
    /// hardware indirect-draw ABI.
    pub draw_calls: &'a wgpu::Buffer,
    /// Same per-group data as `draw_calls`, reordered to wgpu's hardware
    /// indirect-draw ABI -- what `multi_draw_indexed_indirect` actually
    /// reads.
    pub indirect: &'a wgpu::Buffer,
    /// Live draw-call group count this frame.
    pub draw_count: u32,
    /// Live instance count this frame (== `instances`'s valid prefix length).
    pub instance_count: u32,
    /// `(material_class, graph_hash, start, count)` ranges over `draw_calls`
    /// -- `start`/`count` index `draw_calls`/`indirect` directly, not
    /// `instances`. One `multi_draw_indexed_indirect` call per range.
    pub opaque_ranges: &'a [(u32, u64, u32, u32)],
    pub transparent_ranges: &'a [(u32, u64, u32, u32)],
    pub forward_ranges: &'a [(u32, u64, u32, u32)],
    /// One-instance indirect draw args per static (non-movable) object,
    /// for the static shadow atlas.
    pub shadow_static_indirect: &'a wgpu::Buffer,
    pub shadow_static_draw_count: u32,
    /// Same, for movable objects (the dynamic shadow atlas).
    pub shadow_movable_indirect: &'a wgpu::Buffer,
    pub shadow_movable_draw_count: u32,
    /// Bumps whenever the static object set's size last changed -- see
    /// `ObjectBatchPass::shadow_static_generation`'s doc. `helio-pass-
    /// shadow`'s static-atlas cache invalidation signal.
    pub shadow_static_generation: u64,
}
