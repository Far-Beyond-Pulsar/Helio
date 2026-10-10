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
//! bounded, ASYNC CPU readback that trails the GPU by two or more frames (see
//! `helio-pass-object-batch`'s `readback` module doc). Draw passes don't take
//! offsets from them: they only tell `helio-pass-occlusion-cull` which
//! material keys exist, and it gives each key a fixed draw segment the GPU
//! fills in the same frame (see [`crate::DrawSegments`]).
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
    /// as read back, frames late -- `start`/`count` index `draw_calls`/
    /// `indirect` directly, not `instances`. Used to find the material keys
    /// present; draws use [`crate::DrawSegments`].
    pub opaque_ranges: &'a [(u32, u64, u32, u32)],
    pub transparent_ranges: &'a [(u32, u64, u32, u32)],
    pub forward_ranges: &'a [(u32, u64, u32, u32)],
    /// Current-frame GPU range tables, packed in deterministic sorted order.
    pub opaque_ranges_gpu: &'a wgpu::Buffer,
    pub transparent_ranges_gpu: &'a wgpu::Buffer,
    pub forward_ranges_gpu: &'a wgpu::Buffer,
    /// Counts at words 0..3 and three indirect dispatch argument blocks after
    /// them. Used by GPU range compaction; no CPU range-count upload needed.
    pub range_counts_gpu: &'a wgpu::Buffer,
    /// GPU-writable counts, including the per-range count regions.
    pub draw_counts_gpu: &'a wgpu::Buffer,
    /// One-instance indirect draw args per static (non-movable) object,
    /// for the static shadow atlas.
    pub shadow_static_indirect: &'a wgpu::Buffer,
    pub shadow_static_draw_count: u32,
    /// Same, for movable objects (the dynamic shadow atlas).
    pub shadow_movable_indirect: &'a wgpu::Buffer,
    pub shadow_movable_draw_count: u32,
    /// Static transparent-only objects: the shadow pass renders these into
    /// the coloured transmittance layer instead of the depth atlas.
    pub shadow_transmissive_indirect: &'a wgpu::Buffer,
    pub shadow_transmissive_draw_count: u32,
    /// Bumps whenever the static object set's size last changed -- see
    /// `ObjectBatchPass::shadow_static_generation`'s doc. `helio-pass-
    /// shadow`'s static-atlas cache invalidation signal.
    pub shadow_static_generation: u64,
    /// The draw counts above as a GPU `u32` array, for
    /// `multi_draw_indexed_indirect_count` (Helio#306); `None` when the
    /// device lacks `MULTI_DRAW_INDIRECT_COUNT`. Layout: `draw_count`,
    /// `shadow_static_draw_count`, `shadow_movable_draw_count`,
    /// `shadow_transmissive_draw_count`, then one count per entry of
    /// `opaque_ranges`, `transparent_ranges` and `forward_ranges` in that
    /// order. Range counts use three fixed-capacity regions; read them through
    /// the `*_count_slot` methods.
    pub draw_counts: Option<&'a wgpu::Buffer>,
    /// Capacity of one shading bucket's range-count region.
    pub range_slot_capacity: u32,
}

impl<'a> ObjectBatchFrameData<'a> {
    fn draw_count_slot(&self, index: usize) -> Option<crate::GpuDrawCount<'a>> {
        self.draw_counts.map(|buffer| crate::GpuDrawCount {
            buffer,
            offset: index as u64 * 4,
        })
    }

    /// GPU count for drawing all `draw_count` groups of `indirect`.
    pub fn all_draws_count_slot(&self) -> Option<crate::GpuDrawCount<'a>> {
        self.draw_count_slot(0)
    }
    pub fn shadow_static_count_slot(&self) -> Option<crate::GpuDrawCount<'a>> {
        self.draw_count_slot(1)
    }
    pub fn shadow_movable_count_slot(&self) -> Option<crate::GpuDrawCount<'a>> {
        self.draw_count_slot(2)
    }
    pub fn shadow_transmissive_count_slot(&self) -> Option<crate::GpuDrawCount<'a>> {
        self.draw_count_slot(3)
    }
    /// GPU count for `opaque_ranges[range]`.
    pub fn opaque_range_count_slot(&self, range: usize) -> Option<crate::GpuDrawCount<'a>> {
        self.draw_count_slot(4 + range)
    }
    /// GPU count for `transparent_ranges[range]`.
    pub fn transparent_range_count_slot(&self, range: usize) -> Option<crate::GpuDrawCount<'a>> {
        self.draw_count_slot(4 + self.range_slot_capacity as usize + range)
    }
    /// GPU count for `forward_ranges[range]`.
    pub fn forward_range_count_slot(&self, range: usize) -> Option<crate::GpuDrawCount<'a>> {
        self.draw_count_slot(4 + self.range_slot_capacity as usize * 2 + range)
    }
}
