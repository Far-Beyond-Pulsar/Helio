//! The final, fully-culled (frustum + Hi-Z occlusion) indirect draw args and
//! compacted instance indices -- what every pass that actually issues
//! `multi_draw_indexed_indirect` calls (`helio-pass-gbuffer`, `helio-pass-
//! shadow` and its `-cull`/`-dirty` siblings, `helio-pass-transparent`,
//! `helio-pass-forward-lit`, `helio-pass-depth-prepass`, `helio-pass-
//! portal-cull`/`-instances`) should read. Produced by `helio-pass-
//! occlusion-cull`'s `OcclusionCullPass` from `helio-pass-indirect-
//! dispatch`'s `IndirectDispatchFrameData`.
//!
//! Defined here for the same reason as [`crate::ObjectBatchFrameData`]:
//! `helio-pass-occlusion-cull` already depends on `helio-pass-gbuffer` (for
//! `ObjectBatchFrameData`), so putting this type in the opposite direction
//! would be a dependency cycle.
#[derive(Clone, Copy)]
pub struct CulledBatchFrameData<'a> {
    /// Per-group indirect draw args, `instance_count` replaced with each
    /// group's final (frustum + occlusion) surviving count.
    pub indirect: &'a wgpu::Buffer,
    /// Final surviving instance slots, packed per draw-call group -- index
    /// `instances` (from [`crate::ObjectBatchFrameData`]) through this.
    pub compacted_indices: &'a wgpu::Buffer,
}
