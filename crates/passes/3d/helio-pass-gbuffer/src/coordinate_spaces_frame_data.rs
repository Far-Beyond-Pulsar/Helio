//! Coordinate-space transforms (portals + sublevels) for this frame -- still
//! written directly by the `Renderer` today. A real pass-owned relocation
//! (reading `SublevelComponent` from SceneDB directly, owned by whichever
//! pass ends up assembling the portal/sublevel registry -- `helio-pass-
//! portal-cull` is the natural candidate) is still-pending future work, not
//! solved by this type's move out of the old central crate -- see the
//! migration doc's known-gaps section.
//!
//! Defined here rather than in `helio-pass-portal-cull` for the same reason
//! as [`crate::ObjectBatchFrameData`]/[`crate::CulledBatchFrameData`]:
//! `helio-pass-portal-cull` already depends on `helio-pass-gbuffer` (for
//! `ObjectBatchFrameData`), so the opposite direction would cycle.
#[derive(Clone, Copy)]
pub struct CoordinateSpacesFrameData<'a> {
    pub coordinate_spaces: &'a wgpu::Buffer,
    pub coordinate_spaces_prev: &'a wgpu::Buffer,
}
