//! GPU-facing per-portal render data.
//!
//! Published by `helio::Scene::flush()` from its private portal registry
//! (`scene::portals`) into a small storage buffer every frame — there are
//! never more than a handful of active portals, so republishing the whole
//! list unconditionally is simpler than dirty-tracking it and costs nothing
//! measurable. Consumed by `helio-pass-portal-cull` (frustum test to select
//! which instances get a duplicate draw) and `helio-pass-portal-instances`
//! (the duplicate draw itself, clipped to the portal's opening).

use bytemuck::{Pod, Zeroable};

/// One active portal's render data. 80 bytes.
///
/// # WGSL equivalent
/// ```wgsl
/// struct GpuPortalView {
///     inverse_transform: mat4x4<f32>,  // 64 bytes
///     half_extent:       vec2<f32>,    // 8 bytes
///     coordinate_space:  u32,          // 4 bytes
///     _pad:               u32,          // 4 bytes
/// }
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuPortalView {
    /// World → portal-local (this portal surface's own inverse transform).
    /// Used by the fragment-shader clip test: a duplicated fragment is kept
    /// only when its world position maps within `half_extent` of local X/Y
    /// and in front of the surface (local Z <= 0).
    pub inverse_transform: [f32; 16],

    /// Half-extent of the portal opening, in its own local X/Y.
    pub half_extent: [f32; 2],

    /// Index into `coordinate_spaces[]` (see `crate::coordinate_space`) —
    /// holds this portal's `pair_map_inverse`, the rigid transform that
    /// places content actually near the portal's other side where it should
    /// appear when seen through this side.
    pub coordinate_space: u32,

    pub _pad: u32,
}
