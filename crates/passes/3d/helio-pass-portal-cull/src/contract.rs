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

/// Deepest a portal chain (see [`GpuPortalChain`]) can go. This is the whole
/// mechanism behind portals reflecting each other automatically: content is
/// mapped through *chains* of portals, not just one at a time, so a portal
/// facing another (or itself, or a loop of several) shows real recursive
/// depth with zero manual authoring. 3 is a deliberately modest default —
/// chain count grows as `portal_count^depth`, and 3 already reads as
/// "infinite" to the eye for a small handful of portals (a 4th bounce is
/// usually too small/dim to distinguish from ambient falloff anyway). Raise
/// it if a scene's portals are large enough that a 4th bounce is legible.
pub const MAX_CHAIN_DEPTH: usize = 3;

/// Hard cap on how many chains `helio::scene::portals` will ever generate,
/// regardless of `portal_count`/`MAX_CHAIN_DEPTH`. Scenes with more active
/// portals than this comfortably supports at the configured depth degrade
/// gracefully (fewer distinct reflection paths get drawn) rather than
/// growing GPU buffers unboundedly — portal counts are meant to stay small
/// (see the module doc above): `6` portals at the default depth `3` uses
/// 258 of these, leaving comfortable headroom without being wasteful.
///
/// This is *not* itself a large allocation (300 × 16 bytes is nothing) —
/// what it also sizes is `helio-pass-portal-cull`'s/`helio-pass-portal-
/// instances`' fixed per-chain-slot *culling output* buffers (which
/// instances survived, per chain — proportional to scene complexity × chain
/// count, unlike this list of plain index sequences). Keep this and those
/// crates' own per-chain capacity constants deliberately modest and sized
/// together, since their product is what actually gets allocated — the old
/// pre-chain system's capacities were generous because they were multiplied
/// by a handful of portal slots; blindly reusing those same numbers here,
/// multiplied by chain slots instead, would have been a real waste.
pub const MAX_PORTAL_CHAINS: usize = 300;

/// One active portal's render data. 144 bytes.
///
/// # WGSL equivalent
/// ```wgsl
/// struct GpuPortalView {
///     transform:         mat4x4<f32>,  // 64 bytes
///     inverse_transform: mat4x4<f32>,  // 64 bytes
///     half_extent:       vec2<f32>,    // 8 bytes
///     coordinate_space:  u32,          // 4 bytes
///     _pad:               u32,          // 4 bytes
/// }
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuPortalView {
    /// Portal-local → world (this portal surface's own pose, `pair.a.transform`).
    /// Used by `helio-pass-portal-mask` to place the opening quad it stamps
    /// into the screen-space visibility mask — see that crate's docs.
    pub transform: [f32; 16],

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

/// One valid portal chain — a sequence of up to [`MAX_CHAIN_DEPTH`] portal
/// indices (indices into the `portal_views` array), `portals[0]` being the
/// *outermost* one (the real, physical surface the main camera actually
/// looks through) and `portals[depth-1]` the innermost/deepest reflection.
/// 16 bytes at the default `MAX_CHAIN_DEPTH = 3`.
///
/// `helio::scene::portals` generates every such sequence (including repeats
/// — `[P, P, P]` is exactly "look through this portal at its own reflection,
/// three times over", the case that makes a single mirror-pair or a
/// self-facing room read as infinite) whenever the portal set changes. Both
/// `helio-pass-portal-cull` and `helio-pass-portal-instances` iterate this
/// list instead of `portal_views` directly, treating a depth-1 chain
/// (`depth == 1`) as exactly the old single-portal behavior — the chain
/// mechanism is a strict generalization, not a separate code path.
///
/// # WGSL equivalent
/// ```wgsl
/// struct GpuPortalChain {
///     portals: array<u32, 3>,  // MAX_CHAIN_DEPTH — bump both in lockstep
///     depth:   u32,
/// }
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuPortalChain {
    /// Portal indices, outermost (`[0]`) to innermost. Entries at or beyond
    /// `depth` are unused padding (always written as 0, never read).
    pub portals: [u32; MAX_CHAIN_DEPTH],

    /// How many of `portals` are valid, `1..=MAX_CHAIN_DEPTH`.
    pub depth: u32,
}
