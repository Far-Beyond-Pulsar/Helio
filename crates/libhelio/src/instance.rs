//! GPU instance data for GPU-driven indirect rendering.
//!
//! All geometry in the scene is submitted as a flat array of `GpuInstanceData`.
//! The GPU culling compute shaders read this array and emit `DrawIndexedIndirect`
//! commands — the CPU never iterates the draw list.

use bytemuck::{Pod, Zeroable};

// ── Instance flags (`GpuInstanceData::flags`) ───────────────────────────────
//
// Distinct from the `FLAG_*` constants in `material`, which live in
// `GpuMaterial::flags`. Same names, different field — check which one you are setting.

/// This instance contributes to the shadow atlas.
pub const INSTANCE_FLAG_CASTS_SHADOW: u32 = 1 << 0;

/// This instance receives shadows.
pub const INSTANCE_FLAG_RECEIVES_SHADOW: u32 = 1 << 1;

/// Skip GPU culling for this instance: it is always considered visible.
///
/// Both the frustum test in `indirect_dispatch.wgsl` and the Hi-Z occlusion test in
/// `occlusion_cull.wgsl` pass unconditionally when this bit is set.
///
/// # When this is the right answer
///
/// Culling in this engine is driven by a single world-space bounding **sphere** per
/// instance ([`GpuInstanceData::bounds`]). That representation degrades badly for very
/// large or very flat geometry: a ground plane's sphere has a radius set by its diagonal,
/// so it is enormous relative to the geometry actually inside it, and it both fails to
/// cull anything useful *and* is easy to get wrong in the direction that deletes visible
/// geometry. For a handful of such objects — ground planes, skyboxes, an interior shell
/// the camera lives inside — testing them at all is worth less than the risk of testing
/// them wrongly.
///
/// # When it is not
///
/// This is an escape hatch, not a fix. An instance carrying this flag is submitted every
/// frame no matter where the camera looks, so setting it broadly gives back exactly the
/// GPU-driven culling this engine exists to do. If you find yourself setting it on many
/// objects, the real answer is almost always to split the geometry into pieces whose
/// bounding spheres are tight.
pub const INSTANCE_FLAG_ALWAYS_VISIBLE: u32 = 1 << 2;

/// This instance's real geometry is suppressed from the main-scene cull and the
/// shadow cull.
///
/// Set on every instance whose owning group is an *active* sublevel (see
/// `helio::Scene::add_sublevel`). Sublevel members carry sublevel-**local**
/// transforms in [`GpuInstanceData::model`] — rendering them through the main
/// camera or the shadow atlas would draw them at the wrong (local, unplaced)
/// position. `SecondaryGBufferPass` renders them instead, through a camera
/// slot that bakes the sublevel's placement in, and `ProxyCompositePass`
/// merges the result into the main G-buffer at the correct placed position.
///
/// Checked *before*, and overriding, [`INSTANCE_FLAG_ALWAYS_VISIBLE`] in both
/// `indirect_dispatch.wgsl::test_instance` and
/// `shadow_cull.wgsl` — an instance can be "always visible to any camera that
/// would otherwise cull it" and "hidden from the main pass because it lives in
/// a sublevel" at the same time; hidden must win, or a sublevel's interior
/// would double-render (once correctly placed via the composite, once
/// incorrectly at its raw local coordinates via the main pass).
pub const INSTANCE_FLAG_SUBLEVEL_HIDDEN: u32 = 1 << 3;

/// This instance is invisible to the main-scene cull (never appears in the
/// G-buffer) but is **not** excluded from the shadow cull, so it still casts.
///
/// Reserved for a sublevel's coarse shadow-proxy volume (design doc §10
/// "Shadows": *"Sublevels... cast via a single proxy volume per sublevel...
/// the same proxy-mesh double publication the foliage plan uses for tree
/// shadows"*) — a plain low-poly box/AABB, placement-transformed, standing in
/// for a whole sublevel's shadow so its interior doesn't need per-object
/// placement in the shadow pipeline.
///
/// The flag and its main-pass exclusion (`indirect_dispatch.wgsl::test_instance`)
/// are wired; the proxy-volume mesh generation and per-sublevel object
/// lifecycle that would actually *set* this flag on something are not
/// implemented yet — sublevels currently cast no shadow at all (a safe,
/// visually-inert gap, not a wrong one), tracked as follow-up work.
pub const INSTANCE_FLAG_SHADOW_ONLY: u32 = 1 << 4;

/// Bit offset of the 4-bit sublevel-membership nibble in
/// [`GpuInstanceData::flags`].
///
/// Value `0` means "not a sublevel member"; `1..=MAX_SUBLEVEL_VIEWS` (see
/// `helio_secondary_core::MAX_SUBLEVEL_VIEWS`) means "member of the sublevel
/// currently occupying secondary view slot `N - 1`". Read by the secondary
/// per-view cull to select which instances belong to a given sublevel's
/// off-screen fill. Kept in already-reserved `flags` bits rather than a new
/// struct field so [`GpuInstanceData`]'s layout and size never change.
pub const INSTANCE_SUBLEVEL_MEMBERSHIP_SHIFT: u32 = 8;

/// Mask over [`GpuInstanceData::flags`] covering the sublevel-membership
/// nibble (4 bits at [`INSTANCE_SUBLEVEL_MEMBERSHIP_SHIFT`], values 0..=15).
pub const INSTANCE_SUBLEVEL_MEMBERSHIP_MASK: u32 = 0xF << INSTANCE_SUBLEVEL_MEMBERSHIP_SHIFT;

/// Encode a sublevel-membership nibble (`1..=15`, `0` = none) into a `flags`
/// value, replacing any previous membership without disturbing the other bits.
pub const fn set_sublevel_membership(flags: u32, membership: u32) -> u32 {
    debug_assert!(membership <= 0xF);
    (flags & !INSTANCE_SUBLEVEL_MEMBERSHIP_MASK)
        | ((membership << INSTANCE_SUBLEVEL_MEMBERSHIP_SHIFT) & INSTANCE_SUBLEVEL_MEMBERSHIP_MASK)
}

/// Decode the sublevel-membership nibble set by [`set_sublevel_membership`].
/// `0` means "not a sublevel member".
pub const fn sublevel_membership(flags: u32) -> u32 {
    (flags & INSTANCE_SUBLEVEL_MEMBERSHIP_MASK) >> INSTANCE_SUBLEVEL_MEMBERSHIP_SHIFT
}

/// Per-instance data for GPU-driven rendering. 208 bytes.
///
/// Uploaded once when instances change (dirty tracking), then read-only on GPU.
/// The vertex shader uses `instance_index` to look up this data from a storage buffer.
///
/// # WGSL equivalent
/// ```wgsl
/// struct GpuInstanceData {
///     transform:    mat4x4<f32>,  // 64 bytes — model matrix
///     normal_mat_0: vec4<f32>,    // 16 bytes — row 0 of normal matrix
///     normal_mat_1: vec4<f32>,    // 16 bytes — row 1
///     normal_mat_2: vec4<f32>,    // 16 bytes — row 2
///     bounds:       vec4<f32>,    // 16 bytes — bounding sphere
///     prev_model:   mat4x4<f32>,  // 64 bytes — previous frame model matrix
///     mesh_id:      u32,          //  4 bytes
///     material_id:  u32,          //  4 bytes
///     flags:        u32,          //  4 bytes
///     lightmap_index: u32,        //  4 bytes — index into lightmap atlas regions buffer
/// }
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuInstanceData {
    /// Model matrix columns 0–3 (column-major, 64 bytes)
    pub model: [f32; 16],
    /// Normal matrix (inverse-transpose of upper-left 3x3, padded to 3×vec4, 48 bytes)
    pub normal_mat: [f32; 12],
    /// Bounding sphere center in world space (xyz) + radius (w)
    pub bounds: [f32; 4],
    /// Previous frame model matrix (column-major, 64 bytes)
    pub prev_model: [f32; 16],
    /// Mesh index into the global mesh table
    pub mesh_id: u32,
    /// Material index into the global material table
    pub material_id: u32,
    /// Flags (bit 0 = casts_shadow, bit 1 = receives_shadow)
    pub flags: u32,
    /// Index into the lightmap atlas regions buffer (0xFFFFFFFF = no lightmap)
    pub lightmap_index: u32,
}

/// Per-instance AABB in world space for GPU culling. 32 bytes.
///
/// # WGSL equivalent
/// ```wgsl
/// struct GpuAabb {
///     min: vec3<f32>,
///     _pad0: f32,
///     max: vec3<f32>,
///     _pad1: f32,
/// }
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuInstanceAabb {
    pub min: [f32; 3],
    pub _pad0: f32,
    pub max: [f32; 3],
    pub _pad1: f32,
}

#[cfg(test)]
mod sublevel_membership_tests {
    use super::*;

    #[test]
    fn round_trips_every_valid_membership_value() {
        for membership in 0u32..=15 {
            let flags = set_sublevel_membership(0, membership);
            assert_eq!(sublevel_membership(flags), membership);
        }
    }

    #[test]
    fn leaves_other_flag_bits_untouched() {
        let base = INSTANCE_FLAG_CASTS_SHADOW | INSTANCE_FLAG_ALWAYS_VISIBLE;
        let flags = set_sublevel_membership(base, 3);
        assert_eq!(flags & INSTANCE_FLAG_CASTS_SHADOW, INSTANCE_FLAG_CASTS_SHADOW);
        assert_eq!(flags & INSTANCE_FLAG_ALWAYS_VISIBLE, INSTANCE_FLAG_ALWAYS_VISIBLE);
        assert_eq!(sublevel_membership(flags), 3);
    }

    #[test]
    fn re_encoding_replaces_the_previous_membership() {
        let flags = set_sublevel_membership(0, 5);
        let flags = set_sublevel_membership(flags, 2);
        assert_eq!(sublevel_membership(flags), 2);
    }

    #[test]
    fn zero_means_no_membership() {
        assert_eq!(sublevel_membership(0), 0);
        assert_eq!(sublevel_membership(INSTANCE_FLAG_SUBLEVEL_HIDDEN), 0);
    }
}

