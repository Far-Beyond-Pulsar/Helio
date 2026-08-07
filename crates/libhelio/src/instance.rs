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

/// Bit offset of the coordinate-space id within [`GpuInstanceData::flags`].
///
/// # Coordinate spaces
///
/// Every instance is drawn through `coordinate_spaces[space_id] * transform`,
/// where `coordinate_spaces` is a small GPU array of rigid transforms
/// (`crates/helio-core/src/scene/managers.rs::CoordinateSpaceBuffer`). Slot 0
/// is always the identity, so an untagged instance (the overwhelming common
/// case) pays one constant-buffer read and one extra `mat4x4` multiply per
/// vertex — no new pass, no branch, no separate pipeline.
///
/// Sublevels and portals are both just consumers of this one mechanism: a
/// sublevel assigns its members a space id once and moves the whole sublevel
/// by writing one matrix (`Scene::move_sublevel`); a portal draws a *second*,
/// clipped copy of nearby geometry through its own space id
/// (`Scene::add_portal`). See `docs/` for the full design.
pub const INSTANCE_COORDINATE_SPACE_SHIFT: u32 = 8;

/// Mask for the 8-bit coordinate-space id within [`GpuInstanceData::flags`].
///
/// 256 values are encodable; [`crate::MAX_COORDINATE_SPACES`] (32) are
/// actually backed by GPU storage, which is enough headroom that running out
/// is not a realistic concern.
pub const INSTANCE_COORDINATE_SPACE_MASK: u32 = 0xFF << INSTANCE_COORDINATE_SPACE_SHIFT;

/// Number of coordinate-space slots backed by GPU storage. Slot 0 is always
/// the identity (world space). Mirrors `MAX_COORDINATE_SPACES` in
/// `coordinate_spaces.wgsl` / every shader that binds the buffer — keep in sync.
pub const MAX_COORDINATE_SPACES: u32 = 32;

/// Packs `space` (a coordinate-space slot, see [`INSTANCE_COORDINATE_SPACE_SHIFT`])
/// into `flags`, replacing whatever coordinate-space id was previously set.
///
/// # Panics
/// Debug-asserts `space < MAX_COORDINATE_SPACES`; release builds mask silently.
pub const fn set_coordinate_space(flags: u32, space: u32) -> u32 {
    debug_assert!(space < MAX_COORDINATE_SPACES);
    (flags & !INSTANCE_COORDINATE_SPACE_MASK)
        | ((space << INSTANCE_COORDINATE_SPACE_SHIFT) & INSTANCE_COORDINATE_SPACE_MASK)
}

/// Reads the coordinate-space id packed into `flags` by [`set_coordinate_space`].
/// `0` (world space / identity) for any instance that never had one set.
pub const fn coordinate_space(flags: u32) -> u32 {
    (flags & INSTANCE_COORDINATE_SPACE_MASK) >> INSTANCE_COORDINATE_SPACE_SHIFT
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
    /// Flags (bit 0 = casts_shadow, bit 1 = receives_shadow, bit 2 = always_visible,
    /// bits 8-15 = coordinate-space id, see [`set_coordinate_space`]/[`coordinate_space`])
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

