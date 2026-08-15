//! Render-authored SceneDB `World` components -- the CPU-side surface a
//! gameplay/editor caller writes to instead of calling into `helio::Scene`
//! directly. See [`crate::subsystem::HelioRenderSubsystem`] for what
//! consumes these each frame.
//!
//! ## Design: purpose-built components, not a decomposed "RenderObject"
//!
//! Each component here is shaped like a specific Unreal component
//! (`UStaticMeshComponent`, a light component, ...): it owns exactly the
//! fields that ONE kind of renderable needs, including its material(s)
//! inline where that's the natural shape, rather than every renderable
//! being one generic entity assembled from several small cross-referenced
//! pieces. `mesh`/`material`/whatever-else-a-given-kind-needs is that
//! component's own business; SceneDB doesn't impose one shape on every
//! renderable, and it's up to Helio how the data actually gets drawn.
//!
//! The one thing every purpose-built component still composes with:
//! [`RenderTransform`]/[`RenderBounds`]/[`RenderFlags`] stay small and
//! shared, because they're genuinely universal -- every renderable needs
//! exactly one of each regardless of what kind it is, the same way every
//! Unreal mesh component inherits transform/visibility from
//! `USceneComponent`/`UPrimitiveComponent` rather than redeclaring it.
//!
//! ## Why some fields are `#[gpu(buffer = "...")]` and others aren't
//!
//! `#[gpu(buffer = "key")]` fits data addressed by a STABLE per-entity row
//! that some consumer reads by index, with no per-frame compaction --
//! materials are exactly this shape (`GpuInstanceData::material_id`/
//! `ObjectDescriptor::material` are stable indices). It does NOT fit data
//! that a system needs densely packed for straight-line shader iteration
//! every frame -- lights are the opposite shape: `helio::Scene::flush()`
//! rebuilds the entire light buffer every call (filters to movable-only,
//! reassigns indices densely), so a `#[gpu(buffer=...)]` field's fixed
//! `row = Entity::index()` would already be stale by the time anything
//! read it.
//!
//! ## No `LightComponent` here
//!
//! This crate deliberately does NOT define its own light component. Earlier
//! revisions had a plain `LightComponent { light: GpuLight, movability }`
//! shadow type that [`crate::subsystem::HelioRenderSubsystem`] watched via
//! the change tracker -- but that only existed to give this crate something
//! to translate from, duplicating data the real, editor-facing
//! `helio_component::LightComponent` (rich, properties-panel-editable,
//! already a genuine `World` value) already carries. Requiring a caller to
//! ALSO author a second, minimal shadow component just so this crate had
//! something to watch is exactly the kind of shim Pulsar-Native#561 exists
//! to eliminate. This crate (`helio-scenedb`) lives in Helio's own,
//! separate Cargo workspace (a git submodule pointing at a standalone repo)
//! and can't depend on `helio_component` (a Pulsar-Native crate in the
//! *outer* workspace) without pulling a large, editor-shaped dependency
//! tree into an otherwise-standalone renderer build -- so the real
//! `LightComponent -> GpuLight` translation
//! (`helio_component::LightComponent::to_gpu_light`) and the dispatch that
//! drives `Scene::insert_light_with_movability`/`update_light`/
//! `remove_light` from it both live on the Pulsar-Native side
//! (`engine_backend`, which already depends on both `helio` and
//! `helio_component`), not here. `HelioRenderSubsystem` still owns
//! materials/meshes -- those already have neutral `RenderTransform`/
//! `RenderBounds`/`RenderFlags` shadow shapes precisely because compaction
//! isn't in the way for them the way it is for lights, and that path is
//! separately established and tested; this doc note is scoped to lights
//! only.

use helio::{GroupMask, MeshId, MultiMeshId, Movability};
use libhelio::material::GpuMaterial;
use pulsar_scenedb::page::Pod;
use pulsar_scenedb::Entity;
use pulsar_scenedb_derive::SceneStore;
use smallvec::SmallVec;

/// `#[repr(transparent)]` wrapper around `libhelio::GpuMaterial`, needed
/// only to satisfy Rust's orphan rule -- neither `pulsar_scenedb::page::Pod`
/// nor `GpuMaterial` is local to this crate, so a direct
/// `unsafe impl Pod for GpuMaterial` isn't legal here even though it would
/// be perfectly sound (`GpuMaterial` is already `#[repr(C)]` and
/// `bytemuck::Pod`). Byte-for-byte identical layout to `GpuMaterial` --
/// this newtype adds no padding, no reordering, nothing observable on the
/// wire or in a bind group.
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct SdbGpuMaterial(pub GpuMaterial);

// SAFETY: `SdbGpuMaterial` is `#[repr(transparent)]` over `GpuMaterial`,
// which is itself `#[repr(C)]` and already `bytemuck::Pod` (see
// `libhelio::material::GpuMaterial`'s own derive) -- every byte pattern
// `GpuMaterial` accepts, `SdbGpuMaterial` accepts identically, with the
// exact same size/alignment/field layout. No interior padding is
// introduced by the wrapper (a `#[repr(transparent)]` single-field tuple
// struct has zero size/alignment overhead over its field by definition).
unsafe impl Pod for SdbGpuMaterial {}

/// `#[repr(transparent)]` wrapper around `helio::MeshId`, same orphan-rule
/// reasoning as [`SdbGpuMaterial`]. Needed because `#[derive(SceneStore)]`
/// generates `unsafe impl Pod` for the WHOLE struct it's applied to (every
/// field, not just `#[gpu]`-tagged ones -- `SceneColumnSet`'s row-memcpy
/// path needs the entire layout to be byte-reinterpretable), so
/// `StaticMeshComponent::mesh` needs to be `Pod` even though it's a plain,
/// non-`#[gpu]` field. `MeshId` is already two `u32`s (`slot`, `generation`)
/// with no padding-sensitive layout, so this wrapper is exactly as sound as
/// `SdbGpuMaterial`'s.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SdbMeshId(pub MeshId);

// SAFETY: see `SdbGpuMaterial`'s SAFETY comment -- identical reasoning,
// `MeshId` in place of `GpuMaterial`. `MeshId` is `#[derive(..., Clone,
// Copy, ...)]` over two plain `u32`s (`helio::handles::define_handle!`),
// no interior padding, no invalid bit patterns for any `u32` value.
unsafe impl Pod for SdbMeshId {}

/// A static mesh with exactly one material -- the common case. Material
/// data lives INLINE on this component; there is no separate material
/// entity or cross-reference for this path. `#[gpu(buffer =
/// "builtin_material")]` means every `StaticMeshComponent`'s (and every
/// [`MaterialSlot`]'s -- see that type's doc) `material`/`data` field
/// shares ONE physical buffer, keyed by whichever entity carries it; the
/// derive doesn't know or care they're different struct types, only that
/// the key matches (validated -- element type, stride, mode -- at
/// registration).
///
/// **One entity = one material row.** `#[gpu(buffer=...)]` fields always
/// write at `row = Entity::index()` (a hard contract -- see
/// `pulsar_scenedb::gpu::slot_allocator`'s module doc), so THIS entity's
/// index is where `HelioRenderSubsystem` finds this material's bytes.
#[derive(SceneStore, Clone, Copy)]
#[repr(C)]
pub struct StaticMeshComponent {
    /// Helio's own `MeshPool` handle -- not `#[gpu]`, SceneDB never touches
    /// vertex/index bytes (see the crate-level integration plan's "stays
    /// with Helio" list). Wrapped in [`SdbMeshId`] only for `Pod`'s sake
    /// (see that type's doc); reach the real `MeshId` via `.mesh.0`.
    pub mesh: SdbMeshId,
    #[gpu(buffer = "builtin_material", mirror = DirtyTracked)]
    pub material: SdbGpuMaterial,
}

impl StaticMeshComponent {
    pub fn new(mesh: MeshId, material: GpuMaterial) -> Self {
        Self { mesh: SdbMeshId(mesh), material: SdbGpuMaterial(material) }
    }
}

/// Bare material data, meant to be referenced by index from a
/// [`MultiMaterialStaticMeshComponent`] -- NOT the default way a single
/// mesh gets its material (that's [`StaticMeshComponent::material`],
/// inline). Exists only because a multi-material object needs N
/// independently addressable materials, and `#[gpu(buffer=...)]` mirrors
/// one field at one row per entity, not an array per entity -- so "N
/// materials" becomes N entities, each a bare `MaterialSlot`, referenced
/// by `Entity` from the owning component.
#[derive(SceneStore, Clone, Copy)]
#[repr(C)]
pub struct MaterialSlot {
    #[gpu(buffer = "builtin_material", mirror = DirtyTracked)]
    pub data: SdbGpuMaterial,
}

impl MaterialSlot {
    pub fn new(material: GpuMaterial) -> Self {
        Self { data: SdbGpuMaterial(material) }
    }
}

/// A multi-section static mesh with one material per section -- Helio's
/// `MultiMeshId`/`insert_sectioned_object` path (a DIFFERENT mesh-asset
/// registration than plain `StaticMeshComponent::mesh: MeshId` -- a
/// multi-mesh is uploaded via `Scene::insert_sectioned_mesh`, not
/// `insert_mesh`). `materials.len()` must equal the mesh's section count
/// (Helio validates this at translation time and the update is a no-op,
/// not a panic, on mismatch -- see `HelioRenderSubsystem`).
///
/// `SmallVec<[Entity; 4]>`: most multi-material meshes have a handful of
/// sections (LOD-adjacent submeshes, a body + accessories, ...) -- four
/// inline slots covers the common case with no heap allocation; more than
/// four spills to the heap exactly like a `Vec` would.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MultiMaterialStaticMeshComponent {
    pub mesh: MultiMeshId,
    /// Each entry references a [`MaterialSlot`] entity, in section order.
    pub materials: SmallVec<[Entity; 4]>,
}

/// World-space model matrix. Column-major, matching
/// `helio::scene::types::ObjectDescriptor::transform` exactly (both are
/// `glam::Mat4`) -- no conversion at the `HelioRenderSubsystem` boundary.
///
/// Changing this alone drives `helio::Scene::update_object_transform` --
/// confirmed genuinely O(1), in-place, no draw-batch rebuild (see the
/// crate-level integration plan) -- so this is the cheap, every-frame-safe
/// field to write for anything that moves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderTransform(pub glam::Mat4);

/// World-space bounding sphere `[center.x, center.y, center.z, radius]`,
/// used for GPU frustum culling -- matches
/// `ObjectDescriptor::bounds`/`GpuInstanceData::bounds` exactly. Like
/// [`RenderTransform`], changing this alone is an O(1) in-place Helio
/// update for a plain `StaticMeshComponent` object (sectioned objects have
/// no equivalent cheap bounds update -- see `HelioRenderSubsystem`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderBounds(pub [f32; 4]);

/// Render flags, visibility groups, and movability -- everything else
/// `ObjectDescriptor` needs besides mesh/material/transform/bounds.
/// `flags` matches `GpuInstanceData::flags`'s bit layout (bit0
/// casts_shadow, bit1 receives_shadow, bit2 always_visible, bits8-15
/// coordinate-space id) -- see `libhelio::instance::GpuInstanceData`'s doc
/// for the authoritative bit assignments.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct RenderFlags {
    pub flags: u32,
    pub movability: Option<Movability>,
    pub groups: GroupMask,
}
