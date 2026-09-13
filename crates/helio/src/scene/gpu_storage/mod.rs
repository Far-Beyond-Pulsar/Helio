//! GPU-native scene state with dirty tracking -- relocated here, into the
//! `helio` frontend crate, from `helio-core` (removed entirely from that
//! crate: `helio-core` must have zero knowledge of any specific scene-object
//! type -- see the workspace's zero-central-type-knowledge mandate).
//!
//! This module is a frontend-owned concern, not part of the shared,
//! pass-agnostic render-graph core. Passes no longer receive any of this
//! through `PassContext`/`PrepareContext` (that per-frame contract is now
//! `scene_buffers: &SceneBufferProjection` -- type-erased, `BufferKey`-keyed
//! SceneDB columns -- plus whichever `libhelio::FrameResources` slots the
//! owning pass publishes). `GpuScene`/`SceneResources` still exist here only
//! because several subsystems (camera, materials, shadow matrices,
//! coordinate spaces, and portals) have not yet been individually
//! migrated to a SceneDB-native buffer or pass-owned storage the way static
//! objects (`helio-pass-object-batch`), lights (`"scene_lights"` SceneDB
//! buffer), and reflection captures (`"reflection_captures"` SceneDB buffer)
//! already have -- each is its own follow-up migration, not a mechanical
//! rename, per the mandate's own recorded precedent (shadow-caster baking is
//! real algorithmic logic built on `GpuScene.lights`, not just CRUD storage).
//!
//! `Scene` (`crate::scene::core`) still owns one `GpuScene` internally for
//! exactly this reason.

mod gpu_scene;
mod managers;
mod resources;

pub use gpu_scene::GpuScene;
pub use managers::*;
pub use resources::SceneResources;
