//! Scene derivations a Helio host runs each frame between the SceneDB
//! snapshot and the render graph (`RendererBuilder::with_scene_derivation`):
//! they turn the rows engines author into the rows passes read.
//!
//! - [`hierarchy_resolve`]: parent-relative transforms to world space.
//! - [`scene_join`]: meshes and lights joined into the scene's draw rows.
//! - [`environment_join`]: the environment's rows (atmosphere, water,
//!   foliage types, decals, coronas, lenses, sprites) into the buffers their
//!   features read.
//! - [`portal_projection`], [`portals`]: authored portals paired and
//!   projected for the portal feature.
//! - [`texture_means`]: per-material texture means for ray-traced lighting.

pub mod environment_join;
pub mod hierarchy_resolve;
pub mod portal_projection;
pub mod portals;
pub mod scene_join;
pub mod texture_means;
