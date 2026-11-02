//! Scene derivations a Helio host runs each frame between the SceneDB
//! snapshot and the render graph (`RendererBuilder::with_scene_derivation`):
//! they turn the rows engines author into the rows passes read.
//!
//! - [`hierarchy_resolve`]: parent-relative transforms to world space.
//! - Meshes and lights are joined by their own crates' derivations
//!   (`helio_pass_object_batch::scene_join::SceneJoin`,
//!   `helio_pass_light_cull::scene_rows::LightJoin`).
//! - Every other kind's rows (fog, post-process volumes, the lens, water,
//!   foliage, the sky, decals, particle emitters, sprites) are placed by a
//!   `helio_scene_join::RowJoin` each kind builds in its own crate.
//! - [`portal_projection`], [`portals`]: authored portals paired and
//!   projected for the portal feature.
//! - [`texture_means`]: per-material texture means for ray-traced lighting.

pub mod hierarchy_resolve;
pub mod portal_projection;
pub mod portals;
pub mod texture_means;

/// Every scene derivation shader as compiled (shader validation without a
/// device): the hierarchy resolve, the meshes' and lights' joins, the row
/// join.
pub fn validation_sources() -> Vec<(String, String)> {
    vec![
        (
            "hierarchy_resolve.wgsl".to_string(),
            helio_scene_join::join_source(include_str!("../shaders/hierarchy_resolve.wgsl")),
        ),
        (
            "scene_join_meshes.wgsl".to_string(),
            helio_pass_object_batch::scene_join::mesh_join_source(),
        ),
        (
            "scene_join_lights.wgsl".to_string(),
            helio_pass_light_cull::scene_rows::light_join_source(),
        ),
        (
            "row_join.wgsl".to_string(),
            helio_scene_join::row_join_source(),
        ),
    ]
}
