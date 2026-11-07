//! Scene derivations a Helio host runs each frame between the SceneDB
//! snapshot and the render graph (`RendererBuilder::with_scene_derivation`):
//! they turn the rows engines author into the rows passes read.
//!
//! - [`hierarchy_resolve`]: parent-relative transforms to world space.
//! - Each object kind's rows are joined by its own crate's derivation (a
//!   lights plugin's light join, a meshes plugin's scene join), or placed by
//!   a `helio_scene_join::RowJoin` the kind builds in its own crate.

pub mod hierarchy_resolve;

/// Every scene derivation shader in Helio as compiled (shader validation
/// without a device): the hierarchy resolve and the row join.
pub fn validation_sources() -> Vec<(String, String)> {
    vec![
        (
            "hierarchy_resolve.wgsl".to_string(),
            helio_scene_join::join_source(include_str!("../shaders/hierarchy_resolve.wgsl")),
        ),
        (
            "row_join.wgsl".to_string(),
            helio_scene_join::row_join_source(),
        ),
    ]
}
