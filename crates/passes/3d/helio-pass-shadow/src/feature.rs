//! The meshes' shadow drawer: it opens the frame's shadow-caster phase,
//! rendering the stale shadow faces the lights' shadow views picked into the
//! tiled atlas.

use std::sync::Arc;

use helio_feature_api::{missing, pass_node, Feature, FeatureRegistrar};
use helio_frame_api::{FrameNode, Phase};
use helio_pass_shadow_dirty::ShadowDirtyPass;
use helio_pass_shadow_matrix::ShadowSettings;

use crate::ShadowPass;

/// Meshes in the shadow atlas ([`crate`]).
pub struct ShadowDrawers;

impl Feature for ShadowDrawers {
    fn id(&self) -> &'static str {
        "helio.mesh.shadow"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        registrar.node(
            pass_node(
                "shadow",
                "Shadows: Render Atlas",
                "Passes: Scene Prep",
                "Renders the dirty shadow faces into the tiled shadow atlas.",
                |build| {
                    let settings = build
                        .env
                        .setting::<ShadowSettings>()
                        .copied()
                        .unwrap_or_default();
                    let (_, atlas_size) =
                        settings.atlas(build.env.device.limits().max_texture_dimension_2d)?;
                    let (face_dirty, face_geom_count, face_pages) = build
                        .graph
                        .find_pass::<ShadowDirtyPass>()
                        .map(|dirty| {
                            (
                                Arc::clone(&dirty.face_dirty_buf),
                                Arc::clone(&dirty.face_geom_count_buf),
                                Arc::clone(&dirty.face_pages_buf),
                            )
                        })
                        .ok_or_else(|| missing("Shadow Maps", "Shadow Dirty Flags"))?;
                    build.graph.add_pass(Box::new(ShadowPass::new_tiled(
                        build.env.device,
                        build.env.queue,
                        face_dirty,
                        face_geom_count,
                        face_pages,
                        atlas_size,
                    )));
                    Ok(())
                },
            )
            .opens(Phase::ShadowCasters, 0),
        );
    }
}
