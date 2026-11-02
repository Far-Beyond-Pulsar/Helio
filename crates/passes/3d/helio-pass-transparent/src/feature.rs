//! The meshes' transparency drawer: alpha-blended meshes over the scene
//! colour the colour chain has where the transparent phase runs (the fogged
//! colour, when fog composited one), each fragment fogged at its own depth.
//! With [`helio_frame_api::color::ReactiveMask`] it also writes the mask the
//! anti-aliasing reads.

use helio_feature_api::{pass_node, Feature, FeatureRegistrar};
use helio_frame_api::{color, FrameNode, Phase};

use crate::TransparentPass;

/// Transparent meshes ([`crate`]).
pub struct TransparentDrawers;

impl Feature for TransparentDrawers {
    fn id(&self) -> &'static str {
        "helio.mesh.transparent"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        registrar.node(
            pass_node(
                "transparent",
                "Transparency: Blend",
                "Passes: Post",
                "Alpha-blended meshes over the scene colour.",
                |build| {
                    let target = color::current(build.env);
                    let pass = TransparentPass::new(build.env.device, build.env.scene_format());
                    let pass = if build.env.setting::<color::ReactiveMask>().is_some() {
                        pass.with_reactive_mask()
                    } else {
                        pass
                    };
                    let pass = if target == color::SCENE_COLOR {
                        pass.with_pre_aa_target()
                    } else {
                        pass.with_fogged_target()
                    };
                    build.graph.add_pass(Box::new(pass));
                    Ok(())
                },
            )
            .drawer(Phase::Transparent, 0),
        );
    }
}
