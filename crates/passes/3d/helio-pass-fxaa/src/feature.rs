//! FXAA as a feature: the node that anti-aliases the fogged colour.

use helio_feature_api::{pass_node, Feature, FeatureRegistrar};
use helio_frame_api::color;

use crate::FxaaPass;

/// Fast approximate anti-aliasing ([`crate`]).
pub struct FxaaFeature;

impl Feature for FxaaFeature {
    fn id(&self) -> &'static str {
        "helio.fxaa"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        // Shader hot reload may swap these alone: they take only
        // renderer-owned handles (camera, debug camera and cull-stats buffers,
        // debug-draw state, SceneDB) and plain configuration, the same objects
        // in the live and the replacement graph (`helio_core::SwapPolicy`).
        registrar.swappable::<FxaaPass>();
        registrar.node(pass_node(
            "fxaa",
            "Anti-Aliasing: FXAA",
            "Passes: Post",
            "Fast approximate anti-aliasing of the scene colour.",
            |build| {
                let format = build.env.scene_format();
                let pass = FxaaPass::new(build.env.device, format)
                    .with_color_input(color::current(build.env))
                    .with_color_alias(color::alias(build.env))
                    .with_intermediate_target(format);
                build.graph.add_pass(Box::new(pass));
                color::advance(build.env, crate::FXAA_COLOR);
                Ok(())
            },
        ));
    }
}
