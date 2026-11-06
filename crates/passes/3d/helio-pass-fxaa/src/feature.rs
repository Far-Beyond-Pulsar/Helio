//! FXAA as a feature: the node that anti-aliases the fogged colour.

use helio_feature_api::{pass_node, Feature, FeatureRegistrar};
use helio_pass_postprocess::FOGGED_HDR;

use crate::FxaaPass;

/// Fast approximate anti-aliasing ([`crate`]).
pub struct FxaaFeature;

impl Feature for FxaaFeature {
    fn id(&self) -> &'static str {
        "helio.fxaa"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        registrar.node(pass_node(
            "fxaa",
            "Anti-Aliasing: FXAA",
            "Passes: Post",
            "Fast approximate anti-aliasing of the fogged colour.",
            |build| {
                let format = build.env.scene_format();
                let pass = FxaaPass::new(build.env.device, format)
                    .with_color_input(FOGGED_HDR)
                    .with_intermediate_target(format);
                build.graph.add_pass(Box::new(pass));
                Ok(())
            },
        ));
    }
}
