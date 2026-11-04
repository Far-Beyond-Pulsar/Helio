//! HLFS as a feature: the hybrid light-field lighting of the HLFS presets.

use helio_feature_api::{pass_node, Feature, FeatureRegistrar};

use crate::HlfsPass;

/// Hybrid light-field shading ([`crate`]).
pub struct HlfsFeature;

impl Feature for HlfsFeature {
    fn id(&self) -> &'static str {
        "helio.hlfs"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        registrar.node(pass_node(
            "hlfs",
            "Lighting: HLFS",
            "Passes: Lighting",
            "Hybrid light-field shading of the GBuffer: the lighting of the HLFS pipelines.",
            |build| {
                let (iw, ih) = build.env.internal_size();
                let queue = build.env.queue;
                let quality = build
                    .env
                    .setting::<helio_spine::SpineSettings>()
                    .cloned()
                    .unwrap_or_default()
                    .shadow_quality;
                let mut pass =
                    HlfsPass::new(build.env.device, queue, iw, ih, build.env.scene_format());
                pass.set_shadow_quality(quality, queue);
                build.graph.add_pass(Box::new(pass));
                Ok(())
            },
        ));
    }
}
