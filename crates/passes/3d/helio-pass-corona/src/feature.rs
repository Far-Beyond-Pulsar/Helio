//! Light coronas as a feature.

use helio_feature_api::{pass_node, Feature, FeatureRegistrar};

use crate::CoronaPass;

/// Light coronas ([`crate`]).
pub struct CoronaFeature;

impl Feature for CoronaFeature {
    fn id(&self) -> &'static str {
        "helio.corona"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        registrar.node(
            pass_node(
                "corona",
                "Overlay: Light Coronas",
                "Passes: Overlays",
                "Light coronas.",
                |build| {
                    let pass = CoronaPass::new(
                        build.env.device,
                        build.env.queue,
                        build.env.camera_buffer,
                        build.env.scene_format(),
                    );
                    build.graph.add_pass(Box::new(pass));
                    Ok(())
                },
            )
            .with_marker_after(),
        );
    }
}
