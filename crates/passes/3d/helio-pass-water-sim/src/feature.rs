//! Water as a feature: the node that simulates and draws water volumes.

use helio_feature_api::{pass_node, Feature, FeatureRegistrar};

use crate::WaterSimPass;

/// Water volumes ([`crate`]).
pub struct WaterFeature;

impl Feature for WaterFeature {
    fn id(&self) -> &'static str {
        "helio.water"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        registrar.node(
            pass_node(
                "water_sim",
                "Water: Simulate + Draw",
                "Passes: Overlays",
                "Simulates and draws water volumes.",
                |build| {
                    let (iw, ih) = build.env.internal_size();
                    let pass = WaterSimPass::new(
                        build.env.device,
                        build.env.camera_buffer,
                        iw,
                        ih,
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
