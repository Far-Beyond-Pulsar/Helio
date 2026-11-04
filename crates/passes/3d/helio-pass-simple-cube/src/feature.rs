//! The test cube as a feature: the one node of the simple preset.

use helio_feature_api::{pass_node, Feature, FeatureRegistrar};

use crate::SimpleCubePass;

/// A spinning test cube ([`crate`]).
pub struct SimpleCubeFeature;

impl Feature for SimpleCubeFeature {
    fn id(&self) -> &'static str {
        "helio.simple_cube"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        registrar.node(pass_node(
            "simple_cube",
            "Test: Simple Cube",
            "Passes: Output",
            "A spinning test cube straight into the output: checks the device and surface.",
            |build| {
                let pass = SimpleCubePass::new(build.env.device, build.env.frame.surface_format);
                build.graph.add_pass(Box::new(pass));
                Ok(())
            },
        ));
    }
}
