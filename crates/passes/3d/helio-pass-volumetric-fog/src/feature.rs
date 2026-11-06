//! Volumetric fog as a feature: the node that accumulates the fog the
//! spine's composite lays over the scene.

use helio_feature_api::{pass_node, Feature, FeatureRegistrar};

use crate::VolumetricFogPass;

/// Volumetric fog ([`crate`]).
pub struct FogFeature;

impl Feature for FogFeature {
    fn id(&self) -> &'static str {
        "helio.fog"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        registrar.node(pass_node(
            "volumetric_fog",
            "Fog: Volumetric",
            "Passes: Post",
            "Accumulates volumetric fog lit by the shadowed lights.",
            |build| {
                let pass = VolumetricFogPass::new(build.env.device);
                build.graph.add_pass(Box::new(pass));
                Ok(())
            },
        ));
    }
}
