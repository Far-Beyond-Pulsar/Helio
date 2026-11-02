//! The sky as a feature: the nodes it adds to preset graphs.
//!
//! The atmosphere's tables run where the preset names them (scene prep); the
//! clouds' trace joins at [`Slot::LightingInputs`], once the depth is
//! complete, since deferred lighting takes the sun's visibility under the
//! clouds from it.

use helio_feature_api::{Feature, FeatureRegistrar, NodeDescriptor, ParamDesc, Slot};

use crate::{AtmospherePass, CloudTracePass};

/// Atmosphere and clouds ([`crate`]).
pub struct SkyFeature;

impl Feature for SkyFeature {
    fn id(&self) -> &'static str {
        "helio.sky"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        registrar.node(
            NodeDescriptor::new(
                "helio.pass.atmosphere",
                "Sky: Atmosphere Tables",
                "Passes: Scene Prep",
                "Atmosphere lookup tables and the resolved sky frame. Idle without an atmosphere component.",
                |build| {
                    build
                        .graph
                        .add_pass(Box::new(AtmospherePass::new(build.env.device)));
                    Ok(())
                },
            )
            .with_param(ParamDesc::boolean("enabled", "Enabled", true))
            .with_generated_pins(),
        );
        registrar.node(
            NodeDescriptor::new(
                "helio.pass.cloud_trace",
                "Sky: Cloud Trace",
                "Passes: Lighting",
                "The clouds' sun visibility over the finished depth, for deferred lighting. Idle without clouds.",
                |build| {
                    build
                        .graph
                        .add_pass(Box::new(CloudTracePass::new(build.env.device)));
                    Ok(())
                },
            )
            .with_param(ParamDesc::boolean("enabled", "Enabled", true))
            .with_generated_pins()
            .in_slot(Slot::LightingInputs, 10),
        );
    }
}
