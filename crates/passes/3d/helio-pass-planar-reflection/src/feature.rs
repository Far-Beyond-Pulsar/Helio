//! Planar reflections as a feature: the node that reflects the scene across
//! its world-space planes.

use helio_feature_api::{pass_node, Feature, FeatureRegistrar};

use crate::PlanarReflectionPass;

/// Planar reflections ([`crate`]).
pub struct PlanarReflectionFeature;

impl Feature for PlanarReflectionFeature {
    fn id(&self) -> &'static str {
        "helio.planar_reflection"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        registrar.node(
            pass_node(
                "planar_reflection",
                "Reflections: Planar",
                "Passes: Lighting",
                "Reflects the scene across world-space planes.",
                |build| {
                    if !helio_core::REFLECTIONS_SUPPORTED {
                        log::warn!("Planar reflections are not supported on this platform");
                        return Ok(());
                    }
                    let pass = PlanarReflectionPass::new(
                        build.env.device,
                        build.env.camera_buffer,
                        build.env.frame.surface_format,
                    );
                    build.graph.add_pass(Box::new(pass));
                    Ok(())
                },
            )
            .with_unsupported_when(
                helio_core::REFLECTIONS_SUPPORTED,
                "reflections are not supported on this platform",
            ),
        );
    }
}
