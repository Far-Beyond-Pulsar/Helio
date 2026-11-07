//! Planar reflections as a feature: the node that reflects the scene across
//! its world-space planes into an indirect-specular layer of the frame, a
//! lighting input. Builds nothing while
//! [`PlanarReflectionSettings::enabled`] is off, or where the lighting model
//! does not composite the layers itself.

use helio_feature_api::{pass_node, Feature, FeatureRegistrar};
use helio_frame_api::lighting::{claim_specular_layer, specular_layers, SpecularLayers};
use helio_frame_api::{FrameNode, Slot};

use crate::PlanarReflectionPass;

/// Planar reflections' settings (a renderer config's `features`).
#[derive(Clone, Debug, Default)]
pub struct PlanarReflectionSettings {
    pub enabled: bool,
}

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
                    let enabled = build
                        .env
                        .setting::<PlanarReflectionSettings>()
                        .is_some_and(|settings| settings.enabled);
                    if !enabled
                        || !helio_core::REFLECTIONS_SUPPORTED
                        || specular_layers(build.env) != SpecularLayers::Read
                    {
                        return Ok(());
                    }
                    let Some(layer) = claim_specular_layer(build.env) else {
                        return Ok(());
                    };
                    let pass = PlanarReflectionPass::new(
                        build.env.device,
                        build.env.camera_buffer,
                        build.env.frame.surface_format,
                    )
                    .with_output(layer);
                    build.graph.add_pass(Box::new(pass));
                    Ok(())
                },
            )
            .with_unsupported_when(
                helio_core::REFLECTIONS_SUPPORTED,
                "reflections are not supported on this platform",
            )
            .in_slot(Slot::LightingInputs, -10),
        );
    }
}
