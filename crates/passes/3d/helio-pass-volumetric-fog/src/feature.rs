//! Fog as a feature: the volumetric medium and its composite over the scene
//! colour, at the frame's media slot. The composite advances the colour
//! chain to the fogged colour.

use helio_feature_api::{pass_node, Feature, FeatureRegistrar};
use helio_frame_api::{color, FrameNode, Slot};

use crate::{FogCompositePass, VolumetricFogPass, FOGGED_HDR};

/// Volumetric fog ([`crate`]).
pub struct FogFeature;

impl Feature for FogFeature {
    fn id(&self) -> &'static str {
        "helio.fog"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        registrar.node(
            pass_node(
                "volumetric_fog",
                "Fog: Volumetric",
                "Passes: Post",
                "Accumulates volumetric fog lit by the shadowed lights.",
                |build| {
                    let pass = VolumetricFogPass::new(build.env.device);
                    build.graph.add_pass(Box::new(pass));
                    Ok(())
                },
            )
            .in_slot(Slot::Media, 0),
        );

        registrar.node(
            pass_node(
                "fog_composite",
                "Fog: Composite",
                "Passes: Post",
                "Composites the fog over the scene colour into the fogged colour.",
                |build| {
                    let input = color::current(build.env);
                    let pass =
                        FogCompositePass::with_format(build.env.device, build.env.scene_format())
                            .with_color_input(input);
                    build.graph.add_pass(Box::new(pass));
                    color::advance(build.env, FOGGED_HDR);
                    Ok(())
                },
            )
            .in_slot(Slot::Media, 10),
        );
    }
}
