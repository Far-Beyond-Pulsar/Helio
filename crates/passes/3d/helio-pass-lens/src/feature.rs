//! The lens as a feature: the node that images the resolved scene through the
//! camera's lens elements.

use helio_feature_api::{pass_node, Feature, FeatureRegistrar, ParamDesc, PinType};

use crate::{LensPass, LensUser};

/// Lens elements ([`crate`]).
pub struct LensFeature;

impl Feature for LensFeature {
    fn id(&self) -> &'static str {
        "helio.lens"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        registrar.node(
            pass_node(
                "lens",
                "Lens: Elements",
                "Passes: Post",
                "Lens elements: water drops, iris shapes and user lenses.",
                |build| {
                    let (iw, ih) = build.env.internal_size();
                    let input = match build.text("color_input").as_str() {
                        "tsr_color" => "tsr_color",
                        "fogged_hdr" => "fogged_hdr",
                        _ => "fxaa_color",
                    };
                    let user = build.env.setting::<LensUser>().map(|user| user.0);
                    let pass = LensPass::new_with_user(build.env.device, iw, ih, user)
                        .with_color_input(input)
                        .with_bundled_resources();
                    build.graph.add_pass(Box::new(pass));
                    Ok(())
                },
            )
            .with_param(ParamDesc::choice(
                "color_input",
                "Colour input",
                &["fxaa_color", "tsr_color", "fogged_hdr"],
                "fxaa_color",
            ))
            .with_alternative_inputs(&["tsr_color", "fxaa_color", "fogged_hdr"], PinType::TEXTURE),
        );
    }
}
