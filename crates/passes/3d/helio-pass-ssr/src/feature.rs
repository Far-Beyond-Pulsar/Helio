//! Screen-space reflections as a feature: the trace, and its composite for
//! lighting that does not read the trace itself (HLFS).

use helio_feature_api::{pass_node, Feature, FeatureRegistrar};

use crate::{SsrCompositePass, SsrPass};

const CATEGORY: &str = "Passes: Lighting";
const UNSUPPORTED: &str = "reflections are not supported on this platform";

/// Screen-space reflections ([`crate`]).
pub struct SsrFeature;

impl Feature for SsrFeature {
    fn id(&self) -> &'static str {
        "helio.ssr"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        let supported = helio_core::REFLECTIONS_SUPPORTED;
        registrar.node(
            pass_node(
                "ssr",
                "Reflections: Screen-Space",
                CATEGORY,
                "Traces glossy reflections in screen space. One of the most expensive passes.",
                |build| {
                    if !helio_core::REFLECTIONS_SUPPORTED {
                        log::warn!("Screen-space reflections are not supported on this platform");
                        return Ok(());
                    }
                    let (iw, ih) = build.env.internal_size();
                    let pass = SsrPass::new(
                        build.env.device,
                        build.env.queue,
                        build.env.camera_buffer,
                        iw,
                        ih,
                    );
                    build.graph.add_pass(Box::new(pass));
                    Ok(())
                },
            )
            .with_unsupported_when(supported, UNSUPPORTED),
        );

        registrar.node(
            pass_node(
                "ssr_composite",
                "Reflections: Composite Screen-Space",
                CATEGORY,
                "Adds the screen-space reflections to the lit colour, for lighting that does not read them itself (HLFS).",
                |build| {
                    if !helio_core::REFLECTIONS_SUPPORTED {
                        return Ok(());
                    }
                    let pass = SsrCompositePass::new(
                        build.env.device,
                        build.env.camera_buffer,
                        build.env.scene_format(),
                    );
                    build.graph.add_pass(Box::new(pass));
                    Ok(())
                },
            )
            .with_unsupported_when(supported, UNSUPPORTED),
        );
    }
}
