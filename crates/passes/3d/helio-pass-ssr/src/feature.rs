//! Screen-space reflections as a feature: the trace publishes an
//! indirect-specular layer of the frame. Where the lighting model composites
//! the layers itself (deferred) the trace is a lighting input; where it does
//! not (HLFS) the trace runs after lighting, over the lit colour, and its
//! composite adds the layer. Builds nothing while [`SsrSettings::enabled`]
//! is off.

use helio_feature_api::{pass_node, BuildEnv, Feature, FeatureRegistrar, NodeBuild};
use helio_frame_api::lighting::{claim_specular_layer, specular_layers, SpecularLayers};
use helio_frame_api::{FrameNode, Slot};

use crate::{SsrCompositePass, SsrPass};

const CATEGORY: &str = "Passes: Lighting";
const UNSUPPORTED: &str = "reflections are not supported on this platform";

/// Screen-space reflections' settings (a renderer config's `features`).
#[derive(Clone, Debug, Default)]
pub struct SsrSettings {
    pub enabled: bool,
}

fn enabled(env: &BuildEnv<'_>) -> bool {
    helio_core::REFLECTIONS_SUPPORTED
        && env
            .setting::<SsrSettings>()
            .is_some_and(|settings| settings.enabled)
}

/// Adds the trace into the next free layer; the layer, or `None` when every
/// layer is taken.
fn add_trace(build: &mut NodeBuild<'_, '_>) -> Option<&'static str> {
    let layer = claim_specular_layer(build.env)?;
    let (iw, ih) = build.env.internal_size();
    let pass = SsrPass::new(
        build.env.device,
        build.env.queue,
        build.env.camera_buffer,
        iw,
        ih,
    )
    .with_output(layer);
    build.graph.add_pass(Box::new(pass));
    Some(layer)
}

/// Screen-space reflections ([`crate`]).
pub struct SsrFeature;

impl Feature for SsrFeature {
    fn id(&self) -> &'static str {
        "helio.ssr"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        // Shader hot reload may swap these alone: they take only
        // renderer-owned handles (camera, debug camera and cull-stats buffers,
        // debug-draw state, SceneDB) and plain configuration, the same objects
        // in the live and the replacement graph (`helio_core::SwapPolicy`).
        registrar.swappable::<SsrPass>();
        registrar.swappable::<SsrCompositePass>();
        let supported = helio_core::REFLECTIONS_SUPPORTED;
        registrar.node(
            pass_node(
                "ssr",
                "Reflections: Screen-Space",
                CATEGORY,
                "Traces glossy reflections in screen space, a specular layer lighting composites. One of the most expensive passes.",
                |build| {
                    if enabled(build.env) && specular_layers(build.env) == SpecularLayers::Read {
                        add_trace(build);
                    }
                    Ok(())
                },
            )
            .with_unsupported_when(supported, UNSUPPORTED)
            .in_slot(Slot::LightingInputs, -20),
        );

        registrar.node(
            pass_node(
                "ssr_composite",
                "Reflections: Screen-Space over the Lit Colour",
                CATEGORY,
                "Traces screen-space reflections of the lit colour and adds them to it, for lighting that does not composite specular layers itself (HLFS).",
                |build| {
                    if !enabled(build.env)
                        || specular_layers(build.env) != SpecularLayers::CompositedAfter
                    {
                        return Ok(());
                    }
                    let Some(layer) = add_trace(build) else {
                        return Ok(());
                    };
                    let pass = SsrCompositePass::new(
                        build.env.device,
                        build.env.camera_buffer,
                        build.env.scene_format(),
                        layer,
                    );
                    build.graph.add_pass(Box::new(pass));
                    Ok(())
                },
            )
            .with_unsupported_when(supported, UNSUPPORTED)
            .in_slot(Slot::Lighting, -10),
        );
    }
}
