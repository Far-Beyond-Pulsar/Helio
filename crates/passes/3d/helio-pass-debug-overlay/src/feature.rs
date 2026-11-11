//! The on-screen debug overlay as a feature: its node.
//!
//! The overlay's text, bars and charts are the host's: it keeps a
//! [`DebugOverlayState`] and fills it (`populate`, or between frames). The
//! host hands that state to the graph as the [`DebugOverlayHost`] setting (a
//! renderer config's `features`); the node draws it over the final image.
//! With none given the node owns a state of its own, hidden until shown, that
//! the host reaches through the built pass (`find_pass_mut::<DebugOverlayPass>()`,
//! [`DebugOverlayPass::shared`]).
//!
//! The node has no default placement: no shipped pipeline draws the overlay.
//! A pipeline that wants it places the node after the display slot's nodes.

use std::sync::{Arc, Mutex};

use helio_feature_api::{pass_node, Feature, FeatureRegistrar, ParamDesc};

use crate::{DebugOverlayPass, DebugOverlayState};

/// Kind of the debug overlay node.
pub const DEBUG_OVERLAY_NODE: &str = "helio.pass.debug_overlay";

/// The host's overlay state, as a setting of the graph being built.
#[derive(Clone)]
pub struct DebugOverlayHost(pub Arc<Mutex<DebugOverlayState>>);

/// The debug overlay ([`crate`]).
pub struct DebugOverlayFeature;

impl Feature for DebugOverlayFeature {
    fn id(&self) -> &'static str {
        "helio.debug_overlay"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        registrar.swappable::<DebugOverlayPass>();
        registrar.node(
            pass_node(
                "debug_overlay",
                "Output: Debug Overlay",
                "Passes: Output",
                "The host's on-screen text, bars and charts over the final image (FPS, frame time, \
                 whatever the host writes into its overlay state). Draws nothing while hidden.",
                |build| {
                    let state = build
                        .env
                        .setting::<DebugOverlayHost>()
                        .map(|host| host.0.clone())
                        .unwrap_or_else(DebugOverlayState::new);
                    if build.boolean("visible") {
                        state.lock().unwrap_or_else(|e| e.into_inner()).enabled = true;
                    }
                    let frame = build.env.frame;
                    build.graph.add_pass(Box::new(DebugOverlayPass::new(
                        build.env.device,
                        build.env.queue,
                        state,
                        frame.surface_format,
                        frame.width,
                        frame.height,
                    )));
                    Ok(())
                },
            )
            .with_param(ParamDesc::boolean("visible", "Visible", false)),
        );
    }
}
