//! The performance overlay as a feature: its draw over the final image, and
//! the markers that time each phase of a graph.

use std::sync::{Arc, Mutex};

use helio_feature_api::{pass_node, BuildEnv, Feature, FeatureRegistrar, MarkerFn, NodeBuild};
use helio_frame_api::{diagnostics, FrameNode, Slot};

use crate::{
    PerfOverlayAnalyzerPass, PerfOverlayCostAnalyzerPass, PerfOverlayMode, PerfOverlayPass,
    PerfOverlayShared,
};

/// The overlay's accumulator, made by whichever node needs it first.
struct PerfState(Arc<Mutex<PerfOverlayShared>>);

fn perf(env: &mut BuildEnv<'_>) -> Arc<Mutex<PerfOverlayShared>> {
    if let Some(state) = env.shared_existing::<PerfState>() {
        return state.0.clone();
    }
    let counter = diagnostics::tile_counter(env);
    let probe = diagnostics::surface_probe(env);
    let (device, (width, height)) = (env.device, env.internal_size());
    env.shared(|| {
        PerfState(PerfOverlayShared::new(
            device, width, height, counter, probe,
        ))
    })
    .0
    .clone()
}

/// The performance overlay ([`crate`]). Its mode is a setting
/// ([`PerfOverlayMode`]).
pub struct PerfOverlayFeature;

impl Feature for PerfOverlayFeature {
    fn id(&self) -> &'static str {
        "helio.perf_overlay"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        // The overlay's passes share one `PerfOverlayShared`: shader hot
        // reload swaps them together (`helio_core::SwapPolicy`).
        {
            use helio_core::graph::type_name_of as name;
            registrar.swap_group(&[
                name::<PerfOverlayAnalyzerPass>(),
                name::<PerfOverlayCostAnalyzerPass>(),
                name::<PerfOverlayPass>(),
            ]);
        }
        // After a node that ends a phase, the overlay's analyzer times it. The
        // graph does not contain them; the compiler adds them. After the node
        // that writes the offered tile counts (a lighting model's light
        // loops) the cost analyzer measures the shading cost too.
        let marker: MarkerFn = Arc::new(|build: &mut NodeBuild<'_, '_>, node_kind: &str| {
            let perf = perf(build.env);
            let counter = perf.lock().unwrap().tile_counter.clone();
            if counter.is_some_and(|counter| counter.after_node == node_kind) {
                build
                    .graph
                    .add_pass(Box::new(PerfOverlayCostAnalyzerPass::new(perf.clone())));
            }
            build
                .graph
                .add_pass(Box::new(PerfOverlayAnalyzerPass::new(perf)));
        });
        registrar.marker(marker);

        registrar.node(
            pass_node(
                "perf_overlay",
                "Output: Performance Overlay",
                "Passes: Output",
                "Draws the performance overlay over the final image.",
                |build| {
                    let shared = perf(build.env);
                    let mode = build
                        .env
                        .setting::<PerfOverlayMode>()
                        .copied()
                        .unwrap_or(PerfOverlayMode::Disabled);
                    let mut pass = PerfOverlayPass::new(
                        build.env.device,
                        shared,
                        build.env.frame.surface_format,
                    );
                    pass.set_mode(mode);
                    build.graph.add_pass(Box::new(pass));
                    Ok(())
                },
            )
            .in_slot(Slot::Display, 20),
        );
    }
}
