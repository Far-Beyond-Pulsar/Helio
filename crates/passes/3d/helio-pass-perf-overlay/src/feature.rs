//! The performance overlay as a feature: its draw over the final image, and
//! the markers that time each phase of a graph.

use std::sync::{Arc, Mutex};

use helio_feature_api::{pass_node, BuildEnv, Feature, FeatureRegistrar, MarkerFn, NodeBuild};

use crate::{
    PerfOverlayAnalyzerPass, PerfOverlayCostAnalyzerPass, PerfOverlayMode, PerfOverlayPass,
    PerfOverlayShared,
};

/// The overlay's accumulator, made by whichever node needs it first.
struct PerfState(Arc<Mutex<PerfOverlayShared>>);

fn perf(env: &mut BuildEnv<'_>) -> Arc<Mutex<PerfOverlayShared>> {
    let (device, (width, height)) = (env.device, env.internal_size());
    env.shared(|| PerfState(PerfOverlayShared::new(device, width, height)))
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
        // After a node that ends a phase, the overlay's analyzer times it. The
        // graph does not contain them; the compiler adds them. After the
        // lighting the cost analyzer measures the shading cost too.
        let marker: MarkerFn = Arc::new(|build: &mut NodeBuild<'_, '_>, node_kind: &str| {
            let perf = perf(build.env);
            if node_kind == "helio.pass.deferred_light" {
                build
                    .graph
                    .add_pass(Box::new(PerfOverlayCostAnalyzerPass::new(perf.clone())));
            }
            build
                .graph
                .add_pass(Box::new(PerfOverlayAnalyzerPass::new(perf)));
        });
        registrar.marker(marker);

        registrar.node(pass_node(
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
                let mut pass =
                    PerfOverlayPass::new(build.env.device, shared, build.env.frame.surface_format);
                pass.set_mode(mode);
                build.graph.add_pass(Box::new(pass));
                Ok(())
            },
        ));
    }
}
