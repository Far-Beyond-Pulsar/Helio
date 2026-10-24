//! Helio's shipped graphs as [`IntoGraph`] pipelines.
//!
//! Each is the `*_with_context` builder of the same name, named so a host can
//! offer them next to project-authored pipelines (see
//! [`register_builtin_pipelines`]). Hosts that add passes of their own (the
//! engine's voxel passes) register their own wrapper under [`DEFAULT`].

use std::sync::Arc;

use helio::{
    IntoGraph, PassBuildContext, PipelineFactory, PipelineInfo, PipelineOrigin,
    RenderPipelineRegistry,
};
use helio_core::RenderGraph;

/// Id of the full deferred pipeline, the default for a level that names none.
pub const DEFAULT: &str = "helio.deferred";
pub const DEFERRED_FXAA: &str = "helio.fxaa";
pub const HLFS: &str = "helio.hlfs";
pub const HLFS_FXAA: &str = "helio.fxaa_hlfs";
pub const FORWARD_OPAQUE: &str = "helio.forward_opaque";
pub const FORWARD_ONLY: &str = "helio.forward_only";
pub const SIMPLE: &str = "helio.simple";
/// The default deferred pipeline built from its node graph
/// ([`crate::graphs::deferred_graph_asset`]) rather than from code.
pub const DEFERRED_GRAPH: &str = "helio.deferred_graph";

/// One shipped graph builder as a pipeline.
#[derive(Clone, Copy)]
pub struct BuiltinPipeline {
    build: fn(PassBuildContext<'_>) -> RenderGraph,
}

impl BuiltinPipeline {
    pub const fn new(build: fn(PassBuildContext<'_>) -> RenderGraph) -> Self {
        Self { build }
    }
}

impl IntoGraph for BuiltinPipeline {
    fn into_graph(self: Box<Self>, ctx: PassBuildContext<'_>) -> RenderGraph {
        (self.build)(ctx)
    }
}

struct Spec {
    id: &'static str,
    label: &'static str,
    description: &'static str,
    build: fn(PassBuildContext<'_>) -> RenderGraph,
}

const SPECS: &[Spec] = &[
    Spec {
        id: DEFAULT,
        label: "Deferred (Default)",
        description: "Full deferred pipeline: shadows, GI, sky, fog, post-processing and TSR.",
        build: crate::build_default_graph_external_with_context,
    },
    Spec {
        id: DEFERRED_FXAA,
        label: "Deferred + FXAA",
        description: "Deferred lighting with FXAA instead of temporal resolve.",
        build: crate::build_fxaa_graph_with_context,
    },
    Spec {
        id: HLFS,
        label: "HLFS",
        description: "Hybrid light-field shading pipeline.",
        build: crate::build_hlfs_graph_with_context,
    },
    Spec {
        id: HLFS_FXAA,
        label: "HLFS + FXAA",
        description: "Hybrid light-field shading with FXAA.",
        build: crate::build_fxaa_hlfs_graph_with_context,
    },
    Spec {
        id: FORWARD_OPAQUE,
        label: "Forward (Opaque)",
        description: "Forward-lit opaque geometry with a depth pre-pass.",
        build: crate::build_forward_opaque_graph_with_context,
    },
    Spec {
        id: FORWARD_ONLY,
        label: "Forward Only",
        description: "Single forward pass, no deferred buffers. Cheapest option.",
        build: crate::build_forward_only_graph_with_context,
    },
    Spec {
        id: SIMPLE,
        label: "Simple",
        description: "A single test pass. Useful to verify the device and surface.",
        build: crate::build_simple_graph_with_context,
    },
];

/// Registers every shipped pipeline as [`PipelineOrigin::BuiltIn`] and makes
/// [`DEFAULT`] the registry default.
pub fn register_builtin_pipelines(registry: &mut RenderPipelineRegistry) {
    for spec in SPECS {
        let build = spec.build;
        let factory: PipelineFactory =
            Arc::new(move || Box::new(BuiltinPipeline::new(build)) as Box<dyn IntoGraph>);
        registry.register(
            PipelineInfo {
                id: spec.id.to_string(),
                label: spec.label.to_string(),
                description: spec.description.to_string(),
                origin: PipelineOrigin::BuiltIn,
            },
            factory,
        );
    }
    registry.register(
        PipelineInfo {
            id: DEFERRED_GRAPH.to_string(),
            label: "Deferred (Node Graph)".to_string(),
            description: "The default deferred pipeline compiled from render-graph nodes. \
                          The template for custom render graphs."
                .to_string(),
            origin: PipelineOrigin::BuiltIn,
        },
        Arc::new(|| {
            let registry = crate::graphs::base_registry();
            Box::new(helio_graph_asset::GraphAssetPipeline::new(
                Arc::new(crate::graphs::deferred_graph_asset(&registry)),
                Arc::new(registry),
                crate::build_default_graph_external_with_context,
            )) as Box<dyn IntoGraph>
        }),
    );
    registry.set_default(DEFAULT);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_every_spec_and_a_default() {
        let mut registry = RenderPipelineRegistry::new();
        register_builtin_pipelines(&mut registry);
        assert_eq!(registry.list().len(), SPECS.len() + 1);
        assert_eq!(registry.default_id(), Some(DEFAULT));
        assert_eq!(registry.resolve(Some("missing")), Some(DEFAULT));
        assert_eq!(registry.resolve(Some(HLFS)), Some(HLFS));
        assert!(registry.create(Some("missing")).is_some());
    }
}
