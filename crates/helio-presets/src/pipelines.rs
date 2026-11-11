//! Helio's shipped graphs as [`IntoGraph`] pipelines.
//!
//! Each is a [`Preset`], named so a host can offer them next to
//! project-authored pipelines (see [`register_builtin_pipelines`]). Features
//! a host adds join every preset ([`crate::Preset::with_features`]).

use std::sync::Arc;

use helio::{IntoGraph, PipelineFactory, PipelineInfo, PipelineOrigin, RenderPipelineRegistry};

use helio_frame_api::lighting::registered_models;

use crate::presets::Preset;

/// Id of the full deferred pipeline, the default for a level that names none.
pub const DEFAULT: &str = "helio.deferred";
pub const DEFERRED_FXAA: &str = "helio.fxaa";
pub const FORWARD_OPAQUE: &str = "helio.forward_opaque";
pub const FORWARD_ONLY: &str = "helio.forward_only";
/// The default deferred pipeline built from its node graph
/// ([`crate::graphs::deferred_graph_asset_for`]): the same nodes and settings
/// as [`DEFAULT`], laid out in sections.
pub const DEFERRED_GRAPH: &str = "helio.deferred_graph";

struct Spec {
    id: &'static str,
    label: &'static str,
    description: &'static str,
    preset: Preset,
}

const SPECS: &[Spec] = &[
    Spec {
        id: DEFAULT,
        label: "Deferred (Default)",
        description: "Full deferred pipeline: shadows, GI, sky, fog, post-processing, and TSR when the project enables it (FXAA otherwise).",
        preset: Preset::Deferred,
    },
    Spec {
        id: DEFERRED_FXAA,
        label: "Deferred + FXAA",
        description: "Deferred lighting with FXAA instead of temporal resolve.",
        preset: Preset::DeferredFxaa,
    },
    Spec {
        id: FORWARD_OPAQUE,
        label: "Forward (Opaque)",
        description: "Forward-lit opaque geometry, no G-buffer (the Forward Only graph).",
        preset: Preset::Forward,
    },
    Spec {
        id: FORWARD_ONLY,
        label: "Forward Only",
        description: "Single forward pass, no deferred buffers. Cheapest option.",
        preset: Preset::Forward,
    },
];

/// Registers every shipped pipeline as [`PipelineOrigin::BuiltIn`] and makes
/// [`DEFAULT`] the registry default. Each builds from the default features
/// and the engine's `extra` features (its plugins').
pub fn register_builtin_pipelines(
    registry: &mut RenderPipelineRegistry,
    extra: &helio_feature_api::FeatureSet,
) {
    let nodes = Arc::new(crate::graphs::registry_with(extra));
    for spec in SPECS {
        let preset = spec.preset;
        let nodes = Arc::clone(&nodes);
        let factory: PipelineFactory = Arc::new(move || {
            Box::new(preset.with_registry(Arc::clone(&nodes))) as Box<dyn IntoGraph>
        });
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
    // The pipelines the lighting models bring: their ids are the plugin's.
    for model in registered_models(&nodes) {
        for pipeline in &model.pipelines {
            let preset = Preset::Model(*pipeline);
            let nodes = Arc::clone(&nodes);
            registry.register(
                PipelineInfo {
                    id: pipeline.id.to_string(),
                    label: pipeline.label.to_string(),
                    description: pipeline.description.to_string(),
                    origin: PipelineOrigin::BuiltIn,
                },
                Arc::new(move || {
                    Box::new(preset.with_registry(Arc::clone(&nodes))) as Box<dyn IntoGraph>
                }),
            );
        }
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
        Arc::new(move || {
            // Made for each build from the renderer's config, like the preset:
            // the TSR node (at the project's quality) or FXAA.
            Box::new(
                helio_graph_asset::GraphAssetPipeline::made(
                    Arc::new(|registry, env| {
                        let config = env
                            .require::<helio::RendererConfig>()
                            .expect("a renderer build supplies its config")
                            .clone();
                        crate::graphs::deferred_graph_asset_for(registry, &config)
                    }),
                    Arc::clone(&nodes),
                )
                .with_swap_policy(crate::default_swap_policy()),
            ) as Box<dyn IntoGraph>
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
        register_builtin_pipelines(&mut registry, &Default::default());
        assert_eq!(registry.list().len(), SPECS.len() + 1);
        assert_eq!(registry.default_id(), Some(DEFAULT));
        assert_eq!(registry.resolve(Some("missing")), Some(DEFAULT));
        assert!(registry.create(Some("missing")).is_some());
    }
}
