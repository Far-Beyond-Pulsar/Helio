//! Helio's shipped pipelines, as node graphs.
//!
//! A preset is a graph of the spine's passes and the frame's slots and draw
//! phases (`helio_frame_api`), made for the renderer config it is built
//! with: the config picks the anti-aliasing (TSR when `tsr_quality` is set)
//! and the TSR node's settings; the lighting model is the preset's. Every
//! other node joins where its feature placed it (a slot, or a phase's
//! drawers and the compute before them), so a feature (meshes, GI, the
//! clouds, foliage, fog, an engine's object kinds) joins
//! every preset without a preset naming it, and builds nothing when its own
//! settings turn it off. Every build,
//! resizes and shader reloads included, makes the graph again from the
//! renderer's current config.
//!
//! A host builds a renderer with one by function (`presets::deferred`) or by
//! value (`with_graph(Preset::Deferred)`). Project-authored graphs are fixed
//! [`GraphAsset`]s built by `helio_graph_asset::GraphAssetPipeline`.
//!
//! Every preset reads its GPU timings a frame late instead of waiting for
//! them (the renderer's own poll, when it owns the device, or the host's
//! submits deliver them), as the engine's graphs always have.

use std::sync::{Arc, OnceLock};

use helio::{IntoGraph, PassBuildContext, RendererConfig};
use helio_core::RenderGraph;
use helio_feature_api::ExtensionPoint;
use helio_frame_api::color::ReactiveMask;
use helio_frame_api::lighting::SpecularLayers;
use helio_frame_api::{Phase, PhaseList, Slot, HOST_INPUTS};
use helio_graph_asset::{
    core_nodes, Connection, GraphAsset, GraphAssetPipeline, NodeInstance, NodeRegistry, Params,
    PinRef,
};
use serde_json::Value;

use crate::features::{kind, SceneColorFormat};

/// One of Helio's shipped pipelines.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Preset {
    /// Deferred lighting; TSR when the config asks for it, FXAA otherwise.
    Deferred,
    /// Deferred lighting with FXAA, never TSR.
    DeferredFxaa,
    /// Hybrid light-field shading; TSR when the config asks for it, no
    /// anti-aliasing otherwise.
    Hlfs,
    /// Hybrid light-field shading with FXAA, never TSR.
    HlfsFxaa,
    /// One forward pass for opaque geometry, no G-buffer; FXAA.
    Forward,
    /// A test cube straight into the output: checks the device and surface.
    Simple,
}

/// The anti-aliasing a preset ends its scene-linear chain with.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Aa {
    Tsr,
    Fxaa,
    None,
}

impl Preset {
    pub const ALL: [Preset; 6] = [
        Preset::Deferred,
        Preset::DeferredFxaa,
        Preset::Hlfs,
        Preset::HlfsFxaa,
        Preset::Forward,
        Preset::Simple,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Preset::Deferred => "Deferred",
            Preset::DeferredFxaa => "Deferred + FXAA",
            Preset::Hlfs => "HLFS",
            Preset::HlfsFxaa => "HLFS + FXAA",
            Preset::Forward => "Forward",
            Preset::Simple => "Simple",
        }
    }

    fn hlfs(self) -> bool {
        matches!(self, Preset::Hlfs | Preset::HlfsFxaa)
    }

    fn aa(self, config: &RendererConfig) -> Aa {
        let tsr = config.tsr_quality.is_some();
        match self {
            Preset::Deferred if tsr => Aa::Tsr,
            Preset::Hlfs if tsr => Aa::Tsr,
            Preset::Hlfs | Preset::Simple => Aa::None,
            Preset::Deferred | Preset::DeferredFxaa | Preset::HlfsFxaa | Preset::Forward => {
                Aa::Fxaa
            }
        }
    }

    /// The draw phases the preset runs, in frame order.
    pub fn phases(self) -> Vec<Phase> {
        match self {
            Preset::Simple => Vec::new(),
            Preset::Forward => vec![
                Phase::ShadowCasters,
                Phase::Forward,
                Phase::Overlay,
                Phase::Transparent,
            ],
            _ => vec![
                Phase::ShadowCasters,
                Phase::GBuffer,
                Phase::Overlay,
                Phase::Transparent,
            ],
        }
    }

    /// How the preset's lighting takes the indirect-specular layers.
    fn specular_layers(self) -> SpecularLayers {
        match self {
            Preset::Forward | Preset::Simple => SpecularLayers::None,
            _ if self.hlfs() => SpecularLayers::CompositedAfter,
            _ => SpecularLayers::Read,
        }
    }

    /// The preset's nodes for `config`, in execution order: the spine's
    /// passes it names, and the frame's slots and draw phases, each expanded
    /// to the nodes the registry holds there.
    pub fn nodes(self, registry: &NodeRegistry, config: &RendererConfig) -> Vec<NodeInstance> {
        let mut nodes = Nodes {
            registry,
            list: Vec::new(),
        };
        if self == Preset::Simple {
            nodes.pass("simple_cube");
            return nodes.list;
        }
        let aa = self.aa(config);

        // Scene prep: batches, per-frame tables, the lights' shadow views and
        // their casters, culling.
        nodes.pass("object_batch");
        nodes.slot(Slot::Prepare);
        nodes.pass("shadow_matrix");
        nodes.pass("shadow_dirty");
        nodes.phase(Phase::ShadowCasters);
        for name in ["background", "indirect_dispatch", "hiz", "occlusion_cull"] {
            nodes.pass(name);
        }
        if !self.hlfs() {
            nodes.pass("light_cull");
        }

        // Geometry and lighting.
        if self == Preset::Forward {
            nodes.phase(Phase::Forward);
            nodes.pass("cloud_trace");
        } else {
            nodes.phase(Phase::GBuffer);
            nodes.slot(Slot::Geometry);
            nodes.slot(Slot::LightingInputs);
            nodes.pass(if self.hlfs() { "hlfs" } else { "deferred_light" });
        }
        nodes.slot(Slot::Lighting);

        // Over the lit scene, then the scene-linear chain: media,
        // transparency, anti-aliasing, the resolved colour, the tone map.
        nodes.phase(Phase::Overlay);
        nodes.slot(Slot::Media);
        nodes.phase(Phase::Transparent);
        match aa {
            Aa::Tsr => {
                // Features that rotate samples over pixels ask TSR for its
                // stochastic mask themselves (`helio_pass_tsr::StochasticSamples`).
                nodes.with(
                    "tsr",
                    [
                        ("quality", Value::from(tsr_quality(config))),
                        ("transparency_reactivity", Value::from(self.hlfs())),
                    ],
                );
            }
            Aa::Fxaa => nodes.pass("fxaa"),
            Aa::None => {}
        }
        nodes.slot(Slot::Resolved);
        nodes.pass("post_process");

        // Over the final image.
        nodes.slot(Slot::Display);
        nodes.slot(Slot::Output);
        nodes.list
    }

    /// The preset for `config` as a graph: On Frame Start, its nodes on one
    /// execution chain, the Output, and the host inputs it reads.
    pub fn asset(self, registry: &NodeRegistry, config: &RendererConfig) -> GraphAsset {
        let mut asset = GraphAsset {
            name: self.name().into(),
            ..Default::default()
        };
        asset.nodes.push(node("start", core_nodes::ON_FRAME_START));
        let chain = self.nodes(registry, config);
        let ids: Vec<String> = chain.iter().map(|node| node.id.clone()).collect();
        asset.nodes.extend(chain);
        asset.nodes.push(node("output", core_nodes::OUTPUT));
        let mut from = "start".to_string();
        for id in ids.into_iter().chain(["output".to_string()]) {
            asset.connections.push(Connection {
                from: PinRef::new(&from, "out"),
                to: PinRef::new(&id, "in"),
            });
            from = id;
        }
        if self != Preset::Simple {
            for descriptor in registry.placed(HOST_INPUTS) {
                let id = descriptor.kind.replace('.', "_");
                asset.nodes.push(node(&id, &descriptor.kind));
            }
        }
        asset
    }

    /// The preset as a pipeline over `registry`'s nodes.
    pub fn pipeline_with(self, registry: Arc<NodeRegistry>) -> GraphAssetPipeline {
        let pipeline = GraphAssetPipeline::made(
            Arc::new(move |registry, env| {
                let config = env
                    .require::<RendererConfig>()
                    .expect("a renderer build supplies its config")
                    .clone();
                if self.hlfs() {
                    let format = helio_pass_hlfs::HlfsPass::preferred_output_format(env.device);
                    env.insert_setting(SceneColorFormat(format));
                }
                env.insert_setting(PhaseList(self.phases()));
                env.insert_setting(self.specular_layers());
                if self.hlfs() && self.aa(&config) == Aa::Tsr {
                    env.insert_setting(ReactiveMask);
                }
                self.asset(registry, &config)
            }),
            registry,
        );
        if self == Preset::Simple {
            pipeline
        } else {
            pipeline.with_swap_policy(crate::default_swap_policy())
        }
    }

    /// The preset as a pipeline over the default features' nodes.
    pub fn pipeline(self) -> GraphAssetPipeline {
        self.pipeline_with(registry())
    }

    /// The preset over the default features and an engine's `extra`
    /// features (a plugin's, for an object kind Helio does not know).
    pub fn with_features(self, extra: &helio_feature_api::FeatureSet) -> PresetGraph {
        self.with_registry(Arc::new(crate::graphs::registry_with(extra)))
    }

    /// The preset over nodes a host made once ([`crate::graphs::registry_with`])
    /// and shares between its pipelines.
    pub fn with_registry(self, registry: Arc<NodeRegistry>) -> PresetGraph {
        PresetGraph {
            preset: self,
            registry,
        }
    }

    /// Builds the preset's graph for `ctx`.
    pub fn build(self, mut ctx: PassBuildContext<'_>) -> RenderGraph {
        ctx.owns_device = false;
        Box::new(self.pipeline()).into_graph(ctx)
    }
}

impl IntoGraph for Preset {
    fn into_graph(self: Box<Self>, ctx: PassBuildContext<'_>) -> RenderGraph {
        self.build(ctx)
    }
}

/// A preset over its own feature list ([`Preset::with_features`]).
pub struct PresetGraph {
    preset: Preset,
    registry: Arc<NodeRegistry>,
}

impl PresetGraph {
    /// The nodes it builds from.
    pub fn registry(&self) -> &Arc<NodeRegistry> {
        &self.registry
    }

    /// Builds the preset's graph for `ctx` ([`Preset::build`] over these
    /// features).
    pub fn build(self, ctx: PassBuildContext<'_>) -> RenderGraph {
        Box::new(self).into_graph(ctx)
    }
}

impl IntoGraph for PresetGraph {
    fn into_graph(self: Box<Self>, mut ctx: PassBuildContext<'_>) -> RenderGraph {
        ctx.owns_device = false;
        Box::new(self.preset.pipeline_with(self.registry)).into_graph(ctx)
    }
}

/// The default features' nodes ([`crate::graphs::base_registry`]), made once.
pub fn registry() -> Arc<NodeRegistry> {
    static REGISTRY: OnceLock<Arc<NodeRegistry>> = OnceLock::new();
    Arc::clone(REGISTRY.get_or_init(|| Arc::new(crate::graphs::base_registry())))
}

pub fn deferred(ctx: PassBuildContext<'_>) -> RenderGraph {
    Preset::Deferred.build(ctx)
}

pub fn deferred_fxaa(ctx: PassBuildContext<'_>) -> RenderGraph {
    Preset::DeferredFxaa.build(ctx)
}

pub fn hlfs(ctx: PassBuildContext<'_>) -> RenderGraph {
    Preset::Hlfs.build(ctx)
}

pub fn hlfs_fxaa(ctx: PassBuildContext<'_>) -> RenderGraph {
    Preset::HlfsFxaa.build(ctx)
}

pub fn forward(ctx: PassBuildContext<'_>) -> RenderGraph {
    Preset::Forward.build(ctx)
}

pub fn simple(ctx: PassBuildContext<'_>) -> RenderGraph {
    Preset::Simple.build(ctx)
}

fn tsr_quality(config: &RendererConfig) -> &'static str {
    use helio::TsrQuality;
    match config.tsr_quality.unwrap_or_default() {
        TsrQuality::Performance => "performance",
        TsrQuality::Balanced => "balanced",
        TsrQuality::Quality => "quality",
        TsrQuality::Native => "native",
    }
}

fn node(id: &str, kind: &str) -> NodeInstance {
    NodeInstance {
        id: id.into(),
        kind: kind.into(),
        params: Params::default(),
        position: [0.0, 0.0],
    }
}

/// A preset's node list as it is made.
struct Nodes<'r> {
    registry: &'r NodeRegistry,
    list: Vec<NodeInstance>,
}

impl Nodes<'_> {
    fn pass(&mut self, name: &str) {
        self.list.push(node(name, &kind(name)));
    }

    fn with<const N: usize>(&mut self, name: &str, params: [(&str, Value); N]) {
        let mut instance = node(name, &kind(name));
        for (key, value) in params {
            instance.params.insert(key.to_string(), value);
        }
        self.list.push(instance);
    }

    /// The nodes placed at `point`, by order then kind; a node's id is its
    /// kind without the `helio.pass.` prefix.
    fn at(&mut self, point: ExtensionPoint) {
        for descriptor in self.registry.placed(point) {
            let id = descriptor
                .kind
                .trim_start_matches("helio.pass.")
                .replace('.', "_");
            self.list.push(node(&id, &descriptor.kind));
        }
    }

    fn slot(&mut self, slot: Slot) {
        self.at(slot.point());
    }

    /// A draw phase: the compute it needs first, the drawer that opens it,
    /// then its other drawers.
    ///
    /// # Panics
    ///
    /// When a phase whose attachments nobody else clears has drawers but not
    /// exactly one opener: its frame would show stale pixels, or clear twice.
    fn phase(&mut self, phase: Phase) {
        let openers = self.registry.placed(phase.open()).len();
        let drawers = self.registry.placed(phase.draw()).len();
        if phase.needs_opener() && drawers > 0 {
            assert!(
                openers == 1,
                "phase `{}` has {drawers} drawers and {openers} openers; it needs exactly one",
                phase.name()
            );
        }
        self.at(phase.before());
        self.at(phase.open());
        self.at(phase.draw());
    }
}
