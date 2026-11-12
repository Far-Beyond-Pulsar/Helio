//! Helio's shipped pipelines, as node graphs.
//!
//! A preset is a graph of the spine's passes and the frame's slots and draw
//! phases (`helio_frame_api`). It is the same data for every renderer config:
//! where it follows the project's anti-aliasing it names the anti-aliasing
//! node (`features::ANTIALIASING`), which builds TSR at the config's quality or
//! the fallback when the graph builds. The preset chooses a lighting model by
//! id (a plugin registers it; the spine's unlit stand-in is used when it is
//! absent). Every other node joins where its feature placed it (a slot, or a
//! phase's drawers and the compute before them), so a feature (meshes, GI, the
//! clouds, an engine's object kinds) joins every preset without a preset
//! naming it, and builds nothing when its own settings turn it off.
//!
//! A host builds a renderer with one by function (`presets::deferred`) or by
//! value (`with_graph(Preset::Deferred)`). Project-authored graphs are fixed
//! [`GraphAsset`]s built by `helio_graph_asset::GraphAssetPipeline`.
//!
//! Every preset reads its GPU timings a frame late instead of waiting for
//! them (the renderer's own poll, when it owns the device, or the host's
//! submits deliver them), as the engine's graphs always have.

use std::sync::{Arc, Mutex, OnceLock};

use helio::{IntoGraph, PassBuildContext, PipelineFailure};
use helio_core::RenderGraph;
use helio_feature_api::ExtensionPoint;
use helio_frame_api::lighting::{
    model, resolve_model, Antialiasing, LightingModel, ModelPipeline, Shading,
};
use helio_frame_api::{Phase, Slot, HOST_INPUTS};
use helio_graph_asset::{
    core_nodes, Connection, GraphAsset, GraphAssetPipeline, NodeInstance, NodeRegistry, Params,
    PinRef,
};
use serde_json::Value;

use crate::features::kind;

/// Logs, once per `wanted` id, that no installed feature registered the
/// lighting model and `fallback` is used instead.
pub(crate) fn warn_model_missing_once(wanted: &str, fallback: &str) {
    static WARNED: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let mut warned = WARNED
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if !warned.iter().any(|known| known == wanted) {
        warned.push(wanted.to_string());
        log::warn!(
            "lighting model `{wanted}` is not registered (is the plugin that brings it disabled?);              rendering with the `{fallback}` stand-in"
        );
    }
}

/// One of Helio's shipped pipelines, or a pipeline a lighting model brings.
/// A preset chooses its lighting model by id; what the model is, where its
/// passes join and what it reads come from the registry
/// ([`resolve_model`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Preset {
    /// The `deferred` model; TSR when the project enables it, FXAA otherwise.
    Deferred,
    /// The `deferred` model with FXAA, never TSR.
    DeferredFxaa,
    /// The `forward` model: one forward pass for opaque geometry, no
    /// G-buffer; FXAA.
    Forward,
    /// A pipeline a registered lighting model brings
    /// ([`helio_frame_api::lighting::LightingModel::pipelines`]).
    Model(ModelPipeline),
}

/// The draw phases a frame runs for a lighting model that shades `shading`,
/// in frame order. The frame setup node publishes them as the graph's
/// [`PhaseList`].
pub(crate) fn phases_for(shading: Shading) -> Vec<Phase> {
    match shading {
        Shading::Forward => vec![
            Phase::ShadowCasters,
            Phase::Forward,
            Phase::Overlay,
            Phase::Transparent,
        ],
        Shading::GBuffer => vec![
            Phase::ShadowCasters,
            Phase::GBuffer,
            Phase::Overlay,
            Phase::Transparent,
        ],
    }
}

impl Preset {
    /// The presets Helio ships, whatever plugins are installed.
    pub const ALL: [Preset; 3] = [Preset::Deferred, Preset::DeferredFxaa, Preset::Forward];

    pub fn name(self) -> &'static str {
        match self {
            Preset::Deferred => "Deferred",
            Preset::DeferredFxaa => "Deferred + FXAA",
            Preset::Forward => "Forward",
            Preset::Model(pipeline) => pipeline.label,
        }
    }

    /// The id of the lighting model the preset chooses.
    pub fn model_id(self) -> &'static str {
        match self {
            Preset::Deferred | Preset::DeferredFxaa => model::DEFERRED,
            Preset::Forward => model::FORWARD,
            Preset::Model(pipeline) => pipeline.model,
        }
    }

    /// The lighting model the preset gets from `registry`: the registered one
    /// it chose, or the spine's unlit stand-in when nobody registered it.
    ///
    /// The first fallback for each missing id logs one warning naming it, so
    /// a project that disabled the lights plugin does not silently get an
    /// unlit scene (the set is touched only on the fallback path, which a
    /// graph build takes, never a frame).
    pub fn model(self, registry: &NodeRegistry) -> &LightingModel {
        let wanted = self.model_id();
        let resolved = resolve_model(registry, wanted);
        if resolved.id != wanted {
            warn_model_missing_once(wanted, resolved.id);
        }
        resolved
    }

    fn antialiasing(self) -> Antialiasing {
        match self {
            Preset::Deferred => Antialiasing::TemporalOrFxaa,
            Preset::DeferredFxaa | Preset::Forward => Antialiasing::Fxaa,
            Preset::Model(pipeline) => pipeline.aa,
        }
    }

    /// The draw phases the preset runs with `registry`'s models, in frame
    /// order.
    pub fn phases(self, registry: &NodeRegistry) -> Vec<Phase> {
        phases_for(self.model(registry).shading)
    }

    /// The preset's nodes, in execution order: the spine's passes it names,
    /// and the frame's slots and draw phases, each expanded to the nodes the
    /// registry holds there. They are the same for every renderer config:
    /// where the preset follows the project's anti-aliasing it names the
    /// anti-aliasing node, which chooses when the graph builds.
    pub fn nodes(self, registry: &NodeRegistry) -> Vec<NodeInstance> {
        let mut nodes = Nodes {
            registry,
            list: Vec::new(),
        };
        let aa = self.antialiasing();
        let model = self.model(registry);

        // What the frame is: the lighting model (and so the draw phases, the
        // specular layers, where the lighting inputs run) and whether the
        // temporal resolve reads a reactive mask. A node, so a `.rendergraph`
        // states the same thing as data.
        nodes.raw(
            "frame",
            crate::features::FRAME_SETUP,
            [
                ("model", Value::from(self.model_id())),
                (
                    "reactive_mask",
                    Value::from(model.reactive_transparency && aa != Antialiasing::Fxaa),
                ),
            ],
        );

        // Scene prep: per-frame tables (an object kind's batches first), the
        // shadow views and what casts into them (the phase's compute and
        // drawers), the background, visibility against the depth pyramid
        // (a kind's culling, the lights' binning).
        nodes.slot(Slot::Prepare);
        nodes.phase(Phase::ShadowCasters);
        nodes.pass("background");
        nodes.slot(Slot::Visibility);

        // Geometry and lighting: the forward phase draws lit; otherwise the
        // G-buffer is resolved by the installed lighting model, or unlit.
        if model.shading == Shading::Forward {
            nodes.phase(Phase::Forward);
            nodes.slot(Slot::LightingInputs);
        } else {
            nodes.phase(Phase::GBuffer);
            nodes.slot(Slot::Geometry);
            nodes.slot(Slot::LightingInputs);
            nodes.lighting_model(model);
        }
        nodes.slot(Slot::Lighting);

        // Over the lit scene, then the scene-linear chain: media,
        // transparency, anti-aliasing, the resolved colour, the tone map.
        nodes.phase(Phase::Overlay);
        nodes.slot(Slot::Media);
        nodes.phase(Phase::Transparent);
        // Features that rotate samples over pixels ask TSR for its
        // stochastic mask themselves (`helio_pass_tsr::StochasticSamples`).
        match aa {
            Antialiasing::Fxaa => nodes.pass("fxaa"),
            Antialiasing::TemporalOrFxaa | Antialiasing::TemporalOrNone => nodes.with(
                "antialiasing",
                [
                    (
                        "fallback",
                        Value::from(if aa == Antialiasing::TemporalOrFxaa {
                            "fxaa"
                        } else {
                            "none"
                        }),
                    ),
                    (
                        "transparency_reactivity",
                        Value::from(model.reactive_transparency),
                    ),
                ],
            ),
        }
        nodes.slot(Slot::Resolved);
        nodes.pass("post_process");

        // Over the final image.
        nodes.slot(Slot::Display);
        nodes.slot(Slot::Output);
        nodes.list
    }

    /// The preset as a graph: On Frame Start, its nodes on one execution
    /// chain, the Output, and the host inputs it reads.
    pub fn asset(self, registry: &NodeRegistry) -> GraphAsset {
        let mut asset = GraphAsset {
            name: self.name().into(),
            ..Default::default()
        };
        asset.nodes.push(node("start", core_nodes::ON_FRAME_START));
        let chain = self.nodes(registry);
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
        for descriptor in registry.placed(HOST_INPUTS) {
            let id = descriptor.kind.replace('.', "_");
            asset.nodes.push(node(&id, &descriptor.kind));
        }
        asset
    }

    /// The preset as a pipeline over `registry`'s nodes.
    pub fn pipeline_with(self, registry: Arc<NodeRegistry>) -> GraphAssetPipeline {
        let asset = Arc::new(self.asset(&registry));
        GraphAssetPipeline::new(asset, registry).with_swap_policy(crate::default_swap_policy())
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

    /// Builds the preset's graph for `ctx`. A preset is the engine's own graph:
    /// one that does not build is a bug, so this panics with the failure (a
    /// renderer reports the panic as the pipeline's failure; the
    /// [`IntoGraph`] impl returns it).
    pub fn build(self, ctx: PassBuildContext<'_>) -> RenderGraph {
        let name = self.name();
        Box::new(self)
            .into_graph(ctx)
            .unwrap_or_else(|failure| panic!("preset `{name}` does not build: {failure}"))
    }
}

impl IntoGraph for Preset {
    fn into_graph(
        self: Box<Self>,
        mut ctx: PassBuildContext<'_>,
    ) -> Result<RenderGraph, PipelineFailure> {
        ctx.owns_device = false;
        Box::new(self.pipeline()).into_graph(ctx)
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
        let name = self.preset.name();
        Box::new(self)
            .into_graph(ctx)
            .unwrap_or_else(|failure| panic!("preset `{name}` does not build: {failure}"))
    }
}

impl IntoGraph for PresetGraph {
    fn into_graph(
        self: Box<Self>,
        mut ctx: PassBuildContext<'_>,
    ) -> Result<RenderGraph, PipelineFailure> {
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

pub fn forward(ctx: PassBuildContext<'_>) -> RenderGraph {
    Preset::Forward.build(ctx)
}

/// The TSR node's `quality` parameter for `quality`.
pub(crate) fn tsr_quality_name(quality: helio::TsrQuality) -> &'static str {
    use helio::TsrQuality;
    match quality {
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
        self.raw(name, &kind(name), params);
    }

    /// A node of any `kind` (not a pass node), with `params`.
    fn raw<const N: usize>(&mut self, id: &str, kind: &str, params: [(&str, Value); N]) {
        let mut instance = node(id, kind);
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

    /// The lighting model's nodes at its point, or the spine's unlit stand-in
    /// when none are placed there (the preset still shows its surfaces,
    /// unlit).
    fn lighting_model(&mut self, model: &LightingModel) {
        match model.point {
            Some(point) if !self.registry.placed(point).is_empty() => self.at(point),
            _ => self.pass("unlit"),
        }
    }

    /// A draw phase: the compute it needs first, the drawer that opens it,
    /// then its other drawers. The G-buffer is the frame's: when no kind
    /// opens its phase, the spine clears it (`gbuffer_clear`), so its
    /// readers and the drawers that load it find it written.
    ///
    /// # Panics
    ///
    /// When a phase whose attachments nobody else clears has more than one
    /// opener (it would clear twice), or drawers and no opener nor a clear of
    /// the spine's (its frame would show stale pixels).
    fn phase(&mut self, phase: Phase) {
        let openers = self.registry.placed(phase.open()).len();
        let drawers = self.registry.placed(phase.draw()).len();
        let spine_clears = phase == Phase::GBuffer && openers == 0;
        if phase.needs_opener() {
            assert!(
                openers <= 1 && (drawers == 0 || openers == 1 || spine_clears),
                "phase `{}` has {drawers} drawers and {openers} openers; it needs exactly one",
                phase.name()
            );
        }
        self.at(phase.before());
        if spine_clears {
            self.pass("gbuffer_clear");
        }
        self.at(phase.open());
        self.at(phase.draw());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use helio_feature_api::{pass_node, FeatureRegistrar, FeatureSet};
    use helio_frame_api::lighting::{InputsOrder, RegisterLightingModel, SpecularLayers};

    const POINT: ExtensionPoint = ExtensionPoint("test.lighting.model");

    /// A plugin that registers a model with a pass at its point.
    struct Plugin(LightingModel);

    impl helio_feature_api::Feature for Plugin {
        fn id(&self) -> &'static str {
            "test.lighting"
        }

        fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
            registrar.lighting_model(self.0.clone());
            registrar.node(pass_node("test_lit", "Test lit", "Test", "", |_| Ok(())).at(POINT, 0));
        }
    }

    fn kinds(preset: Preset, features: &FeatureSet) -> Vec<String> {
        let registry = crate::graphs::registry_with(features);
        preset
            .nodes(&registry)
            .into_iter()
            .map(|node| node.kind)
            .collect()
    }

    #[test]
    fn an_unregistered_model_resolves_to_the_unlit_stand_in() {
        let registry = crate::graphs::base_registry();
        for preset in Preset::ALL {
            assert_eq!(
                preset.model(&registry).id,
                model::UNLIT,
                "{}",
                preset.name()
            );
        }
        let kinds = kinds(Preset::Deferred, &FeatureSet::new());
        assert!(kinds.contains(&"helio.pass.unlit".to_string()));
        // The forward preset has no model of its own either: it draws the
        // G-buffer unlit.
        assert!(Preset::Forward.phases(&registry).contains(&Phase::GBuffer));
    }

    /// Captures the warnings that name the test's own model id.
    struct Capture(Mutex<Vec<String>>);

    impl log::Log for Capture {
        fn enabled(&self, _: &log::Metadata<'_>) -> bool {
            true
        }
        fn log(&self, record: &log::Record<'_>) {
            let message = record.args().to_string();
            if message.contains("test.once.missing") {
                self.0.lock().unwrap().push(message);
            }
        }
        fn flush(&self) {}
    }

    #[test]
    fn a_missing_model_warns_once_naming_its_id() {
        static CAPTURE: Capture = Capture(Mutex::new(Vec::new()));
        log::set_logger(&CAPTURE).expect("this is the only test that installs a logger");
        log::set_max_level(log::LevelFilter::Warn);
        let registry = crate::graphs::base_registry();
        let preset = Preset::Model(ModelPipeline {
            id: "test.once",
            label: "Once",
            description: "",
            model: "test.once.missing",
            aa: Antialiasing::TemporalOrNone,
        });
        for _ in 0..3 {
            assert_eq!(preset.model(&registry).id, model::UNLIT);
        }
        let warnings = CAPTURE.0.lock().unwrap();
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("`test.once.missing`") && warnings[0].contains("`unlit`"));
    }

    #[test]
    fn a_registered_model_is_placed_and_read_by_id() {
        let registered = LightingModel::new(model::DEFERRED, Shading::GBuffer)
            .at(POINT)
            .with_specular_layers(SpecularLayers::CompositedAfter)
            .with_inputs_order(InputsOrder::AfterLighting);
        let features = FeatureSet::new().with(Plugin(registered));
        let registry = crate::graphs::registry_with(&features);
        let resolved = Preset::DeferredFxaa.model(&registry);
        assert_eq!(resolved.id, model::DEFERRED);
        assert_eq!(resolved.specular_layers, SpecularLayers::CompositedAfter);
        let kinds = kinds(Preset::Deferred, &features);
        assert!(kinds.contains(&"helio.pass.test_lit".to_string()));
        assert!(!kinds.contains(&"helio.pass.unlit".to_string()));
        // Another id is still the stand-in.
        assert_eq!(Preset::Forward.model(&registry).id, model::UNLIT);
    }
}
