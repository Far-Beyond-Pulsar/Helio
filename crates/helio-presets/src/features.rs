//! The features Helio's presets are made of, and the renderer's own nodes.
//!
//! Every pass node is registered by the crate that owns the pass: the spine's
//! by `helio-spine`, each feature's by its crate (`helio_pass_ssr::
//! SsrFeature`, …). What is left here is what belongs to the renderer:
//!
//! - the debug-draw nodes, whose pass and state are the renderer's;
//! - a preparer that turns the renderer's config into the settings the
//!   features read ([`helio_spine::SpineSettings`], the lighting's
//!   [`helio_frame_api::lighting::LightingSettings`], the reflections'
//!   switches, the perf overlay's mode),
//!   so no feature reads `RendererConfig`;
//! - [`default_features`], the feature list Helio's presets build from.

use std::sync::{Arc, Mutex};

use helio::{DebugDrawState, RendererConfig};
use helio_feature_api::{
    pass_node, Feature, FeatureRegistrar, FeatureSet, NodeBuild, NodeDescriptor, NodeRegistry,
    ParamDesc, Params, PinDesc,
};
use helio_frame_api::color::ReactiveMask;
use helio_frame_api::lighting::{model as model_ids, LightingModel, LightingModelId};
use helio_frame_api::{FrameNode, Phase, PhaseList, Slot};

pub use helio_feature_api::{pass_kind as kind, SceneColorFormat};
pub use helio_frame_api::{host_kind, HOST_INPUTS};

/// Helio's features, in registration order: the spine, then every feature
/// that joins the frame's slots and phases, then the renderer's own nodes.
/// An engine adds its features to this set ([`crate::Preset::with_features`]):
/// the features of object kinds Helio does not know (meshes, a voxel world)
/// come from the plugin that owns the kind.
pub fn default_features() -> FeatureSet {
    FeatureSet::new()
        .with(helio_spine::SpineFeature)
        .with(helio_pass_ssr::SsrFeature)
        .with(helio_pass_planar_reflection::PlanarReflectionFeature)
        .with(helio_pass_fxaa::FxaaFeature)
        .with(helio_pass_buffer_view::BufferViewFeature)
        .with(helio_pass_perf_overlay::PerfOverlayFeature)
        .with(helio_pass_debug_overlay::DebugOverlayFeature)
        .with(RendererNodes)
}

/// Kind of the frame setup node: the first node of every shipped graph.
pub const FRAME_SETUP: &str = "helio.frame.setup";

/// The frame setup node: which lighting model the graph resolves its lighting
/// with, and so the draw phases it runs, how the indirect-specular layers are
/// taken and where the lighting inputs run, as the settings the nodes after
/// it read. A `reactive_mask` asks the temporal resolve for the reactive mask
/// (transparency that a model leaves to TSR) when the frame resolves temporally.
///
/// `models` are the lighting models the registry's features registered; a
/// model nobody registered resolves to the spine's unlit stand-in, with one
/// warning. It adds no pass, so it is the same data in a `.rendergraph` as in a
/// preset.
pub(crate) fn frame_setup_node(models: Vec<LightingModel>) -> NodeDescriptor {
    NodeDescriptor::new(
        FRAME_SETUP,
        "Frame: Lighting Model",
        "Passes: Scene Prep",
        "Chooses the lighting model the frame resolves with (the draw phases, specular layers and lighting-input order follow it). Place it first.",
        move |build| {
            let wanted = build.text("model");
            let model = models
                .iter()
                .find(|model| model.id == wanted)
                .or_else(|| models.iter().find(|model| model.id == model_ids::UNLIT))
                .ok_or_else(|| "the spine registers no unlit lighting model".to_string())?;
            if model.id != wanted {
                crate::presets::warn_model_missing_once(&wanted, model.id);
            }
            build.env.insert_setting(LightingModelId(model.id));
            build
                .env
                .insert_setting(PhaseList(crate::presets::phases_for(model.shading)));
            build.env.insert_setting(model.specular_layers);
            build.env.insert_setting(model.inputs_order);
            // The mask is for the temporal resolve: a frame that does not
            // resolve temporally (the project's TSR is off) has no reader for it.
            let temporal = build
                .env
                .setting::<RendererConfig>()
                .is_some_and(|config| config.tsr_quality.is_some());
            if build.boolean("reactive_mask") && temporal {
                build.env.insert_setting(ReactiveMask);
            }
            Ok(())
        },
    )
    .with_param(ParamDesc::text("model", "Lighting model", model_ids::DEFERRED))
    .with_param(ParamDesc::boolean("reactive_mask", "Reactive mask", false))
    .with_pins(vec![PinDesc::flow("in")], vec![PinDesc::flow("out")])
}

/// Kind of the anti-aliasing node that follows the project's setting.
pub const ANTIALIASING: &str = "helio.pass.antialiasing";

/// The scene-linear chain's anti-aliasing as the renderer's config chooses it:
/// the TSR node at the config's quality when the project resolves temporally,
/// otherwise its `fallback` (the FXAA node, or nothing). It runs the TSR and
/// FXAA nodes' own constructors, so a graph that names it is the same data
/// whatever the project's setting, and follows the setting as a preset does.
fn antialiasing_node() -> NodeDescriptor {
    // The nodes it chooses between, as their owners register them.
    let mut owners = NodeRegistry::new();
    FeatureSet::new()
        .with(helio_spine::SpineFeature)
        .with(helio_pass_fxaa::FxaaFeature)
        .register_into(&mut owners);
    let tsr = owners.get(&kind("tsr")).cloned();
    let fxaa = owners.get(&kind("fxaa")).cloned();
    pass_node(
        "antialiasing",
        "Anti-Aliasing: Project Setting",
        "Passes: Post",
        "TSR at the project's quality when the project enables temporal resolve, otherwise the fallback (FXAA or none).",
        move |build| {
            let quality = build
                .env
                .setting::<RendererConfig>()
                .and_then(|config| config.tsr_quality)
                .map(crate::presets::tsr_quality_name);
            let (node, params) = match quality {
                Some(quality) => {
                    let mut params = Params::default();
                    params.insert("quality".into(), quality.into());
                    params.insert(
                        "transparency_reactivity".into(),
                        build.boolean("transparency_reactivity").into(),
                    );
                    (tsr.as_ref(), params)
                }
                None if build.text("fallback") == "fxaa" => (fxaa.as_ref(), Params::default()),
                None => return Ok(()),
            };
            let node = node.ok_or("the TSR or FXAA node is not registered")?;
            let params = node.effective_params(&params);
            let mut chosen = NodeBuild {
                graph: &mut *build.graph,
                env: &mut *build.env,
                params: &params,
                node_id: build.node_id,
            };
            node.run(&mut chosen)
        },
    )
    .with_param(ParamDesc::choice(
        "fallback",
        "Without TSR",
        &["fxaa", "none"],
        "fxaa",
    ))
    .with_param(ParamDesc::boolean(
        "transparency_reactivity",
        "Transparency reactivity",
        false,
    ))
}

/// The renderer's nodes and the config bridge.
struct RendererNodes;

impl Feature for RendererNodes {
    fn id(&self) -> &'static str {
        "helio.renderer"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        // Shader hot reload may swap these alone: they take only
        // renderer-owned handles (camera, debug camera and cull-stats buffers,
        // debug-draw state, SceneDB) and plain configuration, the same objects
        // in the live and the replacement graph (`helio_core::SwapPolicy`).
        registrar.swappable::<helio::DebugDrawPass>();
        registrar.prepare(Arc::new(|env| {
            let Some(config) = env.setting::<RendererConfig>().cloned() else {
                return;
            };
            env.insert_setting(helio_spine::SpineSettings {
                background_color: config.background_color,
            });
            env.insert_setting(helio_frame_api::lighting::LightingSettings {
                debug_mode: config.debug_mode,
                environment_reflections: config.enable_environment_reflections,
            });
            env.insert_setting(helio_pass_ssr::SsrSettings {
                enabled: config.enable_ssr,
            });
            env.insert_setting(helio_pass_planar_reflection::PlanarReflectionSettings {
                enabled: config.enable_planar_reflections,
            });
            env.insert_setting(crate::convert_perf_mode(config.perf_overlay_mode));
        }));

        registrar.node(antialiasing_node());

        registrar.node(
            pass_node(
                "debug_draw_scene",
                "Overlay: Grid + Volume Bounds (Editor)",
                "Passes: Overlays",
                "Editor grid and scene volume bounds, depth-tested against the scene.",
                |build| {
                    let state = build.env.require::<Arc<Mutex<DebugDrawState>>>()?.clone();
                    build.graph.add_pass(Box::new(helio::DebugDrawPass::new(
                        build.env.device,
                        build.env.debug_camera_buffer,
                        build.env.scene_format(),
                        state,
                        true,
                        true,
                    )));
                    Ok(())
                },
            )
            .drawer(Phase::Overlay, 40),
        );

        registrar.node(
            pass_node(
                "debug_draw",
                "Output: Debug Lines",
                "Passes: Output",
                "User debug lines and triangles over the final image.",
                |build| {
                    let state = build.env.require::<Arc<Mutex<DebugDrawState>>>()?.clone();
                    build.graph.add_pass(Box::new(helio::DebugDrawPass::new(
                        build.env.device,
                        build.env.debug_camera_buffer,
                        build.env.frame.surface_format,
                        state,
                        false,
                        false,
                    )));
                    Ok(())
                },
            )
            .in_slot(Slot::Display, 30),
        );
    }
}
