//! Helio's default pipeline as render-graph nodes.
//!
//! One node per stage of the default deferred graph, so a render-graph asset can
//! reuse, reorder, replace or surround them. A stage builds the passes the
//! default graph builds there; its **pins are not written here**. They come from
//! [`probe_pins`], which builds the stages on a device and reads what their
//! passes declare (see `helio_graph_asset::probe`). A stage with no probe result
//! has flow pins only, which is all building needs.
//!
//! - [`base_registry`]: core nodes and the stages, flow pins only.
//! - [`node_registry`]: the same with the probed pins, plus the SceneDB node.
//! - [`default_graph_asset`]: the stages in order, with a wire for every
//!   resource a stage reads from an earlier stage and every SceneDB buffer it
//!   consumes.
//!
//! Stages share the performance-overlay state and the HDR lighting
//! configuration through [`StageState`], created by the first stage built.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use helio::RendererConfig;
use helio_graph_asset::{
    core_nodes, probe, BuildEnv, Connection, GraphAsset, NodeBuild, NodeDescriptor, NodeInstance,
    NodePins, NodeRegistry, PinDesc, PinKind, PinRef, PinType, Probe, SCENEDB_PREFIX,
};
use helio_pass_perf_overlay::PerfOverlayShared;

use crate::{
    add_common_early_passes, add_geometry_passes, add_final_passes, add_late_passes,
    add_scene_linear_chain, declare_common_external_inputs, DecalPass, DeferredLightPass,
    LightCullPass, PerfOverlayAnalyzerPass, PerfOverlayCostAnalyzerPass, PlanarReflectionPass,
    PostProcessPass, SceneAa, SsrPass, TsrPass, FOGGED_HDR_FORMAT,
};

pub const SCENE_INPUTS: &str = "helio.scene_inputs";
/// Node kind of the node for SceneDB GPU buffer `key`.
pub fn scenedb_kind(key: &str) -> String {
    format!("helio.scenedb.{key}")
}
pub const EARLY: &str = "helio.stage.early";
pub const GEOMETRY: &str = "helio.stage.geometry";
pub const LIGHTING: &str = "helio.stage.lighting";
pub const ATMOSPHERE_OVERLAYS: &str = "helio.stage.late";
pub const POST_PROCESS: &str = "helio.stage.post_process";
pub const FINAL: &str = "helio.stage.final";

/// The default pipeline's stages in build order, with their node ids.
const STAGES: &[(&str, &str)] = &[
    ("early", EARLY),
    ("geometry", GEOMETRY),
    ("lighting", LIGHTING),
    ("late", ATMOSPHERE_OVERLAYS),
    ("post", POST_PROCESS),
    ("final", FINAL),
];

/// What the stages of one build share.
struct StageState {
    perf: Arc<Mutex<PerfOverlayShared>>,
    /// Lighting and everything drawn into it stay scene-linear FP16 until
    /// post-processing tone maps once.
    lighting_config: RendererConfig,
}

/// A stage descriptor. Pins are flow-only here; [`register_nodes`] replaces them
/// with probed ones.
fn stage(
    kind: &str,
    title: &str,
    description: &str,
    build: impl Fn(&mut NodeBuild<'_, '_>) -> Result<(), String> + Send + Sync + 'static,
) -> NodeDescriptor {
    NodeDescriptor::new(kind, title, "Helio Stages", description, build)
        .with_pins(vec![PinDesc::flow("in")], vec![PinDesc::flow("out")])
}

/// The stage state, or an error if the stage that makes it has not run: the
/// later stages need what the early stage builds.
fn state<'a>(build: &'a mut NodeBuild<'_, '_>, stage: &str) -> Result<&'a mut StageState, String> {
    build
        .env
        .shared_existing::<StageState>()
        .ok_or_else(|| format!("{stage} needs the Early stage to come before it"))
}

/// Registers the Helio nodes. Where `probe` has pins for a node kind they are
/// the node's pins; the SceneDB node lists every SceneDB buffer the probe saw.
pub fn register_nodes(registry: &mut NodeRegistry, probe: &Probe) {
    let with_probed = |descriptor: NodeDescriptor| -> NodeDescriptor {
        match probe.nodes.get(&descriptor.kind) {
            Some(pins) => {
                let (inputs, outputs) = (pins.inputs.clone(), pins.outputs.clone());
                descriptor.with_pins(inputs, outputs)
            }
            None => descriptor,
        }
    };

    registry.register(with_probed(
        NodeDescriptor::new(
            SCENE_INPUTS,
            "Helio Scene Inputs",
            "Helio Stages",
            "The resources Helio's passes expect the renderer to supply every frame.",
            |build| {
                declare_common_external_inputs(build.graph);
                Ok(())
            },
        )
        .with_pins(Vec::new(), vec![PinDesc::flow("out")]),
    ));

    // One node per SceneDB GPU buffer, whatever registered it: a node offers the
    // buffer as a Buffer pin, and a pass that consumes it has an input for it.
    for key in &probe.scene_buffers {
        registry.register(
            NodeDescriptor::new(
                &scenedb_kind(key),
                &format!("SceneDB::GPU::{key}"),
                "SceneDB",
                "A GPU buffer SceneDB keeps for the scene, supplied to passes by the renderer.",
                |_| Ok(()),
            )
            .with_pins(
                Vec::new(),
                vec![PinDesc::resource(PinType::BUFFER, format!("{SCENEDB_PREFIX}{key}"))],
            ),
        );
    }

    registry.register(with_probed(stage(
        EARLY,
        "Scene Prep + Shadows",
        "Scene preparation, shadow maps, Hi-Z, occlusion culling and light culling.",
        |build| {
            let (iw, ih) = build.env.internal_size();
            let lighting_config =
                RendererConfig { surface_format: FOGGED_HDR_FORMAT, ..build.env.config };
            let env = &*build.env;
            let perf = add_common_early_passes(
                build.graph,
                env.device,
                env.queue,
                env.camera_buffer,
                &lighting_config,
                env.cull_stats_buffer,
                iw,
                ih,
                env.scene_db.clone(),
            );
            build.graph.add_pass(Box::new(LightCullPass::new(env.device, iw, ih)));
            build.env.shared(|| StageState { perf, lighting_config });
            Ok(())
        },
    )));

    registry.register(with_probed(stage(
        GEOMETRY,
        "Geometry (GBuffer)",
        "Opaque geometry into the GBuffer, including foliage and virtual geometry.",
        |build| {
            let perf = state(build, "Geometry")?.perf.clone();
            let env = &*build.env;
            add_geometry_passes(
                build.graph,
                env.device,
                env.camera_buffer,
                &env.config,
                &perf,
                env.scene_db.clone(),
            );
            Ok(())
        },
    )));

    registry.register(with_probed(stage(
        LIGHTING,
        "Deferred Lighting",
        "Decals, reflections (when enabled in settings) and deferred lighting of the GBuffer.",
        |build| {
            let (perf, lighting_config) = {
                let state = state(build, "Lighting")?;
                (state.perf.clone(), state.lighting_config)
            };
            let (iw, ih) = build.env.internal_size();
            let env = &*build.env;
            let (device, queue, camera, config) =
                (env.device, env.queue, env.camera_buffer, env.config);
            let graph = &mut *build.graph;
            graph.add_pass(Box::new(DecalPass::new(device, queue, camera, iw, ih)));
            if config.enable_ssr && helio_core::REFLECTIONS_SUPPORTED {
                graph.add_pass(Box::new(SsrPass::new(device, queue, camera, iw, ih)));
            }
            if config.enable_planar_reflections && helio_core::REFLECTIONS_SUPPORTED {
                graph.add_pass(Box::new(PlanarReflectionPass::new(
                    device,
                    camera,
                    config.surface_format,
                )));
            }
            let mut deferred =
                DeferredLightPass::new(device, queue, camera, lighting_config.surface_format);
            deferred.set_shadow_quality(config.shadow_quality, queue);
            deferred.debug_mode = config.debug_mode;
            deferred.set_env_reflections(config.enable_environment_reflections);
            graph.add_pass(Box::new(deferred));
            graph.add_pass(Box::new(PerfOverlayCostAnalyzerPass::new(perf.clone())));
            graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(perf)));
            Ok(())
        },
    )));

    registry.register(with_probed(stage(
        ATMOSPHERE_OVERLAYS,
        "Atmosphere + Overlays",
        "Sky composite, billboards, coronas, water and the editor's debug drawing.",
        |build| {
            let (perf, lighting_config) = {
                let state = state(build, "Atmosphere + Overlays")?;
                (state.perf.clone(), state.lighting_config)
            };
            let (iw, ih) = build.env.internal_size();
            let env = &*build.env;
            add_late_passes(
                build.graph,
                env.device,
                env.queue,
                env.camera_buffer,
                &lighting_config,
                &perf,
                env.debug_state.clone(),
                env.debug_camera_buffer,
                iw,
                ih,
                env.scene_db.clone(),
            );
            Ok(())
        },
    )));

    registry.register(with_probed(stage(
        POST_PROCESS,
        "Fog, Anti-Aliasing + Post Process",
        "Volumetric fog, transparency, TSR or FXAA, lens, tone mapping and colour grading.",
        |build| {
            let (iw, ih) = build.env.internal_size();
            let env = &*build.env;
            let (device, queue, config) = (env.device, env.queue, env.config);
            let aa = match config.tsr_quality {
                Some(quality) => SceneAa::Tsr(TsrPass::new(
                    device,
                    iw,
                    ih,
                    config.width,
                    config.height,
                    config.surface_format,
                    quality,
                )),
                None => SceneAa::Fxaa,
            };
            let transparent =
                helio_pass_transparent::TransparentPass::new(device, crate::FOGGED_HDR_FORMAT);
            let resolved = add_scene_linear_chain(
                build.graph,
                device,
                FOGGED_HDR_FORMAT,
                Some(transparent),
                aa,
                iw,
                ih,
                None,
            );
            build.graph.add_pass(Box::new(
                PostProcessPass::new_with_user_effects(
                    device,
                    queue,
                    config.width,
                    config.height,
                    config.surface_format,
                    None,
                )
                .with_color_input(resolved),
            ));
            Ok(())
        },
    )));

    registry.register(with_probed(stage(
        FINAL,
        "Final Overlays",
        "Buffer views, sprites, the performance overlay and debug lines over the final image.",
        |build| {
            let perf = state(build, "Final Overlays")?.perf.clone();
            let env = &*build.env;
            add_final_passes(
                build.graph,
                env.device,
                env.queue,
                &env.config,
                &perf,
                env.debug_state.clone(),
                env.debug_camera_buffer,
                None,
            );
            Ok(())
        },
    )));
}

/// Core nodes and the stages with flow pins only: enough to build the default
/// graph, which is all a probe needs.
pub fn base_registry() -> NodeRegistry {
    let mut registry = NodeRegistry::with_core();
    register_nodes(&mut registry, &Probe::default());
    registry
}

/// Core nodes and the stages with the pins `probe` found.
pub fn node_registry(probe: &Probe) -> NodeRegistry {
    let mut registry = NodeRegistry::with_core();
    register_nodes(&mut registry, probe);
    registry
}

/// Builds the default graph on `env`'s device and returns the pins its passes
/// declare. Needs a device and a SceneDB mirror, so a host that has none handy
/// makes a headless pair (see `engine_backend`'s pipeline module).
pub fn probe_pins(env: &mut BuildEnv<'_>) -> Result<Probe, String> {
    // Build every pass the stages can add, so each one's declarations are seen.
    // (Temporal AA and FXAA are alternatives; TSR is probed.)
    env.config = RendererConfig {
        enable_portals: true,
        enable_ssr: true,
        enable_planar_reflections: true,
        tsr_quality: Some(helio_pass_tsr::TsrQuality::Performance),
        ..env.config
    };
    probe(&default_graph_asset(&Probe::default()), &base_registry(), env).map_err(|diagnostics| {
        diagnostics.iter().map(ToString::to_string).collect::<Vec<_>>().join("; ")
    })
}

fn node(asset: &mut GraphAsset, id: &str, kind: &str, x: f32, y: f32) {
    asset.nodes.push(NodeInstance {
        id: id.into(),
        kind: kind.into(),
        params: Default::default(),
        position: [x, y],
    });
}

fn connect(asset: &mut GraphAsset, from: (&str, &str), to: (&str, &str)) {
    asset.connections.push(Connection {
        from: PinRef::new(from.0, from.1),
        to: PinRef::new(to.0, to.1),
    });
}

/// The default deferred pipeline as a graph: the host inputs and SceneDB nodes,
/// each stage in order, then the output.
///
/// Wires follow the pins: a stage input is fed by the last earlier stage that
/// outputs the same resource, and a SceneDB input by the SceneDB node. Host
/// (Scene) inputs are not wired to the stages: the renderer supplies them each
/// frame, and the Scene Inputs node, which is flow-wired into the first stage,
/// declares them. With no probed pins the stages
/// only have flow pins and the graph is the flow chain.
pub fn default_graph_asset(probe: &Probe) -> GraphAsset {
    const COLUMN: f32 = 420.0;
    let mut asset = GraphAsset { name: "Deferred".into(), ..Default::default() };
    node(&mut asset, "inputs", SCENE_INPUTS, 0.0, -420.0);
    for (i, &(id, kind)) in STAGES.iter().enumerate() {
        node(&mut asset, id, kind, COLUMN * (i as f32 + 1.0), 0.0);
    }
    node(&mut asset, "output", core_nodes::OUTPUT, COLUMN * (STAGES.len() as f32 + 1.0), 0.0);

    // Flow: the host inputs first (they declare what the renderer supplies, so
    // the graph must build them), then stage to stage, then the output.
    connect(&mut asset, ("inputs", "out"), (STAGES[0].0, "in"));
    for pair in STAGES.windows(2) {
        connect(&mut asset, (pair[0].0, "out"), (pair[1].0, "in"));
    }
    connect(&mut asset, (STAGES[STAGES.len() - 1].0, "out"), ("output", "in"));

    // SceneDB buffers: one node per buffer a stage consumes, in a column left of
    // the host inputs.
    let mut scenedb_nodes: Vec<String> = Vec::new();
    // Data: the last producer of each resource so far.
    let mut latest: HashMap<String, (&str, String)> = HashMap::new();
    for &(id, kind) in STAGES {
        let Some(pins): Option<&NodePins> = probe.nodes.get(kind) else { continue };
        for input in pins.inputs.iter().filter(|pin| pin.ty.kind != PinKind::Flow) {
            if let Some(key) = input.resource.strip_prefix(SCENEDB_PREFIX) {
                let node_id = format!("scenedb_{key}");
                if !scenedb_nodes.contains(&node_id) {
                    let row = scenedb_nodes.len() as f32;
                    node(&mut asset, &node_id, &scenedb_kind(key), -COLUMN, row * 130.0);
                    scenedb_nodes.push(node_id.clone());
                }
                connect(&mut asset, (&node_id, &input.resource), (id, &input.name));
            } else if let Some((producer, pin)) = latest.get(&input.resource) {
                connect(&mut asset, (producer, pin), (id, &input.name));
            }
        }
        for output in pins.outputs.iter().filter(|pin| pin.ty.kind != PinKind::Flow) {
            latest.insert(output.resource.clone(), (id, output.name.clone()));
        }
    }
    asset
}

#[cfg(test)]
mod tests {
    use super::*;
    use helio_graph_asset::{output_pin, validate, Severity};

    /// A probe result small enough to write out: the shape the derivation
    /// produces, without a device.
    fn sample_probe() -> Probe {
        let mut probe = Probe::default();
        probe.nodes.insert(
            EARLY.into(),
            NodePins {
                inputs: vec![PinDesc::flow("in")],
                outputs: vec![
                    PinDesc::flow("out"),
                    output_pin(PinType::BUFFER, "object_batch"),
                    output_pin(PinType::TEXTURE, "pre_aa"),
                ],
            },
        );
        probe.nodes.insert(
            GEOMETRY.into(),
            NodePins {
                inputs: vec![
                    PinDesc::flow("in"),
                    PinDesc::resource(PinType::BUFFER, "object_batch"),
                    PinDesc::resource(PinType::BUFFER, format!("{SCENEDB_PREFIX}materials")),
                ],
                outputs: vec![PinDesc::flow("out")],
            },
        );
        probe.scene_buffers = vec!["materials".into()];
        probe
    }

    #[test]
    fn flow_only_default_graph_validates_and_orders_the_stages() {
        let registry = base_registry();
        let asset = default_graph_asset(&Probe::default());
        let validation = validate(&asset, &registry);
        assert!(!validation.has_errors(), "{:?}", validation.diagnostics);
        let plan = validation.plan.expect("valid");
        let ids: Vec<_> = plan.order.iter().map(|i| asset.nodes[*i].id.as_str()).collect();
        let position = |id: &str| ids.iter().position(|k| *k == id).unwrap();
        assert!(position("early") < position("geometry"));
        assert!(position("post") < position("final"));
        assert_eq!(ids.last(), Some(&"output"));
    }

    #[test]
    fn probed_pins_become_wires_between_the_stages_that_share_a_resource() {
        let probe = sample_probe();
        let registry = node_registry(&probe);
        let asset = default_graph_asset(&probe);
        let has = |from: (&str, &str), to: (&str, &str)| {
            asset.connections.contains(&Connection {
                from: PinRef::new(from.0, from.1),
                to: PinRef::new(to.0, to.1),
            })
        };
        assert!(has(("early", "out_object_batch"), ("geometry", "object_batch")));
        assert!(has(("scenedb_materials", "scenedb.materials"), ("geometry", "scenedb.materials")));
        let validation = validate(&asset, &registry);
        assert!(validation.diagnostics.iter().all(|d| d.severity != Severity::Error));
    }
}
