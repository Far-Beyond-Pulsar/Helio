//! Helio's default pipelines as node graphs.
//!
//! The shipped pipelines are graphs of the pass nodes in [`crate::features`]
//! ([`crate::presets`] makes them for a renderer config). This module has the
//! registry that holds those nodes and the editor's laid-out template of the
//! default deferred pipeline.
//!
//! - [`base_registry`]: the core nodes, every pass node, the host-input nodes
//!   and the performance marker. Pins are flow-only until a probe has run.
//! - [`probe_pins`]: builds every registered pass once on a device and returns
//!   what they declare; [`apply_probe`] gives the nodes those pins and adds a
//!   node per SceneDB buffer.
//! - [`deferred_graph_asset`]: the default deferred pipeline, laid out. **On Frame
//!   Start** begins an execution chain through every pass in build order to the
//!   Output. With probed pins the data is wired too: every input to the node that
//!   supplies it, and each pass gets its own copy of the host and SceneDB source
//!   nodes it reads, beside it, so no wire runs across the graph.

use std::collections::HashMap;

use helio_graph_asset::{
    auto_layout, core_nodes, probe, BuildEnv, Connection, GraphAsset, NodeDescriptor, NodeInstance,
    NodeRegistry, PinDesc, PinKind, PinRef, PinType, Probe, SCENEDB_PREFIX,
};

use helio_frame_api::HOST_INPUTS;

use crate::features;

/// The config the editor's template and the pin probe use: the renderer's
/// defaults (FXAA on, no screen-space or planar reflections).
pub fn template_config() -> helio::RendererConfig {
    helio::RendererConfig::new(1920, 1080, wgpu::TextureFormat::Rgba8UnormSrgb)
}

/// The default pipeline's nodes in execution order, as (id, kind).
fn deferred_nodes(registry: &NodeRegistry) -> Vec<(String, String)> {
    crate::presets::Preset::Deferred
        .nodes(registry, &template_config())
        .into_iter()
        .map(|node| (node.id, node.kind))
        .collect()
}

/// Core nodes plus everything `features` register.
pub fn registry_for(features: &helio_feature_api::FeatureSet) -> NodeRegistry {
    let mut registry = helio_graph_asset::core_registry();
    features.register_into(&mut registry);
    registry
}

/// Core nodes, the default features and an engine's `extra` features.
pub fn registry_with(extra: &helio_feature_api::FeatureSet) -> NodeRegistry {
    registry_for(&features::default_features().extended(extra))
}

/// Core nodes and the default features ([`features::default_features`]).
pub fn base_registry() -> NodeRegistry {
    registry_for(&features::default_features())
}

fn pass_kinds(registry: &NodeRegistry) -> Vec<String> {
    registry
        .list()
        .into_iter()
        .map(|descriptor| descriptor.kind.clone())
        .filter(|kind| kind.starts_with("helio.pass."))
        .collect()
}

fn add_node(asset: &mut GraphAsset, id: &str, kind: &str) {
    asset.nodes.push(NodeInstance {
        id: id.into(),
        kind: kind.into(),
        params: Default::default(),
        position: [0.0, 0.0],
    });
}

fn add_buffer_node(asset: &mut GraphAsset, id: &str, kind: &str, buffer: &str) {
    asset.nodes.push(NodeInstance {
        id: id.into(),
        kind: kind.into(),
        params: [("buffer".to_string(), serde_json::Value::from(buffer))].into(),
        position: [0.0, 0.0],
    });
}

/// A node for every host input the registry's features declared.
fn add_host_inputs(asset: &mut GraphAsset, registry: &NodeRegistry) {
    for descriptor in registry.placed(HOST_INPUTS) {
        add_node(asset, &descriptor.kind.replace('.', "_"), &descriptor.kind);
    }
}

fn connect(asset: &mut GraphAsset, from: (&str, &str), to: (&str, &str)) {
    asset.connections.push(Connection {
        from: PinRef::new(from.0, from.1),
        to: PinRef::new(to.0, to.1),
    });
}

/// The graph a probe builds: On Frame Start, the host inputs, the default
/// pipeline's passes in order, then every other registered pass, then the output.
/// It has no wires; nodes build in this order. Every pass is in it, so every pass
/// gets pins.
fn probe_asset(registry: &NodeRegistry) -> GraphAsset {
    let mut asset = GraphAsset {
        name: "Pin probe".into(),
        ..Default::default()
    };
    add_node(&mut asset, "start", core_nodes::ON_FRAME_START);
    add_host_inputs(&mut asset, registry);
    let mut placed: Vec<String> = Vec::new();
    for (id, kind) in deferred_nodes(registry) {
        add_node(&mut asset, &id, &kind);
        placed.push(kind);
    }
    let mut others = pass_kinds(registry);
    others.retain(|k| !placed.contains(k));
    others.sort();
    for other in others {
        let id = other.trim_start_matches("helio.pass.").to_string();
        add_node(&mut asset, &id, &other);
    }
    add_node(&mut asset, "output", core_nodes::OUTPUT);
    asset
}

/// Builds every registered pass once on `env`'s device and returns the pins
/// their declarations give the nodes, and the SceneDB buffers in play.
///
/// Needs a device and a SceneDB mirror. A host with a window uses its own device;
/// others make a headless pair (see `engine_backend`'s pipeline module).
pub fn probe_pins(registry: &NodeRegistry, env: &mut BuildEnv<'_>) -> Result<Probe, String> {
    probe(&probe_asset(registry), registry, env).map_err(|diagnostics| {
        diagnostics
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("; ")
    })
}

/// Gives the generated nodes the pins `probe` found, and registers one node per
/// SceneDB GPU buffer: `SceneDB::GPU::<key>`, offering the buffer as a Buffer pin.
pub fn apply_probe(registry: &mut NodeRegistry, probe: &Probe) {
    probe.apply(registry);
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
                vec![PinDesc::resource(
                    PinType::BUFFER,
                    format!("{SCENEDB_PREFIX}{key}"),
                )],
            )
            .with_data_only(),
        );
    }
}

/// Node kind of the node for SceneDB GPU buffer `key`.
pub fn scenedb_kind(key: &str) -> String {
    format!("scenedb.{key}")
}

/// The default deferred pipeline as a laid-out graph.
///
/// Execution: On Frame Start, every pass in order, then the Output, chained by
/// flow wires. With probed pins every input is also wired to what supplies it:
/// a pass to the last earlier pass that writes the resource, a *history* input to
/// the last pass that writes it in the frame, host inputs and SceneDB buffers to
/// source nodes made for that pass and placed beside it. Without pins there is only
/// the chain (and the host nodes, which declare what the renderer supplies), which
/// builds the same pipeline.
pub fn deferred_graph_asset(registry: &NodeRegistry) -> GraphAsset {
    let wired = registry.has_generated_pins();
    let mut asset = GraphAsset {
        name: "Deferred".into(),
        description: "Helio's default deferred pipeline, one node per pass.".into(),
        ..Default::default()
    };

    let nodes = deferred_nodes(registry);
    add_node(&mut asset, "start", core_nodes::ON_FRAME_START);
    for (id, kind) in &nodes {
        add_node(&mut asset, id, kind);
    }
    add_node(&mut asset, "output", core_nodes::OUTPUT);

    // The execution chain.
    connect(&mut asset, ("start", "out"), (&nodes[0].0, "in"));
    for pair in nodes.windows(2) {
        connect(&mut asset, (&pair[0].0, "out"), (&pair[1].0, "in"));
    }
    connect(
        &mut asset,
        (&nodes[nodes.len() - 1].0, "out"),
        ("output", "in"),
    );

    if !wired {
        add_host_inputs(&mut asset, registry);
        auto_layout(&mut asset, registry);
        helio_graph_asset::sections::sectionize(&mut asset, registry);
        return asset;
    }

    // Pins of each pass.
    let pins: HashMap<&str, helio_graph_asset::NodePins> = nodes
        .iter()
        .filter_map(|(id, kind)| {
            registry
                .get(kind)
                .map(|descriptor| (id.as_str(), descriptor.pins(&Default::default())))
        })
        .collect();

    // Every resource some pass reads: only those need a Set where they are written.
    let read_somewhere: std::collections::HashSet<String> = nodes
        .iter()
        .filter_map(|(id, _)| pins.get(id.as_str()))
        .flat_map(|node_pins| node_pins.inputs.iter())
        .filter(|pin| pin.ty.kind != PinKind::Flow)
        .map(|pin| pin.resource.clone())
        .collect();

    // The buffers: each pass reads its inputs through Get Buffer nodes on its left
    // and writes what others read through Set Buffer nodes on its right. A buffer
    // is a variable: the Gets read the latest Set before the pass using them, so no
    // wire runs between passes.
    for (name, _) in &nodes {
        let name = name.as_str();
        let Some(node_pins) = pins.get(name) else {
            continue;
        };
        for input in node_pins
            .inputs
            .iter()
            .filter(|p| p.ty.kind != PinKind::Flow)
        {
            let Some(entry) = helio_graph_asset::buffers::entry(input.ty, &input.resource) else {
                continue;
            };
            let id = format!("get:{}@{name}", input.resource);
            add_buffer_node(&mut asset, &id, core_nodes::BUFFER_GET, &entry);
            connect(&mut asset, (&id, &input.resource), (name, &input.name));
        }
        for output in node_pins
            .outputs
            .iter()
            .filter(|p| p.ty.kind != PinKind::Flow)
        {
            if !read_somewhere.contains(&output.resource) {
                continue;
            }
            let Some(entry) = helio_graph_asset::buffers::entry(output.ty, &output.resource) else {
                continue;
            };
            let id = format!("set:{}@{name}", output.resource);
            add_buffer_node(&mut asset, &id, core_nodes::BUFFER_SET, &entry);
            connect(&mut asset, (name, &output.name), (&id, &output.resource));
        }
    }

    // The host declares what the renderer supplies; the Gets of those are left
    // unresolved on purpose (the renderer supplies them each frame).
    add_host_inputs(&mut asset, registry);

    // Lay out once so the lanes (runs of one category) are known, then make each lane a
    // section: a collapsed node holding its passes and the buffers around them.
    auto_layout(&mut asset, registry);
    helio_graph_asset::sections::sectionize(&mut asset, registry);
    asset
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::kind;
    use helio_frame_api::host_kind;
    use helio_graph_asset::{validate, NodePins, Severity};

    #[test]
    fn every_default_pass_is_registered() {
        let registry = base_registry();
        let config = template_config();
        for preset in crate::presets::Preset::ALL {
            for node in preset.nodes(&registry, &config) {
                assert!(
                    registry.get(&node.kind).is_some(),
                    "{}: no node for `{}`",
                    preset.name(),
                    node.kind
                );
            }
        }
        let hosts: Vec<String> = registry
            .placed(HOST_INPUTS)
            .iter()
            .map(|d| d.kind.clone())
            .collect();
        for host in ["material_textures", "render_environment"] {
            assert!(hosts.contains(&host_kind(host)), "{host}: {hosts:?}");
        }
        assert!(registry.get(core_nodes::ON_FRAME_START).is_some());
        assert!(registry.marker().is_some());
    }

    #[test]
    fn the_unwired_default_is_one_execution_chain_and_validates() {
        let registry = base_registry();
        assert!(!registry.has_generated_pins());
        let asset = deferred_graph_asset(&registry);
        let validation = validate(&asset, &registry);
        assert!(!validation.has_errors(), "{:?}", validation.diagnostics);
        assert!(
            !validation
                .diagnostics
                .iter()
                .any(|d| d.message.contains("execution chain")),
            "every pass is on the chain: {:?}",
            validation.diagnostics
        );
        let flat = validation.flat.expect("expanded");
        let plan = validation.plan.expect("valid");
        let ids: Vec<_> = plan
            .order
            .iter()
            .map(|i| flat.nodes[*i].id.as_str())
            .collect();
        let at = |id: &str| ids.iter().position(|k| *k == id).unwrap();
        assert!(at("start") < at("background") && at("background") < at("hiz"));
        // Without the lights, the G-buffer resolves unlit.
        assert!(at("hiz") < at("unlit"));
        assert!(at("unlit") < at("post_process"));
        // The execution chain — the nodes with flow pins — ends at the Output; the
        // host declarations are data-only nodes and build after the chain.
        let chain: Vec<&str> = plan
            .order
            .iter()
            .filter(|i| {
                let node = &flat.nodes[**i];
                registry.get(&node.kind).is_some_and(|d| {
                    let pins = d.pins(&node.params);
                    pins.inputs
                        .iter()
                        .chain(&pins.outputs)
                        .any(|pin| pin.ty.kind == PinKind::Flow)
                })
            })
            .map(|i| flat.nodes[*i].id.as_str())
            .collect();
        assert_eq!(chain.last(), Some(&"output"), "{chain:?}");
        // Laid out: each lane is a section, and the main graph is the chain through them.
        assert!(asset.sections.len() >= 4, "{:?}", asset.sections.len());
        assert!(asset.nodes.iter().any(|n| asset.section_of(n).is_some()));
        assert!(
            asset.nodes.len() < 12,
            "the passes are inside sections: {}",
            asset.nodes.len()
        );
    }

    fn pins(inputs: Vec<PinDesc>, outputs: Vec<PinDesc>) -> NodePins {
        let mut all_inputs = vec![PinDesc::flow("in")];
        all_inputs.extend(inputs);
        let mut all_outputs = vec![PinDesc::flow("out")];
        all_outputs.extend(outputs);
        NodePins {
            inputs: all_inputs,
            outputs: all_outputs,
        }
    }

    #[test]
    fn probed_pins_become_buffer_nodes_beside_each_pass() {
        use helio_graph_asset::output_pin;
        let mut registry = base_registry();
        let mut probe = Probe::default();
        probe.nodes.insert(
            kind("hiz"),
            pins(
                // Hi-Z reads last frame's depth, which the lighting reads later.
                vec![PinDesc::resource(PinType::DEPTH, "depth").history()],
                vec![output_pin(PinType::TEXTURE, "hiz")],
            ),
        );
        probe.nodes.insert(
            kind("unlit"),
            pins(
                vec![
                    PinDesc::resource(PinType::BUFFER, format!("{SCENEDB_PREFIX}materials")),
                    PinDesc::resource(PinType::SCENE, "material_textures"),
                    PinDesc::resource(PinType::TEXTURE, "hiz"),
                ],
                vec![output_pin(PinType::DEPTH, "depth")],
            ),
        );
        probe.scene_buffers = vec!["materials".into()];
        apply_probe(&mut registry, &probe);

        let asset = deferred_graph_asset(&registry);
        // The wiring inside the sections, which is where the buffers are.
        let all: Vec<&Connection> = asset
            .connections
            .iter()
            .chain(asset.sections.iter().flat_map(|s| s.connections.iter()))
            .collect();
        let has = |from: (&str, &str), to: (&str, &str)| {
            all.contains(&&Connection {
                from: PinRef::new(from.0, from.1),
                to: PinRef::new(to.0, to.1),
            })
        };
        // A pass reads through its own Get and writes through its own Set; nothing is
        // wired from one pass to another.
        assert!(has(
            ("get:hiz@unlit", "hiz"),
            ("unlit", "hiz")
        ));
        assert!(has(("hiz", "out_hiz"), ("set:hiz@hiz", "hiz")));
        assert!(has(("get:depth@hiz", "depth"), ("hiz", "depth")));
        assert!(has(
            ("unlit", "out_depth"),
            ("set:depth@unlit", "depth")
        ));
        assert!(has(
            ("get:scenedb.materials@unlit", "scenedb.materials"),
            ("unlit", "scenedb.materials")
        ));
        // Nothing is wired from one pass to another but the chain's flow wire.
        assert!(all
            .iter()
            .filter(|c| c.from.node == "hiz" && c.to.node == "unlit")
            .all(|c| c.from.pin == "out" && c.to.pin == "in"));

        // The buffers resolve: the lighting reads what Hi-Z set, and Hi-Z reads
        // the depth the lighting sets later in the frame (history).
        let validation = validate(&asset, &registry);
        assert!(
            validation
                .diagnostics
                .iter()
                .all(|d| d.severity != Severity::Error),
            "{:?}",
            validation.diagnostics
        );
        let flat = validation.flat.expect("flattened");
        let flat_has = |from: (&str, &str), to: (&str, &str)| {
            flat.connections.contains(&Connection {
                from: PinRef::new(from.0, from.1),
                to: PinRef::new(to.0, to.1),
            })
        };
        assert!(flat_has(("hiz", "out_hiz"), ("unlit", "hiz")));
        assert!(flat_has(("unlit", "out_depth"), ("hiz", "depth")));
        // The execution chain runs from On Frame Start through the sections.
        assert!(flat_has(("start", "out"), ("background", "in")));

        // A Get sits left of the pass that reads it, a Set right of the pass that writes.
        let x = |id: &str| {
            asset
                .sections
                .iter()
                .flat_map(|s| s.nodes.iter())
                .find(|n| n.id == id)
                .unwrap()
                .position[0]
        };
        assert!(x("get:scenedb.materials@unlit") < x("unlit"));
        assert!(x("set:depth@unlit") > x("unlit"));
    }
}
