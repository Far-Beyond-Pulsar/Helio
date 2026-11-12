//! Helio's default pipelines as node graphs.
//!
//! The shipped pipelines are graphs of the pass nodes in [`crate::features`]
//! ([`crate::presets`]). This module has the
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
    let preset = crate::presets::Preset::Deferred;
    for node in preset.nodes(registry) {
        placed.push(node.kind.clone());
        asset.nodes.push(node);
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
    graph_asset(crate::presets::Preset::Deferred, registry)
}

/// `preset` as a laid-out graph (see [`deferred_graph_asset`]). The same for
/// every renderer config: the anti-aliasing node follows the project's setting
/// when the graph builds.
///
/// Every node comes from the preset's own node list, so the nodes that
/// features placed at the frame's slots and draw phases (any plugin's, by
/// `NodeDescriptor::at`/`in_slot`/`drawer`) are in the asset in their place, and
/// [`auto_layout`] and `sectionize` put them in their lane by their category.
pub fn graph_asset(preset: crate::presets::Preset, registry: &NodeRegistry) -> GraphAsset {
    let wired = registry.has_generated_pins();
    let mut asset = GraphAsset {
        name: preset.name().into(),
        description: format!(
            "{}: the frame's slots and draw phases with every installed node, one node per pass.",
            preset.name()
        ),
        ..Default::default()
    };

    // The preset's own instances, parameters included (TSR's quality, the frame
    // setup's lighting model).
    let instances = preset.nodes(registry);
    let nodes: Vec<(String, String)> = instances
        .iter()
        .map(|node| (node.id.clone(), node.kind.clone()))
        .collect();
    add_node(&mut asset, "start", core_nodes::ON_FRAME_START);
    asset.nodes.extend(instances);
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
        for preset in crate::presets::Preset::ALL {
            for node in preset.nodes(&registry) {
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
        for host in ["render_environment"] {
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

    type Built = Vec<(String, String, helio_graph_asset::Params)>;

    /// The executing nodes of an asset in the order the compiler builds them,
    /// sections expanded.
    fn built_order(asset: &GraphAsset, registry: &NodeRegistry) -> Built {
        let validation = validate(asset, registry);
        assert!(!validation.has_errors(), "{:?}", validation.diagnostics);
        let flat = validation.flat.expect("expanded");
        let plan = validation.plan.expect("valid");
        plan.order
            .iter()
            .map(|i| &flat.nodes[*i])
            .filter(|node| {
                registry.get(&node.kind).is_some_and(|d| {
                    let pins = d.pins(&node.params);
                    pins.inputs
                        .iter()
                        .chain(&pins.outputs)
                        .any(|pin| pin.ty.kind == PinKind::Flow)
                })
            })
            .map(|node| (node.id.clone(), node.kind.clone(), node.params.clone()))
            .collect()
    }

    /// The preset's chain: On Frame Start, the preset's nodes, the Output.
    fn preset_order(preset: crate::presets::Preset, registry: &NodeRegistry) -> Built {
        let mut expected: Built = vec![(
            "start".into(),
            core_nodes::ON_FRAME_START.into(),
            Default::default(),
        )];
        expected.extend(
            preset
                .nodes(registry)
                .into_iter()
                .map(|node| (node.id, node.kind, node.params)),
        );
        expected.push((
            "output".into(),
            core_nodes::OUTPUT.into(),
            Default::default(),
        ));
        expected
    }

    /// The graph is the preset: for each shipped preset the laid-out,
    /// sectioned asset builds its nodes in the preset's order with the
    /// preset's parameters. (The `add_pass` builders this replaces are
    /// gone; the preset's node list is the reference the graph is held to.)
    #[test]
    fn the_graph_builds_the_presets_nodes_in_the_presets_order() {
        use crate::presets::Preset;
        let registry = base_registry();
        for preset in Preset::ALL {
            let asset = graph_asset(preset, &registry);
            assert_eq!(
                built_order(&asset, &registry),
                preset_order(preset, &registry),
                "{}",
                preset.name()
            );
        }
    }

    /// The deferred graph follows the project's anti-aliasing through one
    /// node, so the same asset is TSR under a TSR config and FXAA without; a
    /// preset that never resolves temporally names FXAA itself.
    #[test]
    fn the_deferred_graph_names_the_antialiasing_node_and_no_tsr_or_fxaa() {
        use crate::presets::Preset;
        let registry = base_registry();
        let has = |order: &Built, name: &str| order.iter().any(|(_, k, _)| *k == kind(name));
        let deferred = built_order(&deferred_graph_asset(&registry), &registry);
        assert!(has(&deferred, "antialiasing"));
        assert!(!has(&deferred, "tsr") && !has(&deferred, "fxaa"));
        let at = |name: &str| {
            deferred
                .iter()
                .position(|(_, k, _)| *k == kind(name))
                .unwrap()
        };
        assert_eq!(deferred[at("antialiasing")].2["fallback"], "fxaa");
        assert!(at("antialiasing") < at("post_process"));
        let fxaa = built_order(&graph_asset(Preset::DeferredFxaa, &registry), &registry);
        assert!(has(&fxaa, "fxaa") && !has(&fxaa, "antialiasing"));
    }

    /// The frame setup leads the chain and carries the lighting model the
    /// preset chose, by id, whether or not a plugin registered it.
    #[test]
    fn the_frame_setup_leads_the_chain() {
        let registry = base_registry();
        let order = built_order(&deferred_graph_asset(&registry), &registry);
        assert_eq!(order[1].1, features::FRAME_SETUP, "{order:?}");
        assert_eq!(order[1].2["model"], "deferred");
        // The unlit stand-in leaves no transparency to TSR.
        assert_eq!(order[1].2["reactive_mask"], false);
    }

    /// Nodes a plugin registered at the frame's slots and phases are in the
    /// default graph, in their place, with no preset naming them: the seam is
    /// `NodeDescriptor::in_slot` / `drawer` (`helio_frame_api::FrameNode`).
    #[test]
    fn a_node_placed_in_a_slot_joins_the_default_graph_and_a_section() {
        use helio_feature_api::{pass_node, Feature, FeatureRegistrar, FeatureSet};
        use helio_frame_api::{FrameNode, Slot};

        struct Plugin;
        impl Feature for Plugin {
            fn id(&self) -> &'static str {
                "test.slot_node"
            }
            fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
                registrar.node(
                    pass_node(
                        "slot_test",
                        "Test: Slot node",
                        "Passes: Geometry",
                        "",
                        |_| Ok(()),
                    )
                    .in_slot(Slot::Geometry, 0),
                );
            }
        }

        let registry = registry_with(&FeatureSet::new().with(Plugin));
        let asset = deferred_graph_asset(&registry);
        let order = built_order(&asset, &registry);
        let at = |name: &str| order.iter().position(|(_, k, _)| *k == kind(name));
        let placed = at("slot_test").expect("the node joined the default graph");
        assert!(at("background").unwrap() < placed && placed < at("unlit").unwrap());
        // It sits in a section of the sectioned graph, not loose on the main graph.
        assert!(asset.nodes.iter().all(|n| n.kind != kind("slot_test")));
        assert!(asset
            .sections
            .iter()
            .any(|s| s.nodes.iter().any(|n| n.kind == kind("slot_test"))));
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
                    PinDesc::resource(PinType::BUFFER, format!("{SCENEDB_PREFIX}rows")),
                    PinDesc::resource(PinType::SCENE, "render_environment"),
                    PinDesc::resource(PinType::TEXTURE, "hiz"),
                ],
                vec![output_pin(PinType::DEPTH, "depth")],
            ),
        );
        probe.scene_buffers = vec!["rows".into()];
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
        assert!(has(("get:hiz@unlit", "hiz"), ("unlit", "hiz")));
        assert!(has(("hiz", "out_hiz"), ("set:hiz@hiz", "hiz")));
        assert!(has(("get:depth@hiz", "depth"), ("hiz", "depth")));
        assert!(has(("unlit", "out_depth"), ("set:depth@unlit", "depth")));
        assert!(has(
            ("get:scenedb.rows@unlit", "scenedb.rows"),
            ("unlit", "scenedb.rows")
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
        assert!(flat_has(("start", "out"), ("frame", "in")));
        assert!(flat_has(("frame", "out"), ("background", "in")));

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
        assert!(x("get:scenedb.rows@unlit") < x("unlit"));
        assert!(x("set:depth@unlit") > x("unlit"));
    }
}
