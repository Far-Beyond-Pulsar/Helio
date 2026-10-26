//! Helio's default pipelines as node graphs.
//!
//! The pipelines that used to be `add_pass` code are graphs of the pass nodes in
//! [`crate::graph_nodes`]. This module has the registry that holds those nodes
//! and the graph of the default deferred pipeline.
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

use crate::graph_nodes::{self, host_kind, kind, HOST_INPUTS};

/// The default deferred pipeline's passes, in the order they execute. This is the
/// order the `add_pass` builders used; it is the execution chain of the graph.
const DEFERRED: &[&str] = &[
    "object_batch",
    "atmosphere",
    "shadow_matrix",
    "shadow_dirty",
    "shadow",
    "background",
    "indirect_dispatch",
    "hiz",
    "occlusion_cull",
    "portal_cull",
    "light_cull",
    "foliage_place",
    "gbuffer",
    "portal_mask",
    "portal_instance",
    "foliage_gbuffer",
    "decal",
    "deferred_light",
    "atmosphere_composite",
    "billboard",
    "corona",
    "portal_editor_overlay",
    "water_sim",
    "debug_draw_scene",
    "post_process_volume_blend",
    "volumetric_fog",
    "fog_composite",
    "transparent",
    "fxaa",
    "lens",
    "lens_flare",
    "post_process",
    "buffer_view",
    "sprite_overlay",
    "perf_overlay",
    "debug_draw",
];

/// Core nodes, every pass node, the host inputs and the performance marker.
pub fn base_registry() -> NodeRegistry {
    let mut registry = NodeRegistry::with_core();
    graph_nodes::register_passes(&mut registry);
    registry
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
    for name in HOST_INPUTS {
        add_node(&mut asset, &format!("host_{name}"), &host_kind(name));
    }
    let mut placed: Vec<String> = Vec::new();
    for &name in DEFERRED {
        add_node(&mut asset, name, &kind(name));
        placed.push(kind(name));
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

    add_node(&mut asset, "start", core_nodes::ON_FRAME_START);
    for &name in DEFERRED {
        add_node(&mut asset, name, &kind(name));
    }
    add_node(&mut asset, "output", core_nodes::OUTPUT);

    // The execution chain.
    connect(&mut asset, ("start", "out"), (DEFERRED[0], "in"));
    for pair in DEFERRED.windows(2) {
        connect(&mut asset, (pair[0], "out"), (pair[1], "in"));
    }
    connect(
        &mut asset,
        (DEFERRED[DEFERRED.len() - 1], "out"),
        ("output", "in"),
    );

    if !wired {
        for name in HOST_INPUTS {
            add_node(&mut asset, &format!("host_{name}"), &host_kind(name));
        }
        auto_layout(&mut asset, registry);
        helio_graph_asset::sections::sectionize(&mut asset, registry);
        return asset;
    }

    // Pins of each pass.
    let pins: HashMap<&str, helio_graph_asset::NodePins> = DEFERRED
        .iter()
        .filter_map(|name| {
            registry
                .get(&kind(name))
                .map(|descriptor| (*name, descriptor.pins(&Default::default())))
        })
        .collect();

    // Every resource some pass reads: only those need a Set where they are written.
    let read_somewhere: std::collections::HashSet<String> = DEFERRED
        .iter()
        .filter_map(|name| pins.get(name))
        .flat_map(|node_pins| node_pins.inputs.iter())
        .filter(|pin| pin.ty.kind != PinKind::Flow)
        .map(|pin| pin.resource.clone())
        .collect();

    // The buffers: each pass reads its inputs through Get Buffer nodes on its left
    // and writes what others read through Set Buffer nodes on its right. A buffer
    // is a variable: the Gets read the latest Set before the pass using them, so no
    // wire runs between passes.
    for &name in DEFERRED {
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
    for name in HOST_INPUTS {
        add_node(&mut asset, &format!("host_{name}"), &host_kind(name));
    }

    // Lay out once so the lanes (runs of one category) are known, then make each lane a
    // section: a collapsed node holding its passes and the buffers around them.
    auto_layout(&mut asset, registry);
    helio_graph_asset::sections::sectionize(&mut asset, registry);
    asset
}

#[cfg(test)]
mod tests {
    use super::*;
    use helio_graph_asset::{validate, NodePins, Severity};

    #[test]
    fn every_default_pass_is_registered() {
        let registry = base_registry();
        for name in DEFERRED {
            assert!(
                registry.get(&kind(name)).is_some(),
                "no node for pass `{name}`"
            );
        }
        for host in HOST_INPUTS {
            assert!(registry.get(&host_kind(host)).is_some());
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
        assert!(at("start") < at("object_batch"));
        assert!(at("shadow_matrix") < at("shadow_dirty") && at("shadow_dirty") < at("shadow"));
        assert!(at("hiz") < at("occlusion_cull"));
        assert!(at("portal_cull") < at("portal_instance"));
        assert!(at("gbuffer") < at("deferred_light"));
        assert!(at("deferred_light") < at("post_process"));
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
                // Hi-Z reads last frame's depth, which the GBuffer writes later.
                vec![PinDesc::resource(PinType::DEPTH, "depth").history()],
                vec![output_pin(PinType::TEXTURE, "hiz")],
            ),
        );
        probe.nodes.insert(
            kind("gbuffer"),
            pins(
                vec![
                    PinDesc::resource(PinType::BUFFER, format!("{SCENEDB_PREFIX}materials")),
                    PinDesc::resource(PinType::SCENE, "material_textures"),
                ],
                vec![output_pin(PinType::DEPTH, "depth")],
            ),
        );
        probe.nodes.insert(
            kind("occlusion_cull"),
            pins(vec![PinDesc::resource(PinType::TEXTURE, "hiz")], vec![]),
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
            ("get:hiz@occlusion_cull", "hiz"),
            ("occlusion_cull", "hiz")
        ));
        assert!(has(("hiz", "out_hiz"), ("set:hiz@hiz", "hiz")));
        assert!(has(("get:depth@hiz", "depth"), ("hiz", "depth")));
        assert!(has(
            ("gbuffer", "out_depth"),
            ("set:depth@gbuffer", "depth")
        ));
        assert!(has(
            ("get:scenedb.materials@gbuffer", "scenedb.materials"),
            ("gbuffer", "scenedb.materials")
        ));
        // Nothing is wired from one pass to another but the chain's flow wire.
        assert!(all
            .iter()
            .filter(|c| c.from.node == "hiz" && c.to.node == "occlusion_cull")
            .all(|c| c.from.pin == "out" && c.to.pin == "in"));

        // The buffers resolve: occlusion cull reads what Hi-Z set, and Hi-Z reads the
        // depth the GBuffer sets later in the frame (history).
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
        assert!(flat_has(("hiz", "out_hiz"), ("occlusion_cull", "hiz")));
        assert!(flat_has(("gbuffer", "out_depth"), ("hiz", "depth")));
        // The execution chain runs from On Frame Start through the sections.
        assert!(flat_has(("start", "out"), ("object_batch", "in")));

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
        assert!(x("get:scenedb.materials@gbuffer") < x("gbuffer"));
        assert!(x("set:depth@gbuffer") > x("gbuffer"));
    }
}
