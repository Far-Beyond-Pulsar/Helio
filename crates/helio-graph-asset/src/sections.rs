//! Sections: groups of nodes shown as one named node.
//!
//! A section is organisation only, like a buffer ([`crate::buffers`]). The
//! editor shows it as a collapsed node that opens onto its own graph; before the
//! graph is checked or built, [`expand`] puts its nodes back in place and joins
//! the wires that crossed its boundary, so the pipeline is the one the nodes
//! make unboxed.
//!
//! A section is placed by one node of kind `section:<id>`. Wires cross its
//! boundary through ports: inside, the pseudo node [`SECTION_ENTRY`] is the
//! source of the wires leaving the section's input pins, and [`SECTION_EXIT`] the
//! target of the wires reaching its output pins.

use std::collections::{HashMap, HashSet};

use crate::asset::{
    Connection, GraphAsset, GraphSection, NodeInstance, PinRef, SectionPort, SECTION_ENTRY,
    SECTION_EXIT, SECTION_PREFIX,
};
use crate::core_nodes::{ON_FRAME_START, OUTPUT};
use crate::layout::LANE_PREFIX;
use crate::pins::{PinDesc, PinType};
use crate::registry::{NodePins, NodeRegistry};
use crate::validate::{Diagnostic, Severity};

/// The id of the section a node kind stands for.
pub fn section_id(kind: &str) -> Option<&str> {
    kind.strip_prefix(SECTION_PREFIX)
}

/// The kind of the node that places the section `id`.
pub fn section_kind(id: &str) -> String {
    format!("{SECTION_PREFIX}{id}")
}

/// The pins of a section's node: its ports.
pub fn section_pins(section: &GraphSection) -> NodePins {
    NodePins {
        inputs: section.inputs.iter().map(SectionPort::to_pin).collect(),
        outputs: section.outputs.iter().map(SectionPort::to_pin).collect(),
    }
}

/// How deeply sections can hold sections; a section holding itself stops here.
const MAX_DEPTH: usize = 32;

/// The graph with every section replaced by its nodes.
pub struct Expanded {
    pub asset: GraphAsset,
    pub diagnostics: Vec<Diagnostic>,
}

/// Replaces each section node by the section's nodes, joining its wires.
pub fn expand(asset: &GraphAsset) -> Expanded {
    let mut diagnostics = Vec::new();
    let mut out = asset.clone();
    out.sections.clear();

    for _ in 0..MAX_DEPTH {
        let Some(position) = out.nodes.iter().position(|n| section_id(&n.kind).is_some()) else {
            return Expanded { asset: out, diagnostics };
        };
        let placeholder = out.nodes.remove(position);
        let id = section_id(&placeholder.kind).unwrap_or_default().to_string();
        let incoming: Vec<Connection> =
            out.connections.iter().filter(|c| c.to.node == placeholder.id).cloned().collect();
        let outgoing: Vec<Connection> =
            out.connections.iter().filter(|c| c.from.node == placeholder.id).cloned().collect();
        out.connections.retain(|c| c.to.node != placeholder.id && c.from.node != placeholder.id);

        let Some(section) = asset.sections.iter().find(|s| s.id == id) else {
            diagnostics.push(Diagnostic {
                severity: Severity::Error,
                node: Some(placeholder.id),
                message: format!("section `{id}` is not defined"),
            });
            continue;
        };

        out.nodes.extend(section.nodes.iter().cloned());
        for wire in &section.connections {
            let from_entry = wire.from.node == SECTION_ENTRY;
            let to_exit = wire.to.node == SECTION_EXIT;
            match (from_entry, to_exit) {
                (false, false) => out.connections.push(wire.clone()),
                (true, false) => {
                    for outer in incoming.iter().filter(|c| c.to.pin == wire.from.pin) {
                        out.connections
                            .push(Connection { from: outer.from.clone(), to: wire.to.clone() });
                    }
                }
                (false, true) => {
                    for outer in outgoing.iter().filter(|c| c.from.pin == wire.to.pin) {
                        out.connections
                            .push(Connection { from: wire.from.clone(), to: outer.to.clone() });
                    }
                }
                (true, true) => {
                    for inner in incoming.iter().filter(|c| c.to.pin == wire.from.pin) {
                        for outer in outgoing.iter().filter(|c| c.from.pin == wire.to.pin) {
                            out.connections.push(Connection {
                                from: inner.from.clone(),
                                to: outer.to.clone(),
                            });
                        }
                    }
                }
            }
        }
    }

    diagnostics.push(Diagnostic {
        severity: Severity::Error,
        node: None,
        message: "sections are nested too deeply (does one hold itself?)".to_string(),
    });
    out.nodes.retain(|n| section_id(&n.kind).is_none());
    Expanded { asset: out, diagnostics }
}

/// Moves the nodes of each lane comment (the groups [`crate::auto_layout`] makes)
/// into a section named after the lane, leaving one node for it in the main
/// graph. On Frame Start and Output stay in the main graph. Wires that crossed a
/// lane's edge become the section's ports. The graph is laid out again after.
pub fn sectionize(asset: &mut GraphAsset, registry: &NodeRegistry) {
    let lanes: Vec<_> =
        asset.comments.iter().filter(|c| c.id.starts_with(LANE_PREFIX)).cloned().collect();
    asset.comments.retain(|c| !c.id.starts_with(LANE_PREFIX));

    for (lane_index, lane) in lanes.iter().enumerate() {
        let kinds: HashMap<&str, &str> =
            asset.nodes.iter().map(|n| (n.id.as_str(), n.kind.as_str())).collect();
        let members: HashSet<String> = lane
            .nodes
            .iter()
            .filter(|id| {
                !matches!(kinds.get(id.as_str()), Some(&ON_FRAME_START) | Some(&OUTPUT) | None)
            })
            .cloned()
            .collect();
        if members.is_empty() {
            continue;
        }
        let id = format!("section_{lane_index}");
        let wires = std::mem::take(&mut asset.connections);
        let pin_of = |node: &str, pin: &str, input: bool| -> PinDesc {
            asset
                .nodes
                .iter()
                .find(|n| n.id == node)
                .and_then(|n| registry.get(&n.kind).map(|d| d.pins(&n.params)))
                .and_then(|pins| {
                    let list = if input { pins.inputs } else { pins.outputs };
                    list.into_iter().find(|p| p.name == pin)
                })
                .unwrap_or_else(|| PinDesc::flow(pin))
        };

        let mut section = GraphSection {
            id: id.clone(),
            name: lane.text.clone(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            nodes: Vec::new(),
            connections: Vec::new(),
        };
        let mut outer: Vec<Connection> = Vec::new();
        let node_id = format!("node_{id}");

        // A port per member pin that a wire crosses at, shared by every such wire.
        let port_for = |section: &mut GraphSection, pin: PinDesc, input: bool, key: &str| {
            let list = if input { &mut section.inputs } else { &mut section.outputs };
            let name = format!("{}{key}", if input { "in_" } else { "out_" });
            if !list.iter().any(|p| p.name == name) {
                list.push(SectionPort { name: name.clone(), ty: pin.ty, resource: pin.resource });
            }
            name
        };

        for wire in wires {
            let from_in = members.contains(&wire.from.node);
            let to_in = members.contains(&wire.to.node);
            match (from_in, to_in) {
                (true, true) => section.connections.push(wire),
                (false, false) => outer.push(wire),
                (false, true) => {
                    let pin = pin_of(&wire.to.node, &wire.to.pin, true);
                    let key = format!("{}_{}", wire.to.node, wire.to.pin);
                    let port = port_for(&mut section, pin, true, &key);
                    section.connections.push(Connection {
                        from: PinRef::new(SECTION_ENTRY, &port),
                        to: wire.to,
                    });
                    outer.push(Connection { from: wire.from, to: PinRef::new(&node_id, port) });
                }
                (true, false) => {
                    let pin = pin_of(&wire.from.node, &wire.from.pin, false);
                    let key = format!("{}_{}", wire.from.node, wire.from.pin);
                    let port = port_for(&mut section, pin, false, &key);
                    section.connections.push(Connection {
                        from: wire.from,
                        to: PinRef::new(SECTION_EXIT, &port),
                    });
                    outer.push(Connection { from: PinRef::new(&node_id, port), to: wire.to });
                }
            }
        }
        drop(pin_of);
        asset.connections = outer;

        let (moved, kept): (Vec<NodeInstance>, Vec<NodeInstance>) =
            std::mem::take(&mut asset.nodes).into_iter().partition(|n| members.contains(&n.id));
        asset.nodes = kept;
        section.nodes = moved;
        asset.nodes.push(NodeInstance {
            id: node_id,
            kind: section_kind(&id),
            params: Default::default(),
            position: [0.0; 2],
        });
        asset.sections.push(section);
    }
    crate::layout::auto_layout(asset, registry);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, kind: &str) -> NodeInstance {
        NodeInstance { id: id.into(), kind: kind.into(), params: Default::default(), position: [0.0; 2] }
    }

    fn wire(from: (&str, &str), to: (&str, &str)) -> Connection {
        Connection { from: PinRef::new(from.0, from.1), to: PinRef::new(to.0, to.1) }
    }

    fn sample() -> GraphAsset {
        let port = |name: &str| SectionPort { name: name.into(), ty: PinType::FLOW, resource: String::new() };
        GraphAsset {
            nodes: vec![node("start", ON_FRAME_START), node("box", "section:s1"), node("end", OUTPUT)],
            connections: vec![wire(("start", "out"), ("box", "in")), wire(("box", "out"), ("end", "in"))],
            sections: vec![GraphSection {
                id: "s1".into(),
                name: "Box".into(),
                inputs: vec![port("in")],
                outputs: vec![port("out")],
                nodes: vec![node("a", "x"), node("b", "x")],
                connections: vec![
                    wire((SECTION_ENTRY, "in"), ("a", "in")),
                    wire(("a", "out"), ("b", "in")),
                    wire(("b", "out"), (SECTION_EXIT, "out")),
                ],
            }],
            ..GraphAsset::default()
        }
    }

    #[test]
    fn expanding_joins_wires_through_the_section() {
        let expanded = expand(&sample());
        assert!(expanded.diagnostics.is_empty(), "{:?}", expanded.diagnostics);
        let ids: Vec<_> = expanded.asset.nodes.iter().map(|n| n.id.as_str()).collect();
        assert!(ids.contains(&"a") && ids.contains(&"b") && !ids.contains(&"box"));
        let c = &expanded.asset.connections;
        assert!(c.contains(&wire(("start", "out"), ("a", "in"))));
        assert!(c.contains(&wire(("a", "out"), ("b", "in"))));
        assert!(c.contains(&wire(("b", "out"), ("end", "in"))));
        assert_eq!(c.len(), 3);
    }

    #[test]
    fn a_missing_or_self_holding_section_is_an_error() {
        let mut asset = sample();
        asset.sections.clear();
        assert!(expand(&asset).diagnostics.iter().any(|d| d.message.contains("not defined")));

        let mut looped = sample();
        looped.sections[0].nodes.push(node("again", "section:s1"));
        assert!(expand(&looped).diagnostics.iter().any(|d| d.message.contains("nested too deeply")));
    }
}
