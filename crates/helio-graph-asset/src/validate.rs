//! Checks a graph asset against the node registry and orders its nodes.
//!
//! Rules a graph follows:
//!
//! * a wire joins an output to an input of the same kind, persistence and
//!   resource name; a non-flow input has at most one wire;
//! * every node builds, ordered by its wires (ties by the order the nodes were
//!   authored). A node with `enabled` set to `false` is bypassed instead;
//! * a *history* input reads the previous frame's value, so its wire does not
//!   order the graph: feedback is not a cycle;
//! * exactly one Output node ends the pipeline.

use std::collections::{HashMap, HashSet};
use std::fmt;

use serde_json::Value;

use crate::asset::{GraphAsset, NodeInstance};
use crate::core_nodes::{ON_FRAME_START, OUTPUT};
use crate::pins::{PinKind, PinType};
use crate::registry::{NodePins, NodeRegistry};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    /// The graph cannot be built.
    Error,
    /// The graph builds; the author probably wants to look.
    Warning,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    pub severity: Severity,
    /// The node it concerns, if one.
    pub node: Option<String>,
    pub message: String,
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let level = match self.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
        };
        match &self.node {
            Some(node) => write!(f, "{level}: {node}: {}", self.message),
            None => write!(f, "{level}: {}", self.message),
        }
    }
}

/// Which nodes to build, and in what order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Plan {
    /// Node indices into [`GraphAsset::nodes`], dependencies first. Disabled
    /// nodes are not in it.
    pub order: Vec<usize>,
    /// Nodes with `enabled: false`, which build nothing.
    pub disabled: Vec<usize>,
}

/// The result of [`validate`]: every finding, and a plan if none is an error.
#[derive(Clone, Debug, Default)]
pub struct Validation {
    pub diagnostics: Vec<Diagnostic>,
    pub plan: Option<Plan>,
    /// The graph the plan is about: the asset with its buffer nodes flattened away
    /// ([`crate::buffers`]). `plan.order` indexes its nodes.
    pub flat: Option<GraphAsset>,
}

impl Validation {
    pub fn errors(&self) -> impl Iterator<Item = &Diagnostic> {
        self.diagnostics.iter().filter(|d| d.severity == Severity::Error)
    }

    pub fn has_errors(&self) -> bool {
        self.errors().next().is_some()
    }
}

/// Whether a node is bypassed.
pub fn is_disabled(node: &NodeInstance) -> bool {
    matches!(node.params.get("enabled"), Some(Value::Bool(false)))
}

/// Checks `asset`: first its sections are expanded and its buffer nodes flattened away, then the resulting graph
/// is checked and ordered.
pub fn validate(asset: &GraphAsset, registry: &NodeRegistry) -> Validation {
    let expanded = crate::sections::expand(asset);
    let flattened = crate::buffers::flatten(&expanded.asset, registry);
    let mut validation = validate_flat(&flattened.asset, registry);
    validation.diagnostics.splice(0..0, expanded.diagnostics.into_iter().chain(flattened.diagnostics));
    if validation.diagnostics.iter().any(|d| d.severity == Severity::Error) {
        validation.plan = None;
    }
    validation.flat = Some(flattened.asset);
    validation
}

fn validate_flat(asset: &GraphAsset, registry: &NodeRegistry) -> Validation {
    let mut diagnostics: Vec<Diagnostic> = Vec::new();
    macro_rules! error {
        ($node:expr, $message:expr $(,)?) => {
            diagnostics.push(Diagnostic {
                severity: Severity::Error,
                node: $node.map(|n| AsRef::<str>::as_ref(n).to_owned()),
                message: $message,
            })
        };
    }

    let mut index_of: HashMap<&str, usize> = HashMap::new();
    for (index, node) in asset.nodes.iter().enumerate() {
        if index_of.insert(node.id.as_str(), index).is_some() {
            error!(Some(&node.id), "node id is used more than once".into());
        }
    }

    // Pins per node; a node of an unknown kind has none.
    let mut pins: Vec<Option<NodePins>> = Vec::with_capacity(asset.nodes.len());
    for node in &asset.nodes {
        match registry.get(&node.kind) {
            Some(descriptor) => {
                let node_pins = descriptor.pins(&node.params);
                for pin in node_pins.inputs.iter().chain(&node_pins.outputs) {
                    if pin.ty.kind != PinKind::Flow && pin.resource.trim().is_empty() {
                        error!(Some(&node.id), format!("pin `{}` needs a resource name", pin.name));
                    }
                }
                pins.push(Some(node_pins));
            }
            None => {
                error!(Some(&node.id), format!("unknown node kind `{}`", node.kind));
                pins.push(None);
            }
        }
    }

    // Wires: (from, to, history).
    let mut edges: Vec<(usize, usize, bool, bool)> = Vec::new();
    let mut feeds: HashMap<(usize, &str), usize> = HashMap::new();
    for connection in &asset.connections {
        let (from, to) = (&connection.from, &connection.to);
        let (Some(&from_idx), Some(&to_idx)) =
            (index_of.get(from.node.as_str()), index_of.get(to.node.as_str()))
        else {
            error!(None::<&str>, format!("wire {}.{} -> {}.{} names a missing node", from.node, from.pin, to.node, to.pin));
            continue;
        };
        let (Some(from_pins), Some(to_pins)) = (&pins[from_idx], &pins[to_idx]) else {
            continue; // the unknown kind is already reported
        };
        let Some(out_pin) = from_pins.outputs.iter().find(|p| p.name == from.pin) else {
            error!(Some(&from.node), format!("has no output pin `{}`", from.pin));
            continue;
        };
        let Some(in_pin) = to_pins.inputs.iter().find(|p| p.name == to.pin) else {
            error!(Some(&to.node), format!("has no input pin `{}`", to.pin));
            continue;
        };
        if !out_pin.connects_to(in_pin) {
            error!(
                Some(&to.node),
                format!(
                    "`{}.{}` ({}) cannot feed `{}` ({}): pins connect when kind, persistence and resource name all match",
                    from.node,
                    out_pin.name,
                    describe(out_pin.ty, &out_pin.resource),
                    in_pin.name,
                    describe(in_pin.ty, &in_pin.resource),
                ),
            );
            continue;
        }
        if in_pin.ty.kind != PinKind::Flow {
            let count = feeds.entry((to_idx, to.pin.as_str())).or_default();
            *count += 1;
            if *count == 2 {
                error!(Some(&to.node), format!("input `{}` has more than one wire", to.pin));
            }
        }
        let flow = out_pin.ty.kind == PinKind::Flow;
        if !edges.iter().any(|e| e.0 == from_idx && e.1 == to_idx && e.2 == in_pin.history && e.3 == flow) {
            edges.push((from_idx, to_idx, in_pin.history, flow));
        }
    }

    let outputs = asset.nodes.iter().filter(|n| n.kind == OUTPUT).count();
    if outputs != 1 {
        error!(None::<&str>, format!("a render graph needs exactly one Output node, found {outputs}"));
    }
    let starts = asset.nodes.iter().filter(|n| n.kind == ON_FRAME_START).count();
    if starts != 1 {
        error!(None::<&str>, format!("a render graph needs exactly one On Frame Start node, found {starts}"));
    }

    // History wires do not order the graph, so only the others can cycle.
    let ordering: Vec<(usize, usize)> =
        edges.iter().filter(|e| !e.2).map(|e| (e.0, e.1)).collect();
    if let Some(node) = find_cycle(asset.nodes.len(), &ordering) {
        error!(Some(&asset.nodes[node].id), "is part of a cycle".into());
    }

    if diagnostics.iter().any(|d| d.severity == Severity::Error) {
        return Validation { diagnostics, plan: None, flat: None };
    }

    let disabled: Vec<usize> =
        asset.nodes.iter().enumerate().filter(|(_, n)| is_disabled(n)).map(|(i, _)| i).collect();
    let disabled_set: HashSet<usize> = disabled.iter().copied().collect();

    // What a bypassed node would have supplied is missing downstream.
    for &(from, to, history, _) in &edges {
        if disabled_set.contains(&from) && !disabled_set.contains(&to) && !history {
            diagnostics.push(Diagnostic {
                severity: Severity::Warning,
                node: Some(asset.nodes[to].id.clone()),
                message: format!("reads from `{}`, which is disabled", asset.nodes[from].id),
            });
        }
    }

    // Everything that executes hangs off On Frame Start: a node with a flow pin that
    // the chain of flow wires does not reach still builds, but nothing orders it
    // except its data wires.
    let start = asset.nodes.iter().position(|n| n.kind == ON_FRAME_START);
    if let Some(start) = start {
        let mut reached: HashSet<usize> = HashSet::from([start]);
        let mut pending = vec![start];
        while let Some(node) = pending.pop() {
            for &(from, to, _, flow) in &edges {
                if flow && from == node && reached.insert(to) {
                    pending.push(to);
                }
            }
        }
        for (index, node_pins) in pins.iter().enumerate() {
            let Some(node_pins) = node_pins else { continue };
            let executes = node_pins
                .inputs
                .iter()
                .chain(&node_pins.outputs)
                .any(|pin| pin.ty.kind == PinKind::Flow);
            if executes && !reached.contains(&index) && !disabled_set.contains(&index) {
                diagnostics.push(Diagnostic {
                    severity: Severity::Warning,
                    node: Some(asset.nodes[index].id.clone()),
                    message: "is not on the execution chain from On Frame Start".to_string(),
                });
            }
        }
    }

    // A resource read that nothing writes: Helio rejects it when the graph locks,
    // unless the host supplies it or the pass has a fallback.
    let fed: HashSet<(usize, &str)> = feeds.keys().copied().collect();
    for (index, node_pins) in pins.iter().enumerate() {
        let Some(node_pins) = node_pins else { continue };
        if disabled_set.contains(&index) {
            continue;
        }
        for pin in &node_pins.inputs {
            // The host supplies Scene inputs and SceneDB buffers; a flow pin
            // carries no data; an optional input has a fallback.
            let host_supplied = pin.ty.kind == PinKind::Scene
                || pin.resource.starts_with(crate::probe::SCENEDB_PREFIX);
            if pin.ty.kind != PinKind::Flow
                && !host_supplied
                && !pin.optional
                && !fed.contains(&(index, pin.name.as_str()))
            {
                diagnostics.push(Diagnostic {
                    severity: Severity::Warning,
                    node: Some(asset.nodes[index].id.clone()),
                    message: format!(
                        "input `{}` is not connected; `{}` must come from a pass outside this graph",
                        pin.name, pin.resource
                    ),
                });
            }
        }
    }

    // Dependencies first, ties by the order nodes were authored.
    let mut order = Vec::new();
    let mut placed: HashSet<usize> = HashSet::new();
    let building: Vec<usize> =
        (0..asset.nodes.len()).filter(|i| !disabled_set.contains(i)).collect();
    while order.len() < building.len() {
        let next = building
            .iter()
            .copied()
            .filter(|i| !placed.contains(i))
            .find(|i| {
                ordering
                    .iter()
                    .filter(|(_, to)| to == i)
                    .all(|(from, _)| placed.contains(from) || disabled_set.contains(from))
            })
            .expect("an acyclic graph always has a node whose dependencies are placed");
        placed.insert(next);
        order.push(next);
    }

    Validation { diagnostics, plan: Some(Plan { order, disabled }), flat: None }
}

fn describe(ty: PinType, resource: &str) -> String {
    let kind = format!("{:?}", ty.kind).to_lowercase();
    let persistent = if ty.persistent { "persistent " } else { "" };
    if resource.is_empty() {
        format!("{persistent}{kind}")
    } else {
        format!("{persistent}{kind} `{resource}`")
    }
}

/// A node on a cycle, if the edges have one.
fn find_cycle(count: usize, edges: &[(usize, usize)]) -> Option<usize> {
    let mut indegree = vec![0usize; count];
    for (_, to) in edges {
        indegree[*to] += 1;
    }
    let mut ready: Vec<usize> = (0..count).filter(|i| indegree[*i] == 0).collect();
    let mut removed = 0;
    while let Some(node) = ready.pop() {
        removed += 1;
        for (_, to) in edges.iter().filter(|(from, _)| *from == node) {
            indegree[*to] -= 1;
            if indegree[*to] == 0 {
                ready.push(*to);
            }
        }
    }
    (removed < count).then(|| (0..count).find(|i| indegree[*i] > 0).expect("a remaining node"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::{Connection, NodeInstance, PinRef};
    use crate::core_nodes::{PERSISTENT_BUFFER, TEXTURE};
    use crate::pins::PinDesc;
    use crate::registry::NodeDescriptor;
    use serde_json::json;

    fn node(id: &str, kind: &str, name: Option<&str>) -> NodeInstance {
        let mut instance = NodeInstance {
            id: id.into(),
            kind: kind.into(),
            params: Default::default(),
            position: [0.0; 2],
        };
        if let Some(name) = name {
            instance.params.insert("name".into(), json!(name));
        }
        instance
    }

    fn wire(a: &str, ap: &str, b: &str, bp: &str) -> Connection {
        Connection { from: PinRef::new(a, ap), to: PinRef::new(b, bp) }
    }

    fn registry() -> NodeRegistry {
        NodeRegistry::with_core()
    }

    /// A graph with the nodes every graph has: `start` and `out`.
    fn graph() -> GraphAsset {
        let mut asset = GraphAsset::default();
        asset.nodes.push(node("start", ON_FRAME_START, None));
        asset.nodes.push(node("out", OUTPUT, None));
        asset
    }

    fn flow_node(kind: &str) -> NodeDescriptor {
        NodeDescriptor::new(kind, "Flow", "Test", "", |_| Ok(()))
            .with_pins(vec![PinDesc::flow("in")], vec![PinDesc::flow("out")])
    }

    fn order_ids(asset: &GraphAsset, plan: &Plan) -> Vec<String> {
        plan.order.iter().map(|i| asset.nodes[*i].id.clone()).collect()
    }

    #[test]
    fn requires_one_start_and_one_output() {
        let result = validate(&GraphAsset::default(), &registry());
        assert!(result.has_errors());
        assert!(result.plan.is_none());
        let mut only_output = GraphAsset::default();
        only_output.nodes.push(node("out", OUTPUT, None));
        assert!(validate(&only_output, &registry())
            .errors()
            .any(|d| d.message.contains("On Frame Start")));
        assert!(!validate(&graph(), &registry()).has_errors());
    }

    #[test]
    fn every_node_builds_and_disabled_ones_are_bypassed() {
        let mut asset = graph();
        asset.nodes.push(node("tex", TEXTURE, Some("a")));
        asset.nodes.push(node("off", TEXTURE, Some("b")));
        asset.nodes[3].params.insert("enabled".into(), json!(false));
        let plan = validate(&asset, &registry()).plan.expect("valid");
        let built = order_ids(&asset, &plan);
        assert!(built.contains(&"tex".to_string()), "{built:?}");
        assert!(!built.contains(&"off".to_string()), "{built:?}");
        assert_eq!(plan.disabled, vec![3]);
    }

    #[test]
    fn a_reader_of_a_disabled_node_is_warned() {
        let mut registry = registry();
        registry.register(
            NodeDescriptor::new("t.read", "Read", "Test", "", |_| Ok(())).with_pins(
                vec![PinDesc::resource(PinType::TEXTURE, "x")],
                vec![PinDesc::flow("out")],
            ),
        );
        let mut asset = graph();
        asset.nodes.push(node("tex", TEXTURE, Some("x")));
        asset.nodes[2].params.insert("enabled".into(), json!(false));
        asset.nodes.push(node("r", "t.read", None));
        asset.connections.push(wire("tex", "x", "r", "x"));
        let validation = validate(&asset, &registry);
        assert!(validation
            .diagnostics
            .iter()
            .any(|d| d.severity == Severity::Warning && d.message.contains("disabled")));
    }

    #[test]
    fn rejects_mismatched_pins_and_unknown_kinds() {
        let mut asset = graph();
        asset.nodes.push(node("buf", PERSISTENT_BUFFER, Some("b")));
        asset.nodes.push(node("mystery", "no.such.node", None));
        asset.connections.push(wire("buf", "b", "out", "in"));
        let result = validate(&asset, &registry());
        let messages: Vec<_> = result.errors().map(|d| d.message.clone()).collect();
        assert!(messages.iter().any(|m| m.contains("cannot feed")), "{messages:?}");
        assert!(messages.iter().any(|m| m.contains("unknown node kind")), "{messages:?}");
    }

    #[test]
    fn orders_dependencies_first() {
        let mut registry = registry();
        registry.register(
            NodeDescriptor::new("t.pass", "Pass", "Test", "", |_| Ok(())).with_pins(
                vec![PinDesc::resource(PinType::TEXTURE, "a"), PinDesc::flow("in")],
                vec![PinDesc::flow("out")],
            ),
        );
        let mut asset = graph();
        asset.nodes.push(node("pass", "t.pass", None));
        asset.nodes.push(node("tex", TEXTURE, Some("a")));
        asset.connections.push(wire("start", "out", "pass", "in"));
        asset.connections.push(wire("tex", "a", "pass", "a"));
        asset.connections.push(wire("pass", "out", "out", "in"));
        let plan = validate(&asset, &registry).plan.expect("valid");
        let ids = order_ids(&asset, &plan);
        let at = |id: &str| ids.iter().position(|k| k == id).unwrap();
        assert!(at("tex") < at("pass") && at("pass") < at("out"), "{ids:?}");
        assert!(at("start") < at("pass"), "{ids:?}");
    }

    #[test]
    fn an_executing_node_off_the_chain_is_warned() {
        let mut registry = registry();
        registry.register(flow_node("t.flow"));
        let mut asset = graph();
        asset.nodes.push(node("on", "t.flow", None));
        asset.nodes.push(node("off", "t.flow", None));
        asset.connections.push(wire("start", "out", "on", "in"));
        asset.connections.push(wire("on", "out", "out", "in"));
        let warned: Vec<_> = validate(&asset, &registry)
            .diagnostics
            .into_iter()
            .filter(|d| d.message.contains("execution chain"))
            .filter_map(|d| d.node)
            .collect();
        assert_eq!(warned, vec!["off".to_string()]);
    }

    #[test]
    fn history_wires_do_not_make_cycles() {
        // `a` writes x and reads last frame's y; `b` reads x and writes y.
        let mut registry = registry();
        registry.register(
            NodeDescriptor::new("t.a", "A", "Test", "", |_| Ok(())).with_pins(
                vec![PinDesc::resource(PinType::TEXTURE, "y").history()],
                vec![PinDesc::resource(PinType::TEXTURE, "x")],
            ),
        );
        registry.register(
            NodeDescriptor::new("t.b", "B", "Test", "", |_| Ok(())).with_pins(
                vec![PinDesc::resource(PinType::TEXTURE, "x")],
                vec![PinDesc::resource(PinType::TEXTURE, "y")],
            ),
        );
        let mut asset = graph();
        asset.nodes.push(node("a", "t.a", None));
        asset.nodes.push(node("b", "t.b", None));
        asset.connections.push(wire("a", "x", "b", "x"));
        asset.connections.push(wire("b", "y", "a", "y")); // feedback
        let result = validate(&asset, &registry);
        assert!(!result.has_errors(), "{:?}", result.diagnostics);
        let plan = result.plan.expect("valid");
        let ids = order_ids(&asset, &plan);
        let at = |id: &str| ids.iter().position(|k| k == id).unwrap();
        assert!(at("a") < at("b"), "a (writer of x) builds before b: {ids:?}");
    }

    #[test]
    fn detects_cycles_and_double_fed_inputs() {
        let mut registry = registry();
        registry.register(flow_node("t.flow"));
        let mut asset = graph();
        asset.nodes.push(node("a", "t.flow", None));
        asset.nodes.push(node("b", "t.flow", None));
        asset.connections.push(wire("a", "out", "b", "in"));
        asset.connections.push(wire("b", "out", "a", "in"));
        assert!(validate(&asset, &registry).errors().any(|d| d.message.contains("cycle")));

        let mut asset = graph();
        asset.nodes.push(node("t1", TEXTURE, Some("x")));
        asset.nodes.push(node("t2", TEXTURE, Some("x")));
        registry.register(
            NodeDescriptor::new("t.read", "Read", "Test", "", |_| Ok(())).with_pins(
                vec![PinDesc::resource(PinType::TEXTURE, "x")],
                vec![PinDesc::flow("out")],
            ),
        );
        asset.nodes.push(node("r", "t.read", None));
        asset.connections.push(wire("t1", "x", "r", "x"));
        asset.connections.push(wire("t2", "x", "r", "x"));
        assert!(validate(&asset, &registry)
            .errors()
            .any(|d| d.message.contains("more than one wire")));
    }
}
