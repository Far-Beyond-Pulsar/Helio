//! Checks a graph asset against the node registry and orders its nodes.

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::asset::GraphAsset;
use crate::core_nodes::OUTPUT;
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
    /// Node indices into [`GraphAsset::nodes`], dependencies first.
    pub order: Vec<usize>,
    /// Nodes that lead nowhere near the output.
    pub skipped: Vec<usize>,
}

/// The result of [`validate`]: every finding, and a plan if none is an error.
#[derive(Clone, Debug, Default)]
pub struct Validation {
    pub diagnostics: Vec<Diagnostic>,
    pub plan: Option<Plan>,
}

impl Validation {
    pub fn errors(&self) -> impl Iterator<Item = &Diagnostic> {
        self.diagnostics.iter().filter(|d| d.severity == Severity::Error)
    }

    pub fn has_errors(&self) -> bool {
        self.errors().next().is_some()
    }
}

pub fn validate(asset: &GraphAsset, registry: &NodeRegistry) -> Validation {
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

    // Wires.
    let mut edges: Vec<(usize, usize)> = Vec::new();
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
        if !edges.contains(&(from_idx, to_idx)) {
            edges.push((from_idx, to_idx));
        }
    }

    let outputs: Vec<usize> =
        asset.nodes.iter().enumerate().filter(|(_, n)| n.kind == OUTPUT).map(|(i, _)| i).collect();
    if outputs.len() != 1 {
        error!(None::<&str>, format!("a render graph needs exactly one Output node, found {}", outputs.len()));
    }

    let cyclic = find_cycle(asset.nodes.len(), &edges);
    if let Some(node) = cyclic {
        error!(Some(&asset.nodes[node].id), "is part of a cycle".into());
    }

    let has_errors = diagnostics.iter().any(|d| d.severity == Severity::Error);
    if has_errors {
        return Validation { diagnostics, plan: None };
    }

    // Keep what leads to the output.
    let mut keep: HashSet<usize> = HashSet::new();
    let mut stack = vec![outputs[0]];
    while let Some(node) = stack.pop() {
        if keep.insert(node) {
            stack.extend(edges.iter().filter(|(_, to)| *to == node).map(|(from, _)| *from));
        }
    }

    let mut skipped = Vec::new();
    for (index, node) in asset.nodes.iter().enumerate() {
        if !keep.contains(&index) {
            skipped.push(index);
            diagnostics.push(Diagnostic {
                severity: Severity::Warning,
                node: Some(node.id.clone()),
                message: "does not lead to the Output and is not built".into(),
            });
        }
    }

    // A resource read that nothing writes: Helio rejects it when the graph locks,
    // unless the host supplies it.
    let fed: HashSet<(usize, &str)> =
        feeds.keys().copied().collect();
    for &index in &keep {
        let Some(node_pins) = &pins[index] else { continue };
        for pin in &node_pins.inputs {
            // The host supplies Scene inputs and SceneDB buffers; a flow pin carries no data.
            let host_supplied = pin.ty.kind == PinKind::Scene
                || pin.resource.starts_with(crate::probe::SCENEDB_PREFIX);
            if pin.ty.kind != PinKind::Flow
                && !host_supplied
                && !fed.contains(&(index, pin.name.as_str()))
            {
                diagnostics.push(Diagnostic {
                    severity: Severity::Warning,
                    node: Some(asset.nodes[index].id.clone()),
                    message: format!(
                        "input `{}` is not connected; `{}` must come from a pass outside this graph or a Scene Input",
                        pin.name, pin.resource
                    ),
                });
            }
        }
    }

    // Dependencies first, ties by the order nodes were authored.
    let mut order = Vec::new();
    let mut placed: HashSet<usize> = HashSet::new();
    while order.len() < keep.len() {
        let next = (0..asset.nodes.len())
            .filter(|i| keep.contains(i) && !placed.contains(i))
            .find(|i| {
                edges.iter().filter(|(_, to)| to == i).all(|(from, _)| placed.contains(from) || !keep.contains(from))
            })
            .expect("an acyclic graph always has a node whose dependencies are placed");
        placed.insert(next);
        order.push(next);
    }

    Validation { diagnostics, plan: Some(Plan { order, skipped }) }
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

    #[test]
    fn requires_exactly_one_output() {
        let asset = GraphAsset::default();
        let result = validate(&asset, &registry());
        assert!(result.has_errors());
        assert!(result.plan.is_none());
    }

    #[test]
    fn unreachable_nodes_are_skipped_with_a_warning() {
        let mut asset = GraphAsset::default();
        asset.nodes.push(node("out", OUTPUT, None));
        asset.nodes.push(node("tex", TEXTURE, Some("a")));
        let result = validate(&asset, &registry());
        let plan = result.plan.expect("valid");
        assert_eq!(plan.order, vec![0]);
        assert_eq!(plan.skipped, vec![1]);
        assert!(result.diagnostics.iter().any(|d| d.severity == Severity::Warning));
    }

    #[test]
    fn rejects_mismatched_pins_and_unknown_kinds() {
        let mut asset = GraphAsset::default();
        asset.nodes.push(node("out", OUTPUT, None));
        asset.nodes.push(node("buf", PERSISTENT_BUFFER, Some("b")));
        asset.nodes.push(node("mystery", "no.such.node", None));
        // A buffer cannot feed a flow input.
        asset.connections.push(wire("buf", "b", "out", "in"));
        let result = validate(&asset, &registry());
        let messages: Vec<_> = result.errors().map(|d| d.message.clone()).collect();
        assert!(messages.iter().any(|m| m.contains("cannot feed")), "{messages:?}");
        assert!(messages.iter().any(|m| m.contains("unknown node kind")), "{messages:?}");
    }

    #[test]
    fn orders_dependencies_first() {
        // A resource feeding a custom node that precedes the output.
        let mut registry = registry();
        registry.register(
            crate::registry::NodeDescriptor::new("t.pass", "Pass", "Test", "", |_| Ok(())).with_pins(
                vec![
                    PinDesc::resource(PinType::TEXTURE, "a"),
                    PinDesc::flow("in"),
                ],
                vec![PinDesc::flow("out")],
            ),
        );
        use crate::pins::PinDesc;
        let mut asset = GraphAsset::default();
        asset.nodes.push(node("out", OUTPUT, None));
        asset.nodes.push(node("pass", "t.pass", None));
        asset.nodes.push(node("tex", TEXTURE, Some("a")));
        asset.connections.push(wire("tex", "a", "pass", "a"));
        asset.connections.push(wire("pass", "out", "out", "in"));
        let plan = validate(&asset, &registry).plan.expect("valid");
        let position = |i| plan.order.iter().position(|n| *n == i).unwrap();
        assert!(position(2) < position(1) && position(1) < position(0), "{:?}", plan.order);
    }

    #[test]
    fn detects_cycles_and_double_fed_inputs() {
        let mut registry = registry();
        use crate::pins::PinDesc;
        registry.register(
            crate::registry::NodeDescriptor::new("t.flow", "Flow", "Test", "", |_| Ok(()))
                .with_pins(vec![PinDesc::flow("in")], vec![PinDesc::flow("out")]),
        );
        let mut asset = GraphAsset::default();
        asset.nodes.push(node("out", OUTPUT, None));
        asset.nodes.push(node("a", "t.flow", None));
        asset.nodes.push(node("b", "t.flow", None));
        asset.connections.push(wire("a", "out", "b", "in"));
        asset.connections.push(wire("b", "out", "a", "in"));
        asset.connections.push(wire("b", "out", "out", "in"));
        let result = validate(&asset, &registry);
        assert!(result.errors().any(|d| d.message.contains("cycle")));

        let mut asset = GraphAsset::default();
        asset.nodes.push(node("out", OUTPUT, None));
        asset.nodes.push(node("t1", TEXTURE, Some("x")));
        asset.nodes.push(node("t2", TEXTURE, Some("x")));
        registry.register(
            crate::registry::NodeDescriptor::new("t.read", "Read", "Test", "", |_| Ok(())).with_pins(
                vec![PinDesc::resource(PinType::TEXTURE, "x")],
                vec![PinDesc::flow("out")],
            ),
        );
        asset.nodes.push(node("r", "t.read", None));
        asset.connections.push(wire("t1", "x", "r", "x"));
        asset.connections.push(wire("t2", "x", "r", "x"));
        asset.connections.push(wire("r", "out", "out", "in"));
        let result = validate(&asset, &registry);
        assert!(result.errors().any(|d| d.message.contains("more than one wire")));
    }
}
