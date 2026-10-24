//! Buffers as variables.
//!
//! A buffer is a named resource in a graph, and a graph can mention it as often
//! as it likes: a **Set Buffer** node takes the value of whatever is wired into it,
//! and a **Get Buffer** node offers that value wherever it is placed. They exist
//! for readability only. Before a graph is validated or built they are flattened
//! away: every consumer of a Get is wired straight to the producer of the Set it
//! reads, so a graph full of buffer nodes compiles to the same pipeline as one
//! full of long wires.
//!
//! A buffer is written once per *version*: a pass that reads `pre_aa` and writes it
//! again sets it again. A Get reads the latest Set made by something that executes
//! before the node using it. A *history* input reads the latest Set of the frame,
//! whatever its position, since it reads last frame's value.
//!
//! A buffer node names its buffer in its `buffer` parameter as `<tag>:<name>`
//! (`tex:pre_aa`, `buf:object_batch`, see [`PinType::prefix`]).

use std::collections::{HashMap, HashSet};

use serde_json::Value;

use crate::asset::{Connection, GraphAsset, NodeInstance, PinRef};
use crate::core_nodes::{BUFFER_GET, BUFFER_SET};
use crate::layout::stable_topological_order;
use crate::pins::{PinKind, PinType};
use crate::probe::SCENEDB_PREFIX;
use crate::registry::NodeRegistry;
use crate::validate::{Diagnostic, Severity};

/// Parses a `buffer` parameter, `<tag>:<name>`.
pub fn parse_entry(text: &str) -> Option<(PinType, String)> {
    let (tag, name) = text.trim().split_once(':')?;
    let ty = PinType::from_prefix(tag.trim())?;
    let name = name.trim();
    (!name.is_empty()).then(|| (ty, name.to_string()))
}

/// Writes a buffer as a `buffer` parameter.
pub fn entry(ty: PinType, name: &str) -> Option<String> {
    ty.prefix().map(|tag| format!("{tag}:{name}"))
}

/// The buffer a node names, if it is a buffer node with a valid parameter.
pub fn buffer_of(node: &NodeInstance) -> Option<(PinType, String)> {
    node.params.get("buffer").and_then(Value::as_str).and_then(parse_entry)
}

/// Every buffer the generated nodes read or write, as `(type, name)` sorted by type
/// then name: what the editor offers as variables.
pub fn known_buffers(registry: &NodeRegistry) -> Vec<(PinType, String)> {
    let mut found: Vec<(PinType, String)> = Vec::new();
    for pins in registry.generated_pins() {
        for pin in pins.inputs.iter().chain(&pins.outputs) {
            if pin.ty.kind == PinKind::Flow || pin.ty.prefix().is_none() {
                continue;
            }
            let item = (pin.ty, pin.resource.clone());
            if !found.contains(&item) {
                found.push(item);
            }
        }
    }
    found.sort_by(|a, b| a.0.prefix().cmp(&b.0.prefix()).then_with(|| a.1.cmp(&b.1)));
    found
}

/// The title of a buffer node: what it does and to which buffer (`Get pre_aa`).
/// `None` for any other node or a node without a valid buffer.
pub fn title_of(node: &NodeInstance) -> Option<String> {
    let (_, name) = buffer_of(node)?;
    match node.kind.as_str() {
        BUFFER_GET => Some(format!("Get {name}")),
        BUFFER_SET => Some(format!("Set {name}")),
        _ => None,
    }
}

pub fn is_buffer_node(node: &NodeInstance) -> bool {
    node.kind == BUFFER_GET || node.kind == BUFFER_SET
}

/// The result of [`flatten`].
pub struct Flattened {
    /// The graph without buffer nodes, their wires replaced by direct ones.
    pub asset: GraphAsset,
    pub diagnostics: Vec<Diagnostic>,
}

struct SetSite {
    producer: usize,
    pin: String,
    rank: usize,
}

/// Removes the Get and Set nodes of `asset`, wiring each consumer of a Get to the
/// producer of the Set it reads.
pub fn flatten(asset: &GraphAsset, registry: &NodeRegistry) -> Flattened {
    if !asset.nodes.iter().any(is_buffer_node) {
        return Flattened { asset: asset.clone(), diagnostics: Vec::new() };
    }

    let mut diagnostics: Vec<Diagnostic> = Vec::new();
    let mut diagnose = |severity: Severity, node: Option<&str>, message: String| {
        diagnostics.push(Diagnostic { severity, node: node.map(str::to_owned), message });
    };

    let index: HashMap<&str, usize> =
        asset.nodes.iter().enumerate().map(|(i, n)| (n.id.as_str(), i)).collect();
    let buffer_nodes: HashMap<usize, (PinType, String)> = asset
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| is_buffer_node(n))
        .filter_map(|(i, n)| {
            let parsed = buffer_of(n);
            if parsed.is_none() {
                diagnose(
                    Severity::Error,
                    Some(&n.id),
                    "needs a buffer: set its `buffer` to `<tag>:<name>`, like `tex:pre_aa`".into(),
                );
            }
            parsed.map(|buffer| (i, buffer))
        })
        .collect();
    let real: Vec<usize> =
        (0..asset.nodes.len()).filter(|i| !is_buffer_node(&asset.nodes[*i])).collect();

    // Pins of the real nodes, to find input pins and which are history.
    let pins_of = |node: usize| {
        registry.get(&asset.nodes[node].kind).map(|d| d.pins(&asset.nodes[node].params))
    };
    let input_pin = |node: usize, name: &str| {
        pins_of(node).and_then(|pins| pins.inputs.into_iter().find(|p| p.name == name))
    };
    let output_pin = |node: usize, name: &str| {
        pins_of(node).and_then(|pins| pins.outputs.into_iter().find(|p| p.name == name))
    };

    // Execution rank of every real node: a stable topological order of the wires
    // between real nodes, history wires aside.
    let mut ordering: Vec<(usize, usize)> = Vec::new();
    for connection in &asset.connections {
        let (Some(&from), Some(&to)) =
            (index.get(connection.from.node.as_str()), index.get(connection.to.node.as_str()))
        else {
            continue;
        };
        if buffer_nodes.contains_key(&from) || buffer_nodes.contains_key(&to) || from == to {
            continue;
        }
        let history = input_pin(to, &connection.to.pin).is_some_and(|p| p.history);
        if !history {
            ordering.push((from, to));
        }
    }
    let rank: HashMap<usize, usize> = stable_topological_order(&real, &ordering)
        .into_iter()
        .enumerate()
        .map(|(position, node)| (node, position))
        .collect();

    // Sets: what is wired into each Set node.
    let mut sets: HashMap<String, Vec<SetSite>> = HashMap::new();
    let mut fed_sets: HashSet<usize> = HashSet::new();
    for connection in &asset.connections {
        let (Some(&from), Some(&to)) =
            (index.get(connection.from.node.as_str()), index.get(connection.to.node.as_str()))
        else {
            continue;
        };
        let Some((ty, name)) = buffer_nodes.get(&to).filter(|_| asset.nodes[to].kind == BUFFER_SET)
        else {
            continue;
        };
        if buffer_nodes.contains_key(&from) {
            diagnose(
                Severity::Error,
                Some(&asset.nodes[to].id),
                "a buffer node cannot be wired straight to another buffer node".into(),
            );
            continue;
        }
        let Some(out) = output_pin(from, &connection.from.pin) else {
            diagnose(
                Severity::Error,
                Some(&asset.nodes[from].id),
                format!("has no output pin `{}`", connection.from.pin),
            );
            continue;
        };
        if out.ty != *ty || out.resource != *name {
            diagnose(
                Severity::Error,
                Some(&asset.nodes[to].id),
                format!(
                    "`{}.{}` ({}) cannot be set into buffer `{name}`",
                    asset.nodes[from].id, out.name, out.resource
                ),
            );
            continue;
        }
        fed_sets.insert(to);
        sets.entry(name.clone()).or_default().push(SetSite {
            producer: from,
            pin: connection.from.pin.clone(),
            rank: rank.get(&from).copied().unwrap_or(0),
        });
    }
    for (&node, (_, name)) in &buffer_nodes {
        if asset.nodes[node].kind == BUFFER_SET && !fed_sets.contains(&node) {
            diagnose(
                Severity::Warning,
                Some(&asset.nodes[node].id),
                format!("sets buffer `{name}` but nothing is wired into it"),
            );
        }
    }

    // Gets: wire each consumer to the Set it reads.
    let mut resolved: Vec<Connection> = Vec::new();
    for connection in &asset.connections {
        let (Some(&from), Some(&to)) =
            (index.get(connection.from.node.as_str()), index.get(connection.to.node.as_str()))
        else {
            continue;
        };
        let Some((ty, name)) = buffer_nodes.get(&from).filter(|_| asset.nodes[from].kind == BUFFER_GET)
        else {
            continue;
        };
        if buffer_nodes.contains_key(&to) {
            continue; // reported with the Set it feeds
        }
        let Some(consumer) = input_pin(to, &connection.to.pin) else {
            diagnose(
                Severity::Error,
                Some(&asset.nodes[to].id),
                format!("has no input pin `{}`", connection.to.pin),
            );
            continue;
        };
        if consumer.ty != *ty || consumer.resource != *name {
            diagnose(
                Severity::Error,
                Some(&asset.nodes[to].id),
                format!(
                    "input `{}` ({}) cannot read buffer `{name}`: pins connect when kind, persistence and name all match",
                    consumer.name, consumer.resource
                ),
            );
            continue;
        }

        let consumer_rank = rank.get(&to).copied().unwrap_or(usize::MAX);
        let candidates = sets.get(name).map(Vec::as_slice).unwrap_or(&[]);
        let pick = if consumer.history {
            candidates.iter().max_by_key(|site| site.rank)
        } else {
            candidates.iter().filter(|site| site.rank < consumer_rank).max_by_key(|site| site.rank)
        };
        match pick {
            Some(site) => resolved.push(Connection {
                from: PinRef::new(&asset.nodes[site.producer].id, &site.pin),
                to: connection.to.clone(),
            }),
            None => {
                // The host supplies scene inputs and SceneDB buffers; so do passes with
                // a fallback. Anything else is a read before any write.
                let supplied = ty.kind == PinKind::Scene
                    || name.starts_with(SCENEDB_PREFIX)
                    || consumer.optional;
                if !supplied {
                    diagnose(
                        Severity::Warning,
                        Some(&asset.nodes[to].id),
                        format!("reads buffer `{name}` before anything sets it"),
                    );
                }
            }
        }
    }

    // The flattened graph: real nodes, their direct wires, and the resolved ones.
    let mut flat = asset.clone();
    flat.nodes.retain(|n| !is_buffer_node(n));
    flat.connections.retain(|c| {
        let touches_buffer = |pin: &PinRef| {
            index.get(pin.node.as_str()).is_some_and(|i| buffer_nodes.contains_key(i))
                || asset
                    .nodes
                    .iter()
                    .any(|n| n.id == pin.node && is_buffer_node(n))
        };
        !touches_buffer(&c.from) && !touches_buffer(&c.to)
    });
    flat.connections.extend(resolved);
    Flattened { asset: flat, diagnostics }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_nodes::{ON_FRAME_START, OUTPUT};
    use crate::pins::PinDesc;
    use crate::registry::NodeDescriptor;
    use serde_json::json;

    fn node(id: &str, kind: &str) -> NodeInstance {
        NodeInstance {
            id: id.into(),
            kind: kind.into(),
            params: Default::default(),
            position: [0.0; 2],
        }
    }

    fn buffer(id: &str, kind: &str, entry: &str) -> NodeInstance {
        let mut n = node(id, kind);
        n.params.insert("buffer".into(), json!(entry));
        n
    }

    fn wire(a: &str, ap: &str, b: &str, bp: &str) -> Connection {
        Connection { from: PinRef::new(a, ap), to: PinRef::new(b, bp) }
    }

    /// A pass that reads and writes `pre_aa`.
    fn registry() -> NodeRegistry {
        let mut registry = NodeRegistry::with_core();
        registry.register(
            NodeDescriptor::new("t.pass", "Pass", "Test", "", |_| Ok(())).with_pins(
                vec![PinDesc::flow("in"), PinDesc::resource(PinType::TEXTURE, "pre_aa")],
                vec![
                    PinDesc::flow("out"),
                    helio_graph_asset_output("pre_aa"),
                ],
            ),
        );
        registry
    }

    fn helio_graph_asset_output(resource: &str) -> PinDesc {
        crate::probe::output_pin(PinType::TEXTURE, resource)
    }

    /// start -> a -> b -> out, each pass reading pre_aa through a Get and writing it
    /// through a Set.
    fn chain() -> GraphAsset {
        let mut asset = GraphAsset::default();
        for (id, kind) in [
            ("start", ON_FRAME_START),
            ("a", "t.pass"),
            ("b", "t.pass"),
            ("out", OUTPUT),
        ] {
            asset.nodes.push(node(id, kind));
        }
        asset.nodes.push(buffer("set_a", BUFFER_SET, "tex:pre_aa"));
        asset.nodes.push(buffer("get_b", BUFFER_GET, "tex:pre_aa"));
        asset.nodes.push(buffer("set_b", BUFFER_SET, "tex:pre_aa"));
        asset.nodes.push(buffer("get_a", BUFFER_GET, "tex:pre_aa"));
        asset.connections = vec![
            wire("start", "out", "a", "in"),
            wire("a", "out", "b", "in"),
            wire("b", "out", "out", "in"),
            wire("get_a", "pre_aa", "a", "pre_aa"),
            wire("a", "out_pre_aa", "set_a", "pre_aa"),
            wire("get_b", "pre_aa", "b", "pre_aa"),
            wire("b", "out_pre_aa", "set_b", "pre_aa"),
        ];
        asset
    }

    #[test]
    fn a_get_reads_the_latest_set_before_its_consumer() {
        let flat = flatten(&chain(), &registry());
        assert!(flat.diagnostics.iter().all(|d| d.severity != Severity::Error), "{:?}", flat.diagnostics);
        assert!(flat.asset.nodes.iter().all(|n| !is_buffer_node(n)));
        // b reads what a set; a reads nothing (host-less read before any set: warned).
        assert!(flat.asset.connections.contains(&wire("a", "out_pre_aa", "b", "pre_aa")));
        assert!(!flat
            .asset
            .connections
            .iter()
            .any(|c| c.to.node == "a" && c.to.pin == "pre_aa"));
        assert!(flat
            .diagnostics
            .iter()
            .any(|d| d.severity == Severity::Warning && d.message.contains("before anything sets it")));
    }

    #[test]
    fn a_pass_does_not_read_its_own_set() {
        // Only b sets the buffer; a reads it: nothing earlier than a sets it.
        let mut asset = chain();
        asset.nodes.retain(|n| n.id != "set_a");
        asset.connections.retain(|c| c.to.node != "set_a");
        let flat = flatten(&asset, &registry());
        assert!(!flat.asset.connections.contains(&wire("b", "out_pre_aa", "a", "pre_aa")));
        assert!(!flat.asset.connections.contains(&wire("b", "out_pre_aa", "b", "pre_aa")));
    }

    #[test]
    fn a_wrong_buffer_type_is_an_error() {
        let mut asset = chain();
        asset.nodes[5] = buffer("get_b", BUFFER_GET, "buf:pre_aa");
        let flat = flatten(&asset, &registry());
        assert!(flat.diagnostics.iter().any(|d| d.severity == Severity::Error));
    }

    #[test]
    fn graphs_without_buffer_nodes_pass_through() {
        let mut asset = chain();
        asset.nodes.retain(|n| !is_buffer_node(n));
        asset.connections.retain(|c| !c.from.node.starts_with("get") && !c.to.node.starts_with("set"));
        let flat = flatten(&asset, &registry());
        assert_eq!(flat.asset, asset);
        assert!(flat.diagnostics.is_empty());
    }

    #[test]
    fn entries_round_trip() {
        assert_eq!(parse_entry("ptex:history"), Some((PinType::PERSISTENT_TEXTURE, "history".into())));
        assert_eq!(parse_entry("nope:x"), None);
        assert_eq!(parse_entry("tex:"), None);
        assert_eq!(entry(PinType::BUFFER, "object_batch").as_deref(), Some("buf:object_batch"));
        assert_eq!(entry(PinType::FLOW, "x"), None);
    }
}
