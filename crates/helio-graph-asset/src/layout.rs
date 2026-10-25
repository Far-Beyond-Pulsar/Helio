//! Automatic layout of a render graph.
//!
//! A layered (Sugiyama-style) layout shaped for execution graphs:
//!
//! 1. **Rank.** Nodes that execute (they have a flow pin) are ordered by a stable
//!    topological sort of the flow and data wires (history wires aside, they point
//!    back in time). On Frame Start comes first, the Output last. That order is the
//!    x axis.
//! 2. **Attach.** A node that only supplies data (host inputs, SceneDB buffers,
//!    resources) is placed in a column directly left of the first node that uses it,
//!    so a wire never has to cross the graph to reach its source.
//! 3. **Lanes.** Runs of executing nodes of one palette category form a lane (a row
//!    of the page): Scene Prep, Geometry, Lighting, and so on. A lane wraps when it
//!    gets too wide. Each lane gets a labelled comment box around it.
//! 4. **Place.** Sizes are estimated with the editor's own metrics, so nodes never
//!    overlap and lanes never collide.
//!
//! Nodes with no consumer go in a lane of their own at the end.

use std::collections::{HashMap, HashSet};

use crate::asset::{GraphAsset, GraphComment};
use crate::core_nodes::{ON_FRAME_START, OUTPUT};
use crate::pins::PinKind;
use crate::registry::{NodePins, NodeRegistry};

/// Prefix of the ids of the comments this module creates. It replaces its own
/// on every run and leaves the author's alone.
pub const LANE_PREFIX: &str = "lane:";

// The editor's node metrics (`blueprint_editor::rendering::layout`).
const HEADER_H: f32 = 28.0;
const SEP_H: f32 = 2.0;
const BODY_PAD: f32 = 8.0;
const PIN_ROW_H: f32 = 18.0;
const PIN_GAP: f32 = 4.0;
const PIN_SIZE: f32 = 12.0;
const HEADER_PAD_X: f32 = 9.0;
const GRID: f32 = 10.0;
/// Average glyph width of the title and pin label fonts, and slack on top of the
/// estimate: an estimate that is a little large only costs space.
const TITLE_GLYPH: f32 = 7.4;
const LABEL_GLYPH: f32 = 6.6;
const SLACK_X: f32 = 24.0;
const SLACK_Y: f32 = 12.0;

const SOURCE_GAP: f32 = 70.0;
const SOURCE_STACK_GAP: f32 = 24.0;
const NODE_GAP: f32 = 110.0;
const ROW_GAP: f32 = 120.0;
const LANE_GAP: f32 = 170.0;
const COMMENT_PAD: f32 = 50.0;
const COMMENT_TITLE: f32 = 56.0;
const MAX_ROW_WIDTH: f32 = 5200.0;

fn snap(value: f32) -> f32 {
    (value / GRID).ceil() * GRID
}

fn label_len(pin: &crate::pins::PinDesc) -> usize {
    if pin.ty.kind == PinKind::Flow {
        0
    } else if pin.history {
        pin.resource.chars().count() + " (prev frame)".len()
    } else {
        pin.resource.chars().count()
    }
}

/// The size the editor will give a node with these pins and this title.
pub fn estimated_size(title: &str, pins: &NodePins) -> (f32, f32) {
    let rows = pins.inputs.len().max(pins.outputs.len()).max(1);
    let rows_f = rows as f32;
    let height = HEADER_H + SEP_H + BODY_PAD * 2.0 + rows_f * PIN_ROW_H + (rows_f - 1.0) * PIN_GAP;

    let header = HEADER_PAD_X * 2.0 + title.chars().count() as f32 * TITLE_GLYPH;
    let body = (0..rows)
        .map(|row| {
            let input = pins.inputs.get(row).map_or(0, label_len) as f32 * LABEL_GLYPH;
            let output = pins.outputs.get(row).map_or(0, label_len) as f32 * LABEL_GLYPH;
            let gap = if input > 0.0 && output > 0.0 {
                12.0
            } else {
                0.0
            };
            BODY_PAD * 2.0 + PIN_SIZE + 10.0 + input + output + gap
        })
        .fold(0.0_f32, f32::max);
    (
        snap(header.max(body).max(160.0)) + SLACK_X,
        snap(height) + SLACK_Y,
    )
}

struct Item {
    kind: String,
    category: String,
    size: (f32, f32),
    executes: bool,
}

/// Lays out a graph that holds sections: each section's own nodes as a graph of
/// their own, then the main graph as a row of nodes (the sections among them) in
/// execution order, wrapping when it gets too wide.
fn layout_sections(asset: &mut GraphAsset, registry: &NodeRegistry) {
    for section in &mut asset.sections {
        let mut inner = GraphAsset {
            nodes: std::mem::take(&mut section.nodes),
            connections: section.connections.clone(),
            ..GraphAsset::default()
        };
        auto_layout(&mut inner, registry);
        section.nodes = inner.nodes;
    }

    let count = asset.nodes.len();
    let index: HashMap<&str, usize> = asset
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.id.as_str(), i))
        .collect();
    let sizes: Vec<(f32, f32)> = asset
        .nodes
        .iter()
        .map(|node| match asset.section_of(node) {
            Some(section) => estimated_size(&section.name, &crate::sections::section_pins(section)),
            None => match registry.get(&node.kind) {
                Some(descriptor) => {
                    let title =
                        crate::buffers::title_of(node).unwrap_or_else(|| descriptor.title.clone());
                    estimated_size(&title, &descriptor.pins(&node.params))
                }
                None => estimated_size(&node.kind, &NodePins::default()),
            },
        })
        .collect();
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for wire in &asset.connections {
        if let (Some(&from), Some(&to)) = (
            index.get(wire.from.node.as_str()),
            index.get(wire.to.node.as_str()),
        ) {
            if from != to && !edges.contains(&(from, to)) {
                edges.push((from, to));
            }
        }
    }
    let all: Vec<usize> = (0..count).collect();
    let order = stable_topological_order(&all, &edges);

    let (mut x, mut y, mut row_h) = (0.0_f32, 0.0_f32, 0.0_f32);
    let mut positions = vec![[0.0_f32; 2]; count];
    for node in order {
        let (w, h) = sizes[node];
        if x > 0.0 && x + w > MAX_ROW_WIDTH {
            y += row_h + ROW_GAP;
            x = 0.0;
            row_h = 0.0;
        }
        positions[node] = [x, y];
        x += w + NODE_GAP;
        row_h = row_h.max(h);
    }
    for (node, position) in asset.nodes.iter_mut().zip(positions) {
        node.position = position;
    }
    asset.comments.retain(|c| !c.id.starts_with(LANE_PREFIX));
}

/// Lays `asset` out. Positions every node and replaces the lane comments.
pub fn auto_layout(asset: &mut GraphAsset, registry: &NodeRegistry) {
    if !asset.sections.is_empty() && asset.nodes.iter().any(|n| asset.section_of(n).is_some()) {
        layout_sections(asset, registry);
        return;
    }
    let count = asset.nodes.len();
    if count == 0 {
        asset.comments.retain(|c| !c.id.starts_with(LANE_PREFIX));
        return;
    }
    let index: HashMap<&str, usize> = asset
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.id.as_str(), i))
        .collect();

    let mut items: Vec<Item> = Vec::with_capacity(count);
    let mut all_pins: Vec<NodePins> = Vec::with_capacity(count);
    for node in &asset.nodes {
        let (title, category, pins) = match registry.get(&node.kind) {
            Some(descriptor) => (
                // A buffer node shows its buffer in its title, as the editor does.
                crate::buffers::title_of(node).unwrap_or_else(|| descriptor.title.clone()),
                descriptor.category.clone(),
                descriptor.pins(&node.params),
            ),
            None => (node.kind.clone(), String::new(), NodePins::default()),
        };
        let executes = pins
            .inputs
            .iter()
            .chain(&pins.outputs)
            .any(|pin| pin.ty.kind == PinKind::Flow);
        items.push(Item {
            kind: node.kind.clone(),
            category,
            size: estimated_size(&title, &pins),
            executes,
        });
        all_pins.push(pins);
    }

    // Wires that order the graph: flow and data, not history.
    let mut forward: Vec<(usize, usize)> = Vec::new();
    for connection in &asset.connections {
        let (Some(&from), Some(&to)) = (
            index.get(connection.from.node.as_str()),
            index.get(connection.to.node.as_str()),
        ) else {
            continue;
        };
        let history = all_pins[to]
            .inputs
            .iter()
            .find(|pin| pin.name == connection.to.pin)
            .is_some_and(|pin| pin.history);
        if !history && from != to && !forward.contains(&(from, to)) {
            forward.push((from, to));
        }
    }

    // 1. Rank: a stable topological order of the executing nodes.
    let executing: Vec<usize> = (0..count).filter(|i| items[*i].executes).collect();
    let order = stable_topological_order(&executing, &forward);

    // 2. Attach each data node to the first executing node that uses it.
    let position_in_order: HashMap<usize, usize> = order
        .iter()
        .enumerate()
        .map(|(rank, node)| (*node, rank))
        .collect();
    let mut reads: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut sets: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut unused: Vec<usize> = Vec::new();
    for node in (0..count).filter(|i| !items[*i].executes) {
        if let Some(user) = first_user(node, &forward, &position_in_order) {
            reads.entry(user).or_default().push(node);
        } else if let Some(producer) = last_producer(node, &forward, &position_in_order) {
            sets.entry(producer).or_default().push(node);
        } else {
            unused.push(node);
        }
    }

    // 3. Lanes: runs of one category. The start joins the lane after it and the
    // output the lane before, so neither sits in a lane alone.
    let mut keys: Vec<String> = order.iter().map(|n| items[*n].category.clone()).collect();
    for rank in 0..order.len() {
        if items[order[rank]].kind == ON_FRAME_START {
            if let Some(next) = (rank + 1..order.len()).find(|r| items[order[*r]].kind != OUTPUT) {
                keys[rank] = keys[next].clone();
            }
        }
    }
    for rank in (0..order.len()).rev() {
        if items[order[rank]].kind == OUTPUT && rank > 0 {
            keys[rank] = keys[rank - 1].clone();
        }
    }
    let mut lanes: Vec<(String, Vec<usize>)> = Vec::new();
    for (rank, node) in order.iter().enumerate() {
        match lanes.last_mut() {
            Some((key, nodes)) if *key == keys[rank] => nodes.push(*node),
            _ => lanes.push((keys[rank].clone(), vec![*node])),
        }
    }

    // 4. Place.
    let mut positions: Vec<[f32; 2]> = vec![[0.0; 2]; count];
    let mut comments: Vec<GraphComment> = Vec::new();
    let mut lane_y = 0.0_f32;
    for (lane_index, (key, nodes)) in lanes.iter().enumerate() {
        let members: Vec<Member> = nodes
            .iter()
            .map(|node| {
                (
                    *node,
                    reads.get(node).cloned().unwrap_or_default(),
                    sets.get(node).cloned().unwrap_or_default(),
                )
            })
            .collect();
        let bounds = place_lane(&members, &items, &mut positions, lane_y);
        comments.push(lane_comment(lane_index, key, &members, bounds));
        lane_y = bounds.3 + COMMENT_PAD + LANE_GAP;
    }
    if !unused.is_empty() {
        let members: Vec<Member> = vec![(usize::MAX, unused.clone(), Vec::new())];
        let bounds = place_lane(&members, &items, &mut positions, lane_y);
        // Host inputs declared for the renderer and not wired anywhere are a group of
        // their own; anything else left over is just unattached.
        let title = if unused.iter().all(|n| items[*n].kind.starts_with("host.")) {
            "Host Inputs"
        } else {
            "Unattached"
        };
        comments.push(lane_comment(lanes.len(), title, &members, bounds));
    }

    for (node, position) in asset.nodes.iter_mut().zip(&positions) {
        node.position = *position;
    }
    let names: Vec<String> = asset.nodes.iter().map(|n| n.id.clone()).collect();
    for comment in &mut comments {
        // Members are stored by index while placing; turn them into ids.
        comment.nodes = comment
            .nodes
            .iter()
            .filter_map(|i| i.parse::<usize>().ok().and_then(|i| names.get(i).cloned()))
            .collect();
    }
    asset.comments.retain(|c| !c.id.starts_with(LANE_PREFIX));
    asset.comments.extend(comments);
}

/// Kahn's algorithm over `nodes`, taking the lowest index among those ready, so
/// graphs that do not constrain two nodes keep the order they were authored in.
/// Nodes on a cycle follow the rest in authoring order.
pub(crate) fn stable_topological_order(nodes: &[usize], edges: &[(usize, usize)]) -> Vec<usize> {
    let member: HashSet<usize> = nodes.iter().copied().collect();
    let mut remaining: HashMap<usize, usize> = nodes.iter().map(|n| (*n, 0)).collect();
    for (from, to) in edges {
        if member.contains(from) && member.contains(to) {
            *remaining.get_mut(to).expect("member") += 1;
        }
    }
    let mut order = Vec::with_capacity(nodes.len());
    let mut done: HashSet<usize> = HashSet::new();
    while order.len() < nodes.len() {
        let next = nodes
            .iter()
            .copied()
            .find(|n| !done.contains(n) && remaining[n] == 0);
        let Some(next) = next else {
            order.extend(nodes.iter().copied().filter(|n| !done.contains(n)));
            break;
        };
        done.insert(next);
        order.push(next);
        for (from, to) in edges {
            if *from == next && member.contains(to) {
                *remaining.get_mut(to).expect("member") -= 1;
            }
        }
    }
    order
}

/// The executing node, earliest in the order, that `node` feeds directly or
/// through other data nodes.
fn first_user(
    node: usize,
    edges: &[(usize, usize)],
    rank: &HashMap<usize, usize>,
) -> Option<usize> {
    let mut best: Option<usize> = None;
    let mut seen: HashSet<usize> = HashSet::from([node]);
    let mut pending = vec![node];
    while let Some(current) = pending.pop() {
        for (from, to) in edges {
            if *from != current || !seen.insert(*to) {
                continue;
            }
            match rank.get(to) {
                Some(r) => {
                    if best.map_or(true, |b| *r < rank[&b]) {
                        best = Some(*to);
                    }
                }
                None => pending.push(*to),
            }
        }
    }
    best
}

/// The executing node, latest in the order, that feeds `node` directly or through
/// other data nodes: where a node that only receives data (a Set Buffer) goes.
fn last_producer(
    node: usize,
    edges: &[(usize, usize)],
    rank: &HashMap<usize, usize>,
) -> Option<usize> {
    let mut best: Option<usize> = None;
    let mut seen: HashSet<usize> = HashSet::from([node]);
    let mut pending = vec![node];
    while let Some(current) = pending.pop() {
        for (from, to) in edges {
            if *to != current || !seen.insert(*from) {
                continue;
            }
            match rank.get(from) {
                Some(r) => {
                    if best.map_or(true, |b| *r > rank[&b]) {
                        best = Some(*from);
                    }
                }
                None => pending.push(*from),
            }
        }
    }
    best
}

/// One member of a lane: an executing node with the data nodes placed on its left
/// (what it reads) and right (what it sets).
type Member = (usize, Vec<usize>, Vec<usize>);

/// Most nodes in one column of a block of data nodes; more go in further columns.
const BLOCK_ROWS: usize = 8;
const BLOCK_COLUMN_GAP: f32 = 30.0;

/// Arranges `nodes` as a grid of columns of at most [`BLOCK_ROWS`], returning each
/// node's offset within the block and the block's size.
fn block_layout(nodes: &[usize], items: &[Item]) -> (Vec<(usize, f32, f32)>, (f32, f32)) {
    if nodes.is_empty() {
        return (Vec::new(), (0.0, 0.0));
    }
    let columns = nodes.len().div_ceil(BLOCK_ROWS);
    let rows = nodes.len().div_ceil(columns);
    let column_w: Vec<f32> = (0..columns)
        .map(|c| {
            nodes
                .iter()
                .enumerate()
                .filter(|(k, _)| k / rows == c)
                .map(|(_, n)| items[*n].size.0)
                .fold(0.0_f32, f32::max)
        })
        .collect();
    let mut offsets = Vec::with_capacity(nodes.len());
    let mut height = 0.0_f32;
    let mut column_y = vec![0.0_f32; columns];
    for (k, node) in nodes.iter().enumerate() {
        let column = k / rows;
        let x: f32 = column_w[..column].iter().sum::<f32>() + BLOCK_COLUMN_GAP * column as f32;
        offsets.push((*node, x, column_y[column]));
        column_y[column] += items[*node].size.1 + SOURCE_STACK_GAP;
        height = height.max(column_y[column] - SOURCE_STACK_GAP);
    }
    let width = column_w.iter().sum::<f32>() + BLOCK_COLUMN_GAP * (columns - 1) as f32;
    (offsets, (width, height))
}

/// Places one lane starting at `y`. Returns its bounds (min x, min y, max x, max y).
/// A member is an executing node (or `usize::MAX` for the unattached lane) with the
/// data nodes beside it.
fn place_lane(
    members: &[Member],
    items: &[Item],
    positions: &mut [[f32; 2]],
    y: f32,
) -> (f32, f32, f32, f32) {
    let mut bounds = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    let mut grow = |x: f32, y: f32, size: (f32, f32)| {
        bounds.0 = bounds.0.min(x);
        bounds.1 = bounds.1.min(y);
        bounds.2 = bounds.2.max(x + size.0);
        bounds.3 = bounds.3.max(y + size.1);
    };

    let (mut x, mut row_y, mut row_h) = (0.0_f32, y, 0.0_f32);
    for (node, reads, sets) in members {
        let (read_offsets, read_block) = block_layout(reads, items);
        let (set_offsets, set_block) = block_layout(sets, items);
        let node_size = if *node == usize::MAX {
            (0.0, 0.0)
        } else {
            items[*node].size
        };
        let left = if reads.is_empty() {
            0.0
        } else {
            read_block.0 + SOURCE_GAP
        };
        let right = if sets.is_empty() {
            0.0
        } else {
            SOURCE_GAP + set_block.0
        };
        let unit = left + node_size.0 + right;

        if x > 0.0 && x + unit > MAX_ROW_WIDTH {
            row_y += row_h + ROW_GAP;
            x = 0.0;
            row_h = 0.0;
        }

        for (source, dx, dy) in read_offsets {
            positions[source] = [x + dx, row_y + dy];
            grow(x + dx, row_y + dy, items[source].size);
        }
        if *node != usize::MAX {
            positions[*node] = [x + left, row_y];
            grow(x + left, row_y, node_size);
        }
        for (sink, dx, dy) in set_offsets {
            let sink_x = x + left + node_size.0 + SOURCE_GAP + dx;
            positions[sink] = [sink_x, row_y + dy];
            grow(sink_x, row_y + dy, items[sink].size);
        }
        row_h = row_h.max(node_size.1).max(read_block.1).max(set_block.1);
        x += unit + NODE_GAP;
    }
    bounds
}

/// A comment box around a lane. Its `nodes` hold node *indices* as text until the
/// caller maps them to ids.
fn lane_comment(
    lane: usize,
    key: &str,
    members: &[Member],
    bounds: (f32, f32, f32, f32),
) -> GraphComment {
    const PALETTE: [[f32; 4]; 6] = [
        [0.20, 0.45, 0.80, 0.16],
        [0.80, 0.50, 0.15, 0.16],
        [0.30, 0.70, 0.40, 0.16],
        [0.70, 0.35, 0.75, 0.16],
        [0.80, 0.30, 0.30, 0.16],
        [0.45, 0.65, 0.75, 0.16],
    ];
    let title = key.strip_prefix("Passes: ").unwrap_or(key);
    let title = if title.is_empty() || title == "Graph" {
        "Frame"
    } else {
        title
    };
    let mut nodes: Vec<String> = Vec::new();
    for (node, reads, sets) in members {
        if *node != usize::MAX {
            nodes.push(node.to_string());
        }
        nodes.extend(reads.iter().chain(sets).map(ToString::to_string));
    }
    GraphComment {
        id: format!("{LANE_PREFIX}{lane}"),
        text: title.to_string(),
        position: [
            bounds.0 - COMMENT_PAD,
            bounds.1 - COMMENT_PAD - COMMENT_TITLE,
        ],
        size: [
            bounds.2 - bounds.0 + COMMENT_PAD * 2.0,
            bounds.3 - bounds.1 + COMMENT_PAD * 2.0 + COMMENT_TITLE,
        ],
        color: PALETTE[lane % PALETTE.len()],
        nodes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::{Connection, NodeInstance, PinRef};
    use crate::pins::{PinDesc, PinType};
    use crate::registry::NodeDescriptor;

    fn node(id: &str, kind: &str) -> NodeInstance {
        NodeInstance {
            id: id.into(),
            kind: kind.into(),
            params: Default::default(),
            position: [0.0; 2],
        }
    }

    fn wire(a: &str, ap: &str, b: &str, bp: &str) -> Connection {
        Connection {
            from: PinRef::new(a, ap),
            to: PinRef::new(b, bp),
        }
    }

    fn registry() -> NodeRegistry {
        let mut registry = NodeRegistry::with_core();
        for (kind, category) in [("t.a", "Passes: Alpha"), ("t.b", "Passes: Beta")] {
            registry.register(
                NodeDescriptor::new(kind, kind, category, "", |_| Ok(())).with_pins(
                    vec![
                        PinDesc::flow("in"),
                        PinDesc::resource(PinType::TEXTURE, "x"),
                    ],
                    vec![
                        PinDesc::flow("out"),
                        PinDesc::resource(PinType::TEXTURE, "y"),
                    ],
                ),
            );
        }
        registry.register(
            NodeDescriptor::new("t.src", "Source", "Data", "", |_| Ok(()))
                .with_pins(Vec::new(), vec![PinDesc::resource(PinType::TEXTURE, "x")])
                .with_data_only(),
        );
        registry
    }

    fn sample() -> GraphAsset {
        let mut asset = GraphAsset::default();
        for (id, kind) in [
            ("start", ON_FRAME_START),
            ("a", "t.a"),
            ("src_a", "t.src"),
            ("b", "t.b"),
            ("src_b1", "t.src"),
            ("src_b2", "t.src"),
            ("out", OUTPUT),
            ("loose", "t.src"),
        ] {
            asset.nodes.push(node(id, kind));
        }
        asset.connections = vec![
            wire("start", "out", "a", "in"),
            wire("a", "out", "b", "in"),
            wire("b", "out", "out", "in"),
            wire("src_a", "x", "a", "x"),
            wire("src_b1", "x", "b", "x"),
            wire("src_b2", "x", "b", "x"),
        ];
        asset
    }

    fn rect(asset: &GraphAsset, registry: &NodeRegistry, id: &str) -> (f32, f32, f32, f32) {
        let node = asset.node(id).unwrap();
        let descriptor = registry.get(&node.kind).unwrap();
        let (w, h) = estimated_size(&descriptor.title, &descriptor.pins(&node.params));
        (
            node.position[0],
            node.position[1],
            node.position[0] + w,
            node.position[1] + h,
        )
    }

    fn overlap(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> bool {
        a.0 < b.2 && b.0 < a.2 && a.1 < b.3 && b.1 < a.3
    }

    #[test]
    fn no_two_nodes_overlap() {
        let registry = registry();
        let mut asset = sample();
        auto_layout(&mut asset, &registry);
        let ids: Vec<_> = asset.nodes.iter().map(|n| n.id.clone()).collect();
        for (i, a) in ids.iter().enumerate() {
            for b in &ids[i + 1..] {
                assert!(
                    !overlap(rect(&asset, &registry, a), rect(&asset, &registry, b)),
                    "{a} and {b} overlap"
                );
            }
        }
    }

    #[test]
    fn execution_runs_left_to_right_and_sources_sit_beside_their_user() {
        let registry = registry();
        let mut asset = sample();
        auto_layout(&mut asset, &registry);
        let x = |id: &str| asset.node(id).unwrap().position[0];
        assert!(x("start") < x("a") && x("a") < x("b") && x("b") < x("out"));
        // A source is left of the node it feeds, and in the same lane row.
        assert!(x("src_a") < x("a"));
        assert!(x("src_b1") < x("b") && x("src_b2") < x("b"));
        assert_eq!(x("src_b1"), x("src_b2"), "one column");
        assert_ne!(
            asset.node("src_b1").unwrap().position[1],
            asset.node("src_b2").unwrap().position[1],
            "stacked"
        );
    }

    #[test]
    fn lanes_follow_categories_and_have_a_comment_around_them() {
        let registry = registry();
        let mut asset = sample();
        auto_layout(&mut asset, &registry);
        let titles: Vec<_> = asset.comments.iter().map(|c| c.text.as_str()).collect();
        // The start joins Alpha, the output joins Beta, and the unfed source is unattached.
        assert_eq!(titles, vec!["Alpha", "Beta", "Unattached"]);
        let alpha = &asset.comments[0];
        assert!(
            alpha.nodes.contains(&"start".to_string()) && alpha.nodes.contains(&"a".to_string())
        );
        let beta = &asset.comments[1];
        assert!(beta.nodes.contains(&"out".to_string()));
        // A comment surrounds the nodes it names.
        for comment in &asset.comments {
            for id in &comment.nodes {
                let r = rect(&asset, &registry, id);
                assert!(r.0 >= comment.position[0] && r.2 <= comment.position[0] + comment.size[0]);
                assert!(r.1 >= comment.position[1] && r.3 <= comment.position[1] + comment.size[1]);
            }
        }
    }

    #[test]
    fn running_it_again_replaces_its_own_comments_and_keeps_the_authors() {
        let registry = registry();
        let mut asset = sample();
        asset.comments.push(GraphComment {
            id: "mine".into(),
            text: "keep me".into(),
            position: [0.0; 2],
            size: [10.0; 2],
            color: [1.0; 4],
            nodes: Vec::new(),
        });
        auto_layout(&mut asset, &registry);
        let first = asset.comments.len();
        auto_layout(&mut asset, &registry);
        assert_eq!(asset.comments.len(), first);
        assert!(asset.comments.iter().any(|c| c.id == "mine"));
    }

    #[test]
    fn wide_lanes_wrap() {
        let registry = registry();
        let mut asset = GraphAsset::default();
        asset.nodes.push(node("start", ON_FRAME_START));
        let mut previous = "start".to_string();
        for i in 0..30 {
            let id = format!("a{i}");
            asset.nodes.push(node(&id, "t.a"));
            asset.connections.push(wire(&previous, "out", &id, "in"));
            previous = id;
        }
        asset.nodes.push(node("out", OUTPUT));
        asset.connections.push(wire(&previous, "out", "out", "in"));
        auto_layout(&mut asset, &registry);
        let rows: HashSet<i64> = asset.nodes.iter().map(|n| n.position[1] as i64).collect();
        assert!(
            rows.len() > 1,
            "a 30-node lane should wrap onto more than one row"
        );
    }

    #[test]
    fn a_node_that_only_receives_data_goes_right_of_its_producer() {
        let mut registry = registry();
        registry.register(
            NodeDescriptor::new("t.sink", "Sink", "Data", "", |_| Ok(()))
                .with_pins(vec![PinDesc::resource(PinType::TEXTURE, "y")], Vec::new())
                .with_data_only(),
        );
        let mut asset = sample();
        asset.nodes.push(node("sink", "t.sink"));
        asset.connections.push(wire("a", "y", "sink", "y"));
        auto_layout(&mut asset, &registry);
        let x = |id: &str| asset.node(id).unwrap().position[0];
        assert!(
            x("sink") > x("a"),
            "a Set sits after the node that produces it"
        );
        assert!(x("sink") < x("b"), "and before the next node");
        for other in ["a", "b", "start", "out", "src_a", "src_b1"] {
            assert!(
                !overlap(
                    rect(&asset, &registry, "sink"),
                    rect(&asset, &registry, other)
                ),
                "sink overlaps {other}"
            );
        }
    }

    #[test]
    fn many_sources_wrap_into_columns_instead_of_one_tall_stack() {
        let registry = registry();
        let mut asset = GraphAsset::default();
        asset.nodes.push(node("start", ON_FRAME_START));
        asset.nodes.push(node("a", "t.a"));
        asset.nodes.push(node("out", OUTPUT));
        asset.connections.push(wire("start", "out", "a", "in"));
        asset.connections.push(wire("a", "out", "out", "in"));
        for i in 0..20 {
            let id = format!("s{i}");
            asset.nodes.push(node(&id, "t.src"));
            asset.connections.push(wire(&id, "x", "a", "x"));
        }
        auto_layout(&mut asset, &registry);
        let columns: HashSet<i64> = (0..20)
            .map(|i| asset.node(&format!("s{i}")).unwrap().position[0] as i64)
            .collect();
        assert!(
            columns.len() > 1,
            "20 sources should not form a single column"
        );
        let tallest = (0..20)
            .map(|i| asset.node(&format!("s{i}")).unwrap().position[1] as i64)
            .max()
            .unwrap();
        assert!(tallest < 20 * 60, "stack height is bounded by the grid");
    }
}
