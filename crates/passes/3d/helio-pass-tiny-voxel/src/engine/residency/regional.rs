//! Compose a complete visible cut from ready target regions and immutable
//! fallback payloads. A fallback keeps its original sampling bounds.
use super::{Node, AIR, BRICK, SOLID};

const SLOT_MASK: u32 = 0xffff;
const ANCESTOR_SHIFT: u32 = 16;

pub(super) struct Readiness {
    remaining: Vec<u32>,
    parents: Vec<usize>,
    job_nodes: Vec<usize>,
    pub changed: bool,
}

impl Readiness {
    pub fn new(nodes: &[Node], job_nodes: Vec<usize>) -> Self {
        let mut remaining = vec![0; nodes.len()];
        let mut parents = vec![usize::MAX; nodes.len()];
        for &index in &job_nodes {
            remaining[index] += 1;
        }
        for (index, node) in nodes.iter().enumerate() {
            if node.child & BRICK == 0 {
                for child in node.child as usize..node.child as usize + 8 {
                    parents[child] = index;
                }
            }
        }
        for index in (1..nodes.len()).rev() {
            let parent = parents[index];
            assert_ne!(parent, usize::MAX, "publication tree must be connected");
            remaining[parent] += remaining[index];
        }
        Self {
            remaining,
            parents,
            job_nodes,
            changed: true,
        }
    }

    pub fn generated(&mut self, job: usize) {
        let mut index = self.job_nodes[job];
        loop {
            self.remaining[index] = self.remaining[index].checked_sub(1).unwrap();
            index = self.parents[index];
            if index == usize::MAX {
                break;
            }
        }
        self.changed = true;
    }
}

// Parent payload slots fit 16 bits. Bits 16..20 encode the distance to its
// original sampling level. Bounds are recovered relative to the unchanged
// root, including its nonzero/negative origin; no descriptor buffer is needed.
fn clipped(old: Node, region: Node) -> Node {
    assert_ne!(old.child & BRICK, 0);
    let child = if old.child == AIR || old.child == SOLID {
        old.child
    } else {
        let payload_level = old.level + ((old.child >> ANCESTOR_SHIFT) & 31);
        let delta = payload_level.checked_sub(region.level).unwrap();
        assert!(delta < 28);
        BRICK | (old.child & SLOT_MASK) | (delta << ANCESTOR_SHIFT)
    };
    Node { child, ..region }
}

pub(super) fn compose(old: &[Node], target: &[Node], readiness: &Readiness) -> Vec<Node> {
    assert_eq!((old[0].low, old[0].level), (target[0].low, target[0].level));
    let mut output = vec![target[0]];
    merge(old[0], old, 0, target, readiness, 0, &mut output);
    // Two bounded complete trees bound their union. Unreachable fallback
    // children are removed when sibling references collapse.
    assert!(output.len() <= old.len() + target.len());
    output
}

fn reserve_children(node: Node, output: &mut Vec<Node>) -> usize {
    assert!(node.level > 0);
    let first = output.len();
    let half = 16i32 << node.level;
    output.extend((0..8).map(|octant| Node {
        low: std::array::from_fn(|a| node.low[a] + ((octant >> a) & 1) * half),
        level: node.level - 1,
        child: AIR,
    }));
    first
}

fn copy_tree(node: Node, source: &[Node], index: usize, output: &mut Vec<Node>) {
    output[index] = node;
    if node.child & BRICK == 0 {
        let first = reserve_children(node, output);
        output[index].child = first as u32;
        for octant in 0..8 {
            copy_tree(
                source[node.child as usize + octant],
                source,
                first + octant,
                output,
            );
        }
    }
}

fn merge(
    old: Node,
    old_tree: &[Node],
    next: usize,
    target: &[Node],
    ready: &Readiness,
    index: usize,
    output: &mut Vec<Node>,
) {
    let node = target[next];
    if ready.remaining[next] == 0 {
        copy_tree(node, target, index, output);
        return;
    }
    if node.child & BRICK != 0 {
        if old.child & BRICK == 0 {
            copy_tree(old, old_tree, index, output);
        } else {
            output[index] = clipped(old, node);
        }
        return;
    }
    let first = reserve_children(node, output);
    output[index] = Node {
        child: first as u32,
        ..node
    };
    for octant in 0..8 {
        let previous = if old.child & BRICK == 0 {
            old_tree[old.child as usize + octant]
        } else {
            old
        };
        merge(
            previous,
            old_tree,
            node.child as usize + octant,
            target,
            ready,
            first + octant,
            output,
        );
    }
    // Splitting a fallback must not force rays through a carpet of equivalent
    // leaves. Retain the single old leaf until one child has useful new data.
    if old.child & BRICK != 0
        && (0..8).all(|i| output[first + i] == clipped(old, output[first + i]))
    {
        output[index] = clipped(old, node);
        output.truncate(first);
    }
}

#[cfg(test)]
mod tests;
