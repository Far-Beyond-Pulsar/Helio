pub mod pass;

use std::collections::HashMap;
use pass::{RenderPass, PassContext};

/// A registered pass node.
struct PassNode {
    label:      String,
    pass:       Box<dyn RenderPass>,
    depends_on: Vec<usize>,  // indices into nodes[]
}

/// Simple render graph: registers passes, topological-sorts on first execute.
/// Pass ordering is fixed after the first frame — this is intentional to avoid
/// re-sorting overhead.
pub struct RenderGraph {
    nodes:  Vec<PassNode>,
    order:  Vec<usize>,        // cached topo order, populated on first execute
    sorted: bool,
}

impl RenderGraph {
    pub fn new() -> Self {
        RenderGraph { nodes: Vec::new(), order: Vec::new(), sorted: false }
    }

    /// Register a pass. Returns its index for adding dependencies.
    pub fn add_pass(&mut self, label: &str, pass: impl RenderPass + 'static) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(PassNode {
            label:      label.to_owned(),
            pass:       Box::new(pass),
            depends_on: Vec::new(),
        });
        self.sorted = false;
        idx
    }

    /// Declare that `from` must run before `to`.
    pub fn add_dependency(&mut self, from: usize, to: usize) {
        self.nodes[to].depends_on.push(from);
        self.sorted = false;
    }

    /// Execute all registered passes in topological order.
    pub fn execute(&mut self, ctx: &mut PassContext) {
        if !self.sorted {
            self.toposort();
            self.sorted = true;
        }

        for &idx in &self.order {
            let label = self.nodes[idx].label.clone();
            ctx.profiler_begin_scope_for(&label);
            self.nodes[idx].pass.execute(ctx);
            ctx.profiler_end_scope_for(&label);
        }
    }

    /// Kahn's BFS topological sort.
    fn toposort(&mut self) {
        let n = self.nodes.len();
        let mut in_degree = vec![0usize; n];
        let mut adj = vec![vec![]; n];

        for (i, node) in self.nodes.iter().enumerate() {
            for &dep in &node.depends_on {
                adj[dep].push(i);
                in_degree[i] += 1;
            }
        }

        let mut queue: std::collections::VecDeque<usize> =
            (0..n).filter(|&i| in_degree[i] == 0).collect();

        let mut order = Vec::with_capacity(n);
        while let Some(u) = queue.pop_front() {
            order.push(u);
            for &v in &adj[u] {
                in_degree[v] -= 1;
                if in_degree[v] == 0 { queue.push_back(v); }
            }
        }

        assert_eq!(order.len(), n, "Render graph has a cycle!");
        self.order = order;
    }
}
