//! Graph builder

use super::RenderGraph;

/// Builder for render graph
pub struct GraphBuilder {
    // TODO: Add builder state
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self {}
    }

    pub fn build(self) -> RenderGraph {
        RenderGraph::new()
    }
}
