//! Render graph executor with automatic profiling.
//!
//! The render graph is the top-level orchestrator of the rendering pipeline. It manages
//! pass execution order, automatic profiling, and resource lifetime management.
//!
//! # Design Pattern: Graph Execution
//!
//! Helio v3 uses a **linear graph executor** (future: DAG with parallelism):
//!
//! 1. **Add passes**: `graph.add_pass(Box::new(ShadowPass::new(...)))`
//! 2. **Execute in order**: Passes run sequentially in the order they were added
//! 3. **Automatic profiling**: CPU scopes and GPU timestamps injected per-pass
//! 4. **Zero-copy contexts**: Each pass receives a `PassContext` with borrowed references
//!
//! # Components
//!
//! - [`RenderGraph`] - Graph executor with automatic profiling
//! - [`ResourceBuilder`] - Resource dependency declaration (future feature)
//! - [`ResourceHandle`] - Resource lifetime management (future feature)
//!
//! # Performance
//!
//! - **O(passes)**: Linear execution (sequential, not parallel yet)
//! - **Zero allocations**: Passes and profiler are pre-allocated
//! - **Zero clones**: PassContext uses borrowed references
//!
//! # Example
//!
//! ```rust,no_run
//! use helio_v3::{RenderGraph, GpuScene};
//! use std::sync::Arc;
//!
//! let mut graph = RenderGraph::new(&device, &queue);
//! let scene = GpuScene::new(Arc::new(device), Arc::new(queue));
//!
//! // Add passes (order matters)
//! // graph.add_pass(Box::new(ShadowPass::new(&device)));
//! // graph.add_pass(Box::new(GBufferPass::new(&device)));
//! // graph.add_pass(Box::new(DeferredLightPass::new(&device)));
//!
//! // Render loop
//! // loop {
//! //     scene.flush();
//! //     graph.execute(&scene, &target_view, &depth_view).unwrap();
//! // }
//! ```

mod executor;
mod resource;

pub use executor::RenderGraph;
pub use resource::{
    ResourceAccess, ResourceBuilder, ResourceDecl, ResourceFormat, ResourceHandle, ResourceSize,
};

