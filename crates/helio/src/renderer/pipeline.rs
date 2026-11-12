//! What a renderer is built from: anything that turns a [`PassBuildContext`]
//! into a [`RenderGraph`], or says why it cannot.
//!
//! [`IntoGraph`] is the one extension seam for render pipelines. Helio's
//! built-in pipelines, a visual render-graph asset and a hand-written
//! `fn(PassBuildContext) -> RenderGraph` all implement it, and
//! [`RendererBuilder::with_graph`](super::RendererBuilder::with_graph) and
//! [`Renderer::set_pipeline`](super::Renderer::set_pipeline) accept any of
//! them. [`RenderPipelineRegistry`] names them so a host can offer the set as
//! a choice (the level editor's World Settings pipeline picker).
//!
//! A pipeline that cannot build returns a [`PipelineFailure`]; the renderer
//! keeps it as [`PipelineStatus::Invalid`](super::PipelineStatus) and draws the
//! reason on the output. Nothing is substituted for it.
//!
//! A graph is rebuilt when graph-level configuration changes and on shader
//! reload. An implementation owns that: it installs a
//! [`GraphRebuilder`](super::GraphRebuilder) with
//! `RenderGraph::set_graph_data`, as the built-in builders do (or hands one to
//! its [`PipelineFailure`] so a failed build can be retried).

use std::collections::BTreeMap;
use std::sync::Arc;

use helio_core::RenderGraph;

use super::{PassBuildContext, PipelineFailure};

/// A render pipeline that can produce its [`RenderGraph`].
///
/// Consumed once per construction; a pipeline that must build again after a
/// graph-level configuration change keeps what it needs in the rebuilder it
/// installs on the graph.
pub trait IntoGraph {
    fn into_graph(
        self: Box<Self>,
        ctx: PassBuildContext<'_>,
    ) -> Result<RenderGraph, PipelineFailure>;
}

/// A hand-written builder always produces its graph (one that cannot panics,
/// and the renderer reports the panic as the pipeline's failure).
impl<F> IntoGraph for F
where
    F: for<'a> FnOnce(PassBuildContext<'a>) -> RenderGraph,
{
    fn into_graph(
        self: Box<Self>,
        ctx: PassBuildContext<'_>,
    ) -> Result<RenderGraph, PipelineFailure> {
        Ok((*self)(ctx))
    }
}

/// A pipeline the host can offer for selection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineInfo {
    /// Stable identifier persisted in level settings, e.g. an asset path.
    pub id: String,
    pub label: String,
    pub description: String,
}

/// Makes a fresh pipeline each time one is selected or the renderer restarts.
pub type PipelineFactory = Arc<dyn Fn() -> Box<dyn IntoGraph> + Send + Sync>;

struct Entry {
    info: PipelineInfo,
    factory: PipelineFactory,
}

/// Named [`IntoGraph`] factories, for a host that offers pipelines by id. It
/// has no default: an id that is not registered makes nothing (the renderer
/// shows why on its status screen). Ordered by id so listings are stable.
#[derive(Default, Clone)]
pub struct RenderPipelineRegistry {
    entries: BTreeMap<String, Arc<Entry>>,
}

impl RenderPipelineRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds or replaces the pipeline named `info.id`.
    pub fn register(&mut self, info: PipelineInfo, factory: PipelineFactory) {
        self.entries
            .insert(info.id.clone(), Arc::new(Entry { info, factory }));
    }

    /// Keeps the pipelines `keep` approves of.
    pub fn retain(&mut self, mut keep: impl FnMut(&PipelineInfo) -> bool) {
        self.entries.retain(|_, entry| keep(&entry.info));
    }

    /// Removes the pipeline named `id`, if registered.
    pub fn remove(&mut self, id: &str) {
        self.entries.remove(id);
    }

    pub fn contains(&self, id: &str) -> bool {
        self.entries.contains_key(id)
    }

    pub fn info(&self, id: &str) -> Option<&PipelineInfo> {
        self.entries.get(id).map(|entry| &entry.info)
    }

    /// Every registered pipeline, ordered by id.
    pub fn list(&self) -> Vec<PipelineInfo> {
        self.entries.values().map(|e| e.info.clone()).collect()
    }

    /// A new instance of the pipeline `id`; `None` when it is not registered.
    pub fn create(&self, id: &str) -> Option<Box<dyn IntoGraph>> {
        self.entries.get(id).map(|entry| (entry.factory)())
    }
}
