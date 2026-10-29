//! What a renderer is built from: anything that turns a [`PassBuildContext`]
//! into a [`RenderGraph`].
//!
//! [`IntoGraph`] is the one extension seam for render pipelines. Helio's
//! built-in pipelines, a visual render-graph asset and a hand-written
//! `fn(PassBuildContext) -> RenderGraph` all implement it, and
//! [`RendererBuilder::with_graph`](super::RendererBuilder::with_graph) accepts
//! any of them. [`RenderPipelineRegistry`] names them so a host can offer the
//! set as a choice (the level editor's World Settings pipeline picker).
//!
//! A graph is rebuilt when graph-level configuration changes and on shader
//! reload. An implementation owns that: it installs a
//! [`GraphRebuilder`](super::GraphRebuilder) with
//! `RenderGraph::set_graph_data`, as the built-in builders do.

use std::collections::BTreeMap;
use std::sync::Arc;

use helio_core::RenderGraph;

use super::PassBuildContext;

/// A render pipeline that can produce its [`RenderGraph`].
///
/// Consumed once per construction; a pipeline that must build again after a
/// graph-level configuration change keeps what it needs in the rebuilder it
/// installs on the graph.
pub trait IntoGraph {
    fn into_graph(self: Box<Self>, ctx: PassBuildContext<'_>) -> RenderGraph;
}

impl<F> IntoGraph for F
where
    F: for<'a> FnOnce(PassBuildContext<'a>) -> RenderGraph,
{
    fn into_graph(self: Box<Self>, ctx: PassBuildContext<'_>) -> RenderGraph {
        (*self)(ctx)
    }
}

/// Where a registered pipeline came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PipelineOrigin {
    /// Ships with Helio.
    BuiltIn,
    /// Authored in a project, e.g. a render-graph asset.
    Project,
}

/// A pipeline the host can offer for selection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineInfo {
    /// Stable identifier persisted in level settings: `fxaa`, or a
    /// project-relative asset path.
    pub id: String,
    pub label: String,
    pub description: String,
    pub origin: PipelineOrigin,
}

/// Makes a fresh pipeline each time one is selected or the renderer restarts.
pub type PipelineFactory = Arc<dyn Fn() -> Box<dyn IntoGraph> + Send + Sync>;

struct Entry {
    info: PipelineInfo,
    factory: PipelineFactory,
}

/// Named [`IntoGraph`] factories. Ordered by id so listings are stable.
#[derive(Default, Clone)]
pub struct RenderPipelineRegistry {
    entries: BTreeMap<String, Arc<Entry>>,
    default_id: Option<String>,
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

    /// Removes every pipeline of `origin` (a project reload drops its assets).
    pub fn clear_origin(&mut self, origin: PipelineOrigin) {
        self.entries.retain(|_, entry| entry.info.origin != origin);
    }

    /// The pipeline used when a level names none, or one that no longer exists.
    pub fn set_default(&mut self, id: impl Into<String>) {
        self.default_id = Some(id.into());
    }

    pub fn default_id(&self) -> Option<&str> {
        self.default_id.as_deref()
    }

    pub fn contains(&self, id: &str) -> bool {
        self.entries.contains_key(id)
    }

    pub fn info(&self, id: &str) -> Option<&PipelineInfo> {
        self.entries.get(id).map(|entry| &entry.info)
    }

    /// Built-ins first, then project pipelines, each ordered by id.
    pub fn list(&self) -> Vec<PipelineInfo> {
        let mut infos: Vec<_> = self.entries.values().map(|e| e.info.clone()).collect();
        infos.sort_by(|a, b| (a.origin, &a.id).cmp(&(b.origin, &b.id)));
        infos
    }

    /// A new pipeline for `id`, or the default when `id` is `None` or unknown.
    pub fn create(&self, id: Option<&str>) -> Option<Box<dyn IntoGraph>> {
        let entry = id.and_then(|id| self.entries.get(id)).or_else(|| {
            self.default_id
                .as_deref()
                .and_then(|id| self.entries.get(id))
        })?;
        Some((entry.factory)())
    }

    /// The id [`create`](Self::create) resolves `id` to.
    pub fn resolve<'a>(&'a self, id: Option<&'a str>) -> Option<&'a str> {
        id.filter(|id| self.entries.contains_key(*id))
            .or(self.default_id.as_deref())
    }
}
