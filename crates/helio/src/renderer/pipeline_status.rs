//! What a renderer knows about the pipeline it was given: none, one that does
//! not build, or one that renders.
//!
//! A renderer does not need a pipeline to exist. With none, or with one that
//! failed, it draws a status screen instead of the scene (see
//! `status_screen`), so the reason is on the output frame as well as in the
//! log. There is no substitute pipeline: what was asked for either renders or
//! is reported.

use std::fmt;

use super::pipeline::IntoGraph;
use super::renderer_impl::GraphRebuilder;

/// Id a pipeline carries when its host did not name it
/// ([`RendererBuilder::with_graph`](super::RendererBuilder::with_graph)).
pub const UNNAMED_PIPELINE: &str = "(unnamed pipeline)";

/// Why a pipeline did not build: the reason, and what a person needs to fix
/// it (validation diagnostics, a missing node and the plugin that is
/// disabled, routing findings, the build error text).
pub struct PipelineFailure {
    /// One line: what went wrong.
    pub reason: String,
    /// The lines behind the reason, in the order they were found.
    pub details: Vec<String>,
    /// How to try again (a graph that did not compile can be rebuilt once a
    /// shader or plugin changes). The renderer keeps it for
    /// [`Renderer::request_graph_rebuild`](super::Renderer::request_graph_rebuild)
    /// and shader reload.
    pub rebuilder: Option<GraphRebuilder>,
}

impl PipelineFailure {
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
            details: Vec::new(),
            rebuilder: None,
        }
    }

    pub fn with_details(mut self, details: impl IntoIterator<Item = String>) -> Self {
        self.details.extend(details);
        self
    }

    pub fn with_rebuilder(mut self, rebuilder: GraphRebuilder) -> Self {
        self.rebuilder = Some(rebuilder);
        self
    }
}

impl fmt::Debug for PipelineFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PipelineFailure")
            .field("reason", &self.reason)
            .field("details", &self.details)
            .field("rebuildable", &self.rebuilder.is_some())
            .finish()
    }
}

impl fmt::Display for PipelineFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.reason)?;
        for detail in &self.details {
            write!(f, "; {detail}")?;
        }
        Ok(())
    }
}

/// A pipeline and the id the host knows it by (the id the status screen and
/// the log name when it fails).
pub struct NamedPipeline {
    pub id: String,
    pub pipeline: Box<dyn IntoGraph>,
}

impl NamedPipeline {
    pub fn new(id: impl Into<String>, pipeline: impl IntoGraph + 'static) -> Self {
        Self {
            id: id.into(),
            pipeline: Box::new(pipeline),
        }
    }

    /// [`new`](Self::new) for an already boxed pipeline (what a
    /// [`RenderPipelineRegistry`](super::RenderPipelineRegistry) makes).
    pub fn boxed(id: impl Into<String>, pipeline: Box<dyn IntoGraph>) -> Self {
        Self {
            id: id.into(),
            pipeline,
        }
    }
}

/// The state of a renderer's pipeline, readable by its host
/// ([`Renderer::pipeline_status`](super::Renderer::pipeline_status)).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum PipelineStatus {
    /// No pipeline is assigned; the output shows the status screen.
    #[default]
    None,
    /// The assigned pipeline did not build; the output shows why.
    Invalid {
        id: String,
        reason: String,
        details: Vec<String>,
    },
    /// The pipeline's graph renders.
    Ready,
}

impl PipelineStatus {
    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready)
    }
}

/// What the renderer does at its next frame boundary
/// ([`Renderer::set_pipeline`](super::Renderer::set_pipeline)).
pub(crate) enum PendingPipeline {
    Assign(NamedPipeline),
    Clear,
    Fail {
        id: String,
        failure: PipelineFailure,
    },
}
