//! Assigning, swapping and clearing a renderer's pipeline.
//!
//! A renderer exists with no pipeline. The host assigns one
//! ([`Renderer::set_pipeline`]) and the change happens at the next frame
//! boundary: the old graph is dropped (its GPU resources with it) and the new
//! one built, or, when it does not build, the status screen takes its place with
//! the reason. Nothing else is ever drawn in a pipeline's stead.

use helio_core::{RenderGraph, RenderSize};

use super::builder::PassBuildContext;
use super::pipeline_status::{
    NamedPipeline, PendingPipeline, PipelineFailure, PipelineStatus, UNNAMED_PIPELINE,
};
use super::renderer_impl::{GraphRebuilder, Renderer};
use super::status_screen;

impl Renderer {
    /// Assigns `pipeline`, or none with `None`, at the next frame boundary
    /// (the start of the next [`render`](Self::render), or
    /// [`apply_pending_pipeline`](Self::apply_pending_pipeline)). A later call
    /// before then replaces this one.
    ///
    /// The previous graph is dropped, and with it everything it owned, except
    /// that a pipeline with the id the renderer already renders keeps its
    /// persistent pass state (it is the same pipeline built again).
    pub fn set_pipeline(&mut self, pipeline: Option<NamedPipeline>) {
        self.pending_pipeline = Some(match pipeline {
            Some(pipeline) => PendingPipeline::Assign(pipeline),
            None => PendingPipeline::Clear,
        });
    }

    /// Reports that the pipeline `id` could not be made at all (the host found
    /// no such pipeline, its asset did not load), at the next frame boundary:
    /// the renderer shows `failure` as it does for a pipeline that does not build.
    pub fn fail_pipeline(&mut self, id: impl Into<String>, failure: PipelineFailure) {
        self.pending_pipeline = Some(PendingPipeline::Fail {
            id: id.into(),
            failure,
        });
    }

    /// Whether the renderer has a pipeline that renders, none, or one that did
    /// not build (and why). Reflects a [`set_pipeline`](Self::set_pipeline)
    /// from the next frame boundary on.
    pub fn pipeline_status(&self) -> &PipelineStatus {
        &self.pipeline_status
    }

    /// The id of the pipeline the renderer was last given, whether it built or
    /// not; `None` with no pipeline.
    pub fn pipeline_id(&self) -> Option<&str> {
        self.pipeline_id.as_deref()
    }

    /// What the status screen says about assigning a pipeline, in the host's
    /// words (where its pipeline picker is). Helio's own text names its API.
    pub fn set_status_hint(&mut self, hint: impl Into<String>) {
        let hint = hint.into();
        if self.status_hint == hint {
            return;
        }
        self.status_hint = hint;
        // A screen being shown says the old hint: draw it again.
        if !self.pipeline_status.is_ready() {
            let status = self.pipeline_status.clone();
            let rebuilder = self.graph_rebuilder.clone();
            self.show_status(status, rebuilder);
        }
    }

    /// Makes the pipeline change [`set_pipeline`](Self::set_pipeline) asked
    /// for, if one is waiting. [`render`](Self::render) does it at the start
    /// of every frame; a host that does not render while hidden calls it to
    /// have [`pipeline_status`](Self::pipeline_status) current.
    pub fn apply_pending_pipeline(&mut self) {
        let Some(pending) = self.pending_pipeline.take() else {
            return;
        };
        helio_core::cpu_scope!("Helio: apply pipeline");
        // The change builds from the renderer's configuration as it is now.
        self.pending_rebuild = false;
        match pending {
            PendingPipeline::Clear => {
                self.pipeline_id = None;
                self.show_status(PipelineStatus::None, None);
            }
            PendingPipeline::Fail { id, failure } => self.show_invalid(id, failure),
            PendingPipeline::Assign(NamedPipeline { id, pipeline }) => {
                let same = self.pipeline_status.is_ready()
                    && self.pipeline_id.as_deref() == Some(id.as_str());
                let config = self.renderer_config();
                let built = self.guarded_build(|renderer| {
                    Some(pipeline.into_graph(PassBuildContext {
                        device: &renderer.device,
                        queue: &renderer.queue,
                        config,
                        debug_state: renderer.debug_state.clone(),
                        camera_buffer: &renderer.camera_buffer,
                        debug_camera_buffer: &renderer.debug_camera_buffer,
                        cull_stats_buffer: &renderer.cull_stats_buffer,
                        owns_device: renderer.owns_device,
                        scene_db: renderer.scene_db.clone(),
                    }))
                });
                match built {
                    Some(Ok(graph)) => {
                        self.install_pipeline_graph(graph, same);
                        self.pipeline_id = Some(id);
                        self.pipeline_status = PipelineStatus::Ready;
                    }
                    Some(Err(failure)) => self.show_invalid(id, failure),
                    None => unreachable!("the closure always builds"),
                }
            }
        }
    }

    /// The current pipeline failed to rebuild: its status is invalid.
    pub(crate) fn fail_current_pipeline(&mut self, mut failure: PipelineFailure) {
        // The rebuilder that failed is the way to try again, unless the failure
        // brings a better one.
        if failure.rebuilder.is_none() {
            failure.rebuilder = self.graph_rebuilder.clone();
        }
        let id = self
            .pipeline_id
            .clone()
            .unwrap_or_else(|| UNNAMED_PIPELINE.to_owned());
        self.show_invalid(id, failure);
    }

    /// Runs `build` (a pipeline's, or a rebuilder's) with what can go wrong in
    /// it made into the pipeline's failure: a panic in a pass constructor, and
    /// a GPU validation error (a shader the GPU rejects). The host's frame
    /// thread survives a pipeline that cannot be built.
    pub(crate) fn guarded_build(
        &self,
        build: impl FnOnce(&Self) -> Option<Result<RenderGraph, PipelineFailure>>,
    ) -> Option<Result<RenderGraph, PipelineFailure>> {
        // Pass constructors create pipelines eagerly, so a validation error is
        // raised here and not at the first frame.
        #[cfg(not(target_arch = "wasm32"))]
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| build(self)));
        #[cfg(not(target_arch = "wasm32"))]
        let gpu_error = pollster::block_on(scope.pop());
        #[cfg(target_arch = "wasm32")]
        let gpu_error: Option<wgpu::Error> = None;

        match built {
            Err(payload) => {
                let message = payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_owned())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown panic".to_owned());
                Some(Err(PipelineFailure::new(
                    "the pipeline panicked while building",
                )
                .with_details([message])))
            }
            Ok(Some(Ok(mut graph))) => match gpu_error {
                None => Some(Ok(graph)),
                Some(error) => {
                    let mut failure = PipelineFailure::new(
                        "the GPU rejected what the pipeline built (an invalid shader or pipeline state)",
                    )
                    .with_details([error.to_string()]);
                    failure.rebuilder = graph.take_graph_data::<GraphRebuilder>();
                    Some(Err(failure))
                }
            },
            Ok(other) => other,
        }
    }

    /// Shows `failure` for the pipeline `id`, and logs it when it is news.
    fn show_invalid(&mut self, id: String, failure: PipelineFailure) {
        let PipelineFailure {
            reason,
            details,
            rebuilder,
        } = failure;
        self.pipeline_id = Some(id.clone());
        self.show_status(
            PipelineStatus::Invalid {
                id,
                reason,
                details,
            },
            rebuilder,
        );
    }

    /// Replaces the graph with the status screen for `status` (not `Ready`),
    /// keeping `rebuilder` so the pipeline can be tried again. Logs a change.
    fn show_status(&mut self, status: PipelineStatus, rebuilder: Option<GraphRebuilder>) {
        let Some(text) = status_screen::text_for(&status, &self.status_hint) else {
            return;
        };
        if self.pipeline_status != status {
            match &status {
                PipelineStatus::None => {
                    log::warn!(target: "helio::pipeline", "no render pipeline assigned: drawing the status screen");
                }
                PipelineStatus::Invalid {
                    id,
                    reason,
                    details,
                } => {
                    log::error!(target: "helio::pipeline", "render pipeline `{id}` is invalid: {reason}");
                    for detail in details {
                        log::error!(target: "helio::pipeline", "  {detail}");
                    }
                }
                PipelineStatus::Ready => {}
            }
        }
        let internal = [
            (((self.output_width as f32) * self.render_scale).ceil() as u32).max(1),
            (((self.output_height as f32) * self.render_scale).ceil() as u32).max(1),
        ];
        let graph = status_screen::graph(
            &self.device,
            &self.queue,
            self.owns_device,
            self.surface_format,
            RenderSize::new(internal, [self.output_width, self.output_height]),
            text,
        );
        // The old graph, the pipeline's persistent state and host passes with
        // it, is dropped here.
        self.graph = graph;
        self.graph_rebuilder = rebuilder;
        self.enable_jitter = false;
        self.clear_target_next_frame = true;
        self.pipeline_status = status;
    }
}
