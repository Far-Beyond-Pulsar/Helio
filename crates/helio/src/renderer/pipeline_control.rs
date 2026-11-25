//! Assigning, swapping and clearing a renderer's pipeline.
//!
//! A renderer exists with no pipeline. The host assigns one
//! ([`Renderer::set_pipeline`]) and the change happens at the next frame
//! boundary: the new graph is built off the frame thread (a pipeline's passes
//! create their GPU pipelines as they are constructed, hundreds of
//! milliseconds), and the frame that finds it done swaps it in. While it builds
//! the renderer keeps drawing what it drew (a pipeline that renders, or the
//! status screen, which says it is building). When the pipeline does not
//! build, the status screen takes its place with the reason. Nothing else is
//! ever drawn in a pipeline's stead.

use std::sync::{Arc, Condvar, Mutex};

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use helio_core::{RenderGraph, RenderSize};

use super::builder::{PassBuildContext, SceneDbHandle};
use super::config::RendererConfig;
use super::debug::DebugDrawState;
use super::pipeline_status::{
    NamedPipeline, PendingPipeline, PipelineFailure, PipelineStatus, UNNAMED_PIPELINE,
};
use super::renderer_impl::{GraphRebuilder, Renderer};
use super::status_screen;

/// What a pipeline's build reads, owned: the build runs on a worker, which
/// borrows nothing of the renderer.
struct BuildInputs {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    config: RendererConfig,
    debug_state: Arc<Mutex<DebugDrawState>>,
    camera_buffer: wgpu::Buffer,
    debug_camera_buffer: wgpu::Buffer,
    stats: helio_core::FrameStats,
    owns_device: bool,
    scene_db: SceneDbHandle,
}

impl BuildInputs {
    fn context(&self) -> PassBuildContext<'_> {
        PassBuildContext {
            device: &self.device,
            queue: &self.queue,
            config: self.config.clone(),
            debug_state: self.debug_state.clone(),
            camera_buffer: &self.camera_buffer,
            debug_camera_buffer: &self.debug_camera_buffer,
            stats: &self.stats,
            owns_device: self.owns_device,
            scene_db: self.scene_db.clone(),
        }
    }
}

type BuildResult = Result<RenderGraph, PipelineFailure>;

/// Where a build leaves its graph for the frame thread to take.
#[derive(Default)]
struct BuildSlot {
    result: Mutex<Option<BuildResult>>,
    done: Condvar,
}

impl BuildSlot {
    fn set(&self, result: BuildResult) {
        *self.result.lock().unwrap_or_else(|e| e.into_inner()) = Some(result);
        self.done.notify_all();
    }

    /// The result when the build has ended; never waits.
    fn take(&self) -> Option<BuildResult> {
        self.result.lock().unwrap_or_else(|e| e.into_inner()).take()
    }

    /// Blocks until the build ends: for a host with no frame to keep drawing
    /// ([`Renderer::apply_pending_pipeline`]).
    fn wait(&self) -> BuildResult {
        let mut result = self.result.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(result) = result.take() {
                return result;
            }
            result = self.done.wait(result).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// What a build off the frame thread makes.
#[derive(Clone, Copy)]
enum BuildKind {
    /// A pipeline [`Renderer::set_pipeline`] assigned. `inherit`: it already
    /// renders under this id, so the new graph takes its passes' persistent
    /// state.
    Pipeline { inherit: bool },
    /// The renderer's own pipeline again from its changed configuration
    /// ([`Renderer::request_graph_rebuild`], a quality or resolution change).
    Rebuild,
}

/// A pipeline being built off the frame thread.
pub(crate) struct PipelineBuild {
    id: String,
    kind: BuildKind,
    slot: Arc<BuildSlot>,
    started: Instant,
    /// The size the build was made for: a resize while it ran is applied to
    /// its graph when it is installed (a graph resizes in place).
    built_for: (u32, u32, f32),
}

impl Renderer {
    /// Assigns `pipeline`, or none with `None`, at the next frame boundary
    /// (the start of the next [`render`](Self::render), or
    /// [`apply_pending_pipeline`](Self::apply_pending_pipeline)). A later call
    /// before then replaces this one, and one while the previous pipeline is
    /// still building drops that build's graph.
    ///
    /// A pipeline builds off the frame thread; the renderer swaps it in at the
    /// first frame boundary after it is done (it keeps drawing what it drew
    /// until then). The previous graph is then dropped, and with it everything
    /// it owned, except that a pipeline with the id the renderer already renders
    /// keeps its persistent pass state (it is the same pipeline built again).
    pub fn set_pipeline(&mut self, pipeline: Option<NamedPipeline>) {
        self.pending_pipeline = Some(match pipeline {
            Some(pipeline) => PendingPipeline::Assign(pipeline),
            None => PendingPipeline::Clear,
        });
    }

    /// Whether a rebuild from a changed configuration (a quality or resolution
    /// change, [`request_graph_rebuild`](Self::request_graph_rebuild)) holds
    /// the frame until it is built (the default: the frame after the request
    /// draws the rebuilt graph), or builds off the frame thread while the old
    /// graph keeps drawing, like an assigned pipeline. A host that shows a
    /// loading state for pipelines (`SceneRenderer` in background mode) wants
    /// the second.
    pub fn set_blocking_builds(&mut self, blocking: bool) {
        self.blocking_builds = blocking;
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

    /// Whether the renderer has a pipeline that renders, none, one that is
    /// building, or one that did not build (and why). Reflects a
    /// [`set_pipeline`](Self::set_pipeline) from the next frame boundary on.
    pub fn pipeline_status(&self) -> &PipelineStatus {
        &self.pipeline_status
    }

    /// The id of the pipeline being built off the frame thread, if one is. A
    /// pipeline replacing one that renders builds while that one still draws,
    /// so [`pipeline_status`](Self::pipeline_status) says nothing about it
    /// until it is swapped in; a host that follows the outcome reads the
    /// status once this is `None`.
    pub fn pipeline_building(&self) -> Option<&str> {
        self.pipeline_build.as_ref().map(|build| build.id.as_str())
    }

    /// Whether a pipeline change is on its way: asked for and not yet started
    /// ([`set_pipeline`](Self::set_pipeline), taken at the next frame boundary)
    /// or building. A host that reports the outcome of a change waits for this
    /// to be `false`, then reads [`pipeline_status`](Self::pipeline_status).
    pub fn pipeline_change_pending(&self) -> bool {
        self.pending_pipeline.is_some() || self.pipeline_build.is_some()
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
        self.redraw_status_screen();
    }

    /// Builds the status screen being shown again (a new hint, an edited
    /// status screen shader). Does nothing while a pipeline renders.
    pub(crate) fn redraw_status_screen(&mut self) {
        if !self.pipeline_status.is_ready() {
            let status = self.pipeline_status.clone();
            let rebuilder = self.graph_rebuilder.clone();
            self.show_status(status, rebuilder);
        }
    }

    /// Makes the pipeline change [`set_pipeline`](Self::set_pipeline) asked
    /// for, if one is waiting, and waits for it to be built: when this returns
    /// [`pipeline_status`](Self::pipeline_status) is current. For a host with
    /// no frame to keep drawing meanwhile (a test, an offline tool, a renderer
    /// being constructed); [`render`](Self::render) does the same at the start
    /// of every frame without waiting.
    pub fn apply_pending_pipeline(&mut self) {
        self.begin_pending_pipeline(false);
        self.finish_pipeline_build(true);
    }

    /// The frame-boundary step of [`render`](Self::render): starts the pending
    /// pipeline change, and installs a build that has ended. Never waits.
    pub(crate) fn advance_pipeline(&mut self) {
        self.begin_pending_pipeline(true);
        self.finish_pipeline_build(false);
    }

    /// Starts what [`set_pipeline`](Self::set_pipeline) asked for. A clear or
    /// a failure is made at once; a pipeline starts building off the frame
    /// thread, and `announce` puts the "building" screen up when what is drawn
    /// is not a pipeline.
    fn begin_pending_pipeline(&mut self, announce: bool) {
        let Some(pending) = self.pending_pipeline.take() else {
            return;
        };
        helio_core::cpu_scope!("Helio: apply pipeline");
        // The change builds from the renderer's configuration as it is now.
        self.pending_rebuild = false;
        // A build still running is replaced; its graph is dropped unread.
        self.pipeline_build = None;
        match pending {
            PendingPipeline::Clear => {
                self.pipeline_id = None;
                self.show_status(PipelineStatus::None, None);
            }
            PendingPipeline::Fail { id, failure } => self.show_invalid(id, failure),
            PendingPipeline::Assign(pipeline) => self.start_pipeline_build(pipeline, announce),
        }
    }

    fn start_pipeline_build(&mut self, NamedPipeline { id, pipeline }: NamedPipeline, announce: bool) {
        let inherit =
            self.pipeline_status.is_ready() && self.pipeline_id.as_deref() == Some(id.as_str());
        let inputs = self.build_inputs();
        let built_for = (
            inputs.config.width,
            inputs.config.height,
            inputs.config.render_scale,
        );
        let slot = Arc::new(BuildSlot::default());
        let started = Instant::now();
        let job = {
            let slot = Arc::clone(&slot);
            move || {
                let device = Arc::clone(&inputs.device);
                let built = guard_build(&device, || Some(pipeline.into_graph(inputs.context())));
                slot.set(built.expect("a pipeline's build always ends in a result"));
            }
        };
        // No threads on the web: the pipeline builds where it is asked for.
        #[cfg(target_arch = "wasm32")]
        job();
        #[cfg(not(target_arch = "wasm32"))]
        helio_tasks::submit(
            helio_tasks::JobSpec::new(format!("Build render pipeline {id}")).owner("helio"),
            move |_| job(),
        );
        if announce && !self.pipeline_status.is_ready() {
            // What is drawn is not a pipeline: say that one is on its way.
            self.pipeline_id = Some(id.clone());
            self.show_status(PipelineStatus::Building { id: id.clone() }, None);
        }
        self.pipeline_build = Some(PipelineBuild {
            id,
            kind: BuildKind::Pipeline { inherit },
            slot,
            started,
            built_for,
        });
    }

    /// Starts the rebuild of the pipeline that renders, from the renderer's
    /// configuration as it is now, off the frame thread: the frames keep
    /// drawing the graph being replaced until [`Self::finish_pipeline_build`]
    /// finds the new one done. Does nothing for a graph installed without a
    /// rebuilder, and waits for a build already running (its configuration was
    /// taken when it started).
    pub(crate) fn begin_rebuild(&mut self) {
        if self.pipeline_build.is_some() {
            self.pending_rebuild = true;
            return;
        }
        let Some(rebuilder) = self.graph_rebuilder.clone() else {
            return;
        };
        let inputs = self.build_inputs();
        let built_for = (
            inputs.config.width,
            inputs.config.height,
            inputs.config.render_scale,
        );
        let id = self
            .pipeline_id
            .clone()
            .unwrap_or_else(|| UNNAMED_PIPELINE.to_owned());
        let slot = Arc::new(BuildSlot::default());
        let started = Instant::now();
        let job = {
            let slot = Arc::clone(&slot);
            move || {
                let device = Arc::clone(&inputs.device);
                let built = guard_build(&device, || {
                    Some(rebuilder(
                        &inputs.device,
                        &inputs.queue,
                        inputs.config.clone(),
                        inputs.debug_state.clone(),
                        &inputs.camera_buffer,
                        &inputs.debug_camera_buffer,
                        &inputs.stats,
                    ))
                });
                slot.set(built.expect("a rebuild always ends in a result"));
            }
        };
        #[cfg(target_arch = "wasm32")]
        job();
        #[cfg(not(target_arch = "wasm32"))]
        helio_tasks::submit(
            helio_tasks::JobSpec::new(format!("Rebuild render pipeline {id}")).owner("helio"),
            move |_| job(),
        );
        self.pipeline_build = Some(PipelineBuild {
            id,
            kind: BuildKind::Rebuild,
            slot,
            started,
            built_for,
        });
    }

    /// What a build reads, owned (the worker borrows nothing of the renderer).
    fn build_inputs(&self) -> BuildInputs {
        BuildInputs {
            device: self.device.clone(),
            queue: self.queue.clone(),
            config: self.renderer_config(),
            debug_state: self.debug_state.clone(),
            camera_buffer: self.camera_buffer.clone(),
            debug_camera_buffer: self.debug_camera_buffer.clone(),
            stats: self.stats.clone(),
            owns_device: self.owns_device,
            scene_db: self.scene_db.clone(),
        }
    }

    /// Installs the pipeline build that has ended (waiting for it when `wait`).
    pub(crate) fn finish_pipeline_build(&mut self, wait: bool) {
        let result = match &self.pipeline_build {
            None => return,
            Some(build) if wait => build.slot.wait(),
            Some(build) => match build.slot.take() {
                Some(result) => result,
                None => return,
            },
        };
        let build = self
            .pipeline_build
            .take()
            .expect("a build that ended is the one in flight");
        helio_core::cpu_scope!("Helio: install pipeline");
        log::info!(
            target: "helio::pipeline",
            "render pipeline `{}` built in {:.0} ms",
            build.id,
            build.started.elapsed().as_secs_f64() * 1000.0
        );
        match (build.kind, result) {
            (BuildKind::Pipeline { inherit }, Ok(graph)) => {
                self.install_pipeline_graph(graph, inherit && self.pipeline_status.is_ready());
                self.pipeline_id = Some(build.id);
                self.pipeline_status = PipelineStatus::Ready;
                self.resize_to_current(build.built_for);
            }
            (BuildKind::Pipeline { .. }, Err(failure)) => self.show_invalid(build.id, failure),
            (BuildKind::Rebuild, Ok(replacement)) => {
                self.install_replacement_graph(replacement);
                self.clear_target_next_frame = true;
                self.resize_to_current(build.built_for);
            }
            // The pipeline no longer builds (a plugin it needs was disabled, a
            // shader broke): say so on the output instead of keeping a graph
            // that is not what was asked for.
            (BuildKind::Rebuild, Err(failure)) => self.fail_current_pipeline(failure),
        }
    }

    /// Resized while a build ran: the graph sizes its textures in place.
    fn resize_to_current(&mut self, built_for: (u32, u32, f32)) {
        let now = (self.output_width, self.output_height, self.render_scale);
        if now != built_for {
            let internal = [
                ((self.output_width as f32 * self.render_scale).ceil() as u32).max(1),
                ((self.output_height as f32 * self.render_scale).ceil() as u32).max(1),
            ];
            self.graph
                .resize(RenderSize::new(internal, [self.output_width, self.output_height]));
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
        let Some(content) =
            status_screen::content_for(&status, &self.status_hint, rebuilder.is_some())
        else {
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
                PipelineStatus::Building { id } => {
                    log::info!(target: "helio::pipeline", "building render pipeline `{id}`: drawing the status screen meanwhile");
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
            content,
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

/// Runs `build` (a pipeline's, or a rebuilder's) with what can go wrong in it
/// made into the pipeline's failure: a panic in a pass constructor, and a GPU
/// validation error (a shader the GPU rejects). The host's frame thread
/// survives a pipeline that cannot be built, and so does a worker.
pub(crate) fn guard_build(
    device: &wgpu::Device,
    build: impl FnOnce() -> Option<Result<RenderGraph, PipelineFailure>>,
) -> Option<Result<RenderGraph, PipelineFailure>> {
    // Pass constructors create pipelines eagerly, so a validation error is
    // raised here and not at the first frame. (An error scope belongs to the
    // thread that pushed it: the one that builds.)
    #[cfg(not(target_arch = "wasm32"))]
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(build));
    #[cfg(not(target_arch = "wasm32"))]
    let gpu_error = pollster::block_on(scope.pop());
    #[cfg(target_arch = "wasm32")]
    let gpu_error: Option<wgpu::Error> = {
        let _ = device;
        None
    };

    match built {
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_owned())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_owned());
            Some(Err(
                PipelineFailure::new(status_screen::REASON_PANIC).with_errors([message])
            ))
        }
        Ok(Some(Ok(mut graph))) => match gpu_error {
            None => Some(Ok(graph)),
            Some(error) => {
                let mut failure = PipelineFailure::new(status_screen::REASON_GPU)
                    .with_errors([error.to_string()]);
                failure.rebuilder = graph.take_graph_data::<GraphRebuilder>();
                Some(Err(failure))
            }
        },
        Ok(other) => other,
    }
}
