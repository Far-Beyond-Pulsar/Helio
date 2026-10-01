//! Shader hot reload: the renderer side (feature `shader-hot-reload`).
//!
//! `helio_core::shader::hot` watches the `.wgsl` files that passes registered
//! through `include_wgsl!`, validates edits on the CPU and records accepted
//! ones. A pass bakes its shader into pipelines in `new()`, so a new shader
//! text only matters once the pass is built again. This module builds a
//! replacement graph from the stored [`GraphRebuilder`](super::GraphRebuilder)
//! -- the same path a resize takes -- and then either
//!
//! * swaps in only the passes whose shaders changed, leaving every other pass
//!   (and the history, residency and caches it owns) exactly as it was
//!   ([`ReloadMode::Selective`]), or
//! * installs the whole replacement ([`ReloadMode::Full`]) when the change
//!   cannot be attributed to passes, the graph's builder does not vouch for
//!   swapping them, or the replacement is not the same pass sequence.
//!
//! Attribution is framework-level: the changed files map to the crates that
//! embedded them (`hot::attribute_changes`) and a pass belongs to a crate when
//! its type path starts with the crate's name. No pass carries reload logic.
//! Which passes may be swapped together is the graph builder's
//! [`SwapPolicy`](helio_core::SwapPolicy); see its docs for the audit.
//!
//! The rebuild runs inside a validation error scope and a panic guard, so a
//! shader that passes the CPU check but fails pipeline creation (or a pass
//! whose `new()` panics on a changed interface) leaves the current graph
//! running, rolls the rejected text back out of the override table and reports
//! the error instead of tearing the frame down.
//!
//! The replacement graph is always built in full (a pass can only be built by
//! its constructor), so the hitch is the same as a resize; what a selective
//! swap saves is state, not time.

use std::time::Instant;

use helio_core::graph::plan_selective_swap;
use helio_core::shader::hot;
use helio_core::RenderGraph;

use super::renderer_impl::Renderer;

/// How the most recent shader reload was applied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReloadMode {
    /// Only these passes were replaced (by [`RenderPass::name`](helio_core::RenderPass::name),
    /// without repeats); every other pass kept its state.
    Selective { passes: Vec<&'static str> },
    /// The whole graph was rebuilt, and the state of passes that do not opt
    /// into `inherit_persistent_state` was reset. `reason` says why a
    /// selective swap was not possible.
    Full { reason: String },
}

/// What the editor shows about shader hot reload.
#[derive(Clone, Debug, Default)]
pub struct ShaderReloadStatus {
    /// Generation of the shader sources the running graph was built from.
    /// `0` until the first successful reload.
    pub generation: u64,
    /// Why the most recent edit or rebuild was rejected, if it was: a
    /// rebuild failure, otherwise the watcher's naga diagnostics (with lines
    /// mapped back to the edited file). A file whose rebuild was rejected is
    /// rolled back to its previous accepted text and keeps its entry until a
    /// later edit to it is accepted and applied.
    pub last_error: Option<String>,
    /// When the running graph was last rebuilt for a shader change.
    pub last_reload: Option<Instant>,
    /// How the last applied reload was done; `None` before the first.
    pub last_mode: Option<ReloadMode>,
}

/// Renderer-owned hot-reload bookkeeping.
#[derive(Default)]
pub(crate) struct ShaderReloadState {
    watcher_started: bool,
    warned_no_rebuilder: bool,
    generation: u64,
    rebuild_error: Option<String>,
    last_reload: Option<Instant>,
    last_mode: Option<ReloadMode>,
}

impl Renderer {
    /// Applies accepted shader edits. Call once per frame at a frame
    /// boundary; `render` already does. Cheap when nothing changed (one
    /// atomic swap).
    ///
    /// The first call starts the file watcher.
    pub fn poll_shader_reload(&mut self) {
        if !self.shader_reload.watcher_started {
            // Nothing to watch until a pass registers a shader it can locate
            // on disk; a build with no source tree never spawns the thread.
            if !hot::enabled() {
                self.shader_reload.watcher_started = true;
                return;
            }
            if !hot::has_registered() {
                return;
            }
            self.shader_reload.watcher_started = true;
            hot::start_watcher(Vec::new());
        }
        if !hot::take_dirty() {
            return;
        }
        helio_core::cpu_scope!("Helio: shader hot reload");
        let generation = hot::generation();
        // Everything accepted so far is this reload's; later edits start the
        // next batch. Dropping the batch commits it.
        let batch = hot::begin_reload();

        if self.pending_resize.is_some() {
            // The pending resize rebuilds the graph from the latest sources
            // anyway (the override table is global); don't build twice.
            self.shader_reload.generation = generation;
            self.shader_reload.rebuild_error = None;
            self.shader_reload.last_reload = Some(Instant::now());
            self.shader_reload.last_mode = Some(ReloadMode::Full {
                reason: "a resize rebuild was already pending".to_owned(),
            });
            return;
        }
        if self.graph_rebuilder.is_none() {
            if !self.shader_reload.warned_no_rebuilder {
                self.shader_reload.warned_no_rebuilder = true;
                log::warn!(
                    "[helio-shader] a shader changed but this renderer has no graph rebuilder; \
                     hot reload needs a graph built with a rebuilder (set_graph_with_builder / \
                     set_rebuilder)"
                );
            }
            return;
        }
        if batch.paths.is_empty() {
            // The edit that set the dirty flag was already folded into an
            // earlier batch; the graph was built from it.
            self.shader_reload.generation = generation;
            return;
        }

        let started = Instant::now();
        let config = self.renderer_config();
        // Pass constructors create pipelines eagerly; a pipeline built from a
        // shader the GPU rejects is a validation error, captured here rather
        // than reported to the uncaptured-error handler.
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.build_replacement_graph(config)
        }));
        let gpu_error = pollster::block_on(scope.pop());

        let replacement = match built {
            Ok(Some(replacement)) => replacement,
            Ok(None) => return,
            Err(payload) => {
                let reason = payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_owned())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown panic".to_owned());
                self.reject_reload(&batch, format!("graph rebuild panicked: {reason}"));
                return;
            }
        };
        if let Some(error) = gpu_error {
            // The replacement is dropped; the old graph (and its persistent
            // state, which nothing has touched yet) keeps running.
            self.reject_reload(&batch, format!("GPU rejected the rebuilt graph: {error}"));
            return;
        }

        let mode = self.apply_replacement(replacement, &batch.paths);
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        match &mode {
            ReloadMode::Selective { passes } => log::info!(
                "[helio-shader] selective reload for shader generation {generation}: swapped {} \
                 ({} pass type(s)), every other pass kept its state ({elapsed_ms:.1}ms)",
                passes.join(", "),
                passes.len()
            ),
            ReloadMode::Full { reason } => log::info!(
                "[helio-shader] full graph rebuild for shader generation {generation} \
                 ({reason}) in {elapsed_ms:.1}ms"
            ),
        }
        self.shader_reload.generation = generation;
        self.shader_reload.rebuild_error = None;
        self.shader_reload.last_reload = Some(Instant::now());
        self.shader_reload.last_mode = Some(mode);
    }

    /// Puts `replacement` into service for the shader edits in `changed`:
    /// selectively if the change can be attributed to swappable passes,
    /// otherwise by replacing the whole graph.
    fn apply_replacement(
        &mut self,
        mut replacement: RenderGraph,
        changed: &[std::path::PathBuf],
    ) -> ReloadMode {
        match self.try_selective_swap(&mut replacement, changed) {
            Ok(passes) => ReloadMode::Selective { passes },
            Err(reason) => {
                // A fresh `PipelineFormatCache` comes with the new graph, so
                // there is no stale pipeline to clear here.
                self.install_replacement_graph(replacement);
                // Phase 1 behaviour: the rebuilt graph may not clear the target.
                self.clear_target_next_frame = true;
                ReloadMode::Full { reason }
            }
        }
    }

    /// Moves the passes the change affects from `replacement` into the live
    /// graph. On `Err` nothing was changed and `replacement` is intact.
    fn try_selective_swap(
        &mut self,
        replacement: &mut RenderGraph,
        changed: &[std::path::PathBuf],
    ) -> Result<Vec<&'static str>, String> {
        let attribution = hot::attribute_changes(changed);
        if !attribution.is_complete() {
            let reason = if attribution.unattributed.is_empty() {
                "the change affects no registered shader".to_owned()
            } else {
                attribution.unattributed.join("; ")
            };
            return Err(format!("change not attributable to passes: {reason}"));
        }

        let live = self.graph.pass_identities();
        let fresh = replacement.pass_identities();
        let plan = plan_selective_swap(
            &live,
            &fresh,
            &attribution.crates(),
            replacement.swap_policy(),
        )?;
        self.graph.check_swap(replacement, &plan.picks)?;

        // New passes get the settings the host applies to every rebuilt graph
        // before they are moved, so they match the passes around them.
        self.apply_post_build_settings(replacement);
        self.graph.swap_passes_from(replacement, &plan.picks)?;
        Ok(plan.pass_names)
    }

    /// What `install_replacement_graph` and `replace_graph` apply to a freshly
    /// built graph, for a graph whose passes are about to be moved instead.
    fn apply_post_build_settings(&self, graph: &mut RenderGraph) {
        if let Some(pass) =
            graph.find_pass_mut::<helio_pass_radiance_cascades::RadianceCascadesPass>()
        {
            pass.set_gi_config(self.graph_config.gi_config);
        }
        if let Some(hook) = &self.graph_rebuild_hook {
            hook(graph, &self.device);
        }
        if let Some(sky) = graph.find_pass_mut::<helio_pass_sky::SkyPass>() {
            sky.set_fallback_sky_enabled(self.fallback_sky_enabled);
        }
    }

    /// Keeps the previous graph and puts the files of `batch` back to the text
    /// they had before it, so the rejected text does not fail later reloads.
    fn reject_reload(&mut self, batch: &hot::ReloadBatch, reason: String) {
        hot::reject_reload(batch, &reason);
        log::error!(
            "[helio-shader] hot reload failed, keeping the previous graph and restoring the \
             previous text of {} file(s): {reason}",
            batch.paths.len()
        );
        self.shader_reload.rebuild_error = Some(reason);
    }

    /// Current hot-reload state, for the editor to display.
    pub fn shader_reload_status(&self) -> ShaderReloadStatus {
        let last_error = self.shader_reload.rebuild_error.clone().or_else(|| {
            let errors = hot::last_errors();
            if errors.is_empty() {
                None
            } else {
                Some(
                    errors
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join("\n"),
                )
            }
        });
        ShaderReloadStatus {
            generation: self.shader_reload.generation,
            last_error,
            last_reload: self.shader_reload.last_reload,
            last_mode: self.shader_reload.last_mode.clone(),
        }
    }
}
