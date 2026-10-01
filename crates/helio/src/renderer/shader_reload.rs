//! Shader hot reload: the renderer side (feature `shader-hot-reload`).
//!
//! `helio_core::shader::hot` watches the `.wgsl` files that passes registered
//! through `include_wgsl!`, validates edits on the CPU and records accepted
//! ones. A pass bakes its shader into pipelines in `new()`, so a new shader
//! text only matters once the pass is built again: this module rebuilds the
//! whole graph from the stored [`GraphRebuilder`](super::GraphRebuilder) -- the
//! same path a resize takes -- and swaps it in if the GPU accepts it.
//!
//! The rebuild runs inside a validation error scope and a panic guard, so a
//! shader that passes the CPU check but fails pipeline creation (or a pass
//! whose `new()` panics on a changed interface) leaves the current graph
//! running and reports the error instead of tearing the frame down.

use std::time::Instant;

use helio_core::shader::hot;

use super::renderer_impl::Renderer;

/// What the editor shows about shader hot reload.
#[derive(Clone, Debug, Default)]
pub struct ShaderReloadStatus {
    /// Generation of the shader sources the running graph was built from.
    /// `0` until the first successful reload.
    pub generation: u64,
    /// Why the most recent edit or rebuild was rejected, if it was: a
    /// rebuild failure, otherwise the watcher's naga diagnostics (with lines
    /// mapped back to the edited file). Cleared when a later edit is accepted
    /// and applied.
    pub last_error: Option<String>,
    /// When the running graph was last rebuilt for a shader change.
    pub last_reload: Option<Instant>,
}

/// Renderer-owned hot-reload bookkeeping.
#[derive(Default)]
pub(crate) struct ShaderReloadState {
    watcher_started: bool,
    warned_no_rebuilder: bool,
    generation: u64,
    rebuild_error: Option<String>,
    last_reload: Option<Instant>,
}

impl Renderer {
    /// Applies accepted shader edits. Call once per frame at a frame
    /// boundary; `render` already does. Cheap when nothing changed (one
    /// atomic swap).
    ///
    /// The first call starts the file watcher.
    pub fn poll_shader_reload(&mut self) {
        if !self.shader_reload.watcher_started {
            self.shader_reload.watcher_started = true;
            hot::start_watcher(Vec::new());
        }
        if !hot::take_dirty() {
            return;
        }
        let generation = hot::generation();

        if self.pending_resize.is_some() {
            // The pending resize rebuilds the graph from the latest sources
            // anyway (the override table is global); don't build twice.
            self.shader_reload.generation = generation;
            self.shader_reload.rebuild_error = None;
            self.shader_reload.last_reload = Some(Instant::now());
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
                self.reject_reload(format!("graph rebuild panicked: {reason}"));
                return;
            }
        };
        if let Some(error) = gpu_error {
            // The replacement is dropped; the old graph (and its persistent
            // state, which `inherit_persistent_state` has not touched yet)
            // keeps running.
            self.reject_reload(format!("GPU rejected the rebuilt graph: {error}"));
            return;
        }

        // A rebuilt graph owns a fresh `PipelineFormatCache`, so there is no
        // stale pipeline to clear.
        self.install_replacement_graph(replacement);
        self.clear_target_next_frame = true;
        self.shader_reload.generation = generation;
        self.shader_reload.rebuild_error = None;
        self.shader_reload.last_reload = Some(Instant::now());
        log::info!(
            "[helio-shader] graph rebuilt for shader generation {generation} in {:.1}ms",
            started.elapsed().as_secs_f64() * 1000.0
        );
    }

    fn reject_reload(&mut self, reason: String) {
        log::error!("[helio-shader] hot reload failed, keeping the previous graph: {reason}");
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
        }
    }
}
