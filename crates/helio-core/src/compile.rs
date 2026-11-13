//! Pipeline compiles off the frame.
//!
//! Creating a pipeline runs the backend's shader compiler: milliseconds for
//! a small pass, seconds to minutes for a long kernel on a loaded machine.
//! A feature whose pipelines depend on content (a voxel world's form and
//! terrain program, known when the volume arrives) starts their compile
//! here when the content arrives and draws the content once they are ready,
//! never waiting for them in a frame. This is UE5's PSO precaching at proxy
//! creation; the driver's pipeline cache (`shader::create_*_pipeline`, kept
//! across runs with `PULSAR_PIPELINE_CACHE`) makes the next run's compiles
//! cheap, as DX12 pipeline libraries and `VkPipelineCache` do.
//!
//! [`Compiling::start`] runs `build` (shader modules and every pipeline of a
//! bundle) as a job on the host's task system (`helio_tasks`), inside its own
//! validation error scope, so a shader the backend rejects is that bundle's
//! [`failed`](Compiling::failed) reason rather than a panic on the frame
//! thread. The frame asks [`ready`](Compiling::ready), which never waits.
//! [`in_flight`] counts the jobs not finished in the process: what a host or
//! a test waits on before it judges a frame.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

/// Compile jobs started and not finished, process-wide.
static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);

/// Compile jobs started and not finished in this process.
pub fn in_flight() -> usize {
    IN_FLIGHT.load(Ordering::Acquire)
}

/// A bundle of pipelines being built off the frame.
pub struct Compiling<T> {
    label: String,
    outcome: Arc<Outcome<T>>,
}

/// A job's result, and a way to wait for it outside frames.
struct Outcome<T> {
    built: OnceLock<Result<T, String>>,
    finished: Mutex<bool>,
    wake: Condvar,
}

impl<T> Outcome<T> {
    fn set(&self, built: Result<T, String>) {
        let _ = self.built.set(built);
        *self.finished.lock().unwrap_or_else(|e| e.into_inner()) = true;
        self.wake.notify_all();
    }
}

impl<T: Send + Sync + 'static> Compiling<T> {
    /// Starts building `T` with `build` on a worker. `label` names the job
    /// (the host's task list, the failure log).
    pub fn start(
        device: &wgpu::Device,
        label: impl Into<String>,
        build: impl FnOnce(&wgpu::Device) -> T + Send + 'static,
    ) -> Self {
        let label = label.into();
        let outcome = Arc::new(Outcome {
            built: OnceLock::new(),
            finished: Mutex::new(false),
            wake: Condvar::new(),
        });
        let device = device.clone();
        let job = {
            let label = label.clone();
            let outcome = Arc::clone(&outcome);
            move || {
                let built = build_checked(&device, build);
                if let Err(reason) = &built {
                    log::error!(target: "helio::compile", "{label}: {reason}");
                }
                IN_FLIGHT.fetch_sub(1, Ordering::AcqRel);
                outcome.set(built);
            }
        };
        IN_FLIGHT.fetch_add(1, Ordering::AcqRel);
        // No threads on the web: the bundle builds where it is asked for.
        #[cfg(target_arch = "wasm32")]
        job();
        #[cfg(not(target_arch = "wasm32"))]
        helio_tasks::submit(
            helio_tasks::JobSpec::new(format!("Compile {label}")).owner("helio"),
            move |_| job(),
        );
        Self { label, outcome }
    }

    /// The bundle once it is built, without waiting.
    pub fn ready(&self) -> Option<&T> {
        self.outcome
            .built
            .get()
            .and_then(|built| built.as_ref().ok())
    }

    /// Why the bundle could not be built, once its compile ended so.
    pub fn failed(&self) -> Option<&str> {
        self.outcome
            .built
            .get()
            .and_then(|built| built.as_ref().err().map(String::as_str))
    }

    /// Whether the compile is still running.
    pub fn is_pending(&self) -> bool {
        self.outcome.built.get().is_none()
    }

    /// Blocks until the compile ends: for tests and offline tools that
    /// have no frame to keep drawing. A frame asks [`Self::ready`] instead.
    pub fn wait(&self) -> Result<&T, &str> {
        let mut finished = self
            .outcome
            .finished
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        while !*finished {
            finished = self
                .outcome
                .wake
                .wait(finished)
                .unwrap_or_else(|e| e.into_inner());
        }
        drop(finished);
        match self.outcome.built.get() {
            Some(Ok(bundle)) => Ok(bundle),
            Some(Err(reason)) => Err(reason),
            None => unreachable!("a finished compile holds its outcome"),
        }
    }

    pub fn label(&self) -> &str {
        &self.label
    }
}

/// Runs `build` inside a validation error scope of its own thread and with
/// its panics caught: a rejected shader is the bundle's failure.
fn build_checked<T>(
    device: &wgpu::Device,
    build: impl FnOnce(&wgpu::Device) -> T,
) -> Result<T, String> {
    #[cfg(not(target_arch = "wasm32"))]
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| build(device)));
    #[cfg(not(target_arch = "wasm32"))]
    let error = pollster::block_on(scope.pop());
    #[cfg(target_arch = "wasm32")]
    let error: Option<wgpu::Error> = None;
    match (built, error) {
        (Ok(bundle), None) => Ok(bundle),
        (Ok(_), Some(error)) => Err(error.to_string()),
        (Err(panic), _) => Err(panic
            .downcast_ref::<&str>()
            .map(|s| (*s).to_owned())
            .or_else(|| panic.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "the build panicked".to_owned())),
    }
}
