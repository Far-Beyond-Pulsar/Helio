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

use std::sync::atomic::Ordering;
use std::sync::{Arc, Condvar, Mutex, OnceLock};

/// Compile jobs started and not finished in this process (the host's count,
/// shared with every copy of Helio: [`crate::runtime`]).
pub fn in_flight() -> usize {
    crate::runtime::shared()
        .compiles_in_flight
        .load(Ordering::Acquire)
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
        let asked = std::time::Instant::now();
        let job = {
            let label = label.clone();
            let outcome = Arc::clone(&outcome);
            move || {
                let started = std::time::Instant::now();
                let built = build_checked(&device, build);
                if let Err(reason) = &built {
                    log::error!(target: "helio::compile", "{label}: {reason}");
                }
                // How long a bundle waited for a worker and took to build:
                // what keeps its content from the frame.
                log::info!(
                    target: "helio::compile",
                    "{label}: waited {:.1} s, built in {:.1} s",
                    (started - asked).as_secs_f64(),
                    started.elapsed().as_secs_f64()
                );
                crate::runtime::shared()
                    .compiles_in_flight
                    .fetch_sub(1, Ordering::AcqRel);
                outcome.set(built);
            }
        };
        crate::runtime::shared()
            .compiles_in_flight
            .fetch_add(1, Ordering::AcqRel);
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

/// The pipelines of a pass that exist in many variants (a material's program,
/// a draw's raster state), each compiled off the frame the first time its key
/// is asked for.
///
/// A frame asks [`get`](Self::get); a key never seen before starts its compile
/// on the host's task system ([`Compiling`]) and answers `None` until it is
/// ready, so a frame that meets a new variant draws without it (or with a
/// stand-in the pass picks) instead of waiting for the backend's shader
/// compiler. A variant that fails to build is logged by its job and then answers
/// `None` like one still compiling; it is not retried until [`clear`](Self::clear).
pub struct Variants<K, T> {
    entries: std::collections::HashMap<K, Compiling<T>>,
}

impl<K, T> Default for Variants<K, T> {
    fn default() -> Self {
        Self {
            entries: Default::default(),
        }
    }
}

impl<K, T> Variants<K, T>
where
    K: std::hash::Hash + Eq + Clone,
    T: Send + Sync + 'static,
{
    pub fn new() -> Self {
        Self::default()
    }

    /// The variant `key` once it is built. Starts building it with `build`
    /// when it was never asked for (`label` names the job); never waits.
    pub fn get(
        &mut self,
        device: &wgpu::Device,
        key: &K,
        label: impl FnOnce() -> String,
        build: impl FnOnce(&wgpu::Device) -> T + Send + 'static,
    ) -> Option<&T> {
        if !self.entries.contains_key(key) {
            self.entries
                .insert(key.clone(), Compiling::start(device, label(), build));
        }
        self.entries.get(key)?.ready()
    }

    /// Whether `key` was asked for (built, failed or still compiling).
    pub fn contains(&self, key: &K) -> bool {
        self.entries.contains_key(key)
    }

    /// The variant `key` if it is built, without starting anything.
    pub fn ready(&self, key: &K) -> Option<&T> {
        self.entries.get(key).and_then(Compiling::ready)
    }

    /// Blocks until `key` is built or has failed: for tests and offline tools
    /// that have no frame to keep drawing. `None` when it was never asked for.
    pub fn wait(&self, key: &K) -> Option<Result<&T, &str>> {
        self.entries.get(key).map(Compiling::wait)
    }

    /// Forgets every variant (the pass's state they were built for changed).
    /// Compiles still running finish and are dropped.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Variants asked for, built or not.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// One pipeline's build: what it could not build, if so.
type Job<'a> = Box<dyn FnOnce() -> Result<(), String> + Send + 'a>;

/// The pipelines of one bundle, compiled side by side.
///
/// A pipeline's compile is the backend's shader compiler at work: seconds for
/// a long kernel, and a bundle holds dozens. One after another they keep a
/// bundle's content from the frame for the sum of them; side by side, for about
/// the longest one (the same shape as UE5's parallel PSO precache). `SideBySide`
/// collects a bundle's pipelines as they are named ([`add`](Self::add)) and
/// compiles them together ([`compile`](Self::compile)) on the host's compute
/// lanes ([`helio_tasks::parallel`]), inside the bundle's own compile job
/// ([`Compiling`]): the lanes are the host's compute pool, so a job that
/// waited on further jobs would wait on the host's few long-task workers it
/// is itself holding. As many lanes as a quarter of the host's parallelism (2 to
/// 8), so the pool keeps serving the frame.
pub struct SideBySide<'a> {
    device: &'a wgpu::Device,
    jobs: Vec<Job<'a>>,
}

/// A pipeline of a [`SideBySide`]: taken once they are compiled.
pub struct Slot<T>(Arc<Mutex<Option<T>>>);

impl<T> Slot<T> {
    /// The pipeline, after [`SideBySide::compile`].
    pub fn take(self) -> T {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .expect("taken once, after its bundle compiled")
    }
}

impl<'a> SideBySide<'a> {
    pub fn new(device: &'a wgpu::Device) -> Self {
        Self {
            device,
            jobs: Vec::new(),
        }
    }

    /// A pipeline `build` makes, compiled with the others.
    pub fn add<T: Send + 'a>(&mut self, build: impl FnOnce() -> T + Send + 'a) -> Slot<T> {
        let slot = Arc::new(Mutex::new(None));
        let (out, device) = (slot.clone(), self.device);
        self.jobs.push(Box::new(move || {
            // Error scopes are a thread's: each build answers for itself, on
            // whichever thread runs it.
            #[cfg(not(target_arch = "wasm32"))]
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let built = build();
            #[cfg(not(target_arch = "wasm32"))]
            let error = ready(scope.pop());
            #[cfg(target_arch = "wasm32")]
            let error: Option<wgpu::Error> = {
                let _ = device;
                None
            };
            *out.lock().unwrap_or_else(|e| e.into_inner()) = Some(built);
            error.map_or(Ok(()), |error| Err(error.to_string()))
        }));
        Slot(slot)
    }

    /// Compiles every pipeline added, side by side, each lane taking the next
    /// one not started. Panics with the first failure's reason: the bundle's
    /// compile job reports it as the bundle's ([`Compiling::failed`]).
    pub fn compile(self) {
        let lanes = (helio_tasks::parallelism() / 4)
            .clamp(2, 8)
            .min(self.jobs.len().max(1));
        let next = Mutex::new(self.jobs.into_iter());
        let failures = Mutex::new(Vec::<String>::new());
        helio_tasks::parallel(lanes, &|_lane| loop {
            let job = next.lock().unwrap_or_else(|e| e.into_inner()).next();
            match job {
                Some(job) => {
                    if let Err(reason) = job() {
                        failures.lock().unwrap_or_else(|e| e.into_inner()).push(reason);
                    }
                }
                None => break,
            }
        });
        let failures = failures.into_inner().unwrap_or_else(|e| e.into_inner());
        if let Some(reason) = failures.first() {
            panic!("{reason}");
        }
    }
}

/// A future that is ready when asked (a native device's error scope).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn ready<T>(future: impl std::future::Future<Output = Option<T>>) -> Option<T> {
    let mut future = std::pin::pin!(future);
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    match future.as_mut().poll(&mut context) {
        std::task::Poll::Ready(value) => value,
        std::task::Poll::Pending => None,
    }
}

#[cfg(test)]
mod side_by_side_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn device() -> Option<wgpu::Device> {
        let instance = wgpu::Instance::new(crate::wgpu_instance_descriptor());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        pollster::block_on(adapter.request_device(&Default::default()))
            .ok()
            .map(|(device, _)| device)
    }

    /// Every pipeline added is built once and taken as its own, several at a
    /// time: a bundle takes about its longest build, not their sum.
    #[test]
    fn a_bundles_builds_run_side_by_side_and_keep_their_places() {
        let Some(device) = device() else { return };
        let (running, most) = (AtomicUsize::new(0), AtomicUsize::new(0));
        let mut bundle = SideBySide::new(&device);
        let slots: Vec<Slot<usize>> = (0..24)
            .map(|n| {
                let (running, most) = (&running, &most);
                bundle.add(move || {
                    let now = running.fetch_add(1, Ordering::SeqCst) + 1;
                    most.fetch_max(now, Ordering::SeqCst);
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    running.fetch_sub(1, Ordering::SeqCst);
                    n * 7
                })
            })
            .collect();
        let started = std::time::Instant::now();
        bundle.compile();
        let took = started.elapsed();
        for (n, slot) in slots.into_iter().enumerate() {
            assert_eq!(slot.take(), n * 7);
        }
        let lanes = (helio_tasks::parallelism() / 4).clamp(2, 8);
        assert!(
            most.load(Ordering::SeqCst) <= lanes,
            "no more than its lanes at a time"
        );
        if helio_tasks::parallelism() >= 2 {
            assert!(most.load(Ordering::SeqCst) >= 2, "side by side");
            assert!(took.as_millis() < 24 * 20, "{took:?} for 24 builds of 20 ms");
        }
    }

    /// A pipeline the backend rejects fails the bundle with the reason,
    /// whichever thread built it.
    #[test]
    fn a_rejected_pipeline_is_the_bundles_failure() {
        let Some(device) = device() else { return };
        let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut bundle = SideBySide::new(&device);
            let _fine = bundle.add(|| 1);
            let _bad = bundle.add(|| {
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("not wgsl"),
                    source: wgpu::ShaderSource::Wgsl("fn broken( {".into()),
                })
            });
            bundle.compile();
        }));
        let reason = failure.expect_err("the bundle fails");
        let reason = reason.downcast_ref::<String>().cloned().unwrap_or_default();
        assert!(!reason.is_empty(), "with the backend's reason");
    }
}
