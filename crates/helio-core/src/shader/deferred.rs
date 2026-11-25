//! Pipelines created in batches: queued as they are named, compiled side by side.
//!
//! A pass constructor names its pipelines one after another and each
//! `create_*_pipeline` ran the backend's shader compiler before the next line:
//! on Direct3D 12 that is DXC and the driver, hundreds of milliseconds for a
//! long kernel, and a render pipeline of fifty nodes held its build for the
//! sum of them. Backends have no pipeline cache to make the second build cheap
//! there (wgpu's `PIPELINE_CACHE` is Vulkan's), so what is left is not
//! compiling them in a row.
//!
//! [`create_compute_pipeline_deferred`] and [`create_render_pipeline_deferred`]
//! take the same descriptors as the immediate ones and answer a
//! [`Deferred`] at once. Inside [`batched`] (a build's scope) the descriptor is
//! queued and nothing compiles; the queue compiles side by side on the host's
//! compute lanes ([`helio_tasks::parallel`]) when the build ends, or at the first
//! use of a pipeline whose compile was not started yet. Outside a batch the
//! pipeline compiles on the spot, so a caller that is not part of a build
//! behaves as before. This is UE5's parallel PSO precompile and Unity's
//! asynchronous shader compilation, with a deterministic end: [`batched`] returns
//! when every queued pipeline is built or has failed.
//!
//! A [`Deferred`] dereferences to the pipeline. After [`batched`] returned it
//! costs one atomic load; before, the first use compiles what is queued. A
//! pipeline the backend rejects fails the batch with the backend's reason
//! (`batched` panics with it, as a rejected immediate pipeline would, and the
//! build job reports a panic as the pipeline's failure).

use std::cell::RefCell;
use std::ops::Deref;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};

/// Names a pipeline a build used before it was built: that use waited for the
/// compile, and a constructor that does so holds its build for it.
struct UsedEarly<'a> {
    label: &'a str,
    since: std::time::Instant,
}

impl Drop for UsedEarly<'_> {
    fn drop(&mut self) {
        let waited = self.since.elapsed();
        if waited >= std::time::Duration::from_millis(30) {
            log::info!(
                target: "helio::pipeline",
                "pipeline `{}` was used at build time and waited {:.0} ms for its compile",
                self.label,
                waited.as_secs_f64() * 1000.0
            );
        }
    }
}

/// A pipeline compiled in a batch, once it is built.
pub struct Deferred<T> {
    cell: Arc<Cell<T>>,
}

/// A compute pipeline made by [`create_compute_pipeline_deferred`].
pub type DeferredCompute = Deferred<wgpu::ComputePipeline>;
/// A render pipeline made by [`create_render_pipeline_deferred`].
pub type DeferredRender = Deferred<wgpu::RenderPipeline>;

impl<T> Clone for Deferred<T> {
    fn clone(&self) -> Self {
        Self {
            cell: Arc::clone(&self.cell),
        }
    }
}

type Job<T> = Box<dyn FnOnce() -> Result<T, String> + Send>;

struct Cell<T> {
    job: Mutex<Option<Job<T>>>,
    value: OnceLock<Result<T, String>>,
    done: Mutex<()>,
    wake: Condvar,
    batch: Weak<Batch>,
    /// The build's abandon flag: a pipeline not started when it is set never is.
    cancel: Option<Arc<AtomicBool>>,
    label: String,
}

trait Queued: Send + Sync {
    /// Runs the compile if nobody started it.
    fn run(&self);
    fn failure(&self) -> Option<String>;
    /// Whether the compile ended (built or failed).
    fn finished(&self) -> bool;
}

impl<T: Send + Sync> Queued for Cell<T> {
    fn run(&self) {
        let job = self.job.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(job) = job {
            let cancelled = self
                .cancel
                .as_ref()
                .is_some_and(|token| token.load(Ordering::Acquire));
            let built = if cancelled {
                Err("the build was abandoned".to_owned())
            } else {
                job()
            };
            let _ = self.value.set(built);
            let _guard = self.done.lock().unwrap_or_else(|e| e.into_inner());
            self.wake.notify_all();
        }
    }

    fn failure(&self) -> Option<String> {
        match self.value.get() {
            Some(Err(reason)) => Some(format!("{}: {reason}", self.label)),
            _ => None,
        }
    }

    fn finished(&self) -> bool {
        self.value.get().is_some()
    }
}

impl<T: Send + Sync> Cell<T> {
    fn get(&self) -> &T {
        loop {
            match self.value.get() {
                Some(Ok(pipeline)) => return pipeline,
                Some(Err(reason)) => panic!("pipeline `{}` did not build: {reason}", self.label),
                None => {}
            }
            // The first use of a pipeline nobody started compiles everything
            // queued, together; one another thread is compiling is waited for.
            let waited = std::time::Instant::now();
            let _used = UsedEarly { label: &self.label, since: waited };
            if let Some(batch) = self.batch.upgrade() {
                batch.flush();
            }
            self.run();
            let mut guard = self.done.lock().unwrap_or_else(|e| e.into_inner());
            while self.value.get().is_none() {
                guard = self
                    .wake
                    .wait_timeout(guard, std::time::Duration::from_millis(50))
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
            }
        }
    }
}

impl<T: Send + Sync> Deref for Deferred<T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.cell.get()
    }
}

impl<T: Send + Sync> Deferred<T> {
    /// Whether the pipeline is built (or failed): a use would not wait.
    pub fn is_ready(&self) -> bool {
        self.cell.value.get().is_some()
    }

    /// A pipeline that is already built (made outside any batch).
    pub fn ready(pipeline: T) -> Self {
        let cell = Cell {
            job: Mutex::new(None),
            value: OnceLock::new(),
            done: Mutex::new(()),
            wake: Condvar::new(),
            batch: Weak::new(),
            cancel: None,
            label: String::new(),
        };
        let _ = cell.value.set(Ok(pipeline));
        Self {
            cell: Arc::new(cell),
        }
    }
}

/// The pipelines queued in one build, compiled by workers while the build
/// goes on and by every lane when it ends.
struct Batch {
    queue: Mutex<VecDeque<Arc<dyn Queued>>>,
    more: Condvar,
    closed: AtomicBool,
    started_workers: AtomicBool,
    everything: Mutex<Vec<Arc<dyn Queued>>>,
    cancel: Option<Arc<AtomicBool>>,
}

/// Workers that compile while a build still names pipelines: a few, since the
/// build itself and the voxel and other compile jobs share the host's long
/// tasks; the rest of the lanes join when the build ends ([`Batch::flush`]).
const EARLY_WORKERS: usize = 3;

impl Batch {
    fn next(&self) -> Option<Arc<dyn Queued>> {
        self.queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop_front()
    }

    /// Starts the early workers, once, on the host's tasks.
    fn start_workers(self: &Arc<Self>) {
        if self.started_workers.swap(true, Ordering::AcqRel) {
            return;
        }
        for _ in 0..EARLY_WORKERS {
            let batch = Arc::clone(self);
            helio_tasks::submit(
                helio_tasks::JobSpec::new("Compile queued pipelines").owner("helio"),
                move |_| batch.worker(),
            );
        }
    }

    /// An early worker: compiles what the build queues until it ends.
    fn worker(&self) {
        loop {
            let item = {
                let mut queue = self.queue.lock().unwrap_or_else(|e| e.into_inner());
                loop {
                    if let Some(item) = queue.pop_front() {
                        break Some(item);
                    }
                    if self.closed.load(Ordering::Acquire) {
                        break None;
                    }
                    queue = self
                        .more
                        .wait_timeout(queue, std::time::Duration::from_millis(20))
                        .unwrap_or_else(|e| e.into_inner())
                        .0;
                }
            };
            match item {
                Some(item) => item.run(),
                None => return,
            }
        }
    }

    /// Compiles what is queued side by side, the caller one of the lanes.
    fn flush(&self) {
        let waiting = self.queue.lock().unwrap_or_else(|e| e.into_inner()).len();
        if waiting == 0 {
            return;
        }
        let lanes = lanes().min(waiting);
        let work = |_lane: usize| {
            while let Some(item) = self.next() {
                item.run();
            }
        };
        if lanes <= 1 {
            work(0);
        } else {
            helio_tasks::parallel(lanes, &work);
        }
    }
}

/// The compile lanes a batch uses: most of the host's compute threads, two
/// kept for the frame and the rest of the editor.
fn lanes() -> usize {
    helio_tasks::parallelism().saturating_sub(2).clamp(2, 24)
}

thread_local! {
    static CURRENT: RefCell<Option<Arc<Batch>>> = const { RefCell::new(None) };
    static CANCEL: RefCell<Option<Arc<AtomicBool>>> = const { RefCell::new(None) };
}

/// Marks this thread's build as abandoned when `token` is set: pipelines not
/// started compile no more and node builds stop at the next node
/// ([`build_cancelled`]). `None` clears it.
pub fn set_build_cancel(token: Option<Arc<AtomicBool>>) {
    CANCEL.with(|cell| *cell.borrow_mut() = token);
}

/// Whether the build running on this thread was abandoned.
pub fn build_cancelled() -> bool {
    CANCEL.with(|cell| {
        cell.borrow()
            .as_ref()
            .is_some_and(|token| token.load(Ordering::Acquire))
    })
}

/// Runs `build` with this thread's deferred pipelines queued, then compiles
/// them side by side before returning. Panics with the first failure's reason.
pub fn batched<R>(build: impl FnOnce() -> R) -> R {
    let cancel = CANCEL.with(|cell| cell.borrow().clone());
    let batch = Arc::new(Batch {
        queue: Mutex::new(VecDeque::new()),
        more: Condvar::new(),
        closed: AtomicBool::new(false),
        started_workers: AtomicBool::new(false),
        everything: Mutex::new(Vec::new()),
        cancel,
    });
    let previous = CURRENT.with(|current| current.borrow_mut().replace(Arc::clone(&batch)));
    let finished = std::panic::catch_unwind(std::panic::AssertUnwindSafe(build));
    CURRENT.with(|current| *current.borrow_mut() = previous);
    // Whatever the build did, nothing may compile after it returns: the
    // workers stop taking new work once the queue is empty, every lane takes
    // what is left, and the pipelines a worker still holds are waited for.
    let started = std::time::Instant::now();
    let total = batch.everything.lock().unwrap_or_else(|e| e.into_inner()).len();
    let left = batch.queue.lock().unwrap_or_else(|e| e.into_inner()).len();
    batch.closed.store(true, Ordering::Release);
    batch.more.notify_all();
    batch.flush();
    let everything = std::mem::take(&mut *batch.everything.lock().unwrap_or_else(|e| e.into_inner()));
    for item in &everything {
        while !item.finished() {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
    if total > 0 {
        log::info!(
            target: "helio::pipeline",
            "{total} pipeline(s) queued by the build ({left} still waiting when it ended, compiled side by side on {} lane(s) in {:.0} ms)",
            lanes().min(left.max(1)),
            started.elapsed().as_secs_f64() * 1000.0
        );
    }
    let result = match finished {
        Ok(result) => result,
        Err(panic) => std::panic::resume_unwind(panic),
    };
    if let Some(reason) = everything.iter().find_map(|item| item.failure()) {
        panic!("{reason}");
    }
    result
}

fn defer<T: Send + Sync + 'static>(label: String, job: Job<T>) -> Deferred<T> {
    let batch = CURRENT.with(|current| current.borrow().clone());
    let Some(batch) = batch else {
        // Not in a build: compile now, as the immediate helpers do.
        return match job() {
            Ok(pipeline) => Deferred::ready(pipeline),
            Err(reason) => panic!("pipeline `{label}` did not build: {reason}"),
        };
    };
    let cell = Arc::new(Cell {
        job: Mutex::new(Some(job)),
        value: OnceLock::new(),
        done: Mutex::new(()),
        wake: Condvar::new(),
        batch: Arc::downgrade(&batch),
        cancel: batch.cancel.clone(),
        label,
    });
    batch
        .queue
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push_back(cell.clone());
    batch.more.notify_one();
    batch.start_workers();
    batch
        .everything
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(cell.clone());
    Deferred { cell }
}

/// Runs `create` inside a validation error scope of the thread it runs on, so
/// a pipeline the backend rejects is that pipeline's failure.
fn checked<T>(device: &wgpu::Device, create: impl FnOnce() -> T) -> Result<T, String> {
    #[cfg(not(target_arch = "wasm32"))]
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let created = create();
    #[cfg(not(target_arch = "wasm32"))]
    let error = crate::compile::ready(scope.pop());
    #[cfg(target_arch = "wasm32")]
    let error: Option<wgpu::Error> = {
        let _ = device;
        None
    };
    match error {
        None => Ok(created),
        Some(error) => Err(error.to_string()),
    }
}

/// [`create_compute_pipeline`](super::create_compute_pipeline), queued in the
/// current [`batched`] build and compiled with the others.
pub fn create_compute_pipeline_deferred(
    device: &wgpu::Device,
    descriptor: wgpu::ComputePipelineDescriptor<'_>,
) -> DeferredCompute {
    let label = descriptor.label.unwrap_or("compute pipeline").to_owned();
    let owned = OwnedCompute::capture(&descriptor);
    let device = device.clone();
    defer(
        label,
        Box::new(move || {
            let started = std::time::Instant::now();
            let pipeline = checked(&device, || {
                owned.with_descriptor(|descriptor| super::create_compute_pipeline(&device, descriptor))
            });
            note(owned.label.as_deref(), started.elapsed());
            pipeline
        }),
    )
}

/// [`create_render_pipeline`](super::create_render_pipeline), queued in the
/// current [`batched`] build and compiled with the others.
pub fn create_render_pipeline_deferred(
    device: &wgpu::Device,
    descriptor: wgpu::RenderPipelineDescriptor<'_>,
) -> DeferredRender {
    let label = descriptor.label.unwrap_or("render pipeline").to_owned();
    let owned = OwnedRender::capture(&descriptor);
    let device = device.clone();
    defer(
        label,
        Box::new(move || {
            let started = std::time::Instant::now();
            let pipeline = checked(&device, || {
                owned.with_descriptor(|descriptor| super::create_render_pipeline(&device, descriptor))
            });
            note(owned.label.as_deref(), started.elapsed());
            pipeline
        }),
    )
}

fn note(label: Option<&str>, took: std::time::Duration) {
    if took >= std::time::Duration::from_millis(250) {
        log::info!(
            target: "helio::pipeline",
            "pipeline `{}` took {:.0} ms",
            label.unwrap_or("unlabelled"),
            took.as_secs_f64() * 1000.0
        );
    }
}

/// A compute descriptor with every borrow owned (a handle is a reference count).
struct OwnedCompute {
    label: Option<String>,
    layout: Option<wgpu::PipelineLayout>,
    module: wgpu::ShaderModule,
    entry_point: Option<String>,
    constants: Vec<(String, f64)>,
    zero_initialize_workgroup_memory: bool,
    cache: Option<wgpu::PipelineCache>,
}

impl OwnedCompute {
    fn capture(descriptor: &wgpu::ComputePipelineDescriptor<'_>) -> Self {
        Self {
            label: descriptor.label.map(str::to_owned),
            layout: descriptor.layout.cloned(),
            module: descriptor.module.clone(),
            entry_point: descriptor.entry_point.map(str::to_owned),
            constants: owned_constants(descriptor.compilation_options.constants),
            zero_initialize_workgroup_memory: descriptor
                .compilation_options
                .zero_initialize_workgroup_memory,
            cache: descriptor.cache.cloned(),
        }
    }

    fn with_descriptor<R>(&self, create: impl FnOnce(wgpu::ComputePipelineDescriptor<'_>) -> R) -> R {
        let constants: Vec<(&str, f64)> = self
            .constants
            .iter()
            .map(|(name, value)| (name.as_str(), *value))
            .collect();
        create(wgpu::ComputePipelineDescriptor {
            label: self.label.as_deref(),
            layout: self.layout.as_ref(),
            module: &self.module,
            entry_point: self.entry_point.as_deref(),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                zero_initialize_workgroup_memory: self.zero_initialize_workgroup_memory,
            },
            cache: self.cache.as_ref(),
        })
    }
}

fn owned_constants(constants: &[(&str, f64)]) -> Vec<(String, f64)> {
    constants
        .iter()
        .map(|(name, value)| ((*name).to_owned(), *value))
        .collect()
}

struct OwnedBuffer {
    array_stride: wgpu::BufferAddress,
    step_mode: wgpu::VertexStepMode,
    attributes: Vec<wgpu::VertexAttribute>,
}

/// A render descriptor with every borrow owned.
struct OwnedRender {
    label: Option<String>,
    layout: Option<wgpu::PipelineLayout>,
    vertex_module: wgpu::ShaderModule,
    vertex_entry: Option<String>,
    vertex_constants: Vec<(String, f64)>,
    vertex_zero_init: bool,
    buffers: Vec<Option<OwnedBuffer>>,
    primitive: wgpu::PrimitiveState,
    depth_stencil: Option<wgpu::DepthStencilState>,
    multisample: wgpu::MultisampleState,
    fragment: Option<OwnedFragment>,
    multiview_mask: Option<std::num::NonZeroU32>,
    cache: Option<wgpu::PipelineCache>,
}

struct OwnedFragment {
    module: wgpu::ShaderModule,
    entry: Option<String>,
    constants: Vec<(String, f64)>,
    zero_init: bool,
    targets: Vec<Option<wgpu::ColorTargetState>>,
}

impl OwnedRender {
    fn capture(descriptor: &wgpu::RenderPipelineDescriptor<'_>) -> Self {
        let vertex = &descriptor.vertex;
        Self {
            label: descriptor.label.map(str::to_owned),
            layout: descriptor.layout.cloned(),
            vertex_module: vertex.module.clone(),
            vertex_entry: vertex.entry_point.map(str::to_owned),
            vertex_constants: owned_constants(vertex.compilation_options.constants),
            vertex_zero_init: vertex.compilation_options.zero_initialize_workgroup_memory,
            buffers: vertex
                .buffers
                .iter()
                .map(|buffer| {
                    buffer.as_ref().map(|buffer| OwnedBuffer {
                        array_stride: buffer.array_stride,
                        step_mode: buffer.step_mode,
                        attributes: buffer.attributes.to_vec(),
                    })
                })
                .collect(),
            primitive: descriptor.primitive,
            depth_stencil: descriptor.depth_stencil.clone(),
            multisample: descriptor.multisample,
            fragment: descriptor.fragment.as_ref().map(|fragment| OwnedFragment {
                module: fragment.module.clone(),
                entry: fragment.entry_point.map(str::to_owned),
                constants: owned_constants(fragment.compilation_options.constants),
                zero_init: fragment.compilation_options.zero_initialize_workgroup_memory,
                targets: fragment.targets.to_vec(),
            }),
            multiview_mask: descriptor.multiview_mask,
            cache: descriptor.cache.cloned(),
        }
    }

    fn with_descriptor<R>(&self, create: impl FnOnce(wgpu::RenderPipelineDescriptor<'_>) -> R) -> R {
        let buffers: Vec<Option<wgpu::VertexBufferLayout<'_>>> = self
            .buffers
            .iter()
            .map(|buffer| {
                buffer.as_ref().map(|buffer| wgpu::VertexBufferLayout {
                    array_stride: buffer.array_stride,
                    step_mode: buffer.step_mode,
                    attributes: &buffer.attributes,
                })
            })
            .collect();
        let vertex_constants: Vec<(&str, f64)> = self
            .vertex_constants
            .iter()
            .map(|(name, value)| (name.as_str(), *value))
            .collect();
        let fragment_constants: Vec<(&str, f64)> = self
            .fragment
            .as_ref()
            .map(|fragment| {
                fragment
                    .constants
                    .iter()
                    .map(|(name, value)| (name.as_str(), *value))
                    .collect()
            })
            .unwrap_or_default();
        create(wgpu::RenderPipelineDescriptor {
            label: self.label.as_deref(),
            layout: self.layout.as_ref(),
            vertex: wgpu::VertexState {
                module: &self.vertex_module,
                entry_point: self.vertex_entry.as_deref(),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &vertex_constants,
                    zero_initialize_workgroup_memory: self.vertex_zero_init,
                },
                buffers: &buffers,
            },
            primitive: self.primitive,
            depth_stencil: self.depth_stencil.clone(),
            multisample: self.multisample,
            fragment: self.fragment.as_ref().map(|fragment| wgpu::FragmentState {
                module: &fragment.module,
                entry_point: fragment.entry.as_deref(),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &fragment_constants,
                    zero_initialize_workgroup_memory: fragment.zero_init,
                },
                targets: &fragment.targets,
            }),
            multiview_mask: self.multiview_mask,
            cache: self.cache.as_ref(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device() -> Option<wgpu::Device> {
        let instance = wgpu::Instance::new(crate::wgpu_instance_descriptor());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        pollster::block_on(adapter.request_device(&Default::default()))
            .ok()
            .map(|(device, _)| device)
    }

    fn module(device: &wgpu::Device, source: &str) -> wgpu::ShaderModule {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("deferred test"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        })
    }

    const KERNEL: &str = "@compute @workgroup_size(1) fn main() {}";

    fn pipeline(device: &wgpu::Device, module: &wgpu::ShaderModule) -> DeferredCompute {
        create_compute_pipeline_deferred(
            device,
            wgpu::ComputePipelineDescriptor {
                label: Some("deferred test"),
                layout: None,
                module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        )
    }

    /// Outside a build a pipeline compiles on the spot, as the immediate
    /// helpers do: nothing is queued and nothing waits.
    #[test]
    fn outside_a_batch_a_pipeline_is_built_at_once() {
        let Some(device) = device() else { return };
        let module = module(&device, KERNEL);
        assert!(pipeline(&device, &module).is_ready());
    }

    /// In a batch nothing compiles until the build ends (or a pipeline is
    /// used): every queued pipeline is built when `batched` returns.
    #[test]
    fn a_batch_queues_and_builds_every_pipeline_by_its_end() {
        let Some(device) = device() else { return };
        let module = module(&device, KERNEL);
        let (queued, outcome) = {
            let mut made = Vec::new();
            let outcome = batched(|| {
                for _ in 0..12 {
                    made.push(pipeline(&device, &module));
                }
                // Queued, not compiled: the workers may have started some,
                // but the build thread has not asked for any.
                made.len()
            });
            (made, outcome)
        };
        assert_eq!(outcome, 12);
        assert!(queued.iter().all(Deferred::is_ready), "all built at the end");
    }

    /// Using a queued pipeline before the build ends compiles it (and what
    /// is queued with it) rather than failing or waiting forever.
    #[test]
    fn using_a_pipeline_early_builds_it() {
        let Some(device) = device() else { return };
        let module = module(&device, KERNEL);
        batched(|| {
            let a = pipeline(&device, &module);
            let b = pipeline(&device, &module);
            // Deref compiles on demand: its bind group layout is readable.
            let _ = a.get_bind_group_layout(0);
            assert!(a.is_ready());
            drop(b);
        });
    }

    /// A build that is abandoned compiles nothing more: its queued pipelines
    /// end as failures nobody reads, and the build is not held for them.
    #[test]
    fn an_abandoned_build_compiles_nothing_more() {
        let Some(device) = device() else { return };
        let module = module(&device, KERNEL);
        let token = Arc::new(AtomicBool::new(true));
        set_build_cancel(Some(token));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            batched(|| {
                for _ in 0..4 {
                    drop(pipeline(&device, &module));
                }
            })
        }));
        set_build_cancel(None);
        let reason = result.expect_err("the abandoned pipelines are the batch's failure");
        let reason = reason.downcast_ref::<String>().cloned().unwrap_or_default();
        assert!(reason.contains("abandoned"), "{reason}");
    }

    /// A pipeline the backend rejects fails the batch with the reason.
    #[test]
    fn a_rejected_pipeline_fails_the_batch_with_the_backends_reason() {
        let Some(device) = device() else { return };
        let module = module(&device, KERNEL);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            batched(|| {
                drop(create_compute_pipeline_deferred(
                    &device,
                    wgpu::ComputePipelineDescriptor {
                        label: Some("no such entry point"),
                        layout: None,
                        module: &module,
                        entry_point: Some("does_not_exist"),
                        compilation_options: Default::default(),
                        cache: None,
                    },
                ));
            })
        }));
        let reason = result.expect_err("the batch fails");
        let reason = reason.downcast_ref::<String>().cloned().unwrap_or_default();
        assert!(reason.contains("no such entry point"), "{reason}");
    }
}
