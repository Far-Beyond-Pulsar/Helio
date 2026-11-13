//! Threads and background jobs that Helio-layer code starts.
//!
//! A host owns its threads: the engine lists every one in its task system
//! (name, role, owner, heartbeat, stalls), which Helio and the plugins' feature
//! and world crates cannot depend on. They start their threads here instead,
//! and the host decides how they run:
//!
//! - [`spawn_service`]: a thread that lives as long as what it serves (a
//!   planner fed by a channel, a pool of query workers). Its loop calls
//!   [`TaskContext::heartbeat`] each pass and waits for work inside
//!   [`TaskContext::wait`], so a host can tell a stalled loop from an idle one.
//! - [`submit`]: one piece of blocking work (a pipeline compile).
//!
//! The engine installs its host once at startup ([`install`]), the moment it
//! installs rayon's global pool. Without one (Helio's own examples and tests)
//! the work runs on named std threads ([`StdThreads`]). One host per process,
//! like rayon's pool and tracing's subscriber: threads are a process resource,
//! and the code that starts them (a voxel world's query workers) often runs
//! outside any renderer.

use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

/// The category a host files Helio-layer work under unless the spec says
/// otherwise.
pub const RENDER: &str = "render";

/// How long a service may stay silent (not idle) before a host reports it
/// stalled, unless its spec says otherwise.
pub const DEFAULT_HEARTBEAT: Duration = Duration::from_secs(5);

/// What a thread is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Role {
    /// A long-lived loop serving something (a planner, a builder).
    Service,
    /// A worker of a data-parallel pool.
    Compute,
    /// Blocking I/O.
    Io,
}

/// A thread that lives as long as what it serves.
#[derive(Clone, Debug)]
pub struct ServiceSpec {
    /// What a host shows for it ("Voxel residency planner").
    pub title: String,
    /// The task category (see [`RENDER`]).
    pub category: &'static str,
    /// The OS thread's name.
    pub thread: String,
    /// The subsystem that owns it ("voxel").
    pub owner: String,
    pub role: Role,
    /// The pool it belongs to, for a set of identical workers.
    pub pool: Option<String>,
    /// Longest silence while not idle before it counts as stalled; `None`
    /// for a loop whose work has no bound (it names its phases instead).
    pub heartbeat: Option<Duration>,
    pub stack_size: Option<usize>,
}

impl ServiceSpec {
    pub fn new(title: impl Into<String>, thread: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            category: RENDER,
            thread: thread.into(),
            owner: String::new(),
            role: Role::Service,
            pool: None,
            heartbeat: Some(DEFAULT_HEARTBEAT),
            stack_size: None,
        }
    }

    pub fn owner(mut self, owner: impl Into<String>) -> Self {
        self.owner = owner.into();
        self
    }

    pub fn category(mut self, category: &'static str) -> Self {
        self.category = category;
        self
    }

    pub fn role(mut self, role: Role) -> Self {
        self.role = role;
        self
    }

    pub fn pool(mut self, pool: impl Into<String>) -> Self {
        self.pool = Some(pool.into());
        self
    }

    pub fn heartbeat(mut self, every: Duration) -> Self {
        self.heartbeat = Some(every);
        self
    }

    pub fn no_heartbeat(mut self) -> Self {
        self.heartbeat = None;
        self
    }

    pub fn stack_size(mut self, bytes: usize) -> Self {
        self.stack_size = Some(bytes);
        self
    }
}

/// One piece of blocking work.
#[derive(Clone, Debug)]
pub struct JobSpec {
    /// What a host shows for it ("Compile voxel draw pipelines").
    pub title: String,
    pub category: &'static str,
    /// The subsystem that owns it.
    pub owner: String,
}

impl JobSpec {
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            category: RENDER,
            owner: String::new(),
        }
    }

    pub fn owner(mut self, owner: impl Into<String>) -> Self {
        self.owner = owner.into();
        self
    }

    pub fn category(mut self, category: &'static str) -> Self {
        self.category = category;
        self
    }
}

/// What a host learns from the work it runs. A host implements it over its
/// own task record.
pub trait ContextHooks: Send + Sync {
    fn heartbeat(&self);
    fn set_idle(&self, idle: bool);
    fn set_phase(&self, phase: &str);
    /// Whether the host asked the work to stop.
    fn stop_requested(&self) -> bool;
}

/// The running work's line to its host.
pub struct TaskContext {
    hooks: Arc<dyn ContextHooks>,
}

impl TaskContext {
    pub fn new(hooks: Arc<dyn ContextHooks>) -> Self {
        Self { hooks }
    }

    /// One pass of a service's loop is done.
    pub fn heartbeat(&self) {
        self.hooks.heartbeat();
    }

    /// The work waits on purpose (silence is then not a stall).
    pub fn set_idle(&self, idle: bool) {
        self.hooks.set_idle(idle);
    }

    /// Names what the work does now.
    pub fn set_phase(&self, phase: &str) {
        self.hooks.set_phase(phase);
    }

    /// Whether the host asked the work to stop.
    pub fn stop_requested(&self) -> bool {
        self.hooks.stop_requested()
    }

    /// Waits for work with `wait` (a channel's `recv`): idle meanwhile, a
    /// heartbeat when it returns.
    pub fn wait<T>(&self, wait: impl FnOnce() -> T) -> T {
        self.set_idle(true);
        let got = wait();
        self.set_idle(false);
        self.heartbeat();
        got
    }
}

/// A host's handle on a service's thread.
pub trait ServiceJoin: Send + Sync {
    fn is_finished(&self) -> bool;
    fn request_stop(&self);
    fn join(self: Box<Self>) -> std::thread::Result<()>;
}

/// A running service. Dropping it detaches the thread, as dropping a
/// `JoinHandle` does: services end when what they serve drops their channel.
pub struct Service {
    join: Box<dyn ServiceJoin>,
}

impl Service {
    pub fn new(join: Box<dyn ServiceJoin>) -> Self {
        Self { join }
    }

    pub fn is_finished(&self) -> bool {
        self.join.is_finished()
    }

    /// Asks the loop to stop; it obeys by watching
    /// [`TaskContext::stop_requested`].
    pub fn request_stop(&self) {
        self.join.request_stop();
    }

    pub fn join(self) -> std::thread::Result<()> {
        self.join.join()
    }
}

/// A service's loop.
pub type ServiceWork = Box<dyn FnOnce(&TaskContext) + Send>;
/// A job's work.
pub type JobWork = Box<dyn FnOnce(&TaskContext) + Send>;

/// How a host runs Helio-layer threads and jobs.
pub trait TaskHost: Send + Sync {
    fn spawn_service(&self, spec: ServiceSpec, work: ServiceWork) -> io::Result<Service>;
    fn submit(&self, spec: JobSpec, work: JobWork);
}

/// Named std threads: the work of a process that installed no host.
pub struct StdThreads;

#[derive(Default)]
struct StdHooks {
    stop: AtomicBool,
}

impl ContextHooks for StdHooks {
    fn heartbeat(&self) {}
    fn set_idle(&self, _idle: bool) {}
    fn set_phase(&self, _phase: &str) {}
    fn stop_requested(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }
}

struct StdJoin {
    handle: std::thread::JoinHandle<()>,
    hooks: Arc<StdHooks>,
}

impl ServiceJoin for StdJoin {
    fn is_finished(&self) -> bool {
        self.handle.is_finished()
    }

    fn request_stop(&self) {
        self.hooks.stop.store(true, Ordering::Release);
    }

    fn join(self: Box<Self>) -> std::thread::Result<()> {
        self.handle.join()
    }
}

impl StdThreads {
    fn start(
        name: String,
        stack_size: Option<usize>,
        work: Box<dyn FnOnce(&TaskContext) + Send>,
    ) -> io::Result<StdJoin> {
        let hooks = Arc::new(StdHooks::default());
        let context = TaskContext::new(hooks.clone());
        let mut builder = std::thread::Builder::new().name(name);
        if let Some(bytes) = stack_size {
            builder = builder.stack_size(bytes);
        }
        let handle = builder.spawn(move || work(&context))?;
        Ok(StdJoin { handle, hooks })
    }
}

impl TaskHost for StdThreads {
    fn spawn_service(&self, spec: ServiceSpec, work: ServiceWork) -> io::Result<Service> {
        let join = Self::start(spec.thread, spec.stack_size, work)?;
        Ok(Service::new(Box::new(join)))
    }

    fn submit(&self, spec: JobSpec, work: JobWork) {
        // Like `std::thread::spawn`, a process that cannot start a thread
        // cannot go on.
        if let Err(error) = Self::start(format!("helio job: {}", spec.title), None, work) {
            panic!("no thread for the job \"{}\": {error}", spec.title);
        }
    }
}

static HOST: OnceLock<Arc<dyn TaskHost>> = OnceLock::new();

/// Makes `host` run every thread and job Helio-layer code starts from now on.
/// Once per process, before such work starts; a second call returns the host
/// it was given.
pub fn install(host: Arc<dyn TaskHost>) -> Result<(), Arc<dyn TaskHost>> {
    HOST.set(host)
}

/// Whether a host was installed ([`install`]).
pub fn is_installed() -> bool {
    HOST.get().is_some()
}

/// The installed host, or [`StdThreads`].
pub fn host() -> &'static dyn TaskHost {
    match HOST.get() {
        Some(host) => &**host,
        None => &StdThreads,
    }
}

/// Starts a service ([`TaskHost::spawn_service`]) on the host.
pub fn spawn_service(
    spec: ServiceSpec,
    work: impl FnOnce(&TaskContext) + Send + 'static,
) -> io::Result<Service> {
    host().spawn_service(spec, Box::new(work))
}

/// Runs a job ([`TaskHost::submit`]) on the host.
pub fn submit(spec: JobSpec, work: impl FnOnce(&TaskContext) + Send + 'static) {
    host().submit(spec, Box::new(work));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    fn a_service_waits_for_work_and_ends_with_its_channel() {
        let (jobs, inbox) = mpsc::channel::<u32>();
        let (outbox, results) = mpsc::channel();
        let service = spawn_service(ServiceSpec::new("doubler", "doubler"), move |ctx| {
            while let Ok(n) = ctx.wait(|| inbox.recv()) {
                outbox.send(n * 2).unwrap();
            }
        })
        .unwrap();
        jobs.send(21).unwrap();
        assert_eq!(results.recv().unwrap(), 42);
        drop(jobs);
        service.join().unwrap();
    }

    #[test]
    fn a_stop_request_reaches_the_loop() {
        let service = StdThreads
            .spawn_service(
                ServiceSpec::new("spinner", "spinner"),
                Box::new(|ctx: &TaskContext| {
                    while !ctx.stop_requested() {
                        std::thread::yield_now();
                    }
                }),
            )
            .unwrap();
        service.request_stop();
        service.join().unwrap();
    }

    #[test]
    fn a_job_runs() {
        let (done, finished) = mpsc::channel();
        submit(JobSpec::new("answer"), move |_| done.send(42).unwrap());
        assert_eq!(finished.recv().unwrap(), 42);
    }
}
