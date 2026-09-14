//! `scenedb-inspector-agent` -- an opt-in, in-process debug agent that
//! streams a live `pulsar_scenedb::World` snapshot into a shared-memory
//! ring buffer for the standalone `scenedb_inspector` egui client to read.
//!
//! Zero-cost when not launched under the inspector: [`maybe_start`] is a
//! no-op unless [`SHM_ENV_VAR`] is set in the process environment, which
//! only the inspector's own launcher sets on the child process it spawns
//! (see that client's launch-only design -- there is no "attach to an
//! already-running, un-instrumented process" mode).
//!
//! # Usage
//!
//! ```ignore
//! use std::sync::{Arc, Mutex};
//! use std::time::Duration;
//!
//! let world = Arc::new(Mutex::new(pulsar_scenedb::World::new()));
//! let world_for_agent = world.clone();
//! let _agent = scenedb_inspector_agent::maybe_start(Duration::from_millis(100), move || {
//!     world_for_agent.lock().unwrap().telemetry_snapshot()
//! });
//! // `_agent` must stay alive (don't let it drop) for as long as you want
//! // the inspector able to see this process.
//! ```
//!
//! `snapshot_fn` runs on the agent's own background thread, so whatever it
//! closes over must be safe to call from a thread other than the one
//! driving your sim/render loop -- wrapping the `World` (or the larger
//! state that owns it) in a `Mutex`/`RwLock` shared with that thread, as
//! above, is the simplest pattern. This only needs to be cheap enough for
//! `poll_interval`-frequency polling (inspector UI refresh rate), not the
//! hot path.

/// Shared-memory ring-buffer framing. Public so `examples/read_target.rs`
/// (and any other in-process manual reader) can exercise the reader-side
/// API directly; the standalone `scenedb_inspector` client hand-copies this
/// file instead of depending on this crate (see its own docs for why) but
/// should stay byte-for-byte compatible with it.
pub mod ring;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use pulsar_scenedb::WorldSnapshot;

/// Environment variable the inspector's launcher sets on the child process
/// it spawns, naming the shared-memory segment to publish into. Absent in
/// any normally-launched process, so [`maybe_start`] is a plain no-op
/// there -- safe to call unconditionally from any host app's startup path.
pub const SHM_ENV_VAR: &str = "SCENEDB_INSPECTOR_SHM";

/// Shared setup: create (or fail loudly) the shared-memory segment
/// `shm_name` names, and initialize the ring-buffer header in it. Used by
/// both the threaded ([`start`]) and inline ([`InlineAgent::maybe_start`])
/// publishers.
fn create_ring(shm_name: &str) -> Option<(shared_memory::Shmem, ring::RingView)> {
    use shared_memory::ShmemConf;

    let slot_count = ring::DEFAULT_SLOT_COUNT;
    let slot_capacity = ring::DEFAULT_SLOT_CAPACITY;
    let size = ring::region_size(slot_count, slot_capacity);

    // The inspector's launcher always starts the target fresh with a
    // newly-chosen segment name (launch-only design, no attach-to-running
    // support), so this process is always the one creating the segment --
    // reopen-on-collision only matters for the pathological case of two
    // agents racing the same name, which `create` failing is the right
    // signal for regardless.
    let shmem = match ShmemConf::new().os_id(shm_name).size(size).create() {
        Ok(s) => s,
        Err(e) => {
            log::error!("scenedb-inspector-agent: failed to create shared memory '{shm_name}': {e}");
            return None;
        }
    };

    // SAFETY: `create()` just mapped a fresh region this call exclusively
    // owns the initialization of.
    let view = unsafe { ring::RingView::init(shmem.as_ptr(), slot_count, slot_capacity) };

    log::info!(
        "scenedb-inspector-agent: publishing to shared memory '{shm_name}' \
         ({size} bytes, {slot_count} slots x {slot_capacity} bytes)"
    );

    Some((shmem, view))
}

fn publish_snapshot(view: &ring::RingView, next_slot: &mut u32, snapshot: &WorldSnapshot) {
    match serde_json::to_vec(snapshot) {
        Ok(bytes) => {
            if !view.publish(*next_slot, &bytes) {
                log::warn!(
                    "scenedb-inspector-agent: snapshot ({} bytes) exceeds slot capacity \
                     ({} bytes); dropped this tick",
                    bytes.len(),
                    view.slot_capacity(),
                );
            } else {
                *next_slot = (*next_slot + 1) % view.slot_count();
            }
        }
        Err(e) => log::warn!("scenedb-inspector-agent: snapshot serialization failed: {e}"),
    }
}

/// No-thread, no-`Mutex` publisher: call [`InlineAgent::publish`] yourself,
/// on whatever cadence you like, from whatever thread already owns the
/// `World` -- the natural fit for a typical single-threaded winit render
/// loop (e.g. Helio's own examples), where wrapping `World` in a `Mutex`
/// just for a background agent thread would be pure overhead and a real
/// (if small) risk surface for no benefit. Prefer [`maybe_start`]/[`start`]
/// instead for a host that already has its `World` behind a `Mutex`/`RwLock`
/// for other reasons, or that wants snapshots to keep flowing even while
/// its own main loop is blocked on something else.
pub struct InlineAgent {
    shmem: shared_memory::Shmem,
    view: ring::RingView,
    next_slot: u32,
}

impl InlineAgent {
    /// Create the agent if (and only if) [`SHM_ENV_VAR`] is set. Returns
    /// `None` otherwise -- safe to call unconditionally from any host app's
    /// startup path.
    pub fn maybe_start() -> Option<Self> {
        let shm_name = std::env::var(SHM_ENV_VAR).ok()?;
        let (shmem, view) = create_ring(&shm_name)?;
        Some(Self {
            shmem,
            view,
            next_slot: 0,
        })
    }

    /// Publish `snapshot`. Cheap enough to call every frame for a typical
    /// scene (a `memcpy` + a few atomics), but throttling to whatever
    /// cadence the inspector UI actually needs (e.g. every 6th frame at
    /// 60 fps) costs nothing either -- there is no polling loop here to
    /// starve by calling this less often.
    pub fn publish(&mut self, snapshot: &WorldSnapshot) {
        publish_snapshot(&self.view, &mut self.next_slot, snapshot);
    }
}

impl Drop for InlineAgent {
    fn drop(&mut self) {
        // `self.shmem`'s Drop unmaps/destroys the OS-level segment; nothing
        // else to clean up for the inline (no background thread) path.
        let _ = &self.shmem;
    }
}

/// Handle to a running *threaded* agent (see [`start`]/[`maybe_start`]).
/// Dropping it stops the background thread. Keep it alive for as long as
/// you want the inspector able to see this process's SceneDB state.
pub struct InspectorAgent {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Drop for InspectorAgent {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Start the agent if (and only if) [`SHM_ENV_VAR`] is set in this
/// process's environment. Returns `None` otherwise.
///
/// `snapshot_fn` is called roughly every `poll_interval` on the agent's own
/// background thread -- see the module docs for the threading contract.
pub fn maybe_start<F>(poll_interval: Duration, snapshot_fn: F) -> Option<InspectorAgent>
where
    F: Fn() -> WorldSnapshot + Send + 'static,
{
    let shm_name = std::env::var(SHM_ENV_VAR).ok()?;
    Some(start(&shm_name, poll_interval, snapshot_fn))
}

/// Start the agent unconditionally against a specific shared-memory segment
/// name. Prefer [`maybe_start`] in normal host code -- this is exposed
/// directly for tests/manual setups that don't go through the inspector's
/// own launcher.
pub fn start<F>(shm_name: &str, poll_interval: Duration, snapshot_fn: F) -> InspectorAgent
where
    F: Fn() -> WorldSnapshot + Send + 'static,
{
    let stop = Arc::new(AtomicBool::new(false));
    let thread_stop = stop.clone();
    let shm_name = shm_name.to_string();

    let thread = std::thread::Builder::new()
        .name("scenedb-inspector-agent".into())
        .spawn(move || run(&shm_name, poll_interval, snapshot_fn, thread_stop))
        .expect("scenedb-inspector-agent: failed to spawn background thread");

    InspectorAgent {
        stop,
        thread: Some(thread),
    }
}

fn run<F>(shm_name: &str, poll_interval: Duration, snapshot_fn: F, stop: Arc<AtomicBool>)
where
    F: Fn() -> WorldSnapshot,
{
    let Some((shmem, view)) = create_ring(shm_name) else {
        return;
    };

    let mut next_slot = 0u32;
    while !stop.load(Ordering::Relaxed) {
        let snapshot = snapshot_fn();
        publish_snapshot(&view, &mut next_slot, &snapshot);
        std::thread::sleep(poll_interval);
    }

    // `shmem` (and the OS-level mapping/object it owns) drops here, once
    // the loop exits after `stop` is set.
    drop(shmem);
}
