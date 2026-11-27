//! The state every copy of Helio in a process shares.
//!
//! A plugin built as a library links its own copy of this crate, and a
//! `static` in it would exist once per copy: shader hot reload would watch
//! only the host's shaders, two pipeline caches would serve one device, and
//! the host would not count the compiles and uploads the library starts. So
//! the state that must be one per process is one value, [`Shared`], owned by
//! the host's copy. A library's copy points at it
//! ([`attach`], through the `_plugin_attach_helio` entry point that
//! [`export_runtime_attach!`](crate::export_runtime_attach) exports, called by
//! the plugin loader before the library runs any code of its own), and
//! installs the host's task host, so the threads and jobs the library starts
//! are the host's tasks and not raw threads nobody lists.
//!
//! This is the only `static` here that matters: the pointer to [`Shared`].
//! `tests/runtime_statics.rs` fails when this crate or `helio-tasks` gains a
//! `static`, `thread_local!` or `inventory::collect!` that is not in its
//! allowed list (the list says why each is harmless per copy).

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, AtomicUsize};
use std::sync::{mpsc, Mutex, OnceLock};

use crate::graph::DriverCacheRecord;

/// What is one per process, whichever copy of Helio asks.
#[derive(Default)]
pub(crate) struct Shared {
    /// The shader hot reload registry and watcher.
    #[cfg(all(feature = "shader-hot-reload", not(target_arch = "wasm32")))]
    pub(crate) hot: crate::shader::hot::HotShared,
    /// The driver pipeline caches, by device address.
    pub(crate) driver_caches: Mutex<HashMap<usize, DriverCacheRecord>>,
    /// Pipeline compile jobs started and not finished.
    pub(crate) compiles_in_flight: AtomicUsize,
    // Read only by debug builds' upload report; kept in release builds so both
    // copies of this struct have one layout whichever profile built them.
    /// Bytes uploaded in the frame so far.
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    pub(crate) upload_bytes: AtomicU64,
    /// The debug upload report's channel.
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    pub(crate) upload_reporter: OnceLock<mpsc::Sender<u64>>,
}

/// This copy's pointer to the process's [`Shared`]: its own until a host
/// attaches it to the host's.
static SHARED: OnceLock<&'static Shared> = OnceLock::new();

/// The process's shared state. The first call in a copy that was not attached
/// makes it (the host's copy, or a test); a library is attached before it can
/// call this.
pub(crate) fn shared() -> &'static Shared {
    SHARED.get_or_init(|| Box::leak(Box::<Shared>::default()))
}

/// The symbol a library exports ([`export_runtime_attach!`](crate::export_runtime_attach)).
pub const ATTACH_SYMBOL: &str = "_plugin_attach_helio";

/// Its signature: the host's [`HostHandle`], whether the library took it.
pub type AttachFn = unsafe extern "C" fn(*const c_void) -> bool;

/// What the host hands a library: the shared state and the task host.
pub struct HostHandle {
    layout: u64,
    shared: &'static Shared,
    tasks: &'static dyn helio_tasks::TaskHost,
}

/// A fingerprint of what two copies must agree on for [`Shared`] to mean the
/// same thing in both: its size and alignment, and the features that change
/// its fields.
fn layout() -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut eat = |value: u64| {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    eat(std::mem::size_of::<Shared>() as u64);
    eat(std::mem::align_of::<Shared>() as u64);
    eat(cfg!(all(feature = "shader-hot-reload", not(target_arch = "wasm32"))) as u64);
    // Bump when `Shared`'s fields change meaning without changing size.
    eat(1);
    hash
}

/// The host's handle: its shared state and the task host it runs (the
/// installed one, else plain threads). Leak it, a library keeps it for the
/// process: plugins are never unloaded.
pub fn host_handle() -> HostHandle {
    HostHandle {
        layout: layout(),
        shared: shared(),
        tasks: helio_tasks::host(),
    }
}

/// Hands this copy's state and task host to a loaded library through its
/// entry point `attach`
/// ([`export_runtime_attach!`](crate::export_runtime_attach)), and returns
/// what the library answered. Call it before anything else of the library
/// runs.
///
/// # Safety
///
/// `attach` is that entry point of a library built from the same sources,
/// loaded for as long as it runs Helio code.
pub unsafe fn share_with(attach: AttachFn) -> bool {
    // Leaked: the library keeps it.
    let handle: &'static HostHandle = Box::leak(Box::new(host_handle()));
    // SAFETY: the caller's contract; the handle lives for the process.
    unsafe { attach((handle as *const HostHandle).cast()) }
}

/// Forwards to the host's task host, which lives for the process.
struct ForwardTasks(&'static dyn helio_tasks::TaskHost);

impl helio_tasks::TaskHost for ForwardTasks {
    fn spawn_service(
        &self,
        spec: helio_tasks::ServiceSpec,
        work: helio_tasks::ServiceWork,
    ) -> std::io::Result<helio_tasks::Service> {
        self.0.spawn_service(spec, work)
    }

    fn submit(&self, spec: helio_tasks::JobSpec, work: helio_tasks::JobWork) {
        self.0.submit(spec, work);
    }
}

/// Points this copy at the host's state and task host.
///
/// Returns `false`, leaving the copy on its own state, when `host` is of
/// another layout, or this copy already ran code that made its own state
/// (attach before anything else in the library runs).
///
/// # Safety
///
/// `host` is a [`HostHandle`] from [`host_handle`], alive for the process
/// (leaked), built by a copy of this crate from the same sources and
/// compiler.
pub unsafe fn attach(host: *const c_void) -> bool {
    if host.is_null() {
        return false;
    }
    // SAFETY: the caller's contract.
    let host = unsafe { &*host.cast::<HostHandle>() };
    if host.layout != layout() {
        return false;
    }
    if SHARED.set(host.shared).is_err() && !std::ptr::eq(shared(), host.shared) {
        return false;
    }
    // A host without a task system hands plain threads: the library already
    // has those, so it installs nothing.
    let _ = helio_tasks::install(std::sync::Arc::new(ForwardTasks(host.tasks)));
    true
}

/// Exports `_plugin_attach_helio` from a library, so the plugin loader can
/// attach this copy of Helio to the host's. Call it once in the crate built
/// as the library.
#[macro_export]
macro_rules! export_runtime_attach {
    () => {
        /// Attach this library's Helio to the host's shared state.
        ///
        /// # Safety
        ///
        /// `host` is the host's `helio_core::runtime::HostHandle`.
        #[no_mangle]
        pub unsafe extern "C" fn _plugin_attach_helio(host: *const ::std::ffi::c_void) -> bool {
            ::std::panic::catch_unwind(|| {
                // SAFETY: the caller's contract.
                unsafe { $crate::runtime::attach(host) }
            })
            .unwrap_or(false)
        }
    };
}
