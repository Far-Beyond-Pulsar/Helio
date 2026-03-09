//! GPU + CPU frame timing profiler — scope-macro driven.
//!
//! Uses `wgpu::QueryType::Timestamp` (requires `Features::TIMESTAMP_QUERY`)
//! to bracket each render/compute pass in the graph with GPU timestamps.
//! Results are read back with a 1-frame delay via double-buffered staging
//! buffers — no GPU stall on the hot path.
//!
//! # Profiling Model
//!
//! All CPU timing is captured via the [`profile_scope!`] macro which creates
//! an RAII guard.  Nested scopes automatically form a tree — the macro pushes
//! onto a thread-local stack on creation and pops on drop.  Completed scopes
//! are dispatched over a channel to a background collector thread with **zero**
//! allocation or locking on the main thread.
//!
//! GPU pass timestamps are recorded by the render graph and merged into the
//! same tree by the collector.
//!
//! The collector thread builds per-frame timing trees and forwards them to the
//! live portal (WebSocket) as delta-compressed snapshots.

use std::cell::RefCell;
use std::sync::{Arc, OnceLock, atomic::{AtomicBool, Ordering}};

/// Maximum pass-level scopes per frame.
/// Each scope uses 2 query slots (begin + end).
const MAX_SCOPES: u32 = 192;
const SLOT_COUNT: u32 = MAX_SCOPES * 2;
const SLOT_BYTES: u64 = SLOT_COUNT as u64 * 8; // each timestamp is a u64

// ═══════════════════════════════════════════════════════════════════════════════
// PROFILE SCOPE MACRO + RAII GUARD
// ═══════════════════════════════════════════════════════════════════════════════

/// Create a profiling scope that automatically measures the enclosing block.
///
/// Nested `profile_scope!` invocations form a tree — inner scopes become
/// children of the nearest outer scope on the same thread.  The timing is
/// sent to a background collector on drop with zero heap allocation.
///
/// ```rust,ignore
/// profile_scope!("render");
/// {
///     profile_scope!("prep");
///     // ... prep work ...
/// }
/// {
///     profile_scope!("graph");
///     // ... graph execution ...
/// }
/// ```
#[macro_export]
macro_rules! profile_scope {
    ($name:expr) => {
        // Single relaxed atomic load — branch-predictor-friendly on steady state.
        // When the portal is disconnected this is a complete no-op: no allocation,
        // no thread-local access, no timestamp.
        let _prof_guard = $crate::profiler::profiling_active()
            .then(|| $crate::profiler::ScopeGuard::new($name));
    };
}

// ── Thread-local scope stack ─────────────────────────────────────────────────

/// Index into the per-frame scope list, used to track parent relationships.
type ScopeIdx = u32;

thread_local! {
    /// Stack of currently-open scope indices on this thread.
    /// Push on `ScopeGuard::new`, pop on `ScopeGuard::drop`.
    static SCOPE_STACK: RefCell<Vec<ScopeIdx>> = RefCell::new(Vec::with_capacity(32));
    /// Flat ordered log of every CPU scope that completes on this thread.
    /// Consumed once per frame by `take_frame_scope_log()`.
    static FRAME_SCOPE_LOG: RefCell<Vec<(&'static str, f32)>> =
        RefCell::new(Vec::with_capacity(64));
}

/// Drain and return all CPU scope timings recorded since the last call.
///
/// Call this once per frame after all render work has finished.  The returned
/// vec is in completion order (innermost scopes appear before outer ones since
/// they drop first).
pub fn take_frame_scope_log() -> Vec<(&'static str, f32)> {
    FRAME_SCOPE_LOG.with(|log| std::mem::take(&mut *log.borrow_mut()))
}

/// An event dispatched from the main thread to the collector.
pub(crate) enum ProfileEvent {
    /// A CPU-timed scope completed.
    Scope {
        name: &'static str,
        parent: Option<ScopeIdx>,
        idx: ScopeIdx,
        elapsed_ms: f32,
    },
    /// GPU pass timings for the current frame (from GpuProfiler readback).
    GpuTimings(Vec<PassTiming>),
    /// Marks the end of a logical frame — collector finalises the tree.
    FrameEnd {
        frame: u64,
        frame_time_ms: f32,
        frame_to_frame_ms: f32,
        total_gpu_ms: f32,
        total_cpu_ms: f32,
    },
}

/// Global channel sender, initialised on first use.
static PROFILE_TX: OnceLock<std::sync::mpsc::Sender<ProfileEvent>> = OnceLock::new();

/// Set to `true` when a live portal is connected, `false` otherwise.
/// `profile_scope!` checks this atomically and becomes a complete no-op
/// when inactive — zero cost on the hot render path.
static PROFILING_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Activate or deactivate CPU scope recording.
/// Call with `true` when a portal client connects, `false` on disconnect.
pub fn set_profiling_active(active: bool) {
    PROFILING_ACTIVE.store(active, Ordering::Relaxed);
}

/// Returns `true` if scope recording is currently active.
#[inline(always)]
pub fn profiling_active() -> bool {
    PROFILING_ACTIVE.load(Ordering::Relaxed)
}

/// Counter for scope indices within a frame.  Reset each frame by FrameEnd.
static SCOPE_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Initialise the profiling background thread and channel.
/// Safe to call multiple times — only the first call starts the thread.
/// Returns a clone of the sender.
pub(crate) fn init_profile_thread() -> std::sync::mpsc::Sender<ProfileEvent> {
    PROFILE_TX.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<ProfileEvent>();
        std::thread::Builder::new()
            .name("helio-profiler".to_string())
            .spawn(move || {
                collector_loop(rx);
            })
            .expect("failed to spawn profiler collector thread");
        tx
    }).clone()
}

/// RAII guard created by [`profile_scope!`].
pub struct ScopeGuard {
    name: &'static str,
    idx: ScopeIdx,
    parent: Option<ScopeIdx>,
    start: std::time::Instant,
}

impl ScopeGuard {
    #[inline]
    pub fn new(name: &'static str) -> Self {
        let idx = SCOPE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let parent = SCOPE_STACK.with(|s| {
            let mut stack = s.borrow_mut();
            let parent = stack.last().copied();
            stack.push(idx);
            parent
        });
        Self {
            name,
            idx,
            parent,
            start: std::time::Instant::now(),
        }
    }
}

impl Drop for ScopeGuard {
    #[inline]
    fn drop(&mut self) {
        let elapsed_ms = self.start.elapsed().as_secs_f32() * 1000.0;
        SCOPE_STACK.with(|s| {
            s.borrow_mut().pop();
        });
        // Write to thread-local log — zero allocation, no locking.
        FRAME_SCOPE_LOG.with(|log| {
            log.borrow_mut().push((self.name, elapsed_ms));
        });
        if let Some(tx) = PROFILE_TX.get() {
            let _ = tx.send(ProfileEvent::Scope {
                name: self.name,
                parent: self.parent,
                idx: self.idx,
                elapsed_ms,
            });
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COLLECTOR THREAD
// ═══════════════════════════════════════════════════════════════════════════════

/// A completed scope node in the per-frame tree.
struct ScopeNode {
    name: &'static str,
    parent: Option<ScopeIdx>,
    elapsed_ms: f32,
}

fn collector_loop(rx: std::sync::mpsc::Receiver<ProfileEvent>) {
    let mut frame_scopes: Vec<ScopeNode> = Vec::with_capacity(64);
    let mut frame_gpu_timings: Vec<PassTiming> = Vec::new();

    loop {
        let event = match rx.recv() {
            Ok(e) => e,
            Err(_) => break, // channel closed
        };

        match event {
            ProfileEvent::Scope { name, parent, idx: _, elapsed_ms } => {
                frame_scopes.push(ScopeNode { name, parent, elapsed_ms });
            }
            ProfileEvent::GpuTimings(timings) => {
                frame_gpu_timings = timings;
            }
            ProfileEvent::FrameEnd { .. } => {
                // TODO: Build tree, compute deltas, forward to portal.
                // For now just clear per-frame state.
                frame_scopes.clear();
                frame_gpu_timings.clear();
                SCOPE_COUNTER.store(0, Ordering::Relaxed);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GPU TIMESTAMP PROFILER (kept for per-pass GPU timing)
// ═══════════════════════════════════════════════════════════════════════════════

/// Timing result for one render/compute pass.
#[derive(Clone, Debug)]
pub struct PassTiming {
    /// Label supplied to `begin_render_pass` / `begin_compute_pass` (the pass name)
    pub name: String,
    /// Wall-clock GPU execution time in milliseconds (0 if not available yet)
    pub gpu_ms: f32,
    /// CPU time from just before to just after `pass.execute()`, in milliseconds.
    pub cpu_ms: f32,
}

// ─────────────────────────────────────────────────────────────────────────────

/// Per-scope bookkeeping while a frame is being recorded.
pub(crate) struct ScopeRecord {
    pub name:       String,
    pub begin_slot: u32,
    pub end_slot:   u32,
    pub cpu_ms:     f32,
}

/// State saved between submission and readback for one frame.
struct PendingFrame {
    scopes:     Vec<ScopeRecord>,
    used_slots: u32,
    ready:      Arc<AtomicBool>,
}

// ─────────────────────────────────────────────────────────────────────────────

/// GPU timestamp profiler.  Created once; held by the `Renderer`.
///
/// CPU-side timing is now handled entirely by [`profile_scope!`] and the
/// collector thread.  This struct only manages GPU query sets and readback.
pub struct GpuProfiler {
    query_set:         wgpu::QuerySet,
    resolve_buf:       wgpu::Buffer, // QUERY_RESOLVE | COPY_SRC
    staging_bufs:      [wgpu::Buffer; 2],
    timestamp_period:  f64, // nanoseconds per GPU tick

    // Current-frame state
    pub(crate) current_scopes: Vec<ScopeRecord>,
    next_slot:     u32,
    write_idx:     usize, // which staging buf we write into this frame

    // In-flight frames waiting for staging buffer readback
    pending: [Option<PendingFrame>; 2],

    // True when resolve() successfully staged data this frame.
    needs_readback: [bool; 2],

    /// Most recent successfully resolved timings (one entry per pass).
    pub last_timings:  Vec<PassTiming>,
    /// Sum of all per-pass GPU times from `last_timings` (ms).
    pub last_total_gpu_ms: f32,
    /// Sum of all per-pass CPU times from `last_timings` (ms).
    pub last_total_cpu_ms: f32,
    /// Wall-clock frame time from last render (ms).
    pub last_frame_time_ms: f32,
    /// Wall-clock time from start of last frame to start of this frame (ms).
    pub last_frame_to_frame_ms: f32,
}

impl GpuProfiler {
    /// Try to create a profiler.  Returns `None` if the device does not have
    /// `Features::TIMESTAMP_QUERY`, which means the caller never requested it.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            log::info!("GpuProfiler: TIMESTAMP_QUERY not available — CPU timings only");
            return None;
        }

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Profiler QuerySet"),
            ty:    wgpu::QueryType::Timestamp,
            count: SLOT_COUNT,
        });

        let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Profiler Resolve Buffer"),
            size:  SLOT_BYTES,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_bufs = std::array::from_fn(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Profiler Staging[{i}]")),
                size:  SLOT_BYTES,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        let timestamp_period = queue.get_timestamp_period() as f64;
        log::info!(
            "GpuProfiler: GPU timestamp period = {:.3} ns/tick",
            timestamp_period
        );

        Some(Self {
            query_set,
            resolve_buf,
            staging_bufs,
            timestamp_period,
            current_scopes: Vec::with_capacity(32),
            next_slot:         0,
            write_idx:         0,
            pending:           [None, None],
            needs_readback:    [false, false],
            last_timings:      Vec::new(),
            last_total_gpu_ms: 0.0,
            last_total_cpu_ms: 0.0,
            last_frame_time_ms: 0.0,
            last_frame_to_frame_ms: 0.0,
        })
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Allocate a (begin_slot, end_slot) pair and register a scope for this frame.
    /// Returns `None` when the slot pool is exhausted (should never happen in practice).
    pub(crate) fn allocate_scope(&mut self, name: &str) -> Option<(u32, u32)> {
        let begin = self.next_slot;
        let end   = begin + 1;
        if end >= SLOT_COUNT {
            log::warn!("GpuProfiler: slot pool full, skipping scope '{name}'");
            return None;
        }
        self.current_scopes.push(ScopeRecord {
            name:       name.to_string(),
            begin_slot: begin,
            end_slot:   end,
            cpu_ms:     0.0,
        });
        self.next_slot = end + 1;
        Some((begin, end))
    }

    /// Update the CPU time for the most recently allocated scope.
    pub(crate) fn set_last_scope_cpu_ms(&mut self, cpu_ms: f32) {
        if let Some(s) = self.current_scopes.last_mut() {
            s.cpu_ms = cpu_ms;
        }
    }

    /// Update the wall-clock frame time (in milliseconds).
    pub fn set_frame_time_ms(&mut self, frame_time_ms: f32) {
        self.last_frame_time_ms = frame_time_ms;
    }

    /// Update the frame-to-frame time (time from start of last frame to start of this frame) in milliseconds.
    pub fn set_frame_to_frame_ms(&mut self, frame_to_frame_ms: f32) {
        self.last_frame_to_frame_ms = frame_to_frame_ms;
    }

    /// Reference to the underlying QuerySet (needed by the graph to call write_timestamp).
    pub(crate) fn query_set(&self) -> &wgpu::QuerySet {
        &self.query_set
    }

    // ── Call-order: begin_frame → (allocate/write per pass) → resolve → submit → poll ──

    /// Reset per-frame state.  Call before the graph begins recording.
    pub fn begin_frame(&mut self) {
        self.current_scopes.clear();
        self.next_slot = 0;
    }

    /// Resolve all used query slots into the staging buffer for this frame.
    /// **Must be called on the same encoder before `encoder.finish()`.**
    /// After calling this, call `begin_readback()` once the encoder has been submitted.
    pub fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let used = self.next_slot;
        if used == 0 {
            return;
        }

        let w = self.write_idx;

        // Guard: if this staging buffer slot still has a pending readback from
        // 2 frames ago (the callback fired but we already have a new frame's
        // data, or it hasn't fired yet), unmap it now so the COPY_DST usage is
        // valid. If the map hasn't resolved yet we have to skip this frame.
        if let Some(pending) = self.pending[w].take() {
            if pending.ready.load(Ordering::Acquire) {
                self.staging_bufs[w].unmap();
            } else {
                // Buffer is still in a pending-map state — we cannot write to
                // it. Put the entry back and bail for this frame.
                self.pending[w] = Some(pending);
                log::warn!(
                    "GpuProfiler: staging buf[{w}] still pending, skipping profiling frame"
                );
                return;
            }
        }

        encoder.resolve_query_set(&self.query_set, 0..used, &self.resolve_buf, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve_buf, 0,
            &self.staging_bufs[w], 0,
            used as u64 * 8,
        );

        // Save scope metadata so we know how to interpret this staging buffer.
        // map_async is intentionally NOT called here — it must be called after
        // queue.submit() or wgpu validation will reject the submit because the
        // buffer is in a pending-map state while used in an encoder command.
        let ready = Arc::new(AtomicBool::new(false));
        self.pending[w] = Some(PendingFrame {
            scopes:     std::mem::take(&mut self.current_scopes),
            used_slots: used,
            ready,
        });
        // Signal that begin_readback() should map this slot.
        self.needs_readback[w] = true;
    }

    /// Schedule async readback of this frame's staging buffer.
    /// **Must be called AFTER `queue.submit()`** (i.e. after the encoder that
    /// contained the `copy_buffer_to_buffer` has been submitted).
    /// Safe to call every frame — no-ops if resolve() bailed this frame.
    pub fn begin_readback(&mut self) {
        let w = self.write_idx;
        if !self.needs_readback[w] {
            return;
        }
        self.needs_readback[w] = false;
        if let Some(pending) = &self.pending[w] {
            let used_bytes = pending.used_slots as u64 * 8;
            let ready_clone = pending.ready.clone();
            self.staging_bufs[w]
                .slice(..used_bytes)
                .map_async(wgpu::MapMode::Read, move |result| {
                    if result.is_ok() {
                        ready_clone.store(true, Ordering::Release);
                    }
                });
        }
    }

    /// Attempt to read back the *previous* frame's results (non-blocking).
    /// Call after `queue.submit()` so the driver has a chance to process callbacks.
    /// Updates `last_timings` and `last_total_gpu_ms` if data is available.
    pub fn poll_results(&mut self, device: &wgpu::Device) {
        let _ = device.poll(wgpu::PollType::Poll);

        let read_idx = 1 - self.write_idx; // the other buffer
        if let Some(pending) = &self.pending[read_idx] {
            if pending.ready.load(Ordering::Acquire) {
                let used_bytes = pending.used_slots as u64 * 8;
                let raw = self.staging_bufs[read_idx].slice(..used_bytes).get_mapped_range();
                let timestamps: &[u64] = bytemuck::cast_slice(&*raw);

                let new_timings: Vec<PassTiming> = pending.scopes.iter().filter_map(|s| {
                    let t0 = timestamps.get(s.begin_slot as usize).copied().unwrap_or(0);
                    let t1 = timestamps.get(s.end_slot   as usize).copied().unwrap_or(0);
                    if t0 == 0 || t1 == 0 || t1 < t0 {
                        // GPU didn't write these slots (pass was skipped or slots invalid)
                        return Some(PassTiming {
                            name:   s.name.clone(),
                            gpu_ms: 0.0,
                            cpu_ms: s.cpu_ms,
                        });
                    }
                    let ns = (t1 - t0) as f64 * self.timestamp_period;
                    Some(PassTiming {
                        name:   s.name.clone(),
                        gpu_ms: (ns / 1_000_000.0) as f32,
                        cpu_ms: s.cpu_ms,
                    })
                }).collect();

                // Only sum top-level passes (no '/') so sub-scopes aren't double-counted.
                let total: f32 = new_timings.iter()
                    .filter(|t| !t.name.contains('/'))
                    .map(|t| t.gpu_ms)
                    .sum();
                drop(raw);
                self.staging_bufs[read_idx].unmap();
                self.pending[read_idx] = None;

                let total_cpu = new_timings.iter().map(|t| t.cpu_ms).sum();
                self.last_timings     = new_timings;
                self.last_total_gpu_ms = total;
                self.last_total_cpu_ms = total_cpu;
            }
        }

        // Advance write buffer for next frame
        self.write_idx = 1 - self.write_idx;
    }
}
