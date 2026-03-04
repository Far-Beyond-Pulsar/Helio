//! GPU + CPU frame timing profiler
//!
//! Uses `wgpu::QueryType::Timestamp` (requires `Features::TIMESTAMP_QUERY`)
//! to bracket each render/compute pass in the graph with GPU timestamps.
//! Results are read back with a 1-frame delay via double-buffered staging
//! buffers — no GPU stall on the hot path.
//!
//! If `TIMESTAMP_QUERY` was not requested in the device descriptor,
//! `GpuProfiler::new` returns `None` and the renderer quietly falls back
//! to CPU-only timing.

use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

/// Maximum pass-level scopes per frame.
/// Each scope uses 2 query slots (begin + end).
/// Budget: 6 top-level + 16 lights + 16×6 faces + 6 RC sub-scopes = ~125; 192 gives headroom.
const MAX_SCOPES: u32 = 192;
const SLOT_COUNT: u32 = MAX_SCOPES * 2;
const SLOT_BYTES: u64 = SLOT_COUNT as u64 * 8; // each timestamp is a u64

// ─────────────────────────────────────────────────────────────────────────────

/// Timing result for one render/compute pass.
#[derive(Clone, Debug)]
pub struct PassTiming {
    /// Label supplied to `begin_render_pass` / `begin_compute_pass` (the pass name)
    pub name: String,
    /// Wall-clock GPU execution time in milliseconds (0 if not available yet)
    pub gpu_ms: f32,
    /// CPU time from just before to just after `pass.execute()`, in milliseconds.
    /// Includes driver overhead + submission but NOT GPU execution.
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

/// GPU + CPU profiler.  Created once; held by the `Renderer`.
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

    // True when resolve() successfully staged data this frame.  Cleared by
    // begin_readback() so it only calls map_async once per staging event.
    needs_readback: [bool; 2],

    /// Most recent successfully resolved timings (one entry per pass).
    pub last_timings:  Vec<PassTiming>,
    /// Sum of all per-pass GPU times from `last_timings` (ms).
    pub last_total_gpu_ms: f32,
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

                self.last_timings     = new_timings;
                self.last_total_gpu_ms = total;
            }
        }

        // Advance write buffer for next frame
        self.write_idx = 1 - self.write_idx;
    }

    // ─────────────────────────────────────────────────────────────────────────

    /// Pretty-print timings to stderr as an indented ANSI tree.
    ///
    /// Supports up to 3 levels (0, 1, or 2 slashes in the name):
    /// ```text
    /// [GPU TIMING]
    ///   shadow          39.21 ms gpu
    ///     light_8       10.21 ms gpu
    ///       face_0       0.03 ms gpu
    ///       face_4       9.87 ms gpu   ← the bad one
    /// ```
    pub fn print_timings(&self) {
        if self.last_timings.is_empty() {
            return;
        }

        const RESET:  &str = "\x1b[0m";
        const BOLD:   &str = "\x1b[1m";
        const DIM:    &str = "\x1b[2m";
        const CYAN:   &str = "\x1b[36m";
        const YELLOW: &str = "\x1b[33m";

        eprintln!("{}[GPU TIMING]{}", BOLD, RESET);

        for t in &self.last_timings {
            let depth = t.name.chars().filter(|&c| c == '/').count();
            let display = t.name.rsplitn(2, '/').next().unwrap_or(&t.name);

            let (indent, name_width) = match depth {
                0 => ("  ",     24usize),
                1 => ("    ",   22usize),
                _ => ("      ", 20usize),
            };

            let gpu_col = if t.gpu_ms >= 0.005 { CYAN } else { DIM };
            let gpu_str = format!("{}{:>7.2} ms{} gpu", gpu_col, t.gpu_ms, RESET);

            let cpu_str = if depth == 0 {
                format!("  {}{:>7.2} ms{} cpu", DIM, t.cpu_ms, RESET)
            } else {
                String::new()
            };

            eprintln!("{}{:<name_width$}{}{}",
                indent, display, gpu_str, cpu_str,
                name_width = name_width);
        }

        eprintln!("  {}─────────────────────────────────{}", DIM, RESET);
        eprintln!("  {}{:<24}{}{:>7.2} ms{} gpu{}",
            BOLD, "total", YELLOW, self.last_total_gpu_ms, RESET, RESET);
    }
}
