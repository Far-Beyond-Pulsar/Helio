//! GPU profiling with timestamp queries.
//!
//! This module provides automatic GPU profiling using **timestamp queries**. Timestamps are
//! written at the start and end of each pass, then read back asynchronously to measure GPU time.
//!
//! # Design Pattern: Timestamp Queries
//!
//! GPU profiling uses **wgpu timestamp queries**:
//!
//! 1. Create a query set with N timestamps (e.g., 256 for 128 passes)
//! 2. Write timestamp at pass start (`encoder.write_timestamp(query_set, start_index)`)
//! 3. Write timestamp at pass end (`encoder.write_timestamp(query_set, end_index)`)
//! 4. Read back timestamps asynchronously (future: async buffer mapping)
//! 5. Calculate delta to get GPU time
//!
//! # Performance
//!
//! - **O(1)**: Writing a timestamp is ~10ns (single GPU command)
//! - **Zero allocations**: Query set is pre-allocated
//! - **Zero cost when disabled**: Feature flag eliminates all queries
//!
//! # Async Readback (Future)
//!
//! Timestamp readback is asynchronous to avoid GPU stalls:
//!
//! ```text
//! Frame N:
//!   Write timestamps -> Submit to GPU
//! Frame N+1:
//!   Map buffer -> Read timestamps -> Record timings
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! # use helio_core::profiling::GpuProfiler;
//! let mut profiler = GpuProfiler::new(&device, &queue);
//!
//! // Write start timestamp
//! profiler.begin_pass(&mut encoder, "ShadowPass");
//!
//! // GPU commands...
//!
//! // Write end timestamp
//! profiler.end_pass(&mut encoder, "ShadowPass");
//! ```

/// GPU profiler using timestamp queries.
///
/// `GpuProfiler` measures GPU time by writing timestamps at the start and end of each pass.
/// Timestamps are read back asynchronously to avoid GPU stalls.
///
/// # Design
///
/// The profiler maintains a query set with N timestamps (e.g., 256 for 128 passes).
/// Each pass uses two query slots: one for start, one for end.
///
/// # Performance
///
/// - **O(1)**: Writing a timestamp is ~10ns (single GPU command)
/// - **Zero allocations**: Query set is pre-allocated
/// - **Async readback**: Timestamps read N frames later (no GPU stalls)
///
/// # Example
///
/// ```rust,no_run
/// # use helio_core::profiling::GpuProfiler;
/// let mut profiler = GpuProfiler::new(&device, &queue);
///
/// profiler.begin_pass(&mut encoder, "ShadowPass");
/// // GPU commands...
/// profiler.end_pass(&mut encoder, "ShadowPass");
/// ```
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

const QUERY_CAPACITY: u32 = 256;
const READBACK_SLOT_COUNT: usize = 3;

type QueryRange = (&'static str, u32, u32);

enum ReadbackState {
    Idle,
    CopySubmitted,
    Mapping(Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>>),
}

struct ReadbackSlot {
    buffer: wgpu::Buffer,
    state: ReadbackState,
    queries: Vec<QueryRange>,
    frame_index: u64,
}

pub struct GpuProfiler {
    query_set: Option<wgpu::QuerySet>,
    query_buffer: Option<wgpu::Buffer>,
    readback_slots: Vec<ReadbackSlot>,
    pending_queries: VecDeque<QueryRange>,
    next_index: u32,
    last_timings: Vec<GpuTimestamp>,
    last_completed_frame: Option<u64>,
    dropped_readbacks: u64,
    query_overflows: u64,
    timestamp_period: f32, // Nanoseconds per timestamp tick
}

impl GpuProfiler {
    /// Creates a new GPU profiler.
    ///
    /// Allocates a query set with 256 timestamps (supports 128 passes).
    ///
    /// # Parameters
    ///
    /// - `device`: GPU device for creating query set
    /// - `queue`: GPU queue (reserved for async readback)
    ///
    /// # Performance
    ///
    /// - **O(1)**: Allocates query set once
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use helio_core::profiling::GpuProfiler;
    /// let profiler = GpuProfiler::new(&device, &queue);
    /// ```
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        // write_timestamp on a command encoder requires BOTH TIMESTAMP_QUERY and
        // TIMESTAMP_QUERY_INSIDE_ENCODERS.  WebGPU browsers typically support neither;
        // guard both so we never call write_timestamp on an unsupported backend.
        let has_timestamps = device.features().contains(wgpu::Features::TIMESTAMP_QUERY)
            && device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);

        let query_set = if has_timestamps {
            Some(device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("GPU Profiler QuerySet"),
                ty: wgpu::QueryType::Timestamp,
                count: QUERY_CAPACITY, // 128 passes * 2 timestamps per pass
            }))
        } else {
            None
        };

        let query_buffer = if has_timestamps {
            Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("GPU Profiler Query Buffer"),
                size: u64::from(QUERY_CAPACITY) * 8,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }))
        } else {
            None
        };

        let readback_slots = if has_timestamps {
            (0..READBACK_SLOT_COUNT)
                .map(|index| ReadbackSlot {
                    buffer: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("GPU Profiler Readback Slot {index}")),
                        size: u64::from(QUERY_CAPACITY) * 8,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    }),
                    state: ReadbackState::Idle,
                    queries: Vec::with_capacity((QUERY_CAPACITY / 2) as usize),
                    frame_index: 0,
                })
                .collect()
        } else {
            Vec::new()
        };

        // Get timestamp period for converting ticks to nanoseconds
        let timestamp_period = queue.get_timestamp_period();

        Self {
            query_set,
            query_buffer,
            readback_slots,
            pending_queries: VecDeque::new(),
            next_index: 0,
            last_timings: Vec::new(),
            last_completed_frame: None,
            dropped_readbacks: 0,
            query_overflows: 0,
            timestamp_period,
        }
    }

    /// Writes a start timestamp for a pass.
    ///
    /// This is called internally by `PassContext::begin_render_pass()`.
    /// **Passes should not call this directly.**
    ///
    /// # Parameters
    ///
    /// - `encoder`: Command encoder to write timestamp into
    /// - `name`: Pass name for debugging
    ///
    /// # Performance
    ///
    /// - **O(1)**: Writes a single GPU command (~10ns)
    ///
    /// # Example (Internal)
    ///
    /// ```rust,no_run
    /// # use helio_core::profiling::GpuProfiler;
    /// # let mut profiler = GpuProfiler::new(&device, &queue);
    /// # let mut encoder = device.create_command_encoder(&Default::default());
    /// profiler.begin_pass(&mut encoder, "ShadowPass");
    /// ```
    pub fn begin_pass(&mut self, encoder: &mut wgpu::CommandEncoder, name: &'static str) {
        if let Some(ref query_set) = self.query_set {
            if self.next_index + 1 >= QUERY_CAPACITY {
                self.query_overflows = self.query_overflows.saturating_add(1);
                return;
            }
            let start_index = self.next_index;
            self.next_index += 1;
            encoder.write_timestamp(query_set, start_index);
            // Push incomplete entry (will be completed by end_pass)
            self.pending_queries.push_back((name, start_index, 0));
        }
    }

    /// Writes an end timestamp for a pass.
    ///
    /// This is called internally by `PassContext` (future - currently TODO).
    /// **Passes should not call this directly.**
    ///
    /// # Parameters
    ///
    /// - `encoder`: Command encoder to write timestamp into
    /// - `name`: Pass name for debugging
    ///
    /// # Performance
    ///
    /// - **O(1)**: Writes a single GPU command (~10ns)
    ///
    /// # Example (Internal)
    ///
    /// ```rust,no_run
    /// # use helio_core::profiling::GpuProfiler;
    /// # let mut profiler = GpuProfiler::new(&device, &queue);
    /// # let mut encoder = device.create_command_encoder(&Default::default());
    /// profiler.end_pass(&mut encoder, "ShadowPass");
    /// ```
    pub fn end_pass(&mut self, encoder: &mut wgpu::CommandEncoder, _name: &'static str) {
        if let Some(ref query_set) = self.query_set {
            let Some((_, _, end_index)) = self.pending_queries.back() else {
                return;
            };
            if *end_index != 0 || self.next_index >= QUERY_CAPACITY {
                return;
            }
            let end_index = self.next_index;
            self.next_index += 1;
            encoder.write_timestamp(query_set, end_index);

            // Update the last pending query with end index
            if let Some(last) = self.pending_queries.back_mut() {
                last.2 = end_index;
            }
        }
    }

    /// Resolves this frame into an idle readback slot without waiting for the
    /// GPU. If all slots are still in flight, the sample is explicitly
    /// dropped rather than stalling or reusing a mapped buffer.
    pub fn resolve_queries(&mut self, encoder: &mut wgpu::CommandEncoder, frame_index: u64) {
        if self.next_index == 0 {
            self.pending_queries.clear();
            return;
        }
        let (Some(query_set), Some(query_buffer)) = (&self.query_set, &self.query_buffer) else {
            self.pending_queries.clear();
            self.next_index = 0;
            return;
        };

        encoder.resolve_query_set(query_set, 0..self.next_index, query_buffer, 0);
        let Some(slot) = self
            .readback_slots
            .iter_mut()
            .find(|slot| matches!(slot.state, ReadbackState::Idle))
        else {
            self.dropped_readbacks = self.dropped_readbacks.saturating_add(1);
            self.pending_queries.clear();
            self.next_index = 0;
            return;
        };

        encoder.copy_buffer_to_buffer(
            query_buffer,
            0,
            &slot.buffer,
            0,
            u64::from(self.next_index) * 8,
        );
        slot.queries.clear();
        slot.queries.extend(self.pending_queries.drain(..));
        slot.frame_index = frame_index;
        slot.state = ReadbackState::CopySubmitted;
        self.next_index = 0;
    }

    /// Read back GPU timestamps (blocking, call after frame completion).
    ///
    /// Calls `device.poll(wait_indefinitely)` — only safe when Helio owns the
    /// wgpu device (i.e., the renderer was created via `Renderer::new`).
    /// When sharing an external device (e.g., GPUI) use
    /// `read_timestamps_deferred` instead.
    pub fn read_timestamps_blocking(&mut self, device: &wgpu::Device) -> &[GpuTimestamp] {
        self.start_submitted_mappings();
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        self.consume_completed_mappings();
        &self.last_timings
    }

    /// Read back GPU timestamps without touching device.poll().
    ///
    /// Queues `map_async` and immediately checks with `try_recv` — no poll
    /// is issued. The callback fires when the **external device owner**
    /// (e.g., GPUI) polls the device on its own cadence. If the data isn't
    /// ready this frame, the previous frame's timings are returned unchanged
    /// (GPU timestamps lag 1-2 frames in practice anyway).
    ///
    /// **Do not call `device.poll()` from Helio when using an external device.**
    /// Even a single `PollType::Poll` call from a non-owning thread causes
    /// "Parent device is lost" panics on DX12/Vulkan.
    pub fn read_timestamps_deferred(&mut self) -> &[GpuTimestamp] {
        self.consume_completed_mappings();
        self.start_submitted_mappings();
        &self.last_timings
    }

    fn start_submitted_mappings(&mut self) {
        for slot in &mut self.readback_slots {
            if !matches!(slot.state, ReadbackState::CopySubmitted) {
                continue;
            }
            let completion = Arc::new(Mutex::new(None));
            let callback_completion = Arc::clone(&completion);
            slot.buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    if let Ok(mut completion) = callback_completion.lock() {
                        *completion = Some(result);
                    }
                });
            slot.state = ReadbackState::Mapping(completion);
        }
    }

    fn consume_completed_mappings(&mut self) {
        loop {
            let next_index = self
                .readback_slots
                .iter()
                .enumerate()
                .filter(|(_, slot)| {
                    let ReadbackState::Mapping(completion) = &slot.state else {
                        return false;
                    };
                    completion
                        .lock()
                        .is_ok_and(|completion| completion.is_some())
                })
                .min_by_key(|(_, slot)| slot.frame_index)
                .map(|(index, _)| index);
            let Some(index) = next_index else {
                break;
            };
            let frame_index = self.readback_slots[index].frame_index;
            let result = match &self.readback_slots[index].state {
                ReadbackState::Mapping(completion) => completion
                    .lock()
                    .expect("GPU timestamp completion mutex is not poisoned")
                    .take()
                    .expect("completed GPU timestamp mapping has a result"),
                _ => unreachable!("selected GPU timestamp slot must be mapping"),
            };
            let slot = &mut self.readback_slots[index];
            if result.is_ok() {
                let data = slot
                    .buffer
                    .slice(..)
                    .get_mapped_range()
                    .expect("completed GPU timestamp mapping must be readable");
                let timestamps: &[u64] = bytemuck::cast_slice(&data);
                self.last_timings.clear();
                for &(name, start_index, end_index) in &slot.queries {
                    if (end_index as usize) < timestamps.len()
                        && (start_index as usize) < timestamps.len()
                    {
                        let duration_ticks = timestamps[end_index as usize]
                            .saturating_sub(timestamps[start_index as usize]);
                        self.last_timings.push(GpuTimestamp {
                            name,
                            duration_ns: (duration_ticks as f32 * self.timestamp_period) as u64,
                        });
                    }
                }
                self.last_completed_frame = Some(frame_index);
                drop(data);
            } else {
                self.dropped_readbacks = self.dropped_readbacks.saturating_add(1);
            }
            slot.buffer.unmap();
            slot.queries.clear();
            slot.state = ReadbackState::Idle;
        }
    }

    /// Get last recorded timings (non-blocking)
    pub fn get_last_timings(&self) -> &[GpuTimestamp] {
        &self.last_timings
    }

    pub const fn supported(&self) -> bool {
        self.query_set.is_some()
    }

    pub const fn last_completed_frame(&self) -> Option<u64> {
        self.last_completed_frame
    }

    pub fn pending_readbacks(&self) -> usize {
        self.readback_slots
            .iter()
            .filter(|slot| !matches!(slot.state, ReadbackState::Idle))
            .count()
    }

    pub const fn dropped_readbacks(&self) -> u64 {
        self.dropped_readbacks
    }

    pub const fn query_overflows(&self) -> u64 {
        self.query_overflows
    }
}

/// GPU timestamp result.
///
/// Represents the GPU time for a single pass. Results are collected from async readback
/// and available for external telemetry systems.
///
/// # Fields
///
/// - `name`: Pass name (e.g., "ShadowPass")
/// - `duration_ns`: GPU time in nanoseconds
///
/// # Example (Future)
///
/// ```rust,ignore
/// let timestamps = profiler.read_timestamps(&queue);
/// for ts in timestamps {
///     println!("{}: {:.2}ms", ts.name, ts.duration_ns as f64 / 1_000_000.0);
/// }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuTimestamp {
    /// Pass name (e.g., "ShadowPass").
    pub name: &'static str,

    /// GPU time in nanoseconds.
    ///
    /// Convert to milliseconds: `duration_ns as f64 / 1_000_000.0`
    pub duration_ns: u64,
}
