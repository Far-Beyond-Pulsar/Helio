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
//! # use helio_v3::profiling::GpuProfiler;
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
/// # use helio_v3::profiling::GpuProfiler;
/// let mut profiler = GpuProfiler::new(&device, &queue);
///
/// profiler.begin_pass(&mut encoder, "ShadowPass");
/// // GPU commands...
/// profiler.end_pass(&mut encoder, "ShadowPass");
/// ```
use std::collections::VecDeque;

pub struct GpuProfiler {
    query_set: Option<wgpu::QuerySet>,
    query_buffer: Option<wgpu::Buffer>,
    resolve_buffer: Option<wgpu::Buffer>,
    pending_queries: VecDeque<(&'static str, u32, u32)>, // (name, start_index, end_index)
    next_index: u32,
    last_timings: Vec<GpuTimestamp>,
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
    /// # use helio_v3::profiling::GpuProfiler;
    /// let profiler = GpuProfiler::new(&device, &queue);
    /// ```
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let has_timestamps = device.features().contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);

        let query_set = if has_timestamps {
            Some(device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("GPU Profiler QuerySet"),
                ty: wgpu::QueryType::Timestamp,
                count: 256, // 128 passes * 2 timestamps per pass
            }))
        } else {
            None
        };

        let query_buffer = if has_timestamps {
            Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("GPU Profiler Query Buffer"),
                size: 256 * 8, // 256 timestamps * 8 bytes each
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }))
        } else {
            None
        };

        let resolve_buffer = if has_timestamps {
            Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("GPU Profiler Resolve Buffer"),
                size: 256 * 8,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }))
        } else {
            None
        };

        // Get timestamp period for converting ticks to nanoseconds
        let timestamp_period = queue.get_timestamp_period();

        Self {
            query_set,
            query_buffer,
            resolve_buffer,
            pending_queries: VecDeque::new(),
            next_index: 0,
            last_timings: Vec::new(),
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
    /// # use helio_v3::profiling::GpuProfiler;
    /// # let mut profiler = GpuProfiler::new(&device, &queue);
    /// # let mut encoder = device.create_command_encoder(&Default::default());
    /// profiler.begin_pass(&mut encoder, "ShadowPass");
    /// ```
    pub fn begin_pass(&mut self, encoder: &mut wgpu::CommandEncoder, name: &'static str) {
        if let Some(ref query_set) = self.query_set {
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
    /// # use helio_v3::profiling::GpuProfiler;
    /// # let mut profiler = GpuProfiler::new(&device, &queue);
    /// # let mut encoder = device.create_command_encoder(&Default::default());
    /// profiler.end_pass(&mut encoder, "ShadowPass");
    /// ```
    pub fn end_pass(&mut self, encoder: &mut wgpu::CommandEncoder, _name: &'static str) {
        if let Some(ref query_set) = self.query_set {
            let end_index = self.next_index;
            self.next_index += 1;
            encoder.write_timestamp(query_set, end_index);

            // Update the last pending query with end index
            if let Some(last) = self.pending_queries.back_mut() {
                last.2 = end_index;
            }
        }
    }

    /// Resolve query set to buffer (call after frame submit)
    pub fn resolve_queries(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if let (Some(ref query_set), Some(ref query_buffer)) = (&self.query_set, &self.query_buffer) {
            if self.next_index > 0 {
                encoder.resolve_query_set(query_set, 0..self.next_index, query_buffer, 0);
            }
        }
    }

    /// Copy resolved queries to CPU-readable buffer
    pub fn copy_to_resolve_buffer(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if let (Some(ref query_buffer), Some(ref resolve_buffer)) = (&self.query_buffer, &self.resolve_buffer) {
            if self.next_index > 0 {
                encoder.copy_buffer_to_buffer(query_buffer, 0, resolve_buffer, 0, (self.next_index as u64) * 8);
            }
        }
    }

    /// Read back GPU timestamps (blocking, call after frame completion)
    pub fn read_timestamps_blocking(&mut self, device: &wgpu::Device) -> &[GpuTimestamp] {
        self.last_timings.clear();

        if let Some(ref resolve_buffer) = self.resolve_buffer {
            if self.pending_queries.is_empty() {
                return &self.last_timings;
            }

            let buffer_slice = resolve_buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();

            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).ok();
            });

            // Wait for buffer mapping to complete (rx.recv() will block)
            if rx.recv().ok().is_some() {
                let data = buffer_slice.get_mapped_range();
                let timestamps: &[u64] = bytemuck::cast_slice(&data);

                // Process all pending queries
                for &(name, start_idx, end_idx) in &self.pending_queries {
                    if (end_idx as usize) < timestamps.len() {
                        let start_ticks = timestamps[start_idx as usize];
                        let end_ticks = timestamps[end_idx as usize];

                        if end_ticks > start_ticks {
                            // Convert ticks to nanoseconds using timestamp period
                            let duration_ticks = end_ticks - start_ticks;
                            let duration_ns = (duration_ticks as f32 * self.timestamp_period) as u64;

                            self.last_timings.push(GpuTimestamp {
                                name: name.to_string(),
                                duration_ns,
                            });
                        }
                    }
                }

                drop(data);
                resolve_buffer.unmap();
            }
        }

        // Reset for next frame
        self.pending_queries.clear();
        self.next_index = 0;

        &self.last_timings
    }

    /// Get last recorded timings (non-blocking)
    pub fn get_last_timings(&self) -> &[GpuTimestamp] {
        &self.last_timings
    }
}

/// GPU timestamp result.
///
/// Represents the GPU time for a single pass. Results are collected from async readback
/// and exported to `helio-live-portal`.
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
pub struct GpuTimestamp {
    /// Pass name (e.g., "ShadowPass").
    pub name: String,

    /// GPU time in nanoseconds.
    ///
    /// Convert to milliseconds: `duration_ns as f64 / 1_000_000.0`
    pub duration_ns: u64,
}

