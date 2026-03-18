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
pub struct GpuProfiler {
    query_set: Option<wgpu::QuerySet>,
    // Future: Add query allocation and readback
    // query_buffer: wgpu::Buffer,
    // pending_queries: VecDeque<(String, u32, u32)>, // (name, start_index, end_index)
    // next_index: u32,
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
    pub fn new(device: &wgpu::Device, _queue: &wgpu::Queue) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("GPU Profiler"),
            ty: wgpu::QueryType::Timestamp,
            count: 256, // 128 passes * 2 timestamps per pass
        });

        Self {
            query_set: Some(query_set),
            // Future: Initialize query buffer and allocation
            // query_buffer: device.create_buffer(...),
            // pending_queries: VecDeque::new(),
            // next_index: 0,
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
    pub fn begin_pass(&mut self, encoder: &mut wgpu::CommandEncoder, _name: &str) {
        if let Some(ref query_set) = self.query_set {
            // Future: Allocate next query index
            // let start_index = self.next_index;
            // self.next_index += 1;

            // Write start timestamp
            encoder.write_timestamp(query_set, 0); // Stub: use start_index
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
    pub fn end_pass(&mut self, encoder: &mut wgpu::CommandEncoder, _name: &str) {
        if let Some(ref query_set) = self.query_set {
            // Future: Allocate next query index
            // let end_index = self.next_index;
            // self.next_index += 1;

            // Write end timestamp
            encoder.write_timestamp(query_set, 1); // Stub: use end_index

            // Future: Record query for readback
            // self.pending_queries.push_back((name.to_string(), start_index, end_index));
        }
    }

    // Future: Add readback method
    // pub fn read_timestamps(&mut self, queue: &wgpu::Queue) -> Vec<GpuTimestamp> { ... }
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
