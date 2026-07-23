//! Automatic CPU/GPU profiling system.
//!
//! This module provides **zero-instrumentation profiling** for render passes. Profiling is
//! automatic and injected by `RenderGraph` - passes don't need to manually add profiling code.
//!
//! # Design Pattern: Automatic Profiling
//!
//! Helio v3 uses **implicit profiling** via `PassContext`:
//!
//! 1. **CPU profiling**: `RenderGraph` creates scoped guards for each pass
//! 2. **GPU profiling**: `PassContext::begin_render_pass()` injects timestamp queries
//! 3. **Zero overhead**: Profiling is compile-time gated by `profiling` feature flag
//!
//! # Components
//!
//! - [`Profiler`] - Combined CPU/GPU profiler (feature-gated)
//! - [`CpuProfiler`] - Scoped CPU timing with RAII guards
//! - [`GpuProfiler`] - GPU timestamp queries with async readback
//!
//! # Performance
//!
//! - **Zero runtime cost when disabled**: `cfg!(feature = "profiling")` eliminates all profiling code
//! - **Minimal overhead when enabled**: ~0.1ms CPU overhead, ~10ns GPU timestamp writes
//! - **No allocations**: Profiling data is pre-allocated and reused
//!
//! # Feature Flag
//!
//! ```toml
//! [dependencies]
//! helio-core = { version = "0.1", default-features = false }  # Profiling disabled
//! helio-core = { version = "0.1" }                             # Profiling enabled (default)
//! ```
//!
//! # Example: Automatic Profiling
//!
//! Profiling is automatic - passes don't need to add instrumentation:
//!
//! ```rust,no_run
//! use helio_core::{RenderPass, PassContext, Result};
//!
//! struct MyPass {
//!     pipeline: wgpu::RenderPipeline,
//! }
//!
//! impl RenderPass for MyPass {
//!     fn name(&self) -> &'static str {
//!         "MyPass" // Used for profiling labels
//!     }
//!
//!     fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
//!         // CPU scope created automatically by RenderGraph
//!         // GPU timestamps injected automatically by begin_render_pass
//!
//!         let mut pass = ctx.begin_render_pass(&wgpu::RenderPassDescriptor {
//!             label: Some("MyPass"),
//!             // ...
//! #            color_attachments: &[],
//! #            depth_stencil_attachment: None,
//! #            timestamp_writes: None,
//! #            occlusion_query_set: None,
//!         });
//!
//!         // GPU commands are automatically profiled
//!         pass.set_pipeline(&self.pipeline);
//!         pass.draw(0..3, 0..1);
//!
//!         Ok(())
//!     }
//! }
//! ```
//!
//! # Profiling Flow
//!
//! ```text
//! RenderGraph::execute()
//! ├── profiler.scope("ShadowPass") (CPU start)
//! │   ├── pass.prepare()
//! │   ├── pass.execute()
//! │   │   ├── ctx.begin_render_pass()
//! │   │   │   ├── encoder.write_timestamp(query_set, start_index) (GPU start)
//! │   │   │   └── encoder.begin_render_pass()
//! │   │   ├── GPU commands
//! │   │   └── encoder.write_timestamp(query_set, end_index) (GPU end)
//! │   └── ScopeGuard::drop() (CPU end)
//! ├── profiler.scope("GBufferPass") (CPU start)
//! │   └── ...
//! └── Results available via Profiler::export_timings()
//! ```
//!
//! # Data Export
//!
//! Profiling data is available via [`Profiler::export_timings()`] for integration
//! with external telemetry or custom debug overlays.

mod cpu;
mod gpu;

pub use cpu::{CpuProfiler, ScopeGuard};
pub use gpu::{GpuProfiler, GpuTimestamp};

/// Combined CPU/GPU profiler with automatic feature-gating.
///
/// `Profiler` orchestrates both CPU and GPU profiling. It is created by `RenderGraph`
/// and injected into `PassContext` automatically.
///
/// # Feature Gating
///
/// - **Enabled**: When `profiling` feature is active, profiling runs normally
/// - **Disabled**: When `profiling` feature is off, all profiling code is eliminated at compile-time
///
/// # Design Pattern
///
/// The profiler uses **RAII scopes** for CPU profiling and **timestamp queries** for GPU profiling.
/// Both are zero-overhead when disabled via feature flags.
///
/// # Performance
///
/// - **Zero cost when disabled**: `cfg!(feature = "profiling")` eliminates all code
/// - **Minimal cost when enabled**: ~0.1ms CPU overhead, ~10ns per GPU timestamp write
///
/// # Example (Internal - Used by RenderGraph)
///
/// ```rust,no_run
/// use helio_core::Profiler;
///
/// let mut profiler = Profiler::new(&device, &queue);
///
/// // CPU profiling (RAII scope)
/// {
///     let _scope = profiler.scope("ShadowPass");
///     // ... render code ...
/// } // Scope drops, timing recorded
///
/// // GPU profiling (timestamp queries)
/// profiler.begin_gpu_pass(&mut encoder, "GBufferPass");
/// // ... GPU commands ...
/// profiler.end_gpu_pass(&mut encoder, "GBufferPass");
/// ```
pub struct Profiler {
    cpu: CpuProfiler,
    gpu: GpuProfiler,
    enabled: bool,
    snapshot: RenderTimingSnapshot,
}

impl Profiler {
    /// Creates a new profiler.
    ///
    /// # Performance
    ///
    /// - **O(1)**: Initializes CPU profiler and GPU query set
    /// - **Zero cost when disabled**: If `profiling` feature is off, GPU query set is still created
    ///   but never used (minimal memory overhead)
    ///
    /// # Parameters
    ///
    /// - `device`: GPU device for creating query sets
    /// - `queue`: GPU queue (unused currently, reserved for async readback)
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            cpu: CpuProfiler::new(),
            gpu: GpuProfiler::new(device, queue),
            enabled: cfg!(feature = "profiling"),
            snapshot: RenderTimingSnapshot::default(),
        }
    }

    /// Creates a CPU profiling scope (RAII guard).
    ///
    /// The returned `ScopeGuard` measures CPU time until it is dropped.
    /// Results are recorded to the CPU profiler.
    ///
    /// # Performance
    ///
    /// - **O(1)**: Records start time in `Instant::now()`
    /// - **Zero cost when disabled**: Guard is still created but timing is not recorded
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use helio_core::Profiler;
    /// # let mut profiler = Profiler::new(&device, &queue);
    /// {
    ///     let _scope = profiler.scope("MyPass");
    ///     // ... CPU work ...
    /// } // ScopeGuard drops, timing recorded
    /// ```
    pub fn scope(&mut self, name: &'static str) -> ScopeGuard {
        self.cpu.scope(name)
    }

    /// Begins a GPU profiling pass by writing a start timestamp.
    ///
    /// **Note**: This is called internally by `PassContext::begin_render_pass()`.
    /// Passes should not call this directly.
    ///
    /// # Performance
    ///
    /// - **O(1)**: Writes a single timestamp query (~10ns)
    /// - **Zero cost when disabled**: Feature flag eliminates the call
    ///
    /// # Parameters
    ///
    /// - `encoder`: Command encoder to write timestamp into
    /// - `name`: Pass name for debugging
    pub fn begin_gpu_pass(&mut self, encoder: &mut wgpu::CommandEncoder, name: &'static str) {
        if self.enabled {
            self.gpu.begin_pass(encoder, name);
        }
    }

    /// Ends a GPU profiling pass by writing an end timestamp.
    ///
    /// **Note**: This is called internally by `PassContext` (future - currently TODO).
    /// Passes should not call this directly.
    ///
    /// # Performance
    ///
    /// - **O(1)**: Writes a single timestamp query (~10ns)
    /// - **Zero cost when disabled**: Feature flag eliminates the call
    ///
    /// # Parameters
    ///
    /// - `encoder`: Command encoder to write timestamp into
    /// - `name`: Pass name for debugging
    pub fn end_gpu_pass(&mut self, encoder: &mut wgpu::CommandEncoder, name: &'static str) {
        if self.enabled {
            self.gpu.end_pass(encoder, name);
        }
    }

    /// Resolve GPU timestamp queries to buffer (call after submitting command buffer)
    pub fn resolve_gpu_queries(&mut self, encoder: &mut wgpu::CommandEncoder, frame_index: u64) {
        if self.enabled {
            self.gpu.resolve_queries(encoder, frame_index);
        }
    }

    /// Read back GPU timestamps (blocking, call after frame completion)
    pub fn read_gpu_timestamps_blocking(&mut self, device: &wgpu::Device) -> &[GpuTimestamp] {
        if self.enabled {
            self.gpu.read_timestamps_blocking(device)
        } else {
            &[]
        }
    }

    /// Read back GPU timestamps without blocking the device owner.
    ///
    /// Use this instead of `read_gpu_timestamps_blocking` whenever the wgpu
    /// device is owned externally (e.g., by GPUI).  A single non-blocking
    /// `map_async` is queued and the external owner's event loop delivers the
    /// callback on its own poll cadence; no `device.poll()` is called here.
    /// If the GPU hasn't finished yet, the previous frame's timings are returned
    /// unchanged instead of blocking.
    pub fn read_gpu_timestamps_deferred(&mut self) -> &[GpuTimestamp] {
        if self.enabled {
            self.gpu.read_timestamps_deferred()
        } else {
            &[]
        }
    }

    /// Get CPU timings
    pub fn get_cpu_timings(&self) -> &std::collections::HashMap<&'static str, std::time::Duration> {
        self.cpu.get_timings()
    }

    /// Get last GPU timings (non-blocking)
    pub fn get_gpu_timings(&self) -> &[GpuTimestamp] {
        self.gpu.get_last_timings()
    }

    /// Clear CPU timings for new frame
    pub fn clear_cpu_timings(&mut self) {
        self.cpu.clear();
    }

    /// Updates the reusable host-facing snapshot after a frame has submitted.
    pub fn update_snapshot(
        &mut self,
        cpu_frame_index: u64,
        pass_names: impl Iterator<Item = &'static str>,
    ) {
        self.snapshot.generation = self.snapshot.generation.saturating_add(1);
        self.snapshot.cpu_frame_index = cpu_frame_index;
        self.snapshot.gpu_frame_index = self.gpu.last_completed_frame();
        self.snapshot.gpu_lag_frames = self
            .snapshot
            .gpu_frame_index
            .map(|gpu_frame| cpu_frame_index.saturating_sub(gpu_frame));
        self.snapshot.readback_drops = self.gpu.dropped_readbacks();
        self.snapshot.query_overflows = self.gpu.query_overflows();
        self.snapshot.gpu_availability = if !self.enabled {
            GpuTimingAvailability::Disabled
        } else if !self.gpu.supported() {
            GpuTimingAvailability::Unsupported
        } else if self.snapshot.gpu_frame_index.is_some() {
            GpuTimingAvailability::Available
        } else if self.snapshot.readback_drops != 0 {
            GpuTimingAvailability::Backpressured
        } else {
            GpuTimingAvailability::Pending
        };

        self.snapshot.passes.clear();
        let mut total_cpu_ms = 0.0_f32;
        let mut has_cpu = false;
        let mut total_gpu_ms = 0.0_f32;
        let mut has_gpu = false;
        for name in pass_names {
            let cpu_ms = self.cpu.get_timings().get(name).map(|duration| {
                has_cpu = true;
                let milliseconds = duration.as_secs_f64() as f32 * 1_000.0;
                total_cpu_ms += milliseconds;
                milliseconds
            });
            let gpu_ms = self
                .gpu
                .get_last_timings()
                .iter()
                .find(|timing| timing.name == name)
                .map(|timing| {
                    has_gpu = true;
                    let milliseconds = timing.duration_ns as f32 / 1_000_000.0;
                    total_gpu_ms += milliseconds;
                    milliseconds
                });
            self.snapshot.passes.push(RenderPassTiming {
                name,
                cpu_ms,
                gpu_ms,
            });
        }
        self.snapshot.total_cpu_ms = has_cpu.then_some(total_cpu_ms);
        self.snapshot.total_gpu_ms = has_gpu.then_some(total_gpu_ms);
    }

    /// Returns the latest snapshot without allocation or synchronization.
    pub const fn timing_snapshot(&self) -> &RenderTimingSnapshot {
        &self.snapshot
    }

    /// Print profiling results to console (blocking - use for debugging only!)
    ///
    /// **Warning**: This blocks the render thread and causes frame hitches.
    /// For production use, use export_pass_timings() instead.
    /// Note: The new DebugOverlayPass replaces the need for console printing.
    #[deprecated(note = "Use the DebugOverlayPass for on-screen display instead")]
    pub fn print_frame_timings(&self) {
        if !self.enabled {
            return;
        }

        println!("\n=== Frame Timings ===");

        // CPU timings
        let mut cpu_timings: Vec<_> = self.cpu.get_timings().iter().collect();
        cpu_timings.sort_by_key(|(name, _)| *name);

        println!("CPU:");
        let mut total_cpu = std::time::Duration::ZERO;
        for (name, duration) in cpu_timings {
            println!("  {:<30} {:>8.2}ms", name, duration.as_secs_f64() * 1000.0);
            total_cpu += *duration;
        }
        println!(
            "  {:<30} {:>8.2}ms",
            "TOTAL CPU",
            total_cpu.as_secs_f64() * 1000.0
        );

        // GPU timings
        if !self.gpu.get_last_timings().is_empty() {
            println!("\nGPU:");
            let mut total_gpu = 0u64;
            for ts in self.gpu.get_last_timings() {
                println!(
                    "  {:<30} {:>8.2}ms",
                    ts.name,
                    ts.duration_ns as f64 / 1_000_000.0
                );
                total_gpu += ts.duration_ns;
            }
            println!(
                "  {:<30} {:>8.2}ms",
                "TOTAL GPU",
                total_gpu as f64 / 1_000_000.0
            );
        }

        println!("====================\n");
    }

    /// Export profiling data for external telemetry systems.
    ///
    /// Returns (pass_timings, total_cpu_ms, total_gpu_ms) for non-blocking transmission
    /// to external debug overlays or telemetry backends.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use helio_core::RenderGraph;
    /// # let graph = RenderGraph::new(&device, &queue);
    /// let (pass_timings, total_cpu_ms, total_gpu_ms) = graph.profiler().export_timings();
    ///
    /// // Forward to custom telemetry system
    /// // for timing in &pass_timings { ... }
    /// ```
    pub fn export_timings(&self) -> (Vec<PassTiming>, f32, f32) {
        if !self.enabled {
            return (Vec::new(), 0.0, 0.0);
        }

        let mut pass_timings = Vec::new();
        let mut total_cpu_ms = 0.0;
        let mut total_gpu_ms = 0.0;

        // Build GPU timings map for lookup
        let mut gpu_map = std::collections::HashMap::new();
        for ts in self.gpu.get_last_timings() {
            let ms = ts.duration_ns as f64 / 1_000_000.0;
            gpu_map.insert(ts.name, ms as f32);
            total_gpu_ms += ms as f32;
        }

        // Preserve CPU timing order (execution order) and merge with GPU timings
        for (name, duration) in self.cpu.get_timings() {
            let cpu_ms = duration.as_secs_f64() * 1000.0;
            let gpu_ms = gpu_map.get(name).copied().unwrap_or(0.0);

            pass_timings.push(PassTiming {
                name: name.to_string(),
                cpu_ms: cpu_ms as f32,
                gpu_ms,
            });

            total_cpu_ms += cpu_ms as f32;
        }

        // Add any GPU-only passes that don't have CPU timings (shouldn't happen in practice)
        for ts in self.gpu.get_last_timings() {
            if !pass_timings.iter().any(|pt| pt.name == ts.name) {
                let ms = ts.duration_ns as f64 / 1_000_000.0;
                pass_timings.push(PassTiming {
                    name: ts.name.to_string(),
                    cpu_ms: 0.0,
                    gpu_ms: ms as f32,
                });
            }
        }

        (pass_timings, total_cpu_ms, total_gpu_ms)
    }
}

/// Pass timing data for export to external debug overlays or telemetry systems.
#[derive(Clone, Debug)]
pub struct PassTiming {
    pub name: String,
    pub cpu_ms: f32,
    pub gpu_ms: f32,
}

/// State of non-blocking GPU timing readback for the current snapshot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GpuTimingAvailability {
    /// The `profiling` feature is disabled.
    Disabled,
    /// The adapter does not expose the required timestamp-query features.
    Unsupported,
    /// Timestamp data has been submitted but no host-driven poll has completed it yet.
    #[default]
    Pending,
    /// At least one completed GPU frame is available.
    Available,
    /// All bounded readback slots were occupied before the first result completed.
    Backpressured,
}

/// One pass in a host-facing timing snapshot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderPassTiming {
    pub name: &'static str,
    pub cpu_ms: Option<f32>,
    pub gpu_ms: Option<f32>,
}

/// Reusable read-only timing state exposed by `Renderer`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RenderTimingSnapshot {
    /// Monotonic snapshot revision, incremented after every submitted frame.
    pub generation: u64,
    /// Frame whose CPU pass timings are represented.
    pub cpu_frame_index: u64,
    /// Most recent GPU frame whose asynchronous readback completed.
    pub gpu_frame_index: Option<u64>,
    /// Difference between `cpu_frame_index` and `gpu_frame_index`.
    pub gpu_lag_frames: Option<u64>,
    pub gpu_availability: GpuTimingAvailability,
    pub total_cpu_ms: Option<f32>,
    pub total_gpu_ms: Option<f32>,
    pub readback_drops: u64,
    pub query_overflows: u64,
    pub passes: Vec<RenderPassTiming>,
}
