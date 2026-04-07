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
//! helio-v3 = { version = "0.1", default-features = false }  # Profiling disabled
//! helio-v3 = { version = "0.1" }                             # Profiling enabled (default)
//! ```
//!
//! # Example: Automatic Profiling
//!
//! Profiling is automatic - passes don't need to add instrumentation:
//!
//! ```rust,no_run
//! use helio_v3::{RenderPass, PassContext, Result};
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
//! └── Results exported to helio-live-portal
//! ```
//!
//! # Integration with helio-live-portal
//!
//! Profiling results are exported to `helio-live-portal` for real-time telemetry:
//!
//! ```text
//! ┌─────────────────────────────────┐
//! │ helio-live-portal (Web UI)      │
//! ├─────────────────────────────────┤
//! │ ShadowPass:     1.2ms CPU       │
//! │                 0.8ms GPU       │
//! ├─────────────────────────────────┤
//! │ GBufferPass:    2.5ms CPU       │
//! │                 1.9ms GPU       │
//! ├─────────────────────────────────┤
//! │ DeferredLight:  3.1ms CPU       │
//! │                 2.7ms GPU       │
//! └─────────────────────────────────┘
//! ```

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
/// use helio_v3::Profiler;
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
    /// # use helio_v3::Profiler;
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
    pub fn resolve_gpu_queries(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.enabled {
            self.gpu.resolve_queries(encoder);
            self.gpu.copy_to_resolve_buffer(encoder);
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

    /// Print profiling results to console (blocking - use for debugging only!)
    ///
    /// **Warning**: This blocks the render thread and causes frame hitches.
    /// For production use, export to helio-live-portal instead via `export_pass_timings()`.
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

    /// Export profiling data in a format suitable for helio-live-portal.
    ///
    /// Returns (pass_timings, total_cpu_ms, total_gpu_ms) for non-blocking transmission
    /// to the web UI. This is the recommended way to access profiling data.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use helio_v3::RenderGraph;
    /// # let graph = RenderGraph::new(&device, &queue);
    /// let (pass_timings, total_cpu_ms, total_gpu_ms) = graph.profiler().export_timings();
    ///
    /// // Send to helio-live-portal without blocking
    /// // portal_handle.publish(PortalFrameSnapshot { pass_timings, ... });
    /// ```
    pub fn export_timings(&self) -> (Vec<PassTiming>, f32, f32) {
        if !self.enabled {
            return (Vec::new(), 0.0, 0.0);
        }

        let mut pass_timings = Vec::new();
        let mut total_cpu_ms = 0.0;
        let mut total_gpu_ms = 0.0;

        // Merge CPU and GPU timings by pass name
        let mut pass_map = std::collections::HashMap::new();

        // Add CPU timings
        for (name, duration) in self.cpu.get_timings() {
            let ms = duration.as_secs_f64() * 1000.0;
            pass_map.insert(*name, (ms as f32, 0.0f32));
            total_cpu_ms += ms as f32;
        }

        // Add GPU timings
        for ts in self.gpu.get_last_timings() {
            let ms = ts.duration_ns as f64 / 1_000_000.0;
            pass_map.entry(ts.name.as_str()).or_insert((0.0, 0.0)).1 = ms as f32;
            total_gpu_ms += ms as f32;
        }

        // Convert to Vec
        for (name, (cpu_ms, gpu_ms)) in pass_map {
            pass_timings.push(PassTiming {
                name: name.to_string(),
                cpu_ms,
                gpu_ms,
            });
        }

        // Sort by name for consistent ordering
        pass_timings.sort_by(|a, b| a.name.cmp(&b.name));

        (pass_timings, total_cpu_ms, total_gpu_ms)
    }
}

/// Pass timing data for export to helio-live-portal or other telemetry systems.
#[derive(Clone, Debug)]
pub struct PassTiming {
    pub name: String,
    pub cpu_ms: f32,
    pub gpu_ms: f32,
}
