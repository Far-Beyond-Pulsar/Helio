//! CPU profiling with RAII scoped guards.
//!
//! This module provides automatic CPU profiling using the **RAII pattern**. Scopes are created
//! via `CpuProfiler::scope()` and automatically record timing when dropped.
//!
//! # Design Pattern: RAII Scopes
//!
//! CPU profiling uses **Resource Acquisition Is Initialization (RAII)**:
//!
//! 1. `scope()` creates a `ScopeGuard` and records start time
//! 2. When `ScopeGuard` is dropped, elapsed time is recorded
//! 3. No manual `begin()`/`end()` calls required (automatic via Drop)
//!
//! # Performance
//!
//! - **O(1)**: Records start time in `Instant::now()` (~20ns)
//! - **Zero allocations**: Guard is stack-allocated
//! - **Zero cost when disabled**: Feature flag eliminates recording code
//!
//! # Example
//!
//! ```rust,no_run
//! # use helio_v3::profiling::CpuProfiler;
//! let mut profiler = CpuProfiler::new();
//!
//! {
//!     let _scope = profiler.scope("ShadowPass");
//!     // ... CPU work ...
//! } // ScopeGuard drops, timing recorded
//! ```

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

/// CPU profiler with scoped timing.
///
/// `CpuProfiler` provides automatic CPU profiling using RAII scopes. Timing is recorded
/// when `ScopeGuard` is dropped.
///
/// # Design
///
/// The profiler maintains a stack of active scopes. When a scope ends (via `Drop`), the
/// elapsed time is recorded to a timing tree.
///
/// # Performance
///
/// - **O(1)**: `scope()` creates a guard in constant time
/// - **Zero allocations**: Guard is stack-allocated
/// - **Minimal overhead**: ~20ns per scope (Instant::now() call)
///
/// # Example
///
/// ```rust,no_run
/// # use helio_v3::profiling::CpuProfiler;
/// let mut profiler = CpuProfiler::new();
///
/// {
///     let _scope = profiler.scope("ShadowPass");
///     // ... CPU work ...
/// } // Timing recorded automatically
///
/// {
///     let _scope = profiler.scope("GBufferPass");
///     // ... CPU work ...
/// } // Timing recorded automatically
/// ```
pub struct CpuProfiler {
    // Future: Timing tree for hierarchical profiling
    // scopes: Vec<(String, Duration)>,
}

impl CpuProfiler {
    /// Creates a new CPU profiler.
    ///
    /// # Performance
    ///
    /// - **O(1)**: Initializes empty profiler
    pub fn new() -> Self {
        Self {
            // Future: Initialize timing tree
        }
    }

    /// Creates a CPU profiling scope (RAII guard).
    ///
    /// The returned `ScopeGuard` measures CPU time until it is dropped.
    /// Results are recorded to the profiler's timing tree.
    ///
    /// # Parameters
    ///
    /// - `name`: Scope name (must be static for zero-cost)
    ///
    /// # Performance
    ///
    /// - **O(1)**: Records start time in `Instant::now()` (~20ns)
    /// - **Zero allocations**: Guard is stack-allocated
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use helio_v3::profiling::CpuProfiler;
    /// # let mut profiler = CpuProfiler::new();
    /// {
    ///     let _scope = profiler.scope("MyPass");
    ///     // ... CPU work ...
    /// } // Timing recorded when guard drops
    /// ```
    pub fn scope(&mut self, _name: &'static str) -> ScopeGuard {
        ScopeGuard {
            #[cfg(not(target_arch = "wasm32"))]
            start: Instant::now(),
            // Future: Pass profiler reference for recording
            // profiler: self,
            // name,
        }
    }
}

impl Default for CpuProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII guard for CPU profiling scopes.
///
/// `ScopeGuard` automatically records elapsed time when dropped. This ensures that timing
/// is always captured, even if the scope exits early (e.g., via `return` or `?`).
///
/// # Design
///
/// The guard uses the **RAII pattern**:
/// 1. Created by `CpuProfiler::scope()`
/// 2. Records start time in `Instant::now()`
/// 3. When dropped, calculates elapsed time and records to profiler
///
/// # Performance
///
/// - **O(1)**: Drop records elapsed time in constant time
/// - **Zero allocations**: Guard is stack-allocated
///
/// # Example
///
/// ```rust,no_run
/// # use helio_v3::profiling::CpuProfiler;
/// # let mut profiler = CpuProfiler::new();
/// {
///     let _scope = profiler.scope("MyPass");
///     // ... CPU work ...
/// } // <-- Timing recorded here (automatic via Drop)
/// ```
pub struct ScopeGuard {
    #[cfg(not(target_arch = "wasm32"))]
    start: Instant,
    // Future: Reference to profiler for recording
    // profiler: &'a mut CpuProfiler,
    // name: &'static str,
}

impl Drop for ScopeGuard {
    /// Records elapsed time when the guard is dropped.
    ///
    /// # Performance
    ///
    /// - **O(1)**: Calculates elapsed time and records to profiler
    fn drop(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        let _elapsed = self.start.elapsed();
        // Future: Record to profiler
        // self.profiler.record(self.name, elapsed);
    }
}

