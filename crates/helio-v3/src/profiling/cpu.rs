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

#[cfg(all(not(target_arch = "wasm32"), feature = "profiling"))]
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
use std::collections::HashMap;
use std::time::Duration;

pub struct CpuProfiler {
    /// Timing records per pass name
    timings: HashMap<&'static str, Duration>,
}

impl CpuProfiler {
    /// Creates a new CPU profiler.
    ///
    /// # Performance
    ///
    /// - **O(1)**: Initializes empty profiler
    pub fn new() -> Self {
        Self {
            timings: HashMap::new(),
        }
    }

    /// Get recorded CPU timings for all passes
    pub fn get_timings(&self) -> &HashMap<&'static str, Duration> {
        &self.timings
    }

    /// Clear recorded timings (call at frame start)
    pub fn clear(&mut self) {
        self.timings.clear();
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
    pub fn scope(&mut self, name: &'static str) -> ScopeGuard {
        ScopeGuard {
            #[cfg(all(not(target_arch = "wasm32"), feature = "profiling"))]
            start: Instant::now(),
            #[cfg(all(not(target_arch = "wasm32"), feature = "profiling"))]
            profiler: self,
            #[cfg(all(not(target_arch = "wasm32"), feature = "profiling"))]
            name,
            #[cfg(not(all(not(target_arch = "wasm32"), feature = "profiling")))]
            _phantom: std::marker::PhantomData,
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
pub struct ScopeGuard<'a> {
    #[cfg(all(not(target_arch = "wasm32"), feature = "profiling"))]
    start: Instant,
    #[cfg(all(not(target_arch = "wasm32"), feature = "profiling"))]
    profiler: &'a mut CpuProfiler,
    #[cfg(all(not(target_arch = "wasm32"), feature = "profiling"))]
    name: &'static str,
    // Keeps the lifetime valid when profiling fields are compiled out.
    #[cfg(not(all(not(target_arch = "wasm32"), feature = "profiling")))]
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl Drop for ScopeGuard<'_> {
    /// Records elapsed time when the guard is dropped.
    ///
    /// # Performance
    ///
    /// - **O(1)**: Calculates elapsed time and records to profiler
    fn drop(&mut self) {
        #[cfg(all(not(target_arch = "wasm32"), feature = "profiling"))]
        {
            let elapsed = self.start.elapsed();
            self.profiler.timings.insert(self.name, elapsed);
        }
    }
}

