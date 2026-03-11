//! Shadow matrix constants (CPU-side).
//! Matrix computation for point lights, directional CSM, and spot lights
//! is now handled fully on the GPU via shadow_matrices.wgsl.

/// World-space distance thresholds for the 4 CSM cascade boundaries.
/// Cascade i covers [CSM_SPLITS[i-1], CSM_SPLITS[i]] (cascade 0 starts at 0).
pub(crate) const CSM_SPLITS: [f32; 4] = [16.0, 80.0, 300.0, 1400.0];
