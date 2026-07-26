//! Bounded GPU residency for the production planetary voxel path.
//!
//! This crate is opt-in and is not registered in Helio's default graph. The
//! current milestone owns only shared page buffers, lookup tables, update
//! ordering, lifecycle rebuilds, and validation. Surface extraction and draws
//! are deliberately separate promotion gates.

mod config;
mod extraction;
mod fixture;
mod gpu;
mod render;
mod table;
mod transvoxel;
mod transvoxel_emit;
mod transvoxel_gpu;
mod transvoxel_transition;
mod transvoxel_transition_gpu;

pub use config::*;
pub use extraction::*;
pub use fixture::*;
pub use gpu::*;
pub use render::*;
pub use table::*;
pub use transvoxel::*;
pub use transvoxel_emit::*;
pub use transvoxel_gpu::*;
pub use transvoxel_transition::*;
pub use transvoxel_transition_gpu::*;

pub const EXTRACTION_LAYOUT_WGSL: &str = include_str!("extraction_layout.wgsl");
pub const RESIDENCY_WGSL: &str = include_str!("residency.wgsl");
pub const TRANSVOXEL_CLASSIFY_WGSL: &str = include_str!("transvoxel_classify.wgsl");
pub const TRANSVOXEL_EMIT_WGSL: &str = include_str!("transvoxel_emit.wgsl");
pub const TRANSVOXEL_TRANSITION_GPU_WGSL: &str = include_str!("transvoxel_transition_gpu.wgsl");

/// Winner of the recorded planetary extraction bake-off. The production
/// request and publication contracts intentionally expose no runtime selector.
pub const PRODUCTION_EXTRACTION_ALGORITHM: &str = "gpu_transvoxel";
