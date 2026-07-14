//! Bounded GPU residency for the production planetary voxel path.
//!
//! This crate is opt-in and is not registered in Helio's default graph. The
//! current milestone owns only shared page buffers, lookup tables, update
//! ordering, lifecycle rebuilds, and validation. Surface extraction and draws
//! are deliberately separate promotion gates.

mod config;
mod gpu;
mod table;

pub use config::*;
pub use gpu::*;
pub use table::*;

pub const RESIDENCY_WGSL: &str = include_str!("residency.wgsl");
