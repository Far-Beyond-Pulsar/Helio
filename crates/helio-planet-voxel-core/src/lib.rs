//! Renderer-facing contracts for the production planetary voxel path.
//!
//! Pulsar owns canonical terrain state. This crate deliberately owns only the
//! bounded messages, GPU layouts, and residency model consumed by Helio. It has
//! no renderer, scene, persistence, physics, or dependency on the retained
//! fixed-volume voxel implementation.

mod cache;
mod contract;
mod gpu;
mod types;

pub use cache::*;
pub use contract::*;
pub use gpu::*;
pub use types::*;

/// WGSL declarations that must remain byte-compatible with the public GPU POD
/// types in this crate.
pub const PLANET_VOXEL_LAYOUT_WGSL: &str = include_str!("planet_voxel_layout.wgsl");
