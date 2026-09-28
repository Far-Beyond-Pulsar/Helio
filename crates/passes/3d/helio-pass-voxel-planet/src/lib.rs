//! Destructible cube-sphere voxel planet.
//!
//! * [`grid`] — equal-angle cube-sphere cells aligned with gravity.
//! * [`field`] — deterministic integer terrain shared bit-for-bit with WGSL.
//! * [`edits`] — ordered brushes with exact integer containment.
//! * [`planet`] — canonical queries, materials and exact ray casts.
//! * `engine` — Helio GBuffer pass with GPU-driven clipmap residency.
pub mod edits;
pub mod field;
pub mod grid;
pub mod journal;
pub mod planet;
pub mod residency;
pub mod windows;
#[cfg(feature = "engine")]
pub mod engine;

pub use edits::{Brush, BrushOp, BrushShape};
pub use grid::{Cell, Grid};
pub use planet::{Planet, PlanetRecipe, RayHit};
