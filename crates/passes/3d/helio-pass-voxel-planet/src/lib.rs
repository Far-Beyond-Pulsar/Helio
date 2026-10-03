//! Destructible voxel worlds: cube-sphere planets and planes.
//!
//! * [`grid`] — equal-angle cube-sphere cells aligned with gravity.
//! * [`terrain`] — pluggable terrain generators, evaluated bit-for-bit
//!   identically on CPU and GPU, built from [`noise`]; [`landform`] holds
//!   the built-in generators.
//! * [`edits`] — ordered brushes with exact integer containment.
//! * [`planet`] — canonical queries, materials and exact ray casts.
//! * `engine` — Helio GBuffer pass with GPU-driven clipmap residency.
pub mod column_index;
pub mod edits;
pub mod grid;
pub mod journal;
pub mod landform;
pub mod noise;
pub mod planet;
pub mod residency;
mod ridge_envelope;
mod visible_priority;
pub mod terrain;
pub mod windows;
#[cfg(feature = "engine")]
pub mod engine;

pub use edits::{Brush, BrushOp, BrushShape};
pub use grid::{Cell, Grid};
pub use planet::{Planet, PlanetRecipe, RayHit};
pub use terrain::{GeneratorInfo, TerrainField, TerrainGenerator, TerrainProgram, TerrainSource};
