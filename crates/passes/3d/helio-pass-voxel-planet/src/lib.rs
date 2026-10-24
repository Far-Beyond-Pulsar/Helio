//! Destructible voxel worlds: cube-sphere planets and planes.
//!
//! * [`grid`] — equal-angle cube-sphere cells aligned with gravity.
//! * [`terrain`] — pluggable terrain generators, evaluated bit-for-bit
//!   identically on CPU and GPU, built from [`noise`]; [`layers`] is the
//!   built-in one (ordered layer stacks, interpreted by [`landform`]).
//! * [`edits`] — ordered brushes with exact integer containment.
//! * [`planet`] — canonical queries, materials and exact ray casts.
//! * [`features`] — procedural content (trees, rocks, buildings) generated
//!   like the terrain (docs/features.md), from [`stamp`] models; [`flora`]
//!   grows natural ones (trees, shrubs, boulders, craters), and a
//!   [`feature_recipe`] describes a world's (what grows where, in metres)
//!   for any voxel size and seed.
//! * `chunks` — chunk trees in an octree (docs/chunk-octree.md): GPU
//!   residency, drawing and lighting, and the Helio pass.
//! * `engine` — what hosts share with the pass (frames, picks) and the
//!   terrain half of the shaders, checked against the CPU field.
pub mod edit_store;
pub mod edits;
pub mod feature_recipe;
pub mod features;
pub mod flora;
pub mod snapshot;
pub mod stamp;
pub mod grid;
pub mod record;
pub mod landform;
pub mod layers;
pub mod noise;
pub mod planet;
mod ridge_envelope;
pub mod terrain;
#[cfg(feature = "engine")]
pub mod engine;
#[cfg(feature = "engine")]
pub mod chunks;
#[cfg(feature = "engine")]
pub mod meadow;

pub use edits::{Brush, BrushOp, BrushShape};
pub use grid::{Cell, Grid};
pub use planet::{Planet, PlanetRecipe, RayHit};
pub use terrain::{GeneratorInfo, TerrainField, TerrainGenerator, TerrainProgram, TerrainSource};
