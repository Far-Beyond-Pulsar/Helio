//! # nebula-ao
//!
//! GPU hemisphere-sampled **ambient occlusion** baking.
//!
//! For each world-space surface point (derived from lightmap UVs), `AoBaker`
//! fires N uniformly-distributed hemisphere rays and counts the fraction that
//! escape the scene without hitting any geometry.  The result is a single-
//! channel R32F texture where 1.0 = fully unoccluded, 0.0 = fully occluded.

pub mod baker;
pub mod config;
pub mod output;

pub use baker::AoBaker;
pub use config::AoConfig;
pub use output::AoOutput;

/// Chunk tag for the `.nebula` binary format — declared here, not in nebula-serialize.
pub use nebula_serialize::ChunkTag;
pub const CHUNK_TAG: ChunkTag = ChunkTag::from_bytes(*b"AAOC");
