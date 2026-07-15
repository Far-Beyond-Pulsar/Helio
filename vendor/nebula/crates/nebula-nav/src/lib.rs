pub mod baker;
pub mod config;
pub mod output;

pub use baker::NavBaker;
pub use config::NavConfig;
pub use output::{NavOutput, NavPolygon, NavVertex, CHUNK_TAG};

use nebula_serialize::chunk::ChunkTag;

/// Chunk tag for baked navigation mesh data.
pub const NAV_TAG: ChunkTag = CHUNK_TAG;
