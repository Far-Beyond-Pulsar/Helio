pub mod baker;
pub mod config;
pub mod output;

pub use baker::PvsBaker;
pub use config::PvsConfig;
pub use output::{PvsOutput, CHUNK_TAG};

use nebula_serialize::chunk::ChunkTag;

/// Chunk tag for baked Potentially Visible Set data.
pub const PVS_TAG: ChunkTag = CHUNK_TAG;
