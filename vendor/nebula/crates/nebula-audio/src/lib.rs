pub mod baker;
pub mod config;
pub mod output;

pub use baker::AcousticBaker;
pub use config::AcousticConfig;
pub use output::{AcousticOutput, ImpulseResponse, ReverbZone, CHUNK_TAG};

use nebula_serialize::chunk::ChunkTag;

/// Chunk tag for baked acoustic / room-impulse-response data.
pub const ACOUSTIC_TAG: ChunkTag = CHUNK_TAG;
