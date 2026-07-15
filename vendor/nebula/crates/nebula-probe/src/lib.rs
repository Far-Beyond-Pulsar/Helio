pub mod baker;
pub mod config;
pub mod output;

pub use baker::ProbeBaker;
pub use config::ProbeConfig;
pub use output::{IrradianceOutput, ReflectionOutput, ShCoeff, IRRADIANCE_CHUNK_TAG, REFLECTION_CHUNK_TAG};

use nebula_serialize::chunk::ChunkTag;

/// Reflection / specular cubemap probe tag.
pub const REFLECTION_TAG: ChunkTag = REFLECTION_CHUNK_TAG;
/// Irradiance SH probe tag.
pub const IRRADIANCE_TAG: ChunkTag = IRRADIANCE_CHUNK_TAG;
