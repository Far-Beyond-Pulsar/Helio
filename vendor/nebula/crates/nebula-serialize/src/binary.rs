use crate::{chunk::*, Compression};
use nebula_core::{traits::{BakeOutput, BakeSerializer}, NebulaError};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BinarySerError {
    #[error(transparent)]
    Chunk(#[from] ChunkError),
    #[error("bincode: {0}")]
    Bincode(String),
}

impl From<BinarySerError> for NebulaError {
    fn from(e: BinarySerError) -> Self { NebulaError::Serialize(e.to_string()) }
}

/// Configuration for the compact binary `.nebula` format.
#[derive(Clone, Debug)]
pub struct NebulaBinarySerializer {
    pub compression: Compression,
}

impl Default for NebulaBinarySerializer {
    fn default() -> Self { Self { compression: Compression::Balanced } }
}

/// Low-level helper: write any `bincode`-serializable value as a single chunk.
pub(crate) fn write_bincode_chunk<W, T>(
    w:           &mut W,
    tag:         ChunkTag,
    value:       &T,
    compression: Compression,
) -> Result<(), BinarySerError>
where
    W: std::io::Write,
    T: serde::Serialize,
{
    let encoded = bincode::serde::encode_to_vec(value, bincode::config::standard())
        .map_err(|e| BinarySerError::Bincode(e.to_string()))?;
    write_chunk(w, tag, &encoded, compression)?;
    Ok(())
}

/// Low-level helper: decode a chunk's data via `bincode`.
pub(crate) fn read_bincode_chunk<T>(data: &[u8]) -> Result<T, BinarySerError>
where
    T: serde::de::DeserializeOwned,
{
    let (value, _) = bincode::serde::decode_from_slice(data, bincode::config::standard())
        .map_err(|e| BinarySerError::Bincode(e.to_string()))?;
    Ok(value)
}
