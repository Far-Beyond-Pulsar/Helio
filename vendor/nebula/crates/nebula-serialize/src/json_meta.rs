/// A simple JSON metadata serializer.
///
/// This serializer writes a human-readable JSON file describing the bake
/// output's parameters and file layout (`*.nebula.json`).  Bulk texel/sample
/// data is referenced by byte offset inside an accompanying `*.nebula` binary.
///
/// Useful for:
/// - Asset pipeline tooling that needs to inspect outputs without decoding binary
/// - Diffing metadata between bake runs in version control
/// - Embedding bake parameters in editor project files

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum JsonSerError {
    #[error("serde_json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
}

pub struct NebulaJsonSerializer {
    pub pretty: bool,
}

impl Default for NebulaJsonSerializer {
    fn default() -> Self { Self { pretty: true } }
}

impl NebulaJsonSerializer {
    pub fn serialize_meta<W, T>(&self, writer: &mut W, value: &T) -> Result<(), JsonSerError>
    where
        W: std::io::Write,
        T: Serialize,
    {
        if self.pretty {
            serde_json::to_writer_pretty(writer, value)?;
        } else {
            serde_json::to_writer(writer, value)?;
        }
        Ok(())
    }

    pub fn deserialize_meta<R, T>(&self, reader: R) -> Result<T, JsonSerError>
    where
        R: std::io::Read,
        T: serde::de::DeserializeOwned,
    {
        Ok(serde_json::from_reader(reader)?)
    }
}

/// Metadata envelope written at the top of every `.nebula.json` file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NebulaMeta {
    pub nebula_version: u32,
    pub bake_date:      String,
    pub scene_name:     String,
    pub pass:           String,
    /// Arbitrary pass-specific key/value pairs.
    pub parameters:     serde_json::Value,
}
