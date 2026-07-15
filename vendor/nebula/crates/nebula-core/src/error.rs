use thiserror::Error;

#[derive(Debug, Error)]
pub enum NebulaError {
    #[error("GPU error: {0}")]
    Gpu(String),

    #[error("serialization error: {0}")]
    Serialize(String),

    #[error("deserialization error: {0}")]
    Deserialize(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid scene geometry: {0}")]
    InvalidScene(String),

    #[error("bake pass '{pass}' failed: {reason}")]
    BakeFailed { pass: String, reason: String },

    #[error("unsupported feature: {0}")]
    Unsupported(String),

    #[error("buffer read-back timed out after {ms}ms")]
    ReadbackTimeout { ms: u64 },
}
