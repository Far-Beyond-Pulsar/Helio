//! # nebula-serialize
//!
//! Two serialization backends for Nebula bake outputs:
//!
//! - [`binary`] — Compact chunked binary format (`.nebula`) with optional
//!   per-chunk zstd compression.
//! - [`json_meta`] — Human-readable JSON metadata format, suitable for tool
//!   inspection and asset pipelines that process metadata separately from bulk
//!   texel data.
//!
//! Both backends implement the [`nebula_core::traits::BakeSerializer`] trait so
//! they are interchangeable in all Nebula APIs.

pub mod binary;
pub mod chunk;
pub mod json_meta;

pub use binary::NebulaBinarySerializer;
pub use json_meta::NebulaJsonSerializer;
pub use chunk::{ChunkTag, ChunkError};

// ── Compression level ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compression {
    None,
    /// zstd level 1 (fast, ~2× size reduction)
    Fast,
    /// zstd level 9 (balanced)
    Balanced,
    /// zstd level 19 (maximum, slow)
    Best,
}

impl Compression {
    pub fn zstd_level(self) -> i32 {
        match self {
            Self::None     => 0,
            Self::Fast     => 1,
            Self::Balanced => 9,
            Self::Best     => 19,
        }
    }
}

impl Default for Compression { fn default() -> Self { Self::Balanced } }
