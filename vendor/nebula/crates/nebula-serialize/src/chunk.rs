use crate::Compression;
use thiserror::Error;
use std::io::{Read, Write};

// ── Chunk tag ─────────────────────────────────────────────────────────────────

/// An open, extensible four-byte chunk type tag.
///
/// `ChunkTag` is a transparent `u32` newtype — **not** a closed enum.  Each
/// baker crate is responsible for declaring its own tag constant:
///
/// ```rust,ignore
/// // inside nebula-light
/// pub const CHUNK_TAG: nebula_serialize::ChunkTag =
///     nebula_serialize::ChunkTag::from_bytes(*b"LMAP");
/// ```
///
/// The central `nebula-serialize` crate only defines the three infrastructure
/// tags below (`HEADER`, `METADATA`, `END`).  This keeps the dependency graph
/// clean: foundation crates never reference their consumers.
///
/// # Tag allocation convention
/// - 4 printable ASCII bytes, big-endian
/// - Uppercase for official Nebula passes, lowercase for third-party
/// - Register your tag in `docs/chunk-tags.md` in the repo
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkTag(pub u32);

impl ChunkTag {
    /// Construct a tag from a four-byte ASCII literal, e.g. `*b"LMAP"`.
    pub const fn from_bytes(b: [u8; 4]) -> Self {
        Self(u32::from_be_bytes(b))
    }

    /// Returns the four raw bytes (big-endian).
    pub const fn to_bytes(self) -> [u8; 4] {
        self.0.to_be_bytes()
    }

    // ── Infrastructure tags (known to nebula-serialize) ────────────────────
    /// File header — always the first chunk.
    pub const HEADER:   Self = Self::from_bytes(*b"NEBU");
    /// Per-chunk UTF-8 JSON metadata sidecar.
    pub const METADATA: Self = Self::from_bytes(*b"META");
    /// Sentinel — final chunk in the file.
    pub const END:      Self = Self::from_bytes(*b"END\0");

    pub fn is_end(self) -> bool { self == Self::END }
}

// ── Chunk header on-disk layout ───────────────────────────────────────────────
//
//  [tag: u32 BE] [flags: u32] [uncompressed_len: u64 LE] [compressed_len: u64 LE] [data: bytes]
//
//  flags bit 0 = compressed (zstd)

#[derive(Clone, Copy, Debug)]
pub struct ChunkFlags(pub u32);
impl ChunkFlags {
    pub const COMPRESSED: u32 = 0x01;
    pub fn is_compressed(self) -> bool { self.0 & Self::COMPRESSED != 0 }
}

// ── Write ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum ChunkError {
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("magic mismatch: expected NEBULA\\0\\0, got {0:?}")]
    MagicMismatch(Vec<u8>),
    #[error("version unsupported: {0}")]
    UnsupportedVersion(u32),
}

/// Current file format version.
pub const FORMAT_VERSION: u32 = 1;
pub const MAGIC: &[u8; 8]     = b"NEBULA\0\0";

pub fn write_file_header<W: Write>(w: &mut W) -> Result<(), ChunkError> {
    w.write_all(MAGIC)?;
    w.write_all(&FORMAT_VERSION.to_le_bytes())?;
    Ok(())
}

pub fn write_chunk<W: Write>(
    w:           &mut W,
    tag:         ChunkTag,
    data:        &[u8],
    compression: Compression,
) -> Result<(), ChunkError> {
    let (flags, payload): (u32, Vec<u8>) = if compression == Compression::None {
        (0, data.to_vec())
    } else {
        let compressed = zstd::encode_all(data, compression.zstd_level())?;
        (ChunkFlags::COMPRESSED, compressed)
    };

    let uncompressed_len = data.len() as u64;
    let compressed_len   = payload.len() as u64;

    w.write_all(&tag.to_bytes())?;
    w.write_all(&flags.to_le_bytes())?;
    w.write_all(&uncompressed_len.to_le_bytes())?;
    w.write_all(&compressed_len.to_le_bytes())?;
    w.write_all(&payload)?;
    Ok(())
}

pub fn write_end_chunk<W: Write>(w: &mut W) -> Result<(), ChunkError> {
    w.write_all(&ChunkTag::END.to_bytes())?;
    w.write_all(&0u32.to_le_bytes())?; // flags
    w.write_all(&0u64.to_le_bytes())?; // uncompressed_len
    w.write_all(&0u64.to_le_bytes())?; // compressed_len
    Ok(())
}

// ── Read ──────────────────────────────────────────────────────────────────────

pub struct RawChunk {
    pub tag:  ChunkTag,
    pub data: Vec<u8>,
}

pub fn read_file_header<R: Read>(r: &mut R) -> Result<u32, ChunkError> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(ChunkError::MagicMismatch(magic.to_vec()));
    }
    let mut ver = [0u8; 4];
    r.read_exact(&mut ver)?;
    let version = u32::from_le_bytes(ver);
    if version > FORMAT_VERSION {
        return Err(ChunkError::UnsupportedVersion(version));
    }
    Ok(version)
}

pub fn read_next_chunk<R: Read>(r: &mut R) -> Result<Option<RawChunk>, ChunkError> {
    let mut tag_bytes = [0u8; 4];
    match r.read_exact(&mut tag_bytes) {
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(ChunkError::Io(e)),
        Ok(_)  => {}
    }
    let tag = ChunkTag(u32::from_be_bytes(tag_bytes));

    if tag.is_end() { return Ok(None); }

    let mut flags_bytes = [0u8; 4];
    r.read_exact(&mut flags_bytes)?;
    let flags = ChunkFlags(u32::from_le_bytes(flags_bytes));

    let mut ul = [0u8; 8]; r.read_exact(&mut ul)?;
    let uncompressed_len = u64::from_le_bytes(ul) as usize;

    let mut cl = [0u8; 8]; r.read_exact(&mut cl)?;
    let compressed_len = u64::from_le_bytes(cl) as usize;

    let mut payload = vec![0u8; compressed_len];
    r.read_exact(&mut payload)?;

    let data = if flags.is_compressed() {
        zstd::decode_all(std::io::Cursor::new(&payload))
            .map_err(ChunkError::Io)?
    } else {
        payload
    };

    debug_assert_eq!(data.len(), uncompressed_len, "chunk decompressed size mismatch");
    Ok(Some(RawChunk { tag, data }))
}
