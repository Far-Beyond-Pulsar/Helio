use nebula_serialize::{chunk::{ChunkTag, ChunkError, FORMAT_VERSION, MAGIC}, Compression};
use std::io::Cursor;

// ── ChunkTag ──────────────────────────────────────────────────────────────────

#[test]
fn chunk_tag_from_bytes_roundtrip() {
    let tag = ChunkTag::from_bytes(*b"LMAP");
    let back = tag.to_bytes();
    assert_eq!(back, *b"LMAP");
}

#[test]
fn chunk_tag_equality() {
    let a = ChunkTag::from_bytes(*b"LMAP");
    let b = ChunkTag::from_bytes(*b"LMAP");
    assert_eq!(a, b);
}

#[test]
fn chunk_tag_inequality() {
    let a = ChunkTag::from_bytes(*b"LMAP");
    let b = ChunkTag::from_bytes(*b"NAVM");
    assert_ne!(a, b);
}

#[test]
fn chunk_tag_infrastructure_header() {
    let tag = ChunkTag::HEADER;
    assert_eq!(tag.to_bytes(), *b"NEBU");
}

#[test]
fn chunk_tag_infrastructure_metadata() {
    let tag = ChunkTag::METADATA;
    assert_eq!(tag.to_bytes(), *b"META");
}

#[test]
fn chunk_tag_infrastructure_end() {
    let tag = ChunkTag::END;
    assert!(tag.is_end());
}

#[test]
fn chunk_tag_non_end_is_not_end() {
    let tag = ChunkTag::from_bytes(*b"LMAP");
    assert!(!tag.is_end());
}

#[test]
fn chunk_tag_u32_inner_is_big_endian() {
    // "LMAP" == 0x4C4D4150 in big-endian
    let tag = ChunkTag::from_bytes(*b"LMAP");
    assert_eq!(tag.0, 0x4C4D4150_u32);
}

#[test]
fn chunk_tag_hash_is_consistent() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(ChunkTag::from_bytes(*b"LMAP"));
    set.insert(ChunkTag::from_bytes(*b"NAVM"));
    set.insert(ChunkTag::from_bytes(*b"AUIR"));
    assert_eq!(set.len(), 3);
    assert!(set.contains(&ChunkTag::from_bytes(*b"LMAP")));
}

#[test]
fn chunk_tag_debug_includes_tag() {
    let tag = ChunkTag::from_bytes(*b"PVSS");
    let s = format!("{:?}", tag);
    assert!(s.contains("ChunkTag"));
}

// ── Compression ───────────────────────────────────────────────────────────────

#[test]
fn compression_none_level_is_zero() {
    assert_eq!(Compression::None.zstd_level(), 0);
}

#[test]
fn compression_fast_level_is_1() {
    assert_eq!(Compression::Fast.zstd_level(), 1);
}

#[test]
fn compression_balanced_level_is_9() {
    assert_eq!(Compression::Balanced.zstd_level(), 9);
}

#[test]
fn compression_best_level_is_19() {
    assert_eq!(Compression::Best.zstd_level(), 19);
}

#[test]
fn compression_levels_are_monotone() {
    let none = Compression::None.zstd_level();
    let fast = Compression::Fast.zstd_level();
    let balanced = Compression::Balanced.zstd_level();
    let best = Compression::Best.zstd_level();
    assert!(none < fast);
    assert!(fast < balanced);
    assert!(balanced < best);
}

#[test]
fn compression_default_is_balanced() {
    assert_eq!(Compression::default(), Compression::Balanced);
}

#[test]
fn compression_equality() {
    assert_eq!(Compression::Fast, Compression::Fast);
    assert_ne!(Compression::Fast, Compression::Best);
}

// ── File header write / read round-trip ──────────────────────────────────────

#[test]
fn write_file_header_writes_magic_bytes() {
    use nebula_serialize::chunk::write_file_header;
    let mut buf = Vec::new();
    write_file_header(&mut buf).expect("write_file_header should succeed");
    assert!(buf.starts_with(MAGIC), "file should start with NEBULA magic");
}

#[test]
fn write_then_read_file_header() {
    use nebula_serialize::chunk::{write_file_header, read_file_header};
    let mut buf = Vec::new();
    write_file_header(&mut buf).expect("write");
    let mut cursor = Cursor::new(&buf);
    read_file_header(&mut cursor).expect("read");
}

#[test]
fn file_header_wrong_magic_returns_error() {
    use nebula_serialize::chunk::read_file_header;
    let garbage = b"GARBAGE!".to_vec();
    let mut cursor = Cursor::new(garbage);
    let result = read_file_header(&mut cursor);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), ChunkError::MagicMismatch(_)));
}

#[test]
fn format_version_is_at_least_one() {
    assert!(FORMAT_VERSION >= 1);
}
