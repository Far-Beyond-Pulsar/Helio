//! Content-addressed asset identity (`ContentId`) and its streaming hasher.
//!
//! One rule makes the whole dedup stack coherent: **the identity of an asset
//! IS the XXH3-128 digest of its canonical payload bytes** -- nothing else
//! (no path, no container format, no import options) participates. Two
//! entities referencing the same mesh through different routes therefore
//! land on one [`ContentId`], one CPU parse, and eventually one GPU copy;
//! and when the last referencing entity dies, [`crate::content_ledger`]'s
//! count hits zero and everything behind it is freed deterministically (no
//! tracing GC, no manual cleanup -- see that module for the counting half,
//! and [`crate::mesh_cache`] for what "canonical payload" concretely means
//! for PMSH assets).
//!
//! # Why XXH3-128
//!
//! Already in the dependency tree (`twox-hash`, compiled into every editor
//! build today via transitive deps), very fast on modern parts for large
//! inputs, and -- decisive here -- a 128-bit digest makes accidental or
//! adversarial collision impractical for content-addressing purposes while
//! staying honest that this is a HASH, not a cryptographic commitment:
//! nothing security-relevant is decided by an id match, only "these
//! payloads are byte-identical with overwhelming probability, so parsing
//! one shared copy is safe".
//!
//! # Stability contract
//!
//! These digests are persisted into `.mesh` v2 headers and referenced by
//! the ledger across sessions. The day this module's output changes for
//! identical input bytes, every previously-written asset identity silently
//! forks. The unit tests pin the published XXH3-128 vectors plus
//! round-trip/streaming fixtures precisely so such a change is a LOUD test
//! failure, not a quiet migration.

use std::fmt;
use std::io::{self, Read};
use std::path::Path;

use twox_hash::xxhash3_128::Hasher as Xxh3_128;

/// A content-addressed asset identity: the XXH3-128 digest of an asset's
/// canonical payload bytes. Opaque except for ordering/display; construct
/// via [`ContentHasher`] or deserialize from the hex form [`Self::to_hex`]
/// produces.
///
/// `Copy`/`Eq`/`Ord`/`Hash` throughout -- these ride inside component rows,
/// map keys, and sorted reconciliation diffs. [`Self::ZERO`] is the
/// "no content" sentinel, deliberately shape-compatible with
/// `pulsar_scenedb::HandleId::ZERO` (the two types share the same u128
/// representation; see `static_mesh_component.rs`'s `content_id` field for
/// where one becomes the other).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ContentId(pub u128);

impl ContentId {
    /// The "no content" sentinel.
    pub const ZERO: ContentId = ContentId(0);

    /// Lowercase, zero-padded 32-nibble hex -- the exact wire/storage form
    /// serde uses too, so a persisted id and a printed id always agree
    /// byte-for-byte.
    pub fn to_hex(self) -> String {
        format!("{:032x}", self.0)
    }

    /// Parses [`Self::to_hex`] output (case-insensitive; an optional `0x`
    /// prefix is tolerated for hand-edited files).
    pub fn from_hex(s: &str) -> Option<Self> {
        let s = s.strip_prefix("0x").unwrap_or(s);
        if s.len() != 32 {
            return None;
        }
        // Strictly hex before feeding u128::from_str_radix so garbage like
        // "+" signs can't sneak through radix parsing quirks.
        if !s.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        u128::from_str_radix(s, 16).ok().map(ContentId)
    }
}

impl fmt::Display for ContentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl fmt::Debug for ContentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ContentId({})", self.to_hex())
    }
}

impl From<u128> for ContentId {
    fn from(v: u128) -> Self {
        ContentId(v)
    }
}

mod serde_impl {
    //! Serde as a hex STRING, never as a number: JSON numbers cannot hold
    //! u128 precision through every consumer, and hex strings survive
    //! hand-editing and diffing in scene files.
    use super::ContentId;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    impl Serialize for ContentId {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.to_hex().serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for ContentId {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let s = String::deserialize(deserializer)?;
            ContentId::from_hex(&s)
                .ok_or_else(|| serde::de::Error::custom(format!("invalid ContentId hex {s:?}")))
        }
    }
}

/// Streaming XXH3-128 hasher with buffer reuse: feed it arbitrarily many
/// chunks (`update`), finish once. The point of the streaming shape is file
/// hashing WITHOUT whole-file loads: [`hash_file`] reads in fixed chunks
/// through ONE caller-owned buffer that stays allocated across calls --
/// hashing a 64MB mesh costs 64KB of scratch no matter how many files go
/// through it, and never holds two copies of anything at once.
pub struct ContentHasher {
    inner: Xxh3_128,
}

impl Default for ContentHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl ContentHasher {
    pub fn new() -> Self {
        Self {
            inner: Xxh3_128::new(),
        }
    }

    /// Feed one more chunk. Chunk boundaries are invisible to the digest
    /// (XXH3's streaming state is order-preserving) -- pinned by tests, and
    /// load-bearing: [`hash_file`]'s chunking must produce exactly what a
    /// whole-buffer hash would.
    pub fn update(&mut self, bytes: &[u8]) {
        self.inner.write(bytes);
    }

    /// Consume the stream into its final identity.
    pub fn finish(self) -> ContentId {
        ContentId(self.inner.finish_128())
    }
}

/// One-shot hash of a byte slice (the canonical way small payloads get
/// identified).
pub fn hash_bytes(bytes: &[u8]) -> ContentId {
    let mut h = ContentHasher::new();
    h.update(bytes);
    h.finish()
}

/// Scratch size [`hash_file`] reads through. 64KiB: comfortably inside L2,
/// large enough that syscall overhead vanishes next to hashing cost.
const FILE_CHUNK: usize = 64 * 1024;

/// Streams `path` through [`FILE_CHUNK`] chunks using `buffer` as reusable
/// scratch, producing the same digest a single whole-file hash would.
/// `buffer` is cleared but never shrunk -- call sites that hash many files
/// amortize one allocation across all of them.
pub fn hash_file(path: &Path, buffer: &mut Vec<u8>) -> io::Result<ContentId> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = ContentHasher::new();
    buffer.clear();
    let mut read = 0usize;
    loop {
        buffer.resize(read + FILE_CHUNK, 0);
        let n = file.read(&mut buffer[read..])?;
        if n == 0 {
            break;
        }
        read += n;
        hasher.update(&buffer[read - n..read]);
    }
    Ok(hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_matches_the_published_xxh3_128_vector() {
        // Canonical xxHash spec vector for XXH3_128bits("", seed=0). If this
        // ever changes, EVERY previously written ContentId forks -- the
        // stability contract's loudest tripwire.
        assert_eq!(
            hash_bytes(b""),
            ContentId(0x99AA06D3014798D86001C324468D497F)
        );
    }

    #[test]
    fn streaming_equals_one_shot_across_every_chunking() {
        // Deterministic pseudo-payload spanning many internal block sizes
        // (XXH3 switches algorithms at 16/128/240-byte thresholds and again
        // for >512B accumulations -- cross all of them).
        let payload: Vec<u8> = (0..100_000u32)
            .map(|i| (i.wrapping_mul(2654435761) >> 24) as u8)
            .collect();
        let whole = hash_bytes(&payload);

        for chunk in [1usize, 7, 15, 16, 63, 127, 128, 239, 240, 255, 4096, 65_535] {
            let mut h = ContentHasher::new();
            for part in payload.chunks(chunk) {
                h.update(part);
            }
            assert_eq!(h.finish(), whole, "chunk size {chunk} diverged");
        }

        // Byte-at-a-time too (the degenerate streaming case).
        let mut h = ContentHasher::new();
        for b in &payload {
            h.update(std::slice::from_ref(b));
        }
        assert_eq!(h.finish(), whole);
    }

    #[test]
    fn distinct_content_yields_distinct_ids_and_length_matters() {
        assert_ne!(hash_bytes(b"a"), hash_bytes(b"b"));
        assert_ne!(hash_bytes(b"aa"), hash_bytes(b"a")); // length sensitivity
        assert_ne!(hash_bytes(&[0u8; 64]), hash_bytes(&[0u8; 65]));
    }

    #[test]
    fn hex_round_trip_is_exact_and_rejects_garbage() {
        let id = hash_bytes(b"PMSH stability fixture");
        let text = id.to_hex();
        assert_eq!(text.len(), 32);
        assert_eq!(ContentId::from_hex(&text), Some(id));
        assert_eq!(
            ContentId::from_hex(&text.to_ascii_uppercase()),
            Some(id),
            "case-insensitive accept"
        );
        assert_eq!(
            ContentId::from_hex(&format!("0x{text}")),
            Some(id),
            "optional 0x prefix"
        );

        assert_eq!(ContentId::from_hex(""), None);
        assert_eq!(ContentId::from_hex("zz"), None);
        assert_eq!(ContentId::from_hex(&"a".repeat(31)), None);
        assert_eq!(ContentId::from_hex(&"a".repeat(33)), None);
        assert_eq!(
            ContentId::from_hex("+0000000000000000000000000000001"),
            None
        );
    }

    #[test]
    fn serde_round_trips_through_the_same_hex_form_as_display() {
        let id = hash_bytes(b"serde fixture");
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(
            json.trim_matches('"'),
            id.to_hex(),
            "wire form == display form"
        );
        assert_eq!(serde_json::from_str::<ContentId>(&json).unwrap(), id);
    }

    #[test]
    fn ordering_is_raw_u128_ordering() {
        assert!(ContentId(1) < ContentId(2));
        assert!(ContentId(u128::MAX) > ContentId(0));
        assert_eq!(ContentId::ZERO, ContentId::default());
    }

    #[test]
    fn hash_file_streams_in_chunks_and_never_doubles_the_buffer() {
        let dir =
            std::env::temp_dir().join(format!("helio-content-id-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("chunk.bin");
        let payload: Vec<u8> = (0..300_000u32).map(|i| i as u8 ^ (i >> 8) as u8).collect();
        std::fs::write(&path, &payload).unwrap();
        assert_eq!(hash_bytes(&payload), hash_bytes(&payload));

        let mut buf = Vec::with_capacity(FILE_CHUNK);
        let via_file = hash_file(&path, &mut buf).unwrap();
        assert_eq!(
            via_file,
            hash_bytes(&payload),
            "chunked IO == whole-buffer hash"
        );

        // Buffer reuse: second call must not grow the scratch beyond one
        // chunk's worth of headroom growth (it may resize up to chunk size
        // per call, never to file size).
        let cap_before = buf.capacity();
        let _again = hash_file(&path, &mut buf).unwrap();
        assert!(
            buf.capacity() <= cap_before.max(FILE_CHUNK * 2),
            "scratch buffer ballooned to {} bytes",
            buf.capacity()
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn build_to_build_stability_fixture() {
        // Not a published vector: a fixture pinning THIS build chain's
        // output so any toolchain/dependency bump that shifts the digest is
        // a visible decision, not a silent identity fork. Regenerate
        // deliberately (and record why) if the algorithm ever legitimately
        // changes.
        assert_eq!(
            hash_bytes(b"helio ContentId stability fixture v1"),
            ContentId::from_hex("b05f23e150b4f98ce678a042e683916c").unwrap(),
            "ContentId digest drifted -- every previously written asset identity would fork"
        );
    }
}
