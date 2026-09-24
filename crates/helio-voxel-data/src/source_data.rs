//! Transport-neutral contract for externally produced voxel chunks.
//!
//! These are CPU-side request/validation types only. They do not own payload
//! bytes, publish SceneDB state, schedule work, or imply GPU residency. The
//! owning service applies accepted data to live SceneDB component state.
//! Persistence/exfiltration is caller-owned.

use std::collections::{HashMap, HashSet};

/// Current version of the canonical raw chunk payload schema.
pub const VOXEL_CHUNK_SCHEMA_VERSION: u16 = 1;
/// Stable 128-bit identity for a registered voxel payload format.
pub type VoxelFormatId = u128;
/// Built-in format for 8 x 8 x 8 material samples in X-major order.
/// Zero is air; nonzero bytes are one-based indices into the owning
/// component's SceneDB material-ID palette. A short payload has an implicit
/// all-air tail, so scripts can send compact mostly-empty chunks.
pub const VOXEL_CHUNK_ENCODING_RAW: VoxelFormatId = 1;
pub const VOXEL_CHUNK_EDGE: usize = 8;
pub const VOXEL_CHUNK_SAMPLES: usize = VOXEL_CHUNK_EDGE * VOXEL_CHUNK_EDGE * VOXEL_CHUNK_EDGE;
/// One byte per sample in the built-in raw material format.
pub const MAX_VOXEL_CHUNK_PAYLOAD_BYTES: usize = VOXEL_CHUNK_SAMPLES;
/// Hard ceiling for a registered non-material payload. Each format may choose
/// a smaller bound; the batch and queue limits still apply independently.
pub const MAX_VOXEL_FORMAT_PAYLOAD_BYTES: usize = 16 * 1024 * 1024;
/// Upper bound on update descriptors in one validated batch.
pub const MAX_VOXEL_BATCH_UPDATES: usize = 65_536;
/// Default aggregate payload ceiling for one producer batch (256 MiB).
/// Services may configure a lower limit to fit their queue/frame budget.
pub const MAX_VOXEL_BATCH_PAYLOAD_BYTES: usize = 256 * 1024 * 1024;

/// Stable identity of one terrain component/source pair, assigned by its owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VoxelTerrainId(pub u128);

/// Stable identity/version namespace of an external generator or edit producer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VoxelSourceId(pub u128);

/// Canonical chunk address. Coordinates are signed chunk-space coordinates;
/// LOD zero is finest, and larger values represent progressively coarser data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VoxelChunkKey {
    pub x: i64,
    pub y: i64,
    pub z: i64,
    pub lod: u8,
}

impl VoxelChunkKey {
    pub const fn new(x: i64, y: i64, z: i64, lod: u8) -> Self {
        Self { x, y, z, lod }
    }
}

/// Domain policy for chunk coordinates. `Bounded` uses one explicit key range
/// at every LOD; `BoundedBase` derives each coarser range from LOD-zero keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoxelDomain {
    Unbounded {
        max_lod: u8,
    },
    Bounded {
        min: [i64; 3],
        max: [i64; 3],
        max_lod: u8,
    },
    /// Bounds are inclusive chunk coordinates at LOD zero. Coarser bounds
    /// are derived by signed floor division by `lod_scale ^ lod`.
    BoundedBase {
        min: [i64; 3],
        max: [i64; 3],
        max_lod: u8,
        lod_scale: u32,
    },
}

impl VoxelDomain {
    pub fn validate_key(self, key: VoxelChunkKey) -> Result<(), VoxelUpdateError> {
        let (max_lod, bounds) = match self {
            Self::Unbounded { max_lod } => (max_lod, None),
            Self::Bounded { min, max, max_lod } => {
                if (0..3).any(|axis| min[axis] > max[axis]) {
                    return Err(VoxelUpdateError::InvalidDomainBounds);
                }
                (max_lod, Some((min, max)))
            }
            Self::BoundedBase {
                min,
                max,
                max_lod,
                lod_scale,
            } => {
                if lod_scale == 0 || (0..3).any(|axis| min[axis] > max[axis]) {
                    return Err(VoxelUpdateError::InvalidDomainBounds);
                }
                let min = min.map(|value| coarsen_bound(value, lod_scale, key.lod));
                let max = max.map(|value| coarsen_bound(value, lod_scale, key.lod));
                (max_lod, Some((min, max)))
            }
        };
        if key.lod > max_lod {
            return Err(VoxelUpdateError::LodOutOfDomain {
                lod: key.lod,
                max_lod,
            });
        }
        if let Some((min, max)) = bounds {
            let coords = [key.x, key.y, key.z];
            if (0..3).any(|axis| coords[axis] < min[axis] || coords[axis] > max[axis]) {
                return Err(VoxelUpdateError::ChunkOutOfDomain(key));
            }
        }
        Ok(())
    }
}

fn coarsen_bound(value: i64, lod_scale: u32, lod: u8) -> i64 {
    match i128::from(lod_scale).checked_pow(u32::from(lod)) {
        Some(divisor) => i64::try_from(i128::from(value).div_euclid(divisor))
            .expect("quotient of an i64 by a positive integer fits i64"),
        None if value < 0 => -1,
        None => 0,
    }
}

/// Borrowed, versioned voxel payload. `encoding` is a stable 128-bit registered data
/// layout independently of the generator/source ID. Encoding 1, schema 1 is
/// the built-in 8³ material grid: `linear = z * 64 + y * 8 + x`, zero is air,
/// and 1..=255 index the SceneDB material-ID palette. Missing tail samples in
/// that format are air. Other formats define their own sample semantics.
/// This type makes no copy and must not be retained as renderer state.
#[derive(Clone, Copy, Debug)]
pub struct VoxelChunkPayload<'a> {
    pub encoding: VoxelFormatId,
    pub schema_version: u16,
    pub bytes: &'a [u8],
}

/// Validation contract for a payload format and schema version. The callback
/// checks format-specific structure after byte limits; no renderer dependency
/// enters the data API. A format must be registered before publication.
#[derive(Clone, Copy, Debug)]
pub struct VoxelFormatDescriptor {
    pub encoding: VoxelFormatId,
    pub schema_version: u16,
    pub min_bytes: usize,
    pub max_bytes: usize,
    pub validate: fn(&[u8]) -> bool,
}

/// Producer-side format capabilities. Cloning a registry copies only its
/// small descriptor table; sessions bind one immutable clone to admission and
/// publication so validation cannot change between those steps.
#[derive(Clone, Debug)]
pub struct VoxelFormatRegistry {
    descriptors: HashMap<(VoxelFormatId, u16), VoxelFormatDescriptor>,
}

impl Default for VoxelFormatRegistry {
    fn default() -> Self {
        let raw = VoxelFormatDescriptor {
            encoding: VOXEL_CHUNK_ENCODING_RAW,
            schema_version: VOXEL_CHUNK_SCHEMA_VERSION,
            min_bytes: 1,
            max_bytes: MAX_VOXEL_CHUNK_PAYLOAD_BYTES,
            validate: |_| true,
        };
        Self {
            descriptors: HashMap::from([((raw.encoding, raw.schema_version), raw)]),
        }
    }
}

impl VoxelFormatRegistry {
    pub fn register(&mut self, descriptor: VoxelFormatDescriptor) -> Result<(), VoxelUpdateError> {
        if descriptor.encoding == 0
            || descriptor.schema_version == 0
            || descriptor.min_bytes == 0
            || descriptor.min_bytes > descriptor.max_bytes
            || descriptor.max_bytes > MAX_VOXEL_FORMAT_PAYLOAD_BYTES
            || self
                .descriptors
                .contains_key(&(descriptor.encoding, descriptor.schema_version))
        {
            return Err(VoxelUpdateError::InvalidFormatDescriptor);
        }
        self.descriptors
            .insert((descriptor.encoding, descriptor.schema_version), descriptor);
        Ok(())
    }

    pub fn validate_payload_envelope(
        &self,
        payload: VoxelChunkPayload<'_>,
    ) -> Result<(), VoxelUpdateError> {
        let Some(descriptor) = self
            .descriptors
            .get(&(payload.encoding, payload.schema_version))
        else {
            if self
                .descriptors
                .keys()
                .any(|(encoding, _)| *encoding == payload.encoding)
            {
                return Err(VoxelUpdateError::UnsupportedSchema(payload.schema_version));
            }
            return Err(VoxelUpdateError::UnsupportedEncoding(payload.encoding));
        };
        if payload.bytes.len() < descriptor.min_bytes || payload.bytes.len() > descriptor.max_bytes
        {
            return Err(VoxelUpdateError::InvalidPayloadLength(payload.bytes.len()));
        }
        Ok(())
    }

    pub fn validate_payload(&self, payload: VoxelChunkPayload<'_>) -> Result<(), VoxelUpdateError> {
        self.validate_payload_envelope(payload)?;
        let descriptor = &self.descriptors[&(payload.encoding, payload.schema_version)];
        if !(descriptor.validate)(payload.bytes) {
            return Err(VoxelUpdateError::MalformedPayload(payload.encoding));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VoxelChunkUpdate<'a> {
    pub key: VoxelChunkKey,
    pub payload: VoxelChunkPayload<'a>,
}

/// One canonical chunk-map operation. Deletes carry no payload bytes.
#[derive(Clone, Copy, Debug)]
pub enum VoxelChunkOp<'a> {
    Upsert(VoxelChunkUpdate<'a>),
    Delete { key: VoxelChunkKey },
}

/// Optimistic source revision transition. A batch is valid only when it moves
/// exactly one revision forward from the currently committed revision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VoxelBatchRevision {
    pub expected: u64,
    pub publish: u64,
}

/// Borrowed batch validated before handing it to the canonical live-state
/// owner. Validation rejects duplicate chunk keys instead of making ordering
/// determine the result.
#[derive(Clone, Copy, Debug)]
pub struct VoxelChunkBatch<'a> {
    pub terrain: VoxelTerrainId,
    pub source: VoxelSourceId,
    pub revision: VoxelBatchRevision,
    pub domain: VoxelDomain,
    pub ops: &'a [VoxelChunkOp<'a>],
}

impl<'a> VoxelChunkBatch<'a> {
    pub fn validate(&self, committed_revision: u64) -> Result<(), VoxelUpdateError> {
        self.validate_with_formats(committed_revision, &VoxelFormatRegistry::default())
    }

    pub fn validate_with_formats(
        &self,
        committed_revision: u64,
        formats: &VoxelFormatRegistry,
    ) -> Result<(), VoxelUpdateError> {
        self.validate_with_formats_inner(committed_revision, formats, true)
    }

    /// Cheap admission check. Format-specific byte parsing is deferred to the
    /// publication worker, where failures remain observable on its ticket.
    pub fn validate_envelope_with_formats(
        &self,
        committed_revision: u64,
        formats: &VoxelFormatRegistry,
    ) -> Result<(), VoxelUpdateError> {
        self.validate_with_formats_inner(committed_revision, formats, false)
    }

    fn validate_with_formats_inner(
        &self,
        committed_revision: u64,
        formats: &VoxelFormatRegistry,
        deep: bool,
    ) -> Result<(), VoxelUpdateError> {
        if self.revision.expected != committed_revision {
            return Err(VoxelUpdateError::StaleRevision {
                expected: self.revision.expected,
                actual: committed_revision,
            });
        }
        let expected_publish = if self.ops.is_empty() {
            Some(self.revision.expected)
        } else {
            self.revision.expected.checked_add(1)
        };
        if expected_publish != Some(self.revision.publish) {
            return Err(VoxelUpdateError::NonSequentialRevision {
                expected_next: expected_publish,
                publish: self.revision.publish,
            });
        }
        if self.ops.len() > MAX_VOXEL_BATCH_UPDATES {
            return Err(VoxelUpdateError::BatchTooLarge(self.ops.len()));
        }
        let mut keys = HashSet::with_capacity(self.ops.len());
        let mut payload_bytes = 0usize;
        for op in self.ops {
            let key = match op {
                VoxelChunkOp::Upsert(update) => {
                    let payload = update.payload;
                    if deep {
                        formats.validate_payload(payload)?;
                    } else {
                        formats.validate_payload_envelope(payload)?;
                    }
                    payload_bytes = payload_bytes
                        .checked_add(payload.bytes.len())
                        .ok_or(VoxelUpdateError::BatchPayloadTooLarge(usize::MAX))?;
                    update.key
                }
                VoxelChunkOp::Delete { key } => *key,
            };
            self.domain.validate_key(key)?;
            if !keys.insert(key) {
                return Err(VoxelUpdateError::DuplicateChunk(key));
            }
        }
        if payload_bytes > MAX_VOXEL_BATCH_PAYLOAD_BYTES {
            return Err(VoxelUpdateError::BatchPayloadTooLarge(payload_bytes));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VoxelUpdateError {
    WrongTerrain {
        expected: VoxelTerrainId,
        actual: VoxelTerrainId,
    },
    WrongSource {
        expected: VoxelSourceId,
        actual: VoxelSourceId,
    },
    StoreLockPoisoned,
    StoreCapacityExceeded,
    InvalidStoredKey([u64; 4]),
    StaleRevision {
        expected: u64,
        actual: u64,
    },
    NonSequentialRevision {
        expected_next: Option<u64>,
        publish: u64,
    },
    InvalidDomainBounds,
    LodOutOfDomain {
        lod: u8,
        max_lod: u8,
    },
    ChunkOutOfDomain(VoxelChunkKey),
    DuplicateChunk(VoxelChunkKey),
    BatchTooLarge(usize),
    UnsupportedEncoding(VoxelFormatId),
    UnsupportedSchema(u16),
    InvalidFormatDescriptor,
    MalformedPayload(VoxelFormatId),
    PayloadHandleMismatch,
    InvalidPayloadLength(usize),
    BatchPayloadTooLarge(usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn update(key: VoxelChunkKey) -> VoxelChunkOp<'static> {
        VoxelChunkOp::Upsert(VoxelChunkUpdate {
            key,
            payload: VoxelChunkPayload {
                encoding: VOXEL_CHUNK_ENCODING_RAW,
                schema_version: VOXEL_CHUNK_SCHEMA_VERSION,
                bytes: &[1, 2, 3],
            },
        })
    }

    fn batch<'a>(ops: &'a [VoxelChunkOp<'a>]) -> VoxelChunkBatch<'a> {
        VoxelChunkBatch {
            terrain: VoxelTerrainId(7),
            source: VoxelSourceId(11),
            revision: VoxelBatchRevision {
                expected: 4,
                publish: 5,
            },
            domain: VoxelDomain::Bounded {
                min: [-8, -8, -8],
                max: [8, 8, 8],
                max_lod: 4,
            },
            ops,
        }
    }

    #[test]
    fn signed_coordinates_and_lod_are_canonical_key_identity() {
        let a = VoxelChunkKey::new(-1, 2, i64::MIN, 3);
        assert_eq!(
            a,
            VoxelChunkKey {
                x: -1,
                y: 2,
                z: i64::MIN,
                lod: 3
            }
        );
        assert_ne!(a, VoxelChunkKey::new(1, 2, i64::MIN, 3));
        assert_ne!(a, VoxelChunkKey::new(-1, 2, i64::MIN, 4));
    }

    #[test]
    fn batch_accepts_valid_negative_coordinates_and_rejects_stale_revision() {
        let ops = [update(VoxelChunkKey::new(-1, 0, 1, 2))];
        let valid = batch(&ops);
        assert_eq!(valid.validate(4), Ok(()));
        assert_eq!(
            valid.validate(5),
            Err(VoxelUpdateError::StaleRevision {
                expected: 4,
                actual: 5
            })
        );
    }

    #[test]
    fn rejects_duplicate_keys_bad_bounds_lod_encoding_and_empty_payload() {
        let key = VoxelChunkKey::new(1, 2, 3, 0);
        let duplicates = [update(key), update(key)];
        assert_eq!(
            batch(&duplicates).validate(4),
            Err(VoxelUpdateError::DuplicateChunk(key))
        );

        let outside = [update(VoxelChunkKey::new(9, 0, 0, 0))];
        assert_eq!(
            batch(&outside).validate(4),
            Err(VoxelUpdateError::ChunkOutOfDomain(VoxelChunkKey::new(
                9, 0, 0, 0
            )))
        );
        let high_lod = [update(VoxelChunkKey::new(0, 0, 0, 5))];
        assert_eq!(
            batch(&high_lod).validate(4),
            Err(VoxelUpdateError::LodOutOfDomain { lod: 5, max_lod: 4 })
        );

        let empty = [VoxelChunkOp::Upsert(VoxelChunkUpdate {
            key,
            payload: VoxelChunkPayload {
                encoding: VOXEL_CHUNK_ENCODING_RAW,
                schema_version: VOXEL_CHUNK_SCHEMA_VERSION,
                bytes: &[],
            },
        })];
        assert_eq!(
            batch(&empty).validate(4),
            Err(VoxelUpdateError::InvalidPayloadLength(0))
        );
        let wrong_encoding = [VoxelChunkOp::Upsert(VoxelChunkUpdate {
            key,
            payload: VoxelChunkPayload {
                encoding: 99,
                schema_version: VOXEL_CHUNK_SCHEMA_VERSION,
                bytes: &[1],
            },
        })];
        assert_eq!(
            batch(&wrong_encoding).validate(4),
            Err(VoxelUpdateError::UnsupportedEncoding(99))
        );
    }

    #[test]
    fn domain_rejects_inverted_bounds_and_revision_overflow() {
        assert_eq!(
            VoxelDomain::Bounded {
                min: [2, 0, 0],
                max: [1, 1, 1],
                max_lod: 1
            }
            .validate_key(VoxelChunkKey::new(0, 0, 0, 0)),
            Err(VoxelUpdateError::InvalidDomainBounds)
        );
        let overflow: [VoxelChunkOp<'static>; 1] = [update(VoxelChunkKey::new(0, 0, 0, 0))];
        let mut b = batch(&overflow);
        b.revision = VoxelBatchRevision {
            expected: u64::MAX,
            publish: 0,
        };
        assert_eq!(
            b.validate(u64::MAX),
            Err(VoxelUpdateError::NonSequentialRevision {
                expected_next: None,
                publish: 0
            })
        );
    }

    #[test]
    fn base_bounds_follow_signed_lod_scale_without_overflow() {
        let domain = VoxelDomain::BoundedBase {
            min: [-8, 0, 0],
            max: [8, 0, 0],
            max_lod: 127,
            lod_scale: 3,
        };
        assert!(domain.validate_key(VoxelChunkKey::new(-3, 0, 0, 1)).is_ok());
        assert!(domain.validate_key(VoxelChunkKey::new(2, 0, 0, 1)).is_ok());
        assert_eq!(
            domain.validate_key(VoxelChunkKey::new(3, 0, 0, 1)),
            Err(VoxelUpdateError::ChunkOutOfDomain(VoxelChunkKey::new(
                3, 0, 0, 1
            )))
        );
        assert!(domain
            .validate_key(VoxelChunkKey::new(-1, 0, 0, 127))
            .is_ok());
        assert!(domain
            .validate_key(VoxelChunkKey::new(0, 0, 0, 127))
            .is_ok());
        assert!(domain
            .validate_key(VoxelChunkKey::new(1, 0, 0, 127))
            .is_err());
    }

    #[test]
    fn delete_is_revisioned_and_cannot_duplicate_an_upsert_key() {
        let key = VoxelChunkKey::new(-4, 5, 0, 2);
        let ops = [update(key), VoxelChunkOp::Delete { key }];
        assert_eq!(
            batch(&ops).validate(4),
            Err(VoxelUpdateError::DuplicateChunk(key))
        );

        let delete = [VoxelChunkOp::Delete { key }];
        assert_eq!(batch(&delete).validate(4), Ok(()));
    }

    #[test]
    fn empty_batch_does_not_advance_revision() {
        let empty: [VoxelChunkOp<'static>; 0] = [];
        let mut b = batch(&empty);
        b.revision.publish = b.revision.expected;
        assert_eq!(b.validate(4), Ok(()));
        b.revision.publish += 1;
        assert!(matches!(
            b.validate(4),
            Err(VoxelUpdateError::NonSequentialRevision { .. })
        ));
    }

    #[test]
    fn canonical_raw_chunk_rejects_more_than_eight_cubed_samples() {
        let too_large = vec![1u8; VOXEL_CHUNK_SAMPLES + 1];
        let ops = [VoxelChunkOp::Upsert(VoxelChunkUpdate {
            key: VoxelChunkKey::new(0, 0, 0, 0),
            payload: VoxelChunkPayload {
                encoding: VOXEL_CHUNK_ENCODING_RAW,
                schema_version: VOXEL_CHUNK_SCHEMA_VERSION,
                bytes: &too_large,
            },
        })];
        let mut b = batch(&ops);
        b.revision.expected = 0;
        b.revision.publish = 1;
        assert_eq!(
            b.validate(0),
            Err(VoxelUpdateError::InvalidPayloadLength(
                VOXEL_CHUNK_SAMPLES + 1
            ))
        );
    }

    #[test]
    fn registered_format_validates_shape_without_changing_raw_contract() {
        let mut formats = VoxelFormatRegistry::default();
        formats
            .register(VoxelFormatDescriptor {
                encoding: 42,
                schema_version: 3,
                min_bytes: 4,
                max_bytes: 8,
                validate: |bytes| bytes[0] == 0xA5,
            })
            .unwrap();
        let bytes = [0xA5, 1, 2, 3];
        let ops = [VoxelChunkOp::Upsert(VoxelChunkUpdate {
            key: VoxelChunkKey::new(-1, 0, 2, 1),
            payload: VoxelChunkPayload {
                encoding: 42,
                schema_version: 3,
                bytes: &bytes,
            },
        })];
        assert_eq!(
            batch(&ops).validate(4),
            Err(VoxelUpdateError::UnsupportedEncoding(42))
        );
        assert_eq!(batch(&ops).validate_with_formats(4, &formats), Ok(()));
        let malformed = [0, 1, 2, 3];
        let bad = [VoxelChunkOp::Upsert(VoxelChunkUpdate {
            key: VoxelChunkKey::new(-1, 0, 2, 1),
            payload: VoxelChunkPayload {
                encoding: 42,
                schema_version: 3,
                bytes: &malformed,
            },
        })];
        assert_eq!(
            batch(&bad).validate_with_formats(4, &formats),
            Err(VoxelUpdateError::MalformedPayload(42))
        );
    }

    #[test]
    fn envelope_admission_defers_format_parser() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static PARSES: AtomicUsize = AtomicUsize::new(0);
        fn parse(bytes: &[u8]) -> bool {
            PARSES.fetch_add(1, Ordering::SeqCst);
            bytes[0] == 0xA5
        }
        PARSES.store(0, Ordering::SeqCst);
        let mut formats = VoxelFormatRegistry::default();
        formats
            .register(VoxelFormatDescriptor {
                encoding: 42,
                schema_version: 1,
                min_bytes: 1,
                max_bytes: 4,
                validate: parse,
            })
            .unwrap();
        let bytes = [0xA5];
        let ops = [VoxelChunkOp::Upsert(VoxelChunkUpdate {
            key: VoxelChunkKey::new(0, 0, 0, 0),
            payload: VoxelChunkPayload {
                encoding: 42,
                schema_version: 1,
                bytes: &bytes,
            },
        })];
        assert_eq!(
            batch(&ops).validate_envelope_with_formats(4, &formats),
            Ok(())
        );
        assert_eq!(PARSES.load(Ordering::SeqCst), 0);
        assert_eq!(batch(&ops).validate_with_formats(4, &formats), Ok(()));
        assert_eq!(PARSES.load(Ordering::SeqCst), 1);
    }
}
