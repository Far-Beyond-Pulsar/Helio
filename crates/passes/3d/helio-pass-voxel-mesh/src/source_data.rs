//! Transport-neutral contract for externally produced voxel chunks.
//!
//! These are CPU-side request/validation types only. They do not own payload
//! bytes, publish SceneDB state, schedule work, or imply GPU residency. The
//! owning service applies accepted data to live SceneDB component state before
//! notifying the render pass. Persistence/exfiltration is caller-owned.

use std::collections::HashSet;

/// Current version of the canonical raw chunk payload schema.
pub const VOXEL_CHUNK_SCHEMA_VERSION: u16 = 1;
/// Stable encoding tag for a producer-defined packed raw voxel byte stream.
pub const VOXEL_CHUNK_ENCODING_RAW: u16 = 1;
/// Defensive per-update validation ceiling (16 MiB); policy may impose a lower limit.
pub const MAX_VOXEL_CHUNK_PAYLOAD_BYTES: usize = 16 * 1024 * 1024;
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

/// Domain policy for chunk coordinates. Bounds are inclusive chunk coordinates
/// at every supported LOD; conversion between LOD address spaces belongs to the
/// producer, avoiding ambiguous negative-coordinate rounding here.
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

/// Opaque, borrowed payload view. Bytes are carried by the producer/service;
/// this type makes no copy and must not be retained as renderer state.
#[derive(Clone, Copy, Debug)]
pub struct VoxelChunkPayload<'a> {
    pub encoding: u16,
    pub schema_version: u16,
    pub bytes: &'a [u8],
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

/// Borrowed batch validated before handing it to the durable SceneDB-facing
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
                    if payload.encoding != VOXEL_CHUNK_ENCODING_RAW {
                        return Err(VoxelUpdateError::UnsupportedEncoding(payload.encoding));
                    }
                    if payload.schema_version != VOXEL_CHUNK_SCHEMA_VERSION {
                        return Err(VoxelUpdateError::UnsupportedSchema(payload.schema_version));
                    }
                    if payload.bytes.is_empty() || payload.bytes.len() > MAX_VOXEL_CHUNK_PAYLOAD_BYTES {
                        return Err(VoxelUpdateError::InvalidPayloadLength(payload.bytes.len()));
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
    UnsupportedEncoding(u16),
    UnsupportedSchema(u16),
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
            Err(VoxelUpdateError::ChunkOutOfDomain(VoxelChunkKey::new(9, 0, 0, 0)))
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
    fn delete_is_revisioned_and_cannot_duplicate_an_upsert_key() {
        let key = VoxelChunkKey::new(-4, 5, 0, 2);
        let ops = [
            update(key),
            VoxelChunkOp::Delete { key },
        ];
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
}
