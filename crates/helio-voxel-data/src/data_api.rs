//! Live SceneDB component-data access for voxel producers and script exports.
//!
//! Canonical payload bytes remain in the voxel component's in-memory SceneDB
//! row. This module performs bounded, revision-checked batches and creates
//! immutable snapshots for caller-owned exfiltration. It performs no file I/O
//! and does not provide persistence settings or policy.

use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, RwLock},
};

use crate::{
    VoxelChunkBatch, VoxelChunkKey, VoxelChunkOp, VoxelChunkPayload, VoxelFormatRegistry,
    VoxelSourceId, VoxelTerrainId, VoxelUpdateError, VOXEL_CHUNK_ENCODING_RAW,
    VOXEL_CHUNK_SCHEMA_VERSION,
};

/// Opaque component-store key. The data API owns its chunk-key conversion so
/// component code does not need a second interpretation of signed coordinates.
pub type VoxelPayloadKey = [u64; 4];

/// Immutable live payload with its format identity. Keeping the descriptor
/// beside the bytes prevents snapshots, retries, or imports from silently
/// reinterpreting a non-material format as raw material cells.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VoxelStoredPayload {
    pub encoding: crate::VoxelFormatId,
    pub schema_version: u16,
    pub bytes: Arc<[u8]>,
}

impl VoxelStoredPayload {
    pub fn raw_material(bytes: impl Into<Arc<[u8]>>) -> Self {
        Self {
            encoding: VOXEL_CHUNK_ENCODING_RAW,
            schema_version: VOXEL_CHUNK_SCHEMA_VERSION,
            bytes: bytes.into(),
        }
    }

    pub fn borrowed(&self) -> VoxelChunkPayload<'_> {
        VoxelChunkPayload {
            encoding: self.encoding,
            schema_version: self.schema_version,
            bytes: &self.bytes,
        }
    }
}

impl AsRef<[u8]> for VoxelStoredPayload {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

impl std::ops::Deref for VoxelStoredPayload {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.bytes
    }
}

pub type VoxelPayloadStore = Arc<RwLock<(u64, HashMap<VoxelPayloadKey, VoxelStoredPayload>)>>;

/// Producer capability bound to one terrain's live component payload store.
#[derive(Clone)]
pub struct VoxelSourceWriter {
    terrain: VoxelTerrainId,
    source: VoxelSourceId,
    store: VoxelPayloadStore,
    formats: Arc<VoxelFormatRegistry>,
}

impl VoxelSourceWriter {
    pub(crate) fn terrain_id(&self) -> VoxelTerrainId {
        self.terrain
    }
    pub(crate) fn source_id(&self) -> VoxelSourceId {
        self.source
    }

    /// Copy only requested immutable payload handles under the component lock.
    pub(crate) fn snapshot_keys(
        &self,
        keys: impl IntoIterator<Item = VoxelChunkKey>,
    ) -> Result<VoxelKeySnapshot, VoxelUpdateError> {
        let state = self
            .store
            .read()
            .map_err(|_| VoxelUpdateError::StoreLockPoisoned)?;
        let chunks = keys
            .into_iter()
            .map(|key| (key, state.1.get(&payload_key(key)).cloned()))
            .collect();
        Ok(VoxelKeySnapshot {
            revision: state.0,
            chunks,
        })
    }
    /// Bind a producer to the store obtained from one SceneDB component.
    /// Higher layers should resolve/authorize `terrain` and `source` before
    /// constructing this handle.
    pub fn new(terrain: VoxelTerrainId, source: VoxelSourceId, store: VoxelPayloadStore) -> Self {
        Self::new_with_formats(
            terrain,
            source,
            store,
            Arc::new(VoxelFormatRegistry::default()),
        )
    }

    pub fn new_with_formats(
        terrain: VoxelTerrainId,
        source: VoxelSourceId,
        store: VoxelPayloadStore,
        formats: Arc<VoxelFormatRegistry>,
    ) -> Self {
        Self {
            terrain,
            source,
            store,
            formats,
        }
    }

    pub fn formats(&self) -> &Arc<VoxelFormatRegistry> {
        &self.formats
    }

    /// Current live data revision for this component's chunk map.
    pub fn revision(&self) -> Result<u64, VoxelUpdateError> {
        self.store
            .read()
            .map(|state| state.0)
            .map_err(|_| VoxelUpdateError::StoreLockPoisoned)
    }

    /// Validate, stage, and atomically publish one bounded chunk batch into
    /// the component-owned live map. Payload allocation/copying occurs before
    /// taking the store's write lock. Call on a producer/worker thread when
    /// updates are large; this method itself does not wait for render work.
    pub fn publish_batch(
        &self,
        batch: &VoxelChunkBatch<'_>,
    ) -> Result<VoxelBatchReceipt, VoxelUpdateError> {
        self.publish_batch_inner(batch, None)
    }

    /// Publish an already-owned batch without copying payload bytes. Handles
    /// correspond to Upsert operations in order and must own those exact
    /// borrowed slices. Validation still runs before the SceneDB write lock.
    pub fn publish_shared_batch(
        &self,
        batch: &VoxelChunkBatch<'_>,
        payloads: &[Arc<[u8]>],
    ) -> Result<VoxelBatchReceipt, VoxelUpdateError> {
        self.publish_batch_inner(batch, Some(payloads))
    }

    fn publish_batch_inner(
        &self,
        batch: &VoxelChunkBatch<'_>,
        payloads: Option<&[Arc<[u8]>]>,
    ) -> Result<VoxelBatchReceipt, VoxelUpdateError> {
        if batch.terrain != self.terrain {
            return Err(VoxelUpdateError::WrongTerrain {
                expected: self.terrain,
                actual: batch.terrain,
            });
        }
        if batch.source != self.source {
            return Err(VoxelUpdateError::WrongSource {
                expected: self.source,
                actual: batch.source,
            });
        }

        let observed_revision = self.revision()?;
        batch.validate_with_formats(observed_revision, &self.formats)?;
        if let Some(payloads) = payloads {
            let mut handles = payloads.iter();
            for op in batch.ops {
                if let VoxelChunkOp::Upsert(update) = op {
                    let Some(handle) = handles.next() else {
                        return Err(VoxelUpdateError::PayloadHandleMismatch);
                    };
                    if !std::ptr::eq(handle.as_ptr(), update.payload.bytes.as_ptr())
                        || handle.len() != update.payload.bytes.len()
                    {
                        return Err(VoxelUpdateError::PayloadHandleMismatch);
                    }
                }
            }
            if handles.next().is_some() {
                return Err(VoxelUpdateError::PayloadHandleMismatch);
            }
        }
        if batch.ops.is_empty() {
            return Ok(VoxelBatchReceipt {
                revision: observed_revision,
                upserted: 0,
                deleted: 0,
                missing_deletes: 0,
            });
        }

        // Prepare all owned bytes before entering the component-store write
        // boundary. The operation limit and total-byte limit were validated
        // above, so these allocations are bounded by the accepted batch.
        let mut handles = payloads.map(|payloads| payloads.iter());
        let staged: Vec<_> = batch
            .ops
            .iter()
            .map(|op| match op {
                VoxelChunkOp::Upsert(update) => Some((
                    payload_key(update.key),
                    VoxelStoredPayload {
                        encoding: update.payload.encoding,
                        schema_version: update.payload.schema_version,
                        bytes: match &mut handles {
                            Some(handles) => {
                                Arc::clone(handles.next().expect("validated shared payload count"))
                            }
                            None => Arc::<[u8]>::from(update.payload.bytes),
                        },
                    },
                )),
                VoxelChunkOp::Delete { .. } => None,
            })
            .collect();

        let mut state = self
            .store
            .write()
            .map_err(|_| VoxelUpdateError::StoreLockPoisoned)?;
        if state.0 != batch.revision.expected {
            return Err(VoxelUpdateError::StaleRevision {
                expected: batch.revision.expected,
                actual: state.0,
            });
        }

        let upsert_count = staged.iter().filter(|entry| entry.is_some()).count();
        state
            .1
            .try_reserve(upsert_count)
            .map_err(|_| VoxelUpdateError::StoreCapacityExceeded)?;

        let mut retired = Vec::with_capacity(batch.ops.len());
        let mut upserted = 0;
        let mut deleted = 0;
        let mut missing_deletes = 0;
        for (op, staged_payload) in batch.ops.iter().zip(staged) {
            match (op, staged_payload) {
                (VoxelChunkOp::Upsert(update), Some((key, payload))) => {
                    if let Some(old) = state.1.insert(key, payload) {
                        retired.push(old);
                    }
                    upserted += 1;
                    debug_assert_eq!(key, payload_key(update.key));
                }
                (VoxelChunkOp::Delete { key }, None) => {
                    if let Some(old) = state.1.remove(&payload_key(*key)) {
                        retired.push(old);
                        deleted += 1;
                    } else {
                        // Deleting a missing chunk is deliberately idempotent.
                        missing_deletes += 1;
                    }
                }
                _ => unreachable!("staged operation shape matches validated input"),
            }
        }
        state.0 = batch.revision.publish;
        let revision = state.0;
        drop(state);
        // Destruction of replaced large payloads happens after releasing the
        // store lock so concurrent readers/writers are not held by dealloc.
        drop(retired);

        Ok(VoxelBatchReceipt {
            revision,
            upserted,
            deleted,
            missing_deletes,
        })
    }

    /// Create a stable, caller-owned snapshot. The map index and Arc handles
    /// are copied under one read lock; payload bytes are not copied. The
    /// snapshot pins replaced payloads until it is dropped.
    pub fn snapshot(&self) -> Result<VoxelTerrainSnapshot, VoxelUpdateError> {
        let state = self
            .store
            .read()
            .map_err(|_| VoxelUpdateError::StoreLockPoisoned)?;
        let mut chunks = Vec::with_capacity(state.1.len());
        for (key, payload) in &state.1 {
            chunks.push((decode_payload_key(*key)?, payload.clone()));
        }
        let revision = state.0;
        drop(state);
        chunks.sort_unstable_by_key(|(key, _)| *key);
        Ok(VoxelTerrainSnapshot { revision, chunks })
    }

    /// Replace this component's complete live chunk map from a caller-owned
    /// snapshot. This is an in-memory import/replay primitive only: callers
    /// choose their own serialization, durable storage, and scheduling policy.
    /// The replacement is one validated revision transition and must run on a
    /// worker/script thread for large snapshots.
    pub fn replace_from_snapshot(
        &self,
        snapshot: &VoxelTerrainSnapshot,
        domain: crate::VoxelDomain,
    ) -> Result<VoxelBatchReceipt, VoxelUpdateError> {
        let current = self.snapshot()?;
        let mut ops = Vec::with_capacity(current.len().saturating_add(snapshot.len()));
        for (key, _) in current.iter() {
            if snapshot.get(key).is_none() {
                ops.push(VoxelChunkOp::Delete { key });
            }
        }
        for (key, payload) in snapshot.iter() {
            ops.push(VoxelChunkOp::Upsert(crate::VoxelChunkUpdate {
                key,
                payload,
            }));
        }
        let revision = current.revision();
        let publish = if ops.is_empty() {
            revision
        } else {
            revision
                .checked_add(1)
                .ok_or(VoxelUpdateError::NonSequentialRevision {
                    expected_next: None,
                    publish: revision,
                })?
        };
        let batch = crate::VoxelChunkBatch {
            terrain: self.terrain,
            source: self.source,
            revision: crate::VoxelBatchRevision {
                expected: revision,
                publish,
            },
            domain,
            ops: &ops,
        };
        let payloads: Vec<_> = snapshot
            .chunks
            .iter()
            .map(|(_, payload)| payload.bytes.clone())
            .collect();
        self.publish_shared_batch(&batch, &payloads)
    }
}

pub(crate) struct VoxelKeySnapshot {
    pub revision: u64,
    pub chunks: BTreeMap<VoxelChunkKey, Option<VoxelStoredPayload>>,
}

impl VoxelKeySnapshot {
    pub fn get(&self, key: &VoxelChunkKey) -> Option<Option<&VoxelStoredPayload>> {
        self.chunks.get(key).map(Option::as_ref)
    }
}

/// Summary of one live SceneDB publication.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VoxelBatchReceipt {
    pub revision: u64,
    pub upserted: usize,
    pub deleted: usize,
    /// Deletes of absent chunks are idempotent and counted here.
    pub missing_deletes: usize,
}

/// Immutable chunk snapshot suitable for script-owned export or persistence.
#[derive(Clone, Debug)]
pub struct VoxelTerrainSnapshot {
    revision: u64,
    chunks: Vec<(VoxelChunkKey, VoxelStoredPayload)>,
}

impl VoxelTerrainSnapshot {
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Read one payload with its format identity. Consumers must inspect the
    /// encoding/schema before decoding its bytes.
    pub fn get(&self, key: VoxelChunkKey) -> Option<VoxelChunkPayload<'_>> {
        self.get_payload(key).map(VoxelStoredPayload::borrowed)
    }

    pub fn get_payload(&self, key: VoxelChunkKey) -> Option<&VoxelStoredPayload> {
        self.chunks
            .binary_search_by_key(&key, |(candidate, _)| *candidate)
            .ok()
            .map(|index| &self.chunks[index].1)
    }

    pub fn iter(&self) -> impl Iterator<Item = (VoxelChunkKey, VoxelChunkPayload<'_>)> {
        self.chunks
            .iter()
            .map(|(key, payload)| (*key, payload.borrowed()))
    }

    /// Convenience only for the built-in 8³ material representation.
    pub fn get_raw_material(&self, key: VoxelChunkKey) -> Option<&[u8]> {
        self.get_payload(key).and_then(|payload| {
            (payload.encoding == VOXEL_CHUNK_ENCODING_RAW
                && payload.schema_version == VOXEL_CHUNK_SCHEMA_VERSION)
                .then_some(payload.as_ref())
        })
    }
}

fn payload_key(key: VoxelChunkKey) -> VoxelPayloadKey {
    [key.x as u64, key.y as u64, key.z as u64, u64::from(key.lod)]
}

fn decode_payload_key(key: VoxelPayloadKey) -> Result<VoxelChunkKey, VoxelUpdateError> {
    let lod = u8::try_from(key[3]).map_err(|_| VoxelUpdateError::InvalidStoredKey(key))?;
    Ok(VoxelChunkKey::new(
        key[0] as i64,
        key[1] as i64,
        key[2] as i64,
        lod,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        VoxelBatchRevision, VoxelChunkPayload, VoxelChunkUpdate, VoxelDomain,
        VoxelFormatDescriptor, VOXEL_CHUNK_ENCODING_RAW, VOXEL_CHUNK_SCHEMA_VERSION,
    };

    fn update(key: VoxelChunkKey, bytes: &'static [u8]) -> VoxelChunkOp<'static> {
        VoxelChunkOp::Upsert(VoxelChunkUpdate {
            key,
            payload: VoxelChunkPayload {
                encoding: VOXEL_CHUNK_ENCODING_RAW,
                schema_version: VOXEL_CHUNK_SCHEMA_VERSION,
                bytes,
            },
        })
    }

    fn batch<'a>(
        terrain: VoxelTerrainId,
        source: VoxelSourceId,
        revision: VoxelBatchRevision,
        ops: &'a [VoxelChunkOp<'a>],
    ) -> VoxelChunkBatch<'a> {
        VoxelChunkBatch {
            terrain,
            source,
            revision,
            domain: VoxelDomain::Unbounded { max_lod: 8 },
            ops,
        }
    }

    #[test]
    fn component_writer_publishes_batches_and_snapshot_exposes_live_bytes() {
        let store = Arc::new(RwLock::new((0, HashMap::new())));
        let terrain = VoxelTerrainId(100);
        let source = VoxelSourceId(200);
        let writer = VoxelSourceWriter::new(terrain, source, store);
        let key_a = VoxelChunkKey::new(-1, 0, 2, 0);
        let key_b = VoxelChunkKey::new(5, -7, 1, 3);
        let ops = [update(key_a, &[1, 2, 3]), update(key_b, &[4, 5])];
        let batch = batch(
            terrain,
            source,
            VoxelBatchRevision {
                expected: 0,
                publish: 1,
            },
            &ops,
        );

        let receipt = writer.publish_batch(&batch).unwrap();
        assert_eq!(receipt.revision, 1);
        assert_eq!(receipt.upserted, 2);
        let snapshot = writer.snapshot().unwrap();
        assert_eq!(snapshot.revision(), 1);
        assert_eq!(snapshot.len(), 2);
        assert_eq!(snapshot.get_raw_material(key_a), Some(&[1, 2, 3][..]));
        assert_eq!(snapshot.get_raw_material(key_b), Some(&[4, 5][..]));
    }

    #[test]
    fn snapshot_and_import_preserve_registered_payload_format() {
        let mut formats = VoxelFormatRegistry::default();
        formats
            .register(VoxelFormatDescriptor {
                encoding: 42,
                schema_version: 2,
                min_bytes: 4,
                max_bytes: 4,
                validate: |bytes| bytes[0] == 0xA5,
            })
            .unwrap();
        let formats = Arc::new(formats);
        let source_store = Arc::new(RwLock::new((0, HashMap::new())));
        let writer = VoxelSourceWriter::new_with_formats(
            VoxelTerrainId(1),
            VoxelSourceId(2),
            source_store,
            formats.clone(),
        );
        let key = VoxelChunkKey::new(-7, 3, 11, 4);
        let bytes = [0xA5, 10, 20, 30];
        let ops = [VoxelChunkOp::Upsert(VoxelChunkUpdate {
            key,
            payload: VoxelChunkPayload {
                encoding: 42,
                schema_version: 2,
                bytes: &bytes,
            },
        })];
        writer
            .publish_batch(&batch(
                VoxelTerrainId(1),
                VoxelSourceId(2),
                VoxelBatchRevision {
                    expected: 0,
                    publish: 1,
                },
                &ops,
            ))
            .unwrap();
        let snapshot = writer.snapshot().unwrap();
        let payload = snapshot.get_payload(key).unwrap();
        assert_eq!((payload.encoding, payload.schema_version), (42, 2));
        assert_eq!(payload.as_ref(), &bytes);
        assert_eq!(snapshot.get_raw_material(key), None);

        let target = VoxelSourceWriter::new_with_formats(
            VoxelTerrainId(3),
            VoxelSourceId(4),
            Arc::new(RwLock::new((0, HashMap::new()))),
            formats,
        );
        target
            .replace_from_snapshot(&snapshot, VoxelDomain::Unbounded { max_lod: 8 })
            .unwrap();
        let imported = target.snapshot().unwrap();
        assert_eq!(imported.get_payload(key), snapshot.get_payload(key));
        assert!(Arc::ptr_eq(
            &imported.get_payload(key).unwrap().bytes,
            &snapshot.get_payload(key).unwrap().bytes,
        ));
    }

    #[test]
    fn shared_publication_rejects_a_different_payload_handle() {
        let writer = VoxelSourceWriter::new(
            VoxelTerrainId(1),
            VoxelSourceId(2),
            Arc::new(RwLock::new((0, HashMap::new()))),
        );
        let bytes: Arc<[u8]> = Arc::from([1u8]);
        let other: Arc<[u8]> = Arc::from([1u8]);
        let ops = [VoxelChunkOp::Upsert(VoxelChunkUpdate {
            key: VoxelChunkKey::new(0, 0, 0, 0),
            payload: VoxelChunkPayload {
                encoding: VOXEL_CHUNK_ENCODING_RAW,
                schema_version: VOXEL_CHUNK_SCHEMA_VERSION,
                bytes: &bytes,
            },
        })];
        assert_eq!(
            writer.publish_shared_batch(
                &batch(
                    VoxelTerrainId(1),
                    VoxelSourceId(2),
                    VoxelBatchRevision {
                        expected: 0,
                        publish: 1,
                    },
                    &ops,
                ),
                &[other],
            ),
            Err(VoxelUpdateError::PayloadHandleMismatch)
        );
        assert_eq!(writer.revision().unwrap(), 0);
    }

    #[test]
    fn stale_and_wrong_identity_batches_do_not_mutate_component_data() {
        let store = Arc::new(RwLock::new((0, HashMap::new())));
        let terrain = VoxelTerrainId(100);
        let source = VoxelSourceId(200);
        let writer = VoxelSourceWriter::new(terrain, source, store);
        let stale_ops = [update(VoxelChunkKey::new(0, 0, 0, 0), &[9])];
        let stale = batch(
            terrain,
            source,
            VoxelBatchRevision {
                expected: 1,
                publish: 2,
            },
            &stale_ops,
        );
        assert_eq!(
            writer.publish_batch(&stale),
            Err(VoxelUpdateError::StaleRevision {
                expected: 1,
                actual: 0
            })
        );

        let wrong_terrain = batch(
            VoxelTerrainId(101),
            source,
            VoxelBatchRevision {
                expected: 0,
                publish: 1,
            },
            &stale_ops,
        );
        assert!(matches!(
            writer.publish_batch(&wrong_terrain),
            Err(VoxelUpdateError::WrongTerrain { .. })
        ));
        assert_eq!(writer.revision().unwrap(), 0);
        assert!(writer.snapshot().unwrap().is_empty());
    }

    #[test]
    fn deletes_are_idempotent_and_commit_once_with_the_batch() {
        let store = Arc::new(RwLock::new((0, HashMap::new())));
        let terrain = VoxelTerrainId(10);
        let source = VoxelSourceId(20);
        let writer = VoxelSourceWriter::new(terrain, source, store);
        let key = VoxelChunkKey::new(1, 2, 3, 0);
        let ops = [
            update(key, &[7]),
            VoxelChunkOp::Delete { key },
            VoxelChunkOp::Delete { key },
        ];
        // Duplicate operations on one key are rejected by batch validation.
        assert!(matches!(
            writer.publish_batch(&batch(
                terrain,
                source,
                VoxelBatchRevision {
                    expected: 0,
                    publish: 1
                },
                &ops,
            )),
            Err(VoxelUpdateError::DuplicateChunk(_))
        ));

        let initial = [update(key, &[7])];
        writer
            .publish_batch(&batch(
                terrain,
                source,
                VoxelBatchRevision {
                    expected: 0,
                    publish: 1,
                },
                &initial,
            ))
            .unwrap();
        let delete = [VoxelChunkOp::Delete { key }];
        let deleted = writer
            .publish_batch(&batch(
                terrain,
                source,
                VoxelBatchRevision {
                    expected: 1,
                    publish: 2,
                },
                &delete,
            ))
            .unwrap();
        assert_eq!(deleted.deleted, 1);
        let missing = writer
            .publish_batch(&batch(
                terrain,
                source,
                VoxelBatchRevision {
                    expected: 2,
                    publish: 3,
                },
                &delete,
            ))
            .unwrap();
        assert_eq!(missing.missing_deletes, 1);
        assert_eq!(writer.snapshot().unwrap().revision(), 3);
        assert!(writer.snapshot().unwrap().is_empty());
    }
}
