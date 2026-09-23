//! Bounded, transient handoff for externally produced voxel chunk batches.
//!
//! The inbox is pending work only: it is neither canonical SceneDB state nor
//! persistence, a GPU cache, or a durability guarantee. Consumers drain whole
//! batches and apply them to the owning SceneDB component through the existing
//! writer API. All operations use `try_lock`; contention is reported as
//! `Busy`, never waited out on a caller (in particular, the render thread).

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex, TryLockError},
};

use crate::{VoxelChunkBatch, VoxelChunkKey, VoxelChunkOp, VoxelSourceId, VoxelTerrainId};

/// Hard bounds for queued work and for any one atomic batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VoxelInboxLimits {
    pub max_pending_batches: usize,
    pub max_pending_payload_bytes: usize,
    pub max_pending_ops: usize,
    pub max_batch_ops: usize,
    pub max_batch_payload_bytes: usize,
}

impl Default for VoxelInboxLimits {
    fn default() -> Self {
        Self {
            max_pending_batches: 64,
            max_pending_payload_bytes: 256 * 1024 * 1024,
            max_pending_ops: 65_536,
            max_batch_ops: 16_384,
            max_batch_payload_bytes: 64 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoxelInboxError {
    Closed,
    Full,
    Busy,
    Invalid(VoxelInboxInvalid),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoxelInboxInvalid {
    ZeroCapacity,
    BatchTooLarge,
    TooManyPayloadHandles,
    PayloadHandleMismatch,
    PayloadByteOverflow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoxelInboxClose {
    /// Stop accepting producers; allow consumers to drain already queued work.
    Drain,
    /// Stop accepting producers and immediately release all queued Arc handles.
    Discard,
}

#[derive(Clone, Debug)]
pub struct VoxelInboxBatch {
    pub terrain: VoxelTerrainId,
    pub source: VoxelSourceId,
    pub revision: crate::VoxelBatchRevision,
    pub domain: crate::VoxelDomain,
    ops: Vec<VoxelInboxOp>,
    payload_bytes: usize,
}

#[derive(Clone, Debug)]
enum VoxelInboxOp {
    Upsert {
        key: VoxelChunkKey,
        bytes: Arc<[u8]>,
    },
    Delete {
        key: VoxelChunkKey,
    },
}

impl VoxelInboxBatch {
    pub fn op_count(&self) -> usize {
        self.ops.len()
    }
    pub fn payload_bytes(&self) -> usize {
        self.payload_bytes
    }

    /// Apply this queued batch to the canonical component store. The caller
    /// keeps ownership of the queued batch on error and may retry or discard
    /// it explicitly. Publication is CPU-side SceneDB state only; GPU upload
    /// is a separate transient pass responsibility.
    pub fn publish_into(
        &self,
        writer: &crate::VoxelSourceWriter,
    ) -> Result<crate::VoxelBatchReceipt, crate::VoxelUpdateError> {
        self.with_borrowed_batch(|batch| writer.publish_batch(batch))
    }

    /// Temporarily present this owned queue item through the existing borrowed
    /// batch contract, e.g. `batch.with_borrowed(|b| writer.publish_batch(b))`.
    pub fn with_borrowed_batch<R>(&self, apply: impl FnOnce(&VoxelChunkBatch<'_>) -> R) -> R {
        let ops: Vec<_> = self
            .ops
            .iter()
            .map(|op| match op {
                VoxelInboxOp::Upsert { key, bytes } => {
                    VoxelChunkOp::Upsert(crate::VoxelChunkUpdate {
                        key: *key,
                        payload: crate::VoxelChunkPayload {
                            encoding: crate::VOXEL_CHUNK_ENCODING_RAW,
                            schema_version: crate::VOXEL_CHUNK_SCHEMA_VERSION,
                            bytes,
                        },
                    })
                }
                VoxelInboxOp::Delete { key } => VoxelChunkOp::Delete { key: *key },
            })
            .collect();
        let batch = VoxelChunkBatch {
            terrain: self.terrain,
            source: self.source,
            revision: self.revision,
            domain: self.domain,
            ops: &ops,
        };
        apply(&batch)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VoxelInboxDrainBudget {
    pub max_ops: usize,
    pub max_payload_bytes: usize,
}

#[derive(Debug, Default)]
pub struct VoxelInboxDrain {
    pub batches: Vec<VoxelInboxBatch>,
    pub ops: usize,
    pub payload_bytes: usize,
    /// True when the FIFO head did not fit; it remains queued intact.
    pub budget_limited: bool,
}

#[derive(Clone)]
pub struct BoundedVoxelInbox {
    inner: Arc<Mutex<InboxState>>,
    limits: VoxelInboxLimits,
}

struct InboxState {
    queue: VecDeque<VoxelInboxBatch>,
    pending_ops: usize,
    pending_bytes: usize,
    closed: bool,
}

impl BoundedVoxelInbox {
    pub fn new(limits: VoxelInboxLimits) -> Result<Self, VoxelInboxError> {
        if limits.max_pending_batches == 0
            || limits.max_pending_ops == 0
            || limits.max_batch_ops == 0
            || limits.max_batch_ops > limits.max_pending_ops
        {
            return Err(VoxelInboxError::Invalid(VoxelInboxInvalid::ZeroCapacity));
        }
        Ok(Self {
            inner: Arc::new(Mutex::new(InboxState {
                queue: VecDeque::new(),
                pending_ops: 0,
                pending_bytes: 0,
                closed: false,
            })),
            limits,
        })
    }

    /// Enqueue a validated batch without copying its payload bytes.
    ///
    /// `payloads` contains one Arc for each Upsert, in operation order. Each
    /// borrowed payload in `batch` must be the exact slice owned by its Arc;
    /// this pointer check makes the zero-copy contract explicit. Arc handles
    /// are cloned only after validation succeeds and before taking the lock.
    pub fn try_submit(
        &self,
        batch: &VoxelChunkBatch<'_>,
        payloads: &[Arc<[u8]>],
    ) -> Result<(), VoxelInboxError> {
        let mut upsert_index = 0;
        let mut payload_bytes = 0usize;
        for op in batch.ops {
            if let VoxelChunkOp::Upsert(update) = op {
                let arc = payloads.get(upsert_index).ok_or(VoxelInboxError::Invalid(
                    VoxelInboxInvalid::TooManyPayloadHandles,
                ))?;
                if !std::ptr::eq(update.payload.bytes.as_ptr(), arc.as_ptr())
                    || update.payload.bytes.len() != arc.len()
                {
                    return Err(VoxelInboxError::Invalid(
                        VoxelInboxInvalid::PayloadHandleMismatch,
                    ));
                }
                payload_bytes =
                    payload_bytes
                        .checked_add(arc.len())
                        .ok_or(VoxelInboxError::Invalid(
                            VoxelInboxInvalid::PayloadByteOverflow,
                        ))?;
                upsert_index += 1;
            }
        }
        if upsert_index != payloads.len() {
            return Err(VoxelInboxError::Invalid(
                VoxelInboxInvalid::TooManyPayloadHandles,
            ));
        }
        if batch.validate(batch.revision.expected).is_err()
            || batch.ops.len() > self.limits.max_batch_ops
            || payload_bytes > self.limits.max_batch_payload_bytes
        {
            return Err(VoxelInboxError::Invalid(VoxelInboxInvalid::BatchTooLarge));
        }

        // Build descriptors and clone only Arc handles before acquiring the
        // queue lock; payload bytes themselves are never copied.
        let mut handles = payloads.iter();
        let mut owned_ops = Vec::with_capacity(batch.ops.len());
        for op in batch.ops {
            match op {
                VoxelChunkOp::Upsert(update) => owned_ops.push(VoxelInboxOp::Upsert {
                    key: update.key,
                    bytes: Arc::clone(handles.next().expect("validated payload handle count")),
                }),
                VoxelChunkOp::Delete { key } => owned_ops.push(VoxelInboxOp::Delete { key: *key }),
            }
        }

        let mut state = self.inner.try_lock().map_err(map_lock_error)?;
        if state.closed {
            return Err(VoxelInboxError::Closed);
        }
        if state.queue.len() >= self.limits.max_pending_batches
            || state
                .pending_ops
                .checked_add(batch.ops.len())
                .map_or(true, |n| n > self.limits.max_pending_ops)
            || state
                .pending_bytes
                .checked_add(payload_bytes)
                .map_or(true, |n| n > self.limits.max_pending_payload_bytes)
        {
            return Err(VoxelInboxError::Full);
        }

        state.pending_ops += batch.ops.len();
        state.pending_bytes += payload_bytes;
        state.queue.push_back(VoxelInboxBatch {
            terrain: batch.terrain,
            source: batch.source,
            revision: batch.revision,
            domain: batch.domain,
            ops: owned_ops,
            payload_bytes,
        });
        Ok(())
    }

    /// Remove whole FIFO batches up to both budgets. A batch that does not fit
    /// remains at the head; no partial batch is ever returned.
    pub fn try_drain(
        &self,
        budget: VoxelInboxDrainBudget,
    ) -> Result<VoxelInboxDrain, VoxelInboxError> {
        let mut state = self.inner.try_lock().map_err(map_lock_error)?;
        let mut result = VoxelInboxDrain::default();
        loop {
            let Some(head) = state.queue.front() else {
                break;
            };
            let fits = result
                .ops
                .checked_add(head.op_count())
                .is_some_and(|n| n <= budget.max_ops)
                && result
                    .payload_bytes
                    .checked_add(head.payload_bytes())
                    .is_some_and(|n| n <= budget.max_payload_bytes);
            if !fits {
                result.budget_limited = true;
                break;
            }
            let batch = state.queue.pop_front().expect("front was present");
            state.pending_ops -= batch.op_count();
            state.pending_bytes -= batch.payload_bytes();
            result.ops += batch.op_count();
            result.payload_bytes += batch.payload_bytes();
            result.batches.push(batch);
        }
        Ok(result)
    }

    /// Close producers. `Drain` preserves queued work; `Discard` cancels it and
    /// releases its payload handles. A contended inbox returns `Busy` unchanged.
    pub fn try_close(&self, mode: VoxelInboxClose) -> Result<usize, VoxelInboxError> {
        let mut state = self.inner.try_lock().map_err(map_lock_error)?;
        state.closed = true;
        if mode == VoxelInboxClose::Drain {
            return Ok(0);
        }
        let discarded = state.queue.len();
        let retired = std::mem::take(&mut state.queue);
        state.pending_ops = 0;
        state.pending_bytes = 0;
        drop(state);
        // Payload releases (and any final backing-allocation deallocation)
        // happen outside the queue lock.
        drop(retired);
        Ok(discarded)
    }

    pub fn try_pending(&self) -> Result<(usize, usize, usize), VoxelInboxError> {
        let state = self.inner.try_lock().map_err(map_lock_error)?;
        Ok((state.queue.len(), state.pending_ops, state.pending_bytes))
    }
}

fn map_lock_error(error: TryLockError<std::sync::MutexGuard<'_, InboxState>>) -> VoxelInboxError {
    match error {
        TryLockError::WouldBlock => VoxelInboxError::Busy,
        TryLockError::Poisoned(_) => VoxelInboxError::Busy,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        VoxelBatchRevision, VoxelChunkPayload, VoxelChunkUpdate, VoxelDomain, VoxelSourceWriter,
        VoxelTerrainSnapshot, VOXEL_CHUNK_ENCODING_RAW, VOXEL_CHUNK_SCHEMA_VERSION,
    };
    use std::collections::HashMap;

    fn inbox() -> BoundedVoxelInbox {
        BoundedVoxelInbox::new(VoxelInboxLimits {
            max_pending_batches: 2,
            max_pending_payload_bytes: 8,
            max_pending_ops: 4,
            max_batch_ops: 2,
            max_batch_payload_bytes: 8,
        })
        .unwrap()
    }

    fn make_batch<'a>(ops: &'a [VoxelChunkOp<'a>], revision: u64) -> VoxelChunkBatch<'a> {
        VoxelChunkBatch {
            terrain: VoxelTerrainId(1),
            source: VoxelSourceId(2),
            revision: VoxelBatchRevision {
                expected: revision,
                publish: revision + 1,
            },
            domain: VoxelDomain::Unbounded { max_lod: 0 },
            ops,
        }
    }

    #[test]
    fn enqueue_is_zero_copy_fifo_and_drain_keeps_atomic_head_on_budget_miss() {
        let q = inbox();
        let a: Arc<[u8]> = Arc::from([1u8, 2]);
        let ops_a = [VoxelChunkOp::Upsert(VoxelChunkUpdate {
            key: VoxelChunkKey::new(0, 0, 0, 0),
            payload: VoxelChunkPayload {
                encoding: VOXEL_CHUNK_ENCODING_RAW,
                schema_version: VOXEL_CHUNK_SCHEMA_VERSION,
                bytes: &a,
            },
        })];
        let batch_a = make_batch(&ops_a, 0);
        q.try_submit(&batch_a, &[Arc::clone(&a)]).unwrap();
        assert_eq!(Arc::strong_count(&a), 2);
        let small = q
            .try_drain(VoxelInboxDrainBudget {
                max_ops: 0,
                max_payload_bytes: 8,
            })
            .unwrap();
        assert!(small.batches.is_empty());
        assert!(small.budget_limited);
        let drained = q
            .try_drain(VoxelInboxDrainBudget {
                max_ops: 1,
                max_payload_bytes: 2,
            })
            .unwrap();
        assert_eq!(drained.batches.len(), 1);
        assert_eq!(drained.batches[0].terrain, VoxelTerrainId(1));
    }

    #[test]
    fn full_close_and_discard_release_pending_work() {
        let q = inbox();
        let a: Arc<[u8]> = Arc::from([3u8]);
        let ops = [VoxelChunkOp::Upsert(VoxelChunkUpdate {
            key: VoxelChunkKey::new(0, 0, 0, 0),
            payload: VoxelChunkPayload {
                encoding: VOXEL_CHUNK_ENCODING_RAW,
                schema_version: VOXEL_CHUNK_SCHEMA_VERSION,
                bytes: &a,
            },
        })];
        let b = make_batch(&ops, 0);
        q.try_submit(&b, &[Arc::clone(&a)]).unwrap();
        q.try_submit(&b, &[Arc::clone(&a)]).unwrap();
        assert_eq!(
            q.try_submit(&b, &[Arc::clone(&a)]),
            Err(VoxelInboxError::Full)
        );
        assert_eq!(q.try_close(VoxelInboxClose::Discard), Ok(2));
        assert_eq!(
            q.try_submit(&b, &[Arc::clone(&a)]),
            Err(VoxelInboxError::Closed)
        );
        assert_eq!(q.try_pending(), Ok((0, 0, 0)));
    }

    #[test]
    fn mismatched_arc_is_invalid_and_does_not_enqueue() {
        let q = inbox();
        let a: Arc<[u8]> = Arc::from([1u8]);
        let other: Arc<[u8]> = Arc::from([1u8]);
        let ops = [VoxelChunkOp::Upsert(VoxelChunkUpdate {
            key: VoxelChunkKey::new(0, 0, 0, 0),
            payload: VoxelChunkPayload {
                encoding: VOXEL_CHUNK_ENCODING_RAW,
                schema_version: VOXEL_CHUNK_SCHEMA_VERSION,
                bytes: &a,
            },
        })];
        let b = make_batch(&ops, 0);
        assert_eq!(
            q.try_submit(&b, &[other]),
            Err(VoxelInboxError::Invalid(
                VoxelInboxInvalid::PayloadHandleMismatch
            ))
        );
        assert_eq!(q.try_pending(), Ok((0, 0, 0)));
    }

    #[test]
    fn contention_is_busy_and_drain_close_preserves_queued_work() {
        let q = inbox();
        let a: Arc<[u8]> = Arc::from([8u8]);
        let ops = [VoxelChunkOp::Upsert(VoxelChunkUpdate {
            key: VoxelChunkKey::new(0, 0, 0, 0),
            payload: VoxelChunkPayload {
                encoding: VOXEL_CHUNK_ENCODING_RAW,
                schema_version: VOXEL_CHUNK_SCHEMA_VERSION,
                bytes: &a,
            },
        })];
        let b = make_batch(&ops, 0);
        {
            let _guard = q.inner.lock().unwrap();
            assert_eq!(
                q.try_submit(&b, &[Arc::clone(&a)]),
                Err(VoxelInboxError::Busy)
            );
            assert_eq!(
                q.try_close(VoxelInboxClose::Drain),
                Err(VoxelInboxError::Busy)
            );
        }
        q.try_submit(&b, &[Arc::clone(&a)]).unwrap();
        assert_eq!(q.try_close(VoxelInboxClose::Drain), Ok(0));
        assert_eq!(
            q.try_submit(&b, &[Arc::clone(&a)]),
            Err(VoxelInboxError::Closed)
        );
        let drained = q
            .try_drain(VoxelInboxDrainBudget {
                max_ops: 1,
                max_payload_bytes: 1,
            })
            .unwrap();
        assert_eq!(drained.batches.len(), 1);
    }

    #[test]
    fn drained_batch_publishes_to_live_store_and_old_snapshot_stays_readable() {
        let q = inbox();
        let terrain = VoxelTerrainId(1);
        let source = VoxelSourceId(2);
        let payload_store = Arc::new(std::sync::RwLock::new((0, HashMap::new())));
        let writer = VoxelSourceWriter::new(terrain, source, payload_store);
        let key = VoxelChunkKey::new(-2, 3, 0, 0);
        let first: Arc<[u8]> = Arc::from([1u8, 2]);
        let ops = [VoxelChunkOp::Upsert(VoxelChunkUpdate {
            key,
            payload: VoxelChunkPayload {
                encoding: VOXEL_CHUNK_ENCODING_RAW,
                schema_version: VOXEL_CHUNK_SCHEMA_VERSION,
                bytes: &first,
            },
        })];
        let batch = VoxelChunkBatch {
            terrain,
            source,
            revision: VoxelBatchRevision {
                expected: 0,
                publish: 1,
            },
            domain: VoxelDomain::Unbounded { max_lod: 0 },
            ops: &ops,
        };
        q.try_submit(&batch, &[Arc::clone(&first)]).unwrap();
        let drained = q
            .try_drain(VoxelInboxDrainBudget {
                max_ops: 1,
                max_payload_bytes: 2,
            })
            .unwrap();
        let queued = drained.batches.into_iter().next().unwrap();
        let receipt = queued.publish_into(&writer).unwrap();
        assert_eq!(receipt.revision, 1);
        let snapshot: VoxelTerrainSnapshot = writer.snapshot().unwrap();
        assert_eq!(snapshot.get(key), Some(&[1, 2][..]));

        let second: Arc<[u8]> = Arc::from([9u8]);
        let replace_ops = [VoxelChunkOp::Upsert(VoxelChunkUpdate {
            key,
            payload: VoxelChunkPayload {
                encoding: VOXEL_CHUNK_ENCODING_RAW,
                schema_version: VOXEL_CHUNK_SCHEMA_VERSION,
                bytes: &second,
            },
        })];
        let replace = VoxelChunkBatch {
            terrain,
            source,
            revision: VoxelBatchRevision {
                expected: 1,
                publish: 2,
            },
            domain: VoxelDomain::Unbounded { max_lod: 0 },
            ops: &replace_ops,
        };
        writer.publish_batch(&replace).unwrap();
        assert_eq!(snapshot.get(key), Some(&[1, 2][..]));
        assert_eq!(writer.snapshot().unwrap().get(key), Some(&[9][..]));
    }
}
