//! Bounded, transient handoff for externally produced voxel chunk batches.
//!
//! The inbox is pending work only: it is neither canonical SceneDB state nor
//! persistence, a GPU cache, or a durability guarantee. Consumers drain whole
//! batches and apply them to the owning SceneDB component through the existing
//! writer API. All operations use `try_lock`; contention is reported as
//! `Busy`, never waited out on a caller (in particular, the render thread).

use std::{
    collections::VecDeque,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Condvar, Mutex, TryLockError,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crate::{VoxelChunkBatch, VoxelChunkKey, VoxelChunkOp, VoxelSourceId, VoxelTerrainId};

/// Hard bounds for queued work and for any one atomic batch. A publication
/// worker may hold one additional in-flight batch after removing it from the
/// queue; total transient admission is at most pending limits plus one batch.
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
    ticket: Option<VoxelPublicationTicket>,
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

    fn detached_clone(&self) -> Self {
        let mut batch = self.clone();
        batch.ticket = None;
        batch
    }

    /// Apply this queued batch to the canonical component store. The caller
    /// keeps ownership of the queued batch on error and may retry or discard
    /// it explicitly. Publication is CPU-side SceneDB state only; GPU upload
    /// is a separate responsibility of the consuming backend.
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
    /// True once producers have been closed, including while draining work.
    pub closed: bool,
}

#[derive(Clone)]
pub struct BoundedVoxelInbox {
    inner: Arc<InboxShared>,
    limits: VoxelInboxLimits,
}

struct InboxShared {
    state: Mutex<InboxState>,
    wake: Condvar,
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
            inner: Arc::new(InboxShared {
                state: Mutex::new(InboxState {
                    queue: VecDeque::new(),
                    pending_ops: 0,
                    pending_bytes: 0,
                    closed: false,
                }),
                wake: Condvar::new(),
            }),
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
        self.submit_with_ticket(batch, payloads, None)
    }

    fn submit_with_ticket(
        &self,
        batch: &VoxelChunkBatch<'_>,
        payloads: &[Arc<[u8]>],
        ticket: Option<VoxelPublicationTicket>,
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

        let mut state = self.inner.state.try_lock().map_err(map_lock_error)?;
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
            ticket,
        });
        // The waiter checks this same mutex before sleeping, so a producer
        // cannot notify between its empty check and the Condvar wait.
        self.inner.wake.notify_one();
        Ok(())
    }

    /// Remove whole FIFO batches up to both budgets. A batch that does not fit
    /// remains at the head; no partial batch is ever returned.
    pub fn try_drain(
        &self,
        budget: VoxelInboxDrainBudget,
    ) -> Result<VoxelInboxDrain, VoxelInboxError> {
        let mut state = self.inner.state.try_lock().map_err(map_lock_error)?;
        Ok(drain_locked(&mut state, budget))
    }

    /// Wait for work or closure without polling. This is a consumer-thread
    /// operation; producers should use only the nonblocking try methods.
    pub fn wait_drain(
        &self,
        budget: VoxelInboxDrainBudget,
    ) -> Result<VoxelInboxDrain, VoxelInboxError> {
        let mut state = self.inner.state.lock().map_err(|_| VoxelInboxError::Busy)?;
        while state.queue.is_empty() && !state.closed {
            state = self
                .inner
                .wake
                .wait(state)
                .map_err(|_| VoxelInboxError::Busy)?;
        }
        Ok(drain_locked(&mut state, budget))
    }

    fn wait_next(&self) -> Result<Option<VoxelInboxBatch>, VoxelInboxError> {
        let mut state = self.inner.state.lock().map_err(|_| VoxelInboxError::Busy)?;
        while state.queue.is_empty() && !state.closed {
            state = self
                .inner
                .wake
                .wait(state)
                .map_err(|_| VoxelInboxError::Busy)?;
        }
        let next = state.queue.pop_front();
        if let Some(batch) = &next {
            state.pending_ops -= batch.op_count();
            state.pending_bytes -= batch.payload_bytes();
        }
        Ok(next)
    }

    fn take_remaining(&self) -> Vec<VoxelInboxBatch> {
        let mut state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
        state.pending_ops = 0;
        state.pending_bytes = 0;
        let remaining: Vec<_> = std::mem::take(&mut state.queue).into_iter().collect();
        drop(state);
        for batch in &remaining {
            if let Some(ticket) = &batch.ticket {
                ticket.set_if_pending(VoxelPublicationTicketState::Unprocessed(
                    batch.detached_clone(),
                ));
            }
        }
        remaining
    }

    /// Close producers, waiting only for the short inbox metadata lock. This
    /// must run off the frame thread if the caller also waits for the worker.
    pub fn close(&self, mode: VoxelInboxClose) -> usize {
        let mut state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
        let (discarded, retired) = close_locked(&mut state, mode, &self.inner.wake);
        drop(state);
        mark_discarded(&retired);
        drop(retired);
        discarded
    }

    /// Close producers. `Drain` preserves queued work; `Discard` cancels it and
    /// releases its payload handles. A contended inbox returns `Busy` unchanged.
    pub fn try_close(&self, mode: VoxelInboxClose) -> Result<usize, VoxelInboxError> {
        let mut state = self.inner.state.try_lock().map_err(map_lock_error)?;
        let (discarded, retired) = close_locked(&mut state, mode, &self.inner.wake);
        drop(state);
        mark_discarded(&retired);
        drop(retired);
        Ok(discarded)
    }

    pub fn try_pending(&self) -> Result<(usize, usize, usize), VoxelInboxError> {
        let state = self.inner.state.try_lock().map_err(map_lock_error)?;
        Ok((state.queue.len(), state.pending_ops, state.pending_bytes))
    }
}

fn drain_locked(state: &mut InboxState, budget: VoxelInboxDrainBudget) -> VoxelInboxDrain {
    let mut result = VoxelInboxDrain::default();
    result.closed = state.closed;
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
    result
}

fn close_locked(
    state: &mut InboxState,
    mode: VoxelInboxClose,
    wake: &Condvar,
) -> (usize, VecDeque<VoxelInboxBatch>) {
    state.closed = true;
    let retired = if mode == VoxelInboxClose::Discard {
        let retired = std::mem::take(&mut state.queue);
        state.pending_ops = 0;
        state.pending_bytes = 0;
        retired
    } else {
        VecDeque::new()
    };
    wake.notify_all();
    (retired.len(), retired)
}

fn mark_discarded(retired: &VecDeque<VoxelInboxBatch>) {
    for batch in retired {
        if let Some(ticket) = &batch.ticket {
            ticket.set(VoxelPublicationTicketState::Discarded);
        }
    }
}

/// A producer-owned receipt for one accepted batch. The ticket remains valid
/// after the worker handle is dropped. Failed and unprocessed states carry a
/// batch handle so the producer can inspect, retry, or rebase it.
#[derive(Clone)]
pub struct VoxelPublicationTicket {
    inner: Arc<TicketShared>,
}

struct TicketShared {
    state: Mutex<VoxelPublicationTicketState>,
    wake: Condvar,
}

impl std::fmt::Debug for VoxelPublicationTicket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("VoxelPublicationTicket")
    }
}

#[derive(Clone, Debug)]
pub enum VoxelPublicationTicketState {
    Pending,
    Published(crate::VoxelBatchReceipt),
    Failed(VoxelPublicationFailure),
    Unprocessed(VoxelInboxBatch),
    Discarded,
}

impl VoxelPublicationTicket {
    fn new() -> Self {
        Self {
            inner: Arc::new(TicketShared {
                state: Mutex::new(VoxelPublicationTicketState::Pending),
                wake: Condvar::new(),
            }),
        }
    }

    fn set(&self, state: VoxelPublicationTicketState) {
        *self.inner.state.lock().unwrap_or_else(|e| e.into_inner()) = state;
        self.inner.wake.notify_all();
    }

    fn set_if_pending(&self, next: VoxelPublicationTicketState) {
        let mut state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
        if matches!(*state, VoxelPublicationTicketState::Pending) {
            *state = next;
            self.inner.wake.notify_all();
        }
    }

    pub fn try_state(&self) -> Result<VoxelPublicationTicketState, VoxelInboxError> {
        self.inner
            .state
            .try_lock()
            .map(|state| state.clone())
            .map_err(|_| VoxelInboxError::Busy)
    }

    /// Wait for this batch's result on a producer/management thread. A later
    /// batch held behind a failed one resolves when the owner calls finish or
    /// drops the worker.
    pub fn wait(&self) -> VoxelPublicationTicketState {
        let mut state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
        while matches!(*state, VoxelPublicationTicketState::Pending) {
            state = self
                .inner
                .wake
                .wait(state)
                .unwrap_or_else(|e| e.into_inner());
        }
        state.clone()
    }
}

/// One CPU-only publisher bound to one live SceneDB component store. The
/// component owner must finish this worker when that component is removed or
/// replaced. This type does not resolve SceneDB entity lifetime on its own.
pub struct VoxelPublicationWorker {
    inbox: BoundedVoxelInbox,
    state: Arc<Mutex<PublicationState>>,
    retried_batches: AtomicU64,
    thread: Option<JoinHandle<()>>,
}

#[derive(Default)]
struct PublicationState {
    published_batches: u64,
    failed_batches: u64,
    last_receipt: Option<crate::VoxelBatchReceipt>,
    last_publication_time: Option<Duration>,
    in_flight_ops: usize,
    in_flight_payload_bytes: usize,
    failure: Option<VoxelPublicationFailure>,
    terminal_inbox_error: Option<VoxelInboxError>,
}

#[derive(Clone, Debug)]
pub enum VoxelPublicationFailureReason {
    Writer(crate::VoxelUpdateError),
    WriterPanicked,
}

/// The failed batch remains owned by the caller for inspection, retry, or
/// rebase. The worker stops on the first failure, retaining later queued work.
#[derive(Clone, Debug)]
pub struct VoxelPublicationFailure {
    pub batch: VoxelInboxBatch,
    pub reason: VoxelPublicationFailureReason,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VoxelPublicationStatus {
    pub published_batches: u64,
    pub failed_batches: u64,
    pub retried_batches: u64,
    pub last_receipt: Option<crate::VoxelBatchReceipt>,
    pub last_publication_time: Option<Duration>,
    pub failed: bool,
    pub pending_batches: usize,
    pub pending_ops: usize,
    pub pending_payload_bytes: usize,
    pub in_flight_ops: usize,
    pub in_flight_payload_bytes: usize,
}

#[derive(Debug)]
pub struct VoxelPublicationOutcome {
    pub published_batches: u64,
    pub failed_batches: u64,
    pub retried_batches: u64,
    pub last_receipt: Option<crate::VoxelBatchReceipt>,
    pub last_publication_time: Option<Duration>,
    pub failure: Option<VoxelPublicationFailure>,
    /// Work not processed after an earlier failure or worker exit.
    pub unprocessed: Vec<VoxelInboxBatch>,
    /// Queued work explicitly cancelled by `finish(Discard)`.
    pub discarded_batches: usize,
    pub terminal_inbox_error: Option<VoxelInboxError>,
    pub worker_panicked: bool,
}

#[derive(Debug)]
pub enum VoxelPublicationStartError {
    Inbox(VoxelInboxError),
    Spawn(std::io::Error),
}

impl VoxelPublicationWorker {
    pub fn start(
        writer: crate::VoxelSourceWriter,
        limits: VoxelInboxLimits,
    ) -> Result<Self, VoxelPublicationStartError> {
        let inbox = BoundedVoxelInbox::new(limits).map_err(VoxelPublicationStartError::Inbox)?;
        let state = Arc::new(Mutex::new(PublicationState::default()));
        let worker_inbox = inbox.clone();
        let worker_state = Arc::clone(&state);
        let thread = thread::Builder::new()
            .name("voxel-scenedb-publisher".into())
            .spawn(move || {
                loop {
                    let batch = match worker_inbox.wait_next() {
                        Ok(Some(batch)) => batch,
                        Ok(None) => break,
                        Err(error) => {
                            worker_inbox.close(VoxelInboxClose::Drain);
                            worker_state
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .terminal_inbox_error = Some(error);
                            break;
                        }
                    };
                    {
                        let mut state = worker_state.lock().unwrap_or_else(|e| e.into_inner());
                        state.in_flight_ops = batch.op_count();
                        state.in_flight_payload_bytes = batch.payload_bytes();
                    }
                    let started = Instant::now();
                    let result = catch_unwind(AssertUnwindSafe(|| batch.publish_into(&writer)));
                    let elapsed = started.elapsed();
                    match result {
                        Ok(Ok(receipt)) => {
                            if let Some(ticket) = &batch.ticket {
                                ticket.set(VoxelPublicationTicketState::Published(receipt));
                            }
                            let mut state = worker_state.lock().unwrap_or_else(|e| e.into_inner());
                            state.published_batches = state.published_batches.saturating_add(1);
                            state.last_receipt = Some(receipt);
                            state.last_publication_time = Some(elapsed);
                            state.in_flight_ops = 0;
                            state.in_flight_payload_bytes = 0;
                        }
                        result => {
                            // Stop admission before publishing the error so
                            // dependent revisions cannot run past it.
                            worker_inbox.close(VoxelInboxClose::Drain);
                            let reason = match result {
                                Ok(Err(error)) => VoxelPublicationFailureReason::Writer(error),
                                Err(_) => VoxelPublicationFailureReason::WriterPanicked,
                                Ok(Ok(_)) => unreachable!(),
                            };
                            if let Some(ticket) = &batch.ticket {
                                ticket.set(VoxelPublicationTicketState::Failed(
                                    VoxelPublicationFailure {
                                        batch: batch.detached_clone(),
                                        reason: reason.clone(),
                                    },
                                ));
                            }
                            let mut state = worker_state.lock().unwrap_or_else(|e| e.into_inner());
                            state.failed_batches = state.failed_batches.saturating_add(1);
                            state.last_publication_time = Some(elapsed);
                            state.in_flight_ops = 0;
                            state.in_flight_payload_bytes = 0;
                            state.failure = Some(VoxelPublicationFailure { batch, reason });
                            break;
                        }
                    }
                }
            })
            .map_err(VoxelPublicationStartError::Spawn)?;
        Ok(Self {
            inbox,
            state,
            retried_batches: AtomicU64::new(0),
            thread: Some(thread),
        })
    }

    /// Admission never waits on the component store. Validation and Arc
    /// descriptor construction happen on the submitting thread. The returned
    /// ticket preserves this batch's fate across worker shutdown or Drop.
    pub fn try_submit(
        &self,
        batch: &VoxelChunkBatch<'_>,
        payloads: &[Arc<[u8]>],
    ) -> Result<VoxelPublicationTicket, VoxelInboxError> {
        let ticket = VoxelPublicationTicket::new();
        self.inbox
            .submit_with_ticket(batch, payloads, Some(ticket.clone()))?;
        Ok(ticket)
    }

    /// Resubmit a retained failed/unprocessed batch after the caller has
    /// reviewed and, if needed, rebased its public revision field. A failed
    /// publisher is terminal; use a new worker bound to the same live store.
    pub fn try_retry(
        &self,
        batch: &VoxelInboxBatch,
    ) -> Result<VoxelPublicationTicket, VoxelInboxError> {
        let payloads: Vec<_> = batch
            .ops
            .iter()
            .filter_map(|op| match op {
                VoxelInboxOp::Upsert { bytes, .. } => Some(Arc::clone(bytes)),
                VoxelInboxOp::Delete { .. } => None,
            })
            .collect();
        let ticket = batch.with_borrowed_batch(|borrowed| self.try_submit(borrowed, &payloads))?;
        self.retried_batches.fetch_add(1, Ordering::Relaxed);
        Ok(ticket)
    }

    pub fn try_status(&self) -> Result<VoxelPublicationStatus, VoxelInboxError> {
        let (pending_batches, pending_ops, pending_payload_bytes) = self.inbox.try_pending()?;
        let state = self.state.try_lock().map_err(|_| VoxelInboxError::Busy)?;
        Ok(VoxelPublicationStatus {
            published_batches: state.published_batches,
            failed_batches: state.failed_batches,
            retried_batches: self.retried_batches.load(Ordering::Relaxed),
            last_receipt: state.last_receipt,
            last_publication_time: state.last_publication_time,
            failed: state.failure.is_some() || state.terminal_inbox_error.is_some(),
            pending_batches,
            pending_ops,
            pending_payload_bytes,
            in_flight_ops: state.in_flight_ops,
            in_flight_payload_bytes: state.in_flight_payload_bytes,
        })
    }

    /// Close admission, wait for the CPU worker, and return every unresolved
    /// batch. Drain publishes until empty or the first error. Discard releases
    /// queued batches immediately; a batch already in publication completes.
    /// Call this on a management/worker thread, never in a frame callback.
    pub fn finish(mut self, mode: VoxelInboxClose) -> VoxelPublicationOutcome {
        let discarded_batches = self.inbox.close(mode);
        let worker_panicked = self
            .thread
            .take()
            .is_some_and(|thread| thread.join().is_err());
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        VoxelPublicationOutcome {
            published_batches: state.published_batches,
            failed_batches: state.failed_batches,
            retried_batches: self.retried_batches.load(Ordering::Relaxed),
            last_receipt: state.last_receipt,
            last_publication_time: state.last_publication_time,
            failure: state.failure.take(),
            unprocessed: self.inbox.take_remaining(),
            discarded_batches,
            terminal_inbox_error: state.terminal_inbox_error.take(),
            worker_panicked,
        }
    }
}

impl Drop for VoxelPublicationWorker {
    fn drop(&mut self) {
        if self.thread.is_some() {
            // Drop discards queued work and marks every ticket. A possible
            // in-flight CPU write still completes and updates its ticket.
            self.inbox.close(VoxelInboxClose::Discard);
            // Dropping JoinHandle detaches a possible in-flight CPU write.
        }
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
        VoxelTerrainSnapshot, VoxelUpdateError, VOXEL_CHUNK_ENCODING_RAW,
        VOXEL_CHUNK_SCHEMA_VERSION,
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
            let _guard = q.inner.state.lock().unwrap();
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

    #[test]
    fn waiter_wakes_for_submission_and_both_close_modes() {
        use std::{sync::mpsc, time::Duration};
        let q = inbox();
        let (tx, rx) = mpsc::channel();
        let waiter = q.clone();
        let thread = std::thread::spawn(move || {
            tx.send(waiter.wait_drain(VoxelInboxDrainBudget {
                max_ops: 1,
                max_payload_bytes: 8,
            }))
            .unwrap();
        });
        let payload: Arc<[u8]> = Arc::from([7u8]);
        let ops = [VoxelChunkOp::Upsert(VoxelChunkUpdate {
            key: VoxelChunkKey::new(0, 0, 0, 0),
            payload: VoxelChunkPayload {
                encoding: VOXEL_CHUNK_ENCODING_RAW,
                schema_version: VOXEL_CHUNK_SCHEMA_VERSION,
                bytes: &payload,
            },
        })];
        q.try_submit(&make_batch(&ops, 0), &[Arc::clone(&payload)])
            .unwrap();
        let drained = rx.recv_timeout(Duration::from_secs(2)).unwrap().unwrap();
        assert_eq!(drained.batches.len(), 1);
        assert!(!drained.closed);
        thread.join().unwrap();

        for mode in [VoxelInboxClose::Drain, VoxelInboxClose::Discard] {
            let q = inbox();
            let (tx, rx) = mpsc::channel();
            let waiter = q.clone();
            let thread = std::thread::spawn(move || {
                tx.send(waiter.wait_drain(VoxelInboxDrainBudget {
                    max_ops: 1,
                    max_payload_bytes: 8,
                }))
                .unwrap();
            });
            assert_eq!(q.close(mode), 0);
            let drained = rx.recv_timeout(Duration::from_secs(2)).unwrap().unwrap();
            assert!(drained.closed);
            assert!(drained.batches.is_empty());
            thread.join().unwrap();
        }
    }

    #[test]
    fn concurrent_producers_cannot_exceed_inbox_limits() {
        let q = inbox();
        let results = std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for index in 0..8 {
                let producer = q.clone();
                handles.push(scope.spawn(move || {
                    let payload: Arc<[u8]> = Arc::from([index as u8]);
                    let ops = [VoxelChunkOp::Upsert(VoxelChunkUpdate {
                        key: VoxelChunkKey::new(index, 0, 0, 0),
                        payload: VoxelChunkPayload {
                            encoding: VOXEL_CHUNK_ENCODING_RAW,
                            schema_version: VOXEL_CHUNK_SCHEMA_VERSION,
                            bytes: &payload,
                        },
                    })];
                    producer.try_submit(&make_batch(&ops, 0), &[Arc::clone(&payload)])
                }));
            }
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        let admitted = results.iter().filter(|result| result.is_ok()).count();
        assert!((1..=2).contains(&admitted));
        assert!(results.iter().all(|result| {
            matches!(
                result,
                Ok(()) | Err(VoxelInboxError::Busy | VoxelInboxError::Full)
            )
        }));
        let pending = q.try_pending().unwrap();
        assert_eq!(pending, (admitted, admitted, admitted));
    }

    #[test]
    fn publication_worker_drains_sequential_revisions() {
        let terrain = VoxelTerrainId(1);
        let source = VoxelSourceId(2);
        let store = Arc::new(std::sync::RwLock::new((0, HashMap::new())));
        let writer = VoxelSourceWriter::new(terrain, source, Arc::clone(&store));
        let worker = VoxelPublicationWorker::start(writer.clone(), inbox().limits).unwrap();
        let mut tickets = Vec::new();
        for revision in 0..2 {
            let payload: Arc<[u8]> = Arc::from([revision as u8 + 1]);
            let ops = [VoxelChunkOp::Upsert(VoxelChunkUpdate {
                key: VoxelChunkKey::new(revision as i64, 0, 0, 0),
                payload: VoxelChunkPayload {
                    encoding: VOXEL_CHUNK_ENCODING_RAW,
                    schema_version: VOXEL_CHUNK_SCHEMA_VERSION,
                    bytes: &payload,
                },
            })];
            tickets.push(
                worker
                    .try_submit(&make_batch(&ops, revision), &[Arc::clone(&payload)])
                    .unwrap(),
            );
        }
        let outcome = worker.finish(VoxelInboxClose::Drain);
        assert_eq!(outcome.published_batches, 2);
        assert_eq!(outcome.failed_batches, 0);
        assert!(outcome.last_publication_time.is_some());
        assert!(outcome.failure.is_none());
        assert!(outcome.unprocessed.is_empty());
        assert!(!outcome.worker_panicked);
        assert_eq!(writer.snapshot().unwrap().revision(), 2);
        assert!(matches!(
            tickets[0].try_state(),
            Ok(VoxelPublicationTicketState::Published(_))
        ));
        assert!(matches!(
            tickets[1].try_state(),
            Ok(VoxelPublicationTicketState::Published(_))
        ));
    }

    #[test]
    fn publication_worker_retains_failed_batch_and_later_queue_for_explicit_recovery() {
        let terrain = VoxelTerrainId(1);
        let source = VoxelSourceId(2);
        let store = Arc::new(std::sync::RwLock::new((0, HashMap::new())));
        let writer = VoxelSourceWriter::new(terrain, source, Arc::clone(&store));
        let worker = VoxelPublicationWorker::start(writer.clone(), inbox().limits).unwrap();
        let mut tickets = Vec::new();
        // Hold the store lock until both revisions have been accepted.
        let guard = store.write().unwrap();
        for revision in [1, 2] {
            let payload: Arc<[u8]> = Arc::from([revision as u8]);
            let ops = [VoxelChunkOp::Upsert(VoxelChunkUpdate {
                key: VoxelChunkKey::new(revision as i64, 0, 0, 0),
                payload: VoxelChunkPayload {
                    encoding: VOXEL_CHUNK_ENCODING_RAW,
                    schema_version: VOXEL_CHUNK_SCHEMA_VERSION,
                    bytes: &payload,
                },
            })];
            tickets.push(
                worker
                    .try_submit(&make_batch(&ops, revision), &[Arc::clone(&payload)])
                    .unwrap(),
            );
        }
        drop(guard);
        let outcome = worker.finish(VoxelInboxClose::Drain);
        assert_eq!(outcome.published_batches, 0);
        assert_eq!(outcome.failed_batches, 1);
        let failure = outcome.failure.unwrap();
        assert_eq!(failure.batch.revision.expected, 1);
        assert!(matches!(
            failure.reason,
            VoxelPublicationFailureReason::Writer(VoxelUpdateError::StaleRevision {
                expected: 1,
                actual: 0
            })
        ));
        assert_eq!(outcome.unprocessed.len(), 1);
        assert_eq!(outcome.unprocessed[0].revision.expected, 2);
        assert_eq!(writer.revision().unwrap(), 0);
        assert!(matches!(
            tickets[0].try_state(),
            Ok(VoxelPublicationTicketState::Failed(_))
        ));
        assert!(matches!(
            tickets[1].try_state(),
            Ok(VoxelPublicationTicketState::Unprocessed(_))
        ));
        let mut retry = failure.batch;
        retry.revision = crate::VoxelBatchRevision {
            expected: 0,
            publish: 1,
        };
        let recovery = VoxelPublicationWorker::start(writer.clone(), inbox().limits).unwrap();
        let retry_ticket = recovery.try_retry(&retry).unwrap();
        let mut later = outcome.unprocessed.into_iter().next().unwrap();
        later.revision = crate::VoxelBatchRevision {
            expected: 1,
            publish: 2,
        };
        let later_ticket = recovery.try_retry(&later).unwrap();
        assert_eq!(recovery.try_status().unwrap().retried_batches, 2);
        let recovered = recovery.finish(VoxelInboxClose::Drain);
        assert_eq!(recovered.retried_batches, 2);
        assert_eq!(recovered.published_batches, 2);
        assert!(matches!(
            retry_ticket.wait(),
            VoxelPublicationTicketState::Published(_)
        ));
        assert!(matches!(
            later_ticket.wait(),
            VoxelPublicationTicketState::Published(_)
        ));
        assert_eq!(writer.revision().unwrap(), 2);
    }

    #[test]
    fn dropped_worker_marks_queued_ticket_discarded_and_in_flight_ticket_completes() {
        use std::{
            sync::mpsc,
            time::{Duration, Instant},
        };
        let store = Arc::new(std::sync::RwLock::new((0, HashMap::new())));
        let writer =
            VoxelSourceWriter::new(VoxelTerrainId(1), VoxelSourceId(2), Arc::clone(&store));
        let worker = VoxelPublicationWorker::start(writer.clone(), inbox().limits).unwrap();
        let guard = store.write().unwrap();
        let first_payload: Arc<[u8]> = Arc::from([1u8]);
        let first_ops = [VoxelChunkOp::Upsert(VoxelChunkUpdate {
            key: VoxelChunkKey::new(0, 0, 0, 0),
            payload: VoxelChunkPayload {
                encoding: VOXEL_CHUNK_ENCODING_RAW,
                schema_version: VOXEL_CHUNK_SCHEMA_VERSION,
                bytes: &first_payload,
            },
        })];
        let first = worker
            .try_submit(&make_batch(&first_ops, 0), &[Arc::clone(&first_payload)])
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if matches!(worker.try_status(), Ok(status) if status.pending_batches == 0) {
                break;
            }
            assert!(Instant::now() < deadline, "worker did not take first batch");
            std::thread::yield_now();
        }
        let second_payload: Arc<[u8]> = Arc::from([2u8]);
        let second_ops = [VoxelChunkOp::Upsert(VoxelChunkUpdate {
            key: VoxelChunkKey::new(1, 0, 0, 0),
            payload: VoxelChunkPayload {
                encoding: VOXEL_CHUNK_ENCODING_RAW,
                schema_version: VOXEL_CHUNK_SCHEMA_VERSION,
                bytes: &second_payload,
            },
        })];
        let second = worker
            .try_submit(&make_batch(&second_ops, 1), &[Arc::clone(&second_payload)])
            .unwrap();
        drop(worker);
        assert!(matches!(
            second.try_state(),
            Ok(VoxelPublicationTicketState::Discarded)
        ));
        drop(guard);
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || tx.send(first.wait()).unwrap());
        assert!(matches!(
            rx.recv_timeout(Duration::from_secs(2)).unwrap(),
            VoxelPublicationTicketState::Published(_)
        ));
        assert_eq!(writer.revision().unwrap(), 1);
    }

    #[test]
    fn explicit_discard_reports_queued_work_and_preserves_in_flight_result() {
        use std::time::{Duration, Instant};
        let store = Arc::new(std::sync::RwLock::new((0, HashMap::new())));
        let writer =
            VoxelSourceWriter::new(VoxelTerrainId(1), VoxelSourceId(2), Arc::clone(&store));
        let worker = VoxelPublicationWorker::start(writer.clone(), inbox().limits).unwrap();
        let guard = store.write().unwrap();
        let first_payload: Arc<[u8]> = Arc::from([1u8]);
        let first_ops = [VoxelChunkOp::Upsert(VoxelChunkUpdate {
            key: VoxelChunkKey::new(0, 0, 0, 0),
            payload: VoxelChunkPayload {
                encoding: VOXEL_CHUNK_ENCODING_RAW,
                schema_version: VOXEL_CHUNK_SCHEMA_VERSION,
                bytes: &first_payload,
            },
        })];
        let first = worker
            .try_submit(&make_batch(&first_ops, 0), &[Arc::clone(&first_payload)])
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if matches!(worker.try_status(), Ok(status) if status.pending_batches == 0) {
                break;
            }
            assert!(Instant::now() < deadline, "worker did not take first batch");
            std::thread::yield_now();
        }
        let second_payload: Arc<[u8]> = Arc::from([2u8]);
        let second_ops = [VoxelChunkOp::Upsert(VoxelChunkUpdate {
            key: VoxelChunkKey::new(1, 0, 0, 0),
            payload: VoxelChunkPayload {
                encoding: VOXEL_CHUNK_ENCODING_RAW,
                schema_version: VOXEL_CHUNK_SCHEMA_VERSION,
                bytes: &second_payload,
            },
        })];
        let second = worker
            .try_submit(&make_batch(&second_ops, 1), &[Arc::clone(&second_payload)])
            .unwrap();
        let finisher = std::thread::spawn(move || worker.finish(VoxelInboxClose::Discard));
        assert!(matches!(
            second.wait(),
            VoxelPublicationTicketState::Discarded
        ));
        drop(guard);
        let outcome = finisher.join().unwrap();
        assert_eq!(outcome.discarded_batches, 1);
        assert_eq!(outcome.published_batches, 1);
        assert!(matches!(
            first.wait(),
            VoxelPublicationTicketState::Published(_)
        ));
        assert_eq!(writer.revision().unwrap(), 1);
    }
}
