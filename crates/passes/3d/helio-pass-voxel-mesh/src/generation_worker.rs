//! Bounded, wake-driven generation and canonical SceneDB publication.

use std::{
    collections::HashSet,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc::{self, SyncSender, TrySendError},
        Arc, Condvar, Mutex,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crate::{
    VoxelBatchReceipt, VoxelBatchRevision, VoxelChunkBatch, VoxelChunkKey, VoxelChunkOp,
    VoxelChunkPayload, VoxelChunkUpdate, VoxelGeneratorDescriptor, VoxelGeneratorRegistry,
    VoxelSourceId, VoxelSourceWriter, VoxelTerrainId, VOXEL_CHUNK_ENCODING_RAW,
    VOXEL_CHUNK_SCHEMA_VERSION,
};

pub const VOXEL_GENERATION_MAX_CHUNKS_PER_JOB: usize = 128;
pub const VOXEL_GENERATION_PENDING_JOBS: usize = 2;

#[derive(Clone, Debug)]
pub struct VoxelGenerationJob {
    pub terrain: VoxelTerrainId,
    pub source: VoxelSourceId,
    pub expected_revision: u64,
    pub descriptor: VoxelGeneratorDescriptor,
    pub keys: Vec<VoxelChunkKey>,
}

#[derive(Clone, Debug)]
pub enum VoxelGenerationTicketState {
    Pending,
    Published(VoxelBatchReceipt),
    Failed {
        job: VoxelGenerationJob,
        error: String,
    },
    Discarded,
}

#[derive(Clone)]
pub struct VoxelGenerationTicket(Arc<(Mutex<VoxelGenerationTicketState>, Condvar)>);

impl VoxelGenerationTicket {
    fn new() -> Self {
        Self(Arc::new((
            Mutex::new(VoxelGenerationTicketState::Pending),
            Condvar::new(),
        )))
    }

    fn set(&self, state: VoxelGenerationTicketState) {
        let (lock, wake) = &*self.0;
        *lock.lock().unwrap_or_else(|error| error.into_inner()) = state;
        wake.notify_all();
    }

    pub fn state(&self) -> VoxelGenerationTicketState {
        self.0
             .0
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    /// Management-thread wait. Callers using a frame callback should poll state.
    pub fn wait(&self) -> VoxelGenerationTicketState {
        let (lock, wake) = &*self.0;
        let mut state = lock.lock().unwrap_or_else(|error| error.into_inner());
        while matches!(*state, VoxelGenerationTicketState::Pending) {
            state = wake.wait(state).unwrap_or_else(|error| error.into_inner());
        }
        state.clone()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VoxelGenerationAdmissionError {
    Invalid(String),
    Full,
    Closed,
}

#[derive(Clone, Debug, Default)]
pub struct VoxelGenerationStatus {
    pub queued_jobs: usize,
    pub active_jobs: usize,
    pub published_jobs: u64,
    pub failed_jobs: u64,
    pub discarded_jobs: u64,
    pub published_chunks: u64,
    pub generated_payload_bytes: u64,
    pub last_duration: Option<Duration>,
    pub last_error: Option<String>,
}

#[derive(Default)]
struct Counters {
    queued: AtomicUsize,
    active: AtomicUsize,
    published: AtomicU64,
    failed: AtomicU64,
    discarded: AtomicU64,
    chunks: AtomicU64,
    bytes: AtomicU64,
    last: Mutex<(Option<Duration>, Option<String>)>,
}

struct QueuedJob {
    job: VoxelGenerationJob,
    ticket: VoxelGenerationTicket,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoxelGenerationClose {
    Drain,
    Discard,
}

/// One CPU worker per SceneDB component capability. The owning service drops
/// or finishes it on row removal/replacement; accepted tickets remain valid.
pub struct VoxelGenerationWorker {
    sender: Option<SyncSender<QueuedJob>>,
    counters: Arc<Counters>,
    discard: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl VoxelGenerationWorker {
    pub fn start(
        writer: VoxelSourceWriter,
        registry: VoxelGeneratorRegistry,
    ) -> Result<Self, std::io::Error> {
        let (sender, receiver) = mpsc::sync_channel::<QueuedJob>(VOXEL_GENERATION_PENDING_JOBS);
        let counters = Arc::new(Counters::default());
        let discard = Arc::new(AtomicBool::new(false));
        let worker_counters = Arc::clone(&counters);
        let worker_discard = Arc::clone(&discard);
        let thread = thread::Builder::new()
            .name("voxel-generator".into())
            .spawn(move || {
                while let Ok(queued) = receiver.recv() {
                    worker_counters.queued.fetch_sub(1, Ordering::AcqRel);
                    if worker_discard.load(Ordering::Acquire) {
                        queued.ticket.set(VoxelGenerationTicketState::Discarded);
                        worker_counters.discarded.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    worker_counters.active.store(1, Ordering::Release);
                    let started = Instant::now();
                    let result = catch_unwind(AssertUnwindSafe(|| {
                        publish_generated(&writer, &registry, &queued.job)
                    }))
                    .unwrap_or_else(|_| Err("voxel generator or publisher panicked".into()));
                    let duration = started.elapsed();
                    worker_counters.active.store(0, Ordering::Release);
                    let mut last = worker_counters
                        .last
                        .lock()
                        .unwrap_or_else(|error| error.into_inner());
                    last.0 = Some(duration);
                    match result {
                        Ok((receipt, chunks, bytes)) => {
                            worker_counters.published.fetch_add(1, Ordering::Relaxed);
                            worker_counters
                                .chunks
                                .fetch_add(chunks as u64, Ordering::Relaxed);
                            worker_counters
                                .bytes
                                .fetch_add(bytes as u64, Ordering::Relaxed);
                            last.1 = None;
                            queued
                                .ticket
                                .set(VoxelGenerationTicketState::Published(receipt));
                        }
                        Err(error) => {
                            worker_counters.failed.fetch_add(1, Ordering::Relaxed);
                            last.1 = Some(error.clone());
                            queued.ticket.set(VoxelGenerationTicketState::Failed {
                                job: queued.job,
                                error,
                            });
                        }
                    }
                }
            })?;
        Ok(Self {
            sender: Some(sender),
            counters,
            discard,
            thread: Some(thread),
        })
    }

    /// Nonblocking admission of at most 128 complete chunks (64 KiB of raw
    /// payload). Generation, validation of produced bytes, and publication run
    /// on the worker. No component-store lock is taken here.
    pub fn try_submit(
        &self,
        job: VoxelGenerationJob,
    ) -> Result<VoxelGenerationTicket, VoxelGenerationAdmissionError> {
        job.descriptor
            .validate()
            .map_err(VoxelGenerationAdmissionError::Invalid)?;
        if job.keys.is_empty() || job.keys.len() > VOXEL_GENERATION_MAX_CHUNKS_PER_JOB {
            return Err(VoxelGenerationAdmissionError::Invalid(
                "generation job must contain 1..=128 chunks".into(),
            ));
        }
        if job.expected_revision == u64::MAX {
            return Err(VoxelGenerationAdmissionError::Invalid(
                "source revision exhausted".into(),
            ));
        }
        let mut unique = HashSet::with_capacity(job.keys.len());
        for &key in &job.keys {
            job.descriptor.domain.validate_key(key).map_err(|error| {
                VoxelGenerationAdmissionError::Invalid(format!("invalid chunk key: {error:?}"))
            })?;
            if !unique.insert(key) {
                return Err(VoxelGenerationAdmissionError::Invalid(
                    "duplicate chunk key".into(),
                ));
            }
        }
        let sender = self
            .sender
            .as_ref()
            .ok_or(VoxelGenerationAdmissionError::Closed)?;
        let ticket = VoxelGenerationTicket::new();
        self.counters.queued.fetch_add(1, Ordering::AcqRel);
        match sender.try_send(QueuedJob {
            job,
            ticket: ticket.clone(),
        }) {
            Ok(()) => Ok(ticket),
            Err(error) => {
                self.counters.queued.fetch_sub(1, Ordering::AcqRel);
                Err(match error {
                    TrySendError::Full(_) => VoxelGenerationAdmissionError::Full,
                    TrySendError::Disconnected(_) => VoxelGenerationAdmissionError::Closed,
                })
            }
        }
    }

    pub fn status(&self) -> VoxelGenerationStatus {
        let last = self
            .counters
            .last
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        VoxelGenerationStatus {
            queued_jobs: self.counters.queued.load(Ordering::Acquire),
            active_jobs: self.counters.active.load(Ordering::Acquire),
            published_jobs: self.counters.published.load(Ordering::Relaxed),
            failed_jobs: self.counters.failed.load(Ordering::Relaxed),
            discarded_jobs: self.counters.discarded.load(Ordering::Relaxed),
            published_chunks: self.counters.chunks.load(Ordering::Relaxed),
            generated_payload_bytes: self.counters.bytes.load(Ordering::Relaxed),
            last_duration: last.0,
            last_error: last.1.clone(),
        }
    }

    /// Wait for accepted work to finish or mark queued work discarded. Run on
    /// a management thread; a currently active publication is allowed to end.
    pub fn finish(mut self, mode: VoxelGenerationClose) -> (VoxelGenerationStatus, bool) {
        if mode == VoxelGenerationClose::Discard {
            self.discard.store(true, Ordering::Release);
        }
        self.sender.take();
        let panicked = self
            .thread
            .take()
            .is_some_and(|thread| thread.join().is_err());
        (self.status(), panicked)
    }
}

impl Drop for VoxelGenerationWorker {
    fn drop(&mut self) {
        if self.thread.is_some() {
            self.discard.store(true, Ordering::Release);
            self.sender.take();
        }
    }
}

fn publish_generated(
    writer: &VoxelSourceWriter,
    registry: &VoxelGeneratorRegistry,
    job: &VoxelGenerationJob,
) -> Result<(VoxelBatchReceipt, usize, usize), String> {
    let mut generated = Vec::with_capacity(job.keys.len());
    let mut chunks = 0;
    for &key in &job.keys {
        let payload = registry.generate(&job.descriptor, key)?;
        chunks += usize::from(payload.is_some());
        generated.push((key, payload));
    }
    let ops: Vec<_> = generated
        .iter()
        .map(|(key, payload)| match payload {
            Some(bytes) => VoxelChunkOp::Upsert(VoxelChunkUpdate {
                key: *key,
                payload: VoxelChunkPayload {
                    encoding: VOXEL_CHUNK_ENCODING_RAW,
                    schema_version: VOXEL_CHUNK_SCHEMA_VERSION,
                    bytes,
                },
            }),
            None => VoxelChunkOp::Delete { key: *key },
        })
        .collect();
    let batch = VoxelChunkBatch {
        terrain: job.terrain,
        source: job.source,
        revision: VoxelBatchRevision {
            expected: job.expected_revision,
            publish: job.expected_revision + 1,
        },
        domain: job.descriptor.domain,
        ops: &ops,
    };
    let receipt = writer
        .publish_batch(&batch)
        .map_err(|error| format!("generation publication failed: {error:?}"))?;
    Ok((receipt, chunks, chunks * crate::VOXEL_CHUNK_SAMPLES))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{VoxelDomain, VOXEL_FLAT_GENERATOR};
    use std::{collections::HashMap, sync::RwLock};

    #[test]
    fn worker_publishes_complete_generated_batch_and_reports_stale_failure() {
        let store = Arc::new(RwLock::new((0, HashMap::new())));
        let writer = VoxelSourceWriter::new(VoxelTerrainId(9), VoxelSourceId(2), store.clone());
        let worker =
            VoxelGenerationWorker::start(writer, VoxelGeneratorRegistry::default()).unwrap();
        let descriptor = VoxelGeneratorDescriptor {
            id: VOXEL_FLAT_GENERATOR.into(),
            version: 1,
            seed: 7,
            shape_mode: 0,
            domain: VoxelDomain::Unbounded { max_lod: 0 },
            origin: [0.0; 3],
            voxel_size: 1.0,
            planet_radius: 1.0,
            base_height: 0.0,
            amplitude: 0.0,
            wavelength: 16.0,
            material_slot: 1,
        };
        let job = VoxelGenerationJob {
            terrain: VoxelTerrainId(9),
            source: VoxelSourceId(2),
            expected_revision: 0,
            descriptor,
            keys: vec![
                VoxelChunkKey::new(0, -1, 0, 0),
                VoxelChunkKey::new(0, 0, 0, 0),
            ],
        };
        assert!(matches!(
            worker.try_submit(job.clone()).unwrap().wait(),
            VoxelGenerationTicketState::Published(_)
        ));
        let state = store.read().unwrap();
        assert_eq!(state.0, 1);
        assert_eq!(state.1.len(), 1);
        drop(state);
        assert!(matches!(
            worker.try_submit(job).unwrap().wait(),
            VoxelGenerationTicketState::Failed { .. }
        ));
        let (status, panicked) = worker.finish(VoxelGenerationClose::Drain);
        assert!(!panicked);
        assert_eq!(status.published_jobs, 1);
        assert_eq!(status.failed_jobs, 1);
    }
}
