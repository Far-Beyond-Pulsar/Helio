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
    VoxelSourceId, VoxelSourceWriter, VoxelTerrainId,
};

pub const VOXEL_GENERATION_MAX_CHUNKS_PER_JOB: usize = 128;
pub const VOXEL_GENERATION_PENDING_JOBS: usize = 2;
pub const VOXEL_GENERATION_MAX_PAYLOAD_BYTES_PER_JOB: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VoxelGenerationLimits {
    pub max_pending_jobs: usize,
    pub max_chunks_per_job: usize,
    pub max_payload_bytes_per_job: usize,
}

impl Default for VoxelGenerationLimits {
    fn default() -> Self {
        Self {
            max_pending_jobs: VOXEL_GENERATION_PENDING_JOBS,
            max_chunks_per_job: VOXEL_GENERATION_MAX_CHUNKS_PER_JOB,
            max_payload_bytes_per_job: VOXEL_GENERATION_MAX_PAYLOAD_BYTES_PER_JOB,
        }
    }
}

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
    limits: VoxelGenerationLimits,
    counters: Arc<Counters>,
    discard: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl VoxelGenerationWorker {
    pub fn start(
        writer: VoxelSourceWriter,
        registry: VoxelGeneratorRegistry,
    ) -> Result<Self, std::io::Error> {
        Self::start_with_limits(writer, registry, VoxelGenerationLimits::default())
    }

    pub fn start_with_limits(
        writer: VoxelSourceWriter,
        registry: VoxelGeneratorRegistry,
        limits: VoxelGenerationLimits,
    ) -> Result<Self, std::io::Error> {
        if limits.max_pending_jobs == 0
            || limits.max_chunks_per_job == 0
            || limits.max_chunks_per_job > crate::MAX_VOXEL_BATCH_UPDATES
            || limits.max_payload_bytes_per_job == 0
            || limits.max_payload_bytes_per_job > crate::MAX_VOXEL_BATCH_PAYLOAD_BYTES
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "invalid voxel generation limits",
            ));
        }
        let (sender, receiver) = mpsc::sync_channel::<QueuedJob>(limits.max_pending_jobs);
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
                        publish_generated(
                            &writer,
                            &registry,
                            &queued.job,
                            limits.max_payload_bytes_per_job,
                        )
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
            limits,
            counters,
            discard,
            thread: Some(thread),
        })
    }

    /// Nonblocking admission of at most the configured number of chunk requests. Generated payload
    /// bytes are capped per job on the worker before canonical publication.
    /// No component-store lock is taken here.
    pub fn try_submit(
        &self,
        job: VoxelGenerationJob,
    ) -> Result<VoxelGenerationTicket, VoxelGenerationAdmissionError> {
        job.descriptor
            .validate()
            .map_err(VoxelGenerationAdmissionError::Invalid)?;
        if job.keys.is_empty() || job.keys.len() > self.limits.max_chunks_per_job {
            return Err(VoxelGenerationAdmissionError::Invalid(format!(
                "generation job must contain 1..={} chunks",
                self.limits.max_chunks_per_job
            )));
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
    max_payload_bytes: usize,
) -> Result<(VoxelBatchReceipt, usize, usize), String> {
    let mut generated = Vec::with_capacity(job.keys.len());
    let mut chunks = 0;
    let mut payload_bytes = 0usize;
    for &key in &job.keys {
        let payload = registry.generate(&job.descriptor, key)?;
        if let Some(payload) = &payload {
            payload_bytes = payload_bytes
                .checked_add(payload.bytes.len())
                .ok_or("generated voxel payload byte count overflowed")?;
            if payload_bytes > max_payload_bytes {
                return Err("generated voxel payloads exceed the per-job byte limit".into());
            }
        }
        chunks += usize::from(payload.is_some());
        generated.push((key, payload));
    }
    let ops: Vec<_> = generated
        .iter()
        .map(|(key, payload)| match payload {
            Some(bytes) => VoxelChunkOp::Upsert(VoxelChunkUpdate {
                key: *key,
                payload: VoxelChunkPayload {
                    encoding: bytes.encoding,
                    schema_version: bytes.schema_version,
                    bytes: &bytes.bytes,
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
    let payloads: Vec<_> = generated
        .iter()
        .filter_map(|(_, payload)| payload.as_ref().map(|payload| Arc::clone(&payload.bytes)))
        .collect();
    let receipt = writer
        .publish_shared_batch(&batch, &payloads)
        .map_err(|error| format!("generation publication failed: {error:?}"))?;
    Ok((receipt, chunks, payload_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        VoxelDomain, VoxelFormatDescriptor, VoxelFormatRegistry, VoxelStoredPayload,
        VOXEL_FLAT_GENERATOR,
    };
    use std::{collections::HashMap, sync::RwLock};

    const FIELD_FORMAT: crate::VoxelFormatId = (1u128 << 96) | 42;

    #[test]
    fn registered_generator_publishes_non_material_payload_without_repacking() {
        struct Field;
        impl crate::VoxelChunkGenerator for Field {
            fn generate(
                &self,
                descriptor: &VoxelGeneratorDescriptor,
                _key: VoxelChunkKey,
            ) -> Result<Option<VoxelStoredPayload>, String> {
                assert_eq!(descriptor.chunk_edge_voxels, 2);
                assert_eq!(descriptor.parameters, "quality=high");
                Ok(Some(VoxelStoredPayload {
                    encoding: FIELD_FORMAT,
                    schema_version: 3,
                    bytes: Arc::from([0xA5, 2, 4, 8]),
                }))
            }
        }
        let mut formats = VoxelFormatRegistry::default();
        formats
            .register(VoxelFormatDescriptor {
                encoding: FIELD_FORMAT,
                schema_version: 3,
                min_bytes: 4,
                max_bytes: 4,
                validate: |bytes| bytes[0] == 0xA5,
            })
            .unwrap();
        let store = Arc::new(RwLock::new((0, HashMap::new())));
        let writer = VoxelSourceWriter::new_with_formats(
            VoxelTerrainId(9),
            VoxelSourceId(2),
            store,
            Arc::new(formats),
        );
        let mut registry = VoxelGeneratorRegistry::default();
        registry.register("test.field", 3, Arc::new(Field)).unwrap();
        let worker = VoxelGenerationWorker::start_with_limits(
            writer.clone(),
            registry.clone(),
            VoxelGenerationLimits {
                max_pending_jobs: 1,
                max_chunks_per_job: 1,
                max_payload_bytes_per_job: 3,
            },
        )
        .unwrap();
        let key = VoxelChunkKey::new(-2, 4, 1, 1);
        let job = VoxelGenerationJob {
            terrain: VoxelTerrainId(9),
            source: VoxelSourceId(2),
            expected_revision: 0,
            descriptor: VoxelGeneratorDescriptor {
                id: "test.field".into(),
                version: 3,
                seed: 77,
                domain: VoxelDomain::Unbounded { max_lod: 2 },
                origin: [0.0; 3],
                voxel_size: 0.1,
                chunk_edge_voxels: 2,
                lod_scale: 3,
                parameters: "quality=high".into(),
            },
            keys: vec![key],
        };
        assert!(matches!(
            worker.try_submit(job.clone()).unwrap().wait(),
            VoxelGenerationTicketState::Failed { .. }
        ));
        assert_eq!(writer.revision().unwrap(), 0);
        worker.finish(VoxelGenerationClose::Drain);
        let worker = VoxelGenerationWorker::start_with_limits(
            writer.clone(),
            registry,
            VoxelGenerationLimits {
                max_pending_jobs: 1,
                max_chunks_per_job: 1,
                max_payload_bytes_per_job: 4,
            },
        )
        .unwrap();
        assert!(matches!(
            worker.try_submit(job).unwrap().wait(),
            VoxelGenerationTicketState::Published(_)
        ));
        let payload = writer.snapshot().unwrap().get_payload(key).unwrap().clone();
        assert_eq!(
            (payload.encoding, payload.schema_version),
            (FIELD_FORMAT, 3)
        );
        assert_eq!(payload.as_ref(), &[0xA5, 2, 4, 8]);
        let (status, panicked) = worker.finish(VoxelGenerationClose::Drain);
        assert!(!panicked);
        assert_eq!(status.generated_payload_bytes, 4);
    }

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
            domain: VoxelDomain::Unbounded { max_lod: 0 },
            origin: [0.0; 3],
            voxel_size: 1.0,
            chunk_edge_voxels: 8,
            lod_scale: 2,
            parameters: String::new(),
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

    #[test]
    fn generation_admission_is_bounded_and_discard_reports_every_queued_job() {
        struct Blocking {
            entered: Mutex<Option<mpsc::Sender<()>>>,
            gate: Arc<(Mutex<bool>, Condvar)>,
        }
        impl crate::VoxelChunkGenerator for Blocking {
            fn generate(
                &self,
                _descriptor: &VoxelGeneratorDescriptor,
                _key: VoxelChunkKey,
            ) -> Result<Option<VoxelStoredPayload>, String> {
                if let Some(entered) = self.entered.lock().unwrap().take() {
                    entered.send(()).unwrap();
                }
                let (lock, wake) = &*self.gate;
                let mut open = lock.lock().unwrap();
                while !*open {
                    open = wake.wait(open).unwrap();
                }
                Ok(Some(VoxelStoredPayload::raw_material(
                    [1; crate::VOXEL_CHUNK_SAMPLES],
                )))
            }
        }
        let (entered_tx, entered_rx) = mpsc::channel();
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let mut registry = VoxelGeneratorRegistry::default();
        registry
            .register(
                "test.blocking",
                1,
                Arc::new(Blocking {
                    entered: Mutex::new(Some(entered_tx)),
                    gate: gate.clone(),
                }),
            )
            .unwrap();
        let store = Arc::new(RwLock::new((0, HashMap::new())));
        let writer = VoxelSourceWriter::new(VoxelTerrainId(4), VoxelSourceId(3), store);
        let worker = VoxelGenerationWorker::start(writer, registry).unwrap();
        let job = VoxelGenerationJob {
            terrain: VoxelTerrainId(4),
            source: VoxelSourceId(3),
            expected_revision: 0,
            descriptor: VoxelGeneratorDescriptor {
                id: "test.blocking".into(),
                version: 1,
                seed: 0,
                domain: VoxelDomain::Unbounded { max_lod: 0 },
                origin: [0.0; 3],
                voxel_size: 1.0,
                chunk_edge_voxels: 8,
                lod_scale: 2,
                parameters: String::new(),
            },
            keys: vec![VoxelChunkKey::new(0, 0, 0, 0)],
        };
        let active = worker.try_submit(job.clone()).unwrap();
        entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let queued_a = worker.try_submit(job.clone()).unwrap();
        let queued_b = worker.try_submit(job.clone()).unwrap();
        assert!(matches!(
            worker.try_submit(job),
            Err(VoxelGenerationAdmissionError::Full)
        ));
        let discard = worker.discard.clone();
        let finishing = thread::spawn(move || worker.finish(VoxelGenerationClose::Discard));
        let deadline = Instant::now() + Duration::from_secs(2);
        while !discard.load(Ordering::Acquire) && Instant::now() < deadline {
            thread::yield_now();
        }
        assert!(discard.load(Ordering::Acquire));
        {
            let (lock, wake) = &*gate;
            *lock.lock().unwrap() = true;
            wake.notify_all();
        }
        let (status, panicked) = finishing.join().unwrap();
        assert!(!panicked);
        assert!(matches!(
            active.wait(),
            VoxelGenerationTicketState::Published(_)
        ));
        assert!(matches!(
            queued_a.wait(),
            VoxelGenerationTicketState::Discarded
        ));
        assert!(matches!(
            queued_b.wait(),
            VoxelGenerationTicketState::Discarded
        ));
        assert_eq!(status.queued_jobs, 0);
        assert_eq!(status.discarded_jobs, 2);
    }
}
