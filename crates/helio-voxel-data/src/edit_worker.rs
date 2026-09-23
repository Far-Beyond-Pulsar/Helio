//! Bounded CPU edit queue over the SceneDB canonical voxel map.

use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc::{self, SyncSender, TrySendError},
        Arc, Condvar, Mutex,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crate::{VoxelBatchReceipt, VoxelDomain, VoxelEditError, VoxelSampleEdit, VoxelSourceWriter};

pub const VOXEL_EDIT_MAX_SAMPLES_PER_JOB: usize = 65_536;
pub const VOXEL_EDIT_PENDING_JOBS: usize = 2;

#[derive(Clone, Debug)]
pub struct VoxelEditJob {
    pub edits: Arc<[VoxelSampleEdit]>,
    pub material_ids: Arc<[u32]>,
    pub domain: VoxelDomain,
}

#[derive(Clone, Debug)]
pub enum VoxelEditTicketState {
    Pending,
    Published(VoxelBatchReceipt),
    Failed {
        job: VoxelEditJob,
        error: VoxelEditError,
    },
    Panicked(VoxelEditJob),
    Discarded,
}

#[derive(Clone)]
pub struct VoxelEditTicket(Arc<(Mutex<VoxelEditTicketState>, Condvar)>);

impl VoxelEditTicket {
    fn new() -> Self {
        Self(Arc::new((
            Mutex::new(VoxelEditTicketState::Pending),
            Condvar::new(),
        )))
    }
    fn set(&self, state: VoxelEditTicketState) {
        let (lock, wake) = &*self.0;
        *lock.lock().unwrap_or_else(|error| error.into_inner()) = state;
        wake.notify_all();
    }
    pub fn state(&self) -> VoxelEditTicketState {
        self.0
             .0
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }
    /// Management-thread wait; frame callbacks should poll state.
    pub fn wait(&self) -> VoxelEditTicketState {
        let (lock, wake) = &*self.0;
        let mut state = lock.lock().unwrap_or_else(|error| error.into_inner());
        while matches!(*state, VoxelEditTicketState::Pending) {
            state = wake.wait(state).unwrap_or_else(|error| error.into_inner());
        }
        state.clone()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VoxelEditAdmissionError {
    Empty,
    TooManySamples(usize),
    TooManyMaterials(usize),
    Full,
    Closed,
}

#[derive(Clone, Debug, Default)]
pub struct VoxelEditWorkerStatus {
    pub queued_jobs: usize,
    pub queued_samples: usize,
    pub active_jobs: usize,
    pub published_jobs: u64,
    pub failed_jobs: u64,
    pub discarded_jobs: u64,
    pub published_samples: u64,
    pub last_duration: Option<Duration>,
    pub last_error: Option<String>,
}

#[derive(Default)]
struct Counters {
    queued: AtomicUsize,
    queued_samples: AtomicUsize,
    active: AtomicUsize,
    published: AtomicU64,
    failed: AtomicU64,
    discarded: AtomicU64,
    samples: AtomicU64,
    last: Mutex<(Option<Duration>, Option<String>)>,
}

struct Queued {
    job: VoxelEditJob,
    ticket: VoxelEditTicket,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoxelEditClose {
    Drain,
    Discard,
}

/// One wake-driven worker per component source. The owning scene service must
/// drop/finish it on row removal or replacement; accepted tickets keep their
/// terminal state even after this handle is gone.
pub struct VoxelEditWorker {
    sender: Option<SyncSender<Queued>>,
    counters: Arc<Counters>,
    discard: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl VoxelEditWorker {
    pub fn start(writer: VoxelSourceWriter) -> Result<Self, std::io::Error> {
        let (sender, receiver) = mpsc::sync_channel::<Queued>(VOXEL_EDIT_PENDING_JOBS);
        let counters = Arc::new(Counters::default());
        let discard = Arc::new(AtomicBool::new(false));
        let worker_counters = counters.clone();
        let worker_discard = discard.clone();
        let thread = thread::Builder::new()
            .name("voxel-edits".into())
            .spawn(move || {
                while let Ok(queued) = receiver.recv() {
                    worker_counters.queued.fetch_sub(1, Ordering::AcqRel);
                    worker_counters
                        .queued_samples
                        .fetch_sub(queued.job.edits.len(), Ordering::AcqRel);
                    if worker_discard.load(Ordering::Acquire) {
                        queued.ticket.set(VoxelEditTicketState::Discarded);
                        worker_counters.discarded.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    worker_counters.active.store(1, Ordering::Release);
                    let started = Instant::now();
                    let result = catch_unwind(AssertUnwindSafe(|| {
                        writer.publish_sample_edits(
                            &queued.job.edits,
                            queued.job.domain,
                            &queued.job.material_ids,
                        )
                    }));
                    let elapsed = started.elapsed();
                    worker_counters.active.store(0, Ordering::Release);
                    let mut last = worker_counters
                        .last
                        .lock()
                        .unwrap_or_else(|error| error.into_inner());
                    last.0 = Some(elapsed);
                    match result {
                        Ok(Ok(receipt)) => {
                            worker_counters.published.fetch_add(1, Ordering::Relaxed);
                            worker_counters
                                .samples
                                .fetch_add(queued.job.edits.len() as u64, Ordering::Relaxed);
                            last.1 = None;
                            queued.ticket.set(VoxelEditTicketState::Published(receipt));
                        }
                        Ok(Err(error)) => {
                            worker_counters.failed.fetch_add(1, Ordering::Relaxed);
                            last.1 = Some(format!("{error:?}"));
                            queued.ticket.set(VoxelEditTicketState::Failed {
                                job: queued.job,
                                error,
                            });
                        }
                        Err(_) => {
                            worker_counters.failed.fetch_add(1, Ordering::Relaxed);
                            last.1 = Some("voxel edit worker panicked".into());
                            queued
                                .ticket
                                .set(VoxelEditTicketState::Panicked(queued.job));
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

    /// Enqueue immutable edit handles with a finite sample and queue bound.
    /// Validation, touched-chunk snapshots, and publication run on the worker.
    pub fn try_submit(
        &self,
        job: VoxelEditJob,
    ) -> Result<VoxelEditTicket, VoxelEditAdmissionError> {
        if job.edits.is_empty() {
            return Err(VoxelEditAdmissionError::Empty);
        }
        if job.edits.len() > VOXEL_EDIT_MAX_SAMPLES_PER_JOB {
            return Err(VoxelEditAdmissionError::TooManySamples(job.edits.len()));
        }
        if job.material_ids.len() > 255 {
            return Err(VoxelEditAdmissionError::TooManyMaterials(
                job.material_ids.len(),
            ));
        }
        let sender = self
            .sender
            .as_ref()
            .ok_or(VoxelEditAdmissionError::Closed)?;
        let ticket = VoxelEditTicket::new();
        self.counters.queued.fetch_add(1, Ordering::AcqRel);
        self.counters
            .queued_samples
            .fetch_add(job.edits.len(), Ordering::AcqRel);
        let count = job.edits.len();
        match sender.try_send(Queued {
            job,
            ticket: ticket.clone(),
        }) {
            Ok(()) => Ok(ticket),
            Err(error) => {
                self.counters.queued.fetch_sub(1, Ordering::AcqRel);
                self.counters
                    .queued_samples
                    .fetch_sub(count, Ordering::AcqRel);
                Err(match error {
                    TrySendError::Full(_) => VoxelEditAdmissionError::Full,
                    TrySendError::Disconnected(_) => VoxelEditAdmissionError::Closed,
                })
            }
        }
    }

    pub fn status(&self) -> VoxelEditWorkerStatus {
        let last = self
            .counters
            .last
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        VoxelEditWorkerStatus {
            queued_jobs: self.counters.queued.load(Ordering::Acquire),
            queued_samples: self.counters.queued_samples.load(Ordering::Acquire),
            active_jobs: self.counters.active.load(Ordering::Acquire),
            published_jobs: self.counters.published.load(Ordering::Relaxed),
            failed_jobs: self.counters.failed.load(Ordering::Relaxed),
            discarded_jobs: self.counters.discarded.load(Ordering::Relaxed),
            published_samples: self.counters.samples.load(Ordering::Relaxed),
            last_duration: last.0,
            last_error: last.1.clone(),
        }
    }

    pub fn finish(mut self, mode: VoxelEditClose) -> (VoxelEditWorkerStatus, bool) {
        if mode == VoxelEditClose::Discard {
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

impl Drop for VoxelEditWorker {
    fn drop(&mut self) {
        if self.thread.is_some() {
            self.discard.store(true, Ordering::Release);
            self.sender.take();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{VoxelChunkKey, VoxelSourceId, VoxelTerrainId};
    use std::{collections::HashMap, sync::RwLock};

    #[test]
    fn large_edit_batch_waits_off_frame_and_publishes_one_canonical_revision() {
        let store = Arc::new(RwLock::new((0, HashMap::new())));
        let writer = VoxelSourceWriter::new(VoxelTerrainId(1), VoxelSourceId(2), store.clone());
        let worker = VoxelEditWorker::start(writer.clone()).unwrap();
        let edits: Arc<[VoxelSampleEdit]> = (0..4096)
            .map(|i| VoxelSampleEdit {
                xyz: [(i % 16) as i64, ((i / 16) % 16) as i64, (i / 256) as i64],
                lod: 0,
                material_slot: 1,
            })
            .collect::<Vec<_>>()
            .into();
        let held = store.write().unwrap();
        let ticket = worker
            .try_submit(VoxelEditJob {
                edits,
                material_ids: Arc::from([7]),
                domain: VoxelDomain::Bounded {
                    min: [0; 3],
                    max: [1; 3],
                    max_lod: 0,
                },
            })
            .unwrap();
        assert!(matches!(ticket.state(), VoxelEditTicketState::Pending));
        drop(held);
        assert!(
            matches!(ticket.wait(), VoxelEditTicketState::Published(receipt) if receipt.revision == 1)
        );
        let snapshot = writer.snapshot().unwrap();
        assert_eq!(snapshot.len(), 8);
        assert!(snapshot.get(VoxelChunkKey::new(1, 1, 1, 0)).is_some());
        let (status, panicked) = worker.finish(VoxelEditClose::Drain);
        assert!(!panicked);
        assert_eq!(status.published_samples, 4096);
    }
}
