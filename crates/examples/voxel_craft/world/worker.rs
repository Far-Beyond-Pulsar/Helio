//! The world worker: owns the authoritative planet, applies the player's
//! edits off the frame, and publishes immutable snapshots for drawing and
//! queries (decoration is part of the planet's recipe: nothing streams).
//!
//! Applying brushes needs the planet by `&mut`, while the renderer and the
//! player read shared snapshots, so (as Pulsar's voxel backend does) edits are
//! applied to a private copy on a thread and handed over as a new
//! `Arc<Planet>`. A snapshot carries the ticket of the last player batch it
//! includes.

use helio_pass_voxel_planet::{Brush, Planet};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Edits to apply, numbered so snapshots can say how far they got.
pub struct Batch {
    pub ticket: u64,
    pub brushes: Vec<Brush>,
}

pub enum Job {
    Edit(Batch),
}

/// A planet with every batch up to `ticket` applied.
pub struct Snapshot {
    pub ticket: u64,
    pub planet: Arc<Planet>,
}

pub struct Worker {
    jobs: Sender<Job>,
    results: Receiver<Snapshot>,
}

/// Snapshots are copied at most this often while edits stream in.
const SNAPSHOT_INTERVAL: Duration = Duration::from_millis(16);

impl Worker {
    /// Starts the thread; it owns `planet`.
    pub fn spawn(mut planet: Planet) -> Self {
        let (jobs, job_rx) = channel::<Job>();
        let (result_tx, results) = channel::<Snapshot>();
        std::thread::Builder::new()
            .name("voxel-craft-world".into())
            .spawn(move || {
                let mut ticket = 0;
                let mut dirty = false;
                let mut last_sent = Instant::now() - SNAPSHOT_INTERVAL;
                loop {
                    let wait = if dirty { Duration::from_millis(1) } else { Duration::from_millis(100) };
                    let mut jobs = Vec::new();
                    match job_rx.recv_timeout(wait) {
                        Ok(job) => jobs.push(job),
                        Err(RecvTimeoutError::Timeout) => {}
                        Err(RecvTimeoutError::Disconnected) => break,
                    }
                    while let Ok(job) = job_rx.try_recv() {
                        jobs.push(job);
                    }
                    for job in jobs {
                        let Job::Edit(batch) = job;
                        ticket = batch.ticket;
                        for brush in batch.brushes {
                            if let Err(error) = planet.apply(brush) {
                                log::warn!("brush rejected: {error}");
                            }
                        }
                        dirty = true;
                    }
                    if dirty && last_sent.elapsed() >= SNAPSHOT_INTERVAL {
                        let snapshot = Snapshot { ticket, planet: Arc::new(planet.clone()) };
                        if result_tx.send(snapshot).is_err() {
                            break;
                        }
                        dirty = false;
                        last_sent = Instant::now();
                    }
                }
            })
            .expect("spawn the world worker");
        Self { jobs, results }
    }

    pub fn submit(&self, job: Job) {
        let _ = self.jobs.send(job);
    }

    /// The newest finished snapshot, if there is one.
    pub fn latest(&self) -> Option<Snapshot> {
        let mut newest = None;
        while let Ok(snapshot) = self.results.try_recv() {
            newest = Some(snapshot);
        }
        newest
    }
}
