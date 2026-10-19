//! The world worker: owns the authoritative planet, applies the player's
//! edits and decorates the land around them off the frame, and publishes
//! immutable snapshots for drawing and queries.
//!
//! Applying brushes needs the planet by `&mut`, while the renderer and the
//! player read shared snapshots, so (as Pulsar's voxel backend does) edits are
//! applied to a private copy on a thread and handed over as a new
//! `Arc<Planet>`. A snapshot carries the ticket of the last player batch it
//! includes.

use super::decorate::Decorator;
use crate::config::DECORATE_BATCH;
use helio_pass_voxel_planet::{Brush, Cell, Planet};
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
    /// Decorate around this cell.
    Focus(Cell),
}

/// A planet with every batch up to `ticket` applied.
pub struct Snapshot {
    pub ticket: u64,
    pub planet: Arc<Planet>,
    /// Regions decorated so far.
    pub regions: usize,
}

pub struct Worker {
    jobs: Sender<Job>,
    results: Receiver<Snapshot>,
}

/// Snapshots are copied at most this often while edits stream in.
const SNAPSHOT_INTERVAL: Duration = Duration::from_millis(16);
/// While only decorating (no player edit waiting), snapshots are rarer: each
/// one makes the renderer regenerate the tiles the new edits touch.
const DECORATION_INTERVAL: Duration = Duration::from_millis(2500);

impl Worker {
    /// Starts the thread; it owns `planet`.
    pub fn spawn(mut planet: Planet) -> Self {
        let (jobs, job_rx) = channel::<Job>();
        let (result_tx, results) = channel::<Snapshot>();
        std::thread::Builder::new()
            .name("voxel-craft-world".into())
            .spawn(move || {
                let mut decorator = Decorator::new();
                let mut focus: Option<Cell> = None;
                let mut ticket = 0;
                let mut dirty = false;
                let mut urgent = false;
                let mut last_sent = Instant::now() - SNAPSHOT_INTERVAL;
                loop {
                    let work_left = focus.is_some_and(|f| decorator.has_work(&planet, f));
                    let wait = if work_left || dirty { Duration::from_millis(1) } else { Duration::from_millis(100) };
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
                        match job {
                            Job::Focus(cell) => focus = Some(cell),
                            Job::Edit(batch) => {
                                ticket = batch.ticket;
                                for brush in batch.brushes {
                                    if let Err(error) = planet.apply(brush) {
                                        log::warn!("brush rejected: {error}");
                                    }
                                }
                                dirty = true;
                                urgent = true;
                            }
                        }
                    }
                    // A batch of regions is built in parallel, then applied in
                    // order; batches stay small so player edits are not kept waiting.
                    let scan_started = Instant::now();
                    // A waiting edit goes out first: decoration resumes after its snapshot.
                    let batch = if urgent { Vec::new() } else { focus.map(|f| decorator.next_batch(&planet, f, DECORATE_BATCH)).unwrap_or_default() };
                    let scan_ms = scan_started.elapsed().as_secs_f64() * 1e3;
                    if !batch.is_empty() {
                        let started = Instant::now();
                        let built = decorator.decorate_all(&planet, &batch);
                        let made: usize = built.iter().map(Vec::len).sum();
                        let built_ms = started.elapsed().as_secs_f64() * 1e3;
                        let all: Vec<Brush> = built.into_iter().flatten().collect();
                        if let Err(error) = planet.apply_sealed(&all) {
                            log::warn!("decoration rejected: {error}");
                        }
                        log::debug!("{} regions: {made} brushes, scan {scan_ms:.0} ms, built in {built_ms:.0} ms, all in {:.0} ms", batch.len(), started.elapsed().as_secs_f64() * 1e3);
                        dirty = true;
                    }
                    let interval = if urgent { SNAPSHOT_INTERVAL } else { DECORATION_INTERVAL };
                    if dirty && last_sent.elapsed() >= interval {
                        let snap_started = Instant::now();
                        let snapshot = Snapshot { ticket, planet: Arc::new(planet.clone()), regions: decorator.regions_done() };
                        if result_tx.send(snapshot).is_err() {
                            break;
                        }
                        log::debug!("snapshot in {:.0} ms", snap_started.elapsed().as_secs_f64() * 1e3);
                        dirty = false;
                        urgent = false;
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
