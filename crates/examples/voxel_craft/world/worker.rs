//! The world worker: owns the authoritative planet, applies the player's
//! edits, loads the features (`decorate`) of the regions around them and
//! unloads those left behind, off the frame, and publishes immutable
//! snapshots for drawing and queries.
//!
//! Applying brushes needs the planet by `&mut`, while the renderer and the
//! player read shared snapshots, so (as Pulsar's voxel backend does) edits are
//! applied to a private copy on a thread and handed over as a new
//! `Arc<Planet>`. A snapshot carries the ticket of the last player batch it
//! includes.

use super::decorate;
use crate::config::{DECORATE_BATCH, REGION_BLOCKS, REGION_KEEP, REGION_RADIUS};
use helio_pass_voxel_planet::features::Region;
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
}

pub struct Worker {
    jobs: Sender<Job>,
    results: Receiver<Snapshot>,
}

/// Snapshots are copied at most this often while edits stream in.
const SNAPSHOT_INTERVAL: Duration = Duration::from_millis(16);
/// While only features change (no player edit waiting), snapshots are
/// rarer: each one makes the renderer generate the tiles they changed.
const FEATURES_INTERVAL: Duration = Duration::from_millis(500);

/// Regions decorated around `focus`: their features loaded nearest first,
/// those past [`REGION_KEEP`] unloaded.
struct Regions {
    radius: i32,
    /// The focus region the wanted list was made for, and the list
    /// (nearest first) with how far its loading got.
    wanted: Option<(Region, Vec<Region>, usize)>,
}

impl Regions {
    fn new() -> Self {
        let radius = std::env::var("VOXEL_CRAFT_RADIUS").ok().and_then(|v| v.parse().ok()).unwrap_or(REGION_RADIUS);
        Self { radius, wanted: None }
    }

    /// Loads the next batch of the regions around `focus` and unloads those
    /// left behind (when the focus moved to another region); whether
    /// anything changed.
    fn update(&mut self, planet: &mut Planet, focus: Cell) -> bool {
        let centre = Region::of(focus, REGION_BLOCKS);
        let regions = planet.grid().cells() / REGION_BLOCKS + 1;
        let mut changed = false;
        if self.wanted.as_ref().is_none_or(|(c, _, _)| *c != centre) {
            let keep = self.radius + REGION_KEEP;
            let gone: Vec<u64> = planet.edits().features.keys().filter(|&k| {
                let r = Region::from_key(k);
                r.face != centre.face || (r.x - centre.x).pow(2) + (r.y - centre.y).pow(2) > keep * keep
            }).collect();
            for key in gone {
                changed |= planet.unload_features(key);
            }
            let wanted = centre.around(self.radius, regions).into_iter().filter(|r| decorate::decorated(planet, *r)).collect();
            self.wanted = Some((centre, wanted, 0));
        }
        let Some((_, wanted, done)) = &mut self.wanted else { return changed };
        let end = (*done + DECORATE_BATCH).min(wanted.len());
        let next: Vec<Region> = wanted[*done..end].iter().filter(|r| !planet.edits().features.contains(r.key())).copied().collect();
        *done = end;
        if next.is_empty() {
            return changed;
        }
        // Built side by side against the same terrain, then loaded in order.
        let started = Instant::now();
        let built = parallel(&next, |r| decorate::decorate(planet, *r));
        let decided = started.elapsed().as_secs_f64() * 1e3;
        let cells: usize = built.iter().map(Vec::len).sum();
        for (region, cells) in next.iter().zip(built) {
            if let Err(error) = planet.load_features(region.key(), cells) {
                log::warn!("features of {region:?} rejected: {error}");
            }
        }
        log::debug!("{} regions ({cells} cells): decided in {decided:.0} ms, loaded in {:.0} ms ({} loaded, {} feature bricks, {:?} baked bricks)", next.len(), started.elapsed().as_secs_f64() * 1e3 - decided, planet.edits().features.len(), planet.edits().features.bricks(), planet.edits().baked.brick_counts());
        true
    }

    /// Whether regions around the last focus are still to load.
    fn loading(&self) -> bool {
        self.wanted.as_ref().is_some_and(|(_, wanted, done)| *done < wanted.len())
    }
}

/// `f` over `items` on every core, in their order.
fn parallel<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(items.len()).max(1);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut out: Vec<(usize, R)> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut mine = Vec::new();
                    loop {
                        let n = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(item) = items.get(n) else { break };
                        mine.push((n, f(item)));
                    }
                    mine
                })
            })
            .collect();
        workers.into_iter().flat_map(|w| w.join().expect("decorate thread")).collect()
    });
    out.sort_by_key(|(n, _)| *n);
    out.into_iter().map(|(_, r)| r).collect()
}

impl Worker {
    /// Starts the thread; it owns `planet`.
    pub fn spawn(mut planet: Planet) -> Self {
        let (jobs, job_rx) = channel::<Job>();
        let (result_tx, results) = channel::<Snapshot>();
        std::thread::Builder::new()
            .name("voxel-craft-world".into())
            .spawn(move || {
                let mut regions = Regions::new();
                let mut focus: Option<Cell> = None;
                let mut ticket = 0;
                let mut dirty = false;
                let mut urgent = false;
                let mut last_sent = Instant::now() - SNAPSHOT_INTERVAL;
                loop {
                    let wait = if dirty || regions.loading() { Duration::from_millis(1) } else { Duration::from_millis(100) };
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
                    // A waiting edit goes out first: features follow its snapshot.
                    if !urgent {
                        if let Some(focus) = focus {
                            dirty |= regions.update(&mut planet, focus);
                        }
                    }
                    let interval = if urgent { SNAPSHOT_INTERVAL } else { FEATURES_INTERVAL };
                    if dirty && last_sent.elapsed() >= interval {
                        let snapshot = Snapshot { ticket, planet: Arc::new(planet.clone()) };
                        if result_tx.send(snapshot).is_err() {
                            break;
                        }
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
