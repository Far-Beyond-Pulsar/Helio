//! One ordered GPU bundle at a time, with replaceable camera/source requests.
//! The producer owns all residency metadata. It may only advance after rendering
//! encodes and acknowledges the prior bundle; prepared GPU work is never dropped.
use super::{Job, Node, Residency, Stats, View, World};
use crate::Params;
use std::{
    sync::{Arc, Condvar, Mutex, TryLockError},
    time::Instant,
};

pub type Batch = (Vec<Job>, Vec<u32>, Arc<World>);

struct Request {
    serial: u64,
    world: Arc<World>,
    params: Params,
}

pub struct Prepared {
    sequence: u64,
    request: u64,
    pub batch: Option<Batch>,
    /// Root integer bounds followed by packed child links, ready for upload.
    pub publication: Option<Vec<u32>>,
    active: Option<(Arc<World>, View)>,
    stats: Stats,
}

pub fn pack(nodes: &[Node]) -> Vec<u32> {
    let mut links = Vec::with_capacity(nodes.len() + 4);
    links.extend(nodes[0].low.map(|v| v as u32));
    links.push(nodes[0].level);
    links.extend(nodes.iter().map(|node| node.child));
    links
}

fn prepare(state: &mut Residency, request: Request) -> Prepared {
    let start = Instant::now();
    state.update(&request.world, &request.params);
    let batch = state.next_batch();
    let publication = state.publish().map(|nodes| pack(&nodes));
    let active = state
        .active_world
        .as_ref()
        .zip(state.active_view)
        .map(|(world, view)| (world.clone(), view));
    let mut stats = state.stats;
    stats.worker_cpu_ms = start.elapsed().as_secs_f64() * 1000.0;
    Prepared {
        sequence: 0,
        request: request.serial,
        batch,
        publication,
        active,
        stats,
    }
}

#[derive(Default)]
struct Mailbox {
    request: Option<Request>,
    ready: Option<Prepared>,
    in_flight: Option<u64>,
    stopped: bool,
    failed: bool,
}
#[derive(Default)]
struct Shared {
    mailbox: Mutex<Mailbox>,
    wake: Condvar,
}

struct Worker(Arc<Shared>);
impl Worker {
    fn spawn(mut process: impl FnMut(Request) -> Prepared + Send + 'static) -> Self {
        let shared = Arc::new(Shared::default());
        let thread = shared.clone();
        std::thread::Builder::new()
            .name("voxel-residency".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut sequence = 0;
                    loop {
                        let request = {
                            let mut mailbox = thread.mailbox.lock().unwrap();
                            while !mailbox.stopped
                                && (mailbox.request.is_none()
                                    || mailbox.ready.is_some()
                                    || mailbox.in_flight.is_some())
                            {
                                mailbox = thread.wake.wait(mailbox).unwrap();
                            }
                            if mailbox.stopped {
                                return;
                            }
                            mailbox.request.take().unwrap()
                        };
                        // All classification/admission/generation bookkeeping and
                        // packing occur outside the mailbox critical section.
                        let mut prepared = process(request);
                        sequence += 1;
                        prepared.sequence = sequence;
                        let mut mailbox = thread.mailbox.lock().unwrap();
                        if mailbox.stopped {
                            return;
                        }
                        assert!(mailbox.ready.is_none() && mailbox.in_flight.is_none());
                        mailbox.ready = Some(prepared);
                    }
                }));
                if result.is_err() {
                    // Surface failure on the next render poll rather than leave an
                    // apparently refining world stuck forever after a worker panic.
                    let mut mailbox = thread.mailbox.lock().unwrap_or_else(|e| e.into_inner());
                    mailbox.failed = true;
                    thread.wake.notify_all();
                }
            })
            .expect("voxel residency worker");
        Self(shared)
    }

    /// A busy mailbox skips this poll; it never makes rendering wait for work.
    fn exchange(&self, request: Request, acknowledgement: Option<u64>) -> Option<Option<Prepared>> {
        let mut mailbox = match self.0.mailbox.try_lock() {
            Ok(mailbox) => mailbox,
            Err(TryLockError::WouldBlock) => return None,
            Err(TryLockError::Poisoned(_)) => panic!("voxel residency mailbox poisoned"),
        };
        assert!(!mailbox.failed, "voxel residency worker failed");
        if let Some(sequence) = acknowledgement {
            assert_eq!(mailbox.in_flight.take(), Some(sequence));
        }
        mailbox.request = Some(request);
        let ready = mailbox.ready.take();
        if let Some(prepared) = &ready {
            assert!(mailbox.in_flight.is_none());
            mailbox.in_flight = Some(prepared.sequence);
        }
        self.0.wake.notify_one();
        Some(ready)
    }

    fn acknowledge(&self, sequence: u64) -> bool {
        let mut mailbox = match self.0.mailbox.try_lock() {
            Ok(mailbox) => mailbox,
            Err(TryLockError::WouldBlock) => return false,
            Err(TryLockError::Poisoned(_)) => panic!("voxel residency mailbox poisoned"),
        };
        assert!(!mailbox.failed, "voxel residency worker failed");
        assert_eq!(mailbox.in_flight.take(), Some(sequence));
        self.0.wake.notify_one();
        true
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        let mut mailbox = self.0.mailbox.lock().unwrap_or_else(|e| e.into_inner());
        mailbox.stopped = true;
        self.0.wake.notify_all();
        // No render-thread join. Work and its owned buffers exit cooperatively;
        // no GPU object is owned or touched by this worker.
    }
}

enum Producer {
    Inline(Box<Residency>),
    Worker(Worker),
}

pub struct Pipeline {
    producer: Producer,
    serial: u64,
    in_flight: Option<u64>,
    accepted: bool,
    acknowledgement: Option<u64>,
    demand: Option<(Arc<World>, View)>,
    active: Option<(Arc<World>, View)>,
    pub stats: Stats,
}
impl Pipeline {
    pub fn new(capacity: usize) -> Self {
        Self::configured(
            capacity,
            // Keep the measured inline control available in the same binary.
            std::env::var_os("HELIO_VOXEL_INLINE_ADMISSION").is_none(),
            std::env::var_os("HELIO_VOXEL_RETARGETING").is_some(),
        )
    }
    fn configured(capacity: usize, asynchronous: bool, retargeting: bool) -> Self {
        let create = move || {
            let mut state = Residency::new(capacity);
            state.retargeting = retargeting;
            state
        };
        let producer = if asynchronous {
            let mut state = None;
            Producer::Worker(Worker::spawn(move |request| {
                prepare(state.get_or_insert_with(create), request)
            }))
        } else {
            Producer::Inline(Box::new(create()))
        };
        Self {
            producer,
            serial: 0,
            in_flight: None,
            accepted: false,
            acknowledgement: None,
            demand: None,
            active: None,
            stats: Stats {
                async_admission: asynchronous,
                ..Stats::default()
            },
        }
    }

    pub fn prepare(&mut self, world: &Arc<World>, params: &Params) -> Option<Prepared> {
        assert!(
            self.in_flight.is_none(),
            "previous voxel bundle was not acknowledged"
        );
        let start = Instant::now();
        self.serial += 1;
        self.demand = Some((world.clone(), View::new(params)));
        let request = Request {
            serial: self.serial,
            world: world.clone(),
            params: *params,
        };
        let prepared = match &mut self.producer {
            Producer::Inline(state) => {
                let mut prepared = prepare(state, request);
                prepared.sequence = self.serial;
                prepared.stats.worker_cpu_ms = 0.0;
                Some(prepared)
            }
            Producer::Worker(worker) => {
                let ready = match worker.exchange(request, self.acknowledgement) {
                    Some(ready) => {
                        self.acknowledgement = None;
                        ready
                    }
                    None => None,
                };
                if ready.is_none() {
                    self.stats.pipeline_misses += 1;
                    // No bundle is attributed to this frame. Do not duplicate
                    // the last worker sample across frames that missed a poll.
                    self.stats.worker_cpu_ms = 0.0;
                    self.stats.update_cpu_ms = 0.0;
                }
                ready
            }
        };
        self.in_flight = prepared.as_ref().map(|p| p.sequence);
        self.accepted = false;
        self.refresh_demand();
        self.stats.prepare_cpu_ms = start.elapsed().as_secs_f64() * 1000.0;
        prepared
    }

    /// Caller has encoded this bundle's generation and uploaded its publication.
    /// Subsequent terrain traversal must use this bundle's active source/grid.
    pub fn accept_after_encode(&mut self, prepared: Prepared) {
        assert_eq!(self.in_flight, Some(prepared.sequence));
        assert!(prepared.request <= self.serial);
        assert!(
            prepared.batch.is_none() && prepared.publication.is_none(),
            "voxel bundle contains unconsumed GPU work"
        );
        let prepare_ms = self.stats.prepare_cpu_ms;
        let misses = self.stats.pipeline_misses;
        let asynchronous = self.stats.async_admission;
        self.stats = prepared.stats;
        self.stats.prepare_cpu_ms = prepare_ms;
        self.stats.pipeline_misses = misses;
        self.stats.async_admission = asynchronous;
        self.active = prepared.active;
        self.refresh_demand();
        self.accepted = true;
    }

    pub fn finish_frame(&mut self) {
        if let Some(sequence) = self.in_flight.take() {
            assert!(self.accepted, "voxel bundle was not encoded");
            if let Producer::Worker(worker) = &self.producer {
                if !worker.acknowledge(sequence) {
                    self.acknowledgement = Some(sequence);
                }
            }
        }
    }
    fn refresh_demand(&mut self) {
        self.stats.refining = self.demand.as_ref().is_some_and(|(world, view)| {
            self.active.as_ref().is_none_or(|(active, selected)| {
                !Arc::ptr_eq(world, active) || selected.changed(*view)
            })
        });
    }
    pub fn active_voxel_step(&self) -> u32 {
        self.active
            .as_ref()
            .map_or(1, |(world, _)| world.voxel_step())
    }
    #[cfg(feature = "canonical-far-experiment")]
    pub fn active_world(&self) -> Option<&Arc<World>> {
        self.active.as_ref().map(|(world, _)| world)
    }
}

#[cfg(test)]
mod tests;
