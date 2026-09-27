//! Latest-demand selection: one executing plan, one replaceable request/result.
//! Superseded work exits between bounded groups of region classifications.
use super::{build_cancellable, Plan, SelectionCache, View, World};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Condvar, Mutex,
};

struct Request {
    serial: u64,
    world: Arc<World>,
    view: View,
}
#[derive(Default)]
struct Mailbox {
    request: Option<Request>,
    result: Option<(u64, Plan)>,
}
#[derive(Default)]
struct Shared {
    mailbox: Mutex<Mailbox>,
    wake: Condvar,
    serial: AtomicU64,
    stopped: AtomicBool,
    cancelled: AtomicU64,
}

pub(super) struct Worker(Arc<Shared>);
impl Worker {
    pub fn new(max_leaves: usize) -> Self {
        let shared = Arc::new(Shared::default());
        let thread = shared.clone();
        std::thread::Builder::new().name("voxel-selection".into()).spawn(move || {
            let mut cache = SelectionCache::default();
            loop {
                let request = {
                    let mut mailbox = thread.mailbox.lock().unwrap();
                    while mailbox.request.is_none() && !thread.stopped.load(Ordering::Acquire) {
                        mailbox = thread.wake.wait(mailbox).unwrap();
                    }
                    if thread.stopped.load(Ordering::Acquire) { return; }
                    mailbox.request.take().unwrap()
                };
                let obsolete = || thread.stopped.load(Ordering::Acquire)
                    || thread.serial.load(Ordering::Acquire) != request.serial;
                let start = std::time::Instant::now();
                let Some(mut plan) = build_cancellable(request.world, request.view, max_leaves, &mut cache, obsolete) else {
                    thread.cancelled.fetch_add(1, Ordering::Relaxed);
                    continue;
                };
                // Sort on the worker, not on the render caller. Generated
                // bricks from interrupted plans can be reused at the new view.
                let eye = plan.view.eye;
                plan.leaves.sort_by(|(_, a), (_, b)| {
                    let distance = |key: &super::Key| {
                        let low = glam::DVec3::from_array(key.low.map(|v| f64::from(v) * 0.1));
                        let high = low + glam::DVec3::splat(f64::from(key.side()) * 0.1);
                        eye.distance_squared(eye.clamp(low, high))
                    };
                    distance(a).total_cmp(&distance(b))
                });
                let mut mailbox = thread.mailbox.lock().unwrap();
                if obsolete() {
                    thread.cancelled.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                eprintln!("VOXEL_PLAN serial={} nodes={} bricks={} pixel_budget={:.3} selection_ms={:.2} classified={} cached={}",
                    request.serial, plan.nodes.len(), plan.leaves.len(), plan.pixels,
                    start.elapsed().as_secs_f64()*1000.0, cache.classified, cache.reused);
                mailbox.result = Some((request.serial, plan));
            }
        }).expect("voxel selection worker");
        Self(shared)
    }

    pub fn submit(&self, world: Arc<World>, view: View) -> u64 {
        let mut mailbox = self.0.mailbox.lock().unwrap();
        let serial = self.0.serial.fetch_add(1, Ordering::AcqRel) + 1;
        mailbox.request = Some(Request {
            serial,
            world,
            view,
        });
        mailbox.result = None;
        self.0.wake.notify_one();
        serial
    }
    pub fn take(&self) -> Option<(u64, Plan)> {
        self.0.mailbox.lock().unwrap().result.take()
    }
    pub fn cancelled(&self) -> u64 {
        self.0.cancelled.load(Ordering::Relaxed)
    }
    #[cfg(test)]
    pub fn has_result(&self) -> bool {
        self.0.mailbox.lock().unwrap().result.is_some()
    }
    #[cfg(test)]
    pub fn lifetime_probe(&self) -> impl Fn() -> bool {
        let weak = Arc::downgrade(&self.0);
        move || weak.strong_count() != 0
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        // Do not join a classifier on the render thread. Its ownership is
        // bounded by this single worker and it observes shutdown cooperatively.
        let mut mailbox = self.0.mailbox.lock().unwrap();
        self.0.stopped.store(true, Ordering::Release);
        mailbox.request = None;
        mailbox.result = None;
        self.0.wake.notify_one();
    }
}
