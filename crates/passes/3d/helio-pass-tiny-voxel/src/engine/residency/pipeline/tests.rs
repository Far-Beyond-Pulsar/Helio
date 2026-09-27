use super::*;
use bytemuck::Zeroable;
use std::sync::atomic::{AtomicUsize, Ordering};

fn wait(mut ready: impl FnMut() -> bool) {
    let start = Instant::now();
    while !ready() {
        assert!(start.elapsed().as_secs() < 30, "pipeline timed out");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}
fn request(serial: u64) -> Request {
    Request {
        serial,
        world: Arc::new(World::default()),
        params: Params::zeroed(),
    }
}
fn empty(request: Request) -> Prepared {
    Prepared {
        sequence: 0,
        request: request.serial,
        batch: None,
        publication: None,
        active: None,
        stats: Stats::default(),
    }
}
fn take(worker: &Worker, serial: u64) -> Prepared {
    let mut result = None;
    wait(|| {
        if let Some(ready) = worker.exchange(request(serial), None) {
            result = ready;
        }
        result.is_some()
    });
    result.unwrap()
}

#[test]
fn worker_coalesces_requests_but_never_discards_or_overtakes_gpu_bundles() {
    let completed = Arc::new(AtomicUsize::new(0));
    let counter = completed.clone();
    let worker = Worker::spawn(move |r| {
        counter.fetch_add(1, Ordering::SeqCst);
        empty(r)
    });
    wait(|| worker.exchange(request(1), None).is_some());
    wait(|| worker.0.mailbox.lock().unwrap().ready.is_some());
    let first = take(&worker, 2);
    assert_eq!((first.sequence, first.request), (1, 1));
    for serial in 3..=20 {
        wait(|| worker.exchange(request(serial), None).is_some());
    }
    assert_eq!(
        completed.load(Ordering::SeqCst),
        1,
        "advanced before GPU encoding"
    );
    {
        let mailbox = worker.0.mailbox.lock().unwrap();
        assert_eq!(mailbox.in_flight, Some(1));
        assert_eq!(mailbox.request.as_ref().unwrap().serial, 20);
        assert!(mailbox.ready.is_none());
        // Held mailbox must cause a missed poll/ack, not a blocking render call.
        assert!(worker.exchange(request(21), None).is_none());
        assert!(!worker.acknowledge(1));
    }
    wait(|| worker.acknowledge(1));
    wait(|| worker.0.mailbox.lock().unwrap().ready.is_some());
    assert_eq!(completed.load(Ordering::SeqCst), 2);
    let second = take(&worker, 22);
    assert_eq!((second.sequence, second.request), (2, 20));
    let weak = Arc::downgrade(&worker.0);
    drop(worker); // also releases a consumed but unacknowledged bundle at teardown
    wait(|| weak.strong_count() == 0);
}

#[test]
fn dropping_pipeline_does_not_join_running_cpu_work() {
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let worker = Worker::spawn(move |r| {
        started_tx.send(()).unwrap();
        release_rx.recv().unwrap();
        empty(r)
    });
    wait(|| worker.exchange(request(1), None).is_some());
    started_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    let weak = Arc::downgrade(&worker.0);
    drop(worker);
    assert_eq!(
        weak.strong_count(),
        1,
        "worker should still own the blocked CPU task"
    );
    release_tx.send(()).unwrap();
    wait(|| weak.strong_count() == 0);
}

#[test]
fn worker_failure_is_observable_instead_of_permanent_refinement() {
    let worker = Worker::spawn(|_| panic!("injected residency failure"));
    wait(|| worker.exchange(request(1), None).is_some());
    wait(|| worker.0.mailbox.lock().unwrap().failed);
    let error = std::panic::catch_unwind(|| worker.exchange(request(2), None));
    assert!(error.is_err());
}

fn params(eye: glam::DVec3) -> Params {
    let v = super::super::tests::view(eye);
    let mut p = Params::zeroed();
    let origin = crate::world::render_origin(eye);
    for a in 0..3 {
        p.origin[a] = origin[a];
        p.fraction[a] = (eye[a] * 10.0 - f64::from(origin[a])) as f32;
        p.forward[a] = v.forward[a] as f32;
        p.up[a] = v.up[a] as f32;
        p.right[a] = v.right[a] as f32;
    }
    p.up[3] = v.tan as f32;
    p.right[3] = v.aspect as f32;
    p.screen[1] = v.height as f32;
    p
}

#[cfg(not(feature = "regional-publication-experiment"))]
#[test]
fn ordered_packets_keep_visible_payloads_and_source_grids_coherent_during_replacement() {
    use super::super::{Key, BRICK, BRICK_CAPACITY, SOLID};
    use glam::DVec3;
    use rustc_hash::FxHashSet;
    struct Consumer {
        pipeline: Pipeline,
        payloads: Vec<Option<(Key, Arc<World>)>>,
        visible: FxHashSet<usize>,
        publications: usize,
        last_sequence: u64,
    }
    impl Consumer {
        fn frame(&mut self, world: &Arc<World>, p: &Params) {
            if let Some(mut prepared) = self.pipeline.prepare(world, p) {
                assert_eq!(prepared.sequence, self.last_sequence + 1);
                self.last_sequence = prepared.sequence;
                if let Some((jobs, references, source)) = prepared.batch.take() {
                    assert!(jobs.len() <= super::super::GENERATION_BATCH);
                    for job in jobs {
                        assert!(
                            !self.visible.contains(&(job.slot as usize)),
                            "overwrote visible slot"
                        );
                        let range = job.pad[0] as usize..(job.pad[0] + job.pad[1]) as usize;
                        assert!(references[range]
                            .iter()
                            .all(|&i| (i as usize) < source.edits.len()));
                        self.payloads[job.slot as usize] = Some((
                            Key {
                                low: job.low,
                                level: job.level,
                            },
                            source.clone(),
                        ));
                    }
                }
                if let Some(links) = prepared.publication.take() {
                    let (source, _) = prepared.active.as_ref().unwrap();
                    let mut pending = vec![(0, std::array::from_fn(|a| links[a] as i32), links[3])];
                    self.visible.clear();
                    while let Some((index, low, level)) = pending.pop() {
                        let child = links[index + 4];
                        if child >= SOLID {
                            continue;
                        }
                        if child & BRICK != 0 {
                            let slot = (child & 0xffff) as usize;
                            assert!(
                                self.visible.insert(slot),
                                "duplicate payload in a complete cut"
                            );
                            let (key, generated) = self.payloads[slot]
                                .as_ref()
                                .expect("ungenerated published slot");
                            assert_eq!(*key, Key { low, level });
                            assert_eq!(generated.voxel_step(), source.voxel_step());
                            // Probe each old/new brush's nearest point in this
                            // payload as well as distributed corners. This
                            // catches an edit confined to a brick's interior.
                            for edit in source.edits.iter().chain(&generated.edits) {
                                let cell = std::array::from_fn(|a| {
                                    edit.cell[a].clamp(low[a], low[a] + key.side() - 1)
                                });
                                if edit.contains(source.sample_cell(cell)) {
                                    let material = |w: &World| {
                                        w.latest_edit(cell).map_or_else(
                                            || crate::world::base_material(w.sample_cell(cell)),
                                            |i| w.edits[i].material,
                                        )
                                    };
                                    assert_eq!(
                                        material(generated),
                                        material(source),
                                        "stale interior edit"
                                    );
                                }
                            }
                            if index % 127 == 0 {
                                for octant in 0..8 {
                                    let cell = std::array::from_fn(|a| {
                                        low[a] + ((octant >> a) & 1) * (key.side() - 1)
                                    });
                                    let material = |w: &World| {
                                        w.latest_edit(cell).map_or_else(
                                            || crate::world::base_material(w.sample_cell(cell)),
                                            |i| w.edits[i].material,
                                        )
                                    };
                                    assert_eq!(
                                        material(generated),
                                        material(source),
                                        "stale source payload"
                                    );
                                }
                            }
                        } else {
                            assert!(level > 0);
                            for octant in 0..8usize {
                                pending.push((
                                    child as usize + octant,
                                    std::array::from_fn(|a| {
                                        low[a] + (((octant >> a) & 1) as i32) * (16 << level)
                                    }),
                                    level - 1,
                                ));
                            }
                        }
                    }
                    self.publications += 1;
                }
                self.pipeline.accept_after_encode(prepared);
            }
            self.pipeline.finish_frame();
        }
        fn settle(&mut self, world: &Arc<World>, p: &Params) {
            wait(|| {
                self.frame(world, p);
                let s = self.pipeline.stats;
                s.ready && !s.refining && !s.planning && s.pending == 0
            });
            assert!(Arc::ptr_eq(
                &self.pipeline.active.as_ref().unwrap().0,
                world
            ));
            assert_eq!(self.pipeline.active_voxel_step(), world.voxel_step());
        }
    }
    let world = Arc::new(World::default());
    let ground = world.ground_spawn(0.0, 0.0, 3.0);
    let mut consumer = Consumer {
        pipeline: Pipeline::configured(BRICK_CAPACITY, true, true),
        payloads: vec![None; BRICK_CAPACITY],
        visible: FxHashSet::default(),
        publications: 0,
        last_sequence: 0,
    };
    consumer.settle(&world, &params(ground + DVec3::Y * 300_000.0));
    wait(|| {
        consumer.frame(&world, &params(ground));
        consumer.pipeline.stats.pending > 256
    });
    let mut replacement = (*world).clone();
    replacement
        .apply_edit(crate::world::Edit {
            cell: crate::world::cell_of(ground),
            radius: 4.0,
            material: 0,
        })
        .unwrap();
    replacement.set_voxel_size(1.0).unwrap();
    let replacement = Arc::new(replacement);
    for i in 0..8 {
        consumer.frame(
            &replacement,
            &params(ground + DVec3::X * (i as f64 * 100.0)),
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    consumer.settle(&replacement, &params(ground + DVec3::X * 100.0));
    // Reverse, undo and restore the authored grid. An older prepared bundle can
    // be consumed first, but it must retain its own visible source/grid.
    consumer.settle(&world, &params(ground - DVec3::X * 100.0));
    let mut resized = params(ground);
    resized.screen[1] = 1080.0;
    resized.up[3] *= 0.8;
    consumer.settle(&world, &resized);
    let mut painted = (*world).clone();
    painted
        .apply_edit(crate::world::Edit {
            cell: crate::world::cell_of(ground),
            radius: 4.0,
            material: 3,
        })
        .unwrap();
    consumer.settle(&Arc::new(painted), &resized); // same-grid source change
    consumer.settle(&world, &resized); // same-grid undo
    assert!(consumer.publications >= 6);
    assert!(consumer.pipeline.stats.cancelled_jobs > 0);
    assert!(consumer.pipeline.stats.pipeline_misses > 0);
    let worker = match &consumer.pipeline.producer {
        Producer::Worker(w) => Arc::downgrade(&w.0),
        _ => unreachable!(),
    };
    drop(consumer);
    wait(|| worker.strong_count() == 0);
}
