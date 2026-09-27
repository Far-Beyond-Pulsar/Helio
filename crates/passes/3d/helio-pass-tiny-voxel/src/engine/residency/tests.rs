use super::*;

#[test]
fn cancelled_selection_preserves_reusable_classification() {
    let world = Arc::new(World::default());
    let camera = view(world.ground_spawn(0.0, 0.0, 3.0));
    let mut cache = SelectionCache::default();
    let checks = std::cell::Cell::new(0);
    let result = build_cancellable(world.clone(), camera, 31_744, &mut cache, || {
        checks.set(checks.get() + 1);
        checks.get() > 32
    });
    assert!(result.is_none());
    assert!(cache.classified > 0 && cache.classified <= 32);
    let resumed = build(world.clone(), camera, 31_744, &mut cache);
    let fresh = build(world, camera, 31_744, &mut SelectionCache::default());
    assert_eq!(resumed.nodes, fresh.nodes);
    assert_eq!(resumed.leaves, fresh.leaves);
}

#[test]
fn interrupted_edit_invalidation_does_not_retag_old_classifications() {
    let world = World::default();
    let cell = crate::world::cell_of(world.ground_spawn(0.0, 0.0, 0.0));
    let keys: Vec<_> = (0..128)
        .map(|x| Key {
            low: [cell[0] + x * 32, cell[1], cell[2]],
            level: 0,
        })
        .collect();
    let mut cache = SelectionCache::default();
    cache.reconcile(&world);
    for &key in &keys {
        cache.classify(&world, key);
    }
    let mut edited = world.clone();
    edited
        .apply_edit(Edit {
            cell,
            radius: 16.0,
            material: 0,
        })
        .unwrap();
    let calls = std::cell::Cell::new(0);
    assert!(!cache.reconcile_cancellable(&edited, &|| {
        calls.set(calls.get() + 1);
        calls.get() > 3
    }));
    assert!(cache.edits.is_empty());
    cache.reconcile(&edited);
    for key in keys {
        assert_eq!(cache.classify(&edited, key), classify(&edited, key));
    }
}

#[test]
fn selection_mailbox_delivers_only_latest_demand_and_shuts_down() {
    let worker = selection::Worker::new(31_744);
    let world = Arc::new(World::default());
    let eye = world.ground_spawn(0.0, 0.0, 3.0);
    let mut serial = 0;
    for offset in [300_000.0, 200.0, 1_000.0, 0.0, 50.0, 4.0] {
        serial = worker.submit(world.clone(), view(eye + DVec3::Y * offset));
    }
    let start = std::time::Instant::now();
    loop {
        if let Some((received, plan)) = worker.take() {
            assert_eq!(received, serial);
            assert_eq!(plan.view.eye, eye + DVec3::Y * 4.0);
            assert!(Arc::ptr_eq(&plan.world, &world));
            break;
        }
        assert!(start.elapsed().as_secs() < 20, "latest selection timed out");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    // Drop while another selection can be running; the worker must release its
    // strong ownership after observing shutdown, rather than wait on a sender.
    let weak = worker.lifetime_probe();
    worker.submit(world, view(eye + DVec3::Y * 300_000.0));
    drop(worker);
    while weak() {
        assert!(start.elapsed().as_secs() < 20, "selection worker leaked");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

#[cfg(not(feature = "regional-publication-experiment"))]
#[test]
fn interrupting_generation_keeps_visible_slots_and_only_reuses_generated_payloads() {
    use bytemuck::Zeroable;
    fn params(eye: DVec3) -> Params {
        let v = view(eye);
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
    fn audit(r: &Residency) {
        let free: HashSet<_> = r.free.iter().copied().collect();
        assert_eq!(free.len(), r.free.len(), "slot was freed twice");
        for (slot, key) in r.occupied.iter().enumerate() {
            assert_eq!(free.contains(&slot), key.is_none());
            if r.active_slots.contains(&slot) {
                assert!(key.is_some());
            }
        }
        if let Some(pending) = &r.pending {
            assert!(
                pending.jobs[pending.cursor..]
                    .iter()
                    .all(|j| j.slot == u32::MAX),
                "queued jobs reserved physical storage"
            );
        }
        let retired: HashSet<_> = r.retired_slots.iter().copied().collect();
        assert_eq!(retired.len(), r.retired_slots.len(), "slot retired twice");
        for slot in &retired {
            assert!(
                r.active_slots.contains(slot),
                "retired slot is no longer pinned"
            );
            let key = r.occupied[*slot].unwrap();
            assert!(r.entries.get(&key).is_none_or(|e| e.slot != *slot));
        }
        for (key, entry) in &r.entries {
            assert_eq!(r.occupied[entry.slot], Some(*key));
        }
        for (slot, key) in r.occupied.iter().enumerate() {
            if let Some(key) = key {
                assert!(
                    retired.contains(&slot) || r.entries.get(key).is_some_and(|e| e.slot == slot),
                    "occupied slot has no owner"
                );
            }
        }
    }
    fn generate(r: &mut Residency, payloads: &mut [Option<(Key, Arc<World>)>]) {
        if let Some((jobs, _, world)) = r.next_batch() {
            for job in jobs {
                assert!(
                    !r.active_slots.contains(&(job.slot as usize)),
                    "overwrote visible payload"
                );
                payloads[job.slot as usize] = Some((
                    Key {
                        low: job.low,
                        level: job.level,
                    },
                    world.clone(),
                ));
            }
        }
    }
    fn settle(
        r: &mut Residency,
        world: &Arc<World>,
        p: &Params,
        payloads: &mut [Option<(Key, Arc<World>)>],
    ) {
        let start = std::time::Instant::now();
        let mut batches = 0;
        loop {
            r.update(world, p);
            generate(r, payloads);
            if let Some(nodes) = r.publish() {
                for node in nodes
                    .iter()
                    .filter(|n| n.child & BRICK != 0 && n.child < SOLID)
                {
                    let (key, source) = payloads[(node.child & 0xffff) as usize]
                        .as_ref()
                        .expect("published ungenerated slot");
                    assert_eq!(
                        *key,
                        Key {
                            low: node.low,
                            level: node.level
                        }
                    );
                    assert_eq!(source.voxel_step(), world.voxel_step());
                    for octant in 0..8 {
                        let cell = std::array::from_fn(|a| {
                            key.low[a] + ((octant >> a) & 1) * (key.side() - 1)
                        });
                        let material = |w: &World| {
                            w.latest_edit(cell).map_or_else(
                                || crate::world::base_material(w.sample_cell(cell)),
                                |i| w.edits[i].material,
                            )
                        };
                        assert_eq!(
                            material(source),
                            material(world),
                            "reused stale edit payload"
                        );
                    }
                }
            }
            batches += 1;
            if batches % 32 == 0 || r.pending.is_none() {
                audit(r);
            }
            if r.stats.ready && !r.stats.refining && !r.stats.planning && r.pending.is_none() {
                break;
            }
            assert!(
                start.elapsed().as_secs() < 30,
                "residency convergence timed out"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
    let world = Arc::new(World::default());
    let ground = world.ground_spawn(0.0, 0.0, 3.0);
    let mut r = Residency::new(BRICK_CAPACITY);
    r.retargeting = true;
    let mut payloads = vec![None; BRICK_CAPACITY];
    settle(
        &mut r,
        &world,
        &params(ground + DVec3::Y * 300_000.0),
        &mut payloads,
    );
    let visible = r.active_slots.clone();
    let start = std::time::Instant::now();
    let slots_before = r.occupied.iter().filter(|key| key.is_some()).count();
    while r.pending.is_none() {
        r.update(&world, &params(ground));
        assert!(start.elapsed().as_secs() < 20);
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(
        r.occupied.iter().filter(|key| key.is_some()).count() <= slots_before,
        "plan installation allocated ungenerated payloads"
    );
    generate(&mut r, &mut payloads);
    assert!(r.pending.as_ref().is_some_and(|p| p.cursor < p.jobs.len()));
    let mut edit = (*world).clone();
    edit.apply_edit(Edit {
        cell: crate::world::cell_of(ground),
        radius: 4.0,
        material: 0,
    })
    .unwrap();
    edit.set_voxel_size(1.0).unwrap();
    let edit = Arc::new(edit);
    let arrival = params(ground + DVec3::X * 100.0);
    r.update(&edit, &arrival);
    assert!(r.stats.cancelled_jobs > 0);
    assert_eq!(
        r.active_slots, visible,
        "cancellation changed visible ownership"
    );
    audit(&r);
    settle(&mut r, &edit, &arrival, &mut payloads);
    settle(&mut r, &world, &params(ground), &mut payloads); // undo and grid restoration
    assert!(Arc::ptr_eq(r.active_world.as_ref().unwrap(), &world));
    let generated = r.stats.generated;
    for step in 1..=4 {
        r.update(&world, &params(ground + DVec3::X * (step as f64 * 8.0)));
        let start = std::time::Instant::now();
        while !r.selector.has_result() {
            assert!(start.elapsed().as_secs() < 20);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        // A completed plan must contribute even when the next frame already
        // has a newer camera pose. New demand must not erase all useful work.
        r.update(
            &world,
            &params(ground + DVec3::X * ((step + 1) as f64 * 8.0)),
        );
        generate(&mut r, &mut payloads);
        r.publish();
        audit(&r);
    }
    assert!(
        r.stats.generated > generated,
        "continuous motion starved generation"
    );
    settle(&mut r, &world, &params(ground), &mut payloads);
}

#[test]
fn generation_budget_counts_evaluations_without_exceeding_dispatch_capacity() {
    let world = Arc::new(World::default());
    for (level, ready, expected) in [(0, true, 64), (4, true, 256), (0, false, 256)] {
        let mut residency = Residency::new(BRICK_CAPACITY);
        residency.stats.ready = ready;
        let mut nodes = vec![Node {
            low: [0; 3],
            level: level + 3,
            child: 0,
        }];
        let mut leaves = Vec::new();
        let mut jobs = Vec::new();
        let mut job_nodes = Vec::new();
        let mut index = 0;
        while index < nodes.len() {
            let n = nodes[index];
            if n.level == level {
                let slot = jobs.len() as u32;
                nodes[index].child = BRICK | slot;
                leaves.push((index, Key { low: n.low, level }));
                jobs.push(Job {
                    low: n.low,
                    level,
                    slot,
                    pad: [0; 3],
                });
                job_nodes.push(index);
            } else {
                nodes[index].child = nodes.len() as u32;
                for octant in 0..8 {
                    nodes.push(Node {
                        low: std::array::from_fn(|a| {
                            n.low[a] + ((octant >> a) & 1) * (16 << n.level)
                        }),
                        level: n.level - 1,
                        child: 0,
                    });
                }
            }
            index += 1;
        }
        residency.pending = Some(Pending::new(
            Plan {
                nodes,
                leaves,
                world: world.clone(),
                view: view(DVec3::ZERO),
                pixels: 1.0,
            },
            jobs,
            job_nodes,
            Arc::new(world.edits.clone()),
        ));
        let (batch, _, _) = residency.next_batch().unwrap();
        assert_eq!(batch.len(), expected);
        assert_eq!(residency.stats.pending, 512 - expected);
    }
}

pub(super) fn view(eye: DVec3) -> View {
    let forward = DVec3::new(0.0, -0.15, -1.0).normalize();
    let right = forward.cross(DVec3::Y).normalize();
    View {
        eye,
        forward,
        right,
        up: right.cross(forward),
        tan: 0.41421356,
        aspect: 16.0 / 9.0,
        height: 720.0,
    }
}

#[test]
fn cached_classification_matches_fresh_after_append_undo_and_replacement() {
    let mut world = World::default();
    let cell = crate::world::cell_of(world.ground_spawn(0.0, 0.0, 0.0));
    let mut cache = SelectionCache::default();
    let keys: Vec<_> = (0..16)
        .flat_map(|level| {
            let side = 32 << level;
            let low = cell.map(|v| v.div_euclid(side) * side);
            [-1, 0, 1].into_iter().map(move |offset| Key {
                low: [low[0] + offset * side, low[1], low[2]],
                level,
            })
        })
        .collect();
    let original = world.clone();
    for step in 0..4 {
        match step {
            1 => world
                .apply_edit(Edit {
                    cell,
                    radius: 8.0,
                    material: 0,
                })
                .unwrap(),
            2 => world = original.clone(),
            3 => world
                .apply_edit(Edit {
                    cell,
                    radius: 32.0,
                    material: 3,
                })
                .unwrap(),
            _ => {}
        }
        cache.reconcile(&world);
        for &key in &keys {
            assert_eq!(
                cache.classify(&world, key),
                classify(&world, key),
                "step={step} key={key:?}"
            );
        }
        cache.reconcile(&world);
        for &key in &keys {
            cache.classify(&world, key);
        }
        assert_eq!(cache.classified, 0);
        assert_eq!(cache.reused, keys.len());
    }
}

#[test]
fn zoom_and_roll_require_new_selection() {
    let reference = view(DVec3::ZERO);
    let mut zoomed = reference;
    zoomed.tan *= 0.5;
    assert!(reference.changed(zoomed));
    let mut rolled = reference;
    rolled.up = reference.right;
    rolled.right = -reference.up;
    assert!(reference.changed(rolled));
    assert!(!reference.changed(reference));
}

#[test]
#[ignore = "CPU selection benchmark; reports cold and cached identical cuts"]
fn selection_flight_benchmark() {
    let world = Arc::new(World::default());
    let ground = world.ground_spawn(0.0, 0.0, 3.0);
    let mut cache = SelectionCache::default();
    for (index, offset) in [
        DVec3::ZERO,
        DVec3::X * 2.0,
        DVec3::X * 4.0,
        DVec3::Y * 200.0,
        DVec3::Y * 1_000.0,
        DVec3::Y * 300_000.0,
        DVec3::ZERO,
    ]
    .into_iter()
    .enumerate()
    {
        let camera = view(ground + offset);
        let start = std::time::Instant::now();
        let cold = build(
            world.clone(),
            camera,
            31_744,
            &mut SelectionCache::default(),
        );
        let cold_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = std::time::Instant::now();
        let warm = build(world.clone(), camera, 31_744, &mut cache);
        let cached_ms = start.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(cold.leaves, warm.leaves);
        assert_eq!(cold.pixels, warm.pixels);
        assert_eq!(cold.nodes.len(), warm.nodes.len());
        for (a, b) in cold.nodes.iter().zip(&warm.nodes) {
            assert_eq!((a.low, a.level, a.child), (b.low, b.level, b.child));
        }
        eprintln!("SELECTION_FLIGHT step={index} cold_ms={cold_ms:.3} cached_ms={cached_ms:.3} nodes={} bricks={} classified={} reused={}",
            warm.nodes.len(), warm.leaves.len(), cache.classified, cache.reused);
    }
}
