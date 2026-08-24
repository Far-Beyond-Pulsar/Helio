//! Adversarial / leak-tier tests for the whole content-dedup stack:
//! [`helio_component::content_ledger::ContentLedger`] (production ledger),
//! the SceneDB counted-handle plumbing driving it through real `World`s,
//! and the shared-parse layer (`subsystems::load_mesh_upload_shared`) --
//! trying hard to BREAK all three: exact drop accounting with leak-
//! counting payloads, churn storms, every removal ordering, rewrite
//! storms, rehydrate loops, double releases, zero-id edges, seeded
//! pseudo-random op storms differential-checked against a naive model,
//! and threaded hammering on overlapping and disjoint id sets.
//!
//! Isolation note: every test constructs its OWN `ContentLedger` rather
//! than the process-global `shared_content_ledger()` -- the global's
//! drop-callback slot is single-registration by design, so claiming it
//! here would couple tests arbitrarily. End-to-end wiring THROUGH the
//! global is covered separately (see `global_ledger_evicts_parse_cache`
//! below), which restores nothing afterwards because eviction semantics
//! don't change when the callback slot goes to the built-in eviction.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use helio_component::content_ledger::ContentLedger;
use helio_component::subsystems::{clear_shared_mesh_cache, load_mesh_upload_shared};
use pulsar_scenedb::handle_ledger::{HandleId, HandleLedger};
use pulsar_scenedb::{Entity, SceneStore, World};

/// The `StaticMeshComponent` shape (var-len `#[gpu]` fields + one
/// `HandleId` field), as a self-contained fixture: same codegen routes,
/// none of the editor machinery.
#[derive(Clone, Debug, SceneStore)]
struct DedupRow {
    content_id: HandleId,
    #[gpu]
    verts: Vec<u32>,
}

fn row(content: u128) -> DedupRow {
    DedupRow {
        content_id: HandleId(content),
        verts: vec![content as u32; 3],
    }
}

fn wired_world(ledger: &Arc<ContentLedger>) -> World {
    let mut world = World::new();
    world.attach_handle_ledger(Arc::clone(ledger) as Arc<dyn HandleLedger>);
    world
}

// ── Exact-drop-accounting harness ────────────────────────────────────────

/// A stand-in for the parse-cache/GPU-residency state behind an id: counts
/// its own destruction. This is the leak detector -- if the stack ever
/// fires a transition twice or loses one, these counters expose it.
#[derive(Default)]
struct DropCounts {
    /// id → how many times the LAST reference died. Every entry must end
    /// the suite at exactly the number of times its lifecycle completed.
    fires: Mutex<HashMap<u128, usize>>,
}

impl DropCounts {
    fn record(&self, id: u128) {
        *self.fires.lock().unwrap().entry(id).or_insert(0) += 1;
    }

    fn get(&self, id: u128) -> usize {
        self.fires.lock().unwrap().get(&id).copied().unwrap_or(0)
    }
}

fn ledger_with_counter(counter: Arc<DropCounts>) -> Arc<ContentLedger> {
    let ledger = Arc::new(ContentLedger::new());
    assert!(
        ledger.set_drop_callback(Box::new(move |c| counter.record(c.0))),
        "callback slot already taken on a fresh ledger -- construction bug"
    );
    ledger
}

// ── Tier B: adversarial ──────────────────────────────────────────────────

#[test]
fn last_reference_out_frees_exactly_once_end_to_end() {
    // Two entities, ONE content. Both die. Exactly one 1→0 transition, one
    // callback fire, zero residual count. The whole product promise in one
    // test.
    let counter = Arc::new(DropCounts::default());
    let ledger = ledger_with_counter(Arc::clone(&counter));
    let mut world = wired_world(&ledger);

    let a = world.spawn();
    let b = world.spawn();
    world.insert(a, row(4242));
    world.insert(b, row(4242));
    assert_eq!(ledger.strong_count(HandleId(4242)), 2);

    world.despawn(a);
    assert_eq!(counter.get(4242), 0, "still referenced -- must NOT free");
    assert_eq!(ledger.strong_count(HandleId(4242)), 1);

    world.despawn(b);
    assert_eq!(
        counter.get(4242),
        1,
        "exactly one free, on the true last unref"
    );
    assert_eq!(ledger.strong_count(HandleId(4242)), 0);
}

#[test]
fn churn_storm_thousands_of_cycles_on_one_entity_net_to_zero() {
    let counter = Arc::new(DropCounts::default());
    let ledger = ledger_with_counter(Arc::clone(&counter));
    let mut world = wired_world(&ledger);
    let e = world.spawn();

    const CYCLES: u32 = 3_000;
    for i in 0..CYCLES {
        let content = (i % 16) as u128 + 1;
        world.insert(e, row(content)); // swap-or-acquire, both shapes occur
        if i % 3 == 0 {
            world.remove::<DedupRow>(e);
            world.insert(e, row(content)); // remove+reinsert = full cycle
        }
    }
    // Drain whatever lives.
    world.despawn(e);

    // Sharper assertion than per-id sums: TOTAL callback fires across all
    // contents must equal TOTAL acquires (nothing escaped, nothing doubled).
    let total_fires: usize = counter.fires.lock().unwrap().values().sum();
    let total_acquires: u32 = (0..CYCLES).map(|i| if i % 3 == 0 { 2 } else { 1 }).sum();
    assert_eq!(
        total_fires, total_acquires as usize,
        "fires must eventually equal acquires after full drain"
    );
}

#[test]
fn rewrite_storm_between_two_assets_only_final_survivor_holds() {
    let counter = Arc::new(DropCounts::default());
    let ledger = ledger_with_counter(Arc::clone(&counter));
    let mut world = wired_world(&ledger);
    let e = world.spawn();

    world.insert(e, row(1));
    const N: usize = 4_000;
    let mut prev = 1u128;
    // Deaths are CUMULATIVE per content: an asset swapped out dies once,
    // and if it's later re-acquired and swapped out again it dies AGAIN.
    // Track the exact expectation instead of assuming "holder never died".
    let mut deaths = [0usize; 3];
    for i in 0..N {
        let next = if i % 2 == 0 { 2u128 } else { 1u128 };
        world.insert(e, row(next)); // pure swaps: old released, new acquired

        if next != prev {
            deaths[prev as usize] += 1;
        }
        let l1 = ledger.strong_count(HandleId(1));
        let l2 = ledger.strong_count(HandleId(2));
        assert_eq!(l1 + l2, 1, "flip {i}: {l1}+{l2} != 1 live refs");
        assert_eq!(
            counter.get(1),
            deaths[1],
            "flip {i}: content 1 deaths drifted"
        );
        assert_eq!(
            counter.get(2),
            deaths[2],
            "flip {i}: content 2 deaths drifted"
        );
        prev = next;
    }

    // Final drop: the survivor dies too.
    world.despawn(e);
    deaths[prev as usize] += 1;
    assert_eq!(counter.get(1), deaths[1]);
    assert_eq!(counter.get(2), deaths[2]);
    assert_eq!(ledger.strong_count(HandleId(1)), 0);
    assert_eq!(ledger.strong_count(HandleId(2)), 0);
}

#[test]
fn repeated_rehydrate_of_same_entity_never_double_acquires() {
    let counter = Arc::new(DropCounts::default());
    let ledger = ledger_with_counter(Arc::clone(&counter));
    let mut world = wired_world(&ledger);
    let e = world.spawn();

    for _ in 0..50 {
        world.insert(e, row(777)); // identical content: swap rule ⇒ full no-op
    }
    assert_eq!(
        ledger.strong_count(HandleId(777)),
        1,
        "49 no-op rewrites changed nothing"
    );
    assert_eq!(counter.get(777), 0);

    world.despawn(e);
    assert_eq!(counter.get(777), 1);
}

#[test]
fn interlocked_share_then_remove_all_orders_of_three_entities() {
    // Entities acquire the SAME content, then die in each of the 6 possible
    // relative orders. Counts step down 3→2→1→0; the free fires ONLY at the
    // end, regardless of order.
    for order in [
        [0usize, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let counter = Arc::new(DropCounts::default());
        let ledger = ledger_with_counter(Arc::clone(&counter));
        let mut world = wired_world(&ledger);
        let ents: Vec<Entity> = (0..3).map(|_| world.spawn()).collect();
        for &e in ents.iter() {
            world.insert(e, row(31337));
        }
        assert_eq!(ledger.strong_count(HandleId(31337)), 3, "order {order:?}");

        for (step, &idx) in order.iter().enumerate() {
            world.despawn(ents[idx]);
            assert_eq!(
                ledger.strong_count(HandleId(31337)),
                (3 - step - 1) as i64,
                "order {order:?} step {step}"
            );
            let expected_fires = if step == 2 { 1 } else { 0 };
            assert_eq!(
                counter.get(31337),
                expected_fires,
                "order {order:?} step {step}"
            );
        }
    }
}

#[test]
fn despawn_with_everything_still_pending_nets_to_zero() {
    // Handles have no deferred flush (events are immediate), so the
    // adversarial reading of "despawn while flush pending" is maximum
    // temporal locality: insert→despawn back-to-back, plus mid-flight
    // swaps, thousands of times, asserting perfect netting throughout.
    let counter = Arc::new(DropCounts::default());
    let ledger = ledger_with_counter(Arc::clone(&counter));
    let mut world = wired_world(&ledger);

    for round in 1..=1_000u128 {
        let e = world.spawn();
        world.insert(e, row(round));
        world.insert(e, row(round + 1)); // immediate swap on top
        world.despawn(e);
        assert_eq!(ledger.strong_count(HandleId(round)), 0, "round {round}");
        assert_eq!(ledger.strong_count(HandleId(round + 1)), 0, "round {round}");
    }
    let total: usize = counter.fires.lock().unwrap().values().sum();
    assert_eq!(
        total, 2_000,
        "each round frees exactly two distinct contents"
    );
}

#[test]
fn double_release_and_release_after_drop_behave_per_build_profile() {
    let counter = Arc::new(DropCounts::default());
    let ledger = ledger_with_counter(Arc::clone(&counter));
    ledger.acquire(HandleId(5150));

    let double_release = || {
        ledger.release(HandleId(5150)); // legit 1→0
        ledger.release(HandleId(5150)); // DOUBLE: unknown id now
    };

    if cfg!(debug_assertions) {
        // Debug builds assert loudly on unknown releases (spec'd behavior)
        // -- the panic must be contained to the caller's frame.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(double_release));
        assert!(
            result.is_err(),
            "debug builds must assert on double-release"
        );
        // State survived: entry gone, no phantom resurrection.
        assert_eq!(ledger.strong_count(HandleId(5150)), 0);
    } else {
        // Release builds: warn + ignore, never wrap, never UB.
        double_release();
        assert_eq!(ledger.strong_count(HandleId(5150)), 0);
    }
    assert_eq!(
        counter.get(5150),
        1,
        "exactly one free despite the extra release"
    );
}

#[test]
fn zero_ids_are_invisible_to_the_entire_stack() {
    let counter = Arc::new(DropCounts::default());
    let ledger = ledger_with_counter(Arc::clone(&counter));
    let mut world = wired_world(&ledger);

    let e = world.spawn();
    // All-zero handles (failed-load hydration shape): no events anywhere.
    world.insert(
        e,
        DedupRow {
            content_id: HandleId::ZERO,
            verts: Vec::new(),
        },
    );
    assert_eq!(ledger.strong_count(HandleId::ZERO), 0);

    // Swap from zero to real: pure acquire.
    world.insert(e, row(64));
    assert_eq!(ledger.strong_count(HandleId(64)), 1);

    // Swap back to zero: pure release.
    world.insert(
        e,
        DedupRow {
            content_id: HandleId::ZERO,
            verts: Vec::new(),
        },
    );
    assert_eq!(ledger.strong_count(HandleId(64)), 0);
    assert_eq!(counter.get(64), 1);

    // Zero-length payload with a real id is still a perfectly good
    // reference (an empty mesh IS content).
    world.insert(
        e,
        DedupRow {
            content_id: HandleId(65),
            verts: Vec::new(),
        },
    );
    assert_eq!(ledger.strong_count(HandleId(65)), 1);
    world.despawn(e);
    assert_eq!(counter.get(65), 1);
}

#[test]
fn get_mut_handle_writes_swap_through_the_real_production_ledger() {
    let counter = Arc::new(DropCounts::default());
    let ledger = ledger_with_counter(Arc::clone(&counter));
    let mut world = wired_world(&ledger);
    let e = world.spawn();

    world.insert(e, row(11));
    {
        let mut guard = world.get_mut::<DedupRow>(e).unwrap();
        guard.content_id = HandleId(12);
    }
    assert_eq!(ledger.strong_count(HandleId(11)), 0);
    assert_eq!(ledger.strong_count(HandleId(12)), 1);

    // Borrow-only guard: silence.
    {
        let guard = world.get_mut::<DedupRow>(e).unwrap();
        assert_eq!(guard.content_id, HandleId(12));
    }
    assert_eq!(
        ledger.strong_count(HandleId(12)),
        1,
        "borrow-only get_mut must not churn"
    );

    world.despawn(e);
    assert_eq!(counter.get(12), 1);
}

// ── Seeded differential storm ────────────────────────────────────────────

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Naive multiset model: what ContentLedger's counts MUST be after any op
/// sequence. Deliberately dumb -- its trustworthiness IS its simplicity.
#[derive(Default)]
struct Model {
    counts: HashMap<u128, i64>,
}

impl Model {
    fn apply(&mut self, op: OpKind, id: u128) {
        match op {
            OpKind::Acquire => *self.counts.entry(id).or_insert(0) += 1,
            OpKind::Release => *self.counts.entry(id).or_insert(0) -= 1,
        }
    }
}

#[derive(Clone, Copy)]
enum OpKind {
    Acquire,
    Release,
}

#[test]
fn seeded_op_storm_differential_matches_model_on_every_step() {
    const OPS: u64 = 10_000;
    const POOL: usize = 10;

    for seed in [0xFEED_u64, 0xBEEF_CAFE, 0x1234_5678_9ABC_DEF0] {
        let counter = Arc::new(DropCounts::default());
        let ledger = ledger_with_counter(Arc::clone(&counter));
        let mut world = wired_world(&ledger);
        let mut model = Model::default();
        let mut rng = SplitMix64(seed);

        let mut ents: Vec<Entity> = Vec::with_capacity(POOL);
        let mut present: Vec<bool> = Vec::with_capacity(POOL);
        for _ in 0..POOL {
            ents.push(world.spawn());
            present.push(false);
        }

        for step in 0..OPS {
            let slot = rng.below(POOL as u64) as usize;
            match rng.below(10) {
                // 6/10: insert (acquire or swap).
                0..=5 => {
                    let content = rng.below(24) as u128 + 1;
                    if present[slot] {
                        if let Some(cur) = world.get::<DedupRow>(ents[slot]) {
                            model.apply(OpKind::Release, cur.content_id.0);
                        }
                    }
                    world.insert(ents[slot], row(content));
                    model.apply(OpKind::Acquire, content);
                    present[slot] = true;
                }
                // 2/10: remove.
                6..=7 => {
                    if present[slot] {
                        if let Some(cur) = world.get::<DedupRow>(ents[slot]) {
                            model.apply(OpKind::Release, cur.content_id.0);
                        }
                        world.remove::<DedupRow>(ents[slot]);
                        present[slot] = false;
                    }
                }
                // 2/10: despawn + respawn slot.
                _ => {
                    if present[slot] {
                        if let Some(cur) = world.get::<DedupRow>(ents[slot]) {
                            model.apply(OpKind::Release, cur.content_id.0);
                        }
                    }
                    world.despawn(ents[slot]);
                    ents[slot] = world.spawn();
                    present[slot] = false;
                }
            }

            // Compare EVERY count after EVERY op (intermediate states, not
            // just the tail -- cancelling bugs are still bugs).
            for (&id, &m) in model.counts.iter() {
                let l = ledger.strong_count(HandleId(id));
                assert_eq!(
                    l, m,
                    "seed {seed:#x} step {step}: content {id} ledger={l} model={m}"
                );
            }
            // And the production audit API must agree with both.
            let live: Vec<(HandleId, u32)> = model
                .counts
                .iter()
                .filter(|(_, c)| **c > 0)
                .map(|(&id, &c)| (HandleId(id), c as u32))
                .collect();
            if let Err(mismatches) = ledger.audit(live) {
                panic!("seed {seed:#x} step {step}: audit found {mismatches:?}");
            }
        }

        // Full drain: everything nets to exactly zero; total fires ==
        // total acquires.
        for (slot, &e) in ents.iter().enumerate() {
            if present[slot] {
                if let Some(cur) = world.get::<DedupRow>(e) {
                    model.apply(OpKind::Release, cur.content_id.0);
                }
            }
            world.despawn(e);
        }
        for (_, &c) in model.counts.iter() {
            // Model says everything should have been released; ledger agrees
            // by construction of the drain above (verified via audit there).
            assert_eq!(c, 0, "seed {seed:#x}: model did not drain to zero");
        }
        for (&id, &m) in model.counts.iter() {
            assert_eq!(ledger.strong_count(HandleId(id)), m);
        }
    }
}

// ── Concurrency ──────────────────────────────────────────────────────────

#[test]
fn threaded_hammer_overlapping_and_disjoint_sets_stays_exact() {
    let counter = Arc::new(DropCounts::default());
    let ledger = ledger_with_counter(Arc::clone(&counter));
    const THREADS: usize = 8;
    const OPS_PER_THREAD: usize = 20_000;

    let barrier = Arc::new(std::sync::Barrier::new(THREADS));
    let mut joins = Vec::new();
    for t in 0..THREADS {
        let ledger = Arc::clone(&ledger);
        let barrier = Arc::clone(&barrier);
        joins.push(std::thread::spawn(move || {
            let hot = t < THREADS / 2; // half contend on one window...
            barrier.wait();
            for i in 0..OPS_PER_THREAD {
                let id = if hot {
                    HandleId((i % 32) as u128 + 900)
                } else {
                    // ...half own private ranges (zero contention).
                    HandleId(1_000_000 + (t as u128) * 500_000 + (i % 128) as u128)
                };
                ledger.acquire(id);
                ledger.release_row(&[id]); // despawn-shaped batch path
            }
        }));
    }
    for j in joins {
        j.join().expect("worker panicked");
    }

    // Every hot id: acquired THREADS/2 × (OPS/32) times, released exactly
    // as often ⇒ zero residue. Fire counts are deliberately BOUNDED, not
    // exact: under contention several holders can die in one 1→0 descent,
    // so deaths-per-id is scheduler-dependent. What must hold exactly:
    // no residue anywhere; every touched id died at least once; no id died
    // more times than it was acquired (impossible to fire without a real
    // removal); and global release count == global acquire count.
    let hot_expected_per_id = (THREADS / 2) * (OPS_PER_THREAD / 32);
    for id in 900u128..932 {
        assert_eq!(ledger.strong_count(HandleId(id)), 0, "hot id {id} residue");
        let fires = counter.get(id);
        assert!(fires >= 1, "hot id {id} never died");
        assert!(
            fires <= hot_expected_per_id,
            "hot id {id} fired {fires} > {hot_expected_per_id} acquires -- phantom transition"
        );
    }
    // Disjoint ids: single owner each ⇒ strictly alternating 0↔1 per op ⇒
    // death count per id is EXACTLY its op count. (OPS_PER_THREAD doesn't
    // divide 128 evenly, so compute the exact per-remainder count instead
    // of a sloppy division.)
    for t in (THREADS / 2)..THREADS {
        for k in 0..128u128 {
            let id = 1_000_000 + (t as u128) * 500_000 + k;
            let expected = if (k as usize) < OPS_PER_THREAD % 128 {
                OPS_PER_THREAD / 128 + 1
            } else {
                OPS_PER_THREAD / 128
            };
            assert_eq!(
                ledger.strong_count(HandleId(id)),
                0,
                "disjoint id {id} residue"
            );
            assert_eq!(
                counter.get(id),
                expected,
                "disjoint id {id}: uncontended death count"
            );
        }
    }
}

// ── Shared-parse layer (disk-backed) ─────────────────────────────────────

fn temp_mesh_dir(tag: &str) -> std::path::PathBuf {
    let dir =
        std::env::temp_dir().join(format!("helio-content-dedup-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir); // stale runs
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_mesh(path: &std::path::Path, seed: f32, verts: usize) {
    use helio::PackedVertex;
    let vertices: Vec<PackedVertex> = (0..verts)
        .map(|i| PackedVertex {
            position: [((i as f32) + seed).sin(), ((i as f32) + seed).cos(), seed],
            ..Default::default()
        })
        .collect();
    let indices: Vec<u32> = (0..verts as u32).collect();
    std::fs::write(
        path,
        helio_component::mesh_cache::encode(&helio::MeshUpload { vertices, indices }),
    )
    .unwrap();
}

#[test]
fn same_asset_parses_once_distinct_assets_parse_individually() {
    let dir = temp_mesh_dir("shared");
    let a = dir.join("a.mesh");
    let b = dir.join("b.mesh");
    write_mesh(&a, 1.0, 256);
    write_mesh(&b, 2.0, 256);
    clear_shared_mesh_cache();

    let s1 = load_mesh_upload_shared(&a).unwrap();
    let s2 = load_mesh_upload_shared(&a).unwrap();
    assert!(
        Arc::ptr_eq(&s1.upload, &s2.upload),
        "same path must return the SAME Arc -- that IS the dedup"
    );

    let sb = load_mesh_upload_shared(&b).unwrap();
    assert_ne!(
        s1.content_id, sb.content_id,
        "different content, different identity"
    );

    // Cross-route identity: byte-identical content at another path yields
    // the SAME id (identity is content, not location).
    let twin = dir.join("twin.mesh");
    std::fs::copy(&a, &twin).unwrap();
    clear_shared_mesh_cache(); // force a fresh parse for the twin
    let st = load_mesh_upload_shared(&twin).unwrap();
    assert_eq!(
        s1.content_id, st.content_id,
        "identical content must share identity cross-route"
    );
    assert_ne!(
        Arc::as_ptr(&s1.upload),
        Arc::as_ptr(&st.upload),
        "...while still being independent parses (cache keyed by path)"
    );

    // Staleness: rewriting the file invalidates by mtime/size fast path.
    write_mesh(&a, 99.0, 256);
    let s3 = load_mesh_upload_shared(&a).unwrap();
    assert_ne!(
        s1.content_id, s3.content_id,
        "edited file must produce a fresh parse+identity"
    );

    clear_shared_mesh_cache();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn v1_files_load_and_upgrade_preserving_identity_through_the_shared_layer() {
    use helio_component::mesh_cache::{content_id_for_payload, decode_detailed};

    let dir = temp_mesh_dir("v1upgrade");
    let path = dir.join("legacy.mesh");
    let upload = {
        use helio::PackedVertex;
        let vertices: Vec<PackedVertex> = (0..128)
            .map(|i| PackedVertex {
                position: [i as f32, 0.0, 0.0],
                ..Default::default()
            })
            .collect();
        helio::MeshUpload {
            vertices,
            indices: vec![0, 1, 2],
        }
    };
    // Hand-encode the LEGACY v1 container (the format module keeps its
    // legacy writer private; duplicating ~10 lines here beats widening the
    // public API for one test).
    let mut v1 = Vec::new();
    v1.extend_from_slice(b"PMSH");
    v1.extend_from_slice(&1u32.to_le_bytes());
    v1.extend_from_slice(&(upload.vertices.len() as u64).to_le_bytes());
    v1.extend_from_slice(&(upload.indices.len() as u64).to_le_bytes());
    v1.extend_from_slice(bytemuck::cast_slice(&upload.vertices));
    v1.extend_from_slice(bytemuck::cast_slice(&upload.indices));
    std::fs::write(&path, &v1).unwrap();
    clear_shared_mesh_cache();

    let shared = load_mesh_upload_shared(&path).unwrap();
    let expected = content_id_for_payload(&upload.vertices, &upload.indices);
    assert_eq!(
        shared.content_id, expected,
        "v1 identity computed from payload"
    );

    // The upgrade rewrote the file to v2; a fresh parse reads the DECLARED
    // id, which must equal what v1 computed (identity convergence on disk).
    let reread = std::fs::read(&path).unwrap();
    let decoded = decode_detailed(&reread).expect("upgraded file decodes");
    assert_eq!(
        decoded.source_version, 2,
        "file must have been upgraded to v2"
    );
    assert_eq!(decoded.content_id, expected);
    clear_shared_mesh_cache();
    let again = load_mesh_upload_shared(&path).unwrap();
    assert_eq!(
        again.content_id, shared.content_id,
        "v1→v2 upgrade must not fork identity"
    );

    clear_shared_mesh_cache();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn global_ledger_evicts_parse_cache_when_last_reference_dies() {
    // The one test allowed to drive the PROCESS-GLOBAL ledger: prove the
    // production wiring (ensure_content_ledger_attached + built-in
    // eviction callback) actually frees CPU parse state on last unref --
    // the "today's callback frees the CPU parse-cache entry" contract.
    let dir = temp_mesh_dir("evict");
    let path = dir.join("doomed.mesh");
    write_mesh(&path, 7.0, 128);
    clear_shared_mesh_cache();

    let mut world = pulsar_scenedb::World::new();
    helio_component::subsystems::ensure_content_ledger_attached(&mut world);

    let first = load_mesh_upload_shared(&path).expect("parse");
    let content = first.content_id;

    // Simulate hydrate acquiring through the World (the real event source):
    let e = world.spawn();
    world.insert(
        e,
        DedupRow {
            content_id: HandleId(content.0),
            verts: vec![1],
        },
    );
    world.insert(
        e,
        DedupRow {
            content_id: HandleId(content.0),
            verts: vec![1],
        },
    ); // no-op rewrite

    // Cache currently holds it...
    let cached_again = load_mesh_upload_shared(&path).unwrap();
    assert!(
        Arc::ptr_eq(&first.upload, &cached_again.upload),
        "cache hit before death"
    );

    // Last reference dies -> drop callback fires synchronously on this
    // thread -> cache entry evicted.
    world.despawn(e);

    // Re-parse MUST be a genuinely fresh Arc if eviction ran. (If the
    // global callback slot had already been claimed by an earlier test in
    // this binary WITHOUT the built-in eviction, this would fail -- which
    // is exactly the coupling this assertion is designed to surface.)
    let fresh = load_mesh_upload_shared(&path).unwrap();
    assert_eq!(
        fresh.content_id, content,
        "identity stable across eviction cycle"
    );
    assert_ne!(
        Arc::as_ptr(&fresh.upload),
        Arc::as_ptr(&first.upload),
        "evicted entry was truly freed"
    );

    clear_shared_mesh_cache();
    let _ = std::fs::remove_dir_all(&dir);
}
