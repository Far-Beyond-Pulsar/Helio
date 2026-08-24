//! Benchmarks for the content-dedup infrastructure (`helio_component`):
//!
//! 1. **Ledger acquire/release throughput** -- single-threaded steady state
//!    and an 8-thread contended run. The performance bar this exists to
//!    enforce: steady-state inc/dec is ALLOCATION-FREE and pays no global
//!    lock (64 high-bit shards).
//! 2. **XXH3-128 hashing throughput** at 1KB / 1MB / 64MB, both one-shot
//!    (the payload-identity shape) and chunk-streamed (the file shape,
//!    proving chunked IO costs ~nothing over whole-buffer hashing).
//! 3. **Hydrate dedup win** -- 100 components loading the SAME `.mesh`
//!    asset versus 100 DISTINCT assets, wall-clock plus allocation counts
//!    via a counting global allocator (cheap here because the bench binary
//!    owns the allocator choice end-to-end).
//! 4. **Codegen swap-path overhead** -- `World::insert` row rewrite of a
//!    handle-bearing component with a ledger attached versus absent, i.e.
//!    the exact marginal cost Phase-1's generated plumbing adds when a
//!    ledger is live (and, by the absence arm, that "absent" stays flat).
//!
//! Criterion follows SceneDB's existing convention (`criterion = "=0.8.2"`,
//! harness = false); Pulsar-Native has no other bench infrastructure to
//! reuse, and these measurements need THIS crate's types.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use helio::PackedVertex;
use helio_component::content_id::ContentHasher;
use helio_component::content_ledger::ContentLedger;
use pulsar_scenedb::handle_ledger::{HandleId, HandleLedger};

// ── Counting allocator (bench 3's alloc counter) ─────────────────────────

struct CountingAlloc {
    enabled: AtomicBool,
    bytes: AtomicUsize,
    count: AtomicUsize,
}

// SAFETY: pure pass-through to System; counters are atomics.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if self.enabled.load(Ordering::Relaxed) {
            self.bytes.fetch_add(layout.size(), Ordering::Relaxed);
            self.count.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc {
    enabled: AtomicBool::new(false),
    bytes: AtomicUsize::new(0),
    count: AtomicUsize::new(0),
};

fn alloc_deltas<T>(f: impl FnOnce() -> T) -> (T, usize, usize) {
    // NOTE: toggles GLOBAL itself (the #[global_allocator]) -- an earlier
    // draft had a second identical-looking static and counted nothing.
    GLOBAL.bytes.store(0, Ordering::Relaxed);
    GLOBAL.count.store(0, Ordering::Relaxed);
    GLOBAL.enabled.store(true, Ordering::Relaxed);
    let out = f();
    GLOBAL.enabled.store(false, Ordering::Relaxed);
    (
        out,
        GLOBAL.bytes.load(Ordering::Relaxed),
        GLOBAL.count.load(Ordering::Relaxed),
    )
}

// ── Shared fixtures ──────────────────────────────────────────────────────

/// A deterministic spread of ids whose TOP bits vary -- shard selection
/// reads the high 6 bits, and real content ids are hash digests uniform
/// across the full width, so fixtures must be avalanched the same way (a
/// plain counter would pile every id onto shard 0 and fake a
/// single-lock benchmark).
fn bench_ids(n: usize) -> Vec<HandleId> {
    (0..n)
        .map(|i| {
            let mut z = (i as u128).wrapping_add(0x9E37_79B9_7F4A_7C15_9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 64)).wrapping_mul(0xBF58_476D_1CE4_E5B9_BF58_476D_1CE4_E5B9);
            HandleId(z ^ (z >> 96))
        })
        .collect()
}

/// Pre-warms every id to count TWO so the measured loop is release+acquire
/// pairs over LIVE entries -- the true STEADY state (counts oscillate 2↔1,
/// no entry-insertion, no removal, no callback fires). Driving counts
/// through zero would measure the rare asset-death path instead.
fn warm(ledger: &ContentLedger, ids: &[HandleId]) {
    for &id in ids {
        ledger.acquire(id);
        ledger.acquire(id);
    }
}

// ── 1. Ledger throughput ─────────────────────────────────────────────────

fn bench_ledger(c: &mut Criterion) {
    let mut group = c.benchmark_group("ledger");
    const OPS: usize = 100_000;

    group.bench_function("acquire_release_single_thread", |b| {
        let ledger = ContentLedger::new();
        let ids = bench_ids(4096);
        warm(&ledger, &ids);
        b.iter(|| {
            for (i, &id) in ids.iter().enumerate() {
                if i % 2 == 0 {
                    ledger.release(id);
                    ledger.acquire(id);
                } else {
                    // Interleave a swap-shaped pair too (release A /
                    // acquire B / release B / acquire A).
                    let other = ids[(i + 1) % ids.len()];
                    ledger.release(id);
                    ledger.acquire(other);
                    ledger.release(other);
                    ledger.acquire(id);
                }
            }
        });
    });

    group.throughput(Throughput::Elements(OPS as u64));
    group.bench_function("acquire_release_contended_8_threads", |b| {
        let ledger = Arc::new(ContentLedger::new());
        let ids = Arc::new(bench_ids(4096));
        warm(&ledger, &ids);
        b.iter_custom(|iters| {
            let barrier = Arc::new(std::sync::Barrier::new(8));
            let mut joins = Vec::new();
            let start = Instant::now();
            for t in 0..8usize {
                let ledger = Arc::clone(&ledger);
                let ids = Arc::clone(&ids);
                let barrier = Arc::clone(&barrier);
                joins.push(std::thread::spawn(move || {
                    barrier.wait();
                    for k in 0..iters as usize {
                        // Overlapping id sets across threads: threads pick
                        // overlapping-but-shifted windows so some shard
                        // locks genuinely contend while others don't.
                        let idx = (k * 7 + t * 511) % ids.len();
                        let id = ids[idx];
                        ledger.release(id);
                        ledger.acquire(id);
                    }
                }));
            }
            for j in joins {
                j.join().unwrap();
            }
            start.elapsed()
        });
    });

    // The allocation-free bar, ASSERTED not just timed: 10k live-id inc/dec
    // must move the counting allocator exactly zero times.
    let ledger = ContentLedger::new();
    let ids = bench_ids(4096);
    warm(&ledger, &ids);
    let (_, bytes, count) = alloc_deltas(|| {
        for &id in ids.iter().take(2500) {
            ledger.release(id);
            ledger.acquire(id);
        }
    });
    assert_eq!(
        (bytes, count),
        (0, 0),
        "steady-state ledger ops allocated {bytes} bytes in {count} allocs"
    );

    group.finish();
}

// ── 2. Hashing throughput ────────────────────────────────────────────────

fn bench_hashing(c: &mut Criterion) {
    let mut group = c.benchmark_group("xxh3_128_payload");
    for size in [1024usize, 1024 * 1024, 64 * 1024 * 1024] {
        let payload: Vec<u8> = (0..size)
            .map(|i: usize| (i.wrapping_mul(2_654_435_761usize) >> 24) as u8)
            .collect();
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &payload, |b, p| {
            b.iter(|| {
                let mut h = ContentHasher::new();
                h.update(p);
                h.finish()
            });
        });
    }

    // Chunk-streamed shape at the largest size: proves the 64KiB-chunk file
    // path holds roughly the same throughput as the one-shot slice (i.e.,
    // chunked IO is effectively free next to the hash itself).
    let size = 64 * 1024 * 1024;
    let payload: Vec<u8> = (0..size)
        .map(|i: usize| (i.wrapping_mul(2_654_435_761usize) >> 24) as u8)
        .collect();
    group.throughput(Throughput::Bytes(size as u64));
    group.bench_function("streamed_64MiB_chunks_64KiB", |b| {
        b.iter(|| {
            let mut h = ContentHasher::new();
            let mut buf = Vec::with_capacity(64 * 1024);
            for chunk in payload.chunks(64 * 1024) {
                buf.clear();
                buf.extend_from_slice(chunk); // simulate read-through-reused-buffer
                h.update(&buf);
            }
            h.finish()
        });
    });
    group.finish();
}

// ── 3. Hydrate dedup win ─────────────────────────────────────────────────

fn write_test_mesh(path: &std::path::Path, seed: u64, verts: usize) {
    use helio_component::mesh_cache::encode;
    let vertices: Vec<PackedVertex> = (0..verts)
        .map(|i| PackedVertex {
            position: [
                ((i as f32) + seed as f32).sin(),
                ((i as f32) * 2.0 + seed as f32).cos(),
                (i as f32) * 0.001 + seed as f32 * 0.01,
            ],
            ..Default::default()
        })
        .collect();
    let indices: Vec<u32> = (0..verts as u32).map(|i| i % verts as u32).collect();
    std::fs::write(path, encode(&helio::MeshUpload { vertices, indices })).unwrap();
}

fn bench_hydrate_dedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("hydrate_dedup");
    const COMPONENTS: usize = 100;

    let dir = std::env::temp_dir().join(format!("helio-bench-meshes-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let shared_path = dir.join("shared.mesh");
    write_test_mesh(&shared_path, 1, 2048);
    let distinct_paths: Vec<std::path::PathBuf> = (0..COMPONENTS)
        .map(|i| {
            let p = dir.join(format!("distinct_{i}.mesh"));
            write_test_mesh(&p, i as u64 + 10, 2048);
            p
        })
        .collect();

    // One-shot allocation accounting (printed, not timed): the dedup win
    // isn't just wall-clock, it's 100× fewer parses ⇒ ~100× fewer payload
    // allocations. Measured cold-cache so BOTH arms pay their parse costs
    // honestly.
    {
        helio_component::subsystems::clear_shared_mesh_cache();
        let (n, bytes, count) = alloc_deltas(|| {
            (0..COMPONENTS)
                .map(|_| {
                    helio_component::subsystems::load_mesh_upload_shared(&shared_path).unwrap()
                })
                .count()
        });
        println!(
            "[alloc] same_asset_x100:      {bytes:>12} bytes in {count:>6} allocs ({n} loads)",
        );
        helio_component::subsystems::clear_shared_mesh_cache();
        let (n, bytes, count) = alloc_deltas(|| {
            distinct_paths
                .iter()
                .map(|p| helio_component::subsystems::load_mesh_upload_shared(p).unwrap())
                .count()
        });
        println!(
            "[alloc] distinct_assets_x100: {bytes:>12} bytes in {count:>6} allocs ({n} loads)",
        );
        helio_component::subsystems::clear_shared_mesh_cache();
    }

    // Same asset 100×: ONE parse serves all (cache hits), each component
    // still clones its own Vecs out of the shared Arc.
    group.bench_function("same_asset_x100", |b| {
        b.iter_batched(
            || {
                helio_component::subsystems::clear_shared_mesh_cache();
                Vec::<std::path::PathBuf>::new()
            },
            |_| {
                let (handles, bytes, count) = alloc_deltas(|| {
                    let mut arcs = Vec::with_capacity(COMPONENTS);
                    for _ in 0..COMPONENTS {
                        arcs.push(
                            helio_component::subsystems::load_mesh_upload_shared(&shared_path)
                                .unwrap(),
                        );
                    }
                    arcs
                });
                std::hint::black_box(handles);
                (bytes, count)
            },
            BatchSize::PerIteration,
        );
    });

    // Distinct assets: 100 real parses -- the honest baseline the dedup
    // win is measured against.
    group.bench_function("distinct_assets_x100", |b| {
        b.iter_batched(
            || {
                helio_component::subsystems::clear_shared_mesh_cache();
                Vec::<std::path::PathBuf>::new()
            },
            |_| {
                let (handles, bytes, count) = alloc_deltas(|| {
                    let mut arcs = Vec::with_capacity(COMPONENTS);
                    for p in &distinct_paths {
                        arcs.push(helio_component::subsystems::load_mesh_upload_shared(p).unwrap());
                    }
                    arcs
                });
                std::hint::black_box(handles);
                (bytes, count)
            },
            BatchSize::PerIteration,
        );
    });

    let _ = std::fs::remove_dir_all(&dir);
    group.finish();
}

// ── 4. Codegen swap-path overhead ────────────────────────────────────────

#[derive(Clone, Copy, Debug, pulsar_scenedb::SceneStore)]
struct BenchHandleRow {
    asset: HandleId,
    pad: u64,
}

fn bench_swap_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("codegen_swap_overhead");
    const REWRITES: usize = 20_000;

    fn setup(with_ledger: bool) -> (pulsar_scenedb::World, pulsar_scenedb::Entity) {
        let mut world = pulsar_scenedb::World::new();
        if with_ledger {
            world.attach_handle_ledger(Arc::new(ContentLedger::new()));
        }
        let entity = world.spawn();
        world.insert(
            entity,
            BenchHandleRow {
                asset: HandleId(1),
                pad: 0,
            },
        );
        (world, entity)
    }

    for with_ledger in [false, true] {
        group.bench_function(
            BenchmarkId::new(
                "row_rewrite_x20k",
                if with_ledger {
                    "ledger_attached"
                } else {
                    "no_ledger"
                },
            ),
            |b| {
                let (mut world, entity) = setup(with_ledger);
                b.iter(|| {
                    for i in 0..REWRITES {
                        let id = (i % 2) as u128 + 1;
                        // In-place rewrite => the generated swap dispatch
                        // compares old/new per handle field; equal ⇒ no-op,
                        // alternate ⇒ release+acquire through the ledger.
                        world.insert(
                            entity,
                            BenchHandleRow {
                                asset: HandleId(id),
                                pad: 0,
                            },
                        );
                    }
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_ledger,
    bench_hashing,
    bench_hydrate_dedup,
    bench_swap_path
);
criterion_main!(benches);
