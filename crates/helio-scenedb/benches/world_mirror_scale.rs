//! Helio#211: AAA-scale measurement for `pulsar_scenedb`'s World<->GPU
//! mirror bridge (SceneDB#24 + follow-ups #25-#29, all merged) — spawn
//! throughput, sustained churn, reallocation latency, concurrent buffer
//! access, and flush cost vs. dirty fraction, at realistic entity counts.
//!
//! Run: `cargo bench -p helio-scenedb --bench world_mirror_scale`
//!
//! `harness = false`, matching `pass_timing.rs`/upstream `legacy_model_bench.rs`'s
//! own choice: this measures real wall-clock cost of GPU-submitting
//! operations end to end (CPU bookkeeping + `queue.write_buffer`/
//! `copy_buffer_to_buffer` + submission), which is exactly what a running
//! game pays per frame — not a pure-CPU microbenchmark criterion is tuned
//! for, and not something a fine-grained GPU-timestamp bracket
//! (`gpu_timing.rs`'s own approach) is the right tool for either, since the
//! CPU-side bookkeeping (locking, dirty-marking, HashMap lookups) is very
//! much part of the cost being measured here, not overhead to isolate away.
//!
//! ## A finding this benchmark surfaced before a single number was taken
//!
//! The original issue proposed a "concurrent mutation" scenario: N worker
//! threads calling `world.insert()` in parallel. That scenario doesn't
//! exist to measure — `World::insert(&mut self, ..)` takes `&mut self`, so
//! the Rust borrow checker already prevents two threads from calling it on
//! the same `World` concurrently; genuinely parallel `world.insert()` calls
//! are not possible today without wrapping `World` itself in a `Mutex`
//! (which would serialize everything at that outer lock, making the
//! per-buffer `RwLock` inside `GrowableSceneBuffer`/`DirtyTrackedSceneBuffer`
//! moot -- contention would show up at the `Mutex<World>`, not there).
//!
//! What IS reachable concurrently today: `SceneGpuStore`'s own public
//! methods (`write_row_bytes_growing`, `mark_gpu_row_dirty`, `with_buffer`
//! accessors) take `&self`, reachable from multiple threads via
//! `Arc<SceneGpuStore>` without going through `World` at all -- exactly the
//! shape a hypothetical future "parallelize the CPU-side World mutation,
//! then scatter the GPU-mirror writes across threads afterward" API would
//! need. [`concurrent_buffer_access`] measures THAT instead: N threads
//! calling `write_row_bytes_growing` directly on disjoint rows of one
//! shared, already-registered buffer. This is the honest scope of what
//! "concurrency" means for this bridge today, and it's what Helio#213
//! (the concurrency-model issue) should be evaluated against.
//!
//! ## Scale sweep
//!
//! {1,000 / 10,000 / 100,000} for every scenario; {1,000,000} additionally
//! for [`spawn_throughput`] and [`reallocation_latency`] specifically (cheap
//! enough to run at that scale in a reasonable bench wall-clock budget).
//! [`steady_state_churn`] and [`flush_cost_vs_dirty_fraction`] run real
//! per-frame simulation loops (tens of frames each) and would push total
//! bench runtime past what's reasonable for a routine `cargo bench` if also
//! run at 1M -- 100,000 is already two orders of magnitude past anything
//! this bridge has been exercised at before this file, and is the number
//! that matters most for the "is this safe to build Stage 3 on" question.
//!
//! ## A dependency-resolution landmine this file's own development tripped
//!
//! Building this bench (or ANY target in this workspace) after `cargo
//! update`-ing `pulsar_scenedb` to pick up the merged World-mirror work can
//! break `wgpu-hal`'s DX12 backend compilation on Windows -- confirmed to be
//! a PRE-EXISTING, latent Cargo.lock issue (two incompatible `windows-core`
//! versions already coexist in the checked-in lockfile; re-resolving
//! ANYTHING, not specifically `pulsar_scenedb`, can flip which one
//! `wgpu-hal`'s own code resolves against), not something this change
//! introduces. Filed and tracked separately, not fixed here.

use pulsar_scenedb::gpu::{EngineGpuContext, GpuMirrorHandle, RegionClassConfig, SceneGpuConfig, SceneGpuStore};
use pulsar_scenedb::World;
use pulsar_scenedb_derive::SceneStore;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Shaped like `libhelio::instance::GpuInstanceData` (208 bytes) -- the
/// actual Stage 3 motivating case, not a toy-sized fixture. `#[gpu(layout =
/// packed)]` so it lands in exactly one buffer, matching how Stage 3 will
/// actually register it.
///
/// `#[gpu(mirror = Once)]`, not the plain `#[gpu]` (`DirtyTracked`) default
/// -- deliberately, so `register_gpu_columns_growable` routes this through
/// `register_growable_gpu_buffer` (the immediate/growable path) rather than
/// `register_dirty_tracked_gpu_buffer` (deferred until an explicit
/// `flush_gpu_mirror` call). Scenarios 1/3/4 below are specifically about
/// `GrowableSceneBuffer`'s real reallocation cost (`copy_buffer_to_buffer`)
/// and `write_row_bytes_growing`'s real write path -- neither of which a
/// `DirtyTracked`-mode field would exercise without an explicit flush this
/// file never calls for those three scenarios (scenario 5, further down,
/// uses its own separate DirtyTracked fixture specifically to measure THAT
/// path instead). An earlier version of this fixture used the plain
/// `#[gpu]` default here and silently measured CPU-only dirty-marking cost
/// for all three scenarios instead of real GPU buffer growth -- caught by
/// noticing scenario 1's reported realloc count was 0 at every scale
/// despite starting from a capacity of 64 and reaching 1,000,000 rows.
#[derive(SceneStore, Clone, Copy)]
#[gpu(layout = packed)]
struct BenchInstance {
    #[gpu(mirror = Once)]
    model: [f32; 16],
    #[gpu(mirror = Once)]
    normal_mat: [f32; 16], // padded to match GpuInstanceData's 3x vec4 (48B) closely enough for a size-realistic fixture; exact WGSL layout doesn't matter for a CPU/GPU-bandwidth benchmark
    #[gpu(mirror = Once)]
    bounds: [f32; 16],
    #[gpu(mirror = Once)]
    mesh_material_flags: [f32; 16], // stand-in for mesh_id/material_id/flags/lightmap_index (4x u32 = 16B); using f32;16 keeps this fixture on the crate's one built-in Pod array size rather than needing a bespoke Pod impl
}

fn test_context() -> EngineGpuContext {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .expect("no adapter -- GPU bench needs a local GPU");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("helio-scenedb-world-mirror-scale-bench"),
        ..Default::default()
    }))
    .expect("device");
    EngineGpuContext::new(Arc::new(device), Arc::new(queue))
}

fn scene_cfg() -> SceneGpuConfig {
    SceneGpuConfig {
        classes: vec![RegionClassConfig { capacity: 64, max_resident_cells: 1 }],
        tombstone_headroom: 8,
        max_cells_metadata: 16,
    }
}

fn sample_instance(i: u32) -> BenchInstance {
    BenchInstance {
        model: [i as f32; 16],
        normal_mat: [i as f32; 16],
        bounds: [i as f32; 16],
        mesh_material_flags: [i as f32; 16],
    }
}

fn fmt_dur(d: Duration) -> String {
    if d.as_secs_f64() >= 1.0 {
        format!("{:.3} s", d.as_secs_f64())
    } else if d.as_micros() >= 1000 {
        format!("{:.3} ms", d.as_secs_f64() * 1e3)
    } else {
        format!("{:.1} us", d.as_secs_f64() * 1e6)
    }
}

// ── Scenario 1: spawn throughput + reallocation count ──────────────────────

fn spawn_throughput() {
    println!("\n=== Scenario 1: spawn throughput (N entities, one packed #[gpu] insert each) ===");
    println!("{:>10} | {:>12} | {:>14} | {:>10}", "N", "wall time", "per-entity", "reallocs");
    for &n in &[1_000u32, 10_000, 100_000, 1_000_000] {
        let ctx = test_context();
        let mut store = SceneGpuStore::new(&ctx, scene_cfg());
        // Small initial capacity -- the point is measuring growth cost as
        // part of ordinary spawn throughput, not avoiding it via a
        // conveniently-large upfront allocation.
        BenchInstance::register_gpu_columns_growable(&mut store, 64, ctx.device());
        let store = Arc::new(store);
        let mut world = World::new();
        world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));
        let id = BenchInstance::packed_gpu_component_id();
        let epoch_before = store.growable_epoch_for_id(id).unwrap_or(0);

        let start = Instant::now();
        for i in 0..n {
            let e = world.spawn();
            world.insert(e, sample_instance(i));
        }
        let elapsed = start.elapsed();

        let epoch_after = store.growable_epoch_for_id(id).unwrap_or(0);
        println!(
            "{:>10} | {:>12} | {:>14} | {:>10}",
            n,
            fmt_dur(elapsed),
            fmt_dur(elapsed / n),
            epoch_after - epoch_before,
        );
    }
}

// ── Scenario 2: steady-state churn ──────────────────────────────────────────

fn steady_state_churn() {
    println!("\n=== Scenario 2: steady-state churn (N live entities, F frames, each frame despawns+respawns a fraction, one flush/frame) ===");
    println!("{:>10} | {:>8} | {:>10} | {:>14} | {:>14}", "N", "churn %", "frames", "total", "per-frame");
    const FRAMES: u32 = 30;
    for &n in &[1_000u32, 10_000, 100_000] {
        for &churn_pct in &[1u32, 10, 50] {
            let ctx = test_context();
            let mut store = SceneGpuStore::new(&ctx, scene_cfg());
            BenchInstance::register_gpu_columns_growable(&mut store, n, ctx.device());
            let store = Arc::new(store);
            let mut world = World::new();
            world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

            let mut entities: Vec<_> = (0..n)
                .map(|i| {
                    let e = world.spawn();
                    world.insert(e, sample_instance(i));
                    e
                })
                .collect();

            let churn_count = ((n as u64 * churn_pct as u64) / 100) as usize;
            let start = Instant::now();
            for frame in 0..FRAMES {
                for slot in 0..churn_count {
                    let idx = (slot * 7919 + frame as usize) % entities.len(); // deterministic pseudo-scatter, not just a contiguous prefix
                    world.despawn(entities[idx]);
                    let e = world.spawn();
                    world.insert(e, sample_instance(idx as u32));
                    entities[idx] = e;
                }
                world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");
            }
            let elapsed = start.elapsed();
            println!(
                "{:>10} | {:>7}% | {:>10} | {:>14} | {:>14}",
                n,
                churn_pct,
                FRAMES,
                fmt_dur(elapsed),
                fmt_dur(elapsed / FRAMES),
            );
        }
    }
}

// ── Scenario 3: reallocation latency in isolation ──────────────────────────

fn reallocation_latency() {
    println!("\n=== Scenario 3: single reallocation latency (doubling N -> 2N, direct ensure_capacity-equivalent via one write past capacity) ===");
    // NOT swept to 1,000,000 here (unlike scenarios 1/3's siblings): doubling
    // 1,000,000 rows of this 256-byte fixture needs a 512,000,000-byte
    // buffer, which EXCEEDS wgpu's default `Limits::max_buffer_size` (256
    // MiB = 268,435,456 bytes) and panics with a wgpu validation error
    // rather than failing gracefully -- confirmed empirically, not a
    // hypothetical. Neither `DynamicGpuBuffer::ensure_capacity` nor
    // `GrowableSceneBuffer` currently check against this limit before
    // doubling. This is real, important evidence for Helio#212
    // (pre-reservation/growth issue): growth needs a documented ceiling
    // check (and either a `Result`-based failure path or a device-limits
    // increase at context-creation time), not just "double forever."
    println!("{:>12} | {:>12} | {:>14}", "N (before)", "-> 2N", "realloc time");
    for &n in &[1_000u32, 10_000, 100_000] {
        let ctx = test_context();
        let mut store = SceneGpuStore::new(&ctx, scene_cfg());
        BenchInstance::register_gpu_columns_growable(&mut store, n, ctx.device());
        let store = Arc::new(store);
        let mut world = World::new();
        world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

        // Fill to capacity N first (untimed) so the timed insert below is
        // the one that actually crosses the threshold and triggers growth.
        for i in 0..n {
            let e = world.spawn();
            world.insert(e, sample_instance(i));
        }

        let start = Instant::now();
        let e = world.spawn();
        world.insert(e, sample_instance(n));
        ctx.device().poll(wgpu::PollType::wait_indefinitely()).expect("poll"); // ensure the GPU-to-GPU copy has actually landed, not just been enqueued
        let elapsed = start.elapsed();

        println!("{:>12} | {:>12} | {:>14}", n, n * 2, fmt_dur(elapsed));
    }
}

// ── Scenario 4: concurrent buffer access (see module doc for scope) ────────

fn concurrent_buffer_access() {
    println!("\n=== Scenario 4: concurrent access to one shared, already-registered buffer (disjoint rows per thread; see module doc for why this replaces \"concurrent world.insert\") ===");
    println!("{:>10} | {:>8} | {:>14} | {:>18}", "N total", "threads", "wall time", "per-write");
    for &n in &[10_000u32, 100_000] {
        for &threads in &[1u32, 4, 8, 16] {
            let ctx = test_context();
            let mut store = SceneGpuStore::new(&ctx, scene_cfg());
            BenchInstance::register_gpu_columns_growable(&mut store, n, ctx.device());
            let store = Arc::new(store);
            let id = BenchInstance::packed_gpu_component_id();
            let per_thread = n / threads;

            let start = Instant::now();
            std::thread::scope(|scope| {
                for t in 0..threads {
                    let store = Arc::clone(&store);
                    let queue = Arc::clone(ctx.queue());
                    scope.spawn(move || {
                        let base = t * per_thread;
                        let value = sample_instance(t);
                        let bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(&value as *const BenchInstance as *const u8, std::mem::size_of::<BenchInstance>())
                        };
                        for row in base..base + per_thread {
                            let _ = store.write_row_bytes_growing(id, &queue, bytes, row);
                        }
                    });
                }
            });
            let elapsed = start.elapsed();
            println!(
                "{:>10} | {:>8} | {:>14} | {:>18}",
                n,
                threads,
                fmt_dur(elapsed),
                fmt_dur(elapsed / n),
            );
        }
    }
}

// ── Scenario 5: flush cost vs. dirty fraction ───────────────────────────────

fn flush_cost_vs_dirty_fraction() {
    println!("\n=== Scenario 5: flush_gpu_mirror cost vs. dirty fraction (N entities registered dirty-tracked, only a fraction marked dirty) ===");
    println!("{:>10} | {:>10} | {:>14}", "N", "dirty %", "flush time");
    for &n in &[1_000u32, 10_000, 100_000] {
        for &dirty_pct in &[0u32, 1, 10, 100] {
            let ctx = test_context();
            let mut store = SceneGpuStore::new(&ctx, scene_cfg());
            // Plain #[gpu] (DirtyTracked default) -- not packed/Once here,
            // this scenario is specifically about the dirty-tracked flush
            // path's cost, so a single-field DirtyTracked fixture keeps the
            // measurement focused.
            #[derive(SceneStore, Clone, Copy)]
            struct DirtyTrackedBenchField {
                #[gpu]
                value: u32,
            }
            DirtyTrackedBenchField::register_gpu_columns_growable(&mut store, n, ctx.device());
            let store = Arc::new(store);
            let mut world = World::new();
            world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

            let entities: Vec<_> = (0..n)
                .map(|i| {
                    let e = world.spawn();
                    world.insert(e, DirtyTrackedBenchField { value: i });
                    e
                })
                .collect();
            world.flush_gpu_mirror(ctx.queue()).expect("mirror attached"); // clear the initial-insert dirty state so only the intended fraction is dirty for the timed flush

            let dirty_count = ((n as u64 * dirty_pct as u64) / 100) as usize;
            for &e in entities.iter().take(dirty_count) {
                world.insert(e, DirtyTrackedBenchField { value: 999 });
            }

            let start = Instant::now();
            world.flush_gpu_mirror(ctx.queue()).unwrap();
            let elapsed = start.elapsed();
            println!("{:>10} | {:>9}% | {:>14}", n, dirty_pct, fmt_dur(elapsed));
        }
    }
}

// ── Scenario 6 (round 2, post-Helio#212): reservation eliminates in-batch growth ──

fn reservation_eliminates_batch_growth() {
    println!("\n=== Scenario 6 (post-#212): reserve_gpu_mirror_capacity before a known-size batch spawn ===");
    println!("{:>10} | {:>18} | {:>18} | {:>14}", "N", "reallocs (cold)", "reallocs (reserved)", "reserved wall time");
    for &n in &[1_000u32, 10_000, 100_000] {
        // Cold: no reservation, same shape as scenario 1.
        let ctx = test_context();
        let mut store = SceneGpuStore::new(&ctx, scene_cfg());
        BenchInstance::register_gpu_columns_growable(&mut store, 64, ctx.device());
        let store = Arc::new(store);
        let mut world = World::new();
        world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));
        let id = BenchInstance::packed_gpu_component_id();
        let epoch_before = store.growable_epoch_for_id(id).unwrap_or(0);
        for i in 0..n {
            let e = world.spawn();
            world.insert(e, sample_instance(i));
        }
        let cold_reallocs = store.growable_epoch_for_id(id).unwrap_or(0) - epoch_before;

        // Reserved: same batch, but reserve_gpu_mirror_capacity(n) first.
        let ctx2 = test_context();
        let mut store2 = SceneGpuStore::new(&ctx2, scene_cfg());
        BenchInstance::register_gpu_columns_growable(&mut store2, 64, ctx2.device());
        let store2 = Arc::new(store2);
        let mut world2 = World::new();
        world2.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store2), Arc::clone(ctx2.queue())));
        world2
            .reserve_gpu_mirror_capacity(ctx2.queue(), n)
            .expect("mirror attached")
            .expect("reserve succeeds");
        let epoch_before2 = store2.growable_epoch_for_id(id).unwrap_or(0);
        let start = Instant::now();
        for i in 0..n {
            let e = world2.spawn();
            world2.insert(e, sample_instance(i));
        }
        let elapsed = start.elapsed();
        let reserved_reallocs = store2.growable_epoch_for_id(id).unwrap_or(0) - epoch_before2;

        println!(
            "{:>10} | {:>18} | {:>18} | {:>14}",
            n, cold_reallocs, reserved_reallocs, fmt_dur(elapsed),
        );
    }
}

// ── Scenario 7 (round 2, post-Helio#213): concurrent mark_dirty ────────────

fn concurrent_mark_dirty() {
    println!("\n=== Scenario 7 (post-#213): concurrent mark_dirty on disjoint rows, pre-reserved (fast read-lock path) ===");
    println!("{:>10} | {:>8} | {:>14} | {:>18}", "N total", "threads", "wall time", "per-mark");
    #[derive(SceneStore, Clone, Copy)]
    struct ConcurrentMarkField {
        #[gpu]
        value: u32,
    }
    for &n in &[10_000u32, 100_000] {
        for &threads in &[1u32, 4, 8, 16] {
            let ctx = test_context();
            let mut store = SceneGpuStore::new(&ctx, scene_cfg());
            ConcurrentMarkField::register_gpu_columns_growable(&mut store, n, ctx.device());
            let store = Arc::new(store);
            let id = ConcurrentMarkField::gpu_columns()[0].field_token.id();
            let per_thread = n / threads;

            let start = Instant::now();
            std::thread::scope(|scope| {
                for t in 0..threads {
                    let store = Arc::clone(&store);
                    scope.spawn(move || {
                        let base = t * per_thread;
                        let value_bytes = (t).to_ne_bytes();
                        for row in base..base + per_thread {
                            store.mark_gpu_row_dirty(id, row, &value_bytes);
                        }
                    });
                }
            });
            let elapsed = start.elapsed();
            println!(
                "{:>10} | {:>8} | {:>14} | {:>18}",
                n,
                threads,
                fmt_dur(elapsed),
                fmt_dur(elapsed / n),
            );
        }
    }
}

fn main() {
    println!("Helio#211 -- World<->GPU mirror bridge AAA-scale benchmark");
    println!("(pulsar_scenedb#24 + follow-ups, real GPU device)");
    spawn_throughput();
    steady_state_churn();
    reallocation_latency();
    concurrent_buffer_access();
    flush_cost_vs_dirty_fraction();
    reservation_eliminates_batch_growth();
    concurrent_mark_dirty();
}
