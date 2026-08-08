# `world_mirror_scale` reference numbers (Helio#211)

Measured 2026-08-08 on the maintainer's dev machine (Windows 11,
`cargo bench -p helio-scenedb --bench world_mirror_scale`, real GPU device,
Vulkan backend — see the "How these were captured" note at the bottom for
why not the workspace's default DX12 backend). **These are a human reference
point for the Helio#212/#213/#214 prioritization decision, not a CI
regression gate** — same status as `pulsar_scenedb`'s own `BASELINE.md`
files upstream.

## Scenario 1 — spawn throughput

| N | wall time | per-entity | reallocations |
|---|---|---|---|
| 1,000 | 18.8 ms | 18.8 µs | 4 |
| 10,000 | 42.3 ms | 4.2 µs | 8 |
| 100,000 | 317.5 ms | 3.2 µs | 11 |
| 1,000,000 | 3.083 s | 3.1 µs | 14 |

Amortized per-entity cost stabilizes around ~3 µs once past a few thousand
entities. Reallocation counts match the expected doubling sequence exactly
(64 → 128 → … → past 1,000,000 is 14 doublings) — direct confirmation the
growth mechanism behaves as designed, not just "eventually correct."

## Scenario 2 — steady-state churn (30 simulated frames, one flush/frame)

| N | churn | per-frame | vs. 16.6 ms (60 fps) budget |
|---|---|---|---|
| 1,000 | 1% | 30.7 µs | fine |
| 1,000 | 10% | 341.8 µs | fine |
| 1,000 | 50% | 1.856 ms | fine |
| 10,000 | 1% | 390.0 µs | fine |
| 10,000 | 10% | 3.914 ms | fine |
| 10,000 | 50% | 21.020 ms | **over budget** |
| 100,000 | 1% | 3.904 ms | fine (but 23% of budget) |
| 100,000 | 10% | 44.011 ms | **over budget (2.6 frames)** |
| 100,000 | 50% | 263.237 ms | **over budget (15.9 frames)** |

**This is the headline finding for Stage 3 planning.** At 100,000 live
entities with a realistic 10% per-frame churn rate (streamed sublevels,
spawned/despawned effects), one frame's worth of despawn+respawn+flush
already costs *2.6x* the entire 60fps frame budget on its own — before
accounting for anything else the frame has to do (rendering, physics, audio,
gameplay). 1% churn at the same scale is within budget but consumes nearly a
quarter of it for this one subsystem alone.

## Scenario 3 — single reallocation latency (isolated)

| N (before) | → 2N | realloc time |
|---|---|---|
| 1,000 | 2,000 | 901.7 µs |
| 10,000 | 20,000 | 3.046 ms |
| 100,000 | 200,000 | 42.734 ms |

Scales roughly linearly with the copied byte count (bandwidth-bound
`copy_buffer_to_buffer`, as expected). **42.7 ms for one reallocation at
100k→200k rows is over 2.5x the entire 60fps frame budget, landing wherever
whichever `world.insert()` call happens to cross the capacity threshold** —
direct, measured evidence for Helio#212's premise (a pre-reservation API is
not a nice-to-have at this scale, it's necessary to avoid an unpredictable,
severe frame hitch).

**A crash, not just a slow number, at the next scale up.** Attempting
1,000,000 → 2,000,000 rows of this benchmark's 256-byte packed fixture
(512,000,000 bytes) panics with a real `wgpu` validation error:

```
wgpu error: Validation Error
Caused by:
  In Device::create_buffer, label = 'BenchInstance::packed'
    Buffer size 512000000 is greater than the maximum buffer size (268435456)
```

Neither `DynamicGpuBuffer::ensure_capacity` nor `GrowableSceneBuffer` check
against `wgpu::Limits::max_buffer_size` (256 MiB by default) before
doubling — growth just keeps doubling until it hits this ceiling and panics
outright, with no graceful failure path. **This is new scope for
Helio#212**, not originally called out in that issue: a real, hard ceiling
exists well within AAA-relevant entity counts for wide per-row records (a
256-byte packed instance record hits it at 1,048,576 rows in one buffer),
and needs either a documented capacity ceiling + graceful error, a
device-limits increase requested at context-creation time, or splitting a
single logical column across multiple physical buffers past the limit —
this needs its own design decision, not just noting the number.

## Scenario 4 — concurrent access to one shared buffer (bypassing `World`)

| N | threads | wall time | per-write |
|---|---|---|---|
| 10,000 | 1 | 23.5 ms | 2.4 µs |
| 10,000 | 4 | 37.4 ms | 3.7 µs |
| 10,000 | 8 | 40.1 ms | 4.0 µs |
| 10,000 | 16 | 43.1 ms | 4.3 µs |
| 100,000 | 1 | 159.0 ms | 1.6 µs |
| 100,000 | 4 | 283.6 ms | 2.8 µs |
| 100,000 | 8 | 246.9 ms | 2.5 µs |
| 100,000 | 16 | 252.9 ms | 2.5 µs |

**Per-write cost gets WORSE, not better, as thread count increases** — the
opposite of what embarrassingly-parallel disjoint-row writes should do. At
10,000 rows, going from 1 to 16 threads makes each write ~79% slower on a
per-write basis despite each thread touching completely disjoint rows. This
is direct, measured confirmation that Helio#213's `RwLock`-contention
concern is real, not speculative — the lock IS the bottleneck here, not
useful parallelism. Justifies prioritizing #213's `mark_dirty`
read-vs-write-lock fix and the sharding/lock-free evaluation, not just the
"fix regardless" small item.

## Scenario 5 — flush cost vs. dirty fraction

| N | dirty % | flush time |
|---|---|---|
| 1,000 | 0% | 0.5 µs |
| 1,000 | 1% | 1.6 µs |
| 1,000 | 10% | 1.6 µs |
| 1,000 | 100% | 2.0 µs |
| 10,000 | 0% | 3.8 µs |
| 10,000 | 1% | 5.3 µs |
| 10,000 | 10% | 6.4 µs |
| 10,000 | 100% | 6.9 µs |
| 100,000 | 0% | 38.5 µs |
| 100,000 | 1% | 59.6 µs |
| 100,000 | 10% | 56.8 µs |
| 100,000 | 100% | 76.9 µs |

**The 0%-dirty baseline cost is real and non-trivial, and barely grows with
dirty fraction.** At 100,000 rows, scanning for zero dirty rows already
costs 38.5 µs — and 1% dirty (59.6 µs) costs about as much as 10% dirty
(56.8 µs, within noise of each other), while 100% dirty (76.9 µs) is barely
2x the 0% baseline despite touching 100x more rows. This is exactly the
signature Helio#214 predicted: the `O(capacity)` scan dominates over the
actual write cost across nearly the whole dirty-fraction range, meaning
sparse per-frame changes (the realistic case for a large, mostly-static
scene) get charged almost the full-capacity cost regardless of how little
actually changed. Confirms #214's proposed fix (an explicit dirty-row list,
`O(dirty)` instead of `O(capacity)`) targets a real, measured cost, not a
theoretical one.

## How these were captured — a dependency-resolution note

Captured with `dx12` locally (uncommitted) removed from this workspace's
`wgpu` feature list and the Vulkan backend used instead. Updating
`pulsar_scenedb` to pick up the merged World-mirror work exposed a
pre-existing, latent Cargo.lock issue — two incompatible `windows-core`
versions (0.58.0 via `gpu-allocator`, 0.62.2 direct in `wgpu-hal` 30.0.0)
already coexist in the checked-in lockfile, and re-resolving *anything* (not
specifically `pulsar_scenedb`) can flip which one `wgpu-hal`'s own DX12
module resolves against, breaking the Windows DX12 backend's compilation.
Confirmed via clean-cache A/B testing this is unrelated to this PR's actual
changes. Filed and tracked separately; this PR's committed `Cargo.toml` does
**not** carry the `dx12`-disabling workaround — that was local-only, used
solely to capture these numbers.
