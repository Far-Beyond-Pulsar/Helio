# The 2M benchmark needed GPU mobility updates and frame-aligned performance records

## Resolution and scope

Implemented and corrected in [18d2458b93509595e61d6101e62bb5030149f15e](https://github.com/Far-Beyond-Pulsar/Helio/commit/18d2458b93509595e61d6101e62bb5030149f15e).
This issue covers the new benchmark and defects found in its initial working
iterations. The demo is a new file in that commit, so Git represents it as
additions rather than pretending its intermediate versions existed in HEAD.
The shared mobility helper has a genuine tracked before/after diff.

## Problems diagnosed

1. Inserting `Movability::Dynamic` only changed a CPU component. GPU batching
   and shadow partitions read `StaticObjectComponent.flags`; the benchmark
   could claim two million dynamic objects while still sending static flags.
2. Tick-only duration excluded database logging and event-loop time, yielding
   an FPS number inconsistent with actual frame cadence. Resize could silently
   change Immediate presentation back to FIFO, changing the measurement.
3. Latest GPU timestamps arrive later, can repeat, and can skip delivered source
   frames. Attaching them directly to the current transition checkpoint mixes
   workloads; graph rebuilds can also reset source indices.
4. Queue writes need submission before polling can drain startup transfers.
   Completion was previously marked before the final render/readback, while
   the application kept rendering indefinitely afterward.

## Implemented corrections

- `set_object_movability` changes both the CPU tag and GPU movable bit and is
  used for spawning and transitions, preserving unrelated flag bits.
- Keep all 300 initial frames static; then promote 100,000 objects at a time.
  Wait three seconds at each checkpoint and collect 120 measurement frames.
- Measure present-to-present wall time for FPS and retain CPU tick, scene
  flush, surface acquisition, renderer/submit and presentation stages separately.
- Preserve selected presentation mode on resize; record hardware/backend,
  dimensions, presentation mode and workload semantics.
- Submit pending startup writes before waiting; store upload ranges/bytes.
- Log transitions and settling too, with `frame_context.is_sample` identifying
  measurement windows. Commit SQLite inserts once per frame.
- Track profiler epochs across index resets. SQL views `gpu_frame_samples`
  and `gpu_pass_samples` deduplicate delivered GPU snapshots and join to the
  source frame's checkpoint. Do not relabel CPU samples with GPU source IDs.
- Drain eight trailing frames, mark completion afterward, checkpoint WAL and
  exit. Fail on renderer errors rather than silently recording success.
- Add a read-only Python analysis script; report actual sample counts and use
  only the last 120 baseline frames for stabilized static comparisons.

## Completed measurements

RTX 4060 Laptop GPU, Vulkan, 800×600, Immediate, timestamps enabled:

| Dynamic objects | Mean GPU ms | Mean cadence ms | Actual FPS |
|---:|---:|---:|---:|
| 0 | 3.9499 | 4.5748 | 218.59 |
| 500,000 | 4.0563 | 5.0538 | 197.87 |
| 1,000,000 | 4.0990 | 5.7543 | 173.78 |
| 1,500,000 | 4.0724 | 5.2125 | 191.85 |
| 2,000,000 | 4.0665 | 5.2349 | 191.02 |

Final cadence median/p95: 4.2325/16.4368 ms; GPU p95: 4.2516 ms. Final CPU
tick mean: 3.1614 ms (render 1.9294, acquire 0.8164, present 0.2944, flush
0.1135 ms). Another 2.0735 ms of cadence lies between ticks; database and
event-loop contributions were not separately instrumented in this capture.
CPU/GPU costs overlap and must not simply be added to predict FPS.

`two_million_verified.sqlite` contains 13,696 frame rows and 12,605 unique GPU
source frames; run.completed=1. Final window: 120 cadence samples, 109 distinct
GPU samples. Maximum delivered lag=2; query overflows=0; readback-slot drops=0;
integrity_check=ok. Final steady SceneDB upload bytes=0. Raw databases remain
in the local workspace, not in GitHub; the results and query tooling are in Git.

## Important workload limitation

Dynamic means the CPU mobility promise **and GPU movable classification**.
The demo does not animate all transforms or deform vertices each frame. These
numbers do not describe two million CPU transform updates, physics bodies,
skinned objects or moving shadows. The early A/B captures also predate the
GPU-flag correction; use their controlled pairs to attribute shader fixes,
not as evidence that the earlier dynamic sweep was valid.

## Reproduction and querying

```powershell
$env:HELIO_PERF_DB = 'two_million_new_run.sqlite'
cargo run --release -p examples --bin two_million_dynamic
python scripts/analyze_two_million.py two_million_new_run.sqlite
```

Overrides: HELIO_INITIAL_FRAMES, HELIO_DYNAMIC_STEP,
HELIO_STABILIZE_SECONDS and HELIO_SAMPLE_FRAMES. Use a fresh DB name to retain
old runs separately. The program also supports multiple runs in one DB.

```sql
SELECT pass_name, COUNT(*) AS samples, AVG(gpu_ms) AS gpu_ms
FROM gpu_pass_samples
WHERE run_id=1 AND dynamic_objects=2000000 AND is_sample=1
GROUP BY pass_name ORDER BY gpu_ms DESC;
```

Release build and the full sweep completed. Related renderer tests passed:
four profiler tests, five object-batch GPU tests, one sparse-light GPU test,
and 49 shadow-matrix tests. This is targeted validation, not a claim that the
entire Helio/Pulsar workspace test suite was run.

## Before/after code

The issue includes the actual helper changes, newly added demo/schema, and
analysis script diff. The full published commit also wires dependencies,
profiling features, README instructions and artifact ignore rules.
