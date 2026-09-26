# Two-million-object diagnosis and verified fixes

Measured 2026-09-26 on an NVIDIA RTX 4060 Laptop GPU, Vulkan, **800 × 600**,
Immediate presentation. These numbers are specific to this scene/device and
include enabled timestamp profiling.

The six retrospective Helio issues, with detailed root causes and actual
syntax-highlighted before/after diffs, are indexed in
[the closed-issue records](issues/README.md).

## Exact causes

1. **Wrong encoder was timed.** Passes such as LightCull return no render-pass
   descriptor but record raw compute commands on the graphics encoder. The
   executor timed only the other encoder. Both command streams now have
   matched per-pass scopes, summed by label. CPU samples include prepare and
   execute, repeated labels export once, and totals do not double-count graph
   envelopes. This uses Helio's existing Pulsar profiling integration.
2. **LightCull scanned sparse entity capacity per tile.** The single light is
   spawned after two million objects. Every tile scanned millions of empty
   light rows. A GPU compaction pass now produces active sparse indices when
   light content changes; tile culling walks only active lights. Original IDs
   are preserved for shading, including IDs above two million.
3. **A huge instance batch starved culling of parallel work.** Frustum and
   occlusion kernels each assign one 64-lane workgroup per draw. Two million
   identical cubes were one draw. ObjectBatch now splits runs at 4,096-instance
   boundaries; adjacent draws still share material ranges. No objects are
   removed to achieve this improvement.
4. **ShadowMatrix treated empty rows as live shadow casters.** Zeroed rows have
   shadow_index=0 and light_type=directional. Millions of invocations computed
   invalid cascades and raced on shadow slot zero. It now rejects zero-intensity
   rows before matrix computation or writes.

## Measured evidence

| GPU region | Before | Final |
|---|---:|---:|
| LightCull | 562.667 ms | 0.0098 ms |
| IndirectDispatch / frustum culling | 42.509 ms | 0.9910 ms |
| OcclusionCull | 36.438 ms | 0.7030 ms |
| ShadowMatrix | 4.622 ms | 0.3714 ms |
| Whole graph | ~648.6 ms | 4.0665 ms |

Preserved causal runs:

- `two_million_exact_before.sqlite`: corrected GPU markers, original shaders.
- `two_million_exact_after_light.sqlite`: LightCull fix alone; graph ~85.2 ms.
- `two_million_exact_after_batch.sqlite`: bounded batches too; graph ~8 ms.
- `two_million_exact_final.sqlite`: full sweep with correct GPU mobility flags,
  before the shadow-matrix fix; final graph 8.49 ms.
- **`two_million_verified.sqlite`**: complete final sweep with all fixes.

The earlier demo only changed a CPU mobility tag. The final sweep sets the
actual GPU movable flag too. This is a **mobility-classification benchmark**:
transforms and vertices are not animated every frame. It is not a measurement
of two million per-frame CPU transform updates, skinning, or physics.

## Final sweep

300 static frames, then 20 steps of 100,000 objects, each settling for three
seconds and collecting 120 cadence samples. The baseline below uses its last
120 frames to exclude initialization. GPU data is deduplicated and assigned to
its **source frame**, not the later frame receiving its readback.

| Dynamic objects | Mean GPU ms | Mean actual frame ms | Actual FPS |
|---:|---:|---:|---:|
| 0 | 3.9499 | 4.5748 | 218.59 |
| 500,000 | 4.0563 | 5.0538 | 197.87 |
| 1,000,000 | 4.0990 | 5.7543 | 173.78 |
| 1,500,000 | 4.0724 | 5.2125 | 191.85 |
| 2,000,000 | 4.0665 | 5.2349 | 191.02 |

Final cadence median: 4.2325 ms; p95: 16.4368 ms. Final GPU p95: 4.2516 ms.
Do not report reciprocal GPU time as actual FPS: the latter includes logging,
event-loop scheduling and presentation. CPU and GPU work overlap.

Final CPU tick averages 3.1614 ms: render/submit 1.9294 ms, surface acquisition
0.8164 ms, presentation 0.2944 ms, scene flush 0.1135 ms, and 0.0079 ms of other
tick work. The remaining 2.0735 ms of average cadence is between ticks
(database logging plus event-loop scheduling); those two components are not
separately instrumented in this capture. Steady final SceneDB uploads: 0 bytes.

Per-pass GPU scopes explain 3.9187 ms of the 4.0665 ms graph span. The 0.1478 ms
difference is between pass timestamp boundaries; the serial executor records
no shader dispatches or draws in those intervals; timestamp/backend command
boundary costs are not individually separated in this capture. Envelope and
pass times are preserved separately instead of disguising this difference as
a shader's cost. Target clear is separately measured at 0.0049 ms outside the
graph. Fog's inactive scope is 0.0030 ms, not the original bottleneck.

13,696 frame records; 12,605 unique GPU source frames. Final sample window:
120 cadence samples, 109 unique GPU samples. Latest-snapshot async delivery can
repeat or supersede GPU frames; the views remove duplicates and expose the
actual sample count. Maximum delivered GPU lag is two frames; query overflows
and readback-slot drops are both zero. SQLite integrity check: `ok`.

## Reproduce and query

From the Helio root:

```powershell
$env:HELIO_PERF_DB = 'two_million_new_run.sqlite'
cargo run --release -p examples --bin two_million_dynamic
python scripts/analyze_two_million.py two_million_new_run.sqlite
```

Defaults are the full sweep above. Overrides: `HELIO_INITIAL_FRAMES`,
`HELIO_DYNAMIC_STEP`, `HELIO_STABILIZE_SECONDS`, `HELIO_SAMPLE_FRAMES`.
The program drains timestamp delivery, checkpoints SQLite WAL and exits.

For GPU queries prefer the epoch-aware `gpu_frame_samples` and
`gpu_pass_samples` views. Raw CPU times are in `pass_timings` joined through
`profiler_frames.frame_id`. `frame_context.is_sample` separates measurement
windows from transitions, settling and final drain frames. `run_metadata`
records the device, presentation mode and measurement semantics.

```sql
SELECT pass_name, COUNT(*) AS samples, AVG(gpu_ms) AS gpu_ms
FROM gpu_pass_samples
WHERE run_id = 1 AND dynamic_objects = 2000000 AND is_sample = 1
GROUP BY pass_name ORDER BY gpu_ms DESC;
```

## Validation

- LightCull GPU regression: sparse index 2,000,003 reaches all tiles; deletion
  and all-empty results are correct.
- ObjectBatch: five GPU tests, including contiguous bounded batches with no
  lost instances, correct indirect arguments, material ranges and shadow counts.
- ShadowMatrix: GPU test proves empty rows cannot touch slot zero and a live
  high-index point light still produces finite matrices; 48 math/layout tests.
- Profiler: four tests, including real GPU dual-encoder timestamps, repeated
  label aggregation, non-double-counted totals and CPU prepare/execute accumulation.
