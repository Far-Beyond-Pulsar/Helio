# Ordered residency admission worker

Residency admission and publication packing now run on a bounded worker by
default. Repeated 720p comparisons remove the measured render-thread admission
tail and improve descent frame tails. They do not establish the planetary
performance or visual targets. Whole-plan retargeting remains opt-in; complete
cut replacement, generation latency and inaccurate far geometry remain open.

## Ownership and ordering

The producer owns the existing residency state, including lazy allocation,
visible-slot pins and retirement. It prepares generation jobs, their immutable
source snapshot and packed publication links. Rendering owns every GPU object.
There is one replaceable demand request and at most one prepared/in-flight
bundle. New requests can coalesce, but prepared bundles cannot be discarded or
reordered. The producer cannot advance until rendering acknowledges encoding
the prior bundle, including traversal. Future GPU work uses the same ordered
queue. This acknowledgement is not a GPU-completion wait.

The render caller uses nonblocking mailbox polls. A busy acknowledgement is
retained and flushed on the next exchange. The source/grid used for rendering
comes from the accepted publication, not the producer's future state. Refining
status is compared against current demand so an older ready result cannot
claim that a new request has settled. Worker failure becomes an explicit render
poll failure. Teardown wakes the producer without joining its CPU work.

Selection still has its own cancellable worker: there are two CPU workers per
terrain. This change does not replace global plans with independent regions,
bound arbitrary edit uploads, or pool workers across many terrain components.
The existing GPU generation/sample budgets and all terrain shaders are unchanged.

`HELIO_VOXEL_INLINE_ADMISSION=1` selects the inline control in the current
binary. `HELIO_VOXEL_RETARGETING=1` independently enables experimental whole-plan
retargeting. Both are unset in the ordinary configuration.

Flight CSVs distinguish render preparation/poll duration, accepted worker
service duration and producer update duration. Durations are elapsed wall time
on the indicated thread, not hardware CPU-cycle measurements. Worker service
excludes queue wait; missed polls have no attributed worker/update sample.
`pipeline_misses` counts frames with no prepared bundle, including startup.

## Same-binary measurements

Windows, Ryzen 5 3400G, RTX 3060/Vulkan; release Rust 1.95. These are profiled,
synchronized offscreen full-graph diagnostics with voxel sunlight. They include
submission and GPU completion, exclude capture encoding/readback, and do not
measure presentation or a populated editor. Native quality still uses TSR.
No compiler or second terrain GPU workload ran during timing; preflight engine
counters had no reported sample over 1%, which is not exclusive GPU ownership.

The comparison binary `pipeline-v2.exe` had SHA-256
`f6343f80529786ebafc44d3d3112890921d6f8d0f76c9041285b42a08efabe59`.
That historical binary defaulted to inline admission and used
`HELIO_VOXEL_ASYNC_ADMISSION=1` for the worker. The current binary reverses the
default and uses the INLINE switch above; do not interchange their flags.

The first 1280x720 matrix ran worker/retarget, inline/deferred,
inline/retarget, worker/deferred in that order:

| Mode | Descent sync p95 / p99, ms | Render prepare p95 / p99, ms | Arrival, ms |
| --- | ---: | ---: | ---: |
| Worker, retargeting | 17.15 / 21.14 | 0.022 / 0.038 | 1506 |
| Inline, deferred | 21.61 / 27.74 | 0.985 / 8.767 | 1532 |
| Inline, retargeting | 24.79 / 26.13 | 8.473 / 9.848 | 875 |
| Worker, deferred | 17.28 / 20.70 | 0.020 / 0.027 | 1424 |

Camera movement is frame-indexed and unthrottled. Faster rendering shortens the
wall time spent descending and changes generation headroom. Worker retargeting
does not retain the inline mode's short arrival result; it remains experimental.

The deferred-demand repeat order was worker A, inline A, inline B, worker B:

| Run | Descent sync p95 / p99, ms | Render prepare p95 / p99, ms | Arrival, ms | Walk sync p95 / p99, ms |
| --- | ---: | ---: | ---: | ---: |
| Worker A | 16.29 / 22.26 | 0.017 / 0.023 | 1471 | 24.27 / 30.89 |
| Inline A | 18.64 / 23.41 | 0.896 / 7.235 | 1393 | 27.02 / 36.64 |
| Inline B | 20.60 / 25.93 | 0.966 / 11.423 | 1404 | 26.04 / 35.52 |
| Worker B | 16.37 / 18.65 | 0.017 / 0.024 | 1411 | 26.25 / 37.20 |

All four repeats generated 161,638 bricks, reused 383,834 and cancelled no plans.
Each worker run missed one startup poll. Descent whole-graph GPU p95 was
12.55/11.78 ms for the worker and 12.45/12.88 ms inline. The retained benefit is
bounded render preparation and reduced measured admission tails, not a shader
speedup or a general frame-time pass. Walking tails still fail the goal.

## Final default and movement checks

The final `pipeline-v3.exe` SHA-256 is
`7d802814ac5dcfc62f84efad54196a2105c49dd6ec7a0d16e72898f07b3db944`.
Its unrecorded runs confirmed worker admission without any admission flag:

| Metric | 1280x720 native | 1920x1080 Quality |
| --- | ---: | ---: |
| Completed frames | 1108 | 1112 |
| Descent sync p95 / p99, ms | 21.86 / 28.63 | 26.89 / 57.48 |
| Descent whole-graph GPU p95, ms | 14.45 | 18.57 |
| Descent terrain pass GPU p95, ms | 13.07 | 16.95 |
| Descent render prepare p95, ms | 0.034 | 0.032 |
| Walking sync p95 / p99, ms | 19.58 / 22.03 | 24.09 / 26.44 |
| Full arrival refinement, ms | 1534 | 1884 |
| Local settle after orbital 4 m brush, ms | 133 | 171 |
| Authored 1 m grid replacement, ms | 4411 | 4044 |

Quality renders terrain at 1440x810 before reconstruction. Resizing changes
output to 1344x756 or 1984x1116 respectively. The final runs show that the earlier
favorable descent repeat is not a reliable 60 FPS result. The largest descent
synchronized frame was 49.48 ms at 720p and 99.33 ms at 1080p; submission and
GPU-wait intervals both exceed worker preparation. No specific external cause
is established. Terrain GPU time remains above the 5 ms target.

The eight comparison runs plus two final runs completed 11,037 frames. Their
200 non-sentinel captures contain 192,003,696 primary pixels and the same number
of sunlight pixels, with no loading/exhausted primary or exhausted/invalid sun
statuses. Every profiled run reports zero query drops/overflows. Each lacks two
drained graph and two voxel frame samples at resize/end boundaries. Status
captures do not validate uncaptured frames, and nested GPU scopes are not added
to their parent scopes.

A separate final-default recording completed 1114 frames with 120 walking and
240 descent captures. Its 367 non-sentinel captures contain 338,510,592 pixels
per domain, with no invalid/loading/exhausted statuses. Recording changes worker
scheduling and is not timing acceptance. Inspection of walking, middle/late
descent and the local crater confirms persistent slope banding and stippling.
This is not full temporal visual acceptance, canonical far fidelity, or the
requested art direction. The current sparse-density far surface remains
noncanonical and can miss visible edit effects.

## Process memory diagnostic

`voxel_flight/measure_memory.ps1` samples the launched PID, records binary hash,
selected flags and counter timestamps, and leaves missing samples empty. The
paired v2 deferred runs were separate from timing. They had 19/21 process
samples and 18/19 valid GPU rollup samples respectively, over about half a
minute each. Peaks below are MiB and cover the entire process/graph/driver:

| Counter | Worker | Inline |
| --- | ---: | ---: |
| Sampled working set | 398.82 | 402.46 |
| Largest observed OS peak working set | 400.88 | 405.36 |
| Sampled private bytes | 2433.83 | 2413.14 |
| GPU dedicated/local usage | 1804.37 | 1795.17 |
| GPU shared/nonlocal usage | 154.25 | 152.25 |
| GPU total committed | 1958.68 | 1947.48 |

Private commitment is not resident CPU memory. Per-process GPU counters include
shared allocations and are not isolated terrain allocations; do not sum them
across processes or add overlapping columns. See Microsoft's
[counter scope explanation](https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/).
This pair does not show a large duplication from the worker, but cannot qualify
per-terrain overhead or long-duration memory behavior.

Logical terrain buffers/textures remain 570,928,328/7,372,816 bytes at 720p and
573,951,176/8,128,528 after resize. At 1080p Quality they are
578,761,928/9,331,216 and 581,291,720/9,963,664. The material pool remains
536,870,912 bytes. These logical reservations differ from whole-process counters.

## Correctness and reproduction

The focused residency suite passed 11 tests with one diagnostic ignored. The
combined canonical/regional/cache/appearance library suite passed 48 with three
diagnostics ignored. The final four pipeline tests passed after strengthening
same-grid paint/undo and brush-interior checks. A CPU simulated GPU consumer
checks ordered packets, no writes to visible slots, generated-before-published
payloads, canonical source/grid coherence, interrupted generation, rapid demand,
replacement, undo, resize and zoom. Mailbox tests cover coalescing, backpressure,
failure and cooperative teardown. These are not a full GPU fidelity proof.
Two full-graph tests pass, including actual rendering, source removal, resize
and optional post-lighting compatibility. The release flight builds.

```powershell
cargo build --release -j2 -p helio-default-graphs --example voxel_flight
$env:HELIO_VOXEL_FLIGHT_PROFILE = '1'
$env:HELIO_VOXEL_FLIGHT_SUN = '1'
Remove-Item Env:HELIO_VOXEL_INLINE_ADMISSION -ErrorAction SilentlyContinue
Remove-Item Env:HELIO_VOXEL_RETARGETING -ErrorAction SilentlyContinue
.\target\release\examples\voxel_flight.exe target/voxel-goal/worker 1280 720 native
$env:HELIO_VOXEL_INLINE_ADMISSION = '1'
.\target\release\examples\voxel_flight.exe target/voxel-goal/inline 1280 720 native
python -B crates/helio-default-graphs/examples/voxel_flight/analyze.py target/voxel-goal/worker
```

For memory diagnostics invoke `measure_memory.ps1 -Executable <binary> -Output
<new-directory>` separately. For movement recording unset PROFILE and INLINE,
set `HELIO_VOXEL_FLIGHT_RECORD=1`, and use another output directory. Keep all raw
captures, CSVs, manifests, executables and logs ignored under `target/voxel-goal`.

Next replace whole-cut completion with local revision-safe region transactions
and compatible visibility-aware derived surface records. Moving admission to a
worker establishes a useful boundary but does not solve generation latency,
far appearance, arbitrary-distance edit fidelity or populated-engine gates.
The planetary goal remains active and both integration PRs remain drafts.
