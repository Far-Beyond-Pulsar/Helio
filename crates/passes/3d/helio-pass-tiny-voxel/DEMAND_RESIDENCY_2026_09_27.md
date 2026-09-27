# Demand scheduling and lazy brick allocation

Whole-plan retargeting improves arrival refinement but regresses descent frame
time. It remains an opt-in experiment (`HELIO_VOXEL_RETARGETING=1`). The default
retains deferred camera demand. Neither mode meets the planetary acceptance
gates. Generation, far geometry and lighting shaders are unchanged by this work.

## Retained storage changes

Missing bricks occupy a job descriptor until admitted to the existing bounded
GPU generation batch. They consume no payload slots. Valid cached bricks stay
reusable, and cancellation drops only the remaining descriptors. Each generated
job patches its target node before that node can become publishable. The GPU
generation commands precede publication/traversal on the same queue.

Slot ownership follows physical slots, including superseded source versions,
instead of only spatial keys. Visible predecessors remain pinned. A retirement
list releases superseded versions at complete publication without scanning the
entire physical pool. Eviction makes room for missing payloads rather than
reserving another slot for every reused leaf. Source edit identities survive
camera movement; append, undo, replacement and grid changes still invalidate
intersecting data. Internal integer lookup tables use the existing Fx hasher.

The optional regional experiment still finishes its current two-cut transaction
before admitting another. It cannot cancel a partially published union safely
without a stronger resident-region ownership contract. Its geometry/seam and
performance limitations from the earlier report remain.

## Selection worker and experimental retargeting

There is one executing selection plus one replaceable request/result mailbox.
Selection checks cancellation before each region classification, also during
edit-cache invalidation, and retains reusable classifications. An interrupted
invalidation does not claim the new edit identity. The classification cache is
bounded during capacity retries as well as between requests. Dropping residency
wakes the worker and requests cooperative shutdown without joining on the
render thread.

An initial complete cut bootstraps coverage. After that, the opt-in mode queues
new camera/source demand while older generation is incomplete. It consumes a
useful completed camera plan from the same source before submitting another;
discarding every previous camera result starved generation under continuous
motion. Current generation continues until a newer plan is available, except
that source replacement cancels it immediately. Completed payloads remain
cacheable. These rules still operate on whole plans, not independent regions.

## Diagnostic development results

The first eager-retargeting trial reserved all missing slots before generation.
One 720p comparison reduced arrival from 2585 to 1987 ms but increased descent
synchronized median/p95 from 11.01/20.05 to 25.50/32.02 ms. It discarded 1,658,121
repeated job reservations. That version is rejected.

Lazy allocation, targeted retirement and source-identity reuse reduced that
churn. In the last pre-default-selection binary (`demand-v4.exe`), the same-binary
comparison was:

| Metric | Deferred demand | Retargeting |
| --- | ---: | ---: |
| Descent synchronized p50 / p95 / p99, ms | 11.66 / 20.10 / 24.12 | 17.24 / 24.94 / 26.83 |
| Descent whole-graph GPU p95, ms | 13.71 | 12.83 |
| Descent residency admission p50 / p95, ms | 0.001 / 0.014 | 5.05 / 10.18 |
| Full arrival refinement, ms | 1609 | 942 |
| Generated bricks over the flight | 161,673 | 150,028 |

Admission statistics include frames without a completed plan to install. They
are render-thread demand work, excluding GPU generation and publication. The
remaining cost is recurring whole-plan admission on that thread. Cancelled-job
counters now count queued descriptors, including repeated requests for a key;
they are neither unique saved bricks nor an estimate of GPU time saved.

All profiled runs are diagnostic synchronized offscreen flights, not presented
frame or populated-editor acceptance measurements. They use directional voxel
sunlight, Vulkan, the RTX 3060 and the full graph at 1280x720 native quality.
Native quality still uses TSR. Resizing changes output to 1344x756. Camera
capture/readback work affects asynchronous selection even when excluded from
the per-frame timer. No compiler or second terrain GPU test ran during timing.

## Final binary repeats and checks

The retained `demand-v5.exe` makes retargeting opt-in. SHA-256:
`fcfdef72e1f0c5b1af7409d195e426db7155e7a5f3f17fff270c352dd430c6d6`.
Run order was default A, retargeting A, retargeting B, default B. All four ran
the same executable with no optional rendering features. Preflight Windows
counters showed about 1.2% DWM and 1.1% ChatGPT 3D activity; this is not a claim
of exclusive GPU access.

| Run | Descent synchronized p50 / p95 / p99, ms | Descent graph GPU p95, ms | Arrival, ms | Generated bricks |
| --- | ---: | ---: | ---: | ---: |
| Default A | 11.67 / 19.79 / 24.82 | 14.15 | 1431 | 161,638 |
| Retargeting A | 16.24 / 27.86 / 32.73 | 12.38 | 1226 | 153,282 |
| Retargeting B | 16.00 / 24.88 / 27.48 | 12.03 | 906 | 150,663 |
| Default B | 10.76 / 18.77 / 23.42 | 12.06 | 1383 | 161,638 |

This repeats the tradeoff and rejects enabling whole-plan retargeting by
default. Default walking synchronized p95 was 24.74/27.70 ms; retargeting was
20.07/27.72 ms. Neither mode passes overall frame-time gates. Final authored
1 m grid replacement still took 4.21-4.71 seconds. Local settling after the 4 m
radius orbital brush was 141-171 ms, above the 100 ms target; this is not an
all-distance fidelity proof.

The four runs completed 4,402 frames and 80 non-sentinel captures containing
74,861,568 primary pixels and the same number of sunlight pixels. Those captures
had zero exhausted/loading primary results and zero exhausted/invalid sunlight
results. They do not audit every uncaptured frame. Each run reports zero GPU
query drops/overflows; two graph and two voxel frames lack drained samples at
the resize/end boundaries. Keep the separate nested GPU scopes separate.

A separate retargeting recording completed 1,110 frames and captured every one
of the 120 walking and 240 descent steps. Its 367 non-sentinel captures contain
338,510,592 primary pixels and the same number of sunlight pixels, with zero
loading/exhausted/invalid results. This is a validity diagnostic; recording
changes worker scheduling. Inspection of walking step 119, descent steps 120
and 239, and the local crater confirms visible voxel detail and the persistent
banding. It is not full temporal visual acceptance or canonical far validation.

Logical terrain allocation in the ordinary binary is 570,928,328 buffer bytes
plus 7,372,816 texture bytes at 720p; after resize it is 573,951,176 plus 8,128,528.
The material pool remains 536,870,912 bytes. These are declared allocations, not
independent physical VRAM or CPU peak measurements.

Validation: release build succeeded; seven focused residency tests passed with
one diagnostic ignored; the combined optional-feature library suite passed
45 tests with three diagnostics ignored. Ownership tests interrupt generation,
change the source/grid, undo, continuously move, compare reused source samples,
and check that queued jobs own no slots, visible slots stay pinned, and every
occupied slot has a unique current or retired owner. Worker tests cover newest
request delivery, cooperative cancellation, partial invalidation and shutdown.

Reproduce the final comparison in PowerShell, using separate output directories:

```powershell
cargo build --release -j2 -p helio-default-graphs --example voxel_flight
$env:HELIO_VOXEL_FLIGHT_PROFILE = '1'
$env:HELIO_VOXEL_FLIGHT_SUN = '1'
Remove-Item Env:HELIO_VOXEL_RETARGETING -ErrorAction SilentlyContinue
.\target\release\examples\voxel_flight.exe target/voxel-goal/default 1280 720 native
$env:HELIO_VOXEL_RETARGETING = '1'
.\target\release\examples\voxel_flight.exe target/voxel-goal/retarget 1280 720 native
python -B crates/helio-default-graphs/examples/voxel_flight/analyze.py target/voxel-goal/retarget
```

Visual inspection of late descent and returned-ground images confirms that
coverage remains visible, but slope banding and the inaccurate sampled far
surface remain unacceptable. A valid ray status does not establish fidelity.
The shaders and sparse far representation have not been improved by scheduling.

## Next architectural boundary

Move demand admission off the render thread behind a bounded ordered pipeline.
Prepared generation/publication bundles must carry immutable source identities
and slot ownership. A bundle accepted for rendering cannot be silently dropped;
predecessor slots cannot be reused before the ordered publication that retires
them. Keep the render caller nonblocking and expose queue/backpressure timings.
Then replace whole-cut completion with independently publishable regions and
compatible derived surface data. Moving CPU work alone will not solve the
remaining generation latency or incorrect far geometry.

Raw executables, frame tables, captures and logs stay ignored under
`target/voxel-goal`. Only source, tests, the analyzer and this written checkpoint
belong in the branch.
