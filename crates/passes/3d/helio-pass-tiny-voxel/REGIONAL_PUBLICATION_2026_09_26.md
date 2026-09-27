# Regional publication experiment

Generated captures, logs, CSVs, manifests and archives are local-only in the ignored `validation/` directory. They are not included in the branch; paths below identify local artifacts.

This opt-in experiment publishes generated regions before a replacement view cut finishes. It demonstrates revision-coherent partial publication, but is rejected for default adoption: mixed sparse-density representations expose visible patch boundaries, and arrival, fidelity and performance gates remain unmet. `regional-publication-experiment` is disabled by default. The implementation builds on `66100ff8`; the archived source snapshot and executable hashes identify the tested candidate.

## Representation and safety

The renderer retains the last complete tree and the target tree. A readiness tree counts unfinished jobs under each node. Composition selects ready target regions and keeps old coverage elsewhere. Nearby jobs are generated first, ordered by distance to the plan's camera. This ordering is not refreshed against later camera positions.

A refined region can temporarily reference an old ancestor brick. Its link stores a 16-bit brick slot and a five-bit ancestor-level difference. GPU traversal clips against the region while sampling the brick in its original bounds, reconstructed relative to the unchanged root. Certified air/solid regions retain their existing meaning. Equivalent sibling fallbacks collapse to avoid unnecessary traversal. No larger renderable voxel size is introduced by this metadata.

Both complete cuts remain reserved in the existing 65,536-slot pool. The composed tree cannot exceed the sum of their node counts; the experimental node buffer reserves twice the original capacity, adding 1 MiB. Source identity, authored grid and root bounds must remain compatible. A different world Arc, an edit, grid replacement or root change retains the existing atomic whole-cut publication. This version does not cancel obsolete plans, interleave source revisions, or solve local edit latency.

Only cuts that actually reference ancestor payloads select the additional decoding shader specialization. Completed cuts use the ordinary specialization. Primary and sunlight diagnostic replays use the same choice. CSV counters include total regional publications, currently visible nodes and ancestor fallback regions. `bricks` and `pixel_budget` continue to describe the last complete cut while a partial cut is visible.

## Correctness evidence

- 33 focused terrain tests passed with regional publication before specialization; 34 passed with both regional publication and the canonical reference feature after specialization. The default path passed 27 tests. One CPU benchmark was ignored in each suite.
- CPU tests cover early child publication, unfinished coarsening, mixed randomly generated trees, negative/non-aligned root coordinates, slot 65,535, source/grid/root changes and the transition back to the ordinary shader specialization.
- GPU rays compare one ancestor payload with eight clipped references: axis, diagonal and nearly parallel rays, levels 1/4/21, first-cell and depth agreement, empty planetary traversal, and a newly empty child that must not expose the old payload inside its replaced region.
- The retained specialized 1280x720 native flight completed 1,168 frames and 488 regional publications; 476 frames referenced ancestor payloads. The maximum composed tree had 101,593 nodes and maximum fallback count was 13,793. Its 367 post-sentinel movement captures had zero loading/exhausted primary results and zero invalid/exhausted sunlight results. These are traversal checks, not image-quality acceptance.

The two earlier 320x180 recorded runs completed 738/734 frames; the earlier always-decoding 720p variant completed 1,168. Their successful traversal audits and raw data are retained alongside the final variant. Initial low-resolution comparisons suggested decoder overhead in settled frames, motivating separate shader specializations; they do not establish a controlled performance effect.

The matching 720p whole-cut control completed 1,164 frames and also passed all 367 primary/sunlight capture checks. It reached complete arrival refinement in 2,686.04 ms versus the regional candidate's 3,377.66 ms. Recorded descent frame p50/p95 was 12.36/22.50 ms for the control and 19.31/31.75 ms for the candidate. Earlier visible updates did not translate into lower complete-refinement latency or cheaper motion in this run; the visible cuts and scheduling also differ, so these are diagnostic workload results rather than an isolated shader benchmark.

A combined-feature 320x180 flight completed 737 frames, 20 partial publications and 20 post-sentinel capture audits. All 864 sampled CPU/GPU coverage and first-cell results agreed. This verifies a narrow compatibility case with the canonical reference renderer; it does not prove transient image continuity. Its primary GPU p95 was 36.58 ms at 200 m and 72.40 ms in orbit, confirming that this reference path remains too expensive for runtime use. Orbital f32 depth error still reached 6.41 cm.

## Measured limitations

These are instrumented, recorded, synchronized offscreen diagnostics on RTX 3060/Vulkan. Recording/readback changes selection scheduling. CPU builds and an older heap probe overlapped parts of the experiments; background GPU use was not excluded. Presented cadence and uncontended acceptance timing remain unmeasured.

| Retained 720p observation | Result |
| --- | --- |
| First partial publication after 200 m view change | 52.15 ms |
| Full refinement for that view | 2,596.87 ms |
| Full refinement after recorded descent stops | 3,377.66 ms |
| Orbital brush edit plus inspection-view change | 237.84 ms |
| Replacement with 1 m authored grid | 4,414.20 ms |
| Maximum recorded logical terrain buffers + textures | 583,128,280 bytes |
| Settled primary GPU p50 / p95 at 200 m | 7.96 / 9.19 ms |
| Settled primary GPU p50 / p95 at 1 km | 5.90 / 6.46 ms |
| Settled primary GPU p50 / p95 in orbit | 9.79 / 12.18 ms |

First publication can update an arbitrary selected region; it does not prove that all resolvable arrival detail is ready. Logical memory excludes CPU trees/caches, pipeline objects, driver allocations and other engine attachments. Primary tracing alone exceeds the 5 ms total-terrain target. Whole-frame and stage distributions are in the analysis archive; neither median values nor validity counters override the unmet p95/p99 gates.

The existing far field still approximates canonical occupancy. Of 144 sampled rays per settled pose, first-cell differences were 51 at 200 m, 111 at 1 km and 136 in orbit. Corresponding hits in canonical air were 29, 57 and 80. Orbital depth error reached kilometres. Nearby returned-ground/edit/1 m samples agreed, but that does not qualify the far representation. Visual inspection still shows green contour/noise artifacts. Camera jitter can depend on frame count, so comparing different runs' hit cells is only valid when their logged rays also match.

## Next architecture work

Inspecting `descent-180.png` exposed abrupt patches between representations in the regional candidate. The same-pose control also has severe contour aliasing but different stale geometry. Both have valid ray-status outputs: validity alone cannot certify continuity. Region publication needs surface data that agrees across refinement boundaries before this mechanism is suitable for normal rendering.

The whole-plan lifetime is still a bottleneck: partial output arrives early, but selection cannot replace an obsolete target until all its jobs finish. The next candidate should maintain a bounded resident hierarchy with ready parent coverage, publish local refinement/coarsening transactions, reclaim slots using actual visible ownership, and reprioritize or cancel generation for the current view. Its capacity proof must include fallback ancestors and in-flight generation. Repeatedly composing an unbounded union of successive cuts is not acceptable.

This scheduler work remains separate from selecting a faithful far representation. Sparse interpolated density has failed canonical fidelity; direct canonical tracing has failed runtime cost. Both results remain part of the decision. Edits need explicit regional revision/ownership rules, and engine consumers must continue to agree with the canonical component data.

## Reproduction and evidence

```powershell
$env:HELIO_VOXEL_FLIGHT_PROFILE = '1'
$env:HELIO_VOXEL_FLIGHT_RECORD = '1'
$env:HELIO_VOXEL_FLIGHT_TRACE_WORK = '1'
$env:HELIO_VOXEL_FLIGHT_CANONICAL = '1'
$env:HELIO_VOXEL_FLIGHT_SUN = '1'
$env:HELIO_VOXEL_FLIGHT_SUN_WORK = '1'
cargo +1.98 run --release -p helio-default-graphs --features helio-pass-tiny-voxel/regional-publication-experiment --example voxel_flight -- target/voxel-regional 1280 720 native
```

Omit the feature for the ordinary whole-cut control. Remove recording and audit flags for timing experiments, and explicitly establish machine contention and independent repeats before acceptance measurements. The full local movement viewer is under `target/voxel-goal/regional-specialized-720/movement.html`.

Raw evidence (local `validation/2026-09-26-regional/evidence.zip`), entry hashes (local `validation/2026-09-26-regional/manifest.json`), binary hashes (local `validation/2026-09-26-regional/binary-hashes.json`), and movement hashes (local `validation/2026-09-26-regional/movement-frame-hashes.json`) preserve the experiment. All goal gates remain as written in `PLANET_TERRAIN_GOAL.md`; this candidate is not enabled in the engine by default.
