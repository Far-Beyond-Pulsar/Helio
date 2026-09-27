# Exact empty-space skipping in the local surface cache

The opt-in bounded surface cache now traverses certified empty 4-cubed
microbricks and 32-cubed tiles in one step. Occupied regions still resolve to
authored cells. No larger cube is rendered, and unavailable data still falls
back to the published canonical source. This improves a local accelerator;
the planetary performance, far appearance and streaming gates remain open.

## Retained implementation

The precise DDA previously compared the selected crossing plane against itself
on every step, invoking the integer predicate unnecessarily. It now advances
that axis directly. Bit-identical plane/direction pairs also return equality
without constructing wide products.

Skipping an empty box needs more than rounding the ray position at its exit.
That shortcut can select the wrong cell or entry face near a corner. The new
path selects the box exit using the existing dyadic predicates, then binary
searches the crossed grid planes on the other axes with the same comparisons.
It crosses truly coincident planes together and retains the original
lowest-axis face/depth convention. The ray bits remain unchanged.

`HELIO_VOXEL_SURFACE_CACHE_SKIP_OFF=1` selects plain per-cell DDA in the same
binary for comparisons. `HELIO_VOXEL_SURFACE_CACHE_OFF=1` still bypasses the
whole accelerator. Both controls retain its allocations. The cache adds no
new storage relative to the published bounded-patch baseline. An engine GPU
timestamp named `voxel_cache_precise` measures the cache dispatch separately
inside its parent `voxel_primary` scope; never add the child to the parent.

## Rejected split-pass experiment

A cheap float pass marked uncertain rays for precise replay. Compacting these
into an indirect GPU work queue did not improve the measured result. The queue
held 149,516 requests out of 2,654,208 reference samples; many replayed rays
still left the known cache prefix and fell back to canonical traversal. After
removing redundant predicate work, the single precise pass was faster than the
split pass. The extra pipelines, queue allocations, bindings, diagnostic flags
and public readback getter have been removed. Trial sources and raw evidence
remain ignored under `target/voxel-goal`.

The split design was checked again at 720p, because small dispatches alone are
not enough to select an architecture. It remained slower: cave motion graph
median/p95/p99 was 8.697/15.362/26.178 ms, and ground motion was
22.063/23.005/25.012 ms. The fast pass itself took 4.407 ms median in the cave
and 10.982 ms on ground. Another trial seeded an exact coordinate correction
with a float estimate; it passed 52,488 rays but made little timing difference
(cave graph median 7.509 ms, versus 7.619/7.736 ms with binary search).
Neither additional mechanism is retained.

## Small reference evidence

The actual primary GPU shader passes 52,488 adversarial rays against the
independent CPU plane-interval oracle, including 47,780 occupied hits. Both
skip settings are checked on 0.1/0.3/1 m grids at positive/negative planetary
anchors, from a mixed tile and a completely empty neighboring tile. Cases
include adjacent float directions around edges/corners. The publication test
also passed 149,288 material queries through source replacement, teleport,
undo and grid changes; that query count depends on worker scheduling.
The full feature library suite passed 42 tests with three diagnostic probes
ignored (`surface-cache-experiment,regional-publication-experiment`).

These are pooled 96x54 reference crops with 16 spatial samples per pixel, two
lights and two camera poses for each of eight fixtures. They are diagnostic
measurements, not screen-resolution performance forecasts. The final delayed
timestamps were not drained: each run has 575 graph and 574 primary/cache
samples for 576 reference frames. Other machine activity was not excluded
throughout these trials.

| Variant | Whole graph median | Primary median | Cache precise median |
| --- | ---: | ---: | ---: |
| Compact repair queue, before predicate fix | 13.489 ms | — | 3.771 ms |
| Split pass, predicate fix | 10.583 ms | 9.110 ms | 0.921 ms |
| Single precise pass, predicate fix | 9.738 ms | 8.445 ms | 0.861 ms |
| Single precise pass, exact empty skips | 9.191 ms | 7.703 ms | 0.311 ms |

The last two variants each match all 2,654,208 published precise reference
samples bit-for-bit in coverage, status, hierarchy level, cell, material,
face, original ray, depth, sunlight, albedo and lit color. Comparisons used
`compare_cache.py --strict` against `surface-patch-all-v10`. This preserves the
previous precision corrections; it does not independently prove the accuracy
of every matching sample or solve shadow-origin rounding.

The final build repeats all 32 cases at 10 cm and twelve close cases each at
30 cm and 1 m. All 4,644,864 samples match their corresponding published
precise captures under `compare_cache.py --strict`, including depth and lit
color. Final artifacts are `surface-skip-grid-{0.1,0.3,1.0}-v8` and
`surface-skip-comparison-*-v8.json` under the ignored evidence directory.

## Ordinary-graph local benchmark

`HELIO_VOXEL_CACHE_BENCH=cave-close` selects a settled fixture in the flight
example without installing the spatial-reference tap. It uses the requested
resolution, ordinary graph, native temporal reconstruction and raytraced sun.
There are 32 warm frames before each 180-frame stationary/motion stage. Motion
contains a continuous side-to-side cycle, vertical movement and yaw, then
returns to the initial pose. Four separately labelled drain frames allow the
asynchronous measured-frame timestamps to arrive.

Use `HELIO_VOXEL_FLIGHT_PROFILE=1` for stage diagnosis. Keep
`HELIO_VOXEL_FLIGHT_RECORD` unset for timing; setting it saves every motion
frame in a separate visual run. `analyze.py` verifies timestamp attribution
and reports frame distributions and query losses. The headless synchronized
frame metric excludes presentation and image readback. This fixture has no
populated editor, planet-scale streaming or far-appearance acceptance claim.

The retained binary-search path was repeated with the same binary at 1280x720
native, in cache-on/off/off/on order. Each row below is one 180-frame motion
stage. Every measured frame has graph and terrain timestamps; missing final
readbacks belong only to the separately labelled drain stage. No query
overflows or dropped readbacks were reported. Values use nearest-rank
quantiles from `analyze.py`.

| Cave variant | Graph median / p95 | Primary median / p95 | Synchronized frame median / p95 / p99 |
| --- | --- | --- | --- |
| Cache + skips A | 7.619 / 8.486 ms | 3.775 / 4.537 ms | 12.329 / 14.402 / 15.611 ms |
| Cache off A | 6.385 / 7.404 ms | 1.750 / 2.210 ms | 10.580 / 12.769 / 14.673 ms |
| Cache off B | 5.546 / 6.423 ms | 1.535 / 1.853 ms | 9.359 / 11.721 / 12.395 ms |
| Cache + skips B | 7.736 / 8.448 ms | 3.795 / 4.623 ms | 12.290 / 14.120 / 15.953 ms |
| Cache, plain per-cell DDA | 12.411 / 13.186 ms | 8.634 / 9.600 ms | 16.517 / 18.609 / 31.662 ms |

The cache completed all 921,600 rays in these settled cave captures. Empty
skipping improves its own traversal substantially, but **the canonical
control remains faster**. Keep the accelerator opt-in. A normal ground fixture
(`ground-close`) also shows that a cache hit rate below 100% leaves the cost
of unsuccessful prefixes plus canonical fallback. This is a reason to revisit
representation/integration rather than promote a local dispatch improvement
into a production claim.

The final same-binary ground run confirms the regression. Cache on/off motion
graph median/p95 was 14.897/16.039 versus 9.703/10.536 ms; primary median was
8.666 versus 3.579 ms. Synchronized frame median/p95/p99 was
18.429/20.577/25.228 versus 14.019/16.150/18.216 ms. Each measured stage contains
180 complete timestamp samples. The final settled cache capture completed
703,812 of 921,600 rays (76.37%). Both variants retain the 512 MiB material
arena; cache-off also retains the experimental allocations. Reported logical
terrain allocation is 578,272,520 buffer bytes plus 7,372,816 texture bytes.

A separate recorded ground run saved all 180 motion frames at 720p. Readbacks
cover 165,888,000 primary results with zero exhausted/loading results, plus
the same number of sunlight results with zero exhausted/invalid values.
Frames 0, 60, 120 and 179 were visually inspected: resolved voxel steps remain,
but the ground is still visually sparse and distant face stippling remains.
This is a short local sway/yaw check, not visual acceptance for walking,
streaming boundaries or orbital travel. Timing from the recorded run is
excluded from the table. The frames are in `surface-skip-motion-v8`.

The 720p builds used Rust 1.95.0, tests Rust 1.98.1, on RTX 3060/Vulkan driver
616.64. No compiler ran during these timing runs. An idle GPU-counter snapshot
reported no engine above 1%; this does not establish exclusive GPU use during
the runs. The on/off repeat variation remains visible above. Physical CPU/VRAM
residency is not measured by these logical allocation counters.

Raw evidence is local and ignored: `surface-skip-720-*-v5`,
`surface-skip-720-*-v6`, and `surface-split-720-*-v7` under `target/voxel-goal`.
The final source build is `surface-skip-v8.exe`, SHA-256
`9c2294f07860c4c64340e6dcbfc82c36662a7e67049e557fa9c90e5ba326b39c`.
The preceding repeat binary was `surface-skip-v5.exe`, SHA-256
`487f0f4d8b1523c7693e80642305a974f7ca7a124caa945f79d953f9f69c921f`;
the retained traversal is unchanged. The benchmark adds a ground fixture in
the final build.
