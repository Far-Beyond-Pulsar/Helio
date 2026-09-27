# Canonical region bounds and corner traversal

Generated captures, logs, CSVs, manifests and archives are local-only in the ignored `validation/` directory. They are not included in the branch; paths below identify local artifacts.

This checkpoint reduces unnecessary brick selection without changing the authored voxel field, and fixes a secondary-ray boundary cycle found by recording movement. It does not qualify frame time, arrival latency, far-field fidelity or the requested visual style.

## Conservative bounds

When a query stays inside one value-noise lattice cell, CPU and WGSL now bound its actual eight coefficients over the query's fixed-point interpolation fractions. Each interpolation is monotone in its endpoint values; for fixed endpoints, its extrema over the fraction interval occur at the endpoints. Applying interval interpolation along all three axes therefore encloses the integer field, including downward rounding for negative differences. Boxes crossing a lattice boundary retain the existing global bound.

The sample function, recipe revision, saved world and edit semantics are unchanged. Tighter certificates allow selection to reject more air and solid interiors before allocating bricks. The fixed 512 MiB material pool is unchanged; fewer requested bricks do not establish a physical VRAM saving.

Three alternating runs of the saved baseline and candidate CPU selection binaries produced these deterministic counts. Timings are medians in milliseconds; “cold” means an empty classification cache, not engine startup.

| View | Bricks before / after | Change | Cold ms before / after | Cached ms before / after |
| --- | ---: | ---: | ---: | ---: |
| Ground | 31,090 / 29,299 | -5.76% | 101.29 / 100.25 | 104.76 / 102.71 |
| Walk +2 m | 31,068 / 29,286 | -5.74% | 96.49 / 100.47 | 40.76 / 36.42 |
| Walk +4 m | 31,025 / 29,255 | -5.71% | 97.04 / 101.75 | 40.80 / 34.90 |
| 200 m | 29,870 / 22,984 | -23.05% | 83.88 / 93.62 | 46.71 / 40.37 |
| 1 km | 30,254 / 25,408 | -16.02% | 65.98 / 68.39 | 32.22 / 29.54 |
| Orbit | 14,917 / 14,917 | 0% | 16.04 / 19.35 | 21.09 / 8.04 |
| Return | 31,090 / 29,299 | -5.76% | 92.22 / 100.77 | 46.16 / 34.44 |

Cold selection is slower at several poses. The adaptive selection may spend the capacity gained on finer detail: the Quality flight's initial pixel budget changed from 2.289 to 1.831, with 27,665 versus 25,604 bricks. These observations do not establish a faster GPU renderer or equal-quality frame-time comparison.

## Boundary cycle exposed by movement

The first candidate recording failed at walking frame 27: one sunlight ray exhausted 2,048 leaf visits after only 3.7733464 m. A separate sunlight-work replay reproduced it at pixel 403554. A small GPU regression then reproduced the same failure in an entirely empty, subdivided region, independently of terrain generation.

At an almost simultaneous Y/Z boundary crossing, adding a secondary ray's camera-relative origin could round the other coordinate behind a plane already crossed. The next leaf crossing reversed that ownership again, creating a cycle. Traversal now preserves monotone integer ownership on every axis. It keeps the original ray and boundary parameter; the visit limit is unchanged.

The regression fails before the fix and passes after it, including 63 nearby floating-point perturbations. The optional sunlight replay exports failed rays and traversal maxima outside normal rendering. It casts from terrain hits and does not account for later mesh coverage or establish shadow fidelity.

## Validation

Parent: Helio `0faa86ff`. Windows, Rust 1.98.1, Vulkan/RTX 3060 driver 616.64, Ryzen 5 3400G. Evidence and executable hashes are in validation/2026-09-26-bounds (local `validation/2026-09-26-bounds`).

- Terrain library: **27 passed, one CPU benchmark ignored**. The benchmark was separately run in six alternating comparisons.
- Noise intervals: **904,932 exact samples**, including 5,822 single-cell and 1,090 crossing boxes, plus explicit lattice-boundary and extreme-coordinate checks.
- GPU certificate parity: **5,344 regions** across procedural and nested-edit cases; **546,940 exact cell checks**, including **465,660 exhaustive checks**, with zero certificate, height or occupancy errors.
- Voxel-field integration: **2 passed, one older frozen-data qualification ignored**.
- Corrected 1080p Quality recording: **1,124 frames**, **368 primary/sunlight capture audits**, and primary replay checks. No exhausted/loading primary hits after initial loading, no invalid/exhausted sunlight output.
- Corrected 720p native diagnostic: **1,165 frames**, **21 primary/sunlight capture audits**, primary replay checks and 864 exact CPU rays. No invalid/exhausted captured results. This run did not record every movement frame.

The Quality run samples 864 exact CPU rays. All sampled hit/miss classifications agree. Returned-ground, orbital-edit and 1 m captures match all sampled occupied first cells. At 200 m, only 97/144 first cells agree; 25 reported GPU cells are canonically air. At 1 km, 34/144 agree and 57 reported cells are air. At orbit, 78 of 136 reported hits are canonically air and depth-error p95 is about 1.249 km. None of these ray sets match the prior Quality capture's jittered directions exactly, so they are individual oracle comparisons, not paired improvement measurements.

Recording, replay, CPU oracles and profiling perturb scheduling. These flights do not qualify performance. The returned images still show dark contour bands and aliasing. The remaining independent far-density representation, regional publication, arrival/edit latency, shading, populated editor and simulation gates remain open in [the full goal](PLANET_TERRAIN_GOAL.md).

In the instrumented Quality recording, ground refinement after descent still took 2.219 s, the edit settled in 245 ms, and grid replacement took 4.516 s. These are diagnostic observations, not accepted latency measurements. They do not meet the goal's arrival/edit targets.

To reproduce the recorded Quality diagnostic after building the release `voxel_flight` example, set `HELIO_VOXEL_FLIGHT_SUN`, `SUN_WORK`, `TRACE_WORK`, `CANONICAL`, `PROFILE` and `RECORD` (each with the full `HELIO_VOXEL_FLIGHT_` prefix) to `1`, then run with arguments `OUTPUT 1920 1080 quality`. The generated `movement.html` uses local captured frames and illustrative playback speed, not measured FPS.
