# Planet terrain profiling checkpoint — 26 September 2026

This checkpoint adds engine-native measurement and traversal correctness evidence. It does **not** qualify a faster renderer or AAA terrain. Empty-brick skipping is disabled by default, and the parent-link experiment was reverted after repeatability failures. Both integration PRs remain drafts.

## Retained changes

- Optional per-stage timestamps use Helio's existing GPU profiler: generation, primary traversal, GBuffer and directional visibility. Samples retain the producing frame ID through asynchronous readback and graph rebuilds. The whole-graph duration is its enclosing GPU scope, not a sum of child passes.
- The flight CSV separates CPU submission and GPU waiting and records wall-clock start times. Logical terrain allocations are exported separately from process VRAM. The analyzer reports missing samples and sums overflow/drop counters across profiler epochs.
- Capture-time diagnostics replay the **actual primary direction bits** through accelerated and unaccelerated traversal of the same resident cut. All 32 hit bytes must match; work counts are written to a separate buffer.
- Exact bricks receive a nonempty certificate. Its full-payload GPU tests cover empty bricks, the last occupied cell, and clearing a reused slot. Maximum allocation cost is 256 KiB; the reduction also has generation cost. Normal traversal does not use the experimental skip.
- Traversal resets to the shared leaf exit plane. Previously accumulated fine-DDA rounding residue could change a later far hit by 0.052 m when an empty region was skipped.
- An optional sunlight flight audits actual binary16 visibility output for exhausted or invalid rays. The graph test now exercises the sunlight binding path.

## Evidence and limits

Platform: Ryzen 5 3400G, RTX 3060 12 GB, 16 GiB RAM, Windows Vulkan driver 616.64. Release builds used Rust 1.98.1. Parent integration revision was Helio `156f6a7b28e3ef224facbd950d9b6f9753c327cc`. See [binary hashes](validation/2026-09-26-goal/binary-hashes.json), [raw evidence archive](validation/2026-09-26-goal/flight-evidence.zip) and its [per-entry manifest](validation/2026-09-26-goal/manifest.json). The archive includes failed trials, profiling data, compiler/test logs and the rejected parent-link source. Large executable/Nsight artifacts remain under the local `target/voxel-goal` directory.

The original profiling diagnostic attributed walking primary traversal p50/p95 to 13.452/14.478 ms, GBuffer p50/p95 to 1.128/1.544 ms, and whole-graph p50/p95 to 16.252/17.219 ms. Those stage measurements identify where time went in that flight; they are not presented-frame performance or populated-engine acceptance.

Empty skipping reduced returned-ground mean exact-cell checks from 551.597 to 120.573 while preserving complete hits. It did not reliably meet the predeclared p95 gates. The video-paused repeat still failed: one candidate had walking primary p95 18.114 ms and descent synchronized p95 29.017 ms. The second candidate's walking p95 was 7.471 ms, demonstrating why favorable averages or a single repeat were insufficient.

The parent-link follow-up passed all captured hit comparisons but also failed qualification:

| 720p walking primary GPU p95 | Pair 1 | Pair 2 |
| --- | ---: | ---: |
| Matched control | 14.000 ms | 18.587 ms |
| Parent links + empty skip | 7.713 ms | 16.879 ms |
| Improvement | 44.9% | 9.2% |

Both pairs needed at least 20%. The independent full-graph/descent bounds passed for this follow-up, but do not override its failed primary gate. See [experiment protocol and dispositions](EMPTY_BRICK_EXPERIMENT.md).

Auditing the actual slow walking frame found no increased traversal work or changed hits. Holding its pose let the transient subside. Observed GPU clocks remained near 1.95 GHz. Neither video playback nor the private ancestor stack was established as the cause. A per-process monitor changed timing enough that the slow samples did not reproduce. Nsight Systems Vulkan capture succeeded; hardware-counter access failed with the driver's insufficient-privilege error. No system profiling permissions were changed.

## Validation of retained source

- Engine release tests: **25 passed, 1 ignored** (the ignored test is a CPU benchmark).
- Full deferred-graph GPU test with sunlight enabled: **1 passed**.
- Analyzer checks: source attribution and summed counters after a profiler reset passed; duplicate source frames were rejected.
- Final 1920×1080 Quality flight with sunlight, profiling and exact primary replay: **1,180 frames completed**. All **21 captures** had zero exhausted/invalid sunlight rays. Complete primary hit equivalence, terrain coverage, graph composition, graph rebuild, orbital edit and the 1 m grid checks passed. Timings from this heavily instrumented run are **not acceptance measurements**.
- The final source's returned-ground sunlight readback contained 1,062,857 visible and 103,543 occluded pixels, with zero exhaustion/invalid output. This checks output validity, not correct shadow appearance or far canonical geometry.

Inspected [ground](validation/2026-09-26-goal/returned-ground.png) and [orbital edit](validation/2026-09-26-goal/orbital-edit.png) captures still show strong dark contour bands, aliasing and limited material/content composition. They fail the requested visual direction. This checkpoint does not supply a new accepted movement sequence. The preceding recorded movement evidence and its failures remain applicable; broad movement, arrival, far-fidelity, collision and populated-editor gates in [the active goal](PLANET_TERRAIN_GOAL.md) remain open.

## Engine profiler integration

Pulsar already collects CPU/thread instrumentation, stores sessions, and reads Helio's `RenderTimingSnapshot` into its GPU diagnostics. The new flight exporter uses that same Helio timing system. Pulsar's current `emit_helio_gpu_passes` reconstructs the GPU lane near readback time and places child durations sequentially; it is not a calibrated CPU/GPU execution timeline. Use it for engine context without inferring precise cross-domain overlap. Carry source-frame identities and terrain sub-stage detail into that workflow in the next integration pass; use Nsight only for questions below the engine's instrumentation.
