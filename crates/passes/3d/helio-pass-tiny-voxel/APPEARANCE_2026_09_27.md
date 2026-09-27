# Current-frame appearance experiment

An opt-in spatial resolve reduces fine face stippling at low cost in the tested
local slope. It does not solve silhouettes, missed thin features, far geometry,
streaming or the planetary goal. Keep it disabled by default. The precise local
cache remains a separate, slower experimental accelerator.

## Capture-order correction

`PassContext::begin_compute_pass` records to a separate command stream that the
serial graph submits before all graphics. Both the earlier surface-reference
capture and the initial appearance trial used it after deferred lighting in the
CPU graph order. They therefore read preceding-frame render attachments. Repeated
static frames could hide this mistake; changing the camera exposed it.

`begin_graphics_compute_pass` explicitly records compute in graphics order. It
rejects an active render pass and, in debug builds, chain-transparent callers.
The new GPU regression changes a rendered value on every frame: the old helper
returns the preceding value and the new helper returns the current value. Both
paths are tested, including the initial frame. This is not a general audit or
repair of every existing compute consumer in the engine.

The reference and filter now use the new helper. Completed reference captures
include schema 2 metadata declaring the stream and whether filtering is active.
The comparison tools reject historical captures and filtered reference means.
Historical surface reports are marked superseded for joint lighting/hit and
appearance claims. Primary hit-only checks and independent traversal timings
retain their original scope.

## Experimental resolve and graph integration

An optional factory list adds backend-owned passes after opaque lighting and
before fog, transparency and AA. Existing builder APIs remain compatible;
factories survive graph rebuilds at the internal render size. This hook is
backend-independent. Pulsar's backend trait does not yet expose this extra hook.

The tiny-voxel experiment reconstructs RGB from nine already-visible, separately
lit pixels. It checks solid-hit status, canonical material, position proximity
and the final voxel receiver marker. A smooth weight activates only when the
pixel footprint exceeds the authored voxel size. It does not average normals,
alter primary hits, write source data, use temporal history or blend sky into a
solid center. Its own output preserves the graph's RGBA8 or RGBA16F format.

The producer publishes its actual active hit/parameter buffers. Only matching
frame numbers and dimensions are accepted; absent/stale inputs use a copy path.
This is biased screen-space reconstruction, not a coverage representation. The
receiver marker identifies the backend, not a particular producer: this trial
is restricted to one tiny terrain producer per graph. Multiple producers,
populated mesh scenes and other formats need explicit validation and ownership
contracts before wider integration.

One output texture plus 176 dummy-buffer bytes adds 3,686,576 logical bytes at
720p RGBA8, or 7,372,976 at RGBA16F. Source buffers are shared handles. These bytes
are additional to `memory.csv`'s terrain-only accounting. Physical VRAM and CPU
memory are not qualified by this allocation calculation.

## Corrected image evidence

Release Vulkan captures on RTX 3060, 0.1 m voxels, 96x54 pixels. Eight fixtures,
two sun directions and two positions 2.5 cm apart produce 32 cases. An unfiltered
256-sample reference contains 42,467,328 subpixel samples. It remains a finite
sampling estimate using the existing engine material model, not a physical
lighting oracle or a convergence proof.

Control and filtered 16-sample runs use the same executable and poses. Across
2,654,208 paired samples, primary status/cell/face/material/ray/depth, albedo and
sunlight match exactly. Across center images, 50,588 resolved solid pixels and
45,032 sky pixels retain identical pre-AA output. All repeated-pose comparisons
and the 2,304 sampled CPU center-ray audits per run pass.

Linear RGB RMSE against the new 256-sample mean, first sun and position:

| Fixture | Control | Filtered |
| --- | ---: | ---: |
| Slope | 0.060316 | 0.027714 |
| Ridge | 0.052559 | 0.036647 |
| Cave | 0.045668 | 0.033455 |
| Thin wall | 0.040597 | 0.036463 |
| Destroyed wall | 0.040797 | 0.036493 |
| Cave close | 0.017965 | 0.017965 |
| Thin wall close | 0.013144 | 0.012545 |
| Destroyed wall close | 0.025459 | 0.025104 |

Whole-image error improves in 28 cases and is unchanged in four. Partial-coverage
error slightly regresses for ridge/light0/move1: 0.220838 to 0.221207. Its large
remaining value makes clear that the silhouette problem is unsolved. Two-pose
delta error never increases; slope/light0 improves from 0.044436 to 0.020979.
This includes real parallax and is not a temporal flicker or continuous-flight
acceptance metric. Inspected reference/filtered previews retain close voxel
steps but show remaining silhouette aliasing; the basic materials are not AAA
art-direction acceptance.

## Timing evidence

Same release executable, native resolution, canonical traversal, no local
surface cache, ray-traced terrain sun, ordinary full graph. Each on/off/off/on
run warms 32 frames, then measures 180 stationary and 180 moving frames.
Motion is a local cycle: 0.5 m sideways, 0.1 m vertically and 0.04 rad yaw.
No compiler or other terrain capture ran concurrently; a pre-run Windows GPU
counter sample had no engine above 1%. This is not proof of exclusive GPU use.
Recording is disabled during timings. All 180 moving-frame timestamp samples
arrived for each run, with zero readback drops or query overflows.

720p slope, milliseconds; synchronized time includes CPU submission and GPU wait
but excludes presentation and capture encoding:

| Run | Graph GPU median / p95 | Resolve GPU p95 | Sync median / p95 / p99 |
| --- | ---: | ---: | ---: |
| On A | 10.026 / 10.924 | 0.122 | 14.383 / 16.281 / 18.072 |
| Off A | 9.854 / 10.733 | — | 13.875 / 16.107 / 17.821 |
| Off B | 9.977 / 11.400 | — | 13.449 / 14.950 / 22.804 |
| On B | 10.000 / 11.017 | 0.121 | 14.631 / 17.397 / 18.814 |

At 320x180 the resolve's p95 is 0.039/0.036 ms; graph GPU p95 is
6.683/6.723 ms with filtering versus 6.681/6.188 without. The final 320x180
RGBA8 image changes at 47,191 of 57,600 pixels in the stationary pair, confirming
that the resolve reaches the displayed output. The effect depends on projected
voxel size; neither resolution should stand in for the other.

The 720p stationary outputs are byte-identical on/off and between repeated
controls: this camera's distance rule classifies all affected pixels as resolved.
The 0.12 ms measurement is therefore the copy/guard path, not the cost of
nine-neighbor reconstruction at 720p. It establishes no 720p image improvement.
Foreshortened faces still form strong stair bands and moire in inspected frames.
A distance-only voxel footprint is insufficient to decide face filtering.

A separate filtered 720p recording covers 180 local-motion frames. All
165,888,000 primary samples are solid, with no exhausted/loading results; all
sun samples are valid and non-exhausted. Frames 0, 60, 120 and 179 were inspected.
This verifies this local path's hit validity and documents the remaining bands;
it does not establish continuous visual stability or planetary movement.

One 720p synchronized p95 exceeds 16.67 ms. No 1080p, presentation, populated
editor, destruction latency, physical memory or space-to-ground gate is met by
these local tests. The filter's stage time is not the terrain's total cost.

## Reproduce and retained files

Build `voxel_flight` with `--release --features voxel-reference,voxel-appearance`.
Set `HELIO_VOXEL_SURFACE_REFERENCE=4` or `16` for 16/256 spatial samples, and
`HELIO_VOXEL_APPEARANCE_FILTER=1` only for the candidate. Use 96x54 native.
Run `analyze_appearance.py CONTROL FILTERED REFERENCE` and
`analyze_surface.py CONTROL REFERENCE` with `python -B`. For timings unset
`HELIO_VOXEL_SURFACE_REFERENCE`, set `HELIO_VOXEL_CACHE_BENCH=slope` and
`HELIO_VOXEL_FLIGHT_PROFILE=1`, and keep recording disabled.

GPU tests: `helio-core --test graphics_compute_order` (one test) and
`helio-default-graphs --features voxel-appearance --test voxel_pass_graph`
(two tests, including source removal/resize). Both pass. The latter is a graph
compatibility check, not an image-quality or populated-editor test.

Raw evidence remains ignored under `target/voxel-goal`: `appearance-v3.exe`,
`appearance-reference-v3-16`, `appearance-control-v3-4`,
`appearance-filtered-v3-4`, `appearance-analysis-v3.json`, and
`appearance-perf-{320,720}-{on-a,off-a,off-b,on-b}` and
`appearance-motion-720-v3`. Executable SHA-256:
`2fc99a32d90e13e8d125129738c91bb6a16d2333604aeb23ccbea2b3f3cdff67`.
Generated captures, reports, logs and binaries are not committed.

## Next architecture decision

Keep the cheap resolve experimental while building visibility-aware far records
that preserve depth, face/material contribution and coverage. They must respond
to ordered edits without regenerating the planet. A local image filter cannot
substitute for that data or for revision-safe, demand-driven regional residency.

[Stochastic Texture Filtering](https://arxiv.org/abs/2305.05810) motivates
filtering separately shaded contributions. [Voxel-based filtered appearance](https://cjsb.github.io/hpg2023/voxel-filtered-appearance.pdf) offers directional
appearance and opacity ideas; its storage/build requirements are not an adopted
planetary solution. These are research motivations, not evidence that this
particular reconstruction satisfies the goal.
