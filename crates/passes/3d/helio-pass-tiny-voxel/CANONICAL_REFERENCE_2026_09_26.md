# Canonical far-tracing experiment

The opt-in `canonical-far-experiment` feature evaluates authored cells directly inside far leaves. It is a fidelity diagnostic and architecture experiment. Its measured cost disqualifies it as the normal terrain renderer. The default renderer retains the existing sampled-density representation and its known far-field errors.

## Source and publication

The experiment uses the same integer recipe and ordered edits as CPU queries, the selected 10 cm through 1 m authored grid, and conservative empty-space steps. Certified air/solid nodes and nearby exact bricks remain in use. Exhaustion stays explicit.

A separate edit buffer belongs to the **published** snapshot. Generation can target the next revision without changing the source seen by rays through the previous complete cut. This adds 2 MiB plus 32 bytes under the feature; the corrected logical allocation report includes both buffers. Far density bricks are still generated and the 512 MiB material pool remains allocated. This is not a memory or streaming improvement.

Edits are scanned linearly. That is not scalable for a heavily modified planet. The generic component contract and saved recipe format do not change.

## Precision and observed fidelity

Rounding `rd * distance` into one position loses sub-cell bits at orbital ranges. The experiment preserves whole-cell displacement and a compensated fractional remainder. At 320x180, the initial trace selected the wrong first cell on 25 of 136 occupied orbital samples. Inline compensation matched all 136, with identical origin and direction bits for all 144 rays.

A rewritten helper regressed 21 orbital cells even though its isolated position test passed. Restoring the original addition order restored agreement on those same rays, but exposed a small local-origin rounding error in the isolated test. The subsequent candidate preserves the camera-fraction order while compensating local-origin displacement separately. The cause of the integration-specific arithmetic sensitivity is not established; this is not a claim of a compiler defect. Failed runs remain in the evidence archive.

The inline-compensated 640x360 sunlight flight completed 924 frames and recorded 368 images. The 367 terrain captures after the composition sentinel had no loading or exhausted primary rays; all sunlight captures had no exhausted or invalid outputs. The six settled poses supplied 864 CPU reference rays: all coverage results and all 812 occupied first cells agreed. Orbital depth still uses a single f32 distance: p95 error was 4.90 cm and maximum 6.63 cm despite correct first cells.

These are sparse first-hit checks. They do not establish supersampled coverage, silhouette fidelity, shadow fidelity, all-distance precision or populated-editor acceptance.

The retained source passed 28 terrain tests (one benchmark ignored), including 4,096 GPU position cases with the original 2-micrometre error bound. Maximum observed position error was 0.000000029802 m. Its 734-frame 320x180 sunlight flight matched all 864 sampled coverage/first-cell results; all 20 post-sentinel capture audits passed. This final flight uses the retained helper, whereas the larger movement recording above uses the earlier inline form. The feature-disabled control completed 735 frames and passed its traversal audits, while continuing to report canonical far-field differences.

## Cost and visual decision

Recorded/instrumented RTX 3060 runs are diagnostics, with concurrent CPU builds and no new claim of an uncontended machine. For the last 60 settled frames at each pose in the 640x360 flight:

| Pose | Primary GPU p50 / p95 | Sun GPU p50 / p95 | Synchronized frame p50 / p95 |
| --- | ---: | ---: | ---: |
| 200 m | 91.40 / 98.99 ms | 21.22 / 23.35 ms | 121.79 / 134.16 ms |
| 1 km | 80.46 / 95.81 ms | 21.38 / 23.75 ms | 108.51 / 128.71 ms |
| Orbit | 198.74 / 213.18 ms | 3.00 / 3.53 ms | 207.86 / 222.81 ms |

At 200 m, rays averaged 317.7 canonical field evaluations and reached 5,549. Even at 320x180 without sunlight, primary tracing cost roughly 29–58 ms p50. The goal is 5 ms p95 terrain GPU time at the accepted render configuration. Direct per-ray recipe evaluation misses that target by a wide margin.

Visual inspection still shows strong contour patterns and noisy subpixel shading. Correct sampled occupancy does not meet the requested Lay of the Land appearance or temporal stability. A bounded, cached far representation derived from canonical occupancy is still needed. This experiment does not select or qualify that representation.

## Reproduce

```powershell
$env:HELIO_VOXEL_FLIGHT_CANONICAL = '1'
$env:HELIO_VOXEL_FLIGHT_PROFILE = '1'
$env:HELIO_VOXEL_FLIGHT_TRACE_WORK = '1'
$env:HELIO_VOXEL_FLIGHT_SUN = '1'
$env:HELIO_VOXEL_FLIGHT_SUN_WORK = '1'
cargo +1.98 run --release -p helio-default-graphs --features helio-pass-tiny-voxel/canonical-far-experiment --example voxel_flight -- target/voxel-reference 320 180 native
```

Start small because the path is expensive. Omit the feature to exercise the default renderer. Record movement separately with `HELIO_VOXEL_FLIGHT_RECORD=1`; playback rate is illustrative.

[Raw evidence](validation/2026-09-26-reference/flight-evidence.zip), [entry hashes](validation/2026-09-26-reference/manifest.json), [binary hashes](validation/2026-09-26-reference/binary-hashes.json), [run dispositions](validation/2026-09-26-reference/dispositions.json), and [movement hashes](validation/2026-09-26-reference/movement-frame-hashes.json) preserve successful and rejected runs. Executables and the full movement viewer remain under local `target/voxel-goal`. The historical run named `canonical-far-final` was rejected.

The broader goal remains open: canonical filtered far rendering, regional publication and arrival latency, bounded edit/streaming cost, physical memory measurements, 1080p/720p frame-time targets, populated-engine integration and the intended visual style.
