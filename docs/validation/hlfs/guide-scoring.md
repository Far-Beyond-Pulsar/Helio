# Guide scoring and confidence work

Continuation of [PR #248](https://github.com/Far-Beyond-Pulsar/Helio/pull/248), September 19, 2026. The baseline is the directional-budget fast path at `a1dba21f126e43a88934ed336b7e34f2905b5f7b`.

## Change

For tiles containing only local lights, guide importance does not depend on the sample's directional budget. With two to four samples, the sampler now scores each guide entry once and advances the stratified reservoirs together in four scalar lanes. It preserves guide order, each reservoir's addition order, probability and random-state remapping. One-sample and mixed-directional tiles keep the original scalar loop. There is no private score array, additional shared memory, dispatch or ray reduction.

The guide's unshadowed BRDF energy is only needed for a confidence bit. Its evaluation moves after selected-light shading. Zero covered energy cannot set that bit, so it skips the guide energy scan. For albedo bounded by one, all guide energy terms are nonnegative: once the partial sum exceeds the existing epsilon and `covered_energy / guide_energy < 0.8`, the remaining terms cannot restore confidence. The loop can stop at that point. HDR albedo can produce negative diffuse terms, so it retains the complete sum. The comparison uses the original division, threshold and accumulation order; no approximate reciprocal or changed cutoff is introduced.

Sampling scale, sample/candidate counts, proposal support, visibility queries, normalization and denoising are unchanged. This is a reduction of repeated scoring work, not tile presampling or ReSTIR.

## Validation and limitations

All [five RT regressions](rt-control/guide-scoring/rt-tests.txt), [14 ScreenSpace GPU regressions](rt-control/guide-scoring/screen-tests.txt) and [two configuration/WGSL checks](rt-control/guide-scoring/unit-tests.txt) pass with the final production shader.

The audit now accepts `HLFS_RT_AUDIT_SAMPLES=1..4` and includes an HDR material case alongside local/mixed grids, 1,024-light overflow and 65,536-light packed-ID overflow. Each case exports 16 frames of linear f32 RGB at 65x49, with moving lights, zero-energy lights, changing blockers and history reuse.

The verified baseline/candidate comparison passes at [one](rt-control/guide-scoring/audit-1.json), [two](rt-control/guide-scoring/audit-2.json), [three](rt-control/guide-scoring/audit-3.json) and [four](rt-control/guide-scoring/audit-4.json) samples. All four overflow cases are bit-identical at every sample count: 2,446,080 compared float values. Local-grid mean error is at most 0.123%, with 4.34-6.70% cross-run NRMSE; mixed-grid mean error is at most 2.06%, with 30.88-62.31% NRMSE. The one-candidate mixed-grid case remains noisy.

The comparator requires bit-identical overflow output. Atomic tile insertion order already varies between baseline runs, so tile-list differences are reported without treating those comparisons as a visual quality pass. The frozen scene/error protocol remains separate.

The rebuilt integrated 1440p sampled cathedral capture is visually unchanged and byte-identical to the preceding commit's frame 31. The executable was verified to embed the final shader. Its dark appearance remains unresolved; this 17-light scene is neither a visual acceptance pass nor many-light performance evidence. [Capture metadata](rt-control/guide-scoring/capture.json).

![Unchanged sampled RT cathedral](rt-control/guide-scoring/1440p-sampled.png)

## Measurement protocol

Three serial A-E-E-A blocks use verified release executables, 120 warmup and 600 measured frames per run. A is the original shader; E combines both changes above. Each implementation contributes 3,600 measured frames at 2560x1440 output, half-width/half-height sampling, two samples, eight candidates, 1,024 moving lights and 256 moving triangle instances. Native 1440p and reconstructed 4K are additional single-run pairs with the same warmup and measurement counts.

GPU timing sums six HLFS stages plus TLAS rebuild using one cached BLAS. It excludes initial BLAS construction, upload GPU cost, real scene CPU preparation, GBuffer, AA and the remaining frame. The harness serializes readback. Changing light generation disables temporal composite repair reuse, but retains the reweighted sampler guide. These are synthetic work-accounting results, not whole-frame timings or production acceptance.

All binaries were built before these measurements. GPU tests and visual capture ran separately. Per-frame CSVs, telemetry and binary hashes are in [the artifact directory](rt-control/guide-scoring/summary.json).

| Implementation | Pooled GPU median | Pooled GPU p95 | Sampling-stage median |
| --- | ---: | ---: | ---: |
| Directional-budget baseline | 9.3015 ms | 10.9210 ms | 6.1204 ms |
| Combined guide scoring | 8.7496 ms | 10.3475 ms | 5.5132 ms |

The pooled median improves by **5.93%**. All six candidate run medians (8.675-8.828 ms) are below all six baseline run medians (8.987-9.432 ms). P95 here uses sorted element `floor(0.95 * (n - 1))`. Stage medians need not sum to the frame median. These fresh paired measurements must not be treated as a matched comparison against last week's machine timings.

| Output and shading | Baseline median / p95 | Combined median / p95 |
| --- | ---: | ---: |
| 1440p, native shading | 32.8530 / 35.5625 ms | 31.2986 / 34.2129 ms |
| 4K, half-width/half-height shading | 20.7442 / 24.5596 ms | 19.0075 / 22.4778 ms |

These wider-resolution rows are single-run coverage, not repeated acceptance blocks.

**The 3-4 ms target remains unmet.** The change is useful incremental progress, but neither this synthetic result nor the small cathedral scene establishes production RT readiness. PR #248 remains Draft / In Progress pending the performance and visual gates.

## Exploratory variants and provenance

The [exploratory run log summaries](rt-control/guide-scoring/exploratory-runs.json) preserve the intermediate probes. A shared-memory guide score cache was rejected: its A-B-B-A development probe showed no gain and it added 4 KiB per workgroup. The [inactive patch](rt-control/guide-scoring/shared-cache.patch) preserves that experiment. Separate [vector-only](rt-control/guide-scoring/vector-only.patch) and [confidence-only](rt-control/guide-scoring/confidence-only.patch) patches preserve the intermediate ideas; only the combined production patch is active. Exploratory timings are not the final acceptance comparison.

An attempted rebuilt baseline retained candidate code because copying an older shader preserved its timestamp. Its timing runs and baseline-comparison audits were discarded. The replacement build explicitly updates the shader timestamp, then verifies that the complete expected shader text is embedded in each executable. The baseline shader is also compared with `git show` at the baseline commit. Final measurements and audits use only these verified A3/E2 executables.

## Reproduce

Use the benchmark and audit commands from [the previous optimization](directional-budget.md#reproduce). Reverse [the production patch](rt-control/guide-scoring/production-change.patch) to build the baseline with the same current harness; restore it to build the candidate. When switching shader copies, update `LastWriteTime` before building and verify the executable contains the expected shader, rather than relying only on Cargo's successful exit. Build both executables before starting serial GPU runs.

For each of one through four samples, set `HLFS_RT_AUDIT_SAMPLES`, export baseline and candidate output with `benchmark_candidate_output_audit`, then invoke `scripts/compare_hlfs_rt_audits.py`. An absent sample variable defaults to two.
