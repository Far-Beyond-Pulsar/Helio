# Closed issues: two-million-object profiling and rendering fixes

The implementation is published in
[18d2458b93509595e61d6101e62bb5030149f15e](https://github.com/Far-Beyond-Pulsar/Helio/commit/18d2458b93509595e61d6101e62bb5030149f15e).
These retrospective issues document the exact causes, implemented fixes,
measurements, reproduction commands, tests and limitations. Each GitHub issue
also includes the actual selected-file `git diff 18d2458b^ 18d2458b` in a
syntax-highlighted `diff` block. The adjacent Markdown files preserve the
narratives locally; the linked implementation commit is the source for code.

| Issue | Fix | Local record |
|---|---|---|
| [#286](https://github.com/Far-Beyond-Pulsar/Helio/issues/286) | Compact sparse light IDs before tiled culling | [LightCull](2m-light-cull.md) |
| [#287](https://github.com/Far-Beyond-Pulsar/Helio/issues/287) | Correct GPU encoder attribution, CPU accumulation and timing totals | [Profiling](2m-profiling.md) |
| [#288](https://github.com/Far-Beyond-Pulsar/Helio/issues/288) | Bound instance batches to distribute culling work | [Batch parallelism](2m-batch-parallelism.md) |
| [#289](https://github.com/Far-Beyond-Pulsar/Helio/issues/289) | Reject empty light rows before shadow matrix writes | [Shadow rows](2m-shadow-rows.md) |
| [#290](https://github.com/Far-Beyond-Pulsar/Helio/issues/290) | Skip inactive volumetric fog dispatch | [Inactive fog](2m-inactive-fog.md) |
| [#291](https://github.com/Far-Beyond-Pulsar/Helio/issues/291) | Synchronize GPU mobility flags and export trustworthy benchmark data | [Benchmark integrity](2m-benchmark-integrity.md) |

## Verification summary

The complete [diagnosis report](../two_million_diagnosis.md) records the A/B
sequence and final sweep. On RTX 4060 Laptop GPU/Vulkan at 800×600, correctly
instrumented original GPU graph time was about 648.6 ms. The completed fixed
sweep ended at 4.0665 ms GPU time and **5.2349 ms actual cadence / 191.02 FPS**.
LightCull fell from 562.667 to 0.0098 ms; frustum culling from 42.509 to
0.9910 ms; occlusion from 36.438 to 0.7030 ms; ShadowMatrix from 4.622 to
0.3714 ms. Fog's inactive scope was 0.00298 ms, without a reliable isolated
before measurement.

The final run used 300 static frames and 20 steps of 100,000 objects, each
with three seconds of settling and 120 cadence samples. Dynamic classification
does not imply per-frame transform/deformation updates. Readback source-frame
alignment matters: the final window has 109 unique GPU samples, not 120.

Targeted validation: 59 passing tests across profiling (4), LightCull GPU (1),
ObjectBatch GPU (5), and ShadowMatrix (49, including one GPU test). The release
benchmark completed; SQLite integrity was `ok`; no query overflows or
readback-slot drops were reported. These results are not a claim of running
the entire workspace test suite or testing active fog appearance.

Databases remain local benchmark artifacts rather than committed binaries.
The versioned analysis tool is `scripts/analyze_two_million.py`; use the
epoch-aware `gpu_frame_samples` and `gpu_pass_samples` views for GPU queries.
