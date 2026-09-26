# ShadowMatrix interpreted empty sparse rows as directional shadow casters

## Resolution

Fixed in [18d2458b93509595e61d6101e62bb5030149f15e](https://github.com/Far-Beyond-Pulsar/Helio/commit/18d2458b93509595e61d6101e62bb5030149f15e).

## Exact failure

SceneDB light rows are sparse and entity-indexed. The pass correctly dispatches
over capacity to include high-index lights, but the shader only rejected
`shadow_index == 0xFFFFFFFFu`. An unused zero-filled row instead has
`shadow_index == 0`, `light_type == 0` (directional), and zero intensity.

Consequently, millions of unused rows entered directional-cascade computation
with invalid zero direction data and targeted the same shadow-matrix base
and caster hash at slot zero. This was both wasted GPU work and a correctness
bug: competing writes could corrupt matrices/hashes and set dirty flags even
though no real light owned those rows. It was not a SceneDB upload bottleneck.

## Fix

Return before all matrix computation, hashing and writes when intensity is
non-positive, as well as when the no-shadow sentinel is present. Positive-
intensity real lights retain the point/directional/spot paths and original
shadow indexing. The pass still scans sparse capacity once in parallel;
this change does not claim to eliminate that bandwidth cost.

## Measurements

RTX 4060 Laptop GPU, Vulkan, 800×600, profiling enabled. ShadowMatrix averaged
**4.622 ms** in the corrected-marker original baseline and **0.3714 ms** in
the completed final sweep. After bounded batches but before this fix, it was
still about 4.45 ms and the graph about 8 ms. A full corrected-mobility sweep
before this fix ended at **8.49 ms** graph time; the final fixed sweep ended
at **4.0665 ms**. These sequential runs support the attribution; they are not
a universal frame-time guarantee for other scenes or devices.

Steady final SceneDB upload bytes were zero. Removing phantom shadow-caster
work, rather than optimizing repeated CPU uploads, explains this improvement.

## Reproduction and validation

Allocate a zeroed light buffer with many holes and dispatch the production
ShadowMatrix shader. Before the fix, those holes pass the sentinel check.
After the fix, an all-empty buffer must leave matrices and dirty flags intact.

`cargo test --release -p helio-pass-shadow-matrix` passed 49 tests: one new
GPU regression plus 48 existing math/layout tests. The GPU test first uses
1,025 empty light rows and verifies every output byte remains zero; then it
enables a point light at row 1,024 and verifies finite, nonzero shadow matrices
and the expected dirty flag. It runs the production shader and reads back
actual GPU outputs, rather than relying on a source-text assertion.

The final two-million-object run also completed all 20 transition checkpoints
with zero query overflows or readback-slot drops. Dynamic in that benchmark
means GPU mobility classification, not per-frame transform animation.

## Before/after code

The exact guard change, test wiring and GPU regression are included below in
the GitHub issue as a syntax-highlighted implementation-commit diff.
