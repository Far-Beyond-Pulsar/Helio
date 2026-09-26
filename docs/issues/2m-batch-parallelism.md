# Unbounded instance batches serialized two million culling tests onto 64 lanes

## Resolution

Fixed in [18d2458b93509595e61d6101e62bb5030149f15e](https://github.com/Far-Beyond-Pulsar/Helio/commit/18d2458b93509595e61d6101e62bb5030149f15e).

## Problem and root cause

After fixing LightCull, the graph still took about 85.2 ms. Accurate pass
markers identified roughly 42 ms of frustum culling and 37 ms of occlusion
culling. The scene contains two million instances of one mesh/material pair.
ObjectBatch merged that run into one draw. Both downstream kernels assign
one 64-lane workgroup per draw and walk its instances with a grid-stride loop.
This put about 31,250 loop iterations on each lane of a single workgroup,
instead of distributing the scene across the GPU. The thread-count scaling
followed draw count, not object count.

## Fix

Bound draw groups to at most 4,096 instances by starting a group at each
4,096-element boundary of the sorted live stream, in addition to the existing
mesh/material-change boundaries. Apply the identical predicate in both the
local group scan and group-start write kernels; otherwise ranks and starts
would disagree and corrupt draws.

Two million identical instances now require 489 bounded draws rather than
one. Each draw retains correct mesh fields, contiguous first-instance/count
ranges and indirect arguments. Adjacent draws still coalesce into their
material/shading ranges. The existing culling shaders can exploit many
workgroups without a new CPU scene walk or deleting objects.

This is a scheduling tradeoff: more draws in exchange for bounded per-workgroup
culling. Fixed stream boundaries may also split a smaller mesh/material run
which straddles a boundary. 4,096 is the measured fix, not a universal optimum.

## Before/after measurements

RTX 4060 Laptop GPU, Vulkan, 800×600, profiling enabled:

| GPU pass | Original corrected-marker baseline | Batch-fix A/B | Final full sweep |
|---|---:|---:|---:|
| IndirectDispatch | 42.509 ms | ~0.952 ms | 0.9910 ms |
| OcclusionCull | 36.438 ms | ~0.634 ms | 0.7030 ms |

The light-only run was ~85.2 ms per GPU graph; adding bounded batches reduced
it to ~8 ms. The later shadow-row fix brought the final graph to 4.0665 ms.
Do not multiply reciprocal GPU times into a claim about displayed FPS:
the corrected final benchmark measured 191.02 FPS including logging and
event-loop/presentation overhead. The earlier A/B demo did not yet set the
GPU dynamic flag; its two runs are compared with one another for causality.

## Reproduction and regression coverage

Run the two-million-object demo with a single cube mesh and material. Use
`gpu_pass_samples` in the final DB, or raw `pass_timings` in the preserved
earlier A/B databases, to inspect these named passes.

`cargo test --release -p helio-pass-object-batch --test gpu_object_batch_validation`
passed all five GPU tests. The new regression sends 10,003 identical live
rows plus 13 dead rows through the real pipeline. It verifies exactly three
bounded draws, contiguous and complete instance coverage, unchanged IDs,
matching indirect arguments, preserved aggregate shadow counts, and one
material range spanning the three draws. Existing tests also cover randomized
CPU/GPU agreement, all-dead rows, range-readback growth, and previous transforms.

## Before/after code

The actual two-kernel boundary change and new regression test are appended to
the GitHub issue as a `diff` code block from the implementation commit.
