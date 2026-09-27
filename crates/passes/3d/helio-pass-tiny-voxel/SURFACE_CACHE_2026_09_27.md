# Exact local surface cache experiment

This is a useful local representation candidate, not a replacement planetary
renderer. `surface-cache-experiment` exposes it for experiments; normal engine
configuration does not use it. The canonical `World`, generic component,
registered backend and source revision contracts remain authoritative.

This document records the initial storage experiment. The later
[bounded patch integration](SURFACE_PATCH_2026_09_27.md) connects it to rendering,
tests revision-safe GPU publication and reports a **negative** overall speed
result. Its precise traversal is an opt-in correctness baseline.

## Representation and source construction

A tile contains 32 cubed **authored** cells: 3.2 m at the default 10 cm grid,
32 m at the 1 m grid. It stores uniform 4 cubed microbricks as descriptors and
mixed microbricks as two 64-bit material planes, deduplicated within the tile.
A fully uniform payload uses four bytes. The sampled planetary tiles use
2,980–2,996 material bytes instead of a dense 8,192 bytes. This is not a
universal compression gain: the current worst case is 10,244 bytes.

Exposed faces are counted separately by axis direction and material. Their
bounds enclose actual face vertices; the bounds are never drawn as terrain.
A one-cell halo makes boundary exposure coherent. A certificate covering the
**whole halo**, including the last overriding edit, can leave uniform regions
implicit. A uniform payload alone cannot certify the absence of exposed faces.

The first implementation called `World::material` for every halo sample. The
existing gameplay cache uses fixed 3.2 m storage chunks, independent of the
authored grid. At 1 m this needlessly generates many copies of the same authored
material. The new builder samples authored cells directly, uses certified
8-cell region classes and spatially selected edit candidates, and shares the
canonical ordered-material evaluator with the gameplay chunk builder. It does
not create a second terrain recipe or change edit semantics.

The prototype cache admits only completed immutable bricks. Edits, undo and
replacement journals invalidate intersecting payloads **and their halos**;
grid/domain changes invalidate all addresses. Unaffected bricks remain shared.
LRU admission observes tile and logical byte budgets. A miss remains a miss,
never air. Callers may retain old immutable snapshots after eviction.

Logical accounting includes packed material bytes and 128 bytes per summary.
It excludes map/allocator overhead, other source caches and predecessor Arcs
held by callers. Consequently it does **not** establish the physical RAM/VRAM
budget. Construction is synchronous and belongs off the render thread. GPU
residency, publication and cancellation are still separate work.

## Correctness checks

The final release library runs passed 33 tests with the normal engine feature
and 40 with the cache, canonical reference and regional publication features
together (three diagnostic benchmarks ignored in each). The two cache probes
were also run explicitly. Builds retain existing unrelated unused-variable
warnings in `helio-core`.

The actual WGSL decoder and traversal are tested on RTX 3060/Vulkan. All
294,912 material queries and 42,696 local-ray results agree with the reference
across air, solid, plane, checker, shell, overlapping walls and edited planetary
tiles at 0.1, 0.3 and 1 m. Ray cases include positive/negative directions,
parallel and near-parallel axes, corners, boundaries and inside starts.

The CPU ray reference sorts every grid-plane event and samples the intervals
between them; it does not reuse packed decoding or the GPU DDA. First cell,
material, entry face and distance are checked. Starts already inside an opaque
cell have no defined entry normal. Traversal exhaustion is explicit rather
than a successful empty result. Skipping reduces fixture iteration totals
from 298,783 to 87,187 while preserving checked results.

The traversal consumes small local f32 ray coordinates. This is **not** proof
that a naive planet-to-local f32 subtraction works from orbit. Integration must
retain the existing compensated canonical position contract. Neither planetary
movement visuals nor filtered-lighting images have been rendered by this cache.

## Construction diagnostic

The probe builds nine adjacent source tiles with an empty source cache, then
performs warm cache lookups and applies a 0.61 m air brush. Four tiles invalidate
in this fixture. These are local CPU measurements, not frame or arrival times.

| Authored grid | Original cold build | Direct authored build | Original edited rebuild | Direct edited rebuild |
| --- | ---: | ---: | ---: | ---: |
| 0.1 m | 142.758 ms | 38.715 ms | 25.711 ms | 16.863 ms |
| 0.3 m | 599.596 ms | 38.467 ms | 22.058 ms | 19.751 ms |
| 1 m | 4,157.188 ms | 40.803 ms | 830.299 ms | 16.395 ms |

Packed bytes and exposed-face counts match the original builder for all three
sets. Warm lookup was about 13 ns in this small cache. That figure excludes
construction, uploads and large-cache LRU scans. The especially large 1 m
improvement removes duplicate storage-grid work; it is not an equivalent
improvement to the current engine renderer, which does not yet use this cache.
Three fresh-process repeats after the final halo-certificate fast path put
cold construction between 36.769 and 42.581 ms, and edited rebuilding between
17.231 and 25.406 ms, across the three grids. These repeats retain identical
packed bytes and face counts; their spread is more informative than treating
one table entry as a fixed latency guarantee.

## Warm GPU diagnostic

The timestamp probe intersects one cached tile with 262,144 coherent oblique
rays per dispatch. It performs plain/skip/skip/plain runs, each with four warmup
and 32 measured dispatches. Every repeated result agrees on coverage and, for
hits, cell/material/face/distance; 256 rays per case also have CPU audits.
Inputs, output writes and traversal are included. Construction, uploads,
hierarchy traversal, shadows, shading and presentation are excluded.

| Tile | Plain median | Skip median | Plain p95 | Skip p95 |
| --- | ---: | ---: | ---: | ---: |
| Plane | 0.179 ms | 0.113 ms | 0.183 ms | 0.115 ms |
| Shell | 0.251 ms | 0.136 ms | 0.256 ms | 0.141 ms |
| Overlapping walls | 0.098 ms | 0.071 ms | 0.100 ms | 0.072 ms |
| Planet 0.1 m | 0.176 ms | 0.104 ms | 0.187 ms | 0.107 ms |
| Planet 0.3 m | 0.217 ms | 0.112 ms | 0.477 ms | 0.114 ms |
| Planet 1 m | 0.308 ms | 0.134 ms | 0.415 ms | 0.136 ms |

These compare the **same compressed layout with and without empty-microbrick
skipping**, not the existing production renderer or a dense-layout GPU control.
The data fit in cache. Checker median slightly regressed, from 0.0547 to
0.0553 ms, and its skip p95 reached 0.280 ms. Timestamp spikes also occur in
other cases. Contention was not independently excluded, so these results are
diagnostic and do not qualify the frame-time gates.

## Rejected appearance shortcut and next integration

Unoccluded face-area weights are explicitly a falsification helper. Two equal
parallel walls with different materials produce a 50/50 +Z face mixture even
though all 1,024 orthographic reference rays see only the front material.
Thus an area histogram is not directional visibility, coverage or opacity.
Do not wire it directly into the GBuffer as the filtered terrain appearance.

This agrees with the motivation of [Filtered Appearance](https://cjsb.github.io/hpg2023/voxel-filtered-appearance.pdf):
directional variation and occlusion matter. Its expensive aggregate format is
not adopted. [Hybrid Voxel Formats](https://arxiv.org/abs/2410.14128) motivates
measuring per-level layout choices; it does not establish this layout as best.

Retain the exact cache candidate and the negative visibility test. The next
step is a bounded patch renderer consuming these bricks through the existing
source/backend contract, with locally coherent publication and compensated
ray entry. Compare its lit output with the retained spatial reference patches.
Develop visibility-aware far records only against those references. Do not
resume global-plan optimizations under the assumption that a face histogram
has solved distant appearance.

## Reproduction and evidence policy

```powershell
cargo +1.98 test -j2 --release -p helio-pass-tiny-voxel --features engine --lib surface_cache -- --test-threads=1 --nocapture
cargo +1.98 test -j2 --release -p helio-pass-tiny-voxel --features engine --lib surface_cache -- --ignored --test-threads=1 --nocapture
```

Generated evidence remains ignored and local under `target/voxel-goal`:
`surface-cache-cpu-before.log`, `surface-cache-probe.log`, and
`surface-cache-final-tests.log`. No capture, CSV, binary, raw timing dump or
manifest belongs in the committed branch. This authored report records the
experiment's interpretation and limits. Keep the planetary goal and PRs open.
