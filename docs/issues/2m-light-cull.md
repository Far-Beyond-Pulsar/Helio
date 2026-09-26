# LightCull scanned millions of empty light rows once per screen tile

## Resolution

Fixed in [18d2458b93509595e61d6101e62bb5030149f15e](https://github.com/Far-Beyond-Pulsar/Helio/commit/18d2458b93509595e61d6101e62bb5030149f15e).

## Reproduction and exact cause

Spawn two million cube entities, then one directional light. SceneDB's GPU
light buffer is indexed by raw entity index, not densely by number of lights.
The light therefore occupies a row above two million; preceding object-only
slots are zero-filled light rows. `prepare()` uses `row_capacity()` so a
high-index real light is not lost.

The tile shader then iterated `0..params.num_lights` separately for every
16×16 screen tile. At 800×600 there are 1,900 tiles. Rejecting empty rows inside
that loop preserves correctness but still causes billions of unnecessary
iterations. Reducing the bound to the live-light count would be incorrect:
the one real light is not at row zero. This is distinct from the older
fixed-capacity correctness issue #267.

## Fix and complexity

Add a GPU `compact_lights` entry point with one invocation per sparse row.
Positive-intensity rows append their original index into a buffer containing
a count followed by active IDs. Rebuild this list only when the SceneDB light
buffer epoch, content generation, or capacity changes, not on camera changes.

The tile kernel walks this compact list and writes original sparse IDs into
the existing tile lists. Deferred/forward shading therefore keeps its original
addressing contract. The list grows with capacity; changed buffers invalidate
bind groups and cached culling results. A separate compaction layout prevents
simultaneous writable/read-only aliases of the same buffer in one bind group.
The active counter is cleared before rebuilding, including deletions.

Complexity changes from O(tiles × sparse capacity) each cull to
O(sparse capacity) on light-data changes plus O(tiles × active lights).
The initial compaction cost is not claimed to be the tiny steady-state cost.

## Measured results

RTX 4060 Laptop GPU, Vulkan, 800×600, timestamp profiling enabled:

| Measurement | Correctly instrumented original | Fixed |
|---|---:|---:|
| LightCull, steady GPU time | 562.667 ms | 0.0098 ms |
| Whole graph, light fix alone | ~648.6 ms | ~85.2 ms |
| Whole graph, all subsequent fixes | ~648.6 ms | 4.0665 ms |

The isolated A/B runs kept the original benchmark workload/classification:
`two_million_exact_before.sqlite` and `two_million_exact_after_light.sqlite`.
The final run also corrects GPU mobility flags; it is not the sole evidence
used to attribute the LightCull improvement. Databases are retained locally,
not attached to this issue or committed as binary artifacts.

## Validation

`cargo test --release -p helio-pass-light-cull --lib` passed the actual GPU
regression using the production kernels. Two directional lights, including
index **2,000,003**, must reach all four test tiles with their original IDs.
Then one is removed, then both are removed; counts and surviving IDs must
match exactly. Shader parsing, pipeline creation, dispatch and readback are
validated, not merely a CPU model. The complete 300-frame/20-step benchmark
also completed with zero query overflows and readback-slot drops.

Limitations: tile lists still cap at 64 lights. Atomic compaction is not a
promise of stable ordering among active lights. No claim is made here about
animated-light workloads or more-than-64-light overflow selection.

## Before/after code

The issue appends the exact shader, Rust cache/buffer management, and GPU-test
diff from the fixing commit in a syntax-highlighted `diff` block.
