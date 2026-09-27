# Bounded surface patch integration: correctness baseline, rejected speed claim

The exact cache now runs through the actual terrain/GBuffer/lighting graph,
behind `voxel-surface-cache` in the flight example and
`surface-cache-experiment` in the backend. Normal engine configuration remains
unchanged. **The current integrated traversal is slower in the decisive 10 cm
and 30 cm captures. Do not enable it as a production optimization.** Retain it
as an opt-in correctness and publication baseline for the next architecture
trial. The compact storage and direct authored sampling remain useful.

## Publication and traversal contract

A worker owns an 8 MiB logical CPU cache and prioritizes a local 8-cubed tile
window. Requests carry a revision, obsolete work is cancelled between bricks,
and an eight-result channel bounds completed payloads. The render thread admits
at most eight results per frame. Each tile needs at most 10,244 payload bytes
plus a four-byte directory address. A changed source/window first invalidates
the directory (2,048 bytes plus a 32-byte domain uniform). Old queued revisions
cannot publish into the new directory.

The fixed GPU cache adds 5,247,008 logical bytes including its directory/domain.
It is additional to the existing terrain arena, not a replacement memory saving.
Allocator overhead, source caches and retained immutable snapshots still need
physical memory measurements. The cache follows the **published** source
revision. This preserves source coherence but does not solve the latency of
the existing complete-cut publication system.

The primary cache pass runs after camera-ray preparation. It accepts a first
hit only when every preceding cell was available from current pages. An
outside camera, missing page, inside-solid start or patch exit preserves the
prepared ray for canonical traversal. Exhaustion is explicit. Completed cache
hits keep their original ray bits and carry diagnostic status bit 27. Their
hierarchy level is zero; this metadata need not match the old resident leaf.
The existing material, GBuffer and directional-visibility passes consume hits.
No enclosing tile or microbrick is drawn as terrain.

The final trial uses authored-cell DDA. Near-equal plane crossings use an integer
dyadic predicate; ordinary well-separated crossings use a conservative float
comparison band. Six u32 words retain the local integer/fraction numerator and
the ray's significand, with its exponent compared separately. This avoids
depending on float cancellation residuals. Both the fma-residual and split-float
trials failed a GPU regression: distinct crossings became equal. WGSL permits
[non-fused fma and floating-point transformations](https://www.w3.org/TR/WGSL/#floating-point-evaluation).
Depth output remains f32 and is not bit-identical to the old traversal.

## Checked evidence

Release tests pass: 42 with cache/reference/regional-publication features,
33 with normal engine features; three diagnostic probes ignored in each.
The actual primary shader passes 13,122 adversarial rays, including 11,964 hits,
on 0.1/0.3/1 m grids with positive/negative planetary anchors. Directions include
adjacent float values around edges and corners. Cell, material, face and
bounded depth error are checked against independent f64 grid-plane intervals.
The queue/publication test checks 138,912 resident material queries while
superseding sources, teleporting, undoing and changing grids before completion.
The exact query count depends on worker scheduling.

The offline audit sorts crossings in canonical units: multiplying each plane
numerator by 0.1 before division can spuriously split a coincident edge even in
f64. It reconstructs the same fixtures and uploaded split origin. It diagnoses
geometry/light-disagreeing samples; it is not an independent CPU audit of every
matching pixel. The existing sparse CPU audits and repeated-frame comparisons
also pass.

All captures use 96x54 crops, 16 regular spatial samples per pixel, two sunlight
directions and two poses separated by 2.5 cm. The 10 cm set includes all 32
reference cases. The 30 cm/1 m sets each include twelve close cave, thin-shell
and destroyed-shell cases. Same-binary controls set
`HELIO_VOXEL_SURFACE_CACHE_OFF=1`; they skip requests/dispatch but still allocate
the experimental cache. This is a steady rendering control, not a memory or
startup comparison.

| Grid / cases | Primary samples | Completed by cache | Coverage/material/ray differences | First-cell / face differences | Lit RGB differences |
| --- | ---: | ---: | --- | --- | ---: |
| 0.1 m / 32 | 2,654,208 | 1,184,472 | 0 / 0 / 0 | 6 / 8 | 12 |
| 0.3 m / 12 | 995,328 | 993,400 | 0 / 0 / 0 | 2 / 2 | 2 |
| 1 m / 12 | 995,328 | 995,328 | 0 / 0 / 0 | 0 / 0 | 0 |

No primary diagnostic failures occur. Every differing first hit agrees with
the independent interval oracle in the cache result. The old renderer is wrong
on ten samples in the 10 cm set and four in the 30 cm set; counts include both
lights. These are corrections, not a reason to erase comparison differences.
At 1 m all captured linear lighting is identical. Maximum f32 depth differences
are 19.1, 7.63 and 5.25 micrometres respectively. The 10 cm run has one sunlight
visibility difference. Shadow-origin rounding and full lighting accuracy remain
open; primary geometry audits do not certify those outputs.

The compared mean-image maximum RGB RMSE across cases is 0.0001412 at 10 cm,
0.0000706 at 30 cm, and zero at 1 m. These crops retain resolved voxel faces and
edited openings. They do not validate a continuous planetary flight or AAA art
quality. There are 5,570/536/0 hierarchy-level metadata differences respectively;
the analyzer reports them explicitly and retains its strict comparison mode.

## Performance disposition

Engine GPU timestamps include the actual graph. These are pooled reference
frames, not repeated uncontended acceptance benchmarks or resolution-scalable
estimates. Compilation/other machine activity was not excluded independently.
The final delayed timestamp entries are not drained: there are 575 whole-graph
and 574 primary samples for the 576-frame set, and 215/214 for each 216-frame
set. Do not claim complete timing coverage.

| Grid / cases | Graph median off/on | Graph p95 off/on | Primary median off/on |
| --- | --- | --- | --- |
| 0.1 m / 32 | 9.781 / 12.984 ms | 14.111 / 17.788 ms | 7.776 / 11.522 ms |
| 0.3 m / 12 | 2.299 / 6.599 ms | 5.158 / 7.869 ms | 0.953 / 5.096 ms |
| 1 m / 12 | 1.628 / 1.878 ms | 3.067 / 3.494 ms | 0.158 / 0.153 ms |

Per-voxel lookup and hoisted lookup integrations matched all 2.65 million
10 cm control samples bit-for-bit, including lighting, but added overhead.
A compact per-tile traversal reduced some work but exposed precision errors.
The final precise path fixes the checked errors yet has substantial overhead
and leaves unresolved rays doing the full canonical traversal. None establishes
an overall speedup. All variants remain local experiments.

The next candidate should keep this authority/reference but separate cheap
compact traversal from rare precise repairs, with bounded work and no false
empty prefixes. Measure register/traversal cost and empty-region skipping.
The dominant unresolved task is still visibility-aware distant appearance and
bounded global residency. Face-area averages cannot replace visibility.
Full flight, arrival/edit latency, physical memory, populated engine integration
and the 1080p/720p performance gates remain open.

## Reproduction

```powershell
cargo +1.98 build -j2 --release -p helio-default-graphs --example voxel_flight --features voxel-surface-cache
$env:HELIO_VOXEL_SURFACE_REFERENCE='4'
$env:HELIO_VOXEL_FLIGHT_PROFILE='1'
.\target\release\examples\voxel_flight.exe target/voxel-goal/cache-capture 96 54 native
$env:HELIO_VOXEL_SURFACE_CACHE_OFF='1'
.\target\release\examples\voxel_flight.exe target/voxel-goal/control-capture 96 54 native
python crates/helio-default-graphs/examples/voxel_flight/compare_cache.py target/voxel-goal/control-capture target/voxel-goal/cache-capture
.\target\release\examples\voxel_flight.exe --audit-cache target/voxel-goal/control-capture target/voxel-goal/cache-capture target/voxel-goal/audit
```

Set `HELIO_VOXEL_FLIGHT_BASE_METRES` for other authored grids. The independent
offline audit supports local differing rays below 300 m. Use separate output
directories. Raw evidence is ignored under `target/voxel-goal`: final binary
`surface-patch-v10.exe` (SHA256
`0547659b80ae6a8a30913c75b26c6ff77bbd9789da03f5bde993bcd8c92281f9`),
`surface-patch-all[-control]-v10`, `surface-patch-grid-*-v10`,
`surface-patch-comparison*-v10.json`, `surface-patch-final-audit-v10`,
and final/default test logs. No generated evidence belongs in Git.
