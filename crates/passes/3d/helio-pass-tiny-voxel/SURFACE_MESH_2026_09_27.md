# Exact surface raster checkpoint: useful local speedup, appearance still open

The new opt-in engine accelerator rasterizes exposed canonical faces and passes
accepted hits to the existing GBuffer and lighting graph. It reduces primary
visibility cost in the measured local cave fixture. It is **not** a qualified
planet renderer, far representation, multisample engine lighting path or art
direction result. Open-ground frame-time tails do not improve. The ordinary
engine configuration is unchanged.

## Representation and ownership

An eight-byte rectangle merges only coplanar exposed faces with one material.
Its coordinates and extents are integer authored cells. The canonical source,
ordered edits, exact cached materials and one-cell halo remain authoritative.
Uniform interiors have no surface mesh. Neither containing pages nor enlarged
voxels become displayed geometry. The tiny backend's three-material encoding
does not constrain the generic voxel component's source/payload formats.

Two triangles per greedy rectangle produced a reproducible interior T-junction
hole. The retained topology puts unit-spaced vertices along each rectangle's
perimeter and triangulates a center fan; a single-cell rectangle uses two
triangles. Seven power-of-two triangle-count buckets generate vertices directly
from rectangle records. There is no triangle-index allocation. Padding is below
2x. The procedural topology reproduces all 80 recorded local outputs from the
preceding indexed topology byte-for-byte.

The existing surface-patch worker produces meshes with the exact brick from the
same immutable published source. Generation/admission remains bounded to eight
completed tiles per frame. A tile exceeding 16,384 rectangles, or a full arena
bucket, is rejected atomically; exact traversal handles it. The seven buckets
each hold at most 65,536 rectangles. A changed source or domain resets readiness
and draw ranges before accepting results from the new revision. Stale queued
results cannot publish into the new domain.

The engine accelerator requires `CONSERVATIVE_RASTERIZATION` (overestimated
coverage). Helio requests it only when the adapter supports it; callers that
do not enable it keep the precise-cache path. This retains potentially nearer
faces that ordinary raster coverage can omit. Conservative overcoverage is
never accepted by itself: candidates outside the actual pixel-ray rectangle
still fall back. Transverse coordinates within 0.001 authored cells of an
integer plane, and very near camera intervals, also fall back; this guard does
not displace the displayed surface. See the [wgpu feature contract](https://docs.rs/wgpu/30.0.1/wgpu/struct.Features.html#associatedconstant.CONSERVATIVE_RASTERIZATION)
and [Vulkan conservative rasterization](https://docs.vulkan.org/spec/latest/chapters/primsrast.html#primsrast-conservativeraster).
The 0.001-cell guard is empirical; it is not a proof for arbitrary grazing
cameras or all GPU backends.

The engine path subtracts integer planetary anchors before float conversion and
uses the same camera rotation, projection and jitter as prepared primary rays.
Internal depth/identity attachments hold faces and unavailable-page markers.
Markers are never written as terrain. An independent page-prefix traversal
using the existing exact dyadic plane predicates rejects a hit if any preceding
mesh page is unavailable, including at quantized raster edges. Outside/solid
camera starts, misses, rejected pages and out-of-rectangle pixel rays retain the
prepared ray for precise cache traversal, then canonical fallback. Completed
raster hits carry diagnostic bit 26; bit 27 remains the cache completion contract.

The final engine identity target is Rg32Uint plus Depth32Float: 12 bytes per
rendered pixel. The arena and readiness directory add 3,672,064 bytes. At native
720p this is 14,731,264 additional logical bytes (14.05 MiB), down from the first
Rgba32Uint trial's 21.08 MiB. Current allocations are included in terrain memory
statistics. Physical allocation, retired in-flight resources and whole-process
memory are not inferred from these logical counts.

## Correctness evidence

**The first ordinary-raster engine variants (v1-v3) failed continuous ground
motion.** Across 182 paired 320x180 captures, 1,212 of 10,483,200 rays disagreed
in geometry and 1,194 in depth; one early frame differs by up to 1.9002 m.
Independent f64 interval replay of all 1,212 disagreements finds the precise
control correct and the ordinary-raster candidate wrong in every case.
A quantized raster edge can omit a thin nearer face and select a valid rear
face. Page readiness and material equality do not certify first visibility.
Internal integer planes of merged rectangles also expose incorrect f32 floor
ownership at exact ties. Endpoint successes below did not establish safety.
The revised candidate requires conservative raster coverage and returns
transverse plane ambiguities to precise traversal; its qualification follows
separate recorded checks.

The corrected v4 and final v5 repeat the entire ground route with **zero
geometry/depth disagreements in 10,483,200 rays**; 5,453,488 rays complete through
rasterization (52.0%, down from the rejected path's 76.2%). These recorded
comparisons include the unchanged canonical fallback outside the local patch.
They cover primary geometry, not separately captured linear lighting throughout
the movement. Inspected ground frames retain pronounced aliasing/banding.

Release combined-feature suite: **53 passed, five diagnostic tests ignored**.
Six device/configuration tests also pass, including optional feature negotiation.
This includes:

- CPU expansion of merged rectangles against independently enumerated exposed
  faces/materials, including shells, checker volumes and halo occlusion.
- Canonical source/mesh comparisons with edits, undo, negative coordinates and
  0.1/0.3/1 m authored grids.
- Local GPU identity comparison: 184,320 pixels, five finite fixtures and six
  directions. Hardware coverage uses a declared 0.01-pixel boundary tolerance;
  88 cases require independent nearby-ray checks. Interior holes are not excused.
  Integer-plane reconstruction fixes an earlier interpolated-depth error.
- Actual conservative engine raster/resolve shaders: 7,158 accepted hits match
  independent f64 interval enumeration across three grids and two planetary
  anchor signs.
  Missing pages, reset, arena/tile rejection, resize and inside-solid starts
  exercise fallback. A dedicated GPU regression forces the internal plane tie
  of a two-cell merged face, retaining ambiguous rays for exact traversal while
  accepting its unambiguous neighbor. Existing queued source/edit/undo/grid
  publication tests also run with worker mesh production enabled.
- First engine reference pair: three close fixtures (cave, thin wall, destroyed
  wall), two lights, two camera positions and four spatial samples per pixel at
  64x36. All 110,592 sampled rays match the precise-cache control in geometry,
  depth, material, albedo, sunlight and linear lighting. Including center hits,
  138,240 paired rays contain 136,392 raster completions and zero geometry/depth
  differences. This is a control comparison, not an independent oracle for all
  those rays. The final conservative repeat also has zero geometry, depth or
  linear-lighting differences; it accepts 26,724 raster hits at this coarse
  reference resolution, with exact traversal handling the others.
- Historical v3 two 720p fixed-jitter endpoints: 1,843,200 paired rays, 1,841,884
  raster completions, zero cell/face/material/depth differences. The stationary
  PNG is byte-identical; the motion endpoint PNG differs after temporal rendering.
  Raw primary-hit equality does not establish postprocess/history equivalence.

The independent GPU audit is intentionally separate from the engine control
comparison. Neither proves exact agreement for every quantized raster boundary
in arbitrary scenes. The old endpoint successes did not detect the subsequent
continuous-motion failure. The corrected v5 passes that recorded route, but
further validation remains required before enabling this experiment normally.

## Full-graph timing diagnostics

Windows, Ryzen 5 3400G, RTX 3060/Vulkan, release Rust 1.95. Measurements use the
existing Helio GPU timestamp profiler, including its enclosing whole-graph
scope. Do not add child scopes to their parents. Each settled fixture has 32
warmup frames and 180 measured stationary/moving frames. Motion is one cycle of
0.5 m sideways, 0.1 m vertical and 0.04 rad yaw. Synchronized wall times exclude
capture/PNG encoding and presentation. No concurrent compiler or second test
GPU workload ran during these measurements. They are diagnostic headless runs,
not populated-editor frame-rate acceptance.

Corrected conservative v5 moving results, milliseconds (control -> raster):

| Fixture | Primary GPU median | Whole graph GPU median | Synchronized frame p95 | Synchronized frame p99 |
|---|---:|---:|---:|---:|
| Cave, native 720p | 3.790 -> 2.398 | 7.634 -> 6.323 | 15.230 -> 14.212 | 33.637 -> 19.339 |
| Cave, 1080p Quality (1440x810 internal) | 4.831 -> 2.502 | 9.885 -> 7.590 | 15.721 -> 14.698 | 16.221 -> 15.667 |
| Ground, native 720p | 8.812 -> 8.189 | 15.067 -> 14.433 | 21.662 -> 22.085 | 24.439 -> 25.981 |

Each row is one sequential control/candidate pair, not a confidence interval.
The cave benefits remain after the visibility correction, but open-ground
frame tails regress and exceed the 16.67 ms target. Ground primary GPU p95 is
10.058 -> 10.086 ms; whole graph GPU p95 is 16.255 -> 17.217 ms. In the candidate,
median precise-cache traversal is 5.478 ms, raster is 0.564 ms and sunlight is
2.863 ms. These nested profiler scopes identify remaining work; they must not
be added to enclosing primary or graph times. GPU timings do not include the
full CPU/presentation cost.

Historical v1-v3 moving cave results, milliseconds (control -> raster). These
variants have the failed visibility contract above; their speedups alone do not
qualify the revised conservative candidate:

| Trial | Primary GPU median | Whole graph GPU median | Synchronized frame p95 |
|---|---:|---:|---:|
| First 720p pair | 3.800 -> 1.345 | 7.797 -> 6.872 | 16.240 -> 15.697 |
| Reversed-order 720p pair | 4.102 -> 1.158 | 7.726 -> 6.092 | 14.130 -> 13.036 |
| Final compact target, 720p fixed jitter | 3.791 -> 1.341 | 7.710 -> 6.988 | 14.572 -> 15.356 |
| Final 1080p Quality (1440x810 internal) | 4.819 -> 1.129 | 9.962 -> 6.526 | 16.548 -> 14.114 |

The historical fixed-jitter wall-time p95 regresses despite the primary gain.
CPU submission, GBuffer and other graph work matter. Local cave results do not
replace the failed global
flight gates in the [worker checkpoint](WORKER_RESIDENCY_2026_09_27.md).

Mesh production also costs CPU time. In the first 720p pair, complete-patch
construction rises from 504.6 ms to 792.7 ms (summed worker construction time,
not visible edit latency). Corrected v5 ground construction is 469.9 -> 662.3 ms;
all 512 mesh pages are accepted and packed mesh uploads total 205,768 bytes.
The current experimental patch still resets and
reuploads its entire 512-page window on source/domain change. Local mesh reuse
and independently publishable regions are necessary follow-ups.

## Separate local appearance experiment

The standalone finite-tile experiment shades each of four raster samples before
resolving synthetic Lambert lighting. Twenty cases cover five shapes, two light
directions and two camera positions. Against a 64-sample spatial reference,
four-sample RGB and partial-coverage error improve in all twenty cases. Example
RGB RMSE (single -> four): slope 0.00763 -> 0.00281; overlapping walls 0.00526 ->
0.00118; destroyed shell 0.01657 -> 0.00543; checker 0.04073 -> 0.01491.

This is not the engine's BRDF, shadows, materials or lighting. Fourteen of 62,484
pixels with constant sampled-reference color still change; do not claim all
resolved pixels are preserved or the finite reference has converged. Two-pose
image deltas include parallax and are not a continuous flicker metric.

Warm, single-tile 720p GPU timestamps show four-sample raster medians around
0.094-0.128 ms for slope/walls/shell, versus 0.158-0.227 ms for one-sample brick
traversal using matched synthetic lighting. The checker reverses that result:
four-sample raster is about 0.260 ms versus 0.118 ms traversal. Its 33,024 quads
consume 264,192 bytes; a two-wall fixture needs only 12 quads/96 bytes. Hybrid
representation selection must follow measured geometry, memory and visibility
cost. A universal raster replacement is not supported by these results.

Nine actual planetary ground tiles have 18.6-26.8 KB of packed quads across the
three grids. Paired CPU construction probes take 44.4-47.8 ms cold and 20.1-24.0
ms after a local brush invalidates four tiles. These are source-plus-mesh CPU
costs, not engine edit-to-visible latency or physical-memory measurements.

## Reproduction and remaining work

Build the flight with `--features voxel-surface-cache`, then set
`HELIO_VOXEL_SURFACE_MESH=1` to enable the new path. Leave it unset for the
precise-cache control in the same executable. `HELIO_VOXEL_CACHE_BENCH=cave-close`
selects the local timing fixture; `HELIO_VOXEL_FLIGHT_PROFILE=1` records engine
GPU scopes. `HELIO_VOXEL_CACHE_FIXED_JITTER=1` and
`HELIO_VOXEL_FLIGHT_SAVE_HITS=1` support matched hit captures.

Use `voxel_flight/analyze.py` for frame/GPU summaries and
`voxel_flight/analyze_mesh_engine.py CONTROL CANDIDATE` for paired engine
captures. `analyze_mesh.py` evaluates the separate synthetic-lighting trial.
Run `voxel_flight --audit-mesh-motion CONTROL CANDIDATE OUTPUT` to replay geometry
disagreements from the fixed-jitter ground recording with the independent CPU
interval oracle; this classified all 1,212 rejected v3 disagreements.
Raw evidence, source snapshots, executables and reports remain ignored under
`target/voxel-goal`; only authored source and this interpretation are versioned.

Frozen executable SHA-256:

- First engine trial: `0dc75ffbfb8e0224322c9df6bdbcb68a1377a3aa7630323fcb60a7806081afdd`.
- Rejected ordinary-raster compact trial: `bb7903e3a630c787ee758db5bd95ed878bb721ee9882a4bd0f92f414822cf229`.
- Final conservative engine trial: `da582fffe4ee48aa82c9e84aece31dcf90160630ae0f5bcad8c27f2e4662b44c`.
- Local procedural topology trial: `d858ca9cd3272c7829b3639f665cb305947c9f0b467b6fc19ce08130a48fb922`.

Inspection still shows existing single-sample stair-step bands. Integrate
visibility-aware, separately shaded face contributions before claiming an
appearance improvement. Canonical far coverage, orbital travel, local visible
edits, arrival latency, long-duration physical memory, populated scene systems
and Lay of the Land art direction remain open. No change here qualifies smooth
fallbacks, visible large-block LOD or distance-limited destruction.
