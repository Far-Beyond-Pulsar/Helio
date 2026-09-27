# Planet terrain goal

Active work authorized on 26 September 2026. Architecture, code, data layout and the shared voxel component may change. Reference integration baseline: Helio `156f6a7b`, Pulsar-Native `1d2947ea3`. The measured limitations in `VALIDATION_2026_09_26.md` remain failures until superseded by evidence.

## Required behavior

- A canonical, editable planet with a default 0.1 m base grid, configurable through 1 m. Rendering, queries, collision and saved edits agree on that grid.
- Crisp voxel geometry where resolvable. Distant filtering preserves coverage, silhouettes, material and edit effects within a declared pixel error. No visible parent cubes, independent smooth replacement terrain, or temporal smearing to conceal missing data.
- Storage and rendering remain bounded through sparse edits, procedural untouched regions, cached detail and conservative hierarchy summaries. Constructing a far summary must not require eagerly generating every 10 cm cell on the planet.
- Destruction has no arbitrary tool-distance cutoff. Source changes propagate consistently to all representations and nearby simulation. Large edits have explicit bounded update work and observable completion; neither silently dropping edits nor unbounded render-thread work is acceptable.
- Continuous orbital travel and rapid reversals retain complete coverage. Refinement publishes independently in safe regions, respects revisions, prioritizes the current view and cancels obsolete work.
- The shared component owns reusable source/material/revision contracts. The terrain backend owns its rendering representation. Engine systems consume the canonical data through explicit adapters.
- Visual direction follows Lay of the Land: readable fine voxel forms, strong terrain/material composition and coherent lighting. A passing hit-status test alone is not visual acceptance.

## Initial acceptance targets

These are targets, not achieved results or guarantees. Revise a target only with an explicit technical rationale recorded before comparing candidates; retain failed experiments.

| Gate | Initial target |
| --- | --- |
| Test platform | Ryzen 5 3400G, RTX 3060, 16 GiB RAM; fixed driver/build provenance; no competing GPU job/compiler |
| Resolution and cadence | 1920×1080 Quality and 1280×720 native; warm full-graph p95 ≤16.67 ms, movement p99 ≤25 ms |
| Terrain GPU work | p95 ≤5 ms including generation/tracing/GBuffer/terrain visibility for the declared scene; stage timings and query losses reported |
| Residency | ≤1 GiB logical terrain GPU allocations initially; separately measure actual process VRAM, CPU storage and allocation spikes |
| Arrival | Complete coverage throughout; resolvable arrival detail converges within 250 ms after stopping the 300 km descent; no multi-second stale surface |
| Destruction | Exact deterministic CPU/GPU agreement for 0.1, 0.3 and 1 m grids, negative coordinates, brick boundaries, overlapping edits, undo and reload; visible local edit within 100 ms for a declared brush workload |
| Far fidelity | Compare actual depth, coverage and material with exact supersampled reference crops; preserve silhouettes and edited features that exceed the pixel-error bound; no missing-data hits |
| Movement visuals | Ground walk, rotation, zoom, horizon, continuous ascent/descent, reversal, teleport, resize and grid replacement; inspect full sequences, not only settled screenshots |
| Engine integration | A populated scene with meshes, foliage, lighting/shadows and nearby collision/query consumers, plus interactive editor verification |

Do not combine nested GPU stage sums with their parent graph timings. Keep synchronized offscreen time distinct from GPU timestamps and presented frame time. Record independent repeats for retained optimizations, including worst-frame and long-tail behavior.

## Work order

1. Export frame-identified graph and terrain GPU stages, CPU submission/wait times, residency and memory usage. Diagnose the current expensive stages and visual defects against exact references.
2. Select and test a coherent hierarchy/visibility representation. Compare candidates by fidelity, edit propagation, memory and measured frame cost; the current sampled-density far path has no privileged status.
3. Implement region-level publication and demand scheduling with measured generation/upload budgets, immutable revisions and stale-work cancellation.
4. Develop shading and footprint filtering together with the representation in step 2; validate with directional visibility enabled and representative materials/content. A single exact hit per pixel is not a filtered appearance reference.
5. Extend shared component adapters, simulation/query integration, persistence and stress coverage. Validate the complete engine/editor workflow before declaring the goal achieved.

Keep PRs drafts while these gates remain open. Do not merge without explicit user authorization.

## Regional publication checkpoint

The opt-in experiment in `REGIONAL_PUBLICATION_2026_09_26.md` publishes ready regions with clipped ancestor payloads and passes focused ownership/revision tests. Its retained recorded 720p flight had no exhausted/loading rays, but visual inspection exposed representation seams, full arrival refinement still took 3.38 seconds, and far fidelity remains incorrect. It is rejected for default adoption. The next scheduling candidate is a bounded resident hierarchy with local publication transactions and cancellation of obsolete demand; it also needs surface data that agrees across refinement boundaries. The existing global plan lifetime and inaccurate sampled far field are both replaceable; no acceptance target has been relaxed.

## Surface appearance and resident hierarchy direction

The range-certificate trial and the exact per-ray noise-cache trial are not adopted. Both preserved the sampled correctness checks, but neither established a broad tail-time improvement. The recorded canonical range + regional flight completed 927 frames, passed 367 movement validity captures and matched 864 sampled first cells; visual inspection still showed strong contour stippling. Its primary GPU p95 reached approximately 83 ms at 200 m and 214 ms in orbit at only 640x360. These diagnostics are not uncontended acceptance timing. Trial source, binaries and generated evidence stay local under `target/voxel-goal`.

The source audit found two distinct problems: sparse density changes distant geometry, while one face-normal/albedo/visibility sample per pixel aliases even with canonical geometry. Lighting an averaged normal cannot generally replace averaging the separately lit faces. Research on [filtered voxel appearance](https://cjsb.github.io/hpg2023/voxel-filtered-appearance.pdf) and [filtering after shading](https://arxiv.org/abs/2305.05810) supports preserving those distributions; neither paper supplies a ready-made planetary terrain solution.

The next architecture candidate keeps canonical integer occupancy, material and ordered edits authoritative. It caches compact exact surface microbricks, with uniform interior regions implicit. Derived surface records above them retain coverage, conservative position/depth bounds and separate axis-face/material contributions. A record may replace finer rendering only when its unresolved geometry is below a declared screen-space error; otherwise it must refine. Its enclosing cube is never rendered as terrain. The three procedural detail frequency bands may help bound omitted displacement without enumerating the entire planet, but that construction remains an unproven candidate.

The existing GBuffer's single normal is insufficient for a general mixture of differently shaded voxel faces. Prototype filtering after lighting first to establish an appearance reference, then evaluate a compact mixture consumed by engine lighting. Preserve sharp resolved faces and silhouettes; do not make an independent smooth terrain surface or use temporal blur to hide incorrect geometry.

Use a bounded resident hierarchy with local publication transactions. Keep valid predecessor coverage until a replacement region and its dependent summaries are ready; prioritize current visible error, edits and predicted arrival, and cancel obsolete region/revision requests. Spatially index GPU edits. The generic component should expose reusable canonical query/edit capabilities to simulation as well as rendering. [Aokana](https://arxiv.org/html/2505.02017v1) provides relevant shallow hierarchy, occupancy-mask and visibility-scheduling ideas, but its parent-voxel LOD and lack of runtime modification do not meet this contract.

The first decisive experiment is a set of canonical patches: slope, silhouette ridge, cave opening, thin wall and destructive edit. Capture spatially supersampled linear-light references with coverage, depth, material, face contribution and sunlight outputs. Move the camera across pixel and region boundaries and rotate the sunlight. Disable temporal history during the initial comparison. Compare cached exact bricks, derived surface records and transitions; measure construction cost, steady rendering, memory and local edit propagation. Reject a candidate that smooths resolvable features, exposes parent cubes, repeatedly evaluates the full source for every ray, or only looks stable after temporal accumulation. Passing this experiment is a prerequisite for returning to the complete planetary flight and engine gates, not a replacement for them.

The historical [surface reference checkpoint](SURFACE_REFERENCE_2026_09_27.md)
contains 32 patch/light/translation cases, but its capture dispatch ran before
current-frame graphics. Its joint appearance measurements are superseded.
The [corrected appearance experiment](APPEARANCE_2026_09_27.md) adds an explicit
graphics-stream compute API and changing-frame GPU regression. Rebuild the
spatial lighting reference before using it to judge a rendering approach.

The [exact local cache experiment](SURFACE_CACHE_2026_09_27.md) now supplies
compact authored-cell bricks, halo-aware edit invalidation, bounded logical
residency and a GPU decoder/traverser. It passes 294,912 material checks and
42,696 local ray comparisons. Direct authored sampling removes severe repeated
storage-grid construction at 1 m. Small warm GPU probes favor empty-microbrick
skipping, but do not qualify frame performance. An overlapping-wall fixture
rejects unoccluded face-area mixtures as a filtered appearance solution.
The [bounded patch integration](SURFACE_PATCH_2026_09_27.md) now runs through the
actual GBuffer and lighting graph, with bounded revision-safe uploads. The
three-grid captures and adversarial ray tests expose and correct near-edge
precision errors. That published precise path is slower in the decisive
10/30 cm cases. The [exact skipping checkpoint](SURFACE_SKIP_2026_09_27.md)
removes redundant precision work and crosses certified empty microbricks/tiles
without changing hit ownership. A split traversal/repair queue was slower than
the improved single precise pass and has been removed. Repeated 720p cave
controls and a 720p natural-ground control still favor the canonical renderer
over the cache; the improved cache remains an opt-in correctness baseline.
All 4.64 million three-grid primary samples match the published precise
captures (the lighting-capture correction above limits joint color claims), and a 180-frame local motion recording passes hit/sunlight validity
checks. Distant face stippling remains visible. The next architecture step
must address visible appearance and duplicated traversal/residency costs,
rather than infer planetary performance from a faster cache dispatch.
Visibility-aware far data, global scheduling, continuous motion, physical
memory and full-engine gates remain open. The planetary goal is still active.

The [current-frame appearance experiment](APPEARANCE_2026_09_27.md) now includes
fresh schema-2 captures. Its opt-in nine-sample resolve lowers whole-image error
in 28 of 32 small reference cases and preserves resolved pre-AA pixels, but it
leaves a silhouette regression and produces no change in the tested 720p slope.
The 720p 0.12 ms stage measurement is its copy/guard path. Strong foreshortened
face bands remain. This rules out treating distance-only screen reconstruction
as the far appearance solution; projected face visibility/coverage and coherent
world-space data remain the next decisive architecture work. The source stays
experimental and disabled by default.
