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
4. Address remaining shading, aliasing and content composition failures; validate with directional visibility enabled and representative materials/content.
5. Extend shared component adapters, simulation/query integration, persistence and stress coverage. Validate the complete engine/editor workflow before declaring the goal achieved.

Keep PRs drafts while these gates remain open. Do not merge without explicit user authorization.
