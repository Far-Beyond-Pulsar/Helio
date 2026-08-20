# Production planetary voxel terrain progress

Status: active umbrella tracker, 2026-08-16

The production terrain source is Pulsar's canonical, sparse, hierarchical 3D
SDF. Helio receives versioned page deltas and owns only bounded, disposable GPU
caches and extracted surfaces. A renderer-side plane, spherical shell, or
procedural fallback is not a planet implementation and must not be exposed as
one.

The existing `voxel_demo` and `voxel_demo_raymarch` remain unchanged regression
baselines. The future exact cubic path keeps 10 cm blocks and has no smooth
fallback; the smooth SDF path uses an explicit per-planet sample scale.

## Current work

- [x] Make the deterministic, pole-free volumetric SDF the only production
  planet authority ([Pulsar-Native#579](https://github.com/Far-Beyond-Pulsar/Pulsar-Native/pull/579)).
- [x] Carry the smooth planet's physical sample scale through planning, edits,
  persistence, renderer payloads, and cache addressing.
- [ ] Publish only canonical Pulsar pages through the transactional Helio surface
  path tracked by Helio#230 and Pulsar-Native#578.
- [ ] Validate ground, orbit, poles, signed boundaries, teleports, destruction,
  save/load, eviction, device recovery, and refinement latency with the live
  `PlanetTerrainComponent`.
- [x] Remove obsolete planar and spherical-shell runtime/demo paths rather than
  maintaining compatibility with them.

## Promotion gates

- [ ] The planet is always visible at astronomical distances with bounded coarse
  coverage and refines around the camera without redundant regeneration.
- [ ] Mixed-LOD extraction is watertight and never publishes simultaneous coarse
  and fine ownership of the same boundary.
- [ ] Relief, overhangs, and caves come from canonical 3D SDF samples and survive
  edits, persistence, eviction, and GPU cache rebuilds.
- [ ] CPU memory, GPU memory, upload bandwidth, generation latency, and
  refinement-to-visible latency remain within documented bounded budgets.
- [ ] Existing mesh and raymarch voxel demos compile and behave unchanged.
- [ ] Pulsar and Helio integration callers pass on Windows, Linux, macOS, and the
  WebGPU fallback tier before either integration PR is promoted.

The umbrella work remains incomplete until every gate passes. A failed
topology, precision, visibility, or memory gate requires fixing the production
path, not introducing a fallback planet.
