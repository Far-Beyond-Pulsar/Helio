# Production planetary voxel terrain progress

Status: active umbrella tracker, 2026-07-30

This document tracks the cross-repository implementation of the architecture in
[`planetary_voxel_renderer_plan.md`](planetary_voxel_renderer_plan.md) and
Pulsar's `docs/planetary-voxel-terrain.md`. The umbrella pull request remains a
draft until every production gate passes. Small issue-scoped pull requests are
reviewed and merged into this branch as they become complete.

Helio `main` is the renderer integration base and Pulsar `main` is authoritative
for terrain state. New work and pull requests use `main` to `main`; the former
Helio `v4` branch is no longer an integration or pull-request target. The
existing `voxel_demo`, `voxel_demo_raymarch`, their passes, and
`helio-voxel-core` remain unchanged regression baselines.

## Completed foundations

- [x] Pulsar architecture: [Pulsar-Native#304](https://github.com/Far-Beyond-Pulsar/Pulsar-Native/pull/304)
- [x] Helio renderer contract: [Helio#54](https://github.com/Far-Beyond-Pulsar/Helio/pull/54)
- [x] Helio meshlet construction, LOD, culling, and debug repair: [Helio#55](https://github.com/Far-Beyond-Pulsar/Helio/pull/55)
- [x] Helio resize and stale input-capture repair: [Helio#57](https://github.com/Far-Beyond-Pulsar/Helio/pull/57)
- [x] Pulsar Helio-v4 integration and caller audit: [Pulsar-Native#305](https://github.com/Far-Beyond-Pulsar/Pulsar-Native/pull/305), [#308](https://github.com/Far-Beyond-Pulsar/Pulsar-Native/pull/308), and [#310](https://github.com/Far-Beyond-Pulsar/Pulsar-Native/pull/310)
- [x] Pulsar deterministic sparse terrain core: [Pulsar-Native#311](https://github.com/Far-Beyond-Pulsar/Pulsar-Native/pull/311)
- [x] Helio bounded renderer-facing residency contract: [Helio#62](https://github.com/Far-Beyond-Pulsar/Helio/pull/62)
- [x] Helio bounded GPU residency atlas and page table: [Helio#65](https://github.com/Far-Beyond-Pulsar/Helio/pull/65)
- [x] Pulsar bounded terrain runtime subsystem and component: [Pulsar-Native#327](https://github.com/Far-Beyond-Pulsar/Pulsar-Native/pull/327)
- [x] Helio bounded GPU manifold candidate and parity oracle: [Helio#106](https://github.com/Far-Beyond-Pulsar/Helio/pull/106)
- [x] Matched extraction bake-off and GPU Transvoxel promotion: [Helio#107](https://github.com/Far-Beyond-Pulsar/Helio/pull/107)

## Active milestones

- [ ] Generation-safe bounded terrain meshlet publication, GPU culling, matched
  page-baseline measurement, and truthful debug views:
  [issue #119](https://github.com/Far-Beyond-Pulsar/Helio/issues/119)
- [ ] Production visual and performance validation of the complete
  Pulsar-to-Helio path, including movement, LOD transitions, residency
  replacement/eviction, resize, and the debug views from issue #119

## Implementation milestones

- [x] Pulsar terrain component/subsystem and asynchronous work queues ([Pulsar-Native#327](https://github.com/Far-Beyond-Pulsar/Pulsar-Native/pull/327))
- [x] Helio bounded GPU page atlas, hash table, upload, eviction, and device-loss recovery ([Helio#65](https://github.com/Far-Beyond-Pulsar/Helio/pull/65))
- [x] Earth-radius camera-local coordinates and precision validation ([Helio#109](https://github.com/Far-Beyond-Pulsar/Helio/pull/109))
- [x] Bounded 2:1 view demand, immutable renderer deltas, and generation-safe
  residency reconciliation ([Pulsar-Native#339](https://github.com/Far-Beyond-Pulsar/Pulsar-Native/pull/339),
  [#343](https://github.com/Far-Beyond-Pulsar/Pulsar-Native/pull/343), and
  [#354](https://github.com/Far-Beyond-Pulsar/Pulsar-Native/pull/354))
- [x] GPU Transvoxel versus manifold dual-contouring extraction bake-off ([Helio#107](https://github.com/Far-Beyond-Pulsar/Helio/pull/107))
- [ ] Generation-safe bounded meshlet publication and indirect drawing
  ([Helio issue #119](https://github.com/Far-Beyond-Pulsar/Helio/issues/119), in
  implementation and validation)
- [ ] Crack-free LOD selection, transition topology, and horizon-scale coverage
- [ ] Exact hierarchical destruction, compaction, snapshots, and recovery
- [ ] Collision, physics, and bounded detached terrain bodies
- [ ] Deterministic replication and late-join reconstruction
- [ ] Terrain tooling, debug views, profiling, and `planet_voxel_demo`
- [ ] Cross-platform production hardening and final Pulsar integration pin

Each milestone receives its own issue and pull request. This list is updated with
those links and measured evidence; checking a box requires the corresponding
acceptance gates, not merely a compiling implementation.

### Issue #119 validation evidence

- The terrain crate passes all 72 CPU, GPU, publication, Transvoxel, layout, and
  randomized culling-parity tests on an RTX 3060 Vulkan adapter. Strict
  terrain-crate Clippy and the `planet_voxel_demo` target are clean; warnings
  printed from other crates are pre-existing.
- The release validation demo passed interactive movement, resize, page/meshlet
  switching, and all truthful debug-view checks on 2026-07-30.
- The retained `voxel_demo` and `voxel_demo_raymarch` sources are unchanged and
  both targets compile.
- A reproducible `planet_voxel_demo --benchmark` run measured a bounded
  culling-heavy 64-page fixture. Initial extraction, copy, meshlet build, and
  atomic publication completed one job per frame with full-pass CPU p50/p95
  `0.7332/1.3179 ms` and GPU p50/p95 `1.568768/1.92 ms`.
- After 60 warmup frames, 240 matched steady-state samples per path measured
  page-indexed CPU p50/p95 `0.0007/0.0012 ms` and GPU p50/p95
  `0.012288/0.014336 ms`; meshlets measured CPU p50/p95
  `0.0007/0.0013 ms` and GPU p50/p95 `0.079872/0.088064 ms`.
  The meshlet path compacted 6,272 resident meshlets to 5,204 indirect draws,
  rejected 1,068 by the frustum, and reported zero stale, overflow, or invalid
  candidates. This fixture does not justify a meshlet performance promotion,
  so page-indexed rendering remains the default, the meshlet path remains
  directly selectable for validation/debugging, and no GPU-speed claim is made.

## Final promotion gates

- [ ] Existing mesh and raymarch voxel demos compile and behave unchanged
- [ ] Exact Rust/WGSL layouts and stale-generation behavior pass executable tests
- [ ] Real-radius precision and camera-origin rebasing remain stable at every tested altitude
- [ ] LOD topology has no holes, overlaps, or cracks across randomized transitions
- [ ] Destruction latency, compaction, save/load, corruption fallback, and replay are bounded
- [ ] CPU memory, GPU memory, upload bandwidth, extraction latency, and frame time stay within documented budgets
- [ ] Resize, minimized windows, device loss, allocation failure, and backpressure recover safely
- [ ] Pulsar integration callers compile and run against the promoted Helio revision
- [ ] Windows, Linux, macOS, and the WebGPU fallback satisfy their documented capability tier

The umbrella pull request must not be marked ready or merged while any final gate
is unchecked. A failed extraction, topology, precision, or memory gate triggers
redesign at that milestone instead of weakening the target.
