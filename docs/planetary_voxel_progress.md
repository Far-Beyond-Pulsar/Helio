# Production planetary voxel terrain progress

Status: active umbrella tracker, 2026-07-14

This document tracks the cross-repository implementation of the architecture in
[`planetary_voxel_renderer_plan.md`](planetary_voxel_renderer_plan.md) and
Pulsar's `docs/planetary-voxel-terrain.md`. The umbrella pull request remains a
draft until every production gate passes. Small issue-scoped pull requests are
reviewed and merged into this branch as they become complete.

Helio `v4` is the renderer integration base. Pulsar `main` is authoritative for
terrain state. The existing `voxel_demo`, `voxel_demo_raymarch`, their passes,
and `helio-voxel-core` remain unchanged regression baselines.

## Completed foundations

- [x] Pulsar architecture: [Pulsar-Native#304](https://github.com/Far-Beyond-Pulsar/Pulsar-Native/pull/304)
- [x] Helio renderer contract: [Helio#54](https://github.com/Far-Beyond-Pulsar/Helio/pull/54)
- [x] Helio meshlet construction, LOD, culling, and debug repair: [Helio#55](https://github.com/Far-Beyond-Pulsar/Helio/pull/55)
- [x] Helio resize and stale input-capture repair: [Helio#57](https://github.com/Far-Beyond-Pulsar/Helio/pull/57)
- [x] Pulsar Helio-v4 integration and caller audit: [Pulsar-Native#305](https://github.com/Far-Beyond-Pulsar/Pulsar-Native/pull/305), [#308](https://github.com/Far-Beyond-Pulsar/Pulsar-Native/pull/308), and [#310](https://github.com/Far-Beyond-Pulsar/Pulsar-Native/pull/310)
- [x] Pulsar deterministic sparse terrain core: [Pulsar-Native#311](https://github.com/Far-Beyond-Pulsar/Pulsar-Native/pull/311)

## Active milestone

- [ ] Helio bounded renderer-facing residency contract: [Helio#62](https://github.com/Far-Beyond-Pulsar/Helio/pull/62) ([issue #59](https://github.com/Far-Beyond-Pulsar/Helio/issues/59))

## Implementation milestones

- [ ] Pulsar terrain component/subsystem and asynchronous work queues
- [ ] Helio bounded GPU page atlas, hash table, upload, eviction, and device-loss recovery
- [ ] Earth-radius camera-local coordinates and precision validation
- [ ] View-driven page demand, streaming, and strict CPU/GPU/VRAM budgets
- [ ] GPU Transvoxel versus manifold dual-contouring extraction bake-off
- [ ] Generation-safe bounded meshlet publication and indirect drawing
- [ ] Crack-free LOD selection, transition topology, and horizon-scale coverage
- [ ] Exact hierarchical destruction, compaction, snapshots, and recovery
- [ ] Collision, physics, and bounded detached terrain bodies
- [ ] Deterministic replication and late-join reconstruction
- [ ] Terrain tooling, debug views, profiling, and `planet_voxel_demo`
- [ ] Cross-platform production hardening and final Pulsar integration pin

Each milestone receives its own issue and pull request. This list is updated with
those links and measured evidence; checking a box requires the corresponding
acceptance gates, not merely a compiling implementation.

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
