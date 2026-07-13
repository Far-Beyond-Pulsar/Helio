# Planetary voxel renderer contract

Status: companion rendering plan for `Pulsar-Native/docs/planetary-voxel-terrain.md`, 2026-07-13

## Scope

Helio will gain a separate production planetary voxel path. The current paths remain untouched:

- `helio-pass-voxel-mesh` plus `voxel_demo`
- `helio-pass-voxel-raymarch` plus `voxel_demo_raymarch`
- `helio-voxel-core` and the fixed 64^3 `VoxelTerrain` example component

The new path does not increase their constants, reuse their fixed pools, change their layouts, or turn either demo into the planet implementation.

## New crates

- `helio-planet-voxel-core`: renderer-facing page keys, upload/eviction messages, GPU POD layouts, limits, and validators.
- `helio-pass-planetary-voxel`: bounded resident page buffers, GPU extraction, meshlet output, culling, G-buffer/depth/shadow drawing, and profiling.
- `planet_voxel_demo`: standalone validation from a 10 cm interaction patch to an Earth-radius horizon.

Pulsar remains authoritative for generation, edits, hierarchy, persistence, networking, physics, and canonical coordinates. Helio accepts versioned page deltas and may evict/rebuild them at any time.

## Render-space contract

Every frame supplies a snapped planet-frame origin and camera-relative page addresses. Canonical `i64`/`f64` planet coordinates never enter existing Helio object transforms or the existing camera uniform as absolute values.

The pass receives:

```text
PlanetFrameUniform
PageUpload { key, generation, lod, cells }
PageEvict { key, generation }
VisiblePageSet { key, generation, transition_mask }
MaterialPaletteDelta
```

It produces normal Helio geometry resources and profiling counters. Generation numbers make late uploads, mesh jobs, and evictions harmless.

## GPU organization

- A fixed-budget cell page atlas in large storage buffers.
- A GPU hash/page table for neighbor and key-to-slot lookup.
- Fixed-budget vertex/index/meshlet arenas with fence-aware reuse.
- Indirect draw/count buffers and per-page bounds.
- Per-microbrick dirty queues and prefix-sum compaction.
- No buffer or bind group per terrain page.
- No mandatory mesh-shader, sparse-texture, ray-tracing, or vendor extension.

The provisional total terrain VRAM budget is 2 GB on the reference tier, partitioned and configurable. Allocation failure triggers scheduled eviction/backpressure, never an unchecked allocation or process-wide GPU failure.

## Meshlet prerequisite gate

The current virtual-geometry meshlet path is not assumed to be production-ready and the planetary pass must not silently inherit it. Before the planet renderer depends on shared meshlet infrastructure, audit and either repair or replace the existing path across `libhelio`, `helio::vg`, `helio-pass-virtual-geometry`, its shaders, the default graphs, and every example/caller.

The initial code audit already identifies risks that require executable tests:

- Position-only vertex welding can merge UV seams, hard-normal splits, tangents, colors, or other distinct attributes.
- Every instance currently expands every meshlet from all eight LODs into frame buffers, making CPU memory/upload work proportional to `instances * all_lods`.
- `lod_error` currently stores an integer LOD label rather than a measured geometric error, while selection uses each meshlet's own projected radius. This can select different LODs within one object and does not provide hierarchical replacement coverage.
- The current path is compute-cull plus classic indexed indirect draws, not a hardware mesh-shader path. WebGPU portability requires this fallback, but it needs measured draw/compaction costs and must not be described as mesh-shader execution.
- Current tests mostly validate mirrored constants and threshold ordering. They do not prove attribute preservation, index/bounds correctness, cone-cull correctness, single-coverage LOD selection, Hi-Z conservatism, or bounded dynamic rebuild behavior.

Use meshoptimizer's documented meshlet builder and bounds conventions as the reference CPU implementation. Preserve full vertex identity, store real simplification error and explicit per-LOD ranges/hierarchy, and use the documented perspective cone test. Compare cluster configurations such as 64 vertices/64 triangles and 64 vertices/96 triangles on representative static assets and generated voxel surfaces instead of hard-coding one vendor claim.

Promotion gates:

- Exact attribute/index preservation against the source mesh, including seam and hard-edge fixtures.
- Bounds contain all referenced vertices; cone culling has zero false rejects in randomized CPU/GPU parity tests.
- Exactly one complete LOD representation covers each object or terrain region; transitions never mix unrelated cluster-local LOD choices.
- Static instancing does not duplicate immutable meshlet descriptors per instance, and transform-only changes do not rebuild or re-upload them.
- Dynamic terrain rebuild publishes generation-tagged ranges atomically, reuses bounded arenas, and leaves the prior generation visible until replacement completes.
- GPU compaction never exceeds output capacity and has an overflow/backpressure counter rather than out-of-bounds writes.
- Benchmarks record build time, meshlet fill, vertex reuse, cull rejection, indirect command count, upload bytes, CPU frame cost, and GPU depth/G-buffer cost against the conventional indexed path.
- Existing virtual-geometry callers, debug modes, graph order, and the two retained voxel demos pass regression checks.

If the repaired generic path cannot satisfy dynamic-terrain rebuild and bounded-memory gates, the planetary pass gets a terrain-specialized meshlet publisher behind the same tested descriptor/draw contract; it does not block on or corrupt static virtual geometry.

## Graph integration

The current default graph is locked during construction. Add an opt-in geometry-stage extension/builder hook in `helio-default-graphs`; do not append the pass after graph lock and do not make it a default dependency of every renderer.

With no hook supplied:

- Existing graph pass order is unchanged.
- Existing constructors and examples behave the same.
- Required wgpu features and limits are unchanged.
- Renderer resize/rebuild behavior is unchanged.

Pulsar explicitly builds the extended graph. Planet terrain participates in depth, G-buffer, shadows, Hi-Z/occlusion, deferred lighting, TAA, post-processing, debug views, and performance overlays.

## Extraction gate

Implement GPU Transvoxel and feature-preserving manifold dual contouring against the same page/cache API. Promote exactly one after measuring dirty-page latency, total frame time, mesh size, geometric error, crack behavior, and manifold topology. The losing implementation is removed from the production graph after its benchmark evidence is recorded.

The render mesh is a cache. All edits and material truth remain voxel data owned by Pulsar.

## Compatibility checks

Before promotion:

- `cargo check -p helio --lib`
- `cargo check -p examples --bin voxel_demo --bin voxel_demo_raymarch`
- Existing terrain unit tests
- Default graph pass-order snapshot with no extension
- New page-layout Rust/WGSL parity tests
- New stale-generation, eviction, resize, device-loss, and budget tests
- New GPU readback tests for extraction and LOD seams
- Meshlet seam/hard-edge fixtures, CPU/GPU bounds and cone parity, LOD single-coverage, overflow, rebuild, and benchmark gates
- Pulsar checks for both direct Helio renderer constructor paths after its pinned revision is updated

Pulsar currently pins Helio `b88e366d`, while the retained voxel baseline is `3210590`. Updating that dependency and adapting the changed renderer constructor is milestone 0, separate from the planetary renderer itself.

## Meshlet research basis

- [meshoptimizer clusterization documentation](https://github.com/zeux/meshoptimizer#clusterization): reference builder, meshlet layouts, vendor-sensitive cluster sizes, spatial splitting, and the exact perspective cone-culling formula.
- [Direct3D 12 mesh shader samples](https://learn.microsoft.com/en-us/samples/microsoft/directx-graphics-samples/d3d12-mesh-shader-samples-win32/): Microsoft's reference separation of conversion, meshlet rendering, culling, instancing, and object-level dynamic LOD selection.
- [NVIDIA Introduction to Turing Mesh Shaders](https://developer.nvidia.com/blog/introduction-turing-mesh-shaders/): task-level cluster culling and the rationale for topology/culling-coherent meshlets.
- [AMD mesh-shader optimization and best practices](https://gpuopen.com/learn/mesh_shaders/mesh_shaders-optimization_and_best_practices/): cross-vendor occupancy, amplification-group, vertex-reuse, and cluster-size guidance.

Helio is wgpu/WebGPU-based today, so the production baseline remains compute culling plus compacted indexed indirect drawing. Native mesh-shader backends may be evaluated later as optional accelerators only after they can preserve the same data contract and fallback behavior.
