# Helio → SceneDB object storage migration

## Objective

Replace `helio`'s hand-rolled CPU bookkeeping for in-scene objects and their
properties — everything currently living in `DenseArena`/`SparsePool`
(`ObjectRecord`, `LightRecord`, `MaterialRecord`, `DecalRecord`,
`WaterVolumeRecord`/`WaterHitboxRecord`, the three foliage record types,
`PostProcessVolumeRecord`, `ReflectionCaptureRecord`, `SublevelRecord`/
`PortalRecord`, `VoxelVolumeRecord`, `VirtualObjectRecord`/
`VirtualMeshRecord`, `MultiMeshRecord`/`SectionedInstanceRecord`,
`TextureRecord`) — with `pulsar_scenedb::World`, the CPU archetype ECS.

**Explicitly out of scope**: `helio_core::GpuScene` and every `Gpu*Buffer`
manager on it (`instances`, `draw_calls`, `indirect`, `visibility`,
`compacted_indices*`, `materials` GPU buffer, `lights` GPU buffer, voxel
pools, BLAS/TLAS, shadow buffers, coordinate spaces). Those are the
rendering-pipeline buffers Helio's executor owns, operates, and manages —
this migration does not touch them. `Scene::flush()`'s writes *into*
`gpu_scene.*` are preserved byte-for-byte in behavior; only what feeds
those writes changes.

This is why Helio is "the ultimate test" of SceneDB: object spawning/
manipulation (`insert_object`, `update_object_transform`, etc.) becomes CPU
ECS traffic against `World`, while the render hot loop keeps reading
GPU-resident data that SceneDB's write path (`gpu_scene`, unchanged) still
produces every frame. Two different SceneDB-adjacent paths, exercised by
one real engine, in the same frame.

## Why this maps cleanly

Every Helio handle (`ObjectId`, `LightId`, `MaterialId`, ...) is already a
`{slot: u32, generation: u32}` generational handle (`handles.rs`) — the
exact same shape as `pulsar_scenedb::Entity`. `DenseArena<T, H>`'s promise
(stable handle across swap-and-pop compaction, unstable physical index) is
also exactly `World`'s promise for `Entity`. So most record types
(`ObjectRecord`, `LightRecord`, `DecalRecord`, water, foliage,
post-process volumes, reflection captures, virtual objects) translate
directly: `DenseArena<T, H>` → one or more `#[derive(SceneStore)]`
components on a `World` entity, `H` → a thin newtype wrapping
`pulsar_scenedb::Entity`.

## The one real fork: `SparsePool`'s stable-slot promise

`MaterialRecord`, `TextureRecord`, and `MultiMeshRecord` use `SparsePool`,
not `DenseArena`, because their **raw slot index** — not just their
handle — is baked directly into other GPU data:
`GpuInstanceData.material_id` is a raw `u32` slot number, not a handle.
`SparsePool` never compacts (`remove` tombstones in place), so that baked
integer never goes stale.

`World`'s `Entity::index()` is stable for the entity's lifetime (same
guarantee), but it is not a *small dense array index* usable directly as
`material_id` — it's an index into `World::entity_slots`, sparse across
every entity in the whole scene, not just materials.

**Resolution**: keep a small, explicit slot allocator (`MaterialSlots`,
`TextureSlots`, `MultiMeshSlots` — a free-list of `u32`, no data, ~20 lines
each, same shape as `SubsystemRegistry`'s `name_to_type` map but for
`Entity → slot`) alongside the `World` entity for GPU-slot-stable record
types. The record's *data* (ref counts, texture refs, graph hash — the
"CPU authority" `flush()` mirrors into `gpu_scene`) lives as `World`
components exactly like every other record type; only the raw integer
identity used for baked GPU cross-references gets its own tiny stable-slot
layer instead of coming from `Entity::index()`. This is a strictly smaller
amount of bespoke code than today's full `SparsePool<T, H>` (no `T`
storage, no generation tracking — `World` already does that) — it is not
"keep SparsePool", it's "replace SparsePool's storage half with `World`,
keep only its slot-stability half".

## Phased rollout

Each phase preserves every existing public `Scene` method signature
(`insert_object`, `update_object_transform`, ... — call sites across the
rest of `helio` and all of Pulsar-Native's `engine_backend` do not change)
and the exact GPU-write sequence documented in `scene/flush.rs`/
`scene/objects/rebuild.rs`. Order chosen by centrality — `ObjectRecord` is
the type everything else (sublevels, portals, multi-mesh sections, bake)
composes on top of, so it goes first and is the architecture's proof
point; the `SparsePool`/stable-slot types go last since they're the one
genuinely new mechanism (everything else is close to a mechanical
`DenseArena` → `World` port).

1. **`ObjectRecord`/`ObjectId`** (this branch's first slice) — `objects:
   DenseArena<ObjectRecord, ObjectId>` → `World` + `ObjectId(Entity)`.
   Rewrites the direct-`dense`-field call sites that depend on
   `DenseArena`'s public field layout: `scene/objects/update.rs:266`
   (`update_lightmap_indices`), `scene/sublevels.rs:100-122`
   (`tag_group_with_coordinate_space`), `scene/groups/*.rs` (group
   membership/transform/visibility, all O(N) scans over
   `objects.get_dense_mut`).
2. **`LightRecord`/`LightId`**, **`DecalRecord`/`DecalId`** — same shape
   as objects, no ref-counting complexity, good second step.
3. **`WaterVolumeRecord`/`WaterHitboxRecord`**,
   **`PostProcessVolumeRecord`**, **`ReflectionCaptureRecord`** — all
   `Pod`/`Zeroable` GPU-mirror types today; straightforward `SceneStore`
   components. Note these are *not* written by `flush()` at all (the
   renderer reads dirty ranges/slices directly) — that consumption path
   is unchanged, only the CPU storage backing `get_..._gpu_slice()`
   changes from `bytemuck::cast_slice(arena.dense)` to a `World` query
   materialized into a scratch `Vec` (or a SceneDB columnar page read, if
   profiling shows the per-flush `Vec` collection is hot — see Open
   Questions).
4. **Foliage** (`FoliageTypeRecord`/`FoliageLayerRecord`/
   `FoliageInteractorRecord`) — same shape, plus `FoliageLayer.types:
   Vec<FoliageTypeId>` is the first cross-entity reference in this
   migration; becomes `Vec<Entity>` (or a `RelationIndex` if the
   many-to-many shape grows — not needed at this cardinality).
5. **`SublevelRecord`/`PortalRecord`** — composes on top of objects
   (tags `ObjectRecord.instance.flags`, already migrated in phase 1) and
   the shared `coordinate_space_free`/`coordinate_space_next` allocator,
   which is untouched (it's already a stable-slot allocator over a fixed
   GPU buffer, the exact pattern phase 6 introduces generally).
6. **`MaterialRecord`/`MaterialId`, `TextureRecord`/`TextureId`,
   `MultiMeshRecord`/`SectionedInstanceRecord`** — the stable-slot fork
   above. `SectionedInstanceRecord` composes on phase 1's objects
   (`section_objects: Vec<ObjectId>`) plus its own `MultiMeshId`.
7. **`VoxelVolumeRecord`**, **virtual geometry**
   (`VirtualObjectRecord`/`VirtualMeshRecord`) — last because they're the
   least central (voxel volumes and VG objects don't compose with or get
   composed by anything else in this list) and each carries its own
   nontrivial payload (`VoxelOctree`, meshlet arrays) worth migrating
   with full attention rather than batched alongside simpler types.
   `VirtualMeshId` also needs to gain a real generation (today it's a
   bare monotonic `u32`, the one handle type in `handles.rs` not built
   from `define_handle!`) as part of becoming a `World` entity.

## Open questions worth resurfacing before phase 3+

- **Per-flush `Vec` materialization for the "renderer reads directly"
  types** (water, post-process, reflection captures): today
  `get_water_volumes_gpu_slice()` returns `&[GpuWaterVolume]` via
  `bytemuck::cast_slice(&arena.dense)` — an actual zero-copy slice over
  contiguous storage. A naive `World` port
  (`world.query::<&WaterVolumeComponent>().map(|(_, c)| c.gpu).collect()`)
  allocates a fresh `Vec` every call. Whether that matters depends on how
  hot these call sites are relative to `flush()`'s existing per-frame
  allocation budget — worth profiling in phase 3, not guessing now.
- **`custom_actors: Vec<Box<dyn SceneActorTrait>>`** is unaffected by
  every phase above (it stores boxed trait objects, not arena records) —
  noted here only so it isn't mistaken for an oversight.

## Ground truth this doc is based on

Every struct field, method signature, and file:line reference above was
pulled from the actual source at branch `scenedb-object-storage`'s base
(`main` @ `50f0b1d4`) — see the exploration notes retained in this
repository's PR description for the full per-type breakdown (all 13
record types + `DenseArena`/`SparsePool`/`handles.rs`/`actor.rs`/
`lifecycle.rs`/`flush.rs`/`objects/rebuild.rs`/`objects/update.rs` in
complete field-and-signature detail).
