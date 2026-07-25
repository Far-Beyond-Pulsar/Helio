# SceneDB renderer migration

Issue: [Helio #121](https://github.com/Far-Beyond-Pulsar/Helio/issues/121)

## Ownership rule

SceneDB owns authoritative scene state and GPU scene buffers. Helio owns render
pipelines and derived per-view data. A Helio pass may borrow SceneDB buffers
directly, but Helio must not rebuild a CPU snapshot or upload a second copy of
the same scene records.

`helio-scenedb` is the only Helio crate that depends on `pulsar_scenedb`. The
workspace pins the same SceneDB revision as Pulsar so public SceneDB types and
wgpu handles have one Cargo package identity.

## Current migration slice

The optional `helio/scenedb` feature adds an owned `SceneDbRenderSource`.
Attaching it to `Renderer` publishes borrowed `SceneDbFrameResources` into the
live `FrameResources` for every frame:

- cull bind group: transforms, instance metadata, slot mirror, generations,
  and mesh metadata;
- draw bind group: clusters, meshlets, cell metadata, and materials;
- SceneDB's global vertex and index arenas.

This path allocates Helio bind groups once at attachment. It performs no scene
copy and no per-frame upload. Detaching or destroying a renderer drops only
Helio-owned bindings and cloned wgpu handles; the SceneDB stores remain alive.

No production pass consumes this slot yet. Existing `Scene`, constructors,
graphs, demos, and downstream integrations therefore retain their exact
behavior when the feature is disabled or no source is attached.

## Compatibility inventory

The following existing Helio state is still authoritative for current passes
and cannot be removed until a replacement consumer is validated:

| State | Current writers | Current consumers | SceneDB migration state |
| --- | --- | --- | --- |
| Object transforms, instance metadata, generations, visibility | `Scene` object/virtual-object APIs and `Scene::flush` | GBuffer, depth, shadow, virtual geometry, picking | SceneDB buffers published; production consumers pending |
| Mesh/cluster/meshlet metadata and geometry | mesh and virtual-geometry insertion/removal | GBuffer, depth, shadow, virtual geometry | SceneDB buffers published; format/PBR integration pending |
| Materials and texture references | material/texture APIs | GBuffer, transparent, decals, lighting | material registry published; texture-view/sampler table pending |
| Lights and shadows | light actor APIs | light cull, shadow, deferred lighting, billboards/coronas | not represented by this adapter |
| Camera and per-view culling | renderer camera update | all view-dependent passes, picking, gizmos | remains Helio-derived per view |
| Groups and editor visibility | group/editor APIs | culling, picking, debug views | compatibility mapping pending |
| Voxels and planetary terrain | voxel actors and terrain passes | voxel mesh/raymarch/planet passes | separate payloads; must remain working |
| Water, decals, post-process volumes, reflection captures | specialized `Scene` APIs | corresponding graph passes | not represented by this adapter |
| Resize and external-device lifecycle | `Renderer` | graph rebuild and all target-sized passes | unchanged; attached source must survive rebuild |

Pulsar currently has additional direct `renderer.scene_mut()` writers in the
static-mesh and light components, and a `sync_scene()` bridge in
`engine_backend` that materializes Pulsar's legacy scene database into Helio
objects. Picking, gizmo dragging, selection, and the physics picker also read
Helio `Scene` handles. That bridge must stay until those writers and readers
have moved to the pinned `pulsar_scenedb` API and the equivalence gates below
pass.

## Removal gates

The legacy path may be removed only after all of these are true:

1. A SceneDB-backed GBuffer path matches the current material, texture,
   normal, object-ID, and motion-vector outputs.
2. Depth, shadow, transparency, decals, reflections, and virtual geometry
   either consume SceneDB directly or have an explicitly retained owner.
3. Picking, selection, gizmos, and debug views resolve stable SceneDB IDs.
4. Light components and all static-mesh writers publish to SceneDB without a
   full-scene synchronization loop.
5. Deletion, stale-generation rejection, streaming eviction/rehydration, and
   dirty-range uploads pass lifecycle tests.
6. Resize, external-device construction, camera movement, meshlet/LOD debug,
   both voxel demos, and the Pulsar editor pass visual validation.
7. CPU frame time, GPU pass time, bytes uploaded, and memory use are no worse
   than matched legacy baselines; the full `sync_scene()` scan is absent.

## Planned stacked work

1. Add the SceneDB GBuffer/cull consumer using the published frame slot.
2. Add PBR material and texture bindings plus stable picking/object IDs.
3. Move Pulsar rendering components to SceneDB writers and validate deltas.
4. Migrate remaining pass families and editor/physics readers.
5. Remove `sync_scene()` and the duplicated Helio `Scene` records only after
   all compatibility and performance gates pass.
