# Every object kind is a plugin

The target the layered rewrite ends at, for every kind of object, and the
order that gets there once. Companion to `layered-rewrite.md` (phases 0-6);
this replaces its per-feature Phase 5 groupings and Phase 6 with one plan.

## 1. The rule (user, 2026-11-02)

- Every object kind is a **Pulsar plugin**. The plugin owns its SceneDB
  component (data and methods), the Helio feature that renders it (its
  passes and their contracts), its scene derivation, its file formats and
  its editor UI.
- **Component methods are the only API** to act on a component. They are
  scriptable, and the editor calls the same methods scripts do. No editor
  side channels: no renderer mailbox messages, `find_pass_mut`, or renderer
  setters for one kind.
- A feature's passes **consume only what components publish** (rows,
  pools, payloads by contract key).
- **Helio knows no object kind**: not mesh, light, material, voxel, VG,
  water, sky, fog, decal, portal, sprite, lens or particle. It has no API
  for one.

## 2. What Helio is, when it knows no kind

How others draw the line:
- Bevy: `core_pipeline` defines the frame's phases (opaque, alpha mask,
  transparent, shadow) and views; `PbrPlugin` and `MeshPlugin` queue phase
  items and own lights and materials. The render graph knows no mesh.
- Unreal: the renderer is a closed set of mesh passes fed by primitive
  scene proxies; kinds plug in as proxies, but light types and the
  lighting model are built in. That is the coupling we are removing.
- Unity SRP: the pipeline asset owns everything; renderer features inject
  passes at named events.

Helio takes Bevy's line, with Unity's injection events as its slots.

**Helio core (kind-agnostic, stays in the Helio repo):**
- `helio-core`: graph, scheduling, resources, uploads, shaders,
  profiling, `SceneDerivation`, `BufferContract`/`PoolContract`.
  Leaves it: `asset_types.rs` (`PackedVertex`, `MeshUpload`), `RayInstance`,
  and the keys `MATERIAL_TEXTURES`, `SHADOW_MATRICES`, `BAKED_LIGHTMAP`,
  `PORTAL_VIEWS` (each to its kind's api crate).
- `helio-feature-api`, `helio-graph-asset`, `helio-settings`.
- `helio` (renderer facade): views, cameras, frame loop, surfaces, XR,
  debug draw (lines/tris; splines draw through it like any client). No
  kind setters: `set_ies_texture_view`, `set_color_grading_lut`,
  `set_material_sampler`, template registries, `set_portal_projection_counts`,
  `set_ray_tracing_frame`, `set_ambient`, `set_shadow_quality`,
  `set_deferred_light_timing_ms` all move to their kind (component data or
  feature settings).
- **The frame (`helio-frame-api` + `helio-spine`)**: the contracts every
  kind meets at, and the passes no kind owns.
  - Views: camera uniforms, jitter, motion vectors, view remaps (what
    portals use, by a generic name).
  - **Draw phases** (the main new extension point): `ShadowCasters`,
    `DepthPrepass`, `GBuffer`, `Forward`, `Transparent`, `Overlay`. A phase
    is a graph slot with an attachment contract (formats, load/store) and
    a per-view list of drawers. Kinds register drawers into phases: meshes
    draw into GBuffer/ShadowCasters/Transparent, voxel into GBuffer and
    ShadowCasters, foliage into GBuffer and ShadowCasters, decals into
    GBuffer (after), billboards/coronas/sprites into Overlay. The spine
    runs phases; it draws nothing itself.
  - **G-buffer layout** and the scene-linear colour (`SceneColorFormat`).
  - **Lighting inputs** as generic contracts with stand-ins, never named
    after a kind: `indirect_diffuse` (GI), `indirect_specular`
    (reflection captures, SSR, planar), `sun_visibility` and
    `transmission` (clouds, water caustics), `sky` (atmosphere's sky
    and sun), `light_list` (whatever the lights plugin publishes). The
    lighting model reads these; it never reads water, portals or captures.
  - **Colour chain**: post features read and write "the current scene
    colour" through a chain the spine provides (scene-linear -> AA -> lens
    -> tone map -> output), so no pass hard-codes `lens_color`,
    `fxaa_color`, `tsr_color`.
  - Spine passes no kind owns: background clear, Hi-Z (from depth), TSR,
    the tone map/bloom/exposure (reading a post-settings contract, not
    post-process volumes), buffer view, perf overlay.
  - **Host inputs are declared by features**, not listed by the spine
    (`billboards`, `corona_emitters` leave `HOST_INPUTS`).
- `helio-presets`: presets are graphs of phases, slots and the spine; the
  feature list comes from the engine (each plugin adds its features). The
  presets name no kind.

## 3. The plugins

Each plugin is `plugins/<kind>/` (Pulsar) with: a component crate
(SceneDB component, methods, runtime, file formats), an api crate (its
contracts: keys, row ABIs, WGSL), a feature crate (Helio feature: nodes,
passes, its scene derivation), and its editor crate. Plugins depend on
Helio core and on other plugins' api crates only.

| Plugin | Components | Feature owns | Leaves Helio/Pulsar today's |
|---|---|---|---|
| `pulsar.materials` | material assets, material graph | helio-mats (GPU material ABI, Radiant templates, texture bindings), material texture means | facade template registries, `set_material_sampler`, `MATERIAL_TEXTURES` |
| `pulsar.meshes` | StaticMesh, StaticMeshDraw, SkeletalMesh, LOD, VirtualGeometry | object batch, culling (occlusion, indirect dispatch), mesh drawers (GBuffer, ShadowCasters, Forward, Transparent), skinning, LOD selection, VG (all `helio-pass-vg*`, `helio-vg-*`), ray proxies + TLAS publication, the mesh part of the scene join | gbuffer/shadow/forward/transparent mesh code, `asset_types.rs`, `RayInstance`, `with_virtual_geometry`, authored LOD derivation, VG host in helio_component |
| `pulsar.lights` | Light, ReflectionCapture | light list, light cull, shadow views + atlas (shadow matrix/dirty/atlas; casters come from drawers), the lighting model (deferred light, forward light function), IES, light icons (as billboards) | light part of the scene join, `GpuLight`/`SCENE_LIGHTS` in forward-lit, facade light setters |
| `pulsar.sky` | Atmosphere | atmosphere tables, clouds, sky composite; publishes `sky`, `sun_visibility` | helio-sky-api readers name `sky` contracts only |
| `pulsar.gi` | (none; settings) | probe clipmap, tracers interface; publishes `indirect_diffuse` | — |
| `pulsar.fog` | GlobalFog, LocalFogVolume | volumetric fog and its composite | fog composite out of the spine's postprocess crate |
| `pulsar.post` | PostProcessVolume, CameraPostProcess | volume blend, publishes post settings | PP volume contracts out of the spine |
| `pulsar.lens` | CameraLens | lens, flare (reads `light_list` via lights api) | `lens_color` names in the spine |
| `pulsar.water` | WaterVolume (+ physics hitbox rows via physics api) | water sim and draw, caustics as `transmission` | water reads in deferred light |
| `pulsar.foliage` | Foliage, Wind | placement, drawers | — |
| `pulsar.voxel` | VoxelTerrain, VoxelTerrainLayers | chunk passes, drawers, GI tracer, foliage ground, water terrain | (in progress: layered-rewrite step 6) |
| `pulsar.decals` | Decal | decal drawer (after GBuffer) | — |
| `pulsar.particles` | ParticleEmitter | corona | `corona_emitters` host input |
| `pulsar.sprites` | Sprite | sprite cull/batch/simulate, 2D radiance cascades | — |
| `pulsar.portals` | Portal, SubLevelDisplay | portal cull, draws, projection derivation, view remaps | portal keys in core, `portal_chain_unmap` in deferred light, facade portal setters, `SpaceSync` portal input |
| `pulsar.splines` | Spline | debug-line drawing through the facade's debug draw | renderer's `debug_set_editor_lines("splines")` call from engine_backend |
| `pulsar.physics` | (exists: physics_plugin + pulsar_physics) | water hitbox rows through water's api | — |

Hard cases, and their answer:
1. **The spine drew meshes** (G-buffer, shadows, object batch, culling,
   forward, transparent): the passes become mesh drawers in
   `pulsar.meshes`; the spine keeps the phases and attachments.
2. **Lighting was built in**: the lighting model belongs to
   `pulsar.lights`; it reads lighting-input contracts. A preset without
   the lights plugin draws unlit (albedo + emissive), which is the right
   failure.
3. **Lights are a cross-kind contract** (flare, GI, sky, voxel, fog read
   `scene_lights`): `light_list` is `pulsar.lights`' api contract; readers
   depend on the api crate, not the plugin.
4. **Monolithic joins**: the scene join and environment join split into
   one derivation per kind, each in its plugin's feature crate; the
   hierarchy resolve stays in core (transforms only). Water's sun is the
   lights api's directional light, not "the first directional light".
5. **Spine postprocess crate owned others' contracts**: the tone map keeps
   only the post-settings and colour-chain contracts.

## 4. Order (do it once)

Phase dependencies run materials -> meshes -> lights; everything else
hangs off the frame contracts. Each step deletes the old path in the same
change, keeps the layer gate at 0, and passes the preset snapshot + real
frame A/B (voxel_craft, an indoor cathedral, a mesh-heavy scene).

**A. Frame extension points in Helio core** (nothing moves yet):
1. `helio-frame-api`: draw phases with drawer registration; lighting-input
   contracts with stand-ins; colour chain; feature-declared host inputs.
2. Spine runs phases; today's mesh passes register as drawers from their
   crates (still in Helio). Deferred light reads lighting inputs; water,
   reflection captures, SSR/planar, portals, GI, sky publish them.
3. Post chain on the colour chain; fog composite to the fog feature; PP
   volumes publish post settings.
4. Facade sheds kind setters (each to its kind's settings or component
   data); `asset_types`/`RayInstance`/kind keys to api crates.
5. Joins split per kind (still in Helio, one derivation per feature crate).

**B. Plugins** (each: Pulsar plugin + its Helio crates move out of the
Helio repo into it; component methods for every editor/game operation;
editor moved onto those methods):
materials, meshes (with LOD, skeletal, VG), lights, sky, gi, fog, post,
lens, water, foliage, voxel (step 6, in progress), decals, particles,
sprites, portals, splines. Physics already is one; it gains water's api.

**C. Helio repo is core only**: helio-core, helio-feature-api,
helio-frame-api, helio-spine, helio-graph-asset, helio-presets, helio,
helio-xr, helio-settings, and a test host (`helio-components` +
examples, a host-layer crate that depends on plugin feature crates for its
own demos). The layer gate gains a rule: no crate in the Helio repo may
name a kind's contract outside the test host.

**Done when**: Helio's repo has no kind code outside its test host; each
plugin's editor and game operations are component methods; Pulsar's
`engine_backend` renderer has no kind-specific code; scripting can do
everything the editor does through component methods.
