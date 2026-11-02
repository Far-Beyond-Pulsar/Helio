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

As built, A1-A3 (2026-11-02, branch `rewrite/p6-frame`):
- **Extension points are generic in the feature API**:
  `ExtensionPoint(&'static str)`, `NodeDescriptor::at(point, order)`,
  `NodeRegistry::placed(point)`, and `NodeDescriptor::with_check` (a check
  run on the passes a node added). `helio-feature-api` names no slot or
  phase; `Slot` moved to `helio-frame-api`.
- **`helio-frame-api`** (layer `feature_api`, beside `helio-feature-api`):
  - `Slot`: `Prepare`, `Geometry`, `LightingInputs`, `Lighting`, `Media`,
    `Resolved`, `Display`, `Output` (new: `Prepare` for per-frame tables,
    `Media` for fog and the post settings, `Resolved` for effects between
    AA and the tone map, `Display` for overlays on the final image).
  - `Phase`: `ShadowCasters`, `DepthPrepass`, `GBuffer`, `Forward`,
    `Transparent`, `Overlay`. A phase has three stages, each an extension
    point: **before** (compute a drawer needs first; foliage placement, GI's
    ray generation, portal culling), **open** (the one drawer that clears
    the attachments) and **draw** (drawers that load them). `FrameNode`
    places a node: `in_slot`, `before_phase`, `opens`, `drawer`. How others
    do it: Bevy's phase node draws every queued phase item of a view in one
    render pass; here each kind keeps its own pass (its culling, pipelines,
    membership) and the graph's subpass fusion merges adjacent drawers whose
    attachments match exactly, which is the same GPU work without a central
    draw loop. Compute a drawer needs goes in the phase's before stage so it
    never lands between two fusing drawers (the rule foliage placement
    followed by hand).
  - The **attachment contract** is checked when a drawer builds: every
    attachment of its phase a drawer's pass declares must have the phase's
    format (`Phase::attachments`; the G-buffer layout is
    `helio_frame_api::gbuffer`). A preset panics when a phase that needs an
    opener (shadow casters, depth prepass, G-buffer) has drawers but not
    exactly one opener: no stale pixels, no double clear.
  - Views: a graph is built for one camera (stereo eyes are its layers), so
    a phase's drawer list is per view; `Phase::views` says the shadow-caster
    phase draws for the lights' shadow faces.
  - **Colour chain** (`color::current` / `color::advance`): a post node
    reads the current scene colour when it builds and advances the chain to
    what it writes. Fog composite, TSR, FXAA, lens, flare, transparency and
    the tone map use it; no node names `lens_color`, `fxaa_color`,
    `tsr_color` or `fogged_hdr` any more, and the tone map lost its
    `color_input` parameter (and its pin alternatives). `ReactiveMask` is
    the preset's request for a reactive mask from what draws after
    lighting.
  - **Lighting inputs** (`lighting`): `INDIRECT_DIFFUSE` (GI's resolve;
    `helio_gi_api::GI_IRRADIANCE` is now this key), `INDIRECT_SPECULAR_LAYERS`
    (two rgba16f layers composited over the environment in order; a
    publisher claims the next free one when it builds, so SSR then planar
    keep today's priority), `ENVIRONMENT_PROBES`/`ENVIRONMENT_CUBES`
    (reflection captures and their cube array), `SUN_VISIBILITY` /
    `DIRECTIONAL_VISIBILITY`, `TRANSMISSION` (radiance media add at a
    surface), `sky` (atmosphere frame, sky table and their WGSL, moved from
    `helio-sky-api`, which re-exports them), `LIGHT_LIST`, and
    `VIEW_REMAPS` (per-chain remap matrices plus the G-buffer tag,
    `view_remap_of` in WGSL). `SpecularLayers` says how the preset's
    lighting takes the layers: read (deferred), composited after lighting
    (HLFS) or none (forward).
  - **Host inputs** are declared by the feature that reads them
    (`host_input(name)`, placed at `HOST_INPUTS`): the spine declares
    `material_textures` and `render_environment`, billboards and corona
    their own. Presets add every registered host input.
- **Phases run, drawers register from their crates** (A2): the meshes'
  drawers are features of their own crates (`GBufferDrawers` opens the
  G-buffer, `ShadowDrawers` opens the shadow casters, `ForwardDrawers`
  opens the forward phase and builds only without a G-buffer phase,
  `TransparentDrawers`); portal mask and draws, foliage, VG, voxel chunks
  and decals are G-buffer drawers; corona, the portal editor overlay,
  water and billboards are overlay drawers; GI's probe update, foliage
  placement and portal culling run before the G-buffer phase. Presets name
  only the spine's passes, the lighting model, the anti-aliasing and the
  slots and phases. Features gate themselves: `PortalSettings`,
  `SsrSettings`, `PlanarReflectionSettings` (and the existing
  `FoliageSettings`) are bridged from the renderer config by the presets'
  preparer; `ShadowSettings` (budget, quality) replaced the spine
  settings' shadow fields.
- **Deferred lighting reads lighting inputs only**: no water volumes, no
  caustics textures, no `ssr_trace`/`planar_reflection`, no
  `portal_chain_unmap`, no `helio-sky-api`/`helio-gi-api`. Water publishes
  `transmission` from a new fullscreen pass (`WaterTransmissionPass`, at
  `LightingInputs`, a member only with water) that evaluates the caustics
  field the lighting used to (a fragment pass, so the field's
  screen-space footprint is identical); the portal projection publishes
  `view_remaps`; SSR and planar claim specular layers; under HLFS the SSR
  node runs after lighting with its composite (`Slot::Lighting`), as
  before.
- **Post** (A3): the fog composite moved to `helio-pass-volumetric-fog`
  (`Slot::Media`), post-process volumes to a new feature crate
  `helio-pass-post-volumes` (group `post`: the blend pass, the volume and
  camera rows, the resolver shader split from the tone map's), which
  publishes the post-settings contract `helio_pass_postprocess::POST_SETTINGS`
  (`GpuPostProcessUniforms`, its WGSL in `post_settings.wgsl`). The tone
  map keeps only that contract and the colour chain, and binds
  `PostProcessSettings::default()` when nothing publishes settings (it drew
  nothing before).
- **Preset snapshot** (every preset x default, TSR, voxel+TSR,
  reflections, portals/foliage-off; all passes, dormant included): the
  differences, all intended: GI's probe update runs before the G-buffer
  (it reads no G-buffer; the voxel drawer traces its rays); portal culling
  runs after light binning (no dependency); decals draw before GI's mesh
  rays (they are G-buffer drawers; the mesh rays read no G-buffer); the
  forward preset lost portal culling and its marker (nothing there read
  it); `WaterTransmission` is a new (dormant without water) pass;
  renamed contract resources (`indirect_diffuse`, `indirect_specular_*`,
  `transmission`, `view_remaps`); moved type paths (fog composite,
  volume blend). Performance markers stay unconditional, so with portals
  off the scene-prep marker follows the (empty) portal-cull node.
- Found, not fixed: the deferred lighting loads the half-resolution SSR
  trace at full-resolution pixels (`textureLoad(layer0, pix)`), as it did
  before; HLFS's composite reconstructs it properly. Transparency still
  reads fog's grid (`fog_accum`/`fog_parameters`) by name to fog each
  fragment, and six post passes keep a "the fogged colour may alias
  `pre_aa`" special case: both want a media contract and a chain alias.

As built, A4 (2026-11-02):
- **One generic seam replaces the kind setters**: `Renderer::set_feature<T>`
  (and `feature::<T>()`, `RendererBuilder::with_feature`) sets a feature's
  settings in the renderer config and rebuilds the graph. Kinds own their
  settings types:
  - `set_shadow_quality` and `RendererConfig::{shadow_quality, shadow_budget}`
    -> `helio_pass_shadow_matrix::ShadowSettings { budget, quality }` (host
    settings map into it in `RendererConfig::with_settings`).
  - `set_ambient` / `with_ambient` -> `helio_frame_api::lighting::AmbientLight`
    (the lighting's stand-in sky light; the renderer publishes it in
    `RenderEnvironment` every frame).
  - `set_ies_texture_view` -> `helio_pass_deferred_light::IesProfiles`
    (handed to the lighting node; the renderer no longer publishes
    `ies_textures`).
  - `set_color_grading_lut` -> `helio_pass_postprocess::ColorGradingLut`
    (handed to the tone map node; no `color_grading_lut` resource).
  - `set_material_sampler` -> `helio_mats::MaterialSampler` (the renderer's
    material bindings rebuild their sampler when it changes).
  - `template_registry_mut` / `transparent_template_registry_mut` ->
    `helio_mats::MaterialTemplates` (shared registries; the G-buffer drawer
    now compiles from the opaque one; before, the renderer's registries
    reached no pass, so `radiant_demo`'s custom templates fell back to the
    default class).
  - `set_deferred_light_timing_ms` -> `RenderPass::set_measured_gpu_ms`,
    which `RenderGraph::feed_measured_gpu_times` (and
    `Renderer::feed_measured_gpu_times`) calls for every pass from the
    timing snapshot: any pass whose output shows its own cost gets it, and
    the host learns whether to redraw.
- `helio-core` shed `asset_types.rs` (`PackedVertex`, `MeshUpload`,
  `SectionedMeshUpload`) and `RayInstance` with its flags to the new api
  crate `helio-mesh-api`; `MATERIAL_TEXTURES` to `helio-mats`,
  `SHADOW_MATRICES` to `helio-pass-shadow-matrix`, `BAKED_LIGHTMAP` to
  `helio-bake-types`; `PORTAL_VIEWS` was dropped (the portal crates' own
  `PORTAL_VIEWS` contract already named it). The facade no longer
  re-exports mesh types.
- Left for the kinds' plugins (phase B), with why:
  - `set_ray_tracing_frame`: the frame's TLAS and the rows after it are
    built by the host's acceleration projection (`helio_components`); they
    become the meshes plugin's publication (a derivation or pass that
    builds the TLAS from SceneDB), not a setting.
  - `set_portal_projection_counts`: the portal passes decide membership
    from the live counts on the CPU, the frame they change; reading them
    back from the `portal_projection_counts` rows instead delays joins by
    a readback. It moves with the portal projection into the portals
    plugin, which can publish the counts with its derivation.
  - `RendererConfig::{enable_portals, enable_ssr, enable_planar_reflections,
    enable_environment_reflections}` stay renderer fields, bridged into
    `PortalSettings`, `SsrSettings`, `PlanarReflectionSettings` and the
    spine settings by the presets' preparer, because host settings
    (`helio-settings`) map onto them; they leave with the presets'
    config bridge.

As built, A5 (2026-11-02): one scene derivation per kind, each in its
crate; the hierarchy resolve stays in `helio-scene-derivations`.
- `helio-scene-join` (layer `scene`): what no kind owns: the owner,
  generation, hidden, transform and mesh-bounds row layouts, the shared
  join WGSL and frame-origin helpers, the hierarchy resolve's output keys,
  `JoinOutput`/`signature` helpers, `DerivationGroup` (several derivations
  run as one, each seeing the earlier ones' buffers), and **`RowJoin`**:
  the environment join's engine made kind-agnostic. A kind describes its
  tables (`RowTable`: source key and row size, output key and words,
  packing, slice, gate, layers written over it, a particle pool, a phase
  period) and picks placements from `placement` (`OWNER_AABB`,
  `GATE_HIDDEN`, `SURFACE_HEIGHT`, `SQUARE_LAYER`, `CENTRED`, `OWNER_BOX`,
  `POOL_RANGE`, `SUN`, `WIND`, `MESH_BOUNDS`, `PLANAR_2D`).
- Each kind's crate has `scene_rows::scene_rows(device, RowJoinKeys,
  sources) -> RowJoin` and owns its source row sizes, output keys and
  capacities: fog (global/local media), post-process volumes (volumes,
  camera baselines), the lens (camera lenses), water (volumes with the
  sun, wind and world phase; hitboxes), foliage (types, layers, wind),
  the sky (atmospheres), decals, corona (emitters and the particle pool),
  sprites. The level's wind rows are the foliage api's
  (`WIND_SOURCE_ROW_BYTES`, `GLOBAL_WIND_MARK_WORD`); water reads them by
  key.
- The scene join split in two: the meshes' `SceneJoin` (objects,
  materials, moved list, structure marker) moved to
  `helio-pass-object-batch::scene_join`; the lights' `LightJoin` (the light
  list and the editor's light billboards) is
  `helio-pass-light-cull::scene_rows`. A host installs the lights' join
  before the row joins (water takes its sun from the light list).
- Pulsar groups the kinds' joins (`DerivationGroup`) in `helio_bridge`
  until each kind's plugin installs its own derivation (phase B).
- Not changed: the placements are still WGSL ops of one shader
  (`row_join.wgsl`) the kinds select by bit; the next step is each kind
  bringing its placement as a snippet, so a new kind needs no engine edit.

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
