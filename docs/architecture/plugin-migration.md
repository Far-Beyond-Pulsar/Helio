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

Each plugin is `plugins/vendor/<kind>/` (Pulsar), where every shipped
first-party plugin lives (physics, skm, input, render_bps, asset_viewer),
with: a component crate
(SceneDB component, methods, runtime, file formats), an api crate (its
contracts: keys, row ABIs, WGSL), a feature crate (Helio feature: nodes,
passes, its scene derivation), its editor crate and, when it has any, an
examples crate. Plugins depend on Helio core and on other plugins' api
crates only. During the rewrite each plugin stays in Pulsar's tree; in
phase C it splits into its own repo (`Plugin_<Kind>`, a submodule at the
same path, like physics).

| Plugin | Components | Feature owns | Leaves Helio/Pulsar today's |
|---|---|---|---|
| `pulsar.materials` | material assets, material graph | helio-mats (GPU material ABI, Radiant templates, texture bindings), material texture means | facade template registries, `set_material_sampler`, `MATERIAL_TEXTURES` |
| `pulsar.meshes` | StaticMesh, StaticMeshDraw, SkeletalMesh, LOD, VirtualGeometry | object batch, culling (occlusion, indirect dispatch), mesh drawers (GBuffer, ShadowCasters, Forward, Transparent), skinning, LOD selection, VG (all `helio-pass-vg*`, `helio-vg-*`), ray proxies + TLAS publication, the mesh part of the scene join | gbuffer/shadow/forward/transparent mesh code, `asset_types.rs`, `RayInstance`, `with_virtual_geometry`, authored LOD derivation, VG host in helio_component |
| `pulsar.lights` | Light, ReflectionCapture | light list, light cull, shadow views + atlas (shadow matrix/dirty/atlas; casters come from drawers), the lighting model (deferred light, forward light function), IES, light icons (as billboards) | light part of the scene join, `GpuLight`/`SCENE_LIGHTS` in forward-lit, facade light setters |
| `pulsar.sky` | Atmosphere | atmosphere tables, clouds, sky composite; publishes `sky`, `sun_visibility` | helio-sky-api readers name `sky` contracts only |
| `pulsar.gi` | (none; settings) | probe clipmap, tracers interface; publishes `indirect_diffuse` | — |
| `pulsar.fog` | GlobalFog, LocalFogVolume | volumetric fog and its composite | fog composite out of the spine's postprocess crate |
| `pulsar.post` | PostProcessVolume, CameraPostProcess | volume blend, publishes post settings | PP volume contracts out of the spine |
| `pulsar.lens` | CameraLens | lens, flare (reads `light_list` via lights api) | done (batch 1): `plugins/vendor/lens/` (`lens_component`, `feature/{lens,flare,optics}`, `lens_plugin`, `lens-examples`); the view overscan through `ViewOverscanRegistration` |
| `pulsar.water` | WaterVolume (+ physics hitbox rows via physics api) | water sim and draw, caustics as `transmission` | water reads in deferred light |
| `pulsar.foliage` | Foliage, Wind | placement, drawers | — |
| `pulsar.voxel` | VoxelTerrain, VoxelTerrainLayers | chunk passes, drawers, GI tracer, foliage ground, water terrain | done (layered-rewrite step 6.6): Pulsar `plugins/vendor/voxel/` with `component` (voxel_component), `api` (helio-voxel-data), `world` (helio-voxel-world), `feature` (helio-pass-voxel-planet, a G-buffer drawer since Phase A), `editor` (voxel_plugin), `examples` (helio-voxel-examples: voxel_craft, voxel_flight, water_world); the Helio repo has no voxel crate. Left: the sculpt tool and terrain panel in `ui_level_editor`, waiting on plugin-provided tool state, undo, commands and save hooks |
| `pulsar.decals` | Decal | decal drawer (after GBuffer) | done (batch 1): `plugins/vendor/decals/` |
| `pulsar.particles` | ParticleEmitter | corona | done (batch 1): `plugins/vendor/particles/` (the `corona_emitters` host input is the feature's since Phase A) |
| `pulsar.sprites` | Sprite | sprite cull/batch/simulate, 2D radiance cascades | done (batch 1): `plugins/vendor/sprites/` |
| `pulsar.portals` | Portal (SubLevelDisplay stays in the engine: it places level instances, which rigid frames use too) | portal cull, draws, projection derivation, view remaps | done (batch 1): `plugins/vendor/portals/`; counts as the `PORTAL_COUNTS` payload, pairing and projection as a space producer |
| `pulsar.splines` | Spline | debug-line drawing through the facade's debug draw | done (batch 1): `plugins/vendor/splines/` (component and editor crates; lines through `EditorLinesRegistration`) |
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

Phase A's frame work runs materials -> meshes -> lights; everything else
hangs off the frame contracts. Phase B moves crates out of the Helio repo
in the opposite order (leaves first, see B). Each step deletes the old path in the same
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
  volume blend); after A4 the lighting no longer reads `ies_textures`
  and the tone map no longer reads `color_grading_lut` (both are node
  settings now). Performance markers stay unconditional (stable perf
  phases whatever a level holds), so with portals off the markers of the
  (now empty) portal-cull and portal-overlay nodes remain where the preset
  used to drop the nodes.
- Found, not fixed: the deferred lighting loads the half-resolution SSR
  trace at full-resolution pixels (`textureLoad(layer0, pix)`), as it did
  before; HLFS's composite reconstructs it properly. Transparency still
  reads fog's grid (`fog_accum`/`fog_parameters`) by name to fog each
  fragment, and six post passes keep a "the fogged colour may alias
  `pre_aa`" special case: both want a media contract and a chain alias.

As built, A4 (2026-11-02):
- **One generic seam replaces the kind setters**: `Renderer::set_feature<T>`
  (and `feature::<T>()`, `RendererBuilder::with_feature`) sets a feature's
  settings in the renderer config and rebuilds the graph;
  `set_feature_live<T>` sets one whose readers read it every frame (the
  ambient light, the material sampler) without a rebuild. Kinds own their
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

Phase A verification (2026-11-02): preset snapshot as above; Helio and
Pulsar `cargo check --workspace --all-targets` clean; layer gate 0;
helio-presets GPU tests pass except the known
`foliage_grows_on_the_voxel_ground`; voxel_craft bursts (alternating
base/new, 4 poses) match to base/base noise (pose means within 0.05);
`indoor_cathedral` captures base/new mean |Δ| 0.11 against base/base 0.19.
`indoor_cathedral_water` drifts +0.3 to +0.7 levels evenly over the whole
frame between runs: its candle flicker follows wall-clock time, so each
capture lands at a different flicker phase (not the transmission input).

**B. Plugins, leaves first** (each: Pulsar plugin + its Helio crates move
out of the Helio repo into it; component methods for every editor/game
operation; editor moved onto those methods).

Pulsar depends on Helio and never the reverse, so a crate can leave the
Helio repo only once no crate left in Helio depends on it (normal, build
or dev dependency). Repo moves therefore go consumers-first: a kind moves
after every kind that reads its crates. The test host (examples,
`helio-components`, `helio-scene-derivations`' tests, `helio-presets`'
default feature list and tests, the wasm/web/android demos, helio-core's
dev-dependencies) does not count: each move drops the kind's entries from
it.

Reverse dependencies on `rewrite/p6-frame` (2026-11-02, `cargo metadata`,
test host excluded):

| Crate | Read by (in Helio) |
|---|---|
| `helio-mats` | facade, asset-compat, decal, deferred-light, forward-lit, gbuffer, gi, hlfs (dev), portal-instances, sprite-batch, transparent, vg-cull/dense/raster/shade, water |
| `helio-pass-shadow-matrix` | facade, spine, deferred-light, flare, gbuffer, gi, hlfs, shadow, shadow-cull/dirty, vg-shade, fog |
| `helio-pass-gbuffer` | facade, asset-compat, snapshot, corona, forward-lit, hlfs, indirect-dispatch, object-batch, occlusion-cull, portal-cull/instances, shadow, shadow-cull/dirty, transparent, vg-bvh, vg-cull (dev), water |
| `helio-pass-light-cull` | spine, forward-lit, transparent |
| `helio-pass-deferred-light` | facade, spine, water |
| `helio-pass-object-batch`, `-occlusion-cull`, `-indirect-dispatch` | spine, water (and each other, hlfs/occlusion dev) |
| `helio-mesh-api` | asset-compat, bake, gbuffer, gi, hlfs, shadow |
| `helio-pass-forward-lit` | facade, asset-compat, snapshot, hlfs (dev) |
| `helio-sky-api` | gi, fog, water (the cloud shadow map) |
| `helio-gi-api` | hlfs (only `GI_IRRADIANCE`, which is the frame's `INDIRECT_DIFFUSE`) |
| `helio-foliage-api` | water (the level's wind rows) |
| `helio-pass-foliage-place` | foliage-gbuffer |
| `helio-pass-post-volumes` | fog (dev) |
| `helio-pass-lens` | flare |
| `helio-pass-portal-cull` | portal-instances |
| `helio-pass-sprite-cull` | sprite-batch |
| `helio-pass-postprocess`, `-hiz`, `-tsr` | spine and post passes (spine passes: they stay) |

Order:
1. **Leaf kinds** (nothing left in Helio reads them): splines (no Helio
   crate; their lines go through the facade's debug draw like any
   client's), particles (corona), sprites (sprite cull, batch, simulate,
   2D radiance cascades), decals, lens (lens optics, lens, flare: flare is
   the lens's only reader and moves with it), portals (portal cull,
   portal instances, the portal projection derivation and its facade
   seam).
2. **Media and placement**: fog and post volumes (fog's tests read the
   post volumes' rows: move them together, or fog first), water (reads
   foliage-api's wind rows), foliage (after water).
3. **gi, then sky.** gi-api's one reader, hlfs, switches to the frame's
   `INDIRECT_DIFFUSE` key (the same buffer) so gi-api leaves with gi. The
   sky goes after gi, fog and water, which read `helio-sky-api`'s cloud
   shadow; by then they are Pulsar plugins and read the sky plugin's api
   crate (or the frame's `SUN_VISIBILITY` input, if the cloud shadow
   becomes it).
4. **meshes** (with VG, LOD, skeletal, `helio-mesh-api`, asset-compat).
   First the spine and facade stop depending on mesh crates (the spine
   keeps phases, Hi-Z, TSR, the tone map; the G-buffer layout is
   frame-api's; the facade loses `find_pass_mut` users of object batch and
   G-buffer), then the crates move.
5. **lights** (light cull, shadow matrix/cull/dirty/atlas, deferred light,
   forward lit, hlfs, IES). The spine and facade drop shadow-matrix and
   deferred-light first; by then flare, gi, fog and VG shade already read
   them from Pulsar.
6. **materials** (`helio-mats`), last: about fifteen pass crates and the
   facade read it, all gone from Helio by now.

Physics already is a plugin; it gains water's api in step 2. Voxel is
done: `plugins/vendor/voxel/` (its sculpt tool and terrain panel still
wait in `ui_level_editor`).

**Examples and tests move with their kind**, as voxel's did:
- Examples that exist to show a kind (named after it, or a demo whose
  subject it is) move to the plugin's `examples` crate
  (`plugins/vendor/<kind>/examples`, package `<kind>-examples`, one
  `[[bin]]` each), with a `common.rs` holding the part of Helio's
  `v3_demo_common` they use; they build from Pulsar
  (`cargo build -p <kind>-examples --bin <name>`) and add the plugin's
  feature to their preset (`Preset::with_features`).
- Examples that use a kind in passing (a benchmark spawning one decal, a
  capture tool turning on depth of field) drop that use and say so in
  their docs; they stay in Helio while their own kinds are there.
- Pass crates keep their own unit and GPU tests. Tests elsewhere in Helio
  that exercise the kind (preset graph tests, scene-derivation and join
  tests, shader checks that include its WGSL by path) move into the
  feature crate's `tests/`, over Helio's presets plus the plugin's
  feature, or drop their kind-specific cases when the test is about
  something else.
- Helio's preset snapshot loses exactly the moved kinds' passes and host
  inputs; Pulsar's engine adds each plugin's feature back
  (`RenderFeatureRegistration`).

As built, batch 1 (2026-11-02, branch `rewrite/p7-leaf-plugins` in both
repos): splines, particles, sprites, decals, lens and portals are Pulsar
plugins; the Helio repo has none of their crates.
- **Layout** (Pulsar `plugins/vendor/<kind>/`): `component`
  (`<kind>_component`: the SceneDB components moved from `helio_component`,
  their methods, their derived source rows and every registration),
  `feature` (the pass crates moved from the Helio repo, unchanged names:
  particles `helio-pass-corona`; decals `helio-pass-decal`; sprites
  `feature/{batch,cull,simulate,radiance-cascades-2d}`; lens
  `feature/{lens,flare,optics}`; portals `feature/{cull,instances}`),
  `editor` (`<kind>_plugin`: the plugin's identity, linked by `ui_core`), and
  `examples` (`<kind>-examples`). Splines have no feature crate: their curve
  is drawn through the facade's editor debug lines. No kind needed an api
  crate yet: nothing outside each plugin reads its contracts (the flare reads
  the lens's from inside the plugin), so the contracts stay in the feature
  crates until a second reader appears.
- **Generic engine seams** (Pulsar, `inventory` registrations a plugin's
  component crate submits; the engine names no kind):
  - `engine_backend::scene::SceneRowsRegistration`: a kind's source-row
    columns (registered with the GPU mirror) and its row join (run by the
    environment join after the engine's own kinds). Particles, sprites,
    decals and the lens use it.
  - `engine_backend::scene::SpaceProducerRegistration` / `SpaceProducer`:
    something besides space-defining objects that takes slots in the
    frame's one coordinate-space table, publishes payloads about the frame
    and brings its derivation. `SpaceSync` runs every registered producer;
    the portals' pairing and projection (`PortalSync`, moved from
    `engine_backend::scene::portals`) is one. `SpaceSync` lost
    `portal_input`; `helio_bridge::portal_projection` is gone.
  - `RenderFeatureRegistration` (from the voxel step) for every plugin's
    features and settings; the portals bring their own settings group
    (`helio_pass_portal_cull::settings::PORTALS`, the "portals" toggle that
    left `helio-settings`' renderer group).
  - `ViewOverscanRegistration`: how much wider than its frame a view
    renders (the lens's `camera_lens_view_overscan`; the renderer called it
    by name, with a TODO asking for this hook).
  - `EditorLinesRegistration` / `EditorLines`: world-space line sets the
    editor viewport draws through `debug_set_editor_lines` (the splines'
    `SplineLines`; the renderer no longer names splines).
  - `AnimatedContentRegistration`: content that moves on its own keeps a
    realtime viewport drawing (particle emitters; the renderer queried
    `ParticleEmitterComponent` itself).
  - `helio_component::texture_assets::TextureBindingRegistration`: how a
    component that shows an image takes in a loaded texture asset (sprites,
    decals); `bind_texture`, `collect_texture` and `texture_slot` are public
    for them.
  - View payloads: `HelioEditorMailbox::set_view_payload(key, payload)` /
    `SceneRenderer::set_view_payload` publish editor-only data for one
    viewport under a feature's payload contract key, beside the scene's
    payloads. It replaced the mailbox's `set_portal_debug_texts` /
    `set_portal_debug_geometry` and their `find_pass_mut` calls: the
    portal overlay reads `PORTAL_DEBUG_TEXTS` / `PORTAL_DEBUG_GEOMETRY`.
- **Helio**: the facade lost `RendererConfig::enable_portals` and
  `set_portal_projection_counts`; helio-core's `RenderFrameInputs` lost
  `projection_counts`. The portal passes read their live view and chain
  counts from the payload `helio_pass_portal_cull::PORTAL_COUNTS`, which the
  portals' space producer publishes each frame (or a standalone host, as the
  portal examples' `common::set_projection_frame` does).
  `helio-scene-derivations` lost the portal projection (now
  `helio_pass_portal_cull::scene_rows`), `helio-components` the kinds' row
  components (now in the plugins' `examples/common.rs`), presets their
  features, swap-policy entries and the portal settings bridge.
- **Component methods**: the spline's edit operations (`reverse`,
  `auto_tangents`, `smooth`, `resample`, `insert_at`, `total_length_m`) are
  script methods now; the decal's `fade_by` and the lens's methods already
  were.
- **Examples and tests moved**: corona_demo; indoor_cathedral_decals;
  lens_showcase, lens_flare_demo; portals_demo, infinite_tunnel, portal_cube,
  portal_rooms; sprite_demo, sprite_stress_test, sprite_10m_demo,
  sprite_dig_demo (with Helio's `assets/sprites`). Each builds over Helio's
  deferred preset plus its plugin's features (`common::preset()`). The voxel
  examples add all seven of the leaf plugins' features, as the engine's
  `render_features` does, so their graphs keep p6's passes. Tests:
  `engine_backend`'s camera lens rows, space table and the decal, emitter
  and sprite cases of its environment join test, and `helio_component`'s
  texture asset and device-loss tests, moved to the plugins' component
  crates.
- Left, with why:
  - The spline tool mode, panel and assistant tools, the sprite placement
    helper and the portal diagnostics overlay are still `ui_level_editor`'s
    (as voxel's sculpt tool): they act through the plugins' component crates
    and methods, but moving them needs plugin-provided tool modes and
    viewport overlays.
  - `row_join.wgsl`'s `POOL_RANGE` placement reads the emitter row's words
    at fixed offsets (the particles' layout): the next step is each kind
    bringing its placement as a snippet.
  - Shader hot reload no longer swaps the moved passes by themselves (the
    presets' swap policy cannot vouch for an engine's features): a reload of
    one rebuilds the graph.
  - `level_migrate` and `pulsar_scene` no longer force-link the portal
    component (they cannot depend on a plugin); a migration run decodes
    portal components only if the plugin is linked.
- **Verified** (against p6 builds of Helio `99b02b21` / Pulsar
  `7a1ef8ee1`):
  - Helio's preset snapshot (four configs, every preset) loses exactly the
    moved passes (Corona, SpriteCull, SpriteBatch, Lens, LensFlare,
    DecalApply, PortalCull/Mask/Instance/EditorOverlay), the
    `corona_emitters` input, and the tone map's read of `lens_color`.
  - Pulsar's `pulsar_game` test `plugin_pass_lists` builds the same
    presets with the features the engine adds (`plugin_features`, settings
    through `render_features::configure`) and dumps the same snapshot:
    byte-identical to p6's (3792 lines).
  - voxel_craft (alternating p6/p7 bursts, 4 poses): identical graph dump
    (Lens and LensFlare scheduled); p6-p7 pixel differences within p6-p6
    run noise.
  - Examples: lens_showcase's 81 captures and its flare probe, and
    portal_cube, are bit-identical to p6's. The first lens captures' floor
    settles differently from run to run in both builds; corona_demo
    animates on wall-clock time.
  - The moved crates' unit and GPU tests pass from Pulsar (components,
    editor crates, features, `engine_backend`'s and `helio_component`'s
    integration tests); the layer gate is clean. `engine_backend`'s
    `gpu::tests::readbacks_and_uploads_reuse_their_buffers` (upload pool
    and readback ring, untouched here) failed once.

**C. Helio repo is core only**: helio-core, helio-feature-api,
helio-frame-api, helio-spine, helio-graph-asset, helio-presets, helio,
helio-xr, helio-settings, and a test host (`helio-components` +
examples over what is still in Helio; a plugin's own demos are its
`examples` crate in Pulsar, since Helio never depends on Pulsar). Each
plugin leaves Pulsar's tree for its own repo (`Plugin_<Kind>`), kept at
`plugins/vendor/<kind>/` as a submodule. The layer gate gains a rule: no
crate in the Helio repo may
name a kind's contract outside the test host.

**Done when**: Helio's repo has no kind code outside its test host; each
plugin's editor and game operations are component methods; Pulsar's
`engine_backend` renderer has no kind-specific code; scripting can do
everything the editor does through component methods.
