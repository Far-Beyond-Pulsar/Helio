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
| `pulsar.materials` | material assets, material graph | helio-mats (GPU material ABI, Radiant templates, texture bindings), material texture means | done (batch 6): `plugins/vendor/materials/` (`api` = helio-mats, `feature` = helio-pass-materials: the material texture table, `component` = materials_component, `editor` = materials_plugin); the Helio repo has no material crate. Left: the material assets, graph compiler and cooked textures in `helio_component`, and the engine's material table being the meshes' join (see "As built, batch 6") |
| `pulsar.meshes` | StaticMesh, StaticMeshDraw, SkeletalMesh, LOD, VirtualGeometry | object batch, culling (occlusion, indirect dispatch), mesh drawers (GBuffer, ShadowCasters, Forward, Transparent), skinning, LOD selection, VG (all `helio-pass-vg*`, `helio-vg-*`), ray proxies + TLAS publication, the mesh part of the scene join | crates done (batch 4): `plugins/vendor/meshes/` (`api` = helio-mesh-api, `feature/*`, `vg/*`, `asset-compat`, `host` = helio-mesh-host, `component` = meshes_component: the features' registrations, `editor` = meshes_plugin, `examples` = meshes-examples, `snapshot`, `web-demos`, `android-demos`); the Helio repo has no mesh crate. Left: the SceneDB components, scene join installation, authored LOD selection, VG host and picking in `helio_component`/`engine_backend` (core crates read them by type; see "As built, batch 4") |
| `pulsar.lights` | Light, ReflectionCapture | light list, light cull, shadow views + atlas (shadow matrix/dirty/atlas; casters come from drawers), the lighting model (deferred light, forward light function), IES, light icons (as billboards) | done (batch 5): `plugins/vendor/lights/` (`api` = helio-light-api, `feature/{cull,shadows,deferred,hlfs,billboard}`, `host` = helio-light-host, `component` = lights_component with `LightComponent`/`ReflectionCaptureComponent`, `editor` = lights_plugin, `examples` = lights-examples); the Helio repo has no light crate |
| `pulsar.sky` | Atmosphere | atmosphere tables, clouds, sky composite; publishes `sky`, `sun_visibility`, `sun_transmittance` | done (batch 3): `plugins/vendor/sky/` (`sky_component`, `feature` = helio-pass-sky, `sky_plugin`, `sky-examples`); no api crate: what others read of it is the frame's |
| `pulsar.gi` | (none; settings) | probe clipmap, tracers interface; publishes `indirect_diffuse` | done (batch 3): `plugins/vendor/gi/` (`api` = helio-gi-api, `gi_component`: registrations and settings, `feature` = helio-pass-gi, `gi_plugin`, `gi-examples`) |
| `pulsar.fog` | GlobalFog, LocalFogVolume | volumetric fog and its composite | done (batch 2): `plugins/vendor/fog/` (`fog_component`, `feature` = helio-pass-volumetric-fog, `fog_plugin`, `fog-examples`); publishes the frame's participating media |
| `pulsar.post` | PostProcessVolume, CameraPostProcess | volume blend, publishes post settings | done (batch 2): `plugins/vendor/post/` (`post_component`, `feature` = helio-pass-post-volumes, `post_plugin`, `post-examples`); blends over the host's `POST_BASELINE` |
| `pulsar.lens` | CameraLens | lens, flare (reads `light_list` via lights api) | done (batch 1): `plugins/vendor/lens/` (`lens_component`, `feature/{lens,flare,optics}`, `lens_plugin`, `lens-examples`); the view overscan through `ViewOverscanRegistration` |
| `pulsar.water` | WaterVolume (+ physics hitbox rows via physics api) | water sim and draw, caustics as `transmission` | done (batch 2): `plugins/vendor/water/` (`water_component`, `api` = helio-water-api, `feature` = helio-pass-water-sim, `water_plugin`, `water-examples`) |
| `pulsar.foliage` | Foliage, Wind | placement, drawers | done (batch 2): `plugins/vendor/foliage/` (`foliage_component`, `api` = helio-foliage-api, `feature/{place,gbuffer}`, `foliage_plugin`, `foliage-examples`) |
| `pulsar.voxel` | VoxelTerrain, VoxelTerrainLayers | chunk passes, drawers, GI tracer, foliage ground, water terrain | done (layered-rewrite step 6.6): Pulsar `plugins/vendor/voxel/` with `component` (voxel_component), `api` (helio-voxel-data), `world` (helio-voxel-world), `feature` (helio-pass-voxel-planet, a G-buffer drawer since Phase A), `editor` (voxel_plugin), `examples` (helio-voxel-examples: voxel_craft, voxel_flight, water_world); the Helio repo has no voxel crate. Left: the sculpt tool and terrain panel in `ui_level_editor`, waiting on plugin-provided tool state, undo, commands and save hooks |
| `pulsar.decals` | Decal | decal drawer (after GBuffer) | done (batch 1): `plugins/vendor/decals/` |
| `pulsar.particles` | ParticleEmitter | corona | done (batch 1): `plugins/vendor/particles/` (the `corona_emitters` host input is the feature's since Phase A) |
| `pulsar.sprites` | Sprite | sprite cull/batch/simulate, 2D radiance cascades | done (batch 1): `plugins/vendor/sprites/` |
| `pulsar.portals` | Portal (SubLevelDisplay stays in the engine: it places level instances, which rigid frames use too) | portal cull, draws, projection derivation, view remaps | done (batch 1): `plugins/vendor/portals/`; counts as the `PORTAL_COUNTS` payload, pairing and projection as a space producer |
| `pulsar.splines` | Spline | debug-line drawing through the facade's debug draw | done (batch 1): `plugins/vendor/splines/` (component and editor crates; lines through `EditorLinesRegistration`) |
| `pulsar.physics` | (exists: physics_plugin + pulsar_physics) | water hitbox rows through water's api | done (batch 2): `pulsar_physics` derives them under `helio_water_api::WATER_HITBOX_SOURCES_BUFFER` |

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
2. **Media and placement** (done, batch 2): fog and post volumes (fog's
   tests read the post volumes' rows: move them together, or fog first),
   water (reads foliage-api's wind rows), foliage (after water).
3. **gi, then sky** (done, batch 3). gi-api's one reader, hlfs, switches to the frame's
   `INDIRECT_DIFFUSE` key (the same buffer) so gi-api leaves with gi. The
   sky goes after gi, fog and water, which read `helio-sky-api`'s cloud
   shadow; by then they are Pulsar plugins and read the sky plugin's api
   crate (or the frame's `SUN_VISIBILITY` input, if the cloud shadow
   becomes it). As built: the frame's world-space `sun_transmittance`
   (see "As built, batch 3").
4. **meshes** (with VG, LOD, skeletal, `helio-mesh-api`, asset-compat).
   First the spine and facade stop depending on mesh crates (the spine
   keeps phases, Hi-Z, TSR, the tone map; the G-buffer layout is
   frame-api's; the facade loses `find_pass_mut` users of object batch and
   G-buffer), then the crates move. Crates done (batch 4; see "As built,
   batch 4" for the components still in the engine).
5. **lights** (light cull, shadow matrix/cull/dirty/atlas, deferred light,
   forward lit, hlfs, IES). The spine and facade drop shadow-matrix and
   deferred-light first; by then flare, gi, fog and VG shade already read
   them from Pulsar. Done (batch 5; see "As built, batch 5": the components
   moved too).
6. **materials** (`helio-mats`), last: about fifteen pass crates and the
   facade read it, all gone from Helio by now. Done (batch 6; see "As
   built, batch 6").

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

As built, batch 2 (2026-11-03, branch `rewrite/p8-media-plugins` in both
repos): fog, post-process volumes, water and foliage are Pulsar plugins; the
Helio repo has none of their crates (helio-pass-volumetric-fog,
helio-pass-post-volumes, helio-pass-water-sim, helio-water-api,
helio-foliage-api, helio-pass-foliage-place, helio-pass-foliage-gbuffer).
- **Layout** as batch 1: `plugins/vendor/{fog,post,water,foliage}/` with
  `component` (`<kind>_component`), `feature` (the moved crates, unchanged
  names; foliage `feature/{place,gbuffer}`), `editor` (`<kind>_plugin`) and
  `examples` (`<kind>-examples`). Water and foliage keep their api crates
  (`water/api` = helio-water-api, `foliage/api` = helio-foliage-api): other
  plugins read them (voxel reads both, water reads foliage's wind rows,
  physics reads water's hitbox rows). Fog's component crate uses the post
  plugin's `FogMode` (the post settings' fog block and a medium share it);
  the voxel component spawns and syncs the fog, post and foliage plugins'
  components (its presets, the planet haze).
- **Generic frame seams** (Helio, `helio-frame-api`; no Helio crate names
  fog any more):
  - `color::advance_aliasing` / `color::alias`: a node may publish an earlier
    image's view as its output (the fog composite when no fog is possible);
    readers of the current colour read the alias too so the pool keeps it.
    FXAA, TSR, the tone map, transparency, and the lens and flare (Pulsar)
    replaced their `"fogged_hdr"` checks with it.
  - `media::publish` / `media::current`: a media feature publishes its
    integrated medium by its own resource names (grid: rgb in-scatter, a
    transmittance, log slices from 0.5 m to the far distance in the
    parameters' second vec4's `y`); transparency composites each fragment
    through whatever was published, a stand-in otherwise. Resource names
    did not change (`fog_accum`, `fog_parameters`, `fogged_hdr` are the fog
    plugin's), which is why the pass lists stay byte-identical.
  - `lighting::DIRECTIONAL_LIGHT`: the lights' join publishes one row (the
    direction toward its directional light, `w` 1; the first enabled
    directional light with intensity, as the old scan picked); the row
    join's `SUN` placement copies it (`RowJoinKeys::directional_light`
    replaced `lights`). Water's sun is that contract, not a light-row scan.
  - `helio_pass_postprocess::POST_BASELINE` (a payload): the host's post
    settings, the tone map's stand-in when nothing publishes `POST_SETTINGS`
    and the base the post-volume blend uses as its defaults. It replaced the
    editor's `find_pass_mut::<PostProcessVolumeBlendPass>().set_defaults`
    (Pulsar publishes it as a view payload) and Helio examples' camera
    post-process rows (`color_grading_demo`, `hdr_demo`, the wasm demos).
  - The fog pass skips orthographic views itself (`proj[15] == 1`), which
    replaced the editor renderer's `apply_view_fog`.
- **Helio**: presets' default features, swap policy and
  `config_with_host_settings` drop the kinds; foliage's host settings left
  `helio-settings`' renderer group for the feature's own group
  (`helio_pass_foliage_place::FOLIAGE`, keys `foliage.enabled`, `quality`,
  `range`, `tufts_per_m2`, `blades_per_tuft`; projects' old
  `rendering.foliage*` values fall back to the defaults, as the portals
  toggle did in batch 1); `helio-components` drops the kinds' row
  components: each feature crate carries them as `host_rows` behind a
  `host-rows` cargo feature (foliage's in its api crate, where the rows are
  defined), which only standalone hosts and tests enable; helio-core's
  shader test and helio-scene-derivations' water/foliage join test moved.
- **Pulsar engine seams** (inventory registrations; the engine names none
  of these kinds):
  - `level_rules::LevelSingletonRegistration`: a plugin's component a level
    holds at most one of (the foliage plugin's global wind); the rules
    (add, copy, enable, load repair) treat it as the sky.
  - `AnimatedContentRegistration` for water and for foliage in a wind (one
    registration per component the answer reads).
  - `HelioRenderer::debug_pass::<T>()` (read-only diagnostics) replaced
    `debug_water_sim_texture`.
  - `environment_join` keeps the atmospheres and runs every plugin's join;
    `ensure_gpu_mirror` still registers physics' water hitbox columns
    (physics is the engine's), whose layout and buffer are water's api's.
- **Component methods**: these kinds have no editor operation besides
  property edits, which go through their reflected properties' accessors
  (the same setters scripts call); the World Settings fog, post-process,
  water and wind panels and the terrain panel's foliage stay in
  `ui_level_editor` and act through the plugins' component crates.
- **Examples and tests moved**: volumetric_fog_demo (fog);
  indoor_cathedral_pp_volumes (post); indoor_cathedral_water, oil_rig_night
  (water); foliage_demo (foliage). Helio examples that used the kinds in
  passing dropped the use (the cathedral incense fog, resolution_bench's
  `--global-fog`/`--pp-fog`/`--water`, the VR gallery's water bay and fog
  volume, vg_stress's haze and temple volume, graph_cost's `content`
  scene); `cathedral_volumetric_fog_demo` is `monumental_arch_hlfs` now.
  engine_backend's environment join test kept its atmosphere case; its fog
  and post, water, and foliage cases moved to the plugins' component crates.
- Left, with why:
  - `row_join.wgsl` still holds water's placements (`SUN`, `WIND`, the world
    phase at water's row offsets) beside the particles' `POOL_RANGE`: the
    step after this is each kind bringing its placement as a snippet.
  - The post settings contract (`GpuPostProcessUniforms`) keeps its fog
    block, which only the fog plugin reads: splitting it changes the ABI
    and every post-volume row, and belongs with the post plugin's next step.
  - `FocusedTimingGroup::VolumetricFog` (helio-core's diagnostic timing
    view) still classifies `VolumetricFogPass` by name, as it does HLFS and
    SSR; it adds no work and no API to act on fog.
  - `helio::renderer::settings` tests `default_settings_are_the_default_config`
    and `every_choice_maps` fail on p7 too: `with_settings` inserts the
    `ShadowSettings` feature value (Phase A4).
- **Verified** (against p7 builds of Helio `06198ade` / Pulsar `b0d82a4f2`):
  - Pulsar's `plugin_pass_lists` (now also asserting VolumetricFogPass,
    FogComposite, PostProcessVolumeBlendPass, WaterSim, WaterTransmission,
    FoliagePlace and FoliageGBuffer in the engine's deferred preset) dumps a
    snapshot byte-identical to p7's (3792 lines).
  - voxel_craft (alternating p7/p8 runs, 4 poses x 16-frame bursts, twice):
    the `[RenderGraph]` dump is identical (30 scheduled, 23 dormant, same
    passes, reads and writes; only the build time differs); settled-frame
    differences p7-p8 (mean |d| 0.04-0.65 levels, pose means within 0.17)
    sit within p7-p7 and p8-p8 noise (0.07-0.69, within 0.15); flicker
    statistics are equal.
  - volumetric_fog_demo `--probe` (fog with the post plugin) is bit-identical
    to p7. indoor_cathedral (Helio): p7-p8 mean |d| 0.04-0.24 against p7-p7
    0.23. indoor_cathedral_water: p7-p8 0.16-0.39 against p7-p7 1.02 (its
    candles flicker on wall-clock time). water_world shoreline: frames
    differ by the waves' wall-clock phase in every pair (p7-p7 4.7, p8-p8
    6.9, p7-p8 5.7-8.4) with equal means (88.6-89.0).
  - Layer gate 0; `cargo check --workspace --all-targets` clean in both
    repos.
  - Tests from Pulsar pass: the plugins' component and editor crates (their
    row, registration and join tests, the joins' GPU tests moved from
    engine_backend), the moved feature crates' unit and GPU tests,
    engine_backend's lib, environment join and bloom toggle tests,
    helio_component, pulsar_physics, the lens and flare, and the voxel
    feature's presets and clouds. Fog's GPU tests needed the adapter's limits
    (their inject stage binds nine storage buffers; they failed the same way
    in the Helio repo). Not passing, each unchanged by this batch:
    water's `gpu_caustics` (it builds its own pipeline layout without the
    slice uniform the caustics shader binds since before p7), the voxel
    `foliage_grows_on_the_voxel_ground` (known since Phase A; it hung once
    here) and the voxel `chunk_pass_builds_settles_and_resizes_in_the_deferred_graph`
    ("a settled still view asked for frame 0"; it fails the same way on
    p7). Run in parallel, voxel_pass_graph's GPU-heavy cases also lost the
    device; single-threaded they pass.

As built, batch 3 (2026-11-03, branch `rewrite/p9-gi-sky` in both repos):
GI and the sky are Pulsar plugins; the Helio repo has none of their crates
(helio-pass-gi, helio-gi-api, helio-pass-sky, helio-sky-api).
- **Layout** as batches 1 and 2: `plugins/vendor/gi/` with `api` (helio-gi-api:
  the probe clipmap's contract, which the voxel terrain traces), `component`
  (`gi_component`: GI has no SceneDB component, so it holds the feature's and
  the settings group's registrations), `feature` (helio-pass-gi), `editor`
  (`gi_plugin`) and `examples` (`gi-examples`: `indoor_cathedral_gi`);
  `plugins/vendor/sky/` with `component` (`sky_component`: the
  `AtmosphereComponent` moved from `helio_component`, its source row and every
  registration), `feature` (helio-pass-sky, with the atmosphere host row behind
  `host-rows`), `editor` (`sky_plugin`) and `examples` (`sky-examples`:
  `render_v2_sky`). The sky has no api crate: nothing outside it reads its
  crates.
- **The clouds' shadow is the frame's sun transmittance**
  (`helio_frame_api::lighting::sun_transmittance`): a world-space Beer shadow
  map along the sun round the camera plus its frame, the WGSL that reads it
  (`sun_transmittance_at(p)`) and a stand-in. It is published by producer name
  the way batch 2's media are (`publish` / `current`): the sky's atmosphere
  node publishes its `cloud_shadow_map` / `cloud_shadow_frame`, and fog's
  froxels and GI's probe update and mesh rays bind whatever was published, or
  the stand-in (fully lit). Why not `SUN_VISIBILITY` /
  `DIRECTIONAL_VISIBILITY`: those are per pixel of the G-buffer, while a
  medium's froxels and the probes' ray hits are points off the surface and
  off the screen; the clouds fold the map into `SUN_VISIBILITY` for surfaces,
  as before. Why not the sky plugin's api crate: every reader would then
  depend on the sky, while the map's form (a Beer shadow map, as Hillaire's
  and Frostbite's clouds and Unreal's cloud shadow on the atmosphere light
  use) is any medium's above the scene (a weather or smoke plugin publishes
  the same and fog and GI shade under it unchanged). With it gone,
  helio-sky-api had nothing left (its sky inputs were frame-api's since
  Phase A): water reads `helio_frame_api::lighting::sky`.
- **Other frame seams**:
  - `lighting::InputsOrder` (a preset setting): the forward preset named the
    clouds' trace node by kind after its draw phase. It now places
    `Slot::LightingInputs` there and says the lighting already ran
    (`AfterLighting`), so publishers only the lighting reads (GI's resolve,
    water's transmission) build nothing, and the clouds' trace (whose clouds
    the composite draws) still does.
  - `helio_frame_api::static_names`, from the transparency pass: the read
    lists of passes whose reads are what a publisher named.
  - HLFS reads the frame's `INDIRECT_DIFFUSE` (gi-api's `GI_IRRADIANCE` was
    the same key).
- **Helio**: presets' default features and swap policy drop GI and the sky;
  `config_with_host_settings` (it bridged only GI's probe spacing) is
  deleted, and the spacing left `helio-settings`' renderer group for GI's own
  group (`helio_pass_gi::GI`, key `gi.probe_spacing`; projects' old
  `rendering.gi_probe_spacing` falls back to the default, as foliage's did in
  batch 2); `helio-components` lost `AtmosphereComponent`; helio-core's WGSL
  validation and the presets' tests lost their GI and sky parts (the GI graph
  tests `mesh_gi_graph` and `vg_ray_proxies` moved to the GI feature).
- **Pulsar engine**: `environment_join` runs only registered joins (the
  atmospheres' is the sky component's `SceneRowsRegistration`) and
  `ensure_gpu_mirror` registers no atmosphere column; `level_rules` lost the
  sky special case: the sky is a `LevelSingletonRegistration` like the
  global wind. `project_settings::renderer_config` is `with_settings`; GI's
  settings come in through its `RenderFeatureRegistration::configure`.
- **Editor**: the World Settings sky section and the environment section
  create the sky with the component crate's default value and edit it through
  its reflected properties (`SetComponentProperty`), as the wind; they and
  the scene-edit helpers read the level's sky through
  `sky_component::level_atmosphere`. GI has no editor operation: its settings
  are a project settings group.
- **Examples**: Helio's examples that spawned a sky or relied on GI say they
  have neither (`v3_demo_common::spawn_sky` is gone; `resolution_bench`'s
  `sky` scene is `open`); `render_v2_sky` moved to the sky's examples, and the
  GI plugin's `indoor_cathedral_gi` is Helio's `indoor_cathedral` over the
  lighting plugins. The other plugins' examples and the voxel examples and
  tests add the sky's and GI's features, as the engine does.
- **Verified against p8**: Pulsar's `plugin_pass_lists` snapshot is
  byte-identical (3792 lines; the test now also asserts the sky's three
  passes and, where mesh rays run, GI's three); Helio's preset snapshot loses
  exactly the moved passes (324 lines: Atmosphere, CloudTrace and
  AtmosphereComposite in 15 graphs, GI, GIMeshRays and GIResolve in 12) and
  gains nothing. voxel_craft (p8 exe vs p9, alternating, 8-frame bursts at 4
  poses): the render-graph dump is identical (29 passes), mean |Δ| p8-p9
  0.07-0.49 against p8-p8 0.05-0.36 and p9-p9 0.12-0.43, run times 29/29/26/24 s.
  The GI cathedral (Helio p8 `indoor_cathedral` vs `indoor_cathedral_gi`):
  graph dumps identical, mean |Δ| 0.03-0.23 across the four p8-p9 pairs
  against 0.21 (p8-p8) and 0.20 (p9-p9). The sky demo has no capture hook;
  the voxel `clouds` GPU tests and the sky feature's tests cover it.
- Left, with why:
  - `scene_inventory`'s ledger still lists the batch 1-3 pass crates as
    Helio pass crates (its scan reads only Helio's `passes/` and voxel, so
    each is "not a pass crate"); its ledger and architecture tests were
    already failing on stale rows before batch 1 (unlisted buffers, sites,
    renderer queries). The atmosphere's class rows point at the sky plugin
    now. Bringing the ledger up to date is its own task.
  - Failing before this batch and unchanged by it: helio-core's
    `wgsl_validation` (it reads `scene_join_common.wgsl` from
    `helio-scene-derivations`; the file is in `helio-scene-join`), so it
    stops before most shaders (the moved WGSL is checked by the sky's
    `shaders` test and the GI and fog GPU tests); helio-settings'
    `host_values_are_clamped_parsed_or_dropped` (a number for a toggle);
    water's `gpu_caustics`; voxel's `foliage_grows_on_the_voxel_ground` and
    `chunk_pass_builds_settles_and_resizes_in_the_deferred_graph`;
    `ui_level_editor`'s two voxel inspector tests (its lib tests do not link
    the voxel editor that registers the terrain stack's editor). GI's
    `vg_ray_proxies` failed once in six runs (the VG wall let rays through;
    the other five passed): intermittent, as the proxies build in the background.

As built, batch 4 (2026-11-04, branch `rewrite/p10-meshes` in both repos):
the meshes' crates are the `pulsar.meshes` plugin's; the Helio repo has no
mesh crate (helio-mesh-api, helio-pass-gbuffer, -object-batch,
-occlusion-cull, -indirect-dispatch, -forward-lit, -transparent, -shadow,
-shadow-dirty, every `helio-pass-vg*`, helio-vg-format/-build/-stream,
helio-asset-compat, helio-snapshot, the web and Android demos).
- **Layout** (Pulsar `plugins/vendor/meshes/`): `api` (helio-mesh-api: the
  meshes' whole public contract, below), `feature/*` (the passes, unchanged names; forward-lit is
  `feature/forward`), `vg/{format,build,stream}`, `asset-compat`, `host`
  (helio-mesh-host, below), `component` (meshes_component: registers the
  features, `RenderFeatureRegistration`), `editor` (meshes_plugin),
  `examples` (meshes-examples: Helio's mesh scenes, with `test.fbx` and
  `models/`), `snapshot` (the asset thumbnailer `ui_common` uses),
  `web-demos` (with the `web` builder that was Helio's root bin; Helio's
  Pages workflow that ran it is gone) and `android-demos`.
- **Frame seams** (Helio), each the narrowest that lets the crates leave:
  - `Slot::Visibility`: culling around the spine's depth pyramid (Hi-Z at
    order 0, a kind's argument setup before it, its tests after). The object
    batch is the meshes' (`ObjectBatchFeature`, first in `Slot::Prepare`), as
    are indirect dispatch and occlusion culling (`MeshCulling`), so the
    spine and the presets name no mesh pass. How others do it: Unreal and
    Bevy cull inside the renderer, which knows every primitive type; here the
    pyramid is the frame's and each kind culls its own content against it.
  - **Shadow views vs casters**: the lights keep the views
    (`ShadowMatrixPass`) and share their per-caster dirty flags
    (`helio_pass_shadow_matrix::LightDirtyFlags`); what casts decides what
    of its own content made faces stale and draws them (the meshes'
    `ShadowDirtyPass` before the shadow-caster phase, `ShadowPass` opening
    it). As in Unreal's virtual shadow maps: primitives invalidate the
    light's cached pages, the light owns them. `helio-pass-shadow-cull` was
    never constructed: deleted.
  - **The frame's ray-traced scene is a payload**
    (`helio_frame_api::rays::SCENE_RAYS`: the TLAS and named buffers, taken
    for one frame): `Renderer::set_ray_tracing_frame`, `RayFrame` and
    `FrameAcceleration` are gone. The renderer hands the TLAS to the frame
    (`RenderEnvironment::tlas`) and each buffer as a graph resource under its
    name, so a second kind of traced content joins the same TLAS without a
    setter. `RAY_TRANSMISSION` is the frame's; the instance rows' and proxy
    pool's keys are helio-mesh-api's. The meshes' projection publishes it
    (`SceneDbRayTracing::publish`).
  - **The G-buffer is the frame's**: when no kind opens its phase, the
    spine clears it (`helio_spine::GBufferClearPass`, `gbuffer_clear`), as
    Bevy's opaque pass node clears with no phase items. Without it a Helio
    preset with no meshes read an unwritten G-buffer; with it an engine
    without the meshes plugin still lights a defined G-buffer, and a voxel
    drawer has one to load.
  - The light list's row ABI (`GpuLight`, `LightType`, `LightRow`,
    `SCENE_LIGHTS`) moved from forward-lit to `helio-pass-light-cull` (the
    lights), the material row (`MaterialRow`, `MATERIALS`) from the G-buffer
    crate to `helio-mats`, and the default surface template (`gbuffer.wgsl`,
    every partial template's base) to `helio_mats::templates::DEFAULT_PBR`
    with the other template sources: the materials' template registry
    already used it as class 0.
  - `helio-bake` takes `BakeMesh` only (`bake_mesh_id` keeps its id
    encoding); converting a mesh is the mesh host's.
  - `helio-wasm` knows no kind: an app registers its columns and names its
    kinds' features (`HelioWasmApp::register_columns`, `::features`).
- **Helio's test host**: `helio-components` keeps the kinds still in Helio
  (lights, materials, billboards, reflection captures); Helio's examples keep
  `simple_graph` and `debug_shapes` (a custom graph, the debug draw). Every
  example that is a scene of meshes is the plugin's: Helio cannot depend on
  Pulsar, and a mesh drawer of its own would be a second mesh renderer.
  Standalone hosts take the meshes' rows, projection, feature list and
  Helio's presets over it from `helio-mesh-host` (`presets::deferred`, ...);
  its row components are helio-mesh-api's `host_rows` (`host-rows`
  feature), as batch 2's kinds did.
- **One owner per mesh concept** (the rule since 2026-11-03: a concept
  other kinds use has one owner; they read it through the owner's api crate
  or a frame contract, never a private copy, and add only their own
  addressing). helio-mesh-api is the meshes' whole public contract: the
  vertex layout and mesh uploads; the rows their components author (`rows`:
  objects, render groups, sublevels, the mesh pools' keys) and, with
  `host-rows`, those rows as host components; the batch's instance and draw
  rows (`draw`: `GpuInstanceData`, `GpuDrawCall`, the indirect args, the
  `INSTANCE_FLAG_*` bits, coordinate spaces); the frame data their passes
  publish (the object batch's, the culled batch's, draw segments, draw
  slots, the multi-draw they are issued with); the ray instance rows and
  the proxy pool; and their WGSL as snippets, `RAY_SURFACE_SNIPPET`
  (`//!use mesh_ray_surface`: the ray instance row and flags, the mesh and
  proxy pools' vertex decode, `ray_triangle`, `ray_triangle_outside`) and
  `DRAW_ROWS_SNIPPET` (`//!use mesh_draw_rows`).
  - Copies deleted: GI's mesh tracer declared its own `RayInstance`, flags
    and vertex and proxy decode and looked the mesh pools up by name (now
    `mesh_ray_surface` and `MESH_VERTICES`/`MESH_INDICES`, the same math);
    portal culling and the portal G-buffer program declared their own
    instance row (culling's in a column layout), draw call, indirect args
    and flag bit; the G-buffer crate kept a copy of helio-core's
    `CoordinateSpacesFrameData`. Portals and particles read the draw slots,
    segments and frame data from the G-buffer and object-batch crates; the
    other kinds' component tests read helio-core's `ENTITY_GENERATIONS_KEY`
    through the object batch; every other kind's example, test and feature
    manifest lost its G-buffer and object-batch dependencies.
  - How others do it: Unreal's primitive and instance data are one
    renderer-owned definition every pass includes (`SceneData.ush`); Bevy's
    `MeshUniform` is bevy_pbr's and other crates `#import` its WGSL. Here
    the owner is the meshes plugin and the import is its snippet, so a
    change to a row lands in one file and every reader compiles against it.
  - Copies that stay, with what they wait on:
    - Portals: culling re-culls the meshes' draw groups per portal chain,
      and the portal G-buffer program (`gbuffer_portal.wgsl`, with the
      G-buffer crate's `create_material_bgl`) is a second G-buffer drawer
      for mesh content. Waits on the meshes' drawers drawing extra views (a
      view remap, the chain's composed transform, with a clip mask): portals
      then publish views and the meshes cull and draw them.
    - Water's reflection capture builds its own graph of the meshes' passes
      (object batch, indirect dispatch, occlusion culling, G-buffer,
      forward). Waits on captures as views of the frame's phases.
    - Helio core: the row join's `MESH_BOUNDS` placement (water hitboxes
      from mesh instances' bounding spheres, `helio-scene-join`) reads a
      mesh row. Waits on an owner-bounds contract the meshes publish (an
      owner's box), which water and physics read.
    - The materials' templates (`DEFAULT_PBR`, `transparent_base.wgsl`)
      declare the instance row and the vertex input: the materials batch
      (the surface function apart from the drawer that hosts it), with the
      portal program's `GpuMaterial` copy. GI's `SceneLight` mirrors
      `GpuLight`: the lights batch.
    - The meshes' own shaders (object batch, culling, the drawers) each
      declare the rows; they are the owner's, and adopt `mesh_draw_rows`
      as they change.
- **Presets**: the default features name no kind; the swap policy keeps what
  Helio vouches for (Hi-Z and the shadow matrices now rebuild the graph on a
  shader edit: their sampler and dirty flags are shared with an object
  kind's passes). `with_virtual_geometry` moved to `helio-pass-vg`.
- **Tests moved with the meshes**: the scene join's (object batch), the VG
  graph's (vg), the limited-native shadow budget (shadow), HLFS's two tests
  of the ray-tracing projection (`helio-mesh-host`, over HLFS's fixture read
  by path), and Helio's example GPU tests (meshes-examples). Helio's preset
  tests keep their frame parts (phases, shadow residency, HLFS's format).
- **Fixed first** (own commits): helio-core's `wgsl_validation` composed the
  joins from a path that had moved and aborted before most shaders; it now
  validates all 60 standalone shaders. GI's `vg_ray_proxies` rendered frames
  while its proxy built, so a slow build left light behind the wall; it now
  prepares without rendering until the proxy is in.
- **Verified**:
  - Helio's preset snapshot (every preset, three configs): after the seams
    commit byte-identical to p9 (1344 lines); after the move the retained
    passes and the host inputs are byte-identical, the graphs lose exactly
    ObjectBatch, ShadowDirty, Shadow, IndirectDispatch, OcclusionCull,
    GBuffer or ForwardLit, TransparentPass (and the perf marker each mesh
    drawer node carried), and the twelve G-buffer graphs gain GBufferClear
    (912 lines).
  - Pulsar's `plugin_pass_lists` (now also asserting the meshes' passes) is
    byte-identical to p9's (3792 lines).
  - voxel_craft (the p9 baseline exe vs p10, alternating, 8-frame bursts at
    4 poses, idle machine): identical render-graph dump; mean |Δ| p9-p10
    0.09-0.49 against p9-p9 0.05-0.44 and p10-p10 0.09-0.53, pose means
    within 0.11. (A first round ran the base during a build: its clouds,
    which follow frame time, had moved on.)
  - The GI cathedral (p9's `indoor_cathedral_gi` exe vs p10, frame 240):
    identical graph dump; mean |Δ| p9-p10 0.016-0.021 against p9-p9 0.020
    and p10-p10 0.015 (an earlier pair took keyboard input and was dropped).
  - Mesh-heavy: `move_benchmark` (16k objects, 9.2 M triangles, raster and
    ray-traced, idle / move one / editor resync; p9 built once from
    ca0a1f47): identical graph dumps. GPU time is bimodal from run to run in
    both builds (about +3.5 ms in the slow mode, clock state); in the fast
    mode p9 7.23 / 8.26 / 8.32 ms against p10 7.19-7.31 / 8.27-8.29 /
    8.34-8.36 ms. The headless `test.fbx` snapshot has identical graphs and
    renders black in both builds (one frame of a GPU-driven pipeline that
    needs a frame of readbacks: not this batch's).
  - `cargo check --workspace --all-targets` clean in both repos; layer gate
    0. Helio's remaining tests pass (216; the hardware ray-query ones are
    ignored as before).
  - Tests from Pulsar: every meshes crate (api, features, vg, asset-compat,
    host, snapshot, component, editor; meshes-examples' GPU tests), GI,
    voxel, the other plugins' component and feature crates, pulsar_game's
    lib and `plugin_pass_lists` / `runtime_render_parity`, engine_backend's
    lib and its scene-join, bloom, material-graph, movement and group-motion
    tests: 951 pass. Not passing, each known from p9 and not touched here:
    voxel `foliage_grows_on_the_voxel_ground` and
    `chunk_pass_builds_settles_and_resizes_in_the_deferred_graph`, water's
    caustics projection test. Voxel's `clouds_cost` (the sky's cloud passes
    under 4.0 ms at 1080p, the same graph as p9) measured 4.18 and 4.65 ms
    and passed on a third run: GPU time at its budget. The `chunks` suite
    was not run (it is the release suite). GI's `vg_ray_proxies` passed 6 of
    6 runs.
- Left, with why:
  - The SceneDB components (`StaticMeshComponent`, `StaticMeshDraw`,
    `SkeletalMeshComponent`, `LODComponent`, `VirtualGeometryComponent`)
    stay in `helio_component`, with the scene join's installation
    (`helio_bridge::scene_join`), the authored LOD selection, the VG host
    (`scene_renderer`'s `VirtualGeometryHost`) and mesh picking
    (`interaction`) in `engine_backend`. Core crates read them by type and
    cannot depend on a plugin: `asset_import` (rigged imports make skeletal
    meshes), `pulsar_package` (cooks mesh assets), `pulsar_class` (prefab and
    class records), `pulsar_world_registry`, `level_migrate`, `pulsar_scene`
    (the loader), besides 62 files across the engine, the editor, physics
    and `skm_editor`. The seams that unblock it, each a registration the
    plugin submits: asset importers and cook steps by asset type; class and
    prefab records as reflected component records; an object-join
    registration the engine runs in its scene-join group ahead of the LOD
    selection (the mesh join must stay before it); a render-host part
    (per-renderer state updated each frame that adds settings to the config:
    the VG host); a pick registration (a ray against an entity's content).
  - The materials' templates are whole G-buffer programs over the meshes'
    vertex layout (`DEFAULT_PBR`); splitting the surface function from the
    drawer that hosts it is the materials' step.
  - The copies of mesh concepts other kinds keep: see "One owner per mesh
    concept" above.
  - Helio's Pages workflow is gone with the web demos; the plugin has no
    publishing workflow yet.

As built, batch 5 (2026-11-04, branch `rewrite/p11-lights` in both repos):
the lights are the `pulsar.lights` plugin's; the Helio repo has no light
crate (helio-pass-light-cull, -shadow-matrix, -deferred-light, -hlfs,
-billboard and the light icon `spotlight.png` moved).
- **Layout** (Pulsar `plugins/vendor/lights/`): `api` (helio-light-api, the
  lights' whole public contract, below), `feature/{cull, shadows, deferred,
  hlfs, billboard}` (the passes, unchanged crate names), `host`
  (helio-light-host: the features and the row components for standalone
  hosts), `component` (lights_component: `LightComponent` and
  `ReflectionCaptureComponent` moved from `helio_component`, the light
  join's installation, the one-directional-light rule and every
  registration), `editor` (lights_plugin) and `examples` (lights-examples:
  `ies_demo`, `light_benchmark`, `billboard_demo`).
- **Frame seams** (Helio), each the narrowest that lets the crates leave:
  - **The lighting model is a kind's**: `helio_frame_api::lighting::LightingModel`
    (a preset setting: `Deferred`, `LightField`, `Forward`) names the
    extension point a model's nodes join (`lighting.model.deferred`,
    `lighting.model.light_field`; forward draws lit in its phase). The
    deferred preset expands its point, the HLFS preset its own; a point that
    holds nothing gets the spine's `UnlitPass` (albedo + emissive over the
    background), the intended failure without the lights.
    `graph_parity::without_a_lighting_model_the_gbuffer_is_drawn_unlit` pins
    it, and that a model at the point replaces it. How others do it: Bevy's
    deferred lighting node is `bevy_pbr`'s, not `core_pipeline`'s (a core
    pipeline without it shows no lit surfaces); Unity SRP's renderer asset
    brings forward or deferred with its passes; Unreal builds the model in,
    which is the coupling removed here. Light-field shading asks for its own
    scene colour format in its preparer (the HLFS preset no longer names
    `HlfsPass`).
  - The lights register where the frame places them: the shadow views
    (`ShadowViewsFeature`) first in the shadow-caster phase's compute (the
    casters' dirty pass reads their flags), light binning (`LightCullFeature`)
    in `Slot::Visibility` after a kind's culling, building nothing under
    light-field shading. The spine lost `shadow_matrix`, `light_cull`,
    `deferred_light` and `scene_buffer_or_dummy`.
  - `helio_frame_api::lighting::LightingSettings` (the debug view and
    environment reflections: what a host asks of whichever model) replaced
    the spine settings' lighting fields.
  - The shadow host settings left `helio-settings`' renderer group for the
    shadows' own group (`helio_light_api::SHADOWS`, keys `shadows.quality`,
    `memory_mb`, `max_resolution`, `updates_per_frame`, `update_megatexels`,
    `max_distance`, `fade_frames`, `hysteresis`, `page_invalidation`;
    projects' old `rendering.shadow_*` values fall back to the defaults, as
    GI's and foliage's did). The facade no longer maps them (which also fixed
    `default_settings_are_the_default_config` and `every_choice_maps`, failing
    since A4) nor re-exports `GpuLight`, `LightType`, `ShadowQuality`.
  - `LIGHT_LIST` and `DIRECTIONAL_LIGHT` are no longer frame contracts: the
    light list is the lights' api contract (hard case 3).
  - `helio_scene_join::RowJoinKeys::directional_light` is an `Option` the
    kind placing a sun names: water's join sets the lights api's directional
    light, so an engine's row keys name no light (hard case 4: water's sun is
    the lights api's directional light).
- **Pulsar engine seams** (inventory registrations; the engine names no
  light):
  - `helio_bridge::ObjectJoinRegistration` / `ObjectJoinContext`: a plugin's
    object join (its source-row columns and its derivation), installed in the
    engine's scene-join group after the meshes' join, ahead of the LOD
    selection and the row joins (the seam batch 4 listed for the meshes'
    join). The lights' join (with the editor's light icons) is the first.
  - `level_rules::LevelSingletonRegistration` grew `held`, `default_is`,
    `enabled_only` and `edits_checked`: the one-directional-light rule is the
    lights plugin's registration (`lights_component::directional_lights`,
    `ONE_DIRECTIONAL_LIGHT`), as the sky and the wind are. `level_rules` names
    no component.
  - `helio_component::components::MovabilityRegistration`: what a plugin's
    instances author of their object's movability (`object_movability` read
    `LightComponent` by type).
  - `pulsar_scene` and `level_migrate` no longer force-link the light
    components (they cannot depend on a plugin; as the portals in batch 1).
- **One owner per light concept** (the user's rule, 2026-11-03).
  helio-light-api is the lights' whole public contract: the light row
  (`GpuLight`, `LightRow`, `LIGHT_LIST`, `MAX_LIGHTS`), the directional
  light (`DIRECTIONAL_LIGHT`, now two `vec4`: toward the light, and its
  radiance), the binning (`TILE_SIZE`, `MAX_LIGHTS_PER_TILE`,
  `ClusterLightGrid`, the tile buffers' keys), the shadow views (matrices,
  budget and residency, `ShadowSettings` and `SHADOWS`, `LightDirtyFlags`,
  the atlases' keys), the sun's occlusion pages (which a terrain publishes),
  and their WGSL: `//!use light_list` (`GpuLight`, the kinds, the shadow
  request and the sun rule, the tile index), `//!use light_forward` (the
  forward light function) and `//!use shadow_atlas` (`LightMatrix`, the cube
  face order, the budgeted atlas's addressing and sampling, the glass tint).
  How others do it: Bevy's `bevy_pbr` exports its light and cluster structs
  and `#import`able WGSL; Unreal's light data is one renderer-owned
  definition every shader includes. Here the owner is the plugin and the
  import is its snippet.
  - Copies deleted: GI's `SceneLight` (probe update and mesh tracer) and its
    first-directional-light scan; the sky's raw-word light row and its scan
    (both now the snippet's `light_is_sun`); fog's `GpuLight`,
    `LightMatrix`, cube-face order, atlas addressing and sampling and glass
    tint; the flare's `GpuLight`, `LightMatrix`, face order and atlas
    addressing; the voxel terrain's `SceneLight`, tile constants and shadow
    request; the meshes' forward drawer's `GpuLight`, tile constants and
    light loop (it placed local lights from a transform buffer it bound to
    another buffer, and read tile counts from the instance rows without a
    binning; now the light list's own positions and a zero stand-in) and
    the materials' transparent template's (both now call `light_forward`); every other kind's string copies of the lights'
    resource names (`"scene_lights"`, the atlases, the tile lists).
  - Copies that stay, with what they wait on:
    - GI and the sky still pick the sun from the light list (by the lights'
      rule) instead of reading `DIRECTIONAL_LIGHT`: standalone hosts write
      the light list directly and run no light join, so they publish no
      directional light. Waits on a directional-light derivation the lights
      give standalone hosts (`helio-light-host`).
    - The lights' own shaders (deferred, HLFS, light cull, the shadow
      matrices) keep their own declarations of the row and the atlas
      functions; they are the owner's and adopt the snippets as they change.
    - Water's reflection capture builds a `DeferredLightPass` in its own
      graph (with the meshes' passes): waits on captures as views of the
      frame's phases (batch 4's note).
    - The perf overlay (a spine diagnostic) reads `scene_lights` and
      `tile_light_counts` by name and marks after `helio.pass.deferred_light`
      by kind, as `FocusedTimingGroup` names fog: it adds no work or API.
    - helio-bake's `LightSource` (the lightmap baker's input) is its own
      light description; it is converted from the light row by
      `helio-mesh-host`.
- **Standalone hosts**: helio-mesh-host's presets add the lights' features
  (`helio_light_host::features`), so the meshes' examples and every
  plugin's examples draw lit as before; their light, billboard and
  reflection-capture rows are `helio_light_host`'s (the api's and the
  feature crates' `host_rows`). helio-components keeps the materials' row.
- **Tests moved with the lights**: Helio's budgeted-shadow residency and the
  billboards joining with their rows (`helio-light-host`), the light join's
  shader validation and the shadow shaders' (shadows), engine_backend's
  light hydration and level tests (`runtime_level`), class reload, and the
  light cases of the scene-join rows (lights_component's tests); the
  helio_component unfinished-components test.
- Left, with why:
  - The HLFS demos (`indoor_cathedral_hlfs`, `indoor_cathedral_fxaa_hlfs`,
    `monumental_arch_hlfs`, with `hlfs_viewer` and `hlfs_capture`) stay in
    meshes-examples: they are the cathedral and arch mesh scenes under the
    light-field model, built from the meshes' scene modules.
  - The light's editor UI (create, environment section, properties) stays
    in `ui_level_editor`, acting through lights_component, as the other
    kinds' tools do; `ObjectType::Light` (the scene format's object kind and
    the editor's create menu) is core's, as `ObjectType::Mesh` is.
- **Verified** (against p10: Helio e621b8a4 / Pulsar 540900129):
  - Helio's preset snapshot (every preset, three configs): after the seams
    commit byte-identical to p10 (912 lines); after the move the graphs lose
    exactly ShadowMatrix, LightCull, DeferredLight or HLFS, Billboard (and
    the perf markers after deferred lighting and billboards) and the
    `billboards` host input, and the twelve G-buffer graphs gain `Unlit`
    (648 lines); the HLFS graphs' scene colour falls back to the default
    format (light-field shading asks for its own).
  - Pulsar's `plugin_pass_lists` (now also asserting ShadowMatrix,
    LightCull, DeferredLight and no `Unlit`) is byte-identical to p10's
    (3792 lines).
  - voxel_craft (the p10 baseline exe vs p11, alternating, 8-frame bursts at
    4 poses, idle machine): identical render-graph dump; mean |Δ| p10-p11
    0.03-0.54 against p10-p10 0.04-0.25 and p11-p11 0.09-0.53, pose means
    within 0.12. (The base runs spent ~100 s in their first frame, compiling
    pipelines; not frame cost.)
  - The GI cathedral (frame 240, p11 against p10's kept captures): mean |Δ|
    0.19-0.34 against p10-p10 0.06-0.17 and p11-p11 0.13-0.25, means within
    0.3 levels; its candles flicker with wall-clock time.
  - `ies_demo` (p10 built once from 540900129 with the same capture hook, vs
    p11, alternating, frame 240): p10-p11 mean |Δ| 0.0003 (max 6), p10-p10
    and p11-p11 0.
  - `cargo check --workspace --all-targets` clean in both repos; layer gate
    0; Helio's tests pass but the known `helio-settings` one (the facade's
    two settings tests, failing since A4, now pass).
  - Tests from Pulsar: the lights' crates (api, features, host, component
    with the moved runtime-level, class-reload and scene-join tests, editor),
    GI, the sky, fog, the flare, water, foliage, voxel component and feature
    (presets, clouds, wgsl, voxel_pass_graph), the meshes' forward,
    transparent, G-buffer (now also validating the transparent template
    composed with the lights' snippets), shadow and host crates,
    helio_component, engine_backend's lib and scene-join rows, pulsar_game's
    lib and `plugin_pass_lists`: 818 pass. Not passing, both known from p9:
    water's caustics projection test and voxel's
    `chunk_pass_builds_settles_and_resizes_in_the_deferred_graph`.

As built, batch 6 (2026-11-04, branch `rewrite/p12-materials` in both
repos): the materials are the `pulsar.materials` plugin's; the Helio repo has
no material crate (helio-mats moved) and, with it, no kind's crate at all
outside its test host. Phase B is done.
- **Layout** (Pulsar `plugins/vendor/materials/`): `api` (helio-mats: the
  materials' whole public contract, below), `feature` (helio-pass-materials:
  `MaterialsFeature`, the material texture table), `component`
  (materials_component: the feature's registration and the material sampler
  from the project's `streaming.texture_*` settings, moved from
  `engine_backend`'s `material_sampler`), `editor` (materials_plugin). No
  examples crate: the material demos (`radiant_demo`,
  `architectural_materials`) are mesh scenes and stay in meshes-examples, as
  the HLFS demos did in batch 5.
- **Frame seams** (Helio):
  - `helio_core::FrameInputPublisher`: a node adds one to the graph
    (`RenderGraph::add_frame_input_publisher`); the host updates every
    publisher once a frame (`update_frame_inputs`) and the graph publishes
    them with the passes' frame inputs. The material texture table is one:
    the facade no longer reads the scene's texture store, makes the white
    fallback and the material sampler, or publishes `material_textures`.
    How others do it: Bevy's material plugin prepares its bindless slabs in
    the render world's prepare stage, before the graph; Unity SRP renderer
    features set globals in `OnCameraSetup`. A publisher is not a pass, so
    the pass lists do not change.
  - `helio_core::TEXTURE_BINDING_ARRAYS` and `TEXTURE_TABLE_LIMIT`: the
    device capabilities a bindless table needs, named for no kind (the
    facade and helio-xr ask devices for them; helio-mats'
    `BINDLESS_MATERIAL_FEATURES`/`MAX_MATERIAL_TEXTURES` are them).
  - The spine declares only `render_environment`; `material_textures` is the
    materials' host input (same node kind, so graph assets and pins are
    unchanged).
  - `helio-components` keeps the row helpers (`same_layout!`, `WorldRows`)
    and the bake: `MaterialComponent` is the materials api's host row
    (`host-rows`), as batches 2-5 did.
- **One owner per material concept**. helio-mats holds every material
  concept another kind reads: the row (`MaterialRow`/`GpuMaterial`,
  `MATERIALS`, flags, classes, `MaterialRow::STAND_IN`, the new foliage
  class), the texture table (`MaterialTextureTable`, `MaterialStandIn`,
  `MaterialBindingConfig`, `MaterialTextureBindings`, `MATERIAL_TEXTURES`,
  `MaterialSampler`), texture uploads and UV transforms (`texture`, from the
  facade), the template system (`radiant`, `templates`, `MaterialTemplates`)
  and the WGSL: `//!use material_row` (`GpuMaterial`, its words for raw
  readers, `MaterialInput`, the flag/class/workflow constants,
  `material_stand_in`, `material_transmission`,
  `material_sheet_transmittance`) and `//!use material_surface` (texture
  metadata, `SurfaceData`, `material_at`, `material_texture_data`,
  `sample_texture`, `default_pbr_surface`, `material_cut_out`), over the
  existing `pbr_eval` and `specular_aa`.
  - **Templates are surface functions; programs are the drawers'**. A
    template is only its `radiant_eval_surface` (or
    `radiant_eval_transparent`) and helpers over `MaterialInput`;
    `RadiantTemplate::compose(program, graph, max_textures, snippets)` puts
    it into a drawer's program. `DEFAULT_PBR` split: its surface function is
    the default template (`templates::DEFAULT_SURFACE`), its vertex layout,
    instance rows (now `//!use mesh_draw_rows`) and G-buffer packing are the
    meshes' `helio_pass_gbuffer::GBUFFER_PROGRAM`. `transparent_base`
    likewise: `TRANSPARENT_DEFAULT` and the meshes'
    `helio_pass_transparent::TRANSPARENT_PROGRAM`. `MaterialInput` (position,
    geometric frame, UV, the view position and the frame) is the material's
    contract with a drawer, as Unreal's material pixel parameters are with a
    vertex factory and Bevy's `PbrInput` with a mesh: templates no longer
    read a drawer's `cameras`/`globals`, so every template works in every
    drawer (custom templates now compile into the forward drawer, which used
    to take a G-buffer program for them).
  - Copies deleted: the forward drawer's whole material evaluation (struct,
    texture metadata, sampling, `evaluate_surface`, its default template);
    the depth-material program's (row, metadata, sampling, cut-out test, now
    `material_cut_out`); the portal G-buffer program's `GpuMaterial`
    (task copy 2), metadata, sampling, `default_pbr_surface` and default
    template (its cut-out threshold had drifted from the G-buffer's: alpha-
    tested materials now cut at the same 0.5); the transparent program's row
    and its light-free default; the object batch's and the meshes' join's
    rows; shadow transmittance's row and its glass formula
    (`material_sheet_transmittance`); VG cull/bin/debug's raw word offsets
    and flag constants; VG shade's slicing of `DEFAULT_PBR` (it takes the
    G-buffer program's view declarations and `compute_velocity`, and the
    materials' snippets); the radiant demo's iridescent template (a whole
    G-buffer program, now a surface function); the asset viewer's and the
    engine's graph-material tests' use of the G-buffer program (each has its
    own small program now); voxel's `SceneGpuMaterial` and its private
    binding-capability mirror; the G-buffer's metadata sentinel (the table's
    now); `engine_backend`'s material sampler.
  - **Voxel's palette** (task copy 1). The per-volume uniform was a private
    material format (`MaterialGpu`: colour, roughness, emission,
    transmission, patches, links) with its own evaluation. It is now an
    array of ordinary material rows (`GpuMaterial`, read with
    `//!use material_row`): base colour, roughness, emission and, for
    leaves, the foliage class with its transmission
    (`MATERIAL_CLASS_FOLIAGE`, `class_params.x`), plus the voxel's own
    variation of each row (`VoxelVariation`: dry/middle/lush patch tints of
    the row's colour). What the voxel keeps is its addressing (per-voxel ids,
    the face mapping, the triplanar texturing of `scene_materials.wgsl`) and
    its procedural per-voxel variation (patches, pigment, edges, `detail`),
    which needs the voxel grid's integer domain and so cannot be a material
    graph's. How others do it: Teardown's voxels index a palette whose
    entries are materials (colour, roughness, emission, glass); Minecraft's
    blocks are textures tinted per biome by the world. Here the entries are
    the engine's material rows and the tint is the voxel's. Emission follows
    the voxel's variation (the row's emissive scaled by the voxel's albedo
    over the row's colour), so a glowing voxel's pigment shows in its glow as
    before. The appearance JSON (levels' `appearance_parameters`, voxel
    world crate, GPU-free) is unchanged and converted to rows on upload.
  - The foliage class (`MATERIAL_CLASS_FOLIAGE` = 5, custom templates start
    at `FIRST_CUSTOM_MATERIAL_CLASS` = 6) has a template for meshes
    (two-sided flag, transmitted colour): a voxel's leaves and a mesh's leaf
    cards are one material.
- **Without the plugin** a drawer binds `helio_mats::MaterialStandIn` (a
  white texture table and one grey row, `MaterialRow::STAND_IN`), and
  `material_at` returns the same grey row for an id past the table: every
  surface draws default grey PBR, as batch 5's `Unlit` is the lights'
  absence. `meshes-examples`'
  `material_stand_in::without_the_materials_a_mesh_draws_the_grey_stand_in`
  draws one box without the plugin and with an authored stand-in row and
  compares.
- Left, with why:
  - **The engine's material table is the meshes' join**: a mesh section's
    row embeds its resolved material (`helio_component::MeshSectionDraw`)
    and the join writes one material row per section. Material rows
    authored by another kind cannot join that table, so voxel volumes carry
    their own rows (ordinary rows, read with the materials' code) instead of
    indexing the scene's table. Waits on material instances as entities:
    material assets (and per-slot overrides) as entities with the materials'
    row component, mesh sections naming them, the join writing an object's
    `material_slot` from that; then voxel slots name material assets too and
    index the one table, and the appearance's colours become material
    assets. Blocked, like the meshes' components, by core crates reading
    `helio_component`'s material assets by type.
  - The voxel appearance JSON describes slot materials by colour, roughness,
    emission and transmission (converted to rows on upload); it becomes a
    list of material references with the step above.
  - GI's mesh tracer binds its own no-table fallback for `vg_alpha`
    (opaque rows, white table); `MaterialStandIn` needs a queue its
    constructor lacks.
  - Material assets, the material graph compiler and cooked textures stay in
    `helio_component` with the meshes' components (core crates read them).
- **Verified** (against p11: Helio 60f0c59e / Pulsar b098737c1, its kept
  snapshots and exes):
  - Helio's preset snapshot (every preset, three configs, 648 lines) is
    p11's with exactly one change: the 15 graphs' inputs lose
    `material_textures`.
  - Pulsar's `plugin_pass_lists` (now also asserting the table is an input)
    is byte-identical to p11's (3792 lines): the table is a frame input, no
    pass, and its node keeps the host input's kind.
  - voxel_craft (`voxel_craft_p11.exe` vs p12, alternating, 8-frame bursts
    at 4 poses): identical render-graph dump; mean |Δ| p11-p12 0.24-0.30
    against p11-p11 0.33 and p12-p12 0.20, image means 61.39 / 61.42 (the
    p11 runs again spent their first frame compiling pipelines). The palette
    is gone with no visible change.
  - The GI cathedral (p11's `indoor_cathedral_gi` exe vs p12, frame 240):
    mean |Δ| 0.09-0.38 against p11-p11 0.31 and p12-p12 0.25 (candles
    flicker by wall-clock time), means within 0.4 levels.
  - `radiant_demo` (now with `HELIO_CAPTURE`) draws every built-in class and
    the demo's opal, glass, water and iridescent templates composed into
    the meshes' programs; there is no p11 build of it to compare.
  - `cargo check --workspace --all-targets` clean in both repos; layer gate
    0. Helio's tests: 427 pass, the known `helio-settings` one fails. Pulsar:
    380 tests pass in the materials crates, every meshes feature and host,
    portals, decals, sprites, GI, voxel (lib, `wgsl`, `presets`),
    meshes-examples' `material_stand_in` and `material_segments`,
    engine_backend's `material_graph_rendering` and pulsar_game's
    `plugin_pass_lists`; new tests compose every template into every mesh
    drawer and the portal program, natively and with expanded bindings.
- **What Phase C needs**, in this order:
  0. **A one-owner audit first.** Only batches 4-6 applied the rule (user,
     2026-11-03), each to its own concepts; batches 1-3 (the leaves, media,
     GI and sky) and the voxel move came before it. Sweep every plugin and
     the frame for private copies of concepts owned elsewhere: the earlier
     plugins' (wind, water, sky, fog, GI, sprites, portals, ...) and the
     frame's own (camera and view structs, the G-buffer layout, the colour
     chain, declared privately in WGSL). Delete each in favour of the owner's
     api or snippet, or record what it waits on. Done (see "As built, Phase C
     step 1").
  1. A plugin that isn't installed means its kind is absent (the stand-ins
     above, `Unlit`, `GBufferClear` are the frame's side of it; the engine
     still links every plugin).
  2. The sculpt tool, the terrain panel and the light UI leave
     `ui_level_editor` (plugin-provided tool modes, panels and undo).
  3. The mesh components (and with them the material assets and the engine's
     material table, above) leave `helio_component`.
  4. The repo split, last: each plugin its own `Plugin_<Kind>` repository at
     `plugins/vendor/<kind>/`, and the layer gate's rule that no Helio crate
     outside the test host names a kind's contract.

As built, Phase C step 1 (2026-11-04, branch `rewrite/p13-one-owner-audit` in
both repos): the one-owner audit. Every cross-kind concept has one owner (a
plugin, or the frame: `helio-core`, `helio-frame-api`, `helio-spine`, the
scene join); everything else reads it through the owner's api crate or WGSL
snippet. Method: the WGSL declarations of every shader were compared, by name
and by field layout, with every snippet's (`wgsl_snippet!` files, the
prelude, the scene join's rows); the Rust string literals used as resource
names were compared with every owner's key consts. The check that keeps it
(`one_owner`, below) is that comparison.

**Concept -> owner -> the copies deleted:**

| Concept | Owner (where) | Copies deleted |
|---|---|---|
| Camera (`GpuCameraUniforms`) | frame: prelude `Camera` | 24 private structs (`Camera`, `CameraUniforms`, `CameraUniform`, `VgCamera`; most named the inverse `inv_view_proj`, light cull's forward `direction_far`): decals, billboards, light cull, shadow matrices, forward/G-buffer/transparent programs, indirect dispatch, VG raster/resolve/debug, four portal shaders, sky LUTs, corona x2, post volume blend, the engine's authored LOD, Helio's tone map, SDF x2, simple cube; the unused `planar_reflection.wgsl`. VG raster/shade, voxel draw and the authored LOD now build through the prelude (`helio_core::shader::module`). Sprites' 2D view is its own (`SpriteCamera`). |
| G-buffer targets (order, formats) | frame: `gbuffer::{TARGETS, TARGET_FORMATS, color_targets, GROUP_MEMBERS, *_FORMAT}` | voxel's `GBUFFER_FORMATS`, foliage's `GBUFFER_TARGET_FORMATS` (its "duplicated rather than imported" note), the meshes' and portals' 8-entry colour-target lists, the meshes' group/target declarations, VG shade's storage formats |
| G-buffer surface tags (lightmap-UV channel) | frame: prelude `GBUFFER_TAG_UNLIGHTMAPPED/OWN_OCCLUSION/TRACED`, `gbuffer_tag_view_remap`, `gbuffer_has_lightmap`, `gbuffer_tagged`, `gbuffer_own_occlusion`, `view_remap_of` | raw `(-1,-2)`/`(-1,-1.5)`/`(-1,-1)`/`(-1,-3-c)` in voxel, foliage, portals, meshes, VG shade; the deferred lighting's `is_stored_voxel`/`is_foliage` (the lighting named two kinds) and the dead VG sentinel `is_vg` (nothing wrote `(-2,-2)`); frame-api's `VIEW_REMAP_SNIPPET` (the chain tag is a G-buffer tag) |
| Motion vectors | frame: `gbuffer::VELOCITY` (the encoding documented there) | TSR's `MOTION_VECTORS` |
| Depth names | frame: `resource_keys::DEPTH` (what the executor resolves), `depth::{FULL_RES_DEPTH, HIZ, HIZ_NEAREST, HIZ_SAMPLER}` | literals in the spine, Hi-Z, SSR, planar, SDF, simple cube, decals, foliage, occlusion culling, billboards, corona, portal overlay, transparency, the facade |
| Scene colour, G-buffer, render environment, ray transmission, environment probes/cubes, sun visibility, ... (every frame key) | frame consts | about 300 literal uses across Helio's own crates, the plugins and the engine |
| Participating-medium grid (depth mapping, composite) | frame: `media::SNIPPET` (`MEDIA_NEAR`, `media_slice_from_view_depth`, `media_view_depth_from_slice`, `media_grid_w`, `media_seen`) | transparency's hand-rolled `log(max(d,0.5)/0.5)/...`; the froxel mapping leaves the prelude (it was the media contract's) |
| Ray transmission rows | frame: `rays::RAY_TRANSMISSION_SNIPPET` | `RayTransmissionData` + accessors in SSR (Helio) and HLFS |
| Glare added in the tone map | frame: `color::{publish_glare, glare}` | the tone map reading the lens flare's `lens_output` by name (Helio named a kind); the flare publishes it |
| Scene-join rows (owner links, pool handles, transforms, frame-origin rebasing) | frame: `helio_scene_join::join_source` (`scene_join_common.wgsl`, `Handle` added), `origin_pair` | the authored LOD's `Owner`/`Transform`/`Handle`, its hand-rolled `object_relative` and private Rust `origin_pair`; `Owner`/`Handle` in the LOD apply and the VG instance derivation; `Handle` in the meshes' joins |
| Shadow views (`LightMatrix`), cascade selection, projection | lights: `//!use shadow_atlas`, new `//!use shadow_lookup` | the meshes' casters' `ShadowFace` x2 and shadow-dirty's `GpuShadowMatrix`; the prelude's `helio_csm_select`/`helio_shadow_project` (a lights concept in the frame) move to `shadow_lookup` for fog and the flare |
| Foliage ground rows | foliage api: `GROUND_SNIPPET` (`GroundQuery`, `GroundHit`, `GroundArgs`, `GROUND_NONE`, `ground_miss`) | placement's and the voxel ground's copies |
| Surface flags | materials: `pbr_eval` (`SURFACE_FLAG_TWO_SIDED`) | voxel's copy |
| Material rows key | materials: `MATERIALS` | the object batch's `MATERIALS_KEY` literal and `"materials"` in eight mesh/portal passes |
| Mesh pools, skin buffers, VG instances, static objects | meshes api: `MESH_VERTICES`, `MESH_INDICES`, `SKELETAL_SKIN_VERTICES`, `SKELETAL_BONE_PALETTE` (moved from `helio_component`), `VG_INSTANCES`, `STATIC_OBJECTS` | literals in five mesh drawers, portals, the mesh host, VG BVH, the engine's authored LOD and VG instance derivation |
| Material stand-in | materials: `MaterialStandIn` | GI's mesh tracer's own no-table fallback (batch 6 leftover, cleared: the tracer takes a queue) |

Name clashes that were not copies were renamed so a name means one thing: fog's
`Medium`/`medium_at` (`FogMedium`, `fog_medium_at`; the atmosphere's are the
frame's), sprites' `Camera` (`SpriteCamera`).

**The check**: Pulsar crate `one_owner` (`crates/core/one_owner`, no
dependencies, reads the tree; CI runs it with the workspace's tests). It fails
on a WGSL struct, const or function that another owner's snippet declares,
on a struct with the layout of another owner's snippet struct holding a
matrix, and on a Rust string literal used as a resource name (in
`ResourceKey::new`, `BufferKey::of`, graph reads and writes, pool lookups,
`reads`/`writes` lists) that another owner defines as a key. It skips an
owner's own files, test code, examples, attribute literals and `same_layout!`.
`exceptions.txt` lists what stays, each with what it waits on; the test also
fails when a listed copy is gone. Proof: `a_planted_copy_is_caught` (a
struct by name, a struct by layout, a const and a key literal, and not a
label, an attribute or test code), and planting `struct DecalView` with the
camera's layout in `decal_apply.wgsl` and `ResourceKey::new("gbuffer_velocity")`
in the decals' feature each failed the tree test.

**Copies that stay, with what each waits on** (`exceptions.txt` holds the
ones the check sees):
- The perf overlay (a frame diagnostic) profiles the default BRDF with its own
  copy and reads `scene_lights`/`tile_light_counts` by name, as
  `FocusedTimingGroup` names fog, HLFS and SSR: waits on a diagnostics seam
  where a kind registers its cost probe and counters.
- Fog reads the post volumes' rows (`post_process_volumes`) for the post
  settings' fog block, and the post settings ABI keeps `lens_enabled`: waits
  on the post plugin splitting both out (batch 2's note).
- The row join (`helio-scene-join`'s `row_join.wgsl`) holds water's and the
  particles' placements (`SUN`, `WIND`, `POOL_RANGE`, `MESH_BOUNDS`): waits
  on each kind bringing its placement as a snippet (batches 1, 2, 4).
- The mesh host's ray-tracing projection refuses scenes holding
  `foliage_layers` or `voxel_volumes` by name: waits on a traced-content
  contract (a kind says it has content rays must see).
- The G-buffer drawers each declare their fragment outputs (foliage writes
  six targets, voxel adds `frag_depth`, so one WGSL struct does not fit);
  their order and formats are `TARGETS`, their pipelines' targets
  `color_targets()`.
- The asset viewer's material program copies the meshes' G-buffer `Globals`
  because compiled material graphs read `globals.time`: waits on the graph
  compiler reading time from `MaterialInput`.
- The owners' own shaders that predate their snippets keep their
  declarations (the lights' `GpuLight`/`LightMatrix`/`budget_*` in deferred,
  HLFS and light cull; the meshes' `GpuInstanceData`, `GpuDrawCall`,
  `DrawIndexedIndirect`, `INSTANCE_FLAG_*` in culling, shadows and the object
  batch; water's `WaterVolume` x7; sprites' `SpriteInstance` x4; portals'
  `GpuPortalView` x4): one owner each, adopting the snippets as they change.
- Batch 4-6 leftovers not cleared: the portals' second G-buffer drawer and
  re-culling (waits on drawers drawing extra views); water's reflection
  capture graph (captures as views); GI and the sky picking the sun from the
  light list (a directional-light derivation for standalone hosts); the
  engine's material table being the meshes' join, voxel volumes' own rows and
  the appearance JSON (material assets as entities); the mesh components and
  material assets in `helio_component` (Phase C step 3).

**Pre-rewrite first-party plugins**, what each owns and how it relates to the
rewrite plugins:

| Plugin (repo) | Owns | Overlap | Decision |
|---|---|---|---|
| `skm_editor` (Plugin_SKM) | `skeletal_assets` (every skeletal format: `.skeleton`, `.animseq`, `.skmesh`, `.blendspace`, `.montage`, `.animgraph`, IK rigs and retargets, pose DBs; the animation runtime; CPU skinning) and the animation editor | `helio_component`'s `SkeletalMeshComponent` decodes `skeletal_assets::skmesh` and calls its skinning; the editor preview has its own `SkinVertex` and skinning WGSL; `material_ids.rs` reads and writes material documents | Owner of **animation** (skeletons, clips, graphs, the pose runtime: `pulsar.animation`), not of skinned drawing: the animation plugin writes joint palettes, `pulsar.meshes` owns the skinned vertex rows and skinning in its drawers and the palette/skin keys (moved to `helio-mesh-api` here). Its preview should draw through the engine renderer (as the asset viewer does), and its material ids go through the materials' component api once material assets leave `helio_component`. |
| `shader_editor` (Plugin_Shaders) | the `.material` graph asset (`shader_graph_save.json`), the graph editor, its own PSGC compile and texture lowering, a Lambert "PBR" preview | a second graph compiler beside `helio_component::material_graph`; the preview is not the engine's BRDF; the import companion overlaps `meshes/asset-compat`'s material converter; the graph file name is a literal in seven crates | **Merge into `pulsar.materials`**: the graph UI becomes the materials plugin's editor, its compile path is deleted for the one compiler (which moves into `materials_component` with the material assets, Phase C step 3), the preview draws Radiant templates through the engine renderer. |
| `matter_editor` (Plugin_Matter) | `.pif` texture painting (layers, history, brushes, tile compositor) | none in code: nothing reads `.pif`, painted textures do not reach the material texture table | **Separate, with a contract**: it produces images; the texture-asset pipeline (`texture_assets`, cooked textures) imports them, as any image. |
| `asset_viewer` (in tree) | viewers for `.fbx`, `.mesh`, `.png`; per-slot default materials of a mesh | its own mesh pipeline and graph-material program; the `Globals` copy above | Its mesh viewer and slot materials become `pulsar.meshes`' editor (drawing through the engine renderer); the image viewer stays generic. |
| `render_bps` (in tree) | the `.rendergraph` editor over `helio-graph-asset` | none (it uses the graph asset and the engine's node registry) | Stays; the frame owns the graph asset. |
| `physics_plugin` (Plugin_Physics) | physics assets, rigid bodies, colliders, joints, Rapier | `RigidBodyComponent` duplicates the legacy `pulsar_physics::RigidbodyComponent` (still read by `level_migrate`) | Owner of physics; the legacy components go once `level_migrate` stops reading them. No rendering overlap (water hitboxes go through water's api). |
| `input_plugin`, `ui_editor`, `blueprint_editor`, `code_editor`, `table_editor` | input actions/maps; widgets; blueprints; script editing; SQLite tables | none with the rewrite plugins | Stay. |

How others split these, and why ours: Unreal keeps skeletal meshes,
animation (AnimInstance, Persona) and materials (one HLSL translator the
Material Editor and the cooker share) as engine modules; Unity splits
`SkinnedMeshRenderer` from the Animator and has Shader Graph compile through
the active pipeline's own code; Bevy splits `bevy_animation` from
`bevy_pbr`'s skinning and has no graph. Ours takes the Bevy/Unity line for
animation (the animation owner writes palettes, the mesh owner skins) and
Unreal's for materials (one compiler, the editor is the owner's editor):
each concept has one owner, an editor is never a second implementation of
what it edits, and a project that drops the animation plugin still draws
bind-posed meshes.

**Superseded repos** (the user deletes from this list; nothing was deleted,
renamed or archived): none of the `Plugin_*` repositories is superseded yet.
Pending, each superseded once the named work lands:
- `Plugin_Shaders`: superseded by `pulsar.materials` (its graph editor merged
  as the materials' editor, its compiler replaced by the materials' one)
  after the material assets leave `helio_component` (Phase C step 3) and the
  merge lands. Do not delete before.
- `Plugin_SKM`: not superseded; it becomes the animation plugin's repository
  (rename, the user's call) at the repo split.
- Not the rewrite's to judge, noted for the user: `Plugin_Trait` (an
  in-tree `plugins/trait_editor` exists), `Plugin_Struct`, `Plugin_Enum`,
  `Plugin_TypeAlias`, `Plugin_DAW`, `Plugin_Diagrams`, `Plugin_Terminal` and
  `Plugins` are not linked by Pulsar's tree.

**Verified** (against p12: Helio 83db5934 / Pulsar 0ca924809, its kept
snapshots and exes):
- Pulsar's `plugin_pass_lists` dump is byte-identical to p12's (3792 lines).
- Helio's preset snapshot (every preset, three configs, 648 lines) is p12's
  with one intended change: the 15 graphs' tone maps, which have no lens,
  no longer read `lens_output` (the glare is published, not named).
- voxel_craft (`voxel_craft_p12.exe` vs p13, alternating, 8-frame bursts at
  4 poses): identical render-graph dumps in all four runs; mean |Δ| p12-p13
  0.22-0.39 against p12-p12 0.29 and p13-p13 0.31, image means 61.39-61.46.
- The GI cathedral (p12's `indoor_cathedral_gi` exe vs p13, frame 240):
  identical graph dumps; mean |Δ| 0.019-0.023 for three of the four runs
  against p12-p12 0.028, the fourth run 0.27 from every other (its candles
  flicker by wall-clock time).
- `portal_cube` (p12 built once in the p12 worktree vs p13, frame 240):
  bit-identical, identical graph dumps.
- Helio: `cargo check --workspace --all-targets` clean, layer gate 0, 427
  tests pass (the known `helio-settings` one fails), including
  `wgsl_validation` over the new snippets and
  `resolving_twice_includes_once`.
- Pulsar: `cargo check --workspace --all-targets` clean; `one_owner` passes
  (and fails on each planted copy); the touched crates' tests pass (110 test
  binaries of the lights, fog, lens, GI, sky, foliage, water, sprites,
  particles, decals, post, portals and meshes crates, voxel's `wgsl` and
  `presets`, engine_backend's lib with the authored LOD's GPU tests and
  `scene_join_rows`, helio_component's lib), but water's known
  `caustics_projection_writes_signal_for_the_cathedral_pool`.

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
