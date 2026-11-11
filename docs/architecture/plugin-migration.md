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
same path, like physics). Done in p18: every plugin below is its own
repository (the table's second column), a submodule at
`plugins/vendor/<kind>/`.

| Plugin | Repository | Components | Feature owns | Leaves Helio/Pulsar today's |
|---|---|---|---|---|
| `pulsar.materials` | [Plugin_Materials](https://github.com/Far-Beyond-Pulsar/Plugin_Materials) | material assets, material graph | helio-mats (GPU material ABI, Radiant templates, texture bindings), material texture means | done (batch 6): `plugins/vendor/materials/` (`api` = helio-mats, `feature` = helio-pass-materials: the material texture table, root = materials_plugin (every registration, p19), `component` = materials_component, `editor` = materials_editor); the Helio repo has no material crate. The material assets and the graph compiler are the plugin's since Phase C step 3 part B (`assets`, `graph`, `graph_preview`; the meshes read them through `helio_mats::asset`); images stay core. Left: the engine's material table being the meshes' join (see "As built, batch 6") |
| `pulsar.meshes` | [Plugin_Meshes](https://github.com/Far-Beyond-Pulsar/Plugin_Meshes) | StaticMesh, StaticMeshDraw, SkeletalMesh, LOD, VirtualGeometry | object batch, culling (occlusion, indirect dispatch), mesh drawers (GBuffer, ShadowCasters, Forward, Transparent), skinning, LOD selection, VG (all `helio-pass-vg*`, `helio-vg-*`), ray proxies + TLAS publication, the mesh part of the scene join | crates done (batch 4): `plugins/vendor/meshes/` (`api` = helio-mesh-api, `feature/*`, `vg/*`, `asset-compat`, `host` = helio-mesh-host, root = meshes_plugin (every registration, p19), `component` = meshes_component, `editor` = meshes_editor, `examples` = meshes-examples, `snapshot`, `web-demos`, `android-demos`); the Helio repo has no mesh crate. The SceneDB components, mesh assets and import, scene join, authored LOD selection, VG host, picking and thumbnails are the plugin's since Phase C step 3 part B, through generic registrations (see "As built, Phase C step 3 part B") |
| `pulsar.lights` | [Plugin_Lights](https://github.com/Far-Beyond-Pulsar/Plugin_Lights) | Light, ReflectionCapture | light list, light cull, shadow views + atlas (shadow matrix/dirty/atlas; casters come from drawers), the lighting model (deferred light, forward light function), IES, light icons (as billboards) | done (batch 5): `plugins/vendor/lights/` (`api` = helio-light-api, `feature/{cull,shadows,deferred,hlfs,billboard}`, `host` = helio-light-host, root = lights_plugin (every registration, p19), `component` = lights_component with `LightComponent`/`ReflectionCaptureComponent`, `editor` = lights_editor, `examples` = lights-examples); the Helio repo has no light crate |
| `pulsar.sky` | [Plugin_Sky](https://github.com/Far-Beyond-Pulsar/Plugin_Sky) | Atmosphere | atmosphere tables, clouds, sky composite; publishes `sky`, `sun_visibility`, `sun_transmittance` | done (batch 3): `plugins/vendor/sky/` (root = `sky_plugin`, `sky_component`, `feature` = helio-pass-sky, `sky_editor`, `sky-examples`); no api crate: what others read of it is the frame's |
| `pulsar.gi` | [Plugin_GI](https://github.com/Far-Beyond-Pulsar/Plugin_GI) | (none; settings) | probe clipmap, tracers interface; publishes `indirect_diffuse` | done (batch 3): `plugins/vendor/gi/` (root = `gi_plugin`: the feature's registration and settings, `api` = helio-gi-api, `feature` = helio-pass-gi, `gi-examples`; no component or editor crate) |
| `pulsar.fog` | [Plugin_Fog](https://github.com/Far-Beyond-Pulsar/Plugin_Fog) | GlobalFog, LocalFogVolume | volumetric fog and its composite | done (batch 2): `plugins/vendor/fog/` (root = `fog_plugin`, `fog_component`, `feature` = helio-pass-volumetric-fog, `fog_editor`, `fog-examples`); publishes the frame's participating media |
| `pulsar.post` | [Plugin_Post](https://github.com/Far-Beyond-Pulsar/Plugin_Post) | PostProcessVolume, CameraPostProcess | volume blend, publishes post settings | done (batch 2): `plugins/vendor/post/` (root = `post_plugin`, `post_component`, `feature` = helio-pass-post-volumes, `post_editor`, `post-examples`); blends over the host's `POST_BASELINE` |
| `pulsar.lens` | [Plugin_Lens](https://github.com/Far-Beyond-Pulsar/Plugin_Lens) | CameraLens | lens, flare (reads `light_list` via lights api) | done (batch 1): `plugins/vendor/lens/` (root = `lens_plugin`, `lens_component`, `feature/{lens,flare,optics}`, `lens-examples`; no editor crate); the view overscan through `ViewOverscanRegistration` |
| `pulsar.water` | [Plugin_Water](https://github.com/Far-Beyond-Pulsar/Plugin_Water) | WaterVolume (+ physics hitbox rows via physics api) | water sim and draw, caustics as `transmission` | done (batch 2): `plugins/vendor/water/` (root = `water_plugin`, `water_component`, `api` = helio-water-api, `feature` = helio-pass-water-sim, `water_editor`, `water-examples`) |
| `pulsar.foliage` | [Plugin_Foliage](https://github.com/Far-Beyond-Pulsar/Plugin_Foliage) | Foliage, Wind | placement, drawers | done (batch 2): `plugins/vendor/foliage/` (root = `foliage_plugin`, `foliage_component`, `api` = helio-foliage-api, `feature/{place,gbuffer}`, `foliage_editor`, `foliage-examples`) |
| `pulsar.voxel` | [Plugin_Voxel](https://github.com/Far-Beyond-Pulsar/Plugin_Voxel) | VoxelTerrain, VoxelTerrainLayers | chunk passes, drawers, GI tracer, foliage ground, water terrain | done (layered-rewrite step 6.6): Pulsar `plugins/vendor/voxel/` with root `voxel_plugin` (every registration, p19), `component` (voxel_component), `api` (helio-voxel-data), `world` (helio-voxel-world), `feature` (helio-pass-voxel-planet, a G-buffer drawer since Phase A), `editor` (voxel_editor), `examples` (helio-voxel-examples: voxel_craft, voxel_flight, water_world); the Helio repo has no voxel crate. Left: the sculpt tool and terrain panel in `ui_level_editor`, waiting on plugin-provided tool state, undo, commands and save hooks |
| `pulsar.decals` | [Plugin_Decals](https://github.com/Far-Beyond-Pulsar/Plugin_Decals) | Decal | decal drawer (after GBuffer) | done (batch 1): `plugins/vendor/decals/` |
| `pulsar.particles` | [Plugin_Particles](https://github.com/Far-Beyond-Pulsar/Plugin_Particles) | ParticleEmitter | corona | done (batch 1): `plugins/vendor/particles/` (the `corona_emitters` host input is the feature's since Phase A) |
| `pulsar.sprites` | [Plugin_Sprites](https://github.com/Far-Beyond-Pulsar/Plugin_Sprites) | Sprite | sprite cull/batch/simulate, 2D radiance cascades | done (batch 1): `plugins/vendor/sprites/` |
| `pulsar.portals` | [Plugin_Portals](https://github.com/Far-Beyond-Pulsar/Plugin_Portals) | Portal (SubLevelDisplay stays in the engine: it places level instances, which rigid frames use too) | portal cull, draws, projection derivation, view remaps | done (batch 1): `plugins/vendor/portals/`; counts as the `PORTAL_COUNTS` payload, pairing and projection as a space producer |
| `pulsar.splines` | [Plugin_Splines](https://github.com/Far-Beyond-Pulsar/Plugin_Splines) | Spline | debug-line drawing through the facade's debug draw | done (batch 1): `plugins/vendor/splines/` (component and editor crates; lines through `EditorLinesRegistration`) |
| `pulsar.physics` | [Plugin_Physics](https://github.com/Far-Beyond-Pulsar/Plugin_Physics) | (exists: physics_plugin + pulsar_physics) | water hitbox rows through water's api | done (batch 2): `pulsar_physics` derives them under `helio_water_api::WATER_HITBOX_SOURCES_BUFFER` |

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
  - **The lighting model is a plugin's registration**
    (Far-Beyond-Pulsar/Helio#393 item 3): a feature registers a
    `helio_frame_api::lighting::LightingModel` (`registrar.lighting_model`,
    stored by type through `FeatureRegistrar::provide`): an id, its shading
    (`GBuffer` or `Forward`), the extension point its nodes join, and what it
    reads (the specular-layer mode, where the lighting inputs run, whether
    TSR gets a reactive mask), plus the pipelines it brings
    (`ModelPipeline`: id, label, anti-aliasing). The lights register
    `deferred` (`lighting.model.deferred`), `light-field` (HLFS,
    `lighting.model.light_field`, the `helio.hlfs` and `helio.fxaa_hlfs`
    pipelines) and `forward` (the drawers light as they draw). A preset
    chooses a model by id (`Preset::Deferred` is `deferred`,
    `Preset::Forward` is `forward`, a model's pipeline is
    `Preset::Model(pipeline)`); the phases, the layer mode and the inputs
    order come from the model, so `helio-presets` names no model beyond those
    two ids. A chosen id nobody registered resolves to the spine's `unlit`
    model (registered by the spine; its `UnlitPass` draws albedo + emissive
    over the background): the intended failure without the lights, so
    forward without the lights is an unlit G-buffer pipeline.
    `graph_parity::without_a_lighting_model_the_gbuffer_is_drawn_unlit` pins
    it, and that a registered model replaces it;
    `a_model_brings_its_own_pipeline_and_reads` pins a model's own pipeline.
    The fallback is not silent: `Preset::model` logs one warning per missing
    id (`lighting model `<id>` is not registered ...`) the first time a graph
    build resolves to the stand-in, and `a_missing_model_warns_once_naming_its_id`
    pins it. A project that disables the lights plugin and selects Forward or
    Deferred therefore gets an unlit scene (albedo + emissive, background
    kept), not black or broken output; before the model registration the
    forward preset without the lights had no lighting at all. Every built-in
    preset builds on the stand-in with the lights absent (the same graph_parity
    test runs `Preset::ALL`); a non-black render of it needs a mesh drawer, an
    object kind's, so it is not asserted in Helio.
    How others do it: Bevy's deferred lighting node is `bevy_pbr`'s, not
    `core_pipeline`'s (a core pipeline without it shows no lit surfaces);
    Unity SRP's renderer asset brings forward or deferred with its passes;
    Unreal builds the model in, which is the coupling removed here. A
    light-field model growing into ReSTIR DI or a path-traced model is a
    plugin change, not a frame change. Light-field shading asks for its own
    scene colour format in its preparer.
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
     still links every plugin). Done (see "As built, Phase C step 2").
  2. The sculpt tool, the terrain panel and the light UI leave
     `ui_level_editor` (plugin-provided tool modes, panels and undo). Done
     (see "As built, Phase C step 3").
  3. The mesh components (and with them the material assets and the engine's
     material table, above) leave `helio_component`. Done, but the material
     table (see "As built, Phase C step 3 part B").
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

As built, Phase C step 2 (2026-11-05, branch `rewrite/p14-absent-plugins` in
both repos): **a plugin a project does not enable is absent.** Its components
are not registered, its Helio features are not in the render graph (the
frame's stand-ins take their place), its settings, tool modes and menus do not
appear, and a game built for the project does not link it. All of it is
Pulsar's; Helio's only change here is this section.

How others do it, and why ours:
- Unreal: a `.uplugin` describes a plugin (modules by type, `EnabledByDefault`,
  plugin dependencies); the `.uproject`'s `Plugins` list enables or disables
  them per project; the editor loads only enabled plugins' module DLLs, and a
  packaged game links only theirs (UnrealBuildTool generates the module list).
- Bevy: `PluginGroup`s (`DefaultPlugins.build().disable::<T>()`), chosen in
  code per binary; no per-project data, and an editor serving many projects
  would need one build each.
- Unity: packages in the project manifest; scripts compile per project.
- Ours is Unreal's model with the list in data, minus DLL loading: one editor
  binary serves every project, so the editor links every installed plugin and
  the project's list decides which register (a runtime gate, the same thing
  Unreal gets from not loading a DLL); a game is built per project, so it
  links only the enabled plugins (Unreal's packaged build). A kind is decided
  in one place (the project file), never by editing a host's dependencies, and
  the gate is generic: no host, registry or editor panel names a plugin to
  gate it.

**The pieces** (Pulsar):
- **Descriptor** `plugins/vendor/<kind>/pulsar-plugin.toml` (16 plugins): `id`,
  `name`, `description`, `enabled_by_default`, `requires` (other plugins its
  modules link), `[modules] runtime` (what a game links: the component crate)
  and `editor` (the editor crate). The runtime module carries it
  (`pulsar_plugins::installed!("../pulsar-plugin.toml")`, which also gives the
  crate `plugin_descriptor()`; the editor crates' `PluginMetadata` name and
  description are the descriptor's now, one owner). Linked = installed.
- **Project list**: `Pulsar.toml`'s `[plugins]` table, `"pulsar.voxel" = false`;
  a plugin the list does not name keeps its default. `pulsar_plugins::resolve`:
  enabled = listed or default; an enabled plugin's requirements are enabled
  with it unless the list turns one off; a plugin whose requirement is off or
  not installed is disabled, with a problem saying why (logged). Today's
  requirements are the component crates' real links: voxel requires sky, fog,
  post, foliage and lights (presets, the sun, its fog and sky rows), fog
  requires post (`FogMode`).
- **The gate** (`crates/core/pulsar_plugins`, no engine dependencies): every
  registration a plugin submits names its **origin**, the package that
  submitted it (`origin: pulsar_plugins::origin!()`); the registries iterate
  `pulsar_plugins::registered::<T>()` instead of `inventory::iter`, which skips
  a registration whose origin is a module of a disabled plugin
  (`origin_enabled`). Types gated: `RenderFeatureRegistration`,
  `AnimatedContentRegistration`, `ViewOverscanRegistration`,
  `EditorLinesRegistration`, `SceneRowsRegistration`, `ObjectJoinRegistration`,
  `LevelSingletonRegistration`, `SpaceProducerRegistration` (engine_backend),
  `SettingsGroupRegistration` (pulsar_settings), `SceneSubsystemRegistration`,
  `ScenePayloadRegistration` (pulsar_world_registry),
  `TextureBindingRegistration`, `MovabilityRegistration`, `GroundMaterialNames`
  (helio_component). World component classes: `WorldComponentRegistration`
  gained `origin` (emitted by `#[register_world_component]`); the registry's
  own copy filters the class list and the classes' ticks, events and
  unfinished reports (cached per activation; `ABI_VERSION` 2). A disabled
  plugin's stored components load as `UnresolvedComponent`s, kept verbatim, so
  saving loses nothing; `level_rules::check_new_components` refuses a class
  nothing registers (`NOT_REGISTERED`).
- **Activation**: `engine_state::set_project_path` calls
  `pulsar_plugins::activate_project` (read, resolve, activate). Nothing
  activated (a game, a test) = every linked plugin enabled.
- **Linking, editor**: `plugins/installed` (`installed_plugins`) depends on and
  re-exports every installed plugin's runtime modules and, with `editor`, its
  editor modules; its manifest blocks and re-exports are generated from the
  descriptors (`tests/installed_set.rs` fails on drift,
  `UPDATE_INSTALLED_PLUGINS=1` rewrites; it also checks every descriptor is
  installed by its runtime module and that a plugin submits registrations only
  from crates its descriptor lists). The editor binary (`pulsar_engine`) links
  it. `ui_core` lost its 16 `*_plugin` dependencies.
- **Linking, game**: `pulsar_game` names no plugin (16 `*_component`
  dependencies, the `extern crate`s and the `scene::LightComponent` re-export
  are gone). `engine_backend/build.rs` bakes the installed descriptors and
  each runtime module's dependency spec (`game_plugins.rs`); the project
  builder writes a plugin block in the generated game's `[dependencies]`
  (regenerated every build from `Pulsar.toml`, a user-owned manifest's own line
  for a module wins) and `src/plugins.rs` (`use <module> as _;` per enabled
  runtime module, `mod plugins;` in `main.rs` and the PIE `lib.rs`).
- **Other hosts**: `scene_inventory` links `installed_plugins`;
  `pulsar_package`'s, `pulsar_game`'s and `ui_level_editor`'s tests reach
  plugin types through `installed_plugins` (dev-dependency, runtime modules)
  where they named the crates.
- **A drawer without its views**: the meshes' shadow drawer (the only
  shadow-caster phase drawer) built a `ShadowDirtyPass` that needs the lights'
  `LightDirtyFlags`, so a preset without the lights failed to build ("Shadow
  Dirty Flags needs the Shadow Matrices node"). The phase's views are the
  lights' shadow faces: without them both nodes now build nothing.
- **Editor UI still in `ui_level_editor`** (step 3 moves it into the plugins):
  minimal runtime gates so a disabled plugin's UI does not appear. Tool modes
  (spline, voxel sculpt, terrain) register only while their plugin is enabled
  (`register_tool_modes`); the Create menu offers a kind only while its class
  is registered (`create::available`); the World Settings sky and wind
  sections, the environment mixer's add buttons, sidebar entries and starter
  setups likewise (`environment::available`). Every add goes through
  `level_rules`, which refuses an unregistered class.

**What step 3 must finish** (also the handover note):
- `ui_level_editor` still depends on `voxel_component`, `lights_component`,
  `splines_component`, `sprites_component`, `portals_component`,
  `fog_component`, `post_component`, `water_component`, `foliage_component`
  and `sky_component`, so the editor links them through it as well as through
  `installed_plugins`. Move the sculpt and terrain modes, the spline mode, the
  terrain panel, the light UI, the World Settings sky/wind/environment
  sections, the portal debug overlay and the create entries into the plugins'
  editor crates as registrations carrying an origin (tool modes, panels,
  create entries, undo, save hooks), then drop those dependencies and the
  `pulsar_plugins::is_enabled("pulsar.*")` calls in `register_tool_modes`.
- `helio_component` and `engine_backend` still name the meshes' and the
  materials' crates (mesh components, material assets, the engine's material
  table, VG host, LOD, picking): the meshes and materials plugins are absent
  from the graph and the registries when disabled, but their crates link
  through the engine. The mesh components leaving `helio_component` (step 3)
  ends that.
- Cross-plugin links recorded as `requires`: voxel's component crate reads
  sky, fog, post, wind and light components; fog reads post's `FogMode`. Each
  should become the owner's api (or an optional contract), so voxel works with
  any of them disabled.
- The pre-rewrite plugins (physics, input, skm, ui_editor, asset_viewer,
  blueprint, code/matter/shader/table editors) have no descriptor; the
  linked editor provider registration (`plugin_manager::LinkedEditorProvider`)
  has no origin (the Blueprint submodule constructs it). Give them descriptors
  at the repo split (step 4).
- `pulsar_scenedb`'s GPU mirror/clear/handle-ledger registrations and
  `pulsar_reflection`'s method/property-editor registrations (external
  crates) carry no origin: a disabled plugin's are inert (no instance of its
  classes exists) but still linked in the editor.

**Verified** (against p13: Helio 6214f73a / Pulsar 22b5b4f69, its kept
snapshots and exes):
- **Absent**: `installed_plugins`' `absent_plugins` test (every installed
  plugin linked, as in the editor; a project disabling voxel, lights and
  materials, activated through `engine_state::set_project_path`): their
  component classes are not registered (others are); their features
  (`helio.voxel`, the lights' five, `pulsar.materials`) are not in the
  renderer's set and no preset's graph has a light or voxel pass or the
  material texture table input, the G-buffer presets resolve through
  `Unlit`; their settings groups are absent; a level holding a light and a
  voxel terrain loads, keeps both records verbatim and refuses new ones; the
  game renderer draws the level's cube unlit in its own albedo (centre pixel
  145, 73, 8); another project enabling them brings everything back.
  engine_backend's `generated_game_links_only_enabled_plugins`: the generated
  manifest and `src/plugins.rs` name the 13 other runtime modules, follow the
  list when it changes, and keep a user's own line.
  `a_project_without_some_plugins_compiles_without_them` (pulsar_game,
  ignored like the drift check, run once: 738 s): the generated game for a
  project disabling the three compiles (`cargo check`), and `cargo tree` of it
  has none of their crates.
- **Nothing else changed**: Pulsar's `plugin_pass_lists` dump is p13's with
  one difference: voxel's `VoxelChunks` now appears in 16 graphs (and TSR
  reads its `tsr_stochastic`). The p13 test named 15 component crates and
  never linked voxel (it came only through `pulsar_game`, which the test
  binary never referenced); it now links the installed set. With those lines
  removed the dump is byte-identical (3792 lines). Helio's preset snapshot is
  unchanged by construction: no Helio code changed.
- voxel_craft (`voxel_craft_p13.exe` vs p14, alternating, 8-frame bursts at 4
  poses): identical render-graph dumps (29 passes); mean |Δ| p13-p14
  0.21-0.36 against p13-p13 0.18 and p14-p14 0.37, image means 61.40-61.45.
  The p13 runs again spent their first frame compiling pipelines (130 s and
  96 s against 22 s).
- `cargo clippy --workspace --all-targets` with CI's flags: no errors (it
  includes `cargo check`). `one_owner` passes. `pulsar_plugins` (8 tests:
  descriptors, project list, defaults, requirements, problems),
  `installed_plugins`' `installed_set` (3), the lib tests of `pulsar_game`,
  `engine_backend`, `ui_level_editor` and the 32 plugin component and editor
  crates (469 pass), `pulsar_world_registry` (all pass). Known failure, not
  this step's: ui_level_editor's
  `sculpt_stamps_do_not_rebuild_the_properties_section` (since p9); its
  sibling `the_layers_stack_offers_headings_and_cards_not_one_giant_row`
  passes now that the tests link the editor modules.

As built, Phase C step 3 (2026-11-05, branch `rewrite/p15-plugin-editor-ui` in
both repos): **the level editor is a shell; every kind's editor UI is its
plugin's.** `ui_level_editor` depends on no plugin crate (its tests reach
the components through `installed_plugins`' runtime modules). Step 2's
runtime hiding (`register_tool_modes`' `is_enabled` calls,
`create::available`, `environment::available`) is deleted: a disabled
plugin's UI is absent because its editor crate's registrations are skipped,
like every other registration. All of it is Pulsar's; Helio's only change is
this section.

How others do it, and why ours:
- Unreal: the LevelEditor module exposes extensibility managers (menu and
  toolbar `FExtender`s), `FEditorModeRegistry`/`UEdMode` for tool modes
  (Landscape and Foliage are plugin modes), the PropertyEditor module's
  per-class detail customizations, placement-mode categories, and
  transactions (`FScopedTransaction`/`Modify`) for undo. Plugins depend on
  the editor modules; the editor names no plugin.
- Unity: attributes discovered by reflection over the loaded assemblies
  (`[EditorTool]`, `[CustomEditor]`, `[MenuItem("GameObject/..")]`) and
  `Undo.RecordObject`; removing a package removes its assembly.
- Godot: an `EditorPlugin` per plugin (`add_control_to_dock`,
  `add_custom_type`, `_forward_3d_gui_input`, `get_undo_redo()` with
  do/undo methods), enabled per project in `project.godot`.
- Bevy has no first-party editor; ecosystem editors take windows from
  plugins through `App` extension traits, chosen at compile time.
- Ours: Unreal's dependency direction (each plugin's editor crate depends
  on the shell's extension points, the shell names no plugin), static
  registration like Bevy's (`inventory`: no assembly scan, no module
  loading, typed at compile time), gated per project as in step 2 (Godot's
  per-project enable, Unreal's `.uproject`). Each extension point is one
  type in `ui_level_editor::extend` (its one owner). Most are data the shell
  renders the same way for every kind (a create entry, a level component's
  section, an environment listing, a preset), so a plugin cannot make a
  menu or a section behave differently from the others; code hooks are kept
  for what is inherently code (a tool mode, an undo step, a save hook).
  Adding a kind adds a crate and edits nothing in the shell; the lists are
  sorted by `order`, read through `pulsar_plugins::registered` (a filter
  over a few dozen statics).

**The extension points** (`ui_level_editor::extend`, each with
`origin: pulsar_plugins::origin!()`):
- `ToolModeRegistration` (a `ToolMode`; `ToolMode::label`/`description`
  return the plugin's own translated text, `ModeLayout::dock_width` replaced
  the shell's per-mode width match). The shell keeps only Level Edit.
- `CreateEntry` (id, Create-menu text, environment-mixer text, hierarchy
  quick-add button, `offered` (one sun), `create`): the Create menu (grouped
  by hundreds of `order`), the mixer's add buttons and the hierarchy's quick
  adds are the registered entries. Entries are addressable by id
  (`extend::create(state, "sky")`), so a plugin uses another's entry
  without linking it (the lights' starter setups add the sky plugin's sky
  when it is enabled). `extend::add_object` is the undoable add-and-select
  most entries are.
- `EnvironmentPreset` (the starter setups, the lights'),
  `EnvironmentListing` (classes the mixer's sidebar lists, with the create
  entry offered when the level has none).
- `LevelComponentSection`: a component a level holds once (the sky's
  atmosphere, the global wind), found by the owner's own method
  (`level_atmosphere`, `level_wind`); the shell renders its World Settings
  section (reflected property rows or a note and a create button) and its
  mixer sidebar entry. The two hand-written sections are deleted.
- `SaveHook` (work before a level save: the voxel worlds' journal
  compaction and kept chunks) and `EditorData` (a key of the level file's
  editor section: the foliage palette, `foliage_sets`; the section is a
  flattened map, so the file format is unchanged and a key no enabled plugin
  reads is kept as it was).
- Undo: `SceneCommand::Extension(Box<dyn ExtensionCommand>)` (a plugin's
  command; the shell captures the objects its `scope` names before and
  after, one undo step: Add Voxel World) and
  `SceneEdit::Extension(Box<dyn ExtensionUndo>)` with
  `SceneDomain::commit_extension_edit` (a plugin's own compact step: a
  sculpt stroke's appended edits). Undo labels are the shell's keys or the
  plugin's text (`UndoLabel`).
- `Interaction` (a tool's interaction across frames: the shell finishes it
  before undo and polls it), `SelectionFollowUp` (the spline tool follows
  selections made by assistant tools), `PropertyFollowUp` (after a property
  edit: choosing a terrain's generator calls the voxel component's new
  `ensure_generator_settings`, the method a script calls), `CameraFraming`
  (Focus frames a voxel world over its ground).
- `viewport::DebugOverlay` (the portals' links panel, badges and ghost
  geometry): shown from the Overlays menu or by its debug view mode (the
  renderer then draws lit); the viewport keeps which are shown by id.
- `AiToolNamespace`: a module of `#[tool]` functions (the assistant's
  spline tools); the registry is rebuilt when the project's plugins change.
- Plugin state: `PluginStates` (`state.editor.plugins`, by type; a state
  holding history opts out of tool-query copies) replaced the shell's
  `TerrainDomain`, `SplineDomain`, `VoxelSculptDomain` and `voxel_stroke`
  fields. `extend::panel_kit` (the tool grid, sections, steppers) is the
  widget kit plugins' tool panels share.

**Where the UI went** (each plugin's editor crate, `level_editor` module,
with its own `locales/`; 214 keys left the shell's 16 locale files):
- voxel: the Voxel Sculpt mode and panels, the Terrain mode (world presets,
  the level's terrains), strokes and their undo step, Add Voxel World, the
  save hooks, the generator follow-up, framing.
- foliage: the foliage palette and brush, now its own **Foliage** mode
  (Unreal splits Landscape and Foliage the same way; it was a tab of the
  voxel plugin's Terrain panel), the palette as editor data, the global
  wind's Create entry and section.
- splines: the Spline mode, panel, preview, curve operations and the
  assistant's spline tools.
- lights: the Create menu's lights, the mixer's lights and reflection
  capture, the three starter setups. sky: the Create entry and section.
  fog, post, water: the mixer's entries and listings. sprites: the
  hierarchy's quick-add. meshes: the Create menu's primitives. portals: the
  debug overlay.

**Tests moved with their kind**: the full Create menu, the plugins' modes
and sections, a starter setup, Add Voxel World's undo, the foliage
palette's save/load and the voxel chunks a save keeps are
`installed_plugins`' `tests/level_editor.rs` (the editor as assembled: the
plugins' editor crates depend on the shell, so linking them into the
shell's own unit tests would link a second copy of it); the voxel
inspectors' are `voxel_plugin`'s `tests/inspector.rs`; the absent-plugins
test now also checks the disabled plugins' tool modes, create entries,
presets and listings are absent and come back.

**Left, with why**:
- `requires` are unchanged: voxel requires sky, fog, post, foliage and
  lights because its component crate links them (its presets compose a
  world of those kinds; its air sync derives the sky and the haze from the
  terrain; its runtime reads the sun), and fog requires post (`FogMode`,
  the post settings' fog block). Dropping voxel's means acting on those
  kinds through the world registry's by-name component methods and
  properties (`invoke_component_method`, `set_component_property`, the
  scripting path) instead of their types; recorded as a follow-up.
- `ObjectType::Light`/`Mesh` (the level format's object kind,
  `pulsar_scene_model`) still give hierarchy icons and the assistant's
  object kinds; `SetMovability` names the mesh and light classes; the
  shell's tests generate the HLFS cathedral with `helio-mesh-api` (a
  dev-dependency).
- The mesh components, material assets and the engine's material table
  leaving `helio_component` (step 3's second part, branch
  `rewrite/p16-mesh-components`) is not started: about 60 files read them
  by type, including the Blueprint and Physics plugin submodules; its scope
  and seams are in the p16 handover note.

**Verified** (against p14: Helio b3e0911c / Pulsar e0b17d138, its kept
snapshot and exe):
- Pulsar's `plugin_pass_lists` dump is byte-identical to p14's (3856
  lines). Helio's preset snapshot is unchanged by construction (no Helio
  code changed).
- voxel_craft (`voxel_craft_p14.exe` vs p15, alternating, 8-frame bursts at
  4 poses, `VOXEL_CRAFT_NO_CLOUDS`: the p14 exe spends 70 s compiling
  pipelines against 22 s, and the clouds follow wall time): mean |Δ|
  p14-p15 0.12-0.28 against p14-p14 0.28 and p15-p15 0.21, image means
  58.53-58.56.
- `cargo clippy --workspace --all-targets` with CI's flags clean;
  `one_owner`, `installed_set`, `absent_plugins` (now with the editor UI),
  `level_editor` (8, including the GPU kept-chunks save), the voxel
  inspector tests, the plugins' lib tests and `ui_level_editor`'s lib (109)
  pass. Known failure: `sculpt_stamps_do_not_rebuild_the_properties_section`
  (since p9). `render_acceptance`'s sky and sprite tests pass;
  `foliage_sways_in_the_global_wind_on_the_frame_clock` fails locally
  (0 texels moved; its code path matches p14's, not run on p14).

As built, Phase C step 3 part B (2026-11-05, branch
`rewrite/p16-mesh-components` in both repos, Plugin_Physics and
Plugin_Blueprints): **the material assets are the `pulsar.materials`
plugin's and the mesh components the `pulsar.meshes` plugin's.**
`helio_component` holds no material asset, graph compiler, graph preview or
mesh component; the meshes read material assets through the materials' api
only, and no core crate names a mesh or material type (what the engine runs
for them is generic registrations, below). All of it is Pulsar's; Helio's
only change is this section.

*Materials.*

How others do it, and why ours:
- Unreal: a mesh component holds `UMaterialInterface`s (the material
  module's interface: `GetRenderProxy`, `GetRelevance`); `UMaterial` and its
  HLSL translator are behind it, so a mesh never sees a graph.
- Unity: `Renderer.sharedMaterials` are `Material`s, an engine type; Shader
  Graph compiles through the pipeline's own code.
- Bevy: the mesh does not know materials at all; `MeshMaterial3d<M>` is a
  second component on the entity and the material plugin queues the mesh's
  draws.
- Ours is Unreal's interface line: a mesh slot names a material asset by id
  (SceneDB's `material` asset kind) and binds the loaded payload as
  `helio_mats::asset::MaterialAsset`, which resolves to the row a section
  draws (`resolve(id, version, scene, fallback)`). The interface is in the
  materials' api crate, so the meshes link no material code, and a project
  without the plugin has no loader for the kind: the asset fails to load and
  the slot draws the mesh's imported surface, as `MaterialStandIn` is the
  table's absence. Bevy's separate component is the direction for
  per-entity material overrides (the "material instances as entities" step
  below); it needs the meshes' join to index a material table it does not
  write, which is that step's work, not this one's.

**The materials' api** (`helio_mats::asset`, new): `MATERIAL_ASSET_KIND`,
`SHADER_GRAPH_FILE`, `SurfaceValues` (scalar PBR values: a surface
document's, and a model import's: the meshes' `ImportedSurfaceMaterial` is
it), `SurfaceValues::row` (the one scalar-surface-to-row formula; the
meshes' `surface_material` copy and the skeletal draw's use of it are
gone), `SurfaceMaterialDocument` (the `.mat` file, moved from
`static_mesh_component`), `ResolvedMaterial`, the `MaterialAsset` trait
(`resolve`, `reads_time`, `warm`, `held_bytes`) and `MaterialPayload` (the
kind's SceneDB payload).

**The plugin** (`materials_component`): `assets` (the loader: a surface
document, or a graph compiled on the loader thread with its textures
decoded; the compiled-graph cache and the texture users for reloads;
`LoadedMaterial` implements `MaterialAsset`), `graph` (the one material
graph compiler, was `helio_component::material_graph`), `graph_preview`
(the asset viewer's preview compile). The asset viewer reads the preview
from it (an editor plugin; games never link it).

**Generic asset seams** (`helio_component::asset_component`, core; each
names no kind):
- `AssetLoaderRegistration { origin, kind, load, size }`: the owner of a
  SceneDB asset kind. `install_asset_loaders` installs every linked one once
  per process; each load checks `pulsar_plugins::origin_enabled`, so a
  disabled plugin's kind fails ("not enabled") and switching projects needs
  nothing. `declared_asset_id(kind, source)` declares an asset with the
  loaders installed; `load_asset_now` loads one on the caller's thread for
  tools (thumbnails, the asset viewer) through the enabled owner.
- `AssetReloadRegistration { origin, kind, sources_for_file }`: which world
  assets a changed file reloads. `engine_backend::scene::reload_updated_asset`
  asks every registration for the event's kind (it named the material,
  mesh and texture functions); the materials register for material files
  and for images (the graphs sampling them), images and meshes for their
  own files.
- `asset_source(path)`: the one authored-path normalization (was
  `material_source`).
How others do it: Unreal's `UFactory`/`UAssetManager` type registry and
Bevy's `AssetLoader` registered per extension by the plugin that owns the
type (`app.register_asset_loader`); ours registers by SceneDB asset kind,
gated per project like every other registration.

**Textures**: images stay core (`texture_assets`, `cooked_texture`,
`texture_detail` in `helio_component`): sprites, decals and materials all
read images, and cooking, residency and the scene texture store are not a
material concept (Unreal's `UTexture` is the engine's, not the material
module's). A material graph's texture slots are the material's: the plugin
asks the image residency for each texture it samples. The texture-store
upload helper moved into `texture_assets`. Deleted: `material_textures`'s
`register_graph_texture` (nothing called it but its own test) and
`engine_backend`'s never-called `reload_animation_asset`.

**Tests moved with their owner**: `material_assets` (load, rebind, reload,
bad input; through the meshes' slots) and `material_graph_rendering` (the
compiler and GPU pixels) are the materials plugin's; the mesh upload and FBX
cache tests that lived in `material_graph_rendering` are the meshes'
`mesh_uploads`.
`absent_plugins` checks the material asset kind is absent with the plugin
disabled (no loader, a slot naming a material draws its imported surface, a
changed material file reloads nothing) and back when enabled.

*Meshes.* **The meshes' components are the `pulsar.meshes` plugin's.**
`meshes_component` holds the SceneDB components (`StaticMeshComponent`,
`StaticMeshDraw`, `SkeletalMeshComponent` with its notifies and its five test
files, `LODComponent`, `VirtualGeometryComponent`), mesh assets
(`mesh_cache`, `mesh_files`: the `.mesh` format, loading, the embedded
primitives), the slots' materials (`mesh_materials`, through the materials'
api), picking geometry (`mesh_spatial`), the virtual-geometry host and
instance derivation, the authored LOD selection (`authored_lod`, from
`engine_backend`, with its two shaders), the virtual-geometry project
settings (`vg_settings`, from `pulsar_settings`), the model importer and the
rigged import (`import`, from `asset_import`) and the legacy record
migrations (`records`, from `pulsar_class` and a copy in `pulsar_scene`). The
meshes' editor crate (`meshes_plugin`) holds the asset-browser thumbnails
(source models through `helio_snapshot`, from `ui_common`; `.mesh` files
rasterised on the CPU). `helio_component` keeps only what is shared: the
asset seams, images (texture assets, cooking, residency), cameras,
movability and its motion gate and play-mode watch, ground material names,
reflected lists, sublevel displays.

No core crate names a mesh type or crate: `engine_backend` lost
`helio-pass-object-batch`, `helio-pass-vg`, `helio-vg-*`,
`helio-asset-compat`, `psgc` and `image`; `asset_import` lost
`helio_component`, `helio-asset-compat` and `skeletal_assets`;
`pulsar_scene` lost `helio`, `helio-asset-compat` and `helio_component`;
`pulsar_game` lost `skeletal_assets` and its mesh re-exports; `ui_common`
lost `helio_snapshot`; `ui_file_manager` lost `helio_component`;
`level_migrate` links `installed_plugins` instead of naming mesh classes.
Kept, with why: `engine_backend` keeps `helio-mesh-api` for the sublevel
actor row its sublevel displays author (`sublevels.rs`; a scene row the
meshes' api defines, an object kind of the engine's own). The level format's
`ObjectType::Mesh` (hierarchy icons, the assistant's object kinds) and the
asset-category enums (`AssetKind::Mesh`/`Material` in `ui_types_common`,
`pulsar_content`, `engine_fs`'s watcher) are file vocabulary, not types;
turning asset categories into registered kinds is the repo split's (step 4).

**The seams** (each generic, each a registration a plugin submits with its
origin, so a disabled plugin's are skipped):
- `ObjectJoinRegistration` gained `order` (the meshes' join first, the
  lights' after) and `owner_bounds` (the instance bounding spheres row joins
  place by: the meshes'; without one the key names rows nobody writes).
  The meshes' scene join and every mesh column registration are the meshes'
  registration; `engine_backend` installs whatever is registered.
- `RenderHostPartRegistration` / `RenderHostPart`
  (`helio_renderer::host_parts`): per-renderer state a plugin adds:
  `configure` (the VG runtime into the renderer config), `derivations`
  (installed between the object and row joins: VG asset readiness, VG
  instances, the authored LOD selection), `prepare_frame` (VG streaming with
  the project's settings), `keeps_near_plane` (authored meshes keep the
  camera's near plane under a voxel ground's certified one). Tests and tools
  reach a part by type (`HelioRenderer::host_part`, `set_up_host_parts`);
  the renderer's VG-specific API (`virtual_geometry_runtime`,
  `set_virtual_geometry_settings`, `SceneRendererOptions::virtual_geometry`)
  is gone, and the meshes' `host_part` module has the two helpers. How
  others do it: Unreal's scene view extensions and per-scene subsystems
  (Nanite's streaming manager), Bevy's render-world resources plugins
  insert; ours is Bevy's line by registration.
- `ContentGeometryRegistration` (`helio_renderer::interaction`): a kind's
  content geometry as the editor reads it: local bounds (framing, the
  picking broad phase), a local ray (picking) and its triangles
  (`content_triangles`: the portals' ghost geometry, which read mesh
  vertices by type). As Bevy's picking backends, one per kind.
- `AnimatedContentRegistration` (existed): the meshes' "a shader graph reads
  the clock" answer; the engine's own mesh check is gone.
- `MovabilityRegistration` (existed for the lights): the meshes' instances
  author movability through it; `object_movability` names no class, and the
  editor's Set Movability edits every component with a `movability`
  property of that type instead of naming the mesh and light classes.
- `TickPhaseRegistration` / `ContentReportRegistration`
  (`engine_backend::scene::game_hooks`): the game tick's phase before the
  systems and after the scripts (the skeletal animation phase and notify
  publication), and a headless run's report lines (the meshes' bound
  count; `HeadlessReport::world` replaces `meshes`). As Unreal's tick groups
  and Bevy's schedule labels.
- `ComponentEventsRegistration` (`engine_backend::scene::component_events`):
  events a class raises that its assets declare (a skeletal mesh's notifies,
  for the Blueprint palette, by class name); replaces the engine's
  re-export of the meshes' catalogue.
- `ImporterRegistration` (`asset_import`): the import service, database and
  options schema (`asset_import::options`, from `mesh_cache`) stay core; each
  format's importer is a plugin's (the meshes' model importer, with
  `Importer::assign_default_materials` for import companions). As Unreal's
  `UFactory` and Bevy's `AssetLoader` registration. `ui_file_manager`'s
  drag and drop asks the registry.
- `RecordMigrationRegistration` (`pulsar_class::records`): a plugin's legacy
  record migrations, run with the engine's on every level load (the meshes
  fold `MaterialOverrideComponent` and turn `props.mesh_asset` into a mesh);
  `RecordMigrations::plugins` reports them. `pulsar_scene`'s second copy of
  the material-override fold is deleted (levels are migrated by
  `pulsar_class` before it ran). As Unreal's per-class `PostLoad` fix-ups.
- `ProjectSettingsRegistration` (`pulsar_settings::project`): a plugin's
  project settings namespace (the meshes' `project.virtual_geometry`, same
  keys, so project files are unchanged).
- `helio_mesh_api::skinning::SkinnedMeshAccess`: the skinned pose a physics
  rig reads (mesh asset, bone ids, model-space pose) and the bones it drives
  and releases. Plugin_Physics' rig reads and drives skinned meshes through
  it (`behavior::read_rows_with`) and links no meshes crate; per the step 1
  audit, the meshes own the skinned mesh, the animation runtime stays
  `skeletal_assets` (the animation plugin's crate, which the skeletal
  component still drives: separating the animation instance from the
  component is the animation plugin's step at the repo split).
- `pulsar_game::scene::{attach_class, has_component}`: a script attaches a
  component by class name with the data a level stores, the path every kind
  takes from a script that links no plugin; the generated starter script
  attaches its cube this way (it named `StaticMeshComponent` through
  `pulsar_game::scene`, which no longer re-exports mesh types).
- Thumbnails: `engine_fs` lost its model-format list
  (`register_mesh_thumbnail_renderer`); `ui_common`'s
  `register_thumbnail_renderers` installs every linked
  `ThumbnailRendererRegistration`; the meshes' editor crate registers the
  model and `.mesh` renderers.

**Tests moved with the meshes** (`meshes_component/tests`): the scene join
rows, movement scaling and group motion (from `engine_backend`), the mesh
uploads, static mesh assets and GPU mirror, content dedup (and its
benchmarks), VG instances (from `helio_component`), the model imports:
companions, multi-mesh, multi-skin, rigged, real sample, the import service
(from `asset_import`; its own tests keep a test-only importer), the record
migrations (from `pulsar_class`), the animation phase order (from
`pulsar_game`), picking (three of `interaction`'s tests). Plugin tests that
place rows by a mesh (lights, sprites, water, materials) dev-depend on
`meshes_component`; the editor's and the game's integration tests reach it
through `installed_plugins`.

**Verified** (against p15: Helio 53248a95 / Pulsar 8d71139e1, its kept
snapshot and exe; p15's own test binaries built once for the comparisons
below):
- Pulsar's `plugin_pass_lists` dump is byte-identical to p15's (3856
  lines), with the meshes' and materials' registrations moved. Helio's
  preset snapshot is unchanged by construction (no Helio code changed).
- voxel_craft (`voxel_craft_p15.exe` vs p16, alternating, 8-frame bursts at
  4 poses, `VOXEL_CRAFT_NO_CLOUDS`), two rounds in opposite order: identical
  render-graph dumps (29 passes); mean |Δ| p15-p16 0.39-0.45 against p15-p15
  0.17 and 0.36, p16-p16 0.14 and 0.18, and the darker build swaps between
  the rounds (image means 58.44-58.57). It is the wall clock (the game clock
  follows frame time), not the build: voxel_craft draws through Helio and
  `helio-mesh-host`, which this step does not touch.
- `cargo clippy --workspace --all-targets` with CI's flags is clean, and
  `cargo check --workspace --all-targets` is clean. `one_owner` passes. The
  submodule pin check passes.
- Tests pass:
  - The meshes' crate: lib (81), and every moved suite: picking, animation
    phase, scene join rows, movement scaling, group motion, mesh uploads,
    static mesh assets and GPU mirror, content dedup and its benchmarks, VG
    instances, the five model-import suites, the import service, record
    migrations.
  - The meshes' editor crate (the thumbnails), `helio-mesh-api`,
    `helio-mats`, `materials_component` (lib, `material_assets`,
    `material_graph_rendering`), `helio_component` (lib 41, the image
    suites), `asset_import`, `pulsar_class`, `pulsar_scene`,
    `pulsar_settings`, `ui_file_manager`, `ui_common`, `physics_plugin`
    (70).
  - `engine_backend`: lib (84) and every test suite.
  - `pulsar_game`: lib (90), `asset_reload`, `runtime_render_parity`,
    `voxel_render_parity`, `plugin_pass_lists`.
  - The lights', sprites' and water's suites.
  - `installed_plugins`: `absent_plugins` (now with the meshes' and the
    materials' absence), `installed_set`, `level_editor`.
  - `ui_level_editor`: `mesh_reimport`, `skeletal_component_property_editors`,
    `virtual_geometry_acceptance`.
- Failing, each failing the same way on p15's binaries, so not this step's:
  - `ui_level_editor` lib: `sculpt_stamps_do_not_rebuild_the_properties_section`
    (since p9).
  - `render_acceptance`: 6 of p15's 10 failures. Both foliage tests, the
    editor graph's perf overlay, the lens prescription, water fog after a
    move, and light shadows off. On p14, the foliage sway test's meadow is
    not drawn at all (the frame shows only the editor grid).
  - `virtual_geometry_dense_acceptance`: 3 of 4 tests.
  - `pulsar_game`'s `random_world_blueprint`: the voxel generator's 8-layer
    limit.
  - `scene_inventory`: below.
  - The generated-game test (`a_project_without_some_plugins_compiles_without_them`,
    ignored and heavy) now disables the meshes too and checks
    `meshes_component` is absent from the game's cargo tree. Pulsar's CI runs
    it with `--ignored`; it was not run locally.

**Left, with why** (each another step's design, not a move this step
could make):
- **The engine's material table is still the meshes' join** (a section's
  row embeds its resolved material). Material assets as entities with the
  materials' row component, sections naming them and voxel slots indexing
  the same table changes the scene join's GPU data, the voxel palette and
  the level format: the material table's design (batch 6's note), which the
  `MaterialAsset` interface does not preclude.
- **Animation inside the skeletal component**: `SkeletalMeshComponent` still
  drives `skeletal_assets`' animation instance (the animation plugin's
  crate). Splitting the animation instance into the animation plugin's own
  component, which writes the palette the meshes skin with, is that
  plugin's step (its repository is Plugin_SKM, renamed at the repo split).
- **Asset categories** (`AssetKind::Mesh`/`Material` in `ui_types_common`,
  `pulsar_content`, `engine_fs`'s watcher; the level format's
  `ObjectType::Mesh`): file vocabulary rather than types; making them
  registered kinds is the repo split's (step 4).
- **`scene_inventory`'s ledger and architecture checks** fail on rows and
  roots older than this step: pass crates that left Helio in Phase B,
  buffers and classes of the moved kinds, and world-crate state and voxel
  editor queries from p14/p15. CI runs them, among the GPU suites that
  crash on its lavapipe.
  This step updated only its own rows (the mesh classes' owners and tests,
  the animated-content query site it removed). What the ledger and the
  renderer roots should mean once each plugin is its own repository is the
  repo split's rule (the layer gate's).
- `engine_backend` keeps `helio-mesh-api` for the sublevel actor row
  (above). The row join's `MESH_BOUNDS` placement in `helio-scene-join`
  stays the `one_owner` exception it was: the bounds key is now the meshes'
  `owner_bounds` registration, but the placement WGSL still names mesh
  bounds.
- **The pre-existing test failures above** (render acceptance, dense VG,
  random world, the sculpt stamps perf test) are the cleanup phase's (p17):
  each fails identically on p15 and none is in code this step changed.

As built, p17 (2026-11-06, branch `rewrite/p17-cleanup` in both repos): the
cleanup phase before the repo split. Every open item of p5-p16 was checked
against the code (some had been fixed since: GI's mesh tracer already binds
`MaterialStandIn`, `DEFAULT_PBR` is gone since the templates became surface
functions drawers compose, `SetMovability` was already generic over a
`movability` property and only its message named kinds).

**New Helio seams** (each generic; no kind named in Helio):
- *A feature brings scene derivations*
  (`FeatureRegistrar::scene_derivation(SceneDerivationFactory)`): the node
  registry keeps them, the compiled graph carries them, and the renderer makes
  each one once, by name, after the host's own derivations
  (`with_scene_derivation`), so it reads what they published. A host needs no
  wiring for a kind's derived buffers.
- *A feature declares its swappable passes* (`FeatureRegistrar::swappable`,
  `swap_group`): merged into the preset's `SwapPolicy` when the graph
  compiles. Phase B had dropped every moved pass from the policy, so a shader
  reload of a plugin's pass rebuilt the whole graph; each plugin feature now
  declares its own (the shadow and Hi-Z groups by the meshes, which link both
  sides).
- *Indirect specular layers at any resolution*
  (`helio_frame_api::lighting::INDIRECT_SPECULAR_SNIPPET`,
  `indirect_specular_at(layer, uv)`): SSR traces at half the internal
  resolution while the deferred lighting `textureLoad`ed the layer at the
  full-resolution pixel, reading the wrong texel (a quarter of the screen,
  magnified). The contract now says a layer may be any resolution and readers
  reconstruct it with the frame's premultiplied 2x2 filter (SSR's compose pass
  had its own copy; both use the snippet now).
- *Diagnostics offers* (`helio_frame_api::diagnostics`): the performance
  overlay measures what kinds offer while a graph builds, a `TileCounter`
  (the lights' tile light counts, written after their deferred node) and a
  `SurfaceProbe` (the materials' default BRDF through `pbr_eval`), with a
  Lambert stand-in. Its GGX copy and the `scene_lights`/`tile_light_counts`
  literals are gone (seven `one_owner` exceptions); its probe lights are
  synthetic, so it no longer read the light list with a 48-byte struct of its
  own over 128-byte rows.

**Lights and sky**: the directional lights are derived once, by one kernel
(`cs_directional_lights`, `light_is_sun`), for every host: the lights' join
records them after joining, and `DirectionalLightsDerivation` (brought by the
binning feature) derives them from a light list a standalone host wrote
itself, unless a derivation before it published them. `DIRECTIONAL_LIGHTS`
is a list (up to 32 suns in light-list order: direction, radiance, emitter
disk; the first is the primary), so a sky with two suns (Tristan's p10
multi-sun atmosphere) keeps working: the atmosphere copies the list into its
frame, GI and water read the primary. Nobody scans the light list.

**How others do it, and why ours**: Unreal's renderer picks the atmosphere
lights once per scene (`FScene::AtmosphereLights`) and every consumer reads
them; Unity HDRP resolves one sun for the frame; Bevy's atmosphere takes the
first directional light. Ours is the same one-pick rule, but owned by the
lights and run on the GPU for any host, which a plugin layout needs: a host
that never heard of a sky still gets the row the sky reads. Unreal hot-reloads
by shader type without rebuilding the renderer's pass setup; ours lets each
owner say which of its passes are self-contained. Unreal traces SSR at reduced
resolution and upsamples in the reflection composite; ours puts the
reconstruction in the frame contract so every reader of every layer agrees.
Unreal's shader-complexity and light-complexity views read what the material
compiler and the light grid report; ours lets the owners offer the same.

**Voxel requires nothing**: `pulsar.voxel` linked the sky, fog, post,
foliage and lights component crates (presets compose those kinds, the air
sync derived the sky and haze from the terrain, the runtime read the sun).
Now:
- *A ground's dependents follow it by method*: the air rules moved to their
  owners as component methods, `AtmosphereComponent::follow_ground` and
  `GlobalFogComponent::follow_ground` (the sky's air presets and the fog's
  haze presets are those plugins' vocabulary); voxel's sync calls
  `follow_ground` on whatever instances of its terrain's object answer it
  (`components_with_method`, change cursors by component id), so it names
  no kind and a third kind can follow a ground the same way.
- *Presets and the sun by class name and property* (`voxel_component::others`):
  the replaced sky, wind, haze and ground cover, the updated sun and look,
  and the runtime's sun are found by `ComponentMeta::class_name` and read
  through `pulsar_world_registry::get_component_property`, the scripting
  path. Voxel's descriptor requires nothing; the five crates are its tests'
  dev-dependencies.
How others do it: Unreal's landscape and sky are independent actors a level
template composes, and Unity's terrain knows no sky; Bevy plugins meet at
components another plugin queries by type, which is a link. Ours composes
by name and method, so a plugin never links another kind's crate, and the
owner of a concept (the sky's air) keeps its rules.

**Foliage on a ground that exists**: since p14 the voxel feature is in
every editor graph and offered its ground to foliage at build time even
with no terrain, so a meadow grew on nothing (both render-acceptance
foliage tests). `GroundGeometry::present` (default true) says whether a
ground has content; the voxel ground reports false without volumes, and
placement grows on its built-in ground meanwhile (an offered ground that is
present but loading still has no chart, so nothing pops).

**Smaller owners' fixes**: the asset viewer's material program had a copy of
the meshes' G-buffer `Globals` only for `frame` and `time`; it has its own
two-field clock (the template's time hook keys on the tag, not the layout).
The owners' own shaders take their declarations from their snippets: the
lights (`GpuLight`, `LightMatrix` in culling, deferred, HLFS, shadows),
water (`WaterVolume`, one snippet for nine shaders), sprites
(`SpriteInstance`) and portals (`GpuPortalView`).

**Tests** (each failure's cause, and whether the test or the code was wrong,
is in the p17 handover note): ui_build's plan tests (test wrong: stale after
the regeneration step), plugin_manager's dependency order (Linux: a plugin
loaded through `/proc/self/fd/N` whose fd was closed made the next plugin
reuse its name), the packaging smoke (the packager now links the installed
plugins, so their record migrations run in the cook), the random world
(example wrong for Earth's 7-layer preset: `remove_layers_of_kind`), the
sculpt stamps (a property type unregistered for JSON made every change read
as shown), water caustics (test rebuilt a stale pipeline: it now drives
`CausticsProjection`), scene_inventory (renderer roots are every plugin's
`feature`/`host` crates; pass crates scanned in every plugin; ledger rows
brought up to date).

**CI**: `.config/nextest.toml`'s `ci` profile leaves out, by name, the suites
that render frames (lavapipe SIGSEGVs and minutes per frame took ~90 tests and
the 60-minute limit on p16); the CPU tests of the same crates still run.

**Resumed p17** (after the coordinator's list, same branches):
- *Tristan's p10 commits* went up every level of both stacks (p11..p17,
  pushed at each level, each level pinning its Helio level). His code moved
  to its owners: the light-icon picking became the engine's generic
  `ViewIconRegistration` (any kind submits a billboard icon; the lights
  submit theirs), the billboard pipeline test went to the billboard crate,
  water's private `GpuLight` copy became the lights' `LIGHT_LIST` snippet,
  the bloom test went to Helio's post-process crate, the sky rig to the
  lights editor's presets.
- *Render acceptance*, every failure a real bug or a stale test: the buffer
  view's depth/normal modes (shader compared 40/41, Rust sends 42/43: the
  shader takes a `normals` flag now); a water row was always an ocean, so a
  lake's fog followed the camera everywhere (`WaterVolumeComponent::bounded`
  clips surface, underwater fog, shafts and caustics to the box; the row
  join keeps `world_phase.zw`); point and spot shadows never darkened (a
  constant NDC bias exceeds forward-perspective depth separation at tens of
  metres: `shadow_receiver_depth` biases local lights two texels toward the
  light in world space, Unreal's receiver-plane normal offset); the foliage
  coverage threshold measured a 256^2 frame against a 192^2 depth buffer (TSR
  render scale); the whole suite ran out of memory because device-loss
  watching kept every `wgpu::Device` alive (entries now release with their
  last clone); the windy lake test asked for wind trains Tristan removed on
  10-29 (wind seas are the spectral FFT ocean; the solver moves only bodies'
  own disturbances, which the test now checks); the lens prescription test
  raced the background bank bake.
- *The graph editor is the materials' editor*: Plugin_Shaders' graph UI is
  `plugins/vendor/materials/graph_editor` (`materials_graph_editor`),
  provided to the editor by `pulsar.materials` through
  `plugin_manager::PluginEditorProvider` (origin-gated, so a project without
  materials has no graph editor); the submodule and `ui_core`'s link are
  gone. **Plugin_Shaders is superseded** by `pulsar.materials` once these
  stacks land (nothing was deleted or renamed).
- *Untraced geometry is declared, not named*: the meshes' ray-tracing host
  refused `foliage_layers`/`voxel_volumes` by name. A feature now declares
  the geometry it draws that the scene's ray tracing does not see
  (`FeatureRegistrar::untraced_geometry`), the compiled graph carries the
  list, and a host reads `Renderer::untraced_geometry()`; foliage and VG
  declare theirs.
- *Fog left the post settings*: Helio's post-process ABI carried a 64-byte
  fog block (`GpuFogUniforms`, `FogMode`, override bits 40..=49) that the
  fog pass read back from post-process volume rows, a second fog model
  beside the fog plugin's media. The block is gone (settings 464 bytes, lens
  at 368; bits 40..=49 retired, `PostProcessProperty::RETIRED`, so saved
  masks keep their meaning); the fog pass reads only global and local media
  rows (no post-process bindings, no `fog_components`); the fog plugin owns
  `FogMode` and requires nothing, and the shader's medium modes are the fog
  row's `fog_mode` constants. A bounded fog pocket is a local fog volume
  (the cathedral example moved). How others do it: Unreal's Exponential
  Height Fog and Local Fog Volumes are their own components, and its post
  process volume carries no fog medium; Unity HDRP's Fog is a volume
  override, one model owned by the fog. Ours is Unreal's: the post plugin
  grades a frame, the fog plugin owns every medium, and neither reads the
  other's rows.

- *Material instances are entities; the material table is theirs*: the
  scene's `"materials"` rows were the meshes' join output (every section of
  every instance embedded a resolved `GpuMaterial`, one row per section,
  nothing shared). A material instance is now an entity carrying
  `helio_mats::instance::MaterialInstanceComponent`: its source (a material
  asset, or an explicit surface) and the surface it falls back to (loading,
  failed, materials plugin disabled). SceneDB mirrors each instance into the
  packed `"materials"` row (`MaterialComponent`) and a `"material_keys"` row
  (shading class, graph hash) at its entity row; its asset is declared to the
  handle ledger, loaded by the materials plugin's loader and bound into the
  instance, so a reload rewrites one row. Users acquire instances by source
  (`acquire`/`release`, interned per world: equal sources share one row).
  The meshes keep a scene subsystem (`MeshMaterialInstances`, change cursors
  on static and skeletal meshes) that gives each slot its instance and the
  mesh's entity a `MeshMaterialRows` (its slots' table rows, a var-len GPU
  row). A section row is `{first, count, material_slot}` (16 bytes, was 128);
  the join reads `rows[slot]` through the mesh's handle, copies it into the
  object row and reads class and hash from the key row (a class change
  regroups draws), writing no material data. Assigning or reloading a
  material writes that small row, never the mesh: a first version wrote the
  instance rows into the mesh component, and every guarded mesh write
  re-dispatches the mesh's geometry and draw rows (about 1.3 ms a mesh at 2k
  meshes, superlinear), which stalled a 50k-mesh scene for minutes. Voxel faces
  that show scene materials (`VoxelMaterialMap`) index the same rows; the
  voxel palette (per-voxel colour, patches, flecks) stays the voxel's own
  appearance data. How others do it: Unreal keeps one uniform buffer per
  material proxy that components reference; Bevy prepares one bind group per
  material asset handle. Ours is Bevy's sharing with SceneDB entities as the
  handles, so drawers index one table by entity row, the table is written
  once per material, and a project without the materials plugin still draws
  every slot with its fallback surface.
- *A kind places its own rows*: the row join's shader carried every kind's
  placement (a decal's box transform, an emitter's pool range, water's sun,
  wind and hitboxes, a sprite's 2D placement, a foliage layer's square, an
  atmosphere's centre), selected by flag bits Helio named. The join now knows
  only what kinds share (attachment, visibility, gates, slices, packing, the
  owner's world box); a kind's placement is its own WGSL hooks
  (`helio_scene_join::Placement`: `placement_placed`, `placement_write_row`,
  `placement_source_header`, `placement_write`, `placement_request`,
  `placement_range`), compiled into a pipeline per placement and kept in the
  kind's feature crate. How others do it: Unreal's render proxies place
  themselves (`CreateSceneProxy`), each primitive type owning its upload;
  ours keeps the one GPU join but lets the owner write its placement, so a
  new kind needs no change to Helio.
- *The G-buffer output has one owner*: the meshes, portals, foliage and
  voxel drawers each declared the eight-target fragment struct. The frame
  api now owns it (`helio_frame_api::gbuffer::OUTPUT_SNIPPET`:
  `GBufferOutput`, and `GBufferDepthOutput` for a drawer that writes depth)
  beside `TARGETS`, so the layout changes in one place.

- *Object types are structural*: `ObjectType` named kinds (`Light(LightType)`,
  `Mesh(MeshType)`, `ParticleSystem`, `AudioSource`) in the scene model, the
  level format, picking, the hierarchy and the AI tools. It is now `Empty`,
  `Folder`, `Camera`, `Blueprint`; what an object draws or lights is its
  components. Levels that named a kind load those objects as `Empty`.
  Picking picks by an object's editor icon or its content's geometry
  (`ContentGeometryRegistration`), never by a box for its type; the
  hierarchy shows the icon a kind registers for objects its components make
  (`ui_level_editor::extend::ObjectIconRegistration`: the lights' bulb, the
  meshes' box); the AI tools' kinds are the structural ones (a lamp is an
  object with a `LightComponent`). `SetMovability` was already generic over a
  `movability` property. How others do it: Unity's GameObject has no type
  (its icon follows its components); Unreal's actor classes are the kind's
  module's. Ours is Unity's, so the scene model names no kind.
- *Every plugin describes itself*: the pre-rewrite plugins (asset viewer,
  code editor, table editor, Matter, animation, physics, UI, input, render
  graph, Blueprints) have `pulsar-plugin.toml` descriptors and submit their
  editor providers from their own crates
  (`plugin_manager::PluginEditorProvider`, origin-gated), so a project that
  disables one has no such editor or file type. `ui_core` names none of them
  (it registers only the Level Editor), and the engine binary links them
  through the installed set. A descriptor may list no runtime module (an
  editor tool, which a game never links); its first module carries it
  (`PluginDescriptor::installing_module`).

- *The animator leaves the skinned mesh*: `SkeletalMeshComponent` was the
  player too (graph, Montages, notifies, root motion, the physics blend). It
  is now the skinned mesh only: its asset, skin stream, inverse binds and
  joint palette, showing the pose it is given
  (`SkeletalMeshComponent::show_pose`) and its bind pose without one. The
  animation plugin's runtime module `animation_component` owns
  `AnimationComponent` (moved as-is: the instance, Montages, notifies, root
  motion, the physics blend, the animation phase and its tests, and the
  skeleton, sequence and graph loaders). The owner's `n`-th animator drives
  its `n`-th skinned mesh: it plays on the skeleton the mesh's header names
  and shows its drawn pose there each time it changes. The two meet in a
  core contract (`helio_component::skinning`): `SkinnedMeshAccess` (the
  meshes': find the mesh, read its binding, show a pose) and
  `SkeletalPoseAccess` (the animator's: the pose a physics rig reads and the
  bones it drives). `helio_mesh_api::skinning` is gone (an api crate may not
  collect registrations). Saved levels and class prefabs migrate on load
  (`animation_component::records::split_animator`, now also run on prefabs
  by `PrefabAsset::load_from_dir`); Blueprint event nodes are the
  animator's, and graphs saved with the mesh's class still compile and
  show their titles. How others do it: Unity's Animator writes the pose a
  SkinnedMeshRenderer skins with its bind poses; Unreal keeps the
  `UAnimInstance` inside `USkeletalMeshComponent`. Ours is Unity's split,
  so a skinned mesh needs no animation plugin and an animator drives any
  skinned mesh through the contract. Behaviour is unchanged (the user,
  2026-11-06: animation is structural only in p17); dead pieces went with
  the move (`SKELETAL_PREVIOUS_BONE_PALETTE_BUFFER`,
  `register_loaded_animation_notifies`).

**Known issues, structural only in p17** (the user, 2026-11-06: virtual
geometry, animation and portals/water are not in their final form; p17 puts
each in its owner's place behind generic seams and does not fix their
behaviour, and a move must not make them worse than before):
- *Dense VG*: `virtual_geometry_dense_acceptance` fails as on p16. The crowd
  is drawn by the hardware raster (tier view blue) with coarse clusters, the
  dense-overlap view is nearly black, and the stats report `dense_tiles` 0
  and `engaged` false even at tau 0.25 and sensitivity 8. Every VG pass runs
  (the late raster classifies); the counters are the software raster's. The
  next look would be the page-residency (107) and error-vs-tau (109) debug
  views, to see whether fine pages never stream in. The suite stays out of
  CI's `ci` profile.
- *Animation*: in the animation plugin as above; its open problems stay
  open.
- *Portals' G-buffer drawer*, *water's reflection capture*: in their own
  plugins as they are; their open rendering problems stay open. Both are
  known exceptions to one owner, waiting on **views** (the next design
  item, below), owned by the portals and water plugins:
  - water's capture graph names the meshes' passes (object batch, indirect
    dispatch, occlusion culling, G-buffer, deferred light) and builds them
    through `PassContext::execute_secondary_graph`: one kind reaching into
    another kind's internals;
  - the portals' `PortalInstancePass` keeps its own mesh G-buffer program
    (`gbuffer_portal.wgsl`): a second copy of the meshes' drawer.

**Views, the next design item** (a new capability, so not p17's: it changes
what reflections and portals show). What it is:
- The preset composes a view graph from the frame's members, as it composes
  the primary graph: a capture asks for phases and slots (scene prep,
  visibility, the G-buffer phase, geometry, lighting) and gets every feature
  registered there, not a pass list it names.
- Per-view mirroring in every drawer: a mirrored view (a planar reflection)
  flips winding for each drawer through the view's description, instead of
  one pass's `set_mirrored_view`.
- Remapped, clip-masked culling and draws: a portal publishes a view (the
  chain's composed transform and its clip mask); the meshes cull and draw it
  with their own program, and the portals' second program goes.
- What it adds: the foliage, decal and virtual-geometry drawers appear in
  reflections and through portals (today only the meshes do), at their cost
  in each view.
- `one_owner`'s list has no entry for either: the check finds WGSL copies
  of snippets and key literals, and neither copy is one (water's is a Rust
  dependency; the portal program reads the meshes' rows through their
  snippets). This note is their record until views land.

**Not done yet** (p17 continues on the same branches; nothing here is
handed off): nothing of p17's design list. Dense VG acceptance is a known
issue (above); views are the next design item, not p17's.

**After the split** (deferred from p17, one issue each; repo, what, waits on):
- Helio + Pulsar-Native (meshes, portals, water): **views**: the preset
  composes a view graph from phase and slot members, every drawer honours
  per-view mirroring, and the meshes cull and draw remapped, clip-masked
  views; removes water's capture graph naming the meshes' passes and the
  portals' copy of the mesh G-buffer program. Waits on: its design (a new
  capability that changes what reflections and portals show).
- Pulsar-Native (meshes, VG): **dense VG acceptance**:
  `virtual_geometry_dense_acceptance` fails as on p16 (hardware raster, no
  dense tiles). Waits on: VG's final structure.
- Pulsar-Native (meshes, VG): **VG's final structure**: virtual geometry is
  in its owner but not in its final form (the user, 2026-11-06). Waits on:
  the user's VG rework.
- Plugin_SKM: **animation runtime behaviour**: the animator moved as it was;
  its open problems stay open. Waits on: the user's animation work.
- Plugin_SKM: **several meshes per animator**: the owner's `n`-th animator
  drives its `n`-th skinned mesh; a modular character (one animator, several
  meshes on one skeleton, Unreal's leader pose) needs the animator to drive
  every mesh on its skeleton. Waits on: the animation runtime work.
- Plugin_SKM + Pulsar-Native (engine_backend): **graph dependency reload**:
  `reload_animation_dependencies` has no caller, so editing a sequence a
  running graph uses does not recompile it. Waits on: an asset-change hook
  for dependency closures.
- Plugin_Physics: **ragdolls need an animator**: the rig drives the pose
  through `SkeletalPoseAccess`, so a skinned mesh with no
  `AnimationComponent` has no pose to drive. Waits on: the animation
  runtime work (a pose without an animator).
- Pulsar-Native (voxel, sky): **voxel and sky test failures**: `clouds_cost`
  (perf) fails as before p17 (it waits on the sky split); the chunks suite
  needs `--release`. `foliage_on_voxels` is fixed (Plugin_Voxel PR #3).
  `voxel_pass_graph`'s chunk-pass settle passes serially (9/9 with
  `--test-threads=1`, `chunk_pass_builds_settles_and_resizes_in_the_deferred_graph`
  settling in 112 frames; the old "settled still view asked for frame 0"
  failure did not reproduce, #1627); the remaining failure mode is the
  parallel crash on concurrent wgpu device creation (#1647). Waits on: the
  sky split, and #1647.
- Pulsar-Native (engine_fs and its users): **one asset-reference
  resolver**: texture, skeleton, image and skeletal-mesh references each
  resolve through the file index in their own crate. Waits on: asset
  categories as registered kinds. Still open after p20 (Pulsar-Native#1605
  folded it in; its plan's section 12 says why).
- Pulsar-Native (ui_types_common, pulsar_content, engine_fs): **asset
  categories** (`AssetKind::Mesh`/`Material`) are file vocabulary rather
  than registered kinds. Still open after p20 (most of Pulsar's kind-name
  budget).
- Plugin_Blueprints: **one prefab model**: the Blueprint editor keeps its
  own `PrefabAsset` beside `pulsar_class::prefab::PrefabAsset` (and reads
  `prefab.json` raw for animation events). Waits on: the Blueprint plugin
  adopting `pulsar_class`'s model.
- Pulsar-Native (sky): **night-sky panoramas as assets**: the sky
  feature's build script downloads 4K deep-space HDR panoramas and embeds
  them (about 25 MB in every binary that links the sky, voxel_craft 49 to
  76 MB), so builds need the network and games carry them unasked; they
  should be project assets the sky component names. Waits on: the user's
  sky work.
- Pulsar-Native (pulsar_game, CI): **generated game links on Linux**:
  `generated_project_compiles` (`--ignored`, CI only) fails as on p16 with
  an undefined `__gxx_personality_v0` (a C++ dependency linked without its
  runtime). Waits on: finding the C++ dependency and its link flags.
- Plugin_Shaders: **superseded**: the shader graph editor is
  `pulsar.materials`' graph editor and compiles through the materials'
  compiler. Waits on: the user's decision on the repo (archive or keep).

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

As built, p18 (2026-11-07, branch `rewrite/p18-repo-split` in Helio,
Pulsar-Native and the plugin repositories that changed): the repo split and
Helio as core only.

**The split.** Sixteen public repositories, `Far-Beyond-Pulsar/Plugin_<Kind>`
(Materials, Meshes, Lights, Sky, GI, Fog, Post, Lens, Water, Foliage, Voxel,
Decals, Particles, Sprites, Portals, Splines; the table in section 3 links
each), hold what was `plugins/vendor/<kind>/` in Pulsar-Native, with its
history (`git subtree split` of Pulsar-Native 838f6acca; the plugins moved
into Pulsar-Native from 2026-11-02, and their earlier history is Helio's).
Each comes back as a submodule at the same path, so workspace member paths,
`[workspace.dependencies]`, the one-owner check, the installed-set
generator, `absent_plugins`, the `scene_inventory` ledger and the game
builder (all of which read the checked-out tree) are unchanged. Like
Plugin_Physics, a plugin repository builds only inside a Pulsar-Native
checkout (`workspace = true` dependencies; Helio from Pulsar's submodule).
Each has a README (what it owns: its crates, runtime and editor modules,
which other kinds' api crates it reads), `AGENTS.md` pointing at the
`helio-architecture` skill (linked, never copied), `CLAUDE.md` and CI that
checks it inside a Pulsar-Native checkout (rustfmt, which Pulsar's CI does
not run on submodules; clippy; nextest's `ci` profile). In Pulsar-Native:
- the root manifest names every crate inside a plugin repository by path:
  the `voxel/*` glob matched the new repository's `.github`, and a glob into
  a repository enrolls whatever directory it adds. CI's vendor-member guard
  checks that rule for every plugin submodule instead of listing seven crate
  names;
- `.agents/CRATES.md` has the plugin table (path, repository, plugin) and
  says how plugin repositories are checked and pinned; `AGENTS.md` says to
  clone recursively.

The asset viewer, the render graph editor (`render_bps`), `plugins/installed`
and `plugins/trait_editor` stay in-tree (decided in p20: the editor-tool
plugins stay plugins of their own).

**Helio is core only: the gate.** `scripts/architecture/check_layers.py`
(`just arch`) fails when a crate outside the test host (`test_host = true`
in `architecture/layers.toml`: `helio-components`, `examples`) names an
object kind. `[kinds]` lists each kind's words; the check reads every
`.rs`/`.wgsl` file's identifiers and key-like string literals (comments,
prose strings and tests left out) and splits them into words, so
`MESH_BOUNDS_ROW_BYTES` and `VolumetricFogPass` both count. Words the frame
owns are not kinds' ("light": the light list and directional lights are
lighting inputs; "sky", "shadow", "reflection"). What still names a kind is
recorded per file and kind in `architecture/kind-names.txt` with what it
waits on; the list may only shrink, and a line without a reason is an
error. How others hold the line: Bevy keeps kinds out of
`bevy_core_pipeline` by crate structure alone, and Unreal's renderer module
names every primitive type. A dependency gate cannot see Helio naming a
kind (Helio never depends on a plugin), so this gate reads the names
themselves, which also catches a key literal or a WGSL copy that a plugin's
rename would leave behind.

**Moved or deleted to pass it** (each was kind code in a core crate):
- *Shared resources swap together by their type* (`SwapPolicy::shares`,
  `FeatureRegistrar::swap_shares::<R, P>()`): the meshes' shadow drawer named
  the lights' `ShadowMatrixPass` in its hot-reload swap group, because both
  hold the lights' dirty flags. Now each declares its own pass on the shared
  handle (`LightDirtyFlags`, the lights' api type), and the passes on one
  handle form one group, so neither names the other's pass. The meshes'
  shadow and VG-shade crates and the voxel feature no longer depend on
  `helio-pass-shadow-matrix` (two of those were dead dependencies), and the
  voxel component's sky and fog feature crates are test-only dependencies.
- *The scene join's owner bounds*: `MESH_BOUNDS_ROW_BYTES`,
  `Placement::mesh_bounded`, `RowJoinKeys::mesh_bounds` and the shader's
  `mesh_bounds` are `OWNER_BOUNDS_ROW_BYTES`, `owner_bounded`,
  `owner_bounds`: the bounding spheres of content attached to owners,
  whichever kind publishes them (the engine names the key, as Pulsar's
  `object_joins().owner_bounds` already did; the meshes publish them today).
- *Post-process volumes*: `GpuPostProcessVolume`,
  `PostProcessVolumeDescriptor`, `PostProcessBlender`, `volume_weight` and
  `MAX_PP_VOLUMES` moved from the tone map's crate to the post plugin's
  feature crate (`helio-pass-post-volumes`); the post-settings ABI and its
  override bits stay the frame's contract.
- *Texture means* (the mean colour of a texture, for shading ray hits) moved
  from `helio-scene-derivations` to the meshes' ray-tracing host, its one
  user. `helio-scene-derivations` is the hierarchy resolve only, in the
  scene layer (which may now use itself: it uses the scene join's blocks),
  without its `helio` and `helio-frame-api` dependencies.
- *The facade*: the wasm C++ allocator bridge moved to `helio-vg-build`,
  the crate that links meshoptimizer (the facade linked `meshopt` and used
  none of it). The unused `helio-bake-types` dependency, the dead
  `shadow_atlas_size`/`shadow_face_capacity` settings and the stale MeshPool
  comment are gone. `VIRTUAL_GEOMETRY_INT64_FEATURES` is
  `INT64_ATOMIC_FEATURES` (a capability, not a kind).
- *helio-core*: the TLAS input's `mesh_id` is `geometry_id`.
  `FocusedTimingGroup` (a profiling view that classified fog, HLFS and shadow
  passes by name, used by nothing) is deleted.
- *Lighting-input key values*: `ENVIRONMENT_PROBES`/`ENVIRONMENT_CUBES` were
  `"reflection_captures"`/`"baked_reflection"`, the lights' and a baker's
  names for a frame contract. They are `"environment_probes"` and
  `"environment_cubes"` (the lights' row attribute and the ledger follow).
- `helio-controls`' unused `PerspectiveLens` is gone, and SSR's local named
  `material` (an ORM texel) is renamed.
- *`helio-pass-sdf` is deleted* (the user, 2026-11-07: "old"): an SDF
  terrain kind with its own sky, sun and fog, superseded by `pulsar.voxel`.
  Its one user, the meshes' web `sdf_demo`, went with it, and so did its
  workspace entries and its ledger row.
- *The simple cube moved to the test host*: `helio-pass-simple-cube` (a
  spinning cube drawn straight into the output, to check a device and
  surface) was a feature in every renderer's default set, behind
  `Preset::Simple` and the built-in pipeline `helio.simple`. Nothing in
  Pulsar used either; Helio's `simple_graph` example did. The pass is now
  that example's own module (`crates/examples/simple_cube.rs`, which builds
  its one-pass graph itself); the crate, the preset, the pipeline and its
  tests (which mirrored the private vertex struct and checked arithmetic)
  are gone.

**Helio passes its own tests from its own clone**: `wgsl_validation` failed
on four shaders on main (since p17's 2026-11-06 commits). The test composed
shaders with its own snippet list and its own per-file rules, and those
commits changed how four shaders are built (the frame's
`indirect_specular` snippet in SSR's compose, the row join's placement
hooks, the perf overlay's offered surface probe, the LUT build's
post-settings prefix); a `//!use` marker nobody provides passes
`resolve_with` silently and fails later as an unknown identifier. Now each
crate that composes shaders offers `validation_sources()` (the file's path
and its source, built by the same function its pass compiles with:
`lut_build_source`, `profile_source`, `row_join_source`), the test takes
those instead of re-implementing them, lists every snippet the repo
declares, and fails on any marker its list does not provide.

**The rest of `crates/passes/3d`, judged**: FXAA, SSR (a Hi-Z march with
the frame's TLAS for misses), the buffer view, the debug overlay, Hi-Z, TSR
and the tone map are the frame's or diagnostics, and name no kind. Planar
reflection names none either: it is a screen-space trace against one plane
given by `PlanarReflectionSettings`, writing an indirect-specular layer. Its
plane is content (a mirror, a water surface) that should come from whatever
shows it, and a true planar capture is a mirrored view: it waits on **views**
(Helio#390), not on a move.

**Recorded, with what each waits on** (`kind-names.txt`; each is a design
decision of its own, not a move):
- *Baked lighting* (`helio-bake`, the vendored nebula baker,
  `helio-bake-types`' `BAKED_LIGHTMAP` and PVS, and the G-buffer's
  lightmap-UV target) has no owner yet. Giving it one means a new plugin
  (a repository this phase did not create) or a split between the lights
  and the meshes: the user's call. Its AO and irradiance-SH outputs have no
  reader.
- *The sky lighting input* carries the sky plugin's cloud and star
  parameters in the frame's `AtmosphereParams`. Waits on splitting the
  contract into the frame's medium, sky table and suns, and the sky's own
  parameters.
- *The perf overlay's shading-cost profiler* models per-material light
  loops. Waits on its cost model coming from the materials' `SurfaceProbe`
  offer.
- *The lens block of the post settings* (bytes 368..464, override bits
  63..83, the tone map's lens input). Waits on the lens settings leaving the
  post-settings contract, as fog's did in p17 (an ABI change, saved masks,
  and whether volumes may blend a lens).

As built, p19 (2026-11-09, branch `rewrite/p19-plugin-catalog` in Helio,
Pulsar-Native and the plugin repos; phases 0 and 1 of the plugin API plan,
Pulsar-Native#1608, `.agents/PLUGIN_API_REFACTOR_PLAN.md` there):

- **Feature-to-node ownership.** `FeatureRegistrar::node` records its
  feature as the node kind's owner; `NodeRegistry::feature_of(kind)` and
  `feature_nodes()` read it. A node registered directly (the core nodes)
  has no owner; a later feature replacing a kind owns it. This is the only
  Helio change: the graph, presets and pass lists are unchanged, and
  `FeatureSet` stays the aggregate graphs build from.
- **Each plugin has one root crate** (Pulsar side): the package at the
  plugin's directory (`<kind>_plugin`), the one place it submits what it
  contributes, its editor registrations behind an `editor` feature. The
  descriptor names it (`root = "<kind>_plugin"`), every registration's
  origin is it, and the installed set and generated games link only roots.
  The editor crates are `<kind>_editor`; crates the move left empty are
  gone (GI's component and editor crates: its root holds the feature set
  and its configuration; the decals', particles' and lens's editor
  crates). The plugin table above names each plugin's crates. A plugin's
  component class is `#[world_component]` at its type and registered by
  the root (`submit_world_component!`); SceneDB GPU-mirror hooks and
  reflection tables stay with their type (no origin, inert without the
  class); api crates hold no registrations.
- **The catalog** (`plugin_manager::catalog`): every registration type
  summarizes itself (`pulsar_plugins::Contribution`), and a render feature
  registration's summary lists each feature with the nodes it registered,
  read from `feature_nodes()`. The catalog's proof checks those node ids
  against the engine's graph registry (`helio_presets::graphs::
  registry_with(plugin_features())`) and the owning feature of each.

As built, p20 (2026-11-09, branch `rewrite/p20-kernel` in Helio,
Pulsar-Native and the plugin repos it touched; Pulsar-Native#1605, the
kernel: section 12 of `.agents/PLUGIN_API_REFACTOR_PLAN.md` there):

- **Helio's rule one level up.** Pulsar's core and editor know no kind,
  subsystem or tool. Pulsar runs this repository's layer checker over its
  own `architecture/layers.toml` (`--root`): the checker gained directory
  layers (`[paths]`), dev-dependency checking, `[scan]` directories,
  `#[path]` test modules and `--update` dropping kind names that are gone.
  Helio's own run is unchanged.
- **The texture table's capacity is the frame's.** `helio_core::
  texture_table_capacity` and `texture_table_uses_binding_arrays` (with
  `TEXTURE_TABLE_RESERVE`) decide how many textures the table holds on a
  device; the materials' binding config and Pulsar's scene texture store
  both read it, so the engine needs no kind's api for its store.
- **Texture mips are core** (`helio_core::texture_mips`, from the
  materials' api): colour, data and normal-map mip filtering is texture
  data, cooked by the engine for every kind.
- **Baked lighting's owner, decided**: a kind of its own, its own plugin
  (like GI), with the G-buffer's lightmap-UV target leaving the frame through
  a `GBufferContributor`. `kind-names.txt` says so on each baked-lighting
  line; the move needs that plugin's repository and the frame change, and no
  engine path bakes today (only the cathedral examples).
- **Sublevel placements** (`SubLevelActorRow`, `SubLevelIndex`) left the
  meshes' api for Pulsar's scene model (they define spaces, and no pass
  reads them on the GPU); the portals' resolver reads them there.
- Pulsar side: physics is the physics plugin's (its api crate is the
  `.physasset` schema), the input plugin joins the game through a session
  hook, the optional editor tools register through `ui_common::
  editor_tools`, each plugin owns its closure-ledger rows, and the four
  editor-tool plugins are workspace members.

As built, VG forest (2026-11-10, branch `feat/vg-forest-demo` in
Plugin_Meshes; the meshes plugin only, no Helio code change; the lighting
fix that came out of its measurements is under layered-rewrite.md, "No
per-invocation table copies"):

- **The forest scene** of `vg_stress` (`--scene forest`): a procedural
  terrain of classic tile meshes and up to 30,000 Poly Haven trees through
  VG, with the sky's and GI's features added to the example's preset
  (`helio_mesh_host::presets::with`). Described in the plugin's
  `examples/VG_FOREST.md`.
- **Sectioned tree assets** (example loader): a tree variant's parts become
  one VG asset with one section per material slot when they share a node
  transform; an instance takes one consecutive object row per section. A
  merged asset whose root exceeds the pinned root budget (`RootTooLarge`) is
  cooked as separate parts: the builder simplifies per material.
- **Monotone feedback bits** (`helio-vg-stream`): the request, used and
  urgent bits only accumulate between ordered readback/reset commands, so the
  shader helpers read the bit and skip `atomicOr` when it is set; racing first
  requesters still elect one request-list append from `atomicOr`'s old value.
  The cull's history and hysteresis marks (`helio-pass-vg-cull`) do the same.
- **Cull traversal** (`helio-pass-vg-cull`): refinement below the hold
  threshold skips the history lookup; stats are summed per thread and flushed
  once; a group's clusters are pushed as workgroup-stack jobs when the group
  fits; a hierarchy node stops at `tau * TAU_HOLD * tile_scale_min`, the
  smallest valve scale over its tiles (coverage-complete: no cluster below
  can have a smaller scale). The valve defaults are now 8 triangles per
  covered pixel and a 32x ceiling (were 16 and 4x).
- **One persistent dispatch per phase** (`helio-pass-vg-cull`): workgroup
  deques, a phase queue for sharing, and an outstanding-work counter for
  termination replace the fixed rounds, whose last round drained whole
  subtrees on a few workgroups. Last frame's marks and the valve scales are
  read from a read-only snapshot (cached loads, not L2 atomics); bin slots
  and tile estimates are reserved once per workgroup wave.
- **Capture host**: keeps up to three frames in flight through
  `Renderer::last_submission` instead of draining the GPU every frame.
- Unchanged: buffer contracts, ownership, page residency and eviction, the
  render graph, the geometric error tolerance.
- Not yet measured as a whole: the persistent traversal with the sectioned
  trees (see `VG_FOREST.md`, "Not yet re-measured").

As built, p22 (2026-11-11, branch `rewrite/p22-plugin-contracts` in Helio,
Pulsar-Native and the plugin repositories; Pulsar-Native#1608 phases 2-5 and
T5, recorded in Pulsar's `.agents/PLUGIN_API_REFACTOR_PLAN.md` section 14):

- **Buffer contracts link to their provider.** Every plugin feature declares
  each scene buffer its passes read with `FeatureRegistrar::consumes` (the
  owner's `BufferContract` const; a repeat is recorded once), and each plugin
  descriptor lists the `buffer` contracts it provides by key. Pulsar's plugin
  catalog links every read to the plugin whose contract covers the key, or
  to the frame: `helio_frame_api::FRAME_BUFFERS` lists the buffers the frame
  defines (entity generations, the camera, environment probes, view remaps).
  New contract consts where a key had only a string (in the plugins' api
  crates): the lights' directional light rows, the meshes' skin, bone
  palette, VG instance, suppression and object structure/moved buffers, the
  portal chain's two.
- Helio's own gate is unchanged (0 violations, 31 recorded kind names).

As built, plugin nodes in graph assets (2026-11-11, Pulsar-Native; Helio side:
`helio-feature-api`, `helio-graph-asset`):

- **A plugin's passes are nodes already, through its feature.** A plugin's root
  submits `RenderFeatureRegistration`; the feature's `register` adds each pass
  as a `NodeDescriptor` (`registrar.node`, kind `helio.pass.<name>`, an `enabled`
  parameter, generated pins, a frame slot or draw-phase placement). The engine's
  node registry is the core nodes, Helio's features and every *enabled*
  plugin's features; the probe derives pins from what the passes declare. There
  is no second registration path: the raw `register_graph_node` was removed.
- **The probe builds nodes with the plugins' settings** (`render_features::
  configure`). A node that builds nothing without its feature's settings (the
  voxel terrain's chunk pass) was probed as no pass and had no pins.
- **A disabled plugin is absent, with the reason.** `NodeRegistry::mark_absent`
  / `absence` (kind-agnostic: a host says why a kind is not registered);
  validation reports `missing node `<kind>` (plugin <id> disabled)` instead of
  `unknown node kind`. The engine marks the nodes of every disabled plugin's
  features.
- Helio still names no kind and no plugin: only a reason string.

As built, routing validation (2026-11-11, Pulsar-Native; Helio side: `helio-core`
`graph/routing.rs`, `helio-graph-asset` `routing.rs`):

- **One check, two front ends.** `helio_core::check_routing` is pure over
  `RoutingUnit`s (reads, writes, `writes_target`) and the host inputs. It lives in
  core because the renderer's guard needs it and `helio-graph-asset` sits above
  `helio`. `helio_graph_asset::routing` derives units from a flattened asset's pins
  and `validate` reports them; `RenderGraph::routing_issues` derives them from
  `PassDeclaration`s and `Renderer::construct` logs them for every graph, legacy
  builders included.
- **The presented target is declared.** `RenderPass::writes_target` (default
  false) is on FXAA, TSR (unless intermediate) and PostProcess; it is carried in
  `PassDeclaration`. No pass writing it is the hard error (the frame stays black).
- Findings and rules: `.agents/RENDER_GRAPH.md`, "Routing validation".

As built, the default graph as data (2026-11-11, `helio-presets`):

- **No second builder to compare with.** The `add_pass` builders were already gone;
  `helio.deferred` is `Preset::Deferred` (slots and phases). `helio.deferred_graph` was the
  one that differed: a fixed FXAA asset (no TSR), and it never set the lighting model's
  settings the preset inserted in code (`PhaseList`, `LightingModelId`, specular layers,
  lighting-input order, `ReactiveMask`). Both are fixed.
- **`graph_asset(preset, registry, config)`** takes `Preset::nodes` instances with their
  parameters (the TSR node and its quality). `deferred_graph_asset_for(registry, config)`;
  `helio.deferred_graph` is a made pipeline over it.
- **`helio.frame.setup`** (Frame: Lighting Model, `features::FRAME_SETUP`) leads every
  preset's chain and publishes those settings; `registry_for` snapshots the registered
  lighting models into it. The pipeline closure inserts nothing now.
- **Plugin nodes need no new seam.** The existing `NodeDescriptor::at` / `FrameNode::in_slot` /
  `drawer` placement (an extension point plus order) already puts a plugin node in the
  default graph; `graph_asset` places and `sectionize` lanes it by its category
  (test: `a_node_placed_in_a_slot_joins_the_default_graph_and_a_section`).
- **Golden `.rendergraph` files** (`helio-presets/assets/`, FXAA and TSR) with a drift test
  and an ignored regenerate test; `graph_parity` builds them and compares the pass lists
  with the preset. A default-graphs plugin can ship them.
- Not done: the editor's wired (Get/Set) form on disk, a debug overlay node, deleting the
  preset path as the engine default.
