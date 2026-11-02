# Helio layered rewrite: Pulsar plugin → (components + Helio features) → passes

Status: proposal, 2026-10-31. Measured against Helio `main` (3ea02d06),
Pulsar-Native `main` (edfbcf35a) and `plugin-api-v2-phase1` (9363348e2,
not merged), SceneDB `main` (30ab68d; Helio and Pulsar pin 017252c).

Goal: every change touches one feature and the contracts it uses, never the
whole tree. A voxel change builds and tests voxel, not the renderer, the
editor and every pass crate.

---

## 1. Where we are (measured)

### Helio

| Fact | Number |
|---|---|
| Workspace crates | 70 (54 pass crates) |
| Pass code | 177k Rust + 47k WGSL, one flat layer |
| `helio-pass-voxel-planet` | 43k Rust + 10k WGSL: a world system shipped as "a pass" |
| SceneDB components defined in Helio | 31 `#[derive(SceneStore)]` structs in 14 pass crates (gbuffer, forward-lit, deferred-light, sky, fog, water, lens, foliage, portals, decal, corona, billboard, postprocess, sprites) |
| `RenderPass` hooks | ~25 (graph, scene, lifecycle, editor, debug, jitter, ray tracing, bundles, transparent chaining) |
| `Renderer` fields | 77 (13 XR, bake, cull readback, frame timing, shader reload, templates…) |
| `build_*_graph*` functions in `helio-default-passes` | 27 (one per combination) |
| Pass crates acting as shared libraries | gbuffer (Mesh/Material/StaticObject/RenderGroup components, DrawSlots, ObjectBatchFrameData: ~20 users), shadow-matrix, sky, gi, postprocess |
| `helio` → specific passes | 13 (TsrQuality, HdrOutputMode, ShadowQuality, FoliageQuality, LightComponent…) |
| Dead crates | underwater, water-caustics, water-surface (23-line stubs); debug, depth-prepass, smaa, ssao (no users) |

Co-change since 2026-10-01: **161 of 408 voxel commits (40%) also changed
other crates**, 36 changed four or more. Most frequent companions: default
graphs (87), examples (48), helio-component (28), helio-core (24),
helio-default-passes (19), helio (19), shadow-matrix, foliage, deferred-light,
TSR, GI, fog, sky (10-14 each). That is the "everything touches everything".

### Pulsar ↔ Helio

- **Repo cycle.** `helio-component` (32k lines) lives in the Helio repo but
  declares `workspace = "../../../../.."`, i.e. Pulsar's root, and
  `include_bytes!`s Pulsar's `assets/`. It depends upward on Pulsar editor
  crates (gpui-ce, ui, ui_common, plugin_editor_api, engine_state…). It cannot
  build from a Helio clone.
- **Helio arrives as a submodule** (`crates/renderer/helio`, pinned one
  commit behind Helio main today) with every crate path-depended. Both repos
  now patch wgpu to the same fork branch (`helio/v30.0.1`).
- **engine_backend depends on 14 pass crates** and downcasts into passes:
  `TsrPass` (history reset), `VolumetricFogPass::set_active`,
  `PostProcessVolumeBlendPass::set_defaults`, `WaterSimPass::sim_texture`,
  `ChunkPass::set_appearance/diagnostics`. `voxel_backend.rs` (2.1k lines)
  names voxel-planet internals 30 times.
- **Path coupling outside Cargo.** `scene_inventory` scans
  `crates/renderer/helio/crates` on disk; a test `#[path]`s into Helio
  examples; `core_project_builder.rs` string-matches a Helio path.
- **Rebuild blast radius.** A helio-core change reaches 87 local packages,
  32 Pulsar workspace members (engine, game, every `ui_*` crate via
  `ui_common → helio-snapshot`, five plugins).
- **SceneDB identity leaks into Helio.** Because Helio defines components,
  every Helio crate must resolve to Pulsar's exact SceneDB/reflection copy
  (Pulsar's `[patch]` `//`-URL trick); helio-wasm already hit a `BufferKey`
  mismatch from a second copy.

### Plugin API v2 (#1532)

- Built on `plugin-api-v2-phase1` (7 commits, not on main): manifests,
  `requires`/`optional`, dependency-ordered loading, `Plugin`/`PluginBuilder`
  registrar as a source layer over the v1 vtable.
- Not built: `p.gpu()`, `p.render()`. No plugin registers a Helio pass or
  node today; `register_graph_node` has zero callers.
- Plugins are linked statically in practice (`builtin`); the DLL path has no
  stable ABI (Rust trait objects, rustc hash must match).
- Planned shape: **Pulsar plugin → render pass** (`p.render().pass_node(..)`),
  with `helio-pass-api` split from helio-core (#1538) and Tier A/B pass moves
  (#1541, #1544), passes moving "with their components".

### SceneDB

Already the right foundation: archetype `World`, `#[derive(SceneStore)]`
schemas, `SceneGpuStore` owns every buffer, published by `BufferKey`, with
`epoch`/`content_generation`/`FlushedRows` for change detection. Helio reads
it only through `SceneBufferProjection` (key → handle, `from_handles`).

---

## 2. Diagnosis

1. **The only unit is a pass; the real units are systems.** GI, shadows, VG,
   water, foliage, voxel are each several passes + glue in
   `helio-default-passes` + settings in `helio` + downcasts in hosts + a
   component in `helio-component`. A feature change spans four layers and
   two repos.
2. **Dependency direction is not enforced.** Passes import passes for types,
   the renderer imports features, presets import the renderer, the bridge
   imports both repos. Nothing stops feature A from knowing feature B.
3. **Components live in the renderer.** SceneDB schemas sit in pass crates
   (base ones in the gbuffer pass), so every culler, shadow and water crate
   imports `helio_pass_gbuffer`, and Helio inherits SceneDB's registry
   identity constraints. Components are SceneDB's data, authored by the
   engine; the renderer only needs the buffers they produce.
4. **Graphs are code.** 27 builders because each combination is a function.
5. **Plugin → pass is one level too low.** A plugin that adds water needs
   several passes, persistent GPU state, settings, debug views and a way to
   talk to terrain. A pass node can express none of that except the pass
   itself; the rest falls back into central crates.

---

## 3. Target architecture

A **Pulsar plugin** registers, side by side: file types, editors, **SceneDB
components**, and **Helio features**. Components produce GPU buffers in
SceneDB. Features (and the passes inside them) consume those buffers by key.
Helio never sees a component.

```
PULSAR PLUGIN  (pulsar.water, pulsar.voxel, pulsar.mesh, pulsar.character …)
  ├─ file types, importers
  ├─ editors, panels
  ├─ components ──────────────► SceneDB  (World, #[derive(SceneStore)], GPU mirror)
  │                                 │ publishes buffers by key: "water.volumes", "mesh.static_objects"
  └─ Helio features ◄───────────────┘ consume buffers by key + row ABI
        └─ passes

HELIO (no components, no SceneStore, no inventory)
  presets         .rendergraph assets + feature lists (data, no code)
  helio           Renderer facade: device, graph, views, FeatureSet (≈15 fields)
  features        sky · gi · fog · water · foliage · vg · lens · decals · portals ·
                  sprites2d · debug · voxel-render      (impl crates, never imported by peers)
  feature APIs    sky-api · water-api · voxel-api …     (buffer contracts, row ABIs, handle traits; no statics)
  spine           visibility · gbuffer · lighting · shadows · TSR/output + extension points
  feature-api     Feature trait, registrar, extension-point traits, settings, debug views
  scene-input     buffer contracts: key + row ABI + version; SceneBuffers projection;
                  materials ABI; ray proxies         (no SceneDB derive, ideally no SceneDB dep)
  graph           render graph, slim RenderPass, views, typed frame blackboard
  gpu             device ctx, FrameUploads, caches, pipelines, shaders, profiling, readback
```

World systems that are not rendering (the voxel generator, edits,
WorldQueries, residency planner, planet maps) are their own crates beside the
renderer. The voxel plugin registers the voxel component, the world system,
and the voxel-render feature.

### Rules (enforced in CI, §6)

1. Dependencies point down only.
2. Helio defines no SceneDB components. It declares **buffer contracts**
   (key, row ABI, version) and reads whatever SceneDB publishes under them.
3. A feature impl crate depends on `gpu`, `graph`, `scene-input`,
   `feature-api`, `spine` extension points, and other features' **`*-api`
   crates only**. Never on another feature's impl.
4. `*-api` and `feature-api` crates contain no `static`, `thread_local!` or
   `inventory::collect!` (same rule as #1536), so they are safe across
   statically linked copies and a future DLL.
5. Hosts never downcast to a pass. They hold a feature handle.
6. Graphs are data. Code registers nodes; it does not assemble pipelines.

### Buffer contracts: the seam between components and features

The consumer defines the format, because the shader reads it. So a
feature's `*-api` crate owns the row ABI; the plugin's component produces
rows in it.

```rust
// water-api (Helio): what the water feature reads
pub const WATER_VOLUMES: BufferContract<WaterVolumeRow> =
    BufferContract::new("water.volumes", /*version*/ 3);

#[repr(C)] #[derive(Pod, Zeroable, Clone, Copy)]
pub struct WaterVolumeRow { pub bounds_min: [f32; 4], pub bounds_max: [f32; 4], /* … */ }
// + the matching WGSL struct, generated or checked by a layout test

// pulsar.water plugin (Pulsar): the authored component, SceneDB's business
#[derive(SceneStore)]
#[gpu(layout = packed, buffer = "water.volumes")]
pub struct WaterVolumeComponent { /* fields laid out as WaterVolumeRow */ }
const _: () = water_api::WATER_VOLUMES.assert_layout::<WaterVolumeComponent>();
```

- The component can match the row directly (as above), map into it through
  SceneDB's upload mapper, or be joined into it by a derivation the plugin
  registers (what `environment_join`/`scene_join` do today with
  helio-component's `*SourceRow`s).
- At bind time the feature checks `BufferHandle.row_bytes` and the contract
  version and reports a named error instead of reading garbage.
- Helio's `scene-input` only needs `(key → wgpu::Buffer, epoch,
  content_generation, flushed rows)`. That is what `SceneBufferProjection`
  already is. If Helio owns that small type and Pulsar fills it from
  `SceneGpuStore`, **Helio stops depending on `pulsar_scenedb` at all**:
  no inventory identity, no shared-resolution `[patch]` tricks, standalone
  Helio builds and tests without Pulsar.
- Base contracts (mesh vertex/index pools, materials, static objects, render
  groups, lights, reflection captures) belong to the spine's API. Their
  components move to Pulsar (a core `pulsar.mesh`/`pulsar.lighting` plugin or
  engine core), together with helio-component's source rows.
- Helio's own examples and tests produce rows without Pulsar components: the
  test host writes contract rows by key (or registers test-only SceneStore
  structs in the example crate, which is allowed: examples are hosts).

### The units

**Pass** (graph layer): one GPU step. `RenderPass` keeps only what the graph
needs: `name`, `declare_resources`/`reads`/`writes`, `declare_pipelines`,
`prepare`, `execute`, `on_resize`. Everything else moves to `Feature`.

**Feature** (Helio): a rendering system that owns its persistent GPU state,
passes, settings and debug views, and declares the buffer contracts it
consumes.

```rust
// helio-feature-api
pub trait Feature: Send + Sync + 'static {
    fn id(&self) -> FeatureId;                       // "helio.water"
    fn requires(&self) -> &[FeatureDep] { &[] }      // other feature ids, by version
    fn register(&self, r: &mut FeatureRegistrar);    // one-time, before the pin probe
    fn instantiate(&self, ctx: &FeatureContext) -> Box<dyn FeatureInstance>;
}

pub trait FeatureInstance: Send + Sync {
    fn prepare(&mut self, frame: &FrameContext) {}           // per frame, before graph prepare
    fn is_member(&mut self, scene: &SceneContent) -> bool;   // present when its buffers have rows
    fn handle(&self) -> Option<&dyn FeatureHandle> { None }  // typed control surface for hosts
}

impl FeatureRegistrar {
    fn consumes<R: Pod>(&mut self, c: BufferContract<R>, need: Need); // Required | Optional
    fn graph_node(&mut self, d: NodeDescriptor);             // nodes for .rendergraph assets
    fn settings(&mut self, s: SettingsSchema);               // helio-settings, per feature
    fn debug_view(&mut self, v: DebugViewDescriptor);
    fn contribute<E: ExtensionPoint>(&mut self, c: E::Contribution);
    fn publish<R: FrameResource>(&mut self, key: FrameKey<R>); // e.g. voxel.surface_height
}
```

**Plugin** (Pulsar): file types + editors + components + features (+ world
systems). `p.components().class(..)` and `p.gpu()` register the SceneDB side
(#1534, #1535); `p.render().feature(WaterFeature)` registers the Helio side and
replaces `p.render().pass_node(..)`; the feature registers its own nodes. The
plugin manifest's `requires` covers both: `pulsar.water` requires
`pulsar.voxel` only if its component or editor needs voxel; its Helio feature
reads terrain through the spine's `SurfaceQuery`, not through the voxel
plugin.

### Extension points (the spine's public surface)

The spine is where features meet. Instead of water importing deferred-light
and voxel-planet, both sides talk to a spine trait:

| Extension point | Contributors today |
|---|---|
| `GBufferContributor` (draws into the G-buffer, shares depth/MV/material ABI) | voxel chunks, VG, foliage, water surface, portals, decals |
| `ShadowCaster` (draws into cascades / sun occlusion pages) | meshes, voxel, foliage, VG |
| `RayProxySource` (TLAS instances, GI/ray-query geometry) | meshes, VG proxies, voxel |
| `LightingContributor` (adds radiance / visibility terms) | GI, reflection captures, sky ambient |
| `ParticipatingMedia` (fog, clouds, underwater) | fog, sky/clouds, water |
| `PostEffect` (ordered post stack slots) | lens, flare, exposure, fog composite |
| `SurfaceQuery` (heights, materials, hits for other features) | voxel → water shore, foliage placement |
| `HistoryInvalidation` (TSR/denoiser reset) | anything that teleports or cuts |

Each existing downcast becomes one of these, a handle method or SceneDB data:
TSR reset → `HistoryInvalidation`; fog `set_active` / post-process defaults →
component rows the plugin writes; `ChunkPass::set_appearance/diagnostics` →
`VoxelHandle`; `WaterSimPass::sim_texture` → a debug view.

### Why this shape

- **Unreal** layers RHI → RenderCore (RDG) → Renderer; game-side components
  hand the renderer *scene proxies*, plugins hook in through
  `SceneViewExtension`. Its weakness is the monolithic deferred renderer
  every feature edits; that is our `helio-default-passes` today.
- **Frostbite** (FrameGraph) and **Unity SRP** (`ScriptableRendererFeature`)
  make a *feature* the plugin unit: it owns its passes and injects them at
  named points.
- **Bevy**: components in the ECS, an extract step into render-world data,
  render plugins with graph nodes. Right split of components vs rendering,
  but no layer rules, so render plugins import each other's internals.

Ours takes the Unreal/Bevy split (components are the engine's, the renderer
consumes derived data) with the Frostbite/SRP unit (features own passes),
and drops the extract step: SceneDB already mirrors rows to the GPU, so the
contract is a buffer key and a row ABI. Cost scales with the number of
features, not feature *pairs*, because features meet only at spine
extension points and `*-api` crates, and a renderer that has no components
can't be dragged into gameplay or editor changes.

---

## 4. Repositories

Splitting repos before the contracts exist turns every change into a
multi-repo PR with pin bumps. Split along contracts, after they are stable.

| Repo | Contents | Depends on |
|---|---|---|
| **SceneDB** | as today | wgpu |
| **Helio** | gpu, graph, scene-input, feature-api, spine, helio facade, core features (sky, post, TSR, fog, GI, decals, portals, sprites, debug) + their `*-api` crates, presets, test host. No components. | wgpu fork (SceneDB only if `scene-input` keeps its types) |
| **Helio-Voxel** | voxel-api, voxel-world, voxel-render feature, voxel test suite | Helio API crates |
| **Helio-VG** (later) | vg-format/build/stream + VG feature | Helio API crates |
| **Pulsar-Native** | engine, editor, plugins. **All components** (helio-component's 23 classes and source rows, plus the 31 schemas now in Helio pass crates) live here, split per plugin (#1543) | Helio, Helio-Voxel by git rev/tag, SceneDB |

Mechanics:
- Until phase 6, Pulsar keeps Helio as a submodule, with every Helio crate
  reached through `[workspace.dependencies]` (no crate-level relative paths).
  The submodule gives each session worktree its own Helio checkout, which
  parallel sessions need; a git dependency would make them share one local
  `[patch]` target (Pulsar's `.cargo/config.toml` is tracked, so a per-clone
  patch file has no clean home). Drift is handled by the branch rules: a
  phase lands with Pulsar's pin on the merged Helio `main`.
- At phase 6, with versioned `*-api` crates, Pulsar takes Helio and the split
  repos as **git dependencies on tags**, and co-development uses a `[patch]`
  passed with `cargo --config <file>`.
- One wgpu fork branch, declared once (already unified on main); CI checks
  the two manifests match.
- No crate reaches into another repo by relative path, `#[path]`,
  `include_bytes!` or a directory scan. Engine primitives are Pulsar assets.
- Version the `*-api` crates. A change to an impl crate is a patch release
  nobody downstream rebuilds against; a change to an API crate (a row ABI,
  a handle trait) is the only thing that forces a coordinated bump.

---

## 5. Migration (strangler; every step green, deletes as it goes)

Each phase keeps voxel_craft, voxel_flight and the Pulsar editor rendering
identically (bursts and frames checked before commit).

### Branches

The old landing branches (`voxel-game-example`, `voxel-quick-setup`) are
merged and retired. The rewrite runs on new branches, one per phase:

| Phase | Branch (same name in every repo it touches) | Repos |
|---|---|---|
| 0 | `rewrite/p0-guard-rails` | Helio, Pulsar-Native |
| 1 | `rewrite/p1-buffer-contracts` | Helio, Pulsar-Native (SceneDB only if the projection type changes) |
| 2 | `rewrite/p2-feature-api` | Helio, Pulsar-Native |
| 3 | `rewrite/p3-spine` | Helio, Pulsar-Native (lands with plugin-api-v2's render section) |
| 4 | `rewrite/p4-voxel` | Helio, Pulsar-Native |
| 5 | `rewrite/p5-<feature>` (one per feature group: `p5-vg`, `p5-water`, `p5-sky`…) | Helio, Pulsar-Native |
| 6 | `rewrite/p6-repo-split` | new Helio-Voxel / Helio-VG repos, Helio, Pulsar-Native |

Rules:
- Phases are **stacked**: phase N+1 is cut from phase N's branch, and its PR
  targets phase N's branch, so work continues while phase N waits for
  review. Phase 5 groups stack on phase 4 side by side.
- PRs merge bottom-up. When phase N merges into `main`, phase N+1's PR is
  retargeted to `main` (GitHub does it when N's branch is deleted) and phase
  N+1 merges `main` in, so the next phase always sits on what landed.
- One PR per repo per phase. Session work for a phase goes onto that
  phase's branch, not into side PRs.
- While a phase is open, Pulsar's branch points its Helio submodule at the
  Helio phase branch. When landing: Helio's PR merges first, then Pulsar's
  branch moves its Helio pin to the merged Helio `main` and merges.
- A phase lands only when its "done when" holds, the layer check's
  allow-list shrank, and the preset frames/bursts match.
- `plugin-api-v2-phase1` is merged into Pulsar `main` before
  `rewrite/p3-spine` is cut, so `p.render().feature(..)` builds on it.

### Phase 0: hygiene and guard rails (small, do first)
- Layer manifest `architecture/layers.toml`: every crate → layer (features
  also → group). `scripts/architecture/check_layers.py` (`just arch`, CI)
  fails on new violations; today's 19 are in
  `architecture/layer-violations.txt`, which may only shrink. A budget on
  `#[derive(SceneStore)]` in Helio (31) may only go down.
- Delete the dead crates (3 stubs, 4 unused passes).
- Move `helio-component` into Pulsar-Native (`crates/core/helio_component`);
  drop the 5-level workspace path and the 5-level asset paths.
- Every Pulsar → Helio dependency goes through `[workspace.dependencies]`
  (engine_backend's VG/inspector paths, ui_common's helio-snapshot path).
- `scripts/architecture/co_change.py` (`just co-change`, CI on PRs).
- Deferred: the submodule stays until phase 6 (§4). Pulsar's remaining
  reads of Helio files (scene_inventory's pass/buffer scan, the
  voxel_example_levels test `#[path]` into voxel_flight's worlds,
  core_project_builder's path test) go with the phases that replace what they
  check: the scan's buffer facts become Helio buffer contracts (phase 1), the
  voxel presets become a voxel-api export (phase 4).

### Phase 1: buffer contracts; components leave Helio
What Helio's 31 `SceneStore` structs are today (measured 2026-11-01):
- Helio's runtime code (passes, joins) uses them only as **row layouts**:
  struct values and buffer keys. Component behaviour (World inserts,
  `register_gpu_columns*`, var-len handles) sits in examples, tests, the wasm
  runner, `helio::bake_scene_from_world` (only Helio examples call it) and
  one runtime site, HLFS's `ray_tracing.rs`, which re-reads mesh pools from
  the World.
- Pulsar uses them as row values too: authoring components build them
  (`to_row()`), pack them into its own `*SourceRow` components, and Helio's
  joins write the pass buffers. Pulsar inserts a few directly
  (`SubLevelActorComponent`, portal rows).

So:
1. **Rows and contracts in Helio.** Each struct becomes a plain
   `#[repr(C)] Pod` row named `*Row` (`AtmosphereComponent` →
   `AtmosphereRow`), with no `SceneStore`/`#[gpu]`, plus
   `pub const ATMOSPHERES: BufferContract<AtmosphereRow>` in its crate.
   `BufferContract<R>` (helio-core `scene_input`, moving with it in phase 2)
   is the key and the row size; `resolve()` checks a buffer's `row_bytes`
   against the row and names the contract on mismatch. Var-len data (mesh
   vertices/indices, portal chains) gets `PoolContract<T>`. Rows keep a
   one-line `unsafe impl pulsar_scenedb::Pod` so a component can embed them.
2. **Components where the data is produced.**
   - Pulsar: newtype components over the rows it inserts directly, in
     `helio_component` (`#[repr(transparent)]`, `Deref` to the row, layout
     equal by construction).
   - Helio's standalone hosts (examples, demos, wasm, snapshot, tests): a
     host-layer crate `helio-components` with the same newtypes, plus the
     mesh/portal-chain components and `bake_scene_from_world`. Nothing below
     the host layer may depend on it (the layer gate enforces this).
3. **Runtime code reads contracts only.** HLFS ray tracing builds from the
   static-object rows' own index/vertex ranges instead of re-reading mesh
   components from the World.
4. Unused rows are deleted (`FogComponent`, and any other with no reader).

Done when `scene_store_derives` is 0, no Helio crate below the host layer
depends on `pulsar_scenedb_derive`, and voxel_craft/voxel_flight frames
match. Stretch (may slip to phase 2): Helio owns the projection type and
drops `pulsar_scenedb` below the host layer.

As built (2026-11-02):
- 27 rows + contracts (`ATMOSPHERES`, `SCENE_LIGHTS`, `MATERIALS`, …), mesh
  pools `MESH_VERTICES`/`MESH_INDICES`, `PortalChain` + its pool keys.
  `SectionedObjectComponent` and `FogComponent` deleted (no reader).
  `SubLevelActorRow` is not a contract (no pass reads it): it is the portal
  resolver's CPU input, and hosts store it as a plain component.
- `helio-components` (host layer): one component per contract, the contract
  row's exact fields; `same_layout!` asserts size, alignment, every field
  offset and the key at compile time, and adds `From` both ways, `Deref` to
  the row, the row's `From` conversions and its builders returning the
  component. Also `bake_scene_from_world`, `publish_portal_projection`,
  `upload_scene_materials`, and the HLFS/GI ray-tracing adapter
  (`ray_tracing`, `ray_proxies`), which read host components.
- Pitfall: SceneDB accepts any `Send + Sync` value as a component, so
  inserting a bare row compiles and is never mirrored. Use
  `WorldRows::insert_row`; builders on components return components for the
  same reason. Three examples inserted meadow rows bare, caught by audit.
- Deleted: the hand-written removal tombstones (fog, lens, post-process).
  `#[derive(SceneStore)]` already zeroes a component's row on removal; the
  duplicates cost frame time.
- Verified: every Helio target builds, layer gate clean (0 derives),
  touched crates' tests pass except failures identical on the phase-0
  baseline (8 fog device-limit, clouds cost, foliage on voxels, voxel graph
  settle), voxel_craft A/B frames and run time match the baseline.

### Phase 2: `helio-feature-api` (reordered, 2026-11-02)
The core split (old phase 2) is deferred: it would re-path 1,929
`helio_core::` uses in 276 files, and its payoff (a statics-free crate a
DLL plugin links) only matters once plugins stop being linked statically.
The coupling we feel lives one level up, in how features get into the
graph, so the feature API comes first.

As built:
- `helio-feature-api` (new layer `feature_api`, depends on helio-core only):
  the node types moved from helio-graph-asset (`NodeDescriptor`,
  `NodeRegistry`, pins, params) and `Feature` / `FeatureRegistrar` /
  `FeatureSet`. A feature registers node types and the buffer contracts its
  passes read (`consumes`).
- `BuildEnv` knows no renderer: frame facts (`FrameDesc`: sizes, surface
  format, XR) plus typed **settings** the graph's builder supplies
  (`insert_setting` / `require::<T>()`). The renderer's config, debug-draw
  state and scene handle are settings now (`helio_graph_asset::build_env`);
  a feature reads its own settings type.
- helio-default-passes registers through `default_features()`: one feature
  per stage (scene prep, geometry, lighting, overlays, post, output, host
  inputs, perf markers). `graphs::registry_for(&FeatureSet)` is what an
  engine extends; `base_registry()` is the default set.
- Still to do in later phases: per-feature settings replacing
  `RendererConfig` fields, extension points, feature handles replacing
  `find_pass` downcasts, node registration moving into each feature's crate
  (voxel first, phase 4).

### Phase 3: spine, extension points, thin renderer (#1539 reshaped)
- Feature trait, registrar, extension-point traits, settings per feature
  (helio-settings stays the schema source).
- Spine = today's core deferred set (gbuffer, deferred-light, forward-lit,
  object-batch, light-cull, shadow*, hiz, occlusion-cull, TSR, postprocess
  output) behind extension points.
- `Renderer` keeps device, graph, views, FeatureSet. XR becomes views; bake,
  cull-stats readback and timing become features or graph services.
- Presets become `.rendergraph` assets + feature lists. Delete the 27
  `build_*_graph*` functions and `pipelines.rs` builders (#1539).
- Pulsar: `p.render().feature(..)`; render_bps lists nodes from the registry
  *after* features registered (drop its `OnceLock` snapshot).

As built (2026-11-02): the renderer depends on no feature crate (layer
violations 19 → 9; the 9 left are feature/spine edges for the API crates).
- `FeatureSettings` (feature API): one value per feature's settings type;
  `RendererConfig::features` carries it (the config is `Clone`, no longer
  `Copy`) and `build_env` hands it to nodes. Foliage's five config fields
  are `helio_pass_foliage_place::FoliageSettings` (host settings mapping
  included); VG's runtime id is `helio_vg_stream::VgSettings`
  (`helio_default_passes::with_virtual_geometry`). Engines build their
  config with `helio_default_passes::config_with_host_settings`.
- `Renderer::install_persistent_pass`: a pass the host keeps in every graph
  (made fresh after each rebuild). Baking is a host operation
  (`helio_components::bake`) that installs its inject pass that way; the
  renderer lost `helio-bake` and its `bake` feature.
- Portal host API (`set_projection_frame`, debug texts and geometry) moved
  to `helio_default_passes::portals`; billboard, sky, portal re-exports and
  the unused corona/fog dependencies left the renderer.
- Not done here (next phases): extension points and the sky/GI `*-api`
  crates for the remaining spine/feature edges, presets as assets replacing
  the `build_*_graph` functions, XR as views.


### Phase 4: voxel pilot (the pain we feel)
- Split `helio-pass-voxel-planet` into `voxel-api` (handle trait, buffer
  contracts, published keys like `voxel.surface`), `voxel-world`
  (generator, edits, edit store, WorldQueries, residency, planet maps,
  streams, flora/features data) and `voxel-render` (chunk passes, drawn,
  sun pages; contributes GBuffer/Shadow/RayProxy/SurfaceQuery).
- Water, foliage, sky, GI read voxel through `SurfaceQuery` / published
  keys, never voxel-planet internals.
- The `pulsar.voxel` plugin owns the voxel components (today in
  helio-component), `voxel_backend.rs` and `scene/voxel_*` (#1544), and
  registers voxel-world + voxel-render.
- Test host (§6) with a voxel suite that needs only spine + voxel.
- **Exit metric: voxel commits touching anything outside voxel crates and
  their API < 10%** (from 36% in October by `just co-change --since
  2026-10-01 --unit voxel`, hosts excluded; 40% with examples). If not met, the extension points are wrong;
  fix them before converting more features.

As built (2026-11-02): layer violations 9 → 0.
- Feature contracts are `crates/api/*` crates (layer `api`: keys, row
  ABIs, WGSL snippets, stand-ins; no state): `helio-sky-api` (atmosphere
  frame and sky table, cloud shadow, sun/directional visibility),
  `helio-gi-api` (probe clipmap, rays/hits/irradiance keys, `gi_common` /
  `gi_sample`), `helio-foliage-api` (the old `contract` module: GPU records,
  packing, placement reference, ground protocol, SceneDB rows, shared WGSL;
  still no wgpu). Deferred lighting, HLFS, fog, water and voxel read these,
  never `helio-pass-sky` / `-gi` / `-foliage-place`. The pass crates
  re-export their api, so existing paths keep working.
- Water names no terrain: `WaterTerrainProvider` is its seam, and the voxel
  adapter (`VoxelShoreTerrain`, `shore_map_from_voxel_planet`) lives where
  the two features meet, `helio_default_passes::water_shore`.
- Voxel split into `helio-voxel-world` (grid, terrain, edits, queries,
  maps, features and their WGSL mirrors; no GPU) and
  `helio-pass-voxel-planet` (chunk passes, residency, sun pages, meadow),
  which re-exports the world so a host names one crate. The `engine`
  cargo feature is gone (the world crate is the GPU-free build).
  `helio-voxel-data` (the scene's journal and shared names) is voxel's
  public face to hosts; no separate `voxel-api` crate yet, because no other
  feature reads voxel: voxel implements the others' APIs (GI tracer,
  foliage ground, water's provider), it does not publish one.
- The voxel scene-material reader moved into the voxel crate
  (`chunks::scene::helio_scene_materials`).
- `voxel_flight` and `water_world` moved to `crates/examples` (hosts):
  `cargo run -p examples --release --bin voxel_flight`. They were 36 of the
  106 preset partners in the October co-change.
- Not done: the GI tracer and foliage-ground wiring stay in the preset's
  code builders (composing features is the preset's job); a voxel node for
  node graphs waits for presets-as-assets (phase 3 leftover), since nothing
  builds voxel from a node graph today. The `pulsar.voxel` plugin crate
  (voxel components, `voxel_backend.rs`, `scene/voxel_*`) is phase 5's
  first Pulsar step. Measure the exit metric from this phase on:
  `co_change.py --since <p4 merge> --unit voxel`.

### Phase 5: remaining features (#1541, #1544 reshaped)
- Group Tier A/B passes into features: VG (5 crates), GI, sky+clouds, fog,
  water, foliage (place+gbuffer), lens+flare, portals, decals, sprites2d,
  debug/perf overlays, SSR/planar reflections, FXAA/SMAA/HLFS post variants.
- Each conversion: Helio feature + api crate, and in Pulsar the plugin that
  registers its components and the feature, in one change per repo, deleting
  the central registration.
- `helio-default-passes` disappears.

As built, step 1 (2026-11-02): features place their own nodes.
- Feature API: `Slot` (extension points of preset graphs: `Geometry`,
  after the spine's G-buffer; `LightingInputs`, before deferred lighting)
  with `NodeDescriptor::in_slot(slot, order)`, and **finishers**
  (`FeatureRegistrar::finish`): steps run once every node is built, where
  features that meet are joined.
- The deferred preset is passes by name plus slots; `deferred_graph_asset`
  expands each slot to what the registry holds there.
- `GiFeature` (helio-pass-gi): update and mesh rays at `Geometry`, resolve
  at `LightingInputs`; `GiSettings` (probe spacing, `terrain_traced`)
  replaces `RendererConfig::gi_probe_spacing`.
- `SkyFeature` (helio-pass-sky): the atmosphere tables, and the clouds'
  trace at `LightingInputs`, which the node graph lacked.
- `VoxelFeature` (helio-pass-voxel-planet): the chunk terrain at `Geometry`
  from a host's `VoxelTerrain` setting (its pass makers). It offers its
  ground through `helio_foliage_api::OfferedGrounds`, and a foliage
  finisher grows placement on it. `helio_default_passes::with_voxel_terrain`
  sets the terrain and GI's `terrain_traced` together.
- `graph_parity` compares the whole pass list now (GI and clouds were
  filtered out), and with voxel terrain: same passes and declarations as
  the legacy factory, foliage on the terrain's ground.
- Next: hosts build from node graphs (voxel_craft, voxel_flight, Pulsar's
  scene renderer through `with_voxel_terrain`); forward, HLFS, FXAA and
  simple presets as graphs; then the `build_*_graph*` functions and
  `VoxelPassFactory` go.

As built, step 2 (2026-11-02): every shipped pipeline is a node graph.
- `helio_default_passes::presets::Preset` (`Deferred`, `DeferredFxaa`,
  `Hlfs`, `HlfsFxaa`, `Forward`, `Simple`) makes its graph asset from the
  renderer config on each build (resize and shader reload included): the
  config picks TSR or FXAA, SSR, planar reflections, portals, foliage and
  the TSR node's params. Slots gained `Lighting` (after the lit colour) and
  `Output` (after the final image).
- `GraphAssetPipeline::made` builds such an asset (no fallback: a preset
  that does not build is a bug and panics with its diagnostics); fixed
  assets keep their fallback, now `presets::deferred`. The compiler keeps
  graphs single-layer under XR, as the code builders did.
- New nodes: `hlfs`, `ssr_composite`, `virtual_geometry` (carries the
  geometry phase's perf marker), `simple_cube`; `transparent` and `tsr`
  gained the HLFS reactive-mask params. The scene-linear format is a
  setting (`SceneColorFormat`, default `FOGGED_HDR_FORMAT`); HLFS presets
  set their lighting's preferred format.
- Host WGSL is a setting: `helio_pass_postprocess::UserEffects`,
  `helio_pass_lens::LensUser`.
- Every host moved: ~70 examples to `presets::*`, voxel_craft,
  voxel_flight, water_world and the voxel tests to `with_voxel_terrain` +
  `Preset::Deferred`, Pulsar's scene renderer puts its voxel backends'
  `VoxelTerrain` in the config, so terrain joins whichever pipeline a level
  runs (HLFS and FXAA presets included; Forward has no G-buffer).
- Deleted: the 27 `build_*_graph*` functions, `GraphPassFactory`,
  `VoxelPassFactory` and their helpers (lib.rs 2319 → ~260 lines).
- Checked before deleting: every preset against its builder for default,
  TSR, voxel+TSR, reflections, and portals/foliage-off configs: pass lists,
  reads, writes and write formats. Deferred and Forward and Simple were
  identical. Intended differences: HLFS clears `pre_aa` in its lighting
  format (the builder cleared it in the display format, then HLFS wrote
  another), and the clouds' trace now runs before HLFS lighting (its
  slot); `DeferredFxaa` now runs FXAA and draws transparency (the builder
  had neither unless TSR was on). `DeclaredResource` records write formats
  so tests see format differences.
- `graph_parity` now tests the presets: the template builds the deferred
  preset's graph, presets follow the config, HLFS keeps one format, voxel
  terrain joins every G-buffer preset.

As built, steps 3 and 4 (2026-11-02): every pass node lives with its pass;
`helio-default-passes` is gone.
- `helio-spine` (layer spine) is the spine as a feature: scene prep,
  shadows, culling, light binning, G-buffer, forward and deferred lighting,
  transparency, TSR, the tone map, the host-input nodes and the background
  pass. Its nodes read `SpineSettings` (shadow budget and quality,
  background colour, debug mode, environment reflections), never
  `RendererConfig`.
- Every feature crate has a `feature.rs` registering its nodes: decals,
  SSR (+ composite), planar reflections, HLFS, foliage (both nodes and the
  ground finisher), VG (new crate `helio-pass-vg`: the chain's glue moved
  out of the presets), portal cull, portal draws, billboards, corona, water,
  volumetric fog, FXAA, lens, flare, sprites, buffer view, perf overlay
  (with the phase markers), simple cube; the sky adds its composite.
- Feature API: preparers (`FeatureRegistrar::prepare`, run before any node
  builds), `SceneColorFormat` + `BuildEnv::scene_format`, the `pass_node`
  and `missing` helpers.
- `helio-presets` (layer preset) is what remains of the presets: `Preset`,
  `features::default_features` (the feature list), the renderer's own
  nodes (debug draw) and the config bridge (a preparer mapping
  `RendererConfig` into `SpineSettings` and the perf overlay's mode),
  `graphs`, `pipelines`, the swap policy, `with_voxel_terrain`,
  `with_virtual_geometry`, `config_with_host_settings`, `water_shore`.
- `helio-scene-derivations` (layer preset) holds the scene derivations:
  scene join, hierarchy resolve, environment join, portal projection,
  texture means, and their shaders. The environment join still knows many
  features' rows; splitting it per feature is a follow-up.
- Every preset's pass list for five configs is identical before and after
  both steps (but for the background pass's type path). Hosts and Pulsar
  name `helio_presets` / `helio_scene_derivations`.

**Rule (user, 2026-11-02): components are the API.** The editor, the game
and Helio's passes communicate only through an object's component: its
methods, and the rows and payloads it publishes. No object-specific
knowledge in Helio, the presets, `engine_backend`'s renderer or other
surrounding crates; no bespoke registries, traits or mailboxes for one kind
of object.

As built, step 5 (2026-11-02): the presets name no feature pair.
- Features declare what they need of the others in their own preparers:
  the voxel feature, while it has terrain, inserts
  `helio_gi_api::TerrainTracer` (GI then updates without a TLAS and spaces
  probes for open ground) and `helio_pass_tsr::StochasticSamples` (TSR
  reads the stochastic mask). `GiSettings::terrain_traced`,
  `with_voxel_terrain` and the presets' voxel check are gone; a host adds a
  feature setting with the generic `RendererConfig::with_feature`.
- Water's terrain seam is `helio-water-api::WaterTerrainProvider`; the
  voxel adapter is `helio_pass_voxel_planet::VoxelShoreTerrain`, and the
  host bakes with `WaterShoreMap::from_terrain`. `helio_presets::water_shore`
  is gone.

Next, step 6: voxel through its component (replaces the earlier
"`pulsar.voxel` plugin crate" item, which would have moved today's bespoke
renderer API into a plugin unchanged). Today, per a survey of Pulsar:
two CPU worlds per terrain (the component's `WORLDS` cache and the render
backend's `CachedPlanet`); sculpting writes the journal from the renderer
(`apply_voxel_brush_commit`), bypassing the component's `append_edits`, so
no block events fire and the component's world goes stale; ground, lift,
clip-range, pick and kept-chunk queries go through `VoxelRenderBackend`,
`HelioRenderer` mailboxes and Arc channels; Helio's pass gets its world
from host-made `VoxelTerrain` closures. Target and order:
1. **Edits through the component.** A brush is a component method
   (`VoxelTerrainComponent::apply_brush`, through `append_edits`): events
   fire, the component's world stays current. The renderer only finds the
   hit.
2. **One world per terrain, owned by the component's runtime.** The
   component builds and caches its `Arc<Planet>` (async, as the backend
   does now, kept chunks included); the backend's copy and `WORLDS` go.
3. **Queries are component methods.** Ground (up, altitude), lift out of
   ground, clip range, focus, CPU ray hits: `VoxelTerrainComponent`
   methods over its world. `VoxelGround`, `voxel_ground`,
   `voxel_altitude`, `lift_out_of_ground` leave the renderer; the game's
   free camera calls the component.
4. **Rendering reads what the component publishes.** The component
   publishes a volume row (entity, transform, voxel size, revision,
   appearance, smoothing) and its world as a SceneDB asset payload keyed by
   the entity; the voxel feature's pass reads rows and payloads from
   SceneDB. `VoxelTerrain`/`ChunkPassMaker`, `SharedPlanetFrame`,
   `VoxelRenderBackend`, `VoxelBackendRegistry` and the `find_pass_mut`
   appearance calls go.
5. **GPU answers come back as rows.** Drawn-surface picks and kept-chunk
   exports are published by the pass into SceneDB rows the component reads
   (`VoxelTerrainComponent::kept_chunks`, a pick method); the
   `VoxelChunkSaves` condvar mailbox and `SharedPicks` go.
6. Then the plugin: the voxel components, their runtime and the editor's
   sculpt tool move into a `pulsar.voxel` plugin that registers them and
   `VoxelFeature`; `helio_component` and `engine_backend` keep no voxel
   code.

### Phase 6: repo split
- Move Helio-Voxel (then VG) out once its API crate has gone ~2 weeks
  without a breaking change. Pulsar pins tags.

### Mapping to Pulsar #1532

| Issue | Change |
|---|---|
| #1533 Phase 1 leftovers | unchanged; add helio-component move-in |
| #1534 component class API | components are plugin-owned; unchanged in spirit |
| #1535 per-plugin GPU schemas | plugin registers the schema; the key/row ABI it writes is a Helio buffer contract |
| #1536 typed cross-plugin access | same rules apply to Helio `*-api` crates |
| #1538 helio-pass-api | becomes `helio-graph` + `helio-feature-api` (+ `helio-scene-input`) |
| #1539 render registration | `p.render().feature(..)`; features register nodes |
| #1541 Tier A moves | grouped as features, not per pass |
| #1543 split helio_component | components go to their plugins; Helio's 31 pass-crate schemas join them |
| #1544 Tier B moves | "passes move with their component" becomes: the plugin owns the component, the feature owns the passes; voxel first as the Phase 4 pilot |

---

## 6. Keeping it from rotting

- **Layer check in CI** (Phase 0): `cargo metadata` against `layers.toml`;
  the allow-list only shrinks.
- **API purity test**: `*-api` crates have no statics (#1536 style).
- **Contract layout tests**: each row ABI has a Rust/WGSL layout test in its
  `*-api` crate; each Pulsar component asserts it matches.
- **Test host** (`helio-test-host`): headless device + spine + chosen
  features + contract rows written by key; golden frames and GPU timings per
  feature. A feature PR runs its own suite + a spine smoke test; the preset
  matrix (default, voxel_craft, voxel_flight, minimal) runs on merge.
- **Co-change report** per PR: crates touched outside the feature and its
  APIs. A PR that touches two impl crates is a review flag.
- **Rebuild budget**: changing a feature impl rebuilds that crate, the preset
  and the final binaries, nothing else. Checked with `cargo build --timings`
  on a sample change per phase.

---

## 7. Open decisions

1. **Voxel world home**: Helio-Voxel repo (recommended: it is GPU-heavy
   engine tech, usable without the editor) or Pulsar.
2. **Which features stay in the Helio repo** vs their own repos. Recommended:
   split only voxel and VG at first; the rest stay in Helio under
   `features/`, with the same layer rules.
3. **Helio's SceneDB dependency**: own the small projection type and drop
   `pulsar_scenedb` (recommended), or keep re-exporting SceneDB's handle
   types.
4. **Dylib plugins**: keep statically linked (today's real mode); the
   statics-free API crates keep dylibs possible later.
5. **Replacing core nodes from a plugin**: explicit `replaces =` in the
   manifest, as #1532's doc proposes.
