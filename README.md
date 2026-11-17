<div align="center">

<img src="./branding/Helio.svg" alt="Helio Renderer" width="400"/>

A kind-agnostic, GPU-driven renderer in Rust on wgpu

[![Rust](https://img.shields.io/badge/rust-stable-orange?logo=rust)](https://www.rust-lang.org/)
[![wgpu](https://img.shields.io/badge/wgpu-30-blue)](https://wgpu.rs/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

Helio is the renderer of the [Pulsar](https://github.com/Far-Beyond-Pulsar/Pulsar-Native) engine.
It owns the frame, the render graph, GPU uploads, shaders, profiling and settings.
It knows no object kind. Meshes, lights, materials, sky, fog, water, voxels, foliage and every other kind are Pulsar plugins that bring their own components, features, passes and contracts.

This README covers what Helio is, its layers, the seams a plugin plugs into, and how to build and test it.
The design and its history are in [`docs/architecture/`](docs/architecture/).

## 1. What Helio is

- The frame: views, draw phases, the G-buffer layout, the scene colour chain and the lighting-input contracts (`helio-frame-api`), and the spine passes no kind owns (`helio-spine`).
- The render graph: passes, resources, scheduling, render-pass fusion and scene membership (`helio-core`), graphs as data (`helio-graph-asset`) and the presets built from them (`helio-presets`).
- GPU uploads: a frame's buffer writes staged through one ring and copied in one flush (`helio_core::upload::FrameUploads`), pipeline compiles off the frame (`helio_core::compile::Compiling`) and the threads Helio-layer code starts (`helio-tasks`, run by the host's task system).
- Shaders: `include_wgsl!`, the shared prelude and the WGSL snippet system (`helio_core::shader`).
- Profiling: CPU scopes through the `profiling` crate of the public [Pulsar-Profiling](https://github.com/Far-Beyond-Pulsar/Pulsar-Profiling) repository, and GPU timestamps per pass (`RenderTimingSnapshot`).
- Settings: every setting a host can change, described once (`helio-settings`).

What Helio is not:

- It has no object kind and no API for one: no mesh, light, material or sky types, setters or passes. The layer gate fails when a crate outside the test host names a kind (`[kinds]` in [`architecture/layers.toml`](architecture/layers.toml)); the names left are in [`architecture/kind-names.txt`](architecture/kind-names.txt), a list that only shrinks.
- It defines no SceneDB components (`scene_store_derives = 0` in the layer manifest). It declares contracts and reads what components publish under them.
- It ships no default pipeline. A host chooses what it renders with. Pulsar's `pulsar.default_pipelines` plugin makes its `.rendergraph` assets from Helio's presets.

## 2. Layers

Every workspace crate has a layer in [`architecture/layers.toml`](architecture/layers.toml), and a crate depends only on the layers its layer's `may_use` lists: those below it, and for some layers its own. `just arch` checks this. Feature crates of one group may use each other, never another group's crates.

| Layer | Crates | What it holds |
|---|---|---|
| leaf | `helio-settings`, `helio-tasks`, `scenedb-inspector-agent` | No workspace dependency. Setting descriptions, the task seam, a SceneDB inspector agent. |
| core | `helio-core` | gpu and graph: `RenderGraph`, `RenderPass`, resources, views, uploads, compiles, shaders, profiling, frame stats, `BufferContract`/`PayloadContract`, `SceneDerivation`. |
| scene | `helio-scene-join`, `helio-scene-derivations` | Scene input building blocks no kind owns: the generic GPU row join (`RowJoin`) and the hierarchy resolve. |
| feature_api | `helio-feature-api`, `helio-frame-api` | What a feature registers (`Feature`, `FeatureRegistrar`, `NodeDescriptor`, `BuildEnv`) and the frame it registers into (slots, phases, G-buffer, colour chain, lighting inputs, views, rays). |
| api | (none in Helio) | A kind's contracts. Each lives in its plugin (`helio-light-api`, `helio-mesh-api`, ...). |
| spine | `helio-spine`, `helio-pass-hiz`, `helio-pass-tsr`, `helio-pass-postprocess`, `helio-pass-debug-overlay`, `helio-pass-status-screen` | The passes every pipeline is built around: background, depth pyramid, unlit stand-in, TSR, tone map, on-screen text, the status screen. |
| feature | `helio-pass-ssr`, `helio-pass-planar-reflection`, `helio-pass-fxaa`, `helio-pass-perf-overlay`, `helio-pass-buffer-view`, `helio-pass-selection-outline` | Kind-agnostic features: screen-space and planar reflections, FXAA, the performance overlay, buffer views, the editor's selection outline. |
| renderer | `helio`, `helio-xr` | The facade: `Renderer`, `RendererBuilder`, `RendererConfig`, cameras, the frame loop, surfaces, the coordinate-space table, debug draw, OpenXR. It knows no feature. |
| preset | `helio-presets`, `helio-graph-asset` | Graphs as data, node registries, the shipped presets (`Preset::Deferred`, `DeferredFxaa`, `Forward`) and the feature list they build from. |
| host | `examples`, `helio-components`, `helio-controls`, `helio-wasm` | Helio's own hosts: examples, row components tied to their contract rows (`same_layout!`, for Helio's tests and the plugins' host rows), camera controllers, the wasm runner. |

The crates are in [`crates/`](crates/), the pass crates in [`crates/passes/3d/`](crates/passes/3d/).

## 3. How a plugin plugs in

A plugin is a Pulsar repository, `Far-Beyond-Pulsar/Plugin_<Kind>`, vendored at `plugins/vendor/<kind>/` in Pulsar-Native. It has an `api` crate (its contracts), a `component` crate (its SceneDB component and the methods scripts and the editor call), a `feature` crate (its Helio feature and passes), an `editor` crate and a root crate that registers everything. The plugins are listed in the [plugins registry](https://github.com/Far-Beyond-Pulsar/Plugins) and in the table of [`docs/architecture/plugin-migration.md`](docs/architecture/plugin-migration.md), section 3.

A plugin meets Helio and other plugins only at the seams below.

### Feature registration

A feature implements `helio_feature_api::Feature`. Its `register` adds node types (each builds passes into a graph), declares the buffer contracts its passes read, and may add preparers (`prepare`), finishers (`finish`), scene derivations (`scene_derivation`), records a frame contract asks for (`provide`) and shader hot-reload policy (`swappable`, `swap_group`). A preset places each node at the slot or phase it names, so no preset names a feature.

```rust
use helio_core::BufferContract;
use helio_feature_api::{pass_kind, pass_node, Feature, FeatureRegistrar, FeatureSet};
use helio_frame_api::{FrameNode, Phase};

/// The row a glow kind publishes. Its api crate owns this layout.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GlowRow {
    pub position_radius: [f32; 4],
    pub color: [f32; 4],
}

pub const GLOWS: BufferContract<GlowRow> = BufferContract::new("glows");

pub struct GlowFeature;

impl Feature for GlowFeature {
    fn id(&self) -> &'static str {
        "example.glow"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        registrar.consumes(GLOWS);
        registrar.node(
            pass_node("glow", "Glow", "Passes: Example", "Glows over the lit scene.", |_build| {
                // build.graph.add_pass(Box::new(GlowPass::new(build.env.device, ..)));
                Ok(())
            })
            .drawer(Phase::Overlay, 10),
        );
    }
}

// An engine adds its plugins' features to Helio's own.
let registry = helio_presets::graphs::registry_with(&FeatureSet::new().with(GlowFeature));
assert_eq!(registry.feature_of(&pass_kind("glow")), Some("example.glow"));
```

In Pulsar, the plugin's root crate hands its features to the engine through `RenderFeatureRegistration` (engine_backend), and the engine builds every pipeline over the features of the plugins a project enables.

### Buffer and pool contracts

A contract is the seam between a component and the passes that read its data. The consumer's api crate owns the row ABI, because its shaders read it.

- `BufferContract<R>` (`helio_core::scene_input`) is a key and a row type. `resolve` checks a published buffer's stride against `R` and names the contract on a mismatch. A producer asserts its layout at compile time with `assert_producer`.
- `PoolContract<T>` is the same type, for variable-length pools (vertices, indices).
- `PayloadContract<T>` is a key and a CPU type for payloads a host hands the renderer (`Renderer::set_scene_payloads`), read by passes from `ctx.scene_buffers.payload(..)`.
- The version lives in the plugin's `pulsar-plugin.toml`: each `provides` entry has an `id`, a `kind` (`buffer`, `api`, `class`, `trait`), its `members` (buffer keys or crate names) and an integer `version`. A `requires` entry names a contract id, and one that pins a version needs exactly that one.
- Row ABIs carry layout tests in their api crates (the meshes api checks each row against its WGSL struct). Pulsar's `one_owner` test fails a WGSL copy of another owner's snippet and a key named by string literal.

### Extension points

The plan's extension-point traits (`GBufferContributor`, `ShadowCaster`, `LightingContributor`, `PostEffect`, ...) were not built as traits. Most became a place in the frame (an `ExtensionPoint` a node is placed at) or a contract in `helio-frame-api`, and two need no frame seam:

| Planned seam | What it is on main |
|---|---|
| `GBufferContributor` | A drawer of `Phase::GBuffer`, writing the layout in `helio_frame_api::gbuffer` |
| `ShadowCaster` | A drawer of `Phase::ShadowCasters`, drawn for every shadow face the lights pick |
| `RayProxySource` | The frame's TLAS payload, `helio_frame_api::rays::SCENE_RAYS`, published by whatever builds it |
| `LightingContributor` | A publisher of a lighting input (`helio_frame_api::lighting`: `INDIRECT_DIFFUSE`, `INDIRECT_SPECULAR_LAYERS`, `SUN_VISIBILITY`, `TRANSMISSION`, ...) at `Slot::LightingInputs`, or a `LightingModel` |
| `ParticipatingMedia` | A publisher of `helio_frame_api::media` at `Slot::Media` |
| `PostEffect` | A node on the scene colour chain (`helio_frame_api::color::current` and `advance`) at `Slot::Resolved` or `Slot::Display` |
| `SurfaceQuery` | No frame seam: a kind that offers its surface implements the reader's api (foliage's ground, `helio-foliage-api`, which voxel terrain implements) |
| `HistoryInvalidation` | No frame seam: TSR is a spine pass, a host resets its history on a cut (`TsrPass::reset_history`), and producers describe their shading through the reactive and stochastic masks |

Slots (`helio_frame_api::Slot`): `Prepare`, `Visibility`, `Geometry`, `LightingInputs`, `Lighting`, `Media`, `Resolved`, `Display`, `Output`. A node joins one with `FrameNode::in_slot(slot, order)`. Host inputs, resources a host supplies every frame, are declared by the feature that reads them (`helio_frame_api::host_input`, placed at `HOST_INPUTS`).

### Draw phases

A phase (`helio_frame_api::Phase`) is a place in the frame where every kind with something to draw there draws into attachments the frame owns: `ShadowCasters`, `DepthPrepass`, `GBuffer`, `Forward`, `Transparent`, `Overlay`, `SelectionMask`. A phase has three stages: compute before it (`before_phase`), one opener that clears its attachments (`opens`), and drawers that load them (`drawer`). Adjacent drawers with the same attachments fuse into one render pass. A drawer's declared attachments are checked against the phase's formats when it builds. The lighting model a preset uses decides which phases run (`Preset::phases`).

### The scene-driven graph

A graph holds every pass its pipeline can use. Each frame it schedules only its members: before a frame it asks every pass `RenderPass::is_member(&mut SceneContent)`. A pass with content-dependent work answers from the scene, usually `SceneContent::holds_rows(key)`, whether a buffer has a live row (read back without waiting). A dormant pass is not prepared or executed and holds no graph textures. It frees its content resources in `on_leave`. A resize keeps the graph and reallocates its textures in place.

### Views

A frame renders its primary view and the captures features ask for (`helio_core::view`). A pass that renders views returns `true` from `RenderPass::renders_views`. At that pass boundary the graph calls its `render_views` with a `ViewContext`, and `ViewContext::render_view` runs a view graph the preset composed from the frame's members (`Preset::view_nodes`). Nodes marked `once_per_frame` run once for the frame, and views read their outputs. `ViewBudget` bounds view depth and count. A drawer reads the view's facts (winding, clip) from `helio_frame_api::view` when it builds. A planar reflection is a capture.

### Coordinate spaces and portals

A portal is not a view. It is a rigid map between coordinate spaces, drawn in the frame's own G-buffer.

- The renderer owns the frame's coordinate-space table (slot 0 the frame's own space, then the host's spaces, current and previous frame), published as `helio_core::CoordinateSpacesFrameData`. Every kind's culling and drawers place an instance through its space.
- World positions are f64 (`Camera::eye`). Each frame renders about an origin the renderer picks near the camera (`Renderer::origin_for`), and pass rows are relative to it.
- A portals plugin publishes its chains through its own api crate. Each drawer draws its content through them. The pixels a chain draws carry its remap (`helio_frame_api::lighting::VIEW_REMAPS`), so lighting and decals take them back to their own world.

## 4. Building a renderer

A host makes a `RendererConfig`, hands `RendererBuilder` the SceneDB GPU mirror and a pipeline, builds the renderer and calls `Renderer::render(&camera, &target)` once a frame.

```rust,no_run
use helio::{HelioGraphicsContext, Renderer, RendererBuilder, RendererConfig, SceneDbHandle};
use helio_feature_api::FeatureSet;
use helio_presets::Preset;

fn renderer(
    graphics: &mut HelioGraphicsContext<'_>,
    scene_db: SceneDbHandle,
    plugin_features: &FeatureSet,
    width: u32,
    height: u32,
) -> helio::Result<Renderer> {
    let config = RendererConfig::new(width, height, graphics.surface_config.format);
    RendererBuilder::new(config, scene_db)
        .with_graph(Preset::Deferred.with_features(plugin_features))
        .build_with_graphics_context(graphics)
}
```

Without the plugins' features nothing draws into the preset's phases, and its lighting model resolves to the spine's unlit stand-in (with one warning naming the missing model). Scenes of meshes, lights and materials are the plugins' examples in Pulsar-Native.

## 5. Build and test

The recipes are in the [`justfile`](justfile):

```sh
just ci      # fmt-check, layer gate, check, clippy, nextest
just arch    # the layer gate (Python 3.11+)
just co-change   # how often a change to one unit also changed others
```

Helio's own examples need no plugin:

```sh
cargo run -p examples --bin simple_graph   # a custom one-pass graph: a spinning test cube
cargo run -p examples --bin debug_shapes   # the facade's debug draw
```

`simple_graph` builds its graph with `RendererBuilder::with_pass_build_context` and draws with no scene content. `debug_shapes` draws lines, spheres, boxes, cones, cylinders, tori and frustums through the renderer's debug draw.

The wasm runner is `helio-wasm`. The web and Android demos are scenes of meshes, so they live in Plugin_Meshes (`plugins/vendor/meshes/web-demos` and `android-demos` in Pulsar-Native).

## 6. Further reading

- [`AGENTS.md`](AGENTS.md): the rules every change follows, and the key commands.
- The [`helio-architecture` skill](.agents/skills/helio-architecture/SKILL.md): the steps for any change to Helio or a plugin's rendering (read, find the owner, design note, invariants, verify).
- [`docs/architecture/layered-rewrite.md`](docs/architecture/layered-rewrite.md): layers, rules, buffer contracts, extension points and why this shape.
- [`docs/architecture/plugin-migration.md`](docs/architecture/plugin-migration.md): every object kind is a plugin, what Helio is, the plugin table and the "As built" notes per phase.
- [`docs/custom-render-pass.md`](docs/custom-render-pass.md) and [`docs/custom-pass-pipeline.md`](docs/custom-pass-pipeline.md): writing a `RenderPass` and composing passes into a graph.
- [Pulsar-Native](https://github.com/Far-Beyond-Pulsar/Pulsar-Native): the engine and editor, and the plugins at `plugins/vendor/`.
- [Far-Beyond-Pulsar/Plugins](https://github.com/Far-Beyond-Pulsar/Plugins): the plugins registry.

## License

Helio is MIT licensed. Copyright 2026 Tristan Poland. See [LICENSE](LICENSE) for the full text.
