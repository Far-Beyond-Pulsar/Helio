<div align="center">

<img src="./branding/Helio.svg" alt="Helio Renderer" width="400"/>

**GPU-driven deferred rendering in pure Rust: modular, pass-based, cross-platform**

[![Rust](https://img.shields.io/badge/rust-stable-orange?logo=rust)](https://www.rust-lang.org/)
[![wgpu](https://img.shields.io/badge/wgpu-30-blue)](https://wgpu.rs/)
[![WebGPU](https://img.shields.io/badge/WebGPU-native%20%2B%20browser-8A2BE2)](https://gpuweb.github.io/gpuweb/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

Helio is a GPU-driven deferred renderer built entirely in Rust on `wgpu`. Culling, LOD selection, indirect-draw dispatch, and light evaluation all run on the GPU, while every CPU-side call stays bounded (typically O(1)). The pass architecture is strictly modular, with each render pass living in its own crate, and the same renderer, scene, and render graph run unchanged on native desktop backends and in the browser through WebGPU.

## Table of contents

[Why Helio](#why-helio) · [Architecture](#architecture) · [Quick start](#quick-start) · [Cross-platform graphics](#cross-platform-graphics) · [The render graph](#the-render-graph) · [Scene and handle API](#scene-and-handle-api) · [Custom material shaders (Radiant)](#custom-material-shaders-radiant) · [Custom post-process shaders](#custom-post-process-shaders) · [Custom render passes](#custom-render-passes) · [Debug tools](#debug-tools) · [Pass reference](#pass-reference) · [Examples and web demos](#examples-and-web-demos) · [Asset pipeline](#asset-pipeline)

---

## Why Helio

The pass architecture is genuinely modular. Every render pass is its own crate (`helio-pass-gbuffer`, `helio-pass-deferred-light`, `helio-pass-taa`, and 40+ more), and the central `helio` crate has zero knowledge of any pass type. Adding a pass means writing a crate and plugging it into a graph builder; the central crates never change.

It is GPU-driven by default. The CPU never iterates draw calls: culling, LOD selection, and indirect-dispatch buffer generation all happen on the GPU. Scene data lives in GPU buffers with dirty-tracked CPU mirrors, so `flush()` uploads only what actually changed since the last frame.

The scene API is handle-based. Every resource (`MeshId`, `MaterialId`, `LightId`, `ObjectId`, and so on) is a lightweight `Copy` handle backed by a generational arena, so inserting, updating, and removing are O(1) with no aliasing. The render graph carries its own rebuilder closure, which means resizing the window transparently rebuilds the pipeline with no manual boilerplate.

Most importantly for this project, Helio is cross-platform from a single source. The renderer, scene, and graph builders are identical on native and web; WebGPU's smaller capability envelope (bindless limits, no multi-draw-indirect) is absorbed inside the engine rather than in your code. A demo written against `HelioWasmApp` compiles to a native window and a browser canvas from the same file.

---

## Architecture

```
crates/helio                Public API: Renderer, Scene, Camera, Radiant, debug helpers
crates/helio-core           Render graph runtime, GpuScene, RenderPass trait, PassContext
crates/libhelio             GPU-shared POD types (GpuLight, GpuMaterial, uniforms)
crates/helio-default-graphs Pre-built graph configurations (deferred, FXAA, user-effects)
crates/helio-pass-*         One crate per render pass (40+ passes)
crates/helio-wasm           Cross-platform app runner and HelioWasmApp trait (native + web)
crates/helio-web-demos      Every example compiled to WASM, one feature flag per demo
crates/helio-asset-compat   FBX / glTF / OBJ / USD loading
crates/helio-voxel-core     Shared voxel terrain component (mesh + ray-march)
crates/examples             Runnable native demos and editor
```

The separation between `helio` and the `helio-pass-*` crates is strict: central crates never import pass types. The `RenderPass` trait lives in `helio-core`, pass crates implement it, and graph builders compose them while storing a `GraphRebuilder` inside the graph. The `Renderer` extracts that rebuilder at construction, which is what gives automatic rebuild on resize without any dependency on specific pass types.

---

## Quick start

```sh
# Native demos
cargo run -p examples --bin indoor_cathedral --release
cargo run -p examples --bin outdoor_city --release
cargo run -p examples --bin load_fbx --release -- path/to/model.fbx

# All demos in the browser (builds WASM and serves on http://127.0.0.1:8000)
cargo run --bin web
```

To build a renderer around your own surface, the pattern is: request the features and limits Helio needs for the adapter, create a `RendererConfig` and `Scene`, build a graph, and hand everything to `Renderer::new`. Because the graph carries its own rebuilder, resize handling and graph reconstruction happen for you.

```rust
use helio::{Camera, DebugDrawState, Renderer, RendererConfig, Scene,
            required_wgpu_features, required_wgpu_limits};
use helio_default_graphs::build_default_graph;

// Request exactly the features/limits Helio needs for this adapter.
let features = required_wgpu_features(adapter.features());
let limits   = required_wgpu_limits(adapter.limits());
// ... create device/queue with those ...

let config = RendererConfig::new(width, height, surface_format);
let scene  = Scene::new(device.clone(), queue.clone());
let debug_state = std::sync::Arc::new(std::sync::Mutex::new(DebugDrawState::default()));
// (debug_camera_buf / cull_stats_buf: small uniform/storage buffers, see examples)

let graph = build_default_graph(
    &device, &queue, &scene, config,
    debug_state.clone(), &debug_camera_buf, &cull_stats_buf, None,
);
let mut renderer = Renderer::new(
    device.clone(), queue.clone(),
    config.surface_format, config.width, config.height, config.render_scale,
    config, scene, graph, debug_state, debug_camera_buf, cull_stats_buf,
);

let camera = Camera::perspective_look_at(
    glam::Vec3::new(0.0, 2.0, 6.0), glam::Vec3::ZERO, glam::Vec3::Y,
    60_f32.to_radians(), width as f32 / height as f32, 0.1, 1000.0,
);

renderer.render(&camera, &surface_view)?;
```

If you want that same code to run in the browser too, don't hand-roll the windowing. Use `HelioWasmApp` (described next), which wraps all of the above for both targets.

---

## Cross-platform graphics

Helio runs on native desktop backends (Vulkan, Metal, DX12, GLES) and in the browser on WebGPU. This is not a separate port; it is the same renderer driven through the same graph.

The parts you touch are identical across targets. Graph building (`RenderGraph::new`, `add_pass`, `lock`, and the `build_default_graph*` family) has zero target cfgs, so it is byte-for-byte the same. The `Renderer` and `Scene` API (`render`, `insert_material`, `find_pass_mut`, the `debug_*` helpers) behaves the same everywhere. What differs lives inside the passes, where the engine adapts automatically, and in the app/windowing layer, which the shared runner in `helio-wasm` absorbs.

The one thing worth understanding is the capability envelope, because WebGPU is smaller than a native backend. The engine handles each difference internally:

| Capability | Native | Web (WebGPU) |
|---|---|---|
| Bindless material textures (`MAX_TEXTURES`) | 256 | 16 (limits clamped automatically) |
| Multi-draw indirect | `multi_draw_indexed_indirect`, one call | per-draw `draw_indexed_indirect` loop, automatic |
| Required features | `TEXTURE_BINDING_ARRAY` + non-uniform indexing + `INDIRECT_FIRST_INSTANCE` | `INDIRECT_FIRST_INSTANCE` only |
| Present mode | backend-selected | `Fifo` |

"Same API" is not the same as "same capabilities." A scene that stays within the web envelope (at most sixteen unique material textures per draw, and so on) looks identical on both targets. A scene that pushes past it can look different or fail on the web, but that is a content ceiling rather than a code fork. To stay on the right side of it, always create your device through the helpers so you request exactly what Helio needs on each target:

```rust
let features = helio::required_wgpu_features(adapter.features()); // differs per target
let limits   = helio::required_wgpu_limits(adapter.limits());     // clamps MAX_TEXTURES
```

### One source, both targets, via `HelioWasmApp`

`helio-wasm` provides the cross-platform app runner. You implement `HelioWasmApp` and call `launch::<T>()`; on native it runs a winit window, and in the browser it attaches a WebGPU canvas. The runner owns the event loop, surface, input, and camera plumbing, leaving you to write the scene and the per-frame logic.

```rust
use std::sync::Arc;
use helio::{Camera, Renderer};
use helio_wasm::{HelioWasmApp, InputState, KeyCode, launch};

struct Demo { /* camera state, handles, ... */ }

impl HelioWasmApp for Demo {
    fn title() -> &'static str { "My Demo" }

    fn init(renderer: &mut Renderer, _device: Arc<wgpu::Device>,
            _queue: Arc<wgpu::Queue>, _w: u32, _h: u32) -> Self {
        // Build the scene: meshes, materials, lights, ambient, clear color.
        renderer.set_ambient([0.4, 0.45, 0.5], 0.15);
        Demo { /* ... */ }
    }

    fn update(&mut self, renderer: &mut Renderer, dt: f32, elapsed: f32,
              input: &InputState) -> Camera {
        // Read input, animate, return the camera for this frame.
        Camera::perspective_look_at(/* ... */ input.aspect_ratio(), 0.1, 1000.0)
    }
}

// Native entry point; the very same type serves the browser build.
fn main() { launch::<Demo>(); }
```

Only `init` and `update` are required; everything else on the trait has a sensible default.

| Method | Purpose |
|---|---|
| `init(renderer, device, queue, w, h) -> Self` | Build the scene once. |
| `update(&mut self, renderer, dt, elapsed, input) -> Camera` | Per-frame logic; return the camera. |
| `title() -> &str` | Window / tab title. |
| `render_scale() -> f32` | Internal render resolution (default `0.75`; use `1.0` for graphs with no TAA upscale). |
| `build_graph(device, queue, scene, config, debug_state, ...) -> Option<RenderGraph>` | Return a custom pipeline (voxel meshing, injected post-process); `None` uses the default deferred graph. |
| `on_resize(&mut self, renderer, w, h)` | React to viewport changes. |
| `grab_cursor_button()` / `release_cursor_on_grab_button_release()` | Mouse-look capture behavior. |

The `InputState` passed to `update` exposes `keys`, `mouse_delta`, `cursor_grabbed`, `mouse_left_just_pressed`, `cursor_pos`, `viewport_size`, and an `aspect_ratio()` helper.

### Building for the web

Every example is compiled to WASM by `helio-web-demos`, with one Cargo feature per demo, each launching a `HelioWasmApp`. The `web` binary is the build tool. Running `cargo run --bin web` opens an interactive TUI that builds every demo and then serves them, while `cargo run --bin web -- --headless` runs the same build non-interactively, writes the full site, and exits non-zero if any demo failed. Headless mode is what CI uses; there is no shell build script. Each demo is built with `wasm-pack` into `target/wasm-prebuilt/<demo>/`, alongside a generated landing page and a master index. Because the C dependencies (`meshopt`) need a wasm-capable `clang`, install LLVM first (`brew install llvm` on macOS, or your distribution's `clang` on Linux).

---

## The render graph

A `RenderGraph` is an ordered list of passes with declared read/write resources. It validates the dependency graph, manages transient texture pools and barriers, and rebuilds itself on resize. You rarely build one by hand, because a graph builder does it for you; `build_default_graph` gives the full deferred pipeline, and `build_default_graph_with_user_effects` adds injected post-process WGSL.

When you do want a custom pipeline, construct a graph and add passes directly. This is exactly what `HelioWasmApp::build_graph` returns, and it is the same code on native and web:

```rust
use helio::{RenderGraph, Renderer};
use helio_pass_voxel_mesh::VoxelMeshPass;
use helio_pass_fxaa::FxaaPass;

let mut graph = RenderGraph::new(device, queue);
graph.add_pass(Box::new(VoxelMeshPass::new(device, queue, config.surface_format)));
graph.add_pass(Box::new(FxaaPass::new(device, config.surface_format)));
graph.lock(config.width, config.height);
```

Passes communicate through named resources such as `"pre_aa"` and `"gbuffer"`; a pass declares what it `reads()` and `writes()`, and the graph wires them together. You can reach into a live graph by pass type at any time with `renderer.find_pass_mut::<FxaaPass>()`.

---

## Scene and handle API

The `Scene` is GPU-native with dirty-tracked uploads, and everything in it is a `Copy` handle. You insert a material, a mesh, an object that references both, and lights, then keep the returned handles to update or remove them later.

```rust
let scene = renderer.scene_mut();

let material = scene.insert_material(GpuMaterial { /* base_color, roughness_metallic, ... */ });
let mesh     = scene.insert_actor(helio::SceneActor::mesh(mesh_upload));
let object   = scene.insert_actor(helio::SceneActor::object(ObjectDescriptor {
    mesh, material, transform, /* bounds, groups, movability, ... */
}));
let light    = scene.insert_actor(helio::SceneActor::light(GpuLight { /* ... */ }));
```

Beyond the basics, objects carry a 64-bit `GroupMask` so you can hide, show, or transform whole groups at once, and meshes can be sectioned into one vertex buffer with several index ranges for Unreal-style multi-material geometry. Voxel volumes are inserted with `scene.insert_voxel_volume(VoxelVolumeDescriptor { .. })` and shared by both the voxel mesh and ray-march passes, while post-process is enabled by inserting a `SceneActor::post_process_volume(..)` (covered below). Whole-scene state such as ambient light, clear color, editor mode, and TAA jitter lives on the `Renderer` (`set_ambient`, `set_clear_color`, `set_editor_mode`, `set_jitter_enabled`), and `scene.clear()` wipes everything.

---

## Custom material shaders (Radiant)

Radiant is Helio's material system. It combines a built-in PBR uber-shader, hand-authored surface templates, and graph-generated WGSL snippets. The GBuffer pass evaluates every material through a shared `radiant_eval_surface()` function that contains injection markers, so custom templates and graph compilers can splice code in without touching the engine.

Radiant works in three tiers over one cost model. Tier 1 covers roughly ninety-five percent of materials by toggling feature flags on the built-in PBR shader, which needs no new pipelines at all. Tier 2 introduces a hand-authored `.wgsl` surface template (optionally paired with a graph snippet) for archetypes like clear coat, skin, hair, fabric, or iridescence, at the cost of one pipeline per template. Tier 3 is a full custom WGSL surface emitted by a graph compiler, one pipeline per snippet, for surfaces that fit no template.

| Tier | Mechanism | PSOs | Use case |
|------|-----------|------|----------|
| 1 | Feature flags on the built-in PBR uber-shader | 1 | ~95% of materials |
| 2 | Hand-authored `.wgsl` surface template (+ optional graph snippet) | per template | Surface archetypes (clear coat, skin, fabric) |
| 3 | Full custom WGSL via a graph compiler | per snippet | Unique surfaces that fit no template |

The GBuffer pass sorts instances by `(material_class, graph_hash)` at flush time and issues one draw per PSO, and shader modules are lazily compiled and cached by `(template_id, graph_hash, feature_flags)`. The lighting pass is never touched, because every variant writes the same GBuffer format.

A material is a plain POD struct. The `flags` field drives Tier 1, the `material_class` selects a template (0 is the built-in PBR shader), and `class_params` passes free parameters into whatever template is active.

```rust
GpuMaterial {
    base_color:         [f32; 4],   // linear RGBA
    emissive:           [f32; 4],   // RGB + strength
    roughness_metallic: [f32; 4],   // x=roughness y=metallic z=IOR w=specular_tint
    tex_base_color, tex_normal, tex_roughness, tex_emissive, tex_occlusion: u32, // bindless indices
    workflow:       u32,
    flags:          u32,            // FLAG_HAS_NORMAL_MAP | FLAG_ALPHA_TEST | ...  (Tier 1)
    material_class: u32,            // 0 = default PBR, 1+ = custom template (Tier 2/3)
    class_params:   [f32; 4],       // free parameters read by the active template
}
```

For Tier 1 you only toggle flags, with no new pipelines:

```rust
scene.set_material_class(material_id,
    0,                                          // class 0 = built-in PBR
    0,                                          // no graph snippet
    Some(FLAG_HAS_NORMAL_MAP | FLAG_ALPHA_TEST),
);
```

A Tier 2 template is a full WGSL file that defines `radiant_eval_surface(...) -> SurfaceData` with two marker comments that a graph snippet, if present, replaces. When there is no snippet, the markers are stripped and the template runs as authored.

```wgsl
// class_params.x = thin-film frequency, class_params.y = intensity
fn radiant_eval_surface(material: GpuMaterial,
                        material_tex: MaterialTextureData,
                        input: VertexOutput) -> SurfaceData {
    var s = default_pbr_surface(material, material_tex, input);

    // ... custom surface math, e.g. thin-film interference on s.f0 ...
    let film_freq = material.class_params.x;

    // RADIANT_OVERRIDE_SURFACE
    // RADIANT_OVERRIDE_END
    return s;
}
```

`crates/examples/shaders/radiant_iridescent.wgsl` is a complete example. Registering templates and graph snippets happens through the scene and the GBuffer pass: you register a compiled snippet with `scene.radiant_graphs.register(graph_hash, wgsl_source)`, load a template through the pass, then point a material at both.

```rust
use helio_pass_gbuffer::GBufferPass;

let reg = renderer.find_pass_mut::<GBufferPass>()
    .map(|p| p.template_registry_mut()).unwrap();
let template_id = reg.load_from_file("templates/clear_coat.wgsl").unwrap();
// or: let template_id = reg.register_str("iridescent", wgsl_source);

scene.set_material_class(material_id, template_id, graph_hash, Some(flags));
```

---

## Custom post-process shaders

The post-process pass (`helio-pass-postprocess`) runs an uber-pipeline of exposure, bloom, tone mapping, grain, vignette, and chromatic aberration, and it lets you inject WGSL at fixed points in that chain. This is how the `vhs_backrooms` demo layers a full camcorder look on top of the deferred image.

Each user effect is a WGSL function body that receives the current color and returns a new one; the engine wraps it in the signature `(color: vec3<f32>, uv: vec2<f32>, dims: vec2<f32>) -> vec3<f32>`. A trivial warm tint, for instance, is just `return color * vec3<f32>(1.0, 0.95, 0.9);`. You pick where it runs with a `UserEffectPosition`:

| Position | Runs |
|---|---|
| `PreBlend` | before exposure / bloom / color grade |
| `PostTonemap` | after tone map, before vignette / CA / grain |
| `PostGrain` | after grain, before DoF / motion blur |
| `Final` | after all built-in effects |

Inside the snippet you can read two engine-provided resources: `pp_custom`, a `array<vec4<f32>>` of per-frame parameters you upload (binding 14), and the tiling `noise_tex` / `noise_samp` pair for grain and dither.

There are two ways to inject. For a whole-frame effect at the `Final` position, pass the WGSL at graph-build time, which is what the VHS demo does:

```rust
const VHS: &str = include_str!("vhs_effects.wgsl"); // defines fn user_effects(color, uv, dims)

let graph = build_default_graph_with_user_effects(
    &device, &queue, &scene, config,
    debug_state, &debug_camera_buf, &cull_stats_buf,
    None,  // debug overlay
    VHS,   // injected WGSL
);
```

Alternatively, add effects to a live pass and commit them, which rebuilds the uber-pipeline:

```rust
use helio_pass_postprocess::{PostProcessPass, UserEffectPosition};

if let Some(pp) = renderer.find_pass_mut::<PostProcessPass>() {
    pp.add_user_effect(UserEffectPosition::PostTonemap, "return color * 0.85;");
    pp.commit_user_effects(&device);
}
```

Whichever route you take, drive the effect each frame by uploading the `vec4` parameters it reads from `pp_custom`:

```rust
if let Some(pp) = renderer.find_pass_mut::<PostProcessPass>() {
    pp.set_custom_params(&[
        [0.0, 0.12, 8.0, 0.2],   // pp_custom[0]: tape jitter, frequency, flicker
        [0.4, elapsed, 0.0, 0.0] // pp_custom[1]: grain amount, animation time
    ]);
}
```

Post-processing is gated by a `PostProcessVolume`, so for a global look you add one unbounded volume once. Keeping its built-in `settings` at default leaves the standard chain neutral so your injected WGSL owns the entire result:

```rust
scene.insert_actor(helio::SceneActor::post_process_volume(PostProcessVolumeDescriptor {
    bounds_min: [-1000.0; 3], bounds_max: [1000.0; 3],
    unbound: true, priority: 100.0, blend_weight: 1.0, blend_radius: 0.0,
    settings: PostProcessSettings::default(),
}));
```

`crates/helio-web-demos/examples-wasm/vhs_effects.wgsl` is a full camcorder shader with chromatic aberration, YIQ chroma drift, tracking noise, a head-switching bar, grain, and flicker.

---

## Custom render passes

A pass is any struct implementing `RenderPass` from `helio-core`. It declares the graph resources it reads and writes, and the graph validates the dependency graph at construction while managing pools and barriers. The `execute` method records commands and borrows scene resources zero-copy through `ctx.scene`, while the optional `prepare` method handles per-frame uploads and target resizing.

```rust
use helio_core::{RenderPass, PassContext, PrepareContext, Result};

struct MyPass { /* pipelines, buffers ... */ }

impl RenderPass for MyPass {
    fn name(&self) -> &'static str { "MyPass" }

    fn reads(&self)  -> &'static [&'static str] { &["gbuffer"] }
    fn writes(&self) -> &'static [&'static str] { &["pre_aa"] }

    fn prepare(&mut self, ctx: &PrepareContext) -> Result<()> {
        // Optional: upload per-frame uniforms, resize targets.
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        // Record commands; begin render/compute passes via ctx.begin_render_pass()
        // or ctx.begin_compute_pass(), and read scene data from ctx.scene.
        Ok(())
    }
}

graph.add_pass(Box::new(MyPass::new(&device)));
```

A pass opts into temporal jitter by overriding `requires_camera_jitter()`, which lets the renderer keep non-temporal graphs such as an FXAA-only pipeline pixel-stable by default. CPU and GPU profiling is injected automatically per pass, and debug builds catch resources that a pass declared but never wrote.

---

## Debug tools

Helio has an immediate-mode debug drawing API on the `Renderer`. With editor mode enabled (`set_editor_mode(true)`), call `debug_clear()` at the start of a frame and then any of `debug_line`, `debug_sphere`, `debug_circle`, `debug_torus`, `debug_cylinder`, `debug_cone`, and `debug_frustum`; the default graph's `DebugDrawPass` renders them. The `debug_shapes` demo is a live gallery of all of these.

At runtime, pressing F2 toggles the debug overlay, which shows FPS, frame timing, and any custom per-frame data you supply through its `populate` hook. F3 and F4 cycle through debug views such as UV, world normals, albedo, roughness, shadow heatmap, LOD heatmap, and overdraw.

---

## Pass reference

There are more than forty pass crates, each independently versioned. A representative selection:

| Crate | Pass | Role |
|---|---|---|
| `helio-pass-depth-prepass` | `DepthPrepassPass` | Early-Z, O(1) CPU |
| `helio-pass-gbuffer` | `GBufferPass` | GPU-driven G-buffer fill; Radiant material eval |
| `helio-pass-deferred-light` | `DeferredLightPass` | Cook-Torrance BRDF + shadows + GI + tone map |
| `helio-pass-shadow` | `ShadowPass` | Cascaded shadow atlas (PCF/PCSS) |
| `helio-pass-light-cull` | `LightCullPass` | Tile/cluster light culling |
| `helio-pass-radiance-cascades` | `RadianceCascadesPass` | Probe-based multi-bounce GI |
| `helio-pass-ssao` | `SsaoPass` | Screen-space ambient occlusion |
| `helio-pass-sky` / `helio-pass-sky-lut` | `SkyPass` | Hillaire 2020 atmosphere + clouds |
| `helio-pass-virtual-geometry` | `VirtualGeometryPass` | Meshlet cull + coverage LOD |
| `helio-pass-hiz` / `helio-pass-occlusion-cull` | Hi-Z + occlusion | Depth pyramid + GPU occlusion tests |
| `helio-pass-taa` | `TaaPass` | Temporal AA (jitter + reprojection) |
| `helio-pass-fxaa` / `helio-pass-smaa` | `FxaaPass` / `SmaaPass` | Spatial AA |
| `helio-pass-postprocess` | `PostProcessPass` | Exposure, bloom, tone map, grain, user WGSL |
| `helio-pass-voxel-mesh` / `helio-pass-voxel-raymarch` | Voxel | Meshlet surface extraction / DDA ray march |
| `helio-pass-water-*` | Water sim/surface/caustics | Simulation, surface, underwater |
| `helio-pass-transparent` | `TransparentPass` | Sorted forward transparency |
| `helio-pass-debug-overlay` / `helio-pass-perf-overlay` | Overlays | Text/graph overlay, GPU heatmaps |

---

## Examples and web demos

Native binaries live in `crates/examples`, and the same demos run in the browser through `crates/helio-web-demos`.

| Binary | Description |
|---|---|
| `indoor_cathedral` | Gothic nave with Radiance-Cascades GI and light shafts |
| `outdoor_city` | Dense night city with street lamps and beacons |
| `outdoor_canyon` | Desert canyon, `Q/E` rotates the sun |
| `space_station` | Orbiting station with solar arrays |
| `ship_flight` | 6-DoF ship through an asteroid field |
| `vhs_backrooms` | Procedural maze with an injected VHS post-process shader |
| `voxel_mesh_demo` | Editable voxel terrain via a custom `VoxelMeshPass` + FXAA graph |
| `debug_shapes` | Immediate-mode debug primitive gallery |
| `editor_demo` | Interactive scene editor: pick, translate, rotate, scale |
| `load_fbx` | Drop-in FBX/glTF/OBJ/USD viewer |
| `light_benchmark` | 128 animated point lights |
| `simple_graph` | Minimal fly-camera scene |

```sh
cargo run -p examples --bin indoor_cathedral --release
cargo run --bin web        # all of the above, in the browser
```

---

## Asset pipeline

Helio loads FBX, glTF, OBJ, and USD through `helio-asset-compat`, and `helio-bake` produces baked AO, lightmaps, reflection probes, and irradiance SH for static geometry, along with pre-computed potentially-visible sets for CPU-side culling. On the web, load assets from embedded bytes with `include_bytes!` rather than from disk, as the `load_fbx_embedded` demo shows.

---

## License

MIT, 2026 Tristan Poland. See [LICENSE](LICENSE).
