<div align="center">

<img src="./branding/Helio.svg" alt="Helio Renderer" width="400"/>

**GPU-driven deferred rendering in pure Rust — modular, pass-based, cross-platform**

[![Rust](https://img.shields.io/badge/rust-stable-orange?logo=rust)](https://www.rust-lang.org/)
[![wgpu](https://img.shields.io/badge/wgpu-30-blue)](https://wgpu.rs/)
[![WebGPU](https://img.shields.io/badge/WebGPU-native%20%2B%20browser-8A2BE2)](https://gpuweb.github.io/gpuweb/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

Helio is a GPU-driven deferred renderer built entirely in Rust on `wgpu`. Culling, LOD selection, indirect-draw dispatch, and light evaluation run on the GPU; every CPU-side call is bounded (typically O(1)). The pass architecture is strictly modular — each render pass is its own crate — and the **same renderer, scene, and render graph run unchanged on native desktop backends and in the browser via WebGPU**.

---

## Table of contents

- [Why Helio](#why-helio)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Cross-platform graphics](#cross-platform-graphics) — one source, native + web
- [The render graph](#the-render-graph) — building and customizing pipelines
- [Scene &amp; handle API](#scene--handle-api)
- [Custom material shaders (Radiant)](#custom-material-shaders-radiant)
- [Custom post-process shaders](#custom-post-process-shaders)
- [Custom render passes](#custom-render-passes)
- [Debug tools](#debug-tools)
- [Pass reference](#pass-reference)
- [Examples &amp; web demos](#examples--web-demos)
- [Asset pipeline](#asset-pipeline)

---

## Why Helio

**Truly modular pass architecture.** Every render pass is its own crate — `helio-pass-gbuffer`, `helio-pass-deferred-light`, `helio-pass-taa`, and 40+ more. The central `helio` crate has zero knowledge of any pass type. Adding a pass means writing a crate and plugging it into a graph builder; central crates never change.

**GPU-driven by default.** The CPU never iterates draw calls. Culling, LOD selection, and indirect-dispatch buffer generation all run on the GPU. Scene data lives in GPU buffers with dirty-tracked CPU mirrors — `flush()` uploads only what changed.

**Handle-based scene API.** Every resource (`MeshId`, `MaterialId`, `LightId`, `ObjectId`, …) is a lightweight `Copy` handle backed by generational arenas. Insert, update, or remove with O(1) operations and no aliasing.

**Render graph with automatic rebuild.** Graphs carry their own rebuilder closure, so resizing the window transparently rebuilds the pipeline — no manual boilerplate.

**Cross-platform from one source.** The renderer, scene, and graph builders are identical on native and web. WebGPU's smaller capability envelope (bindless limits, no multi-draw-indirect) is absorbed *inside* the engine — your code doesn't branch on target. A demo written against `HelioWasmApp` compiles to a native window and a browser canvas from the same file.

---

## Architecture

```
crates/helio                Public API: Renderer, Scene, Camera, Radiant, debug helpers
crates/helio-core           Render graph runtime, GpuScene, RenderPass trait, PassContext
crates/libhelio             GPU-shared POD types (GpuLight, GpuMaterial, uniforms)
crates/helio-default-graphs Pre-built graph configurations (deferred, FXAA, user-effects…)
crates/helio-pass-*         One crate per render pass (40+ passes)
crates/helio-wasm           Cross-platform app runner + HelioWasmApp trait (native + web)
crates/helio-web-demos      Every example compiled to WASM, one feature flag per demo
crates/helio-asset-compat   FBX / glTF / OBJ / USD loading
crates/helio-voxel-core     Shared voxel terrain component (mesh + ray-march)
crates/examples             Runnable native demos and editor
```

The separation between `helio` and the `helio-pass-*` crates is strict: **central crates never import pass types.** The `RenderPass` trait lives in `helio-core`; pass crates implement it; graph builders compose them and store a `GraphRebuilder` inside the graph. The `Renderer` extracts that at construction, giving automatic rebuild on resize with no dependency on specific pass types.

---

## Quick start

```sh
# Native demos
cargo run -p examples --bin indoor_cathedral --release
cargo run -p examples --bin outdoor_city --release
cargo run -p examples --bin load_fbx --release -- path/to/model.fbx

# All demos in the browser (builds WASM + serves on http://127.0.0.1:8000)
cargo run --bin web
```

### Minimal native setup

Build a renderer around any surface. The graph carries its own rebuilder, so resize handling and graph reconstruction are automatic.

```rust
use helio::{Camera, DebugDrawState, Renderer, RendererConfig, Scene,
            required_wgpu_features, required_wgpu_limits};
use helio_default_graphs::build_default_graph;

// Request exactly the features/limits Helio needs for this adapter.
let features = required_wgpu_features(adapter.features());
let limits   = required_wgpu_limits(adapter.limits());
// … create device/queue with those …

let config = RendererConfig::new(width, height, surface_format);
let scene  = Scene::new(device.clone(), queue.clone());
let debug_state = std::sync::Arc::new(std::sync::Mutex::new(DebugDrawState::default()));
// (debug_camera_buf / cull_stats_buf: small uniform/storage buffers — see examples)

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

If you want the same code to also run in the browser, don't hand-roll the windowing — use `HelioWasmApp` (next section), which wraps all of the above for both targets.

---

## Cross-platform graphics

Helio runs on native desktop backends (Vulkan / Metal / DX12 / GLES) **and** in the browser on WebGPU. This is not a separate port — it is the same renderer.

### What is identical vs adapted

| Layer | Native ↔ Web |
|---|---|
| **Graph building** (`RenderGraph::new`, `add_pass`, `lock`, `build_default_graph*`) | **Byte-for-byte identical.** Zero target cfgs in the graph layer. |
| **Renderer & Scene API** (`render`, `insert_material`, `find_pass_mut`, `debug_*`, …) | **Identical.** Same calls, same behavior. |
| **Pass internals** | Same API; the engine cfg-adapts internally (see below). Invisible to your code. |
| **App / windowing** | The only real difference — absorbed by the shared runner (`helio-wasm`). |

The renderer auto-adapts to WebGPU's smaller capability envelope **inside** the passes:

| Capability | Native | Web (WebGPU) |
|---|---|---|
| Bindless material textures (`MAX_TEXTURES`) | 256 | **16** (limits clamped automatically) |
| Multi-draw indirect | `multi_draw_indexed_indirect` (one call) | per-draw `draw_indexed_indirect` loop (automatic) |
| Required features | `TEXTURE_BINDING_ARRAY` + non-uniform indexing + `INDIRECT_FIRST_INSTANCE` | `INDIRECT_FIRST_INSTANCE` only |
| Present mode | backend-selected | `Fifo` |

> **Content ceiling, not a code fork.** "Same API" ≠ "same capabilities." A scene that stays within the web envelope (≤16 unique material textures per draw, etc.) looks identical on both. A scene that pushes past it can look different or fail on the web — that's a content limit, not divergent code.

Always create your device with the helper functions so you request exactly what Helio needs on each target:

```rust
let features = helio::required_wgpu_features(adapter.features()); // differs per target
let limits   = helio::required_wgpu_limits(adapter.limits());     // clamps MAX_TEXTURES
```

### One source, both targets: `HelioWasmApp`

`helio-wasm` provides the cross-platform app runner. Implement `HelioWasmApp` and call `launch::<T>()` — on native it runs a winit window, in the browser it attaches a WebGPU canvas. The runner owns the event loop, surface, input, and camera plumbing; you write scene + per-frame logic.

```rust
use std::sync::Arc;
use helio::{Camera, Renderer};
use helio_wasm::{HelioWasmApp, InputState, KeyCode, launch};

struct Demo { /* camera state, handles, … */ }

impl HelioWasmApp for Demo {
    fn title() -> &'static str { "My Demo" }

    fn init(renderer: &mut Renderer, _device: Arc<wgpu::Device>,
            _queue: Arc<wgpu::Queue>, _w: u32, _h: u32) -> Self {
        // Build the scene: meshes, materials, lights, ambient, clear color…
        renderer.set_ambient([0.4, 0.45, 0.5], 0.15);
        Demo { /* … */ }
    }

    fn update(&mut self, renderer: &mut Renderer, dt: f32, elapsed: f32,
              input: &InputState) -> Camera {
        // Read input, animate, return the camera for this frame.
        Camera::perspective_look_at(/* … */ input.aspect_ratio(), 0.1, 1000.0)
    }
}

// Native entry point — the very same type serves the browser build.
fn main() { launch::<Demo>(); }
```

Trait surface (everything but `init`/`update` has a default):

| Method | Purpose |
|---|---|
| `init(renderer, device, queue, w, h) -> Self` | Build the scene once. |
| `update(&mut self, renderer, dt, elapsed, input) -> Camera` | Per-frame logic; return the camera. |
| `title() -> &str` | Window / tab title. |
| `render_scale() -> f32` | Internal render resolution (default `0.75`; use `1.0` for graphs without TAA upscale). |
| `build_graph(device, queue, scene, config, debug_state, …) -> Option<RenderGraph>` | Return a **custom pipeline** (voxel meshing, injected post-process, …); `None` uses the default deferred graph. |
| `on_resize(&mut self, renderer, w, h)` | React to viewport changes. |
| `grab_cursor_button()` / `release_cursor_on_grab_button_release()` | Mouse-look capture behavior. |

`InputState` gives you `keys`, `mouse_delta`, `cursor_grabbed`, `mouse_left_just_pressed`, `cursor_pos`, `viewport_size`, and `aspect_ratio()`.

### Building for the web

Every example is compiled to WASM by `helio-web-demos`, one Cargo feature per demo, each launching a `HelioWasmApp`. The `web` binary is the build tool:

```sh
cargo run --bin web              # interactive TUI: builds all demos, then serves them
cargo run --bin web -- --headless  # CI mode: build all, write the site, exit non-zero on failure
```

It builds each demo with `wasm-pack` into `target/wasm-prebuilt/<demo>/`, writes each landing page + a master index, and (interactively) serves on `http://127.0.0.1:8000`. Headless mode is what CI runs — there is no shell build script. A wasm-capable `clang` is required for C dependencies (`meshopt`): on macOS `brew install llvm`, on Linux install `clang`.

---

## The render graph

A `RenderGraph` is an ordered list of passes with declared read/write resources. The graph validates the DAG, manages transient texture pools and barriers, and rebuilds itself on resize. You rarely build one by hand — a graph builder does it:

```rust
use helio_default_graphs::{
    build_default_graph,                 // full deferred pipeline
    build_default_graph_with_user_effects, // + injected post-process WGSL
};
```

To assemble a **custom pipeline** (the same code on native and web), construct a graph and add passes directly. This is exactly what `HelioWasmApp::build_graph` returns:

```rust
use helio::{RenderGraph, Renderer};
use helio_pass_voxel_mesh::VoxelMeshPass;
use helio_pass_fxaa::FxaaPass;

let mut graph = RenderGraph::new(device, queue);
graph.add_pass(Box::new(VoxelMeshPass::new(device, queue, config.surface_format)));
graph.add_pass(Box::new(FxaaPass::new(device, config.surface_format)));
graph.lock(config.width, config.height);
```

Passes communicate through named resources (`"pre_aa"`, `"gbuffer"`, …); a pass declares what it `reads()` and `writes()`, and the graph wires them. Reach into a live graph by pass type at any time:

```rust
if let Some(pass) = renderer.find_pass_mut::<FxaaPass>() { /* tweak it */ }
```

---

## Scene &amp; handle API

The `Scene` is GPU-native with dirty-tracked uploads. Everything is a `Copy` handle:

```rust
let scene = renderer.scene_mut();

let material = scene.insert_material(GpuMaterial { /* base_color, roughness_metallic, … */ });
let mesh     = scene.insert_actor(helio::SceneActor::mesh(mesh_upload));
let object   = scene.insert_actor(helio::SceneActor::object(ObjectDescriptor {
    mesh, material, transform, /* bounds, groups, movability, … */
}));
let light    = scene.insert_actor(helio::SceneActor::light(GpuLight { /* … */ }));
```

- **Groups** — a 64-bit `GroupMask` per object; hide/show/transform whole groups at once.
- **Sectioned meshes** — one vertex buffer + N index ranges (Unreal-style multi-material).
- **Voxel volumes** — `scene.insert_voxel_volume(VoxelVolumeDescriptor { … })`, shared by the mesh and ray-march voxel passes.
- **Post-process volumes** — `SceneActor::post_process_volume(…)` (see below).
- `scene.clear()` wipes everything; `set_ambient`, `set_clear_color`, `set_editor_mode`, `set_jitter_enabled` live on the `Renderer`.

---

## Custom material shaders (Radiant)

Radiant is Helio's material system: it combines a built-in PBR uber-shader, hand-authored **surface templates**, and graph-generated WGSL snippets. The GBuffer pass evaluates every material through a shared `radiant_eval_surface()` function that contains injection markers; custom templates and graph compilers splice code in without touching the engine.

### Three tiers, one cost model

| Tier | Mechanism | PSOs | Use case |
|------|-----------|------|----------|
| **1** | Feature flags on the built-in PBR uber-shader | 1 | ~95% of materials — just toggle flags |
| **2** | Hand-authored `.wgsl` surface template (+ optional graph snippet) | per template | Surface archetypes: clear coat, skin, hair, fabric, iridescence |
| **3** | Full custom WGSL via a graph compiler | per snippet | Unique surfaces that fit no template |

The GBuffer pass sorts instances by `(material_class, graph_hash)` at flush time and issues one draw per PSO. Shader modules are lazily compiled and cached by `(template_id, graph_hash, feature_flags)`. **The lighting pass is never touched** — every variant writes the same GBuffer format.

### The material struct

```rust
GpuMaterial {
    base_color:         [f32; 4],   // linear RGBA
    emissive:           [f32; 4],   // RGB + strength
    roughness_metallic: [f32; 4],   // x=roughness y=metallic z=IOR w=specular_tint
    tex_base_color, tex_normal, tex_roughness, tex_emissive, tex_occlusion: u32, // bindless indices
    workflow:       u32,
    flags:          u32,            // FLAG_HAS_NORMAL_MAP | FLAG_ALPHA_TEST | …  (Tier 1)
    material_class: u32,            // 0 = default PBR, 1+ = custom template (Tier 2/3)
    class_params:   [f32; 4],       // free parameters read by the active template
}
```

**Tier 1** is just flags — no new pipelines:

```rust
scene.set_material_class(material_id,
    0,                                          // class 0 = built-in PBR
    0,                                          // no graph snippet
    Some(FLAG_HAS_NORMAL_MAP | FLAG_ALPHA_TEST),
);
```

### Authoring a template (Tier 2)

A template is a full WGSL file that defines `radiant_eval_surface(...) -> SurfaceData` with two marker comments the graph snippet (if any) replaces:

```wgsl
// class_params.x = thin-film frequency, class_params.y = intensity
fn radiant_eval_surface(material: GpuMaterial,
                        material_tex: MaterialTextureData,
                        input: VertexOutput) -> SurfaceData {
    var s = default_pbr_surface(material, material_tex, input);

    // … custom surface math, e.g. thin-film interference on s.f0 …
    let film_freq = material.class_params.x;

    // RADIANT_OVERRIDE_SURFACE
    // RADIANT_OVERRIDE_END
    return s;
}
```

When no graph snippet is present, the markers are stripped and the template runs as-authored. See `crates/examples/shaders/radiant_iridescent.wgsl` for a complete example.

### Registering templates and graphs

```rust
use helio_pass_gbuffer::GBufferPass;

// Register a compiled graph snippet (typically emitted by your node-graph compiler)
scene.radiant_graphs.register(graph_hash, wgsl_source);

// Load a surface template through the GBuffer pass
let reg = renderer.find_pass_mut::<GBufferPass>()
    .map(|p| p.template_registry_mut()).unwrap();
let template_id = reg.load_from_file("templates/clear_coat.wgsl").unwrap();
// or: let template_id = reg.register_str("iridescent", wgsl_source);

// Point a material at the template (+ optional graph snippet, + optional flag override)
scene.set_material_class(material_id, template_id, graph_hash, Some(flags));
```

---

## Custom post-process shaders

The post-process pass (`helio-pass-postprocess`) runs an uber-pipeline (exposure → bloom → tone map → grain → vignette/CA → …) and lets you **inject WGSL** at fixed points in that chain. This is how the `vhs_backrooms` demo layers a full camcorder look on top of the deferred image.

### The injection contract

Each user effect is a WGSL function body spliced into the pipeline. It receives the current color and returns a new one:

```wgsl
// signature the engine wraps around your body:
//   (color: vec3<f32>, uv: vec2<f32>, dims: vec2<f32>) -> vec3<f32>
return color * vec3<f32>(1.0, 0.95, 0.9);  // e.g. warm tint
```

Choose where it runs with `UserEffectPosition`:

| Position | Runs |
|---|---|
| `PreBlend` | before exposure / bloom / color grade |
| `PostTonemap` | after tone map, before vignette / CA / grain |
| `PostGrain` | after grain, before DoF / motion blur |
| `Final` | after all built-in effects |

Your snippet can read two engine-provided resources:

- `pp_custom: array<vec4<f32>>` — per-frame parameters you upload (binding 14).
- `noise_tex` / `noise_samp` — a tiling blue-noise texture for grain/dither.

### Two ways to inject

**At graph-build time** (whole-frame effect, `Final` position) — used by the VHS demo:

```rust
const VHS: &str = include_str!("vhs_effects.wgsl"); // defines fn user_effects(color, uv, dims)

let graph = build_default_graph_with_user_effects(
    &device, &queue, &scene, config,
    debug_state, &debug_camera_buf, &cull_stats_buf,
    None,  // debug overlay
    VHS,   // injected WGSL
);
```

**At runtime** — add effects to a live pass and commit:

```rust
use helio_pass_postprocess::{PostProcessPass, UserEffectPosition};

if let Some(pp) = renderer.find_pass_mut::<PostProcessPass>() {
    pp.add_user_effect(UserEffectPosition::PostTonemap, "return color * 0.85;");
    pp.commit_user_effects(&device); // rebuilds the uber-pipeline
}
```

### Driving parameters per frame

Upload `vec4` params your shader reads from `pp_custom`:

```rust
if let Some(pp) = renderer.find_pass_mut::<PostProcessPass>() {
    pp.set_custom_params(&[
        [0.0, 0.12, 8.0, 0.2],   // pp_custom[0]: tape jitter, frequency, flicker…
        [0.4, elapsed, 0.0, 0.0] // pp_custom[1]: grain amount, animation time…
    ]);
}
```

### Scene-wide activation

Post-process is gated by a `PostProcessVolume`. For a global look, add an unbounded volume once:

```rust
scene.insert_actor(helio::SceneActor::post_process_volume(PostProcessVolumeDescriptor {
    bounds_min: [-1000.0; 3], bounds_max: [1000.0; 3],
    unbound: true, priority: 100.0, blend_weight: 1.0, blend_radius: 0.0,
    settings: PostProcessSettings::default(), // built-in chain stays neutral; your WGSL owns the look
}));
```

See `crates/helio-web-demos/examples-wasm/vhs_effects.wgsl` for a full camcorder shader (chromatic aberration, YIQ chroma drift, tracking noise, head-switching bar, grain, flicker).

---

## Custom render passes

A pass is any struct implementing `RenderPass` from `helio-core`. It declares the graph resources it reads/writes; the graph validates the DAG at construction and manages pools and barriers.

```rust
use helio_core::{RenderPass, PassContext, PrepareContext, Result};

struct MyPass { /* pipelines, buffers … */ }

impl RenderPass for MyPass {
    fn name(&self) -> &'static str { "MyPass" }

    fn reads(&self)  -> &'static [&'static str] { &["gbuffer"] }
    fn writes(&self) -> &'static [&'static str] { &["pre_aa"] }

    fn prepare(&mut self, ctx: &PrepareContext) -> Result<()> {
        // Optional: upload per-frame uniforms, resize targets.
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        // Record commands. Borrow scene resources zero-copy via ctx.scene;
        // begin render/compute passes via ctx.begin_render_pass()/begin_compute_pass().
        Ok(())
    }
}

graph.add_pass(Box::new(MyPass::new(&device)));
```

Passes opt into temporal jitter via `requires_camera_jitter()`; the renderer keeps non-temporal graphs (e.g. FXAA-only) pixel-stable by default. CPU/GPU profiling is automatic per pass, and debug builds catch unwritten resources.

---

## Debug tools

- **Immediate-mode debug drawing** — call `renderer.debug_clear()` then `debug_line`, `debug_sphere`, `debug_circle`, `debug_torus`, `debug_cylinder`, `debug_cone`, `debug_frustum` each frame (requires `set_editor_mode(true)`). Rendered by the default graph's `DebugDrawPass`. See the `debug_shapes` demo.
- **F2** — toggle the debug overlay (FPS, frame timing, custom per-frame data via a `populate` hook).
- **F3 / F4** — cycle debug views: UV, world normals, albedo, roughness, shadow heatmap, LOD heatmap, overdraw, and more.

---

## Pass reference

40+ pass crates, each independently versioned. A selection:

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
| `helio-pass-postprocess` | `PostProcessPass` | Exposure, bloom, tone map, grain, **user WGSL** |
| `helio-pass-voxel-mesh` / `helio-pass-voxel-raymarch` | Voxel | Meshlet surface extraction / DDA ray march |
| `helio-pass-water-*` | Water sim/surface/caustics | Simulation, surface, underwater |
| `helio-pass-transparent` | `TransparentPass` | Sorted forward transparency |
| `helio-pass-debug-overlay` / `helio-pass-perf-overlay` | Overlays | Text/graph overlay, GPU heatmaps |

---

## Examples &amp; web demos

Native binaries live in `crates/examples`; the same demos run in the browser via `crates/helio-web-demos`.

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
| `editor_demo` | Interactive scene editor — pick, translate, rotate, scale |
| `load_fbx` | Drop-in FBX/glTF/OBJ/USD viewer |
| `light_benchmark` | 128 animated point lights |
| `simple_graph` | Minimal fly-camera scene |

```sh
cargo run -p examples --bin indoor_cathedral --release
cargo run --bin web        # all of the above, in the browser
```

---

## Asset pipeline

FBX, glTF, OBJ, and USD via `helio-asset-compat`. Baked AO, lightmaps, reflection probes, and irradiance SH for static geometry (`helio-bake`); pre-computed potentially-visible sets for CPU-side culling. On the web, load assets from embedded bytes (`include_bytes!`) rather than disk — see the `load_fbx_embedded` demo.

---

## License

MIT © 2026 Tristan Poland. See [LICENSE](LICENSE).
