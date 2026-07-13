<div align="center">

<img src="./branding/Helio.svg" alt="Helio Renderer" width="400"/>

**GPU-driven deferred rendering in pure Rust — modular, pass-based, zero-overhead**

[![Rust](https://img.shields.io/badge/rust-stable-orange?logo=rust)](https://www.rust-lang.org/)
[![wgpu](https://img.shields.io/badge/wgpu-28-blue)](https://wgpu.rs/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

Helio is a GPU-driven deferred renderer built entirely in Rust on `wgpu`. Every CPU-side call is bounded (typically O(1)) while culling, LOD selection, indirect-draw dispatch, and light evaluation happen entirely on the GPU.

## What makes Helio special

**Truly modular pass architecture.** Every render pass is its own crate — `helio-pass-gbuffer`, `helio-pass-fxaa`, `helio-pass-taa`, and so on. The central `helio` crate has zero knowledge of any pass type. Adding a new pass means writing a crate and plugging it into a graph builder; central crates never change. This keeps the core small and makes experimentation trivial.

**GPU-driven by default.** The CPU never iterates draw calls. Culling, LOD selection, and indirect-dispatch buffer generation all run on the GPU. Scene data lives in GPU buffers with dirty-tracked CPU mirrors — `flush()` uploads only what changed.

**Handle-based scene API.** Every resource (`MeshId`, `MaterialId`, `LightId`, `ObjectId`, …) is a lightweight `Copy` stable handle backed by generational arenas. Insert, update, or remove objects with O(1) operations and no aliasing.

**Render graph with automatic rebuild.** Graphs carry their own rebuilder closure, so resizing the window transparently rebuilds the entire pipeline. No manual `set_rebuild` boilerplate needed — it just works.

---

## Architecture

```
crates/helio               Public API: Renderer, Scene, Camera, debug helpers
crates/helio-core          Render graph runtime, GpuScene, RenderPass trait
crates/libhelio            GPU-shared types (GpuLight, GpuMaterial, uniforms)
crates/helio-pass-*        One crate per render pass (30+ passes available)
crates/helio-default-graphs Pre-built graph configurations
crates/helio-asset-compat  FBX / glTF / OBJ / USD loading
crates/examples            Runnable demos and editor
```

The separation between `helio` (the central crate) and `helio-pass-*` (pass crates) is strict: **central crates never import pass types**. The `RenderPass` trait lives in `helio-core`. Pass crates implement it. Graph builder functions compose passes and store a `GraphRebuilder` inside the graph. The `Renderer` extracts it at construction time, giving automatic rebuild on resize without any dependency on specific pass types.

---

## Quick start

```sh
cargo run -p examples --bin indoor_cathedral --release
cargo run -p examples --bin outdoor_city --release
cargo run -p examples --bin load_fbx --release -- path/to/model.fbx
```

### Minimal setup

```rust
use helio::{Camera, DebugDrawState, Renderer, RendererConfig, Scene,
            required_wgpu_features, required_wgpu_limits};
use helio_default_graphs::build_default_graph;

let features = required_wgpu_features(adapter.features());
let limits   = required_wgpu_limits(adapter.limits());

let config = RendererConfig::new(width, height, surface_format);
let scene = Scene::new(device.clone(), queue.clone());
let debug_camera_buf = device.create_buffer(&wgpu::BufferDescriptor { … });
let cull_stats_buf = device.create_buffer(&wgpu::BufferDescriptor { … });
let debug_state = Arc::new(std::sync::Mutex::new(DebugDrawState::default()));
let graph = build_default_graph(&device, &queue, &scene, config,
    debug_state.clone(), &debug_camera_buf, &cull_stats_buf, None);
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

**The graph carries its own rebuilder.** You don't need to create a closure or call `set_rebuilder`. Resize handling, depth recreation, and graph reconstruction happen transparently.

---

## Features

### Rendering pipeline
- GPU-driven GBuffer — culling, LOD, indirect-draw dispatch all on GPU
- Virtual geometry — per-meshlet frustum/backface culling, coverage LOD
- Hi-Z occlusion culling — min-reduction pyramid + GPU occlusion tests
- Deferred lighting — Cook-Torrance BRDF, metallic-roughness, IOR, specular tint
- Cascaded shadow maps — 4-split CSM with PCF/PCSS, quality presets
- Tile/cluster light culling — O(tiles) evaluation
- Screen-space ambient occlusion
- Radiance Cascades GI — multi-bounce probe-based global illumination
- Volumetric sky — Hillaire 2020 model with clouds

### Post-processing
- TAA (temporal AA with jitter + reprojection)
- FXAA / FXAA+HLFS hybrid
- Tone mapping (integrated into deferred light pass)
- Debug visualisations — UV, normals, albedo, shadow heatmap, LOD heatmap

### Scene management
- Handle-based API — `MeshId`, `MaterialId`, `LightId`, `ObjectId`, etc.
- Group system — 64-bit bitmask per object; per-group hide/show/transform
- Sectioned meshes — single VB + N index ranges (Unreal-style multi-material)
- Reusable voxel terrain component shared by mesh and ray-march rendering
- GPU-native scene with dirty-tracked uploads

### Pass system
- 30+ pass crates, each independently versioned
- Render graph automatically rebuilds on resize (no explicit rebuilder needed)
- Debug-build tracking catches unwritten resources
- Automatic CPU/GPU profiling per pass
- Central crates have zero knowledge of pass types

---

## Pass reference

| Crate | Pass | Description |
|---|---|---|
| `helio-pass-depth-prepass` | `DepthPrepassPass` | Early-Z, O(1) CPU |
| `helio-pass-gbuffer` | `GBufferPass` | GPU-driven G-buffer fill |
| `helio-pass-deferred-light` | `DeferredLightPass` | BRDF + shadows + GI + tone map |
| `helio-pass-shadow` | `ShadowPass` | Shadow atlas |
| `helio-pass-sky` | `SkyPass` | Fullscreen atmospheric background |
| `helio-pass-virtual-geometry` | `VirtualGeometryPass` | Meshlet cull + LOD |
| `helio-pass-fxaa` | `FxaaPass` | Fullscreen FXAA |
| `helio-pass-taa` | `TaaPass` | Temporal AA |
| `helio-pass-hlfs` | `HlfsPass` | Hybrid lighting |
| `helio-pass-ssao` | `SsaoPass` | Screen-space ambient occlusion |
| `helio-pass-hiz` | `HiZBuildPass` | Hi-Z mip chain |
| `helio-pass-occlusion-cull` | `OcclusionCullPass` | GPU occlusion culling |
| `helio-pass-debug-overlay` | `DebugOverlayPass` | Text/graph overlay (F2) |
| `helio-pass-perf-overlay` | `PerfOverlayPass` | GPU performance heatmaps |

Full pass reference, debug view tables, GPU layout docs, and asset pipeline details are available in the crate documentation.

---

## Debug overlay

Press **F2** to toggle the debug overlay — shows FPS, frame timing, and optional user data. The overlay pass includes a `populate` callback hook for custom per-frame data. Press **F3** / **F4** to cycle through debug rendering views (UV, normals, shadow heatmap, LOD heatmap, etc.).

---

## Examples

| Binary | Description |
|---|---|
| `indoor_cathedral` | Gothic nave with RC GI, stained-glass light shafts |
| `indoor_cathedral_fxaa` | FXAA anti-aliasing variant |
| `indoor_cathedral_hlfs` | Hybrid lighting variant |
| `outdoor_city` | Dense city block at dusk |
| `outdoor_canyon` | Desert canyon, `Q/E` rotates sun |
| `space_station` | Massive orbital station |
| `load_fbx` | Drop-in FBX/glTF/OBJ/USD viewer |
| `editor_demo` | Interactive scene editor — pick, translate, scale |
| `editor_demo_mini` | Compact editor with FXAA |
| `light_benchmark` | 150 simultaneous point lights |
| `voxel_demo` | Editable voxel terrain rendered as GPU-generated meshlets |
| `voxel_demo_raymarch` | The same terrain rendered with per-pixel DDA ray marching |
| `simple_graph` | Minimal single-pass example |

```sh
cargo run -p examples --bin indoor_cathedral --release
cargo run -p examples --bin load_fbx --release -- path/to/model.fbx
```

---

## Asset pipeline

FBX, glTF, OBJ, and USD support via `helio_asset_compat`. Baked AO, lightmaps, reflection probes, and irradiance SH for static geometry. Pre-computed potentially-visible sets for CPU-side culling.

---

## Custom passes

Each pass is a struct implementing the `RenderPass` trait from `helio-core`. Passes register the resources they read and write; the graph validates the DAG at construction time and automatically manages texture pools and barriers.

```rust
use helio_core::{PrepareContext, PassContext, RenderPass, Result};

struct MyPass { … }

impl RenderPass for MyPass {
    fn prepare(&mut self, ctx: &PrepareContext) -> Result<()> { … }
    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> { … }
}

// Add to a graph builder:
graph.add_pass(Box::new(MyPass::new(&device)));
```

---

## License

MIT
