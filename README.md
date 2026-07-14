<div align="center">

<img src="./branding/Helio.svg" alt="Helio Renderer" width="400"/>

**GPU-driven deferred rendering for browser WebGPU, in Rust and WASM**

[![Rust](https://img.shields.io/badge/rust-stable-orange?logo=rust)](https://www.rust-lang.org/)
[![wgpu](https://img.shields.io/badge/wgpu-30-blue)](https://wgpu.rs/)
[![Target](https://img.shields.io/badge/target-browser%20WebGPU-purple)](https://www.w3.org/TR/webgpu/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

Helio is a browser engine built on the browser's native WebGPU implementation through
`wgpu` and `wasm32-unknown-unknown`. Culling, LOD selection, indirect draw generation,
lighting, shadows, GI, water, and post-processing run on the GPU. The public scene API
uses stable handles for meshes, materials, objects, lights, and other resources.

This repository intentionally has one graphics target.

| Platform/API | Status | Included implementation |
|---|---:|---|
| Browser + WASM + WebGPU | Target | `wgpu` browser WebGPU backend and WGSL |
| Browser + WebGL | Not supported | Removed |
| Vulkan | Not supported | Removed |
| Metal | Not supported | Removed |
| Direct3D 12 | Not supported | Removed |
| Native GLES | Not supported | Removed |
| Native Rust executables | Not supported | Removed |

Non-browser builds fail deliberately at compile time.

## WebGPU engine coverage

Browser WebGPU is the real renderer, not a preview or compatibility rasterizer.

| Engine area | Browser WebGPU status | Notes |
|---|---:|---|
| Deferred PBR / MRT G-buffer | Full | Four color targets plus depth |
| Compute passes | Full | Culling, Hi-Z, light bins, GI, SDF, water |
| GPU-generated indirect draws | Full | Requires `indirect-first-instance` |
| Virtual geometry | Full rendering | Meshlet culling and LOD remain GPU-side |
| Dynamic and cached shadows | Full rendering | Browser records individual indirect draws |
| Radiance Cascades GI | Full | Real-time path replaces baked lighting |
| SSAO, TAA, FXAA, SMAA | Full | Selected by render graph |
| Sky, corona, water, caustics | Full | Render and compute pipelines |
| Materials and mipmapping | Full, bounded table | 16 texture/sampler slots per scene bind group |
| CPU profiling | Full | Browser clock through `web-time` |
| GPU timestamps | Optional | Enabled only when the adapter exposes both required timestamp features |
| Hardware ray tracing | Not available | Not part of the retained WebGPU core path |
| Offline baking / PVS / snapshots | Removed | Real-time visibility and lighting only |

WebGPU does not expose wgpu's native multi-draw extensions. Helio therefore records one
`draw_indexed_indirect` command per GPU-generated slot. This preserves pixels and GPU-side
visibility decisions, but command encoding can cost more CPU time in very draw-heavy scenes.

## Vendored wgpu scope

`vendor/wgpu` is a source slice, not the upstream multi-platform workspace.

| Retained | Removed |
|---|---|
| High-level `wgpu` Rust API | `wgpu-core` and `wgpu-hal` |
| Browser WebGPU backend | Vulkan, Metal, DX12, GLES, WebGL, noop |
| Generated WebGPU JS bindings | Native platform integrations |
| `wgpu-types` and `naga-types` | Naga translators and CLI tooling |
| WGSL shader input | GLSL, SPIR-V, and Naga-IR inputs |
| Required licenses | Upstream examples, tests, benches, docs tooling |

Helio enables only:

```toml
wgpu = { path = "vendor/wgpu/wgpu", default-features = false, features = ["webgpu", "wgsl"] }
```

## Build and run

Install the Rust target and `wasm-bindgen-cli`, then build one demo:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli
./build-wasm.sh render_v2_basic
npx serve target/wasm-prebuilt
```

Open `http://localhost:3000/render_v2_basic/` in a WebGPU-capable browser. The page must
be served from a secure context (`localhost` is accepted); opening it directly as a file
will not work.

To build every retained demo:

```sh
./build-wasm.sh
```

For a fast Rust-side validation without `wasm-bindgen`:

```sh
cargo check --workspace --target wasm32-unknown-unknown
```

## Minimal application

Implement `HelioWasmApp`; `helio-wasm` owns the canvas, browser event loop, WebGPU adapter,
surface, and presentation lifecycle.

```rust
use std::sync::Arc;
use glam::Vec3;
use helio::{Camera, Renderer};
use helio_wasm::{HelioWasmApp, InputState};

struct App;

impl HelioWasmApp for App {
    fn init(
        _renderer: &mut Renderer,
        _device: Arc<wgpu::Device>,
        _queue: Arc<wgpu::Queue>,
        _width: u32,
        _height: u32,
    ) -> Self {
        Self
    }

    fn update(
        &mut self,
        _renderer: &mut Renderer,
        _dt: f32,
        _elapsed: f32,
        _input: &InputState,
    ) -> Camera {
        Camera::perspective_look_at(
            Vec3::new(0.0, 2.0, 6.0), Vec3::ZERO, Vec3::Y,
            60_f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0,
        )
    }
}

#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn run() {
    helio_wasm::launch::<App>();
}
```

The browser runner requests `wgpu::Backends::BROWSER_WEBGPU` explicitly. Device creation
requires `INDIRECT_FIRST_INSTANCE`; timestamp queries are requested only when supported.

## Architecture

| Path | Responsibility |
|---|---|
| `crates/helio` | Public renderer, scene, material, mesh, light, and editor APIs |
| `crates/helio-v3` | Render graph, pass contexts, GPU scene, profiling |
| `crates/libhelio` | GPU-shared layouts and browser material bindings |
| `crates/helio-pass-*` | Modular render and compute passes |
| `crates/helio-wasm` | Browser canvas, input, WebGPU initialization, frame loop |
| `crates/helio-web-demos` | Feature-selected WASM demos and generated HTML |
| `crates/helio-asset-compat` | Optional in-browser asset conversion |
| `vendor/wgpu` | Browser-only wgpu source slice |

The default frame is approximately:

```text
scene dirty uploads
  -> shadow matrices and shadow atlases
  -> depth prepass
  -> Hi-Z and occlusion/light culling compute
  -> G-buffer and virtual geometry
  -> deferred PBR + shadows + Radiance Cascades
  -> sky / transparency / water / billboards
  -> TAA and debug/performance overlays
  -> browser WebGPU surface presentation
```

## Materials

WebGPU baseline limits guarantee a finite number of sampled textures and samplers per shader
stage. Helio uses 16 fixed texture bindings and 16 sampler bindings, selected from WGSL with
explicit gradients so transformed UVs retain correct mip selection. Missing material textures
are filled with fallback views.

For scenes needing more than 16 simultaneously resident material textures, the next scaling
step is material paging, atlases, or compatible texture arrays; native unbounded binding arrays
are intentionally not part of this target.

## Asset loading

The base demos do not pull in SolidRS. Asset conversion is optional and enabled only for the
asset demos:

| Demo feature | Asset conversion |
|---|---:|
| `render_v2_basic` and procedural demos | No |
| `load_fbx_embedded` | Yes |
| `ship_flight` | Yes |
| `outdoor_rocks` | Yes |

Browser applications load embedded bytes or fetched bytes; there is no native filesystem path.

## Demo features

Each build selects one feature so unused scenes and optional import code are eliminated:

| Category | Features |
|---|---|
| Core | `render_v2_basic`, `render_v2_sky`, `simple_graph` |
| Indoor | `indoor_room`, `indoor_corridor`, `indoor_cathedral`, `indoor_server_room` |
| Outdoor | `outdoor_night`, `outdoor_canyon`, `outdoor_city`, `outdoor_volcano`, `outdoor_rocks` |
| Systems | `debug_shapes`, `light_benchmark`, `rc_benchmark`, `sdf_demo`, `editor_demo` |
| Assets / flight | `load_fbx`, `load_fbx_embedded`, `ship_flight`, `space_station` |

## License

[MIT](LICENSE). Vendored wgpu components retain their upstream MIT/Apache-2.0 licenses.
