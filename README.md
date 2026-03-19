<div align="center">
  <img src="./branding/Helio.svg" alt="Helio" width="360" />
</div>

Helio is a Rust rendering workspace built around `wgpu`. The repo currently contains three main layers:

- `helio-v3`: the low-level render-graph and pass runtime
- `helio`: a higher-level wrapper with stable handles, scene mutation helpers, and a forward renderer
- `helio-pass-*`: modular render-pass crates that plug into `helio-v3`

This repository is actively evolving. Some older examples and comments still reflect earlier APIs, so this README focuses on the code paths that exist in the workspace today.

## What is in this repo

The workspace is split into a few clear pieces.

### Core crates

| Crate | What it does |
| --- | --- |
| `crates/helio-v3` | Core render graph, `RenderPass` trait, profiling, GPU scene buffers, and zero-copy pass contexts |
| `crates/helio` | Higher-level wrapper over `helio-v3` with stable handles for meshes, materials, lights, textures, and objects |
| `crates/libhelio` | Shared GPU structs, frame resources, and common rendering data types |
| `crates/helio-asset-compat` | Asset loading bridge for FBX, glTF, OBJ, and USD via Solid3D |
| `crates/examples` | Native example binaries |
| `crates/helio-live-portal` | Live telemetry / dashboard integration |
| `crates/helio-wasm-app` | WASM-facing app crate |
| `crates/helio-wasm-examples` | WASM example harnesses |

### Pass crates

These are normal Rust crates. Each crate exposes one pass type that implements `helio_v3::RenderPass`.

Examples:

- `helio-pass-shadow` -> `ShadowPass`
- `helio-pass-gbuffer` -> `GBufferPass`
- `helio-pass-sky-lut` -> `SkyLutPass`
- `helio-pass-sky` -> `SkyPass`
- `helio-pass-ssao` -> `SsaoPass`
- `helio-pass-hiz` -> `HiZBuildPass`
- `helio-pass-billboard` -> `BillboardPass`
- `helio-pass-taa` -> `TaaPass`
- `helio-pass-fxaa` -> `FxaaPass`
- `helio-pass-smaa` -> `SmaaPass`
- `helio-pass-radiance-cascades` -> `RadianceCascadesPass`

Some pass crates are further along than others. A few are still placeholders or are being reshaped.

## The two main ways to use Helio

### 1. Use `helio` when you want a higher-level scene API

`helio` gives you a renderer plus stable IDs for scene resources. It is the easier entry point if you want to:

- insert meshes, materials, textures, lights, and objects
- update them through handles
- keep CPU-side scene mutation straightforward
- optionally attach a render graph with custom passes

The high-level type to start from is usually `helio::Renderer`.

### 2. Use `helio-v3` when you want to build the graph yourself

`helio-v3` is the lower-level layer. You work directly with:

- `GpuScene`
- `RenderGraph`
- `RenderPass`
- `PassContext`
- `PrepareContext`

This is the right level if you want explicit control over pass ordering and resource wiring.

## How passes are actually imported and used

This is the part that tends to be confusing at first:

Passes are not loaded automatically.

They are not discovered by filename.

They are not registered through a config file.

They are just Rust types from other crates that implement `helio_v3::RenderPass`.

You use them by adding the pass crate as a dependency, importing the pass type, constructing it, and then pushing it into a `RenderGraph` or into `helio::Renderer::add_pass(...)`.

### Step 1: add pass crates to `Cargo.toml`

```toml
[dependencies]
helio = { path = "crates/helio" }
helio-v3 = { path = "crates/helio-v3" }
helio-pass-shadow = { path = "crates/helio-pass-shadow" }
helio-pass-gbuffer = { path = "crates/helio-pass-gbuffer" }
helio-pass-sky-lut = { path = "crates/helio-pass-sky-lut" }
helio-pass-sky = { path = "crates/helio-pass-sky" }
```

### Step 2: import the pass types in Rust

```rust
use helio_v3::RenderGraph;
use helio_pass_gbuffer::GBufferPass;
use helio_pass_shadow::ShadowPass;
use helio_pass_sky::SkyPass;
use helio_pass_sky_lut::SkyLutPass;
```

### Step 3: construct the passes with the GPU resources they need

Pass constructors usually take explicit buffers or views. They are not magical: you pass in the camera buffer, instance buffer, shadow matrices, LUT views, target format, and so on.

```rust
let mut scene = helio_v3::GpuScene::new(device.clone(), queue.clone());
let mut graph = RenderGraph::new(&device, &queue);

let sky_lut = SkyLutPass::new(&device, scene.camera.buffer());
let sky = SkyPass::new(&device, scene.camera.buffer(), &sky_lut.sky_lut_view, surface_format);
let shadow = ShadowPass::new(&device, scene.shadow_matrices.buffer(), scene.instances.buffer());
let gbuffer = GBufferPass::new(
    &device,
    scene.camera.buffer(),
    scene.instances.buffer(),
    width,
    height,
);

graph.add_pass(Box::new(sky_lut));
graph.add_pass(Box::new(shadow));
graph.add_pass(Box::new(gbuffer));
graph.add_pass(Box::new(sky));
```

The key idea is that pass wiring happens in Rust code, not through a registry.

### Step 4: if you are using `helio::Renderer`, you can add passes there too

The `helio` wrapper exposes:

```rust
renderer.add_pass(Box<dyn RenderPass>)
```

Internally that creates or reuses a `RenderGraph` and appends the pass. The relevant entry point today is `crates/helio/src/renderer.rs`.

## Why the examples may not show this clearly

Not every example in this repository is demonstrating the same layer of the stack.

In particular:

- some examples use the `helio` wrapper directly
- some older examples still reference earlier renderer APIs
- pass crates live in the workspace as modular building blocks, but not every demo is a small “build your own graph” sample

So if you were looking for a single `main.rs` that imports every pass crate and wires the whole pipeline together, that is exactly what this README is trying to make explicit: the pass system exists, but it is composed in Rust by the application that chooses to use it.

## What Helio is aiming for architecturally

The current direction of the codebase is consistent in a few places:

- pass-oriented rendering through `helio-v3`
- zero-copy pass contexts where passes borrow GPU resources instead of cloning them
- dirty-tracked GPU scene buffers
- constant-time or bounded CPU-side scene updates where possible
- modular crates so features can be developed independently

That architecture shows up in files like:

- `crates/helio-v3/src/context.rs`
- `crates/helio-v3/src/graph/executor.rs`
- `crates/helio-v3/src/scene/gpu_scene.rs`
- `crates/helio-v3/src/scene/managers.rs`
- `crates/helio/src/renderer.rs`

## Running the examples

The example binaries live in `crates/examples/Cargo.toml`.

Some useful current entry points:

```powershell
cargo run -p examples --bin load_fbx
cargo run -p examples --bin load_fbx_embedded
cargo run -p examples --bin ship_flight
```

If you only want a compile check:

```powershell
cargo check -p helio --quiet
cargo check -p examples --bin load_fbx --bin load_fbx_embedded --bin ship_flight --quiet
```

## Asset loading

`helio-asset-compat` is the current bridge for scene import.

It loads data through Solid3D and converts it into Helio-friendly meshes, textures, and materials. At the moment that path is where FBX, glTF, OBJ, and USD support lives.

Relevant crate:

- `crates/helio-asset-compat`

## A practical mental model for the repo

If you are trying to understand the workspace quickly, this is the easiest way to think about it:

- `libhelio` defines shared rendering data
- `helio-v3` defines the render-graph runtime and pass interfaces
- `helio-pass-*` crates define individual rendering stages
- `helio` wraps the lower level in a more ergonomic scene API
- `examples` exercise different pieces of the stack

## Current rough edges

A few things are worth saying plainly:

- the repository is in transition
- not every crate is equally finished
- some examples and comments still describe earlier APIs
- the root README used to point at crates and commands that no longer match the current workspace

If you are trying to find the truth, trust the workspace manifests and the crate sources first.

## Where to start reading

If you want the big picture:

1. `crates/helio-v3/src/lib.rs`
2. `crates/helio-v3/src/graph/executor.rs`
3. `crates/helio-v3/src/context.rs`
4. `crates/helio/src/renderer.rs`
5. `crates/examples`

If you want to understand scene data and uploads:

1. `crates/helio/src/scene.rs`
2. `crates/helio/src/mesh.rs`
3. `crates/helio-v3/src/scene/managers.rs`
4. `crates/helio-v3/src/scene/gpu_scene.rs`

If you want to understand passes:

1. `crates/helio-pass-shadow`
2. `crates/helio-pass-gbuffer`
3. `crates/helio-pass-sky-lut`
4. `crates/helio-pass-sky`
5. `crates/helio-pass-ssao`

## License

This project is licensed under MIT or Apache-2.0.
