<div align="center">

<img src="./branding/Helio.svg" alt="Helio Renderer" width="400"/>

**GPU-driven deferred rendering in pure Rust, modular and cross-platform**

[![Rust](https://img.shields.io/badge/rust-stable-orange?logo=rust)](https://www.rust-lang.org/)
[![wgpu](https://img.shields.io/badge/wgpu-30-blue)](https://wgpu.rs/)
[![WebGPU](https://img.shields.io/badge/WebGPU-native%20%2B%20browser-8A2BE2)](https://gpuweb.github.io/gpuweb/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

Helio is a GPU-driven deferred renderer written entirely in Rust on top of `wgpu`. The idea that runs through the whole project is that the GPU should do the heavy lifting. Culling, level-of-detail selection, indirect-draw dispatch, and light evaluation all happen on the GPU, and the CPU side of every frame stays bounded, usually constant time regardless of how much is on screen. It is built out of small, independent pieces, and the exact same renderer runs on a native desktop backend and in a browser through WebGPU without you writing two versions of anything.

This README is the tour: how the engine is put together, how to draw a frame, how the native and web builds share one codebase, and how to push past the defaults with your own material shaders, post-process effects, and render passes.

## How the engine is put together

The thing that shapes everything else is that every render pass is its own crate. The G-buffer fill, the deferred lighting, temporal anti-aliasing, the sky, the water simulation, and forty-odd others each live in a `helio-pass-*` crate, and the central `helio` crate has no idea any of them exist. A pass is just a struct that implements a trait from `helio-core` and declares which named resources it reads and writes. A graph builder strings passes together into a pipeline. Adding a brand new effect never means editing the core; you write a crate and drop it into a graph. That constraint keeps the middle of the engine small and makes it genuinely pleasant to experiment in.

Because the GPU drives the draws, the CPU never loops over draw calls. Scene data lives in GPU buffers, with dirty-tracked mirrors on the CPU side, so when you change one object and call `flush()` only that object's bytes get uploaded. The scene itself is handle-based: a mesh, a material, a light, or an object is a small `Copy` handle backed by a generational arena, and inserting, updating, or removing any of them is a constant-time operation with no dangling references to worry about.

The render graph is a little smart about its own lifecycle. When you build one, the builder tucks a rebuilder closure inside it, and the renderer pulls that out at construction time. The practical result is that resizing the window rebuilds the whole pipeline, recreates depth targets, and rewires everything for you, with no resize boilerplate in your code.

If you want to see where things live: `helio` is the public API you program against (the renderer, scene, camera, the Radiant material system, debug helpers), `helio-core` is the graph runtime and the `RenderPass` trait, `libhelio` holds the plain GPU-shared structs like `GpuLight` and `GpuMaterial`, `helio-default-graphs` has the ready-made pipelines, the `helio-pass-*` crates are the passes, `helio-wasm` is the cross-platform app runner, `helio-web-demos` compiles every example to WebAssembly, and `helio-asset-compat` handles model loading. The runnable native demos and the editor are in `crates/examples`.

## Drawing a frame

The fastest way to see something is to run one of the demos:

```sh
cargo run -p examples --bin indoor_cathedral --release
cargo run -p examples --bin outdoor_city --release
cargo run --bin web                # build every demo to WASM and serve it locally
```

To wire the renderer up to your own window, the shape of it is always the same. You ask Helio which GPU features and limits it needs for the adapter you have, create a config and a scene, build a graph, and hand all of it to `Renderer::new`. From then on you call `render` once per frame with a camera and a surface view.

```rust
use helio::{Camera, DebugDrawState, Renderer, RendererConfig, Scene,
            required_wgpu_features, required_wgpu_limits};
use helio_default_graphs::build_default_graph;

let features = required_wgpu_features(adapter.features());
let limits   = required_wgpu_limits(adapter.limits());
// ... create your device and queue with those ...

let config = RendererConfig::new(width, height, surface_format);
let scene  = Scene::new(device.clone(), queue.clone());
let debug_state = std::sync::Arc::new(std::sync::Mutex::new(DebugDrawState::default()));

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

That is the low-level path, and it is worth understanding once. In practice, if you want your code to also run in the browser, you should not hand-roll the windowing at all. That is what the next section is about.

## One codebase, native and web

Helio runs on the native backends (Vulkan, Metal, DX12, GLES) and in the browser on WebGPU, and this is not a port that drifts out of sync. It is the same renderer driven through the same graph. The parts you actually touch are identical on both targets. Building a graph, adding passes, locking it to a size, inserting materials and lights, calling `render`, drawing debug shapes: none of that has any target-specific code in it. The differences that do exist live down inside the passes, where the engine handles them for you, and up in the windowing layer, which a shared runner handles for you.

The one thing worth carrying in your head is that WebGPU is a smaller target than a native driver, and Helio quietly adapts to it. On a desktop backend a material can reference up to two hundred and fifty-six bindless textures; on the web that ceiling is sixteen, and the limits are clamped for you automatically. Where native issues a single multi-draw-indirect call, the web build loops and issues one indirect draw at a time, because WebGPU has no multi-draw. Native asks the adapter for bindless texture arrays and non-uniform indexing; the web build only requires the indirect-first-instance feature. You do not write any of this yourself, but it does mean "same API" is not quite "same capabilities." A scene that stays inside the web envelope looks identical in both places, and a scene that blows past it (hundreds of unique material textures in a single draw, say) can look different or fail to start in the browser. That is a content budget, not a fork in the code. The way you stay on the right side of it is to always build your device through `required_wgpu_features` and `required_wgpu_limits`, which request exactly what Helio needs for whichever target you are on.

The abstraction that makes single-source demos work is a trait called `HelioWasmApp` in the `helio-wasm` crate. You implement it and call `launch::<T>()`, and on native that spins up a winit window while in the browser it attaches a WebGPU canvas. The runner owns the event loop, the surface, the input, and the camera plumbing, and you are left with the two methods that actually matter: `init`, where you build the scene once, and `update`, where you read input, animate, and hand back the camera for this frame.

```rust
use std::sync::Arc;
use helio::{Camera, Renderer};
use helio_wasm::{HelioWasmApp, InputState, launch};

struct Demo { /* camera state, handles, whatever you need */ }

impl HelioWasmApp for Demo {
    fn title() -> &'static str { "My Demo" }

    fn init(renderer: &mut Renderer, _device: Arc<wgpu::Device>,
            _queue: Arc<wgpu::Queue>, _w: u32, _h: u32) -> Self {
        renderer.set_ambient([0.4, 0.45, 0.5], 0.15);
        // build meshes, materials, lights here
        Demo { /* ... */ }
    }

    fn update(&mut self, renderer: &mut Renderer, dt: f32, elapsed: f32,
              input: &InputState) -> Camera {
        // read input.keys / input.mouse_delta, move the camera, return it
        Camera::perspective_look_at(/* ... */ input.aspect_ratio(), 0.1, 1000.0)
    }
}

fn main() { launch::<Demo>(); }
```

Everything else on the trait has a default, so you only override what you care about. You can set the window title, choose an internal render scale (it defaults to three-quarters resolution and upscales, which you would set back to `1.0` for a pipeline that has no temporal upscale step), adjust the mouse-look capture behaviour, react to resizes, and, most powerfully, return a completely custom render graph from `build_graph`. That last one is how the voxel and VHS demos plug their own pipelines in while still running on both targets; returning `None` just uses the standard deferred graph. The `InputState` you get each frame carries the held keys, the mouse delta, whether the cursor is grabbed, a one-frame left-click edge, the cursor position, the viewport size, and an `aspect_ratio()` helper.

Building the web version is its own small tool rather than a shell script. Running `cargo run --bin web` opens a terminal UI that builds every demo to WebAssembly and then serves the lot on a local port, and `cargo run --bin web -- --headless` does the same thing without the UI, writes the finished site out, and exits with a failure code if any demo did not build. Headless mode is what continuous integration runs. Under the hood it invokes `wasm-pack` per demo and writes each landing page plus a master index. The one prerequisite is a wasm-capable `clang` for the C dependencies, which means installing LLVM (`brew install llvm` on macOS, or your distribution's `clang` package on Linux).

## Building pipelines with the render graph

A render graph is an ordered set of passes, each declaring the named resources it reads and produces, and the graph validates that dependency structure, manages the transient textures and barriers between passes, and rebuilds itself when the window changes size. Most of the time you never construct one directly, because a builder does it: `build_default_graph` gives you the full deferred pipeline, and `build_default_graph_with_user_effects` gives you the same thing with a slot for injected post-process WGSL.

When you do want something bespoke, you build the graph yourself, and it reads exactly the same whether it ends up on a desktop or in a browser. This snippet, for instance, is the entire voxel pipeline, a mesh-extraction pass feeding an FXAA pass:

```rust
use helio::RenderGraph;
use helio_pass_voxel_mesh::VoxelMeshPass;
use helio_pass_fxaa::FxaaPass;

let mut graph = RenderGraph::new(device, queue);
graph.add_pass(Box::new(VoxelMeshPass::new(device, queue, config.surface_format)));
graph.add_pass(Box::new(FxaaPass::new(device, config.surface_format)));
graph.lock(config.width, config.height);
```

Passes talk to each other through resource names like `"gbuffer"` and `"pre_aa"`; a pass says what it reads and what it writes, and the graph connects the wires. Once a graph is running you can reach back into it and grab any pass by its type with `renderer.find_pass_mut::<FxaaPass>()`, which is how you tweak a pass's settings or feed it per-frame data.

## Working with the scene

The scene is GPU-native and everything in it is a handle. You insert a material and get a `MaterialId`, insert a mesh and get a handle, insert an object that points at both, and insert lights. You hold onto those handles to update or remove things later, and the uploads are dirty-tracked so nothing you did not change gets re-sent.

```rust
let scene = renderer.scene_mut();

let material = scene.insert_material(GpuMaterial { /* base_color, roughness_metallic, ... */ });
let mesh     = scene.insert_actor(helio::SceneActor::mesh(mesh_upload));
let object   = scene.insert_actor(helio::SceneActor::object(ObjectDescriptor {
    mesh, material, transform, /* bounds, groups, movability, ... */
}));
let light    = scene.insert_actor(helio::SceneActor::light(GpuLight { /* ... */ }));
```

There is more under the surface when you need it. Every object carries a sixty-four-bit group mask, so you can hide, show, or transform whole groups of objects in one call. Meshes can be split into sections, one vertex buffer with several index ranges, which is the Unreal-style way of putting several materials on one model. Voxel volumes go in through `insert_voxel_volume` and are shared by both the meshing and ray-marching voxel passes. Whole-scene knobs like ambient light, the clear color, editor mode, and temporal jitter live on the renderer as `set_ambient`, `set_clear_color`, `set_editor_mode`, and `set_jitter_enabled`, and `scene.clear()` wipes the slate.

## Writing your own material shaders

Materials in Helio go through a system called Radiant, which is a deliberate middle path between "one fixed shader for everything" and "every material is a bespoke pipeline." It blends a built-in physically based shader, hand-authored surface templates, and WGSL snippets generated by an external graph compiler, and it does it by having the G-buffer pass evaluate every material through a single shared function with marked injection points. Custom code splices in at those markers without the engine ever knowing about it.

There are three tiers, and they share one cost model. The overwhelming majority of materials never leave the first tier, which is just feature flags on the built-in shader; you flip a normal-map bit or an alpha-test bit and nothing new gets compiled. The second tier is for surface archetypes that genuinely behave differently, things like clear coat, skin, hair, fabric, or thin-film iridescence. You write a small WGSL template for the surface, optionally pair it with a graph snippet, and pay for exactly one pipeline per template. The third tier is a fully custom surface emitted by a graph compiler, one pipeline per unique snippet, for the surfaces that fit no template at all. Whichever tier a material uses, the G-buffer pass sorts instances by their material class and graph hash, issues one draw per pipeline, and caches the compiled shader by the combination of template, graph, and flags. Crucially the lighting pass never changes, because every variant writes the same G-buffer format.

A material is a plain struct. Its base color, emissive, and packed roughness/metallic/IOR/tint values are the ordinary PBR inputs, the texture fields are bindless indices, the `flags` field drives the first tier, `material_class` selects a template (zero being the built-in shader), and `class_params` is four free floats that whatever template is active can interpret however it likes.

```rust
GpuMaterial {
    base_color:         [f32; 4],   // linear RGBA
    emissive:           [f32; 4],   // RGB + strength
    roughness_metallic: [f32; 4],   // x = roughness, y = metallic, z = IOR, w = specular tint
    tex_base_color, tex_normal, tex_roughness, tex_emissive, tex_occlusion: u32,
    workflow:       u32,
    flags:          u32,            // FLAG_HAS_NORMAL_MAP | FLAG_ALPHA_TEST | ...   (tier 1)
    material_class: u32,            // 0 = built-in PBR, 1+ = a template            (tier 2/3)
    class_params:   [f32; 4],       // free parameters read by the active template
}
```

Staying in the first tier is a single call that just toggles flags:

```rust
scene.set_material_class(material_id, 0, 0, Some(FLAG_HAS_NORMAL_MAP | FLAG_ALPHA_TEST));
```

A tier-two template is a full WGSL file that defines `radiant_eval_surface`, returning the surface data the rest of the pipeline consumes. It contains two marker comments, and if a graph snippet is attached, the compiler replaces everything between them; if not, the markers are simply stripped and your template runs as written. The example below reads two of the free `class_params` to drive a thin-film effect, and `crates/examples/shaders/radiant_iridescent.wgsl` is a complete working version.

```wgsl
fn radiant_eval_surface(material: GpuMaterial,
                        material_tex: MaterialTextureData,
                        input: VertexOutput) -> SurfaceData {
    var s = default_pbr_surface(material, material_tex, input);

    let film_freq = material.class_params.x;
    // ... your surface math, e.g. thin-film interference on s.f0 ...

    // RADIANT_OVERRIDE_SURFACE
    // RADIANT_OVERRIDE_END
    return s;
}
```

Registering templates and graphs is a short bit of setup. You register a compiled snippet on the scene with `scene.radiant_graphs.register(graph_hash, wgsl_source)`, load a template through the G-buffer pass, and then point a material at both the template and the snippet.

```rust
use helio_pass_gbuffer::GBufferPass;

let reg = renderer.find_pass_mut::<GBufferPass>()
    .map(|p| p.template_registry_mut()).unwrap();
let template_id = reg.load_from_file("templates/clear_coat.wgsl").unwrap();
// or reg.register_str("iridescent", wgsl_source);

scene.set_material_class(material_id, template_id, graph_hash, Some(flags));
```

## Writing your own post-process shaders

The post-process pass runs the usual chain of exposure, bloom, tone mapping, grain, vignette, and chromatic aberration, and it lets you splice your own WGSL into fixed points along that chain. This is how the backrooms demo lays a full VHS camcorder look over the rendered image.

An effect is a WGSL function body that takes the current color and returns a new one. The engine wraps it in the signature `(color: vec3<f32>, uv: vec2<f32>, dims: vec2<f32>) -> vec3<f32>`, so the simplest possible effect is just `return color * vec3<f32>(1.0, 0.95, 0.9);` for a warm tint. You choose where in the chain it runs by position: before the blend stage, after the tonemap, after the grain, or right at the end after every built-in effect. Inside the snippet you have access to two things the engine provides, a storage array called `pp_custom` full of `vec4` parameters you upload each frame, and a tiling noise texture and sampler for grain and dithering.

There are two ways to get your WGSL in. For a whole-frame effect that runs at the end, you pass it at graph-build time, which is what the VHS demo does:

```rust
const VHS: &str = include_str!("vhs_effects.wgsl");

let graph = build_default_graph_with_user_effects(
    &device, &queue, &scene, config,
    debug_state, &debug_camera_buf, &cull_stats_buf,
    None,   // debug overlay
    VHS,    // your injected WGSL
);
```

Or you can add effects to a live pass at runtime and commit them, which rebuilds the pipeline:

```rust
use helio_pass_postprocess::{PostProcessPass, UserEffectPosition};

if let Some(pp) = renderer.find_pass_mut::<PostProcessPass>() {
    pp.add_user_effect(UserEffectPosition::PostTonemap, "return color * 0.85;");
    pp.commit_user_effects(&device);
}
```

Either way, you drive the effect frame to frame by uploading the parameters it reads out of `pp_custom`:

```rust
if let Some(pp) = renderer.find_pass_mut::<PostProcessPass>() {
    pp.set_custom_params(&[
        [0.0, 0.12, 8.0, 0.2],    // tape jitter, frequency, flicker
        [0.4, elapsed, 0.0, 0.0], // grain amount, animation time
    ]);
}
```

The whole chain is gated behind a post-process volume, so to get a global effect you insert one unbounded volume once. Leaving its built-in settings at their defaults keeps the standard chain neutral, which lets your injected shader own the entire look:

```rust
scene.insert_actor(helio::SceneActor::post_process_volume(PostProcessVolumeDescriptor {
    bounds_min: [-1000.0; 3], bounds_max: [1000.0; 3],
    unbound: true, priority: 100.0, blend_weight: 1.0, blend_radius: 0.0,
    settings: PostProcessSettings::default(),
}));
```

For a real, non-trivial example, `crates/helio-web-demos/examples-wasm/vhs_effects.wgsl` is a complete camcorder shader with chromatic aberration, YIQ chroma drift, tracking noise, a head-switching bar at the bottom of frame, grain, and flicker.

## Writing your own render passes

If you need to do something the existing passes do not, you write your own. A pass is any struct that implements the `RenderPass` trait from `helio-core`. It gives itself a name, tells the graph which resources it reads and writes, and does its work in `execute`, where it records GPU commands and reads scene data straight out of the context with no copies. There is an optional `prepare` step for per-frame uploads and resizing.

```rust
use helio_core::{RenderPass, PassContext, PrepareContext, Result};

struct MyPass { /* pipelines, buffers, ... */ }

impl RenderPass for MyPass {
    fn name(&self) -> &'static str { "MyPass" }

    fn reads(&self)  -> &'static [&'static str] { &["gbuffer"] }
    fn writes(&self) -> &'static [&'static str] { &["pre_aa"] }

    fn prepare(&mut self, ctx: &PrepareContext) -> Result<()> { Ok(()) }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        // begin a render or compute pass on ctx, set pipelines and bind groups,
        // read scene resources from ctx.scene, issue your draws or dispatches
        Ok(())
    }
}

graph.add_pass(Box::new(MyPass::new(&device)));
```

A pass that reconstructs temporal data opts into camera jitter, which lets the renderer keep non-temporal pipelines pixel-stable by default. Profiling, both CPU and GPU, is injected around every pass automatically, and debug builds will tell you if a pass declared a resource it never actually wrote.

## Debugging

Helio has an immediate-mode debug drawing API right on the renderer. Turn on editor mode with `set_editor_mode(true)`, call `debug_clear()` at the top of a frame, and then draw lines, spheres, circles, tori, cylinders, cones, and frustums by calling the matching `debug_*` methods; the default graph's debug pass renders them. The `debug_shapes` demo is a live gallery of every primitive. While a demo is running, F2 toggles an overlay showing frame rate and timing (with a hook for your own per-frame text), and F3 and F4 cycle through debug views like UVs, world-space normals, albedo, roughness, the shadow heatmap, the LOD heatmap, and overdraw.

## What the passes give you

The full pipeline is assembled out of the pass crates, and it is worth knowing roughly what is in the box. Geometry goes through an early depth prepass and a GPU-driven G-buffer fill that also evaluates Radiant materials, with meshlet-level virtual geometry culling and a hierarchical-Z occlusion system keeping hidden triangles off the GPU. Lighting is deferred, using a Cook-Torrance BRDF with tile and cluster light culling so hundreds of lights stay cheap, cascaded shadow maps with soft filtering, screen-space ambient occlusion, and a Radiance Cascades global illumination pass for multi-bounce indirect light. The sky is a Hillaire atmospheric model with volumetric clouds. Anti-aliasing comes in temporal and spatial flavours (TAA, FXAA, SMAA), and the post-process pass handles exposure, bloom, tonemapping, and the user WGSL described above. On top of all that there are specialised passes for voxel terrain (both a meshing path and a per-pixel ray-marching path), water simulation and surface rendering with caustics and an underwater look, sorted forward transparency, and the debug and performance overlays. Every one of these is an independent crate, and you compose only the ones a given pipeline needs.

## Examples and demos

The `crates/examples` directory has the native binaries, and the very same demos run in the browser through `helio-web-demos`. There is an indoor cathedral lit by Radiance Cascades global illumination and shafts of stained-glass light, a dense night city, a desert canyon where Q and E rotate the sun, an orbital space station, a six-degree-of-freedom ship flying through an asteroid field, the VHS backrooms with its injected post-process shader, editable voxel terrain rendered through a custom mesh-plus-FXAA graph, the debug shapes gallery, an interactive editor that picks and moves objects, a drop-in FBX/glTF/OBJ/USD viewer, a benchmark pushing a hundred and twenty-eight animated point lights, and a bare-bones fly-camera scene to start from. Run any of them natively with `cargo run -p examples --bin <name> --release`, or run `cargo run --bin web` to build them all for the browser at once.

## Assets

Model loading covers FBX, glTF, OBJ, and USD through `helio-asset-compat`. For static geometry there is a baking crate that precomputes ambient occlusion, lightmaps, reflection probes, and irradiance spherical harmonics, along with potentially-visible sets for CPU-side culling. On the web you load assets from bytes embedded at compile time with `include_bytes!` rather than from disk, which the `load_fbx_embedded` demo shows.

## License

Helio is MIT licensed. Copyright 2026 Tristan Poland. See [LICENSE](LICENSE) for the full text.
