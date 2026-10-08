# Composing a custom Helio pass pipeline

This guide covers how to assemble multiple `RenderPass` implementations into a graph, supply it to Helio, and preserve its resource and resize behavior. The graph API is in `crates/helio-core/src/graph/execution.rs`; high-level graph construction uses `PassBuildContext` from `crates/helio/src/renderer/builder.rs`.

## Pick the composition point

- **Add an effect to an existing 3D pipeline:** use the relevant `helio-default-graphs` builder or its pass-factory arguments. Those builders add passes at defined pipeline stages, declare common external inputs, and install a graph rebuilder.
- **Build a standalone/custom pipeline:** create a `RenderGraph`, add each pass, declare host-provided resources, then `lock` it at the intended render dimensions.
- **Use Helio's `Renderer`:** provide a graph-building closure with `RendererBuilder::with_pass_build_context`. The closure receives the device, queue, renderer config, camera/debug buffers, and the frontend's GPU-only SceneDB handle. The closure builds the graph before the renderer starts; it is not a per-frame callback.

The full deferred pipeline builders are in `crates/helio-default-graphs/src/lib.rs`. `hlfs_viewer.rs` demonstrates wrapping a default graph builder to adjust a pass, and `helio-wasm/src/lib.rs` documents the custom-graph hook for wasm applications.

## Build a graph before locking it

Every pass must be added before `lock`. Locking collects declarations, validates resource dependencies, allocates transient textures, prepares pipeline registries, detects render-pass chains, and creates reusable render bundles. Adding a pass after lock with `add_pass` panics. Use `add_pass_live` only for intentional runtime graph mutation; it relocks the graph and schedules resources for the new pass.

```rust
use helio_core::RenderGraph;

fn build_graph(
    device: &std::sync::Arc<wgpu::Device>,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
) -> RenderGraph {
    let mut graph = RenderGraph::new(device, queue);

    graph.add_pass(Box::new(UploadOrSimulationPass::new(device)));
    graph.add_pass(Box::new(SceneDrawPass::new(device)));
    graph.add_pass(Box::new(ToneMapPass::new(device)));

    graph.lock(width.max(1), height.max(1));
    graph
}
```

`RenderGraph::new` marks the graph as owning its device. If the application owns the device, construct with `RenderGraph::new_with_external_device` instead. Match this choice to `RendererBuilder::with_external_device`; the high-level default graph builders use `PassBuildContext::owns_device` for this reason.

## Express the edges between passes

Pass list order is not a replacement for resource dependencies. Each producer and consumer must declare its named reads and writes using `RenderPass::declare_resources` (or the older `reads`/`writes` methods for legacy slot-based resources). A minimal texture edge looks like this:

```rust
fn declare_resources(&self, builder: &mut helio_core::graph::ResourceBuilder) {
    builder.write_color(
        "scene_color",
        helio_core::graph::ResourceFormat::Rgba16Float,
        helio_core::graph::ResourceSize::MatchSurface,
    );
}
```

The next pass declares `builder.read("scene_color")`. `write_color` makes the graph allocate and lifetime-manage a texture; the producer chooses that view in its render descriptor and the graph routes named graph textures into the registry. For compound attachments, use `write_group`. For a storage buffer the pass allocates itself, use `write_buffer` to declare the dependency; this tracks ordering but does not allocate the buffer.

If a resource comes from the host before pass zero, declare it once with `graph.declare_external_input("resource_name")`. For example, `helio-default-graphs` declares host-fed `material_textures`, `render_environment`, `vg`, `billboards`, and `corona_emitters`. A read with neither an earlier writer nor an external-input declaration fails graph dependency validation.

Named textures appear in `PassContext::registry` by `ResourceKey`. Pass-owned non-graph resources (such as a buffer or a view held by a pass) need the producer to publish them with `publish_frame_inputs` when consumers need them before execution, or `publish` after execution. The graph invokes publication at the corresponding frame stage. The `ResourceRegistry` is per-frame; do not cache its borrowed views across frames.

## Compose from `PassBuildContext`

When you give a custom graph to `RendererBuilder`, its closure should build the whole graph from the provided context. This compact example shows the correct lifetime and construction shape; adapt ownership and dimensions to the application:

```rust
use helio::{PassBuildContext, RendererBuilder};
use helio_core::RenderGraph;

fn build_custom_graph(ctx: PassBuildContext<'_>) -> RenderGraph {
    let mut graph = if ctx.owns_device {
        RenderGraph::new(ctx.device, ctx.queue)
    } else {
        RenderGraph::new_with_external_device(ctx.device, ctx.queue)
    };

    graph.add_pass(Box::new(SceneDrawPass::new(ctx.device)));
    graph.add_pass(Box::new(MyPostProcessPass::new(ctx.device)));
    graph.lock(ctx.config.width.max(1), ctx.config.height.max(1));
    graph
}

// The frontend creates and populates SceneDB, attaches its GPU mirror, then:
// let renderer = RendererBuilder::new(config, scene_db_gpu_mirror)
//     .with_pass_build_context(Box::new(build_custom_graph))
//     .build(device, queue, width, height, surface_format);
```

This uses the SceneDB handle only as `RendererBuilder` input. Passes see its registered buffers through `PassContext::scene_buffers` / `PrepareContext::scene_buffers`; Helio does not pass a typed scene world into the graph. A graph that reads SceneDB rows must use the keys and byte layouts registered by its host.

## Resize and rebuild policy

For a directly owned `RenderGraph`, `set_render_size` reallocates graph resources and calls `on_resize` on its passes. The `Renderer` can rebuild a fresh graph on resize when a `GraphRebuilder` is stored on the graph or supplied through `set_graph_with_builder`; otherwise it resizes the existing graph. Use a rebuilder when your pipeline's pass set or constructors depend on current config/dimensions. `GraphRebuilder` is an `Arc` closure over the device, queue, config, debug state, and renderer buffers; default graph builders store one on the graph.

Passes whose persistent state can safely move to the rebuilt graph may implement `inherit_persistent_state`. The replacement pass must check that its type/configuration matches, avoid retaining old graph-pool views, and return `true` only when it actually moved the state. The graph calls `on_resize` after successful transfer.

## Render-pass chaining, external inputs, and runtime additions

- A descriptor-backed graphics pass is eligible for executor-managed chaining; use the executor-provided pass in `execute`.
- A `chain_transparent` pass must not touch a render encoder; it is for passes that only use the compute encoder while sharing a render-chain position.
- `declare_frame_demands` is for optional pass outputs that can be skipped when no downstream consumer requested them.
- `declare_pipelines` lets the executor resolve and cache format-specific pipeline recipes before execution.
- `declare_bindings` and `reflected_shader` are advanced reflected-binding paths; use them only when the WGSL and pipeline layout match the declared interface.
- `add_pass_live` is a graph edit, not the ordinary frame loop. It relocks resources and invalidates graph execution caches to account for the new pass.

For model examples, compare `helio-default-graphs/src/lib.rs` for production 3D stage composition, `crates/examples/sprite_demo.rs` for a small `SpriteCullPass` → `SpriteBatchPass` graph, and `crates/examples/sprite_dig_demo.rs` for a SceneDB-backed 2D graph that flushes its GPU mirror before execution.
