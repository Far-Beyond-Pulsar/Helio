# Composing a custom Helio pass pipeline

This guide covers how to assemble multiple `RenderPass` implementations into a graph, supply it to Helio, and preserve its resource and resize behavior. The graph API is in `crates/helio-core/src/graph/execution.rs`; high-level graph construction uses `PassBuildContext` from `crates/helio/src/renderer/builder.rs`.

## Pick the composition point

- **Add an effect to an existing 3D pipeline:** make it a feature (`helio_feature_api::Feature`) whose node joins the presets at a `Slot` or draws in a `Phase` (`helio_frame_api`), and add it to the features the presets build from (`Preset::with_features`, or `helio_presets::graphs::registry_with` for a registry). The presets place it, declare the host inputs it brings, and rebuild it with the graph. The repository README's section 3 has a compiled example.
- **Build a standalone/custom pipeline:** create a `RenderGraph`, add each pass, declare host-provided resources, then `lock` it at the intended render dimensions.
- **Use Helio's `Renderer`:** provide a graph-building closure with `RendererBuilder::with_pass_build_context`. The closure receives the device, queue, renderer config, camera/debug buffers, and the frontend's GPU-only SceneDB handle. The closure builds the graph before the renderer starts; it is not a per-frame callback.

The shipped pipelines are node graphs in `crates/helio-presets/src/presets.rs`; each pass's node lives with its pass (the spine's in `helio-spine`, each feature's in its crate's `feature.rs`, a plugin's in its feature crate). `crates/examples/simple_graph.rs` builds a one-pass graph through `with_pass_build_context`, and `helio-wasm/src/lib.rs` documents the custom-graph hook for wasm applications.

## Build a graph before locking it

Every pass must be added before `lock`. Locking decides the first members (with no scene yet, passes that need scene content start dormant; see Scene membership below), then schedules them: it collects their declarations, validates resource dependencies, allocates their graph textures and detects render-pass chains. Adding a pass after lock with `add_pass` panics. Use `add_pass_live` only for intentional runtime graph mutation; it relocks the graph and schedules resources for the new pass.

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

`RenderGraph::new` marks the graph as owning its device. If the application owns the device, construct with `RenderGraph::new_with_external_device` instead. Match this choice to `RendererBuilder::with_external_device`; a graph builder reads it from `PassBuildContext::owns_device`.

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

If a resource comes from the host before pass zero, declare it once with `graph.declare_external_input("resource_name")`. In a preset, the feature whose passes read a host input registers a host-input node for it (`helio_frame_api::host_input`, placed at `HOST_INPUTS`), and the preset adds every one registered; the spine's own is `render_environment`. A read with neither an earlier writer nor an external-input declaration fails graph dependency validation.

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

A resize keeps the graph and its passes: `RenderGraph::resize(RenderSize)` (or `set_render_size` for one size) reallocates graph textures in place and calls `on_resize` on every pass, members and dormant ones alike, with both the internal (render) and output (presented) sizes. The `Renderer` does this on every window resize. It builds a fresh graph only on `request_graph_rebuild` (config changes, TSR quality), through its `GraphRebuilder` (`set_graph_with_builder`, `set_rebuilder`); a graph installed without one is not rebuilt, and the rebuilt graph inherits the old graph's members. Use a rebuilder when your pipeline's pass set or constructors depend on config.

Passes whose persistent state can safely move to the rebuilt graph may implement `inherit_persistent_state`. The replacement pass must check that its type/configuration matches, avoid retaining old graph-pool views, and return `true` only when it actually moved the state. The graph calls `on_resize` after successful transfer.

## Scene membership

The graph holds every pass a pipeline can use, but each frame schedules only its members. Before a frame the graph asks each pass `is_member(&mut SceneContent)`; a pass that answers false is dormant: not prepared or executed, given no graph textures, but still found by `find_pass` and still told about debug mode, editor mode, frame inputs and resizes. A pass whose work depends on scene content answers from that content, usually `scene.holds_rows("key")` (whether a SceneDB buffer has a live row, read back asynchronously by the graph) or from state a host set (a debug mode, a projection count). Passes allocate their content resources when they join (first `prepare`) and free them in `on_leave`. When the members change, the graph schedules again, keeping every graph texture whose declaration is unchanged; a new member joins a frame or two after its rows appear. Code builders and graph assets get this alike, because the predicate lives in the pass.

## Render-pass chaining, external inputs, and runtime additions

- A descriptor-backed graphics pass is eligible for executor-managed chaining; use the executor-provided pass in `execute`.
- A `chain_transparent` pass must not touch a render encoder; it is for passes that only use the compute encoder while sharing a render-chain position.
- `declare_frame_demands` is for optional pass outputs that can be skipped when no downstream consumer requested them.
- `declare_bindings` and `reflected_shader` are advanced reflected-binding paths; use them only when the WGSL and pipeline layout match the declared interface.
- `add_pass_live` is a graph edit, not the ordinary frame loop. It relocks resources and invalidates graph execution caches to account for the new pass.

For model examples, compare `helio-presets/src/presets.rs` for how a preset composes slots and phases, and `crates/examples/simple_graph.rs` for a graph of one pass with no scene content. Scenes of an object kind are that kind's plugin's examples (Pulsar-Native `plugins/vendor/<kind>/examples`).
