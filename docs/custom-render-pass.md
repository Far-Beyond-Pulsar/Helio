# Creating a custom Helio render pass

This guide follows the current Helio 3.0 `RenderPass` API in `helio-core`. The pass trait and contexts live in `crates/helio-core/src/traits.rs` and `crates/helio-core/src/context.rs`; working implementations are in `crates/passes/`.

## 1. Put the pass in its own crate

Production passes are separate `helio-pass-*` crates under `crates/passes/2d/` or `crates/passes/3d/`. A minimal crate usually depends on `helio-core` and `wgpu`; add other dependencies only for APIs the pass actually uses. Keep shader files beside the implementation and load them with `helio_core::include_wgsl!` as the existing passes do.

## 2. Keep persistent GPU state on the pass

Store pipelines, bind-group layouts, reusable buffers, samplers, and cached bind groups on the pass struct. Construct size-independent resources once, usually in `new`. `prepare` is the per-frame place to upload small changing values through `PrepareContext::write_buffer` or `write_texture`; avoid allocating a new pipeline or buffer there every frame.

```rust
use helio_core::{PassContext, PrepareContext, RenderFrameStorage, RenderPass, Result};

pub struct TintPass {
    pipeline: wgpu::RenderPipeline,
}

impl RenderPass for TintPass {
    fn name(&self) -> &'static str { "Tint" }

    // The executor opens this pass and provides ctx.active_render_pass_ptr().
    fn render_pass_descriptor_with_storage<'a>(
        &'a self,
        target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a helio_core::ResourceRegistry<'a>,
        storage: &'a mut RenderFrameStorage,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        let attachments = storage.retain_boxed_slice(Box::new([
            Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            }),
        ]));
        Some(wgpu::RenderPassDescriptor {
            label: Some("Tint"),
            color_attachments: attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn prepare(&mut self, _ctx: &PrepareContext) -> Result<()> {
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        // A descriptor-backed graphics pass records into the executor's pass.
        let pass = unsafe { &mut *ctx.active_render_pass_ptr().unwrap() };
        pass.set_pipeline(&self.pipeline);
        pass.draw(0..3, 0..1); // e.g. a full-screen triangle
        Ok(())
    }
}
```

The example omits pipeline construction and its matching WGSL. Its pipeline must use the color format and attachment state expected by the graph/host. Use the `*_with_storage` descriptor hook when the descriptor contains temporary slices; `RenderFrameStorage::retain_boxed_slice` keeps those slices alive for the executor. The older `render_pass_descriptor` hook is suitable when the descriptor can borrow all of its data from `self` and the supplied views.

For graphics work, returning a descriptor is the preferred route: Helio owns the render-pass lifetime and can apply pass chaining and store-op handling. In that route, `execute` records through `ctx.active_render_pass_ptr()`; that accessor is unsafe to dereference because the executor owns the pass lifetime. If the descriptor returns `None`, `execute` may open its own pass with `ctx.begin_render_pass` or `ctx.begin_compute_pass`. That self-managed route remains supported, but cannot participate in executor-managed render-pass chaining. A compute-only pass normally returns `None` for the graphics descriptor.

`execute` is required. `prepare`, `on_resize`, and the descriptor have defaults in the trait, though graphics passes should deliberately choose their descriptor route. `prepare` may return a Helio `Result` error. Implement `on_resize` only when the pass owns resources that depend on the render size; otherwise the default no-op is correct.

## 3. Declare named resource dependencies

Override `declare_resources` for graph-owned textures and pass dependencies. For example, a pass consuming `pre_aa` should call `builder.read("pre_aa")`; a pass producing a transient color target should declare it with `builder.write_color(name, format, size)`. `ResourceSize` is in `helio_core::graph`. The graph allocates declared transient textures and validates that reads have a writer or were declared external inputs.

Declaration does not itself make a texture accessible under every API. The executor routes graph-declared color outputs through the per-frame `ResourceRegistry`; a consumer commonly reads one with `ctx.registry.read(ResourceKey::new("pre_aa"), self.name())`. A pass-owned resource that is not a graph texture must be published by the producer through `publish` or `publish_frame_inputs`. For attachment setup, see `GBufferPass` and `FxaaPass`; for named view publication, see `FogCompositePass` and `ResourceRegistry::route_named_texture`.

Use `declare_pipelines` and the pipeline recipe APIs when you want the graph executor to cache a pipeline across attachment-format variants. The pass crate's `PipelineRegistry` handles are available from `PassContext::pipelines`. Manual pipeline creation is also used in existing passes; if formats can vary, ensure the pipeline matches the actual attachments and recreate or cache it accordingly.

## 4. Read scene data through the current boundary

The current `PassContext` has `scene_buffers: &SceneBufferProjection`, not the old typed `ctx.scene` fields described in some older comments and examples. SceneDB GPU columns are type-erased and resolved by key:

```rust
let Some(sprite_rows) = ctx.scene_buffers.get(
    pulsar_scenedb::gpu::BufferKey::of("sprite_instances"),
) else {
    return Ok(()); // this graph/host did not provide the optional column
};
// Bind sprite_rows.buffer according to the registered SceneDB GPU layout.
```

The renderer does not discover component types for a pass. The host must register and upload a component's GPU columns in SceneDB before exposing them in `SceneBufferProjection`; the pass owns the interpretation of the named buffer layout. `PrepareContext` also exposes the read-only projection when the pass needs to detect or bind current buffers before execution.

## 5. Keep per-frame behavior bounded

Do not walk large CPU scene collections or rebuild scene data in `execute`. Bind the authoritative GPU columns and do GPU-side work where practical. Reuse bind groups until their buffer/view identity changes; invalidate size- or format-dependent state in `on_resize`. Use `ctx.camera`, `ctx.camera_data`, and `ctx.camera_generation` for the universal frame camera. Opt into `requires_camera_jitter` only if the pass consumes a jittered projection. Use `ctx.begin_render_pass` / `begin_compute_pass` in self-managed paths for the context helpers, and let the graph provide pass profiling.

Useful source references:

- `crates/helio-core/src/traits.rs` — `RenderPass` hooks and resource declarations.
- `crates/helio-core/src/context.rs` — `PassContext`, `PrepareContext`, and encoder helpers.
- `crates/passes/3d/helio-pass-fxaa/src/lib.rs` — named input/output and descriptor-backed graphics.
- `crates/passes/2d/helio-pass-sprite-batch/src/lib.rs` — SceneDB buffer consumption and resize/bind-group invalidation.
