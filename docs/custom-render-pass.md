# Creating a custom Helio render pass

This guide follows the `RenderPass` API in `helio-core`. The pass trait and contexts live in `crates/helio-core/src/traits.rs` and `crates/helio-core/src/context.rs`; working implementations are in `crates/passes/`.

## 1. Put the pass in its own crate

Helio's own passes are `helio-pass-*` crates under `crates/passes/3d/`. A pass that draws or reads an object kind belongs to that kind's plugin (its `feature` crate in Pulsar-Native `plugins/vendor/<kind>/`), never to Helio. A minimal crate usually depends on `helio-core` and `wgpu`; add other dependencies only for APIs the pass actually uses. Keep shader files beside the implementation and load them with `helio_core::include_wgsl!` as the existing passes do.

## 2. Keep persistent GPU state on the pass

Store pipelines, bind-group layouts, reusable buffers, samplers, and cached bind groups on the pass struct. Construct size-independent resources once, usually in `new`. `prepare` is the per-frame place to upload small changing values through `PrepareContext::write_buffer` or `write_texture`; avoid allocating a new pipeline or buffer there every frame.

```rust
use helio_core::{PassContext, PrepareContext, RenderFrameStorage, RenderPass, Result};

pub struct TintPass {
    pipeline: wgpu::RenderPipeline,
}

impl RenderPass for TintPass {
    fn name(&self) -> &'static str { "Tint" }

    // The executor opens this pass and provides it through ctx.render_cmds().
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
        let mut pass = ctx.render_cmds().unwrap();
        pass.set_pipeline(&self.pipeline);
        pass.draw(0..3, 0..1); // e.g. a full-screen triangle
        Ok(())
    }
}
```

The example omits pipeline construction and its matching WGSL. Its pipeline must use the color format and attachment state expected by the graph/host. Use the `*_with_storage` descriptor hook when the descriptor contains temporary slices; `RenderFrameStorage::retain_boxed_slice` keeps those slices alive for the executor. The older `render_pass_descriptor` hook is suitable when the descriptor can borrow all of its data from `self` and the supplied views.

For graphics work, returning a descriptor is the preferred route: Helio owns the render-pass lifetime and can apply pass chaining and store-op handling. In that route, `execute` records through `ctx.render_cmds()`, a `RenderCmds` handle to the pass the executor owns; the pass stays open after the handle is dropped. If the descriptor returns `None`, `execute` may open its own pass with `ctx.begin_render_pass` or `ctx.begin_compute_pass`, which return `RenderCmds` / `ComputeCmds` handles that end the pass when dropped. Copies, clears and queries go through `ctx.graphics_cmds()` or `ctx.compute_cmds()`, which return a `CommandRecorder` for that command stream. That self-managed route remains supported, but cannot participate in executor-managed render-pass chaining. A compute-only pass normally returns `None` for the graphics descriptor.

`execute` is required. `prepare`, `on_resize`, and the descriptor have defaults in the trait, though graphics passes should deliberately choose their descriptor route. `prepare` may return a Helio `Result` error. Implement `on_resize` only when the pass owns resources that depend on the render size (`size.internal` for scene-resolution work, `size.output` for presented-resolution work); otherwise the default no-op is correct. A pass that only has work when the scene holds some content (water volumes, emitters, decals) implements `is_member` and `on_leave` so it costs nothing in scenes without it; see `docs/custom-pass-pipeline.md`, Scene membership.

## 3. Declare named resource dependencies

Override `declare_resources` for graph-owned textures and pass dependencies. For example, a pass consuming `pre_aa` should call `builder.read("pre_aa")`; a pass producing a transient color target should declare it with `builder.write_color(name, format, size)`. `ResourceSize` is in `helio_core::graph`. The graph allocates declared transient textures and validates that reads have a writer or were declared external inputs.

Declaration does not itself make a texture accessible under every API. The executor routes graph-declared color outputs through the per-frame `ResourceRegistry`; a consumer commonly reads one with `ctx.registry.read(ResourceKey::new("pre_aa"), self.name())`. A pass-owned resource that is not a graph texture must be published by the producer through `publish` or `publish_frame_inputs`. For attachment setup, see `FxaaPass`; for publishing a pass-owned resource, see `HiZBuildPass::publish` and `ResourceRegistry::route_named_texture`.

Create pipelines when the pass is made, through `helio_core::shader` (`module`, `create_render_pipeline`, `create_compute_pipeline`, which use the driver's pipeline cache). A pipeline that depends on content known only later compiles off the frame with `helio_core::compile::Compiling` and draws once it is ready. If formats can vary, make sure the pipeline matches the actual attachments and recreate or cache it accordingly.

## 4. Read scene data through the current boundary

`PassContext` has `scene_buffers: &SceneBufferProjection`. SceneDB GPU columns are type-erased; a pass reads one through the contract its owner's api crate declares (`BufferContract<Row>`), never by a string literal:

```rust
// `GLOWS` is a `helio_core::BufferContract<GlowRow>` from the owner's api crate.
let Some(glows) = GLOWS
    .resolve(ctx.scene_buffers)
    .map_err(|e| helio_core::Error::InvalidPassConfig(e.to_string()))?
else {
    return Ok(()); // nothing published under the key this frame
};
// Bind glows.buffer; its stride is checked against GlowRow.
```

`resolve` fails with an error naming the contract when the published stride is not the row's. The renderer does not discover component types for a pass. The host (a plugin's component) registers and uploads the GPU columns in SceneDB; the contract's api crate owns the row layout, and the pass's feature declares that it reads it (`FeatureRegistrar::consumes`). `PrepareContext` also exposes the read-only projection when the pass needs to detect or bind current buffers before execution.

## 5. Keep per-frame behavior bounded

Do not walk large CPU scene collections or rebuild scene data in `execute`. Bind the authoritative GPU columns and do GPU-side work where practical. Reuse bind groups until their buffer/view identity changes; invalidate size- or format-dependent state in `on_resize`. Use `ctx.camera`, `ctx.camera_data`, and `ctx.camera_generation` for the universal frame camera. Opt into `requires_camera_jitter` only if the pass consumes a jittered projection. Use `ctx.begin_render_pass` / `begin_compute_pass` in self-managed paths for the context helpers, and let the graph provide pass profiling.

Useful source references:

- `crates/helio-core/src/traits.rs`: `RenderPass` hooks and resource declarations.
- `crates/helio-core/src/context.rs`: `PassContext`, `PrepareContext`, and encoder helpers.
- `crates/passes/3d/helio-pass-fxaa/src/lib.rs`: named input/output and descriptor-backed graphics.
- `crates/passes/3d/helio-pass-hiz/src/lib.rs`: a graph-owned texture, bind groups rebuilt when its views change, and a published resource.
