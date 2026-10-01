# Command interface (Helio#311)

Passes record GPU work through core-owned handles instead of raw wgpu objects.
This is what lets the core, later, record a frame once into native reusable
command buffers (Vulkan / D3D12) and resubmit it, with wgpu still owning
devices, resources, pipelines, bind groups and shaders.

## Status

| Step | State |
|---|---|
| 1. Core command interface (`helio_core::cmd`), wgpu-forwarding backend, passes migrated | in progress |
| 2. Data-driven frames: per-frame CPU branches moved into GPU data; `recording_key` per pass | API added (`RenderPass::recording_key`, `RenderGraph::recording_blockers`); passes still return `None` |
| 3. Native Vulkan/D3D12 backend behind the same handles | wgpu patched (see `vendor/README.md`): raw pipeline / layout / bind-group handles and reusable Vulkan buffers exist; the recorder itself is not written |
| 4. Metal / WebGPU / GL keep the wgpu backend | falls out of 1 |

## The types

All in `helio_core::cmd`, re-exported at the crate root. Method names and
signatures match wgpu's, so a migration is a type swap.

| Type | Replaces | Obtained from |
|---|---|---|
| `RenderCmds<'a>` | `&mut wgpu::RenderPass` | `ctx.render_cmds()` (graph-opened pass), `CommandRecorder::begin_render_pass`, `ctx.begin_render_pass` |
| `ComputeCmds<'a>` | `wgpu::ComputePass` | `CommandRecorder::begin_compute_pass`, `ctx.begin_compute_pass`, `ctx.begin_graphics_compute_pass` |
| `CommandRecorder<'a>` | `&mut wgpu::CommandEncoder` | `ctx.graphics_cmds()`, `ctx.compute_cmds()`; `CommandRecorder::from_encoder(&mut enc)` for tests/offline code |

`CommandRecorder` has: `begin_render_pass`, `begin_compute_pass`,
`copy_buffer_to_buffer`, `copy_buffer_to_texture`, `copy_texture_to_buffer`,
`copy_texture_to_texture`, `clear_buffer`, `clear_texture`, `write_timestamp`,
`resolve_query_set`, debug groups/markers.

`RenderCmds` has: `set_pipeline`, `set_bind_group`, `set_vertex_buffer`,
`set_index_buffer`, `set_viewport`, `set_scissor_rect`, `set_stencil_reference`,
`set_blend_constant`, `set_immediates`, `draw`, `draw_indexed`, `draw_indirect`,
`draw_indexed_indirect`, `multi_draw_indirect`, `multi_draw_indexed_indirect`,
`multi_draw_indirect_count`, `multi_draw_indexed_indirect_count`,
`execute_bundles`, `write_timestamp`, debug groups/markers.

`ComputeCmds` has: `set_pipeline`, `set_bind_group`, `set_immediates`,
`dispatch_workgroups`, `dispatch_workgroups_indirect`, `write_timestamp`, debug
groups/markers.

Resources stay wgpu: commands take `&wgpu::Buffer`, `&wgpu::BindGroup`,
`&wgpu::RenderPipeline`, `wgpu::BufferSlice`, and so on.

### Streams

* `ctx.graphics_cmds()` is the graphics stream, in graph order. It is what
  `ctx.encoder_ptr` was.
* `ctx.compute_cmds()` is the pre-graphics compute stream, submitted before all
  graphics regardless of the pass's position. It is what
  `ctx.compute_encoder_ptr` was.
* Keep using the stream a pass used before. Do not "fix" which stream a command
  goes to during migration.

### Lifetimes

`ctx.render_cmds()`, `ctx.graphics_cmds()` and `ctx.compute_cmds()` return
handles that do **not** borrow `ctx`, because passes read `ctx.camera`, the
registry etc. while recording (same as the raw pointers). They are valid until
`execute()` returns; keep one live per stream at a time.

`CommandRecorder::begin_*_pass(&mut self)` borrows the recorder. So this does
not compile (temporary dropped while borrowed):

```rust
let mut pass = ctx.graphics_cmds().begin_compute_pass(&desc);   // wrong
```

Write it as two statements:

```rust
let mut cmds = ctx.graphics_cmds();
let mut pass = cmds.begin_compute_pass(&desc);
```

## Migration rules (mechanical)

| Before | After |
|---|---|
| `unsafe { &mut *ctx.active_render_pass_ptr().unwrap() }` | `ctx.render_cmds().expect("<Pass> requires the graph render pass")` (bind with `let mut`) |
| `let Some(p) = ctx.active_render_pass_ptr() else {..}; let rp = unsafe { &mut *p };` | `let Some(mut rp) = ctx.render_cmds() else {..};` |
| `ctx.active_render_pass_ptr().is_some()` | `ctx.render_cmds().is_some()` |
| `unsafe { &mut *ctx.encoder_ptr }` | `ctx.graphics_cmds()` (bind: `let mut enc = ctx.graphics_cmds();`) |
| `unsafe { &mut *ctx.compute_encoder_ptr }` | `ctx.compute_cmds()` |
| `let ce = ctx.encoder_ptr; ... unsafe { &mut *ce }` | `let mut ce = ctx.graphics_cmds();` and use `ce` directly |
| `enc.begin_compute_pass(&desc)` on those | same call on the `CommandRecorder` (bind the recorder first, see Lifetimes) |
| `enc.begin_render_pass(&desc)` | same call on the `CommandRecorder` |
| `ctx.begin_render_pass(..)` / `begin_compute_pass(..)` / `begin_graphics_compute_pass(..)` | unchanged call; the return type is now `RenderCmds` / `ComputeCmds` |
| `fn helper(.., encoder: &mut wgpu::CommandEncoder)` called from `execute` | `fn helper(.., encoder: &mut CommandRecorder<'_>)`; call with `&mut ctx.graphics_cmds()` |
| `fn helper(.., rp: &mut wgpu::RenderPass<'_>)` / `&mut wgpu::RenderPass<'static>` | `rp: &mut RenderCmds<'_>` |
| `fn helper(.., cp: &mut wgpu::ComputePass<'_>)` | `cp: &mut ComputeCmds<'_>` |
| Tests/benchmarks that call a helper with their own `wgpu::CommandEncoder` | `&mut CommandRecorder::from_encoder(&mut encoder)` |
| Tests that open their own `wgpu::RenderPass`/`ComputePass` and hand it to a helper | `RenderCmds::from_wgpu(pass)` / `ComputeCmds::from_wgpu(pass)` |

Import what you name: `use helio_core::{CommandRecorder, ComputeCmds, RenderCmds};`.

Rules:

1. Behaviour must not change: same commands, same order, same stream.
2. Stay inside your assigned crates. If something is missing from the
   interface (a wgpu method not listed above, a type the pass needs a raw
   encoder for), **do not work around it with unsafe or by keeping a raw
   pointer**. Leave a `// TODO(Helio#311): needs core API: <what>` at the site,
   leave that site on the old API, and report it.
3. Encoders a pass creates itself with `device.create_command_encoder` for
   one-time setup/baking and submits on its own are out of scope. Only
   `PassContext`-derived command recording is migrated.
4. Resource creation, `queue.write_buffer`, bind-group building, pipeline
   building: unchanged.
5. Do not run cargo. Verify by reading and grepping. Re-read each edited
   region for borrow problems (a `CommandRecorder` is `&mut`-borrowed while a
   pass it opened is alive).

Done-check for a crate (should print nothing for `src/`, `tests/`, `benches/`):

```
rg -n "encoder_ptr|active_render_pass_ptr|active_compute_pass_ptr|wgpu::(CommandEncoder|RenderPass<|ComputePass<)" <crate>
```

(`wgpu::CommandEncoder` left over for rule 3 is allowed; say so in the report.)

## Step 2: data-driven frames (audit first)

A frame can be recorded once only if every pass's `execute()` records the same
commands each frame while its inputs are unchanged. Anything that decides
*which commands* to record from per-frame CPU data must move to the GPU
(indirect draws/dispatches, GPU-side skip flags, GPU counts):

* `helio-pass-shadow`: per-face pass selection from CPU dirty flags
  (`need_static`, `any_dirty_caster`, `objects_moved`).
* `helio-pass-object-batch`: skips its dispatch chain when settled
  (`skip_this_frame`).
* geometry passes: one pipeline per material range from a CPU readback (#307).
* the graph rebuilds reflected bind groups every frame (#307); GPU timestamp
  query indices change per frame; the output target rotates (one recording per
  target slot, see `RenderGraph::frame_recording_key`).

A pass opts in by returning `Some(key)` from `RenderPass::recording_key`
(see its docs for the contract). `RenderGraph::recording_blockers()` lists the
passes still returning `None`.

## Step 3: native backend (wgpu patched, recorder pending)

**Update:** Helio now vendors wgpu / wgpu-core / wgpu-hal 30.0.1 with the
missing hal accessors and a reusable-Vulkan-buffer switch (`vendor/README.md`).
What follows is the original analysis of why that was needed; what remains is
the recorder, barriers and queue synchronisation.

Checked against wgpu / wgpu-core / wgpu-hal 30.0.1: the public hal escape
hatches cover `Device`, `Queue`, `CommandEncoder` (`as_hal_mut`), `Buffer`,
`Texture`, `TextureView`, `Adapter`, `Instance`, `Surface` and acceleration
structures. They do **not** cover `RenderPipeline`, `ComputePipeline`,
`BindGroup`, `BindGroupLayout` or `PipelineLayout`.

A native command buffer has to bind `VkPipeline` / `ID3D12PipelineState` and
descriptor sets / root tables, so "wgpu keeps owning pipelines and bind groups,
Helio records natively" cannot be built on stock wgpu. The ways forward are:

1. Patch wgpu (`[patch.crates-io]` with vendored wgpu, wgpu-core, wgpu-hal)
   to expose raw handles for those four object kinds, and keep them alive for
   as long as a native buffer referencing them can be submitted (wgpu frees
   descriptor sets when a bind group drops).
2. Create pipelines and bind groups for the hot passes directly through
   wgpu-hal, side-stepping wgpu's objects for those passes.

Both are project-scale, backend-specific, and need Vulkan and D3D12 hardware
to validate, plus barrier derivation from `declare_resources` and queue
synchronisation with wgpu's own submissions. The handle types are enums so
either can be added later without touching passes. Until then the frame-level
win comes from step 2: do less per frame, and decide less on the CPU.
