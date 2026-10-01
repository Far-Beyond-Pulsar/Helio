# Command interface and recording cache (Helio#311)

Passes record GPU work through core-owned handles instead of raw wgpu objects.
The core decides what a recorded command does: encode it into wgpu directly,
or capture it so a frame whose commands did not change is resubmitted from
already-encoded, reusable command buffers instead of being encoded and
validated again. wgpu keeps owning devices, resources, pipelines, bind groups
and shaders.

## Status

| Step | State |
|---|---|
| 1. Core command interface (`helio_core::cmd`); all passes migrated | done |
| 2. Record once, resubmit: the recording cache (below) | done; on by default on Vulkan and D3D12 |
| 3. Native reusable command buffers | done, inside wgpu: the vendored wgpu keeps a frame's native Vulkan/D3D12 command buffers and resubmits them (`vendor/README.md`) |
| 4. Metal / WebGPU / GL fall back to encoding every frame | done (the cache switches itself off) |
| Raising the hit rate: passes that record different commands every frame | ongoing; see "Making passes cacheable" |

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

## The recording cache

Each frame, every unit (one pass, or one fused chain) records through the
same handles, but in cache mode the handles append owned commands to two
streams per unit (`cmd_ir`: compute and graphics) instead of encoding. Then,
per unit (`graph/recording_cache.rs`):

* **Hit**: the streams equal a cached recording of this unit (same commands,
  same arguments, same wgpu objects by identity; debug labels ignored). The
  cached `wgpu::ReusableCommandBuffer`s are submitted again. wgpu skips
  encoding and validation and only records per-submission barriers,
  memory-init clears and lifetime tracking.
* **Miss**: the streams are encoded once into reusable command buffers and
  cached as a new variant. Up to four variants per unit, so ping-pong passes
  alternate between two cached recordings. A variant unused for 16 frames is
  dropped with the resources it keeps alive.

Correctness never depends on a pass: any CPU decision that changes the
recorded commands produces different streams and misses. `prepare()`,
`publish()` and `execute()` still run every frame; only wgpu encoding is
skipped on a hit.

Per-pass GPU timing keeps working: a cached unit writes its timestamps at
fixed indices into its own query set and resolves them into its own buffer
(both part of the cached commands), and the graph profiler copies those
results into its frame readback.

Switches: `RenderGraph::set_recording_cache(false)` or
`HELIO_RECORDING_CACHE=0`. The cache is also off for the finish-breakdown
diagnostic, on the web, and on backends that cannot reuse command buffers
(detected on the first frame).

Units that can never be reused (acceleration-structure builds or uses, surface
textures) are recorded straight into wgpu every frame, as without the cache.

### Making passes cacheable

`RenderGraph::recording_cache_stats()` (or `HELIO_RECORDING_CACHE_LOG=1`, which
prints it every 300 frames) reports per unit: hits, misses, cached
variants, and the last miss reason (which stream, which command, whether its
kind or only its arguments changed). A unit that misses every frame records
different commands every frame. Typical causes, from the pass audit:

* **Bind groups or buffers created every frame** (often keyed on raw pointers
  of views). Create them once, rebuild only when an input really changes.
* **CPU counts as draw/dispatch arguments** (`instance_count`, `emitter_count`,
  light counts). Use indirect draws/dispatches with GPU-written counts.
* **CPU skip/early-return on data** (`count == 0`, dirty flags, `settled`).
  Record the work unconditionally and let an indirect count of zero do nothing.
* **Readback state machines** (`copy when Idle`): fine, they alternate between
  a few variants.
* **Optional per-frame timestamp queries owned by the pass** with changing
  indices: use fixed indices.

Each fix is purely a hit-rate improvement and can land pass by pass.

## Worker recording (Helio#309)

Units of one dependency layer record concurrently on resident worker
threads (`HELIO_PARALLEL_RECORDING=0` to switch off). With or without
workers, the cache handles each unit on the thread that recorded it.
