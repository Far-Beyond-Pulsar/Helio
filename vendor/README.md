# Vendored dependencies

## `wgpu/`, `wgpu-core/`, `wgpu-hal/` (patched)

The crates.io `wgpu`, `wgpu-core` and `wgpu-hal` **30.0.1** sources, with the
patch below. They are wired in through `[patch.crates-io]` in both the Helio
workspace manifest and the Pulsar-Native root manifest, so every crate that
depends on `wgpu` 30 gets them. The patch is additive: code that never calls
the new methods sees stock wgpu behaviour.

Every edit is marked `HELIO PATCH`; `rg "HELIO PATCH" vendor` lists them all.

### What it adds: reusable command buffers (Helio#311)

Stock wgpu consumes a command buffer when it is submitted, so a frame has to
be encoded (and validated) again every frame. The patch lets a finished
command buffer be kept and submitted any number of times:

```rust
let mut encoder = device.create_command_encoder(&Default::default());
encoder.mark_reusable();              // false: backend can't, use it normally
/* record */
let frame = encoder.finish().into_reusable().unwrap_or_else(|cb| todo!("submit cb once"));
queue.submit_mixed([wgpu::SubmitItem::Reusable(&frame)]);   // every frame
```

Each submission of a reusable command buffer still gets a fresh per-submission
"transit" command buffer built by wgpu itself, exactly as for a normal one:
memory-initialisation clears, and barriers from the device tracker's current
resource states into the states the retained commands expect. The device
tracker then advances to the commands' end states, and the submission keeps
every resource they use alive until the GPU is done. So queue writes,
mapping, resource destruction and wgpu's own later submissions all stay
correct; only encoding and its validation are skipped. Destroyed or mapped
resources are still rejected at submit.

| Crate | Change |
|---|---|
| `wgpu-hal` | `CommandEncoder::set_reusable` (default: unsupported). Vulkan begins buffers with `SIMULTANEOUS_USE` instead of `ONE_TIME_SUBMIT`; D3D12 supports it as is. Forwarded through `DynCommandEncoder`. |
| `wgpu-core` | `command/reusable.rs`: `ReusableCommandBuffer`, `CommandBuffer::make_reusable`, `Global::{command_encoder_mark_reusable, command_buffer_make_reusable, queue_submit_mixed}`. `Queue::submit` goes through `submit_items`, which handles both kinds. In-flight submissions keep a reusable buffer's commands alive and count its resources as in use (`EncoderInFlight::reusable`, `life.rs`). Pooled HAL encoders are switched back to single use when recycled. |
| `wgpu` | `CommandEncoder::mark_reusable`, `CommandBuffer::into_reusable`, `ReusableCommandBuffer`, `Queue::submit_mixed`, `SubmitItem`. |

Limits: Vulkan and D3D12 only (Metal command buffers are single-use; WebGPU
has no wgpu-core). Command buffers that build or use acceleration structures,
touch surface textures, or carry deferred actions (`map_buffer_on_submit`,
`on_submitted_work_done`) cannot be made reusable; `into_reusable` hands them
back unchanged. D3D12 re-executes a closed command list while an earlier
execution may still be in flight, which the D3D12 model permits.

Check after any change: `cargo test -p helio-core --test reusable_command_buffer`.

### Updating

To move to a newer wgpu, vendor the new release and re-apply the `HELIO PATCH`
hunks. When upstream gains equivalent API, delete the patch and the
`[patch.crates-io]` entries.

## `nebula/`

Based on upstream Nebula commit `b9b7cfc9d071aab999ae1e8ce98f576e67c82c80` from
<https://github.com/Far-Beyond-Pulsar/Nebula>. Its workspace `wgpu` dependency
is kept compatible with the wgpu release Helio uses.
