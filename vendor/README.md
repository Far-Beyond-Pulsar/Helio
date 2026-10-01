# Vendored dependencies

## `wgpu/`, `wgpu-core/`, `wgpu-hal/` (patched)

The crates.io `wgpu`, `wgpu-core` and `wgpu-hal` **30.0.1** sources, with the
small patch below. They are wired in through `[patch.crates-io]` in both the
Helio workspace manifest and the Pulsar-Native root manifest, so every crate
that depends on `wgpu` 30 gets them. The patch is purely additive: nothing
upstream changes behaviour, and code that never calls the new methods cannot
tell the difference.

Every edit is marked `HELIO PATCH`; `rg "HELIO PATCH" vendor` lists them all.

| Crate | Change |
|---|---|
| `wgpu-core` | `Global::{render_pipeline,compute_pipeline,pipeline_layout,bind_group}_as_hal` (`as_hal.rs`); `RawResourceAccess for BindGroup` (`binding_model.rs`) |
| `wgpu` | `RenderPipeline::as_hal`, `ComputePipeline::as_hal`, `PipelineLayout::as_hal`, `BindGroup::as_hal`, mirroring `Buffer::as_hal` (`api/*.rs`, `backend/wgpu_core.rs`) |
| `wgpu-hal` (Vulkan) | `CommandEncoder::set_reusable(bool)`: begin command buffers without `ONE_TIME_SUBMIT` so a recorded buffer can be submitted again |

Why: a native (Vulkan / D3D12) recorder for Helio#311 has to bind
`VkPipeline` / descriptor sets, which stock wgpu does not expose. See
`docs/command_interface.md`, step 3.

Check after any change: `cargo test -p helio-core --test hal_handle_access`.

### Updating

To move to a newer wgpu, vendor the new release and re-apply the `HELIO PATCH`
hunks (about 120 lines). When upstream gains equivalent API, delete the patch
and the `[patch.crates-io]` entries.

## `nebula/`

Based on upstream Nebula commit `b9b7cfc9d071aab999ae1e8ce98f576e67c82c80` from
<https://github.com/Far-Beyond-Pulsar/Nebula>. Its workspace `wgpu` dependency
is kept compatible with the wgpu release Helio uses.
