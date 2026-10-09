# Bindless material rendering

Helio's material texture binding mode is selected once from the device's
capabilities by `MaterialBindingConfig::from_capabilities` in `helio-mats`.
`BindingArray` is the preferred mode and is selected when both
`TEXTURE_BINDING_ARRAY` and
`SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING` are available.
Otherwise Helio uses `Expanded`, which exposes separate texture and sampler
bindings. Web builds always use `Expanded`. Texture counts are clamped to the
device's per-stage and per-bind-group limits in either mode.

Material texture indices and object/material data are read from shared tables;
mesh vertex and index data are shared pools. A mode change is not a runtime
toggle: host code must request the features and limits returned by
`required_wgpu_features` and `required_wgpu_limits` before creating the device.

## Bind group lifetime

Reflected pass bind groups are cached by `RenderGraph` and reused while the
identities and epochs of their bound registry resources remain unchanged.
Pass-owned groups follow the same change-driven rule: for example, GBuffer
and ForwardLit track their buffer/material resources, Transparent tracks its
frame and fog resources, and Debug Overlay rebuilds only when its bindings are
marked dirty. Graph texture inputs such as DOF's source views can change when
the graph reallocates or routes textures, so those groups are rebuilt when
their inputs change.

When adding a pass-owned bind group, keep its bound resource identity or
generation alongside the group and rebuild only when one of those bindings
changes. Do not use a per-frame dirty flag for bindings whose resources have
stable lifetimes.

## Current material dispatch boundary

Object Batch builds GPU-side draw and material-range tables. The GBuffer,
ForwardLit, Transparent, and Shadow passes currently consume CPU-visible
range metadata to choose the material-class/graph-hash pipeline and the
indirect range. The bounded asynchronous readback in Object Batch exists for
that dispatch contract; it is not a readback of per-object instance data.
Therefore bindless texture lookup does not yet mean GPU-only material-class
dispatch, and material ranges can still cause pipeline switches.

Removing that readback requires changing the dispatch contract across those
passes. A viable design must keep compatible material shader behavior while
letting the GPU provide per-class draw counts (or dispatch through a shared
shader); changing only the range readback would leave the passes without the
CPU values they currently need to select pipelines and issue indirect draws.

## GPU-driven draw encoding (Helio #306)

Native device setup requests `MULTI_DRAW_INDIRECT_COUNT` when the adapter
supports it. The shared `multi_draw_indexed_indirect` helper uses the count
variant on that path, falls back to ordinary multi-draw when unavailable, and
uses individual indirect draws on WebGPU. The object-batch geometry passes,
shadow paths, and virtual-geometry pass route through this helper.

Occlusion culling compacts surviving indirect draw records inside each
material range on the GPU and writes the survivor count into the per-range
draw-count table. Range counts and dispatch dimensions are GPU-generated;
the CPU records a fixed three indirect compute dispatches. The count-capable
path skips culled draw slots. The no-count fallback receives packed records
with zeroed tails, and WebGPU continues to use individual indirect calls.

The CPU still reads material-range metadata asynchronously to select the
matching material pipeline. Removing that readback requires a stable
material-class catalog or an uber-shader dispatch model across GBuffer,
ForwardLit, Transparent, and Shadow. Until that dispatch contract is in
place, a current-frame GPU range table cannot safely replace the one-frame
delayed pipeline metadata.

Object Batch now assigns range-table slots with a GPU prefix scan instead of
atomic allocation. Opaque, transparent, and forward ranges are packed in
sorted order, giving a stable mapping for each table while the readback is
still in use.
