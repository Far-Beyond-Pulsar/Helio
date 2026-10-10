# Bindless material rendering

Helio's material texture binding mode is selected once from the device's
capabilities by `MaterialBindingConfig::from_capabilities` in `helio-mats`.

- **`BindingArray` is the default.** It is selected whenever the adapter
  supports both `TEXTURE_BINDING_ARRAY` and
  `SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING`;
  `required_wgpu_features` requests them whenever the adapter has them.
  Materials index one descriptor-indexed texture table (up to 256 textures).
- **`Expanded` is the fallback**, with separate texture and sampler bindings,
  on adapters without both features and on web builds.

Texture counts are clamped to the device's per-stage and per-bind-group
limits in either mode.

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

## Material dispatch: per-material draw segments (Helio#330)

Object Batch sorts draw groups by material key (`material_class`, graph hash)
and builds the range tables on the GPU. A draw pass needs a pipeline per key
and a CPU-side offset to draw from, but the CPU only sees the range tables
through Object Batch's asynchronous readback, two or more frames late.
Drawing a late range table against the current buffers drew some groups with
a neighbouring material's pipeline whenever the layout shifted.

Passes now draw per-material segments instead
(`helio_pass_gbuffer::DrawSegments`, built by `OcclusionCullPass`):

- The CPU keeps a table of every material key seen in the range tables, per
  shading bucket (opaque, transparent, forward). Each key owns a fixed region
  of a segment indirect buffer, twice its observed group count rounded up to a
  power of two, at least 64 records. Keys unseen for 300 frames are dropped.
  The table changes, and is uploaded, only when a key appears, disappears or
  outgrows its region.
- Range compaction appends each range's surviving draws to its key's region
  and counts them, on the GPU in the same frame. Ranges sharing a key (the sort
  interleaves them by mesh) append to the same region.
- GBuffer, Transparent and ForwardLit draw each segment of their bucket with
  its key's pipeline, at its fixed offset, with the GPU count capped at the
  region's capacity.

So the readback only decides which keys have regions: a key seen for the
first time is skipped until it has one, and survivors past a region's
capacity are skipped until it grows. A draw is never made with another key's
pipeline. Before any key has been read back, the passes draw every group with
the default pipeline. Because the table rarely changes, the recorded draw
commands stay the same frame to frame and the recording cache keeps hitting.

Shadow, DepthPrepass and PortalInstances draw every group with one pipeline
and don't use ranges.

Object Batch's readback still provides the live group and instance counts
that size the culling dispatches; those trail the GPU by the same frames.

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

Object Batch assigns range-table slots with a GPU prefix scan instead of
atomic allocation. Opaque, transparent, and forward ranges are packed in
sorted order.
