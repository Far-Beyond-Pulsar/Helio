# Pass isolation specification

Status: architecture specification, 2026-09-11. Defines a binding rule for `helio-core` and
`libhelio` going forward, plus the full pass-facing API (current and proposed) that makes the
rule enforceable rather than aspirational. Supersedes no prior doc — this is the specification
version of the plan discussed in the dynamic-rendering-executor design thread; that thread's
phase numbering is kept so the two documents cross-reference cleanly.

Revised same day: §13–14 added after an explicit review of whether this spec's API-hygiene work
alone reaches top-tier renderer quality. It does not, on its own — §13 adds four verified,
concrete gaps (one of them, §13.2, a documentation-accuracy bug found by re-reading the actual
allocator rather than trusting its doc comment) as binding requirements, and §13.4 records one
permanent ceiling imposed by this engine's graphics API dependency rather than overclaiming
something that dependency cannot provide.

---

## 1. The rule

> **No crate in the core set (`helio-core`, `libhelio`) may contain any field, struct, enum
> variant, match arm, or string literal that names a specific pass or a specific pass's
> resource.** The core knows about *shapes* — a single view, a named group of views, an
> externally-supplied input, a pipeline recipe — and never about *instances* of those shapes
> (`gbuffer`, `billboards`, `ssao`, `hiz_sampler`, …).
>
> **Adding a new pass must require zero changes to any core crate.** The only crate that is
> ever allowed to change is a composition crate (`helio-default-graphs` today; any future
> alternative graph-composition crate), and only if the new pass is being wired into one of
> the predefined default graphs. A pass that a host application wires up itself (not through a
> default graph) requires **no changes anywhere in this repo** — not core, not
> `helio-default-graphs`, nothing.

This is a hard rule, not a style guideline, and §9 gives the enforcement mechanism: it is
satisfied by construction (the core has no closed enum of resource names left to extend) plus a
CI check that fails the build if a banned pattern reappears.

### 1.1 Crate roles

| Crate | Role | May contain pass-specific code? |
|---|---|---|
| `helio-core` | Trait definitions, executor, graph scheduling, resource pool, caches | **Never** |
| `libhelio` | Cross-cutting shared types (`FrameResources`, `SceneResources`, sky/wind data shapes) | **Never** |
| `helio-pass-*` (one crate per pass) | One pass's implementation | Yes — this *is* the pass-specific code, and it is the only place it is allowed to live |
| `helio-default-graphs` | Composes predefined graphs (forward, deferred, editor, …) out of pass crates | Only pass *construction and ordering* — never a pass's internal resource names, types, or behavior |
| `helio` | Host-facing `Renderer`, config, public API surface | Only what the host needs to drive the graph generically (camera, resize, editor-mode) — never a specific pass's resource names |

### 1.2 A second, orthogonal rule: render resources vs. scene resources

§1's rule governs *pass-identity* leakage into core crates. There is a second, independent
boundary this spec also enforces, and it applies one level up — to `helio` (the host `Renderer`)
as well as `helio-core`/`libhelio`:

> **No crate in the Helio workspace may be a second authority for scene content.** Anything with
> authoring identity or persistence — a light, a mesh instance, a foliage placement, a water
> volume, a billboard, a particle emitter, a portal — is a `pulsar_scenedb::World` component,
> registered through `helio-component`, full stop. What a pass, `PassContext`, or the executor
> may hold is a **borrowed GPU-handle projection** of that SceneDB-owned data for the current
> frame — a `&wgpu::Buffer` or `&wgpu::TextureView` reference — never a second CPU-side copy with
> its own lifecycle, and never something written by calling a `Renderer` method that stores it in
> a `Vec` the renderer itself owns.

This is not a new principle invented for this spec — it is the host workspace's existing
non-negotiable rule for scene ownership ("the renderer must not share or lock the CPU scene
world... no parallel metadata scene database may remain in production," from that workspace's
own `.agents/SCENEDB_MIGRATION.md` — outside this repository, so not linked here since this doc
also lives in Helio's own standalone repo), which that rule already enforces at the editor/engine
layer. §15 audits the same failure mode inside Helio itself, where it had not yet been checked
against.

### 1.3 Why this is stricter than "avoid special cases"

The audit behind this spec (see §2) found the core already violates the rule in three
independent places, all for the same underlying reason: the core's resource contract
(`FrameResources`) is a **closed struct**, and a closed struct can only grow by someone editing
the crate that owns it. As long as the resource contract is closed, "add a pass with a new
resource" and "edit a core crate" are the same action. §4 replaces the closed struct with an
**open registry**, which is what actually makes the rule possible to satisfy, not just
desirable.

---

## 2. Audit: current violations (as of `befeb33a`)

| # | Violation | Location | Why it exists |
|---|---|---|---|
| V1 | `FrameResources` has one hand-declared field per resource across the entire engine (50+ fields: `gbuffer`, `ssao`, `hiz`, `billboards`, `vg`, `corona_emitters`, `foliage_*`, `baked_*`, `hlfs_*`, …), each requiring 4 synchronized edits (struct field, `empty()`, `reset_tracking()`'s `reset_field!` list, and — for pool-routed ones — `route_named_texture`'s match arm) | [libhelio/src/frame.rs:188](../crates/libhelio/src/frame.rs) | No generic "named slot" abstraction existed when the struct was first grown; each pass's author added a field the same way the last one did |
| V2 | GBuffer's 4-view bundle is special-cased three separate times: exact-string-name detection in the generic texture allocator, a no-op match arm in the router, and a bespoke `Tracked<GBufferViews>` type distinct from every other slot | [resource_lifetime.rs:176-227](../crates/helio-core/src/graph/resource_lifetime.rs), [execution.rs:996](../crates/helio-core/src/graph/execution.rs) | No declarable "compound resource" shape existed; the one pass that needed one got a hand-rolled exception in the generic allocator |
| V3 | `validate_dependencies()` hardcodes the literal pass-name-adjacent strings `"main_scene"`, `"vg"`, `"billboards"`, `"corona_emitters"`, `"depth_texture"` as always-available, with no registered link to where they're actually supplied | [execution.rs:229-233](../crates/helio-core/src/graph/execution.rs) | No API existed for a host to declare "I supply this externally"; the validator's author hardcoded the known cases instead |
| V4 | Three parallel, undocumented mechanisms for "how does a pass receive data" coexist with no single answer for a new pass author: pool-routed named resources, ad hoc host `.write()` calls into `FrameResources`, and (as of this session) `AttachmentSlot`/pool lookups for dynamic attachments | [render.rs:471-598](../crates/helio/src/renderer/render.rs), [execution.rs](../crates/helio-core/src/graph/execution.rs), [attachments.rs](../crates/helio-core/src/graph/attachments.rs) | Each mechanism was added to solve one problem in isolation, without unifying with what came before |

V1–V3 are the ones this spec's rule (§1) makes structurally illegal going forward. V4 is
resolved by this spec having exactly one mechanism (§4–§6) rather than three.

---

## 3. Design principle: shapes, not instances

Everywhere the old design had the core know a resource's *name*, the new design has the core
know only a resource's *shape*. A pass declares a shape; the core allocates, tracks, and routes
based on the shape's generic structure; nothing about the declaration's payload (its name, its
format, which pass wrote it) is ever pattern-matched by string or hardcoded by identity anywhere
in `helio-core` or `libhelio`.

Four shapes cover every case found in the audit:

1. **Single view** — one `wgpu::TextureView` or `wgpu::Buffer`, tracked and routed by a typed key. Covers the ~45 simple `FrameResources` fields today.
2. **View group** — a fixed-size named bundle of views produced and consumed together. Covers GBuffer (4), HLFS's clip stack (4), foliage terrain (2).
3. **External input** — a resource no pass in the graph writes; the host supplies it before `execute()`. Covers `billboards`, `vg`, `corona_emitters`, `main_scene`.
4. **Pipeline recipe** — a format-keyed, executor-cached GPU pipeline (from the dynamic-rendering work already merged; formalized here as a first-class shape rather than a bolt-on).

---

## 4. The resource registry (replaces `FrameResources`' field list)

### 4.1 `ResourceKey<T>`

```rust
/// A typed handle to one resource slot. `T` is the Rust type stored in the
/// slot (`&wgpu::TextureView`, `&wgpu::Buffer`, a small POD struct for a
/// view group, …). Two keys are the same slot iff their `name`s are equal;
/// `T` gives compile-time type safety on read/write without the registry
/// itself needing to know what `T` is.
///
/// Declared as a `const` by whoever owns the resource semantically — today
/// that's almost always the writing pass's crate, colocated with the type
/// it stores. The core never declares one.
pub struct ResourceKey<T> {
    name: &'static str,
    _marker: PhantomData<fn() -> T>,
}

impl<T> ResourceKey<T> {
    pub const fn new(name: &'static str) -> Self;
}
```

A pass crate declares its own keys next to the types they carry:

```rust
// crates/passes/3d/helio-pass-billboard/src/lib.rs
pub const BILLBOARDS: ResourceKey<BillboardFrameData<'static>> = ResourceKey::new("billboards");
```

No core crate ever writes the string `"billboards"` again.

### 4.2 `ResourceRegistry`

Replaces `FrameResources` as the type threaded through `PassContext`/`PrepareContext`.
Keeps every behavior `Tracked<T>` already provides (debug-mode "who wrote this" tracking,
`was_written()`, panic-on-unwritten-read in debug builds) but through an open map instead of a
closed struct:

```rust
pub struct ResourceRegistry<'a> { /* opaque */ }

impl<'a> ResourceRegistry<'a> {
    pub fn empty() -> Self;

    /// Write `value` into the slot named by `key`, recording the writer's
    /// name in debug builds (mirrors `Tracked::write`).
    pub fn write<T: Copy + 'a>(&mut self, key: ResourceKey<T>, value: T, writer: &'static str);

    /// Debug-tracked read: panics in debug builds if `key` was never
    /// written this frame (mirrors `Tracked::read`).
    pub fn read<T: Copy + 'a>(&self, key: ResourceKey<T>, reader: &'static str) -> Option<T>;

    /// Untracked read, for legitimately-optional resources (mirrors `Tracked::get`).
    pub fn get<T: Copy + 'a>(&self, key: ResourceKey<T>) -> Option<T>;

    pub fn was_written<T>(&self, key: ResourceKey<T>) -> bool;

    /// Resets debug-tracking markers for the next frame (mirrors
    /// `FrameResources::reset_tracking`) — generic over every stored slot,
    /// no per-field macro invocation list to maintain.
    pub fn reset_tracking(&mut self, writer: &'static str);
}
```

Internally this is a `HashMap<&'static str, ErasedSlot>` (or, if profiling shows the hash lookup
matters on the hot path, a `Vec<(&'static str, ErasedSlot)>` populated once at `lock()` time in
declaration order and looked up by a pass-cached index — an internal optimization, invisible to
the API in §4.1–4.2 either way). **`ResourceRegistry` never has a method or field naming a
specific resource.** That absence is what makes V1 structurally impossible to reintroduce: there
is no closed field list left to hardcode against.

### 4.3 Migration shim

`libhelio::FrameResources` is kept for one deprecation window as a thin wrapper generating its
existing fields from a single table (a macro or build-script-generated list, not four
hand-synced call sites) and internally backed by a `ResourceRegistry`. Existing passes reading
`ctx.resources.billboards.read(...)` keep compiling unchanged; new passes are written directly
against `ResourceRegistry` and `ResourceKey`. `FrameResources` is removed once every in-tree pass
has migrated (tracked as its own follow-up, not blocking this spec).

---

## 5. View groups (replaces the GBuffer special case)

```rust
impl ResourceBuilder {
    /// Declares a named group of `N` color views produced and consumed as
    /// one unit. The allocator groups these by declaration — not by
    /// pattern-matching exact string suffixes — so any future compound
    /// resource (a clip stack, a terrain capture pair) gets the same
    /// handling GBuffer needed, with no new code in `resource_lifetime.rs`.
    pub fn write_group<const N: usize>(
        &mut self,
        group_name: &'static str,
        members: [(&'static str, ResourceFormat); N],
        size: ResourceSize,
    );
}

/// A resolved view group, generic over its arity — the type a pass reads
/// out of the registry for a declared `write_group`. Replaces the
/// GBuffer-specific `GBufferViews` struct; `GBufferViews` becomes a type
/// alias `ViewGroup<4>` (or is kept as a pass-crate-local newtype wrapping
/// it, at the pass author's option) rather than a core-defined type.
pub struct ViewGroup<'a, const N: usize> {
    pub views: [&'a wgpu::TextureView; N],
    pub names: [&'static str; N],
}
```

`allocate_textures()` groups any `write_group` declaration by its `group_name` — a generic
operation on the declaration, not a scan for four specific strings. `GBufferPass` becomes just
another user of `write_group("gbuffer", [...], ...)`; `helio-core` never again contains the
substring `"gbuffer_albedo"`.

---

## 6. External inputs (replaces the hardcoded `validate_dependencies` list)

```rust
impl RenderGraph {
    /// Registers `name` as supplied by the host rather than by any pass in
    /// the graph. Called once at graph-build time by whoever writes the
    /// value every frame (today: `helio`'s `Renderer`, for
    /// `billboards`/`vg`/`corona_emitters`/`main_scene`).
    ///
    /// `validate_dependencies()` treats every registered external input as
    /// available from pass index 0, replacing the hardcoded literal list —
    /// a resource that no `declare_external_input` call and no pass's
    /// `write_group`/`write_color` covers is now a real validation error
    /// instead of a silent hardcoded exception.
    pub fn declare_external_input(&mut self, name: &'static str);
}
```

**Staging note.** This is deliberately `&'static str`, matching the string-keyed convention
`ResourceBuilder`/`reads()`/`writes()` already use — Phase 1 ships against today's
`libhelio::FrameResources`, before `ResourceKey<T>`/`ResourceRegistry` (§4) exist. Once Phase 3
lands, `declare_external_input` gains a second, typed overload (or is renamed and this one
deprecated — implementer's call at that point) that takes a `ResourceKey<T>` directly; the
string-keyed form is not removed until every caller has migrated, per §4.3's shim policy.

This closes V3 and V4 together: there is now exactly one place (`declare_external_input`) that
says "this resource comes from outside the graph," and `validate_dependencies` reads that
registry instead of a literal list that could silently drift from where the host actually writes
the value.

---

## 7. Dynamic pipeline construction: attachments, format caching, and shader-driven automation

### 7.1 Shipped today

These already exist in `helio-core::graph` and satisfy the rule as designed — included here for
completeness, since they're part of the same unified API surface:

- `AttachmentSlot::{Target, Depth, Named(&'static str)}` — symbolic attachment reference, resolved against the pool. The `Named` variant takes an arbitrary caller-supplied name; the core never enumerates which names exist.
- `ColorAttachmentIntent`, `DepthAttachmentIntent` — declarative load/store + slot, resolved to a real `wgpu::RenderPassColorAttachment`/`DepthStencilAttachment` on demand.
- `PipelineFormatKey`, `PipelineFormatCache` — format-keyed pipeline cache, keyed by the *pass's own* `name()` string plus resolved formats, never by a core-known pass identity.
- Dynamic resizing: `set_render_size` reallocates the pool and pulses `PrepareContext::resize`; a pass's next `render_pass_descriptor_with_pool`/`attachment_format` call picks up the new view/format automatically, with no pass re-initialization required. This is the "resizing propagates without pass changes" guarantee from the original four-task brief, and it is satisfied by the shipped code, not a proposal.

### 7.2 Proposed — Phase 4: executor-owned pipeline recipes

`PipelineRecipe` + `RenderPass::declare_pipelines()`: a pass declares *what* to build (a closure)
and *which slots* key its format; the executor decides *when* to rebuild, resolving formats and
hitting `PipelineFormatCache` before `execute()` runs, so a pass's `execute()` only ever reads a
ready `Arc<wgpu::RenderPipeline>` out of `ctx.pipelines`. Same rule applies throughout: the
recipe's `handle` is a pass-local opaque id, never a core-known name.

### 7.3 Proposed — Phase 5: shader-reflected bind groups and directive-driven pipeline state

The deeper end of "the pass should just write shaders": `naga` (already a dependency, currently
used only for WGSL validation) reflects each pass's `.wgsl` module at `lock()` time to derive
`BindGroupLayoutEntry`s and `PipelineLayout` automatically — deleting the hand-written
layout-entry boilerplate every pass currently repeats (see `helio-pass-billboard`'s `bgl_0`
construction as the representative example). Two things naga's reflection cannot recover from
WGSL text, and which the core must still take as an explicit, small, pass-local declaration
rather than infer:

- **Binding identity.** Naga knows binding 0 is a `texture_2d<f32>`; it does not know it should
  bind to a particular `ResourceKey`. Resolved by *name-matching* — a shader variable named
  `t_pre_aa` auto-binds to the `ResourceKey` named `"pre_aa"` — with an explicit
  `declare_bindings()` override map for the rare mismatch. This is a **second naming contract**
  (shader variable name ↔ `ResourceKey` name) alongside the existing one (`ResourceKey` name ↔
  what the writer published); see §11 item 5 for why this is accepted as a real, tracked cost
  rather than an invisible one.
- **Fixed-function pipeline state.** Blend mode, depth compare/write, cull mode, and primitive
  topology have no WGSL representation at all. Proposed: reuse the `//!use helio_prelude`-style
  directive-comment convention already established in `helio-core/src/shader/mod.rs` — e.g.
  `//!blend alpha`, `//!depth less_equal no_write`, `//!cull back` — so this state is declared in
  the same `.wgsl` file rather than in a separate Rust struct, keeping the "pass authoring
  surface" to one file for the passes that opt all the way in.

At full opt-in, a pass's `execute()` reduces to `ctx.draw(vertices, instances)` /
`ctx.dispatch(x, y, z)` — no `set_pipeline`, no `set_bind_group`, no pipeline-descriptor code.
Compute passes (`light-cull`, `hiz`, `occlusion-cull`, `foliage-place`) are the recommended first
implementation target: no attachment-format matching to fight, so the win is isolated and
easiest to verify correct before extending to graphics passes.

---

## 8. The complete `RenderPass` API surface

Every method a pass crate may implement, in the order the executor calls them. **[Current]**
methods exist today at `befeb33a`; **[Proposed]** methods are part of this spec's target state
(§7, §10) and not yet implemented. Nothing in this list, current or proposed, ever requires a
pass-specific edit outside the pass's own crate to add a new pass.

| Method | Status | Required? | Purpose |
|---|---|---|---|
| `fn name(&self) -> &'static str` | Current | **Required** | Unique id for profiling, the pipeline-cache key, and `find_pass::<T>()` |
| `fn requires_camera_jitter(&self) -> bool` | Current | Optional (default `false`) | Opt in to jittered projection (TAA-style passes) |
| `fn reads(&self) -> &'static [&'static str]` | Current (legacy) | Optional (default `&[]`) | Superseded by `declare_resources`; kept for backward compat |
| `fn writes(&self) -> &'static [&'static str]` | Current (legacy) | Optional (default `&[]`) | Superseded by `declare_resources`; kept for backward compat |
| `fn declare_resources(&self, builder: &mut ResourceBuilder)` | Current | Optional (default no-op) | Declares reads/writes/groups; the one source of truth for the dependency graph and allocator |
| `fn set_editor_mode(&mut self, enabled: bool)` | Current | Optional (default no-op) | Per-frame editor/game-mode toggle |
| `fn set_debug_mode(&mut self, mode: u32)` | Current | Optional (default no-op) | Renderer-wide debug visualization mode |
| `fn debug_views(&self) -> &'static [DebugViewDescriptor]` | Current | Optional (default `&[]`) | Advertises this pass's debug views |
| `fn on_resize(&mut self, device: &wgpu::Device, width: u32, height: u32)` | Current | Optional (default no-op) | Rebuild size-dependent pass-owned resources |
| `fn prepare(&mut self, ctx: &PrepareContext) -> Result<()>` | Current | Optional (default `Ok(())`) | CPU-side per-frame uniform upload, before `execute()` |
| `fn render_pass_descriptor(&self, target, depth, resources) -> Option<RenderPassDescriptor>` | Current | **Required** (no default) | Legacy/manual descriptor construction; `None` for compute-only or self-managed passes |
| `fn render_pass_descriptor_with_pool(&self, target, depth, resources, pool) -> Option<RenderPassDescriptor>` | Current | Optional (defaults to forwarding to `render_pass_descriptor`) | Pool-aware descriptor construction — the dynamic-rendering opt-in |
| `fn declare_pipelines(&self, declare: &mut PipelineRecipeBuilder)` | **Proposed (Phase 4)** | Optional (default no-op) | Declares format-keyed pipeline recipes the executor builds/caches before `execute()` |
| `fn declare_bindings(&self, declare: &mut BindingOverrideBuilder)` | **Proposed (Phase 5)** | Optional (default no-op — pure name-matching applies) | Overrides shader-variable-name-to-`ResourceKey` matching for the rare case a name-match is wrong or ambiguous (§7.3) |
| `fn build_gpu_render_bundle(&mut self, device, resources) -> Option<wgpu::RenderBundle>` | Current | Optional (default `None`) | Pre-recorded bundle for passes with zero per-frame CPU work |
| `fn chain_transparent(&self) -> bool` | Current | Optional (default `false`) | Opts into being bridged across a fused render-pass chain without touching the encoder |
| `fn execute(&mut self, ctx: &mut PassContext) -> Result<()>` | Current | **Required** (no default) | Records GPU commands. **This signature never changes, in any phase.** |
| `fn publish<'a>(&'a self, registry: &mut ResourceRegistry<'a>)` | Current (signature updates from `libhelio::FrameResources` to `ResourceRegistry` under §4.3's migration) | Optional (default no-op) | Publishes this pass's outputs for downstream passes to read |

### 8.1 Supporting traits and types (current, unchanged by this spec)

- `AsAny` — blanket-implemented downcast support for `find_pass`/`find_pass_mut`. Never implemented by hand.
- `MaybeSend` / `MaybeSync` — platform-conditional bounds (native requires `Send`/`Sync`; wasm32 relaxes them). Blanket-implemented.
- `DebugViewDescriptor { name, debug_mode, description }` — plain data, returned from `debug_views()`.

### 8.2 `PassContext<'a>` fields (current)

| Field | Type | Notes |
|---|---|---|
| `encoder_ptr` | `*mut wgpu::CommandEncoder` | Render encoder; access via `unsafe`, or `begin_render_pass()` |
| `compute_encoder_ptr` | `*mut wgpu::CommandEncoder` | Always-available compute encoder |
| `target` | `&'a wgpu::TextureView` | This frame's color target |
| `depth` | `&'a wgpu::TextureView` | This frame's depth target |
| `scene` | `SceneResources<'a>` | Zero-copy scene buffers |
| `frame_num` | `u64` | Monotonic frame counter |
| `width`, `height` | `u32` | Internal render resolution |
| `device` | `&'a wgpu::Device` | For rare in-`execute()` bind-group creation |
| `resources` | `&'a libhelio::FrameResources<'a>` (→ `&'a ResourceRegistry<'a>` post-§4.3) | Per-frame resource registry |
| `subpass_index`, `subpass_count` | `u32` | Position within a fused render-pass chain |
| `owns_device` | `bool` | Whether Helio owns the wgpu device |
| `resource_pool` | `&'a GraphTexturePool` | The executor's texture registry |
| `active_render_pass`, `active_compute_pass` | `Option<*mut wgpu::RenderPass<'static>>` / `Option<*mut wgpu::ComputePass<'static>>` | Set by the executor before `execute()` when a descriptor was returned |
| `components` | `&'a ComponentRegistry` | Type-erased component storage |
| `pipeline_cache` | `&'a PipelineFormatCache` | Format-keyed pipeline cache, shared across every pass this frame |

Plus methods: `active_render_pass_ptr() -> Option<*mut wgpu::RenderPass<'static>>`,
`active_compute_pass_ptr() -> Option<*mut wgpu::ComputePass<'static>>`, `begin_render_pass(&self, desc) -> wgpu::RenderPass`
(legacy/manual path, panics in debug builds if `chain_transparent` is set).

### 8.3 `PrepareContext<'a>` fields (current)

`device: &'a wgpu::Device`, `queue: &'a wgpu::Queue`, `frame_num: u64`, `scene: &'a GpuScene`,
`frame_resources: &'a libhelio::FrameResources<'a>` (→ `ResourceRegistry`), `resize: bool`
(one-frame pulse after `set_render_size`), `width: u32`, `height: u32`, `delta_time: f32`.

### 8.4 `ResourceBuilder` methods (current + proposed)

| Method | Status |
|---|---|
| `read(&mut self, name: &'static str)` | Current |
| `write_color(&mut self, name, format: ResourceFormat, size: ResourceSize)` | Current |
| `write_depth(&mut self, name, size: ResourceSize)` | Current |
| `write_color_raw(&mut self, name, format: wgpu::TextureFormat, size)` | Current |
| `write_buffer(&mut self, name: &'static str)` | Current |
| `with_layers(&mut self, layers: u32) -> &mut Self` | Current |
| `with_extra_usage(&mut self, usage: wgpu::TextureUsages) -> &mut Self` | Current |
| `write_group<const N: usize>(&mut self, group_name, members, size)` | **Proposed (§5)** |
| `declarations(&self) -> &[ResourceDecl]` | Current |

### 8.5 `RenderGraph` public API used by composition crates (current + proposed)

`new`, `new_with_external_device`, `set_delta_time`, `with_xr_mode`, `requires_camera_jitter`,
`set_graph_data`/`take_graph_data`, `set_render_size`, `init_transients`, `add_pass`,
`find_pass`/`find_pass_mut`, `pass_index_of`, `set_editor_mode`, `replace_pass_at`,
`iter_passes_mut`, `collect_debug_views`, `set_debug_mode`, `validate_dependencies`,
`dump_dependency_graph`, `profiler`, `collect_frame_debug_data`, `execute`,
`execute_with_frame_resources`, `lock` — all current, all unchanged by this spec.
**`declare_external_input` (§6)** is the one addition.

---

## 9. Enforcement

The rule in §1 is satisfied two ways, and both are required — policy alone has already failed
once (V1–V3 accreted gradually under a rule that was implicit, not written down):

1. **Structural (primary).** Once `FrameResources`' closed field list is replaced by
   `ResourceRegistry` (§4) and the GBuffer-shaped special case is replaced by `write_group`
   (§5), there is no longer a closed enum, struct, or match statement in `helio-core`/`libhelio`
   that a new pass's resource could be added to. "Add a pass" and "edit the core" stop being the
   same action because the core has nothing left shaped like a per-pass list.
2. **Mechanical (backstop).** A CI check (new test, `helio-core/tests/pass_isolation.rs`) scans
   `helio-core/src` and `libhelio/src` for two patterns and fails the build on either:
   - A string literal matching a denylist seeded from every resource name found in the audit (§2) — catches a regression that reintroduces one of the exact violations found here.
   - (Stretch goal, not blocking) A `match` arm or struct field whose identifier isn't one of the generic vocabulary words (`resource`, `slot`, `key`, `group`, `pipeline`, `format`, …) — a heuristic, not a proof, but cheap to add and catches most future drift by construction of the pattern rather than an exact string.

---

## 10. Phased rollout

Reusing the numbering from the prior planning discussion so the two documents track each other:

| Phase | Delivers | Status |
|---|---|---|
| 0 | Attachment slot resolution + pipeline format cache + resize propagation (§7.1) | **Done** (`befeb33a`) |
| 1 | `declare_external_input`, removes V3/V4's hardcoded list | Not started |
| 2 | `write_group`, removes V2 (GBuffer's 3-way special case) | Not started |
| 3 | `ResourceRegistry`, removes V1 (the `FrameResources` field list); `FrameResources` becomes a deprecated shim per §4.3 | Not started |
| 4 | `PipelineRecipe` + `declare_pipelines()`, executor-owned pipeline lifecycle (§7.2) | Not started |
| 5 | naga-reflected bind groups + directive-driven fixed-function state (§7.3), starting with compute passes | Not started |
| 6 | `PassBuildContext`, shrinks `helio-default-graphs`'s per-pass constructor wiring | Not started |
| 7 | `pass_isolation.rs` CI check (§9, item 2) | Ship alongside Phase 3, once the denylist has something real to check against |

Each phase ships and is tested independently; no phase requires any other phase to be in flight
simultaneously, and every phase preserves every currently-shipped pass unchanged (per §1.3's
"required = zero core edits" rule applying retroactively to existing passes too — none of them
are forced to migrate off `FrameResources`/`reads()`/`writes()` on any timeline this spec sets).

---

## 11. Known limitations and explicit non-goals

Written down so these are decisions, not gaps someone rediscovers and re-litigates later.

1. **Bindless / array bindings never get the Phase 5 treatment.** `forward_lit.wgsl`'s
   `enable wgpu_binding_array;` and the virtual-texturing path (`vt_binder.rs`/`vt_sample.wgsl`)
   bind an *array* of resources selected per-draw, not "one name, one resource." Name-matched
   reflection (§7.3) does not generalize to this. These passes stay at a lower rung of the hook
   ladder permanently — that is a correct, final state under this spec, not a temporary gap.
   Nothing in §1's rule requires every pass to reach full automation; it only requires that
   *not* reaching it costs the core nothing.
2. **GPU-driven indirect draws are out of scope for draw-call automation.** `occlusion-cull`,
   `indirect-dispatch`, and `light-cull`'s actual complexity is `draw_indirect`/`dispatch_indirect`
   with GPU-computed offsets — not pipeline or bind-group plumbing. Phase 5 shrinks those passes'
   *setup* code, never their draw call. `ctx.draw(vertices, instances)` in §7.3 describes the
   simple case only.
3. **One shader, multiple pipeline variants (opaque/blend split, depth-prepass vs. full-shade)
   is an open design question for Phase 5**, not a solved one. The directive syntax as sketched
   in §7.3 is file-scoped and singular; expressing "build two variants from this file" needs
   either repeated directives with a variant tag or a second file, and which is cleaner has not
   been decided. Do not assume Phase 5 ships with this solved.
4. **Executor override seams survive full automation.** XR's forced `multiview_mask = 0b11` and
   subpass-chain store-op patching already reach in and override what a pass's own descriptor
   says, even in today's fully-manual model. A reflected/directive-driven pipeline still needs
   these seams open — Phase 5 is not a "hand the final artifact over and never touch it again"
   design, and should not be built as one.
5. **New risk surface accepted, not eliminated, by Phase 5:**
   - A bug in the reflection layer can silently miscompile every pass using it at once, with the
     wgpu validation error pointing at generated code rather than the `.wgsl` file a human wrote
     — a real debuggability regression versus one bespoke bug in one hand-written pass today.
   - The second naming contract from §7.3 (shader variable name ↔ `ResourceKey` name) is a new
     thing to keep in sync, on top of the existing one (`ResourceKey` name ↔ publisher). Neither
     is compiler-checked; a typo on either side fails at runtime, not build time.
   - Auto-built bind groups need their own cache, invalidated by a resource-generation counter
     bumped on pool reallocation — a new place to reintroduce the exact class of stale-view bug
     `render_graph_resize_contract.rs` already exists to catch for attachments. **Phase 5 does
     not ship without an equivalent contract test for the bind-group cache** — this is a
     requirement on the phase, not a nice-to-have.
6. **This spec does not shrink `helio-default-graphs`'s bespokeness for its own sake.** Phase 6
   (`PassBuildContext`) reduces repeated argument plumbing (`device`, `queue`, `camera_buf`,
   `config.surface_format`) but does not and should not try to make pass construction itself
   generic — a pass's constructor legitimately takes whatever pass-specific arguments it needs;
   only the *repeated, shared-across-most-passes* arguments are worth centralizing.

---

## 12. Worked example: adding a pass under this spec

Adding a new post-process pass, `helio-pass-vignette`, that reads `pre_aa` and writes back into
it, once Phases 1–3 are complete:

1. **New crate**, `crates/passes/3d/helio-pass-vignette/`, implementing `RenderPass`. Declares
   its own `ResourceKey`s if it needs any beyond the standard `"pre_aa"` name it already knows
   from `declare_resources(builder) { builder.read("pre_aa"); builder.write_color("pre_aa", ...); }`.
2. **If and only if** it's meant to run by default: one new `graph.add_pass(Box::new(VignettePass::new(...)))` line in `helio-default-graphs/src/lib.rs`, at the point in the pass order it belongs.

That's it. No edit to `helio-core`, no edit to `libhelio`, no new `FrameResources` field, no new
`route_named_texture` arm, no new `validate_dependencies` literal — because none of those exist
in a form a new pass could need to extend. A pass a host application wires up itself, outside
any default graph, requires change *nowhere in this repository* — it's a new crate, full stop.

---

## 13. Production-grade performance and robustness requirements

§1–§12 make the API surface clean. They do not, by themselves, make the engine competitive with
a shipping high-end renderer — that requires closing four specific, verified gaps between what
the executor currently does and what a top-tier console/PC renderer does. These are binding
requirements, not aspirations: a phase in this section is not done until its stated verification
condition passes.

### 13.1 Parallel command recording — P0

**Current state.** `execute_with_frame_resources` records every pass on a single thread in a
sequential `for` loop ([execution.rs:403](../crates/helio-core/src/graph/execution.rs)).
`graph/barriers.rs`, the file that would own cross-thread resource-state tracking, is a two-line
stub with no implementation. `RenderPass`'s own doc comment calls parallel compilation "a future
feature" — it has been future since the trait was written.

**Requirement.** The executor must record independent passes' GPU commands concurrently across a
worker thread pool, submitting the resulting command buffers to the single `wgpu::Queue` in an
order that preserves every real data dependency.

**Design.**

1. Generalize the existing reads/writes dependency analysis (`chain_read_write_sets`, already
   used for subpass-chain fusion) from "adjacent-pair chain detection" into a full DAG, then
   partition it into **layers** by longest-path-from-source (standard topological layering): all
   passes in layer *N* depend only on passes in layers `< N`, and passes within one layer have no
   edge between them in either direction.
2. Each worker thread records one pass's `execute()` into its own `wgpu::CommandEncoder`,
   producing an independent `wgpu::CommandBuffer`. Passes in the same layer are dispatched to the
   pool together; the executor joins before moving to the next layer.
3. Submit finished command buffers to `queue.submit(...)` **in layer order** (layer 0's buffers
   before layer 1's, etc.) — wgpu tracks resource usage across command buffers within one
   ordered submission and inserts the necessary transitions itself, so correct layer ordering is
   sufficient; the executor does not need to hand-roll barrier logic to get this part right.
4. Every piece of state a pass touches during `execute()` must be safe under this scheme:
   - `GraphTexturePool` is already read-only for the duration of a frame's execution (allocation
     happens at `lock()`/resize, not per-pass) — no change needed here.
   - `PipelineFormatCache`'s `RefCell` is **not** safe for concurrent access from multiple
     worker threads and must not be mutated during the parallel recording phase. Resolved by
     13.3: once pipeline pre-warming (13.3) guarantees every pipeline a frame will need already
     exists in the cache before recording starts, the parallel phase only ever calls
     `get_or_create` on a guaranteed hit — read-only in practice. `PipelineFormatCache` gains a
     debug-assertion that panics on an actual cache miss during parallel recording, so a
     violation of this precondition is a loud bug, not a silent race.
   - `Profiler`'s per-pass CPU/GPU scopes move to thread-local accumulation, merged into one
     `RenderTimingSnapshot` after the parallel phase joins, rather than the current sequential
     `&mut self.profiler` per pass.
5. New clause on the `RenderPass` contract (documentation only — no signature change): a pass's
   `execute()` must not depend on being called in any particular order relative to a pass it has
   no declared read/write relationship with, and must not hold hidden global mutable state beyond
   what `PassContext`/`self` provide. This was already true in spirit (the zero-copy-borrow
   discipline the trait's docs already require); this makes it load-bearing rather than
   incidental.

**Verification.** A new contract test constructs a small graph with two independent passes (no
declared dependency) and asserts identical output regardless of which one the scheduler happens
to record first — a linearizability check, not a timing check, so it is deterministic and does
not flake.

### 13.2 Real, whole-frame memory aliasing — P0

**Current state, verified in this session.** `GraphTexturePool::allocate` unconditionally calls
`device.create_texture(...)` on every allocation, regardless of `alias_group`. `alias_refs` is
written to and decremented but **never read to decide reuse**, and `release()` — its only other
caller — is never invoked anywhere in the codebase. The struct-level doc comment ("non-overlapping
textures in the same alias group share a single `wgpu::Texture` allocation") describes a feature
that does not exist. Every resource, aliased or not, gets its own full physical texture today.
This is a correctness-of-documentation bug independent of anything else in this section and is
fixed first, before the scope is widened.

**Requirement, two tiers:**

- **Tier 1 (must-fix, achievable entirely within wgpu's public API).** `allocate()` must actually
  check, before calling `create_texture`, whether an existing texture already assigned to
  `desc.alias_group` has a compatible descriptor (same format, same-or-larger extent, same usage
  flags, currently released) and reuse that `wgpu::Texture`/view rather than creating a new one.
  This alone delivers the VRAM savings the current doc comment already (incorrectly) claims.
- **Tier 2 (explicit stretch, not required for this spec's baseline).** True sub-resource memory
  placement — two *different*-shaped resources sharing the same physical GPU memory range, the
  way a low-level graphics API's placed/heap-suballocated resources work — is not reachable
  through wgpu's public `Device::create_texture` at all; it requires a native per-backend path
  through wgpu's hardware-abstraction layer, which is a separate, much larger, platform-specific
  effort. Tier 2 is recorded here as the ceiling, explicitly out of scope until a native-backend
  investment is separately justified.
- **Scope widening.** Independently of the Tier 1 fix, `assign_chain_aware_alias_groups` — which
  currently only ever aliases resources *within one fused subpass chain* — is extended to run an
  interval-overlap analysis (`first_write_pass..last_read_pass`, already tracked per
  `ResourceLifetime`) across **every** declared resource in the frame, chain-local or not, so two
  compatible-shape resources that are alive at non-overlapping points anywhere in the pass order
  can share a group, not only within a fused chain.

**Verification.** A new contract test declares two resources with non-overlapping lifetimes and
compatible descriptors, asserts they resolve to the same underlying `wgpu::Texture` after
`lock()`, and asserts total pool texture count for a known graph shape matches the expected
post-aliasing count rather than the pre-aliasing declaration count.

### 13.3 Persistent, pre-warmed pipeline cache — P0

**Current state.** The `PipelineFormatCache` shipped this session (§7.1) is in-memory and
per-session only. The first frame that hits a given format combination pays a full
`create_render_pipeline` cost inline, on the frame thread — a stutter, and one that recurs every
time the process restarts, even for a format combination seen thousands of times before.

**Requirement.**

1. **Cross-session persistence.** wgpu (pinned at 30.0.0 in this workspace) already exposes
   `wgpu::PipelineCache` — a `device.create_pipeline_cache()` object backed by a driver-validated
   binary blob, accepted via the `cache:` field already present on
   `RenderPipelineDescriptor`/`ComputePipelineDescriptor` (used as `cache: None` in every pass
   today). `PipelineFormatCache` is extended to own one `wgpu::PipelineCache`, load its blob from
   disk at startup, pass it to every `create_render_pipeline`/`create_compute_pipeline` call, and
   persist the updated blob back to disk periodically and at shutdown. The cache's own driver/
   hardware validation key means a stale or foreign blob is safely ignored, never misused, on a
   driver or GPU change.
2. **Pre-warming.** Once pipeline recipes are declarative (Phase 4, §7.2), the executor knows the
   full set of `(pass, formats)` combinations a graph can ever need *before* the first frame that
   needs one. `RenderGraph::lock()` enumerates every declared recipe against every attachment
   format the host has configured as reachable (a small, explicit list the host provides — not
   inferred) and compiles them during graph construction, off the frame thread, before the first
   `execute()` call. A shipped build with pre-warming and a valid persisted cache should have
   **zero** pipeline-cache misses during normal gameplay; a miss becomes a logged event, not a
   silent stutter.
3. **Never block the frame thread on a genuine miss.** For the residual case a shipped title must
   still tolerate (a truly new combination — a dynamic quality-setting change, a modded asset), a
   cache miss is compiled on a background thread; the pass whose pipeline isn't ready yet skips
   its draw for that frame (logged) rather than blocking the render thread on
   `create_render_pipeline`. A visible pop-in for one frame is an acceptable, bounded failure
   mode; a frame-thread stall is not.

**Verification.** A contract test asserts: (a) a second `PipelineFormatCache` constructed from a
persisted blob produced by a first one does not recompile an already-seen key from scratch
(measurable via a build-count counter passed through the recipe closure, not wall-clock timing);
(b) pre-warming a graph with a known recipe/format set leaves zero cache misses across N
simulated frames.

### 13.4 Compute/graphics scheduling: documented ceiling, not a requirement

Investigated and **explicitly not pursued**, for a concrete technical reason rather than lack of
interest: wgpu exposes exactly one `wgpu::Queue` per `wgpu::Device`, with no public API for
multiple hardware queues. Genuine async-compute overlap — compute work executing concurrently
with unrelated graphics work on separate hardware engines — is a property of multi-queue
submission on a low-level graphics API and is **not reachable through wgpu's public surface at
all**, at any effort level, short of a native per-backend path through wgpu's hardware
abstraction layer (a separate, large, platform-specific undertaking, not justified by this spec).

What *is* available and worth doing: the driver can still find some overlap within a single
queue's submission when there is no synchronization point forcing serialization between two
pieces of work. The engine's contribution here is negative, not positive — correctly-scoped
`declare_resources()` calls that don't manufacture an artificial dependency between two passes
that don't actually need one leave whatever overlap opportunity the driver can find on the table;
an over-broad or incorrect resource declaration forecloses it. No new API is proposed for this;
it falls out of getting §5/§6 right, and is recorded here so nobody spends effort chasing
multi-queue scheduling against an API that cannot provide it.

### 13.5 In-engine graph introspection tooling — P1

**Current state.** `dump_dependency_graph` emits Graphviz text; `collect_frame_debug_data`
collects per-pass/per-resource data as plain structs. Both are real and useful, but neither is a
visual, timeline-correlated capture — reading either means correlating text against separately-
gathered profiler numbers by hand.

**Requirement.** Feed `FrameDebugData` plus the existing CPU/GPU `RenderTimingSnapshot` into a
per-pass timeline view rendered in-editor, not a new standalone tool: this workspace already has
a GPU-backed flame-chart/timeline UI component under active development
(`crates/ui/wgpui-component`) — the render-graph inspector is a new data source feeding that
existing widget (pass name, CPU ms, GPU ms, declared reads/writes, alias-group membership, and
which subpass chain/parallel layer it landed in per §13.1), not a reason to build new timeline UI
from scratch.

**Verification.** No contract test — this is a developer-tooling deliverable, verified by use.
Its acceptance condition is qualitative: an engineer can answer "why does this frame cost what it
costs, and which resource is holding the most VRAM" from the in-editor view alone, without
reading `helio-core` source.

---

## 14. Updated phased rollout

Supersedes the phase count in §10 (kept there for the API-surface work; this table adds the
performance/robustness phases from §13). Phases 8–10 have no dependency on Phases 1–7 completing
first and may be worked in parallel with them.

| Phase | Delivers | Status |
|---|---|---|
| 8 | Tier 1 real texture-object reuse for alias groups + whole-frame liveness widening (§13.2) | Not started — **P0, and independently a documentation-accuracy bug fix** |
| 9 | Parallel command recording (§13.1) | Not started — **P0**, depends on Phase 4 (declarative recipes) for its pipeline-cache-safety precondition |
| 10 | Persistent + pre-warmed pipeline cache (§13.3) | Not started — **P0**, depends on Phase 4 |
| 11 | In-engine graph inspector on `wgpui-component` (§13.5) | Not started — P1 |
| 12 | Delete `helio-core::component::ComponentRegistry` (§15.1) | Not started — **P0**, zero behavior change, pure dead-code removal |
| 13 | Register `BillboardComponent`/`CoronaEmitterComponent` in `helio-component`; migrate `vg` to an existing or new typed component; thin `render.rs`'s ad hoc `Vec` state down to a GPU-handle projection (§15.2) | Not started — P1, unblocks by Phase 1 landing first (§15.3) |

§13.4 is a documented ceiling, not a phase — there is nothing to schedule.

---

## 15. Render/scene boundary audit (§1.2)

### 15.1 `helio-core::component::ComponentRegistry` — dead, and a standing invitation to regress

**Verified.** `ComponentRegistry` ([component.rs](../crates/helio-core/src/component.rs)) is a
TypeId-keyed, type-erased `Vec<T>` store, constructed empty by `GpuScene::new`
(`components: ComponentRegistry::new()`), exposed read-only via `PassContext.components` and
mutably via `GpuScene::components_mut()`. Its own doc comment calls it "the new Entity-Component
system." Grepping every crate in this workspace for a call to `.register::<T>()` on it returns
**zero results outside its own definition and construction site** — no pass, no host code,
nothing anywhere populates it with a single component type. It is unused.

Its problem is not the wasted bytes; it is that it sits inside a core crate as a second,
generic-looking ECS primitive, which makes it exactly the kind of thing a future contributor
reaches for instead of registering a proper `pulsar_scenedb::World` component — the same
"parallel scene database" failure mode `.agents/SCENEDB_MIGRATION.md` is actively fighting
elsewhere in this codebase, just not yet noticed inside Helio itself.

**Requirement.** Delete `ComponentRegistry`, `Component`, `ComponentVec`, and the
`components`/`components_mut()` fields/methods on `GpuScene` and `PassContext`. Zero behavior
change — nothing reads it — so this is unconditional, not staged behind any other phase.

### 15.2 Billboards, corona emitters, and virtual-geometry data are a second scene authority

**Verified.** `Renderer` owns `billboard_scratch: Vec<BillboardInstance>` and
`corona_emitters: Vec<libhelio::GpuCoronaEmitter>` directly
([renderer_impl.rs:106,113](../crates/helio/src/renderer/renderer_impl.rs)), populated by
whatever public `Renderer` methods the host app calls, and packed into `FrameResources` by hand
every frame ([render.rs:507,518](../crates/helio/src/renderer/render.rs)). `vg_frame_data()`
follows the same shape via `self.scene`. None of the three ever touches `pulsar_scenedb::World`.
Contrast with `helio-component`, which already has typed, GPU-mirrored, SceneDB-registered
components for eleven other scene domains (light, static mesh, foliage, water volume, portal,
reflection capture, post-process volume, planet terrain, LOD, material override, script) — and
with `MainSceneResources`'s mesh/material buffers, which *are* already correct under §1.2: plain
borrowed handles into SceneDB's own `VarLenGpuPool`, not a second copy.

Billboards, corona emitters, and virtual-geometry instance data are the only three scene-content
types in Helio that skipped SceneDB registration — which is not a coincidence: they are also
exactly the three names hardcoded in `validate_dependencies` (V3, §2). V3 is not purely an
API-hygiene bug; it is the symptom of these three not having a proper component-backed source to
declare as external in the first place.

**Requirement.**

1. Register `BillboardComponent` and `CoronaEmitterComponent` in `helio-component` following the
   existing pattern (`#[derive(SceneStore)]`, `#[gpu]`-mirrored fields), and fold
   virtual-geometry instance data into an existing typed component (most likely alongside
   `StaticMeshComponent`/LOD, since VG is a mesh-rendering strategy, not a distinct scene-object
   kind) or a new one if the domains genuinely don't fit.
2. Migrate `render.rs`'s hand-packed `.billboards.write(...)`/`.corona_emitters.write(...)`/
   `.vg.write(...)` calls to read the resulting GPU-mirrored buffers the same way
   `MainSceneResources`'s mesh buffers already do — a borrowed handle, not a host-side `Vec`
   the renderer maintains itself.
3. `Renderer::billboard_scratch`/`corona_emitters` and their public `add_billboard`/
   `add_corona_emitter`-style mutators are removed once every caller goes through the SceneDB
   component API instead.

### 15.3 Relationship to Phase 1

`declare_external_input` (§6, Phase 1) is not superseded by this — it is the validator-side fix,
independent of where a resource's value ultimately comes from, and a resource genuinely supplied
once per frame by a SceneDB-integration layer (rather than by any pass in the graph) is a
legitimate use of it regardless of whether that layer is today's ad hoc `Renderer` state or
tomorrow's proper component query. Phase 1 ships first, unchanged by this section; Phase 13
(§15.2) is what eventually shrinks the *set* of things Phase 1's mechanism needs to register, as
`billboards`/`corona_emitters`/`vg` stop being external-to-the-graph host state and become
ordinary SceneDB-component-backed buffer projections instead.
