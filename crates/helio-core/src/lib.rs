//! # Helio v3: GPU-Driven Rendering Core
//!
//! **Helio v3** is a production-grade, modular rendering framework designed for game engines.
//! It provides a zero-overhead abstraction for building complex GPU-driven render pipelines with
//! automatic profiling, zero-copy resource access, and constant-time performance guarantees.
//!
//! ## Design Philosophy
//!
//! Helio v3 is built on four core principles that ensure AAA-grade performance:
//!
//! 1. **Zero-Copy Access**: Passes receive borrowed references (`&wgpu::Buffer`) to GPU resources,
//!    never owned copies. This eliminates clones and ensures O(1) resource access.
//!
//! 2. **GPU-Native Scene**: All scene state (lights, meshes, materials, camera) lives on the GPU
//!    with dirty-tracked CPU mirrors. Updates use delta-upload patterns with zero cost at steady state.
//!
//! 3. **Implicit Profiling**: CPU and GPU profiling happens automatically via `PassContext`.
//!    No manual instrumentation required - just implement `RenderPass` and profiling is injected.
//!
//! 4. **Trait-Based Modularity**: Render passes are separate crates implementing core traits.
//!    This enables compile-time polymorphism and hot-swappable pipelines without runtime overhead.
//!
//! ## Performance Guarantees
//!
//! Helio v3 enforces strict performance guarantees that match engine standards:
//!
//! - **Zero per-frame allocations** in the render path (all buffers pre-allocated)
//! - **Zero clones** in the render path (all access is by reference)
//! - **Zero locks** in the render path (no `Arc<Mutex<_>>` or `RwLock<_>`)
//! - **O(1) CPU time** regardless of scene complexity (constant-time render loop)
//! - **Dirty tracking** eliminates redundant GPU uploads (zero cost at steady state)
//!
//! ## Architecture Overview
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                        helio-core (core)                          │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Traits                                                          │
//! │ ├── RenderPass        : Execute render/compute passes           │
//! │ ├── GpuSceneManager   : Manage GPU buffers with dirty tracking  │
//! │ └── GpuResource       : Auto-growing GPU buffers                │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Scene                                                           │
//! │ ├── GpuScene          : GPU-native scene container              │
//! │ ├── SceneResources    : Zero-copy resource references           │
//! │ └── Managers          : LightBuffer, MeshBuffer, MaterialBuffer │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Graph                                                           │
//! │ ├── RenderGraph       : Graph executor with auto-profiling      │
//! │ └── ResourceBuilder   : Declare pass dependencies               │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Profiling                                                       │
//! │ ├── Profiler          : Combined CPU/GPU profiler               │
//! │ ├── CpuProfiler       : Scoped CPU timing with RAII guards      │
//! │ └── GpuProfiler       : GPU timestamp queries                   │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Context                                                         │
//! │ ├── PassContext       : Zero-copy context for execute()         │
//! │ └── PrepareContext    : Context for prepare() (uploads)         │
//! └─────────────────────────────────────────────────────────────────┘
//!
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                  Pass Crates (user-defined)                     │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ helio-gbuffer         : GBuffer geometry pass                   │
//! │ helio-shadow          : Cascaded shadow maps                    │
//! │ helio-deferred-light  : Deferred lighting pass                  │
//! │ helio-ssao            : Screen-space ambient occlusion          │
//! │ helio-bloom           : HDR bloom post-process                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Quick Start: Creating a Renderer
//!
//! ```rust,no_run
//! use helio_core::{RenderGraph, RenderPass, PassContext, Result};
//! use std::sync::Arc;
//!
//! // Define a simple pass
//! struct MyPass {
//!     pipeline: wgpu::RenderPipeline,
//! }
//!
//! impl RenderPass for MyPass {
//!     fn name(&self) -> &'static str {
//!         "MyPass"
//!     }
//!
//!     fn render_pass_descriptor<'a>(
//!         &'a self,
//!         _: &'a wgpu::TextureView,
//!         _: &'a wgpu::TextureView,
//!         _: &'a helio_core::ResourceRegistry<'a>,
//!     ) -> Option<wgpu::RenderPassDescriptor<'a>> {
//!         None
//!     }
//!
//!     fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
//!         // PassContext provides zero-copy access to scene resources
//!         let color_attachments = [Some(wgpu::RenderPassColorAttachment {
//!                 view: ctx.target,
//!                 resolve_target: None,
//!                 depth_slice: None,
//!                 ops: wgpu::Operations {
//!                     load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
//!                     store: wgpu::StoreOp::Store,
//!                 },
//!             })];
//!         let descriptor = wgpu::RenderPassDescriptor {
//!             label: Some("MyPass"),
//!             color_attachments: &color_attachments,
//!             depth_stencil_attachment: None,
//!             timestamp_writes: None,
//!             occlusion_query_set: None,
//!             multiview_mask: None,
//!         };
//!         let mut pass = ctx.begin_render_pass(&descriptor);
//!
//!         pass.set_pipeline(&self.pipeline);
//!         // Access scene resources via ctx.scene
//!         // e.g., pass.set_bind_group(0, ctx.scene.lights.bind_group(), &[]);
//!         pass.draw(0..3, 0..1);
//!
//!         Ok(())
//!     }
//! }
//!
//! // Build the render graph
//! fn create_renderer(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) {
//!     let mut graph = RenderGraph::new(&device, &queue);
//!     // Add passes (order matters)
//!     // graph.add_pass(Box::new(ShadowPass::new(&device)));
//!     // graph.add_pass(Box::new(GBufferPass::new(&device)));
//!     // graph.add_pass(Box::new(DeferredLightPass::new(&device)));
//!
//!     // Render loop
//!     // let target = surface.get_current_texture().unwrap();
//!     // let view = target.texture.create_view(&Default::default());
//!     // graph.execute(&scene_input, &view, &depth_view).unwrap();
//! }
//! ```
//!
//! ## How Passes Work
//!
//! A **render pass** is a single stage in the rendering pipeline. Passes implement the `RenderPass` trait:
//!
//! ```rust,no_run
//! use helio_core::{RenderPass, PassContext, PrepareContext, Result};
//!
//! struct MyPass {
//!     pipeline: wgpu::RenderPipeline,
//!     uniform_buffer: wgpu::Buffer,
//! }
//!
//! impl RenderPass for MyPass {
//!     fn name(&self) -> &'static str {
//!         "MyPass" // Used for profiling labels
//!     }
//!
//!     fn render_pass_descriptor<'a>(
//!         &'a self,
//!         _: &'a wgpu::TextureView,
//!         _: &'a wgpu::TextureView,
//!         _: &'a helio_core::ResourceRegistry<'a>,
//!     ) -> Option<wgpu::RenderPassDescriptor<'a>> {
//!         None
//!     }
//!
//!     fn prepare(&mut self, ctx: &PrepareContext) -> Result<()> {
//!         // Optional: Upload per-frame uniforms (called before execute)
//!         // ctx.queue.write_buffer(&self.uniform_buffer, 0, data);
//!         Ok(())
//!     }
//!
//!     fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
//!         // Record GPU commands using ctx.encoder
//!         // Access scene resources via ctx.scene (zero-copy)
//!         Ok(())
//!     }
//! }
//! ```
//!
//! **Key Points:**
//! - `prepare()` runs on CPU before GPU submission (for uniform uploads)
//! - `execute()` records GPU commands into `ctx.encoder`
//! - `PassContext` provides zero-copy access to scene resources via `ctx.scene`
//! - Profiling is automatic (CPU scope + GPU timestamps)
//!
//! ## How Scene State Works
//!
//! Authored scene state (lights, meshes, materials) is owned by SceneDB and
//! exposed to passes through the type-erased `SceneInput::scene_buffers()` projection.
//!
//! ```text
//! The frontend performs the SceneDB update/flush boundary and supplies the
//! resulting borrowed GPU buffer projection to `RenderGraph::execute()`.
//! ```
//!
//! **Key Points:**
//! - SceneDB remains the only owner of authored scene data
//! - Passes receive borrowed SceneDB GPU buffer handles (zero-copy)
//!
//! ## How Profiling Works
//!
//! Profiling is **automatic** and happens implicitly via `PassContext`:
//!
//! ```rust,no_run
//! use helio_core::{RenderPass, PassContext, Result};
//!
//! struct MyPass {
//!     pipeline: wgpu::RenderPipeline,
//! }
//!
//! impl RenderPass for MyPass {
//!     fn name(&self) -> &'static str { "MyPass" }
//!
//!     fn render_pass_descriptor<'a>(
//!         &'a self,
//!         _: &'a wgpu::TextureView,
//!         _: &'a wgpu::TextureView,
//!         _: &'a helio_core::ResourceRegistry<'a>,
//!     ) -> Option<wgpu::RenderPassDescriptor<'a>> {
//!         None
//!     }
//!
//!     fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
//!         // CPU profiling: Automatic scope created by RenderGraph
//!         // GPU profiling: Automatic timestamps via begin_render_pass
//!
//!         let descriptor = wgpu::RenderPassDescriptor {
//!             label: Some("MyPass"), // Used for GPU timestamp label
//!             // ...
//! #            color_attachments: &[],
//! #            depth_stencil_attachment: None,
//! #            timestamp_writes: None,
//! #            occlusion_query_set: None,
//! #            multiview_mask: None,
//!         };
//!         let mut pass = ctx.begin_render_pass(&descriptor);
//!
//!         // GPU timestamps automatically inserted at begin/end
//!         pass.set_pipeline(&self.pipeline);
//!         pass.draw(0..3, 0..1);
//!
//!         Ok(())
//!     }
//! }
//! ```
//!
//! **Key Points:**
//! - CPU profiling: `RenderGraph` creates scopes for each pass
//! - GPU profiling: `begin_render_pass` injects timestamp queries
//! - Zero instrumentation cost (compile-time feature flag `profiling`)
//! - Results available via `Profiler::export_timings()` for external telemetry
//!
//! ## Migration from helio-render-v2
//!
//! If you're migrating from `helio-render-v2`, here are the key differences:
//!
//! | v2 (Monolithic)                     | v3 (Modular)                          |
//! |-------------------------------------|---------------------------------------|
//! | `Renderer::render()`                | `RenderGraph::execute()`              |
//! | `Renderer::add_light()`             | SceneDB component update             |
//! | `prepare_env(SceneEnv)`             | Frontend SceneDB flush               |
//! | Passes in `passes/` folder          | Separate crates (`helio-gbuffer`)     |
//! | Manual profiling calls              | Automatic via `PassContext`           |
//! | `Arc<Mutex<GpuScene>>`              | frontend-owned SceneDB projection    |
//! | Owned `wgpu::Buffer` in passes      | `&wgpu::Buffer` via SceneDB keys      |
//!
//! **Example Migration:**
//!
//! ```rust,no_run
//! // v2 (old)
//! // let mut renderer = Renderer::new(device, queue);
//! // renderer.add_light(light);
//! // renderer.render(&target, &depth);
//!
//! // v3 (new)
//! use helio_core::RenderGraph;
//! use std::sync::Arc;
//!
//! # fn example(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) {
//! let mut graph = RenderGraph::new(&device, &queue);
//! // The frontend supplies a SceneInput implementation backed by SceneDB.
//! // graph.execute(&scene_input, &target, &depth);
//! # }
//! ```
//!
//! ## Feature Flags
//!
//! - `profiling` (default): Enable automatic CPU/GPU profiling
//!   - Disable for maximum performance in shipping builds: `default-features = false`
//!
//! ## Performance Tips
//!
//! 1. **Minimize `flush()` calls**: Only call `scene.flush()` after batched updates
//! 2. **Pre-allocate buffers**: Set initial capacity for managers to avoid GPU buffer reallocations
//! 3. **Use dirty tracking**: Managers automatically skip uploads when nothing changed
//! 4. **Avoid per-frame allocations**: All data structures are persistent and reused
//! 5. **Profile with `profiling` feature**: Monitor CPU/GPU timings via `Profiler::export_timings()`
//!
//! ## Architecture Patterns
//!
//! ### Zero-Copy Resource Access
//!
//! Passes never own GPU resources - they borrow them:
//!
//! ```rust,no_run
//! use helio_core::{RenderPass, PassContext, Result};
//!
//! struct MyPass {
//!     bind_group_layout: wgpu::BindGroupLayout,
//! }
//!
//! impl RenderPass for MyPass {
//!     fn name(&self) -> &'static str { "MyPass" }
//!
//!     fn render_pass_descriptor<'a>(
//!         &'a self,
//!         _: &'a wgpu::TextureView,
//!         _: &'a wgpu::TextureView,
//!         _: &'a helio_core::ResourceRegistry<'a>,
//!     ) -> Option<wgpu::RenderPassDescriptor<'a>> {
//!         None
//!     }
//!
//!     fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
//!         // Zero-copy: borrow GPU buffers from scene
//!         // let light_buffer = ctx.scene.lights.buffer(); // &wgpu::Buffer
//!         // let mesh_buffer = ctx.scene.meshes.buffer();   // &wgpu::Buffer
//!
//!         // Create bind groups on-the-fly (cheap with no allocations)
//!         // let bind_group = ctx.device.create_bind_group(...);
//!
//!         Ok(())
//!     }
//! }
//! ```
//!
//! ### Dirty Tracking Pattern
//!
//! Managers track dirty state and skip uploads when nothing changed:
//!
//! ```text
//! The frontend batches SceneDB changes and flushes once before executing the
//! graph; the graph itself only observes borrowed GPU handles.
//! ```
//!
//! ### Automatic Profiling Pattern
//!
//! Profiling is injected automatically by `RenderGraph`:
//!
//! ```text
//! RenderGraph::execute()
//! ├── CPU Scope: "ShadowPass" (automatic)
//! │   ├── prepare() (user code)
//! │   └── execute()
//! │       ├── GPU Timestamp: Start (automatic)
//! │       ├── GPU commands (user code)
//! │       └── GPU Timestamp: End (automatic)
//! ├── CPU Scope: "GBufferPass" (automatic)
//! │   └── ...
//! └── CPU Scope: "DeferredLightPass" (automatic)
//!     └── ...
//! ```
//!
//! ## See Also
//!
//! - [`RenderPass`] - Core trait for implementing render/compute passes
//! - [`GpuScene`] - GPU-native scene container with persistent state
//! - [`RenderGraph`] - Graph executor with automatic profiling
//! - [`PassContext`] - Zero-copy context passed to `execute()`
//! - [`Profiler`] - Automatic CPU/GPU profiling system

/// Whether reflection features (screen-space, planar, and cubemap-capture
/// reflections) are available on this target.
///
/// Apple targets render them incorrectly, so the whole reflection group is
/// compiled out there: `SsrPass` and `PlanarReflectionPass` are left out of the
/// graph, and `DeferredLightPass` tells its shader to skip both composites and
/// the reflection-capture cube array. Everything else in deferred lighting —
/// direct light, ambient, RC GI — is untouched.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub const REFLECTIONS_SUPPORTED: bool = false;
/// See the macOS/iOS variant above.
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub const REFLECTIONS_SUPPORTED: bool = true;

pub mod acceleration;
pub mod actor;
pub mod asset_types;
pub mod camera;
pub mod context;
pub mod frame_storage;
pub mod entity;
pub mod error;
pub mod graph;
pub mod movability;
pub mod profiling;
pub mod registry;
pub mod render_environment;
pub mod scene_input;
pub mod shader;
pub mod temporal;
pub mod traits;
pub mod upload;

// Generic types owned directly by helio-core: graph scheduling, the open
// resource registry, and render primitives (camera, mobility) that name no
// specific pass or scene-object type. Pass-specific GPU shapes (lights,
// materials, geometry instances, ...) live in their owning `helio-pass-*`
// crate, never here.
pub use asset_types::{MeshUpload, PackedVertex, SectionedMeshUpload};
pub use camera::GpuCameraUniforms;
pub use movability::Movability;
pub use registry::{ResourceKey, ResourceRegistry, Tracked, ViewGroup};
pub use render_environment::RenderEnvironment;

// Re-export managers
pub use crate::acceleration::{AccelerationError, BlasGeometry, BlasManager, FrameAcceleration, TlasInstanceInput, TlasManager};
// Re-export core types
pub use actor::Actor;
pub use context::{PassContext, PrepareContext};
pub use frame_storage::RenderFrameStorage;
pub use entity::Entity;
pub use error::{Error, Result};
pub use graph::{
    BindingOverrideBuilder, DebugPassInfo, DebugResourceInfo, FrameDebugData, GraphTimelineData,
    GraphTimelinePass, PipelineFormatCache, PipelineFormatKey, PipelineFormatSet, PipelineHandle,
    PipelineRecipeBuilder, PipelineRegistry, RenderGraph,
};
pub use profiling::{
    FocusedTiming, FocusedTimingGroup, FocusedTimingReport, GpuTimingAvailability, Profiler,
    RenderPassTiming, RenderTimingSnapshot,
};
pub use scene_input::{BufferHandle, BufferKey, SceneBufferProjection, SceneInput};
pub use shader::{populate_bind_group_entries, ReflectedShader};
pub use traits::{AsAny, DebugViewDescriptor, MaybeSend, MaybeSync, RenderPass};
