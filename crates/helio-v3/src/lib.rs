//! # Helio v3: AAA GPU-Driven Rendering Core
//!
//! **Helio v3** is a production-grade, modular rendering framework designed for AAA game engines.
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
//! ## AAA Performance Guarantees
//!
//! Helio v3 enforces strict performance guarantees that match AAA engine standards:
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
//! │                        helio-v3 (core)                          │
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
//! use helio_v3::{RenderGraph, GpuScene, RenderPass, PassContext, Result};
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
//!     fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
//!         // PassContext provides zero-copy access to scene resources
//!         let mut pass = ctx.begin_render_pass(&wgpu::RenderPassDescriptor {
//!             label: Some("MyPass"),
//!             color_attachments: &[Some(wgpu::RenderPassColorAttachment {
//!                 view: ctx.target,
//!                 resolve_target: None,
//!                 ops: wgpu::Operations {
//!                     load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
//!                     store: wgpu::StoreOp::Store,
//!                 },
//!             })],
//!             depth_stencil_attachment: None,
//!             timestamp_writes: None,
//!             occlusion_query_set: None,
//!         });
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
//!     let scene = GpuScene::new(device.clone(), queue.clone());
//!
//!     // Add passes (order matters)
//!     // graph.add_pass(Box::new(ShadowPass::new(&device)));
//!     // graph.add_pass(Box::new(GBufferPass::new(&device)));
//!     // graph.add_pass(Box::new(DeferredLightPass::new(&device)));
//!
//!     // Render loop
//!     // let target = surface.get_current_texture().unwrap();
//!     // let view = target.texture.create_view(&Default::default());
//!     // graph.execute(&scene, &view, &depth_view).unwrap();
//! }
//! ```
//!
//! ## How Passes Work
//!
//! A **render pass** is a single stage in the rendering pipeline. Passes implement the `RenderPass` trait:
//!
//! ```rust,no_run
//! use helio_v3::{RenderPass, PassContext, PrepareContext, Result};
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
//! Scene state (lights, meshes, materials) is managed by `GpuScene`:
//!
//! ```rust,no_run
//! use helio_v3::GpuScene;
//! use std::sync::Arc;
//!
//! let mut scene = GpuScene::new(
//!     Arc::new(device),
//!     Arc::new(queue),
//! );
//!
//! // Add scene objects (future API)
//! // let light_id = scene.lights.add(PointLight { ... });
//! // scene.lights.remove(light_id);
//! // scene.lights.update(light_id, PointLight { ... });
//!
//! // Flush dirty data to GPU (zero-cost at steady state)
//! scene.flush();
//!
//! // Get zero-copy resource references for passes
//! let resources = scene.resources();
//! // resources.lights.buffer() -> &wgpu::Buffer
//! // resources.meshes.buffer() -> &wgpu::Buffer
//! ```
//!
//! **Key Points:**
//! - All scene data lives on GPU with dirty-tracked CPU mirrors
//! - `flush()` uploads only changed data (O(changed) not O(total))
//! - At steady state (no changes), `flush()` is a no-op (zero cost)
//! - Passes receive `SceneResources<'_>` with `&wgpu::Buffer` references (zero-copy)
//!
//! ## How Profiling Works
//!
//! Profiling is **automatic** and happens implicitly via `PassContext`:
//!
//! ```rust,no_run
//! use helio_v3::{RenderPass, PassContext, Result};
//!
//! struct MyPass;
//!
//! impl RenderPass for MyPass {
//!     fn name(&self) -> &'static str { "MyPass" }
//!
//!     fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
//!         // CPU profiling: Automatic scope created by RenderGraph
//!         // GPU profiling: Automatic timestamps via begin_render_pass
//!
//!         let mut pass = ctx.begin_render_pass(&wgpu::RenderPassDescriptor {
//!             label: Some("MyPass"), // Used for GPU timestamp label
//!             // ...
//! #            color_attachments: &[],
//! #            depth_stencil_attachment: None,
//! #            timestamp_writes: None,
//! #            occlusion_query_set: None,
//!         });
//!
//!         // GPU timestamps automatically inserted at begin/end
//!         pass.set_pipeline(&self.pipeline);
//!         pass.draw(0..3, 0..1);
//!
//!         Ok(())
//!     }
//! #    pipeline: wgpu::RenderPipeline,
//! }
//! #
//! # impl MyPass {
//! #     fn new(pipeline: wgpu::RenderPipeline) -> Self { Self { pipeline } }
//! # }
//! ```
//!
//! **Key Points:**
//! - CPU profiling: `RenderGraph` creates scopes for each pass
//! - GPU profiling: `begin_render_pass` injects timestamp queries
//! - Zero instrumentation cost (compile-time feature flag `profiling`)
//! - Results exported to `helio-live-portal` for real-time telemetry
//!
//! ## Migration from helio-render-v2
//!
//! If you're migrating from `helio-render-v2`, here are the key differences:
//!
//! | v2 (Monolithic)                     | v3 (Modular)                          |
//! |-------------------------------------|---------------------------------------|
//! | `Renderer::render()`                | `RenderGraph::execute()`              |
//! | `Renderer::add_light()`             | `GpuScene::lights.add()`              |
//! | `prepare_env(SceneEnv)`             | `GpuScene::flush()` (automatic)       |
//! | Passes in `passes/` folder          | Separate crates (`helio-gbuffer`)     |
//! | Manual profiling calls              | Automatic via `PassContext`           |
//! | `Arc<Mutex<GpuScene>>`              | `&GpuScene` (zero locks)              |
//! | Owned `wgpu::Buffer` in passes      | `&wgpu::Buffer` via `SceneResources`  |
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
//! use helio_v3::{RenderGraph, GpuScene};
//! use std::sync::Arc;
//!
//! let mut graph = RenderGraph::new(&device, &queue);
//! let mut scene = GpuScene::new(Arc::new(device), Arc::new(queue));
//!
//! // scene.lights.add(light);
//! scene.flush(); // Upload changes
//! // graph.execute(&scene, &target, &depth);
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
//! 5. **Profile with `profiling` feature**: Monitor CPU/GPU timings in `helio-live-portal`
//!
//! ## Architecture Patterns
//!
//! ### Zero-Copy Resource Access
//!
//! Passes never own GPU resources - they borrow them:
//!
//! ```rust,no_run
//! use helio_v3::{RenderPass, PassContext, Result};
//!
//! struct MyPass {
//!     bind_group_layout: wgpu::BindGroupLayout,
//! }
//!
//! impl RenderPass for MyPass {
//!     fn name(&self) -> &'static str { "MyPass" }
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
//! ```rust,no_run
//! use helio_v3::GpuScene;
//! use std::sync::Arc;
//!
//! let mut scene = GpuScene::new(Arc::new(device), Arc::new(queue));
//!
//! // Frame 1: Add lights (dirty = true)
//! // scene.lights.add(light1);
//! // scene.lights.add(light2);
//! scene.flush(); // Uploads to GPU
//!
//! // Frame 2: No changes (dirty = false)
//! scene.flush(); // No-op (zero cost)
//!
//! // Frame 3: Update one light (dirty = true)
//! // scene.lights.update(light1_id, new_light);
//! scene.flush(); // Uploads only changed data
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

pub mod traits;
pub mod error;
pub mod context;
pub mod profiling;
pub mod scene;
pub mod graph;
pub mod upload;

// Re-export libhelio types for convenience
pub use libhelio::{
    GpuCameraUniforms, GpuInstanceData, GpuInstanceAabb, GpuDrawCall,
    GpuLight, GpuMaterial, GpuShadowMatrix, DrawIndexedIndirectArgs,
    GBufferViews, FrameResources, SkyContext, SkyUniforms,
};
// Re-export managers
pub use crate::scene::managers::*;
// Re-export core types
pub use traits::{RenderPass, GpuSceneManager, GpuResource, MaybeSend, MaybeSync};
pub use error::{Error, Result};
pub use context::{PassContext, PrepareContext};
pub use profiling::Profiler;
pub use scene::{GpuScene, SceneResources};
pub use graph::RenderGraph;
