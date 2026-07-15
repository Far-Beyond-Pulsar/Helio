//! Core traits for helio-core modular renderer.
//!
//! This module defines the fundamental traits that power helio-core's modular architecture:
//!
//! - [`RenderPass`] - Trait for implementing render/compute passes
//! - [`GpuSceneManager`] - Trait for managing GPU buffers with dirty tracking
//! - [`GpuResource`] - Trait for auto-growing GPU buffers
//!
//! # Design Pattern: Trait-Based Modularity
//!
//! Helio v3 uses trait-based modularity to enable compile-time polymorphism with zero runtime overhead.
//! Each render pass is a separate crate that implements `RenderPass`, allowing:
//!
//! - **Hot-swappable pipelines**: Change passes without recompiling the core
//! - **Zero abstraction cost**: Traits are monomorphized at compile-time (no vtable overhead)
//! - **Type-safe composition**: RenderGraph enforces correct pass ordering at compile-time
//!
//! # Example: Implementing a Custom Pass
//!
//! ```rust,no_run
//! use helio_core::{RenderPass, PassContext, PrepareContext, Result};
//!
//! pub struct MyCustomPass {
//!     pipeline: wgpu::RenderPipeline,
//!     uniform_buffer: wgpu::Buffer,
//! }
//!
//! impl RenderPass for MyCustomPass {
//!     fn name(&self) -> &'static str {
//!         "MyCustomPass"
//!     }
//!
//!     fn prepare(&mut self, ctx: &PrepareContext) -> Result<()> {
//!         // Upload per-frame uniforms (runs on CPU before GPU submission)
//!         let uniforms = MyUniforms {
//!             time: ctx.frame as f32,
//!         };
//!         ctx.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
//!         Ok(())
//!     }
//!
//!     fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
//!         // Record GPU commands (profiling is automatic)
//!         let mut pass = ctx.begin_render_pass(&wgpu::RenderPassDescriptor {
//!             label: Some("MyCustomPass"),
//!             color_attachments: &[Some(wgpu::RenderPassColorAttachment {
//!                 view: ctx.target,
//!                 resolve_target: None,
//!                 ops: wgpu::Operations {
//!                     load: wgpu::LoadOp::Load,
//!                     store: wgpu::StoreOp::Store,
//!                 },
//!             })],
//!             depth_stencil_attachment: None,
//!             timestamp_writes: None,
//!             occlusion_query_set: None,
//!         });
//!
//!         pass.set_pipeline(&self.pipeline);
//!         // Access scene resources: ctx.scene.lights, ctx.scene.meshes, etc.
//!         pass.draw(0..3, 0..1);
//!
//!         Ok(())
//!     }
//! }
//! # #[repr(C)]
//! # #[derive(Copy, Clone)]
//! # struct MyUniforms { time: f32 }
//! # unsafe impl bytemuck::Pod for MyUniforms {}
//! # unsafe impl bytemuck::Zeroable for MyUniforms {}
//! ```

// ── Platform-conditional Send + Sync bounds ────────────────────────────────
// On native targets, wgpu types are Send + Sync, so RenderPass requires them.
// On wasm32, wgpu's internal dyn-context types are not Send + Sync (WASM is
// single-threaded), so we relax the bounds via blanket no-op traits.

/// Blanket bound: `Send` on native, nothing on `wasm32`.
#[cfg(not(target_arch = "wasm32"))]
pub trait MaybeSend: Send {}
#[cfg(not(target_arch = "wasm32"))]
impl<T: Send> MaybeSend for T {}

#[cfg(target_arch = "wasm32")]
pub trait MaybeSend {}
#[cfg(target_arch = "wasm32")]
impl<T> MaybeSend for T {}

/// Blanket bound: `Sync` on native, nothing on `wasm32`.
#[cfg(not(target_arch = "wasm32"))]
pub trait MaybeSync: Sync {}
#[cfg(not(target_arch = "wasm32"))]
impl<T: Sync> MaybeSync for T {}

#[cfg(target_arch = "wasm32")]
pub trait MaybeSync {}
#[cfg(target_arch = "wasm32")]
impl<T> MaybeSync for T {}

use crate::graph::ResourceBuilder;
use crate::{PassContext, PrepareContext, Result};

/// Describes a debug visualisation mode that a render pass provides.
///
/// Passes return a static slice of these from [`RenderPass::debug_views`] so
/// the renderer can enumerate all available debug modes at runtime.
#[derive(Debug, Clone, Copy)]
pub struct DebugViewDescriptor {
    pub name: &'static str,
    pub debug_mode: u32,
    pub description: &'static str,
}

/// Supertrait that provides safe `Any`-based downcasting for render passes.
///
/// Blanket-implemented for every `T: 'static`, so no concrete pass needs to
/// write `fn as_any` / `fn as_any_mut` — the compiler generates the correct
/// body automatically and it can never be written incorrectly.
///
/// `RenderGraph::find_pass` / `find_pass_mut` use this to downcast a
/// `Box<dyn RenderPass>` back to a concrete pass type in O(1).
pub trait AsAny: std::any::Any {
    /// Returns `self` as a `&dyn Any` so `downcast_ref` can be called.
    fn as_any(&self) -> &dyn std::any::Any;
    /// Returns `self` as a `&mut dyn Any` so `downcast_mut` can be called.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// Blanket implementation — every concrete type gets this for free.
impl<T: std::any::Any> AsAny for T {
    #[inline]
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    #[inline]
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Core trait for all rendering passes.
///
/// `RenderPass` is the fundamental building block of helio-core. Each pass is a self-contained
/// stage in the rendering pipeline (e.g., shadow mapping, GBuffer, deferred lighting, post-process).
///
/// # Contract
///
/// Implementations must:
/// - Be **thread-safe** (`Send + Sync`) for parallel pass compilation (future feature)
/// - Return a **unique name** for profiling and debugging
/// - **Record GPU commands** in `execute()` without blocking the CPU
/// - **Upload uniforms** in `prepare()` if needed (optional)
///
/// # Lifecycle
///
/// For each frame, the `RenderGraph` calls methods in this order:
///
/// 1. `declare_resources()` (once at graph construction)
/// 2. `prepare()` (per-frame, CPU-side uniform uploads)
/// 3. `execute()` (per-frame, record GPU commands)
///
/// # Performance
///
/// - **O(1) CPU time**: `execute()` should only record commands, not iterate large datasets
/// - **Zero allocations**: Reuse pre-allocated buffers and bind groups
/// - **Zero clones**: Borrow scene resources via `PassContext::scene`
///
/// # Profiling
///
/// Profiling is **automatic**:
/// - CPU scope: `RenderGraph` creates a scope for each pass using `name()`
/// - GPU timestamps: `PassContext::begin_render_pass()` injects timestamp queries
///
/// # Examples
///
/// ## Minimal Pass (No Prepare)
///
/// ```rust,no_run
/// use helio_core::{RenderPass, PassContext, Result};
///
/// struct SimplePass {
///     pipeline: wgpu::RenderPipeline,
/// }
///
/// impl RenderPass for SimplePass {
///     fn name(&self) -> &'static str {
///         "SimplePass"
///     }
///
///     fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
///         let mut pass = ctx.begin_render_pass(&wgpu::RenderPassDescriptor {
///             label: Some("SimplePass"),
///             color_attachments: &[Some(wgpu::RenderPassColorAttachment {
///                 view: ctx.target,
///                 resolve_target: None,
///                 ops: wgpu::Operations {
///                     load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
///                     store: wgpu::StoreOp::Store,
///                 },
///             })],
///             depth_stencil_attachment: None,
///             timestamp_writes: None,
///             occlusion_query_set: None,
///         });
///
///         pass.set_pipeline(&self.pipeline);
///         pass.draw(0..3, 0..1);
///
///         Ok(())
///     }
/// }
/// ```
///
/// ## Pass with Prepare (Uniform Upload)
///
/// ```rust,no_run
/// use helio_core::{RenderPass, PassContext, PrepareContext, Result};
///
/// struct PassWithUniforms {
///     pipeline: wgpu::RenderPipeline,
///     uniform_buffer: wgpu::Buffer,
///     bind_group: wgpu::BindGroup,
/// }
///
/// impl RenderPass for PassWithUniforms {
///     fn name(&self) -> &'static str {
///         "PassWithUniforms"
///     }
///
///     fn prepare(&mut self, ctx: &PrepareContext) -> Result<()> {
///         // Upload per-frame data (called before execute)
///         let uniforms = MyUniforms {
///             view_proj: [[0.0; 4]; 4], // Compute from camera
///         };
///         ctx.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
///         Ok(())
///     }
///
///     fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
///         let mut pass = ctx.begin_render_pass(&wgpu::RenderPassDescriptor {
///             label: Some("PassWithUniforms"),
///             color_attachments: &[Some(wgpu::RenderPassColorAttachment {
///                 view: ctx.target,
///                 resolve_target: None,
///                 ops: wgpu::Operations {
///                     load: wgpu::LoadOp::Load,
///                     store: wgpu::StoreOp::Store,
///                 },
///             })],
///             depth_stencil_attachment: None,
///             timestamp_writes: None,
///             occlusion_query_set: None,
///         });
///
///         pass.set_pipeline(&self.pipeline);
///         pass.set_bind_group(0, &self.bind_group, &[]);
///         pass.draw(0..3, 0..1);
///
///         Ok(())
///     }
/// }
/// # #[repr(C)]
/// # #[derive(Copy, Clone)]
/// # struct MyUniforms { view_proj: [[f32; 4]; 4] }
/// # unsafe impl bytemuck::Pod for MyUniforms {}
/// # unsafe impl bytemuck::Zeroable for MyUniforms {}
/// ```
///
/// # Implementation Notes
///
/// - **Pass State**: Store pipelines, bind groups, and uniform buffers as fields
/// - **Resource Access**: Use `ctx.scene` to borrow GPU buffers (zero-copy)
/// - **Error Handling**: Return `Result<()>` for GPU errors (e.g., shader compilation)
/// - **Profiling**: Use `ctx.begin_render_pass()` instead of `ctx.encoder.begin_render_pass()`
///   to get automatic GPU timestamp injection
pub trait RenderPass: AsAny + MaybeSend + MaybeSync {
    /// Returns a unique name for this pass.
    ///
    /// Used for profiling labels and debugging. Should be a human-readable identifier
    /// (e.g., "ShadowPass", "GBuffer", "DeferredLighting").
    ///
    /// # Performance
    ///
    /// - **O(1)**: Returns a static string (no allocations)
    fn name(&self) -> &'static str;

    /// Whether this pass consumes a deliberately jittered camera projection.
    ///
    /// Temporal reconstruction passes opt in so the renderer can keep FXAA and
    /// other non-temporal graphs pixel-stable by default.
    fn requires_camera_jitter(&self) -> bool {
        false
    }

    /// Resources this pass reads. Checked at graph construction time.
    /// Override to declare dependencies on prior-pass outputs.
    /// Return graph resource name strings (e.g. `"pre_aa"`, `"gbuffer"`).
    fn reads(&self) -> &'static [&'static str] {
        &[]
    }

    /// Resources this pass writes/publishes. Checked at graph construction time.
    /// Override to declare outputs for downstream passes.
    fn writes(&self) -> &'static [&'static str] {
        &[]
    }

    /// Executes the pass by recording GPU commands.
    ///
    /// This is the main entry point for rendering. Implementations should:
    /// 1. Begin a render/compute pass using `ctx.begin_render_pass()` or `ctx.begin_compute_pass()`
    /// 2. Set pipelines, bind groups, and issue draw/dispatch calls
    /// 3. Access scene resources via `ctx.scene` (zero-copy)
    ///
    /// # Parameters
    ///
    /// - `ctx`: Zero-copy context with scene resources and profiler
    ///
    /// # Performance
    ///
    /// - **O(1) CPU time**: Only record commands, don't iterate large datasets on CPU
    /// - **Zero allocations**: Reuse pre-allocated buffers
    /// - **Zero clones**: Borrow `ctx.scene` resources by reference
    ///
    /// # Profiling
    ///
    /// - CPU scope is automatically created by `RenderGraph` using `name()`
    /// - GPU timestamps are injected by `ctx.begin_render_pass()`
    ///
    /// # Errors
    ///
    /// Returns `Err` if GPU command recording fails (rare).
    fn execute(&mut self, ctx: &mut PassContext) -> Result<()>;

    /// Publishes outputs into the shared frame-resource contract for later passes.
    ///
    /// Passes should expose only stable resource contracts here (e.g. GBuffer,
    /// shadow atlas, SSAO, pre-AA) rather than pass-specific implementation types.
    fn publish<'a>(&'a self, _frame: &mut libhelio::FrameResources<'a>) {}

    /// Build a reusable render bundle for passes that require no per-frame CPU work.
    ///
    /// If the pass can record all GPU draw commands in advance, return `Some(bundle)`.
    /// The graph will skip `prepare()` / `execute()` at runtime and replay the bundle
    /// inside the pass's render pass descriptor.
    ///
    /// The default implementation returns `None`, which means the pass will execute
    /// dynamically every frame.
    fn build_gpu_render_bundle(
        &mut self,
        _device: &wgpu::Device,
        _resources: &libhelio::FrameResources<'_>,
    ) -> Option<wgpu::RenderBundle> {
        None
    }

    /// Declares the render pass this pass needs, or `None` if the pass only
    /// performs compute dispatches.  Every pass **must** implement this —
    /// there is no default.  Compute-only passes return `None`; render passes
    /// return `Some(descriptor)` so the executor can manage the render pass
    /// lifecycle (subpass chaining, store-op patching, tile-memory fusion).
    fn render_pass_descriptor<'a>(
        &'a self,
        target: &'a wgpu::TextureView,
        depth: &'a wgpu::TextureView,
        resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>>;

    /// Returns true if this pass's `execute()` never touches the main render
    /// encoder (`ctx.encoder_ptr` / `ctx.active_render_pass`) — only
    /// `ctx.compute_encoder_ptr` / `ctx.begin_compute_pass()`. Such passes may be
    /// placed inside an active subpass chain without closing the chain's open
    /// render pass, since their GPU work is recorded on a separate encoder that
    /// cannot conflict with it.
    ///
    /// Default `false`. Only opt in after auditing every encoder touch in
    /// `execute()` — a single stray `ctx.encoder_ptr` use while the chain's
    /// render pass is open is a real hazard, not just a missed optimization.
    fn chain_transparent(&self) -> bool {
        false
    }

    /// Optionally prepares per-frame data before GPU execution.
    ///
    /// Called once per frame **before** `execute()`. Use this to upload per-frame uniforms
    /// (e.g., camera matrices, time, light data) to GPU buffers.
    ///
    /// # Parameters
    ///
    /// - `ctx`: Context with device, queue, and frame counter
    ///
    /// # Performance
    ///
    /// - **Minimize uploads**: Only upload changed data (use dirty tracking)
    /// - **Batch writes**: Use `write_buffer()` instead of multiple small uploads
    ///
    /// # Errors
    ///
    /// Returns `Err` if GPU upload fails (rare).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use helio_core::{RenderPass, PassContext, PrepareContext, Result};
    /// # struct MyPass { uniform_buffer: wgpu::Buffer }
    /// # impl RenderPass for MyPass {
    /// #     fn name(&self) -> &'static str { "MyPass" }
    /// #     fn execute(&mut self, _: &mut PassContext) -> Result<()> { Ok(()) }
    /// fn prepare(&mut self, ctx: &PrepareContext) -> Result<()> {
    ///     let uniforms = MyUniforms {
    ///         time: ctx.frame as f32,
    ///     };
    ///     ctx.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    ///     Ok(())
    /// }
    /// # }
    /// # #[repr(C)]
    /// # #[derive(Copy, Clone)]
    /// # struct MyUniforms { time: f32 }
    /// # unsafe impl bytemuck::Pod for MyUniforms {}
    /// # unsafe impl bytemuck::Zeroable for MyUniforms {}
    /// ```
    fn prepare(&mut self, _ctx: &PrepareContext) -> Result<()> {
        Ok(())
    }

    /// Called when the render target is resized.
    ///
    /// Override to recreate size-dependent resources (pipelines, textures, bind groups).
    /// The default is a no-op, so passes that don't need resize handling can ignore it.
    fn on_resize(&mut self, _device: &wgpu::Device, _width: u32, _height: u32) {}

    /// Updates the renderer-wide debug visualisation mode.
    ///
    /// Passes that expose [`debug_views`](Self::debug_views) must override this
    /// hook and update the state consumed by their next `prepare`/`execute`.
    /// The default keeps passes without debug visualisations source-compatible.
    fn set_debug_mode(&mut self, _mode: u32) {}

    /// Returns the debug visualisation modes this pass provides.
    ///
    /// The renderer aggregates these from all passes to build a discoverable
    /// list of debug views that can be cycled at runtime.
    fn debug_views(&self) -> &'static [DebugViewDescriptor] {
        &[]
    }

    /// Declares the inter-pass resources this pass reads and writes.
    ///
    /// The graph uses these declarations to:
    /// - Create and own all inter-pass textures (removing per-pass allocation)
    /// - Route texture views through `FrameResources` automatically
    /// - Alias non-overlapping resources to reduce peak VRAM
    /// - Fuse linear A→B chains into subpasses (zero intermediate storage)
    ///
    /// Override this to declare dependencies:
    ///
    /// ```rust,ignore
    /// fn declare_resources(&self, builder: &mut ResourceBuilder) {
    ///     builder.read("gbuffer_albedo");
    ///     builder.read("depth");
    ///     builder.write_color("pre_aa", wgpu::TextureFormat::Rgba16Float, ResSize::Internal);
    /// }
    /// ```
    ///
    /// The default implementation returns reads/writes from the legacy
    /// `ResourceSlot`-based [`reads`](Self::reads) / [`writes`](Self::writes)
    /// methods for backward compatibility.
    fn declare_resources(&self, _builder: &mut ResourceBuilder) {}
}
