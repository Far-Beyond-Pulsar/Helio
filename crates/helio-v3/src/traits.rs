//! Core traits for helio-v3 modular renderer.
//!
//! This module defines the fundamental traits that power helio-v3's modular architecture:
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
//! use helio_v3::{RenderPass, PassContext, PrepareContext, Result};
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

use crate::{PassContext, PrepareContext, Result};

/// Core trait for all rendering passes.
///
/// `RenderPass` is the fundamental building block of helio-v3. Each pass is a self-contained
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
/// use helio_v3::{RenderPass, PassContext, Result};
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
/// use helio_v3::{RenderPass, PassContext, PrepareContext, Result};
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
pub trait RenderPass: MaybeSend + MaybeSync {
    /// Returns a unique name for this pass.
    ///
    /// Used for profiling labels and debugging. Should be a human-readable identifier
    /// (e.g., "ShadowPass", "GBuffer", "DeferredLighting").
    ///
    /// # Performance
    ///
    /// - **O(1)**: Returns a static string (no allocations)
    fn name(&self) -> &'static str;

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
    /// # use helio_v3::{RenderPass, PassContext, PrepareContext, Result};
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

    /// Returns a shared reference to `self` as `dyn Any` for downcasting.
    ///
    /// Required by every concrete pass so `RenderGraph::find_pass` can downcast.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Returns a mutable reference to `self` as `dyn Any` for downcasting.
    ///
    /// Required by every concrete pass so `RenderGraph::find_pass_mut` can downcast.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

