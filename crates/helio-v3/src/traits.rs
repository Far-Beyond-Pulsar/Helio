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

use crate::graph::ResourceBuilder;
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

    /// Optionally declares resource dependencies (future feature).
    ///
    /// Used for automatic resource lifetime management and pass reordering.
    /// Currently a stub for future graph optimization.
    ///
    /// # Parameters
    ///
    /// - `builder`: Resource dependency builder
    ///
    /// # Example (Future API)
    ///
    /// ```rust,no_run
    /// # use helio_v3::{RenderPass, PassContext, Result};
    /// # use helio_v3::graph::ResourceBuilder;
    /// # struct MyPass;
    /// # impl RenderPass for MyPass {
    /// #     fn name(&self) -> &'static str { "MyPass" }
    /// #     fn execute(&mut self, _: &mut PassContext) -> Result<()> { Ok(()) }
    /// fn declare_resources(&self, builder: &mut ResourceBuilder) {
    ///     builder.read("gbuffer_albedo");
    ///     builder.read("gbuffer_normal");
    ///     builder.write("final_color");
    /// }
    /// # }
    /// ```
    fn declare_resources(&self, _builder: &mut ResourceBuilder) {}

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

    /// Returns a shared reference to `self` as `dyn Any` for downcasting.
    ///
    /// Implement this to allow the render graph to retrieve a typed reference
    /// to a specific pass via `RenderGraph::find_pass_mut`. The default returns
    /// `None`; return `Some(self)` in concrete implementations.
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }

    /// Returns a mutable reference to `self` as `dyn Any` for downcasting.
    ///
    /// Implement this to allow the render graph to retrieve a typed mutable
    /// reference to a specific pass via `RenderGraph::find_pass_mut`. The
    /// default returns `None`; return `Some(self)` in concrete implementations.
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        None
    }
}

/// Trait for GPU scene managers (lights, meshes, materials).
///
/// `GpuSceneManager` provides a unified interface for managing GPU buffers with **dirty tracking**.
/// Each manager (e.g., `LightBuffer`, `MeshBuffer`) maintains a CPU mirror of GPU data and uploads
/// only changed data to the GPU via `flush()`.
///
/// # Contract
///
/// Implementations must:
/// - Track **dirty state** internally (e.g., `dirty: bool` flag)
/// - Skip GPU upload in `flush()` if `dirty == false` (zero cost at steady state)
/// - Provide **zero-copy access** to GPU buffers via `buffer()`
///
/// # Performance
///
/// - **O(changed)**: `flush()` uploads only changed data, not entire buffer
/// - **Zero cost at steady state**: If no changes, `flush()` is a no-op
/// - **No clones**: `buffer()` returns `&wgpu::Buffer` (borrowed reference)
///
/// # Example Implementation
///
/// ```rust,no_run
/// use helio_v3::GpuSceneManager;
///
/// pub struct LightBuffer {
///     buffer: wgpu::Buffer,
///     lights: Vec<GpuLight>,
///     dirty: bool,
/// }
///
/// impl GpuSceneManager for LightBuffer {
///     fn flush(&mut self, queue: &wgpu::Queue) {
///         if !self.dirty {
///             return; // Zero cost at steady state
///         }
///         queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&self.lights));
///         self.dirty = false;
///     }
///
///     fn buffer(&self) -> &wgpu::Buffer {
///         &self.buffer
///     }
///
///     fn count(&self) -> u32 {
///         self.lights.len() as u32
///     }
/// }
/// # #[repr(C)]
/// # #[derive(Copy, Clone)]
/// # struct GpuLight { position: [f32; 3], _pad: f32 }
/// # unsafe impl bytemuck::Pod for GpuLight {}
/// # unsafe impl bytemuck::Zeroable for GpuLight {}
/// ```
pub trait GpuSceneManager: MaybeSend + MaybeSync {
    /// Flushes dirty data to the GPU.
    ///
    /// Uploads changed data from the CPU mirror to the GPU buffer. Should be a **no-op**
    /// if nothing changed (dirty tracking).
    ///
    /// # Performance
    ///
    /// - **O(changed)**: Upload only changed data, not entire buffer
    /// - **O(1) at steady state**: If `dirty == false`, this is a no-op
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use helio_v3::GpuSceneManager;
    /// # struct LightBuffer { buffer: wgpu::Buffer, lights: Vec<u8>, dirty: bool }
    /// # impl GpuSceneManager for LightBuffer {
    /// #     fn buffer(&self) -> &wgpu::Buffer { &self.buffer }
    /// #     fn count(&self) -> u32 { 0 }
    /// fn flush(&mut self, queue: &wgpu::Queue) {
    ///     if !self.dirty {
    ///         return; // Zero cost at steady state
    ///     }
    ///     queue.write_buffer(&self.buffer, 0, &self.lights);
    ///     self.dirty = false;
    /// }
    /// # }
    /// ```
    fn flush(&mut self, queue: &wgpu::Queue);

    /// Returns a reference to the GPU buffer.
    ///
    /// Provides zero-copy access to the underlying `wgpu::Buffer`. Used by passes
    /// to create bind groups or access buffer metadata.
    ///
    /// # Performance
    ///
    /// - **O(1)**: Returns a borrowed reference (no clones)
    fn buffer(&self) -> &wgpu::Buffer;

    /// Optionally returns the bind group for this resource.
    ///
    /// If the manager maintains a persistent bind group, return it here.
    /// Otherwise, passes can create bind groups on-the-fly using `buffer()`.
    ///
    /// # Performance
    ///
    /// - **O(1)**: Returns a borrowed reference (no clones)
    fn bind_group(&self) -> Option<&wgpu::BindGroup> {
        None
    }

    /// Returns the current number of items in the buffer.
    ///
    /// Used by passes to set dynamic offsets or dispatch counts.
    ///
    /// # Performance
    ///
    /// - **O(1)**: Returns cached count
    fn count(&self) -> u32;
}

/// Trait for GPU resources with automatic capacity management.
///
/// `GpuResource` extends `GpuSceneManager` with automatic buffer growth. When item count
/// exceeds capacity, the buffer is reallocated with increased capacity (e.g., 2x growth).
///
/// # Contract
///
/// Implementations must:
/// - Provide **capacity tracking** (current buffer size)
/// - Implement **buffer growth** logic (allocate new buffer, copy old data)
///
/// # Performance
///
/// - **Amortized O(1)**: Rare reallocations amortize to constant time
/// - **No shrinking**: Buffers never shrink (reduces reallocations)
///
/// # Example Implementation
///
/// ```rust,no_run
/// use helio_v3::GpuResource;
///
/// pub struct GrowableBuffer {
///     buffer: wgpu::Buffer,
///     count: u32,
///     capacity: u32,
/// }
///
/// impl GpuResource for GrowableBuffer {
///     fn buffer(&self) -> &wgpu::Buffer {
///         &self.buffer
///     }
///
///     fn count(&self) -> u32 {
///         self.count
///     }
///
///     fn capacity(&self) -> u32 {
///         self.capacity
///     }
///
///     fn grow(&mut self, device: &wgpu::Device, new_capacity: u32) {
///         let new_buffer = device.create_buffer(&wgpu::BufferDescriptor {
///             label: Some("Growable Buffer"),
///             size: (new_capacity * std::mem::size_of::<u32>() as u32) as u64,
///             usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
///             mapped_at_creation: false,
///         });
///         self.buffer = new_buffer;
///         self.capacity = new_capacity;
///     }
/// }
/// ```
pub trait GpuResource {
    /// Returns a reference to the GPU buffer.
    ///
    /// # Performance
    ///
    /// - **O(1)**: Returns a borrowed reference (no clones)
    fn buffer(&self) -> &wgpu::Buffer;

    /// Returns the current number of items in the buffer.
    ///
    /// # Performance
    ///
    /// - **O(1)**: Returns cached count
    fn count(&self) -> u32;

    /// Returns the current buffer capacity (maximum items before reallocation).
    ///
    /// # Performance
    ///
    /// - **O(1)**: Returns cached capacity
    fn capacity(&self) -> u32;

    /// Checks if the buffer needs to grow to accommodate `new_count` items.
    ///
    /// Default implementation: `new_count > self.capacity()`
    ///
    /// # Performance
    ///
    /// - **O(1)**: Simple comparison
    fn needs_grow(&self, new_count: u32) -> bool {
        new_count > self.capacity()
    }

    /// Grows the buffer to `new_capacity`.
    ///
    /// Allocates a new GPU buffer with increased capacity. The old buffer is dropped
    /// automatically. Implementations should copy old data to the new buffer if needed.
    ///
    /// # Performance
    ///
    /// - **Rare operation**: Only happens when capacity is exceeded
    /// - **Amortized O(1)**: Exponential growth (e.g., 2x) ensures rare reallocations
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use helio_v3::GpuResource;
    /// # struct MyBuffer { buffer: wgpu::Buffer, capacity: u32 }
    /// # impl GpuResource for MyBuffer {
    /// #     fn buffer(&self) -> &wgpu::Buffer { &self.buffer }
    /// #     fn count(&self) -> u32 { 0 }
    /// #     fn capacity(&self) -> u32 { self.capacity }
    /// fn grow(&mut self, device: &wgpu::Device, new_capacity: u32) {
    ///     self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
    ///         label: Some("Growable Buffer"),
    ///         size: (new_capacity * 16) as u64, // 16 bytes per item
    ///         usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    ///         mapped_at_creation: false,
    ///     });
    ///     self.capacity = new_capacity;
    /// }
    /// # }
    /// ```
    fn grow(&mut self, device: &wgpu::Device, new_capacity: u32);
}

