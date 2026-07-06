//! Context types passed to render passes.
//!
//! This module defines two context types that provide passes with access to GPU resources:
//!
//! - [`PassContext`] - Passed to `RenderPass::execute()` for recording GPU commands
//! - [`PrepareContext`] - Passed to `RenderPass::prepare()` for uploading uniforms
//!
//! # Design Pattern: Zero-Copy Access
//!
//! Contexts provide **borrowed references** to GPU resources, never owned copies.
//! This ensures:
//!
//! - **Zero clones**: All access is by reference (`&wgpu::Buffer`, not `wgpu::Buffer`)
//! - **Zero allocations**: No `Box`, `Arc`, or `Vec` in the hot path
//! - **Zero locks**: No `Arc<Mutex<_>>` or `RwLock<_>` (single-threaded per frame)
//!
//! # Lifecycle
//!
//! For each frame, contexts are created by `RenderGraph`:
//!
//! ```text
//! RenderGraph::execute()
//! ├── scene.flush() (upload dirty data)
//! └── for each pass:
//!     ├── PrepareContext created
//!     ├── pass.prepare(&ctx) (upload uniforms)
//!     ├── PassContext created
//!     └── pass.execute(&mut ctx) (record GPU commands)
//! ```
//!
//! # Performance
//!
//! - **O(1)**: Context creation is constant-time (no allocations)
//! - **Zero-copy**: All fields are references (no clones)
//! - **RAII**: Contexts are scoped to each pass (automatic cleanup)
//!
//! # Example: Using PassContext
//!
//! ```rust,no_run
//! use helio_v3::{RenderPass, PassContext, Result};
//!
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
//!         // Access render target and depth buffer
//!         let target = ctx.target;
//!         let depth = ctx.depth;
//!
//!         // Access scene resources (zero-copy)
//!         // let light_buffer = ctx.scene.lights.buffer();
//!         // let mesh_buffer = ctx.scene.meshes.buffer();
//!
//!         // Record GPU commands with automatic profiling
//!         let mut pass = ctx.begin_render_pass(&wgpu::RenderPassDescriptor {
//!             label: Some("MyPass"),
//!             color_attachments: &[Some(wgpu::RenderPassColorAttachment {
//!                 view: target,
//!                 resolve_target: None,
//!                 ops: wgpu::Operations {
//!                     load: wgpu::LoadOp::Load,
//!                     store: wgpu::StoreOp::Store,
//!                 },
//!             })],
//!             depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
//!                 view: depth,
//!                 depth_ops: Some(wgpu::Operations {
//!                     load: wgpu::LoadOp::Clear(1.0),
//!                     store: wgpu::StoreOp::Store,
//!                 }),
//!                 stencil_ops: None,
//!             }),
//!             timestamp_writes: None,
//!             occlusion_query_set: None,
//!         });
//!
//!         pass.set_pipeline(&self.pipeline);
//!         pass.draw(0..3, 0..1);
//!
//!         Ok(())
//!     }
//! }
//! ```

use crate::scene::GpuScene;
use crate::{Profiler, SceneResources};

/// Context passed to `RenderPass::execute()` for recording GPU commands.
///
/// `PassContext` provides zero-copy access to:
/// - **Render targets**: `target` (color) and `depth` (depth/stencil)
/// - **Scene resources**: `scene` (lights, meshes, materials via borrowed buffers)
/// - **Command encoder**: `encoder` (for recording GPU commands)
/// - **Profiler**: Automatic CPU/GPU profiling (injected by `RenderGraph`)
///
/// # Lifetime
///
/// The `'a` lifetime ensures that all borrowed resources outlive the context.
/// This prevents dangling references and ensures zero-copy safety.
///
/// # Design Pattern: Zero-Copy Access
///
/// All fields are **borrowed references** (`&`, not owned types). This ensures:
/// - **Zero clones**: No `Arc::clone()` or `buffer.clone()`
/// - **Zero allocations**: No `Box`, `Vec`, or heap allocations
/// - **Type safety**: Rust's borrow checker enforces exclusive access to `encoder`
///
/// # Profiling
///
/// The `profiler` field is **private** and injected by `RenderGraph`.
/// Profiling is automatic when using `begin_render_pass()` or `begin_compute_pass()`.
///
/// # Example
///
/// ```rust,no_run
/// use helio_v3::{RenderPass, PassContext, Result};
///
/// struct MyPass {
///     pipeline: wgpu::RenderPipeline,
/// }
///
/// impl RenderPass for MyPass {
///     fn name(&self) -> &'static str {
///         "MyPass"
///     }
///
///     fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
///         // Access render targets
///         let target = ctx.target;
///         let depth = ctx.depth;
///
///         // Access frame metadata
///         let frame_num = ctx.frame_num;
///         let (width, height) = (ctx.width, ctx.height);
///
///         // Record GPU commands (automatic profiling)
///         let mut pass = ctx.begin_render_pass(&wgpu::RenderPassDescriptor {
///             label: Some("MyPass"),
///             color_attachments: &[Some(wgpu::RenderPassColorAttachment {
///                 view: target,
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
pub struct PassContext<'a> {
    /// Command encoder for non-render-pass GPU ops (buffer clears, copies).
    /// Passes do NOT call begin_render_pass on this — the executor does that.
    /// Access via `unsafe { &mut *ctx.encoder_ptr }`.
    pub encoder_ptr: *mut wgpu::CommandEncoder,

    /// Separate compute encoder for compute dispatches (always available, even
    /// during a render pass on the render encoder).  Access via unsafe.
    pub compute_encoder_ptr: *mut wgpu::CommandEncoder,

    /// Color render target (main framebuffer or offscreen texture).
    pub target: &'a wgpu::TextureView,

    /// Depth/stencil buffer.
    pub depth: &'a wgpu::TextureView,

    /// Zero-copy scene resources (lights, meshes, materials).
    pub scene: SceneResources<'a>,

    /// Profiler (automatic - injected by RenderGraph).
    #[allow(dead_code)]
    pub(crate) profiler: &'a mut Profiler,

    /// Current frame number (starts at 0).
    pub frame_num: u64,

    /// Render target width in pixels.
    pub width: u32,

    /// Render target height in pixels.
    pub height: u32,

    /// Device reference for creating bind groups in execute() if needed (rare).
    pub device: &'a wgpu::Device,

    /// Per-frame transient resource views.
    pub resources: &'a libhelio::FrameResources<'a>,

    /// Subpass index within a fused render-pass chain.
    pub subpass_index: u32,

    /// When `false`, Helio does not own the wgpu device.
    pub owns_device: bool,

    /// Graph-owned texture pool.
    pub resource_pool: &'a crate::graph::GraphTexturePool,

    /// Active render pass, or None if not in a render pass.
    /// Set by the executor BEFORE calling execute(). The pass draws into
    /// this instead of calling begin_render_pass().
    pub active_render_pass: Option<*mut wgpu::RenderPass<'static>>,
    /// Active compute pass, or None if not in a compute pass.
    pub active_compute_pass: Option<*mut wgpu::ComputePass<'static>>,
}

impl<'a> PassContext<'a> {
    /// Returns a raw pointer to the active render pass, if any.
    /// Cast to a reference in the pass: `let rp = unsafe { &mut *ctx.active_render_pass()? };`
    #[inline]
    pub fn active_render_pass_ptr(&self) -> Option<*mut wgpu::RenderPass<'static>> {
        self.active_render_pass
    }

    /// Returns a raw pointer to the active compute pass, if any.
    #[inline]
    pub fn active_compute_pass_ptr(&self) -> Option<*mut wgpu::ComputePass<'static>> {
        self.active_compute_pass
    }
}

impl<'a> PassContext<'a> {
    /// Begins a render pass with automatic GPU profiling.
    ///
    /// This is a wrapper around `encoder.begin_render_pass()` that automatically
    /// injects GPU timestamp queries for profiling. **Always use this instead of
    /// calling `encoder.begin_render_pass()` directly.**
    ///
    /// # Profiling
    ///
    /// - GPU timestamps are written at the start and end of the pass
    /// - Results are available for external telemetry systems
    /// - Zero overhead when `profiling` feature is disabled
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use helio_v3::{RenderPass, PassContext, Result};
    /// # struct MyPass;
    /// # impl RenderPass for MyPass {
    /// #     fn name(&self) -> &'static str { "MyPass" }
    /// fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
    ///     let mut pass = ctx.begin_render_pass(&wgpu::RenderPassDescriptor {
    ///         label: Some("MyPass"),
    ///         color_attachments: &[Some(wgpu::RenderPassColorAttachment {
    ///             view: ctx.target,
    ///             resolve_target: None,
    ///             ops: wgpu::Operations {
    ///                 load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
    ///                 store: wgpu::StoreOp::Store,
    ///             },
    ///         })],
    ///         depth_stencil_attachment: None,
    ///         timestamp_writes: None,
    ///         occlusion_query_set: None,
    ///     });
    ///
    ///     // Record GPU commands
    ///     // pass.set_pipeline(&self.pipeline);
    ///     // pass.draw(0..3, 0..1);
    ///
    ///     Ok(())
    /// }
    /// # }
    /// ```
    pub fn begin_render_pass<'b>(
        &'b mut self,
        desc: &'b wgpu::RenderPassDescriptor<'b>,
    ) -> wgpu::RenderPass<'b> {
        // TODO: GPU profiling with begin/end_gpu_pass (needs lifetime fixes)
        unsafe { (*self.encoder_ptr).begin_render_pass(desc) }
    }

    /// Begins a compute pass with automatic GPU profiling.
    ///
    /// This is a wrapper around `encoder.begin_compute_pass()` that automatically
    /// injects GPU timestamp queries for profiling. **Always use this instead of
    /// calling `encoder.begin_compute_pass()` directly.**
    ///
    /// # Profiling
    ///
    /// - GPU timestamps are written at the start and end of the pass
    /// - Results are available for external telemetry systems
    /// - Zero overhead when `profiling` feature is disabled
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use helio_v3::{RenderPass, PassContext, Result};
    /// # struct MyComputePass { pipeline: wgpu::ComputePipeline }
    /// # impl RenderPass for MyComputePass {
    /// #     fn name(&self) -> &'static str { "MyComputePass" }
    /// fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
    ///     let mut pass = ctx.begin_compute_pass(&wgpu::ComputePassDescriptor {
    ///         label: Some("MyComputePass"),
    ///         timestamp_writes: None,
    ///     });
    ///
    ///     pass.set_pipeline(&self.pipeline);
    ///     pass.dispatch_workgroups(256, 1, 1);
    ///
    ///     Ok(())
    /// }
    /// # }
    /// ```
    pub fn begin_compute_pass<'b>(
        &'b mut self,
        desc: &'b wgpu::ComputePassDescriptor<'b>,
    ) -> wgpu::ComputePass<'b> {
        // Uses the separate compute encoder so compute work never conflicts with
        // an active render pass on the render encoder (migrated path).
        unsafe { (*self.compute_encoder_ptr).begin_compute_pass(desc) }
    }
}

/// Context passed to `RenderPass::prepare()` for uploading per-frame uniforms.
///
/// `PrepareContext` provides access to:
/// - **Device**: For creating buffers/textures (if needed)
/// - **Queue**: For uploading data to GPU (`write_buffer()`, `write_texture()`)
/// - **Frame counter**: For time-based effects
///
/// # Lifecycle
///
/// `prepare()` is called **before** `execute()` for each pass. Use it to upload
/// per-frame data (e.g., camera matrices, time, light counts) to GPU buffers.
///
/// ```text
/// for each pass:
///     prepare(&ctx)  <- Upload uniforms (CPU -> GPU)
///     execute(&ctx)  <- Record GPU commands
/// ```
///
/// # Performance
///
/// - **Minimize uploads**: Only upload changed data (use dirty tracking)
/// - **Batch writes**: Use `write_buffer()` instead of many small uploads
/// - **Pre-allocate**: Create buffers once in pass constructor, not in `prepare()`
///
/// # Example
///
/// ```rust,no_run
/// use helio_v3::{RenderPass, PassContext, PrepareContext, Result};
///
/// struct MyPass {
///     pipeline: wgpu::RenderPipeline,
///     uniform_buffer: wgpu::Buffer,
/// }
///
/// impl RenderPass for MyPass {
///     fn name(&self) -> &'static str {
///         "MyPass"
///     }
///
///     fn prepare(&mut self, ctx: &PrepareContext) -> Result<()> {
///         // Upload per-frame uniforms
///         let uniforms = MyUniforms {
///             time: ctx.frame as f32,
///             resolution: [1920, 1080],
///         };
///         ctx.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
///         Ok(())
///     }
///
///     fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
///         // Use uploaded uniforms
///         let mut pass = ctx.begin_render_pass(&wgpu::RenderPassDescriptor {
///             label: Some("MyPass"),
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
///         // pass.set_bind_group(0, &self.bind_group, &[]);
///         pass.draw(0..3, 0..1);
///
///         Ok(())
///     }
/// }
/// # #[repr(C)]
/// # #[derive(Copy, Clone)]
/// # struct MyUniforms { time: f32, resolution: [u32; 2] }
/// # unsafe impl bytemuck::Pod for MyUniforms {}
/// # unsafe impl bytemuck::Zeroable for MyUniforms {}
/// ```
pub struct PrepareContext<'a> {
    /// GPU device for creating buffers/textures (if needed).
    ///
    /// **Note**: Avoid creating resources in `prepare()` - create them once in
    /// the pass constructor for better performance.
    pub device: &'a wgpu::Device,

    /// Command queue for uploading data to GPU.
    ///
    /// Use `write_buffer()` to upload uniform data, or `write_texture()` for images.
    pub queue: &'a wgpu::Queue,

    /// Current frame number (starts at 0).
    ///
    /// Useful for time-based effects (e.g., animations, TAA jitter).
    pub frame_num: u64,

    /// Zero-copy scene resource references for prepare().
    pub scene: &'a GpuScene,

    /// Per-frame transient resource views (for passes that need them in prepare).
    pub frame_resources: &'a libhelio::FrameResources<'a>,

    /// True if the render target was resized this frame.
    pub resize: bool,

    /// Render target width.
    pub width: u32,

    /// Render target height.
    pub height: u32,

    /// Elapsed time since the previous frame, in seconds.
    ///
    /// Set by `RenderGraph::set_delta_time()` before each `execute()` call.
    /// The high-level `Renderer` updates this automatically.  Direct
    /// `RenderGraph` users should call `graph.set_delta_time(dt)` at the top
    /// of their render loop.
    ///
    /// Passes that need frame-rate-independent motion (water simulation,
    /// particle systems, material time-animations) should read this rather
    /// than hard-coding `0.016`.  Returns `0.0` if the host has not yet
    /// called `set_delta_time()`.
    pub delta_time: f32,
}

impl<'a> PrepareContext<'a> {
    /// Upload bytes into a GPU buffer while participating in Helio's debug upload accounting.
    pub fn write_buffer(&self, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
        crate::upload::write_buffer(self.queue, buffer, offset, data);
    }

    /// Upload bytes into a GPU texture while participating in Helio's debug upload accounting.
    pub fn write_texture(
        &self,
        texture: wgpu::TexelCopyTextureInfo<'_>,
        data: &[u8],
        data_layout: wgpu::TexelCopyBufferLayout,
        size: wgpu::Extent3d,
    ) {
        crate::upload::write_texture(self.queue, texture, data, data_layout, size);
    }
}

