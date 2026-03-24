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
    /// Command encoder for recording GPU commands.
    ///
    /// Use this to begin render passes, compute passes, or issue GPU commands.
    /// Passes receive **exclusive access** (`&mut`) to ensure no data races.
    pub encoder: &'a mut wgpu::CommandEncoder,

    /// Color render target (main framebuffer or offscreen texture).
    ///
    /// This is the texture that the pass renders into. For the final pass,
    /// this is typically the swapchain texture.
    pub target: &'a wgpu::TextureView,

    /// Depth/stencil buffer.
    ///
    /// Used for depth testing and stencil operations. Shared across all passes
    /// in a frame.
    pub depth: &'a wgpu::TextureView,

    /// Zero-copy scene resources (lights, meshes, materials).
    ///
    /// Provides borrowed references to GPU buffers. Passes access scene data
    /// via `scene.lights.buffer()`, `scene.meshes.buffer()`, etc.
    pub scene: SceneResources<'a>,

    /// Profiler (automatic - injected by RenderGraph).
    ///
    /// Used internally by `begin_render_pass()` and `begin_compute_pass()` to
    /// inject GPU timestamp queries. Not directly accessible to passes.
    pub(crate) profiler: &'a mut Profiler,

    /// Current frame number (starts at 0).
    ///
    /// Useful for time-based effects (e.g., animations, TAA jitter).
    pub frame_num: u64,

    /// Render target width in pixels.
    pub width: u32,

    /// Render target height in pixels.
    pub height: u32,

    /// Device reference for creating bind groups in execute() if needed (rare).
    pub device: &'a wgpu::Device,

    /// Per-frame transient resource views (GBuffer, HiZ, shadow atlas, sky LUT, etc.)
    pub frame: &'a libhelio::FrameResources<'a>,
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
    /// - Results are exported to `helio-live-portal` for real-time telemetry
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
        let label = desc.label.unwrap_or("unnamed");
        self.profiler.begin_gpu_pass(self.encoder, label);

        self.encoder.begin_render_pass(desc)
        // TODO: Wrap in AutoProfiledRenderPass with Drop for end_gpu_pass
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
    /// - Results are exported to `helio-live-portal` for real-time telemetry
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
        let label = desc.label.unwrap_or("unnamed");
        self.profiler.begin_gpu_pass(self.encoder, label);

        self.encoder.begin_compute_pass(desc)
        // TODO: Wrap in AutoProfiledComputePass
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
    pub frame: u64,

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

