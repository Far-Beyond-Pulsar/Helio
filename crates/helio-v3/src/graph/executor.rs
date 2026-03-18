//! Render graph executor with automatic profiling.
//!
//! The render graph orchestrates pass execution with automatic profiling injection.
//! It is the top-level coordinator of the rendering pipeline.
//!
//! # Design Pattern: Graph Execution
//!
//! Helio v3 uses a **linear graph executor** (future: DAG with parallelism):
//!
//! 1. **Add passes**: `graph.add_pass(Box::new(ShadowPass::new(...)))`
//! 2. **Execute in order**: Passes run sequentially in the order they were added
//! 3. **Automatic profiling**: CPU scopes and GPU timestamps injected per-pass
//! 4. **Zero-copy contexts**: Each pass receives a `PassContext` with borrowed references
//!
//! # Architecture
//!
//! ```text
//! RenderGraph
//! ├── passes: Vec<Box<dyn RenderPass>>
//! ├── profiler: Profiler (automatic CPU/GPU profiling)
//! └── execute()
//!     ├── Create command encoder
//!     ├── For each pass:
//!     │   ├── profiler.scope(pass.name()) (CPU profiling)
//!     │   ├── pass.prepare(&ctx)          (upload uniforms)
//!     │   ├── pass.execute(&mut ctx)      (record GPU commands)
//!     │   │   ├── ctx.begin_render_pass() (GPU profiling)
//!     │   │   └── GPU commands
//!     │   └── ScopeGuard::drop()          (CPU profiling end)
//!     └── queue.submit(encoder.finish())
//! ```
//!
//! # Performance
//!
//! - **O(passes)**: Linear execution (future: parallel with DAG)
//! - **Zero allocations**: Passes and profiler are pre-allocated
//! - **Zero clones**: PassContext uses borrowed references
//!
//! # Example
//!
//! ```rust,no_run
//! use helio_v3::{RenderGraph, GpuScene};
//! use std::sync::Arc;
//!
//! let mut graph = RenderGraph::new(&device, &queue);
//! let scene = GpuScene::new(Arc::new(device), Arc::new(queue));
//!
//! // Add passes (order matters)
//! // graph.add_pass(Box::new(ShadowPass::new(&device)));
//! // graph.add_pass(Box::new(GBufferPass::new(&device)));
//! // graph.add_pass(Box::new(DeferredLightPass::new(&device)));
//! // graph.add_pass(Box::new(BloomPass::new(&device)));
//!
//! // Render loop
//! // loop {
//! //     let target = surface.get_current_texture().unwrap();
//! //     let view = target.texture.create_view(&Default::default());
//! //     graph.execute(&scene, &view, &depth_view).unwrap();
//! //     target.present();
//! // }
//! ```

use crate::{RenderPass, GpuScene, PassContext, Profiler, Result};

/// Render graph executor with automatic profiling.
///
/// `RenderGraph` orchestrates pass execution with:
/// - **Automatic profiling**: CPU scopes and GPU timestamps injected per-pass
/// - **Zero-copy contexts**: Passes receive borrowed references to scene resources
/// - **Linear execution**: Passes run in the order they were added (future: DAG with parallelism)
///
/// # Design
///
/// The graph maintains a list of passes (`Vec<Box<dyn RenderPass>>`) and executes them
/// sequentially. Each pass receives a `PassContext` with zero-copy access to scene resources.
///
/// # Lifecycle
///
/// ```text
/// RenderGraph::new(device, queue)
/// ├── Add passes: graph.add_pass(Box::new(...))
/// └── Execute: graph.execute(&scene, target, depth)
///     ├── Create command encoder
///     ├── For each pass:
///     │   ├── CPU scope (automatic profiling)
///     │   ├── pass.prepare(&ctx) (upload uniforms)
///     │   ├── pass.execute(&mut ctx) (record GPU commands)
///     │   └── GPU timestamps (automatic profiling)
///     └── Submit to queue
/// ```
///
/// # Performance
///
/// - **O(passes)**: Linear execution (sequential, not parallel yet)
/// - **Zero allocations**: Passes and profiler are pre-allocated
/// - **Zero clones**: PassContext uses borrowed references
///
/// # Example
///
/// ```rust,no_run
/// use helio_v3::{RenderGraph, GpuScene, RenderPass, PassContext, Result};
/// use std::sync::Arc;
///
/// // Define a simple pass
/// struct SimplePass {
///     pipeline: wgpu::RenderPipeline,
/// }
///
/// impl RenderPass for SimplePass {
///     fn name(&self) -> &'static str { "SimplePass" }
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
///
/// // Build the render graph
/// let mut graph = RenderGraph::new(&device, &queue);
/// let scene = GpuScene::new(Arc::new(device), Arc::new(queue));
///
/// // Add passes
/// // graph.add_pass(Box::new(SimplePass { pipeline }));
///
/// // Execute
/// // graph.execute(&scene, &target_view, &depth_view);
/// ```
pub struct RenderGraph {
    /// List of render passes (executed in order).
    ///
    /// Passes are stored as trait objects (`Box<dyn RenderPass>`) for polymorphism.
    /// Future: Replace with a DAG for parallel execution.
    passes: Vec<Box<dyn RenderPass>>,

    /// Profiler for automatic CPU/GPU profiling.
    ///
    /// Injected into `PassContext` to provide automatic profiling for passes.
    profiler: Profiler,
}

impl RenderGraph {
    /// Creates a new render graph.
    ///
    /// Initializes an empty pass list and a profiler.
    ///
    /// # Parameters
    ///
    /// - `device`: GPU device for creating profiler query sets
    /// - `queue`: GPU queue (reserved for async profiling readback)
    ///
    /// # Performance
    ///
    /// - **O(1)**: Initializes empty vector and profiler
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use helio_v3::RenderGraph;
    ///
    /// let graph = RenderGraph::new(&device, &queue);
    /// ```
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            passes: Vec::new(),
            profiler: Profiler::new(device, queue),
        }
    }

    /// Adds a render pass to the graph.
    ///
    /// Passes are executed in the order they are added. For a typical deferred pipeline:
    /// 1. Shadow passes (depth-only)
    /// 2. GBuffer pass (geometry)
    /// 3. Deferred lighting pass (fullscreen quad)
    /// 4. Post-process passes (bloom, TAA, FXAA, etc.)
    ///
    /// # Parameters
    ///
    /// - `pass`: Boxed trait object implementing `RenderPass`
    ///
    /// # Performance
    ///
    /// - **O(1)**: Appends to vector (amortized)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use helio_v3::{RenderGraph, RenderPass, PassContext, Result};
    /// # struct ShadowPass;
    /// # impl RenderPass for ShadowPass {
    /// #     fn name(&self) -> &'static str { "ShadowPass" }
    /// #     fn execute(&mut self, _: &mut PassContext) -> Result<()> { Ok(()) }
    /// # }
    /// # let mut graph = RenderGraph::new(&device, &queue);
    /// graph.add_pass(Box::new(ShadowPass));
    /// ```
    pub fn add_pass(&mut self, pass: Box<dyn RenderPass>) {
        self.passes.push(pass);
    }

    /// Executes the render graph with automatic profiling.
    ///
    /// This is the main entry point for rendering. It:
    /// 1. Creates a command encoder
    /// 2. Executes each pass in order with automatic profiling
    /// 3. Submits the command buffer to the GPU queue
    ///
    /// # Parameters
    ///
    /// - `scene`: GPU scene with dirty-tracked state (must call `scene.flush()` first)
    /// - `target`: Color render target (swapchain texture or offscreen buffer)
    /// - `depth`: Depth/stencil buffer (shared across all passes)
    ///
    /// # Performance
    ///
    /// - **O(passes)**: Linear execution (sequential, not parallel yet)
    /// - **Zero allocations**: Encoder and context are stack-allocated
    /// - **Zero clones**: All resource access is by reference
    ///
    /// # Profiling
    ///
    /// - CPU scopes created automatically for each pass (using `pass.name()`)
    /// - GPU timestamps injected via `PassContext::begin_render_pass()`
    /// - Results exported to `helio-live-portal` for real-time telemetry
    ///
    /// # Errors
    ///
    /// Returns `Err` if any pass fails (rare - typically shader compilation errors).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use helio_v3::{RenderGraph, GpuScene};
    /// # use std::sync::Arc;
    /// # let mut graph = RenderGraph::new(&device, &queue);
    /// # let mut scene = GpuScene::new(Arc::new(device), Arc::new(queue));
    /// # let target = &view;
    /// # let depth = &depth_view;
    /// // Update scene objects
    /// // scene.lights.add(light);
    /// // scene.meshes.update(id, mesh);
    ///
    /// // Flush changes to GPU (zero-cost if nothing changed)
    /// scene.flush();
    ///
    /// // Execute render graph (automatic profiling)
    /// graph.execute(&scene, target, depth).unwrap();
    /// ```
    ///
    /// # Profiling Flow
    ///
    /// ```text
    /// execute()
    /// ├── Create encoder
    /// ├── Pass 1: "ShadowPass"
    /// │   ├── CPU scope start (automatic)
    /// │   ├── prepare(&ctx)
    /// │   ├── execute(&mut ctx)
    /// │   │   ├── GPU timestamp start (automatic)
    /// │   │   ├── GPU commands
    /// │   │   └── GPU timestamp end (automatic)
    /// │   └── CPU scope end (automatic)
    /// ├── Pass 2: "GBufferPass"
    /// │   └── ...
    /// └── Submit to queue
    /// ```
    pub fn execute(
        &mut self,
        scene: &GpuScene,
        target: &wgpu::TextureView,
        depth: &wgpu::TextureView,
    ) -> Result<()> {
        let mut encoder = scene.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Graph"),
        });

        // Execute passes with automatic profiling
        for pass in &mut self.passes {
            // CPU profiling scope (automatic)
            let _scope = self.profiler.scope(pass.name());

            // Build context with zero-copy references (scoped to each pass)
            let mut ctx = PassContext {
                encoder: &mut encoder,
                target,
                depth,
                scene: scene.resources(),
                profiler: &mut self.profiler,
                frame: scene.frame_count,
                width: scene.width,
                height: scene.height,
            };

            // Execute pass (GPU profiling injected via ctx.begin_render_pass)
            pass.execute(&mut ctx)?;
        }

        // Submit command buffer to GPU
        scene.queue.submit([encoder.finish()]);
        Ok(())
    }
}
