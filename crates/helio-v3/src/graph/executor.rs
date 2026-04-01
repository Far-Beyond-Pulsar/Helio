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

use super::resource::{ResourceAccess, ResourceBuilder};
use crate::{GpuScene, PassContext, PrepareContext, Profiler, RenderPass, Result};
use std::collections::HashMap;

/// Transient texture managed by the render graph.
///
/// Created based on pass resource declarations, owned by the graph,
/// and borrowed during execution via FrameResources.
///
/// # Performance
///
/// - **Zero-copy**: Texture views borrowed via references
/// - **Persistent allocation**: Textures created once at graph construction
struct TransientTexture {
    /// The GPU texture (owned by the graph)
    texture: wgpu::Texture,
    /// Texture view for binding (created once, reused every frame)
    view: wgpu::TextureView,
    /// Resource name (matches declaration name)
    name: &'static str,
}

/// Render graph executor with automatic profiling and resource management.
///
/// `RenderGraph` orchestrates pass execution with:
/// - **Automatic profiling**: CPU scopes and GPU timestamps injected per-pass
/// - **Automatic resource management**: Transient textures created from declarations
/// - **Zero-copy contexts**: Passes receive borrowed references to scene resources
/// - **Linear execution**: Passes run in the order they were added
///
/// # Design
///
/// The graph maintains:
/// 1. A list of passes (`Vec<Box<dyn RenderPass>>`)
/// 2. Transient textures created from resource declarations
/// 3. A profiler for automatic CPU/GPU profiling
///
/// # Lifecycle
///
/// ```text
/// RenderGraph::new(device, queue)
/// ├── Add passes: graph.add_pass(Box::new(...))
/// │   ├── Call pass.declare_resources() to collect declarations
/// │   ├── Create transient textures for declared writes
/// │   └── Store pass in passes vector
/// └── Execute: graph.execute(&scene, target, depth)
///     ├── Create command encoder
///     ├── Populate FrameResources with transient texture views
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
/// - **Zero allocations** in render loop: All textures pre-allocated
/// - **Zero clones**: PassContext uses borrowed references
///
/// # Example
///
/// ```rust,no_run
/// use helio_v3::{RenderGraph, GpuScene, RenderPass, PassContext, Result};
/// use helio_v3::graph::{ResourceBuilder, ResourceFormat, ResourceSize};
/// use std::sync::Arc;
///
/// // Define a pass with resource declarations
/// struct BloomPass {
///     pipeline: wgpu::RenderPipeline,
/// }
///
/// impl RenderPass for BloomPass {
///     fn name(&self) -> &'static str { "BloomPass" }
///
///     fn declare_resources(&self, builder: &mut ResourceBuilder) {
///         builder.read("hdr_main");  // Read from deferred lighting
///         builder.write_color("bloom_result", ResourceFormat::Rgba16Float, ResourceSize::MatchSurface);
///     }
///
///     fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
///         // Graph automatically provides "hdr_main" and creates "bloom_result"
///         Ok(())
///     }
/// }
///
/// // Build the render graph
/// let mut graph = RenderGraph::new(&device, &queue);
/// let scene = GpuScene::new(Arc::new(device), Arc::new(queue));
///
/// // Add passes (graph auto-creates transient textures)
/// // graph.add_pass(Box::new(BloomPass { pipeline }));
///
/// // Execute (zero allocations, auto resource routing)
/// // graph.execute(&scene, &target_view, &depth_view);
/// ```
pub struct RenderGraph {
    /// List of render passes (executed in order).
    ///
    /// Passes are stored as trait objects (`Box<dyn RenderPass>`) for polymorphism.
    passes: Vec<Box<dyn RenderPass>>,

    /// Profiler for automatic CPU/GPU profiling.
    ///
    /// Injected into `PassContext` to provide automatic profiling for passes.
    profiler: Profiler,

    /// Transient textures created from resource declarations.
    ///
    /// Maps resource name → texture/view. Created during add_pass(),
    /// borrowed during execute() via FrameResources.
    ///
    /// # Performance
    ///
    /// - **Pre-allocated**: Created once during graph construction
    /// - **Zero-copy**: Views borrowed via FrameResources
    transient_textures: HashMap<&'static str, TransientTexture>,

    /// GPU device (needed for creating transient textures)
    device: std::sync::Arc<wgpu::Device>,

    /// Surface width (for ResourceSize::MatchSurface)
    width: u32,

    /// Surface height (for ResourceSize::MatchSurface)
    height: u32,
}

impl RenderGraph {
    /// Creates a new render graph.
    ///
    /// Initializes an empty pass list, profiler, and prepares for transient texture creation.
    ///
    /// # Parameters
    ///
    /// - `device`: GPU device for creating profiler query sets and transient textures
    /// - `queue`: GPU queue (reserved for async profiling readback)
    ///
    /// # Performance
    ///
    /// - **O(1)**: Initializes empty vectors and profiler
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use helio_v3::RenderGraph;
    /// use std::sync::Arc;
    ///
    /// let graph = RenderGraph::new(&Arc::new(device), &queue);
    /// ```
    pub fn new(device: &std::sync::Arc<wgpu::Device>, queue: &wgpu::Queue) -> Self {
        Self {
            passes: Vec::new(),
            profiler: Profiler::new(device, queue),
            transient_textures: HashMap::new(),
            device: device.clone(),
            width: 0,  // Set via set_render_size() before first execute()
            height: 0, // Set via set_render_size() before first execute()
        }
    }

    /// Sets the render target size and recreates transient textures.
    ///
    /// Must be called after adding passes and before first execute().
    /// Call again when window is resized to recreate size-dependent textures.
    ///
    /// # Parameters
    ///
    /// - `width`: Render target width in pixels
    /// - `height`: Render target height in pixels
    ///
    /// # Performance
    ///
    /// - **O(transient_textures)**: Recreates all graph-managed textures
    /// - Call only on resize, not every frame
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use helio_v3::RenderGraph;
    /// # use std::sync::Arc;
    /// # let device = Arc::new(todo!());
    /// # let queue = todo!();
    /// let mut graph = RenderGraph::new(&device, &queue);
    /// // Add passes...
    /// graph.set_render_size(1920, 1080);  // Before first execute()
    /// ```
    pub fn set_render_size(&mut self, width: u32, height: u32) {
        if self.width == width && self.height == height {
            return; // No change, avoid recreation
        }
        self.width = width;
        self.height = height;
        self.recreate_transient_textures();
    }

    /// Adds a render pass to the graph.
    ///
    /// Collects resource declarations from the pass and prepares for transient texture creation.
    /// Transient textures are created when `set_render_size()` is called.
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
    /// - Declarations collected but textures not created until set_render_size()
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use helio_v3::{RenderGraph, RenderPass, PassContext, Result};
    /// # use helio_v3::graph::{ResourceBuilder, ResourceFormat, ResourceSize};
    /// # struct DeferredLightPass;
    /// # impl RenderPass for DeferredLightPass {
    /// #     fn name(&self) -> &'static str { "DeferredLightPass" }
    /// #     fn declare_resources(&self, builder: &mut ResourceBuilder) {
    /// #         builder.write_color("hdr_main", ResourceFormat::Rgba16Float, ResourceSize::MatchSurface);
    /// #     }
    /// #     fn execute(&mut self, _: &mut PassContext) -> Result<()> { Ok(()) }
    /// # }
    /// # use std::sync::Arc;
    /// # let device = Arc::new(todo!());
    /// # let queue = todo!();
    /// # let mut graph = RenderGraph::new(&device, &queue);
    /// graph.add_pass(Box::new(DeferredLightPass));
    /// graph.set_render_size(1920, 1080);  // Creates transient textures
    /// ```
    pub fn add_pass(&mut self, pass: Box<dyn RenderPass>) {
        // Note: We just push the pass for now. Declarations are collected
        // and textures created in set_render_size() by iterating all passes.
        self.passes.push(pass);
    }

    /// Returns a mutable reference to the first pass of type `T`, if present.
    ///
    /// Uses `RenderPass::as_any_mut()` for downcasting. Only passes that return
    /// `Some` from `as_any_mut()` can be found; others are skipped safely.
    pub fn find_pass_mut<T: RenderPass + 'static>(&mut self) -> Option<&mut T> {
        self.passes.iter_mut().find_map(|p| {
            p.as_any_mut()?.downcast_mut::<T>()
        })
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
        let frame_resources = libhelio::FrameResources::empty();
        self.execute_with_frame_resources(scene, target, depth, &frame_resources)
    }

    pub fn execute_with_frame_resources(
        &mut self,
        scene: &GpuScene,
        target: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        frame_resources: &libhelio::FrameResources<'_>,
    ) -> Result<()> {
        let mut encoder = scene
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Graph"),
            });
        for pass_index in 0..self.passes.len() {
            let (published_passes, pending_passes) = self.passes.split_at_mut(pass_index);
            let (pass, _) = pending_passes
                .split_first_mut()
                .expect("pass_index within bounds");
            let _scope = self.profiler.scope(pass.name());

            // Populate frame resources with transient textures + pass-published resources
            let mut visible_frame_resources = *frame_resources;

            // Map well-known transient resource names to FrameResources fields
            if let Some(tex) = self.transient_textures.get("pre_aa") {
                visible_frame_resources.pre_aa = Some(&tex.view);
            }
            if let Some(tex) = self.transient_textures.get("sky_lut") {
                visible_frame_resources.sky_lut = Some(&tex.view);
            }
            if let Some(tex) = self.transient_textures.get("ssao") {
                visible_frame_resources.ssao = Some(&tex.view);
            }

            // Add pass-published resources (e.g., GBufferPass publishes gbuffer views)
            for published_pass in published_passes {
                published_pass.publish(&mut visible_frame_resources);
            }

            let prepare_ctx = PrepareContext {
                device: &scene.device,
                queue: &scene.queue,
                frame: scene.frame_count,
                scene,
                frame_resources: &visible_frame_resources,
                resize: false,
                width: scene.width,
                height: scene.height,
            };
            pass.prepare(&prepare_ctx)?;

            let mut ctx = PassContext {
                encoder: &mut encoder,
                target,
                depth,
                scene: scene.resources(),
                profiler: &mut self.profiler,
                frame_num: scene.frame_count,
                width: scene.width,
                height: scene.height,
                device: &scene.device,
                frame: &visible_frame_resources,
            };

            pass.execute(&mut ctx)?;
        }

        scene.queue.submit([encoder.finish()]);
        crate::upload::finish_frame();
        Ok(())
    }

    /// Recreates all transient textures based on pass resource declarations.
    ///
    /// Called by `set_render_size()` when size changes or after adding all passes.
    /// Collects write declarations from all passes and creates GPU textures.
    ///
    /// # Performance
    ///
    /// - **O(passes × declarations)**: Iterates all passes, collects all writes
    /// - Call only on size change, not every frame
    /// - Textures owned by graph, views borrowed zero-copy during execute()
    fn recreate_transient_textures(&mut self) {
        // Clear existing textures
        self.transient_textures.clear();

        // Collect all write declarations from all passes
        for pass in &self.passes {
            let mut builder = ResourceBuilder::new();
            pass.declare_resources(&mut builder);

            for decl in builder.declarations() {
                if decl.access == ResourceAccess::Write {
                    // Only create textures for writes (reads are inputs from other passes)
                    if self.transient_textures.contains_key(decl.name) {
                        // Texture already created by an earlier pass
                        continue;
                    }

                    let format = decl.format.expect("Write declaration must specify format");
                    let size = decl.size.expect("Write declaration must specify size");

                    let (width, height) = match size {
                        super::resource::ResourceSize::MatchSurface => (self.width, self.height),
                        super::resource::ResourceSize::Absolute { width, height } => {
                            (width, height)
                        }
                        super::resource::ResourceSize::Scaled { divisor } => {
                            (self.width / divisor, self.height / divisor)
                        }
                    };

                    // Create the transient texture
                    let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some(decl.name),
                        size: wgpu::Extent3d {
                            width: width.max(1),
                            height: height.max(1),
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: format.to_wgpu(),
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    });

                    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

                    self.transient_textures.insert(
                        decl.name,
                        TransientTexture {
                            texture,
                            view,
                            name: decl.name,
                        },
                    );
                }
            }
        }
    }
}

