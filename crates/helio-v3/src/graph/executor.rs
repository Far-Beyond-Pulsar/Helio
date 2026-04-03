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

use crate::{GpuScene, PassContext, PrepareContext, Profiler, RenderPass, Result};
use std::any::TypeId;
use std::collections::HashMap;

/// Type-erased routing function: wires a graph-managed transient texture view
/// into the appropriate field of `FrameResources` each frame.
///
/// Registered via `RenderGraph::register_transient_route`.
type TransientRoute = Box<
    dyn for<'frame> Fn(
        &'frame wgpu::TextureView,
        &mut libhelio::FrameResources<'frame>,
    ) + Send + Sync,
>;

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

    /// TypeId → pass index map for true O(1) `find_pass` / `find_pass_mut` lookups.
    /// Only the first pass of each type is stored; duplicates are harmlessly ignored.
    pass_index_map: HashMap<TypeId, usize>,

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

    /// Elapsed frame time in seconds, set by `set_delta_time()` before `execute()`.
    ///
    /// Forwarded into `PrepareContext::delta_time` so passes can implement
    /// frame-rate-independent updates without querying a global clock.
    delta_time: f32,

    /// Registered name → FrameResources routing functions for transient textures.
    ///
    /// The three legacy names ("pre_aa", "sky_lut", "ssao") are pre-registered in
    /// `new()`.  Custom passes add their own routes via `register_transient_route()`.
    resource_routes: Vec<(&'static str, TransientRoute)>,
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
        // Pre-register the three built-in transient routes so existing graphs
        // that rely on these names continue to work without any changes.
        let mut resource_routes: Vec<(&'static str, TransientRoute)> = Vec::new();
        resource_routes.push(("pre_aa",  Box::new(|view, fr| fr.pre_aa  = Some(view))));
        resource_routes.push(("sky_lut", Box::new(|view, fr| fr.sky_lut = Some(view))));
        resource_routes.push(("ssao",    Box::new(|view, fr| fr.ssao    = Some(view))));

        Self {
            passes: Vec::new(),
            pass_index_map: HashMap::new(),
            profiler: Profiler::new(device, queue),
            transient_textures: HashMap::new(),
            device: device.clone(),
            width: 0,  // Set via set_render_size() before first execute()
            height: 0, // Set via set_render_size() before first execute()
            delta_time: 0.0,
            resource_routes,
        }
    }

    /// Sets the elapsed frame time (seconds) forwarded to `PrepareContext::delta_time`.
    ///
    /// Call once per frame, **before** `execute()`.  The high-level `Renderer` does
    /// this automatically.  Direct `RenderGraph` users should call this themselves:
    ///
    /// ```rust,ignore
    /// let dt = last_frame.elapsed().as_secs_f32().min(0.1);
    /// last_frame = Instant::now();
    /// graph.set_delta_time(dt);
    /// graph.execute(&scene, &target, &depth)?;
    /// ```
    pub fn set_delta_time(&mut self, dt: f32) {
        self.delta_time = dt;
    }

    /// Registers a routing function that populates a `FrameResources` field from
    /// a graph-managed transient texture each frame.
    ///
    /// Call this in your graph builder after adding a pass that declares a
    /// transient texture write.  Without a route, the texture is created and
    /// resized correctly but never wired into `ctx.frame` for downstream passes.
    ///
    /// The three built-in names (`"pre_aa"`, `"sky_lut"`, `"ssao"`) are already
    /// registered in `RenderGraph::new()` for backward compatibility.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// graph.register_transient_route("ssr_result", |view, fr| {
    ///     fr.ssr_result = Some(view);
    /// });
    /// ```
    pub fn register_transient_route<F>(&mut self, name: &'static str, route: F)
    where
        F: for<'frame> Fn(
                &'frame wgpu::TextureView,
                &mut libhelio::FrameResources<'frame>,
            ) + Send + Sync + 'static,
    {
        self.resource_routes.push((name, Box::new(route)));
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
        for pass in &mut self.passes {
            pass.on_resize(&self.device, width, height);
        }
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
        let type_id = pass.as_any().type_id();
        // Only store the first occurrence so find_pass() returns the first matching pass.
        self.pass_index_map.entry(type_id).or_insert(self.passes.len());
        self.passes.push(pass);
    }

    /// Returns a mutable reference to the first pass of type `T`, if present.
    ///
    /// Uses `pass_index_map` for true O(1) lookup by `TypeId`.
    pub fn find_pass_mut<T: RenderPass + 'static>(&mut self) -> Option<&mut T> {
        let idx = *self.pass_index_map.get(&TypeId::of::<T>())?;
        self.passes[idx].as_any_mut().downcast_mut::<T>()
    }

    /// Returns an immutable reference to the first pass of type `T`, if present.
    ///
    /// Uses `pass_index_map` for true O(1) lookup by `TypeId`.
    pub fn find_pass<T: RenderPass + 'static>(&self) -> Option<&T> {
        let idx = *self.pass_index_map.get(&TypeId::of::<T>())?;
        self.passes[idx].as_any().downcast_ref::<T>()
    }

    /// Returns mutable references to **all** passes of type `T`.
    ///
    /// Unlike `find_pass_mut`, which returns only the first match, this iterates
    /// the full pass list. Use when multiple instances of the same pass type exist
    /// in the graph (e.g., two `DebugDrawPass` nodes).
    pub fn iter_passes_mut<T: RenderPass + 'static>(&mut self) -> impl Iterator<Item = &mut T> {
        self.passes
            .iter_mut()
            .filter_map(|p| p.as_any_mut().downcast_mut::<T>())
    }

    /// Returns a reference to the profiler for reading timing data.
    ///
    /// Use this to access CPU and GPU profiling results after calling `execute()`.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use helio_v3::RenderGraph;
    /// # let mut graph = RenderGraph::new(&device, &queue);
    /// graph.execute(&scene, &target, &depth).unwrap();
    ///
    /// // Print timing results
    /// graph.profiler().print_frame_timings();
    ///
    /// // Or access raw data
    /// for (name, duration) in graph.profiler().get_cpu_timings() {
    ///     println!("{}: {:?}", name, duration);
    /// }
    /// ```
    pub fn profiler(&self) -> &Profiler {
        &self.profiler
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
        // Clear CPU timings from previous frame
        self.profiler.clear_cpu_timings();

        let mut encoder = scene
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Graph"),
            });

        // Apply resource routes once (transient texture pointer → FrameResources field).
        // Routes are stable across passes, so there is no reason to re-apply them inside
        // the per-pass loop (previously O(passes × routes), now O(routes)).
        let mut base_frame_resources = *frame_resources;
        for (name, route) in &self.resource_routes {
            if let Some(tex) = self.transient_textures.get(name) {
                route(&tex.view, &mut base_frame_resources);
            }
        }

        for pass_index in 0..self.passes.len() {
            let (published_passes, pending_passes) = self.passes.split_at_mut(pass_index);
            let (pass, _) = pending_passes
                .split_first_mut()
                .expect("pass_index within bounds");

            // Populate frame resources with transient textures + pass-published resources
            let mut visible_frame_resources = base_frame_resources;

            // Add pass-published resources (e.g., GBufferPass publishes gbuffer views)
            for published_pass in published_passes {
                published_pass.publish(&mut visible_frame_resources);
            }

            // CPU profiling scope for prepare()
            {
                let _scope = self.profiler.scope(pass.name());
                let prepare_ctx = PrepareContext {
                    device: &scene.device,
                    queue: &scene.queue,
                    frame_num: scene.frame_count,
                    scene,
                    frame_resources: &visible_frame_resources,
                    resize: false,
                    width: scene.width,
                    height: scene.height,
                    delta_time: self.delta_time,
                };
                pass.prepare(&prepare_ctx)?;
            } // Scope ends here, profiler is released

            // GPU profiling: write start timestamp
            let pass_name = pass.name();
            self.profiler.begin_gpu_pass(&mut encoder, pass_name);

            // execute() with GPU profiling (handled via PassContext)
            {
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
                    resources: &visible_frame_resources,
                };

                pass.execute(&mut ctx)?;
            }

            // GPU profiling: write end timestamp
            self.profiler.end_gpu_pass(&mut encoder, pass_name);
        }

        // Resolve GPU timestamp queries
        self.profiler.resolve_gpu_queries(&mut encoder);

        scene.queue.submit([encoder.finish()]);
        crate::upload::finish_frame();

        // Read back GPU timestamps (blocks until GPU completes)
        self.profiler.read_gpu_timestamps_blocking(&scene.device);

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
    }
}

