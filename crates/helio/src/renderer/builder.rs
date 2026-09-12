//! Builder for [`Renderer`] that eliminates boilerplate by creating internal
//! buffers, the scene, and debug state automatically.

use std::sync::{Arc, Mutex};

use helio_core::RenderGraph;
use pulsar_scenedb::gpu::GpuMirrorHandle;

use crate::scene::Scene;

use super::config::RendererConfig;
use super::debug::DebugDrawState;
use super::renderer_impl::Renderer;

/// Closure that builds a [`RenderGraph`] at init time.
///
/// The arguments match the public `helio_default_graphs::build_*` functions,
/// so any of them can be called from inside the closure.
///
/// # Arguments
///
/// * `device`   – the WGPU device
/// * `queue`    – the WGPU queue
/// * `scene`    – the newly-created scene (ready for population)
/// * `config`   – the [`RendererConfig`]
/// * `debug_state` – shared debug-draw state
/// * `debug_camera_buffer` – camera-uniform buffer (64‑byte, `UNIFORM | COPY_DST`)
/// * `cull_stats_buffer`   – cull‑statistics buffer (`STORAGE | COPY_DST`)
pub type GraphBuilderFn = Box<
    dyn FnOnce(
        &Arc<wgpu::Device>,
        &Arc<wgpu::Queue>,
        &Scene,
        RendererConfig,
        Arc<Mutex<DebugDrawState>>,
        &wgpu::Buffer,
        &wgpu::Buffer,
    ) -> RenderGraph,
>;

/// Shared construction inputs for passes and graph composition. New graph
/// builders can accept this context instead of repeating the seven common
/// arguments used by the legacy `GraphBuilderFn` ABI.
pub struct PassBuildContext<'a> {
    pub device: &'a Arc<wgpu::Device>,
    pub queue: &'a Arc<wgpu::Queue>,
    pub scene: &'a Scene,
    pub config: RendererConfig,
    pub debug_state: Arc<Mutex<DebugDrawState>>,
    pub camera_buffer: &'a wgpu::Buffer,
    pub cull_stats_buffer: &'a wgpu::Buffer,
    /// Whether the renderer owns the device and graph allocations.
    pub owns_device: bool,
    /// Frontend-owned authoritative SceneDB handle. It is guaranteed present
    /// when this context is handed to graph composition.
    pub scene_db: SceneDbHandle,
}

/// Context-based graph builder. The higher-ranked lifetime keeps the context
/// borrowed for construction only; no renderer-owned GPU reference escapes.
pub type PassGraphBuilderFn = Box<dyn for<'a> FnOnce(PassBuildContext<'a>) -> RenderGraph>;

/// Shared handle to the application's authoritative SceneDB.
/// Read-only GPU projection of the frontend-owned SceneDB.
///
/// This is deliberately not a handle to `SceneDb`/`World`: the frontend owns
/// and mutates those values at its frame boundary.  Helio receives only the
/// cloneable GPU projection established during initialization, so render and
/// resize paths never lock or alias the authoritative CPU database.
pub type SceneDbHandle = GpuMirrorHandle;

/// Builder for [`Renderer`] that eliminates the repetitive boilerplate of
/// creating internal buffers, the scene, and debug state.
///
/// You **must** provide a [`RenderGraph`] via one of:
///
/// * [`with_pass_build_context`](Self::with_pass_build_context) – a closure
///   that builds the graph from one shared construction context.
/// * [`with_graph`](Self::with_graph) – the legacy seven-argument ABI, retained
///   for custom graphs and existing applications.
///
/// # Example – default deferred graph
///
/// ```rust,ignore
/// let config = RendererConfig::new(1920, 1080, surface_format);
/// // `scene_db` is a `GpuMirrorHandle` obtained by attaching a GPU mirror to
/// // the frontend's SceneDB `World` *before* the renderer is constructed.
/// let mut renderer = RendererBuilder::new(config, scene_db)
///     .with_external_device()
///     .with_editor_mode(true)
///     .with_ambient([0.0, 0.0, 0.0], 0.0)
///     .with_pass_build_context(Box::new(
///         helio_default_graphs::build_default_graph_external_with_context,
///     ))
///     .build(device, queue, 1920, 1080, surface_format);
/// ```
///
/// # Example – custom graph
///
/// ```rust,ignore
/// let renderer = RendererBuilder::new(config, scene_db)
///     .with_pass_build_context(Box::new(
///         helio_default_graphs::build_fxaa_graph_with_context,
///     ))
///     .build(device, queue, 1920, 1080, fmt);
/// ```
pub struct RendererBuilder {
    config: RendererConfig,
    graph_fn: Option<GraphBuilderFn>,
    pass_graph_fn: Option<PassGraphBuilderFn>,
    editor_mode: bool,
    ambient_color: [f32; 3],
    ambient_intensity: f32,
    clear_color: [f32; 4],
    owns_device: bool,
    scene_db: SceneDbHandle,
}

impl RendererBuilder {
    /// Start building a [`Renderer`] with the given configuration and the
    /// frontend's authoritative SceneDB GPU projection.
    ///
    /// `scene_db` is not optional: SceneDB is the sole scene authority, so
    /// there is no `RendererBuilder` state — and no `Renderer` — that can
    /// exist without one. The frontend must attach its GPU mirror (see
    /// `engine_backend::scene::helio_bridge::ensure_gpu_mirror` or the
    /// equivalent seam) *before* calling this, not after `build()`.
    pub fn new(config: RendererConfig, scene_db: SceneDbHandle) -> Self {
        Self {
            config,
            graph_fn: None,
            pass_graph_fn: None,
            editor_mode: false,
            ambient_color: [0.05, 0.05, 0.08],
            ambient_intensity: 1.0,
            clear_color: [0.02, 0.02, 0.03, 1.0],
            owns_device: true,
            scene_db,
        }
    }

    /// Provide a closure that builds the [`RenderGraph`].
    ///
    /// **Required** – `build()` will panic if no graph builder has been set.
    pub fn with_graph(mut self, f: GraphBuilderFn) -> Self {
        self.graph_fn = Some(f);
        self.pass_graph_fn = None;
        self
    }

    /// Provide a graph builder using a single shared construction context.
    pub fn with_pass_build_context(mut self, f: PassGraphBuilderFn) -> Self {
        self.pass_graph_fn = Some(f);
        self.graph_fn = None;
        self
    }

    /// Mark the renderer as sharing a device created externally.
    /// Equivalent to the old `new_with_external_device()`.
    pub fn with_external_device(mut self) -> Self {
        self.owns_device = false;
        self
    }

    /// Enable editor mode (shows the `EDITOR` group, enables gizmo drawing).
    pub fn with_editor_mode(mut self, enabled: bool) -> Self {
        self.editor_mode = enabled;
        self
    }

    /// Override the ambient light colour and intensity.
    pub fn with_ambient(mut self, color: [f32; 3], intensity: f32) -> Self {
        self.ambient_color = color;
        self.ambient_intensity = intensity;
        self
    }

    /// Override the clear colour applied each frame.
    pub fn with_clear_color(mut self, color: [f32; 4]) -> Self {
        self.clear_color = color;
        self
    }

    /// Configure the virtual-texture tile size for frontend projections.
    ///
    /// Tile residency is owned by the SceneDB/virtual-geometry pass. The
    /// renderer keeps this builder entry as a harmless construction-time
    /// input so callers do not need to reach into a scene container.
    pub fn with_vt_tile_size(self, _tile_size: u32) -> Self {
        self
    }

    /// Configure the texture residency budget owned by the frontend.
    pub fn with_texture_streaming(self, _budget_mb: u32) -> Self {
        self
    }

    /// Build the [`Renderer`].
    ///
    /// Creates the scene, debug buffers, and debug state internally, then calls
    /// the closure registered via [`with_graph`](Self::with_graph) to obtain the
    /// render graph, and finally constructs the renderer.
    ///
    /// # Panics
    ///
    /// Panics if `with_graph` was not called before `build`.
    pub fn build(
        self,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        width: u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
    ) -> Renderer {
        let width = width.max(1);
        let height = height.max(1);

        let debug_state = Arc::new(Mutex::new(DebugDrawState::default()));
        let debug_camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Camera Buffer"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cull_stats_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cull Stats Buffer"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scene = Scene::new(device.clone(), queue.clone());

        let config = self.config;
        let graph = if let Some(pass_graph_fn) = self.pass_graph_fn {
            pass_graph_fn(PassBuildContext {
                device: &device,
                queue: &queue,
                scene: &scene,
                config,
                debug_state: debug_state.clone(),
                camera_buffer: &debug_camera_buffer,
                cull_stats_buffer: &cull_stats_buffer,
                owns_device: self.owns_device,
                scene_db: self.scene_db.clone(),
            })
        } else {
            let graph_fn = self.graph_fn.expect(
                "RendererBuilder::build: call .with_graph() or .with_pass_build_context() first",
            );
            graph_fn(
                &device,
                &queue,
                &scene,
                config,
                debug_state.clone(),
                &debug_camera_buffer,
                &cull_stats_buffer,
            )
        };

        let scene_db = self.scene_db;
        let mut renderer = Renderer::construct(
            device,
            queue,
            surface_format,
            width,
            height,
            config.render_scale,
            config,
            scene,
            graph,
            debug_state,
            debug_camera_buffer,
            cull_stats_buffer,
            scene_db,
        );

        renderer.owns_device = self.owns_device;
        renderer.set_editor_mode(self.editor_mode);
        renderer.set_ambient(self.ambient_color, self.ambient_intensity);
        renderer.set_clear_color(self.clear_color);

        renderer
    }
}
