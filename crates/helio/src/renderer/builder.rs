//! Builder for [`Renderer`] that eliminates boilerplate by creating internal
//! buffers, the scene, and debug state automatically.

use std::sync::{Arc, Mutex};

use helio_core::RenderGraph;

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

/// Builder for [`Renderer`] that eliminates the repetitive boilerplate of
/// creating internal buffers, the scene, and debug state.
///
/// You **must** provide a [`RenderGraph`] via one of:
///
/// * [`with_graph`](Self::with_graph) – a closure that builds the graph (supports
///   any `helio_default_graphs::build_*` function, custom graphs, or both).
///
/// # Example – default deferred graph
///
/// ```rust,ignore
/// let config = RendererConfig::new(1920, 1080, surface_format);
/// let mut renderer = RendererBuilder::new(config)
///     .with_external_device()
///     .with_editor_mode(true)
///     .with_ambient([0.0, 0.0, 0.0], 0.0)
///     .with_graph(Box::new(|d, q, s, cfg, ds, cb, csb| {
///         helio_default_graphs::build_default_graph_external(
///             d, q, s, cfg, ds, cb, csb, None,
///         )
///     }))
///     .build(device, queue, 1920, 1080, surface_format);
/// ```
///
/// # Example – custom graph
///
/// ```rust,ignore
/// let renderer = RendererBuilder::new(config)
///     .with_graph(Box::new(|d, q, s, cfg, ds, cb, csb| {
///         helio_default_graphs::build_fxaa_graph(d, q, s, cfg, ds, cb, csb, None)
///     }))
///     .build(device, queue, 1920, 1080, fmt);
/// ```
pub struct RendererBuilder {
    config: RendererConfig,
    graph_fn: Option<GraphBuilderFn>,
    editor_mode: bool,
    ambient_color: [f32; 3],
    ambient_intensity: f32,
    clear_color: [f32; 4],
    owns_device: bool,
}

impl RendererBuilder {
    /// Start building a [`Renderer`] with the given configuration.
    pub fn new(config: RendererConfig) -> Self {
        Self {
            config,
            graph_fn: None,
            editor_mode: false,
            ambient_color: [0.05, 0.05, 0.08],
            ambient_intensity: 1.0,
            clear_color: [0.02, 0.02, 0.03, 1.0],
            owns_device: true,
        }
    }

    /// Provide a closure that builds the [`RenderGraph`].
    ///
    /// **Required** – `build()` will panic if no graph builder has been set.
    pub fn with_graph(mut self, f: GraphBuilderFn) -> Self {
        self.graph_fn = Some(f);
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
        let graph_fn = self
            .graph_fn
            .expect("RendererBuilder::build: call .with_graph() first");
        let graph = graph_fn(
            &device,
            &queue,
            &scene,
            config,
            debug_state.clone(),
            &debug_camera_buffer,
            &cull_stats_buffer,
        );

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
        );

        renderer.owns_device = self.owns_device;
        renderer.set_editor_mode(self.editor_mode);
        renderer.set_ambient(self.ambient_color, self.ambient_intensity);
        renderer.set_clear_color(self.clear_color);

        renderer
    }
}
