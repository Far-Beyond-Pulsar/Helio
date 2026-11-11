//! Builder for [`Renderer`] that eliminates boilerplate by creating internal
//! GPU buffers and debug state automatically.

use std::sync::{Arc, Mutex};

use helio_core::RenderGraph;
use pulsar_scenedb::gpu::GpuMirrorHandle;

use super::config::RendererConfig;
use super::debug::DebugDrawState;
use super::pipeline::IntoGraph;
use super::pipeline_status::{NamedPipeline, UNNAMED_PIPELINE};
use super::renderer_impl::Renderer;

/// Shared construction inputs for passes and graph composition. New graph
pub struct PassBuildContext<'a> {
    pub device: &'a Arc<wgpu::Device>,
    pub queue: &'a Arc<wgpu::Queue>,
    pub config: RendererConfig,
    pub debug_state: Arc<Mutex<DebugDrawState>>,
    pub camera_buffer: &'a wgpu::Buffer,
    pub debug_camera_buffer: &'a wgpu::Buffer,
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
/// A pipeline is optional: a renderer built without one draws the status screen
/// ("No render pipeline assigned") until the host assigns one with
/// [`Renderer::set_pipeline`]. Give it one with:
///
/// * [`with_graph`](Self::with_graph) / [`with_pipeline`](Self::with_pipeline) –
///   anything that implements [`IntoGraph`];
/// * [`with_pass_build_context`](Self::with_pass_build_context) – a closure
///   that builds the graph from one shared construction context.
///
/// A pipeline that does not build does not fail `build`: the renderer comes
/// back with [`PipelineStatus::Invalid`](super::PipelineStatus) and draws why.
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
///     .with_feature(helio::AmbientLight { color: [0.0; 3], intensity: 0.0 })
///     .with_pass_build_context(Box::new(
///         helio_presets::deferred,
///     ))
///     .build(device, queue, 1920, 1080, surface_format);
/// ```
///
/// # Example – custom graph
///
/// ```rust,ignore
/// let renderer = RendererBuilder::new(config, scene_db)
///     .with_pass_build_context(Box::new(
///         helio_presets::deferred_fxaa,
///     ))
///     .build(device, queue, 1920, 1080, fmt);
/// ```
pub struct RendererBuilder {
    config: RendererConfig,
    pipeline: Option<NamedPipeline>,
    status_hint: Option<String>,
    editor_mode: bool,
    scene_derivations: Vec<Box<dyn helio_core::SceneDerivation>>,
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
            pipeline: None,
            status_hint: None,
            editor_mode: false,
            scene_derivations: Vec::new(),
            clear_color: [0.02, 0.02, 0.03, 1.0],
            owns_device: true,
            scene_db,
        }
    }

    /// Provide a graph builder using a single shared construction context.
    pub fn with_pass_build_context(self, f: PassGraphBuilderFn) -> Self {
        self.with_pipeline(UNNAMED_PIPELINE, f)
    }

    /// Start with `pipeline`, known to the status screen and the host as `id`.
    pub fn with_pipeline(
        mut self,
        id: impl Into<String>,
        pipeline: impl IntoGraph + 'static,
    ) -> Self {
        self.pipeline = Some(NamedPipeline::new(id, pipeline));
        self
    }

    /// What the status screen says about assigning a pipeline
    /// ([`Renderer::set_status_hint`]).
    pub fn with_status_hint(mut self, hint: impl Into<String>) -> Self {
        self.status_hint = Some(hint.into());
        self
    }

    /// Replace the renderer configuration before graph construction. This is
    /// used by [`HelioGraphicsContext`](crate::HelioGraphicsContext) to apply
    /// display-derived output settings without exposing XR setup to apps.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn with_renderer_config(mut self, config: RendererConfig) -> Self {
        self.config = config;
        self
    }

    /// Build this application renderer on Helio's automatically selected
    /// desktop or XR device. Use this before any app-owned GPU resources are
    /// created so the SceneDB mirror and renderer share the context device.
    /// Applications that need controller actions can call
    /// [`HelioGraphicsContext::build_renderer`] directly and retain its input
    /// handle; render-only apps can use this compact path.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn build_with_graphics_context(
        self,
        graphics: &mut crate::HelioGraphicsContext<'_>,
    ) -> helio_core::Result<Renderer> {
        let config = self.config.clone();
        let (renderer, _input) = graphics.build_renderer(self, config)?;
        Ok(renderer)
    }

    /// Build the render graph from anything that implements [`IntoGraph`]: a
    /// built-in pipeline, a render-graph asset or a plain function.
    pub fn with_graph(self, pipeline: impl IntoGraph + 'static) -> Self {
        self.with_boxed_graph(Box::new(pipeline))
    }

    /// [`with_graph`](Self::with_graph) for an already boxed pipeline.
    pub fn with_boxed_graph(mut self, pipeline: Box<dyn IntoGraph>) -> Self {
        self.pipeline = Some(NamedPipeline::boxed(UNNAMED_PIPELINE, pipeline));
        self
    }

    /// Mark the renderer as sharing a device created externally.
    /// Equivalent to the old `new_with_external_device()`.
    pub fn with_external_device(mut self) -> Self {
        self.owns_device = false;
        self
    }

    /// Enable editor mode (shows the `EDITOR` group, enables gizmo drawing).
    /// Add a scene derivation, run each frame (in the order added) between
    /// the SceneDB snapshot and the graph; see `helio_core::scene_derivation`.
    pub fn with_scene_derivation(
        mut self,
        derivation: Box<dyn helio_core::SceneDerivation>,
    ) -> Self {
        self.scene_derivations.push(derivation);
        self
    }

    pub fn with_editor_mode(mut self, enabled: bool) -> Self {
        self.editor_mode = enabled;
        self
    }

    /// Sets a feature's settings in the renderer config (see
    /// [`RendererConfig::with_feature`]).
    pub fn with_feature<T: std::any::Any + Send + Sync>(mut self, setting: T) -> Self {
        self.config.features.insert(setting);
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
    /// Creates the debug buffers and debug state internally, constructs the
    /// renderer and builds the pipeline given to it, if any. With none, or one
    /// that does not build, the renderer shows the status screen
    /// ([`Renderer::pipeline_status`] says which).
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
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scene Camera Buffer"),
            size: (std::mem::size_of::<helio_core::GpuCameraUniforms>() * 2) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
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

        let config = self.config;
        let mut renderer = Renderer::construct(
            device,
            queue,
            surface_format,
            width,
            height,
            config.render_scale,
            config,
            debug_state,
            camera_buffer,
            debug_camera_buffer,
            cull_stats_buffer,
            self.scene_db,
        );

        renderer.owns_device = self.owns_device;
        renderer.set_editor_mode(self.editor_mode);
        // The host's derivations first; the ones the pipeline's features bring
        // join after them (below), and may read what the host's published.
        renderer.scene_derivations = self.scene_derivations;
        renderer.set_clear_color(self.clear_color);
        if let Some(hint) = self.status_hint {
            renderer.status_hint = hint;
        }
        // The pipeline last: its features' derivations join the host's, and it
        // builds with the renderer's own buffers, device ownership and config.
        renderer.set_pipeline(self.pipeline);
        renderer.apply_pending_pipeline();

        renderer
    }
}
