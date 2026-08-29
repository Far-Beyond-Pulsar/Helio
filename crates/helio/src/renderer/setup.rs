use std::sync::{Arc, Mutex};

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use crate::radiant::RadiantTemplateRegistry;
use crate::scene::Scene;
use helio_core::RenderGraph;

use super::config::RendererConfig;
use super::debug::DebugDrawState;
use super::renderer_impl::{CullStatsReadbackState, DebugCameraUniform, GraphRebuilder, Renderer};

impl Renderer {
    pub(crate) fn create_depth_resources(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Helio Depth Texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    /// Create a two-layer array depth texture for the OpenXR multiview render
    /// path. Both eye layers are cleared/written in a single pass via
    /// `multiview_mask = 0b11`.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn create_xr_depth_resources(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Helio XR Depth Texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 2,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        // Array view: the depth-stencil attachment for the multiview render
        // passes (multiview_mask = 0b11 writes both eye layers).
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Helio XR Depth View"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            base_array_layer: 0,
            array_layer_count: Some(2),
            ..Default::default()
        });
        // Layer-0 D2 view: for passes that *sample* the rendered depth as a
        // plain `texture_depth_2d` (HiZ, lens flare, ...). A D2Array view
        // cannot be bound to a D2 bind-group entry.
        let layer0_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Helio XR Depth Layer0 View"),
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_array_layer: 0,
            array_layer_count: Some(1),
            ..Default::default()
        });
        (texture, view, layer0_view)
    }

    /// Construct a [`Renderer`] with all dependencies provided explicitly.
    ///
    /// Prefer [`RendererBuilder`](super::builder::RendererBuilder) for new code — it
    /// creates the scene, debug state, and internal buffers automatically.
    #[allow(deprecated)]
    pub(crate) fn construct(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        render_scale: f32,
        config: RendererConfig,
        mut scene: Scene,
        mut graph: RenderGraph,
        debug_state: Arc<Mutex<DebugDrawState>>,
        debug_camera_buffer: wgpu::Buffer,
        cull_stats_buffer: wgpu::Buffer,
    ) -> Self {
        scene.set_shadow_face_capacity(config.shadow_face_capacity);
        scene.set_render_size(width, height);

        assert!(
            device
                .features()
                .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE),
            "Helio requires INDIRECT_FIRST_INSTANCE because GPU-driven object and meshlet \
             draws use non-zero indirect first_instance values; create the device with \
             helio::required_wgpu_features(adapter.features())"
        );

        let internal_w = config.internal_width();
        let internal_h = config.internal_height();

        let (depth_texture, depth_view) =
            Self::create_depth_resources(&device, internal_w, internal_h);

        let (full_res_depth_texture, full_res_depth_view) = if render_scale < 1.0 {
            let (t, v) = Self::create_depth_resources(&device, width, height);
            (Some(t), Some(v))
        } else {
            (None, None)
        };

        // In XR (multiview) mode the depth-stencil attachment of the render
        // passes must be a 2-layer array view (the executor forces
        // `multiview_mask = 0b11` on every pass). The OpenXR swapchain image is
        // `width × height × 2`, so the array depth is allocated at the internal
        // resolution. It is kept separate from the desktop `depth_texture`;
        // passes that sample scene depth receive a plain D2 view of layer 0.
        #[cfg(not(target_arch = "wasm32"))]
        let (xr_depth_texture, xr_depth_view, xr_depth_view_layer0) = if config.enable_xr {
            let (t, v, l0) = Self::create_xr_depth_resources(&device, internal_w, internal_h);
            (Some(t), Some(v), Some(l0))
        } else {
            (None, None, None)
        };
        #[cfg(target_arch = "wasm32")]
        let (xr_depth_texture, xr_depth_view, xr_depth_view_layer0) = (None, None, None);

        let water_volumes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Water Volumes Buffer"),
            size: 256 * 256,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let water_hitboxes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Water Hitboxes Buffer"),
            size: 256 * 80,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Fixed 256-interactor ceiling, matching the water-hitbox buffer above. The
        // interaction field is 64 m across; more than a couple of hundred bodies inside it
        // at once is a gameplay problem, not a rendering one, and a fixed size keeps this
        // off the per-frame allocation path entirely.
        let foliage_interactors_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Foliage Interactors Buffer"),
            size: 256 * std::mem::size_of::<crate::scene::GpuFoliageInteractor>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pp_volumes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PostProcess Volumes Buffer"),
            size: 256 * std::mem::size_of::<libhelio::GpuPostProcessVolume>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let postprocess_buf_size = std::mem::size_of::<libhelio::GpuPostProcessUniforms>() as u64;
        let postprocess_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PostProcess Uniforms Buffer"),
            size: postprocess_buf_size,
            // COPY_SRC: VolumetricFogPass copies the fog block out of this buffer
            // rather than mirroring the whole 368-byte struct in its shader.
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cull_stats_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CullStats Staging"),
            size: 32,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Camera jitter is only valid when a temporal pass reconstructs it.
        // Applying it to FXAA/non-temporal graphs shifts the final image every
        // frame and presents as whole-scene shimmer.
        let enable_jitter = graph.requires_camera_jitter();

        let graph_rebuilder = graph.take_graph_data::<GraphRebuilder>();
        // Captured before `scene` is moved into `Self`.
        let scene_has_sky = scene.sky_context().has_sky;

        Self {
            device,
            queue,
            graph,
            scene,
            depth_texture,
            depth_view,
            output_width: width,
            output_height: height,
            render_scale,
            full_res_depth_texture,
            full_res_depth_view,
            surface_format,
            debug_camera_buffer,
            ambient_color: [0.05, 0.05, 0.08],
            ambient_intensity: 1.0,
            clear_color: [0.02, 0.02, 0.03, 1.0],
            gi_config: config.gi_config,
            shadow_quality: config.shadow_quality,
            shadow_atlas_size: config.shadow_atlas_size,
            shadow_face_capacity: config.shadow_face_capacity,
            enable_ssr: config.enable_ssr,
            enable_foliage: config.enable_foliage,
            foliage_blades_per_m2: config.foliage_blades_per_m2,
            enable_portals: config.enable_portals,
            enable_planar_reflections: config.enable_planar_reflections,
            enable_environment_reflections: config.enable_environment_reflections,
            debug_mode: config.debug_mode,
            editor_mode: false,
            debug_state,
            billboard_instances: Vec::new(),
            billboard_scratch: Vec::new(),
            billboard_dirty: true,
            billboard_cached_light_count: usize::MAX,
            billboard_cached_light_gen: u64::MAX,
            billboard_cached_editor_hidden: false,
            billboard_cached_corona_gen: u64::MAX,
            billboard_generation: 0,
            corona_emitters: Vec::new(),
            corona_emitter_generation: 0,
            water_volumes_buffer,
            water_hitboxes_buffer,
            foliage_interactors_buffer,
            pp_volumes_buffer,
            postprocess_buffer,
            last_render_time: Instant::now(),
            delta_time: 0.0,
            color_grading_lut_view: None,
            ies_texture_view: None,
            cull_stats_staging,
            cull_stats_readback_state: CullStatsReadbackState::Idle,
            cull_stats: [0; 8],
            graph_time_ms: 0.0,
            frame_times: vec![0.0; 200],
            frame_times_cursor: 0,
            enable_jitter,
            #[cfg(feature = "bake")]
            bake_pending: None,
            #[cfg(feature = "bake")]
            baked_data: None,
            clear_target_next_frame: true,
            graph_has_sky: scene_has_sky,
            xr_stage_transform: glam::Mat4::IDENTITY,
            owns_device: true,
            pending_resize: None,
            gizmo_camera: None,
            gizmo_viewport_height: 0.0,
            cull_stats_buffer,
            graph_rebuilder,
            tsr_quality: config.tsr_quality,
            template_registry: std::sync::Arc::new(std::sync::RwLock::new(
                RadiantTemplateRegistry::new(),
            )),
            transparent_template_registry: std::sync::Arc::new(std::sync::RwLock::new(
                RadiantTemplateRegistry::new(),
            )),
            render_mode: config.render_mode,
            enable_xr: config.enable_xr,
            #[cfg(not(target_arch = "wasm32"))]
            xr_instance: None,
            #[cfg(not(target_arch = "wasm32"))]
            xr: None,
            #[cfg(not(target_arch = "wasm32"))]
            xr_swapchain: None,
            #[cfg(not(target_arch = "wasm32"))]
            xr_depth_texture,
            #[cfg(not(target_arch = "wasm32"))]
            xr_depth_view,
            #[cfg(not(target_arch = "wasm32"))]
            xr_depth_view_layer0,
            #[cfg(not(target_arch = "wasm32"))]
            xr_idle_skips: 0,
            #[cfg(not(target_arch = "wasm32"))]
            xr_camera: None,
            #[cfg(not(target_arch = "wasm32"))]
            xr_mirror_pipeline: None,
            #[cfg(not(target_arch = "wasm32"))]
            xr_mirror_bgl: None,
            #[cfg(not(target_arch = "wasm32"))]
            xr_mirror_sampler: None,
            #[cfg(not(target_arch = "wasm32"))]
            xr_mirror_bind_group: None,
            #[cfg(not(target_arch = "wasm32"))]
            xr_mirror_format: None,
        }
    }

    /// Create a [`Renderer`] that owns its device and queue.
    ///
    /// This is the original full-signature constructor kept for backward
    /// compatibility.  Prefer [`RendererBuilder`](super::builder::RendererBuilder)
    /// for new code — it creates the scene, debug state, and internal buffers
    /// automatically.
    #[deprecated(since = "0.20.0", note = "use RendererBuilder instead")]
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        render_scale: f32,
        config: RendererConfig,
        scene: Scene,
        graph: RenderGraph,
        debug_state: Arc<Mutex<DebugDrawState>>,
        debug_camera_buffer: wgpu::Buffer,
        cull_stats_buffer: wgpu::Buffer,
    ) -> Self {
        Self::construct(
            device,
            queue,
            surface_format,
            width,
            height,
            render_scale,
            config,
            scene,
            graph,
            debug_state,
            debug_camera_buffer,
            cull_stats_buffer,
        )
    }

    /// Create a [`Renderer`] that shares a device/queue owned externally.
    ///
    /// Equivalent to [`new()`] with `owns_device = false`.
    #[deprecated(since = "0.20.0", note = "use RendererBuilder instead")]
    pub fn new_with_external_device(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        render_scale: f32,
        config: RendererConfig,
        scene: Scene,
        graph: RenderGraph,
        debug_state: Arc<Mutex<DebugDrawState>>,
        debug_camera_buffer: wgpu::Buffer,
        cull_stats_buffer: wgpu::Buffer,
    ) -> Self {
        let mut renderer = Self::construct(
            device,
            queue,
            surface_format,
            width,
            height,
            render_scale,
            config,
            scene,
            graph,
            debug_state,
            debug_camera_buffer,
            cull_stats_buffer,
        );
        renderer.owns_device = false;
        renderer
    }
}
