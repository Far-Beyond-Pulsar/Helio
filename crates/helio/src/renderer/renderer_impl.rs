use std::sync::{Arc, Mutex};

#[cfg(target_arch = "wasm32")]
use web_time::Instant;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use arrayvec::ArrayVec;
use helio_pass_debug::{DebugVertex};
use helio_pass_debug_overlay::DebugOverlayState;
use helio_pass_deferred_light::DeferredLightPass;
use helio_pass_perf_overlay::{PerfOverlayMode, PerfOverlayPass};
use helio_pass_virtual_geometry::VirtualGeometryPass;
use helio_v3::{RenderGraph, RenderPass, Result as HelioResult};
use helio_pass_debug::DebugCameraUniform;
const MAX_TEXTURES: usize = crate::material::MAX_TEXTURES;

use crate::groups::GroupId;
use crate::mesh::MeshBuffers;
use crate::scene::{Camera, Scene};

use super::config::{GiConfig, RendererConfig};
use super::debug::{DebugDrawPass, DebugDrawState};
use super::graph::{build_default_graph, build_simple_graph, create_depth_resources};

type CustomGraphBuilder = Arc<dyn Fn(&Arc<wgpu::Device>, &Arc<wgpu::Queue>, &Scene, RendererConfig, Arc<Mutex<DebugDrawState>>, &wgpu::Buffer, &wgpu::Buffer, Option<&Arc<Mutex<DebugOverlayState>>>) -> RenderGraph + Send + Sync>;

const HALTON_JITTER: [[f32; 2]; 16] = [
    [0.5,     0.333333],
    [0.25,    0.666667],
    [0.75,    0.111111],
    [0.125,   0.444444],
    [0.625,   0.777778],
    [0.375,   0.222222],
    [0.875,   0.555556],
    [0.0625,  0.888889],
    [0.5625,  0.037037],
    [0.3125,  0.37037 ],
    [0.8125,  0.703704],
    [0.1875,  0.148148],
    [0.6875,  0.481481],
    [0.4375,  0.814815],
    [0.9375,  0.259259],
    [0.03125, 0.592593],
];

pub struct Renderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    graph: RenderGraph,
    graph_kind: GraphKind,
    scene: Scene,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    output_width: u32,
    output_height: u32,
    render_scale: f32,
    full_res_depth_texture: Option<wgpu::Texture>,
    full_res_depth_view: Option<wgpu::TextureView>,
    surface_format: wgpu::TextureFormat,
    debug_camera_buffer: wgpu::Buffer,
    cull_stats_buffer: wgpu::Buffer,
    ambient_color: [f32; 3],
    ambient_intensity: f32,
    clear_color: [f32; 4],
    gi_config: GiConfig,
    shadow_quality: libhelio::ShadowQuality,
    debug_mode: u32,
    perf_overlay_mode: PerfOverlayMode,
    debug_depth_test: bool,
    editor_mode: bool,
    custom_graph_builder: Option<CustomGraphBuilder>,
    custom_graph_config: Option<RendererConfig>,
    debug_state: Arc<Mutex<DebugDrawState>>,
    debug_overlay_shared: Arc<Mutex<DebugOverlayState>>,
    billboard_instances: Vec<helio_pass_billboard::BillboardInstance>,
    billboard_scratch: Vec<helio_pass_billboard::BillboardInstance>,
    /// True when billboard_instances was updated since the last rebuild.
    billboard_dirty: bool,
    /// Cached light count at last rebuild — detects add/remove.
    billboard_cached_light_count: usize,
    /// Cached movable_lights_generation at last rebuild — detects light updates.
    billboard_cached_light_gen: u64,
    /// Cached editor-hidden state at last rebuild.
    billboard_cached_editor_hidden: bool,
    /// Cached corona emitter generation at last rebuild.
    billboard_cached_corona_gen: u64,
    /// Monotonic generation for billboard GPU uploads.
    billboard_generation: u64,

    // ── Corona particle emitters ──────────────────────────────────────────
    corona_emitters: Vec<libhelio::GpuCoronaEmitter>,
    corona_emitter_generation: u64,

    water_volumes_buffer: wgpu::Buffer,
    water_hitboxes_buffer: wgpu::Buffer,
    /// Instant of the previous `render()` call, used to compute real `delta_time`.
    last_render_time: Instant,
    /// Frame delta time in seconds (for debug overlay display).
    delta_time: f32,
    /// Time spent in graph execution (milliseconds, previous frame).
    graph_time_ms: f32,
    /// Staging buffer for GPU culling stats readback (8 u32 counters).
    cull_stats_staging: wgpu::Buffer,
    /// Last frame's culling statistics, updated via GPU readback.
    cull_stats: [u32; 8],
    /// Frame time history for the timing graph (ring buffer).
    frame_times: Vec<f32>,
    frame_times_cursor: usize,
    /// Precomputed TAA jitter translation matrices (16-sample Halton sequence).
    /// Cached per resolution to avoid per-frame matrix construction overhead.
    jitter_matrices: [glam::Mat4; 16],
    /// Cached resolution for jitter matrix precomputation.
    jitter_cache_width: u32,
    jitter_cache_height: u32,
    /// Camera and viewport stored by the caller via [`set_gizmo_camera`] so that
    /// gizmo drawing and hit-testing can compute a screen-space-consistent size.
    gizmo_camera: Option<crate::scene::Camera>,
    gizmo_viewport_height: f32,

    /// Optional live portal handle for non-blocking profiling telemetry
    #[cfg(feature = "live-portal")]
    portal_handle: Option<helio_live_portal::LivePortalHandle>,

    // ── Baking ────────────────────────────────────────────────────────────
    /// Pending bake configuration.  Consumed in the first call to `render()`,
    /// blocking until all passes complete before the first draw.
    #[cfg(feature = "bake")]
    bake_pending: Option<helio_bake::BakeRequest>,
    /// GPU-resident baked data (AO, lightmaps, probes, PVS).
    /// Populated once, published into `FrameResources` every subsequent frame.
    #[cfg(feature = "bake")]
    baked_data: Option<std::sync::Arc<helio_bake::BakedData>>,
    /// Whether Helio owns the wgpu device (true) or is using an externally-owned
    /// device (false, e.g. GPUI). When false, device.poll() must never be called.
    owns_device: bool,
    /// Pending resize dimensions.  Set by `set_render_size`; consumed and applied
    /// (graph rebuild + texture recreation) at the start of the next `render()` call
    /// so that rapid resize events during window dragging only trigger one rebuild
    /// per rendered frame rather than one per pixel of drag movement.
    pending_resize: Option<(u32, u32)>,
    /// Force-clears the output target on the next frame. Set after resize so
    /// stale swapchain contents from the old size cannot leak into the new frame.
    clear_target_next_frame: bool,
}

enum GraphKind {
    Default,
    Simple,
    Custom,
}

pub struct DebugBatch<'a> {
    state: &'a mut DebugDrawState,
    lines_changed: bool,
    tris_changed: bool,
}

impl<'a> DebugBatch<'a> {
    pub fn line(&mut self, from: [f32; 3], to: [f32; 3], color: [f32; 4]) {
        self.state.user_lines.push(DebugVertex { position: from, _pad: 0.0, color });
        self.state.user_lines.push(DebugVertex { position: to, _pad: 0.0, color });
        self.lines_changed = true;
    }

    pub fn tri(&mut self, v0: [f32; 3], v1: [f32; 3], v2: [f32; 3], color: [f32; 4]) {
        self.state.user_tris.push(DebugVertex { position: v0, _pad: 0.0, color });
        self.state.user_tris.push(DebugVertex { position: v1, _pad: 0.0, color });
        self.state.user_tris.push(DebugVertex { position: v2, _pad: 0.0, color });
        self.tris_changed = true;
    }

    pub fn sphere(&mut self, center: [f32; 3], radius: f32, color: [f32; 4], segments: u32) {
        if segments < 4 { return; }
        for plane in 0..3 {
            let mut prev = glam::Vec3::ZERO;
            for i in 0..=segments {
                let theta = i as f32 / segments as f32 * std::f32::consts::TAU;
                let pos = match plane {
                    0 => glam::Vec3::new(radius * theta.cos(), radius * theta.sin(), 0.0),
                    1 => glam::Vec3::new(radius * theta.cos(), 0.0, radius * theta.sin()),
                    _ => glam::Vec3::new(0.0, radius * theta.cos(), radius * theta.sin()),
                } + glam::Vec3::from(center);
                if i > 0 {
                    self.line(prev.to_array(), pos.to_array(), color);
                }
                prev = pos;
            }
        }
    }

    pub fn cone(&mut self, apex: [f32; 3], axis: [f32; 3], height: f32, base_radius: f32, color: [f32; 4], segments: u32) {
        if segments < 3 { return; }
        let apex_v = glam::Vec3::from(apex);
        let dir = glam::Vec3::from(axis).normalize_or_zero();
        let base = apex_v + dir * height;
        let up = if dir.cross(glam::Vec3::Y).length_squared() < 1e-8 { glam::Vec3::X } else { glam::Vec3::Y };
        let tangent = dir.cross(up).normalize_or_zero();
        let bitangent = dir.cross(tangent).normalize_or_zero();
        let mut prev = base + tangent * base_radius;
        for i in 1..=segments {
            let theta = i as f32 / segments as f32 * std::f32::consts::TAU;
            let cur = base + (tangent * theta.cos() + bitangent * theta.sin()) * base_radius;
            self.line(prev.to_array(), cur.to_array(), color);
            self.line(cur.to_array(), apex_v.to_array(), color);
            prev = cur;
        }
    }

    pub fn filled_cone(&mut self, apex: [f32; 3], axis: [f32; 3], height: f32, base_radius: f32, color: [f32; 4], segments: u32) {
        if segments < 3 { return; }
        let apex_v = glam::Vec3::from(apex);
        let dir    = glam::Vec3::from(axis).normalize_or_zero();
        let base   = apex_v + dir * height;
        let up = if dir.cross(glam::Vec3::Y).length_squared() < 1e-8 { glam::Vec3::X } else { glam::Vec3::Y };
        let tangent   = dir.cross(up).normalize_or_zero();
        let bitangent = dir.cross(tangent).normalize_or_zero();
        let mut prev = base + tangent * base_radius;
        for i in 1..=segments {
            let theta = i as f32 / segments as f32 * std::f32::consts::TAU;
            let cur = base + (tangent * theta.cos() + bitangent * theta.sin()) * base_radius;
            self.tri(apex_v.to_array(), prev.to_array(), cur.to_array(), color);
            self.tri(base.to_array(), cur.to_array(), prev.to_array(), color);
            prev = cur;
        }
    }

    pub fn filled_box(&mut self, center: [f32; 3], half: f32, color: [f32; 4]) {
        let c = glam::Vec3::from(center);
        let h = half;
        let corners = [
            c + glam::Vec3::new(-h, -h, -h),
            c + glam::Vec3::new( h, -h, -h),
            c + glam::Vec3::new( h,  h, -h),
            c + glam::Vec3::new(-h,  h, -h),
            c + glam::Vec3::new(-h, -h,  h),
            c + glam::Vec3::new( h, -h,  h),
            c + glam::Vec3::new( h,  h,  h),
            c + glam::Vec3::new(-h,  h,  h),
        ];
        let quads: [[usize; 4]; 6] = [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 4, 7, 3],
            [1, 2, 6, 5],
            [0, 1, 5, 4],
            [3, 7, 6, 2],
        ];
        for [a, b, cc, d] in quads {
            self.tri(corners[a].to_array(), corners[b].to_array(), corners[cc].to_array(), color);
            self.tri(corners[a].to_array(), corners[cc].to_array(), corners[d].to_array(), color);
        }
    }

    fn finish(self) {
        if self.lines_changed {
            self.state.user_lines_generation = self.state.user_lines_generation.wrapping_add(1);
        }
        if self.tris_changed {
            self.state.user_tris_generation = self.state.user_tris_generation.wrapping_add(1);
        }
    }
}

impl Renderer {
    /// Precompute all 16 TAA jitter translation matrices for the given resolution.
    /// This avoids per-frame matrix construction overhead in render().
    fn compute_jitter_matrices(width: u32, height: u32) -> [glam::Mat4; 16] {
        let mut matrices = [glam::Mat4::IDENTITY; 16];
        for (i, raw) in HALTON_JITTER.iter().enumerate() {
            let jx = ((raw[0] - 0.5) * 2.0) / (width as f32);
            let jy = ((raw[1] - 0.5) * 2.0) / (height as f32);
            matrices[i] = glam::Mat4::from_translation(glam::Vec3::new(jx, jy, 0.0));
        }
        matrices
    }

    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, config: RendererConfig) -> Self {
        // On WebGPU, non-zero firstInstance in indirect draw calls requires the
        // `indirect-first-instance` feature (wgpu::Features::INDIRECT_FIRST_INSTANCE).
        // Without it, every draw with firstInstance>0 is silently skipped by the browser,
        // so only the first object (dense_index=0, the floor) would render.
        #[cfg(target_arch = "wasm32")]
        if !device.features().contains(wgpu::Features::INDIRECT_FIRST_INSTANCE) {
            log::error!(
                "helio: INDIRECT_FIRST_INSTANCE (WebGPU indirect-first-instance) is not \
                 available on this device. Only the first object in every scene will render. \
                 Please use a browser that supports the indirect-first-instance WebGPU feature \
                 (Chrome 113+, Firefox 122+, Safari 17+)."
            );
        }

        let mut scene = Scene::new(device.clone(), queue.clone());
        scene.set_render_size(config.width, config.height);

        let debug_state = Arc::new(Mutex::new(DebugDrawState::default()));

        let debug_camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Camera Buffer"),
            size: std::mem::size_of::<DebugCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let debug_overlay_shared = DebugOverlayState::new();
        let cull_stats_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cull Stats Buffer"),
            size: 32,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut graph = build_default_graph(
            &device, &queue, &scene, config, debug_state.clone(), &debug_camera_buffer,
            &cull_stats_buffer,
            Some(&debug_overlay_shared),
        );

        let (depth_texture, depth_view) = create_depth_resources(
            &device,
            config.internal_width(),
            config.internal_height(),
        );

        let (full_res_depth_texture, full_res_depth_view) = if config.render_scale < 1.0 {
            let (t, v) = create_depth_resources(&device, config.width, config.height);
            (Some(t), Some(v))
        } else {
            (None, None)
        };

        // Water volumes buffer (256 max volumes * 256 bytes each = 64KB)
        let water_volumes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Water Volumes Buffer"),
            size: 256 * 256, // Max 256 volumes, 256 bytes each
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Water hitboxes buffer (256 max hitboxes * 80 bytes each ≈ 20KB)
        let water_hitboxes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Water Hitboxes Buffer"),
            size: 256 * 80,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Precompute jitter matrices for TAA
        let internal_w = config.internal_width();
        let internal_h = config.internal_height();
        let jitter_matrices = Self::compute_jitter_matrices(internal_w, internal_h);

        let cull_stats_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CullStats Staging"),
            size: 32,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut renderer = Self {
            device,
            queue,
            graph,
            graph_kind: GraphKind::Default,
            scene,
            depth_texture,
            depth_view,
            output_width: config.width,
            output_height: config.height,
            render_scale: config.render_scale,
            full_res_depth_texture,
            full_res_depth_view,
            surface_format: config.surface_format,
            debug_camera_buffer,
            ambient_color: [0.05, 0.05, 0.08],
            ambient_intensity: 1.0,
            clear_color: [0.02, 0.02, 0.03, 1.0],
            gi_config: config.gi_config,
            shadow_quality: config.shadow_quality,
            debug_mode: config.debug_mode,
            debug_depth_test: true,
            editor_mode: false,
            custom_graph_builder: None,
            custom_graph_config: None,
            perf_overlay_mode: config.perf_overlay_mode,
            debug_state,
            debug_overlay_shared,
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
            last_render_time: Instant::now(),
            delta_time: 0.0,
            cull_stats_staging,
            cull_stats: [0; 8],
            graph_time_ms: 0.0,
            frame_times: vec![0.0; 200],
            frame_times_cursor: 0,
            jitter_matrices,
            jitter_cache_width: internal_w,
            jitter_cache_height: internal_h,
            #[cfg(feature = "live-portal")]
            portal_handle: None,
            #[cfg(feature = "bake")]
            bake_pending: None,
            #[cfg(feature = "bake")]
            baked_data: None,
            clear_target_next_frame: true,
            owns_device: true,
            pending_resize: Some((config.width, config.height)),
            gizmo_camera: None,
            gizmo_viewport_height: 0.0,
            cull_stats_buffer,
        };

        // Automatically start live performance dashboard if feature is enabled
        #[cfg(feature = "live-portal")]
        {
            match helio_live_portal::start_live_portal("127.0.0.1:3030") {
                Ok(handle) => {
                    log::info!("🌐 Live performance dashboard: {}", handle.url);
                    renderer.portal_handle = Some(handle);
                }
                Err(e) => {
                    log::warn!("Failed to start live performance dashboard: {}", e);
                }
            }
        }

        renderer
    }

    /// Create a `Renderer` backed by a **externally-owned** wgpu device.
    ///
    /// Use this when the `device` and `queue` are owned by another system —
    /// e.g., when Helio is embedded inside a UI framework such as GPUI that
    /// already manages the wgpu device lifecycle and event loop.
    ///
    /// The key difference from [`new`](Self::new) is that this renderer will
    /// **never** call `device.poll(wait_indefinitely)`.  Anything that requires
    /// blocking readback (GPU timestamp queries) falls back to a single
    /// non-blocking `PollType::Poll` tick per frame; if the data is not yet
    /// ready the previous frame's values are reused.  The device owner is
    /// responsible for calling `device.poll` at an appropriate cadence — GPUI
    /// does this through winit's `RedrawRequested` handler.
    ///
    /// # Example (GPUI integration)
    ///
    /// ```rust,ignore
    /// let surface = window.create_wgpu_surface(width, height, format)?;
    /// let device = Arc::new(surface.device().clone());
    /// let queue  = Arc::new(surface.queue().clone());
    ///
    /// let renderer = Renderer::new_with_external_device(
    ///     device,
    ///     queue,
    ///     RendererConfig::new(width, height, format),
    /// );
    /// ```
    pub fn new_with_external_device(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        config: RendererConfig,
    ) -> Self {
        let mut scene = Scene::new(device.clone(), queue.clone());
        scene.set_render_size(config.width, config.height);

        let debug_state = Arc::new(Mutex::new(DebugDrawState::default()));

        let debug_camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Camera Buffer"),
            size: std::mem::size_of::<DebugCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let debug_overlay_shared = DebugOverlayState::new();
        let cull_stats_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cull Stats Buffer"),
            size: 32,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut graph = super::graph::build_default_graph_external(
            &device,
            &queue,
            &scene,
            config,
            debug_state.clone(),
            &debug_camera_buffer,
            &cull_stats_buffer,
            Some(&debug_overlay_shared),
        );

        let (depth_texture, depth_view) =
            super::graph::create_depth_resources(&device, config.internal_width(), config.internal_height());

        let (full_res_depth_texture, full_res_depth_view) = if config.render_scale < 1.0 {
            let (t, v) = super::graph::create_depth_resources(&device, config.width, config.height);
            (Some(t), Some(v))
        } else {
            (None, None)
        };

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

        let internal_w = config.internal_width();
        let internal_h = config.internal_height();
        let jitter_matrices = Self::compute_jitter_matrices(internal_w, internal_h);
        let cull_stats_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CullStats Staging"),
            size: 32,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            graph,
            graph_kind: GraphKind::Default,
            scene,
            depth_texture,
            depth_view,
            output_width: config.width,
            output_height: config.height,
            render_scale: config.render_scale,
            full_res_depth_texture,
            full_res_depth_view,
            surface_format: config.surface_format,
            debug_camera_buffer,
            ambient_color: [0.05, 0.05, 0.08],
            ambient_intensity: 1.0,
            clear_color: [0.02, 0.02, 0.03, 1.0],
            gi_config: config.gi_config,
            shadow_quality: config.shadow_quality,
            debug_mode: config.debug_mode,
            debug_depth_test: true,
            editor_mode: false,
            custom_graph_builder: None,
            custom_graph_config: None,
            perf_overlay_mode: config.perf_overlay_mode,
            debug_state,
            debug_overlay_shared,
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
            last_render_time: Instant::now(),
            delta_time: 0.0,
            cull_stats_staging,
            cull_stats: [0; 8],
            graph_time_ms: 0.0,
            frame_times: vec![0.0; 200],
            frame_times_cursor: 0,
            jitter_matrices,
            jitter_cache_width: internal_w,
            jitter_cache_height: internal_h,
            #[cfg(feature = "live-portal")]
            portal_handle: None,
            #[cfg(feature = "bake")]
            bake_pending: None,
            #[cfg(feature = "bake")]
            baked_data: None,
            gizmo_camera: None,
            gizmo_viewport_height: 0.0,
            owns_device: false,
            pending_resize: Some((config.width, config.height)),
            clear_target_next_frame: true,
            cull_stats_buffer,
        }
    }

    pub fn set_gi_config(&mut self, gi_config: GiConfig) {
        self.gi_config = gi_config;
    }

    pub fn gi_config(&self) -> GiConfig {
        self.gi_config
    }

    pub fn set_shadow_quality(&mut self, quality: libhelio::ShadowQuality) {
        self.shadow_quality = quality;
        if matches!(self.graph_kind, GraphKind::Default) {
            if let Some(pass) = self.graph.find_pass_mut::<DeferredLightPass>() {
                pass.set_shadow_quality(quality, &self.queue);
            }
        }
    }

    pub fn set_debug_mode(&mut self, mode: u32) {
        self.debug_mode = mode;
        if matches!(self.graph_kind, GraphKind::Default) {
            if let Some(pass) = self.graph.find_pass_mut::<DeferredLightPass>() {
                pass.set_debug_mode(mode);
            }
            if let Some(pass) = self.graph.find_pass_mut::<VirtualGeometryPass>() {
                pass.debug_mode = mode;
            }
        }
    }

    pub fn available_debug_views(&self) -> Vec<helio_v3::DebugViewDescriptor> {
        self.graph.collect_debug_views()
    }

    pub fn set_perf_overlay_mode(&mut self, mode: PerfOverlayMode) {
        self.perf_overlay_mode = mode;
        if matches!(self.graph_kind, GraphKind::Default) {
            if let Some(pass) = self.graph.find_pass_mut::<PerfOverlayPass>() {
                pass.set_mode(mode);
            }
        }
    }

    pub fn set_debug_overlay_enabled(&mut self, enabled: bool) {
        if let Ok(mut state) = self.debug_overlay_shared.lock() {
            state.enabled = enabled;
        }
    }

    pub fn set_debug_depth_test(&mut self, enabled: bool) {
        self.debug_depth_test = enabled;
        // Both pipelines are pre-compiled inside DebugPass; toggling the flag is O(1)
        // and requires no pipeline or graph rebuild.
        for pass in self.graph.iter_passes_mut::<DebugDrawPass>() {
            pass.set_depth_test(enabled);
        }
    }

    pub fn set_editor_mode(&mut self, enabled: bool) {
        self.editor_mode = enabled;
        if enabled {
            self.scene.show_group(GroupId::EDITOR);
        } else {
            self.scene.hide_group(GroupId::EDITOR);
        }
        if let Ok(mut s) = self.debug_state.lock() {
            s.editor_enabled = enabled;
        }
    }

    pub fn is_editor_mode(&self) -> bool {
        self.editor_mode
    }

    /// Clear the per-frame debug geometry.
    ///
    /// **Must** be called every frame before any [`debug_batch`](Self::debug_batch)
    /// or direct debug-draw calls.  The generation counter is bumped so the
    /// render pass detects the change and uploads the (now-empty) buffer.
    pub fn debug_clear(&mut self) {
        if let Ok(mut s) = self.debug_state.lock() {
            if !s.user_lines.is_empty() {
                s.user_lines_generation = s.user_lines_generation.wrapping_add(1);
            }
            s.user_lines.clear();
            if !s.user_tris.is_empty() {
                s.user_tris_generation = s.user_tris_generation.wrapping_add(1);
            }
            s.user_tris.clear();
        }
    }

    pub fn debug_batch<F>(&mut self, f: F)
    where
        F: FnOnce(&mut DebugBatch<'_>),
    {
        if let Ok(mut s) = self.debug_state.lock() {
            let mut batch = DebugBatch {
                state: &mut s,
                lines_changed: false,
                tris_changed: false,
            };
            f(&mut batch);
            batch.finish();
        }
    }

    pub fn debug_line(&mut self, from: [f32; 3], to: [f32; 3], color: [f32; 4]) {
        if let Ok(mut s) = self.debug_state.lock() {
            s.user_lines.push(DebugVertex { position: from, _pad: 0.0, color });
            s.user_lines.push(DebugVertex { position: to, _pad: 0.0, color });
            s.user_lines_generation = s.user_lines_generation.wrapping_add(1);
        }
    }

    /// Submit a single filled triangle.  Every call queues 3 vertices.
    ///
    /// Triangles are rendered with alpha blending in a separate pass after all
    /// lines, so semi-transparent fills don't occlude one another in paint order.
    ///
    /// Vertex winding: CCW = front face (but cull_mode is None so both sides render).
    pub fn debug_tri(&mut self, v0: [f32; 3], v1: [f32; 3], v2: [f32; 3], color: [f32; 4]) {
        if let Ok(mut s) = self.debug_state.lock() {
            s.user_tris.push(DebugVertex { position: v0, _pad: 0.0, color });
            s.user_tris.push(DebugVertex { position: v1, _pad: 0.0, color });
            s.user_tris.push(DebugVertex { position: v2, _pad: 0.0, color });
            s.user_tris_generation = s.user_tris_generation.wrapping_add(1);
        }
    }

    /// Fill a disk (flat circle) with triangles.
    ///
    /// `normal` is the outward-facing side direction.  The disk lies in the
    /// plane perpendicular to `normal` centred at `center`.
    pub fn debug_filled_disk(&mut self, center: [f32; 3], normal: [f32; 3], radius: f32, color: [f32; 4], segments: u32) {
        if segments < 3 { return; }
        let c = glam::Vec3::from(center);
        let n = glam::Vec3::from(normal).normalize_or_zero();
        let up = if n.abs_diff_eq(glam::Vec3::Y, 1e-5) { glam::Vec3::X } else { glam::Vec3::Y };
        let tangent   = n.cross(up).normalize_or_zero();
        let bitangent = n.cross(tangent).normalize_or_zero();
        let mut prev = c + tangent * radius;
        for i in 1..=segments {
            let theta = i as f32 / segments as f32 * std::f32::consts::TAU;
            let cur = c + (tangent * theta.cos() + bitangent * theta.sin()) * radius;
            self.debug_tri(c.to_array(), prev.to_array(), cur.to_array(), color);
            prev = cur;
        }
    }

    /// Fill a cone with triangles.  Solid sides + base cap.
    ///
    /// `apex` is the tip.  `axis` points from apex toward the base.
    /// `height` is the apex-to-base distance.  `base_radius` is the base circle radius.
    pub fn debug_filled_cone(&mut self, apex: [f32; 3], axis: [f32; 3], height: f32, base_radius: f32, color: [f32; 4], segments: u32) {
        if segments < 3 { return; }
        let apex_v = glam::Vec3::from(apex);
        let dir    = glam::Vec3::from(axis).normalize_or_zero();
        let base   = apex_v + dir * height;
        // Use cross-product length to detect parallel-to-Y in either direction (+Y or -Y).
        let up = if dir.cross(glam::Vec3::Y).length_squared() < 1e-8 { glam::Vec3::X } else { glam::Vec3::Y };
        let tangent   = dir.cross(up).normalize_or_zero();
        let bitangent = dir.cross(tangent).normalize_or_zero();
        let mut prev = base + tangent * base_radius;
        for i in 1..=segments {
            let theta = i as f32 / segments as f32 * std::f32::consts::TAU;
            let cur = base + (tangent * theta.cos() + bitangent * theta.sin()) * base_radius;
            // Lateral face: apex + two base-ring points (CCW outward).
            self.debug_tri(apex_v.to_array(), prev.to_array(), cur.to_array(), color);
            // Base cap: reverse winding so cap faces away from apex.
            self.debug_tri(base.to_array(), cur.to_array(), prev.to_array(), color);
            prev = cur;
        }
    }

    /// Fill an axis-aligned box with triangles.  `half` is the half-extent on each side.
    pub fn debug_filled_box(&mut self, center: [f32; 3], half: f32, color: [f32; 4]) {
        let c = glam::Vec3::from(center);
        let h = half;
        // 8 corners: naming convention is (±x, ±y, ±z)
        let corners = [
            c + glam::Vec3::new(-h, -h, -h), // 0
            c + glam::Vec3::new( h, -h, -h), // 1
            c + glam::Vec3::new( h,  h, -h), // 2
            c + glam::Vec3::new(-h,  h, -h), // 3
            c + glam::Vec3::new(-h, -h,  h), // 4
            c + glam::Vec3::new( h, -h,  h), // 5
            c + glam::Vec3::new( h,  h,  h), // 6
            c + glam::Vec3::new(-h,  h,  h), // 7
        ];
        // 6 faces, 2 triangles each.  Winding is CCW when viewed from outside.
        let quads: [[usize; 4]; 6] = [
            [0, 3, 2, 1], // -Z (front)
            [4, 5, 6, 7], // +Z (back)
            [0, 4, 7, 3], // -X (left)
            [1, 2, 6, 5], // +X (right)
            [0, 1, 5, 4], // -Y (bottom)
            [3, 7, 6, 2], // +Y (top)
        ];
        for [a, b, cc, d] in quads {
            self.debug_tri(corners[a].to_array(), corners[b].to_array(), corners[cc].to_array(), color);
            self.debug_tri(corners[a].to_array(), corners[cc].to_array(), corners[d].to_array(), color);
        }
    }

    pub fn debug_circle(&mut self, center: [f32; 3], radius: f32, color: [f32; 4], segments: u32) {
        if segments < 3 { return; }
        let (cx, cy, cz) = (center[0], center[1], center[2]);
        let step = std::f32::consts::TAU / segments as f32;
        let mut last = (cx + radius, cy, cz);
        for i in 1..=segments {
            let theta = i as f32 * step;
            let next = (cx + radius * theta.cos(), cy, cz + radius * theta.sin());
            self.debug_line([last.0, last.1, last.2], [next.0, next.1, next.2], color);
            last = next;
        }
    }

    pub fn debug_sphere(&mut self, center: [f32; 3], radius: f32, color: [f32; 4], segments: u32) {
        if segments < 4 { return; }
        for plane in 0..3 {
            let mut prev = glam::Vec3::ZERO;
            for i in 0..=segments {
                let theta = i as f32 / segments as f32 * std::f32::consts::TAU;
                let pos = match plane {
                    0 => glam::Vec3::new(radius * theta.cos(), radius * theta.sin(), 0.0),
                    1 => glam::Vec3::new(radius * theta.cos(), 0.0, radius * theta.sin()),
                    _ => glam::Vec3::new(0.0, radius * theta.cos(), radius * theta.sin()),
                } + glam::Vec3::from(center);
                if i > 0 {
                    self.debug_line(prev.to_array(), pos.to_array(), color);
                }
                prev = pos;
            }
        }
    }

    pub fn debug_torus(&mut self, center: [f32; 3], normal: [f32; 3], major_radius: f32, minor_radius: f32, color: [f32; 4], major_segments: u32, minor_segments: u32) {
        if major_segments < 3 || minor_segments < 3 { return; }
        let c = glam::Vec3::from(center);
        let n = glam::Vec3::from(normal).normalize_or_zero();
        let up = if n.abs_diff_eq(glam::Vec3::Y, 1e-6) { glam::Vec3::X } else { glam::Vec3::Y };
        let tangent = n.cross(up).normalize_or_zero();
        let bitangent = n.cross(tangent).normalize_or_zero();

        for j in 0..major_segments {
            let theta0 = 2.0 * std::f32::consts::TAU * (j as f32) / (major_segments as f32);
            let theta1 = 2.0 * std::f32::consts::TAU * ((j + 1) as f32) / (major_segments as f32);
            let center0 = c + (tangent * theta0.cos() + bitangent * theta0.sin()) * major_radius;
            let center1 = c + (tangent * theta1.cos() + bitangent * theta1.sin()) * major_radius;

            let mut pprev0 = center0 + (n * minor_radius);
            let mut pprev1 = center1 + (n * minor_radius);
            for i in 1..=minor_segments {
                let phi = 2.0 * std::f32::consts::TAU * (i as f32) / (minor_segments as f32);
                let offset = (n * phi.cos() + (tangent * theta0.cos() + bitangent * theta0.sin()) * phi.sin()).normalize_or_zero() * minor_radius;
                let cur0 = center0 + offset;
                let offset1 = (n * phi.cos() + (tangent * theta1.cos() + bitangent * theta1.sin()) * phi.sin()).normalize_or_zero() * minor_radius;
                let cur1 = center1 + offset1;

                self.debug_line(pprev0.to_array(), cur0.to_array(), color);
                self.debug_line(pprev1.to_array(), cur1.to_array(), color);
                self.debug_line(pprev0.to_array(), pprev1.to_array(), color);

                pprev0 = cur0;
                pprev1 = cur1;
            }
        }
    }

    pub fn debug_cylinder(&mut self, base_center: [f32; 3], axis: [f32; 3], height: f32, radius: f32, color: [f32; 4], segments: u32) {
        if segments < 3 { return; }
        let base = glam::Vec3::from(base_center);
        let dir = glam::Vec3::from(axis).normalize_or_zero();
        let top = base + dir * height;
        let up = if dir.abs_diff_eq(glam::Vec3::Y, 1e-5) { glam::Vec3::X } else { glam::Vec3::Y };
        let tangent = dir.cross(up).normalize_or_zero();
        let bitangent = dir.cross(tangent).normalize_or_zero();
        let mut prev_base = base + tangent * radius;
        let mut prev_top = top + tangent * radius;
        for i in 1..=segments {
            let theta = i as f32 / segments as f32 * std::f32::consts::TAU;
            let dir_circle = tangent * theta.cos() + bitangent * theta.sin();
            let cur_base = base + dir_circle * radius;
            let cur_top = top + dir_circle * radius;
            self.debug_line(prev_base.to_array(), cur_base.to_array(), color);
            self.debug_line(prev_top.to_array(), cur_top.to_array(), color);
            self.debug_line(prev_base.to_array(), prev_top.to_array(), color);
            prev_base = cur_base;
            prev_top = cur_top;
        }
    }

    pub fn debug_cone(&mut self, apex: [f32; 3], axis: [f32; 3], height: f32, base_radius: f32, color: [f32; 4], segments: u32) {
        if segments < 3 { return; }
        let apex_v = glam::Vec3::from(apex);
        let dir = glam::Vec3::from(axis).normalize_or_zero();
        let base = apex_v + dir * height;
        // Use cross-product length to detect parallel-to-Y in either direction (+Y or -Y).
        let up = if dir.cross(glam::Vec3::Y).length_squared() < 1e-8 { glam::Vec3::X } else { glam::Vec3::Y };
        let tangent = dir.cross(up).normalize_or_zero();
        let bitangent = dir.cross(tangent).normalize_or_zero();
        let mut prev = base + tangent * base_radius;
        for i in 1..=segments {
            let theta = i as f32 / segments as f32 * std::f32::consts::TAU;
            let cur = base + (tangent * theta.cos() + bitangent * theta.sin()) * base_radius;
            self.debug_line(prev.to_array(), cur.to_array(), color);
            self.debug_line(cur.to_array(), apex_v.to_array(), color);
            prev = cur;
        }
    }

    pub fn debug_frustum(&mut self, origin: [f32; 3], forward: [f32; 3], up: [f32; 3], fov_y: f32, aspect: f32, near: f32, far: f32, color: [f32; 4]) {
        let o = glam::Vec3::from(origin);
        let fwd = glam::Vec3::from(forward).normalize_or_zero();
        let upv = glam::Vec3::from(up).normalize_or_zero();
        let rightv = fwd.cross(upv).normalize_or_zero();
        let n_center = o + fwd * near;
        let f_center = o + fwd * far;
        let nh = (fov_y * 0.5).tan() * near;
        let nw = nh * aspect;
        let fh = (fov_y * 0.5).tan() * far;
        let fw = fh * aspect;

        let n = [
            n_center + upv * nh - rightv * nw,
            n_center + upv * nh + rightv * nw,
            n_center - upv * nh + rightv * nw,
            n_center - upv * nh - rightv * nw,
        ];
        let f = [
            f_center + upv * fh - rightv * fw,
            f_center + upv * fh + rightv * fw,
            f_center - upv * fh + rightv * fw,
            f_center - upv * fh - rightv * fw,
        ];

        for i in 0..4 {
            self.debug_line(n[i].to_array(), n[(i + 1) % 4].to_array(), color);
            self.debug_line(f[i].to_array(), f[(i + 1) % 4].to_array(), color);
            self.debug_line(n[i].to_array(), f[i].to_array(), color);
        }
    }

    pub fn shadow_quality(&self) -> libhelio::ShadowQuality {
        self.shadow_quality
    }

    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    pub fn scene_mut(&mut self) -> &mut Scene {
        &mut self.scene
    }

    pub fn debug_state(&self) -> Arc<Mutex<DebugDrawState>> {
        self.debug_state.clone()
    }

    pub fn debug_camera_buf(&self) -> &wgpu::Buffer {
        &self.debug_camera_buffer
    }

    pub fn cull_stats_buf(&self) -> &wgpu::Buffer {
        &self.cull_stats_buffer
    }

    pub fn debug_overlay_shared(&self) -> &Arc<Mutex<DebugOverlayState>> {
        &self.debug_overlay_shared
    }

    pub fn camera_buffer(&self) -> &wgpu::Buffer {
        self.scene.gpu_scene().camera.buffer()
    }

    pub fn mesh_buffers(&self) -> MeshBuffers<'_> {
        self.scene.mesh_buffers()
    }

    pub fn dynamic_mesh_buffers(&self) -> MeshBuffers<'_> {
        self.scene.dynamic_mesh_buffers()
    }

    pub fn add_pass(&mut self, pass: Box<dyn helio_v3::RenderPass>) {
        self.graph.add_pass(pass);
    }

    /// Returns a typed mutable reference to the first pass of type `T` in the graph.
    ///
    /// Requires the pass to implement `RenderPass::as_any_mut()` (returning `Some(self)`).
    /// Use this to configure a custom pass after it has been added to the graph without
    /// holding a raw pointer:
    ///
    /// ```rust,ignore
    /// renderer.find_pass_mut::<SdfPass>()?.add_edit(edit);
    /// ```
    pub fn find_pass_mut<T: RenderPass + 'static>(&mut self) -> Option<&mut T> {
        self.graph.find_pass_mut::<T>()
    }

    /// Returns a typed immutable reference to the first pass of type `T` in the graph.
    ///
    /// Requires the pass to implement `RenderPass::as_any()` (returning `Some(self)`).
    pub fn find_pass<T: RenderPass + 'static>(&self) -> Option<&T> {
        self.graph.find_pass::<T>()
    }

    /// Queue a render size change.  The actual graph rebuild and texture
    /// recreation is deferred to the next `render()` call so that rapid
    /// `Resized` events during a window drag only trigger **one** rebuild per
    /// rendered frame instead of one rebuild per pixel of drag movement.
    pub fn set_render_size(&mut self, width: u32, height: u32) {
        // Always update the logical dimensions immediately so that callers
        // querying `output_width` / `output_height` see the current value,
        // and so that aspect-ratio computations in the same frame are correct.
        self.output_width = width;
        self.output_height = height;
        self.pending_resize = Some((width, height));
    }

    /// Perform the actual graph rebuild and depth-texture recreation for a
    /// pending resize.  Called at the top of `render()`.
    fn apply_resize_now(&mut self, width: u32, height: u32) {
        let resize_start = Instant::now();
        
        let scene_start = Instant::now();
        self.scene.set_render_size(width, height);
        log::trace!("apply_resize_now: scene.set_render_size {}ms", scene_start.elapsed().as_secs_f64() * 1000.0);
        
        let config = RendererConfig {
            width,
            height,
            surface_format: self.surface_format,
            gi_config: self.gi_config,
            shadow_quality: self.shadow_quality,
            debug_mode: self.debug_mode,
            render_scale: self.render_scale,
            perf_overlay_mode: self.perf_overlay_mode,
            shadow_atlas_size: 1024,
        };
        
        let depth_start = Instant::now();
        let (depth_texture, depth_view) = create_depth_resources(
            &self.device,
            config.internal_width(),
            config.internal_height(),
        );
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
        log::trace!("apply_resize_now: internal depth {}x{} {}ms", config.internal_width(), config.internal_height(), depth_start.elapsed().as_secs_f64() * 1000.0);
        
        let full_depth_start = Instant::now();
        if self.render_scale < 1.0 {
            let (t, v) = create_depth_resources(&self.device, width, height);
            self.full_res_depth_texture = Some(t);
            self.full_res_depth_view = Some(v);
            log::trace!("apply_resize_now: full-res depth {}x{} {}ms", width, height, full_depth_start.elapsed().as_secs_f64() * 1000.0);
        } else {
            self.full_res_depth_texture = None;
            self.full_res_depth_view = None;
        }

        // Ensure the first frame after resize starts from a known target state.
        self.clear_target_next_frame = true;

        let graph_start = Instant::now();
        match self.graph_kind {
            GraphKind::Default => {
                // Safety-first path: rebuild the default graph on resize so Helio
                // re-creates and rebinds all pass-owned resources from a single
                // source of truth. This avoids stale views/buffers after resize.
                self.graph = if self.owns_device {
                    build_default_graph(
                        &self.device,
                        &self.queue,
                        &self.scene,
                        config,
                        self.debug_state.clone(),
                        &self.debug_camera_buffer,
                        &self.cull_stats_buffer,
                        Some(&self.debug_overlay_shared),
                    )
                } else {
                    super::graph::build_default_graph_external(
                        &self.device,
                        &self.queue,
                        &self.scene,
                        config,
                        self.debug_state.clone(),
                        &self.debug_camera_buffer,
                        &self.cull_stats_buffer,
                        Some(&self.debug_overlay_shared),
                    )
                };
                log::trace!("apply_resize_now: graph rebuild {}ms", graph_start.elapsed().as_secs_f64() * 1000.0);

                // The rebuilt graph contains fresh pass instances; mark water
                // volumes dirty so simulation params get re-applied next frame.
                let water_start = Instant::now();
                self.scene.mark_water_volumes_dirty();
                log::trace!("apply_resize_now: mark_water_volumes_dirty {}ms", water_start.elapsed().as_secs_f64() * 1000.0);
            }
            GraphKind::Simple => {
                self.graph.set_render_size(width, height);
                log::trace!("apply_resize_now: simple graph set_render_size {}ms", graph_start.elapsed().as_secs_f64() * 1000.0);
            }
            GraphKind::Custom => {
                if let Some(builder) = &self.custom_graph_builder {
                    if let Some(prev_config) = self.custom_graph_config {
                        let new_cfg = RendererConfig {
                            width,
                            height,
                            ..prev_config
                        };
                        self.graph = builder(
                            &self.device,
                            &self.queue,
                            &self.scene,
                            new_cfg,
                            self.debug_state.clone(),
                            &self.debug_camera_buffer,
                            &self.cull_stats_buffer,
                            Some(&self.debug_overlay_shared),
                        );
                        self.custom_graph_config = Some(new_cfg);
                        log::trace!("apply_resize_now: custom graph rebuild {}ms", graph_start.elapsed().as_secs_f64() * 1000.0);
                        
                        // Same as Default: re-dirty so the new pass gets wind params.
                        let water_start = Instant::now();
                        self.scene.mark_water_volumes_dirty();
                        log::trace!("apply_resize_now: mark_water_volumes_dirty {}ms", water_start.elapsed().as_secs_f64() * 1000.0);
                    } else {
                        self.graph.set_render_size(width, height);
                        log::trace!("apply_resize_now: custom graph set_render_size {}ms", graph_start.elapsed().as_secs_f64() * 1000.0);
                    }
                } else {
                    self.graph.set_render_size(width, height);
                }
            }
        }
        
        log::trace!("apply_resize_now: total resize {}ms", resize_start.elapsed().as_secs_f64() * 1000.0);
    }

    pub fn set_render_scale(&mut self, scale: f32) {
        self.render_scale = scale.clamp(0.25, 1.0);
        self.set_render_size(self.output_width, self.output_height);
    }

    pub fn render_scale(&self) -> f32 {
        self.render_scale
    }

    pub fn set_clear_color(&mut self, color: [f32; 4]) {
        self.clear_color = color;
    }

    pub fn set_ambient(&mut self, color: [f32; 3], intensity: f32) {
        self.ambient_color = color;
        self.ambient_intensity = intensity;
    }

    pub fn set_graph(&mut self, graph: RenderGraph) {
        self.graph = graph;
        self.graph_kind = GraphKind::Custom;
        self.custom_graph_builder = None;
        self.custom_graph_config = None;
    }

    pub fn set_graph_custom(
        &mut self,
        graph: RenderGraph,
        config: RendererConfig,
        builder: CustomGraphBuilder,
    ) {
        self.graph = graph;
        self.graph_kind = GraphKind::Custom;
        self.custom_graph_builder = Some(builder);
        self.custom_graph_config = Some(config);
    }

    pub fn use_simple_graph(&mut self) {
        self.graph = build_simple_graph(&self.device, &self.queue, self.surface_format);
        self.graph_kind = GraphKind::Simple;
    }

    pub fn use_default_graph(&mut self) {
        let config = RendererConfig {
            width: self.output_width,
            height: self.output_height,
            surface_format: self.surface_format,
            gi_config: self.gi_config,
            shadow_quality: self.shadow_quality,
            debug_mode: self.debug_mode,
            render_scale: self.render_scale,
            perf_overlay_mode: self.perf_overlay_mode,
            shadow_atlas_size: 1024,
        };
        self.graph = if self.owns_device {
            build_default_graph(
                &self.device,
                &self.queue,
                &self.scene,
                config,
                self.debug_state.clone(),
                &self.debug_camera_buffer,
                &self.cull_stats_buffer,
                Some(&self.debug_overlay_shared),
            )
        } else {
            super::graph::build_default_graph_external(
                &self.device,
                &self.queue,
                &self.scene,
                config,
                self.debug_state.clone(),
                &self.debug_camera_buffer,
                &self.cull_stats_buffer,
                Some(&self.debug_overlay_shared),
            )
        };
        self.graph_kind = GraphKind::Default;
    }

    pub fn optimize_scene_layout(&mut self) {
        self.scene.optimize_scene_layout();
    }

    /// Queue a bake to run **once**, blocking before the first rendered frame.
    ///
    /// Helio will call [`helio_bake::run_bake_blocking`] at the top of the next
    /// `render()` invocation, writing cache files to `request.config.cache_dir`
    /// (skipping any pass whose cache file already exists).  After the bake,
    /// baked AO is injected directly into `SsaoPass` so that pass skips its
    /// per-frame screen-space computation, and all baked resources are available
    /// via `FrameResources` fields for use by downstream passes.
    ///
    /// Calling this a second time replaces any previous pending request.
    #[cfg(feature = "bake")]
    pub fn configure_bake(&mut self, request: helio_bake::BakeRequest) {
        self.bake_pending = Some(request);
    }

    /// Automatically configure baking using all static objects and lights in the scene.
    ///
    /// This is a convenience method that extracts static geometry from the scene
    /// automatically, eliminating the need to manually build a `SceneGeometry`.
    /// Equivalent to calling `scene.build_static_bake_scene()` and then
    /// `configure_bake()`, but with a cleaner API.
    ///
    /// # Example
    /// ```ignore
    /// // After populating your scene normally with insert_object, insert_light, etc...
    /// renderer.auto_bake(BakeConfig::fast("indoor_cathedral_water"));
    /// ```
    ///
    /// # See Also
    /// - [`Scene::build_static_bake_scene`] - for manual control over what gets baked
    /// - [`configure_bake`](Self::configure_bake) - for explicit scene geometry specification
    #[cfg(feature = "bake")]
    pub fn auto_bake(&mut self, config: helio_bake::BakeConfig) {
        let scene = self.scene.build_static_bake_scene();
        self.configure_bake(helio_bake::BakeRequest { scene, config });
    }


    pub fn set_billboard_instances(&mut self, instances: &[helio_pass_billboard::BillboardInstance]) {
        self.billboard_instances.clear();
        self.billboard_instances.extend_from_slice(instances);
        self.billboard_dirty = true;
    }

    pub fn set_corona_emitters(&mut self, emitters: &[libhelio::GpuCoronaEmitter]) {
        self.corona_emitters.clear();
        self.corona_emitters.extend_from_slice(emitters);
        self.corona_emitter_generation = self.corona_emitter_generation.wrapping_add(1);
    }

    /// Store the current camera and viewport height for screen-space gizmo sizing.
    ///
    /// Call this **before** [`draw_gizmos`](crate::EditorState::draw_gizmos) / [`update_hover`](crate::EditorState::update_hover)
    /// each frame so that gizmos are sized consistently regardless of camera distance.
    /// Typically the same camera that is passed to [`render`](Self::render).
    pub fn set_gizmo_camera(&mut self, camera: &crate::scene::Camera, viewport_height: f32) {
        self.gizmo_camera = Some(*camera);
        self.gizmo_viewport_height = viewport_height;
    }

    /// Returns the camera and viewport height previously set via [`set_gizmo_camera`],
    /// or [`None`] if no camera has been set yet.
    pub fn gizmo_camera_info(&self) -> Option<(&crate::scene::Camera, f32)> {
        self.gizmo_camera.as_ref().map(|c| (c, self.gizmo_viewport_height))
    }

    pub fn output_width(&self) -> u32 {
        self.output_width
    }

    pub fn output_height(&self) -> u32 {
        self.output_height
    }

    pub fn render(&mut self, camera: &Camera, target: &wgpu::TextureView) -> HelioResult<()> {
        // ── Deferred resize: apply at most once per frame ─────────────────
        // `set_render_size` only records a pending size to avoid rebuilding
        // the full render graph on every pixel of a window drag.  We flush it
        // here, at the top of the first render call after the resize settles.
        if let Some((w, h)) = self.pending_resize.take() {
            self.apply_resize_now(w, h);
        }

        // ── Baking: run once, blocking, before the first drawn frame ──────
        #[cfg(feature = "bake")]
        if let Some(request) = self.bake_pending.take() {
            let obj_count = request.scene.meshes.len();
            let light_count = request.scene.lights.len();
            
            log::info!(
                "[helio-bake] Starting pre-frame-1 bake for scene '{}' (cache: {})…",
                request.config.scene_name,
                request.config.cache_dir.display(),
            );
            
            let bake_start = Instant::now();
            let baked = helio_bake::run_bake_blocking(
                &self.device,
                &self.queue,
                &request.scene,
                &request.config,
            )
            .map_err(|e| helio_v3::Error::InvalidPassConfig(e.to_string()))?;
            let bake_duration = bake_start.elapsed();
            
            let baked = std::sync::Arc::new(baked);
            // Bypass per-frame SSAO — SsaoPass holds an Arc to the baked AO view
            // and skips GPU execution while that override is set.
            if let Some(pass) = self.graph.find_pass_mut::<helio_pass_ssao::SsaoPass>() {
                pass.set_baked_ao(baked.ao_view());
            }
            
            // Log detailed stats
            log::info!(
                "[helio-bake] ✓ Bake complete in {:.2}s — {} objects, {} lights (avg {:.1}ms/obj)",
                bake_duration.as_secs_f32(),
                obj_count,
                light_count,
                if obj_count > 0 { bake_duration.as_millis() as f32 / obj_count as f32 } else { 0.0 }
            );
            
            self.baked_data = Some(baked.clone());

            // Upload lightmap atlas regions to GBuffer pass
            if let Some(gbuffer_pass) = self.graph.find_pass_mut::<helio_pass_gbuffer::GBufferPass>() {
                let regions_gpu = baked.lightmap_atlas_regions_gpu();
                gbuffer_pass.upload_lightmap_atlas_regions(&self.device, &self.queue, &regions_gpu);
            }

            // Update lightmap indices in scene objects
            self.scene.update_lightmap_indices(baked.lightmap_atlas_regions());
        }

        // ── Check for invalidated bake ────────────────────────────────────
        #[cfg(feature = "bake")]
        if self.baked_data.is_some() && self.scene.is_bake_invalidated() {
            log::warn!(
                "[helio-bake] ⚠️  Static geometry or lights have been added since the last bake!\n\
                 The baked lighting is now out of date. Call renderer.auto_bake() again to rebake the scene."
            );
        }

        // Compute real frame delta, capped at 100 ms to avoid spiral-of-death on
        // slow frames (e.g. first frame, window unfocus/refocus, GPU stalls).
        let now = Instant::now();
        let dt = now.duration_since(self.last_render_time).as_secs_f32().min(0.1);
        self.last_render_time = now;
        self.delta_time = dt;
        self.frame_times[self.frame_times_cursor] = dt;
        self.frame_times_cursor = (self.frame_times_cursor + 1) % self.frame_times.len();
        self.graph.set_delta_time(dt);

        // OPTIMIZATION: Use precomputed jitter matrices (avoid per-frame Mat4 construction).
        // Recompute cache only if internal resolution changes (rare).
        let internal_w = (((self.output_width as f32) * self.render_scale).ceil() as u32).max(1);
        let internal_h = (((self.output_height as f32) * self.render_scale).ceil() as u32).max(1);
        if internal_w != self.jitter_cache_width || internal_h != self.jitter_cache_height {
            self.jitter_matrices = Self::compute_jitter_matrices(internal_w, internal_h);
            self.jitter_cache_width = internal_w;
            self.jitter_cache_height = internal_h;
        }

        let frame_idx = self.scene.gpu_scene().frame_count;
        let jitter_mat = self.jitter_matrices[(frame_idx % 16) as usize];
        let raw = HALTON_JITTER[(frame_idx % 16) as usize]; // Still needed for jx/jy below
        let jx = ((raw[0] - 0.5) * 2.0) / (internal_w as f32);
        let jy = ((raw[1] - 0.5) * 2.0) / (internal_h as f32);
        let jittered_m = jitter_mat * camera.proj * camera.view;
        let col = jittered_m.to_cols_array();
        let debug_camera_uniform = DebugCameraUniform {
            view_proj: [
                [col[0],  col[1],  col[2],  col[3]],
                [col[4],  col[5],  col[6],  col[7]],
                [col[8],  col[9],  col[10], col[11]],
                [col[12], col[13], col[14], col[15]],
            ],
        };
        self.queue.write_buffer(
            &self.debug_camera_buffer,
            0,
            bytemuck::bytes_of(&debug_camera_uniform),
        );

        let mut jittered_camera = *camera;
        jittered_camera.proj = jitter_mat * camera.proj;
        jittered_camera.jitter = [jx, jy];
        self.scene.update_camera(jittered_camera);
        self.scene.flush();

        let editor_hidden = self.scene.is_group_hidden(GroupId::EDITOR);
        let light_count = self.scene.gpu_scene().lights.len();
        let light_gen = self.scene.gpu_scene().movable_lights_generation;
        let corona_gen = self.corona_emitter_generation;
        if self.billboard_dirty
            || light_count != self.billboard_cached_light_count
            || light_gen != self.billboard_cached_light_gen
            || editor_hidden != self.billboard_cached_editor_hidden
            || corona_gen != self.billboard_cached_corona_gen
        {
            self.billboard_scratch.clear();
            self.billboard_scratch.extend_from_slice(&self.billboard_instances);
            if !editor_hidden {
                for light in self.scene.gpu_scene().lights.as_slice() {
                    if light.light_type == libhelio::LightType::Point as u32
                        || light.light_type == libhelio::LightType::Spot as u32
                    {
                        let [x, y, z, _] = light.position_range;
                        let [r, g, b, _] = light.color_intensity;
                        self.billboard_scratch.push(helio_pass_billboard::BillboardInstance {
                            world_pos: [x, y, z, 0.0],
                            scale_flags: [0.25, 0.25, 0.0, 0.0],
                            color: [r, g, b, 1.0],
                        });
                    }
                }
                // Corona emitter billboards — cyan tint to distinguish from lights
                for emitter in &self.corona_emitters {
                    let [x, y, z, _] = emitter.transform[3];
                    self.billboard_scratch.push(helio_pass_billboard::BillboardInstance {
                        world_pos: [x, y, z, 0.0],
                        scale_flags: [0.25, 0.25, 0.0, 0.0],
                        color: [0.2, 0.8, 1.0, 1.0],
                    });
                }
            }
            self.billboard_generation = self.billboard_generation.wrapping_add(1);
            self.billboard_dirty = false;
            self.billboard_cached_light_count = light_count;
            self.billboard_cached_light_gen = light_gen;
            self.billboard_cached_editor_hidden = editor_hidden;
            self.billboard_cached_corona_gen = corona_gen;
        }

        // Upload water volumes to GPU only when the descriptor has changed.
        // get_water_volumes_gpu_slice() avoids heap allocations at steady state.
        // NOTE: must happen before the `texture_views` ArrayVec is built, since
        // clear_water_volumes_dirty() requires `&mut self.scene` and cannot
        // coexist with the immutable borrows held by that ArrayVec.
        let water_volume_count = self.scene.water_volumes_count();
        if water_volume_count > 0 && self.scene.water_volumes_dirty() {
            let water_volumes = self.scene.get_water_volumes_gpu_slice();
            let water_volume_dirty_range = self.scene.water_volumes_dirty_range();
            if let Some(pass) = self.graph.find_pass_mut::<helio_pass_water_sim::WaterSimPass>() {
                let vol = &water_volumes[0];
                pass.set_sim_dynamics(vol.sim_dynamics[0], vol.sim_dynamics[1]);
                pass.set_wave_scale(vol.sim_dynamics[2]);
                pass.set_wave_speed(vol.wave_params[2]);
                pass.set_wind([vol.wind_params[0], vol.wind_params[1]], vol.wind_params[2]);
            }
            if let Some((start, end)) = water_volume_dirty_range {
                self.queue.write_buffer(
                    &self.water_volumes_buffer,
                    (start * std::mem::size_of::<libhelio::GpuWaterVolume>()) as u64,
                    bytemuck::cast_slice(&water_volumes[start..end]),
                );
            }
            self.scene.clear_water_volumes_dirty();
        }

        // Upload water hitboxes to GPU only when they have changed.
        let water_hitbox_count = self.scene.water_hitboxes_count();
        if water_hitbox_count > 0 && self.scene.water_hitboxes_dirty() {
            let water_hitboxes = self.scene.get_water_hitboxes_gpu_slice();
            let water_hitbox_dirty_range = self.scene.water_hitboxes_dirty_range();
            if let Some((start, end)) = water_hitbox_dirty_range {
                self.queue.write_buffer(
                    &self.water_hitboxes_buffer,
                    (start * std::mem::size_of::<libhelio::GpuWaterHitbox>()) as u64,
                    bytemuck::cast_slice(&water_hitboxes[start..end]),
                );
            }
            self.scene.clear_water_hitboxes_dirty();
        }

        let mut texture_views = ArrayVec::<&wgpu::TextureView, MAX_TEXTURES>::new();
        let mut samplers = ArrayVec::<&wgpu::Sampler, MAX_TEXTURES>::new();
        for slot in 0..crate::material::MAX_TEXTURES {
            texture_views.push(self.scene.texture_view_for_slot(slot));
            samplers.push(self.scene.texture_sampler_for_slot(slot));
        }

        let mesh_buffers = self.scene.mesh_buffers();
        let dynamic_mesh_buffers = self.scene.dynamic_mesh_buffers();
        if let Ok(mut state) = self.debug_state.lock() {
            state.camera_position = camera.position;
        }
        let rc_radius = self.gi_config.rc_radius;
        let rc_min = [camera.position.x - rc_radius, camera.position.y - rc_radius, camera.position.z - rc_radius];
        let rc_max = [camera.position.x + rc_radius, camera.position.y + rc_radius, camera.position.z + rc_radius];

        // Baked data references — None when bake feature is disabled.
        #[cfg(feature = "bake")]
        let baked_ao = self.baked_data.as_deref().and_then(|d| d.ao_view_ref());
        #[cfg(not(feature = "bake"))]
        let baked_ao = None;
        #[cfg(feature = "bake")]
        let baked_ao_sampler = self.baked_data.as_deref().and_then(|d| d.ao_sampler_ref());
        #[cfg(not(feature = "bake"))]
        let baked_ao_sampler = None;
        #[cfg(feature = "bake")]
        let baked_lightmap = self.baked_data.as_deref().and_then(|d| d.lightmap_view_ref());
        #[cfg(not(feature = "bake"))]
        let baked_lightmap = None;
        #[cfg(feature = "bake")]
        let baked_lightmap_sampler = self.baked_data.as_deref().and_then(|d| d.lightmap_sampler_ref());
        #[cfg(not(feature = "bake"))]
        let baked_lightmap_sampler = None;
        #[cfg(feature = "bake")]
        let baked_reflection = self.baked_data.as_deref().and_then(|d| d.reflection_view_ref());
        #[cfg(not(feature = "bake"))]
        let baked_reflection = None;
        #[cfg(feature = "bake")]
        let baked_reflection_sampler = self.baked_data.as_deref().and_then(|d| d.reflection_sampler_ref());
        #[cfg(not(feature = "bake"))]
        let baked_reflection_sampler = None;
        #[cfg(feature = "bake")]
        let baked_irradiance_sh = self.baked_data.as_deref().and_then(|d| d.irradiance_sh_buf_ref());
        #[cfg(not(feature = "bake"))]
        let baked_irradiance_sh = None;
        #[cfg(feature = "bake")]
        let baked_pvs = self.baked_data.as_deref().and_then(|d| d.pvs_ref());
        #[cfg(not(feature = "bake"))]
        let baked_pvs = None;

        let mut frame_resources = libhelio::FrameResources::empty();
        frame_resources.main_scene.write(
            libhelio::MainSceneResources {
                mesh_buffers: libhelio::MeshBuffers {
                    vertices: mesh_buffers.vertices,
                    indices: mesh_buffers.indices,
                    dynamic_vertices: dynamic_mesh_buffers.vertices,
                    dynamic_indices: dynamic_mesh_buffers.indices,
                },
                material_textures: libhelio::MaterialTextureBindings {
                    material_textures: self.scene.material_texture_buffer(),
                    texture_views: texture_views.as_slice(),
                    samplers: samplers.as_slice(),
                    version: self.scene.texture_binding_version(),
                },
                clear_color: self.clear_color,
                ambient_color: self.ambient_color,
                ambient_intensity: self.ambient_intensity,
                rc_world_min: rc_min,
                rc_world_max: rc_max,
            },
            "Renderer",
        );
        if !self.billboard_scratch.is_empty() {
            frame_resources.billboards.write(
                libhelio::BillboardFrameData {
                    instances: bytemuck::cast_slice(&self.billboard_scratch),
                    count: self.billboard_scratch.len() as u32,
                    generation: self.billboard_generation,
                },
                "Renderer",
            );
        }

        if !self.corona_emitters.is_empty() {
            frame_resources.corona_emitters.write(
                libhelio::CoronaEmitterFrameData {
                    emitters: bytemuck::cast_slice(&self.corona_emitters),
                    count: self.corona_emitters.len() as u32,
                    generation: self.corona_emitter_generation,
                    max_particles: libhelio::CORONA_MAX_PARTICLES,
                },
                "Renderer",
            );
        }
        if water_volume_count > 0 {
            frame_resources.water_volumes.write(&self.water_volumes_buffer, "Renderer");
        }
        frame_resources.water_volume_count = water_volume_count;
        if water_hitbox_count > 0 {
            frame_resources.water_hitboxes.write(&self.water_hitboxes_buffer, "Renderer");
        }
        frame_resources.water_hitbox_count = water_hitbox_count;
        frame_resources.depth_texture.write(&self.depth_texture, "Renderer");
        if let Some(v) = self.full_res_depth_view.as_ref().map(|v| v as &wgpu::TextureView) {
            frame_resources.full_res_depth.write(v, "Renderer");
        }
        if let Some(t) = self.full_res_depth_texture.as_ref().map(|t| t as &wgpu::Texture) {
            frame_resources.full_res_depth_texture.write(t, "Renderer");
        }
        if let Some(vg_data) = self.scene.vg_frame_data() {
            frame_resources.vg.write(vg_data, "Renderer");
        }
        frame_resources.sky = self.scene.sky_context();
        if let Some(ao) = baked_ao {
            frame_resources.baked_ao.write(ao, "Renderer");
        }
        if let Some(ao_sampler) = baked_ao_sampler {
            frame_resources.baked_ao_sampler.write(ao_sampler, "Renderer");
        }
        if let Some(lightmap) = baked_lightmap {
            frame_resources.baked_lightmap.write(lightmap, "Renderer");
        }
        if let Some(lightmap_sampler) = baked_lightmap_sampler {
            frame_resources.baked_lightmap_sampler.write(lightmap_sampler, "Renderer");
        }
        if let Some(reflection) = baked_reflection {
            frame_resources.baked_reflection.write(reflection, "Renderer");
        }
        if let Some(reflection_sampler) = baked_reflection_sampler {
            frame_resources.baked_reflection_sampler.write(reflection_sampler, "Renderer");
        }
        if let Some(irradiance_sh) = baked_irradiance_sh {
            frame_resources.baked_irradiance_sh.write(irradiance_sh, "Renderer");
        }
        if let Some(pvs) = baked_pvs {
            frame_resources.baked_pvs.write(pvs, "Renderer");
        }

        if self.clear_target_next_frame {
            let clear = wgpu::Color {
                r: self.clear_color[0] as f64,
                g: self.clear_color[1] as f64,
                b: self.clear_color[2] as f64,
                a: self.clear_color[3] as f64,
            };
            let mut clear_encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Renderer Resize Target Clear"),
                });
            {
                let _pass = clear_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Renderer Resize Target Clear Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(clear),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
            }
            self.queue.submit(std::iter::once(clear_encoder.finish()));
            self.clear_target_next_frame = false;
        }

        // Populate debug overlay data before executing passes
        {
            if let Ok(mut state) = self.debug_overlay_shared.lock() {
                if state.enabled {
                    state.clear();

                    // Size grid to fill screen
                    let cols = (self.output_width / 14).min(280);
                    let rows = (self.output_height / 24).min(90);
                    state.set_grid_size(cols, rows);
                    let half = cols / 2;
                    let sw = self.output_width as f32;
                    let sh = self.output_height as f32;

                    let debug_data = self.graph.collect_frame_debug_data();
                    let fps = if self.delta_time > 0.0 { (1.0 / self.delta_time) as u32 } else { 0 };
                    let frame_ms = self.delta_time * 1000.0;

                    // ── LEFT SECTION: perf info ──
                    let mut l = 0u32;
                    let other_ms = (frame_ms - self.graph_time_ms).max(0.0);
                    let (timings, total_cpu, total_gpu) = self.graph.profiler().export_timings();
                    state.write_text(0, l, &format!("Helio  FPS: {}  Frame: {:.1} ms  Graph: {:.2} ms  Other: {:.2} ms",
                        fps, frame_ms, self.graph_time_ms, other_ms)); l += 1;
                    l += 1;

                    if !timings.is_empty() {
                        state.write_text(0, l, &format!("  CPU-prepare: {:.2} ms  GPU-compute: {:.2} ms", total_cpu, total_gpu)); l += 1;
                        for pt in &timings {
                            if l >= rows { break; }
                            state.write_text(0, l, &format!("  {:.3}c/{:.3}g ms  {}", pt.cpu_ms, pt.gpu_ms, pt.name));
                            l += 1;
                        }
                        l += 1;
                    }
                    l += 1;

                    state.write_text(0, l, &format!("Graph VRAM: {} KB ({} MB)  Subpass chains: {}",
                        debug_data.total_vram_kb, debug_data.total_vram_kb / 1024, debug_data.subpass_chains.len())); l += 1;
                    for ch in &debug_data.subpass_chains {
                        if l >= rows { break; }
                        state.write_text(0, l, &format!("  {}", ch)); l += 1;
                    }

                    // ── CULLING STATS ──
                    l += 1;
                    if l < rows {
                        let cs = self.cull_stats;
                        let total = cs[0];
                        let frustum = cs[1];
                        let subpixel = cs[2];
                        let frustum_visible = cs[3];
                        let occ_raw = cs[4];
                        // Cap occlusion-culled at frustum_visible (occlusion pass may
                        // process frustum-culled draws due to timing/overlap).
                        let occ = occ_raw.min(frustum_visible);
                        let visible = frustum_visible - occ;
                        let sh_total = cs[5];
                        let sh_visible = cs[6];
                        let sh_occ_raw = cs[7];
                        let sh_occ = sh_occ_raw.min(sh_visible);
                        let sh_frustum = sh_total.saturating_sub(sh_visible + sh_occ);
                        state.write_text(0, l, &format!("── Culling Stats ──────────────────────")); l += 1;
                        state.write_text(0, l, &format!("  Total draws:     {:>6}", total)); l += 1;
                        let pct = |n: u32, d: u32| -> f64 { if d == 0 { 0.0 } else { n as f64 / d as f64 * 100.0 } };
                        state.write_text(0, l, &format!("  Frustum culled:  {:>6}  {:>5.1}%", frustum, pct(frustum, total))); l += 1;
                        state.write_text(0, l, &format!("  Sub-pixel culled:{:>6}  {:>5.1}%", subpixel, pct(subpixel, total))); l += 1;
                        state.write_text(0, l, &format!("  Occlusion culled:{:>6}  {:>5.1}%", occ, pct(occ, total))); l += 1;
                        state.write_text(0, l, &format!("  Visible:         {:>6}  {:>5.1}%", visible, pct(visible, total))); l += 1;
                        l += 1;
                        let sh_vis_final = sh_visible.saturating_sub(sh_occ);
                        state.write_text(0, l, &format!("  Shadow casters:  {:>6}", sh_total)); l += 1;
                        state.write_text(0, l, &format!("    Visible:       {:>6}  {:>5.1}%", sh_vis_final, pct(sh_vis_final, sh_total))); l += 1;
                        state.write_text(0, l, &format!("    Frustum culled:{:>6}  {:>5.1}%", sh_frustum, pct(sh_frustum, sh_total))); l += 1;
                        state.write_text(0, l, &format!("    Occlusion cull:{:>6}  {:>5.1}%", sh_occ, pct(sh_occ, sh_total))); l += 1;
                    }

                    // ── RIGHT SECTION: column-aligned texture table (anchored to right edge) ──
                    let mut table_rows: Vec<Vec<String>> = Vec::new();
                    for res in &debug_data.resources {
                        table_rows.push(vec![
                            res.name.clone(),
                            format!("{}x{}", res.width, res.height),
                            res.layers.to_string(),
                            res.format_name.clone(),
                            format!("{}KB", res.size_kb),
                            res.alias.clone(),
                        ]);
                    }
                    let mut col_widths = vec![4u32; 6];
                    for row in &table_rows {
                        for (i, val) in row.iter().enumerate() {
                            col_widths[i] = col_widths[i].max(val.chars().count() as u32);
                        }
                    }
                    let header = ["name", "size", "layers", "format", "KB", "alias"];
                    for (i, h) in header.iter().enumerate() {
                        col_widths[i] = col_widths[i].max(h.chars().count() as u32);
                    }
                    let total_table_w: u32 = col_widths.iter().sum::<u32>() + (col_widths.len() as u32 - 1);
                    let right_x = cols.saturating_sub(total_table_w);

                    let mut t = 0u32;
                    let mut x = right_x;
                    for (i, h) in header.iter().enumerate() {
                        state.write_text(x, t, h);
                        x += col_widths[i] + 1;
                    }
                    t += 1;
                    let mut sep = String::new();
                    for w in &col_widths { for _ in 0..*w { sep.push('-'); } sep.push(' '); }
                    state.write_text(right_x, t, &sep); t += 1;

                    for row in &table_rows {
                        if t >= rows { break; }
                        let mut x = right_x;
                        for (i, val) in row.iter().enumerate() {
                            let w = col_widths[i] as usize;
                            let display: String = val.chars().take(w).collect();
                            state.write_text(x, t, &display);
                            x += col_widths[i] + 1;
                        }
                        t += 1;
                    }

                    // ── Pass pipeline (right-anchored table) ──
                    t += 1;
                    if t < rows {
                        let mut pass_rows: Vec<Vec<String>> = Vec::new();
                        for pi in &debug_data.passes {
                            if pi.index == 999 { continue; }
                            let ws = if pi.writes.is_empty() { String::new() } else { pi.writes.join(", ") };
                            pass_rows.push(vec![
                                pi.index.to_string(),
                                pi.kind.clone(),
                                pi.chain_marker.clone(),
                                pi.name.clone(),
                                ws,
                            ]);
                        }
                        let mut pw = vec![2u32, 1, 6, 12, 0];
                        for row in &pass_rows {
                            for (i, val) in row.iter().enumerate() {
                                pw[i] = pw[i].max(val.chars().count() as u32);
                            }
                        }
                        for (i, h) in ["#", "", "chain", "pass", "writes"].iter().enumerate() {
                            pw[i] = pw[i].max(h.chars().count() as u32);
                        }
                        let pass_total: u32 = pw.iter().sum::<u32>() + (pw.len() as u32 - 1);
                        let pass_x = cols.saturating_sub(pass_total);

                        state.write_text(pass_x, t, "Pass pipeline:"); t += 1;
                        let mut px = pass_x;
                        for (i, h) in ["#", "", "chain", "pass", "writes"].iter().enumerate() {
                            state.write_text(px, t, h);
                            px += pw[i] + 1;
                        }
                        t += 1;

                        for row in &pass_rows {
                            if t >= rows { break; }
                            let mut px = pass_x;
                            for (i, val) in row.iter().enumerate() {
                                let display: String = val.chars().take(pw[i] as usize).collect();
                                state.write_text(px, t, &display);
                                px += pw[i] + 1;
                            }
                            t += 1;
                        }
                    }

                    // ── LOWER CHARTS: FPS graph (left) + VRAM pie (right) ──
                    let chart_y = sh - 150.0;
                    let graph_w = 220.0;
                    let graph_h = 110.0;
                    let graph_x = 10.0;
                    let pie_r = 80.0;
                    let pie_cx = sw - pie_r - 60.0;
                    let pie_cy = chart_y + graph_h * 0.5;

                    // ── FPS GRAPH ──
                    let num_samples = self.frame_times.len();
                    let bar_w = graph_w / num_samples as f32;
                    let max_dt = 0.05;

                    for (ms, y_frac, label) in [(0.050, 0.0, "50ms"), (0.033, 0.34, "33ms"), (0.016, 0.66, "16ms")] {
                        let dy = chart_y + graph_h * (1.0 - y_frac);
                        state.add_bar(graph_x, dy, graph_w, 1.0, 0.5, 0.5, 0.5, 0.5);
                        let lcol = ((graph_x + graph_w + 4.0) / 8.0) as u32;
                        let lrow = ((dy - 5.0) / 12.0) as u32;
                        if lcol < state.small_cols() && lrow < state.small_rows() {
                            state.write_small(lcol, lrow, label);
                        }
                    }

                    for (i, &ft) in self.frame_times.iter().enumerate() {
                        let bar_h = (ft / max_dt * graph_h).min(graph_h);
                        let bx = graph_x + i as f32 * bar_w;
                        let by = chart_y + graph_h - bar_h;
                        let color = if ft < 0.016 { (0.3, 0.8, 0.3, 0.8) }
                                     else if ft < 0.033 { (0.9, 0.9, 0.2, 0.8) }
                                     else { (0.9, 0.3, 0.3, 0.8) };
                        state.add_bar(bx, by, bar_w.max(2.0), bar_h.max(1.0), color.0, color.1, color.2, color.3);
                    }

                    // ── VRAM PIE CHART ──
                    if debug_data.total_vram_kb > 0 {
                        let vram_total = debug_data.total_vram_kb as f32;
                        let mut angle = 0.0f32;
                        let pie_colors = [
                            (0.3, 0.6, 1.0, 0.9), (0.3, 1.0, 0.6, 0.9),
                            (1.0, 0.6, 0.3, 0.9), (1.0, 0.3, 0.6, 0.9),
                            (0.6, 0.3, 1.0, 0.9), (0.6, 1.0, 0.3, 0.9),
                        ];

                        for (i, res) in debug_data.resources.iter().enumerate() {
                            let frac = res.size_kb as f32 / vram_total;
                            let end = angle + frac * std::f32::consts::TAU;
                            let ci = pie_colors[i % pie_colors.len()];
                            state.add_pie_slice(pie_cx, pie_cy, pie_r, end, ci.0, ci.1, ci.2, ci.3);

                            let mid = angle + frac * std::f32::consts::PI;
                            let edge_x = pie_cx + mid.cos() * pie_r;
                            let edge_y = pie_cy + mid.sin() * pie_r;
                            let pct = (frac * 100.0) as u32;
                            let label = format!("{} {}%", res.name, pct);
                            let lw = label.chars().count() as u32;

                            // Dynamic gap: right-aligned text (extends toward pie from tip) needs gap >= label width
                            let prefer_left = mid.cos() >= 0.0;
                            let min_gap = if !prefer_left { (lw as f32 + 2.0) * 8.0 } else { 20.0 };
                            let gap = min_gap.max(20.0).min(200.0);
                            let lx = pie_cx + mid.cos() * (pie_r + gap);
                            let ly = pie_cy + mid.sin() * (pie_r + gap);
                            state.add_line(edge_x, edge_y, lx, ly, 1.0, 1.0, 1.0, 0.7);

                            let sm_cols = state.small_cols();
                            let sm_rows = state.small_rows();
                            let lrow = ((ly - 4.0) / 12.0) as u32;

                            // Auto-anchor: prefer outward from pie center, fallback to opposite if off-screen
                            let tip_col = lx / 8.0;
                            let left_col = tip_col + 1.0;
                            let right_col = tip_col - lw as f32 - 1.0;
                            let left_ok = left_col + lw as f32 <= sm_cols as f32;
                            let right_ok = right_col >= 0.0;

                            let lcol = if prefer_left && left_ok {
                                left_col as u32
                            } else if !prefer_left && right_ok {
                                right_col as u32
                            } else if left_ok {
                                left_col as u32
                            } else if right_ok {
                                right_col as u32
                            } else {
                                0u32
                            };
                            if lrow < sm_rows {
                                let max_w = sm_cols.saturating_sub(lcol);
                                let truncated: String = label.chars().take(max_w as usize).collect();
                                state.write_small(lcol, lrow, &truncated);
                            }
                            angle = end;
                        }
                    }
                }
            }
        }

        // Clear culling stats before graph execution (zero out all 8 atomic counters).
        {
            let mut clear_encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("CullStats Clear"),
            });
            clear_encoder.clear_buffer(&self.cull_stats_buffer, 0, Some(32));
            self.queue.submit(std::iter::once(clear_encoder.finish()));
        }

        let _graph_start = Instant::now();
        self.graph.execute_with_frame_resources(
            self.scene.gpu_scene(),
            target,
            &self.depth_view,
            &frame_resources,
        )?;
        self.graph_time_ms = _graph_start.elapsed().as_secs_f64() as f32 * 1000.0;

        // Read back culling stats: copy GPU buffer → staging, then map + read.
        {
            let mut read_encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("CullStats Readback"),
            });
            read_encoder.copy_buffer_to_buffer(
                &self.cull_stats_buffer, 0,
                &self.cull_stats_staging, 0,
                32,
            );
            self.queue.submit(std::iter::once(read_encoder.finish()));
        }

        // Poll and map the staging buffer to read the 8 counters.
        // Skip when the renderer doesn't own the device (can't poll without stalling).
        if self.owns_device {
            let staging_slice = self.cull_stats_staging.slice(..);
            staging_slice.map_async(wgpu::MapMode::Read, |_| {});
            self.device.poll(wgpu::PollType::wait_indefinitely());
            {
                let mapped = staging_slice.get_mapped_range();
                if mapped.len() >= 32 {
                    let ptr = mapped.as_ptr() as *const u32;
                    self.cull_stats = unsafe { std::ptr::read_unaligned(ptr.cast()) };
                }
                drop(mapped);
            }
            self.cull_stats_staging.unmap();
        }

        // Send profiling data to live portal (non-blocking)
        #[cfg(feature = "live-portal")]
        if let Some(ref portal) = self.portal_handle {
            let (pass_timings, total_cpu_ms, total_gpu_ms) = self.graph.profiler().export_timings();

            // Convert helio_v3::PassTiming to helio_live_portal::PortalPassTiming
            let portal_timings: Vec<_> = pass_timings.iter().map(|pt| {
                helio_live_portal::PortalPassTiming {
                    name: pt.name.clone(),
                    gpu_ms: pt.gpu_ms,
                    cpu_ms: pt.cpu_ms,
                }
            }).collect();

            // Create stage timing hierarchy for node graph visualization
            // Top-level stage represents the entire GPU render pipeline
            let render_stage_id = "gpu_render";
            let stage_timings = vec![
                helio_live_portal::PortalStageTiming {
                    id: render_stage_id.to_string(),
                    name: "GPU Render".to_string(),
                    ms: total_gpu_ms,
                    children: Vec::new(),
                }
            ];

            let snapshot = helio_live_portal::PortalFrameSnapshot {
                frame: self.scene.gpu_scene().frame_count,
                timestamp_ms: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis(),
                frame_time_ms: dt * 1000.0,
                frame_to_frame_ms: dt * 1000.0,
                total_gpu_ms,
                total_cpu_ms,
                pass_timings: portal_timings,
                pipeline_order: pass_timings.iter().map(|pt| pt.name.clone()).collect(),
                pipeline_stage_id: Some(render_stage_id.to_string()),
                scene_delta: None,
                object_count: self.scene.gpu_scene().instances.len(),
                light_count: self.scene.gpu_scene().lights.len(),
                billboard_count: self.billboard_instances.len(),
                draw_calls: helio_live_portal::DrawCallMetrics {
                    total: self.scene.gpu_scene().resources().draw_count as usize,
                    opaque: self.scene.gpu_scene().resources().draw_count as usize,
                    transparent: 0,
                },
                mesh_stats: {
                    let (verts, tris, meshes) = self.scene.mesh_stats();
                    let (drawn_verts, drawn_tris) = self.scene.drawn_mesh_stats();
                    helio_live_portal::MeshStats {
                        total_vertices: verts,
                        total_triangles: tris,
                        unique_meshes: meshes,
                        drawn_vertices: drawn_verts,
                        drawn_triangles: drawn_tris,
                    }
                },
                stage_timings,
            };

            portal.publish(snapshot);
        }

        // Explicit drops needed: both ArrayVecs borrow from `self.scene` and must
        // be released before `advance_frame()` takes `&mut self.scene`.
        drop(texture_views);
        drop(samplers);
        self.scene.advance_frame();
        Ok(())
    }

    /// Start the live performance portal web dashboard on http://127.0.0.1:3030
    ///
    /// **Note**: When the `live-portal` feature is enabled (default), the dashboard
    /// starts automatically during `Renderer::new()`. You only need to call this manually
    /// if you want to restart the server or handle startup errors differently.
    ///
    /// Once started, profiling data will automatically be sent to the web UI
    /// on every frame without blocking the render thread.
    ///
    /// Returns the URL on success.
    #[cfg(feature = "live-portal")]
    pub fn start_live_portal_default(&mut self) -> std::io::Result<String> {
        let handle = helio_live_portal::start_live_portal("127.0.0.1:3030")?;
        let url = handle.url.clone();
        self.portal_handle = Some(handle);
        Ok(url)
    }

    /// Enter true DXGI exclusive fullscreen, bypassing DWM composition entirely.
    ///
    /// # Platform behaviour
    ///
    /// | Platform | Backend | Effect |
    /// |---|---|---|
    /// | Windows | DX12 | 1. Calls `MakeWindowAssociation(hwnd, 0)` to lift DXGI's default Alt-Enter / window-change locks. 2. Obtains the `IDXGISwapChain3` from the HAL surface and calls `SetFullscreenState(TRUE, None)` to enter hardware exclusive mode, bypassing DWM. |
    /// | Windows | Vulkan | Logs a warning — `VK_EXT_full_screen_exclusive` must be requested at instance-creation time. |
    /// | Non-Windows | any | Logs an unsupported warning. |
    ///
    /// Call `exit_exclusive_fullscreen` before the window is destroyed or
    /// minimised, as DXGI requires `SetFullscreenState(FALSE)` before the
    /// swap chain is released.
    ///
    /// Returns `true` when all platform calls succeeded.
    ///
    /// # Safety
    /// - `raw_hwnd` must be a valid, live `HWND` for the window this renderer
    ///   is presenting to.
    /// - `surface` must be the `wgpu::Surface` associated with that window and
    ///   must already be configured (i.e. `surface.configure()` has been called).
    pub unsafe fn request_exclusive_fullscreen(
        &self,
        surface: &wgpu::Surface<'_>,
        raw_hwnd: *mut std::ffi::c_void,
    ) -> bool {
        #[cfg(not(target_os = "windows"))]
        {
            let _ = (surface, raw_hwnd);
            log::warn!("helio: request_exclusive_fullscreen is not supported on this platform");
            return false;
        }

        #[cfg(target_os = "windows")]
        exclusive_fullscreen_win(&self.device, surface, raw_hwnd)
    }

    /// Exit DXGI exclusive fullscreen.
    ///
    /// Must be called before the window is closed or minimised when exclusive
    /// fullscreen is active.  Safe to call even if not currently in exclusive
    /// fullscreen — DXGI will simply no-op.
    ///
    /// # Safety
    /// `surface` must be the `wgpu::Surface` that was passed to
    /// `request_exclusive_fullscreen`.
    pub unsafe fn exit_exclusive_fullscreen(&self, surface: &wgpu::Surface<'_>) {
        #[cfg(not(target_os = "windows"))]
        {
            let _ = surface;
        }

        #[cfg(target_os = "windows")]
        exit_exclusive_fullscreen_win(surface);
    }
}

// ── Exclusive fullscreen — Windows DX12 / Vulkan ─────────────────────────────

/// Enter DXGI exclusive fullscreen on Windows.
///
/// Steps:
/// 1. `MakeWindowAssociation(hwnd, 0)` — lifts DXGI's default Alt-Enter /
///    window-change locks so the driver is *permitted* to flip.
/// 2. `IDXGISwapChain::SetFullscreenState(TRUE, None)` — actually transitions
///    the display into exclusive mode, bypassing DWM entirely.
#[cfg(target_os = "windows")]
fn exclusive_fullscreen_win(
    device: &wgpu::Device,
    surface: &wgpu::Surface<'_>,
    raw_hwnd: *mut std::ffi::c_void,
) -> bool {
    use windows::Win32::{
        Foundation::HWND,
        Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1, DXGI_MWA_FLAGS},
    };

    let hwnd = HWND(raw_hwnd);

    // ── DX12 path ────────────────────────────────────────────────────────────
    let is_dx12 = unsafe { device.as_hal::<wgpu::hal::api::Dx12>() }.is_some();
    if is_dx12 {
        let result: windows::core::Result<()> = (|| unsafe {
            // Step 1: lift DXGI window-association restrictions.
            // CreateDXGIFactory1 is correct here — MakeWindowAssociation is
            // effective regardless of which factory owns the swap chain (MSDN).
            let factory: IDXGIFactory1 = CreateDXGIFactory1()?;
            factory.MakeWindowAssociation(hwnd, DXGI_MWA_FLAGS(0))?;

            // Step 2: call SetFullscreenState(TRUE) on the actual swap chain.
            // This is the call that makes the transition hardware-exclusive.
            let swap_chain = surface
                .as_hal::<wgpu::hal::api::Dx12>()
                .and_then(|s| s.swap_chain())
                .ok_or_else(|| windows::core::Error::from(
                    windows::Win32::Foundation::E_FAIL,
                ))?;
            // None → let DXGI pick the output (current monitor).
            swap_chain.SetFullscreenState(true, None)
        })();
        return match result {
            Ok(()) => true,
            Err(e) => {
                log::warn!("helio: DX12 exclusive fullscreen failed: {e}");
                false
            }
        };
    }

    // ── Vulkan path ──────────────────────────────────────────────────────────
    let is_vulkan = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }.is_some();
    if is_vulkan {
        log::warn!(
            "helio: exclusive fullscreen on Vulkan requires VK_EXT_full_screen_exclusive \
             to be enabled at instance-creation time; it cannot be activated post-hoc"
        );
        return false;
    }

    log::warn!("helio: request_exclusive_fullscreen: unrecognised or unsupported backend");
    false
}

/// Exit DXGI exclusive fullscreen on Windows.
///
/// Calls `SetFullscreenState(FALSE)` on the swap chain.
/// DXGI requires this before the swap chain is destroyed.
#[cfg(target_os = "windows")]
fn exit_exclusive_fullscreen_win(surface: &wgpu::Surface<'_>) {
    unsafe {
        if let Some(swap_chain) = surface
            .as_hal::<wgpu::hal::api::Dx12>()
            .and_then(|s| s.swap_chain())
        {
            if let Err(e) = swap_chain.SetFullscreenState(false, None) {
                log::warn!("helio: exit_exclusive_fullscreen: SetFullscreenState(FALSE) failed: {e}");
            }
        }
    }
}
