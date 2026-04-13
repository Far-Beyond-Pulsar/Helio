use std::sync::{Arc, Mutex};

use arrayvec::ArrayVec;
use helio_pass_debug::{DebugVertex};
use helio_pass_deferred_light::DeferredLightPass;
use helio_pass_perf_overlay::{PerfOverlayMode, PerfOverlayPass};
use helio_v3::{RenderGraph, RenderPass, Result as HelioResult};
use helio_pass_debug::DebugCameraUniform;
const MAX_TEXTURES: usize = crate::material::MAX_TEXTURES;

use crate::groups::GroupId;
use crate::mesh::MeshBuffers;
use crate::scene::{Camera, Scene};

use super::config::{GiConfig, RendererConfig};
use super::debug::{DebugDrawPass, DebugDrawState};
use super::graph::{build_default_graph, build_simple_graph, create_depth_resources};

type CustomGraphBuilder = Arc<dyn Fn(&Arc<wgpu::Device>, &Arc<wgpu::Queue>, &Scene, RendererConfig, Arc<Mutex<DebugDrawState>>, &wgpu::Buffer) -> RenderGraph + Send + Sync>;

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
    water_volumes_buffer: wgpu::Buffer,
    water_hitboxes_buffer: wgpu::Buffer,
    /// Instant of the previous `render()` call, used to compute real `delta_time`.
    last_render_time: std::time::Instant,
    /// Precomputed TAA jitter translation matrices (16-sample Halton sequence).
    /// Cached per resolution to avoid per-frame matrix construction overhead.
    jitter_matrices: [glam::Mat4; 16],
    /// Cached resolution for jitter matrix precomputation.
    jitter_cache_width: u32,
    jitter_cache_height: u32,
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
}

enum GraphKind {
    Default,
    Simple,
    Custom,
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
        let mut scene = Scene::new(device.clone(), queue.clone());
        scene.set_render_size(config.width, config.height);

        let debug_state = Arc::new(Mutex::new(DebugDrawState::default()));

        let debug_camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Camera Buffer"),
            size: std::mem::size_of::<DebugCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let graph = build_default_graph(&device, &queue, &scene, config, debug_state.clone(), &debug_camera_buffer);

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
            billboard_instances: Vec::new(),
            billboard_scratch: Vec::new(),
            billboard_dirty: true,
            billboard_cached_light_count: usize::MAX,
            billboard_cached_light_gen: u64::MAX,
            billboard_cached_editor_hidden: false,
            water_volumes_buffer,
            water_hitboxes_buffer,
            last_render_time: std::time::Instant::now(),
            jitter_matrices,
            jitter_cache_width: internal_w,
            jitter_cache_height: internal_h,
            #[cfg(feature = "live-portal")]
            portal_handle: None,

            #[cfg(feature = "bake")]
            bake_pending: None,
            #[cfg(feature = "bake")]
            baked_data: None,
            owns_device: true,
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

        let graph = super::graph::build_default_graph_external(
            &device,
            &queue,
            &scene,
            config,
            debug_state.clone(),
            &debug_camera_buffer,
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
            billboard_instances: Vec::new(),
            billboard_scratch: Vec::new(),
            billboard_dirty: true,
            billboard_cached_light_count: usize::MAX,
            billboard_cached_light_gen: u64::MAX,
            billboard_cached_editor_hidden: false,
            water_volumes_buffer,
            water_hitboxes_buffer,
            last_render_time: std::time::Instant::now(),
            jitter_matrices,
            jitter_cache_width: internal_w,
            jitter_cache_height: internal_h,
            #[cfg(feature = "live-portal")]
            portal_handle: None,

            #[cfg(feature = "bake")]
            bake_pending: None,
            #[cfg(feature = "bake")]
            baked_data: None,
            owns_device: false,
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
        }
    }

    pub fn set_perf_overlay_mode(&mut self, mode: PerfOverlayMode) {
        self.perf_overlay_mode = mode;
        if matches!(self.graph_kind, GraphKind::Default) {
            if let Some(pass) = self.graph.find_pass_mut::<PerfOverlayPass>() {
                pass.set_mode(mode);
            }
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

    pub fn debug_clear(&mut self) {
        if let Ok(mut s) = self.debug_state.lock() {
            s.user_lines.clear();
            s.user_tris.clear();
        }
    }

    pub fn debug_line(&mut self, from: [f32; 3], to: [f32; 3], color: [f32; 4]) {
        if let Ok(mut s) = self.debug_state.lock() {
            s.user_lines.push(DebugVertex { position: from, _pad: 0.0, color });
            s.user_lines.push(DebugVertex { position: to, _pad: 0.0, color });
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

    pub fn camera_buffer(&self) -> &wgpu::Buffer {
        self.scene.gpu_scene().camera.buffer()
    }

    pub fn mesh_buffers(&self) -> MeshBuffers<'_> {
        self.scene.mesh_buffers()
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

    pub fn set_render_size(&mut self, width: u32, height: u32) {
        self.output_width = width;
        self.output_height = height;
        self.scene.set_render_size(width, height);
        let config = RendererConfig {
            width,
            height,
            surface_format: self.surface_format,
            gi_config: self.gi_config,
            shadow_quality: self.shadow_quality,
            debug_mode: self.debug_mode,
            render_scale: self.render_scale,
            perf_overlay_mode: self.perf_overlay_mode,
        };
        let (depth_texture, depth_view) = create_depth_resources(
            &self.device,
            config.internal_width(),
            config.internal_height(),
        );
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
        if self.render_scale < 1.0 {
            let (t, v) = create_depth_resources(&self.device, width, height);
            self.full_res_depth_texture = Some(t);
            self.full_res_depth_view = Some(v);
        } else {
            self.full_res_depth_texture = None;
            self.full_res_depth_view = None;
        }

        match self.graph_kind {
            GraphKind::Default => {
                self.graph = if self.owns_device {
                    build_default_graph(
                        &self.device,
                        &self.queue,
                        &self.scene,
                        config,
                        self.debug_state.clone(),
                        &self.debug_camera_buffer,
                    )
                } else {
                    super::graph::build_default_graph_external(
                        &self.device,
                        &self.queue,
                        &self.scene,
                        config,
                        self.debug_state.clone(),
                        &self.debug_camera_buffer,
                    )
                };
                // The new graph contains a fresh WaterSimPass with default (no-wind)
                // settings. Re-dirty the water volumes so the next frame re-applies
                // the descriptor's wind/sim params to the new pass.
                self.scene.mark_water_volumes_dirty();
            }
            GraphKind::Simple => {
                self.graph = build_simple_graph(&self.device, &self.queue, self.surface_format);
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
                        );
                        self.custom_graph_config = Some(new_cfg);
                        // Same as Default: re-dirty so the new pass gets wind params.
                        self.scene.mark_water_volumes_dirty();
                    } else {
                        self.graph.set_render_size(width, height);
                    }
                } else {
                    self.graph.set_render_size(width, height);
                }
            }
        }
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
        };
        self.graph = if self.owns_device {
            build_default_graph(
                &self.device,
                &self.queue,
                &self.scene,
                config,
                self.debug_state.clone(),
                &self.debug_camera_buffer,
            )
        } else {
            super::graph::build_default_graph_external(
                &self.device,
                &self.queue,
                &self.scene,
                config,
                self.debug_state.clone(),
                &self.debug_camera_buffer,
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

    pub fn render(&mut self, camera: &Camera, target: &wgpu::TextureView) -> HelioResult<()> {
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
            
            let bake_start = std::time::Instant::now();
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
        let now = std::time::Instant::now();
        let dt = now.duration_since(self.last_render_time).as_secs_f32().min(0.1);
        self.last_render_time = now;
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
        if self.billboard_dirty
            || light_count != self.billboard_cached_light_count
            || light_gen != self.billboard_cached_light_gen
            || editor_hidden != self.billboard_cached_editor_hidden
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
            }
            self.billboard_dirty = false;
            self.billboard_cached_light_count = light_count;
            self.billboard_cached_light_gen = light_gen;
            self.billboard_cached_editor_hidden = editor_hidden;
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

        let frame_resources = libhelio::FrameResources {
            gbuffer: None,
            gbuffer_lightmap_uv: None,
            shadow_atlas: None,
            static_shadow_atlas: None,
            shadow_sampler: None,
            hiz: None,
            hiz_sampler: None,
            static_hiz: None,
            static_hiz_sampler: None,
            sky_lut: None,
            sky_lut_sampler: None,
            ssao: None,
            pre_aa: None,
            main_scene: Some(libhelio::MainSceneResources {
                mesh_buffers: libhelio::MeshBuffers {
                    vertices: mesh_buffers.vertices,
                    indices: mesh_buffers.indices,
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
            }),
            tile_light_lists: None,
            tile_light_counts: None,
            full_res_depth: self.full_res_depth_view.as_ref().map(|v| v as &wgpu::TextureView),
            full_res_depth_texture: self.full_res_depth_texture.as_ref().map(|t| t as &wgpu::Texture),
            sky: self.scene.sky_context(),
            billboards: if self.billboard_scratch.is_empty() {
                None
            } else {
                Some(libhelio::BillboardFrameData {
                    instances: bytemuck::cast_slice(&self.billboard_scratch),
                    count: self.billboard_scratch.len() as u32,
                })
            },
            vg: self.scene.vg_frame_data(),
            water_caustics: None,
            water_volumes: if water_volume_count > 0 {
                Some(&self.water_volumes_buffer)
            } else {
                None
            },
            water_volume_count,
            water_sim_texture: None,
            water_sim_sampler: None,
            water_hitboxes: if water_hitbox_count > 0 {
                Some(&self.water_hitboxes_buffer)
            } else {
                None
            },
            water_hitbox_count,
            depth_texture: Some(&self.depth_texture),
            // Baked data — populated after configure_bake() completes.
            baked_ao,
            baked_ao_sampler,
            baked_lightmap,
            baked_lightmap_sampler,
            baked_reflection,
            baked_reflection_sampler,
            baked_irradiance_sh,
            baked_pvs,
        };

        self.graph.execute_with_frame_resources(
            self.scene.gpu_scene(),
            target,
            &self.depth_view,
            &frame_resources,
        )?;

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
