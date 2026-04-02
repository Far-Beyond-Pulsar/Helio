use std::sync::{Arc, Mutex};

use arrayvec::ArrayVec;
use helio_pass_debug::{DebugVertex};
use helio_pass_deferred_light::DeferredLightPass;
use helio_v3::{RenderGraph, RenderPass, Result as HelioResult};
use helio_pass_debug::DebugCameraUniform;
const MAX_TEXTURES: usize = crate::material::MAX_TEXTURES;

use crate::groups::GroupId;
use crate::mesh::MeshBuffers;
use crate::scene::{Camera, Scene};

use super::config::{GiConfig, RendererConfig};
use super::debug::{DebugDrawPass, DebugDrawState};
use super::graph::{build_default_graph, build_simple_graph, create_depth_resources};

type CustomGraphBuilder = Arc<dyn Fn(&Arc<wgpu::Device>, &Arc<wgpu::Queue>, &Scene, RendererConfig, Arc<Mutex<DebugDrawState>>, &wgpu::Buffer, bool) -> RenderGraph + Send + Sync>;

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
    debug_depth_test: bool,
    editor_mode: bool,
    custom_graph_builder: Option<CustomGraphBuilder>,
    custom_graph_config: Option<RendererConfig>,
    debug_state: Arc<Mutex<DebugDrawState>>,
    billboard_instances: Vec<helio_pass_billboard::BillboardInstance>,
    billboard_scratch: Vec<helio_pass_billboard::BillboardInstance>,
    water_volumes_buffer: wgpu::Buffer,
    water_hitboxes_buffer: wgpu::Buffer,
    /// Instant of the previous `render()` call, used to compute real `delta_time`.
    last_render_time: std::time::Instant,
}

enum GraphKind {
    Default,
    Simple,
    Custom,
}

impl Renderer {
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

        let graph = build_default_graph(&device, &queue, &scene, config, debug_state.clone(), &debug_camera_buffer, true);

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
            debug_mode: 0,
            debug_depth_test: true,
            editor_mode: false,
            custom_graph_builder: None,
            custom_graph_config: None,
            debug_state,
            billboard_instances: Vec::new(),
            billboard_scratch: Vec::new(),
            water_volumes_buffer,
            water_hitboxes_buffer,
            last_render_time: std::time::Instant::now(),
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
        }
    }

    pub fn debug_line(&mut self, from: [f32; 3], to: [f32; 3], color: [f32; 4]) {
        if let Ok(mut s) = self.debug_state.lock() {
            s.user_lines.push(DebugVertex { position: from, _pad: 0.0, color });
            s.user_lines.push(DebugVertex { position: to, _pad: 0.0, color });
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
        let up = if dir.abs_diff_eq(glam::Vec3::Y, 1e-5) { glam::Vec3::X } else { glam::Vec3::Y };
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
                self.graph = build_default_graph(
                    &self.device,
                    &self.queue,
                    &self.scene,
                    config,
                    self.debug_state.clone(),
                    &self.debug_camera_buffer,
                    self.debug_depth_test,
                );
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
                            self.debug_depth_test,
                        );
                        self.custom_graph_config = Some(new_cfg);
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
        };
        self.graph = build_default_graph(
            &self.device,
            &self.queue,
            &self.scene,
            config,
            self.debug_state.clone(),
            &self.debug_camera_buffer,
            self.debug_depth_test,
        );
        self.graph_kind = GraphKind::Default;
    }

    pub fn optimize_scene_layout(&mut self) {
        self.scene.optimize_scene_layout();
    }

    pub fn set_billboard_instances(&mut self, instances: &[helio_pass_billboard::BillboardInstance]) {
        self.billboard_instances.clear();
        self.billboard_instances.extend_from_slice(instances);
    }

    pub fn render(&mut self, camera: &Camera, target: &wgpu::TextureView) -> HelioResult<()> {
        // Compute real frame delta, capped at 100 ms to avoid spiral-of-death on
        // slow frames (e.g. first frame, window unfocus/refocus, GPU stalls).
        let now = std::time::Instant::now();
        let dt = now.duration_since(self.last_render_time).as_secs_f32().min(0.1);
        self.last_render_time = now;
        self.graph.set_delta_time(dt);

        let frame_idx = self.scene.gpu_scene().frame_count;
        let raw = HALTON_JITTER[(frame_idx % 16) as usize];
        let internal_w = (((self.output_width as f32) * self.render_scale).ceil() as u32).max(1);
        let internal_h = (((self.output_height as f32) * self.render_scale).ceil() as u32).max(1);
        let jx = ((raw[0] - 0.5) * 2.0) / (internal_w as f32);
        let jy = ((raw[1] - 0.5) * 2.0) / (internal_h as f32);
        let jitter_mat = glam::Mat4::from_translation(glam::Vec3::new(jx, jy, 0.0));
        let jittered_m = jitter_mat * camera.proj * camera.view;
        let col = jittered_m.to_cols_array();
        let debug_camera_uniform = DebugCameraUniform {
            view_proj: [
                [col[0], col[1], col[2], col[3]],
                [col[4], col[5], col[6], col[7]],
                [col[8], col[9], col[10], col[11]],
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

        self.billboard_scratch.clear();
        self.billboard_scratch.extend_from_slice(&self.billboard_instances);
        if !self.scene.is_group_hidden(GroupId::EDITOR) {
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

        // Upload water volumes to GPU only when the descriptor has changed.
        // get_water_volumes_gpu() allocates a Vec; skipping it at steady state
        // eliminates the per-frame heap allocation and GPU write.
        // NOTE: must happen before the `texture_views` ArrayVec is built, since
        // clear_water_volumes_dirty() requires `&mut self.scene` and cannot
        // coexist with the immutable borrows held by that ArrayVec.
        let water_volume_count = self.scene.water_volumes_count();
        if water_volume_count > 0 && self.scene.water_volumes_dirty() {
            let water_volumes = self.scene.get_water_volumes_gpu();
            // Bridge descriptor sim/wind params into WaterSimPass so the update
            // shader sees them. The descriptor is the source of truth — the pass's
            // own fields are updated when the volume descriptor changes.
            if let Some(pass) = self.graph.find_pass_mut::<helio_pass_water_sim::WaterSimPass>() {
                let vol = &water_volumes[0];
                pass.set_sim_dynamics(vol.sim_dynamics[0], vol.sim_dynamics[1]);
                pass.set_wave_scale(vol.sim_dynamics[2]);
                pass.set_wave_speed(vol.wave_params[2]);
                pass.set_wind([vol.wind_params[0], vol.wind_params[1]], vol.wind_params[2]);
            }
            self.queue.write_buffer(
                &self.water_volumes_buffer,
                0,
                bytemuck::cast_slice(&water_volumes),
            );
            self.scene.clear_water_volumes_dirty();
        }

        // Upload water hitboxes to GPU only when they have changed.
        let water_hitbox_count = self.scene.water_hitboxes_count();
        if water_hitbox_count > 0 && self.scene.water_hitboxes_dirty() {
            let water_hitboxes = self.scene.get_water_hitboxes_gpu();
            self.queue.write_buffer(
                &self.water_hitboxes_buffer,
                0,
                bytemuck::cast_slice(&water_hitboxes),
            );
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

        let frame_resources = libhelio::FrameResources {
            gbuffer: None,
            shadow_atlas: None,
            shadow_sampler: None,
            hiz: None,
            hiz_sampler: None,
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
        };

        self.graph.execute_with_frame_resources(
            self.scene.gpu_scene(),
            target,
            &self.depth_view,
            &frame_resources,
        )?;

        // Explicit drops needed: both ArrayVecs borrow from `self.scene` and must
        // be released before `advance_frame()` takes `&mut self.scene`.
        drop(texture_views);
        drop(samplers);
        self.scene.advance_frame();
        Ok(())
    }

    /// Start the live performance portal web dashboard on http://127.0.0.1:3030
    /// Returns the URL on success.
    #[cfg(feature = "live-portal")]
    pub fn start_live_portal_default(&mut self) -> std::io::Result<String> {
        let handle = helio_live_portal::start_live_portal("127.0.0.1:3030")?;
        let url = handle.url.clone();
        // TODO: Store handle and integrate frame snapshots
        Ok(url)
    }
}
