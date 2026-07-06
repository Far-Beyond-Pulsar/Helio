use std::sync::{Arc, Mutex};

#[cfg(target_arch = "wasm32")]
use web_time::Instant;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use helio_pass_debug::DebugVertex;
use helio_pass_debug_overlay::DebugOverlayState;
use helio_pass_deferred_light::DeferredLightPass;
use helio_pass_perf_overlay::{PerfOverlayMode, PerfOverlayPass};
use helio_pass_virtual_geometry::VirtualGeometryPass;
use helio_v3::{RenderGraph, RenderPass};

use crate::groups::GroupId;
use crate::mesh::MeshBuffers;
use crate::scene::Scene;

use super::config::{GiConfig, RendererConfig};
use super::debug::DebugDrawState;
use super::graph::{build_default_graph, build_default_graph_external, build_simple_graph};

type CustomGraphBuilder = Arc<dyn Fn(&Arc<wgpu::Device>, &Arc<wgpu::Queue>, &Scene, RendererConfig, Arc<Mutex<DebugDrawState>>, &wgpu::Buffer, &wgpu::Buffer, Option<&Arc<Mutex<DebugOverlayState>>>) -> RenderGraph + Send + Sync>;

pub(crate) const HALTON_JITTER: [[f32; 2]; 16] = [
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
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    pub(crate) graph: RenderGraph,
    pub(crate) graph_kind: GraphKind,
    pub(crate) scene: Scene,
    pub(crate) depth_texture: wgpu::Texture,
    pub(crate) depth_view: wgpu::TextureView,
    pub(crate) output_width: u32,
    pub(crate) output_height: u32,
    pub(crate) render_scale: f32,
    pub(crate) full_res_depth_texture: Option<wgpu::Texture>,
    pub(crate) full_res_depth_view: Option<wgpu::TextureView>,
    pub(crate) surface_format: wgpu::TextureFormat,
    pub(crate) debug_camera_buffer: wgpu::Buffer,
    pub(crate) cull_stats_buffer: wgpu::Buffer,
    pub(crate) ambient_color: [f32; 3],
    pub(crate) ambient_intensity: f32,
    pub(crate) clear_color: [f32; 4],
    pub(crate) gi_config: GiConfig,
    pub(crate) shadow_quality: libhelio::ShadowQuality,
    pub(crate) debug_mode: u32,
    pub(crate) perf_overlay_mode: PerfOverlayMode,
    pub(crate) debug_depth_test: bool,
    pub(crate) editor_mode: bool,
    pub(crate) custom_graph_builder: Option<CustomGraphBuilder>,
    pub(crate) custom_graph_config: Option<RendererConfig>,
    pub(crate) debug_state: Arc<Mutex<DebugDrawState>>,
    pub(crate) debug_overlay_shared: Arc<Mutex<DebugOverlayState>>,
    pub(crate) billboard_instances: Vec<helio_pass_billboard::BillboardInstance>,
    pub(crate) billboard_scratch: Vec<helio_pass_billboard::BillboardInstance>,
    pub(crate) billboard_dirty: bool,
    pub(crate) billboard_cached_light_count: usize,
    pub(crate) billboard_cached_light_gen: u64,
    pub(crate) billboard_cached_editor_hidden: bool,
    pub(crate) billboard_cached_corona_gen: u64,
    pub(crate) billboard_generation: u64,
    pub(crate) corona_emitters: Vec<libhelio::GpuCoronaEmitter>,
    pub(crate) corona_emitter_generation: u64,
    pub(crate) water_volumes_buffer: wgpu::Buffer,
    pub(crate) water_hitboxes_buffer: wgpu::Buffer,
    pub(crate) last_render_time: Instant,
    pub(crate) delta_time: f32,
    pub(crate) graph_time_ms: f32,
    pub(crate) cull_stats_staging: wgpu::Buffer,
    pub(crate) cull_stats: [u32; 8],
    pub(crate) frame_times: Vec<f32>,
    pub(crate) frame_times_cursor: usize,
    pub(crate) jitter_matrices: [glam::Mat4; 16],
    pub(crate) jitter_cache_width: u32,
    pub(crate) jitter_cache_height: u32,
    pub(crate) gizmo_camera: Option<crate::scene::Camera>,
    pub(crate) gizmo_viewport_height: f32,
    #[cfg(feature = "bake")]
    pub(crate) bake_pending: Option<helio_bake::BakeRequest>,
    #[cfg(feature = "bake")]
    pub(crate) baked_data: Option<std::sync::Arc<helio_bake::BakedData>>,
    pub(crate) owns_device: bool,
    pub(crate) pending_resize: Option<(u32, u32)>,
    pub(crate) clear_target_next_frame: bool,
}

pub(crate) enum GraphKind {
    Default,
    Simple,
    Custom,
}

pub struct DebugBatch<'a> {
    pub(crate) state: &'a mut DebugDrawState,
    pub(crate) lines_changed: bool,
    pub(crate) tris_changed: bool,
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

    pub(crate) fn finish(self) {
        if self.lines_changed {
            self.state.user_lines_generation = self.state.user_lines_generation.wrapping_add(1);
        }
        if self.tris_changed {
            self.state.user_tris_generation = self.state.user_tris_generation.wrapping_add(1);
        }
    }
}

impl Renderer {
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

    pub fn find_pass_mut<T: RenderPass + 'static>(&mut self) -> Option<&mut T> {
        self.graph.find_pass_mut::<T>()
    }

    pub fn find_pass<T: RenderPass + 'static>(&self) -> Option<&T> {
        self.graph.find_pass::<T>()
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
            build_default_graph_external(
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

    #[cfg(feature = "bake")]
    pub fn configure_bake(&mut self, request: helio_bake::BakeRequest) {
        self.bake_pending = Some(request);
    }

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

    pub fn set_gizmo_camera(&mut self, camera: &crate::scene::Camera, viewport_height: f32) {
        self.gizmo_camera = Some(*camera);
        self.gizmo_viewport_height = viewport_height;
    }

    pub fn gizmo_camera_info(&self) -> Option<(&crate::scene::Camera, f32)> {
        self.gizmo_camera.as_ref().map(|c| (c, self.gizmo_viewport_height))
    }

    pub fn output_width(&self) -> u32 {
        self.output_width
    }

    pub fn output_height(&self) -> u32 {
        self.output_height
    }
}
