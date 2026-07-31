use std::sync::{Arc, Mutex};

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use bytemuck::{Pod, Zeroable};
use helio_core::{RenderGraph, RenderPass};

use super::config::{PerfOverlayMode, RenderMode, RendererConfig};

/// Closure that rebuilds the render graph on resize.
pub type GraphRebuilder = Arc<
    dyn Fn(
            &Arc<wgpu::Device>,
            &Arc<wgpu::Queue>,
            &Scene,
            RendererConfig,
            Arc<Mutex<DebugDrawState>>,
            &wgpu::Buffer,
            &wgpu::Buffer,
        ) -> RenderGraph
        + Send
        + Sync,
>;

use crate::groups::GroupId;
use crate::mesh::MeshBuffers;
use crate::radiant::RadiantTemplateRegistry;
use crate::scene::Scene;

use super::config::GiConfig;
use super::debug::DebugDrawState;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DebugCameraUniform {
    pub view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct DebugVertex {
    pub position: [f32; 3],
    pub _pad: f32,
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BillboardInstance {
    pub world_pos: [f32; 4],
    pub scale_flags: [f32; 4],
    pub color: [f32; 4],
}

pub(crate) enum CullStatsReadbackState {
    Idle,
    Mapping(Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>>),
    Disabled,
}

pub struct Renderer {
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    pub(crate) graph: RenderGraph,
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
    pub(crate) shadow_atlas_size: u32,
    pub(crate) shadow_face_capacity: u32,
    /// Preserved across graph rebuilds (resize) so an opt-in is not silently lost.
    pub(crate) enable_ssr: bool,
    /// Persisted so the resize rebuild reconstructs the same graph. Every flag the
    /// rebuilder needs has to live here — a literal in `resize.rs` silently produces a
    /// *different* pipeline after the first resize, which is exactly what happened when
    /// this field was first added.
    pub(crate) enable_foliage: bool,
    pub(crate) foliage_blades_per_m2: Option<f32>,
    pub(crate) enable_planar_reflections: bool,
    pub(crate) enable_environment_reflections: bool,
    /// TSR quality preset, preserved across graph rebuilds.
    pub(crate) tsr_quality: Option<helio_pass_tsr::TsrQuality>,
    pub(crate) debug_mode: u32,
    pub(crate) editor_mode: bool,
    pub(crate) debug_state: Arc<Mutex<DebugDrawState>>,
    pub(crate) billboard_instances: Vec<BillboardInstance>,
    pub(crate) billboard_scratch: Vec<BillboardInstance>,
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
    pub(crate) foliage_interactors_buffer: wgpu::Buffer,
    pub(crate) pp_volumes_buffer: wgpu::Buffer,
    pub(crate) postprocess_buffer: wgpu::Buffer,
    pub(crate) last_render_time: Instant,
    pub(crate) delta_time: f32,
    /// Optional 3D LUT texture view for colour grading, set by the application.
    pub(crate) color_grading_lut_view: Option<wgpu::TextureView>,
    pub(crate) ies_texture_view: Option<wgpu::TextureView>,
    pub(crate) graph_time_ms: f32,
    pub(crate) cull_stats_staging: wgpu::Buffer,
    pub(crate) cull_stats_readback_state: CullStatsReadbackState,
    pub(crate) cull_stats: [u32; 8],
    pub(crate) frame_times: Vec<f32>,
    pub(crate) frame_times_cursor: usize,
    /// Whether per-frame subpixel camera jitter is applied. Graphs with a
    /// temporal accumulation pass (TaaPass, TsrPass) need this; non-temporal
    /// graphs (e.g. FXAA-only) disable it to avoid visible shimmer.
    pub(crate) enable_jitter: bool,
    pub(crate) gizmo_camera: Option<crate::scene::Camera>,
    pub(crate) gizmo_viewport_height: f32,
    #[cfg(feature = "bake")]
    pub(crate) bake_pending: Option<helio_bake::BakeRequest>,
    #[cfg(feature = "bake")]
    pub(crate) baked_data: Option<std::sync::Arc<helio_bake::BakedData>>,
    pub(crate) owns_device: bool,
    pub(crate) pending_resize: Option<(u32, u32)>,
    pub(crate) clear_target_next_frame: bool,
    pub(crate) graph_rebuilder: Option<GraphRebuilder>,

    /// Whether the graph was built with the sky passes present.
    ///
    /// `SkyLutPass`/`SkyPass` are added conditionally on `Scene::sky_context().has_sky` at
    /// graph *build* time, but the natural call order is `Renderer::new(scene, graph)` and
    /// only then populate the scene — so a scene that gains a sky afterwards has a graph
    /// that will never draw it. Tracking what the graph was built with is what lets
    /// `rebuild_graph_if_sky_changed` notice.
    pub(crate) graph_has_sky: bool,

    /// Engine-world transform of the headset's stage origin — the locomotion hook.
    ///
    /// Identity means the player stands at the world origin. Translating/rotating this
    /// moves the player through the world without touching the scene, which is what
    /// joystick locomotion drives. Applied to the located eye poses, so it moves the
    /// cameras and nothing else.
    pub(crate) xr_stage_transform: glam::Mat4,
    /// Templates registered by the user for the gbuffer (opaque) path.
    /// Preserved across graph rebuilds (resize).
    pub(crate) template_registry: RadiantTemplateRegistry,

    /// Templates registered by the user for the transparent path.
    /// Used by TransparentPass for alpha-blended materials (water, glass, etc.).
    pub(crate) transparent_template_registry: RadiantTemplateRegistry,
    pub(crate) render_mode: RenderMode,
    /// Whether the render graph was built in OpenXR multiview mode. Mirrors
    /// `config.enable_xr` and is preserved across graph rebuilds (resize) so an
    /// opt-in is not silently lost.
    pub(crate) enable_xr: bool,
    #[cfg(not(target_arch = "wasm32"))]
    /// Live OpenXR instance (owned so the runtime binding is kept alive).
    pub(crate) xr_instance: Option<helio_xr::instance::XrInstance>,
    #[cfg(not(target_arch = "wasm32"))]
    /// Live OpenXR session (frame waiter/stream, spaces). `None` in desktop
    /// mirror mode.
    pub(crate) xr: Option<helio_xr::session::XrSession>,
    #[cfg(not(target_arch = "wasm32"))]
    /// OpenXR swapchain whose images are wrapped as wgpu textures.
    pub(crate) xr_swapchain: Option<helio_xr::swapchain::XrSwapchain>,
    #[cfg(not(target_arch = "wasm32"))]
    /// Two-layer array depth texture for the multiview render pass. Distinct
    /// from `depth_texture` (single-layer, used by passes that *sample* scene
    /// depth as a plain `texture_depth_2d`).
    pub(crate) xr_depth_texture: Option<wgpu::Texture>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) xr_depth_view: Option<wgpu::TextureView>,
    #[cfg(not(target_arch = "wasm32"))]
    /// Layer-0 `D2` view of `xr_depth_texture` for depth-sampling passes.
    pub(crate) xr_depth_view_layer0: Option<wgpu::TextureView>,
    #[cfg(not(target_arch = "wasm32"))]
    /// Consecutive frames skipped because the session was not focused/visible or
    /// the runtime asked us not to render. Used to rate-limit diagnostic logs.
    pub(crate) xr_idle_skips: u64,
    #[cfg(not(target_arch = "wasm32"))]
    /// Application-provided camera template for XR frames: supplies
    /// `postprocess_settings`, near/far and the representative position used
    /// for RC bounds and the debug state. The per-eye view/proj are overridden
    /// by the headset each frame.
    pub(crate) xr_camera: Option<crate::scene::Camera>,
    #[cfg(not(target_arch = "wasm32"))]
    /// PC mirror blit: samples the acquired XR swapchain image (2-layer array)
    /// and draws both eyes side-by-side to the mirror window surface.
    pub(crate) xr_mirror_pipeline: Option<wgpu::RenderPipeline>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) xr_mirror_bgl: Option<wgpu::BindGroupLayout>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) xr_mirror_sampler: Option<wgpu::Sampler>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) xr_mirror_bind_group: Option<(u32, wgpu::BindGroup)>,
    #[cfg(not(target_arch = "wasm32"))]
    /// Colour format of the PC mirror window surface; the blit pipeline's color
    /// target must match it (usually Bgra8UnormSrgb on Windows), not the XR
    /// swapchain format. Set via [`Renderer::set_xr_mirror_format`].
    pub(crate) xr_mirror_format: Option<wgpu::TextureFormat>,
}

pub struct DebugBatch<'a> {
    pub(crate) state: &'a mut DebugDrawState,
    pub(crate) lines_changed: bool,
    pub(crate) tris_changed: bool,
}

impl<'a> DebugBatch<'a> {
    pub fn line(&mut self, from: [f32; 3], to: [f32; 3], color: [f32; 4]) {
        self.state.user_lines.push(DebugVertex {
            position: from,
            _pad: 0.0,
            color,
        });
        self.state.user_lines.push(DebugVertex {
            position: to,
            _pad: 0.0,
            color,
        });
        self.lines_changed = true;
    }

    pub fn tri(&mut self, v0: [f32; 3], v1: [f32; 3], v2: [f32; 3], color: [f32; 4]) {
        self.state.user_tris.push(DebugVertex {
            position: v0,
            _pad: 0.0,
            color,
        });
        self.state.user_tris.push(DebugVertex {
            position: v1,
            _pad: 0.0,
            color,
        });
        self.state.user_tris.push(DebugVertex {
            position: v2,
            _pad: 0.0,
            color,
        });
        self.tris_changed = true;
    }

    pub fn sphere(&mut self, center: [f32; 3], radius: f32, color: [f32; 4], segments: u32) {
        if segments < 4 {
            return;
        }
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

    pub fn cone(
        &mut self,
        apex: [f32; 3],
        axis: [f32; 3],
        height: f32,
        base_radius: f32,
        color: [f32; 4],
        segments: u32,
    ) {
        if segments < 3 {
            return;
        }
        let apex_v = glam::Vec3::from(apex);
        let dir = glam::Vec3::from(axis).normalize_or_zero();
        let base = apex_v + dir * height;
        let up = if dir.cross(glam::Vec3::Y).length_squared() < 1e-8 {
            glam::Vec3::X
        } else {
            glam::Vec3::Y
        };
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

    pub fn filled_cone(
        &mut self,
        apex: [f32; 3],
        axis: [f32; 3],
        height: f32,
        base_radius: f32,
        color: [f32; 4],
        segments: u32,
    ) {
        if segments < 3 {
            return;
        }
        let apex_v = glam::Vec3::from(apex);
        let dir = glam::Vec3::from(axis).normalize_or_zero();
        let base = apex_v + dir * height;
        let up = if dir.cross(glam::Vec3::Y).length_squared() < 1e-8 {
            glam::Vec3::X
        } else {
            glam::Vec3::Y
        };
        let tangent = dir.cross(up).normalize_or_zero();
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
            c + glam::Vec3::new(h, -h, -h),
            c + glam::Vec3::new(h, h, -h),
            c + glam::Vec3::new(-h, h, -h),
            c + glam::Vec3::new(-h, -h, h),
            c + glam::Vec3::new(h, -h, h),
            c + glam::Vec3::new(h, h, h),
            c + glam::Vec3::new(-h, h, h),
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
            self.tri(
                corners[a].to_array(),
                corners[b].to_array(),
                corners[cc].to_array(),
                color,
            );
            self.tri(
                corners[a].to_array(),
                corners[cc].to_array(),
                corners[d].to_array(),
                color,
            );
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
    }

    /// Overrides the graph-derived per-frame camera-jitter setting.
    ///
    /// Graphs containing a temporal reconstruction pass enable jitter
    /// automatically. FXAA and other non-temporal graphs leave it disabled so
    /// the final image remains pixel-stable.
    pub fn set_jitter_enabled(&mut self, enabled: bool) {
        self.enable_jitter = enabled;
    }

    pub fn set_debug_mode(&mut self, mode: u32) {
        self.debug_mode = mode;
        self.graph.set_debug_mode(mode);
    }

    pub fn available_debug_views(&self) -> Vec<helio_core::DebugViewDescriptor> {
        self.graph.collect_debug_views()
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

    pub fn camera_buffer(&self) -> &wgpu::Buffer {
        self.scene.gpu_scene().camera.buffer()
    }

    /// Latest frame timing state. Reading it performs no GPU polling,
    /// synchronization, or allocation.
    pub fn timing_snapshot(&self) -> &helio_core::RenderTimingSnapshot {
        self.graph.profiler().timing_snapshot()
    }

    pub fn mesh_buffers(&self) -> MeshBuffers<'_> {
        self.scene.mesh_buffers()
    }

    pub fn dynamic_mesh_buffers(&self) -> MeshBuffers<'_> {
        self.scene.dynamic_mesh_buffers()
    }

    pub fn add_pass(&mut self, pass: Box<dyn helio_core::RenderPass>) {
        self.graph.add_pass(pass);
    }

    pub fn find_pass_mut<T: RenderPass + 'static>(&mut self) -> Option<&mut T> {
        self.graph.find_pass_mut::<T>()
    }

    pub fn graph_pass_index<T: RenderPass + 'static>(&self) -> Option<usize> {
        self.graph.pass_index_of::<T>()
    }

    pub fn replace_graph_pass(&mut self, index: usize, pass: Box<dyn RenderPass>) {
        self.graph.replace_pass_at(index, pass);
    }

    pub fn find_pass<T: RenderPass + 'static>(&self) -> Option<&T> {
        self.graph.find_pass::<T>()
    }

    /// Access the gbuffer template registry (preserved across graph rebuilds).
    /// Register custom surface templates here instead of through the pass
    /// directly to ensure they survive window resize.
    pub fn template_registry_mut(&mut self) -> &mut RadiantTemplateRegistry {
        &mut self.template_registry
    }

    /// Access the transparent template registry for alpha-blended materials.
    pub fn transparent_template_registry_mut(&mut self) -> &mut RadiantTemplateRegistry {
        &mut self.transparent_template_registry
    }

    /// Sync the renderer's template registries into the GpuScene so passes
    /// can find them across graph rebuilds.
    pub(crate) fn sync_template_registry_to_scene(&mut self) {
        // Gbuffer templates
        let reg: Box<dyn std::any::Any + Send + Sync> = Box::new(self.template_registry.clone());
        self.scene.set_template_registry(reg);

        // Transparent templates (separate registry to avoid binding layout conflicts)
        let treg: Box<dyn std::any::Any + Send + Sync> = Box::new(self.transparent_template_registry.clone());
        self.scene.set_transparent_template_registry(treg);
    }

    pub fn set_clear_color(&mut self, color: [f32; 4]) {
        self.clear_color = color;
    }

    pub fn set_ambient(&mut self, color: [f32; 3], intensity: f32) {
        self.ambient_color = color;
        self.ambient_intensity = intensity;
    }

    pub fn set_graph(&mut self, mut graph: RenderGraph) {
        // Extract rebuilder stored in the graph by the builder function
        self.graph_rebuilder = graph.take_graph_data::<GraphRebuilder>();
        self.graph = graph;
    }

    pub fn set_graph_with_builder(&mut self, graph: RenderGraph, rebuilder: GraphRebuilder) {
        self.graph = graph;
        self.graph_rebuilder = Some(rebuilder);
    }

    pub fn set_rebuilder(&mut self, rebuilder: GraphRebuilder) {
        self.graph_rebuilder = Some(rebuilder);
    }

    #[cfg(feature = "bake")]
    pub fn configure_bake(&mut self, mut request: helio_bake::BakeRequest) {
        self.sync_reflection_capture_probes(&mut request.config);
        self.bake_pending = Some(request);
    }

    /// Point the probe bake at the scene's static reflection captures, and bind
    /// each capture to the cube array layer its probe will land in.
    ///
    /// Both halves come from one ordered traversal, which is the whole point:
    /// hand-authored probe positions and hand-set layer indices are two lists
    /// that silently rot apart the moment a capture moves or is deleted.
    #[cfg(feature = "bake")]
    fn sync_reflection_capture_probes(&mut self, config: &mut helio_bake::BakeConfig) {
        let positions = self.scene.static_reflection_capture_positions();
        if positions.is_empty() {
            return;
        }
        match config.probes.as_mut() {
            Some(spec) => spec.positions = positions,
            None => {
                config.probes = Some(helio_bake::ProbeSpec {
                    positions,
                    config: helio_bake::ProbeConfig::default(),
                })
            }
        }
        self.scene.assign_reflection_capture_layers();
    }

    #[cfg(feature = "bake")]
    pub fn auto_bake(&mut self, config: helio_bake::BakeConfig) {
        let scene = self.scene.build_static_bake_scene();
        self.configure_bake(helio_bake::BakeRequest { scene, config });
    }

    pub fn set_billboard_instances(&mut self, instances: &[BillboardInstance]) {
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
        self.gizmo_camera = Some(camera.clone());
        self.gizmo_viewport_height = viewport_height;
    }

    pub fn gizmo_camera_info(&self) -> Option<(&crate::scene::Camera, f32)> {
        self.gizmo_camera
            .as_ref()
            .map(|c| (c, self.gizmo_viewport_height))
    }

    pub fn output_width(&self) -> u32 {
        self.output_width
    }

    pub fn output_height(&self) -> u32 {
        self.output_height
    }

    /// Snapshot the renderer settings needed to build a replacement graph.
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    /// Returns a reference to the GPU device.
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    pub fn renderer_config(&self) -> RendererConfig {
        RendererConfig {
            width: self.output_width,
            height: self.output_height,
            surface_format: self.surface_format,
            gi_config: self.gi_config,
            shadow_quality: self.shadow_quality,
            debug_mode: self.debug_mode,
            render_scale: self.render_scale,
            perf_overlay_mode: PerfOverlayMode::Disabled,
            shadow_atlas_size: self.shadow_atlas_size,
            shadow_face_capacity: self.shadow_face_capacity,
            enable_ssr: self.enable_ssr,
            enable_planar_reflections: self.enable_planar_reflections,
            enable_environment_reflections: self.enable_environment_reflections,
            tsr_quality: self.tsr_quality,
            hdr_output_mode: libhelio::HdrOutputMode::Ldr,
            render_mode: self.render_mode,
            enable_xr: self.enable_xr,
            enable_foliage: self.enable_foliage,
            foliage_blades_per_m2: self.foliage_blades_per_m2,
        }
    }

    /// Hand the renderer an OpenXR session and swapchain created by the
    /// application (via `helio-xr`). All three are optional: `None` fields keep
    /// the renderer in desktop (window) mode, which is the fallback when no
    /// headset is connected.
    ///
    /// The swapchain must have been created from `session` and both must belong
    /// to the same `wgpu::Device` the renderer was built with.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_xr_session(
        &mut self,
        instance: Option<helio_xr::instance::XrInstance>,
        session: Option<helio_xr::session::XrSession>,
        swapchain: Option<helio_xr::swapchain::XrSwapchain>,
    ) {
        self.xr_instance = instance;
        self.xr = session;
        self.xr_swapchain = swapchain;
    }

    /// Predicted display time of the most recent XR frame, used to locate
    /// controller poses (see `helio_xr::XrInput::grip_pose_matrices`) at the
    /// same instant the eye views were located. `None` in desktop mode or
    /// before the first XR frame has been waited on.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn xr_last_display_time(&self) -> Option<helio_xr::Time> {
        self.xr.as_ref().map(|session| session.last_display_time)
    }

    /// Set the camera template used for XR frames (postprocess settings,
    /// near/far planes, RC bounds position). The per-eye view/projection
    /// matrices are overridden by the headset pose each frame; only the
    /// post-processing settings and clip distances are read back.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_xr_camera(&mut self, camera: crate::scene::Camera) {
        self.xr_camera = Some(camera);
    }

    /// Set the PC mirror window's colour format (from the mirror surface's
    /// capabilities) so the XR mirror blit pipeline's color target matches.
    /// Usually `Bgra8UnormSrgb` on Windows.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_xr_mirror_format(&mut self, format: wgpu::TextureFormat) {
        if self.xr_mirror_format != Some(format) {
            self.xr_mirror_format = Some(format);
            // The pipeline is keyed on the format; drop it so it is rebuilt.
            self.xr_mirror_pipeline = None;
        }
    }

    /// Whether the renderer was built with the OpenXR multiview path enabled.
    pub fn xr_enabled(&self) -> bool {
        self.enable_xr
    }

    /// Returns a reference to the post-process uniform buffer.
    pub fn postprocess_buffer(&self) -> &wgpu::Buffer {
        &self.postprocess_buffer
    }

    /// Set or clear the 3D colour grading LUT texture.
    /// Call with `None` to disable LUT grading.
    /// The LUT texture must be `Rgba16Float`, `TextureDimension::D3`.
    pub fn set_color_grading_lut(&mut self, view: Option<wgpu::TextureView>) {
        self.color_grading_lut_view = view;
    }

    /// Returns the current colour grading LUT view, if any.
    pub fn color_grading_lut(&self) -> Option<&wgpu::TextureView> {
        self.color_grading_lut_view.as_ref()
    }

    /// Upload an IES profile texture to the scene's IES texture array.
    ///
    /// The texture should be a single R8Unorm 256×256 layer containing the
    /// IES angular intensity distribution. Returns the layer index to use
    /// as `ies_profile_index` / `light_function_index` on GpuLight.
    ///
    /// Currently creates a new array texture each call (single-profile).
    /// Multi-profile support can be added by growing the array.
    pub fn upload_ies_texture(&mut self, data: &[u8], width: u32, height: u32) -> Option<u32> {
        if data.len() < (width * height) as usize {
            return None;
        }
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("IES Texture Array"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        self.ies_texture_view = Some(view);
        Some(0) // always layer 0 for now
    }

    /// Returns the current IES texture array view, if any.
    pub fn ies_texture_view(&self) -> Option<&wgpu::TextureView> {
        self.ies_texture_view.as_ref()
    }

    /// Set the IES texture array view directly (for multi-layer arrays).
    pub fn set_ies_texture_view(&mut self, view: wgpu::TextureView) {
        self.ies_texture_view = Some(view);
    }
}
