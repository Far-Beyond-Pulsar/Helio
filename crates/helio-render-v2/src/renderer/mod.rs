//! Main renderer implementation

mod config;
mod init;
mod scene_prep;
mod uniforms;
mod shadow_math;
mod helpers;
mod portal;

pub use config::RendererConfig;

use crate::resources::ResourceManager;
use crate::features::{FeatureRegistry, FeatureContext, PrepareContext, RadianceCascadesFeature};
use crate::pipeline::{PipelineCache, PipelineVariant};
use crate::graph::{RenderGraph, GraphContext};
use crate::passes::{DebugDrawPass, SkyPass, SkyLutPass, SKY_LUT_W, SKY_LUT_H, SKY_LUT_FORMAT, ShadowCullLight, DepthPrepassPass, GBufferPass, GBufferTargets, DeferredLightingPass, TransparentPass, AntiAliasingMode, FxaaPass, SmaaPass, TaaPass, IndirectDispatchPass};
use crate::mesh::{GpuMesh, DrawCall, GpuDrawCall};
use crate::camera::Camera;
use crate::scene::Scene;
use crate::debug_draw::{self, DebugDrawBatch, DebugShape};
use crate::features::lighting::{GpuLight, MAX_LIGHTS};
use crate::features::BillboardsFeature;
use crate::material::{Material, GpuMaterial, MaterialUniform, DefaultMaterialViews, build_gpu_material};
use crate::profiler::{GpuProfiler, PassTiming};
use crate::{Result, Error};
use helio_live_portal::{
    LivePortalHandle,
    PortalFrameSnapshot,
    PortalPassTiming,
    PortalStageTiming,
    PortalSceneLayout,
    PortalSceneLayoutDelta,
    PortalSceneCamera,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, atomic::{AtomicU32, Ordering}};
use std::time::{SystemTime, UNIX_EPOCH};
use wgpu::util::DeviceExt;

use self::uniforms::GlobalsUniform;
use self::portal::{compute_scene_delta, open_url_in_browser};

/// Main renderer
pub struct Renderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    resources: ResourceManager,
    graph: RenderGraph,
    pipelines: PipelineCache,
    features: FeatureRegistry,

    // Uniform buffers
    camera_buffer: wgpu::Buffer,
    globals_buffer: wgpu::Buffer,

    // Bind groups
    global_bind_group: wgpu::BindGroup,
    lighting_bind_group: Arc<wgpu::BindGroup>,
    lighting_layout: Arc<wgpu::BindGroupLayout>,
    default_material_bind_group: Arc<wgpu::BindGroup>,

    // Default 1×1 texture views + sampler shared by all materials
    default_material_views: DefaultMaterialViews,

    // Draw list (shared with GeometryPass)
    draw_list: Arc<Mutex<Vec<DrawCall>>>,
    // Debug draw primitives queued by user each frame.
    debug_shapes: Arc<Mutex<Vec<DebugShape>>>,
    // GPU batch built from debug_shapes before graph execution.
    debug_batch: Arc<Mutex<Option<DebugDrawBatch>>>,

    // Light buffer for scene writes
    light_buffer: Arc<wgpu::Buffer>,
    light_buffer_capacity_lights: u32,
    lighting_shadow_view: Arc<wgpu::TextureView>,
    lighting_shadow_sampler: Arc<wgpu::Sampler>,
    lighting_env_cube_view: Arc<wgpu::TextureView>,
    lighting_rc_view: Arc<wgpu::TextureView>,
    lighting_env_sampler: Arc<wgpu::Sampler>,
    // Shadow light-space matrix buffer (shared with ShadowPass)
    shadow_matrix_buffer: Arc<wgpu::Buffer>,
    // Shared light count for ShadowPass (updated each frame before graph exec)
    light_count_arc: Arc<AtomicU32>,
    // Per-light active face counts: 6=point, 4=directional, 1=spot
    light_face_counts: Arc<std::sync::Mutex<Vec<u8>>>,
    // Per-light position/range/type for ShadowPass draw-call culling
    shadow_cull_lights: Arc<std::sync::Mutex<Vec<ShadowCullLight>>>,
    // Current scene ambient (updated by render_scene)
    scene_ambient_color: [f32; 3],
    scene_ambient_intensity: f32,
    scene_light_count: u32,
    scene_sky_color: [f32; 3],
    scene_has_sky: bool,
    /// CSM cascade split distances uploaded each frame to GlobalsUniform.
    scene_csm_splits: [f32; 4],
    // RC world bounds (set from RadianceCascadesFeature, zeroed if disabled)
    rc_world_min: [f32; 3],
    rc_world_max: [f32; 3],

    // Sky pass resources
    sky_uniform_buffer: wgpu::Buffer,
    sky_bind_group: Arc<wgpu::BindGroup>,

    // Depth buffer (Depth32Float, recreated on resize)
    depth_texture:      wgpu::Texture,
    depth_view:         wgpu::TextureView,
    /// Depth-only view (DepthOnly aspect) bound into the G-buffer read bind group.
    depth_sample_view:  wgpu::TextureView,

    // ── Deferred G-buffer ──────────────────────────────────────────────────
    gbuf_albedo_texture:   wgpu::Texture,
    gbuf_normal_texture:   wgpu::Texture,
    gbuf_orm_texture:      wgpu::Texture,
    gbuf_emissive_texture: wgpu::Texture,
    /// Shared with GBufferPass.  Swapped on resize.
    gbuffer_targets: Arc<Mutex<GBufferTargets>>,
    /// Shared with DeferredLightingPass.  Swapped on resize.
    deferred_bg: Arc<Mutex<Arc<wgpu::BindGroup>>>,

    // ── AO and AA resources ───────────────────────────────────────────────
    enable_ssao: bool,
    ssao_texture: Option<wgpu::Texture>,
    ssao_view: Option<wgpu::TextureView>,
    aa_mode: AntiAliasingMode,
    pre_aa_texture: wgpu::Texture,
    pre_aa_view: wgpu::TextureView,
    fxaa_pass: Option<FxaaPass>,
    smaa_pass: Option<SmaaPass>,
    taa_pass: Option<TaaPass>,
    fxaa_bind_group: Option<wgpu::BindGroup>,
    smaa_bind_group: Option<wgpu::BindGroup>,
    taa_bind_group: Option<wgpu::BindGroup>,

    // ── GPU-DRIVEN RENDERING (optional indirect rendering) ─────────────────
    gpu_driven: bool,
    async_compute: bool,
    /// Indirect draw buffer for opaque draws (built by compute each frame)
    indirect_opaque_buffer: Option<Arc<wgpu::Buffer>>,
    /// Indirect draw buffer for transparent draws (built by compute each frame)
    indirect_transparent_buffer: Option<Arc<wgpu::Buffer>>,
    /// Opaque draw count written by preprocessing compute
    opaque_draw_count: Arc<std::sync::atomic::AtomicU32>,
    /// Transparent draw count written by preprocessing compute
    transparent_draw_count: Arc<std::sync::atomic::AtomicU32>,
    /// Draw list upload buffer (GpuDrawCall array sent to compute shader)
    draw_list_gpu_buffer: Option<Arc<wgpu::Buffer>>,
    /// Material ID assignment: bind group pointer -> unique ID
    material_id_map: HashMap<usize, u32>,
    /// Next material ID to assign
    next_material_id: u32,

    // ── TEMPORAL CACHING ──────────────────────────────────────────────────
    /// Cached sky state — skip LUT re-render if unchanged
    cached_sky_color: [f32; 3],
    cached_sky_has_sky: bool,
    cached_sky_sun_direction: [f32; 3],
    cached_sky_sun_intensity: f32,
    sky_state_changed: bool,

    /// Monotonically increasing counter.  Bumped only when the draw-list structure
    /// actually changes between frames.
    draw_list_generation: u64,
    draw_list_structural_hash: u64,
    scene_fingerprint: u64,
    canonical_draw_list: Vec<DrawCall>,

    /// Cached light list state — skip sort if unchanged
    cached_light_count: usize,
    cached_light_position_hash: u64,
    cached_camera_pos: [f32; 3],
    camera_move_threshold: f32,

    scratch_gpu_lights: Vec<GpuLight>,
    scratch_shadow_mats: Vec<uniforms::GpuShadowMatrix>,
    scratch_shadow_matrix_hashes: Vec<u64>,
    scratch_batches: HashMap<(usize, usize), Vec<[f32; 16]>>,
    scratch_example_idx: HashMap<(usize, usize), usize>,
    scratch_sorted_light_indices: Vec<usize>,
    instance_buffer_cache: HashMap<(usize, usize), (Arc<wgpu::Buffer>, u32)>,

    // Frame state
    frame_count: u64,
    width: u32,
    height: u32,
    last_frame_start: Option<std::time::Instant>,
    last_frame_end: Option<std::time::Instant>,

    /// GPU + CPU pass-level profiler.  `None` when TIMESTAMP_QUERY is unavailable.
    profiler: Option<GpuProfiler>,

    /// When true, GPU timing stats are printed to stderr every frame.
    debug_printout: bool,
    /// Optional live web dashboard handle for real-time pipeline/perf telemetry.
    live_portal: Option<LivePortalHandle>,
    /// Last captured scene layout snapshot forwarded to the portal thread.
    latest_scene_layout: Option<PortalSceneLayout>,
    /// Previous scene layout for delta computation (reduces bandwidth).
    previous_scene_layout: Option<PortalSceneLayout>,
    /// Per-stage CPU timings (ms) from the most recent `render_scene()` call.
    /// [0]=draws, [1]=lights, [2]=shadows, [3]=billboards, [4]=sky.
    pending_scene_stage_ms: [f32; 5],
    /// True when render_scene() rebuilt the full layout (objects/lights/billboards changed).
    pending_layout_changed: bool,
    portal_scene_key: (usize, usize, usize),
    /// Pass names cached once after `graph.build()` to avoid a `Vec<String>` alloc every frame.
    cached_pass_names: Vec<String>,
}

impl Renderer {
    // ── Draw submission ───────────────────────────────────────────────────────

    /// Queue a mesh to be drawn this frame using the default white material
    pub fn draw_mesh(&self, mesh: &GpuMesh) {
        self.draw_list.lock().unwrap().push(DrawCall::new(mesh, self.default_material_bind_group.clone(), false));
    }

    /// Queue a mesh with a custom GPU material (preserves transparency mode)
    pub fn draw_mesh_with_gpu_material(&self, mesh: &GpuMesh, material: &GpuMaterial) {
        self.draw_list.lock().unwrap().push(DrawCall::new(mesh, material.bind_group.clone(), material.transparent_blend));
    }

    /// Queue a mesh with a custom material bind group (legacy opaque path)
    pub fn draw_mesh_with_material(&self, mesh: &GpuMesh, material: Arc<wgpu::BindGroup>) {
        self.draw_list.lock().unwrap().push(DrawCall::new(mesh, material, false));
    }

    pub fn debug_shape(&self, shape: DebugShape) {
        self.debug_shapes.lock().unwrap().push(shape);
    }

    pub fn debug_line(&self, start: glam::Vec3, end: glam::Vec3, color: [f32; 4], thickness: f32) {
        self.debug_shape(DebugShape::Line { start, end, color, thickness });
    }

    pub fn debug_cone(&self, apex: glam::Vec3, direction: glam::Vec3, height: f32, radius: f32, color: [f32; 4], thickness: f32) {
        self.debug_shape(DebugShape::Cone { apex, direction, height, radius, color, thickness });
    }

    pub fn debug_box(&self, center: glam::Vec3, half_extents: glam::Vec3, rotation: glam::Quat, color: [f32; 4], thickness: f32) {
        self.debug_shape(DebugShape::Box { center, half_extents, rotation, color, thickness });
    }

    pub fn debug_sphere(&self, center: glam::Vec3, radius: f32, color: [f32; 4], thickness: f32) {
        self.debug_shape(DebugShape::Sphere { center, radius, color, thickness });
    }

    pub fn debug_capsule(&self, start: glam::Vec3, end: glam::Vec3, radius: f32, color: [f32; 4], thickness: f32) {
        self.debug_shape(DebugShape::Capsule { start, end, radius, color, thickness });
    }

    pub fn clear_debug_shapes(&self) {
        self.debug_shapes.lock().unwrap().clear();
    }

    // ── Feature enable/disable ────────────────────────────────────────────────

    pub fn enable_feature(&mut self, name: &str) -> Result<()> {
        self.features.enable(name)?;
        let flags = self.features.active_flags();
        self.pipelines.set_active_features(flags);
        log::trace!("Enabled feature: {}", name);
        Ok(())
    }

    pub fn disable_feature(&mut self, name: &str) -> Result<()> {
        self.features.disable(name)?;
        let flags = self.features.active_flags();
        self.pipelines.set_active_features(flags);
        log::trace!("Disabled feature: {}", name);
        Ok(())
    }

    pub fn get_feature_mut<T: crate::features::Feature + 'static>(&mut self, name: &str) -> Option<&mut T> {
        self.features.get_typed_mut::<T>(name)
    }

    /// Toggle the per-frame GPU timing printout.
    pub fn debug_key_pressed(&mut self) {
        self.debug_printout = !self.debug_printout;
        eprintln!("[Helio] GPU timing printout: {}",
            if self.debug_printout { "ON  (press 4 to hide)" } else { "OFF" });
    }

    /// Start the live performance dashboard and open it in the user's browser.
    pub fn start_live_portal(&mut self, bind_addr: &str) -> Result<String> {
        if let Some(portal) = &self.live_portal {
            return Ok(portal.url.clone());
        }
        let handle = helio_live_portal::start_live_portal(bind_addr)
            .map_err(|e| Error::Resource(format!("Failed to start live portal on {bind_addr}: {e}")))?;
        let url = handle.url.clone();
        open_url_in_browser(&url);
        log::info!("Helio live portal started at {url}");
        self.live_portal = Some(handle);
        Ok(url)
    }

    /// Convenience: start live portal on the default port.
    pub fn start_live_portal_default(&mut self) -> Result<String> {
        self.start_live_portal("0.0.0.0:7878")
    }

    pub fn is_gpu_driven(&self) -> bool { self.gpu_driven }
    pub fn is_async_compute(&self) -> bool { self.async_compute }

    // ── Frame rendering ───────────────────────────────────────────────────────

    /// Render a frame.  Call `draw_mesh()` BEFORE calling this.
    pub fn render(&mut self, camera: &Camera, target: &wgpu::TextureView, delta_time: f32) -> Result<()> {
        let frame_start = std::time::Instant::now();
        log::trace!("Rendering frame {}", self.frame_count);

        // ── Preparation: camera upload, feature prepare, globals, debug batch, GPU-driven ──
        let prep_start = std::time::Instant::now();

        // Upload camera uniform (features may use camera-dependent logic in prepare).
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera));

        // Prepare features (upload lights etc.)
        let prep_ctx = PrepareContext::new(
            &self.device, &self.queue, &self.resources,
            self.frame_count, delta_time, camera,
        );
        self.features.prepare_all(&prep_ctx)?;

        // Pull live RC bounds after feature prepare so GI volume can follow camera.
        log::trace!("Attempting to pull RC bounds from feature registry");
        if let Some(rc) = self.features.get_typed_mut::<RadianceCascadesFeature>("radiance_cascades") {
            let (mn, mx) = rc.world_bounds();
            self.rc_world_min = mn;
            self.rc_world_max = mx;
            log::trace!("✓ RC bounds pulled from feature: [{:?} .. {:?}]", mn, mx);
        } else {
            log::warn!("✗ FAILED to pull RC bounds - feature not found in registry!");
        }

        // Update globals after feature prepare so all per-frame feature outputs are current.
        let globals = GlobalsUniform {
            frame: self.frame_count as u32,
            delta_time,
            light_count: self.scene_light_count,
            ambient_intensity: self.scene_ambient_intensity,
            ambient_color: [self.scene_ambient_color[0], self.scene_ambient_color[1], self.scene_ambient_color[2], 0.0],
            rc_world_min: [self.rc_world_min[0], self.rc_world_min[1], self.rc_world_min[2], 0.0],
            rc_world_max: [self.rc_world_max[0], self.rc_world_max[1], self.rc_world_max[2], 0.0],
            csm_splits: self.scene_csm_splits,
        };
        log::trace!("Uploading globals: RC bounds min=[{:.1} {:.1} {:.1}] max=[{:.1} {:.1} {:.1}]",
                   self.rc_world_min[0], self.rc_world_min[1], self.rc_world_min[2],
                   self.rc_world_max[0], self.rc_world_max[1], self.rc_world_max[2]);
        self.queue.write_buffer(&self.globals_buffer, 0, bytemuck::bytes_of(&globals));

        // Build GPU debug batch from shapes submitted since the previous frame.
        let debug_shapes = {
            let mut shapes = self.debug_shapes.lock().unwrap();
            std::mem::take(&mut *shapes)
        };
        if debug_shapes.is_empty() {
            *self.debug_batch.lock().unwrap() = None;
        } else {
            *self.debug_batch.lock().unwrap() = debug_draw::build_batch(&self.device, &debug_shapes);
        }

        // ── GPU-DRIVEN RENDERING: Assign material IDs and upload draw list ──────
        if self.gpu_driven {
            if let Some(ref draw_list_buf) = self.draw_list_gpu_buffer {
                let gpu_draws = {
                    let mut draw_calls = self.draw_list.lock().unwrap();
                    for dc in draw_calls.iter_mut() {
                        let mat_ptr = Arc::as_ptr(&dc.material_bind_group) as usize;
                        if !self.material_id_map.contains_key(&mat_ptr) {
                            self.material_id_map.insert(mat_ptr, self.next_material_id);
                            self.next_material_id += 1;
                        }
                        dc.material_id = *self.material_id_map.get(&mat_ptr).unwrap();
                    }
                    draw_calls.iter().map(|dc| {
                        GpuDrawCall {
                            vertex_offset: 0,
                            index_offset: 0,
                            index_count: dc.index_count,
                            vertex_count: dc.vertex_count,
                            material_id: dc.material_id,
                            transparent_blend: if dc.transparent_blend { 1 } else { 0 },
                            _pad0: 0,
                            _pad1: 0,
                            bounds_center: dc.bounds_center,
                            bounds_radius: dc.bounds_radius,
                        }
                    }).collect::<Vec<_>>()
                };
                if !gpu_draws.is_empty() {
                    self.queue.write_buffer(draw_list_buf, 0, bytemuck::cast_slice(&gpu_draws));
                    log::trace!("GPU-driven: Uploaded {} draw calls for indirect rendering", gpu_draws.len());
                }
                if let (Some(ref _opaque_buf), Some(ref _transparent_buf)) =
                    (&self.indirect_opaque_buffer, &self.indirect_transparent_buffer) {
                    self.opaque_draw_count.store(0, Ordering::Release);
                    self.transparent_draw_count.store(0, Ordering::Release);
                }
            }
        }

        let prep_ms = prep_start.elapsed().as_secs_f32() * 1000.0;

        // ── Execute render graph ──────────────────────────────────────────────
        let graph_start = std::time::Instant::now();

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        // Determine render target: if AA is enabled, write to pre_aa texture first
        let graph_target = match self.aa_mode {
            AntiAliasingMode::None | AntiAliasingMode::Msaa(_) => target,
            _ => &self.pre_aa_view,
        };

        // Clear pre_aa_texture at the start of each frame when AA is enabled
        if !matches!(self.aa_mode, AntiAliasingMode::None | AntiAliasingMode::Msaa(_)) {
            let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Pre-AA Clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.pre_aa_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }

        let mut graph_ctx = GraphContext {
            encoder: &mut encoder,
            resources: &self.resources,
            target: graph_target,
            depth_view: &self.depth_view,
            frame: self.frame_count,
            global_bind_group: &self.global_bind_group,
            lighting_bind_group: &self.lighting_bind_group,
            sky_color: self.scene_sky_color,
            has_sky: self.scene_has_sky,
            sky_state_changed: self.sky_state_changed,
            sky_bind_group: None,
            camera_position: camera.position,
            camera_forward: camera.forward(),
            draw_list_generation: self.draw_list_generation,
        };

        let profiling_active = self.debug_printout || self.live_portal.is_some();

        if profiling_active {
            if let Some(p) = &mut self.profiler { p.begin_frame(); }
            self.graph.execute(&mut graph_ctx, self.profiler.as_mut())?;
        } else {
            self.graph.execute(&mut graph_ctx, None)?;
        }
        let graph_ms = graph_start.elapsed().as_secs_f32() * 1000.0;

        // ── Anti-aliasing post-processing ─────────────────────────────────────
        let aa_start = std::time::Instant::now();
        if let Some(fxaa) = &self.fxaa_pass {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("FXAA Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target, resolve_target: None, depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
            });
            if let Some(bind_group) = &self.fxaa_bind_group { fxaa.execute_draw(&mut pass, bind_group); }
        } else if let Some(smaa) = &self.smaa_pass {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SMAA Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target, resolve_target: None, depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
            });
            if let Some(bind_group) = &self.smaa_bind_group { smaa.execute_draw(&mut pass, bind_group); }
        } else if let Some(taa) = &self.taa_pass {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("TAA Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target, resolve_target: None, depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
            });
            if let Some(bind_group) = &self.taa_bind_group { taa.execute_draw(&mut pass, bind_group); }
        }
        let aa_ms = aa_start.elapsed().as_secs_f32() * 1000.0;

        // ── Resolve GPU timestamp queries ─────────────────────────────────────
        let resolve_start = std::time::Instant::now();
        if profiling_active {
            if let Some(p) = &mut self.profiler { p.resolve(&mut encoder); }
        }
        let resolve_ms = resolve_start.elapsed().as_secs_f32() * 1000.0;

        // ── Finalize command buffer ───────────────────────────────────────────
        let finish_start = std::time::Instant::now();
        let cmd_buf = encoder.finish();
        let finish_ms = finish_start.elapsed().as_secs_f32() * 1000.0;

        // ── Submit to GPU ─────────────────────────────────────────────────────
        let submit_start = std::time::Instant::now();
        self.queue.submit(Some(cmd_buf));
        let submit_ms = submit_start.elapsed().as_secs_f32() * 1000.0;
        if self.debug_printout && (finish_ms + submit_ms) > 10.0 {
            eprintln!("⚠️  encoder.finish()={:.2}ms  queue.submit()={:.2}ms", finish_ms, submit_ms);
        }

        *self.debug_batch.lock().unwrap() = None;

        // ── Profiler readback ─────────────────────────────────────────────────
        let poll_start = std::time::Instant::now();
        if profiling_active {
            if let Some(p) = &mut self.profiler {
                p.begin_readback();
                p.poll_results(&self.device);
            }
        }
        let poll_ms = poll_start.elapsed().as_secs_f32() * 1000.0;

        // ── Frame timing bookkeeping ──────────────────────────────────────────
        let frame_to_frame_ms = if let Some(last_start) = self.last_frame_start {
            frame_start.duration_since(last_start).as_secs_f32() * 1000.0
        } else {
            0.0
        };

        let frame_time_ms = frame_start.elapsed().as_secs_f32() * 1000.0;
        let untracked_ms = frame_to_frame_ms - frame_time_ms;

        self.last_frame_start = Some(frame_start);
        self.last_frame_end = Some(std::time::Instant::now());

        // ── Debug printout (every 60 frames) ──────────────────────────────────
        if self.debug_printout && self.frame_count % 60 == 0 {
            if let Some(p) = &mut self.profiler {
                p.set_frame_time_ms(frame_time_ms);
                p.set_frame_to_frame_ms(frame_to_frame_ms);
            }
            if let Some(p) = &self.profiler {
                if !p.last_timings.is_empty() {
                    let timings = p.last_timings.clone();
                    let total_gpu   = p.last_total_gpu_ms;
                    let total_cpu   = p.last_total_cpu_ms;
                    std::thread::spawn(move || {
                        crate::profiler::GpuProfiler::print_snapshot(timings, total_gpu, total_cpu, frame_time_ms, frame_to_frame_ms);
                    });
                }
            }
        }

        // ── Live portal snapshot ──────────────────────────────────────────────
        if let (Some(portal), Some(p)) = (&self.live_portal, &self.profiler) {
            let now_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0);

            let frame_num = self.frame_count;
            let ft_time = frame_time_ms;
            let ft_ff = frame_to_frame_ms;
            let tot_gpu = p.last_total_gpu_ms;
            let tot_cpu = p.last_total_cpu_ms;

            let pass_clone = p.last_timings.clone();
            let pipeline_clone = self.cached_pass_names.clone();

            let current_layout = self.latest_scene_layout.clone();
            let previous_layout = self.previous_scene_layout.clone();
            let layout_changed = self.pending_layout_changed;
            let layout_for_update = current_layout.clone();

            let (draw_total, draw_opaque, draw_transparent) = {
                let draws = self.draw_list.lock().unwrap();
                let total = draws.len();
                let opaque = draws.iter().filter(|dc| !dc.transparent_blend).count();
                let transparent = total.saturating_sub(opaque);
                (total, opaque, transparent)
            };

            let (obj_count, light_count, bb_count) = if let Some(layout) = &current_layout {
                (layout.objects.len(), layout.lights.len(), layout.billboards.len())
            } else { (0, 0, 0) };

            let prep = prep_ms;
            let graphm = graph_ms;
            let aam = aa_ms;
            let resolve = resolve_ms;
            let finish = finish_ms;
            let submitm = submit_ms;
            let pollm = poll_ms;
            let untracked_val = untracked_ms;
            let stage_copy = self.pending_scene_stage_ms;

            let tx = portal.sender();

            std::thread::spawn(move || {
                let scene_delta = current_layout.as_ref().map(|cur| {
                    if layout_changed {
                        compute_scene_delta(cur, previous_layout.as_ref())
                    } else {
                        let prev_cam = previous_layout.as_ref().and_then(|p| p.camera.clone());
                        let mut d = PortalSceneLayoutDelta::default();
                        if cur.camera != prev_cam {
                            d.camera = Some(cur.camera.clone());
                        }
                        d
                    }
                });

                let snapshot = PortalFrameSnapshot {
                    frame: frame_num,
                    frame_time_ms: ft_time,
                    frame_to_frame_ms: ft_ff,
                    total_gpu_ms: tot_gpu,
                    total_cpu_ms: tot_cpu,
                    pass_timings: pass_clone
                        .iter()
                        .map(|t| PortalPassTiming { name: t.name.clone(), gpu_ms: t.gpu_ms, cpu_ms: t.cpu_ms })
                        .collect(),
                    pipeline_order: pipeline_clone,
                    scene_delta,
                    timestamp_ms: now_ms,
                    object_count: obj_count,
                    light_count,
                    billboard_count: bb_count,
                    draw_calls: helio_live_portal::DrawCallMetrics { total: draw_total, opaque: draw_opaque, transparent: draw_transparent },
                    prep_ms: prep,
                    graph_ms: graphm,
                    aa_ms: aam,
                    resolve_ms: resolve,
                    finish_ms: finish,
                    submit_ms: submitm,
                    poll_ms: pollm,
                    untracked_ms: untracked_val,
                    stage_timings: {
                        let s = &stage_copy;
                        let scene_prep_total: f32 = s.iter().sum();
                        let app_ms = (untracked_val - scene_prep_total).max(0.0);
                        vec![
                            PortalStageTiming { id: "app".into(),          name: "App".into(),              ms: app_ms },
                            PortalStageTiming { id: "scene_draws".into(),  name: "Scene: Draws".into(),     ms: s[0] },
                            PortalStageTiming { id: "scene_lights".into(), name: "Scene: Lights".into(),    ms: s[1] },
                            PortalStageTiming { id: "scene_shadow".into(), name: "Scene: Shadows".into(),   ms: s[2] },
                            PortalStageTiming { id: "scene_bb".into(),     name: "Scene: Billboards".into(),ms: s[3] },
                            PortalStageTiming { id: "scene_sky".into(),    name: "Scene: Sky".into(),       ms: s[4] },
                            PortalStageTiming { id: "prep".into(),         name: "Prep".into(),             ms: prep },
                            PortalStageTiming { id: "pipeline".into(),     name: "Render Pipeline".into(),  ms: graphm },
                            PortalStageTiming { id: "aa".into(),           name: "AA".into(),               ms: aam },
                            PortalStageTiming { id: "resolve".into(),      name: "Resolve".into(),          ms: resolve },
                            PortalStageTiming { id: "finish".into(),       name: "Encode".into(),           ms: finish },
                            PortalStageTiming { id: "submit".into(),       name: "Submit".into(),           ms: submitm },
                            PortalStageTiming { id: "poll".into(),         name: "Poll".into(),             ms: pollm },
                        ]
                    },
                    pipeline_stage_id: Some("pipeline".to_string()),
                };

                let _ = tx.send(snapshot);
            });

            if self.pending_layout_changed {
                self.previous_scene_layout = layout_for_update.clone();
            } else if let (Some(ref mut prev), Some(ref cur)) =
                (&mut self.previous_scene_layout, &layout_for_update)
            {
                prev.camera = cur.camera.clone();
            } else {
                self.previous_scene_layout = layout_for_update.clone();
            }
        }

        // finally, clear the draw list now that portal has sampled it
        self.draw_list.lock().unwrap().clear();

        self.frame_count += 1;
        Ok(())
    }

    // ── Resize ────────────────────────────────────────────────────────────────

    pub fn resize(&mut self, width: u32, height: u32) {
        log::trace!("Resizing renderer to {}x{}", width, height);
        self.width = width;
        self.height = height;

        let (tex, view, sample_view) = helpers::create_depth_texture(&self.device, width, height);
        self.depth_texture = tex;
        self.depth_view = view;
        self.depth_sample_view = sample_view;

        let (albedo_tex, normal_tex, orm_tex, emissive_tex, new_targets) =
            helpers::create_gbuffer_textures(&self.device, width, height);
        self.gbuf_albedo_texture  = albedo_tex;
        self.gbuf_normal_texture  = normal_tex;
        self.gbuf_orm_texture     = orm_tex;
        self.gbuf_emissive_texture = emissive_tex;

        *self.gbuffer_targets.lock().unwrap() = new_targets;

        let new_bg = Arc::new(helpers::create_gbuffer_bind_group(
            &self.device,
            &self.resources.bind_group_layouts.gbuffer_read,
            &*self.gbuffer_targets.lock().unwrap(),
            &self.depth_sample_view,
        ));
        *self.deferred_bg.lock().unwrap() = new_bg;

        self.pre_aa_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Pre-AA Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.pre_aa_texture.format(),
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.pre_aa_view = self.pre_aa_texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.fxaa_bind_group = self.fxaa_pass.as_ref().map(|p| p.create_bind_group(&self.device, &self.pre_aa_view));
        self.smaa_bind_group = self.smaa_pass.as_ref().map(|p| p.create_bind_group(&self.device, &self.pre_aa_view));
        self.taa_bind_group = self.taa_pass.as_ref().map(|p| p.create_bind_group(&self.device, &self.pre_aa_view, &self.depth_view));
    }

    pub fn frame_count(&self) -> u64 { self.frame_count }

    // ── Profiling ─────────────────────────────────────────────────────────────

    pub fn last_frame_timings(&self) -> &[PassTiming] {
        self.profiler.as_ref()
            .map(|p| p.last_timings.as_slice())
            .unwrap_or(&[])
    }

    pub fn print_timings_every(&self, interval: u64) {
        if self.frame_count % interval != 0 { return; }
        if let Some(p) = &self.profiler {
            let timings = p.last_timings.clone();
            let total_gpu   = p.last_total_gpu_ms;
            let total_cpu   = p.last_total_cpu_ms;
            let frame_time  = p.last_frame_time_ms;
            let frame_to_frame = p.last_frame_to_frame_ms;
            std::thread::spawn(move || {
                crate::profiler::GpuProfiler::print_snapshot(timings, total_gpu, total_cpu, frame_time, frame_to_frame);
            });
        } else {
            log::trace!("[TIMING] TIMESTAMP_QUERY unavailable — add wgpu::Features::TIMESTAMP_QUERY to device descriptor");
        }
    }

    pub fn device(&self) -> &wgpu::Device { &self.device }
    pub fn queue(&self) -> &wgpu::Queue { &self.queue }
}
