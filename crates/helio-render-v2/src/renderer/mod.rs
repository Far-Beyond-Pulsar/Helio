//! Main renderer implementation

mod config;
mod init;
mod scene_prep;
mod uniforms;
mod shadow_math;
mod helpers;
mod portal;
mod gpu_light_scene;

pub use config::RendererConfig;

use crate::resources::ResourceManager;
use crate::features::{FeatureRegistry, FeatureContext, PrepareContext, RadianceCascadesFeature};
use crate::pipeline::{PipelineCache, PipelineVariant};
use crate::graph::{RenderGraph, GraphContext};
use crate::passes::{DebugDrawPass, SkyPass, SkyLutPass, SKY_LUT_W, SKY_LUT_H, SKY_LUT_FORMAT, ShadowCullLight, DepthPrepassPass, GBufferPass, GBufferTargets, DeferredLightingPass, TransparentPass, AntiAliasingMode, FxaaPass, SmaaPass, TaaPass, IndirectDispatchPass};
use crate::passes::depth_prepass::{BundleInbox, PrecompiledBundles, sort_opaque_indices, build_depth_bundle};
use crate::passes::gbuffer::build_gbuffer_bundle;
use crate::mesh::{GpuMesh, DrawCall, GpuDrawCall};
use crate::camera::Camera;
use crate::scene::{ObjectId, SceneLight};
use crate::debug_draw::{self, DebugDrawBatch, DebugShape};
use crate::features::BillboardsFeature;
use crate::material::{Material, GpuMaterial, MaterialUniform, DefaultMaterialViews, build_gpu_material};
use crate::profiler::GpuProfiler;
use crate::gpu_transfer;
use crate::gpu_scene::GpuScene;
use crate::{Result, Error};
use helio_live_portal::{
    LivePortalHandle,
    PortalFrameSnapshot,
    PortalPassTiming,
    PortalStageTiming,
    PortalSceneLayout,
    PortalSceneLayoutDelta,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, atomic::{AtomicU32, Ordering}};
use std::time::{SystemTime, UNIX_EPOCH};
use wgpu::util::DeviceExt;

use self::uniforms::GlobalsUniform;
use self::portal::{compute_scene_delta, open_url_in_browser};

/// Lightweight per-frame message sent from the render loop to the portal
/// worker thread.  Avoids per-frame `thread::spawn` and redundant cloning.
struct PortalFrameMsg {
    frame: u64,
    timestamp_ms: u128,
    frame_time_ms: f32,
    frame_to_frame_ms: f32,
    total_gpu_ms: f32,
    total_cpu_ms: f32,
    pass_timings: Vec<crate::profiler::PassTiming>,
    current_layout: Option<PortalSceneLayout>,
    layout_changed: bool,
    draw_total: usize,
    draw_opaque: usize,
    draw_transparent: usize,
    prep_ms: f32,
    graph_ms: f32,
    aa_ms: f32,
    resolve_ms: f32,
    finish_ms: f32,
    submit_ms: f32,
    poll_ms: f32,
    untracked_ms: f32,
    stage_ms: [f32; 5],
    cpu_scope_log: Vec<(&'static str, f32)>,
}

/// Snapshot of per-frame environment data supplied by `set_scene_env`.
/// Kept as a plain struct so `render()` can cheaply read it without locking.
pub struct SceneEnv {
    pub lights:            Vec<SceneLight>,
    pub ambient_color:     [f32; 3],
    pub ambient_intensity: f32,
    pub sky_color:         [f32; 3],
    pub sky_atmosphere:    Option<crate::scene::SkyAtmosphere>,
    pub skylight:          Option<crate::scene::Skylight>,
    pub billboards:        Vec<crate::features::BillboardInstance>,
}

impl Default for SceneEnv {
    fn default() -> Self {
        Self {
            lights:            Vec::new(),
            ambient_color:     [0.0; 3],
            ambient_intensity: 0.0,
            sky_color:         [0.0; 3],
            sky_atmosphere:    None,
            skylight:          None,
            billboards:        Vec::new(),
        }
    }
}

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

    // Draw list (shared with GeometryPass / TransparentPass)
    draw_list: Arc<Mutex<Vec<DrawCall>>>,
    // Debug draw primitives queued by user each frame.
    debug_shapes: Arc<Mutex<Vec<DebugShape>>>,
    // GPU batch built from debug_shapes before graph execution.
    debug_batch: Arc<Mutex<Option<DebugDrawBatch>>>,

    // Light buffer for scene writes
    lighting_shadow_view: Arc<wgpu::TextureView>,
    lighting_shadow_sampler: Arc<wgpu::Sampler>,
    lighting_env_cube_view: Arc<wgpu::TextureView>,
    lighting_rc_view: Arc<wgpu::TextureView>,
    lighting_env_sampler: Arc<wgpu::Sampler>,
    // Shared light count for ShadowPass (updated each frame before graph exec)
    light_count_arc: Arc<AtomicU32>,
    // Per-light active face counts: 6=point, 4=directional, 1=spot
    light_face_counts: Arc<std::sync::Mutex<Vec<u8>>>,
    // Per-light position/range/type for ShadowPass draw-call culling
    shadow_cull_lights: Arc<std::sync::Mutex<Vec<ShadowCullLight>>>,
    // Current scene ambient (updated by prepare_env)
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

    // ── Post-processing ────────────────────────────────────────────────────
    enable_ssao: bool,
    ssao_texture: Option<wgpu::Texture>,
    ssao_view:    Option<wgpu::TextureView>,

    aa_mode: AntiAliasingMode,
    pre_aa_texture: wgpu::Texture,
    pre_aa_view:    wgpu::TextureView,
    fxaa_pass:      Option<FxaaPass>,
    smaa_pass:      Option<SmaaPass>,
    taa_pass:       Option<TaaPass>,
    fxaa_bind_group: Option<wgpu::BindGroup>,
    smaa_bind_group: Option<wgpu::BindGroup>,
    taa_bind_group:  Option<wgpu::BindGroup>,

    // ── GPU-driven indirect rendering ─────────────────────────────────────
    gpu_driven:  bool,
    async_compute: bool,
    indirect_opaque_buffer:      Option<Arc<wgpu::Buffer>>,
    indirect_transparent_buffer: Option<Arc<wgpu::Buffer>>,
    opaque_draw_count:      Arc<AtomicU32>,
    transparent_draw_count: Arc<AtomicU32>,
    draw_list_gpu_buffer: Option<Arc<wgpu::Buffer>>,
    material_id_map:  HashMap<usize, u32>,
    next_material_id: u32,

    // ── Sky LUT caching ───────────────────────────────────────────────────
    cached_sky_color:         [f32; 3],
    cached_sky_has_sky:       bool,
    cached_sky_sun_direction: [f32; 3],
    cached_sky_sun_intensity: f32,
    sky_state_changed:        bool,

    // ── Temporal shadow / light caching ──────────────────────────────────
    /// Monotonically increasing counter.  Bumped only when the registered object
    /// set changes (add_object / remove_object).  Does NOT change on camera moves.
    draw_list_generation: u64,
    /// Number of persistent draw calls at the start of draw_list (from gpu_scene).
    /// One-frame draws from `draw_mesh()` sit after this index and are truncated
    /// at the start of the next `render()`.
    persistent_draw_count: usize,
    /// Generation value when persistent draw_list was last rebuilt.
    cached_draw_list_gen: u64,

    // ── Shadow draw list ──────────────────────────────────────────────────
    /// Pointers to all registered-proxy DrawCalls, rebuilt O(N) each frame by
    /// pointer-cloning.  ShadowPass culls this by light range on its own thread.
    shadow_draw_list: Arc<Mutex<Vec<DrawCall>>>,

    // ── Parallel bundle pre-compilation (Unreal FParallelCommandListSet) ──
    /// Shared inbox through which the Renderer delivers pre-compiled bundles
    /// to DepthPrepassPass and GBufferPass before graph execution.
    bundle_inbox: Arc<BundleInbox>,
    /// Pipeline copies kept on the Renderer for the pre-compile thread pool.
    depth_pipeline: Arc<wgpu::RenderPipeline>,
    gbuffer_pipeline: Arc<wgpu::RenderPipeline>,
    /// draw_list_generation value at the time bundles were last parallel pre-compiled.
    last_precompile_gen: u64,

    // ════════════════════════════════════════════════════════════════════════
    // GPU-RESIDENT SCENE (Unreal FGPUScene equivalent)
    // ════════════════════════════════════════════════════════════════════════
    //
    // All per-object instance data lives in a single GPU storage buffer +
    // a parallel vertex buffer for the mat4 transforms.  Objects are
    // assigned stable slots via a free-list; transforms are uploaded as a
    // single delta write per frame.  Static objects have ZERO CPU→GPU cost.
    //
    // The application API is identical to before:
    //   add_object(mesh, material, transform) -> ObjectId
    //   remove_object(id)
    //   update_transform(id, transform)
    //
    // Internally, add/remove/update delegate to GpuScene which manages
    // slot allocation, dirty tracking, and the per-frame flush.
    // ────────────────────────────────────────────────────────────────────────
    /// GPU-resident scene buffer (slot allocator + dirty tracking + delta upload).
    gpu_scene: GpuScene,
    /// GPU-resident light + shadow matrix buffers (dirty-bit delta uploads).
    gpu_light_scene: gpu_light_scene::GpuLightScene,

    // ── Environment (lights, sky, ambient, billboards) ────────────────────
    /// Set by `set_scene_env` each frame (or once for static scenes).
    /// Fully replaces `render_scene`'s scene-struct argument.
    pending_env: Option<SceneEnv>,

    // Frame state
    frame_count: u64,
    width: u32,
    height: u32,
    last_frame_start: Option<std::time::Instant>,
    last_frame_end: Option<std::time::Instant>,

    /// GPU + CPU pass-level profiler.  `None` when TIMESTAMP_QUERY is unavailable.
    profiler: Option<GpuProfiler>,

    /// Optional live web dashboard handle for real-time pipeline/perf telemetry.
    live_portal: Option<LivePortalHandle>,
    /// Last captured scene layout snapshot forwarded to the portal thread.
    latest_scene_layout: Option<PortalSceneLayout>,
    /// Previous scene layout for delta computation (reduces bandwidth).
    previous_scene_layout: Option<PortalSceneLayout>,
    /// Per-stage CPU timings (ms) from the most recent env-prepare call.
    pending_scene_stage_ms: [f32; 5],
    /// True when object/light/billboard count changed this frame.
    pending_layout_changed: bool,
    portal_scene_key: (usize, usize, usize),
    /// Pass names cached once after `graph.build()` to avoid a `Vec<String>` alloc every frame.
    cached_pass_names: Vec<String>,

    /// Channel to the persistent portal worker thread (spawned once in
    /// `start_live_portal`).  `SyncSender` with a small bound means the
    /// main thread never blocks — if the worker is behind we just drop the
    /// frame via `try_send`.
    portal_worker_tx: Option<std::sync::mpsc::SyncSender<PortalFrameMsg>>,
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

        // Spawn persistent worker thread.  Bounded channel (4 frames) means
        // try_send on the main thread is non-blocking; if the worker falls
        // behind we simply drop the frame data.
        let (wtx, wrx) = std::sync::mpsc::sync_channel::<PortalFrameMsg>(4);
        let portal_tx = handle.sender();
        let pipeline_order = self.cached_pass_names.clone();
        std::thread::Builder::new()
            .name("helio-portal-worker".into())
            .spawn(move || portal_worker_loop(wrx, portal_tx, pipeline_order))
            .expect("failed to spawn portal worker thread");
        self.portal_worker_tx = Some(wtx);

        self.live_portal = Some(handle);
        Ok(url)
    }

    /// Convenience: start live portal on the default port.
    pub fn start_live_portal_default(&mut self) -> Result<String> {
        self.start_live_portal("0.0.0.0:7878")
    }

    pub fn is_gpu_driven(&self) -> bool { self.gpu_driven }
    pub fn is_async_compute(&self) -> bool { self.async_compute }

    // ── Render-pass toggle ────────────────────────────────────────────────────

    /// Toggle a render pass on/off by name.  Returns the new enabled state.
    pub fn toggle_pass(&mut self, name: &str) -> bool {
        self.graph.toggle_pass(name)
    }

    /// Set whether a render pass is enabled.
    pub fn set_pass_enabled(&mut self, name: &str, enabled: bool) {
        self.graph.set_pass_enabled(name, enabled);
    }

    /// Returns true if the named pass is currently enabled.
    pub fn is_pass_enabled(&self, name: &str) -> bool {
        self.graph.is_pass_enabled(name)
    }

    /// Return all pass names in execution order.
    pub fn pass_names(&self) -> Vec<String> {
        self.graph.execution_pass_names()
    }

    // ══════════════════════════════════════════════════════════════════════════
    // PERSISTENT SCENE PROXY API  (Unreal FScene::AddPrimitive / RemovePrimitive)
    // ══════════════════════════════════════════════════════════════════════════

    /// Register a mesh into the GPU-resident scene.
    ///
    /// Equivalent to `UPrimitiveComponent::RegisterComponent()` → `FScene::AddPrimitive()`.
    /// Allocates a slot in the GPU scene buffer and writes the initial transform.
    /// Returns a stable [`ObjectId`] for the object's lifetime in the scene.
    ///
    /// Cost: one slot alloc + one CPU-mirror write.  GPU upload is deferred to
    /// `flush()` which batches all dirty slots into a single `write_buffer`.
    pub fn add_object(
        &mut self,
        mesh:      &GpuMesh,
        material:  Option<&GpuMaterial>,
        transform: glam::Mat4,
    ) -> ObjectId {
        let id = self.gpu_scene.add_object(
            &self.device,
            mesh,
            material,
            &self.default_material_bind_group,
            transform,
        );

        // Structural change bumps generation so TransparentPass invalidates cached sorts.
        self.draw_list_generation = self.gpu_scene.generation;

        id
    }

    /// Remove an object from the GPU-resident scene.
    ///
    /// Equivalent to `UPrimitiveComponent::UnregisterComponent()` → `FScene::RemovePrimitive()`.
    /// Frees the slot in the GPU scene buffer; the zeroed slot is uploaded next flush.
    ///
    /// Cost: one slot free + one CPU-mirror zero.
    pub fn remove_object(&mut self, id: ObjectId) {
        self.gpu_scene.remove_object(id);
        self.draw_list_generation = self.gpu_scene.generation;
    }

    /// Update the world transform of a registered object.
    ///
    /// Equivalent to `USceneComponent::SendRenderTransform()` → proxy `SetTransform()`.
    /// Only marks the slot dirty if the FNV-1a hash changed — static geometry has zero cost.
    /// GPU upload is deferred to `flush()`.
    pub fn update_transform(&mut self, id: ObjectId, transform: glam::Mat4) {
        self.gpu_scene.update_transform(id, transform);
    }

    /// Batch-update many transforms in a single call.
    pub fn update_transforms(&mut self, updates: &[(ObjectId, glam::Mat4)]) {
        self.gpu_scene.update_transforms(updates);
    }

    /// Number of objects currently registered in the GPU scene.
    pub fn object_count(&self) -> usize {
        self.gpu_scene.object_count() as usize
    }

    /// Enable an object so it appears in all draw lists.
    pub fn enable_object(&mut self, id: ObjectId) {
        self.gpu_scene.set_object_enabled(id, true);
        self.draw_list_generation = self.gpu_scene.generation;
    }

    /// Disable an object so it is excluded from all draw lists (keeps its GPU slot).
    pub fn disable_object(&mut self, id: ObjectId) {
        self.gpu_scene.set_object_enabled(id, false);
        self.draw_list_generation = self.gpu_scene.generation;
    }

    /// Returns `true` if the object exists and is currently enabled.
    pub fn is_object_enabled(&self, id: ObjectId) -> bool {
        self.gpu_scene.is_object_enabled(id)
    }

    /// Override the bounding sphere radius used for culling for a registered object.
    pub fn set_object_bounds(&mut self, id: ObjectId, radius: f32) {
        self.gpu_scene.set_object_bounds(id, radius);
    }

    // ══════════════════════════════════════════════════════════════════════════
    // SCENE ENVIRONMENT API  (lights, sky, ambient, billboards)
    // ══════════════════════════════════════════════════════════════════════════

    /// Supply the per-frame environment snapshot: lights, sky, ambient, and billboards.
    ///
    /// Call once per frame (or whenever any environmental value changes).  The renderer
    /// consumes this at the top of the next `render()` call via `prepare_env` and then
    /// discards it — no allocation carry-over between frames.
    ///
    /// This replaces the old `render_scene(&Scene, ...)` call for the light/sky portion.
    /// Registered objects are completely independent and never need to appear here.
    pub fn set_scene_env(&mut self, env: SceneEnv) {
        self.pending_env = Some(env);
    }



    /// Render a frame.
    ///
    /// Before calling this, supply the per-frame environment via `set_scene_env()`.
    /// Registered objects (added via `add_object`) are included automatically —
    /// no other per-frame object submission is required at steady state.
    pub fn render(&mut self, camera: &Camera, target: &wgpu::TextureView, delta_time: f32) -> Result<()> {
        let frame_start = std::time::Instant::now();
        log::trace!("Rendering frame {}", self.frame_count);

        // ── Step 0: process pending env (lights/sky/billboards) ──────────────
        if let Some(env) = self.pending_env.take() {
            self.prepare_env(env, camera);
        }

        // ── Step 1: flush GPU scene (single delta write_buffer) ───────────────
        //
        // Uploads only the dirty slot range to both the storage buffer (for future
        // compute culling) and the vertex buffer (for draw-call instance binding).
        // At steady state with no moved objects, this is a complete no-op.
        self.gpu_scene.flush(&self.queue);

        // ── Step 2: populate draw_list from GPU scene (persistent cache) ────
        //
        // The draw list is rebuilt ONLY when gpu_scene.generation changes
        // (add_object / remove_object / update_material).  At steady state
        // this block does NOTHING — zero Arc clones, zero iteration.
        //
        // One-frame draws from `draw_mesh()` that were pushed between frames
        // sit after the persistent portion and are truncated here.
        {
            crate::profile_scope!("dl_rebuild");
            let mut dl  = self.draw_list.lock().unwrap();
            let mut sdl = self.shadow_draw_list.lock().unwrap();

            if self.gpu_scene.generation != self.cached_draw_list_gen {
                // Structural change — full rebuild from gpu_scene cache.
                let (scene_dl, scene_sdl) = self.gpu_scene.draw_lists();

                dl.clear();
                dl.extend_from_slice(scene_dl);

                sdl.clear();
                sdl.extend_from_slice(scene_sdl);

                self.persistent_draw_count = dl.len();
                self.cached_draw_list_gen = self.gpu_scene.generation;
            } else {
                // Steady state — trim any one-frame draws from previous frame.
                dl.truncate(self.persistent_draw_count);
                sdl.truncate(self.persistent_draw_count);
            }
        }

        // ── Preparation: camera upload, feature prepare, globals, debug batch, GPU-driven ──
        let prep_start = std::time::Instant::now();

        // Upload camera uniform (features may use camera-dependent logic in prepare).
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera));
        gpu_transfer::track_upload(std::mem::size_of::<Camera>() as u64);

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
        gpu_transfer::track_upload(std::mem::size_of::<GlobalsUniform>() as u64);

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

        // ── Parallel bundle pre-compilation (Unreal FParallelCommandListSet) ────
        //
        // When the draw list changes (add/remove/enable/disable object), split the
        // sorted opaque draw list into N chunks and compile each chunk into its own
        // RenderBundle on a worker thread.  This is equivalent to:
        //   UE4: FParallelCommandListSet → RHICmdList[i].DrawIndexedPrimitive(…)
        //   D3D12: N secondary command lists → ExecuteCommandLists(N, pLists)
        //   wgpu:  N RenderBundles → pass.execute_bundles(&bundles)
        //
        // The pre-compiled bundles are deposited into the shared BundleInbox.
        // DepthPrepassPass and GBufferPass pick them up in their execute() methods,
        // avoiding any inline compilation.  At steady state (no scene changes),
        // this block is skipped entirely.
        if self.draw_list_generation != self.last_precompile_gen {
            crate::profile_scope!("parallel_precompile");
            let draw_calls: Vec<DrawCall> = self.draw_list.lock().unwrap().clone();
            let sorted = sort_opaque_indices(&draw_calls);

            if sorted.len() >= 64 {
                let num_chunks = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4)
                    .min(8)
                    .max(1);
                let chunk_size = (sorted.len() + num_chunks - 1) / num_chunks;

                // Rebind Arc-wrapped fields to local shared refs so scoped threads
                // can capture them without borrowing `self`.
                let device: &wgpu::Device        = &self.device;
                let depth_pl: &wgpu::RenderPipeline  = &self.depth_pipeline;
                let gbuf_pl: &wgpu::RenderPipeline   = &self.gbuffer_pipeline;
                let global_bg: &wgpu::BindGroup  = &self.global_bind_group;
                let lighting_bg: &wgpu::BindGroup = &self.lighting_bind_group;
                let generation     = self.draw_list_generation;
                let global_bg_ptr  = global_bg as *const _ as usize;
                let lighting_bg_ptr = lighting_bg as *const _ as usize;
                let dc_ref: &[DrawCall] = &draw_calls;
                let sorted_ref: &[usize] = &sorted;

                // Pre-collect chunks so spawn closures capture simple slices.
                let chunks: Vec<&[usize]> = sorted_ref.chunks(chunk_size).collect();

                let (depth_bundles, depth_kept, gbuf_bundles, gbuf_kept) = std::thread::scope(|s| {
                    // Spawn N depth-prepass chunk threads.
                    let depth_handles: Vec<_> = chunks.iter()
                        .map(|&chunk| {
                            s.spawn(move || build_depth_bundle(device, depth_pl, dc_ref, chunk, global_bg, lighting_bg))
                        })
                        .collect();

                    // Spawn N gbuffer chunk threads (in parallel with depth).
                    let gbuf_handles: Vec<_> = chunks.iter()
                        .map(|&chunk| {
                            s.spawn(move || build_gbuffer_bundle(device, gbuf_pl, dc_ref, chunk, global_bg))
                        })
                        .collect();

                    // Collect depth results.
                    let mut d_bundles = Vec::with_capacity(depth_handles.len());
                    let mut d_kept = Vec::new();
                    for h in depth_handles {
                        let (bundle, kept) = h.join().unwrap();
                        d_bundles.push(bundle);
                        d_kept.extend(kept);
                    }

                    // Collect gbuffer results.
                    let mut g_bundles = Vec::with_capacity(gbuf_handles.len());
                    let mut g_kept = Vec::new();
                    for h in gbuf_handles {
                        let (bundle, kept) = h.join().unwrap();
                        g_bundles.push(bundle);
                        g_kept.extend(kept);
                    }

                    (d_bundles, d_kept, g_bundles, g_kept)
                });

                eprintln!(
                    "⚠️ [Parallel Precompile] {} depth + {} gbuffer bundles, {} chunks, {} draws (gen {})",
                    depth_bundles.len(), gbuf_bundles.len(), num_chunks, sorted.len(), generation,
                );

                // Deposit into inbox — passes will .take() during graph execution.
                let sorted_clone = sorted.clone();
                *self.bundle_inbox.depth_prepass.lock().unwrap() = Some(PrecompiledBundles {
                    bundles: depth_bundles,
                    kept_arcs: depth_kept,
                    sorted_indices: sorted_clone,
                    generation,
                    global_bg_ptr,
                    lighting_bg_ptr,
                });
                *self.bundle_inbox.gbuffer.lock().unwrap() = Some(PrecompiledBundles {
                    bundles: gbuf_bundles,
                    kept_arcs: gbuf_kept,
                    sorted_indices: sorted,
                    generation,
                    global_bg_ptr,
                    lighting_bg_ptr,
                });
            }

            self.last_precompile_gen = self.draw_list_generation;
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
                    let draw_bytes = (gpu_draws.len() * std::mem::size_of::<GpuDrawCall>()) as u64;
                    self.queue.write_buffer(draw_list_buf, 0, bytemuck::cast_slice(&gpu_draws));
                    gpu_transfer::track_upload(draw_bytes);
                    log::trace!("GPU-driven: Uploaded {} draw calls for indirect rendering", gpu_draws.len());
                }
                if let (Some(ref _opaque_buf), Some(ref _transparent_buf)) =
                    (&self.indirect_opaque_buffer, &self.indirect_transparent_buffer) {
                    self.opaque_draw_count.store(0, Ordering::Release);
                    self.transparent_draw_count.store(0, Ordering::Release);
                }
            }
        }

        // ── Encoder + pre-AA clear (still part of prep) ──────────────────────
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

        let prep_ms = prep_start.elapsed().as_secs_f32() * 1000.0;

        // ── Execute render graph ──────────────────────────────────────────────
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

        let profiling_active = self.live_portal.is_some();

        let graph_start = std::time::Instant::now();
        if profiling_active {
            if let Some(p) = &mut self.profiler { p.begin_frame(); }
            self.graph.execute(&mut graph_ctx, self.profiler.as_mut())?;
        } else {
            self.graph.execute(&mut graph_ctx, None)?;
        }
        let graph_ms = graph_start.elapsed().as_secs_f32() * 1000.0;
        // Collect CPU scope timings emitted by profile_scope! calls inside the passes.
        let cpu_scope_log = crate::profiler::take_frame_scope_log();

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

        // ── Live portal snapshot ──────────────────────────────────────────────
        // Pack cheap scalars + one `take`'d Vec into a message and hand it to
        // the persistent worker thread.  No thread::spawn, no redundant clones.
        if let (Some(wtx), Some(p)) = (&self.portal_worker_tx, &mut self.profiler) {
            let (draw_total, draw_opaque, draw_transparent) = {
                let draws = self.draw_list.lock().unwrap();
                let total = draws.len();
                let opaque = draws.iter().filter(|dc| !dc.transparent_blend).count();
                (total, opaque, total.saturating_sub(opaque))
            };

            let msg = PortalFrameMsg {
                frame: self.frame_count,
                timestamp_ms: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_millis())
                    .unwrap_or(0),
                frame_time_ms,
                frame_to_frame_ms,
                total_gpu_ms: p.last_total_gpu_ms,
                total_cpu_ms: p.last_total_cpu_ms,
                // take() is O(1) — profiler refills this next frame from readback.
                pass_timings: std::mem::take(&mut p.last_timings),
                current_layout: self.latest_scene_layout.take(),
                layout_changed: self.pending_layout_changed,
                draw_total,
                draw_opaque,
                draw_transparent,
                prep_ms,
                graph_ms,
                aa_ms,
                resolve_ms,
                finish_ms,
                submit_ms,
                poll_ms,
                untracked_ms,
                stage_ms: self.pending_scene_stage_ms,
                cpu_scope_log,
            };
            // Non-blocking: if the worker is behind, drop this frame's data.
            let _ = wtx.try_send(msg);
        }

        // One-frame draws (from draw_mesh()) sit after `persistent_draw_count`
        // and are truncated at the start of the next render() call.
        // The persistent portion survives across frames — zero-cost at steady state.

        self.frame_count += 1;
        gpu_transfer::end_frame();
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

    pub fn device(&self) -> &wgpu::Device { &self.device }
    pub fn queue(&self) -> &wgpu::Queue { &self.queue }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PORTAL WORKER THREAD
// ═══════════════════════════════════════════════════════════════════════════════

/// Persistent background thread that assembles `PortalFrameSnapshot` from the
/// lightweight `PortalFrameMsg` and forwards it to the portal bridge.
/// Owns the previous scene layout so the main thread never clones it.
fn portal_worker_loop(
    rx: std::sync::mpsc::Receiver<PortalFrameMsg>,
    portal_tx: std::sync::mpsc::Sender<PortalFrameSnapshot>,
    pipeline_order: Vec<String>,
) {
    let mut previous_layout: Option<PortalSceneLayout> = None;

    while let Ok(m) = rx.recv() {
        let scene_delta = m.current_layout.as_ref().map(|cur| {
            if m.layout_changed {
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

        let (obj_count, light_count, bb_count) = if let Some(l) = &m.current_layout {
            (l.objects.len(), l.lights.len(), l.billboards.len())
        } else {
            (0, 0, 0)
        };

        let s = &m.stage_ms;
        let scene_prep_total: f32 = s.iter().sum();
        let app_ms = (m.untracked_ms - scene_prep_total).max(0.0);
        let scope_map: HashMap<&str, f32> =
            m.cpu_scope_log.iter().map(|(n, ms)| (*n, *ms)).collect();
        let mut stage_timings = vec![
            PortalStageTiming { id: "app".into(),          name: "App".into(),              ms: app_ms },
            PortalStageTiming { id: "scene_draws".into(),  name: "Scene: Draws".into(),     ms: s[0] },
            PortalStageTiming { id: "scene_lights".into(), name: "Scene: Lights".into(),    ms: s[1] },
            PortalStageTiming { id: "scene_shadow".into(), name: "Scene: Shadows".into(),   ms: s[2] },
            PortalStageTiming { id: "scene_bb".into(),     name: "Scene: Billboards".into(),ms: s[3] },
            PortalStageTiming { id: "scene_sky".into(),    name: "Scene: Sky".into(),       ms: s[4] },
            PortalStageTiming { id: "prep".into(),         name: "Prep".into(),             ms: m.prep_ms },
            PortalStageTiming { id: "pipeline".into(),     name: "Render Pipeline".into(),  ms: m.graph_ms },
            PortalStageTiming { id: "aa".into(),           name: "AA".into(),               ms: m.aa_ms },
            PortalStageTiming { id: "resolve".into(),      name: "Resolve".into(),          ms: m.resolve_ms },
            PortalStageTiming { id: "finish".into(),       name: "Encode".into(),           ms: m.finish_ms },
            PortalStageTiming { id: "submit".into(),       name: "Submit".into(),           ms: m.submit_ms },
            PortalStageTiming { id: "poll".into(),         name: "Poll".into(),             ms: m.poll_ms },
        ];
        for &(name, id) in &[
            ("dl_rebuild",            "dl_rebuild"),
            ("depth_prepass/compile", "depth_prepass_compile"),
            ("depth_prepass/replay",  "depth_prepass_replay"),
            ("gbuffer/compile",       "gbuffer_compile"),
            ("gbuffer/replay",        "gbuffer_replay"),
        ] {
            let ms = scope_map.get(name).copied().unwrap_or(0.0);
            stage_timings.push(PortalStageTiming {
                id: id.to_string(), name: name.to_string(), ms,
            });
        }

        let snapshot = PortalFrameSnapshot {
            frame: m.frame,
            frame_time_ms: m.frame_time_ms,
            frame_to_frame_ms: m.frame_to_frame_ms,
            total_gpu_ms: m.total_gpu_ms,
            total_cpu_ms: m.total_cpu_ms,
            pass_timings: m.pass_timings
                .iter()
                .map(|t| PortalPassTiming { name: t.name.clone(), gpu_ms: t.gpu_ms, cpu_ms: t.cpu_ms })
                .collect(),
            pipeline_order: pipeline_order.clone(),
            scene_delta,
            timestamp_ms: m.timestamp_ms,
            object_count: obj_count,
            light_count,
            billboard_count: bb_count,
            draw_calls: helio_live_portal::DrawCallMetrics {
                total: m.draw_total,
                opaque: m.draw_opaque,
                transparent: m.draw_transparent,
            },
            prep_ms: m.prep_ms,
            graph_ms: m.graph_ms,
            aa_ms: m.aa_ms,
            resolve_ms: m.resolve_ms,
            finish_ms: m.finish_ms,
            submit_ms: m.submit_ms,
            poll_ms: m.poll_ms,
            untracked_ms: m.untracked_ms,
            stage_timings,
            pipeline_stage_id: Some("pipeline".to_string()),
        };

        // Update previous layout tracking.
        if m.layout_changed {
            previous_layout = m.current_layout;
        } else if let (Some(ref mut prev), Some(ref cur)) =
            (&mut previous_layout, &m.current_layout)
        {
            prev.camera = cur.camera.clone();
        } else if previous_layout.is_none() {
            previous_layout = m.current_layout;
        }

        let _ = portal_tx.send(snapshot);
    }
}
