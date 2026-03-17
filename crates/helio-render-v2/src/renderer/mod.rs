//! Main renderer implementation

mod config;
mod init;
mod scene_prep;
mod uniforms;
mod shadow_math;
mod helpers;
#[cfg(feature = "live-portal")]
mod portal;
mod gpu_light_scene;

pub use config::RendererConfig;

use crate::resources::ResourceManager;
use crate::features::{FeatureRegistry, FeatureContext, PrepareContext, RadianceCascadesFeature};
use crate::pipeline::{PipelineCache, PipelineVariant};
use crate::graph::{RenderGraph, GraphContext};
use crate::passes::{DebugDrawPass, SkyPass, SkyLutPass, SKY_LUT_W, SKY_LUT_H, SKY_LUT_FORMAT, ShadowCullLight, DepthPrepassPass, GBufferPass, GBufferTargets, DeferredLightingPass, TransparentPass, AntiAliasingMode, FxaaPass, SmaaPass, TaaPass, IndirectDispatchPass, ShadowMatrixPass};
use crate::mesh::{GpuMesh, DrawCall, PackedVertex};
use crate::camera::Camera;
use crate::scene::{ObjectId, SceneLight, LightId, BillboardId};
use crate::debug_draw::{self, DebugDrawBatch, DebugShape};
use crate::debug_viz;
use crate::features::BillboardsFeature;
use crate::material::{Material, GpuMaterial, MaterialUniform, DefaultMaterialViews, build_gpu_material};
use crate::profiler::GpuProfiler;
use crate::gpu_transfer;
use crate::gpu_scene::{GpuScene, MaterialRange};
use crate::buffer_pool::GpuBufferPool;
use crate::{Result, Error};

// Portal/dashboard support is optional.  it brings in a large async
// networking stack which doesn't compile for wasm, so gate it with a feature.
#[cfg(feature = "live-portal")]
use helio_live_portal::{
    LivePortalHandle,
    PortalFrameSnapshot,
    PortalPassTiming,
    PortalStageTiming,
    PortalSceneLayout,
    PortalSceneLayoutDelta,
    PortalSceneObject,
    PortalSceneLight,
    PortalSceneBillboard,
    PortalSceneCamera,
};
use std::sync::{Arc, Mutex, atomic::{AtomicU32, Ordering}};
use std::time::{SystemTime, UNIX_EPOCH};
use crate::time::Instant;
use wgpu::util::DeviceExt;

use self::uniforms::GlobalsUniform;
#[cfg(feature = "live-portal")]
use self::portal::{compute_scene_delta, open_url_in_browser};

/// Persistent scene environment — ambient, sky, and their dirty flags.
///
/// Written by `set_ambient` / `set_sky_atmosphere` / `set_skylight` / `set_sky_color`.
/// Read once per frame in `flush_scene_state`.  At steady state (nothing changed)
/// all dirty flags are false → zero CPU work, zero GPU uploads.
struct SceneState {
    ambient_color:     [f32; 3],
    ambient_intensity: f32,
    sky_color:         [f32; 3],
    sky_atmosphere:    Option<crate::scene::SkyAtmosphere>,
    skylight:          Option<crate::scene::Skylight>,

    /// SkyLutPass needs to re-render + SkyUniform needs upload.
    sky_lut_dirty:   bool,
    /// RC ambient color needs update.
    ambient_dirty:   bool,
    /// Cached for sun-direction change detection.
    cached_sun_direction: [f32; 3],
    cached_sun_intensity: f32,
}

impl Default for SceneState {
    fn default() -> Self {
        Self {
            ambient_color:     [0.0; 3],
            ambient_intensity: 0.0,
            sky_color:         [0.0; 3],
            sky_atmosphere:    None,
            skylight:          None,
            sky_lut_dirty:     true,  // first frame always renders LUT
            ambient_dirty:     true,
            cached_sun_direction: [0.0, -1.0, 0.0],
            cached_sun_intensity: 1.0,
        }
    }
}

/// Legacy per-frame environment snapshot — now internal only.
///
/// Use the persistent API instead:
///   `add_light` / `remove_light` / `update_light` for lights,
///   `add_billboard` / `remove_billboard` / `update_billboard` for billboards,
///   `set_ambient` / `set_sky_atmosphere` / `set_skylight` / `set_sky_color` for sky.
struct SceneEnv {
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
    /// Kept alive so lighting_bind_group resources are valid; not read after init.
    _lighting_layout: Arc<wgpu::BindGroupLayout>,
    default_material_bind_group: Arc<wgpu::BindGroup>,

    // Default 1×1 texture views + sampler shared by all materials
    default_material_views: DefaultMaterialViews,

    // Opaque+transparent draw list — used by TransparentPass and RadianceCascadesPass.
    draw_list: Arc<Mutex<Vec<DrawCall>>>,
    // Debug draw primitives queued by user each frame.
    debug_shapes: Arc<Mutex<Vec<DebugShape>>>,
    // GPU batch built from debug_shapes before graph execution.
    debug_batch: Arc<Mutex<Option<DebugDrawBatch>>>,

    // Light buffer for scene writes — views/samplers kept alive for lighting bind group.
    _lighting_shadow_view: Arc<wgpu::TextureView>,
    _lighting_shadow_sampler: Arc<wgpu::Sampler>,
    _lighting_env_cube_view: Arc<wgpu::TextureView>,
    _lighting_rc_view: Arc<wgpu::TextureView>,
    _lighting_env_sampler: Arc<wgpu::Sampler>,
    // Shared light count for ShadowPass (updated each frame before graph exec)
    light_count_arc: Arc<AtomicU32>,
    // Per-light active face counts: 6=point, 4=directional, 1=spot
    light_face_counts: Arc<std::sync::Mutex<Vec<u8>>>,
    // Per-light position/range/type for ShadowPass draw-call culling
    shadow_cull_lights: Arc<std::sync::Mutex<Vec<ShadowCullLight>>>,
    // Current scene ambient / sky state (updated by flush_scene_state)
    scene_state: SceneState,
    /// Cached light count forwarded to GlobalsUniform (from gpu_light_scene).
    scene_light_count: u32,
    /// Resolved ambient color (base + skylight contribution) for GlobalsUniform.
    scene_ambient_color: [f32; 3],
    scene_ambient_intensity: f32,
    /// Sky color forwarded to GraphContext.
    scene_sky_color: [f32; 3],
    scene_has_sky: bool,
    /// CSM cascade split distances uploaded each frame to GlobalsUniform.
    scene_csm_splits: [f32; 4],
    // RC world bounds (set from RadianceCascadesFeature, zeroed if disabled)
    rc_world_min: [f32; 3],
    rc_world_max: [f32; 3],

    // Sky pass resources
    sky_uniform_buffer: wgpu::Buffer,
    /// Held here to keep sky bind group alive; SkyPass holds its own Arc.
    _sky_bind_group: Arc<wgpu::BindGroup>,

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
    gbuf_specular_texture: wgpu::Texture,
    /// Shared with GBufferPass.  Swapped on resize.
    gbuffer_targets: Arc<Mutex<GBufferTargets>>,
    /// Shared with DeferredLightingPass.  Swapped on resize.
    deferred_bg: Arc<Mutex<Arc<wgpu::BindGroup>>>,

    // ── Post-processing ────────────────────────────────────────────────────
    // TODO: SSAO pass is not yet wired into the render graph; kept as placeholder.
    _enable_ssao: bool,
    _ssao_texture: Option<wgpu::Texture>,
    _ssao_view:    Option<wgpu::TextureView>,

    aa_mode: AntiAliasingMode,
    pre_aa_texture: wgpu::Texture,
    pre_aa_view:    wgpu::TextureView,
    fxaa_pass:      Option<FxaaPass>,
    smaa_pass:      Option<SmaaPass>,
    taa_pass:       Option<TaaPass>,
    fxaa_bind_group: Option<wgpu::BindGroup>,
    smaa_bind_group: Option<wgpu::BindGroup>,
    taa_bind_group:  Option<wgpu::BindGroup>,

    // ── Sky LUT change flag (forwarded to GraphContext each frame) ────────
    sky_state_changed: bool,

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

    // ── GPU-driven indirect rendering ─────────────────────────────────────
    /// Unified geometry pool (128 MB VB + 64 MB IB). All pool-allocated meshes share these buffers.
    buffer_pool: GpuBufferPool,
    /// Indirect dispatch pass (held here, NOT in graph — called manually before graph).
    indirect_dispatch: IndirectDispatchPass,
    /// Shadow matrix computation pass (GPU-driven, called before shadow pass).
    shadow_matrix_pass: ShadowMatrixPass,
    /// Indirect draw command buffer written by IndirectDispatchPass each frame. Shared with geometry passes.
    shared_indirect_buf: Arc<Mutex<Option<Arc<wgpu::Buffer>>>>,
    /// Per-material draw ranges in the indirect buffer (opaque). Shared with geometry passes.
    shared_material_ranges: Arc<Mutex<Vec<MaterialRange>>>,
    /// Raw GpuScene draw-call buffer, refreshed each frame. Used by ShadowPass for GPU indirect shadow cull.
    shared_shadow_draw_call_buf: Arc<Mutex<Option<Arc<wgpu::Buffer>>>>,

    // Frame state
    frame_count: u64,
    width: u32,
    height: u32,
    last_frame_start: Option<Instant>,
    last_frame_end: Option<Instant>,

    /// GPU + CPU pass-level profiler.  `None` when TIMESTAMP_QUERY is unavailable.
    profiler: Option<GpuProfiler>,

    #[cfg(feature = "live-portal")]
    /// Optional live web dashboard handle for real-time pipeline/perf telemetry.
    live_portal: Option<LivePortalHandle>,
    #[cfg(feature = "live-portal")]
    /// Last captured scene layout snapshot forwarded to the portal thread.
    latest_scene_layout: Option<PortalSceneLayout>,
    #[cfg(feature = "live-portal")]
    /// Previous scene layout for delta computation (reduces bandwidth).
    previous_scene_layout: Option<PortalSceneLayout>,
    #[cfg(feature = "live-portal")]
    /// True when object/light/billboard count changed this frame.
    pending_layout_changed: bool,
    #[cfg(feature = "live-portal")]
    portal_scene_key: (usize, usize, usize),
    /// Pass names cached once after `graph.build()` to avoid a `Vec<String>` alloc every frame.
    cached_pass_names: Vec<String>,

    // ── Debug Visualization System ────────────────────────────────────────
    /// Pluggable overlay system.  Toggle master switch with F3; individual
    /// renderers can be enabled via `debug_viz_mut().set_enabled(name, flag)`.
    debug_viz: debug_viz::DebugVizSystem,
    /// When true, each registered light automatically gets an icon billboard
    /// at its world position, and the billboard follows light transforms.
    editor_mode: bool,
    /// Tracks the billboard icon spawned per light so it can be updated/removed.
    editor_billboard_ids: std::collections::HashMap<LightId, BillboardId>,
    /// Shader debug visualization mode: 0=normal, 1=UV grid, 2=texture direct
    debug_mode: u32,
}

impl Renderer {
    // ── Draw submission ───────────────────────────────────────────────────────

    /// Queue a mesh to be drawn this frame using the default white material
    pub fn draw_mesh(&self, mesh: &GpuMesh) {
        self.draw_list.lock().unwrap().push(DrawCall::new(mesh, 0, self.default_material_bind_group.clone(), false));
    }

    /// Queue a mesh with a custom GPU material (preserves transparency mode)
    pub fn draw_mesh_with_gpu_material(&self, mesh: &GpuMesh, material: &GpuMaterial) {
        self.draw_list.lock().unwrap().push(DrawCall::new(mesh, 0, material.bind_group.clone(), material.transparent_blend));
    }

    /// Queue a mesh with a custom material bind group (legacy opaque path)
    pub fn draw_mesh_with_material(&self, mesh: &GpuMesh, material: Arc<wgpu::BindGroup>) {
        self.draw_list.lock().unwrap().push(DrawCall::new(mesh, 0, material, false));
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

    // ── Pool mesh creation ────────────────────────────────────────────────────

    /// Upload raw geometry into the unified geometry pool.
    ///
    /// Returns a pool-allocated `GpuMesh` when the pool has capacity.
    /// When the pool has reached the VRAM cap, falls back transparently to a
    /// standalone sys-mem buffer (`pool_allocated = false`); a throttled warning
    /// is emitted by the pool.  Callers never need to change.
    pub fn create_mesh(&mut self, vertices: &[PackedVertex], indices: &[u32]) -> GpuMesh {
        if let Some(mesh) = GpuMesh::upload_to_pool(&self.queue, &mut self.buffer_pool, vertices, indices) {
            return mesh;
        }
        // Pool at VRAM cap — fall back to per-mesh sys-mem buffers.
        GpuMesh::upload_standalone(&self.device, vertices, indices)
    }

    pub fn create_mesh_unit_cube(&mut self) -> GpuMesh {
        let (v, i) = GpuMesh::cube_data([0.0, 0.0, 0.0], 0.5);
        self.create_mesh(&v, &i)
    }

    pub fn create_mesh_cube(&mut self, center: [f32; 3], half_size: f32) -> GpuMesh {
        let (v, i) = GpuMesh::cube_data(center, half_size);
        self.create_mesh(&v, &i)
    }

    pub fn create_mesh_plane(&mut self, center: [f32; 3], half_extent: f32) -> GpuMesh {
        let (v, i) = GpuMesh::plane_data(center, half_extent);
        self.create_mesh(&v, &i)
    }

    pub fn create_mesh_rect3d(&mut self, center: [f32; 3], half_extents: [f32; 3]) -> GpuMesh {
        let (v, i) = GpuMesh::rect3d_data(center, half_extents);
        self.create_mesh(&v, &i)
    }

    pub fn create_mesh_sphere(&mut self, center: [f32; 3], radius: f32, subdivisions: u32) -> GpuMesh {
        let (v, i) = GpuMesh::sphere_data(center, radius, subdivisions);
        self.create_mesh(&v, &i)
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
        #[cfg(feature = "live-portal")]
        {
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
        #[cfg(not(feature = "live-portal"))]
        {
            Err(Error::Resource("live portal feature not enabled".into()))
        }
    }

    /// Convenience: start live portal on the default port.
    pub fn start_live_portal_default(&mut self) -> Result<String> {
        self.start_live_portal("0.0.0.0:7878")
    }

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

    /// Upload a Material to the GPU and return a GpuMaterial that can be used with add_object.
    ///
    /// This uploads the material uniforms (base color, metallic, roughness, etc.) and all
    /// textures (base_color, normal, ORM, emissive) to the GPU, creating a bind group.
    ///
    /// # Example
    /// ```no_run
    /// # use helio_render_v2::*;
    /// # let mut renderer: Renderer = todo!();
    /// let mut mat = Material::new();
    /// mat.base_color = [1.0, 0.5, 0.25, 1.0];
    /// mat.metallic = 0.8;
    /// mat.roughness = 0.6;
    /// let gpu_mat = renderer.upload_material(&mat);
    /// ```
    pub fn upload_material(&self, material: &Material) -> GpuMaterial {
        build_gpu_material(
            &self.device,
            &self.queue,
            &self.resources.bind_group_layouts.material,
            material,
            &self.default_material_views,
        )
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
    // PERSISTENT SCENE ENVIRONMENT API  (lights, sky, ambient, billboards)
    //
    // Lights and billboards follow the same persistent-proxy pattern as mesh
    // objects: add once, update/remove as needed, zero per-frame cost at
    // steady state.  Sky/ambient are set via dirty-flagged setters.
    // ══════════════════════════════════════════════════════════════════════════

    // ── Lights ───────────────────────────────────────────────────────────────

    /// Register a new light.  Returns a stable [`LightId`] valid until
    /// [`remove_light`] is called.  Equivalent to `ULightComponent::RegisterComponent`.
    ///
    /// In editor mode, automatically spawns a billboard icon at the light's
    /// world position to make it selectable in the viewport.
    pub fn add_light(&mut self, light: SceneLight) -> LightId {
        let id = self.gpu_light_scene.add_light(light.clone());
        if self.editor_mode {
            self.spawn_editor_light_billboard(id, &light);
        }
        id
    }

    /// Remove a light permanently.
    ///
    /// Also removes the associated editor billboard if one exists.
    pub fn remove_light(&mut self, id: LightId) {
        if let Some(bb_id) = self.editor_billboard_ids.remove(&id) {
            self.remove_billboard(bb_id);
        }
        self.gpu_light_scene.remove_light(id);
    }

    /// Replace all parameters of an existing light (position, color, intensity, etc.).
    ///
    /// Syncs the editor billboard color/position if editor mode is on.
    pub fn update_light(&mut self, id: LightId, light: SceneLight) {
        self.gpu_light_scene.update_light(id, light.clone());
        if let Some(&bb_id) = self.editor_billboard_ids.get(&id) {
            let [r, g, b] = light.color;
            let inst = crate::features::BillboardInstance::new(light.position, [0.35, 0.35])
                .with_color([r, g, b, 1.0])
                .with_screen_scale(true);
            self.update_billboard(bb_id, inst);
        }
    }

    /// Update only the world-space position of a light.  O(1), no shadow-matrix
    /// recompute until `flush_scene_state` runs.
    ///
    /// Moves the editor billboard to the new position.
    pub fn move_light(&mut self, id: LightId, position: [f32; 3]) {
        self.gpu_light_scene.move_light(id, position);
        if let Some(&bb_id) = self.editor_billboard_ids.get(&id) {
            self.move_billboard(bb_id, position);
        }
    }

    /// Update only color + intensity of a light (does not dirty shadow matrices).
    pub fn set_light_params(&mut self, id: LightId, color: [f32; 3], intensity: f32) {
        self.gpu_light_scene.set_light_params(id, color, intensity);
    }

    // ── Editor mode ──────────────────────────────────────────────────────────

    /// Enable or disable editor mode.
    ///
    /// In editor mode, every registered light automatically shows a billboard
    /// icon at its world position.  Enabling this mid-session spawns icons for
    /// all currently active lights; disabling removes them all.
    pub fn set_editor_mode(&mut self, enabled: bool) {
        if self.editor_mode == enabled { return; }
        self.editor_mode = enabled;

        if enabled {
            // Snapshot existing lights (can't borrow mutably and iterate at the same time).
            let lights: Vec<(LightId, SceneLight)> = self.gpu_light_scene
                .iter_lights()
                .map(|(id, l)| (id, l.clone()))
                .collect();
            for (id, light) in lights {
                self.spawn_editor_light_billboard(id, &light);
            }
        } else {
            // Remove all editor icons.
            let ids: Vec<BillboardId> = self.editor_billboard_ids.drain().map(|(_, v)| v).collect();
            for bb_id in ids {
                self.remove_billboard(bb_id);
            }
        }
    }

    /// Returns `true` when editor mode is active.
    pub fn is_editor_mode(&self) -> bool { self.editor_mode }

    /// Set shader debug visualization mode:
    /// - 0 = Normal rendering (with normal mapping)
    /// - 1 = UV grid (shows texture coordinates as colors)
    /// - 2 = Texture direct (G-buffer write, bypasses material multiply)
    /// - 3 = Lit without normal mapping (uses geometry normals only)
    /// - 4 = G-buffer readback test (reads albedo from G-buffer without lighting)
    /// - 5 = World normals (remaps N from [-1,1] to [0,1] as RGB)
    pub fn set_debug_mode(&mut self, mode: u32) {
        self.debug_mode = mode.min(5); // Clamp to valid range
    }

    /// Get current shader debug mode
    pub fn debug_mode(&self) -> u32 {
        self.debug_mode
    }

    /// Helper: spawn an editor billboard for `light` keyed under `id`.
    fn spawn_editor_light_billboard(&mut self, id: LightId, light: &SceneLight) {
        let [r, g, b] = light.color;
        let inst = crate::features::BillboardInstance::new(light.position, [0.35, 0.35])
            .with_color([r, g, b, 1.0])
            .with_screen_scale(true);
        let bb_id = self.add_billboard(inst);
        if bb_id != BillboardId::INVALID {
            self.editor_billboard_ids.insert(id, bb_id);
        }
    }

    // ── Debug Viz accessors ──────────────────────────────────────────────────

    /// Immutable access to the debug visualization system.
    ///
    /// Useful for reading which overlays are currently registered/enabled.
    pub fn debug_viz(&self) -> &debug_viz::DebugVizSystem {
        &self.debug_viz
    }

    /// Mutable access to the debug visualization system.
    ///
    /// ```no_run
    /// // Bind to F3 in your event loop:
    /// renderer.debug_viz_mut().enabled ^= true;
    ///
    /// // Toggle individual overlay:
    /// renderer.debug_viz_mut().set_enabled("grid", false);
    /// ```
    pub fn debug_viz_mut(&mut self) -> &mut debug_viz::DebugVizSystem {
        &mut self.debug_viz
    }

    // ── Billboards ────────────────────────────────────────────────────────────

    /// Register a billboard instance.  Returns a stable [`BillboardId`].
    pub fn add_billboard(&mut self, instance: crate::features::BillboardInstance) -> BillboardId {
        if let Some(bb) = self.features.get_typed_mut::<BillboardsFeature>("billboards") {
            bb.add_billboard_persistent(instance)
        } else {
            BillboardId::INVALID
        }
    }

    /// Remove a billboard instance.
    pub fn remove_billboard(&mut self, id: BillboardId) {
        if let Some(bb) = self.features.get_typed_mut::<BillboardsFeature>("billboards") {
            bb.remove_billboard_persistent(id);
        }
    }

    /// Update an existing billboard instance.
    pub fn update_billboard(&mut self, id: BillboardId, instance: crate::features::BillboardInstance) {
        if let Some(bb) = self.features.get_typed_mut::<BillboardsFeature>("billboards") {
            bb.update_billboard_persistent(id, instance);
        }
    }

    /// Update only the world-space position of a billboard.
    pub fn move_billboard(&mut self, id: BillboardId, position: [f32; 3]) {
        if let Some(bb) = self.features.get_typed_mut::<BillboardsFeature>("billboards") {
            bb.move_billboard(id, position);
        }
    }

    // ── Ambient / Sky ─────────────────────────────────────────────────────────

    /// Set the base ambient light color and intensity.
    pub fn set_ambient(&mut self, color: [f32; 3], intensity: f32) {
        if self.scene_state.ambient_color != color
            || self.scene_state.ambient_intensity.to_bits() != intensity.to_bits()
        {
            self.scene_state.ambient_color = color;
            self.scene_state.ambient_intensity = intensity;
            self.scene_state.ambient_dirty = true;
        }
    }

    /// Set the background sky color (used when no `SkyAtmosphere` is active).
    pub fn set_sky_color(&mut self, color: [f32; 3]) {
        if self.scene_state.sky_color != color {
            self.scene_state.sky_color = color;
            self.scene_state.sky_lut_dirty = true;
        }
    }

    /// Set or clear the physical atmosphere.  Triggers SkyLut re-render.
    pub fn set_sky_atmosphere(&mut self, atm: Option<crate::scene::SkyAtmosphere>) {
        self.scene_state.sky_atmosphere = atm;
        self.scene_state.sky_lut_dirty = true;
    }

    /// Set or clear the sky-driven ambient light.
    pub fn set_skylight(&mut self, skylight: Option<crate::scene::Skylight>) {
        self.scene_state.skylight = skylight;
        self.scene_state.ambient_dirty = true;
    }

    // ── Internal compat helper (not public) ──────────────────────────────────

    #[allow(dead_code)]
    fn set_scene_env(&mut self, env: SceneEnv) {
        // Lights — delegate to sync_lights (compat diff path).
        self.gpu_light_scene.sync_lights(&env.lights);

        // Billboards — delegate to set_billboards_slice (dirty-checked).
        if let Some(bb) = self.features.get_typed_mut::<BillboardsFeature>("billboards") {
            bb.set_billboards_slice(&env.billboards);
        }

        // Ambient.
        self.set_ambient(env.ambient_color, env.ambient_intensity);

        // Sky.
        let had_sky = self.scene_state.sky_atmosphere.is_some();
        let has_sky = env.sky_atmosphere.is_some();
        if self.scene_state.sky_color != env.sky_color || had_sky != has_sky {
            self.scene_state.sky_color = env.sky_color;
            self.scene_state.sky_lut_dirty = true;
        }
        if let Some(ref new_atm) = env.sky_atmosphere {
            let sun_dir = env.lights.iter()
                .find(|l| matches!(l.light_type, crate::features::LightType::Directional))
                .map(|l| {
                    let d = glam::Vec3::from(l.direction).normalize();
                    [-d.x, -d.y, -d.z]
                })
                .unwrap_or([0.0, 1.0, 0.0]);
            let sun_moved =
                (sun_dir[0] - self.scene_state.cached_sun_direction[0]).abs() > 0.01
                || (sun_dir[1] - self.scene_state.cached_sun_direction[1]).abs() > 0.01
                || (sun_dir[2] - self.scene_state.cached_sun_direction[2]).abs() > 0.01;
            if sun_moved
                || (new_atm.sun_intensity - self.scene_state.cached_sun_intensity).abs() > 0.01
                || !had_sky
            {
                self.scene_state.sky_lut_dirty = true;
                self.scene_state.cached_sun_direction = sun_dir;
                self.scene_state.cached_sun_intensity = new_atm.sun_intensity;
            }
        } else if had_sky {
            self.scene_state.sky_lut_dirty = true;
        }
        self.scene_state.sky_atmosphere = env.sky_atmosphere;
        self.scene_state.skylight = env.skylight;
    }



    /// Render a frame.
    ///
    /// Before calling this, supply the per-frame environment via `set_scene_env()`.
    /// Registered objects (added via `add_object`) are included automatically —
    /// no other per-frame object submission is required at steady state.
    pub fn render(&mut self, camera: &Camera, target: &wgpu::TextureView, delta_time: f32) -> Result<()> {
        let frame_start = Instant::now();
        // Profiling is tied to the live portal; if the feature is disabled
        // or unavailable simply treat profiling as inactive.
        let profiling_active = {
            #[cfg(feature = "live-portal")]
            {
                self.live_portal.is_some()
            }
            #[cfg(not(feature = "live-portal"))]
            {
                false
            }
        };
        crate::profiler::set_profiling_active(profiling_active);
        log::trace!("Rendering frame {}", self.frame_count);

        // ── Step 0: flush scene state (lights/sky/ambient) ───────────────────
        let camera_moved = {
            crate::profile_scope!("Scene");
            self.flush_scene_state(camera)
        };
        // `flush_scene_state` now returns `camera_moved` from update_shadow_matrices

        // ── Step 1: flush GPU scene (single delta write_buffer) ───────────────
        self.gpu_scene.flush(&self.queue);

        // ── Step 2: populate draw_list from GPU scene (persistent cache) ────
        {
            crate::profile_scope!("DrawList");
            let mut dl = self.draw_list.lock().unwrap();

            if self.gpu_scene.generation != self.cached_draw_list_gen {
                let scene_dl = self.gpu_scene.draw_lists();
                dl.clear();
                dl.extend_from_slice(scene_dl);
                self.persistent_draw_count = dl.len();
                self.cached_draw_list_gen = self.gpu_scene.generation;
            } else {
                dl.truncate(self.persistent_draw_count);
            }
        }

        // ── Step 2b: flush GPU draw calls + run indirect dispatch ────────────
        {
            // Ensure draw lists are rebuilt (so material_ranges is current).
            self.gpu_scene.draw_lists();

            // Upload GPU draw call buffer (sorted by material_slot).
            self.gpu_scene.flush_draw_calls(&self.device, &self.queue);
            let draw_count = self.gpu_scene.draw_call_count;

            // Update shared material ranges (for geometry passes).
            {
                let mut ranges = self.shared_material_ranges.lock().unwrap();
                ranges.clear();
                ranges.extend_from_slice(&self.gpu_scene.material_ranges);
            }

            // Update raw draw-call buffer ref for ShadowPass GPU indirect cull.
            *self.shared_shadow_draw_call_buf.lock().unwrap() =
                if draw_count > 0 { Some(self.gpu_scene.draw_call_buffer().clone()) } else { None };

            // Run IndirectDispatchPass: GPU culls + writes indirect commands.
            if draw_count > 0 {
                let new_buf = self.indirect_dispatch.update(
                    &self.device,
                    self.gpu_scene.draw_call_buffer(),
                    &self.camera_buffer,
                    draw_count,
                )?;
                *self.shared_indirect_buf.lock().unwrap() = new_buf;
            } else {
                *self.shared_indirect_buf.lock().unwrap() = None;
            }
        }

        // ── Preparation: camera upload, feature prepare, globals, debug batch ──
        {
            crate::profile_scope!("Prep");

            self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera));
            gpu_transfer::track_upload(std::mem::size_of::<Camera>() as u64);

            let prep_ctx = PrepareContext::new(
                &self.device, &self.queue, &self.resources,
                self.frame_count, delta_time, camera,
            );
            self.features.prepare_all(&prep_ctx)?;

            if let Some(rc) = self.features.get_typed_mut::<RadianceCascadesFeature>("radiance_cascades") {
                let (mn, mx) = rc.world_bounds();
                self.rc_world_min = mn;
                self.rc_world_max = mx;
            }

            let globals = GlobalsUniform {
                frame: self.frame_count as u32,
                delta_time,
                light_count: self.scene_light_count,
                ambient_intensity: self.scene_ambient_intensity,
                ambient_color: [self.scene_ambient_color[0], self.scene_ambient_color[1], self.scene_ambient_color[2], 0.0],
                rc_world_min: [self.rc_world_min[0], self.rc_world_min[1], self.rc_world_min[2], 0.0],
                rc_world_max: [self.rc_world_max[0], self.rc_world_max[1], self.rc_world_max[2], 0.0],
                csm_splits: self.scene_csm_splits,
                debug_mode: self.debug_mode,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            self.queue.write_buffer(&self.globals_buffer, 0, bytemuck::bytes_of(&globals));
            gpu_transfer::track_upload(std::mem::size_of::<GlobalsUniform>() as u64);

            // ── Debug Visualization System ────────────────────────────────────
            // Collect overlay shapes into debug_shapes before the batch is built.
            if self.debug_viz.enabled {
                let bounds = self.gpu_scene.collect_world_bounds();
                let ctx = debug_viz::DebugRenderContext {
                    lights: &self.gpu_light_scene.cached_scene_lights,
                    object_bounds: &bounds,
                    camera_pos: camera.position,
                    camera_forward: camera.forward(),
                    dt: delta_time,
                };
                let mut shapes = self.debug_shapes.lock().unwrap();
                self.debug_viz.collect(&ctx, &mut shapes);
            }

            let debug_shapes = {
                let mut shapes = self.debug_shapes.lock().unwrap();
                std::mem::take(&mut *shapes)
            };
            if debug_shapes.is_empty() {
                *self.debug_batch.lock().unwrap() = None;
            } else {
                *self.debug_batch.lock().unwrap() = debug_draw::build_batch(&self.device, &debug_shapes);
            }
        } // profile_scope!("Prep") drops here

        // ── Indirect compute dispatch + render passes ─────────────────────────
        let error_scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        self.indirect_dispatch.dispatch(&self.queue, &mut encoder);

        // ── GPU shadow matrix computation (runs before shadow pass) ──────────
        {
            crate::profile_scope!("ShadowMatrices");
            let light_count = self.gpu_light_scene.active_count;

            if light_count > 0 {
                // Upload dirty flags to GPU
                self.gpu_light_scene.upload_shadow_dirty_flags(&self.queue);

                // Bind resources (light buffer, shadow matrix buffer, camera, dirty flags, hashes)
                self.shadow_matrix_pass.bind_resources(
                    &self.device,
                    &self.gpu_light_scene.light_buffer,
                    &self.gpu_light_scene.shadow_matrix_buffer,
                    &self.camera_buffer,
                    self.gpu_light_scene.shadow_dirty_buffer(),
                    self.gpu_light_scene.shadow_hash_buffer(),
                );

                // Execute compute shader (GPU computes shadow matrices)
                self.shadow_matrix_pass.execute(
                    &mut encoder,
                    &self.queue,
                    camera_moved,
                    light_count,
                );

                // Clear dirty flags after uploading (GPU will process them asynchronously)
                self.gpu_light_scene.clear_shadow_dirty_flags();
            }
        }

        let graph_target = match self.aa_mode {
            AntiAliasingMode::None | AntiAliasingMode::Msaa(_) => target,
            _ => &self.pre_aa_view,
        };

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

        // ── Execute render graph ──────────────────────────────────────────────
        {
            crate::profile_scope!("Graph");
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
                transparent_start: self.gpu_scene.transparent_start,
            };

            if profiling_active {
                if let Some(p) = &mut self.profiler { p.begin_frame(); }
                self.graph.execute(&mut graph_ctx, self.profiler.as_mut())?;
            } else {
                self.graph.execute(&mut graph_ctx, None)?;
            }
        } // profile_scope!("Graph") drops here

        // ── Anti-aliasing post-processing ─────────────────────────────────────
        {
            crate::profile_scope!("AA");
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
        } // profile_scope!("AA") drops here

        // ── Encode: resolve timestamps, finish, submit ────────────────────────
        {
            crate::profile_scope!("Encode");
            {
                crate::profile_scope!("Encode::Resolve");
                if profiling_active {
                    if let Some(p) = &mut self.profiler { p.resolve(&mut encoder); }
                }
            }
            let cmd_buf = {
                crate::profile_scope!("Encode::Finish");
                encoder.finish()
            };
            {
                crate::profile_scope!("Encode::Submit");
                self.queue.submit(Some(cmd_buf));
            }
            // Pop the validation error scope and block until the GPU has processed
            // the frame — gives us the exact shader/buffer error before TDR fires.
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(e) = pollster::block_on(error_scope.pop()) {
                panic!("[GPU VALIDATION ERROR] {}", e);
            }
            #[cfg(target_arch = "wasm32")]
            {
                // cannot block on wasm; spawn a local task to panic later if there is
                // an error (should be rare in release builds).
                use wasm_bindgen_futures::spawn_local;
                let fut = error_scope.pop();
                spawn_local(async move {
                    if let Some(e) = fut.await {
                        panic!("[GPU VALIDATION ERROR] {}", e);
                    }
                });
            }
        } // profile_scope!("Encode") drops here

        // ── GPU readback (non-blocking) ───────────────────────────────────────
        {
            crate::profile_scope!("Poll");
            if profiling_active {
                if let Some(p) = &mut self.profiler {
                    p.begin_readback();
                    p.poll_results(&self.device);
                }
            }
        } // profile_scope!("Poll") drops here

        // Collect scope log AFTER all profile_scope! guards above have dropped.
        let cpu_scope_log = crate::profiler::take_frame_scope_log();

        *self.debug_batch.lock().unwrap() = None;

        // ── Frame timing bookkeeping ──────────────────────────────────────────
        let frame_to_frame_ms = if let Some(last_start) = self.last_frame_start {
            frame_start.duration_since(last_start).as_secs_f32() * 1000.0
        } else {
            0.0
        };
        let frame_time_ms = frame_start.elapsed().as_secs_f32() * 1000.0;

        self.last_frame_start = Some(frame_start);
        self.last_frame_end = Some(Instant::now());

        // ── Live portal snapshot ──────────────────────────────────────────────
        #[cfg(feature = "live-portal")]
        if let Some(portal) = &self.live_portal {
            let draw_total = { self.draw_list.lock().unwrap().len() };
            let draw_transparent = draw_total.saturating_sub(self.gpu_scene.transparent_start);
            let draw_opaque      = draw_total - draw_transparent;

            let new_key = (
                self.gpu_scene.object_count() as usize,
                self.gpu_light_scene.active_count as usize,
                self.features.get_typed::<BillboardsFeature>("billboards")
                    .map(|bb| bb.proxy_count()).unwrap_or(0),
            );
            let layout_changed = new_key != self.portal_scene_key;
            let layout: Option<PortalSceneLayout> = if layout_changed || self.latest_scene_layout.is_none() {
                self.portal_scene_key = new_key;
                Some(self.build_portal_layout(camera))
            } else {
                self.latest_scene_layout.take()
            };
            self.pending_layout_changed = layout_changed;

            // Compute delta against previous layout.
            let scene_delta = layout.as_ref().map(|cur| {
                if layout_changed {
                    compute_scene_delta(cur, self.previous_scene_layout.as_ref())
                } else {
                    let prev_cam = self.previous_scene_layout.as_ref().and_then(|p| p.camera.clone());
                    let mut d = PortalSceneLayoutDelta::default();
                    if cur.camera != prev_cam { d.camera = Some(cur.camera.clone()); }
                    d
                }
            });

            let (obj_count, light_count, bb_count) = layout.as_ref()
                .map(|l| (l.objects.len(), l.lights.len(), l.billboards.len()))
                .unwrap_or((0, 0, 0));

            // Advance previous layout tracking.
            if layout_changed {
                self.previous_scene_layout = layout;
            } else if let Some(cur) = layout {
                match &mut self.previous_scene_layout {
                    Some(prev) => prev.camera = cur.camera,
                    None       => self.previous_scene_layout = Some(cur),
                }
            }

            let stage_timings = build_scope_tree(&cpu_scope_log);

            let (total_gpu_ms, total_cpu_ms, pass_timings) = self.profiler.as_mut()
                .map(|p| (p.last_total_gpu_ms, p.last_total_cpu_ms, std::mem::take(&mut p.last_timings)))
                .unwrap_or((0.0, 0.0, vec![]));

            let snapshot = PortalFrameSnapshot {
                frame: self.frame_count,
                timestamp_ms: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_millis())
                    .unwrap_or(0),
                frame_time_ms,
                frame_to_frame_ms,
                total_gpu_ms,
                total_cpu_ms,
                pass_timings: pass_timings.iter()
                    .map(|t| PortalPassTiming { name: t.name.clone(), gpu_ms: t.gpu_ms, cpu_ms: t.cpu_ms })
                    .collect(),
                pipeline_order: self.cached_pass_names.clone(),
                pipeline_stage_id: Some("graph_passes".to_string()),
                scene_delta,
                object_count: obj_count,
                light_count,
                billboard_count: bb_count,
                draw_calls: helio_live_portal::DrawCallMetrics {
                    total: draw_total,
                    opaque: draw_opaque,
                    transparent: draw_transparent,
                },
                stage_timings,
            };
            portal.publish(snapshot);
        }

        // One-frame draws (from draw_mesh()) sit after `persistent_draw_count`
        // and are truncated at the start of the next render() call.
        // The persistent portion survives across frames — zero-cost at steady state.

        self.frame_count += 1;
        gpu_transfer::end_frame();
        Ok(())
    }

    // ── Resize ────────────────────────────────────────────────────────────────
    // ── Portal helpers ────────────────────────────────────────────────────────

    #[cfg(feature = "live-portal")]
    /// Build a `PortalSceneLayout` from the renderer's live state.
    ///
    /// Objects come from the persistent draw_list (bounds data already cached).
    /// Lights come from the GPU light scene's CPU mirror.
    /// Billboards come from the billboard feature's proxy count.
    /// This is called only when the portal is active and the scene has changed.
    fn build_portal_layout(&self, camera: &Camera) -> PortalSceneLayout {
        // Objects: use the persistent draw_list (opaque + transparent).
        let objects = {
            let dl = self.draw_list.lock().unwrap();
            dl[..self.persistent_draw_count.min(dl.len())]
                .iter()
                .enumerate()
                .map(|(id, dc)| PortalSceneObject {
                    id:           id as u32,
                    bounds_center: dc.bounds_center,
                    bounds_radius: dc.bounds_radius,
                    has_material: true,
                })
                .collect::<Vec<_>>()
        };

        // Lights: CPU mirror in GpuLightScene (pub(super) from this module).
        let lights = self.gpu_light_scene.cached_scene_lights
            [..self.gpu_light_scene.active_count as usize]
            .iter()
            .enumerate()
            .map(|(id, l)| PortalSceneLight {
                id:        id as u32,
                position:  l.position,
                color:     l.color,
                intensity: l.intensity,
                range:     l.range,
            })
            .collect::<Vec<_>>();

        // Billboards: proxy count from feature.
        let billboards = self.features
            .get_typed::<BillboardsFeature>("billboards")
            .map(|bb| {
                (0..bb.proxy_count() as u32)
                    .map(|id| PortalSceneBillboard { id, position: [0.0; 3], scale: [1.0; 2] })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let fwd = camera.forward();
        PortalSceneLayout {
            objects,
            lights,
            billboards,
            camera: Some(PortalSceneCamera {
                position: [camera.position.x, camera.position.y, camera.position.z],
                forward:  [fwd.x, fwd.y, fwd.z],
            }),
        }
    }


    pub fn resize(&mut self, width: u32, height: u32) {
        log::trace!("Resizing renderer to {}x{}", width, height);
        self.width = width;
        self.height = height;

        let (tex, view, sample_view) = helpers::create_depth_texture(&self.device, width, height);
        self.depth_texture = tex;
        self.depth_view = view;
        self.depth_sample_view = sample_view;

        let (albedo_tex, normal_tex, orm_tex, emissive_tex, specular_tex, new_targets) =
            helpers::create_gbuffer_textures(&self.device, width, height);
        self.gbuf_albedo_texture  = albedo_tex;
        self.gbuf_normal_texture  = normal_tex;
        self.gbuf_orm_texture     = orm_tex;
        self.gbuf_emissive_texture = emissive_tex;
        self.gbuf_specular_texture = specular_tex;

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

    /// Hard-set the render resolution.  Unlike `resize()` (which only
    /// recreates textures), this also forces the sky LUT to regenerate and
    /// bumps the draw-list generation so every pass picks up the change.
    pub fn set_render_size(&mut self, width: u32, height: u32) {
        let width  = width.max(1);
        let height = height.max(1);
        log::info!("set_render_size: {}x{} (was {}x{})", width, height, self.width, self.height);
        if width == self.width && height == self.height {
            return;
        }
        self.resize(width, height);
        // Force every dirty flag so the next frame fully regenerates.
        self.scene_state.sky_lut_dirty = true;
        self.sky_state_changed = true;
        self.draw_list_generation = self.draw_list_generation.wrapping_add(1);
        self.cached_draw_list_gen = u64::MAX; // force draw-list rebuild
    }

    pub fn frame_count(&self) -> u64 { self.frame_count }

    pub fn device(&self) -> &wgpu::Device { &self.device }
    pub fn queue(&self) -> &wgpu::Queue { &self.queue }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PORTAL WORKER THREAD
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "live-portal")]
/// Convert a flat `Vec<CompletedScope>` — ordered by drop (innermost first) —
/// into a `PortalStageTiming` tree that the portal can render.
///
/// The scope log is in *completion* order: inner scopes complete (drop) before
/// their parent.  We reconstruct the tree from the `parent` index field, which
/// is set by the thread-local `SCOPE_STACK` at the moment the scope opened.
fn build_scope_tree(
    scopes: &[crate::profiler::CompletedScope],
) -> Vec<PortalStageTiming> {
    use std::collections::HashMap;

    // Map scope idx → position in the slice.
    let idx_map: HashMap<u32, usize> = scopes
        .iter()
        .enumerate()
        .map(|(i, s)| (s.idx, i))
        .collect();

    // Build parent → [child indices] adjacency list.
    let mut children_of: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut roots: Vec<u32> = Vec::new();

    for scope in scopes {
        match scope.parent {
            Some(p) => children_of.entry(p).or_default().push(scope.idx),
            None    => roots.push(scope.idx),
        }
    }

    // Sort children by their own idx (open-order = declaration order in code).
    for kids in children_of.values_mut() {
        kids.sort_unstable();
    }
    roots.sort_unstable();

    fn to_node(
        idx:         u32,
        scopes:      &[crate::profiler::CompletedScope],
        idx_map:     &HashMap<u32, usize>,
        children_of: &HashMap<u32, Vec<u32>>,
    ) -> PortalStageTiming {
        let scope = &scopes[idx_map[&idx]];
        let id = scope.name
            .replace([' ', ':', '/', '!'], "_")
            .to_ascii_lowercase();
        let kids = children_of
            .get(&idx)
            .map(|v| v.iter().map(|&c| to_node(c, scopes, idx_map, children_of)).collect())
            .unwrap_or_default();
        PortalStageTiming {
            id,
            name:     scope.name.to_string(),
            ms:       scope.elapsed_ms,
            children: kids,
        }
    }

    roots
        .iter()
        .map(|&r| to_node(r, scopes, &idx_map, &children_of))
        .collect()
}
