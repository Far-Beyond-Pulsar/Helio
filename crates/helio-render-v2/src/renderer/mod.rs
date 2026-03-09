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
use crate::passes::{DebugDrawPass, SkyPass, SkyLutPass, SKY_LUT_W, SKY_LUT_H, SKY_LUT_FORMAT, ShadowCullLight, DepthPrepassPass, GBufferPass, GBufferTargets, DeferredLightingPass, TransparentPass, AntiAliasingMode, FxaaPass, SmaaPass, TaaPass, IndirectDispatchPass, PostProcessPass, PhysicalBloomPass, BloomConfig, SsrPass, GodRaysPass, DofPass};
use crate::mesh::{GpuMesh, DrawCall, GpuDrawCall, INSTANCE_STRIDE};
use crate::gpu_scene::{GpuScene, GpuPrimitive, PrimitiveSlot, PRIM_TRANSPARENT};
use crate::camera::Camera;
use crate::scene::{ObjectId, SceneLight};
use crate::debug_draw::{self, DebugDrawBatch, DebugShape};
use crate::features::lighting::{GpuLight, MAX_LIGHTS};
use crate::features::BillboardsFeature;
use crate::material::{Material, GpuMaterial, MaterialUniform, DefaultMaterialViews};
use crate::material_registry::MaterialRegistry;
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

// ── Persistent proxy entry — one per registered object ───────────────────────
//
// Mirrors Unreal Engine's `FPrimitiveSceneProxy`.  Created once by `add_object`,
// destroyed by `remove_object`.  Frustum culling, enable/disable, and
// transform updates never allocate or free GPU resources.
//
// The transform is stored in CPU memory and batched into a shared per-frame
// `scene_instance_buf` by `render()` Step 1, grouped with all other objects
// that share the same (vertex_buffer, index_buffer, material_bind_group) key.
// This mirrors Unreal's per-frame GPU scene instance buffer construction.
struct RegisteredProxy {
    /// Draw call template: vertex/index buffers + material bind group.
    dc: DrawCall,
    /// Slot in the GPU Scene persistent storage buffer.
    /// This is both the `first_instance` for the per-frame prim-id staging
    /// buffer AND the index the vertex shader uses to read model transforms.
    slot: PrimitiveSlot,
    /// FNV-1a hash of the last set transform matrix for change detection.
    transform_hash: u64,
    /// Whether this proxy renders this frame.
    enabled: bool,
    /// World-space bounding sphere centre (for CPU frustum culling).
    bounding_center: glam::Vec3,
    /// World-space bounding sphere radius (f32::INFINITY = never culled).
    bounding_radius: f32,
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

    // ── HDR intermediate buffer + new post-process stack ──────────────────
    /// Rgba16Float render target.  All scene passes write here; post-process
    /// stack reads here; PostProcess writes to pre_aa_view / swapchain.
    hdr_texture:      wgpu::Texture,
    hdr_view:         wgpu::TextureView,
    /// Copy of HDR used as SSR scene_color input (avoids read-write hazard).
    hdr_prev_texture: wgpu::Texture,
    hdr_prev_view:    wgpu::TextureView,
    bloom_pass:       PhysicalBloomPass,
    ssr_pass:         SsrPass,
    god_rays_pass:    GodRaysPass,
    dof_pass:         DofPass,
    post_process_pass: PostProcessPass,
    /// Last sun screen-space position for god rays (updated each frame).
    sun_screen_pos: [f32; 2],

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

    /// Cached light list state — skip sort if unchanged
    cached_light_count:        usize,
    cached_light_position_hash: u64,
    cached_camera_pos:         [f32; 3],
    camera_move_threshold:     f32,

    // ── Per-frame scratch (reused allocations, no heap churn) ─────────────
    scratch_gpu_lights:              Vec<GpuLight>,
    scratch_shadow_mats:             Vec<uniforms::GpuShadowMatrix>,
    scratch_shadow_matrix_hashes:    Vec<u64>,
    scratch_sorted_light_indices:    Vec<usize>,

    // ── Shadow draw list ──────────────────────────────────────────────────
    /// Batched DrawCalls for all enabled objects, rebuilt each frame by render()
    /// Step 1.  ShadowPass culls this by light range / face frustum.
    shadow_draw_list: Arc<Mutex<Vec<DrawCall>>>,

    // ── Shared per-frame primitive ID buffer ──────────────────────────────
    /// Single GPU buffer holding all per-instance `primitive_id` u32 values
    /// for the current frame, laid out as: [camera_batch_0..N | shadow_batch_0..N].
    /// Each DrawCall's `instance_buffer_offset` points into the right region.
    /// 16× smaller than the old mat4 staging buffer (4 bytes vs 64 bytes per instance).
    scene_instance_buf: Option<Arc<wgpu::Buffer>>,
    /// Capacity in number of u32 primitive IDs.
    scene_instance_buf_cap: u32,

    // ── GPU Scene (persistent per-object data) ────────────────────────────
    /// Persistent storage buffer holding one GpuPrimitive (144 bytes) per
    /// registered object.  Dirty-tracked: only changed slots are re-uploaded
    /// each frame, so a completely static scene pays zero upload cost.
    gpu_scene: GpuScene,

    // ════════════════════════════════════════════════════════════════════════
    // PERSISTENT SCENE PROXY REGISTRY
    // ════════════════════════════════════════════════════════════════════════
    //
    // This is the Unreal FScene equivalent.  The application calls:
    //   add_object(mesh, material, transform) -> ObjectId   (like AddPrimitive)
    //   remove_object(id)                                   (like RemovePrimitive)
    //   update_transform(id, transform)                     (like SendRenderTransform)
    //
    // The renderer stores one `RegisteredProxy` per id.  Every frame `render()`
    // rebuilds `draw_list` and `shadow_draw_list` by pointer-cloning the existing
    // DrawCall arcs — O(N) pointer increments, zero GPU allocations at steady state.
    //
    // Frustum culling, camera rotation, LOD paging — none of these touch this map.
    // The ONLY writes are from add_object / remove_object / update_transform.
    // ────────────────────────────────────────────────────────────────────────
    /// Stable proxy map keyed by ObjectId.
    registered_objects: HashMap<u64, RegisteredProxy>,
    /// Next id to hand out.  Starts at 1; zero is INVALID.
    next_object_id: u64,

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

    /// When true, GPU timing stats are printed to stderr every frame.
    debug_printout: bool,
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
    // ── Material / texture registry ───────────────────────────────────────
    /// Content-addressed cache: same material bytes → same GPU bind group.
    /// Prevents duplicate texture uploads and sampler allocations (Unreal: FMaterialRenderProxy).
    material_registry: MaterialRegistry,

    cached_pass_names: Vec<String>,
}

// ── Frustum culling helpers ──────────────────────────────────────────────────

/// Extract 6 world-space frustum planes from a view-projection matrix.
///
/// Uses the Gribb-Hartmann method.  Planes are normalised so the signed
/// distance `dot(plane.xyz, p) + plane.w` is in world-space metres.
/// Convention: **positive** = inside, **negative** = outside.
///
/// Matches wgpu / WebGPU NDC: x∈[-1,1], y∈[-1,1], z∈[0,1].
fn extract_frustum_planes(vp: glam::Mat4) -> [glam::Vec4; 6] {
    // Transpose so `t.x_axis` = row 0 of vp, etc. (glam is column-major).
    let t  = vp.transpose();
    let r0 = t.x_axis;
    let r1 = t.y_axis;
    let r2 = t.z_axis;
    let r3 = t.w_axis;
    let raw = [
        r3 + r0,  // left
        r3 - r0,  // right
        r3 + r1,  // bottom
        r3 - r1,  // top
        r2,       // near  (z_ndc >= 0)
        r3 - r2,  // far   (z_ndc <= 1)
    ];
    raw.map(|p| {
        let len = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
        if len > 1e-8 { p / len } else { p }
    })
}

/// Returns `true` if the bounding sphere is fully or partially inside *all*
/// six frustum planes (i.e. not definitley outside any single plane).
///
/// `radius == f32::INFINITY` always returns `true` (culling disabled).
#[inline]
fn sphere_in_frustum(planes: &[glam::Vec4; 6], center: glam::Vec3, radius: f32) -> bool {
    if radius == f32::INFINITY { return true; }
    for plane in planes {
        // Signed distance of centre from plane.
        let dist = plane.x * center.x + plane.y * center.y + plane.z * center.z + plane.w;
        if dist < -radius { return false; }  // sphere entirely outside this plane
    }
    true
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

    // ══════════════════════════════════════════════════════════════════════════
    // PERSISTENT SCENE PROXY API  (Unreal FScene::AddPrimitive / RemovePrimitive)
    // ══════════════════════════════════════════════════════════════════════════

    /// Register a mesh into the persistent scene proxy registry.
    ///
    /// Equivalent to `UPrimitiveComponent::RegisterComponent()` → `FScene::AddPrimitive()`.
    /// Creates a dedicated 64-byte per-object instance buffer on the GPU and stores
    /// the initial `transform` in it.  Returns a stable [`ObjectId`] that uniquely
    /// identifies this object for its entire lifetime in the scene.
    ///
    /// Cost: one `create_buffer_init` + one `HashMap::insert`.  Never called again
    /// for the same object — subsequent transforms are O(1) `write_buffer`.
    pub fn add_object(
        &mut self,
        mesh:      &GpuMesh,
        material:  Option<&GpuMaterial>,
        transform: glam::Mat4,
    ) -> ObjectId {
        let id = ObjectId(self.next_object_id);
        self.next_object_id += 1;

        let mat: [f32; 16] = transform.to_cols_array();
        let transform_hash = fnv1a_mat(&mat);

        let (bind_group, transparent) = match material {
            Some(m) => (Arc::clone(&m.bind_group), m.transparent_blend),
            None    => (Arc::clone(&self.default_material_bind_group), false),
        };

        let dc = DrawCall::new(mesh, bind_group, transparent);

        let bounding_center = transform.w_axis.truncate();
        let flags = if transparent { PRIM_TRANSPARENT } else { 0 };
        let prim  = GpuPrimitive::from_transform(
            transform,
            bounding_center.to_array(),
            f32::MAX,   // culling disabled until set_object_bounds() is called
            0,
            flags,
        );
        let slot = self.gpu_scene.alloc(prim, &self.device);

        self.registered_objects.insert(id.0, RegisteredProxy {
            dc,
            slot,
            transform_hash,
            enabled: true,
            bounding_center,
            bounding_radius: f32::INFINITY,
        });

        self.draw_list_generation = self.draw_list_generation.wrapping_add(1);

        id
    }

    /// Set the world-space bounding sphere radius for frustum culling.
    ///
    /// Call once after `add_object` for objects that should be culled when outside
    /// the camera frustum (e.g. voxel chunk entities with known extents).
    /// The bounding center is kept in sync automatically by `update_transform`.
    ///
    /// Set `radius = f32::INFINITY` to disable culling for this object (the default).
    pub fn set_object_bounds(&mut self, id: ObjectId, radius: f32) {
        if let Some(proxy) = self.registered_objects.get_mut(&id.0) {
            proxy.bounding_radius = radius;
        }
    }

    /// Remove an object from the scene proxy registry.
    ///
    /// Equivalent to `UPrimitiveComponent::UnregisterComponent()` → `FScene::RemovePrimitive()`.
    /// No per-object GPU buffer is held (transforms live in the shared `scene_instance_buf`),
    /// so this is a pure CPU operation with no GPU resource lifetime concerns.
    ///
    /// Cost: one `HashMap::remove`.
    pub fn remove_object(&mut self, id: ObjectId) {
        if let Some(proxy) = self.registered_objects.remove(&id.0) {
            self.gpu_scene.free(proxy.slot, &self.device);
            self.draw_list_generation = self.draw_list_generation.wrapping_add(1);
        }
    }

    /// Update the world transform of a registered object.
    ///
    /// Equivalent to `USceneComponent::SendRenderTransform()` → proxy `SetTransform()`.
    /// Stores the new matrix in CPU memory only; the GPU upload happens once per frame
    /// inside `render()` Step 1 as part of the batched `scene_instance_buf` write.
    /// Static geometry that never moves incurs zero cost (hash check skips the copy).
    ///
    /// Cost: ~1 µs hash check + conditional 64-byte CPU copy.
    pub fn update_transform(&mut self, id: ObjectId, transform: glam::Mat4) {
        if let Some(proxy) = self.registered_objects.get_mut(&id.0) {
            let mat  = transform.to_cols_array();
            let hash = fnv1a_mat(&mat);
            if hash != proxy.transform_hash {
                proxy.transform_hash  = hash;
                proxy.bounding_center = transform.w_axis.truncate();
                // Build a new GpuPrimitive and mark the slot dirty.
                // flush_dirty() at the top of render() will upload it.
                let flags = if proxy.dc.transparent_blend { PRIM_TRANSPARENT } else { 0 };
                let new_prim = GpuPrimitive::from_transform(
                    transform,
                    proxy.bounding_center.to_array(),
                    proxy.bounding_radius,
                    0,
                    flags,
                );
                self.gpu_scene.update(proxy.slot, new_prim);
            }
        }
    }

    /// Batch-update many transforms in a single call.  More efficient than calling
    /// `update_transform` in a loop because the loop over changed entries is inlined.
    pub fn update_transforms(&mut self, updates: &[(ObjectId, glam::Mat4)]) {
        for &(id, transform) in updates {
            self.update_transform(id, transform);
        }
    }

    /// Exclude a registered object from rendering without deallocating its GPU resources.
    ///
    /// Equivalent to hiding a component in Unreal without unregistering it.
    /// Zero GPU cost — only flips a `bool`.  The instance buffer and draw call
    /// remain allocated; re-enabling is instant.
    ///
    /// Use this for chunk streaming: call when a chunk deactivates instead of
    /// `remove_object`, and `enable_object` when it reactivates.  This avoids
    /// GPU buffer alloc/free churn on every streaming event.
    pub fn disable_object(&mut self, id: ObjectId) {
        if let Some(proxy) = self.registered_objects.get_mut(&id.0) {
            proxy.enabled = false;
        }
    }

    /// Re-include a previously disabled object in rendering.
    pub fn enable_object(&mut self, id: ObjectId) {
        if let Some(proxy) = self.registered_objects.get_mut(&id.0) {
            proxy.enabled = true;
        }
    }

    /// Returns `true` if the object is registered and currently enabled.
    pub fn is_object_enabled(&self, id: ObjectId) -> bool {
        self.registered_objects.get(&id.0).map_or(false, |p| p.enabled)
    }

    /// Number of objects currently registered (enabled + disabled).
    pub fn object_count(&self) -> usize {
        self.registered_objects.len()
    }

    /// Number of objects currently enabled (contributing to rendering).
    pub fn enabled_object_count(&self) -> usize {
        self.registered_objects.values().filter(|p| p.enabled).count()
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

        // ── Step 1: CPU-driven instance batching ─────────────────────────────
        //
        // Groups enabled proxies by (vertex_buffer, index_buffer, material_bind_group,
        // transparent_blend) — the "batch key".  Objects sharing the same key have
        // identical GPU pipeline and resource state so they can be drawn with a
        // single instanced draw_indexed call (one per batch, not one per object).
        //
        // This mirrors Unreal's draw policy grouping:
        //   N objects × M passes  →  K batches × M passes   (K << N at steady state)
        //
        // All transforms for the current frame are packed into `scene_instance_buf`:
        //   [camera_batch_0 | camera_batch_1 | ... | shadow_batch_0 | shadow_batch_1 | ...]
        // Each batched DrawCall carries the byte offset into that buffer.
        // One write_buffer uploads all transforms to the GPU in one call.
        {
            let mut dl  = self.draw_list.lock().unwrap();
            let mut sdl = self.shadow_draw_list.lock().unwrap();
            dl.clear();
            sdl.clear();

            // Camera frustum (Gribb-Hartmann, wgpu NDC z∈[0,1]).
            let frustum = extract_frustum_planes(camera.view_proj);

            // Collect (batch_key, proxy_ref) pairs for camera and shadow sets.
            // batch_key encodes unique (mesh, material) state as raw pointer values.
            type BatchKey = (usize, usize, usize, bool);
            let mut camera_items: Vec<(BatchKey, &RegisteredProxy)> = Vec::new();
            let mut shadow_items: Vec<(BatchKey, &RegisteredProxy)> = Vec::new();

            for proxy in self.registered_objects.values() {
                if !proxy.enabled { continue; }
                let key: BatchKey = (
                    Arc::as_ptr(&proxy.dc.vertex_buffer)       as usize,
                    Arc::as_ptr(&proxy.dc.index_buffer)        as usize,
                    Arc::as_ptr(&proxy.dc.material_bind_group) as usize,
                    proxy.dc.transparent_blend,
                );
                // Shadow: all enabled (light sees geometry outside camera frustum).
                shadow_items.push((key, proxy));
                // Camera: frustum-culled.
                if sphere_in_frustum(&frustum, proxy.bounding_center, proxy.bounding_radius) {
                    camera_items.push((key, proxy));
                }
            }

            camera_items.sort_unstable_by_key(|&(k, _)| k);
            shadow_items.sort_unstable_by_key(|&(k, _)| k);

            let camera_total = camera_items.len();
            let shadow_total = shadow_items.len();
            let buf_need     = camera_total + shadow_total;

            // Grow the shared instance buffer when capacity is exceeded.
            // Rounded up to the next power of two to amortise reallocation.
            if buf_need > self.scene_instance_buf_cap as usize
                || self.scene_instance_buf.is_none()
            {
                let new_cap = ((buf_need as u32).max(256)).next_power_of_two();
                let buf = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label:              Some("Scene Instance Buffer"),
                    size:               new_cap as u64 * INSTANCE_STRIDE,
                    usage:              wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                self.scene_instance_buf     = Some(buf);
                self.scene_instance_buf_cap = new_cap;
            }
            let scene_buf = Arc::clone(self.scene_instance_buf.as_ref().unwrap());

            // Pack all per-instance primitive_id u32 values + build batched DrawCalls.
            // Each value is the GPU Scene slot index; the vertex shader reads the
            // model transform from gpu_primitives[primitive_id].transform.
            // 16× smaller than the old [f32;16] transform staging buffer.
            let mut staging: Vec<u32> = Vec::with_capacity(buf_need);

            // ── Camera-visible batches → main draw list ───────────────────────
            {
                let mut i = 0;
                while i < camera_items.len() {
                    let key = camera_items[i].0;
                    let batch_offset = staging.len() as u64;
                    let mut j = i;
                    while j < camera_items.len() && camera_items[j].0 == key {
                        staging.push(camera_items[j].1.slot.0);
                        j += 1;
                    }
                    let mut dc = camera_items[i].1.dc.clone();
                    dc.bounds_center          = camera_items[i].1.bounding_center.to_array();
                    dc.instance_buffer        = Some(Arc::clone(&scene_buf));
                    dc.instance_buffer_offset = batch_offset * INSTANCE_STRIDE;
                    dc.instance_count         = (j - i) as u32;
                    dl.push(dc);
                    i = j;
                }
            }

            // ── All-enabled batches → shadow draw list (no frustum cull) ─────
            {
                let mut i = 0;
                while i < shadow_items.len() {
                    let key = shadow_items[i].0;
                    let batch_offset = staging.len() as u64;
                    let mut j = i;
                    while j < shadow_items.len() && shadow_items[j].0 == key {
                        staging.push(shadow_items[j].1.slot.0);
                        j += 1;
                    }
                    let mut dc = shadow_items[i].1.dc.clone();
                    dc.bounds_center          = shadow_items[i].1.bounding_center.to_array();
                    dc.instance_buffer        = Some(Arc::clone(&scene_buf));
                    dc.instance_buffer_offset = batch_offset * INSTANCE_STRIDE;
                    dc.instance_count         = (j - i) as u32;
                    sdl.push(dc);
                    i = j;
                }
            }

            // Upload all transforms in a single GPU write.
            if !staging.is_empty() {
                self.queue.write_buffer(&scene_buf, 0, bytemuck::cast_slice(&staging));
            }
        }

        // ── Step 0.5: Upload dirty GPU Scene primitives ───────────────────────
        //
        // Must happen before the render graph executes so the vertex shaders see
        // up-to-date model transforms.  flush_dirty() coalesces contiguous dirty
        // slots into the fewest possible write_buffer calls.
        self.gpu_scene.flush_dirty(&self.queue);

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

        // ── Encoder + pre-AA clear (still part of prep) ──────────────────────
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        // Always render scene to HDR intermediate buffer.
        // The post-process stack converts HDR → LDR (→ pre_aa_view when AA enabled, else swapchain).
        let graph_target = &self.hdr_view;

        // Clear HDR buffer at the start of each frame
        {
            let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("HDR Clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.hdr_view,
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
            gpu_scene_bind_group: &self.gpu_scene.bind_group,
            sky_color: self.scene_sky_color,
            has_sky: self.scene_has_sky,
            sky_state_changed: self.sky_state_changed,
            sky_bind_group: None,
            camera_position: camera.position,
            camera_forward: camera.forward(),
            draw_list_generation: self.draw_list_generation,
        };

        let profiling_active = self.debug_printout || self.live_portal.is_some();

        let graph_start = std::time::Instant::now();
        if profiling_active {
            if let Some(p) = &mut self.profiler { p.begin_frame(); }
            self.graph.execute(&mut graph_ctx, self.profiler.as_mut())?;
        } else {
            self.graph.execute(&mut graph_ctx, None)?;
        }
        let graph_ms = graph_start.elapsed().as_secs_f32() * 1000.0;

        // ── HDR post-process stack ────────────────────────────────────────────
        // SSR → God Rays → Bloom → DoF → PostProcess (HDR → LDR)
        let hdr_start = std::time::Instant::now();

        // SSR, God Rays, Physical Bloom, and DoF are disabled — core BRDF improvements first.
        // TODO: re-enable SSR, god_rays_pass, bloom_pass, dof_pass once core pipeline is solid.

        // Final PostProcess: HDR → LDR (→ pre_aa_view when AA enabled, else direct to swapchain)
        {
            use crate::passes::PostProcessUniforms;
            let uniforms = PostProcessUniforms {
                frame: self.frame_count as u32,
                ..Default::default()
            };
            self.queue.write_buffer(&self.post_process_pass.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        }
        let pp_target = match self.aa_mode {
            AntiAliasingMode::None | AntiAliasingMode::Msaa(_) => target,
            _ => &self.pre_aa_view,
        };
        self.post_process_pass.execute(&mut encoder, pp_target)?;

        let _hdr_ms = hdr_start.elapsed().as_secs_f32() * 1000.0;

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

            // object_count: registered persistent proxies (enabled + disabled).
            // light/billboard counts come from the latest scene env layout when available.
            let obj_count = self.registered_objects.len();
            let (light_count, bb_count) = if let Some(layout) = &current_layout {
                (layout.lights.len(), layout.billboards.len())
            } else { (0, 0) };

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

        // Recreate HDR intermediate buffer and all bind groups that reference it.
        let (hdr_texture, hdr_view) = helpers::create_hdr_texture(&self.device, width, height);
        let (hdr_prev_texture, hdr_prev_view) = helpers::create_hdr_texture(&self.device, width, height);
        self.hdr_texture      = hdr_texture;
        self.hdr_view         = hdr_view;
        self.hdr_prev_texture = hdr_prev_texture;
        self.hdr_prev_view    = hdr_prev_view;

        // Rebuild G-buffer views (normal + ORM + depth) for SSR
        let gbuf_normal_view = self.gbuf_normal_texture.create_view(&Default::default());
        let gbuf_orm_view    = self.gbuf_orm_texture.create_view(&Default::default());

        self.post_process_pass.create_bind_group(&self.device, &self.hdr_view);
        // Bloom mip textures must be resized; recreate the whole pass.
        let bloom_config = self.bloom_pass.config;
        if let Ok(new_bloom) = PhysicalBloomPass::new(&self.device, width, height, bloom_config) {
            self.bloom_pass = new_bloom;
        }
        self.bloom_pass.create_bind_groups(&self.device, &self.hdr_view);
        self.ssr_pass.create_bind_group(
            &self.device, &gbuf_normal_view, &gbuf_orm_view,
            &self.depth_sample_view, &self.hdr_prev_view,
        );
        self.god_rays_pass.create_bind_group(&self.device, &self.hdr_prev_view, &self.depth_view);
        self.dof_pass.create_bind_groups_resized(&self.device, &self.hdr_view, &self.depth_sample_view, &self.camera_buffer, width, height);
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

// ── Module-level helpers ──────────────────────────────────────────────────────

/// FNV-1a hash of a single [f32; 16] transform matrix.
/// Used by `add_object` and `update_transform` to skip redundant `write_buffer`s.
#[inline]
fn fnv1a_mat(mat: &[f32; 16]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &f in mat {
        h ^= f.to_bits() as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
