use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use helio_core::{RenderFrameInputs, RenderGraph, RenderPass};

use super::builder::SceneDbHandle;
use super::config::{RenderMode, RendererConfig};
#[cfg(not(target_arch = "wasm32"))]
use super::stereo_mirror::StereoMirror;

/// Closure that rebuilds the render graph when graph-level configuration changes.
pub type GraphRebuilder = Arc<
    dyn Fn(
            &Arc<wgpu::Device>,
            &Arc<wgpu::Queue>,
            RendererConfig,
            Arc<Mutex<DebugDrawState>>,
            &wgpu::Buffer,
            &wgpu::Buffer,
            &wgpu::Buffer,
        ) -> RenderGraph
        + Send
        + Sync,
>;

/// Reapplies application-owned pass settings after a graph rebuild.
pub type GraphRebuildHook = Arc<dyn Fn(&mut RenderGraph, &wgpu::Device) + Send + Sync>;

use crate::camera::Camera;
use helio_mats::radiant::{RadiantTemplateRegistry, SharedTemplateRegistry};

use super::debug::DebugDrawState;

/// Backend-only fallback bindings for the material sampling contract.
///
/// Material parameters and texture references are authored in SceneDB. The
/// renderer keeps only the descriptor objects required by the passes to bind
/// that data. SceneDB texture-store slots supply cached views; vacant slots use the white
/// fallback. Frontends must clear material slot references before freeing them.
pub(crate) struct MaterialBindingResources {
    pub(crate) _fallback_texture: wgpu::Texture,
    pub(crate) fallback_view: wgpu::TextureView,
    pub(crate) fallback_sampler: wgpu::Sampler,
    pub(crate) texture_count: usize,
    pub(crate) version: u64,
    pub(crate) scene_views: Vec<Option<(wgpu::Texture, wgpu::TextureView)>>,
}

pub use helio_pass_billboard::BillboardInstance;

pub(crate) enum CullStatsReadbackState {
    Idle,
    Mapping(Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>>),
    Disabled,
}

pub struct Renderer {
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    pub(crate) graph: RenderGraph,
    pub(crate) camera_buffer: wgpu::Buffer,
    pub(crate) camera_data: helio_core::GpuCameraUniforms,
    /// Bumped only when [`CameraIdentity`] changes -- see
    /// [`Renderer::note_camera`].
    pub(crate) camera_generation: u64,
    /// The camera `camera_generation` currently describes.
    pub(crate) camera_identity: Option<CameraIdentity>,
    pub(crate) frame_count: u64,
    pub(crate) ray_frame: helio_core::FrameAcceleration,
    /// Previous projection per stable camera/view id, and the world origin
    /// it is expressed in. Stereo callers submit eyes sequentially, so a
    /// single matrix would make each eye reproject from the other eye's
    /// camera.
    pub(crate) prev_view_proj_by_view: HashMap<u32, (glam::Mat4, glam::DVec3)>,
    /// GPU work deriving drawable scene buffers from the frontend's authored
    /// ones, run each frame between the SceneDB snapshot and the graph. See
    /// `helio_core::scene_derivation`.
    pub(crate) scene_derivations: Vec<Box<dyn helio_core::SceneDerivation>>,
    /// This frame's world origin: every frame is camera-relative, its
    /// coordinates world positions minus this. Follows the eye
    /// ([`Renderer::origin_for`]).
    pub(crate) world_origin: glam::DVec3,
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
    pub(crate) material_bindings: MaterialBindingResources,
    pub(crate) ambient_color: [f32; 3],
    pub(crate) ambient_intensity: f32,
    pub(crate) clear_color: [f32; 4],
    /// The configuration the current graph was built from: the recipe a
    /// rebuild hands the graph builder, so the new graph has the same passes
    /// and pass settings (Helio#254/#255).
    ///
    /// Opaque to the renderer: it never interprets pass-specific fields
    /// (shadow atlas, SSR, foliage, portals, TSR, reflections), it only
    /// stores what the graph was built with. The size, scale, surface format,
    /// debug mode, render mode and XR flag have live fields of their own;
    /// [`Renderer::renderer_config`] overlays those onto this.
    pub(crate) graph_config: RendererConfig,
    /// Coordinate-space transforms supplied by the frontend for portal and
    /// sublevel instances. Kept on Renderer so graph rebuilds cannot reset
    /// the G-buffer's table back to identity.
    pub(crate) coordinate_spaces: Vec<glam::Mat4>,
    /// Active dense portal projection counts supplied by the resolver bridge.
    /// Growable SceneDB buffers are capacity-sized, so passes must not infer
    /// active rows from their allocation size.
    pub(crate) portal_projection_counts: Option<(u32, u32)>,
    /// Whether the current renderer output is an XR eye view.
    pub(crate) xr_mode: bool,
    /// TSR quality preset, preserved across graph rebuilds.
    pub(crate) tsr_quality: Option<helio_pass_tsr::TsrQuality>,
    pub(crate) debug_mode: u32,
    pub(crate) editor_mode: bool,
    pub(crate) debug_state: Arc<Mutex<DebugDrawState>>,
    pub(crate) last_render_time: Instant,
    pub(crate) delta_time: f32,
    /// Application-authored textures handed to the graph (Helio#257).
    ///
    /// Same accepted exception as `template_registry`: the application
    /// creates these once and passes them in, and the renderer only forwards
    /// them into the frame registry (`"color_grading_lut"` for PostProcessPass,
    /// `"ies_textures"` for DeferredLightPass). The renderer never creates,
    /// sizes or formats them, so it holds no pass-internal state here.
    pub(crate) color_grading_lut_view: Option<wgpu::TextureView>,
    pub(crate) ies_texture_view: Option<wgpu::TextureView>,
    pub(crate) graph_time_ms: f32,
    /// The last frame's graph submission. See [`Self::last_submission`].
    pub(crate) last_submission: Option<wgpu::SubmissionIndex>,
    pub(crate) cull_stats_staging: wgpu::Buffer,
    pub(crate) cull_stats_readback_state: CullStatsReadbackState,
    pub(crate) cull_stats: [u32; 8],
    pub(crate) frame_times: Vec<f32>,
    pub(crate) frame_times_cursor: usize,
    /// Whether per-frame subpixel camera jitter is applied. Graphs with a
    /// temporal accumulation pass (TaaPass, TsrPass) need this; non-temporal
    /// graphs (e.g. FXAA-only) disable it to avoid visible shimmer.
    pub(crate) enable_jitter: bool,
    pub(crate) camera_jitter_override: Option<[f32; 2]>,
    pub(crate) frame_delta_override: Option<f32>,
    /// The frame clock animation reads (`PrepareContext::time`), in seconds.
    pub(crate) frame_clock: f64,
    /// How far the host's clock advances the frame clock each frame; `None`
    /// follows the frame delta (wall time, or `frame_delta_override`).
    pub(crate) frame_clock_delta: Option<f32>,
    /// A bake to run before the next frame. Its result is owned by the
    /// graph's `BakeInjectPass`, never by the renderer (Helio#256).
    #[cfg(feature = "bake")]
    pub(crate) bake_pending: Option<helio_bake::BakeRequest>,
    /// Optional CPU bake projection supplied by the SceneDB/frontend owner.
    /// The renderer may execute it, but never traverses or synthesizes scene
    /// entities to build it.
    #[cfg(feature = "bake")]
    pub(crate) bake_scene: Option<helio_bake::SceneGeometry>,
    pub(crate) owns_device: bool,
    pub(crate) pending_resize: Option<(u32, u32)>,
    /// A pending size update also needs a fresh graph because graph-level
    /// settings or the pass topology changed. Ordinary viewport resizes keep
    /// the existing passes and resize their resources in place.
    pub(crate) pending_graph_rebuild: bool,
    pub(crate) clear_target_next_frame: bool,
    pub(crate) graph_rebuilder: Option<GraphRebuilder>,
    pub(crate) graph_rebuild_hook: Option<GraphRebuildHook>,
    /// Shader hot-reload bookkeeping (see `shader_reload.rs`).
    #[cfg(all(feature = "shader-hot-reload", not(target_arch = "wasm32")))]
    pub(crate) shader_reload: super::shader_reload::ShaderReloadState,
    /// Frontend-owned SceneDB GPU projection. The CPU SceneDB remains outside
    /// Helio and is flushed by its owner at the frame boundary.
    pub(crate) scene_db: SceneDbHandle,

    /// Engine-world transform of the headset's stage origin — the locomotion hook.
    ///
    /// Identity means the player stands at the world origin. Translating/rotating this
    /// moves the player through the world without touching the scene, which is what
    /// joystick locomotion drives. Applied to the located eye poses, so it moves the
    /// cameras and nothing else.
    pub(crate) xr_stage_transform: glam::Mat4,
    /// Templates registered by the user for the gbuffer (opaque) path.
    /// Preserved across graph rebuilds. Shared (not cloned) with
    /// passes and the GPU scene via `Arc<RwLock<_>>` — see
    /// `SharedTemplateRegistry` for why cloning this must be avoided.
    pub(crate) template_registry: SharedTemplateRegistry,

    /// Templates registered by the user for the transparent path.
    /// Used by TransparentPass for alpha-blended materials (water, glass, etc.).
    pub(crate) transparent_template_registry: SharedTemplateRegistry,
    pub(crate) render_mode: RenderMode,
    /// Whether the render graph was built in OpenXR multiview mode. Mirrors
    /// `config.enable_xr` and is preserved across graph rebuilds so an
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
    /// `view_id`, near/far and the representative position used
    /// for the debug state. The per-eye view/proj are overridden
    /// by the headset each frame.
    pub(crate) xr_camera: Option<Camera>,
    #[cfg(not(target_arch = "wasm32"))]
    /// PC mirror blit: samples the acquired XR swapchain image (2-layer array)
    /// and draws both eyes side-by-side to the mirror window surface.
    pub(crate) xr_mirror: Option<StereoMirror>,
    #[cfg(not(target_arch = "wasm32"))]
    /// Colour format of the PC mirror window surface.
    pub(crate) xr_mirror_format: Option<wgpu::TextureFormat>,
    /// Pixel size of the current app window mirror target.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) xr_mirror_size: [u32; 2],
}

impl Renderer {
    /// Publish an already-built frontend acceleration projection (its TLAS
    /// and the rows that follow its instance order) for the next frame
    /// only. Call after flushing SceneDB and preparing its BLAS/TLAS; an
    /// omitted update expires instead of reusing potentially stale geometry.
    pub fn set_ray_tracing_frame(&mut self, rays: helio_core::RayFrame<'_>) {
        self.ray_frame.publish(self.frame_count, rays);
    }

    /// Raw depth-buffer texture (`Depth32Float`, already `COPY_SRC`) for
    /// external debug capture (e.g. an example dumping it to a PNG to
    /// answer "is anything actually being rasterized"). Every other
    /// render-graph buffer (G-buffer, shadow atlas, etc.) is privately
    /// owned inside its own pass crate and only reachable as a
    /// `wgpu::TextureView` via `ResourceRegistry` -- no path back to the
    /// owning `Texture` a GPU readback needs -- so this is deliberately
    /// the one buffer `Renderer` itself still owns directly, not a general
    /// "every buffer" debug API.
    pub fn debug_depth_texture(&self) -> &wgpu::Texture {
        &self.depth_texture
    }

    pub(crate) fn upload_camera(&mut self, camera: &Camera) {
        // This view's last projection, rebased from the origin it was in; a
        // view's first frame has none (no motion).
        let previous_projection = match self.prev_view_proj_by_view.get(&camera.view_id) {
            Some(&(previous, origin)) => helio_core::temporal::rebase_previous_projection(
                previous,
                self.world_origin - origin,
            ),
            None => camera.view_proj_in(self.world_origin),
        };
        let uniforms = helio_core::GpuCameraUniforms::new(
            camera.view_in(self.world_origin),
            camera.proj,
            camera.position_in(self.world_origin),
            camera.near,
            camera.far,
            self.frame_count as u32,
            camera.jitter,
            previous_projection,
        )
        .with_view_id(camera.view_id);
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniforms));
        self.prev_view_proj_by_view.insert(
            camera.view_id,
            (
                glam::Mat4::from_cols_array(&uniforms.view_proj),
                self.world_origin,
            ),
        );
        self.camera_data = uniforms;
    }

    /// Advance `camera_generation` if `camera` (unjittered) differs from the
    /// one it last described. Call with the camera as supplied, before any
    /// per-frame jitter.
    ///
    /// The generation promises "the view or projection changed"; passes cache
    /// camera-dependent work on it (light-cull tile lists, the Hi-Z max
    /// pyramid). It used to advance on every upload, so those caches never
    /// hit (Pulsar-Native#834). TAA/TSR jitter and the frame counter change
    /// the uploaded uniforms every frame by design and are not part of it.
    pub(crate) fn note_camera(&mut self, camera: &crate::Camera) {
        let identity = CameraIdentity::of(camera);
        if self.camera_identity != Some(identity) {
            self.camera_identity = Some(identity);
            self.camera_generation = self.camera_generation.wrapping_add(1);
        }
    }

    /// The world origin of the next frame, for a camera with its eye at
    /// `eye`: the origin follows the eye in cells of
    /// [`crate::camera::WORLD_ORIGIN_CELL`] and moves only when the eye leaves
    /// its cell. [`Renderer::render`] calls it with the camera's eye; a host
    /// that writes data in frame coordinates (relative to the origin) calls it
    /// first, with the same eye, to know the origin.
    pub fn origin_for(&mut self, eye: glam::DVec3) -> glam::DVec3 {
        self.world_origin = crate::camera::follow_origin(Some(self.world_origin), eye);
        self.world_origin
    }

    /// The world origin of the last (or current) frame.
    pub fn world_origin(&self) -> glam::DVec3 {
        self.world_origin
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn upload_stereo_camera(
        &mut self,
        left: &helio_core::GpuCameraUniforms,
        right: &helio_core::GpuCameraUniforms,
    ) {
        // Preserve temporal reprojection independently for each eye. The XR
        // path submits eyes sequentially, but both eyes belong to the same
        // headset frame; `xr_view_to_camera` initializes prev_view_proj to the
        // current matrix, which would make every frame behave like a camera
        // cut and leave stale history behind during head motion.
        let mut left = *left;
        let mut right = *right;
        for camera in [&mut left, &mut right] {
            let view_id = camera.jitter_frame[3].to_bits();
            let current = glam::Mat4::from_cols_array(&camera.view_proj);
            let previous =
                self.prev_view_proj_by_view
                    .get(&view_id)
                    .map_or(current, |&(previous, origin)| {
                        helio_core::temporal::rebase_previous_projection(
                            previous,
                            self.world_origin - origin,
                        )
                    });
            camera.prev_view_proj = previous.to_cols_array();
            self.prev_view_proj_by_view
                .insert(view_id, (current, self.world_origin));
        }
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[left, right]));
        self.camera_data = left;
        // A tracked headset moves every frame; treat each stereo upload as a
        // new view, and make the next mono frame compare afresh.
        self.camera_identity = None;
        self.camera_generation = self.camera_generation.wrapping_add(1);
    }

    /// Change the shadow quality. The passes that use it are configured at
    /// graph build, so this rebuilds the graph before the next frame; it
    /// used to only take effect at the next window resize.
    pub fn set_shadow_quality(&mut self, quality: helio_pass_shadow_matrix::ShadowQuality) {
        if self.graph_config.shadow_quality != quality {
            self.graph_config.shadow_quality = quality;
            self.request_graph_rebuild();
        }
    }

    /// Rebuild the graph from [`Self::renderer_config`] before the next frame.
    /// No-op for graphs installed without a rebuilder.
    pub fn request_graph_rebuild(&mut self) {
        self.pending_graph_rebuild = true;
        self.pending_resize = Some((self.output_width, self.output_height));
    }

    /// Overrides the graph-derived per-frame camera-jitter setting.
    ///
    /// Graphs containing a temporal reconstruction pass enable jitter
    /// automatically. FXAA and other non-temporal graphs leave it disabled so
    /// the final image remains pixel-stable.
    pub fn set_jitter_enabled(&mut self, enabled: bool) {
        self.enable_jitter = enabled;
    }

    /// Set a deterministic projection offset in internal render-pixel units
    /// for the standard, non-XR render path.
    /// `None` restores the graph's normal temporal sampling sequence.
    pub fn set_camera_jitter_override(&mut self, jitter: Option<[f32; 2]>) {
        assert!(jitter.is_none_or(|v| v.iter().all(|x| x.is_finite())));
        self.camera_jitter_override = jitter;
    }

    /// Fix temporal-filter time for offline captures in the standard, non-XR
    /// render path so readback latency does not alter history weights.
    /// `None` restores measured frame time.
    pub fn set_frame_delta_override(&mut self, seconds: Option<f32>) {
        assert!(seconds.is_none_or(|v| v.is_finite() && v > 0.0));
        self.frame_delta_override = seconds;
    }

    /// Drive the frame clock animation reads (material graph `time`, foliage
    /// wind, particles; `PrepareContext::time`) from the host's clock: each
    /// frame advances it by `seconds`, until changed. `Some(0.0)` freezes it
    /// (an editor viewport that is not realtime, a paused game); a game
    /// passes its clock's delta every frame, so animation follows pause and
    /// time dilation. `None` (the default) advances it by the frame delta:
    /// wall time, or [`Self::set_frame_delta_override`].
    pub fn set_frame_clock_delta(&mut self, seconds: Option<f32>) {
        assert!(seconds.is_none_or(|v| v.is_finite() && v >= 0.0));
        self.frame_clock_delta = seconds;
    }

    /// The frame clock (`PrepareContext::time`) as of the last frame, in
    /// seconds.
    pub fn frame_clock(&self) -> f64 {
        self.frame_clock
    }

    /// Advance the frame clock for a frame whose delta is `delta_time` and
    /// hand it to the graph.
    pub(crate) fn advance_frame_clock(&mut self, delta_time: f32) {
        let advance = self.frame_clock_delta.unwrap_or(delta_time);
        self.frame_clock += f64::from(advance);
        self.graph.set_frame_clock(self.frame_clock as f32, advance);
    }

    /// Set the renderer-wide debug visualization mode.
    pub fn set_debug_mode(&mut self, mode: u32) {
        self.debug_mode = mode;
        self.graph.set_debug_mode(mode);
    }

    /// Return owned descriptors for the debug views advertised by the graph.
    pub fn available_debug_views(&self) -> Vec<helio_core::DebugViewDescriptor> {
        self.graph.collect_debug_views()
    }

    /// Color legends advertised by passes for the editor's debug views.
    pub fn available_debug_view_legends(&self) -> Vec<(u32, helio_core::DebugViewLegend)> {
        self.graph.collect_debug_view_legends()
    }

    /// Menu categories (`(debug_mode, category)`) of the debug views whose
    /// pass names one.
    pub fn available_debug_view_categories(&self) -> Vec<(u32, &'static str)> {
        self.graph.collect_debug_view_categories()
    }

    pub fn set_editor_mode(&mut self, enabled: bool) {
        self.editor_mode = enabled;
        if let Ok(mut s) = self.debug_state.lock() {
            s.editor_enabled = enabled;
        }
    }

    /// Set the world-anchored pixel text shown by the editor's portal overlay.
    pub fn set_portal_debug_texts(
        &mut self,
        texts: Vec<helio_pass_portal_instances::PortalDebugText>,
    ) -> bool {
        self.find_pass_mut::<helio_pass_portal_instances::PortalEditorOverlayPass>()
            .is_some_and(|pass| pass.set_debug_texts(texts))
    }

    /// Set transient portal-through geometry for the editor diagnostics view.
    pub fn set_portal_debug_geometry(&mut self, vertices: Vec<[f64; 3]>) -> bool {
        self.find_pass_mut::<helio_pass_portal_instances::PortalEditorOverlayPass>()
            .is_some_and(|pass| pass.set_ghost_vertices(vertices))
    }

    /// Pass the latest asynchronous DeferredLight timestamp to its cost view.
    pub fn set_deferred_light_timing_ms(&mut self, milliseconds: f32) -> bool {
        self.find_pass_mut::<helio_pass_deferred_light::DeferredLightPass>()
            .is_some_and(|pass| pass.set_measured_lighting_ms(milliseconds))
    }

    pub fn is_editor_mode(&self) -> bool {
        self.editor_mode
    }

    pub fn shadow_quality(&self) -> helio_pass_shadow_matrix::ShadowQuality {
        self.graph_config.shadow_quality
    }

    /// Return the frontend-owned SceneDB GPU projection.
    /// construction. The handle is exposed for pass integration and debug
    /// tooling; Helio does not take ownership of scene content.
    pub fn scene_db(&self) -> SceneDbHandle {
        self.scene_db.clone()
    }

    /// Return the shared debug-drawing state used by debug passes.
    pub fn debug_state(&self) -> Arc<Mutex<DebugDrawState>> {
        self.debug_state.clone()
    }

    /// The scene camera buffer (`STORAGE | UNIFORM | COPY_DST`, two
    /// `GpuCameraUniforms`) — what passes bind as *the* camera. Not to be
    /// confused with [`Self::debug_camera_buf`], a 64-byte `UNIFORM`-only
    /// view-projection buffer that only `DebugDrawPass` may read; binding it
    /// where a pass declares a storage camera fails wgpu validation.
    pub fn camera_buf(&self) -> &wgpu::Buffer {
        &self.camera_buffer
    }

    pub fn debug_camera_buf(&self) -> &wgpu::Buffer {
        &self.debug_camera_buffer
    }

    pub fn cull_stats_buf(&self) -> &wgpu::Buffer {
        &self.cull_stats_buffer
    }

    /// Latest frame timing state. Reading it performs no GPU polling,
    /// synchronization, or allocation.
    pub fn timing_snapshot(&self) -> &helio_core::RenderTimingSnapshot {
        self.graph.profiler().timing_snapshot()
    }

    /// Process-local identity for the timing snapshot's graph profiler.
    /// Combine this with a GPU frame index to disambiguate graph rebuilds.
    pub fn profiling_instance_id(&self) -> u64 {
        self.graph.profiler().instance_id()
    }

    /// GPU time across the graph's compute and graphics work, excluding CPU
    /// work and presentation. Unlike a legacy per-pass sum, this is `None`
    /// until a completed whole-graph timing scope is available.
    pub fn gpu_frame_ms(&self) -> Option<f32> {
        self.graph.profiler().gpu_frame_ms()
    }

    /// Latest graph topology/resource timeline for editor diagnostics.
    ///
    /// This is a host-facing snapshot: it contains no live wgpu handles and
    /// can safely be copied across the renderer/UI boundary.
    pub fn graph_timeline(&self) -> helio_core::GraphTimelineData {
        self.graph.collect_graph_timeline()
    }

    /// The queue submission carrying the last rendered frame's graph, for a
    /// host that must wait on or hand off the frame (a compositor sampling
    /// the target). Saves the host an extra empty `queue.submit` just to
    /// obtain an index.
    pub fn last_submission(&self) -> Option<wgpu::SubmissionIndex> {
        self.last_submission.clone()
    }

    /// Per-pass `CommandEncoder::finish` cost for the last frame; see
    /// [`helio_core::RenderGraph::set_finish_breakdown`]. Enable it with the
    /// `HELIO_FINISH_BREAKDOWN` environment variable so rebuilt graphs keep it.
    pub fn finish_breakdown(&self) -> &[helio_core::FinishSegment] {
        self.graph.finish_breakdown()
    }

    /// Which passes the recording cache resubmitted or re-encoded, and why;
    /// see [`helio_core::RenderGraph::set_recording_cache`]. The cache is on
    /// by default; `HELIO_RECORDING_CACHE=0` switches it off for every graph,
    /// rebuilt ones included.
    pub fn recording_cache_stats(&self) -> helio_core::RecordingCacheStats {
        self.graph.recording_cache_stats()
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

    /// Configure passes in the current graph and every graph created by resize.
    /// Use this for settings applied directly to passes, which a fresh graph
    /// would otherwise reset to defaults.
    pub fn set_graph_rebuild_hook(
        &mut self,
        hook: impl Fn(&mut RenderGraph, &wgpu::Device) + Send + Sync + 'static,
    ) {
        hook(&mut self.graph, &self.device);
        self.graph_rebuild_hook = Some(Arc::new(hook));
    }

    /// Set the coordinate-space table through the generic graph input seam
    /// (the non-identity tail: slot 0 is the world and implicit). The host owns
    /// the one table: every producer of spaces (rigid frames, sublevel
    /// placements, portal projections) takes its slots from the host's
    /// [`helio_core::CoordinateSpaceAllocator`], and the host hands the result
    /// here. The owning pass uploads its GPU table during prepare.
    pub fn set_coordinate_spaces(&mut self, spaces: &[glam::Mat4]) {
        self.coordinate_spaces.clear();
        self.coordinate_spaces.extend_from_slice(spaces);
        self.apply_coordinate_spaces();
    }

    /// Upload a resolver-generated portal projection frame built with its own
    /// table (`PortalProjectionBridge::build`): its spaces become the table.
    /// A host that shares the table with other producers builds with
    /// `build_into` and calls [`Self::set_coordinate_spaces`] and
    /// [`Self::set_portal_projection_counts`] instead.
    pub fn set_portal_projection_frame(
        &mut self,
        frame: &helio_pass_portal_cull::PortalProjectionFrame,
    ) {
        self.coordinate_spaces.clear();
        self.coordinate_spaces
            .extend_from_slice(frame.renderer_coordinate_spaces());
        self.set_portal_projection_counts(
            frame.counts.portal_view_count,
            frame.counts.portal_chain_count,
        );
    }

    /// The live portal view and chain counts of a projection whose spaces the
    /// host placed in the table itself.
    pub fn set_portal_projection_counts(&mut self, portal_views: u32, portal_chains: u32) {
        self.portal_projection_counts = Some((portal_views, portal_chains));
        self.apply_coordinate_spaces();
    }

    pub(crate) fn apply_coordinate_spaces(&mut self) {
        let inputs = RenderFrameInputs {
            coordinate_spaces: &self.coordinate_spaces,
            projection_counts: self
                .portal_projection_counts
                .map(|(view_count, chain_count)| [view_count, chain_count]),
            xr_mode: self.xr_mode,
        };
        self.graph.set_frame_inputs(&inputs);
    }

    /// Select the renderer's XR quality path. Hosts that submit eye views
    /// through `render()` rather than `render_xr()` should enable this once
    /// their XR session starts and disable it when the session ends.
    pub(crate) fn set_xr_mode(&mut self, enabled: bool) {
        if self.xr_mode != enabled {
            self.xr_mode = enabled;
            self.apply_coordinate_spaces();
        }
    }

    /// Access the gbuffer template registry (preserved across graph rebuilds).
    /// Register custom surface templates here instead of through the pass
    /// directly to ensure they survive window resize.
    pub fn template_registry_mut(
        &mut self,
    ) -> std::sync::RwLockWriteGuard<'_, RadiantTemplateRegistry> {
        self.template_registry
            .write()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// Access the transparent template registry for alpha-blended materials.
    pub fn transparent_template_registry_mut(
        &mut self,
    ) -> std::sync::RwLockWriteGuard<'_, RadiantTemplateRegistry> {
        self.transparent_template_registry
            .write()
            .unwrap_or_else(|e| e.into_inner())
    }

    pub fn set_clear_color(&mut self, color: [f32; 4]) {
        self.clear_color = color;
    }

    /// Replace the shared sampler used by SceneDB material texture slots.
    /// Incrementing the binding version refreshes pass descriptor caches.
    pub fn set_material_sampler(&mut self, descriptor: &wgpu::SamplerDescriptor<'_>) {
        self.material_bindings.fallback_sampler = self.device.create_sampler(descriptor);
        self.material_bindings.version = self.material_bindings.version.wrapping_add(1);
    }

    pub fn set_ambient(&mut self, color: [f32; 3], intensity: f32) {
        self.ambient_color = color;
        self.ambient_intensity = intensity;
    }

    pub fn set_graph(&mut self, mut graph: RenderGraph) {
        // Extract rebuilder stored in the graph by the builder function
        self.graph_rebuilder = graph.take_graph_data::<GraphRebuilder>();
        self.replace_graph(graph);
    }

    pub fn set_graph_with_builder(&mut self, graph: RenderGraph, rebuilder: GraphRebuilder) {
        self.replace_graph(graph);
        self.graph_rebuilder = Some(rebuilder);
    }

    pub fn set_rebuilder(&mut self, rebuilder: GraphRebuilder) {
        self.graph_rebuilder = Some(rebuilder);
    }

    #[cfg(feature = "bake")]
    pub fn configure_bake(&mut self, request: helio_bake::BakeRequest) {
        self.bake_pending = Some(request);
    }

    /// Supply the explicit CPU bake projection assembled by the SceneDB
    /// owner. This replaces the removed renderer scene traversal.
    #[cfg(feature = "bake")]
    pub fn set_bake_scene(&mut self, scene: helio_bake::SceneGeometry) {
        self.bake_scene = Some(scene);
    }

    /// Bake the scene last given to [`Self::set_bake_scene`] before the next
    /// frame. Build it with [`crate::bake_scene_from_world`]. Returns whether a
    /// bake was queued, so a missing scene is not a silent no-op (Helio#256).
    #[cfg(feature = "bake")]
    pub fn auto_bake(&mut self, config: helio_bake::BakeConfig) -> bool {
        let Some(scene) = self.bake_scene.clone() else {
            log::error!(
                "Renderer::auto_bake requires set_bake_scene(helio::bake_scene_from_world(..)) first; nothing was baked"
            );
            return false;
        };
        self.configure_bake(helio_bake::BakeRequest { scene, config });
        true
    }

    /// Hand a finished bake to the graph. `BakeInjectPass` owns it from here
    /// and publishes it every frame; a re-bake replaces the previous one.
    #[cfg(feature = "bake")]
    pub(crate) fn install_baked_data(&mut self, baked: std::sync::Arc<helio_bake::BakedData>) {
        let pass = Box::new(helio_bake::BakeInjectPass::new(baked));
        match self.graph.pass_index_of::<helio_bake::BakeInjectPass>() {
            Some(index) => self.graph.replace_pass_at(index, pass),
            None => self.graph.add_pass_live(pass),
        }
    }

    /// The baked data the current graph publishes, for carrying it into a
    /// rebuilt graph.
    #[cfg(feature = "bake")]
    pub(crate) fn installed_baked_data(&self) -> Option<std::sync::Arc<helio_bake::BakedData>> {
        self.graph
            .find_pass::<helio_bake::BakeInjectPass>()
            .map(|pass| pass.baked_data().clone())
    }

    /// Replace the graph, keeping what the old one owned that a freshly
    /// built graph cannot know about (a finished bake, the debug mode for
    /// passes that only learn it through `set_debug_mode`).
    pub(crate) fn replace_graph(&mut self, mut graph: RenderGraph) {
        graph.set_debug_mode(self.debug_mode);
        #[cfg(feature = "bake")]
        let baked = self.installed_baked_data();
        self.graph = graph;
        #[cfg(feature = "bake")]
        if let Some(baked) = baked {
            self.install_baked_data(baked);
        }
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

    /// The config a rebuilt graph is built from: what the current graph was
    /// built with, at the current size, scale and modes.
    pub fn renderer_config(&self) -> RendererConfig {
        RendererConfig {
            width: self.output_width,
            height: self.output_height,
            surface_format: self.surface_format,
            debug_mode: self.debug_mode,
            render_scale: self.render_scale,
            tsr_quality: self.tsr_quality,
            render_mode: self.render_mode,
            enable_xr: self.enable_xr,
            ..self.graph_config
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
    pub(crate) fn set_xr_session(
        &mut self,
        instance: Option<helio_xr::instance::XrInstance>,
        session: Option<helio_xr::session::XrSession>,
        swapchain: Option<helio_xr::swapchain::XrSwapchain>,
    ) {
        self.set_xr_mode(session.is_some());
        self.xr_instance = instance;
        self.xr = session;
        self.xr_swapchain = swapchain;
    }

    /// Install a fully bootstrapped Helio XR runtime and return its input
    /// actions. The renderer must have been built with the runtime's device;
    /// construct `RendererConfig` from `runtime.eye_extent()` and the XR
    /// swapchain format before creating the renderer.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn install_xr_runtime(
        &mut self,
        runtime: helio_xr::XrRuntime,
    ) -> helio_core::Result<helio_xr::XrInput> {
        if !Arc::ptr_eq(&self.device, &runtime.device) {
            return Err(helio_core::Error::InvalidPassConfig(
                "XR runtime and renderer must share the same wgpu device".into(),
            ));
        }

        let helio_xr::XrRuntime {
            instance,
            session,
            swapchain,
            input,
            ..
        } = runtime;
        self.set_xr_session(Some(instance), Some(session), Some(swapchain));
        self.set_jitter_enabled(false);
        Ok(input)
    }

    /// Predicted display time of the most recent XR frame, used to locate
    /// controller poses (see `helio_xr::XrInput::grip_pose_matrices`) at the
    /// same instant the eye views were located. `None` in desktop mode or
    /// before the first XR frame has been waited on.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn xr_last_display_time(&self) -> Option<helio_xr::Time> {
        self.xr.as_ref().map(|session| session.last_display_time)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn xr_session_focused(&self) -> bool {
        self.xr
            .as_ref()
            .is_some_and(helio_xr::session::XrSession::is_focused)
    }

    /// Current OpenXR headset refresh rate in Hz, if the active runtime can
    /// report or infer it. `None` in desktop mode or before enough XR timing
    /// data has arrived.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn xr_display_refresh_rate_hz(&self) -> Option<f32> {
        self.xr
            .as_ref()
            .and_then(helio_xr::session::XrSession::display_refresh_rate_hz)
    }

    /// Ask the OpenXR runtime for an exact supported headset display rate.
    /// The request is optional and may be declined;
    /// query [`Self::xr_display_refresh_rate_hz`] on subsequent frames to see
    /// which rate the runtime actually adopted.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn request_xr_display_refresh_rate_hz(
        &self,
        requested_hz: f32,
    ) -> helio_xr::Result<bool> {
        match self.xr.as_ref() {
            Some(session) => session.request_display_refresh_rate_hz(requested_hz),
            None => Ok(false),
        }
    }

    /// Set the camera template used for XR frames (postprocess settings,
    /// near/far planes, position). The per-eye view/projection
    /// matrices are overridden by the headset pose each frame; only the
    /// post-processing settings and clip distances are read back.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn set_xr_camera(&mut self, camera: Camera) {
        self.xr_camera = Some(camera);
    }

    /// Set the PC mirror window's colour format (from the mirror surface's
    /// capabilities) so the XR mirror blit pipeline's color target matches.
    /// Usually `Bgra8UnormSrgb` on Windows.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn set_xr_mirror_format(&mut self, format: wgpu::TextureFormat) {
        if self.xr_mirror_format != Some(format) {
            self.xr_mirror_format = Some(format);
            self.xr_mirror = None;
        }
    }

    /// Whether the renderer was built with the OpenXR multiview path enabled.
    pub(crate) fn xr_enabled(&self) -> bool {
        self.enable_xr
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

    /// Returns the current IES texture array view, if any.
    pub fn ies_texture_view(&self) -> Option<&wgpu::TextureView> {
        self.ies_texture_view.as_ref()
    }

    /// Set the IES profile texture array view (`R8Unorm`, `D2Array`, one
    /// layer per profile; a light's `ies_profile_index` selects the layer).
    /// The application creates the texture; see `examples/ies_demo.rs`.
    pub fn set_ies_texture_view(&mut self, view: wgpu::TextureView) {
        self.ies_texture_view = Some(view);
    }
}

/// The camera as the scene sees it: what `camera_generation` tracks. Excludes
/// the per-frame jitter and frame counter, which change every frame by design.
#[derive(Clone, Copy, PartialEq)]
pub(crate) struct CameraIdentity {
    eye: glam::DVec3,
    rotation: glam::Mat4,
    proj: glam::Mat4,
    near: f32,
    far: f32,
    view_id: u32,
}

impl CameraIdentity {
    pub(crate) fn of(camera: &crate::Camera) -> Self {
        Self {
            eye: camera.eye,
            rotation: camera.rotation,
            proj: camera.proj,
            near: camera.near,
            far: camera.far,
            view_id: camera.view_id,
        }
    }
}

#[cfg(test)]
mod camera_identity_tests {
    use super::CameraIdentity;
    use glam::{DVec3, Mat4, Vec3};

    fn camera() -> crate::Camera {
        crate::Camera::perspective_look_at(
            Vec3::new(0.0, 2.0, 5.0),
            Vec3::ZERO,
            Vec3::Y,
            1.0,
            16.0 / 9.0,
            0.1,
            100.0,
        )
    }

    #[test]
    fn jitter_is_not_a_camera_change() {
        let still = camera();
        let mut jittered = still.clone();
        jittered.jitter = [0.3, -0.2];
        assert!(CameraIdentity::of(&still) == CameraIdentity::of(&jittered));
    }

    #[test]
    fn moving_or_reprojecting_is() {
        let still = camera();
        let mut moved = still.clone();
        moved.eye += DVec3::X;
        let mut turned = still.clone();
        turned.rotation = Mat4::from_rotation_y(0.1) * turned.rotation;
        let mut zoomed = still.clone();
        zoomed.proj = helio_core::depth::perspective(0.5, 16.0 / 9.0, 0.1);
        assert!(CameraIdentity::of(&still) != CameraIdentity::of(&moved));
        assert!(CameraIdentity::of(&still) != CameraIdentity::of(&turned));
        assert!(CameraIdentity::of(&still) != CameraIdentity::of(&zoomed));
    }
}
