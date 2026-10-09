use std::sync::{Arc, Mutex};

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use helio_core::{RenderFrameInputs, RenderGraph, RenderPass};

use super::builder::SceneDbHandle;
use super::config::{RenderMode, RendererConfig};

/// Closure that rebuilds the render graph on resize.
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

/// Reapplies application-owned pass settings after a resize rebuilds the graph.
pub type GraphRebuildHook = Arc<dyn Fn(&mut RenderGraph, &wgpu::Device) + Send + Sync>;

use crate::camera::Camera;
use helio_mats::radiant::{RadiantTemplateRegistry, SharedTemplateRegistry};

use super::config::GiConfig;
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
    pub(crate) prev_view_proj: glam::Mat4,
    /// World origin used to express the previous local camera projection.
    pub(crate) previous_world_origin: Option<glam::DVec3>,
    /// GPU work deriving drawable scene buffers from the frontend's authored
    /// ones, run each frame between the SceneDB snapshot and the graph. See
    /// `helio_core::scene_derivation`.
    pub(crate) scene_derivations: Vec<Box<dyn helio_core::SceneDerivation>>,
    pub(crate) world_origin: Option<glam::DVec3>,
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
    /// rebuild (resize, a config change) hands the graph builder, so the new
    /// graph has the same passes and pass settings (Helio#254/#255).
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
    /// Preserved across graph rebuilds (resize). Shared (not cloned) with
    /// passes and the GPU scene via `Arc<RwLock<_>>` — see
    /// `SharedTemplateRegistry` for why cloning this must be avoided.
    pub(crate) template_registry: SharedTemplateRegistry,

    /// Templates registered by the user for the transparent path.
    /// Used by TransparentPass for alpha-blended materials (water, glass, etc.).
    pub(crate) transparent_template_registry: SharedTemplateRegistry,
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
    /// `view_id`, near/far and the representative position used
    /// for RC bounds and the debug state. The per-eye view/proj are overridden
    /// by the headset each frame.
    pub(crate) xr_camera: Option<Camera>,
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

impl Renderer {
    /// Publish an already-built frontend acceleration projection for the next
    /// frame only. Call after flushing SceneDB and preparing its BLAS/TLAS; an
    /// omitted update expires instead of reusing potentially stale geometry.
    pub fn set_ray_tracing_frame(&mut self, tlas: Option<&wgpu::Tlas>) {
        self.ray_frame.publish(self.frame_count, tlas);
    }

    /// Publish a TLAS and its matching per-instance transmission buffer for one
    /// frame. Material rows must follow the exact TLAS instance ordering.
    pub fn set_ray_tracing_frame_with_transmission(
        &mut self,
        tlas: Option<&wgpu::Tlas>,
        transmission: Option<&wgpu::Buffer>,
    ) {
        self.ray_frame
            .publish_with_transmission(self.frame_count, tlas, transmission);
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
        let previous_projection = match (self.previous_world_origin, self.world_origin) {
            (Some(previous), Some(current)) => helio_core::temporal::rebase_previous_projection(
                self.prev_view_proj,
                current - previous,
            ),
            (None, None) => self.prev_view_proj,
            // Switching coordinate spaces invalidates the old projection.
            _ => camera.proj * camera.view,
        };
        let uniforms = helio_core::GpuCameraUniforms::new(
            camera.view,
            camera.proj,
            camera.position,
            camera.near,
            camera.far,
            self.frame_count as u32,
            camera.jitter,
            previous_projection,
        ).with_view_id(camera.view_id);
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniforms));
        self.prev_view_proj = glam::Mat4::from_cols_array(&uniforms.view_proj);
        self.previous_world_origin = self.world_origin;
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

    /// Set the double-precision world origin for camera-relative frames.
    /// Geometry submitted to this frame must use the same local coordinates.
    pub fn set_world_origin(&mut self, origin: Option<glam::DVec3>) {
        self.world_origin = origin;
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn upload_stereo_camera(
        &mut self,
        left: &helio_core::GpuCameraUniforms,
        right: &helio_core::GpuCameraUniforms,
    ) {
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[*left, *right]),
        );
        self.camera_data = *left;
        self.prev_view_proj = glam::Mat4::from_cols_array(&left.view_proj);
        // A tracked headset moves every frame; treat each stereo upload as a
        // new view, and make the next mono frame compare afresh.
        self.camera_identity = None;
        self.camera_generation = self.camera_generation.wrapping_add(1);
    }

    pub fn set_gi_config(&mut self, gi_config: GiConfig) {
        self.graph_config.gi_config = gi_config;
        if let Some(pass) = self
            .graph
            .find_pass_mut::<helio_pass_radiance_cascades::RadianceCascadesPass>()
        {
            pass.set_gi_config(gi_config);
        }
    }

    pub fn gi_config(&self) -> GiConfig {
        self.graph_config.gi_config
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

    pub fn set_editor_mode(&mut self, enabled: bool) {
        self.editor_mode = enabled;
        if let Ok(mut s) = self.debug_state.lock() {
            s.editor_enabled = enabled;
        }
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

    /// Forward portal/sublevel coordinate spaces through the generic graph
    /// input seam. The owning pass uploads its own GPU table during prepare.
    pub fn set_coordinate_spaces(&mut self, spaces: &[glam::Mat4]) {
        self.coordinate_spaces.clear();
        self.coordinate_spaces.extend_from_slice(spaces);
        self.apply_coordinate_spaces();
    }

    /// Upload a resolver-generated portal projection frame while preserving
    /// the renderer's existing coordinate-space seam. The frame owns slot
    /// zero as identity and this facade accepts the non-identity tail because
    /// The graph broadcasts the frame to whichever pass families consume it;
    /// the renderer does not need to know their concrete types.
    pub fn set_portal_projection_frame(
        &mut self,
        frame: &helio_pass_portal_cull::PortalProjectionFrame,
    ) {
        self.coordinate_spaces.clear();
        self.coordinate_spaces
            .extend_from_slice(frame.renderer_coordinate_spaces());
        self.portal_projection_counts = Some((
            frame.counts.portal_view_count,
            frame.counts.portal_chain_count,
        ));
        self.apply_coordinate_spaces();
    }

    pub(crate) fn apply_coordinate_spaces(&mut self) {
        let inputs = RenderFrameInputs {
            coordinate_spaces: &self.coordinate_spaces,
            projection_counts: self
                .portal_projection_counts
                .map(|(view_count, chain_count)| [view_count, chain_count]),
        };
        self.graph.set_frame_inputs(&inputs);
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
    /// built graph cannot know about (a finished bake).
    pub(crate) fn replace_graph(&mut self, mut graph: RenderGraph) {
        if let Some(pass) = graph
            .find_pass_mut::<helio_pass_radiance_cascades::RadianceCascadesPass>()
        {
            pass.set_gi_config(self.graph_config.gi_config);
        }
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
    pub fn set_xr_camera(&mut self, camera: Camera) {
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
    view: glam::Mat4,
    proj: glam::Mat4,
    position: glam::Vec3,
    near: f32,
    far: f32,
    view_id: u32,
}

impl CameraIdentity {
    pub(crate) fn of(camera: &crate::Camera) -> Self {
        Self {
            view: camera.view,
            proj: camera.proj,
            position: camera.position,
            near: camera.near,
            far: camera.far,
            view_id: camera.view_id,
        }
    }
}

#[cfg(test)]
mod camera_identity_tests {
    use super::CameraIdentity;
    use glam::{Mat4, Vec3};

    fn camera() -> crate::Camera {
        crate::Camera::perspective_look_at(Vec3::new(0.0, 2.0, 5.0), Vec3::ZERO, Vec3::Y, 1.0, 16.0 / 9.0, 0.1, 100.0)
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
        moved.view = Mat4::from_translation(Vec3::X) * moved.view;
        let mut zoomed = still.clone();
        zoomed.proj = Mat4::perspective_rh(0.5, 16.0 / 9.0, 0.1, 100.0);
        assert!(CameraIdentity::of(&still) != CameraIdentity::of(&moved));
        assert!(CameraIdentity::of(&still) != CameraIdentity::of(&zoomed));
    }
}
