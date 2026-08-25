#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use super::config::{PerfOverlayMode, RendererConfig};
use super::renderer_impl::Renderer;

impl Renderer {
    pub fn set_render_size(&mut self, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        if self.output_width == width && self.output_height == height {
            return; // no-op: preserve pass state registered before first frame
        }
        self.output_width = width;
        self.output_height = height;
        self.pending_resize = Some((width, height));
    }

    pub(crate) fn apply_resize_now(&mut self, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        let resize_start = Instant::now();

        self.scene.set_render_size(width, height);

        let internal_w = (((width as f32) * self.render_scale).ceil() as u32).max(1);
        let internal_h = (((height as f32) * self.render_scale).ceil() as u32).max(1);

        let depth_start = Instant::now();
        let (depth_texture, depth_view) =
            Self::create_depth_resources(&self.device, internal_w, internal_h);
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
        log::trace!(
            "apply_resize_now: internal depth {}x{} {}ms",
            internal_w,
            internal_h,
            depth_start.elapsed().as_secs_f64() * 1000.0
        );

        if self.render_scale < 1.0 {
            let (t, v) = Self::create_depth_resources(&self.device, width, height);
            self.full_res_depth_texture = Some(t);
            self.full_res_depth_view = Some(v);
        } else {
            self.full_res_depth_texture = None;
            self.full_res_depth_view = None;
        }

        // Recreate the multiview depth array when the internal resolution
        // changes (XR mode keeps it in sync with the window-driven resize so a
        // headset-less rebuild can't leave it stale).
        #[cfg(not(target_arch = "wasm32"))]
        if self.enable_xr {
            let (t, v, l0) = Self::create_xr_depth_resources(&self.device, internal_w, internal_h);
            self.xr_depth_texture = Some(t);
            self.xr_depth_view = Some(v);
            self.xr_depth_view_layer0 = Some(l0);
        }

        self.clear_target_next_frame = true;

        if let Some(rebuilder) = &self.graph_rebuilder {
            let config = RendererConfig {
                width,
                height,
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
                enable_portals: self.enable_portals,
                virtual_texturing_enabled: self.virtual_texturing_enabled,
                texture_stream_pool_mb: self.texture_stream_pool_mb,
                vt_tile_size_px: self.vt_tile_size_px,
            };
            self.graph = rebuilder(
                &self.device,
                &self.queue,
                &self.scene,
                config,
                self.debug_state.clone(),
                &self.debug_camera_buffer,
                &self.cull_stats_buffer,
            );
        } else {
            self.graph.set_render_size(internal_w, internal_h);
        }

        self.graph_has_sky = self.scene.sky_context().has_sky;
        self.scene.mark_water_volumes_dirty();

        log::trace!(
            "apply_resize_now: total resize {}ms",
            resize_start.elapsed().as_secs_f64() * 1000.0
        );
    }

    pub fn set_render_scale(&mut self, scale: f32) {
        self.render_scale = scale.clamp(0.25, 1.0);
        self.set_render_size(self.output_width, self.output_height);
    }

    pub fn render_scale(&self) -> f32 {
        self.render_scale
    }
}

impl Renderer {
    /// Rebuild the graph when the scene gains or loses its sky.
    ///
    /// `SkyLutPass` and `SkyPass` are added conditionally on
    /// `Scene::sky_context().has_sky` when the graph is *built*, but the natural
    /// construction order is to hand `Renderer::new` a graph and an empty scene and then
    /// populate the scene. A sky added after that point never gets its passes.
    ///
    /// The symptom is worse than a missing sky: `SkyPass` is what establishes `pre_aa`
    /// each frame, and every later colour pass loads that target rather than clearing it.
    /// With no sky pass nothing initialises it, so the image accumulates frame over frame
    /// and — in the dual-pass XR path, where both eyes share the graph's internal
    /// targets — eye over eye. It reads as geometry smearing over itself.
    ///
    /// Desktop hid this because the first window resize rebuilds the graph after the
    /// scene exists. The XR path renders into the runtime's swapchain and never resizes,
    /// so it kept the empty-scene graph forever.
    pub(crate) fn rebuild_graph_if_sky_changed(&mut self) {
        let has_sky = self.scene.sky_context().has_sky;
        if has_sky == self.graph_has_sky {
            return;
        }
        self.graph_has_sky = has_sky;

        let Some(rebuilder) = self.graph_rebuilder.clone() else {
            return;
        };
        log::info!("[graph] sky presence changed to {has_sky}; rebuilding render graph");
        let config = self.renderer_config();
        self.graph = rebuilder(
            &self.device,
            &self.queue,
            &self.scene,
            config,
            self.debug_state.clone(),
            &self.debug_camera_buffer,
            &self.cull_stats_buffer,
        );
    }
}

impl Renderer {
    /// Engine-world transform of the headset's stage origin.
    ///
    /// This is the locomotion hook: the headset reports poses relative to its stage
    /// origin, and this matrix places that origin in the world. Translating it walks the
    /// player forward; rotating it turns them. Scene content is untouched, so nothing has
    /// to move to make the player move.
    ///
    /// Keep it a rigid transform. Scale here would scale the interpupillary distance
    /// along with everything else, which is a reliable way to make people motion-sick.
    pub fn set_xr_stage_transform(&mut self, world_from_stage: glam::Mat4) {
        self.xr_stage_transform = world_from_stage;
    }

    /// The current stage transform.
    pub fn xr_stage_transform(&self) -> glam::Mat4 {
        self.xr_stage_transform
    }
}
