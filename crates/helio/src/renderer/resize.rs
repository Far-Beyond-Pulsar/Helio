#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use helio_core::RenderGraph;

use super::config::RendererConfig;
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

        // Same recipe the current graph was built from, at the new size:
        // one stored config, not a field per pass-specific flag, so no
        // setting can be dropped by a rebuild (Helio#254/#255).
        let config = RendererConfig {
            width,
            height,
            ..self.renderer_config()
        };
        if let Some(replacement) = self.build_replacement_graph(config) {
            self.install_replacement_graph(replacement);
        } else {
            self.graph.set_render_size(internal_w, internal_h);
        }

        log::trace!(
            "apply_resize_now: total resize {}ms",
            resize_start.elapsed().as_secs_f64() * 1000.0
        );
    }

    /// Builds a fresh graph from `config` with the stored rebuilder, or `None`
    /// if the renderer has none. Shared by resize and shader hot reload; it
    /// does not touch the current graph.
    pub(crate) fn build_replacement_graph(&self, config: RendererConfig) -> Option<RenderGraph> {
        let rebuilder = self.graph_rebuilder.clone()?;
        Some(rebuilder(
            &self.device,
            &self.queue,
            config,
            self.debug_state.clone(),
            &self.camera_buffer,
            &self.debug_camera_buffer,
            &self.cull_stats_buffer,
        ))
    }

    /// Swaps `replacement` in for the current graph, carrying over what a
    /// rebuild must not lose and reapplying application-owned settings.
    pub(crate) fn install_replacement_graph(&mut self, mut replacement: RenderGraph) {
        // Passes carry persistent state (voxel residency, histories)
        // into the rebuilt graph; replace_graph restores GI and bakes.
        replacement.inherit_persistent_state(&mut self.graph);
        self.replace_graph(replacement);
        if let Some(hook) = &self.graph_rebuild_hook {
            hook(&mut self.graph, &self.device);
        }
        if let Some(sky) = self.graph.find_pass_mut::<helio_pass_sky::SkyPass>() {
            sky.set_fallback_sky_enabled(self.fallback_sky_enabled);
            sky.set_planetary_sky(self.planetary_sky);
        }
        if let Some(light) = self.graph.find_pass_mut::<helio_pass_deferred_light::DeferredLightPass>() {
            light.set_planetary_atmosphere(
                self.planetary_sky.map(|s| (s.eye_m, s.radius_m, s.sun_direction)),
            );
        }
    }

    pub fn set_render_scale(&mut self, scale: f32) {
        self.render_scale = scale.clamp(0.25, 1.0);
        self.set_render_size(self.output_width, self.output_height);
    }

    /// Change temporal resolve when scene content becomes available after the
    /// renderer was constructed. Rebuild even if the window size is unchanged:
    /// the pass set and internal attachment sizes both depend on this choice.
    pub fn set_tsr_quality(&mut self, quality: Option<helio_pass_tsr::TsrQuality>) {
        let scale = quality.map_or(0.75, helio_pass_tsr::TsrQuality::render_scale);
        if self.tsr_quality == quality && (self.render_scale - scale).abs() < f32::EPSILON {
            return;
        }
        self.tsr_quality = quality;
        self.render_scale = scale;
        self.pending_resize = Some((self.output_width, self.output_height));
    }

    pub fn render_scale(&self) -> f32 {
        self.render_scale
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
