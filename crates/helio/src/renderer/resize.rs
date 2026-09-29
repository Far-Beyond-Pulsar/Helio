#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

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

        if let Some(rebuilder) = self.graph_rebuilder.clone() {
            // Same recipe the current graph was built from, at the new size:
            // one stored config, not a field per pass-specific flag, so no
            // setting can be dropped by a rebuild (Helio#254/#255).
            let config = RendererConfig {
                width,
                height,
                ..self.renderer_config()
            };
            let graph = rebuilder(
                &self.device,
                &self.queue,
                config,
                self.debug_state.clone(),
                &self.camera_buffer,
                &self.debug_camera_buffer,
                &self.cull_stats_buffer,
            );
            self.replace_graph(graph);
            if let Some(hook) = &self.graph_rebuild_hook {
                hook(&mut self.graph, &self.device);
            }
        } else {
            self.graph.set_render_size(internal_w, internal_h);
        }

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
