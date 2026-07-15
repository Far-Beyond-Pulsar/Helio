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
