use web_time::Instant;

use super::config::RendererConfig;
use super::graph::{build_default_graph, build_default_graph_external, create_depth_resources};
use super::renderer_impl::{GraphKind, Renderer};

impl Renderer {
    pub fn set_render_size(&mut self, width: u32, height: u32) {
        self.output_width = width;
        self.output_height = height;
        self.pending_resize = Some((width, height));
    }

    pub(crate) fn apply_resize_now(&mut self, width: u32, height: u32) {
        let resize_start = Instant::now();

        let scene_start = Instant::now();
        self.scene.set_render_size(width, height);
        log::trace!(
            "apply_resize_now: scene.set_render_size {}ms",
            scene_start.elapsed().as_secs_f64() * 1000.0
        );

        let config = RendererConfig {
            width,
            height,
            surface_format: self.surface_format,
            gi_config: self.gi_config,
            shadow_quality: self.shadow_quality,
            debug_mode: self.debug_mode,
            render_scale: self.render_scale,
            perf_overlay_mode: self.perf_overlay_mode,
            shadow_atlas_size: self.shadow_atlas_size,
        };

        let depth_start = Instant::now();
        let (depth_texture, depth_view) = create_depth_resources(
            &self.device,
            config.internal_width(),
            config.internal_height(),
        );
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
        log::trace!(
            "apply_resize_now: internal depth {}x{} {}ms",
            config.internal_width(),
            config.internal_height(),
            depth_start.elapsed().as_secs_f64() * 1000.0
        );

        let full_depth_start = Instant::now();
        if self.render_scale < 1.0 {
            let (t, v) = create_depth_resources(&self.device, width, height);
            self.full_res_depth_texture = Some(t);
            self.full_res_depth_view = Some(v);
            log::trace!(
                "apply_resize_now: full-res depth {}x{} {}ms",
                width,
                height,
                full_depth_start.elapsed().as_secs_f64() * 1000.0
            );
        } else {
            self.full_res_depth_texture = None;
            self.full_res_depth_view = None;
        }

        self.clear_target_next_frame = true;

        let graph_start = Instant::now();
        match self.graph_kind {
            GraphKind::Default => {
                self.graph = if self.owns_device {
                    build_default_graph(
                        &self.device,
                        &self.queue,
                        &self.scene,
                        config,
                        self.debug_state.clone(),
                        &self.debug_camera_buffer,
                        &self.cull_stats_buffer,
                        Some(&self.debug_overlay_shared),
                    )
                } else {
                    build_default_graph_external(
                        &self.device,
                        &self.queue,
                        &self.scene,
                        config,
                        self.debug_state.clone(),
                        &self.debug_camera_buffer,
                        &self.cull_stats_buffer,
                        Some(&self.debug_overlay_shared),
                    )
                };
                log::trace!(
                    "apply_resize_now: graph rebuild {}ms",
                    graph_start.elapsed().as_secs_f64() * 1000.0
                );

                let water_start = Instant::now();
                self.scene.mark_water_volumes_dirty();
                log::trace!(
                    "apply_resize_now: mark_water_volumes_dirty {}ms",
                    water_start.elapsed().as_secs_f64() * 1000.0
                );
            }
            GraphKind::Simple => {
                self.graph.set_render_size(width, height);
                log::trace!(
                    "apply_resize_now: simple graph set_render_size {}ms",
                    graph_start.elapsed().as_secs_f64() * 1000.0
                );
            }
            GraphKind::Custom => {
                if let Some(builder) = &self.custom_graph_builder {
                    if let Some(prev_config) = self.custom_graph_config {
                        let new_cfg = RendererConfig {
                            width,
                            height,
                            ..prev_config
                        };
                        self.graph = builder(
                            &self.device,
                            &self.queue,
                            &self.scene,
                            new_cfg,
                            self.debug_state.clone(),
                            &self.debug_camera_buffer,
                            &self.cull_stats_buffer,
                            Some(&self.debug_overlay_shared),
                        );
                        self.custom_graph_config = Some(new_cfg);
                        log::trace!(
                            "apply_resize_now: custom graph rebuild {}ms",
                            graph_start.elapsed().as_secs_f64() * 1000.0
                        );

                        let water_start = Instant::now();
                        self.scene.mark_water_volumes_dirty();
                        log::trace!(
                            "apply_resize_now: mark_water_volumes_dirty {}ms",
                            water_start.elapsed().as_secs_f64() * 1000.0
                        );
                    } else {
                        self.graph.set_render_size(width, height);
                        log::trace!(
                            "apply_resize_now: custom graph set_render_size {}ms",
                            graph_start.elapsed().as_secs_f64() * 1000.0
                        );
                    }
                } else {
                    self.graph.set_render_size(width, height);
                }
            }
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
