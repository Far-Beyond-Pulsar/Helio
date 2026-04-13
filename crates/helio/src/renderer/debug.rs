use std::sync::{Arc, Mutex};

use helio_pass_debug::{DebugPass, DebugVertex};
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

pub struct DebugDrawState {
    pub editor_enabled: bool,
    pub camera_position: glam::Vec3,
    pub user_lines: Vec<DebugVertex>,
    pub user_lines_generation: u64,
    /// Filled triangles (TriangleList) from the current frame — cleared by `debug_clear`.
    pub user_tris: Vec<DebugVertex>,
    pub user_tris_generation: u64,
}

impl Default for DebugDrawState {
    fn default() -> Self {
        Self {
            editor_enabled: false,
            camera_position: glam::Vec3::ZERO,
            user_lines: Vec::new(),
            user_lines_generation: 0,
            user_tris: Vec::new(),
            user_tris_generation: 0,
        }
    }
}

pub struct DebugDrawPass {
    pass: DebugPass,
    state: Arc<Mutex<DebugDrawState>>,
    editor_mode: bool,
    cached_line_gen: u64,
    cached_tri_gen: u64,
    editor_grid_cache: Vec<DebugVertex>,
    editor_frame_lines: Vec<DebugVertex>,
    editor_last_key: Option<(bool, i32, i32, i32)>,
    editor_last_cam: Option<[f32; 3]>,
}

impl DebugDrawPass {
    pub fn new(
        device: &wgpu::Device,
        camera_buf: &wgpu::Buffer,
        surface_format: wgpu::TextureFormat,
        state: Arc<Mutex<DebugDrawState>>,
        depth_test: bool,
        editor_mode: bool,
    ) -> Self {
        Self {
            pass: DebugPass::new(device, camera_buf, surface_format, depth_test),
            state,
            editor_mode,
            cached_line_gen: u64::MAX,
            cached_tri_gen: u64::MAX,
            editor_grid_cache: Vec::new(),
            editor_frame_lines: Vec::new(),
            editor_last_key: None,
            editor_last_cam: None,
        }
    }

    /// Toggle depth-test at runtime. Propagates to the inner `DebugPass`
    /// without any pipeline rebuild (both pipelines are pre-compiled).
    pub fn set_depth_test(&mut self, enabled: bool) {
        self.pass.set_depth_test(enabled);
    }

    fn rebuild_editor_grid_cache(&mut self, center_x: f32, center_z: f32, grid_step: f32) {
        self.editor_grid_cache.clear();

        let minor_color: [f32; 4] = [0.25, 0.25, 0.25, 1.0];
        let major_color: [f32; 4] = [0.5, 0.5, 0.5, 1.0];
        let axis_color_x: [f32; 4] = [1.0, 0.2, 0.2, 1.0];
        let axis_color_z: [f32; 4] = [0.2, 1.0, 0.2, 1.0];

        let range: f32 = 40.0;
        let count = (range / grid_step).ceil() as i32;

        for i in -count..=count {
            let x = center_x + i as f32 * grid_step;
            let z = center_z + i as f32 * grid_step;
            let x_color = if i.rem_euclid(5) == 0 { major_color } else { minor_color };
            let z_color = if i.rem_euclid(5) == 0 { major_color } else { minor_color };

            self.editor_grid_cache.push(DebugVertex { position: [x, 0.0, center_z - range], _pad: 0.0, color: if x.abs() < 0.01 { axis_color_x } else { x_color } });
            self.editor_grid_cache.push(DebugVertex { position: [x, 0.0, center_z + range], _pad: 0.0, color: if x.abs() < 0.01 { axis_color_x } else { x_color } });
            self.editor_grid_cache.push(DebugVertex { position: [center_x - range, 0.0, z], _pad: 0.0, color: if z.abs() < 0.01 { axis_color_z } else { z_color } });
            self.editor_grid_cache.push(DebugVertex { position: [center_x + range, 0.0, z], _pad: 0.0, color: if z.abs() < 0.01 { axis_color_z } else { z_color } });
        }

        let origin_color = [1.0, 1.0, 0.0, 1.0];
        self.editor_grid_cache.push(DebugVertex { position: [-3.0, 0.0, 0.0], _pad: 0.0, color: origin_color });
        self.editor_grid_cache.push(DebugVertex { position: [3.0, 0.0, 0.0], _pad: 0.0, color: origin_color });
        self.editor_grid_cache.push(DebugVertex { position: [0.0, 0.0, -3.0], _pad: 0.0, color: origin_color });
        self.editor_grid_cache.push(DebugVertex { position: [0.0, 0.0, 3.0], _pad: 0.0, color: origin_color });
    }
}

impl RenderPass for DebugDrawPass {
    fn name(&self) -> &'static str {
        "DebugDraw"
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let state_arc = Arc::clone(&self.state);
        let state = state_arc.lock().unwrap();

        if self.editor_mode {
            let editor_enabled = state.editor_enabled;
            let cam = state.camera_position;
            drop(state);

            if !editor_enabled {
                if self.cached_line_gen != 0 {
                    self.pass.update_lines(ctx.queue, &[]);
                    self.cached_line_gen = 0;
                }
                self.editor_last_key = None;
                self.editor_last_cam = None;
                self.pass.update_tris(ctx.queue, &[]);
                self.cached_tri_gen = 0;
                return Ok(());
            }

            let cam_dist = cam.length();
            let (grid_step, step_index) = if cam_dist < 20.0_f32 {
                (1.0_f32, 0)
            } else if cam_dist < 60.0_f32 {
                (2.0_f32, 1)
            } else if cam_dist < 150.0_f32 {
                (5.0_f32, 2)
            } else if cam_dist < 300.0_f32 {
                (10.0_f32, 3)
            } else {
                (20.0_f32, 4)
            };

            let center_x = (cam.x / grid_step).round() * grid_step;
            let center_z = (cam.z / grid_step).round() * grid_step;
            let key = (
                true,
                step_index,
                (center_x * 1000.0) as i32,
                (center_z * 1000.0) as i32,
            );
            if self.editor_last_key != Some(key) {
                self.rebuild_editor_grid_cache(center_x, center_z, grid_step);
                self.editor_last_key = Some(key);
            }

            let cam_arr = [cam.x, cam.y, cam.z];
            if self.editor_last_cam != Some(cam_arr) || self.cached_line_gen == u64::MAX {
                self.editor_frame_lines.clear();
                self.editor_frame_lines
                    .extend_from_slice(&self.editor_grid_cache);

                let camera_marker_color = [0.0, 1.0, 1.0, 1.0];
                let mark = cam;
                self.editor_frame_lines.push(DebugVertex { position: [mark.x - 0.3, mark.y, mark.z], _pad: 0.0, color: camera_marker_color });
                self.editor_frame_lines.push(DebugVertex { position: [mark.x + 0.3, mark.y, mark.z], _pad: 0.0, color: camera_marker_color });
                self.editor_frame_lines.push(DebugVertex { position: [mark.x, mark.y - 0.3, mark.z], _pad: 0.0, color: camera_marker_color });
                self.editor_frame_lines.push(DebugVertex { position: [mark.x, mark.y + 0.3, mark.z], _pad: 0.0, color: camera_marker_color });
                self.editor_frame_lines.push(DebugVertex { position: [mark.x, mark.y, mark.z - 0.3], _pad: 0.0, color: camera_marker_color });
                self.editor_frame_lines.push(DebugVertex { position: [mark.x, mark.y, mark.z + 0.3], _pad: 0.0, color: camera_marker_color });

                self.pass.update_lines(ctx.queue, &self.editor_frame_lines);
                self.editor_last_cam = Some(cam_arr);
                self.cached_line_gen = self.cached_line_gen.wrapping_add(1);
            }

            if self.cached_tri_gen != 0 {
                self.pass.update_tris(ctx.queue, &[]);
                self.cached_tri_gen = 0;
            }
            return Ok(());
        }

        let user_lines_generation = state.user_lines_generation;
        let user_tris_generation = state.user_tris_generation;

        if user_lines_generation != self.cached_line_gen {
            self.pass.update_lines(ctx.queue, &state.user_lines);
            self.cached_line_gen = user_lines_generation;
        }
        if user_tris_generation != self.cached_tri_gen {
            self.pass.update_tris(ctx.queue, &state.user_tris);
            self.cached_tri_gen = user_tris_generation;
        }
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let target_view = if self.editor_mode {
            ctx.resources.pre_aa.unwrap_or(ctx.target)
        } else {
            ctx.target
        };

        let previous_target = ctx.target;
        ctx.target = target_view;

        let res = self.pass.execute(ctx);

        ctx.target = previous_target;

        res
    }

}
