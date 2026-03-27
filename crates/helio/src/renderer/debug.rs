use std::sync::{Arc, Mutex};

use helio_pass_debug::{DebugPass, DebugVertex, DebugCameraUniform};
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

pub struct DebugDrawState {
    pub editor_enabled: bool,
    pub camera_position: glam::Vec3,
    pub user_lines: Vec<DebugVertex>,
}

impl Default for DebugDrawState {
    fn default() -> Self {
        Self {
            editor_enabled: false,
            camera_position: glam::Vec3::ZERO,
            user_lines: Vec::new(),
        }
    }
}

pub struct DebugDrawPass {
    pass: DebugPass,
    state: Arc<Mutex<DebugDrawState>>,
    editor_mode: bool,
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
        }
    }

    fn build_frame_vertices(&self) -> Vec<DebugVertex> {
        let state = self.state.lock().unwrap();

        let mut output = if self.editor_mode {
            Vec::new()
        } else {
            state.user_lines.clone()
        };

        let cam = state.camera_position;
        let cam_dist = cam.length();

        if self.editor_mode && state.editor_enabled {
            let grid_step = if cam_dist < 20.0_f32 {
                1.0_f32
            } else if cam_dist < 60.0_f32 {
                2.0_f32
            } else if cam_dist < 150.0_f32 {
                5.0_f32
            } else if cam_dist < 300.0_f32 {
                10.0_f32
            } else {
                20.0_f32
            };

            let minor_color: [f32; 4] = [0.25, 0.25, 0.25, 1.0];
            let major_color: [f32; 4] = [0.5, 0.5, 0.5, 1.0];
            let axis_color_x: [f32; 4] = [1.0, 0.2, 0.2, 1.0];
            let axis_color_z: [f32; 4] = [0.2, 1.0, 0.2, 1.0];

            let range: f32 = 40.0;
            let count = (range / grid_step).ceil() as i32;
            let center_x = (cam.x / grid_step).round() * grid_step;
            let center_z = (cam.z / grid_step).round() * grid_step;

            for i in -count..=count {
                let x = center_x + i as f32 * grid_step;
                let z = center_z + i as f32 * grid_step;
                let x_color = if i.rem_euclid(5) == 0 { major_color } else { minor_color };
                let z_color = if i.rem_euclid(5) == 0 { major_color } else { minor_color };

                output.push(DebugVertex { position: [x, 0.0, center_z - range], _pad: 0.0, color: if x.abs() < 0.01 { axis_color_x } else { x_color } });
                output.push(DebugVertex { position: [x, 0.0, center_z + range], _pad: 0.0, color: if x.abs() < 0.01 { axis_color_x } else { x_color } });
                output.push(DebugVertex { position: [center_x - range, 0.0, z], _pad: 0.0, color: if z.abs() < 0.01 { axis_color_z } else { z_color } });
                output.push(DebugVertex { position: [center_x + range, 0.0, z], _pad: 0.0, color: if z.abs() < 0.01 { axis_color_z } else { z_color } });
            }

            let origin_color = [1.0, 1.0, 0.0, 1.0];
            output.push(DebugVertex { position: [-3.0, 0.0, 0.0], _pad: 0.0, color: origin_color });
            output.push(DebugVertex { position: [3.0, 0.0, 0.0], _pad: 0.0, color: origin_color });
            output.push(DebugVertex { position: [0.0, 0.0, -3.0], _pad: 0.0, color: origin_color });
            output.push(DebugVertex { position: [0.0, 0.0, 3.0], _pad: 0.0, color: origin_color });

            let camera_marker_color = [0.0, 1.0, 1.0, 1.0];
            let mark = cam;
            output.push(DebugVertex { position: [mark.x - 0.3, mark.y, mark.z], _pad: 0.0, color: camera_marker_color });
            output.push(DebugVertex { position: [mark.x + 0.3, mark.y, mark.z], _pad: 0.0, color: camera_marker_color });
            output.push(DebugVertex { position: [mark.x, mark.y - 0.3, mark.z], _pad: 0.0, color: camera_marker_color });
            output.push(DebugVertex { position: [mark.x, mark.y + 0.3, mark.z], _pad: 0.0, color: camera_marker_color });
            output.push(DebugVertex { position: [mark.x, mark.y, mark.z - 0.3], _pad: 0.0, color: camera_marker_color });
            output.push(DebugVertex { position: [mark.x, mark.y, mark.z + 0.3], _pad: 0.0, color: camera_marker_color });
        }

        output
    }
}

impl RenderPass for DebugDrawPass {
    fn name(&self) -> &'static str {
        "DebugDraw"
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let lines = self.build_frame_vertices();
        self.pass.update_lines(ctx.queue, &lines);
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let target_view = if self.editor_mode {
            ctx.frame.pre_aa.unwrap_or(ctx.target)
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
