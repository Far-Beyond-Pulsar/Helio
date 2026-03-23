//! WASM twin of `debug_shapes` — gallery of coloured solid props.

use std::sync::Arc;

use glam::Vec3;
use helio::{Camera, Renderer};
use helio_wasm::{HelioWasmApp, InputState};

use crate::common::{box_mesh, directional_light, insert_object, make_material, point_light};

const LOOK_SENS: f32 = 0.0024;
const FLY_SPEED: f32 = 5.0;

pub struct Demo {
    cam_pos:   Vec3,
    cam_yaw:   f32,
    cam_pitch: f32,
}

impl HelioWasmApp for Demo {
    fn title() -> &'static str { "Helio — Debug Shapes" }

    fn init(renderer: &mut Renderer, _device: Arc<wgpu::Device>,
            _queue: Arc<wgpu::Queue>, _w: u32, _h: u32) -> Self {
        // Materials
        let red     = renderer.insert_material(make_material([0.9, 0.2, 0.2, 1.0], 0.5, 0.0, [0.0;3], 0.0));
        let green   = renderer.insert_material(make_material([0.2, 0.8, 0.3, 1.0], 0.6, 0.0, [0.0;3], 0.0));
        let blue    = renderer.insert_material(make_material([0.2, 0.4, 0.9, 1.0], 0.4, 0.2, [0.0;3], 0.0));
        let yellow  = renderer.insert_material(make_material([0.9, 0.8, 0.1, 1.0], 0.7, 0.0, [0.0;3], 0.0));
        let cyan    = renderer.insert_material(make_material([0.1, 0.8, 0.9, 1.0], 0.3, 0.5, [0.0;3], 0.0));
        let magenta = renderer.insert_material(make_material([0.9, 0.1, 0.7, 1.0], 0.5, 0.1, [0.0;3], 0.0));
        let white   = renderer.insert_material(make_material([0.9, 0.9, 0.9, 1.0], 0.9, 0.0, [0.0;3], 0.0));
        let glow    = renderer.insert_material(make_material([0.1, 0.1, 0.1, 1.0], 0.9, 0.0, [1.0, 0.5, 0.1], 3.0));
        let metal   = renderer.insert_material(make_material([0.8, 0.8, 0.9, 1.0], 0.2, 1.0, [0.0;3], 0.0));

        // Floor
        let floor = renderer.insert_mesh(box_mesh([0.0, -0.1, 0.0], [10.0, 0.1, 10.0]));
        let _ = insert_object(renderer, floor, white, glam::Mat4::IDENTITY, 12.0);

        // Gallery of shapes arranged in a grid
        let shapes_data = [
            ([ -6.0, 0.5,  -6.0], [0.5, 0.5, 0.5], red),
            ([ -3.0, 0.75, -6.0], [0.4, 0.75, 0.4], green),
            ([  0.0, 1.0,  -6.0], [0.6, 1.0, 0.3], blue),
            ([  3.0, 0.5,  -6.0], [0.5, 0.5, 0.5], yellow),
            ([  6.0, 1.2,  -6.0], [0.3, 1.2, 0.3], cyan),
            ([ -6.0, 0.4,   0.0], [1.2, 0.4, 0.4], magenta),
            ([ -3.0, 0.6,   0.0], [0.6, 0.6, 0.6], metal),
            ([  0.0, 0.3,   0.0], [1.5, 0.3, 1.5], white),
            ([  3.0, 0.8,   0.0], [0.4, 0.8, 0.8], glow),
            ([  6.0, 0.5,   0.0], [0.5, 0.5, 0.5], red),
        ];
        for (pos, ext, mat) in shapes_data {
            let mesh = renderer.insert_mesh(box_mesh(pos, ext));
            let _ = insert_object(renderer, mesh, mat, glam::Mat4::IDENTITY, 2.0);
        }

        // Lights
        renderer.insert_light(directional_light([-0.3, -0.8, -0.5], [1.0, 0.95, 0.85], 1.5));
        renderer.insert_light(point_light([-6.0, 4.0, -6.0], [1.0, 0.2, 0.1], 8.0, 12.0));
        renderer.insert_light(point_light([ 6.0, 4.0,  0.0], [0.2, 0.5, 1.0], 8.0, 12.0));
        renderer.set_ambient([0.25, 0.28, 0.35], 0.1);

        Self {
            cam_pos: Vec3::new(0.0, 4.0, 14.0),
            cam_yaw: 0.0, cam_pitch: -0.25,
        }
    }

    fn update(&mut self, _renderer: &mut Renderer, dt: f32, _elapsed: f32,
              input: &InputState) -> Camera {
        self.cam_yaw   += input.mouse_delta.0 * LOOK_SENS;
        self.cam_pitch  = (self.cam_pitch - input.mouse_delta.1 * LOOK_SENS).clamp(-1.55, 1.55);

        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let fwd   = Vec3::new(sy * cp, sp, -cy * cp);
        let right = Vec3::new(cy, 0.0, sy);

        if input.keys.contains(&helio_wasm::KeyCode::KeyW) { self.cam_pos += fwd   * FLY_SPEED * dt; }
        if input.keys.contains(&helio_wasm::KeyCode::KeyS) { self.cam_pos -= fwd   * FLY_SPEED * dt; }
        if input.keys.contains(&helio_wasm::KeyCode::KeyA) { self.cam_pos -= right * FLY_SPEED * dt; }
        if input.keys.contains(&helio_wasm::KeyCode::KeyD) { self.cam_pos += right * FLY_SPEED * dt; }
        if input.keys.contains(&helio_wasm::KeyCode::Space)     { self.cam_pos.y += FLY_SPEED * dt; }
        if input.keys.contains(&helio_wasm::KeyCode::ShiftLeft) { self.cam_pos.y -= FLY_SPEED * dt; }

        Camera::perspective_look_at(
            self.cam_pos, self.cam_pos + fwd, Vec3::Y,
            std::f32::consts::FRAC_PI_4, 1280.0 / 720.0, 0.1, 200.0,
        )
    }
}
