//! WASM twin of `render_v2_basic` — 3 cubes + ground, three orbiting point lights.

use std::sync::Arc;

use glam::Vec3;
use helio::{Camera, LightId, MeshId, Renderer};
use helio_wasm::{HelioWasmApp, InputState};

use crate::common::{box_mesh, cube_mesh, insert_object, make_material, plane_mesh, point_light};

const LOOK_SENS: f32 = 0.0024;
const FLY_SPEED: f32 = 5.0;

pub struct Demo {
    cube1: MeshId,
    cube2: MeshId,
    cube3: MeshId,
    _ground: MeshId,
    light_p0: LightId,

    cam_pos: Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
}

impl HelioWasmApp for Demo {
    fn title() -> &'static str {
        "Helio — Basic Render"
    }

    fn init(
        renderer: &mut Renderer,
        _device: Arc<wgpu::Device>,
        _queue: Arc<wgpu::Queue>,
        _w: u32,
        _h: u32,
    ) -> Self {
        let mat = renderer.scene_mut().insert_material(make_material(
            [0.7, 0.7, 0.72, 1.0],
            0.7,
            0.0,
            [0.0, 0.0, 0.0],
            0.0,
        ));

        let cube1 = renderer.scene_mut().insert_mesh(cube_mesh([0.0, 0.5, 0.0], 0.5));
        let cube2 = renderer.scene_mut().insert_mesh(cube_mesh([-2.0, 0.4, -1.0], 0.4));
        let cube3 = renderer.scene_mut().insert_mesh(cube_mesh([2.0, 0.3, 0.5], 0.3));
        let ground = renderer.scene_mut().insert_mesh(plane_mesh([0.0, 0.0, 0.0], 5.0));

        let _ = insert_object(renderer, cube1, mat, glam::Mat4::IDENTITY, 0.5);
        let _ = insert_object(renderer, cube2, mat, glam::Mat4::IDENTITY, 0.4);
        let _ = insert_object(renderer, cube3, mat, glam::Mat4::IDENTITY, 0.3);
        let _ = insert_object(renderer, ground, mat, glam::Mat4::IDENTITY, 5.0);

        let light_p0 =
            renderer.scene_mut().insert_light(point_light([0.0, 2.2, 0.0], [1.0, 0.55, 0.15], 6.0, 5.0));
        renderer.scene_mut().insert_light(point_light([-3.5, 2.0, -1.5], [0.25, 0.5, 1.0], 5.0, 6.0));
        renderer.scene_mut().insert_light(point_light([3.5, 1.5, 1.5], [1.0, 0.3, 0.5], 5.0, 6.0));

        Self {
            cube1,
            cube2,
            cube3,
            _ground: ground,
            light_p0,
            cam_pos: Vec3::new(0.0, 2.5, 7.0),
            cam_yaw: 0.0,
            cam_pitch: -0.2,
        }
    }

    fn update(
        &mut self,
        renderer: &mut Renderer,
        dt: f32,
        elapsed: f32,
        input: &InputState,
    ) -> Camera {
        // Mouse look
        self.cam_yaw += input.mouse_delta.0 * LOOK_SENS;
        self.cam_pitch = (self.cam_pitch - input.mouse_delta.1 * LOOK_SENS).clamp(-1.55, 1.55);

        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let fwd = Vec3::new(sy * cp, sp, -cy * cp);
        let right = Vec3::new(cy, 0.0, sy);

        if input.keys.contains(&helio_wasm::KeyCode::KeyW) {
            self.cam_pos += fwd * FLY_SPEED * dt;
        }
        if input.keys.contains(&helio_wasm::KeyCode::KeyS) {
            self.cam_pos -= fwd * FLY_SPEED * dt;
        }
        if input.keys.contains(&helio_wasm::KeyCode::KeyA) {
            self.cam_pos -= right * FLY_SPEED * dt;
        }
        if input.keys.contains(&helio_wasm::KeyCode::KeyD) {
            self.cam_pos += right * FLY_SPEED * dt;
        }
        if input.keys.contains(&helio_wasm::KeyCode::Space) {
            self.cam_pos.y += FLY_SPEED * dt;
        }
        if input.keys.contains(&helio_wasm::KeyCode::ShiftLeft) {
            self.cam_pos.y -= FLY_SPEED * dt;
        }

        // Animate light p0
        let p0 = [0.0_f32, 2.2 + (elapsed * 0.7).sin() * 0.3, 0.0];
        let _ = renderer.scene_mut().update_light(self.light_p0, point_light(p0, [1.0, 0.55, 0.15], 6.0, 5.0));

        Camera::perspective_look_at(
            self.cam_pos,
            self.cam_pos + fwd,
            Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            1280.0 / 720.0,
            0.1,
            200.0,
        )
    }
}

