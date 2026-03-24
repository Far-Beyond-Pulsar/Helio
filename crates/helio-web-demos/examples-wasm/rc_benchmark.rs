//! WASM twin of `rc_benchmark` — Cornell box GI benchmark with 3 point lights.
//!
//! Controls: WASD fly, mouse look. Plus(=) / Minus to adjust light intensity.

use std::sync::Arc;

use glam::Vec3;
use helio::{Camera, LightId, Renderer};
use helio_wasm::{HelioWasmApp, InputState};

use crate::common::{box_mesh, insert_object, make_material, point_light};

const LOOK_SENS: f32 = 0.0020;
const SPEED: f32 = 4.0;

const LIGHT_BASE: &[([f32; 3], [f32; 3], f32, f32)] = &[
    ([0.0, 4.8, 0.0], [1.0, 1.0, 1.0], 18.0, 6.0),
    ([-3.5, 3.0, -2.0], [1.0, 0.7, 0.4], 12.0, 5.0),
    ([3.5, 3.0, 2.0], [0.4, 0.7, 1.0], 12.0, 5.0),
];

pub struct Demo {
    light_ids: [LightId; 3],
    intensity: f32,
    cam_pos: Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
}

impl HelioWasmApp for Demo {
    fn title() -> &'static str {
        "Helio — RC Benchmark"
    }

    fn init(
        renderer: &mut Renderer,
        _device: Arc<wgpu::Device>,
        _queue: Arc<wgpu::Queue>,
        _w: u32,
        _h: u32,
    ) -> Self {
        let mat_white =
            renderer.insert_material(make_material([0.9, 0.9, 0.9, 1.0], 0.9, 0.0, [0.0; 3], 0.0));
        let mat_red =
            renderer.insert_material(make_material([0.8, 0.1, 0.1, 1.0], 0.9, 0.0, [0.0; 3], 0.0));
        let mat_green =
            renderer.insert_material(make_material([0.1, 0.7, 0.1, 1.0], 0.9, 0.0, [0.0; 3], 0.0));
        let mat_cube = renderer.insert_material(make_material(
            [0.8, 0.78, 0.72, 1.0],
            0.85,
            0.0,
            [0.0; 3],
            0.0,
        ));

        let mut add_box = |cx: f32, cy: f32, cz: f32, hx: f32, hy: f32, hz: f32, mat| {
            let m = renderer.insert_mesh(box_mesh([cx, cy, cz], [hx, hy, hz]));
            let _ = insert_object(
                renderer,
                m,
                mat,
                glam::Mat4::IDENTITY,
                (hx * hx + hy * hy + hz * hz).sqrt(),
            );
        };
        add_box(0.0, -0.05, 0.0, 5.0, 0.05, 5.0, mat_white); // floor
        add_box(0.0, 5.05, 0.0, 5.0, 0.05, 5.0, mat_white); // ceiling
        add_box(0.0, 2.5, -5.05, 5.0, 2.5, 0.05, mat_white); // back wall
        add_box(0.0, 2.5, 5.05, 5.0, 2.5, 0.05, mat_white); // front wall
        add_box(5.05, 2.5, 0.0, 0.05, 2.5, 5.0, mat_green); // right (green)
        add_box(-5.05, 2.5, 0.0, 0.05, 2.5, 5.0, mat_red); // left (red)
        add_box(-2.0, 0.5, -2.0, 0.5, 0.5, 0.5, mat_cube);
        add_box(2.0, 0.5, 2.0, 0.5, 0.5, 0.5, mat_cube);
        add_box(0.0, 0.7, 0.0, 0.7, 0.7, 0.7, mat_cube);
        add_box(-3.0, 1.0, 1.5, 1.0, 1.0, 1.0, mat_cube);
        add_box(3.0, 0.6, -1.5, 0.6, 0.6, 0.6, mat_cube);

        let mut ids_vec: Vec<LightId> = LIGHT_BASE
            .iter()
            .map(|&(pos, col, int, rng)| renderer.insert_light(point_light(pos, col, int, rng)))
            .collect();
        let light_ids = [ids_vec.remove(0), ids_vec.remove(0), ids_vec.remove(0)];

        renderer.set_ambient([0.02, 0.02, 0.03], 1.0);

        Self {
            light_ids,
            intensity: 1.0,
            cam_pos: Vec3::new(0.0, 2.5, 8.0),
            cam_yaw: 0.0,
            cam_pitch: 0.0,
        }
    }

    fn update(
        &mut self,
        renderer: &mut Renderer,
        dt: f32,
        _elapsed: f32,
        input: &InputState,
    ) -> Camera {
        self.cam_yaw += input.mouse_delta.0 * LOOK_SENS;
        self.cam_pitch = (self.cam_pitch - input.mouse_delta.1 * LOOK_SENS).clamp(-1.5, 1.5);

        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let fwd = Vec3::new(sy * cp, sp, -cy * cp);
        let right = Vec3::new(cy, 0.0, sy);

        if input.keys.contains(&helio_wasm::KeyCode::KeyW) {
            self.cam_pos += fwd * SPEED * dt;
        }
        if input.keys.contains(&helio_wasm::KeyCode::KeyS) {
            self.cam_pos -= fwd * SPEED * dt;
        }
        if input.keys.contains(&helio_wasm::KeyCode::KeyA) {
            self.cam_pos -= right * SPEED * dt;
        }
        if input.keys.contains(&helio_wasm::KeyCode::KeyD) {
            self.cam_pos += right * SPEED * dt;
        }
        if input.keys.contains(&helio_wasm::KeyCode::Space) {
            self.cam_pos.y += SPEED * dt;
        }
        if input.keys.contains(&helio_wasm::KeyCode::ShiftLeft) {
            self.cam_pos.y -= SPEED * dt;
        }
        if input.keys.contains(&helio_wasm::KeyCode::Equal) {
            self.intensity = (self.intensity + 0.5 * dt).min(5.0);
        }
        if input.keys.contains(&helio_wasm::KeyCode::Minus) {
            self.intensity = (self.intensity - 0.5 * dt).max(0.1);
        }

        for (id, &(pos, col, int, rng)) in self.light_ids.iter().zip(LIGHT_BASE.iter()) {
            let _ = renderer.update_light(*id, point_light(pos, col, int * self.intensity, rng));
        }

        Camera::perspective_look_at(
            self.cam_pos,
            self.cam_pos + fwd,
            Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            1280.0 / 720.0,
            0.1,
            50.0,
        )
    }
}

