//! WASM twin of `simple_graph` — fly camera around the hardcoded debug cube.
//!
//! The native version renders via `SimpleCubePass` with no scene objects.
//! This WASM twin approximates the same feel with a white unit cube and a
//! free-fly camera at the same default position.
//!
//! Controls: WASD/Space/Shift move, mouse look, left-click to grab cursor.

use std::sync::Arc;

use glam::{EulerRot, Quat, Vec3};
use helio::{Camera, Renderer};
use helio_wasm::{HelioWasmApp, InputState};

use crate::common::{cube_mesh, directional_light, insert_object, make_material, point_light};

const LOOK_SENS: f32 = 0.002;
const FLY_SPEED: f32 = 3.0;
const DRAG: f32 = 8.0;

pub struct Demo {
    cam_pos: Vec3,
    yaw: f32,
    pitch: f32,
    velocity: Vec3,
}

impl HelioWasmApp for Demo {
    fn title() -> &'static str {
        "Helio — Simple Graph"
    }

    fn init(
        renderer: &mut Renderer,
        _device: Arc<wgpu::Device>,
        _queue: Arc<wgpu::Queue>,
        _w: u32,
        _h: u32,
    ) -> Self {
        // Single white cube at origin
        let mat = renderer.scene_mut().insert_material(make_material(
            [0.95, 0.95, 0.95, 1.0],
            0.5,
            0.05,
            [0.0; 3],
            0.0,
        ));
        let mesh = renderer.scene_mut().insert_mesh(cube_mesh([0.0, 0.0, 0.0], 1.0));
        let _ = insert_object(renderer, mesh, mat, glam::Mat4::IDENTITY, 1.0);

        // Simple lighting
        renderer.scene_mut().insert_light(directional_light([0.4, -0.8, 0.5], [1.0, 1.0, 1.0], 1.2));
        renderer.scene_mut().insert_light(point_light([3.0, 2.0, 2.0], [0.5, 0.7, 1.0], 6.0, 12.0));
        renderer.set_ambient([0.4, 0.45, 0.5], 0.15);

        Self {
            cam_pos: Vec3::new(3.0, 2.0, 5.0),
            yaw: 0.0,
            pitch: -0.25,
            velocity: Vec3::ZERO,
        }
    }

    fn update(
        &mut self,
        _renderer: &mut Renderer,
        dt: f32,
        _elapsed: f32,
        input: &InputState,
    ) -> Camera {
        let (dx, dy) = input.mouse_delta;
        self.yaw -= dx * LOOK_SENS;
        self.pitch = (self.pitch - dy * LOOK_SENS).clamp(-1.5, 1.5);

        let orientation = Quat::from_euler(EulerRot::YXZ, self.yaw, self.pitch, 0.0);
        let forward = orientation * -Vec3::Z;
        let right = orientation * Vec3::X;

        let mut accel = Vec3::ZERO;
        if input.keys.contains(&helio_wasm::KeyCode::KeyW) {
            accel += forward;
        }
        if input.keys.contains(&helio_wasm::KeyCode::KeyS) {
            accel -= forward;
        }
        if input.keys.contains(&helio_wasm::KeyCode::KeyA) {
            accel -= right;
        }
        if input.keys.contains(&helio_wasm::KeyCode::KeyD) {
            accel += right;
        }
        if input.keys.contains(&helio_wasm::KeyCode::Space) {
            accel.y += 1.0;
        }
        if input.keys.contains(&helio_wasm::KeyCode::ShiftLeft) {
            accel.y -= 1.0;
        }
        if accel.length_squared() > 0.0 {
            accel = accel.normalize();
        }

        self.velocity += accel * FLY_SPEED * dt;
        self.velocity /= 1.0 + DRAG * dt;
        self.cam_pos += self.velocity * dt;

        // Soft boundary: pull back if too far from origin
        let dist = self.cam_pos.length();
        if dist > 20.0 {
            self.cam_pos *= 20.0 / dist;
            self.velocity = Vec3::ZERO;
        }

        let target = self.cam_pos + forward;
        let up = orientation * Vec3::Y;
        Camera::perspective_look_at(
            self.cam_pos,
            target,
            up,
            std::f32::consts::FRAC_PI_4,
            1280.0 / 720.0,
            0.01,
            100.0,
        )
    }
}

