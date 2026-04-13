//! WASM twin of `space_station` — orbiting space station.
//!
//! Controls: WASD fly (6DoF — no gravity), mouse look.

use std::sync::Arc;

use glam::Vec3;
use helio::{Camera, LightId, Renderer};
use helio_wasm::{HelioWasmApp, InputState};

use crate::common::{
    box_mesh, cube_mesh, directional_light, insert_object, make_material, point_light,
};

const LOOK_SENS: f32 = 0.0024;
const FLY_SPEED: f32 = 6.0;

pub struct Demo {
    port_light: LightId,
    star_light: LightId,
    cam_pos: Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
}

impl HelioWasmApp for Demo {
    fn title() -> &'static str {
        "Helio — Space Station"
    }

    fn init(
        renderer: &mut Renderer,
        _device: Arc<wgpu::Device>,
        _queue: Arc<wgpu::Queue>,
        _w: u32,
        _h: u32,
    ) -> Self {
        let hull_m = renderer.scene_mut().insert_material(make_material(
            [0.75, 0.75, 0.78, 1.0],
            0.4,
            0.6,
            [0.0; 3],
            0.0,
        ));
        let panel_m = renderer.scene_mut().insert_material(make_material(
            [0.18, 0.22, 0.35, 1.0],
            0.5,
            0.3,
            [0.02, 0.04, 0.1],
            0.3,
        ));
        let solar_m = renderer.scene_mut().insert_material(make_material(
            [0.08, 0.12, 0.18, 1.0],
            0.3,
            0.1,
            [0.0, 0.02, 0.05],
            0.5,
        ));
        let window_m = renderer.scene_mut().insert_material(make_material(
            [0.5, 0.55, 0.7, 1.0],
            0.05,
            0.95,
            [0.1, 0.15, 0.3],
            0.8,
        ));
        let red_m = renderer.scene_mut().insert_material(make_material(
            [0.15, 0.0, 0.0, 1.0],
            1.0,
            0.0,
            [1.0, 0.05, 0.05],
            5.0,
        ));
        let green_m = renderer.scene_mut().insert_material(make_material(
            [0.0, 0.15, 0.0, 1.0],
            1.0,
            0.0,
            [0.05, 1.0, 0.1],
            5.0,
        ));
        let truss_m = renderer.scene_mut().insert_material(make_material(
            [0.6, 0.6, 0.62, 1.0],
            0.5,
            0.5,
            [0.0; 3],
            0.0,
        ));
        let thruster_m = renderer.scene_mut().insert_material(make_material(
            [0.3, 0.3, 0.35, 1.0],
            0.6,
            0.4,
            [0.1, 0.15, 0.2],
            0.5,
        ));

        // Central hub
        let hub = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [2.5, 2.5, 4.0])));
        insert_object(renderer, hub, hull_m, glam::Mat4::IDENTITY, 4.0).unwrap();

        // Main truss spine
        let spine = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [0.4, 0.4, 28.0])));
        insert_object(renderer, spine, truss_m, glam::Mat4::IDENTITY, 28.0).unwrap();

        // Habitation rings (4 around the hub)
        for (i, angle) in (0..4).map(|i| (i, i as f32 * std::f32::consts::FRAC_PI_2)) {
            let x = angle.cos() * 5.0;
            let y = angle.sin() * 5.0;
            let ring = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([x, y, 0.0], [1.5, 1.5, 3.0])));
            insert_object(renderer, ring, hull_m, glam::Mat4::IDENTITY, 3.0).unwrap();
            // Spoke
            let sx = x * 0.5;
            let sy = y * 0.5;
            let spoke = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([sx, sy, 0.0], [0.18, 0.18, 3.2])));
            insert_object(renderer, spoke, truss_m, glam::Mat4::IDENTITY, 3.2).unwrap();
            // Ring windows
            let wm = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([x, y, 1.0], [1.4, 1.4, 0.1])));
            insert_object(renderer, wm, window_m, glam::Mat4::IDENTITY, 1.4).unwrap();
            let wm2 = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([x, y, -1.0], [1.4, 1.4, 0.1])));
            insert_object(renderer, wm2, window_m, glam::Mat4::IDENTITY, 1.4).unwrap();
            let _ = i;
        }

        // Solar array wings
        for sx in [-1.0_f32, 1.0] {
            for (sz, z_off) in [(1.0_f32, 10.0_f32), (-1.0, -10.0)] {
                // Boom
                let boom = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh(
                    [0.0, 0.0, z_off * 0.5],
                    [0.12, 0.12, z_off.abs() - 2.0],
                ));
                insert_object(renderer, boom, truss_m, glam::Mat4::IDENTITY, z_off.abs()).unwrap();
                // Two solar panel rows per wing
                for py in [-0.9_f32, 0.9] {
                    let panel =
                        renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([sx * 8.0, py, z_off], [6.0, 0.04, 1.8])));
                    insert_object(renderer, panel, solar_m, glam::Mat4::IDENTITY, 6.0).unwrap();
                    let panel2 =
                        renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([sx * 16.0, py, z_off], [6.0, 0.04, 1.8])));
                    insert_object(renderer, panel2, solar_m, glam::Mat4::IDENTITY, 6.0).unwrap();
                    // Panel grid lines
                    for k in -2..=2 {
                        let grid = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh(
                            [sx * 8.0, py, z_off + k as f32 * 0.35],
                            [6.0, 0.015, 0.01],
                        ));
                        insert_object(renderer, grid, panel_m, glam::Mat4::IDENTITY, 6.0).unwrap();
                    }
                }
                let _ = sz;
            }
        }

        // Docking port (forward)
        let dock = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 5.5], [1.0, 1.0, 1.5])));
        insert_object(renderer, dock, hull_m, glam::Mat4::IDENTITY, 1.5).unwrap();
        let dock_ring = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 6.5], [1.3, 1.3, 0.15])));
        insert_object(renderer, dock_ring, truss_m, glam::Mat4::IDENTITY, 1.3).unwrap();

        // Engine cluster (aft)
        for ex in [-0.7_f32, 0.7] {
            for ey in [-0.7_f32, 0.7] {
                let eng = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([ex, ey, -5.5], [0.3, 0.3, 0.8])));
                insert_object(renderer, eng, thruster_m, glam::Mat4::IDENTITY, 0.8).unwrap();
            }
        }

        // Nav lights (port=red, starboard=green)
        let port_cube = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(cube_mesh([sx_for_nav(-1.0), 0.0, 0.0], 0.08)));
        insert_object(renderer, port_cube, red_m, glam::Mat4::IDENTITY, 0.08).unwrap();
        let star_cube = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(cube_mesh([sx_for_nav(1.0), 0.0, 0.0], 0.08)));
        insert_object(renderer, star_cube, green_m, glam::Mat4::IDENTITY, 0.08).unwrap();

        let port_light =
            renderer.scene_mut().insert_actor(helio::SceneActor::light(point_light([-22.5, 0.0, 0.0], [1.0, 0.05, 0.05], 12.0, 8.0)));
        let star_light =
            renderer.scene_mut().insert_actor(helio::SceneActor::light(point_light([22.5, 0.0, 0.0], [0.05, 1.0, 0.1], 12.0, 8.0)));

        // Sunlight (directional, no atmosphere)
        let sun_dir = Vec3::new(-0.5, -0.4, 0.8).normalize();
        renderer.scene_mut().insert_actor(helio::SceneActor::light(directional_light(
            [sun_dir.x, sun_dir.y, sun_dir.z],
            [1.0, 0.98, 0.95],
            1.0,
        ));
        renderer.set_ambient([0.05, 0.06, 0.1], 0.005);
        renderer.set_clear_color([0.0, 0.0, 0.0, 1.0]);

        Self {
            port_light,
            star_light,
            cam_pos: Vec3::new(25.0, 10.0, 30.0),
            cam_yaw: std::f32::consts::PI + 0.4,
            cam_pitch: -0.3,
        }
    }

    fn update(
        &mut self,
        renderer: &mut Renderer,
        dt: f32,
        elapsed: f32,
        input: &InputState,
    ) -> Camera {
        self.cam_yaw += input.mouse_delta.0 * LOOK_SENS;
        self.cam_pitch = (self.cam_pitch - input.mouse_delta.1 * LOOK_SENS).clamp(-1.55, 1.55);

        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let fwd = Vec3::new(sy * cp, sp, -cy * cp);
        let right = Vec3::new(cy, 0.0, sy);
        let up = Vec3::Y;

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
            self.cam_pos += up * FLY_SPEED * dt;
        }
        if input.keys.contains(&helio_wasm::KeyCode::ShiftLeft) {
            self.cam_pos -= up * FLY_SPEED * dt;
        }

        // Nav light blink (1 Hz)
        let blink = ((elapsed * std::f32::consts::TAU).sin() > 0.0) as u8 as f32;
        let _ = renderer.scene_mut().update_light(
            self.port_light,
            point_light([-22.5, 0.0, 0.0], [1.0, 0.05, 0.05], 12.0 * blink, 8.0),
        );
        let _ = renderer.scene_mut().update_light(
            self.star_light,
            point_light(
                [22.5, 0.0, 0.0],
                [0.05, 1.0, 0.1],
                12.0 * (1.0 - blink),
                8.0,
            ),
        );

        Camera::perspective_look_at(
            self.cam_pos,
            self.cam_pos + fwd,
            Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            1280.0 / 720.0,
            0.1,
            300.0,
        )
    }
}

fn sx_for_nav(s: f32) -> f32 {
    s * 22.5
}



