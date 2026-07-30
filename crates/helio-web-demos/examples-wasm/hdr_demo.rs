use std::sync::Arc;
use glam::Vec3;
use helio::{Camera, Renderer};
use helio_wasm::{HelioWasmApp, InputState, KeyCode};
use crate::common::{cube_mesh, directional_light, insert_object, make_material, plane_mesh, point_light};

const LOOK_SENS: f32 = 0.0024;
const FLY_SPEED: f32 = 5.0;

pub struct Demo {
    cam_pos: Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    hdr_mode: u32,
}

impl HelioWasmApp for Demo {
    fn title() -> &'static str { "Helio — HDR Display Output" }

    fn render_scale() -> f32 { 1.0 }

    fn init(renderer: &mut Renderer, _device: Arc<wgpu::Device>,
            _queue: Arc<wgpu::Queue>, _w: u32, _h: u32) -> Self {
        let white = renderer.scene_mut().insert_material(make_material(
            [0.9, 0.9, 0.92, 1.0], 0.6, 0.0, [0.0, 0.0, 0.0], 0.0,
        ));
        let emissive_red = renderer.scene_mut().insert_material(make_material(
            [1.0, 0.1, 0.1, 1.0], 0.3, 0.0, [10.0, 0.5, 0.5], 10.0,
        ));
        let emissive_green = renderer.scene_mut().insert_material(make_material(
            [0.1, 1.0, 0.1, 1.0], 0.3, 0.0, [0.5, 10.0, 0.5], 10.0,
        ));
        let emissive_blue = renderer.scene_mut().insert_material(make_material(
            [0.1, 0.1, 1.0, 1.0], 0.3, 0.0, [0.5, 0.5, 10.0], 10.0,
        ));
        let emissive_sun = renderer.scene_mut().insert_material(make_material(
            [1.0, 0.9, 0.7, 1.0], 0.2, 0.0, [50.0, 45.0, 35.0], 50.0,
        ));
        let metal = renderer.scene_mut().insert_material(make_material(
            [0.95, 0.93, 0.88, 1.0], 0.1, 1.0, [0.0, 0.0, 0.0], 0.0,
        ));

        let ground = renderer.scene_mut().insert_actor(
            helio::SceneActor::mesh(plane_mesh([0.0, 0.0, 0.0], 8.0)));
        let _ = insert_object(renderer, ground, white, glam::Mat4::IDENTITY, 8.0);

        let red_cube = renderer.scene_mut().insert_actor(
            helio::SceneActor::mesh(cube_mesh([-1.5, 0.5, -1.0], 0.5)));
        let _ = insert_object(renderer, red_cube, emissive_red,
                              glam::Mat4::IDENTITY, 0.5);

        let green_cube = renderer.scene_mut().insert_actor(
            helio::SceneActor::mesh(cube_mesh([1.5, 0.5, -1.0], 0.5)));
        let _ = insert_object(renderer, green_cube, emissive_green,
                              glam::Mat4::IDENTITY, 0.5);

        let blue_cube = renderer.scene_mut().insert_actor(
            helio::SceneActor::mesh(cube_mesh([0.0, 0.5, 1.5], 0.5)));
        let _ = insert_object(renderer, blue_cube, emissive_blue,
                              glam::Mat4::IDENTITY, 0.5);

        let metal_cube = renderer.scene_mut().insert_actor(
            helio::SceneActor::mesh(cube_mesh([-1.5, 0.5, 2.5], 0.5)));
        let _ = insert_object(renderer, metal_cube, metal,
                              glam::Mat4::IDENTITY, 0.5);

        let sun_sphere = renderer.scene_mut().insert_actor(
            helio::SceneActor::mesh(cube_mesh([3.0, 4.0, -3.0], 0.4)));
        let _ = insert_object(renderer, sun_sphere, emissive_sun,
                              glam::Mat4::IDENTITY, 0.4);

        renderer.scene_mut().insert_actor(
            helio::SceneActor::light(directional_light(
                [0.3, -0.8, 0.5], [1.0, 0.95, 0.85], 15.0)));
        renderer.scene_mut().insert_actor(
            helio::SceneActor::light(point_light(
                [3.0, 4.0, -3.0], [1.0, 0.9, 0.7], 20.0, 15.0)));

        Self {
            cam_pos: Vec3::new(0.0, 3.0, 6.0),
            cam_yaw: 0.0,
            cam_pitch: -0.3,
            hdr_mode: 0,
        }
    }

    fn update(&mut self, renderer: &mut Renderer, dt: f32,
              _elapsed: f32, input: &InputState) -> Camera {
        self.cam_yaw += input.mouse_delta.0 * LOOK_SENS;
        self.cam_pitch = (self.cam_pitch - input.mouse_delta.1 * LOOK_SENS).clamp(-1.55, 1.55);
        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let fwd = Vec3::new(sy * cp, sp, -cy * cp);
        let right = Vec3::new(cy, 0.0, sy);

        if input.keys.contains(&KeyCode::KeyW) { self.cam_pos += fwd * FLY_SPEED * dt; }
        if input.keys.contains(&KeyCode::KeyS) { self.cam_pos -= fwd * FLY_SPEED * dt; }
        if input.keys.contains(&KeyCode::KeyA) { self.cam_pos -= right * FLY_SPEED * dt; }
        if input.keys.contains(&KeyCode::KeyD) { self.cam_pos += right * FLY_SPEED * dt; }
        if input.keys.contains(&KeyCode::Space) { self.cam_pos.y += FLY_SPEED * dt; }
        if input.keys.contains(&KeyCode::ShiftLeft) { self.cam_pos.y -= FLY_SPEED * dt; }

        if input.keys.contains(&KeyCode::KeyH) {
            self.hdr_mode = (self.hdr_mode + 1) % 4;
            let name = match self.hdr_mode {
                0 => "LDR (sRGB)",
                1 => "HDR10 (PQ)",
                2 => "scRGB",
                _ => "Passthrough",
            };
            log::info!("[HDR Demo] Switched to {}", name);
        }

        let mut camera = Camera::perspective_look_at(
            self.cam_pos, self.cam_pos + fwd, Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            input.aspect_ratio(), 0.1, 200.0,
        );
        match self.hdr_mode {
            0 => {
                camera.postprocess_settings.hdr_output_mode = helio::HdrOutputMode::Ldr;
                camera.postprocess_settings.tonemap_operator = helio::TonemapOperator::Aces;
            }
            1 => {
                camera.postprocess_settings.hdr_output_mode = helio::HdrOutputMode::Hdr10;
                camera.postprocess_settings.tonemap_operator = helio::TonemapOperator::None;
            }
            2 => {
                camera.postprocess_settings.hdr_output_mode = helio::HdrOutputMode::ScRgb;
                camera.postprocess_settings.tonemap_operator = helio::TonemapOperator::None;
            }
            _ => {
                camera.postprocess_settings.hdr_output_mode = helio::HdrOutputMode::Passthrough;
                camera.postprocess_settings.tonemap_operator = helio::TonemapOperator::None;
            }
        }
        camera
    }
}
