use std::sync::Arc;
use glam::Vec3;
use helio::{Camera, Renderer, TonemapOperator};
use helio_wasm::{HelioWasmApp, InputState, KeyCode};
use crate::common::{make_material, plane_mesh, insert_object};

const LOOK_SENS: f32 = 0.0024;
const FLY_SPEED: f32 = 5.0;

pub struct Demo {
    cam_pos: Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    light_ids: [helio::LightId; 3],
    ies_enabled: [bool; 3],
    gobo_enabled: bool,
}

impl HelioWasmApp for Demo {
    fn title() -> &'static str { "Helio — IES Light Profiles" }

    fn render_scale() -> f32 { 1.0 }

    fn init(renderer: &mut Renderer, _device: Arc<wgpu::Device>,
            _queue: Arc<wgpu::Queue>, _w: u32, _h: u32) -> Self {
        let floor_mat = renderer.scene_mut().insert_material(make_material(
            [0.15, 0.15, 0.16, 1.0], 0.8, 0.0, [0.0, 0.0, 0.0], 0.0,
        ));
        let ground = renderer.scene_mut().insert_actor(
            helio::SceneActor::mesh(plane_mesh([0.0, 0.0, 0.0], 6.0)));
        let _ = insert_object(renderer, ground, floor_mat, glam::Mat4::IDENTITY, 6.0);

        let light_ids: [helio::LightId; 3] = std::array::from_fn(|i| {
            let mut light = helio::GpuLight::default();
            light.light_type = helio::LightType::Spot as u32;
            light.color_intensity = match i {
                0 => [1.0, 0.3, 0.2, 8.0],
                1 => [0.2, 1.0, 0.3, 8.0],
                _ => [0.3, 0.4, 1.0, 8.0],
            };
            light.direction_outer = match i {
                0 => [0.0, -1.0, 0.0, 0.85],
                1 => [0.0, -1.0, 0.0, 0.75],
                _ => [0.0, -1.0, 0.0, 0.50],
            };
            light.position_range = match i {
                0 => [-1.5, 3.0, -1.0, 8.0],
                1 => [1.5, 3.0, -1.0, 8.0],
                _ => [0.0, 3.0, 2.0, 8.0],
            };
            light.inner_angle = match i {
                0 => 0.98,
                1 => 0.92,
                _ => 0.80,
            };
            renderer.scene_mut().insert_actor(helio::SceneActor::light(light)).as_light().unwrap()
        });

        Self {
            cam_pos: Vec3::new(0.0, 2.0, 5.0),
            cam_yaw: 0.0,
            cam_pitch: -0.2,
            light_ids,
            ies_enabled: [false, false, false],
            gobo_enabled: false,
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

        // Toggle IES on keypress
        if input.keys.contains(&KeyCode::Digit1) { self.ies_enabled[0] = !self.ies_enabled[0]; }
        if input.keys.contains(&KeyCode::Digit2) { self.ies_enabled[1] = !self.ies_enabled[1]; }
        if input.keys.contains(&KeyCode::Digit3) { self.ies_enabled[2] = !self.ies_enabled[2]; }
        if input.keys.contains(&KeyCode::KeyG) { self.gobo_enabled = !self.gobo_enabled; }

        for (i, &enabled) in self.ies_enabled.iter().enumerate() {
            let mut light = helio::GpuLight::default();
            light.light_type = helio::LightType::Spot as u32;
            light.color_intensity = match i {
                0 => [1.0, 0.3, 0.2, 8.0],
                1 => [0.2, 1.0, 0.3, 8.0],
                _ => [0.3, 0.4, 1.0, 8.0],
            };
            light.direction_outer = match i {
                0 => [0.0, -1.0, 0.0, 0.85],
                1 => [0.0, -1.0, 0.0, 0.75],
                _ => [0.0, -1.0, 0.0, 0.50],
            };
            light.position_range = match i {
                0 => [-1.5, 3.0, -1.0, 8.0],
                1 => [1.5, 3.0, -1.0, 8.0],
                _ => [0.0, 3.0, 2.0, 8.0],
            };
            light.inner_angle = match i {
                0 => 0.98,
                1 => 0.92,
                _ => 0.80,
            };
            if enabled {
                light.ies_profile_index = 0;
                light.ies_angle_scale = match i { 0 => 0.5, 1 => 1.0, _ => 2.0 };
            }
            if self.gobo_enabled {
                light.light_function_index = 0;
            }
            let _ = renderer.scene_mut().update_light(self.light_ids[i], light);
        }

        let camera = Camera::perspective_look_at(
            self.cam_pos, self.cam_pos + fwd, Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            input.aspect_ratio(), 0.1, 200.0,
        );
        camera
    }
}
