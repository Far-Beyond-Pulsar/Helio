use std::sync::Arc;
use glam::Vec3;
use helio::{Camera, Renderer, TonemapOperator};
use helio_wasm::{HelioWasmApp, InputState, KeyCode};
use helio_pass_postprocess::LutBuilder;
use crate::common::{cube_mesh, insert_object, make_material, plane_mesh, point_light};

const LOOK_SENS: f32 = 0.0024;
const FLY_SPEED: f32 = 5.0;

pub struct Demo {
    cam_pos: Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    grading_mode: u32,
    hue_shift: f32,
    lut_intensity: f32,
    lut_size: u32,
    lut_generation: u32,
    lut_builder: Option<LutBuilder>,
}

impl HelioWasmApp for Demo {
    fn title() -> &'static str { "Helio — Color Grading Demo" }

    fn render_scale() -> f32 { 1.0 }

    fn init(renderer: &mut Renderer, device: Arc<wgpu::Device>,
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

        renderer.scene_mut().insert_actor(
            helio::SceneActor::light(point_light(
                [3.0, 4.0, -3.0], [1.0, 0.9, 0.7], 20.0, 15.0)));

        let lut_builder = LutBuilder::new(&device, 16);

        Self {
            cam_pos: Vec3::new(0.0, 3.0, 6.0),
            cam_yaw: 0.0,
            cam_pitch: -0.3,
            grading_mode: 0,
            hue_shift: 0.0,
            lut_intensity: 1.0,
            lut_size: 16,
            lut_generation: 1,
            lut_builder: Some(lut_builder),
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

        // Mode / parameter changes
        if input.keys.contains(&KeyCode::Digit1) { self.grading_mode = 0; }
        if input.keys.contains(&KeyCode::Digit2) { self.grading_mode = 1; }
        if input.keys.contains(&KeyCode::Digit3) { self.grading_mode = 2; self.lut_generation += 1; }
        if input.keys.contains(&KeyCode::KeyQ) { self.hue_shift = (self.hue_shift - 5.0).clamp(-180.0, 180.0); }
        if input.keys.contains(&KeyCode::KeyE) { self.hue_shift = (self.hue_shift + 5.0).clamp(-180.0, 180.0); }
        if input.keys.contains(&KeyCode::KeyR) { self.lut_intensity = (self.lut_intensity - 0.1).clamp(0.0, 1.0); }
        if input.keys.contains(&KeyCode::KeyF) { self.lut_intensity = (self.lut_intensity + 0.1).clamp(0.0, 1.0); }

        let mut camera = Camera::perspective_look_at(
            self.cam_pos, self.cam_pos + fwd, Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            input.aspect_ratio(), 0.1, 200.0,
        );
        camera.postprocess_settings.tonemap_operator = TonemapOperator::Aces;

        match self.grading_mode {
            0 => {
                camera.postprocess_settings.lut_platform = 0;
                camera.postprocess_settings.color_contrast = [1.2, 1.2, 1.2];
                camera.postprocess_settings.color_saturation = [1.3, 1.3, 1.3];
            }
            1 => {
                camera.postprocess_settings.lut_platform = 0;
                camera.postprocess_settings.lift_color = [0.02, 0.01, 0.04];
                camera.postprocess_settings.gamma_color = [0.95, 1.0, 1.05];
                camera.postprocess_settings.gain_color = [1.1, 1.0, 0.95];
                camera.postprocess_settings.hue_shift = self.hue_shift;
            }
            _ => {
                camera.postprocess_settings.lut_platform = 1;
                camera.postprocess_settings.lut_intensity = self.lut_intensity;
                camera.postprocess_settings.hue_shift = self.hue_shift;
            }
        }

        if let Some(ref mut builder) = self.lut_builder {
            let rebuilt = builder.build_if_needed(
                renderer.device(),
                renderer.queue(),
                renderer.postprocess_buffer(),
                self.lut_generation,
            );
            if rebuilt || renderer.color_grading_lut().is_none() {
                let view = builder.lut_view().clone();
                renderer.set_color_grading_lut(Some(view));
            }
        }

        camera
    }
}
