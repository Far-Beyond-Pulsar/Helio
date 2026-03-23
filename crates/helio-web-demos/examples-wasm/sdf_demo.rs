//! WASM twin of `sdf_demo` — SDF clipmap with rolling terrain and edits.
//!
//! Controls: WASD fly (click to disable auto-orbit), mouse look.

use std::sync::Arc;

use glam::{Mat4, Vec3};
use helio::{Camera, Renderer};
use helio_pass_sdf::{
    SdfClipmapPass,
    edit_list::{BooleanOp, SdfEdit},
    primitives::{SdfShapeParams, SdfShapeType},
    terrain::TerrainConfig,
};
use helio_wasm::{HelioWasmApp, InputState};

const SPEED: f32 = 25.0;
const SENS:  f32 = 0.0020;

pub struct Demo {
    cam_pos:    Vec3,
    cam_yaw:    f32,
    cam_pitch:  f32,
    orbit_mode: bool,
}

impl HelioWasmApp for Demo {
    fn title() -> &'static str { "Helio — SDF Clipmap Demo" }

    fn init(renderer: &mut Renderer, device: Arc<wgpu::Device>,
            _queue: Arc<wgpu::Queue>, w: u32, h: u32) -> Self {
        let _ = (w, h);
        let mut sdf = {
            let camera_buf = renderer.camera_buffer();
            // The surface format used by the runner is fixed; use a sRGB-compatible format.
            // We use Bgra8UnormSrgb as a reasonable default; the pass itself will
            // adapts if the renderer provides a different target format.
            SdfClipmapPass::new(&device, camera_buf, wgpu::TextureFormat::Bgra8UnormSrgb)
        };

        sdf.set_terrain(TerrainConfig::rolling());

        sdf.add_edit(SdfEdit {
            shape:        SdfShapeType::Sphere,
            op:           BooleanOp::Union,
            transform:    Mat4::from_translation(Vec3::new(0.0, 12.0, 0.0)),
            params:       SdfShapeParams::sphere(6.0),
            blend_radius: 2.0,
        });
        sdf.add_edit(SdfEdit {
            shape:        SdfShapeType::Capsule,
            op:           BooleanOp::Union,
            transform:    Mat4::from_translation(Vec3::new(20.0, 8.0, 5.0)),
            params:       SdfShapeParams::capsule(2.0, 8.0),
            blend_radius: 1.0,
        });
        sdf.add_edit(SdfEdit {
            shape:        SdfShapeType::Sphere,
            op:           BooleanOp::Subtraction,
            transform:    Mat4::from_translation(Vec3::new(0.0, 2.0, 0.0)),
            params:       SdfShapeParams::sphere(5.0),
            blend_radius: 1.5,
        });

        renderer.add_pass(Box::new(sdf));

        renderer.set_ambient([0.15, 0.18, 0.28], 0.04);
        renderer.set_clear_color([0.4, 0.55, 0.85, 1.0]);

        Self {
            cam_pos:    Vec3::new(0.0, 30.0, 80.0),
            cam_yaw:    std::f32::consts::PI,
            cam_pitch:  -0.18,
            orbit_mode: true,
        }
    }

    fn update(&mut self, _renderer: &mut Renderer, dt: f32, elapsed: f32,
              input: &InputState) -> Camera {
        // Exit orbit on first movement input
        if input.keys.contains(&helio_wasm::KeyCode::KeyW)
            || input.keys.contains(&helio_wasm::KeyCode::KeyS)
            || input.keys.contains(&helio_wasm::KeyCode::KeyA)
            || input.keys.contains(&helio_wasm::KeyCode::KeyD)
            || input.keys.contains(&helio_wasm::KeyCode::Space)
        {
            self.orbit_mode = false;
        }

        self.cam_yaw   += input.mouse_delta.0 * SENS;
        self.cam_pitch  = (self.cam_pitch - input.mouse_delta.1 * SENS).clamp(-1.55, 1.55);

        if self.orbit_mode {
            let angle  = elapsed * 0.07;
            let radius = 100.0_f32;
            let height = 30.0 + 15.0 * (elapsed * 0.022).sin();
            self.cam_pos = Vec3::new(radius * angle.cos(), height, radius * angle.sin());
            let dir = (-self.cam_pos).normalize();
            self.cam_yaw   = dir.z.atan2(dir.x);
            self.cam_pitch = dir.y.asin();
        } else {
            let (sy, cy) = self.cam_yaw.sin_cos();
            let (sp, cp) = self.cam_pitch.sin_cos();
            let fwd   = Vec3::new(sy * cp, sp, -cy * cp);
            let right = Vec3::new(cy, 0.0, sy);
            if input.keys.contains(&helio_wasm::KeyCode::KeyW) { self.cam_pos += fwd   * SPEED * dt; }
            if input.keys.contains(&helio_wasm::KeyCode::KeyS) { self.cam_pos -= fwd   * SPEED * dt; }
            if input.keys.contains(&helio_wasm::KeyCode::KeyA) { self.cam_pos -= right * SPEED * dt; }
            if input.keys.contains(&helio_wasm::KeyCode::KeyD) { self.cam_pos += right * SPEED * dt; }
            if input.keys.contains(&helio_wasm::KeyCode::Space)     { self.cam_pos.y += SPEED * dt; }
            if input.keys.contains(&helio_wasm::KeyCode::ShiftLeft) { self.cam_pos.y -= SPEED * dt; }
        }

        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let fwd = Vec3::new(sy * cp, sp, -cy * cp);

        Camera::perspective_look_at(
            self.cam_pos, self.cam_pos + fwd, Vec3::Y,
            std::f32::consts::FRAC_PI_4, 1280.0 / 720.0, 0.5, 2000.0,
        )
    }
}
