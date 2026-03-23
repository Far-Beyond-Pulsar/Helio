//! WASM twin of `load_fbx_embedded` — embedded FBX showcase.
//!
//! Loads `test.fbx` from workspace root via `include_bytes!`.
//! Controls: WASD fly, mouse look.

use std::sync::Arc;

use glam::Vec3;
use helio::{Camera, Renderer};
use helio_asset_compat::{load_scene_bytes_with_config, upload_scene, LoadConfig};
use helio_wasm::{HelioWasmApp, InputState};

use crate::common::{
    box_mesh, directional_light, insert_object, make_material, plane_mesh, spot_light,
};

const EMBEDDED_BYTES: &[u8] = include_bytes!("../../../test.fbx");
const LOOK_SENS: f32 = 0.002;

pub struct Demo {
    cam_pos:   Vec3,
    cam_yaw:   f32,
    cam_pitch: f32,
    speed:     f32,
}

impl HelioWasmApp for Demo {
    fn title() -> &'static str { "Helio — Load FBX Embedded" }

    fn init(renderer: &mut Renderer, _device: Arc<wgpu::Device>,
            _queue: Arc<wgpu::Queue>, _w: u32, _h: u32) -> Self {
        let base_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..").join("..");

        let mut cam_pos   = Vec3::new(1.5, 1.0, 5.0);
        let mut cam_yaw   = std::f32::consts::PI;
        let mut cam_pitch = -0.1_f32;
        let mut speed     = 8.0_f32;

        match load_scene_bytes_with_config(
            EMBEDDED_BYTES,
            "fbx",
            Some(base_dir.as_path()),
            LoadConfig::default().with_uv_flip(false),
        ) {
            Ok(scene) => {
                // Compute AABB from mesh vertices
                let mut min = Vec3::splat(f32::INFINITY);
                let mut max = Vec3::splat(f32::NEG_INFINITY);
                for mesh in &scene.meshes {
                    for v in &mesh.vertices {
                        let p = Vec3::from([v.position[0], v.position[1], v.position[2]]);
                        min = min.min(p);
                        max = max.max(p);
                    }
                }
                let center  = (min + max) * 0.5;
                let extents = (max - min).max(Vec3::splat(0.1));
                let radius  = extents.length().max(2.5);

                // Upload all meshes + materials in one pass
                if let Ok(uploaded) = upload_scene(renderer, &scene) {
                    for (i, mesh_data) in scene.meshes.iter().enumerate() {
                        if let Some(mesh_id) = uploaded.mesh_ids.get(i).copied() {
                            let mat_id = uploaded
                                .mesh_material(i, mesh_data)
                                .or_else(|| uploaded.material_ids.first().copied());
                            if let Some(mat_id) = mat_id {
                                let _ = insert_object(renderer, mesh_id, mat_id,
                                                      glam::Mat4::IDENTITY, radius);
                            }
                        }
                    }
                }

                // Stage + lighting
                let floor_y = min.y - radius * 0.08;
                let floor_m = renderer.insert_material(make_material([0.07, 0.08, 0.10, 1.0], 0.16, 0.02, [0.0;3], 0.0));
                let floor = renderer.insert_mesh(plane_mesh([center.x, floor_y, center.z], radius * 1.55));
                insert_object(renderer, floor, floor_m, glam::Mat4::IDENTITY, radius * 1.55).unwrap();

                let ped_m = renderer.insert_material(make_material([0.11, 0.12, 0.15, 1.0], 0.28, 0.04, [0.0;3], 0.0));
                let ped = renderer.insert_mesh(box_mesh(
                    [center.x, floor_y + radius * 0.05, center.z],
                    [radius * 0.62, radius * 0.05, radius * 0.62],
                ));
                insert_object(renderer, ped, ped_m, glam::Mat4::IDENTITY, radius).unwrap();

                let back_m = renderer.insert_material(make_material([0.04, 0.05, 0.08, 1.0], 0.82, 0.0, [0.04, 0.06, 0.12], 0.03));
                let back = renderer.insert_mesh(box_mesh(
                    [center.x, floor_y + radius * 0.62, center.z - radius * 1.35],
                    [radius * 1.35, radius * 0.62, radius * 0.05],
                ));
                insert_object(renderer, back, back_m, glam::Mat4::IDENTITY, radius * 1.5).unwrap();

                let focus = center + Vec3::new(0.0, (max.y - min.y) * 0.18, 0.0);
                let r     = radius;
                let key  = focus + Vec3::new(r * 0.22, r * 0.34, r * 0.24);
                let fill = focus + Vec3::new(-r * 0.26, r * 0.14, r * 0.28);
                let rim  = focus + Vec3::new(-r * 0.30, r * 0.22, -r * 0.32);
                renderer.insert_light(spot_light(key.to_array(),  (focus - key).normalize().to_array(),  [1.0, 0.80, 0.62], 18.0, r * 0.62, 0.20, 0.38));
                renderer.insert_light(spot_light(fill.to_array(), (focus - fill).normalize().to_array(), [0.52, 0.66, 1.0], 6.5,  r * 0.59, 0.28, 0.46));
                renderer.insert_light(spot_light(rim.to_array(),  (focus - rim).normalize().to_array(),  [0.36, 0.55, 1.0], 14.0, r * 0.57, 0.22, 0.40));
                renderer.insert_light(directional_light([0.15, -1.0, 0.1], [0.07, 0.09, 0.14], 0.3));
                renderer.set_ambient([0.0, 0.0, 0.0], 0.0);

                cam_pos   = center + Vec3::new(r * 0.55, r * 0.28, r * 1.55);
                cam_yaw   = std::f32::consts::PI + 0.1;
                cam_pitch = -0.12;
                speed     = (r * 0.85).clamp(8.0, 42.0);
            }
            Err(e) => {
                log::warn!("Failed to load embedded FBX: {e:?}. Showing empty scene.");
                renderer.insert_light(directional_light([0.2, -1.0, 0.4], [1.0, 0.95, 0.85], 0.01));
                renderer.set_ambient([0.1, 0.12, 0.18], 0.05);
                renderer.set_clear_color([0.02, 0.02, 0.04, 1.0]);
            }
        };

        Self { cam_pos, cam_yaw, cam_pitch, speed }
    }

    fn update(&mut self, _renderer: &mut Renderer, dt: f32, _elapsed: f32,
              input: &InputState) -> Camera {
        self.cam_yaw   += input.mouse_delta.0 * LOOK_SENS;
        self.cam_pitch  = (self.cam_pitch - input.mouse_delta.1 * LOOK_SENS).clamp(-1.5, 1.5);

        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let fwd   = Vec3::new(sy * cp, sp, -cy * cp);
        let right = Vec3::new(cy, 0.0, sy);
        let up    = Vec3::Y;

        if input.keys.contains(&helio_wasm::KeyCode::KeyW) { self.cam_pos += fwd   * self.speed * dt; }
        if input.keys.contains(&helio_wasm::KeyCode::KeyS) { self.cam_pos -= fwd   * self.speed * dt; }
        if input.keys.contains(&helio_wasm::KeyCode::KeyA) { self.cam_pos -= right * self.speed * dt; }
        if input.keys.contains(&helio_wasm::KeyCode::KeyD) { self.cam_pos += right * self.speed * dt; }
        if input.keys.contains(&helio_wasm::KeyCode::Space)     { self.cam_pos += up * self.speed * dt; }
        if input.keys.contains(&helio_wasm::KeyCode::ShiftLeft) { self.cam_pos -= up * self.speed * dt; }

        Camera::perspective_look_at(
            self.cam_pos, self.cam_pos + fwd, Vec3::Y,
            std::f32::consts::FRAC_PI_4, 1280.0 / 720.0, 0.05, 500.0,
        )
    }
}
