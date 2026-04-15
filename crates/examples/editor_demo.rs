//! Shipyard Demo — Helio v3
//!
//! A waterfront shipyard scene featuring stacks of shipping containers,
//! loading cranes, dock infrastructure, and area lighting.
//!
//! The shipping container mesh is loaded from `container with textures.fbx`
//! and instanced ~300 times across the yard in stacked bays.
//!
//! # Controls
//!
//! | Input                  | Action                                      |
//! |------------------------|---------------------------------------------|
//! | **Right-click hold**   | Capture cursor for free-fly camera          |
//! | **Right-click release**| Release cursor for object picking           |
//! | WASD                   | Fly forward / left / back / right (RMB)     |
//! | Space / L-Shift        | Fly up / down (hold RMB)                    |
//! | **Left-click**         | Pick object under cursor (cursor free)      |
//! | G                      | Switch to **Translate** gizmo               |
//! | R                      | Switch to **Rotate** gizmo                  |
//! | S                      | Switch to **Scale** gizmo                   |
//! | Ctrl+D                 | **Duplicate** selected object               |
//! | Delete                 | **Delete** selected object                  |
//! | Tab                    | Toggle editor grid                          |
//! | **F11** / **Alt+Enter**| Toggle fullscreen                           |
//! | Escape                 | Deselect → exit                             |

mod v3_demo_common;

use helio::{
    Camera, EditorState, GizmoMode, Renderer, RendererConfig,
    SceneActor, ScenePicker, required_wgpu_features, required_wgpu_limits,
};
use helio_asset_compat::{load_scene_bytes_with_config, upload_sectioned_scene, LoadConfig};
use v3_demo_common::{
    box_mesh, insert_object_with_movability, make_material, plane_mesh, point_light,
    sphere_mesh,
};

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use std::collections::HashSet;
use std::sync::Arc;

const CRATES_FBX: &[u8] = include_bytes!("../../models/source/container with textures.fbx");

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App { state: None };
    event_loop.run_app(&mut app).expect("run");
}

// ─────────────────────────────────────────────────────────────────────────────
// App scaffold
// ─────────────────────────────────────────────────────────────────────────────

struct App {
    state: Option<AppState>,
}

struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    surface_format: wgpu::TextureFormat,
    renderer: Renderer,
    last_frame: std::time::Instant,

    // ── Camera ────────────────────────────────────────────────────────────
    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    right_mouse_held: bool,
    mouse_delta: (f32, f32),
    cursor_pos: (f32, f32),
    cam_speed: f32,

    // ── Editor ────────────────────────────────────────────────────────────
    editor: EditorState,
    picker: ScenePicker,
    grid_enabled: bool,
    is_fullscreen: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        // ── Window & wgpu setup ───────────────────────────────────────────
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Helio — Shipyard Demo")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
                )
                .expect("window"),
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
            ..Default::default()
        });
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: required_wgpu_features(adapter.features()),
                required_limits: required_wgpu_limits(adapter.limits()),
                ..Default::default()
            },
        ))
        .expect("device");
        device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| {
            panic!("[GPU] {:?}", e);
        }));
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let sz = window.inner_size();
        surface.configure(
            &device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: sz.width,
                height: sz.height,
                present_mode: wgpu::PresentMode::AutoVsync,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            },
        );

        // ── Renderer ──────────────────────────────────────────────────────
        let mut renderer = Renderer::new(
            device.clone(),
            queue.clone(),
            RendererConfig::new(sz.width, sz.height, format),
        );
        renderer.set_editor_mode(true);
        // Night sky — deep navy
        renderer.set_clear_color([0.03, 0.05, 0.10, 1.0]);
        // Moonlight ambient — bright enough to read the container colours
        renderer.set_ambient([0.18, 0.22, 0.32], 0.35);

        // ── Scene construction ────────────────────────────────────────────
        let mut picker = ScenePicker::new();

        // ── Materials ─────────────────────────────────────────────────────
        // Dock concrete
        let mat_dock = renderer.scene_mut().insert_material(
            make_material([0.42, 0.40, 0.38, 1.0], 0.95, 0.0, [0.0; 3], 0.0));
        // Road apron
        let mat_road = renderer.scene_mut().insert_material(
            make_material([0.22, 0.22, 0.22, 1.0], 0.95, 0.0, [0.0; 3], 0.0));
        // Safety stripe yellow
        let mat_stripe = renderer.scene_mut().insert_material(
            make_material([0.92, 0.75, 0.05, 1.0], 0.7, 0.0, [0.0; 3], 0.0));
        // Crane steel
        let mat_steel = renderer.scene_mut().insert_material(
            make_material([0.25, 0.26, 0.28, 1.0], 0.15, 0.6, [0.0; 3], 0.0));
        // Crane safety orange
        let mat_orange = renderer.scene_mut().insert_material(
            make_material([0.85, 0.35, 0.05, 1.0], 0.3, 0.4, [0.0; 3], 0.0));
        // Warning beacon emissive red
        let mat_warning = renderer.scene_mut().insert_material(
            make_material([1.0, 0.1, 0.05, 1.0], 0.4, 0.0, [1.0, 0.05, 0.0], 1.5));
        // Harbour water
        let mat_water = renderer.scene_mut().insert_material(
            make_material([0.04, 0.12, 0.20, 1.0], 0.05, 0.95, [0.0; 3], 0.0));
        // Bollard dark iron
        let mat_bollard = renderer.scene_mut().insert_material(
            make_material([0.18, 0.14, 0.10, 1.0], 0.85, 0.0, [0.0; 3], 0.0));
        // Mast pole lamp housing
        let mat_lamp = renderer.scene_mut().insert_material(
            make_material([0.90, 0.85, 0.50, 1.0], 0.3, 0.0, [0.6, 0.55, 0.1], 0.8));

        // ── Ground — large dock apron ──────────────────────────────────────
        let dock_upload = plane_mesh([0.0, 0.0, 0.0], 100.0);
        let dock_mesh = renderer.scene_mut()
            .insert_actor(SceneActor::mesh(dock_upload.clone())).as_mesh().unwrap();
        picker.register_mesh(dock_mesh, &dock_upload);
        insert_object_with_movability(
            &mut renderer, dock_mesh, mat_dock,
            glam::Mat4::IDENTITY, 120.0, None,
        ).ok();

        // ── Harbour water ──────────────────────────────────────────────────
        let water_upload = plane_mesh([0.0, -0.15, 0.0], 80.0);
        let water_mesh = renderer.scene_mut()
            .insert_actor(SceneActor::mesh(water_upload.clone())).as_mesh().unwrap();
        picker.register_mesh(water_mesh, &water_upload);
        insert_object_with_movability(
            &mut renderer, water_mesh, mat_water,
            glam::Mat4::from_translation(glam::Vec3::new(115.0, 0.0, 0.0)),
            90.0, None,
        ).ok();

        // ── Quay wall ─────────────────────────────────────────────────────
        let quay_upload = box_mesh([0.0, 0.0, 0.0], [95.0, 3.5, 1.2]);
        let quay_mesh = renderer.scene_mut()
            .insert_actor(SceneActor::mesh(quay_upload.clone())).as_mesh().unwrap();
        picker.register_mesh(quay_mesh, &quay_upload);
        insert_object_with_movability(
            &mut renderer, quay_mesh, mat_dock,
            glam::Mat4::from_translation(glam::Vec3::new(0.0, -1.75, -51.0)),
            50.0, None,
        ).ok();

        // ── Road apron strip ───────────────────────────────────────────────
        let road_upload = box_mesh([0.0, 0.0, 0.0], [95.0, 0.05, 6.0]);
        let road_mesh = renderer.scene_mut()
            .insert_actor(SceneActor::mesh(road_upload.clone())).as_mesh().unwrap();
        picker.register_mesh(road_mesh, &road_upload);
        insert_object_with_movability(
            &mut renderer, road_mesh, mat_road,
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.01, -44.0)),
            52.0, None,
        ).ok();

        // Yellow safety stripes (two parallel lines)
        let stripe_upload = box_mesh([0.0, 0.0, 0.0], [95.0, 0.06, 0.4]);
        let stripe_mesh = renderer.scene_mut()
            .insert_actor(SceneActor::mesh(stripe_upload.clone())).as_mesh().unwrap();
        picker.register_mesh(stripe_mesh, &stripe_upload);
        for sz_off in [-1.5_f32, 1.5_f32] {
            insert_object_with_movability(
                &mut renderer, stripe_mesh, mat_stripe,
                glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.02, -44.0 + sz_off)),
                52.0, None,
            ).ok();
        }

        // ── Bollards along the quay edge ───────────────────────────────────
        let bollard_upload = box_mesh([0.0, 0.0, 0.0], [0.28, 0.6, 0.28]);
        let bollard_mesh = renderer.scene_mut()
            .insert_actor(SceneActor::mesh(bollard_upload.clone())).as_mesh().unwrap();
        picker.register_mesh(bollard_mesh, &bollard_upload);
        for bi in 0..18i32 {
            let bx = -85.0 + bi as f32 * 10.0;
            insert_object_with_movability(
                &mut renderer, bollard_mesh, mat_bollard,
                glam::Mat4::from_translation(glam::Vec3::new(bx, 0.6, -43.5)),
                0.8, None,
            ).ok();
        }

        // ── Portal cranes — two units at each end of the yard ─────────────
        for crane_i in 0i32..2 {
            let cx = -50.0 + crane_i as f32 * 100.0;
            let cz = -38.0_f32;

            // Leg pair
            let leg_upload = box_mesh([0.0; 3], [1.0, 18.0, 1.0]);
            let leg_mesh = renderer.scene_mut()
                .insert_actor(SceneActor::mesh(leg_upload.clone())).as_mesh().unwrap();
            picker.register_mesh(leg_mesh, &leg_upload);
            for leg_dx in [-5.0_f32, 5.0_f32] {
                insert_object_with_movability(
                    &mut renderer, leg_mesh, mat_steel,
                    glam::Mat4::from_translation(glam::Vec3::new(cx + leg_dx, 9.0, cz)),
                    10.0, None,
                ).ok();
            }
            // Cross-beam
            let beam_upload = box_mesh([0.0; 3], [13.0, 1.0, 1.0]);
            let beam_mesh = renderer.scene_mut()
                .insert_actor(SceneActor::mesh(beam_upload.clone())).as_mesh().unwrap();
            picker.register_mesh(beam_mesh, &beam_upload);
            insert_object_with_movability(
                &mut renderer, beam_mesh, mat_orange,
                glam::Mat4::from_translation(glam::Vec3::new(cx, 18.0, cz)),
                8.0, None,
            ).ok();
            // Boom extending over the water side
            let boom_sign = if crane_i == 0 { -1.0_f32 } else { 1.0_f32 };
            let boom_upload = box_mesh([0.0; 3], [18.0, 0.8, 0.8]);
            let boom_mesh = renderer.scene_mut()
                .insert_actor(SceneActor::mesh(boom_upload.clone())).as_mesh().unwrap();
            picker.register_mesh(boom_mesh, &boom_upload);
            insert_object_with_movability(
                &mut renderer, boom_mesh, mat_steel,
                glam::Mat4::from_translation(glam::Vec3::new(cx + boom_sign * 10.0, 17.5, cz)),
                12.0, None,
            ).ok();
            // Warning beacon on boom tip
            let beacon_upload = sphere_mesh([0.0; 3], 0.45);
            let beacon_mesh = renderer.scene_mut()
                .insert_actor(SceneActor::mesh(beacon_upload.clone())).as_mesh().unwrap();
            picker.register_mesh(beacon_mesh, &beacon_upload);
            insert_object_with_movability(
                &mut renderer, beacon_mesh, mat_warning,
                glam::Mat4::from_translation(glam::Vec3::new(cx + boom_sign * 27.5, 17.5, cz)),
                0.6, None,
            ).ok();
            // Operator cab
            let cab_upload = box_mesh([0.0; 3], [3.5, 2.5, 3.0]);
            let cab_mesh = renderer.scene_mut()
                .insert_actor(SceneActor::mesh(cab_upload.clone())).as_mesh().unwrap();
            picker.register_mesh(cab_mesh, &cab_upload);
            insert_object_with_movability(
                &mut renderer, cab_mesh, mat_orange,
                glam::Mat4::from_translation(glam::Vec3::new(cx, 17.0, cz - 0.5)),
                3.0, None,
            ).ok();
        }

        // ── Lamp mast posts ────────────────────────────────────────────────
        // 7×3 grid of sodium masts spanning the expanded yard
        let mast_xs: &[f32] = &[-84.0, -56.0, -28.0, 0.0, 28.0, 56.0, 84.0];
        let mast_zs: &[f32] = &[-32.0, 4.0, 38.0];
        let mast_upload = box_mesh([0.0; 3], [0.35, 12.0, 0.35]);
        let mast_mesh = renderer.scene_mut()
            .insert_actor(SceneActor::mesh(mast_upload.clone())).as_mesh().unwrap();
        picker.register_mesh(mast_mesh, &mast_upload);
        let lamp_upload = box_mesh([0.0; 3], [1.2, 0.4, 1.2]);
        let lamp_mesh = renderer.scene_mut()
            .insert_actor(SceneActor::mesh(lamp_upload.clone())).as_mesh().unwrap();
        picker.register_mesh(lamp_mesh, &lamp_upload);
        for &mz in mast_zs {
            for &mx in mast_xs {
                insert_object_with_movability(
                    &mut renderer, mast_mesh, mat_steel,
                    glam::Mat4::from_translation(glam::Vec3::new(mx, 6.0, mz)),
                    7.0, None,
                ).ok();
                insert_object_with_movability(
                    &mut renderer, lamp_mesh, mat_lamp,
                    glam::Mat4::from_translation(glam::Vec3::new(mx, 12.4, mz)),
                    1.0, None,
                ).ok();
            }
        }

        // ── Shipping containers — ~900 stacked in shipyard bays ───────────
        {
            let crates_base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join("models/source");

            match load_scene_bytes_with_config(
                CRATES_FBX,
                "fbx",
                Some(crates_base.as_path()),
                LoadConfig::default()
                    .with_uv_flip(false)
                    .with_merge_meshes(true)
                    .with_import_scale(glam::Vec3::splat(1.0 / 200.0)),
            ) {
                Ok(scene) => {
                    match upload_sectioned_scene(&mut renderer, &scene) {
                        Ok((multi_mesh_id, section_mat_ids)) => {
                            let sm = scene.sectioned_mesh.as_ref().unwrap();

                            // Measure local-space AABB to derive step sizes
                            let mut bb_min = glam::Vec3::splat(f32::INFINITY);
                            let mut bb_max = glam::Vec3::splat(f32::NEG_INFINITY);
                            for v in &sm.vertices {
                                let p = glam::Vec3::from(v.position);
                                bb_min = bb_min.min(p);
                                bb_max = bb_max.max(p);
                            }
                            let local_center = (bb_min + bb_max) * 0.5;
                            let local_size   = bb_max - bb_min;
                            let radius = (local_size * 0.5).length().max(0.5);

                            eprintln!(
                                "[shipyard] container: {} sections {} verts  \
                                 size={:.2?} center={:.2?} r={radius:.2}",
                                sm.sections.len(), sm.vertices.len(), local_size, local_center
                            );

                            // Step sizes: tiny gap so edges don't z-fight
                            let step_x = local_size.x + 0.04;
                            let step_z = local_size.z + 0.06;
                            let step_y = local_size.y;

                            // ── Bay layout ────────────────────────────────────
                            // Total across all bays: ~900 containers.
                            // oz is the Z origin of the *nearest* row in each bay.
                            struct Bay { ox: f32, oz: f32, cols: u32, rows: u32, layers: u32 }
                            let bays: &[Bay] = &[
                                // Bay A — left main block (14×7×6 = 588)
                                Bay { ox: -95.0, oz: -26.0, cols: 14, rows: 7, layers: 6 },
                                // Bay B — right main block (12×6×5 = 360)
                                Bay { ox:  18.0, oz: -26.0, cols: 12, rows: 6, layers: 5 },
                                // Bay C — rear overflow (10×5×3 = 150) — shorter stacks
                                Bay { ox: -60.0, oz:  36.0, cols: 10, rows: 5, layers: 3 },
                                // Bay D — dock-side loose row (18×3×2 = 108)
                                //         freshly offloaded, only 2 high
                                Bay { ox: -95.0, oz: -38.0, cols: 18, rows: 3, layers: 2 },
                            ];

                            let mut count = 0u32;
                            for bay in bays {
                                for layer in 0..bay.layers {
                                    for row in 0..bay.rows {
                                        for col in 0..bay.cols {
                                            // World-space centre of this container slot
                                            let wx = bay.ox + col as f32 * step_x;
                                            // Bottom of layer 0 sits exactly on y = 0
                                            let wy = layer as f32 * step_y + local_size.y * 0.5;
                                            let wz = bay.oz + row as f32 * step_z;

                                            // Alternate 180° every other column for variety.
                                            // Rotation must happen around the mesh centre so
                                            // we use: T(world) * R * T(-local_center)
                                            let rot = if (col + layer) % 2 == 1 {
                                                glam::Mat4::from_rotation_y(std::f32::consts::PI)
                                            } else {
                                                glam::Mat4::IDENTITY
                                            };
                                            let placement =
                                                glam::Mat4::from_translation(
                                                    glam::Vec3::new(wx, wy, wz),
                                                ) * rot
                                                * glam::Mat4::from_translation(-local_center);
                                            // World centre is always exactly (wx, wy, wz)
                                            // regardless of rotation

                                            match renderer.scene_mut().insert_sectioned_object(
                                                multi_mesh_id,
                                                &section_mat_ids,
                                                placement,
                                                [wx, wy, wz, radius],
                                                None, // Static — containers never move
                                            ) {
                                                Ok(_) => count += 1,
                                                Err(e) => eprintln!(
                                                    "[shipyard] insert failed: {e:?}"
                                                ),
                                            }
                                        }
                                    }
                                }
                            }
                            eprintln!("[shipyard] {count} containers inserted");

                            // Register sections with the picker once
                            if let Some(section_ids) =
                                renderer.scene().sectioned_section_mesh_ids(multi_mesh_id)
                            {
                                let section_ids: Vec<_> = section_ids.to_vec();
                                for (section_mesh_id, sec) in
                                    section_ids.iter().zip(sm.sections.iter())
                                {
                                    picker.register_mesh(
                                        *section_mesh_id,
                                        &helio::MeshUpload {
                                            vertices: sm.vertices.clone(),
                                            indices: sec.indices.clone(),
                                        },
                                    );
                                }
                            }
                        }
                        Err(e) => eprintln!("[shipyard] upload_sectioned_scene ERR: {e}"),
                    }
                }
                Err(e) => eprintln!("[shipyard] Failed to load container FBX: {e}"),
            }
        }

        picker.rebuild_instances(renderer.scene());

        // ── Lights ────────────────────────────────────────────────────────
        // Sodium mast floodlights — warm amber, high mounted, wide range
        for &mz in mast_zs {
            for &mx in mast_xs {
                renderer.scene_mut().insert_actor(SceneActor::light(point_light(
                    [mx, 14.0, mz],
                    [1.0, 0.80, 0.40],
                    280.0,
                    52.0,
                )));
            }
        }

        // Crane work lights — cool white, tight cone, very bright
        renderer.scene_mut().insert_actor(SceneActor::light(point_light(
            [-90.0, 20.0, -38.0], [0.90, 0.96, 1.0], 400.0, 35.0,
        )));
        renderer.scene_mut().insert_actor(SceneActor::light(point_light(
            [ 90.0, 20.0, -38.0], [0.90, 0.96, 1.0], 400.0, 35.0,
        )));

        // Boom tip warning lights (red, matching emissive beacons)
        renderer.scene_mut().insert_actor(SceneActor::light(point_light(
            [-117.5, 17.5, -38.0], [1.0, 0.05, 0.02], 50.0, 7.0,
        )));
        renderer.scene_mut().insert_actor(SceneActor::light(point_light(
            [ 117.5, 17.5, -38.0], [1.0, 0.05, 0.02], 50.0, 7.0,
        )));

        // Ground-level fill lights between stack rows — sodium spill colour
        // These light up the lower sides of containers the masts can't reach.
        let fill_pts: &[(f32, f32, f32)] = &[
            // Bay A alleys
            (-80.0, 3.5, -12.0), (-52.0, 3.5, -12.0), (-24.0, 3.5, -12.0),
            // Bay B alleys
            ( 30.0, 3.5, -12.0), ( 58.0, 3.5, -12.0),
            // Row between bay A and bay D (dock-side alley)
            (-80.0, 3.5, -33.0), (-52.0, 3.5, -33.0), (-24.0, 3.5, -33.0),
            // Bay C alleys (rear)
            (-50.0, 3.5,  48.0), (-22.0, 3.5,  48.0),
        ];
        for &(fx, fy, fz) in fill_pts {
            renderer.scene_mut().insert_actor(SceneActor::light(point_light(
                [fx, fy, fz], [1.0, 0.76, 0.38], 90.0, 28.0,
            )));
        }

        // Harbour water sheen — deep teal reflections off the harbour side
        renderer.scene_mut().insert_actor(SceneActor::light(point_light(
            [120.0, 2.0, -20.0], [0.20, 0.55, 0.85], 60.0, 60.0,
        )));
        renderer.scene_mut().insert_actor(SceneActor::light(point_light(
            [130.0, 2.0,  10.0], [0.15, 0.45, 0.75], 50.0, 55.0,
        )));

        self.state = Some(AppState {
            window,
            surface,
            device,
            surface_format: format,
            renderer,
            last_frame: std::time::Instant::now(),
            // Start back and high for a wide overview of the full expanded yard
            cam_pos: glam::Vec3::new(0.0, 55.0, 150.0),
            cam_yaw: 0.0,
            cam_pitch: -0.38,
            keys: HashSet::new(),
            right_mouse_held: false,
            mouse_delta: (0.0, 0.0),
            cursor_pos: (640.0, 360.0),
            cam_speed: 18.0,
            editor: EditorState::new(),
            picker,
            grid_enabled: true,
            is_fullscreen: false,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = &mut self.state else { return };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(sz) if sz.width > 0 && sz.height > 0 => {
                state.surface.configure(
                    &state.device,
                    &wgpu::SurfaceConfiguration {
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        format: state.surface_format,
                        width: sz.width,
                        height: sz.height,
                        present_mode: wgpu::PresentMode::AutoVsync,
                        alpha_mode: wgpu::CompositeAlphaMode::Auto,
                        view_formats: vec![],
                        desired_maximum_frame_latency: 2,
                    },
                );
                state.renderer.set_render_size(sz.width, sz.height);
            }

            WindowEvent::CursorMoved { position, .. } => {
                state.cursor_pos = (position.x as f32, position.y as f32);
                if !state.right_mouse_held {
                    let (ray_o, ray_d) = state.build_ray();
                    state.editor.update_hover(ray_o, ray_d, state.renderer.scene());
                    if state.editor.is_dragging() {
                        state.editor.update_drag(ray_o, ray_d, state.renderer.scene_mut());
                    }
                }
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: ks,
                        ..
                    },
                ..
            } => {
                match ks {
                    ElementState::Pressed => {
                        state.keys.insert(code);
                        match code {
                            KeyCode::F11 => state.toggle_fullscreen(),
                            KeyCode::Enter | KeyCode::NumpadEnter
                                if state.keys.contains(&KeyCode::AltLeft)
                                    || state.keys.contains(&KeyCode::AltRight) =>
                            {
                                state.toggle_fullscreen();
                            }
                            KeyCode::Escape => {
                                if state.editor.selected().is_some() {
                                    state.editor.deselect();
                                } else {
                                    event_loop.exit();
                                }
                            }
                            KeyCode::Delete if !state.right_mouse_held => {
                                if state.editor.delete_selected(state.renderer.scene_mut()) {
                                    state.picker.rebuild_instances(state.renderer.scene());
                                }
                            }
                            KeyCode::KeyG if !state.right_mouse_held => {
                                state.editor.set_gizmo_mode(GizmoMode::Translate)
                            }
                            KeyCode::KeyR if !state.right_mouse_held => {
                                state.editor.set_gizmo_mode(GizmoMode::Rotate)
                            }
                            KeyCode::KeyS if !state.right_mouse_held => {
                                state.editor.set_gizmo_mode(GizmoMode::Scale)
                            }
                            KeyCode::KeyD
                                if !state.right_mouse_held
                                && (state.keys.contains(&KeyCode::ControlLeft)
                                    || state.keys.contains(&KeyCode::ControlRight)) =>
                            {
                                if let Some(_new_id) =
                                    state.editor.duplicate_selected(&mut state.renderer)
                                {
                                    state.picker.rebuild_instances(state.renderer.scene());
                                }
                            }
                            KeyCode::Tab => {
                                state.grid_enabled = !state.grid_enabled;
                                state.renderer.set_editor_mode(state.grid_enabled);
                            }
                            KeyCode::KeyL if !state.right_mouse_held => {
                                let pos = state.cam_pos.to_array();
                                state.renderer.scene_mut().insert_actor(
                                    SceneActor::light(point_light(
                                        pos, [0.2, 0.5, 1.0], 500.0, 150.0,
                                    )),
                                );
                            }
                            _ => {}
                        }
                    }
                    ElementState::Released => {
                        state.keys.remove(&code);
                    }
                }
            }

            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Right,
                ..
            } => {
                if !state.right_mouse_held {
                    let _ = state
                        .window
                        .set_cursor_grab(CursorGrabMode::Confined)
                        .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked));
                    state.window.set_cursor_visible(false);
                    state.right_mouse_held = true;
                }
            }

            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Right,
                ..
            } => {
                let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                state.window.set_cursor_visible(true);
                state.right_mouse_held = false;
            }

            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if !state.right_mouse_held {
                    let (ray_o, ray_d) = state.build_ray();
                    if !state.editor.try_start_drag(ray_o, ray_d, state.renderer.scene()) {
                        state.picker.rebuild_instances(state.renderer.scene());
                        if let Some(hit) =
                            state.picker.cast_ray(state.renderer.scene(), ray_o, ray_d)
                        {
                            state.editor.select(hit.actor_id);
                        } else {
                            state.editor.deselect();
                        }
                    }
                }
            }

            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => {
                state.editor.end_drag();
            }

            WindowEvent::MouseWheel { delta, .. } => {
                if let Some(state) = self.state.as_mut() {
                    let lines = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 20.0,
                    };
                    state.cam_speed = (state.cam_speed * 1.15_f32.powf(lines))
                        .clamp(0.5, 500.0);
                }
            }

            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                let dt = (now - state.last_frame).as_secs_f32();
                state.last_frame = now;
                state.update_camera(dt);
                state.render();
                state.window.request_redraw();
            }

            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _: &ActiveEventLoop,
        _: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        let Some(s) = &mut self.state else { return };
        if let DeviceEvent::MouseMotion { delta } = event {
            if s.right_mouse_held {
                s.mouse_delta.0 += delta.0 as f32;
                s.mouse_delta.1 += delta.1 as f32;
            }
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(s) = &self.state {
            s.window.request_redraw();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-frame update & render
// ─────────────────────────────────────────────────────────────────────────────

impl AppState {
    fn build_ray(&self) -> (glam::Vec3, glam::Vec3) {
        let sz     = self.window.inner_size();
        let width  = sz.width as f32;
        let height = sz.height as f32;
        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let fwd    = glam::Vec3::new(sy * cp, sp, -cy * cp);
        let aspect = width / height.max(1.0);
        let proj   = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.1, 800.0);
        let view   = glam::Mat4::look_at_rh(self.cam_pos, self.cam_pos + fwd, glam::Vec3::Y);
        let vp_inv = (proj * view).inverse();
        EditorState::ray_from_screen(self.cursor_pos.0, self.cursor_pos.1, width, height, vp_inv)
    }

    fn update_camera(&mut self, dt: f32) {
        const LOOK: f32 = 0.0025;

        if self.right_mouse_held {
            self.cam_yaw += self.mouse_delta.0 * LOOK;
            self.cam_pitch -= self.mouse_delta.1 * LOOK;
            self.cam_pitch = self
                .cam_pitch
                .clamp(-std::f32::consts::FRAC_PI_2 * 0.99, std::f32::consts::FRAC_PI_2 * 0.99);

            let (sy, cy) = self.cam_yaw.sin_cos();
            let fwd   = glam::Vec3::new(sy, 0.0, -cy);
            let right = glam::Vec3::new(cy, 0.0, sy);

            let mut vel = glam::Vec3::ZERO;
            if self.keys.contains(&KeyCode::KeyW) { vel += fwd; }
            if self.keys.contains(&KeyCode::KeyS) { vel -= fwd; }
            if self.keys.contains(&KeyCode::KeyD) { vel += right; }
            if self.keys.contains(&KeyCode::KeyA) { vel -= right; }
            if self.keys.contains(&KeyCode::Space)     { vel += glam::Vec3::Y; }
            if self.keys.contains(&KeyCode::ShiftLeft) { vel -= glam::Vec3::Y; }
            if vel.length_squared() > 0.0 {
                self.cam_pos += vel.normalize() * self.cam_speed * dt;
            }
        }
        self.mouse_delta = (0.0, 0.0);
    }

    fn toggle_fullscreen(&mut self) {
        use winit::window::Fullscreen;
        if self.is_fullscreen {
            #[cfg(target_os = "windows")]
            unsafe { self.renderer.exit_exclusive_fullscreen(&self.surface); }
            self.window.set_fullscreen(None);
            self.is_fullscreen = false;
        } else {
            let monitor = self.window.current_monitor();
            self.window.set_fullscreen(Some(Fullscreen::Borderless(monitor)));
            #[cfg(target_os = "windows")]
            {
                use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
                if let Ok(handle) = self.window.window_handle() {
                    if let RawWindowHandle::Win32(h) = handle.as_raw() {
                        let hwnd = h.hwnd.get() as *mut std::ffi::c_void;
                        unsafe {
                            self.renderer.request_exclusive_fullscreen(&self.surface, hwnd);
                        }
                    }
                }
            }
            self.is_fullscreen = true;
        }
    }

    fn render(&mut self) {
        let sz = self.window.inner_size();
        let aspect = sz.width as f32 / sz.height.max(1) as f32;

        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let fwd = glam::Vec3::new(sy * cp, sp, -cy * cp);

        let camera = Camera::perspective_look_at(
            self.cam_pos,
            self.cam_pos + fwd,
            glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            aspect,
            0.1,
            800.0,
        );

        self.renderer.debug_clear();
        self.editor.draw_gizmos(&mut self.renderer);

        if !self.right_mouse_held {
            let hit_point = self.cam_pos + fwd * 0.1;
            let r = 0.004;
            self.renderer.debug_line(
                (hit_point - glam::Vec3::X * r).to_array(),
                (hit_point + glam::Vec3::X * r).to_array(),
                [1.0, 1.0, 1.0, 0.7],
            );
            self.renderer.debug_line(
                (hit_point - glam::Vec3::Y * r).to_array(),
                (hit_point + glam::Vec3::Y * r).to_array(),
                [1.0, 1.0, 1.0, 0.7],
            );
        }

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(e) => { log::warn!("surface: {:?}", e); return; }
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        if let Err(e) = self.renderer.render(&camera, &view) {
            log::error!("render: {:?}", e);
        }
        output.present();
    }
}
