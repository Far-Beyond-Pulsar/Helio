//! Outdoor rocks demo — dozens of photorealistic rock meshes scattered across a
//! terrain, with animated billboard markers floating above them, and the ship
//! FBX parked nearby.
//!
//! Three rock types are loaded from the `3d/` directory:
//!   - Chiseled Rock (rafue)
//!   - Granite Rock (pjtsT)
//!   - Granite Rock (pkeeM)
//!
//! Controls:
//!   WASD        — move forward/left/back/right
//!   Space/Shift — move up/down
//!   Q/E         — rotate sun (time of day)
//!   Mouse drag  — look around (click to grab cursor)
//!   Escape      — release cursor / exit
//!   F3          — toggle virtual geometry triangle debug view

mod v3_demo_common;

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use glam::{EulerRot, Mat4, Quat, Vec3};
use helio::{
    required_wgpu_features, required_wgpu_limits, BillboardInstance, Camera, LightId, Renderer,
    RendererConfig, VirtualMeshUpload, VirtualObjectDescriptor,
};
use helio_asset_compat::{
    load_scene_bytes_with_config, load_scene_file_with_config, upload_scene_materials, LoadConfig,
};
use v3_demo_common::{cube_mesh, directional_light, make_material, point_light};
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

// ── Embedded ship asset ───────────────────────────────────────────────────────
const SHIP_BYTES: &[u8] = include_bytes!("../../test.fbx");

// ── Rock scatter parameters ───────────────────────────────────────────────────
const ROCK_COUNT_PER_TYPE: usize = 30;
const FIELD_RADIUS: f32 = 80.0;
const BILLBOARD_EVERY_N: usize = 4; // place a billboard above every Nth rock

fn base_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

/// Simple LCG random number generator (identical to ship_flight's).
fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*seed >> 33) as f32) / (u32::MAX as f32)
}
fn rand_s(seed: &mut u64) -> f32 {
    lcg(seed) * 2.0 - 1.0
}

// ── Marker color palette (RGBA, linear) ──────────────────────────────────────
const MARKER_COLORS: [[f32; 4]; 6] = [
    [1.0, 0.8, 0.1, 0.85], // amber
    [0.2, 0.8, 1.0, 0.85], // cyan
    [1.0, 0.3, 0.3, 0.85], // red
    [0.4, 1.0, 0.4, 0.85], // green
    [0.9, 0.4, 1.0, 0.85], // violet
    [1.0, 1.0, 1.0, 0.85], // white
];

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
    last_frame: Instant,
    start_time: Instant,

    // Billboards — positions stay fixed; alpha/scale pulse in update()
    billboard_positions: Vec<(Vec3, usize)>, // (world_pos, color_index)

    cam_pos: Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

    sun_light_id: LightId,
    sun_angle: f32,

    // ── VG debug mode ─────────────────────────────────────────────
    /// True when F3 VG triangle debug is active (debug_mode 20).
    vg_debug: bool,

    // ── CPU-side profiling ───────────────────────────────────────────────
    frame_count: u64,
    prof_frame_total: f64, // ms
    prof_update: f64,      // ms: sun, billboards, camera update
    prof_render: f64,      // ms: renderer.render() call
    prof_present: f64,     // ms: get_current_texture + present
    prof_frame_min: f64,
    prof_frame_max: f64,
}

impl App {
    fn new() -> Self {
        Self { state: None }
    }
}

// ── Camera free-fly helper ────────────────────────────────────────────────────
impl AppState {
    fn update_camera(&mut self, dt: f32) -> Vec3 {
        const SPEED: f32 = 28.0;
        const LOOK_SENS: f32 = 0.002;

        self.cam_yaw += self.mouse_delta.0 * LOOK_SENS;
        self.cam_pitch = (self.cam_pitch - self.mouse_delta.1 * LOOK_SENS).clamp(-1.48, 1.48);
        self.mouse_delta = (0.0, 0.0);

        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let forward = Vec3::new(sy * cp, sp, -cy * cp);
        let right = Vec3::new(cy, 0.0, sy);

        if self.keys.contains(&KeyCode::KeyW) {
            self.cam_pos += forward * SPEED * dt;
        }
        if self.keys.contains(&KeyCode::KeyS) {
            self.cam_pos -= forward * SPEED * dt;
        }
        if self.keys.contains(&KeyCode::KeyA) {
            self.cam_pos -= right * SPEED * dt;
        }
        if self.keys.contains(&KeyCode::KeyD) {
            self.cam_pos += right * SPEED * dt;
        }
        if self.keys.contains(&KeyCode::Space) {
            self.cam_pos.y += SPEED * dt;
        }
        if self.keys.contains(&KeyCode::ShiftLeft) {
            self.cam_pos.y -= SPEED * dt;
        }

        forward
    }
}

// ── ApplicationHandler ────────────────────────────────────────────────────────
impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        // ── Window + wgpu setup ────────────────────────────────────────────
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Helio — Outdoor Rocks")
                        .with_inner_size(winit::dpi::PhysicalSize::new(1280, 720)),
                )
                .expect("failed to create window"),
        );

        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .expect("failed to create surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("no adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: required_wgpu_features(adapter.features()),
            required_limits: required_wgpu_limits(adapter.limits()),
            ..Default::default()
        }))
        .expect("no device");

        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        surface.configure(
            &device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: surface_format,
                width: size.width,
                height: size.height,
                present_mode: wgpu::PresentMode::AutoNoVsync,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            },
        );

        let mut renderer = Renderer::new(
            device.clone(),
            queue,
            RendererConfig::new(size.width, size.height, surface_format),
        );
        renderer.set_clear_color([0.34, 0.48, 0.72, 1.0]); // overcast sky blue
        renderer.set_ambient([0.38, 0.44, 0.50], 1.3);

        // ── Sun light ─────────────────────────────────────────────────────
        let sun_angle: f32 = 0.62; // radians above horizon
        let sun_dir = Vec3::new(-sun_angle.cos(), -sun_angle.sin(), -0.6).normalize();
        let sun_light_id = renderer.scene_mut().insert_actor(helio::SceneActor::light(directional_light(
            sun_dir.to_array(),
            [1.0, 0.93, 0.75],
            4.2,
        ))).as_light().unwrap();

        // Small fill lights to break up flatness
        let _ = renderer.scene_mut().insert_actor(helio::SceneActor::light(point_light([0.0, 8.0, 0.0], [0.6, 0.7, 1.0], 12.0, 50.0)));
        let _ = renderer.scene_mut().insert_actor(helio::SceneActor::light(point_light([60.0, 4.0, -40.0], [1.0, 0.85, 0.5], 8.0, 30.0)));

        // ── Ground plane ──────────────────────────────────────────────────
        let ground_mat = renderer.scene_mut().insert_material(make_material(
            [0.28, 0.23, 0.18, 1.0],
            0.92,
            0.0,
            [0.0, 0.0, 0.0],
            0.0,
        ));
        let ground_mesh = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(v3_demo_common::plane_mesh([0.0, 0.0, 0.0], 250.0))).as_mesh().unwrap();        let _ = v3_demo_common::insert_object(
            &mut renderer,
            ground_mesh,
            ground_mat,
            Mat4::IDENTITY,
            250.0,
        );

        // ── Load rock FBX files as virtual geometry (meshlets) ────────────
        // Each rock type is meshletised once at upload time; the GPU cull
        // compute then handles per-meshlet frustum + backface culling O(1) CPU.
        let asset_dir = base_dir().join("3d");
        let rock_paths = [
            asset_dir.join("Chiseled_Rock_rafue_Raw.fbx"),
            asset_dir.join("Granite_Rock_pjtsT_Raw.fbx"),
            asset_dir.join("Granite_Rock_pkeeM_Raw.fbx"),
        ];

        // Fallback cube material/mesh for any type that failed to load
        let fallback_mat = renderer.scene_mut().insert_material(make_material(
            [0.35, 0.30, 0.25, 1.0],
            0.85,
            0.0,
            [0.0, 0.0, 0.0],
            0.0,
        ));
        let fallback_mesh = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(cube_mesh([0.0, 0.0, 0.0], 0.5))).as_mesh().unwrap();
        // rock_vg[type] = Some(vec of (VirtualMeshId, material_slot_u32))
        let rock_vg: Vec<Option<Vec<(helio::VirtualMeshId, u32)>>> = rock_paths
            .iter()
            .map(|path| {
                let scene = match load_scene_file_with_config(path, LoadConfig::default()) {
                    Ok(s) => s,
                    Err(e) => {
                        log::warn!("Could not load rock '{}': {e}", path.display());
                        return None;
                    }
                };
                let mat_ids = match upload_scene_materials(&mut renderer, &scene) {
                    Ok(ids) => ids,
                    Err(e) => {
                        log::warn!(
                            "Could not upload rock materials for '{}': {e}",
                            path.display()
                        );
                        return None;
                    }
                };
                let entries: Vec<(helio::VirtualMeshId, u32)> = scene
                    .meshes
                    .iter()
                    .map(|mesh| {
                        let vm_id = renderer.scene_mut().insert_actor(helio::SceneActor::virtual_mesh(VirtualMeshUpload {
                            vertices: mesh.vertices.clone(),
                            indices: mesh.indices.clone(),
                        })).as_virtual_mesh().unwrap();
                        let mat_id = mesh
                            .material_index
                            .and_then(|idx| mat_ids.get(idx))
                            .or_else(|| mat_ids.first())
                            .copied()
                            .unwrap_or(fallback_mat);
                        (vm_id, mat_id.slot())
                    })
                    .collect();
                Some(entries)
            })
            .collect();

        // ── Scatter rocks ─────────────────────────────────────────────────
        let mut seed: u64 = 0xDEAD_BEEF_CAFE_1234;
        let mut billboard_positions: Vec<(Vec3, usize)> = Vec::new();
        let mut global_rock_idx: usize = 0;

        for rock_type in 0..3usize {
            let vg_entries = rock_vg[rock_type].as_deref();

            for _ in 0..ROCK_COUNT_PER_TYPE {
                // Random position in a disc
                let angle = lcg(&mut seed) * std::f32::consts::TAU;
                let dist = FIELD_RADIUS * lcg(&mut seed).sqrt();
                let pos = Vec3::new(angle.cos() * dist, 0.0, angle.sin() * dist);

                // Random scale — small floor rocks, no overlapping giants
                let base_scale = 0.08 + lcg(&mut seed) * 0.22; // 0.08..0.30
                let scale = Vec3::new(
                    base_scale * (0.8 + lcg(&mut seed) * 0.4),
                    base_scale * (0.5 + lcg(&mut seed) * 0.5),
                    base_scale * (0.8 + lcg(&mut seed) * 0.4),
                );
                let rot = Quat::from_euler(
                    EulerRot::XYZ,
                    0.0,
                    rand_s(&mut seed) * std::f32::consts::PI,
                    0.0,
                );
                let transform = Mat4::from_scale_rotation_translation(scale, rot, pos);

                // World-space bounding sphere used for instance-level culling.
                let center = pos + Vec3::Y * scale.y * 0.5;
                let bounds_radius = base_scale * 1.2;

                match vg_entries {
                    None => {
                        let _ = v3_demo_common::insert_object(
                            &mut renderer,
                            fallback_mesh,
                            fallback_mat,
                            transform,
                            base_scale,
                        );
                    }
                    Some(entries) => {
                        for &(vm_id, mat_slot) in entries {
                            let _ = renderer.scene_mut().insert_actor(helio::SceneActor::virtual_object(VirtualObjectDescriptor {
                                virtual_mesh: vm_id,
                                material_id: mat_slot,
                                transform,
                                bounds: [center.x, center.y, center.z, bounds_radius],
                                flags: 0,
                                groups: helio::GroupMask::NONE,
                            }));
                        }
                    }
                }

                // Every Nth rock gets a billboard marker
                if global_rock_idx % BILLBOARD_EVERY_N == 0 {
                    let color_idx = (global_rock_idx / BILLBOARD_EVERY_N) % MARKER_COLORS.len();
                    let billboard_pos = pos + Vec3::Y * (scale.y + 1.5);
                    billboard_positions.push((billboard_pos, color_idx));
                }
                global_rock_idx += 1;
            }
        }

        // ── Ship (parked nearby) ───────────────────────────────────────────
        let ship_pos = Vec3::new(18.0, 0.0, -12.0);
        let load_result = load_scene_bytes_with_config(
            SHIP_BYTES,
            "fbx",
            Some(base_dir().as_path()),
            LoadConfig::default().with_uv_flip(false),
        );
        match load_result {
            Ok(scene) => {
                // Upload meshes + materials in a single traversal.
                match upload_scene_materials(&mut renderer, &scene) {
                    Ok(mat_ids) => {
                        let ship_transform =
                            Mat4::from_rotation_translation(Quat::from_rotation_y(0.4), ship_pos);
                        for mesh in &scene.meshes {
                            let radius = mesh
                                .vertices
                                .iter()
                                .map(|v| Vec3::from_array(v.position).length())
                                .fold(0.5_f32, f32::max);
                            let mesh_id = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(helio::MeshUpload {
                                vertices: mesh.vertices.clone(),
                                indices: mesh.indices.clone(),
                            })).as_mesh().unwrap();
                            let mat = mesh
                                .material_index
                                .and_then(|idx| mat_ids.get(idx))
                                .or_else(|| mat_ids.first())
                                .copied()
                                .unwrap_or(fallback_mat);
                            let _ = v3_demo_common::insert_object(
                                &mut renderer,
                                mesh_id,
                                mat,
                                ship_transform,
                                radius,
                            );
                        }
                    }
                    Err(e) => log::warn!("Could not upload ship materials: {e}"),
                }
            }
            Err(e) => {
                log::warn!("Could not load ship FBX: {e} — placing fallback cube");
                let ship_mesh = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(cube_mesh([0.0, 0.0, 0.0], 1.5))).as_mesh().unwrap();                let ship_mat = renderer.scene_mut().insert_material(make_material(
                    [0.55, 0.70, 0.90, 1.0],
                    0.25,
                    0.75,
                    [0.0, 0.0, 0.0],
                    0.0,
                ));
                let transform = Mat4::from_translation(ship_pos);
                let _ = v3_demo_common::insert_object(
                    &mut renderer,
                    ship_mesh,
                    ship_mat,
                    transform,
                    1.5,
                );
            }
        }

        // ── Initial billboard upload ──────────────────────────────────────
        let bill_instances: Vec<BillboardInstance> = billboard_positions
            .iter()
            .map(|&(pos, color_idx)| BillboardInstance {
                world_pos: [pos.x, pos.y, pos.z, 0.0],
                scale_flags: [1.2, 1.2, 0.0, 0.0],
                color: MARKER_COLORS[color_idx],
            })
            .collect();
        renderer.set_billboard_instances(&bill_instances);

        self.state = Some(AppState {
            window,
            surface,
            device,
            surface_format,
            renderer,
            last_frame: Instant::now(),
            start_time: Instant::now(),
            billboard_positions,
            cam_pos: Vec3::new(0.0, 12.0, 50.0),
            cam_yaw: 0.0,
            cam_pitch: -0.18,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            sun_light_id,
            sun_angle,
            vg_debug: false,
            frame_count: 0,
            prof_frame_total: 0.0,
            prof_update: 0.0,
            prof_render: 0.0,
            prof_present: 0.0,
            prof_frame_min: f64::MAX,
            prof_frame_max: 0.0,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                ..
            } => {
                if state.cursor_grabbed {
                    state.cursor_grabbed = false;
                    state.window.set_cursor_visible(true);
                    let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                } else {
                    event_loop.exit();
                }
            }

            WindowEvent::Resized(size) => {
                state.surface.configure(
                    &state.device,
                    &wgpu::SurfaceConfiguration {
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        format: state.surface_format,
                        width: size.width,
                        height: size.height,
                        present_mode: wgpu::PresentMode::AutoNoVsync,
                        alpha_mode: wgpu::CompositeAlphaMode::Opaque,
                        view_formats: vec![],
                        desired_maximum_frame_latency: 2,
                    },
                );
                state.renderer.set_render_size(size.width, size.height);
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::F3),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                state.vg_debug = !state.vg_debug;
                state
                    .renderer
                    .set_debug_mode(if state.vg_debug { 20 } else { 0 });
                state.window.set_title(if state.vg_debug {
                    "Helio — Outdoor Rocks  [VG TRIANGLE DEBUG — F3 to exit]"
                } else {
                    "Helio — Outdoor Rocks"
                });
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => match key_state {
                ElementState::Pressed => {
                    state.keys.insert(code);
                }
                ElementState::Released => {
                    state.keys.remove(&code);
                }
            },

            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if !state.cursor_grabbed {
                    let ok = state
                        .window
                        .set_cursor_grab(CursorGrabMode::Confined)
                        .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked))
                        .is_ok();
                    if ok {
                        state.cursor_grabbed = true;
                        state.window.set_cursor_visible(false);
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                let frame_start = Instant::now();
                let now = frame_start;
                let dt = now.duration_since(state.last_frame).as_secs_f32().min(0.05);
                let t = now.duration_since(state.start_time).as_secs_f32();
                state.last_frame = now;

                // ── Sun rotation ──────────────────────────────────────────
                let t_update_start = Instant::now();
                const SUN_SPEED: f32 = 0.6;
                if state.keys.contains(&KeyCode::KeyQ) {
                    state.sun_angle += SUN_SPEED * dt;
                }
                if state.keys.contains(&KeyCode::KeyE) {
                    state.sun_angle -= SUN_SPEED * dt;
                }
                state.sun_angle = state.sun_angle.clamp(-1.48, 1.48);
                let sun_dir = Vec3::new(
                    -state.sun_angle.cos(),
                    -state.sun_angle.sin().abs() - 0.3,
                    -0.6,
                )
                .normalize();
                let _ = state.renderer.scene_mut().update_light(
                    state.sun_light_id,
                    directional_light(sun_dir.to_array(), [1.0, 0.93, 0.75], 4.2),
                );

                // ── Animate billboards (pulse alpha + gentle bob) ─────────
                let bill_instances: Vec<BillboardInstance> = state
                    .billboard_positions
                    .iter()
                    .enumerate()
                    .map(|(i, &(base_pos, color_idx))| {
                        let phase = t * 1.4 + i as f32 * 0.43;
                        let bob = (phase * 0.9).sin() * 0.35;
                        let alpha = 0.55 + 0.35 * (phase * 1.1).sin().abs();
                        let mut color = MARKER_COLORS[color_idx];
                        color[3] = alpha;
                        BillboardInstance {
                            world_pos: [base_pos.x, base_pos.y + bob, base_pos.z, 0.0],
                            scale_flags: [
                                1.1 + 0.15 * (phase * 0.7).cos(),
                                1.1 + 0.15 * (phase * 0.7).cos(),
                                0.0,
                                0.0,
                            ],
                            color,
                        }
                    })
                    .collect();
                state.renderer.set_billboard_instances(&bill_instances);

                // ── Camera ────────────────────────────────────────────────
                let forward = state.update_camera(dt);
                let size = state.window.inner_size();
                let camera = Camera::perspective_look_at(
                    state.cam_pos,
                    state.cam_pos + forward,
                    Vec3::Y,
                    std::f32::consts::FRAC_PI_4,
                    size.width as f32 / size.height.max(1) as f32,
                    0.15,
                    2000.0,
                );
                let t_update_ms = t_update_start.elapsed().as_secs_f64() * 1000.0;

                // ── Render ────────────────────────────────────────────────
                let t_render_start = Instant::now();
                let output = match state.surface.get_current_texture() {
                    Ok(t) => t,
                    Err(e) => {
                        log::warn!("surface error: {e:?}");
                        return;
                    }
                };
                let view = output
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                if let Err(e) = state.renderer.render(&camera, &view) {
                    log::error!("render error: {e:?}");
                }
                let t_render_ms = t_render_start.elapsed().as_secs_f64() * 1000.0;

                let t_present_start = Instant::now();
                output.present();
                let t_present_ms = t_present_start.elapsed().as_secs_f64() * 1000.0;

                // ── Profiling accumulate + print every 100 frames ─────────
                let frame_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
                state.prof_frame_total += frame_ms;
                state.prof_update += t_update_ms;
                state.prof_render += t_render_ms;
                state.prof_present += t_present_ms;
                if frame_ms < state.prof_frame_min {
                    state.prof_frame_min = frame_ms;
                }
                if frame_ms > state.prof_frame_max {
                    state.prof_frame_max = frame_ms;
                }
                state.frame_count += 1;

                const REPORT_EVERY: u64 = 100;
                if state.frame_count % REPORT_EVERY == 0 {
                    let n = REPORT_EVERY as f64;
                    let avg = state.prof_frame_total / n;
                    let fps = 1000.0 / avg;
                    let upd = state.prof_update / n;
                    let rnd = state.prof_render / n;
                    let pres = state.prof_present / n;
                    let other = avg - upd - rnd - pres;
                    eprintln!(
                        "[PROF #{:>6}] avg {:.2}ms ({:.0} fps) | min {:.2}ms max {:.2}ms",
                        state.frame_count, avg, fps, state.prof_frame_min, state.prof_frame_max
                    );
                    eprintln!(
                        "             update {:.2}ms  render {:.2}ms  present {:.2}ms  other {:.2}ms",
                        upd, rnd, pres, other
                    );
                    state.prof_frame_total = 0.0;
                    state.prof_update = 0.0;
                    state.prof_render = 0.0;
                    state.prof_present = 0.0;
                    state.prof_frame_min = f64::MAX;
                    state.prof_frame_max = 0.0;
                }
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        let Some(state) = &mut self.state else { return };
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if state.cursor_grabbed {
                state.mouse_delta.0 += dx as f32;
                state.mouse_delta.1 += dy as f32;
            }
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

fn main() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();
    let event_loop = EventLoop::new().expect("failed to create event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("event loop error");
}



