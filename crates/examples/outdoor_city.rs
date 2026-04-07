//! Outdoor city example – high complexity
//!
//! A dense downtown city block at dusk: 21 buildings of varying heights
//! arranged across two city blocks, 10 sodium streetlamps lining the main
//! avenue, 4 neon signs on landmark buildings, sidewalk strips, a central
//! plaza with a fountain base, and a controllable sun casting long shadows.
//!
//! Controls:
//!   WASD        — move forward/left/back/right
//!   Space/Shift — move up/down
//!   Q/E         — rotate sun (time of day)
//!   Mouse drag  — look around (click to grab cursor)
//!   Escape      — release cursor / exit

mod v3_demo_common;
use helio::{
    required_wgpu_features, required_wgpu_limits, Camera, LightId, MeshId, Renderer, RendererConfig,
};
use v3_demo_common::{
    box_mesh, directional_light, make_material, plane_mesh, point_light, spot_light,
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

// ── Scene data ────────────────────────────────────────────────────────────────

/// Buildings: (center_x, center_z, half_w, half_d, half_h)
/// half_h also serves as Y center (so the base sits on the ground plane).
const BUILDINGS: &[(f32, f32, f32, f32, f32)] = &[
    // West city block
    (-16.0, -20.0, 5.0, 4.0, 13.0),
    (-10.0, -24.0, 3.5, 3.0, 9.0),
    (-20.0, -5.0, 4.5, 7.0, 6.0),
    (-11.0, 7.0, 3.0, 4.0, 15.0),
    (-19.0, 17.0, 4.0, 3.5, 5.0),
    (-7.0, 22.0, 2.5, 3.0, 8.0),
    // East city block
    (16.0, -20.0, 4.5, 4.5, 18.0), // tallest tower
    (10.0, -24.0, 3.0, 3.5, 8.0),
    (20.0, -5.0, 4.0, 6.5, 10.0),
    (11.0, 7.0, 3.5, 4.0, 12.0),
    (19.0, 17.0, 4.5, 4.0, 5.0),
    (8.0, 22.0, 2.5, 3.5, 7.0),
    // Background skyline towers
    (-7.0, -33.0, 2.5, 2.5, 24.0),
    (0.0, -30.0, 5.5, 4.5, 8.0),
    (7.0, -33.0, 2.5, 2.5, 21.0),
    // Foreground low shops
    (-6.0, 31.0, 3.5, 2.5, 4.0),
    (0.0, 29.0, 4.5, 3.0, 3.0),
    (6.0, 31.0, 3.0, 3.0, 5.5),
    // Central plaza features
    (-2.5, -7.0, 1.2, 1.2, 2.5), // kiosk A
    (2.5, -7.0, 1.2, 1.2, 2.5),  // kiosk B
    (0.0, 0.0, 1.8, 1.8, 0.45),  // fountain plinth
];

/// Streetlamps: (x, z)
const LAMPS: &[(f32, f32)] = &[
    (-4.5, -22.0),
    (4.5, -22.0),
    (-4.5, -12.0),
    (4.5, -12.0),
    (-4.5, -2.0),
    (4.5, -2.0),
    (-4.5, 8.0),
    (4.5, 8.0),
    (-4.5, 18.0),
    (4.5, 18.0),
];

/// Neon signs: (x, y, z, r, g, b)
const NEONS: &[(f32, f32, f32, f32, f32, f32)] = &[
    (16.0, 14.5, -19.5, 1.0, 0.05, 0.85), // magenta on east tower
    (-7.0, 20.0, -32.5, 0.05, 0.85, 1.0), // cyan on bg tower
    (-11.0, 12.0, 6.5, 0.1, 1.0, 0.2),    // green on west tower
    (7.0, 17.0, -32.5, 1.0, 0.5, 0.0),    // amber on east bg tower
];

// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("run");
}

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

    _ground: MeshId,
    _buildings: Vec<MeshId>,
    _lamp_poles: Vec<MeshId>,
    _sidewalks: Vec<MeshId>,
    _road_center: MeshId,

    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),
    sun_angle: f32,

    // Scene state
    sun_light_id: LightId,
    lamp_light_ids: Vec<LightId>,
    neon_light_ids: Vec<LightId>,
}

impl App {
    fn new() -> Self {
        Self { state: None }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Helio – Outdoor City")
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
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Device"),
            required_features: required_wgpu_features(adapter.features()),
            required_limits: required_wgpu_limits(adapter.limits()),
            ..Default::default()
        }))
        .expect("device");
        device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| {
            panic!("[GPU UNCAPTURED ERROR] {:?}", e);
        }));
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let caps = surface.get_capabilities(&adapter);
        let format = caps
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
                format,
                width: size.width,
                height: size.height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            },
        );

        let mut renderer = Renderer::new(
            device.clone(),
            queue.clone(),
            RendererConfig::new(size.width, size.height, format),
        );

        let mat = renderer.scene_mut().insert_material(make_material(
            [0.75, 0.75, 0.75, 1.0],
            0.8,
            0.0,
            [0.0, 0.0, 0.0],
            0.0,
        ));

        let _ground = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(plane_mesh([0.0, 0.0, 0.0], 40.0)))
            .as_mesh()
            .unwrap();
        let _ =
            v3_demo_common::insert_object(&mut renderer, _ground, mat, glam::Mat4::IDENTITY, 40.0);

        let _road_center = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(box_mesh(
                [0.0, 0.0, 0.0],
                [4.0, 0.01, 32.0],
            )))
            .as_mesh()
            .unwrap();
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            _road_center,
            mat,
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.01, 0.0)),
            32.0,
        );

        let _sidewalks: Vec<MeshId> = vec![
            renderer
                .scene_mut()
                .insert_actor(helio::SceneActor::mesh(box_mesh(
                    [0.0, 0.0, 0.0],
                    [0.35, 0.04, 32.0],
                )))
                .as_mesh()
                .unwrap(),
            renderer
                .scene_mut()
                .insert_actor(helio::SceneActor::mesh(box_mesh(
                    [0.0, 0.0, 0.0],
                    [0.35, 0.04, 32.0],
                )))
                .as_mesh()
                .unwrap(),
            renderer
                .scene_mut()
                .insert_actor(helio::SceneActor::mesh(box_mesh(
                    [0.0, 0.0, 0.0],
                    [32.0, 0.04, 0.35],
                )))
                .as_mesh()
                .unwrap(),
            renderer
                .scene_mut()
                .insert_actor(helio::SceneActor::mesh(box_mesh(
                    [0.0, 0.0, 0.0],
                    [32.0, 0.04, 0.35],
                )))
                .as_mesh()
                .unwrap(),
        ];
        for (&m, t) in _sidewalks.iter().zip(
            [
                glam::Mat4::from_translation(glam::Vec3::new(-4.2, 0.04, 0.0)),
                glam::Mat4::from_translation(glam::Vec3::new(4.2, 0.04, 0.0)),
                glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.04, -32.0)),
                glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.04, 32.0)),
            ]
            .iter(),
        ) {
            let _ = v3_demo_common::insert_object(&mut renderer, m, mat, *t, 32.0);
        }

        let _buildings: Vec<MeshId> = BUILDINGS
            .iter()
            .map(|&(_, _, hw, hd, hh)| {
                renderer
                    .scene_mut()
                    .insert_actor(helio::SceneActor::mesh(box_mesh(
                        [0.0, 0.0, 0.0],
                        [hw, hh, hd],
                    )))
                    .as_mesh()
                    .unwrap()
            })
            .collect();
        for (&m, &(cx, cz, _hw, _hd, hh)) in _buildings.iter().zip(BUILDINGS.iter()) {
            let _ = v3_demo_common::insert_object(
                &mut renderer,
                m,
                mat,
                glam::Mat4::from_translation(glam::Vec3::new(cx, hh, cz)),
                15.0,
            );
        }

        let _lamp_poles: Vec<MeshId> = LAMPS
            .iter()
            .map(|&(_x, _z)| {
                renderer
                    .scene_mut()
                    .insert_actor(helio::SceneActor::mesh(box_mesh(
                        [0.0, 0.0, 0.0],
                        [0.08, 2.75, 0.08],
                    )))
                    .as_mesh()
                    .unwrap()
            })
            .collect();
        for (&m, &(x, z)) in _lamp_poles.iter().zip(LAMPS.iter()) {
            let _ = v3_demo_common::insert_object(
                &mut renderer,
                m,
                mat,
                glam::Mat4::from_translation(glam::Vec3::new(x, 2.75, z)),
                3.0,
            );
        }

        let sun_light_id = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::light(directional_light(
                [-0.35, -0.38, -0.45],
                [1.0, 0.9, 0.7],
                0.005,
            )))
            .as_light()
            .unwrap();
        let mut lamp_light_ids = Vec::new();
        for &(x, z) in LAMPS {
            let p = [x, 5.55, z];
            lamp_light_ids.push(
                renderer
                    .scene_mut()
                    .insert_actor(helio::SceneActor::light(spot_light(
                        p,
                        [0.0, -1.0, 0.0],
                        [1.0, 0.72, 0.30],
                        0.0,
                        14.0,
                        0.96,
                        1.22,
                    )))
                    .as_light()
                    .unwrap(),
            );
        }
        let mut neon_light_ids = Vec::new();
        for &(x, y, z, r, g, b) in NEONS {
            let p = [x, y, z];
            neon_light_ids.push(
                renderer
                    .scene_mut()
                    .insert_actor(helio::SceneActor::light(point_light(
                        p,
                        [r, g, b],
                        3.0,
                        12.0,
                    )))
                    .as_light()
                    .unwrap(),
            );
        }
        renderer.set_ambient([0.5, 0.5, 0.55], 0.06);
        renderer.set_clear_color([0.55, 0.65, 0.9, 1.0]);

        self.state = Some(AppState {
            window,
            surface,
            device,
            surface_format: format,
            renderer,
            last_frame: std::time::Instant::now(),
            _ground,
            _buildings,
            _lamp_poles,
            _sidewalks,
            _road_center,
            cam_pos: glam::Vec3::new(0.0, 5.0, 30.0),
            cam_yaw: std::f32::consts::PI,
            cam_pitch: -0.1,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            sun_angle: 0.38,
            sun_light_id,
            lamp_light_ids,
            neon_light_ids,
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
                    let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                    state.window.set_cursor_visible(true);
                } else {
                    event_loop.exit();
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ks,
                        physical_key: PhysicalKey::Code(key),
                        ..
                    },
                ..
            } => match ks {
                ElementState::Pressed => {
                    state.keys.insert(key);
                }
                ElementState::Released => {
                    state.keys.remove(&key);
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
                        state.window.set_cursor_visible(false);
                        state.cursor_grabbed = true;
                    }
                }
            }
            WindowEvent::Resized(s) if s.width > 0 && s.height > 0 => {
                state.surface.configure(
                    &state.device,
                    &wgpu::SurfaceConfiguration {
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        format: state.surface_format,
                        width: s.width,
                        height: s.height,
                        present_mode: wgpu::PresentMode::Fifo,
                        alpha_mode: wgpu::CompositeAlphaMode::Auto,
                        view_formats: vec![],
                        desired_maximum_frame_latency: 2,
                    },
                );
                state.renderer.set_render_size(s.width, s.height);
            }
            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                let dt = (now - state.last_frame).as_secs_f32();
                state.last_frame = now;
                state.render(dt);
                state.window.request_redraw();
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: winit::event::DeviceId, event: DeviceEvent) {
        let Some(state) = &mut self.state else { return };
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if state.cursor_grabbed {
                state.mouse_delta.0 += dx as f32;
                state.mouse_delta.1 += dy as f32;
            }
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(s) = &self.state {
            s.window.request_redraw();
        }
    }
}

impl AppState {
    fn render(&mut self, dt: f32) {
        const SPEED: f32 = 8.0;
        const SENS: f32 = 0.002;
        const SUN_SPEED: f32 = 0.4;

        if self.keys.contains(&KeyCode::KeyQ) {
            self.sun_angle -= SUN_SPEED * dt;
        }
        if self.keys.contains(&KeyCode::KeyE) {
            self.sun_angle += SUN_SPEED * dt;
        }

        self.cam_yaw += self.mouse_delta.0 * SENS;
        self.cam_pitch = (self.cam_pitch - self.mouse_delta.1 * SENS).clamp(-1.4, 1.4);
        self.mouse_delta = (0.0, 0.0);

        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let forward = glam::Vec3::new(sy * cp, sp, -cy * cp);
        let right = glam::Vec3::new(cy, 0.0, sy);

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
            self.cam_pos += glam::Vec3::Y * SPEED * dt;
        }
        if self.keys.contains(&KeyCode::ShiftLeft) {
            self.cam_pos -= glam::Vec3::Y * SPEED * dt;
        }

        let size = self.window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;

        let camera = Camera::perspective_look_at(
            self.cam_pos,
            self.cam_pos + forward,
            glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            aspect,
            0.1,
            1000.0,
        );

        let sun_dir_v =
            glam::Vec3::new(self.sun_angle.cos() * 0.35, self.sun_angle.sin(), 0.45).normalize();
        let light_dir = [-sun_dir_v.x, -sun_dir_v.y, -sun_dir_v.z];
        let sun_elev = sun_dir_v.y.clamp(-1.0, 1.0);
        let sun_lux = (sun_elev * 3.0).clamp(0.0, 1.0);
        let warmth = (1.0 - sun_elev).clamp(0.0, 1.0);
        let sun_color = [
            (1.0 + warmth * 0.55_f32).min(1.0),
            (0.72 + sun_elev * 0.28).clamp(0.0, 1.0),
            (0.50 + sun_elev * 0.38).clamp(0.0, 1.0),
        ];

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(e) => {
                log::warn!("Surface: {:?}", e);
                return;
            }
        };
        let view = output.texture.create_view(&Default::default());

        let lamp_on = (1.0 - sun_lux).clamp(0.0, 1.0);

        let _ = self.renderer.scene_mut().update_light(
            self.sun_light_id,
            directional_light(light_dir, sun_color, (sun_lux * 0.45).max(0.005)),
        );

        for (i, &id) in self.lamp_light_ids.iter().enumerate() {
            let (x, z) = LAMPS[i];
            let p = [x, 5.55, z];
            let _ = self.renderer.scene_mut().update_light(
                id,
                spot_light(
                    p,
                    [0.0, -1.0, 0.0],
                    [1.0, 0.72, 0.30],
                    5.5 * lamp_on,
                    14.0,
                    0.96,
                    1.22,
                ),
            );
        }

        let neon_boost = 0.6 + lamp_on * 0.4;
        for (i, &id) in self.neon_light_ids.iter().enumerate() {
            let (x, y, z, r, g, b) = NEONS[i];
            let p = [x, y, z];
            let _ = self
                .renderer
                .scene_mut()
                .update_light(id, point_light(p, [r, g, b], 5.0 * neon_boost, 12.0));
        }

        if let Err(e) = self.renderer.render(&camera, &view) {
            log::error!("Render: {:?}", e);
        }
        output.present();
    }
}
