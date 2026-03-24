//! Outdoor volcano example – high complexity
//!
//! An active volcanic island: a multi-layered cone built from five stacked
//! rect3d slabs of decreasing radius, a glowing crater pit, two lava-flow
//! channels snaking down the slopes, scattered boulders and rock formations,
//! and a ring of lava-pool planes at the base.
//!
//! Eight fire/lava glow lights in deep red-orange fill the scene with
//! hellish warmth.  A cool blue "ocean ambient" directional light provides
//! just enough contrast to read the dark rock silhouettes.
//!
//! Controls:
//!   WASD        — move forward/left/back/right
//!   Space/Shift — move up/down
//!   Mouse drag  — look around (click to grab cursor)
//!   Escape      — release cursor / exit

mod v3_demo_common;
use helio::{
    required_wgpu_features, required_wgpu_limits, Camera, LightId, MeshId, Renderer, RendererConfig,
};
use v3_demo_common::{
    box_mesh, cube_mesh, directional_light, make_material, plane_mesh, point_light,
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

// Lava/fire lights: (x, y, z, r, g, b, intensity, range)
const LAVA_LIGHTS: &[(f32, f32, f32, f32, f32, f32, f32, f32)] = &[
    // Crater eruption — hottest, most intense
    (0.0, 33.5, -10.0, 1.0, 0.35, 0.05, 18.0, 35.0),
    // Lava lake surface glow
    (0.0, 30.2, -10.0, 1.0, 0.20, 0.02, 8.0, 20.0),
    // Left lava flow channel
    (-12.0, 3.0, 4.0, 1.0, 0.30, 0.04, 5.0, 14.0),
    (-18.0, 0.8, 14.0, 1.0, 0.25, 0.03, 4.0, 12.0),
    // Right lava flow channel
    (14.0, 2.5, 2.0, 1.0, 0.28, 0.04, 5.0, 13.0),
    (20.0, 0.8, 10.0, 0.9, 0.22, 0.03, 4.0, 11.0),
    // Fumarole vents on mid-slope
    (-6.0, 14.0, -4.0, 1.0, 0.45, 0.1, 3.0, 8.0),
    (6.0, 12.0, -6.0, 1.0, 0.40, 0.08, 3.0, 8.0),
];

// Boulder/rock formations: (x, y_half, z, half_size)
const BOULDERS: &[(f32, f32, f32, f32)] = &[
    (-22.0, 1.4, 12.0, 1.4),
    (25.0, 1.1, 8.0, 1.1),
    (-16.0, 0.8, 20.0, 0.8),
    (18.0, 1.6, 18.0, 1.6),
    (-8.0, 0.9, 24.0, 0.9),
    (6.0, 1.2, 22.0, 1.2),
    (-28.0, 0.7, -4.0, 0.7),
    (28.0, 1.0, -12.0, 1.0),
    (-4.0, 2.1, -2.0, 2.1), // large rock near base
    (10.0, 1.8, -18.0, 1.8),
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

    _island_ground: MeshId,
    _cone_l1: MeshId,
    _cone_l2: MeshId,
    _cone_l3: MeshId,
    _cone_l4: MeshId,
    _cone_l5: MeshId,
    _crater_rim: MeshId,
    _lava_lake: MeshId,
    _flow_left: Vec<MeshId>,
    _flow_right: Vec<MeshId>,
    _lava_pools: Vec<MeshId>,
    _boulders: Vec<MeshId>,
    _scorch_patches: Vec<MeshId>,

    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

    // Scene state
    _ocean_light_id: LightId,
    lava_light_ids: Vec<LightId>,
    start_time: std::time::Instant,
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
                        .with_title("Helio – Outdoor Volcano")
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

        let rock_mat = renderer.insert_material(make_material(
            [0.25, 0.2, 0.18, 1.0],
            0.9,
            0.0,
            [0.0, 0.0, 0.0],
            0.0,
        ));
        let lava_mat = renderer.insert_material(make_material(
            [0.3, 0.08, 0.02, 1.0],
            0.9,
            0.0,
            [1.0, 0.35, 0.05],
            3.0,
        ));

        let _island_ground = renderer.insert_mesh(plane_mesh([0.0, 0.0, 0.0], 55.0));
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            _island_ground,
            rock_mat,
            glam::Mat4::IDENTITY,
            55.0,
        );

        let _cone_l1 = renderer.insert_mesh(box_mesh([0.0, 5.0, -10.0], [22.0, 5.0, 20.0]));
        let _cone_l2 = renderer.insert_mesh(box_mesh([0.0, 11.5, -10.0], [15.5, 6.5, 14.0]));
        let _cone_l3 = renderer.insert_mesh(box_mesh([0.0, 18.0, -10.0], [10.0, 6.5, 9.5]));
        let _cone_l4 = renderer.insert_mesh(box_mesh([0.0, 24.0, -10.0], [5.5, 6.0, 5.5]));
        let _cone_l5 = renderer.insert_mesh(box_mesh([0.0, 28.5, -10.0], [2.8, 4.5, 2.8]));
        let _crater_rim = renderer.insert_mesh(box_mesh([0.0, 30.5, -10.0], [3.2, 0.4, 3.2]));
        let _lava_lake = renderer.insert_mesh(box_mesh([0.0, 30.1, -10.0], [2.2, 0.05, 2.2]));
        for &m in &[
            _cone_l1,
            _cone_l2,
            _cone_l3,
            _cone_l4,
            _cone_l5,
            _crater_rim,
        ] {
            let _ = v3_demo_common::insert_object(
                &mut renderer,
                m,
                rock_mat,
                glam::Mat4::IDENTITY,
                22.0,
            );
        }
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            _lava_lake,
            lava_mat,
            glam::Mat4::IDENTITY,
            3.0,
        );

        let _flow_left: Vec<MeshId> = vec![
            renderer.insert_mesh(box_mesh([-5.5, 23.0, -6.0], [1.0, 0.12, 2.5])),
            renderer.insert_mesh(box_mesh([-9.0, 16.0, -2.0], [1.2, 0.12, 3.5])),
            renderer.insert_mesh(box_mesh([-13.0, 6.5, 3.0], [1.4, 0.12, 5.0])),
            renderer.insert_mesh(box_mesh([-17.0, 1.5, 10.0], [1.5, 0.1, 6.0])),
        ];
        let _flow_right: Vec<MeshId> = vec![
            renderer.insert_mesh(box_mesh([5.0, 22.0, -7.0], [1.0, 0.12, 2.5])),
            renderer.insert_mesh(box_mesh([9.0, 15.5, -3.0], [1.2, 0.12, 3.5])),
            renderer.insert_mesh(box_mesh([13.5, 6.0, 2.0], [1.4, 0.12, 4.5])),
            renderer.insert_mesh(box_mesh([19.0, 1.5, 8.0], [1.5, 0.1, 5.5])),
        ];
        for &m in _flow_left.iter().chain(_flow_right.iter()) {
            let _ = v3_demo_common::insert_object(
                &mut renderer,
                m,
                lava_mat,
                glam::Mat4::IDENTITY,
                6.0,
            );
        }

        let _lava_pools: Vec<MeshId> = vec![
            renderer.insert_mesh(box_mesh([-18.0, 0.06, 16.0], [4.0, 0.06, 3.0])),
            renderer.insert_mesh(box_mesh([22.0, 0.06, 12.0], [3.5, 0.06, 2.5])),
            renderer.insert_mesh(box_mesh([0.0, 0.06, 22.0], [2.5, 0.06, 2.0])),
        ];
        for &m in &_lava_pools {
            let _ = v3_demo_common::insert_object(
                &mut renderer,
                m,
                lava_mat,
                glam::Mat4::IDENTITY,
                4.0,
            );
        }

        let _boulders: Vec<MeshId> = BOULDERS
            .iter()
            .map(|&(x, yh, z, hs)| renderer.insert_mesh(cube_mesh([x, yh, z], hs)))
            .collect();
        for &m in &_boulders {
            let _ = v3_demo_common::insert_object(
                &mut renderer,
                m,
                rock_mat,
                glam::Mat4::IDENTITY,
                2.0,
            );
        }

        let _scorch_patches: Vec<MeshId> = vec![
            renderer.insert_mesh(box_mesh([-10.0, 0.02, 8.0], [4.5, 0.02, 3.5])),
            renderer.insert_mesh(box_mesh([12.0, 0.02, 6.0], [3.5, 0.02, 3.0])),
            renderer.insert_mesh(box_mesh([2.0, 0.02, 16.0], [3.0, 0.02, 4.0])),
            renderer.insert_mesh(box_mesh([-20.0, 0.02, -2.0], [3.0, 0.02, 2.5])),
            renderer.insert_mesh(box_mesh([22.0, 0.02, -8.0], [3.5, 0.02, 2.5])),
        ];
        for &m in &_scorch_patches {
            let _ = v3_demo_common::insert_object(
                &mut renderer,
                m,
                rock_mat,
                glam::Mat4::IDENTITY,
                4.0,
            );
        }

        let ocean_dir = glam::Vec3::new(-0.3, -0.6, 0.2).normalize();
        let _ocean_light_id = renderer.insert_light(directional_light(
            [ocean_dir.x, ocean_dir.y, ocean_dir.z],
            [0.3, 0.5, 1.0],
            0.04,
        ));
        let mut lava_light_ids = Vec::new();
        for &(x, y, z, r, g, b, intensity, range) in LAVA_LIGHTS {
            let p = [x, y, z];
            lava_light_ids.push(renderer.insert_light(point_light(p, [r, g, b], intensity, range)));
        }
        renderer.set_ambient([0.5, 0.1, 0.02], 0.04);
        renderer.set_clear_color([0.06, 0.01, 0.01, 1.0]);

        self.state = Some(AppState {
            window,
            surface,
            device,
            surface_format: format,
            renderer,
            last_frame: std::time::Instant::now(),
            _island_ground,
            _cone_l1,
            _cone_l2,
            _cone_l3,
            _cone_l4,
            _cone_l5,
            _crater_rim,
            _lava_lake,
            _flow_left,
            _flow_right,
            _lava_pools,
            _boulders,
            _scorch_patches,
            cam_pos: glam::Vec3::new(0.0, 8.0, 38.0),
            cam_yaw: std::f32::consts::PI,
            cam_pitch: -0.15,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            _ocean_light_id,
            lava_light_ids,
            start_time: std::time::Instant::now(),
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
        const SPEED: f32 = 10.0;
        const SENS: f32 = 0.002;

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
        let time = self.start_time.elapsed().as_secs_f32();

        let camera = Camera::perspective_look_at(
            self.cam_pos,
            self.cam_pos + forward,
            glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            aspect,
            0.1,
            1000.0,
        );

        // Per-light flicker: each light gets a unique phase
        let f = |phase: f32, freq: f32, amp: f32| 1.0 + (time * freq + phase).sin() * amp;

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(e) => {
                log::warn!("Surface: {:?}", e);
                return;
            }
        };
        let view = output.texture.create_view(&Default::default());

        // Update lava lights with per-light flicker
        for (i, &id) in self.lava_light_ids.iter().enumerate() {
            let (x, y, z, r, g, b, intensity, range) = LAVA_LIGHTS[i];
            let phase = i as f32 * 1.37;
            let fi = f(phase, 8.0 + i as f32 * 1.1, 0.06 + (i % 3) as f32 * 0.03);
            let p = [x, y, z];
            let _ = self
                .renderer
                .update_light(id, point_light(p, [r, g, b], intensity * fi, range));
        }
        if let Err(e) = self.renderer.render(&camera, &view) {
            log::error!("Render: {:?}", e);
        }
        output.present();
    }
}

