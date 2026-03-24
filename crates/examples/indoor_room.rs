//! Indoor room example – simple scene, low complexity
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
use v3_demo_common::{box_mesh, make_material, plane_mesh, point_light};

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use std::collections::HashSet;
use std::sync::Arc;

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("Event loop error");
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
    start_time: std::time::Instant,

    floor: MeshId,
    ceiling: MeshId,
    wall_n: MeshId,
    wall_s: MeshId,
    wall_e: MeshId,
    wall_w: MeshId,
    table: MeshId,
    bookcase: MeshId,
    sofa: MeshId,

    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

    overhead_light_id: LightId,
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
                        .with_title("Helio – Indoor Room")
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
        let info = adapter.get_info();
        println!(
            "[WGPU] Backend: {:?}, Device: {}, Driver: {}",
            info.backend, info.name, info.driver
        );
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

        let mat = renderer.insert_material(make_material(
            [0.7, 0.68, 0.62, 1.0],
            0.8,
            0.0,
            [0.0, 0.0, 0.0],
            0.0,
        ));

        let floor = renderer.insert_mesh(plane_mesh([0.0, 0.0, 0.0], 4.0));
        let ceiling = renderer.insert_mesh(plane_mesh([0.0, 3.0, 0.0], 4.0));
        let wall_n = renderer.insert_mesh(box_mesh([0.0, 1.5, -4.0], [4.0, 1.5, 0.05]));
        let wall_s = renderer.insert_mesh(box_mesh([0.0, 1.5, 4.0], [4.0, 1.5, 0.05]));
        let wall_e = renderer.insert_mesh(box_mesh([4.0, 1.5, 0.0], [0.05, 1.5, 4.0]));
        let wall_w = renderer.insert_mesh(box_mesh([-4.0, 1.5, 0.0], [0.05, 1.5, 4.0]));
        let table = renderer.insert_mesh(box_mesh([1.5, 0.4, 1.5], [0.8, 0.4, 0.5]));
        let bookcase = renderer.insert_mesh(box_mesh([-3.4, 1.0, -2.5], [0.3, 1.0, 1.2]));
        let sofa = renderer.insert_mesh(box_mesh([-1.5, 0.35, 2.5], [1.2, 0.35, 0.5]));

        let _ = v3_demo_common::insert_object(&mut renderer, floor, mat, glam::Mat4::IDENTITY, 4.0);
        let _ =
            v3_demo_common::insert_object(&mut renderer, ceiling, mat, glam::Mat4::IDENTITY, 4.0);
        let _ =
            v3_demo_common::insert_object(&mut renderer, wall_n, mat, glam::Mat4::IDENTITY, 4.0);
        let _ =
            v3_demo_common::insert_object(&mut renderer, wall_s, mat, glam::Mat4::IDENTITY, 4.0);
        let _ =
            v3_demo_common::insert_object(&mut renderer, wall_e, mat, glam::Mat4::IDENTITY, 4.0);
        let _ =
            v3_demo_common::insert_object(&mut renderer, wall_w, mat, glam::Mat4::IDENTITY, 4.0);
        let _ = v3_demo_common::insert_object(&mut renderer, table, mat, glam::Mat4::IDENTITY, 0.8);
        let _ =
            v3_demo_common::insert_object(&mut renderer, bookcase, mat, glam::Mat4::IDENTITY, 1.0);
        let _ = v3_demo_common::insert_object(&mut renderer, sofa, mat, glam::Mat4::IDENTITY, 1.2);

        let overhead_pos = [0.0f32, 2.85, 0.0];
        let overhead_light_id =
            renderer.insert_light(point_light(overhead_pos, [1.0, 0.85, 0.6], 4.0, 7.0));
        renderer.insert_light(point_light([-2.5, 0.9, -2.5], [1.0, 0.55, 0.2], 2.5, 5.0));
        renderer.insert_light(point_light([2.5, 0.9, 2.5], [1.0, 0.75, 0.35], 2.0, 4.5));
        renderer.set_ambient([1.0, 0.95, 0.85], 0.05);
        renderer.set_clear_color([0.02, 0.02, 0.06, 1.0]);

        self.state = Some(AppState {
            window,
            surface,
            device,
            surface_format: format,
            renderer,
            last_frame: std::time::Instant::now(),
            start_time: std::time::Instant::now(),
            floor,
            ceiling,
            wall_n,
            wall_s,
            wall_e,
            wall_w,
            table,
            bookcase,
            sofa,
            cam_pos: glam::Vec3::new(0.0, 1.6, 2.0),
            cam_yaw: std::f32::consts::PI,
            cam_pitch: 0.0,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            overhead_light_id,
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
        const SPEED: f32 = 3.0;
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
            100.0,
        );

        // Overhead ceiling light pulses very slightly (candle-like flicker)
        let flicker = 1.0 + (time * 11.3).sin() * 0.04 + (time * 7.7).cos() * 0.02;
        let _ = self.renderer.update_light(
            self.overhead_light_id,
            point_light([0.0, 2.85, 0.0], [1.0, 0.85, 0.6], 4.0 * flicker, 7.0),
        );

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(e) => {
                log::warn!("Surface: {:?}", e);
                return;
            }
        };
        let view = output.texture.create_view(&Default::default());

        if let Err(e) = self.renderer.render(&camera, &view) {
            log::error!("Render: {:?}", e);
        }
        output.present();
    }
}

