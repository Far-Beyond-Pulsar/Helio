//! Terra Forge Demo — simple voxel sphere.
//!
//! Renders a 128^3 voxel grid containing a sphere using DDA ray marching.
//!
//! Controls:
//!   Mouse + Left click  - look around
//!   W/A/S/D             - fly
//!   Space/Shift         - up / down
//!   Scroll              - adjust fly speed
//!   Escape              - release / quit

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use glam::{EulerRot, Quat, Vec3};
use helio::{
    required_wgpu_features, required_wgpu_limits, Camera, RenderGraph, Renderer, RendererConfig,
};
use helio_pass_terra_forge::{TerraForgePass, DEFAULT_PLANET_RADIUS};
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

const LOOK_SENS: f32 = 0.002;
const INITIAL_FLY_SPEED: f32 = 50.0;
const DRAG: f32 = 6.0;

struct App {
    state: Option<AppState>,
}

struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    #[allow(dead_code)]
    queue: Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat,
    renderer: Renderer,
    last_frame: Instant,
    cam_pos: Vec3,
    yaw: f32,
    pitch: f32,
    velocity: Vec3,
    fly_speed: f32,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),
}

impl AppState {
    fn update(&mut self, dt: f32) {
        let (dx, dy) = self.mouse_delta;
        self.mouse_delta = (0.0, 0.0);
        self.yaw -= dx * LOOK_SENS;
        self.pitch = (self.pitch - dy * LOOK_SENS).clamp(-1.5, 1.5);

        let orientation = Quat::from_euler(EulerRot::YXZ, self.yaw, self.pitch, 0.0);
        let forward = orientation * -Vec3::Z;
        let right = orientation * Vec3::X;

        let mut accel = Vec3::ZERO;
        if self.keys.contains(&KeyCode::KeyW) {
            accel += forward;
        }
        if self.keys.contains(&KeyCode::KeyS) {
            accel -= forward;
        }
        if self.keys.contains(&KeyCode::KeyA) {
            accel -= right;
        }
        if self.keys.contains(&KeyCode::KeyD) {
            accel += right;
        }
        if self.keys.contains(&KeyCode::Space) {
            accel += Vec3::Y;
        }
        if self.keys.contains(&KeyCode::ShiftLeft) {
            accel -= Vec3::Y;
        }

        self.velocity += accel * self.fly_speed * dt;
        self.velocity /= 1.0 + DRAG * dt;
        self.cam_pos += self.velocity * dt;
    }

    fn camera(&self, width: u32, height: u32) -> Camera {
        let orientation = Quat::from_euler(EulerRot::YXZ, self.yaw, self.pitch, 0.0);
        let target = self.cam_pos + orientation * -Vec3::Z;
        let up = orientation * Vec3::Y;
        Camera::perspective_look_at(
            self.cam_pos,
            target,
            up,
            std::f32::consts::FRAC_PI_4,
            width as f32 / height.max(1) as f32,
            0.1,
            100_000.0,
        )
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
                        .with_title("Terra Forge — Voxel Sphere")
                        .with_inner_size(winit::dpi::PhysicalSize::new(1280u32, 720)),
                )
                .unwrap(),
        );
        let size = window.inner_size();

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("No suitable GPU adapter found");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: required_wgpu_features(adapter.features()),
            required_limits: required_wgpu_limits(adapter.limits()),
            ..Default::default()
        }))
        .expect("Failed to create device");

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        device.on_uncaptured_error(Arc::new(|error| {
            log::error!("wgpu uncaptured error: {}", error);
        }));

        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);

        surface.configure(
            &device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: surface_format,
                width: size.width,
                height: size.height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            },
        );

        let config =
            RendererConfig::new(size.width, size.height, surface_format).with_render_scale(1.0);
        let mut renderer = Renderer::new(device.clone(), queue.clone(), config);

        let pass = TerraForgePass::new(&device, &queue, size.width, size.height, surface_format);

        let mut graph = RenderGraph::new(&device, &queue);
        graph.add_pass(Box::new(pass));
        renderer.set_graph(graph);

        // Start camera just above the planet surface (not 2× radius away!).
        let planet_r = DEFAULT_PLANET_RADIUS;
        let cam_pos = Vec3::new(0.0, planet_r + 50.0, 0.0);

        self.state = Some(AppState {
            window,
            surface,
            device,
            queue,
            surface_format,
            renderer,
            last_frame: Instant::now(),
            cam_pos,
            yaw: 0.0,
            pitch: -0.5, // Look slightly downward toward the surface
            velocity: Vec3::ZERO,
            fly_speed: INITIAL_FLY_SPEED,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state: key_state,
                        ..
                    },
                ..
            } => match key_state {
                ElementState::Pressed => {
                    state.keys.insert(key);
                    if key == KeyCode::Escape {
                        if state.cursor_grabbed {
                            let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                            state.window.set_cursor_visible(true);
                            state.cursor_grabbed = false;
                        } else {
                            event_loop.exit();
                        }
                    }
                }
                ElementState::Released => {
                    state.keys.remove(&key);
                }
            },

            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                state.surface.configure(
                    &state.device,
                    &wgpu::SurfaceConfiguration {
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        format: state.surface_format,
                        width: size.width,
                        height: size.height,
                        present_mode: wgpu::PresentMode::Fifo,
                        alpha_mode: wgpu::CompositeAlphaMode::Opaque,
                        view_formats: vec![],
                        desired_maximum_frame_latency: 2,
                    },
                );
                state.renderer.set_render_size(size.width, size.height);
            }

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

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.1,
                };
                state.fly_speed = (state.fly_speed * (1.0 + scroll * 0.1)).clamp(0.1, 2000.0);
                log::info!("Fly speed: {:.1} m/s", state.fly_speed);
            }

            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(state.last_frame).as_secs_f32().min(0.05);
                state.last_frame = now;
                state.update(dt);

                let size = state.window.inner_size();
                let camera = state.camera(size.width, size.height);

                let output = match state.surface.get_current_texture() {
                    Ok(t) => t,
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Other) => {
                        let size = state.window.inner_size();
                        if size.width > 0 && size.height > 0 {
                            state.surface.configure(
                                &state.device,
                                &wgpu::SurfaceConfiguration {
                                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                                    format: state.surface_format,
                                    width: size.width,
                                    height: size.height,
                                    present_mode: wgpu::PresentMode::Fifo,
                                    alpha_mode: wgpu::CompositeAlphaMode::Opaque,
                                    view_formats: vec![],
                                    desired_maximum_frame_latency: 2,
                                },
                            );
                            state.renderer.set_render_size(size.width, size.height);
                        }
                        state.window.request_redraw();
                        return;
                    }
                    Err(e) => {
                        log::warn!("surface error: {:?}", e);
                        state.window.request_redraw();
                        return;
                    }
                };
                let view = output
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                if let Err(e) = state.renderer.render(&camera, &view) {
                    log::error!("render error: {:?}", e);
                }
                output.present();
                state.window.request_redraw();
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
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App { state: None };
    event_loop.run_app(&mut app).unwrap();
}
