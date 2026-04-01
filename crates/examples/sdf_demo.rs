//! SDF Demo — sparse signed distance field terrain with editable shapes.
//!
//! Uses `helio-pass-sdf` as a custom render pass on top of the Helio renderer.
//! A procedural terrain is generated via noise, and the user can place shapes
//! into the scene to demonstrate union/subtraction/intersection operations.
//!
//! Controls:
//!   Mouse + Left click  – look around
//!   W/A/S/D             – fly
//!   Space/Shift         – up / down
//!   1                   – place sphere (union)
//!   2                   – subtract sphere (subtraction)
//!   3                   – place cube (union)
//!   4                   – place smooth-blended sphere
//!   5                   – toggle debug clip-level visualisation
//!   R                   – clear all edits
//!   Escape              – release / quit

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use glam::{EulerRot, Mat4, Quat, Vec3};
use helio::{required_wgpu_features, required_wgpu_limits, Camera, Renderer, RendererConfig};
use helio::{
    RenderGraph,
};
use helio_pass_sdf::{
    BooleanOp, SdfEdit, SdfPass, SdfShapeParams, SdfShapeType, TerrainConfig,
};
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

// ── constants ─────────────────────────────────────────────────────────────────

const LOOK_SENS: f32 = 0.002;
const FLY_SPEED: f32 = 8.0;
const DRAG: f32 = 6.0;

// ── app ───────────────────────────────────────────────────────────────────────

struct App {
    state: Option<AppState>,
}

struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat,
    renderer: Renderer,
    sdf_pass: *mut SdfPass,
    last_frame: Instant,
    cam_pos: Vec3,
    yaw: f32,
    pitch: f32,
    velocity: Vec3,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),
}

// SAFETY: SdfPass is only accessed from the main thread (winit event loop).
unsafe impl Send for AppState {}
unsafe impl Sync for AppState {}

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
        if self.keys.contains(&KeyCode::KeyW) { accel += forward; }
        if self.keys.contains(&KeyCode::KeyS) { accel -= forward; }
        if self.keys.contains(&KeyCode::KeyA) { accel -= right; }
        if self.keys.contains(&KeyCode::KeyD) { accel += right; }
        if self.keys.contains(&KeyCode::Space) { accel += Vec3::Y; }
        if self.keys.contains(&KeyCode::ShiftLeft) { accel -= Vec3::Y; }

        self.velocity += accel * FLY_SPEED * dt;
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
            0.01,
            2000.0,
        )
    }

    fn sdf(&mut self) -> &mut SdfPass {
        unsafe { &mut *self.sdf_pass }
    }

    fn place_edit(&mut self, shape: SdfShapeType, op: BooleanOp, params: SdfShapeParams, blend: f32) {
        let orientation = Quat::from_euler(EulerRot::YXZ, self.yaw, self.pitch, 0.0);
        let forward = orientation * -Vec3::Z;
        let pos = self.cam_pos + forward * 5.0;
        let transform = Mat4::from_translation(pos);

        self.sdf().add_edit(SdfEdit {
            shape,
            op,
            transform,
            params,
            blend_radius: blend,
        });
        log::info!("Placed {:?} {:?} at {:?} (blend={})", shape, op, pos, blend);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }

        let attrs = Window::default_attributes()
            .with_title("Helio SDF Demo")
            .with_inner_size(winit::dpi::PhysicalSize::new(1280, 720));
        let window = Arc::new(event_loop.create_window(attrs).unwrap());

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
        }))
        .expect("No suitable GPU adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: required_wgpu_features(adapter.features()),
                required_limits: required_wgpu_limits(adapter.limits()),
                ..Default::default()
            },
        ))
        .expect("Device request failed");

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Capture GPU validation errors so we can debug pipeline issues
        device.on_uncaptured_error(Arc::new(|error| {
            log::error!("wgpu uncaptured error: {}", error);
        }));

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps.formats.iter()
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

        // ── Renderer + custom graph with SDF pass ─────────────────────────────
        let config = RendererConfig::new(size.width, size.height, surface_format);
        let mut renderer = Renderer::new(device.clone(), queue.clone(), config);

        // Build a minimal graph: SDF pass only
        let mut graph = RenderGraph::new(&device, &queue);
        let sdf_pass = SdfPass::new(&device, surface_format, Some(TerrainConfig::rolling()));
        let sdf_box = Box::new(sdf_pass);
        let sdf_ptr = &*sdf_box as *const SdfPass as *mut SdfPass;
        graph.add_pass(sdf_box);
        renderer.set_graph(graph);

        self.state = Some(AppState {
            window,
            surface,
            device,
            queue,
            surface_format,
            renderer,
            sdf_pass: sdf_ptr,
            last_frame: Instant::now(),
            cam_pos: Vec3::new(0.0, 4.0, 15.0),
            yaw: 0.0,
            pitch: -0.2,
            velocity: Vec3::ZERO,
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
                event: KeyEvent {
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

            // ── SDF edit keys ─────────────────────────────────────────────────
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    state: ElementState::Pressed,
                    physical_key: PhysicalKey::Code(code),
                    repeat: false,
                    ..
                },
                ..
            } => {
                match code {
                    KeyCode::Digit1 => {
                        state.place_edit(
                            SdfShapeType::Sphere, BooleanOp::Union,
                            SdfShapeParams::sphere(2.0), 0.0,
                        );
                    }
                    KeyCode::Digit2 => {
                        state.place_edit(
                            SdfShapeType::Sphere, BooleanOp::Subtraction,
                            SdfShapeParams::sphere(3.0), 0.5,
                        );
                    }
                    KeyCode::Digit3 => {
                        state.place_edit(
                            SdfShapeType::Cube, BooleanOp::Union,
                            SdfShapeParams::cube(1.5, 1.5, 1.5), 0.0,
                        );
                    }
                    KeyCode::Digit4 => {
                        state.place_edit(
                            SdfShapeType::Sphere, BooleanOp::Union,
                            SdfShapeParams::sphere(2.5), 1.5,
                        );
                    }
                    KeyCode::Digit5 => {
                        state.sdf().toggle_debug();
                        log::info!("Debug clip-level vis toggled");
                    }
                    KeyCode::KeyR => {
                        state.sdf().clear_edits();
                        log::info!("All SDF edits cleared");
                    }
                    _ => { state.keys.insert(code); }
                }
            }

            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    physical_key: PhysicalKey::Code(code),
                    state: ElementState::Released,
                    ..
                },
                ..
            } => {
                state.keys.remove(&code);
            }

            WindowEvent::Resized(size) => {
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
                    let ok = state.window
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
                let now = Instant::now();
                let dt = now.duration_since(state.last_frame).as_secs_f32().min(0.05);
                state.last_frame = now;
                state.update(dt);

                let size = state.window.inner_size();
                let camera = state.camera(size.width, size.height);

                let output = match state.surface.get_current_texture() {
                    Ok(t) => t,
                    Err(e) => {
                        log::warn!("surface error: {:?}", e);
                        return;
                    }
                };
                let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
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
