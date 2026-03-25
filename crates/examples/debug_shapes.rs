//! Debug Shapes — helio v3
//!
//! The v2 debug drawing primitives (debug_line, debug_sphere, etc.) are
//! not available in helio v3.  This demo instead displays a gallery of
//! richly coloured solid-geometry props that showcase the material/light
//! system while still serving as a visual debugging reference.
//!
//! Controls:
//!   WASD / Space / Shift — fly  (5 m/s)
//!   Mouse drag           — look (click to grab cursor)
//!   Escape               — release cursor / exit

mod v3_demo_common;
use v3_demo_common::{box_mesh, directional_light, insert_object, make_material, point_light};

use helio::{required_wgpu_features, required_wgpu_limits, Camera, Renderer, RendererConfig};

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
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App { state: None };
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
    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),
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
                        .with_title("Helio Debug Shapes (v3)")
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
            required_features: required_wgpu_features(adapter.features()),
            required_limits: required_wgpu_limits(adapter.limits()),
            ..Default::default()
        }))
        .expect("device");
        device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| {
            panic!("[GPU] {:?}", e)
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
        let size = window.inner_size();
        surface.configure(
            &device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: size.width,
                height: size.height,
                present_mode: wgpu::PresentMode::AutoVsync,
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
        renderer.set_clear_color([0.05, 0.05, 0.08, 1.0]);
        renderer.set_ambient([0.20, 0.22, 0.30], 0.18);

        // ── Materials ─────────────────────────────────────────────────────────────
        let mat_floor = renderer.insert_material(make_material(
            [0.28, 0.28, 0.28, 1.0],
            0.90,
            0.00,
            [0.0; 3],
            0.0,
        ));
        let mat_red = renderer.insert_material(make_material(
            [0.90, 0.15, 0.15, 1.0],
            0.60,
            0.00,
            [0.0; 3],
            0.0,
        ));
        let mat_green = renderer.insert_material(make_material(
            [0.15, 0.85, 0.20, 1.0],
            0.60,
            0.00,
            [0.0; 3],
            0.0,
        ));
        let mat_blue = renderer.insert_material(make_material(
            [0.15, 0.35, 0.90, 1.0],
            0.60,
            0.00,
            [0.0; 3],
            0.0,
        ));
        let mat_orange = renderer.insert_material(make_material(
            [1.00, 0.45, 0.10, 1.0],
            0.50,
            0.00,
            [0.0; 3],
            0.0,
        ));
        let mat_cyan = renderer.insert_material(make_material(
            [0.10, 0.80, 0.90, 1.0],
            0.50,
            0.00,
            [0.0; 3],
            0.0,
        ));
        let mat_magenta = renderer.insert_material(make_material(
            [0.90, 0.10, 0.80, 1.0],
            0.50,
            0.00,
            [0.0; 3],
            0.0,
        ));
        let mat_yellow = renderer.insert_material(make_material(
            [1.00, 0.90, 0.10, 1.0],
            0.40,
            0.00,
            [0.0; 3],
            0.0,
        ));
        let mat_metal = renderer.insert_material(make_material(
            [0.70, 0.70, 0.70, 1.0],
            0.30,
            0.85,
            [0.0; 3],
            0.0,
        ));
        let mat_glow = renderer.insert_material(make_material(
            [0.05, 0.05, 0.05, 1.0],
            0.90,
            0.00,
            [0.4, 0.8, 1.0],
            3.5,
        ));

        // ── Geometry ───────────────────────────────────────────────────────────────
        let mut add =
            |r: &mut Renderer, cx: f32, cy: f32, cz: f32, hx: f32, hy: f32, hz: f32, mat| {
                let m = r.insert_mesh(box_mesh([0.0, 0.0, 0.0], [hx, hy, hz]));
                let _ = insert_object(
                    r,
                    m,
                    mat,
                    glam::Mat4::from_translation(glam::Vec3::new(cx, cy, cz)),
                    (hx * hx + hy * hy + hz * hz).sqrt(),
                );
            };

        // Ground
        add(&mut renderer, 0.0, -0.05, 0.0, 12.0, 0.05, 12.0, mat_floor);

        // Axis-colour columns (representing a coordinate frame)
        add(&mut renderer, -6.0, 1.5, 0.0, 0.4, 1.5, 0.4, mat_red); // -X
        add(&mut renderer, 6.0, 1.5, 0.0, 0.4, 1.5, 0.4, mat_orange); // +X
        add(&mut renderer, 0.0, 1.5, -6.0, 0.4, 1.5, 0.4, mat_green); // -Z
        add(&mut renderer, 0.0, 1.5, 6.0, 0.4, 1.5, 0.4, mat_cyan); // +Z
        add(&mut renderer, 0.0, 3.0, 0.0, 0.25, 3.0, 0.25, mat_blue); // Y

        // Gallery row (Z = 3) — various shapes/sizes
        add(&mut renderer, -4.0, 0.5, 3.0, 0.5, 0.5, 0.5, mat_magenta);
        add(&mut renderer, -2.0, 0.7, 3.0, 0.4, 0.7, 0.8, mat_yellow);
        add(&mut renderer, 0.0, 1.2, 3.0, 0.6, 1.2, 0.4, mat_metal);
        add(&mut renderer, 2.0, 0.5, 3.0, 0.8, 0.5, 0.5, mat_glow);
        add(&mut renderer, 4.0, 0.9, 3.0, 0.3, 0.9, 0.9, mat_red);

        // Gallery row (Z = -3)
        add(&mut renderer, -4.0, 0.35, -3.0, 0.6, 0.35, 0.6, mat_green);
        add(&mut renderer, -2.0, 0.8, -3.0, 0.3, 0.8, 0.6, mat_blue);
        add(&mut renderer, 0.0, 0.55, -3.0, 1.1, 0.55, 0.4, mat_orange);
        add(&mut renderer, 2.0, 0.7, -3.0, 0.5, 0.7, 0.5, mat_cyan);
        add(&mut renderer, 4.0, 0.45, -3.0, 0.7, 0.45, 0.3, mat_metal);

        // AABB corner markers (draw_aabb analogue)
        draw_aabb_corners(
            &mut renderer,
            [-2.5, 0.0, -0.5],
            [-1.5, 1.5, 0.5],
            mat_yellow,
        );

        // ── Lights ──────────────────────────────────────────────────────────────
        let _ = renderer.insert_light(directional_light(
            [-0.5, -1.0, -0.5],
            [1.0, 0.95, 0.88],
            5.0,
        ));
        let _ = renderer.insert_light(point_light([0.0, 6.0, 0.0], [0.60, 0.75, 1.0], 40.0, 20.0));
        let _ = renderer.insert_light(point_light([5.0, 3.0, -5.0], [1.00, 0.60, 0.3], 20.0, 15.0));

        self.state = Some(AppState {
            window,
            surface,
            device,
            surface_format: format,
            renderer,
            last_frame: std::time::Instant::now(),
            cam_pos: glam::Vec3::new(0.0, 3.0, 10.0),
            cam_yaw: 0.0,
            cam_pitch: -0.15,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
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
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: ks,
                        ..
                    },
                ..
            } => match ks {
                ElementState::Pressed => {
                    state.keys.insert(code);
                    if code == KeyCode::Escape {
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
                    state.keys.remove(&code);
                }
            },
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if !state.cursor_grabbed {
                    if state
                        .window
                        .set_cursor_grab(CursorGrabMode::Confined)
                        .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked))
                        .is_ok()
                    {
                        state.window.set_cursor_visible(false);
                        state.cursor_grabbed = true;
                    }
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

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        let Some(s) = &mut self.state else { return };
        if let DeviceEvent::MouseMotion { delta } = event {
            if s.cursor_grabbed {
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

impl AppState {
    fn update_camera(&mut self, dt: f32) {
        const LOOK_SPEED: f32 = 0.003;
        const MOVE_SPEED: f32 = 5.0;
        if self.cursor_grabbed {
            self.cam_yaw += self.mouse_delta.0 * LOOK_SPEED;
            self.cam_pitch -= self.mouse_delta.1 * LOOK_SPEED;
            self.cam_pitch = self.cam_pitch.clamp(
                -std::f32::consts::FRAC_PI_2 * 0.99,
                std::f32::consts::FRAC_PI_2 * 0.99,
            );
            self.mouse_delta = (0.0, 0.0);
        }
        let fwd = glam::Vec3::new(self.cam_yaw.sin(), 0.0, -self.cam_yaw.cos());
        let right = glam::Vec3::new(self.cam_yaw.cos(), 0.0, self.cam_yaw.sin());
        let up = glam::Vec3::Y;
        let mut vel = glam::Vec3::ZERO;
        if self.keys.contains(&KeyCode::KeyW) {
            vel += fwd;
        }
        if self.keys.contains(&KeyCode::KeyS) {
            vel -= fwd;
        }
        if self.keys.contains(&KeyCode::KeyD) {
            vel += right;
        }
        if self.keys.contains(&KeyCode::KeyA) {
            vel -= right;
        }
        if self.keys.contains(&KeyCode::Space) {
            vel += up;
        }
        if self.keys.contains(&KeyCode::ShiftLeft) {
            vel -= up;
        }
        if vel.length_squared() > 0.0 {
            self.cam_pos += vel.normalize() * MOVE_SPEED * dt;
        }
    }

    fn render(&mut self) {
        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(e) => {
                log::warn!("surface: {:?}", e);
                return;
            }
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let fwd = glam::Vec3::new(sy * cp, sp, -cy * cp);
        let size = self.window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;
        let camera = Camera::perspective_look_at(
            self.cam_pos,
            self.cam_pos + fwd,
            glam::Vec3::Y,
            70.0_f32.to_radians(),
            aspect,
            0.1,
            1000.0,
        );

        if let Err(e) = self.renderer.render(&camera, &view) {
            log::error!("render: {:?}", e);
        }
        output.present();
    }
}

/// Place one small cube at each corner of an AABB to visualise its extent.
fn draw_aabb_corners(
    renderer: &mut Renderer,
    min: [f32; 3],
    max: [f32; 3],
    mat: helio::MaterialId,
) {
    let hs = 0.06_f32;
    for &x in &[min[0], max[0]] {
        for &y in &[min[1], max[1]] {
            for &z in &[min[2], max[2]] {
                let m = renderer.insert_mesh(box_mesh([x, y, z], [hs, hs, hs]));
                let _ = insert_object(renderer, m, mat, glam::Mat4::IDENTITY, hs * 1.8);
            }
        }
    }
}

