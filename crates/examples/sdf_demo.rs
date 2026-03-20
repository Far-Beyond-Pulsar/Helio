//! SDF Demo — helio v3
//!
//! The interactive SDF terrain-editing features from the original v2 demo are
//! not available in helio v3.  This replacement shows a static rocky-terrain
//! scene that demonstrates basic lighting and material variety.
//!
//! Controls:
//!   WASD / Space / Shift — fly  (20 m/s)
//!   Mouse drag           — look (click to grab cursor)
//!   Escape               — release cursor / exit

mod v3_demo_common;
use v3_demo_common::{box_mesh, make_material, point_light, directional_light, insert_object};

use helio::{required_wgpu_features, required_wgpu_limits, Camera, Renderer, RendererConfig};

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId, CursorGrabMode},
};

use std::collections::HashSet;
use std::sync::Arc;

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App { state: None };
    event_loop.run_app(&mut app).expect("run");
}

struct App { state: Option<AppState> }

struct AppState {
    window:         Arc<Window>,
    surface:        wgpu::Surface<'static>,
    device:         Arc<wgpu::Device>,
    surface_format: wgpu::TextureFormat,
    renderer:       Renderer,
    last_frame:     std::time::Instant,
    cam_pos:        glam::Vec3,
    cam_yaw:        f32,
    cam_pitch:      f32,
    keys:           HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta:    (f32, f32),
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }

        let window = Arc::new(event_loop.create_window(
            Window::default_attributes()
                .with_title("Helio SDF Demo (v3)")
                .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
        ).expect("window"));

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(), ..Default::default()
        });
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })).expect("adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: required_wgpu_features(adapter.features()),
                required_limits:   required_wgpu_limits(adapter.limits()),
                ..Default::default()
            },
            None,
        )).expect("device");
        device.on_uncaptured_error(Box::new(|e| panic!("[GPU] {:?}", e)));
        let device = Arc::new(device);
        let queue  = Arc::new(queue);
        let caps   = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
        let size   = window.inner_size();
        surface.configure(&device, &wgpu::SurfaceConfiguration {
            usage:  wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width:  size.width, height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode:   caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        });

        let mut renderer = Renderer::new(device.clone(), queue.clone(),
            RendererConfig::new(size.width, size.height, format),
        );
        renderer.set_clear_color([0.10, 0.14, 0.18, 1.0]);
        renderer.set_ambient([0.28, 0.32, 0.40], 0.22);

        // ── Materials ──────────────────────────────────────────────────────
        let mat_ground = renderer.insert_material(make_material([0.40, 0.38, 0.35, 1.0], 0.95, 0.00, [0.0,0.0,0.0], 0.0));
        let mat_stone  = renderer.insert_material(make_material([0.55, 0.52, 0.48, 1.0], 0.88, 0.05, [0.0,0.0,0.0], 0.0));
        let mat_bright = renderer.insert_material(make_material([0.82, 0.78, 0.70, 1.0], 0.80, 0.00, [0.0,0.0,0.0], 0.0));
        let mat_dark   = renderer.insert_material(make_material([0.22, 0.20, 0.18, 1.0], 0.95, 0.02, [0.0,0.0,0.0], 0.0));

        // ── Geometry ───────────────────────────────────────────────────────
        let m = renderer.insert_mesh(box_mesh([0.0, -0.1, 0.0], [30.0, 0.1, 30.0]));
        let _ = insert_object(&mut renderer, m, mat_ground, glam::Mat4::IDENTITY, 42.0);

        for &(cx, cz, hx, hy, hz, bright) in &[
            (-5.0_f32, -5.0, 2.0, 0.6, 2.0, false),
            ( 5.0,  3.0, 3.0, 1.2, 2.5, true),
            (-8.0,  2.0, 1.5, 0.8, 1.5, false),
            ( 2.0, -7.0, 2.5, 0.5, 2.0, true),
            (-3.0,  7.0, 1.8, 1.5, 1.8, false),
            ( 8.0, -3.0, 1.5, 1.0, 3.0, true),
            ( 0.0, -9.0, 4.0, 0.4, 3.5, false),
            (-6.0,  8.5, 2.2, 0.7, 2.2, true),
        ] {
            let mesh = renderer.insert_mesh(box_mesh([cx, hy, cz], [hx, hy, hz]));
            let mat  = if bright { mat_bright } else { mat_stone };
            let r    = (hx*hx + hy*hy + hz*hz).sqrt();
            let _ = insert_object(&mut renderer, mesh, mat, glam::Mat4::IDENTITY, r);
        }
        for &[cx, cz] in &[[-6.0_f32, 6.0], [6.0, -6.0], [-2.0, -9.0], [9.5, 7.0]] {
            let mesh = renderer.insert_mesh(box_mesh([cx, 2.0, cz], [0.8, 2.0, 0.8]));
            let _ = insert_object(&mut renderer, mesh, mat_dark, glam::Mat4::IDENTITY, 2.9);
        }

        // ── Lights ─────────────────────────────────────────────────────────
        let _ = renderer.insert_light(directional_light([-0.4, -1.0, -0.5], [1.0, 0.95, 0.85], 6.0));
        let _ = renderer.insert_light(point_light([0.0, 8.0, 0.0], [0.5, 0.70, 1.0], 30.0, 25.0));
        let _ = renderer.insert_light(point_light([-8.0, 3.0, -8.0], [0.9, 0.60, 0.3], 15.0, 18.0));

        self.state = Some(AppState {
            window, surface, device, surface_format: format, renderer,
            last_frame: std::time::Instant::now(),
            cam_pos: glam::Vec3::new(0.0, 2.0, 8.0),
            cam_yaw: 0.0, cam_pitch: -0.1,
            keys: HashSet::new(), cursor_grabbed: false, mouse_delta: (0.0, 0.0),
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event: KeyEvent { state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::Escape), .. }, ..
            } => {
                if state.cursor_grabbed {
                    state.cursor_grabbed = false;
                    let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                    state.window.set_cursor_visible(true);
                } else { event_loop.exit(); }
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent { state: ks, physical_key: PhysicalKey::Code(key), .. }, ..
            } => {
                match ks {
                    ElementState::Pressed  => { state.keys.insert(key); }
                    ElementState::Released => { state.keys.remove(&key); }
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed, button: MouseButton::Left, ..
            } => {
                if !state.cursor_grabbed {
                    let ok = state.window.set_cursor_grab(CursorGrabMode::Confined)
                        .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked))
                        .is_ok();
                    if ok { state.window.set_cursor_visible(false); state.cursor_grabbed = true; }
                }
            }
            WindowEvent::Resized(sz) if sz.width > 0 && sz.height > 0 => {
                state.surface.configure(&state.device, &wgpu::SurfaceConfiguration {
                    usage:  wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format: state.surface_format,
                    width:  sz.width, height: sz.height,
                    present_mode: wgpu::PresentMode::AutoVsync,
                    alpha_mode:   wgpu::CompositeAlphaMode::Auto,
                    view_formats: vec![],
                    desired_maximum_frame_latency: 2,
                });
                state.renderer.set_render_size(sz.width, sz.height);
            }
            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                let dt  = (now - state.last_frame).as_secs_f32();
                state.last_frame = now;
                state.render(dt);
                state.window.request_redraw();
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: winit::event::DeviceId, event: DeviceEvent) {
        let Some(s) = &mut self.state else { return };
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if s.cursor_grabbed { s.mouse_delta.0 += dx as f32; s.mouse_delta.1 += dy as f32; }
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(s) = &self.state { s.window.request_redraw(); }
    }
}

impl AppState {
    fn render(&mut self, dt: f32) {
        // silence unused warning — dt drives camera movement
        let _ = dt;
        const SPEED: f32 = 20.0;
        const SENS:  f32 = 0.002;

        self.cam_yaw   += self.mouse_delta.0 * SENS;
        self.cam_pitch  = (self.cam_pitch - self.mouse_delta.1 * SENS).clamp(-1.5, 1.5);
        self.mouse_delta = (0.0, 0.0);

        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let fwd   = glam::Vec3::new(sy * cp, sp, -cy * cp);
        let right = glam::Vec3::new(cy, 0.0, sy);
        if self.keys.contains(&KeyCode::KeyW)      { self.cam_pos += fwd   * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyS)      { self.cam_pos -= fwd   * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyA)      { self.cam_pos -= right * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyD)      { self.cam_pos += right * SPEED * dt; }
        if self.keys.contains(&KeyCode::Space)     { self.cam_pos.y += SPEED * dt; }
        if self.keys.contains(&KeyCode::ShiftLeft) { self.cam_pos.y -= SPEED * dt; }

        let sz     = self.window.inner_size();
        let aspect = sz.width as f32 / sz.height.max(1) as f32;
        let camera = Camera::perspective_look_at(
            self.cam_pos, self.cam_pos + fwd, glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4, aspect, 0.1, 5000.0,
        );

        let output = match self.surface.get_current_texture() {
            Ok(t)  => t,
            Err(e) => { log::warn!("surface: {:?}", e); return; }
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        if let Err(e) = self.renderer.render(&camera, &view) {
            log::error!("render: {:?}", e);
        }
        output.present();
    }
}

// ── dead code kept to avoid remove-old-method confusion ──────────────────────

fn _use_tool_removed_in_v3() {
    // The SDF pick/edit API does not exist in helio v3.
}

