//! Radiance Cascades benchmark — GI performance test (helio v3)
//!
//! Cornell box with colored walls and a few cubes demonstrating multi-bounce GI.
//! Three point lights whose intensity can be adjusted live.
//!
//! Controls:
//!   WASD / Space / Shift — fly
//!   Mouse drag           — look (click to grab cursor)
//!   +/-                  — increase/decrease all light intensity
//!   Escape               — release cursor / exit

mod v3_demo_common;
use v3_demo_common::{box_mesh, make_material, point_light, insert_object};

use helio::{required_wgpu_features, required_wgpu_limits, Camera, LightId, Renderer, RendererConfig};

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId, CursorGrabMode},
};

use std::collections::HashSet;
use std::sync::Arc;

/// Base light parameters: (position, color, intensity, range)
const LIGHT_BASE: &[([f32; 3], [f32; 3], f32, f32)] = &[
    ([0.0, 4.8, 0.0],   [1.0, 1.0, 1.0], 18.0, 6.0),
    ([-3.5, 3.0, -2.0], [1.0, 0.7, 0.4], 12.0, 5.0),
    ([3.5,  3.0,  2.0], [0.4, 0.7, 1.0], 12.0, 5.0),
];

fn main() {
    env_logger::init();
    EventLoop::new()
        .expect("event loop")
        .run_app(&mut App { state: None })
        .expect("run");
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
    light_ids:      [LightId; 3],
    light_intensity_multiplier: f32,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }

        let window = Arc::new(
            event_loop.create_window(
                Window::default_attributes()
                    .with_title("Helio — RC Benchmark (v3)")
                    .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
            ).expect("window"),
        );

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

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let fmt  = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
        surface.configure(&device, &wgpu::SurfaceConfiguration {
            usage:  wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: fmt,
            width:  size.width, height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 1,
            alpha_mode:   caps.alpha_modes[0],
            view_formats: vec![],
        });

        let mut renderer = Renderer::new(device.clone(), queue.clone(),
            RendererConfig::new(size.width, size.height, fmt),
        );
        renderer.set_ambient([0.02, 0.02, 0.03], 1.0);

        // ── Materials ─────────────────────────────────────────────────────────────
        let mat_white = renderer.insert_material(make_material([0.9, 0.9, 0.9, 1.0],    0.9, 0.0, [0.0,0.0,0.0], 0.0));
        let mat_red   = renderer.insert_material(make_material([0.8, 0.1, 0.1, 1.0],    0.9, 0.0, [0.0,0.0,0.0], 0.0));
        let mat_green = renderer.insert_material(make_material([0.1, 0.7, 0.1, 1.0],    0.9, 0.0, [0.0,0.0,0.0], 0.0));
        let mat_cube  = renderer.insert_material(make_material([0.8, 0.78, 0.72, 1.0], 0.85, 0.0, [0.0,0.0,0.0], 0.0));

        // ── Geometry ───────────────────────────────────────────────────────────────
        let mut add_box = |cx: f32, cy: f32, cz: f32, hx: f32, hy: f32, hz: f32, mat| {
            let m = renderer.insert_mesh(box_mesh([cx,cy,cz],[hx,hy,hz]));
            let _ = insert_object(&mut renderer, m, mat, glam::Mat4::IDENTITY,
                                  (hx*hx + hy*hy + hz*hz).sqrt());
        };
        add_box( 0.0, -0.05,  0.0, 5.0, 0.05, 5.0, mat_white); // floor
        add_box( 0.0,  5.05,  0.0, 5.0, 0.05, 5.0, mat_white); // ceiling
        add_box( 0.0,  2.5,  -5.05, 5.0, 2.5, 0.05, mat_white); // back wall
        add_box( 0.0,  2.5,   5.05, 5.0, 2.5, 0.05, mat_white); // front wall
        add_box( 5.05, 2.5,   0.0, 0.05, 2.5, 5.0,  mat_green); // right (green)
        add_box(-5.05, 2.5,   0.0, 0.05, 2.5, 5.0,  mat_red);   // left (red)
        add_box(-2.0, 0.5,  -2.0, 0.5,  0.5, 0.5,  mat_cube);
        add_box( 2.0, 0.5,   2.0, 0.5,  0.5, 0.5,  mat_cube);
        add_box( 0.0, 0.7,   0.0, 0.7,  0.7, 0.7,  mat_cube);
        add_box(-3.0, 1.0,   1.5, 1.0,  1.0, 1.0,  mat_cube);
        add_box( 3.0, 0.6,  -1.5, 0.6,  0.6, 0.6,  mat_cube);

        // ── Lights ───────────────────────────────────────────────────────────────
        let light_ids: [LightId; 3] = LIGHT_BASE
            .iter()
            .map(|&(pos, col, int, rng)| renderer.insert_light(point_light(pos, col, int, rng)))
            .collect::<Vec<_>>()
            .try_into()
            .expect("3 lights");

        self.state = Some(AppState {
            window, surface, device, surface_format: fmt, renderer,
            last_frame: std::time::Instant::now(),
            cam_pos: glam::Vec3::new(0.0, 2.5, 8.0),
            cam_yaw: 0.0, cam_pitch: 0.0,
            keys: HashSet::new(), cursor_grabbed: false, mouse_delta: (0.0, 0.0),
            light_ids,
            light_intensity_multiplier: 1.0,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => { event_loop.exit(); }

            WindowEvent::KeyboardInput { event: KeyEvent {
                state: ElementState::Pressed,
                physical_key: PhysicalKey::Code(KeyCode::Escape), ..
            }, .. } => {
                if state.cursor_grabbed {
                    state.cursor_grabbed = false;
                    let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                    state.window.set_cursor_visible(true);
                } else { event_loop.exit(); }
            }

            WindowEvent::KeyboardInput { event: KeyEvent {
                state: ElementState::Pressed,
                physical_key: PhysicalKey::Code(KeyCode::Minus), ..
            }, .. } => {
                state.light_intensity_multiplier = (state.light_intensity_multiplier - 0.1).max(0.1);
                eprintln!("Light intensity: {:.1}x", state.light_intensity_multiplier);
            }

            WindowEvent::KeyboardInput { event: KeyEvent {
                state: ElementState::Pressed,
                physical_key: PhysicalKey::Code(KeyCode::Equal), ..
            }, .. } => {
                state.light_intensity_multiplier = (state.light_intensity_multiplier + 0.1).min(5.0);
                eprintln!("Light intensity: {:.1}x", state.light_intensity_multiplier);
            }

            WindowEvent::KeyboardInput { event: KeyEvent {
                state: ks, physical_key: PhysicalKey::Code(key), ..
            }, .. } => {
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

            WindowEvent::Resized(sz) => {
                state.surface.configure(&state.device, &wgpu::SurfaceConfiguration {
                    usage:  wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format: state.surface_format,
                    width:  sz.width, height: sz.height,
                    present_mode: wgpu::PresentMode::AutoVsync,
                    desired_maximum_frame_latency: 1,
                    alpha_mode: wgpu::CompositeAlphaMode::Auto,
                    view_formats: vec![],
                });
                state.renderer.set_render_size(sz.width, sz.height);
            }

            WindowEvent::RedrawRequested => { state.render(); }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(s) = &self.state { s.window.request_redraw(); }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: winit::event::DeviceId, event: DeviceEvent) {
        let Some(s) = &mut self.state else { return };
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if s.cursor_grabbed { s.mouse_delta.0 += dx as f32; s.mouse_delta.1 += dy as f32; }
        }
    }
}

impl AppState {
    fn render(&mut self) {
        const SPEED: f32 = 4.0;
        const MOUSE_SENS: f32 = 0.002;

        let now = std::time::Instant::now();
        let dt  = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        self.cam_yaw   += self.mouse_delta.0 * MOUSE_SENS;
        self.cam_pitch  = (self.cam_pitch - self.mouse_delta.1 * MOUSE_SENS).clamp(-1.5, 1.5);
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
            std::f32::consts::FRAC_PI_4, aspect, 0.1, 300.0,
        );

        let output = match self.surface.get_current_texture() {
            Ok(t)  => t,
            Err(e) => { log::warn!("Surface error: {:?}", e); return; }
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        for (i, &id) in self.light_ids.iter().enumerate() {
            let (pos, col, base_int, range) = LIGHT_BASE[i];
            let _ = self.renderer.update_light(id,
                point_light(pos, col, base_int * self.light_intensity_multiplier, range));
        }
        if let Err(e) = self.renderer.render(&camera, &view) {
            log::error!("Render error: {:?}", e);
        }
        output.present();
    }
}