//! Indoor server room — helio v3
//!
//! A data-centre floor: four rows of eight server racks (32 total), cold/hot-aisle
//! walls, overhead cable trays, eight ceiling fluorescent panels, four rear cooling
//! units, and per-row status LED lighting (green/amber/red health states).
//!
//! Controls:
//!   WASD / Space / Shift — fly  (4 m/s)
//!   Mouse drag           — look (click to grab cursor)
//!   Escape               — release cursor / exit

mod v3_demo_common;
use v3_demo_common::{box_mesh, make_material, spot_light, point_light, insert_object};

use helio::{required_wgpu_features, required_wgpu_limits, Camera, LightId, Renderer, RendererConfig};

use winit::{
    application::ApplicationHandler, event::*, event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey}, window::{Window, WindowId, CursorGrabMode},
};
use std::collections::HashSet;
use std::sync::Arc;

// ── Scene data ────────────────────────────────────────────────────────────────

const RACK_ROWS: &[(f32, u8)] = &[
    (-7.5, 0),
    (-2.5, 1),
    ( 2.5, 2),
    ( 7.5, 3),
];
const RACK_Z_OFFSETS: &[f32] = &[-4.2, -3.0, -1.8, -0.6, 0.6, 1.8, 3.0, 4.2];
const CEILING_PANEL_XZ: &[(f32, f32)] = &[
    (-7.5, -3.5), (-7.5, 3.5),
    (-2.5, -3.5), (-2.5, 3.5),
    ( 2.5, -3.5), ( 2.5, 3.5),
    ( 7.5, -3.5), ( 7.5, 3.5),
];

fn row_color(tag: u8) -> [f32; 3] {
    match tag {
        0 => [0.0, 1.0, 0.2],
        1 => [0.0, 0.9, 0.5],
        2 => [1.0, 0.65, 0.0],
        _ => [1.0, 0.05, 0.05],
    }
}

// ─────────────────────────────────────────────────────────────────────────────


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
    _light_ids:     Vec<LightId>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }

        let window = Arc::new(event_loop.create_window(
            Window::default_attributes()
                .with_title("Helio — Indoor Server Room (v3)")
                .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
        ).expect("window"));

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(), ..Default::default()
        });
        let surface  = instance.create_surface(window.clone()).expect("surface");
        let adapter  = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface), force_fallback_adapter: false,
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format, width: size.width, height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![], desired_maximum_frame_latency: 2,
        });

        let mut renderer = Renderer::new(device.clone(), queue.clone(),
            RendererConfig::new(size.width, size.height, format)
                .with_shadow_quality(helio::ShadowQuality::Ultra),
        );
        renderer.set_clear_color([0.02, 0.02, 0.04, 1.0]);
        renderer.set_ambient([0.6, 0.72, 1.0], 0.06);

        // ── Materials ─────────────────────────────────────────────────────────────
        let mat_floor   = renderer.insert_material(make_material([0.18, 0.20, 0.18, 1.0], 0.90, 0.00, [0.0; 3], 0.0));
        let mat_ceiling = renderer.insert_material(make_material([0.85, 0.90, 0.95, 1.0], 0.80, 0.00, [0.0; 3], 0.0));
        let mat_wall    = renderer.insert_material(make_material([0.82, 0.86, 0.90, 1.0], 0.85, 0.00, [0.0; 3], 0.0));
        let mat_rack    = renderer.insert_material(make_material([0.10, 0.10, 0.12, 1.0], 0.40, 0.70, [0.0; 3], 0.0));
        let mat_panel   = renderer.insert_material(make_material([0.88, 0.93, 1.00, 1.0], 0.90, 0.00, [0.5, 0.6, 0.8], 3.0));
        let mat_cooling = renderer.insert_material(make_material([0.40, 0.50, 0.60, 1.0], 0.50, 0.60, [0.0; 3], 0.0));
        let mat_door    = renderer.insert_material(make_material([0.40, 0.45, 0.50, 1.0], 0.60, 0.30, [0.0; 3], 0.0));
        let mat_tray    = renderer.insert_material(make_material([0.30, 0.30, 0.35, 1.0], 0.40, 0.80, [0.0; 3], 0.0));

        // ── Geometry ───────────────────────────────────────────────────────────────
        let mut add = |r: &mut Renderer, cx: f32, cy: f32, cz: f32,
                       hx: f32, hy: f32, hz: f32, mat| {
            let m = r.insert_mesh(box_mesh([cx, cy, cz], [hx, hy, hz]));
            let _ = insert_object(r, m, mat, glam::Mat4::IDENTITY, (hx*hx+hy*hy+hz*hz).sqrt());
        };

        // Room shell: 24 m × 4 m × 12 m
        add(&mut renderer,   0.0, -0.05,  0.0, 12.0, 0.05, 6.0, mat_floor);
        add(&mut renderer,   0.0,  4.05,  0.0, 12.0, 0.05, 6.0, mat_ceiling);
        add(&mut renderer,   0.0,  2.0,  -6.0, 12.0, 2.0, 0.05, mat_wall);
        add(&mut renderer,   0.0,  2.0,   6.0, 12.0, 2.0, 0.05, mat_wall);
        add(&mut renderer,  12.0,  2.0,   0.0, 0.05, 2.0, 6.0,  mat_wall);
        add(&mut renderer, -12.0,  2.0,   0.0, 0.05, 2.0, 6.0,  mat_wall);

        // Raised floor tiles (5x3 grid)
        for xi in -2_i32..=2 {
            for zi in -1_i32..=1 {
                add(&mut renderer, xi as f32 * 4.0, 0.03, zi as f32 * 3.5, 1.9, 0.03, 1.7, mat_floor);
            }
        }

        // Server racks (4 rows x 8 each)
        for &(rx, _) in RACK_ROWS {
            for &rz in RACK_Z_OFFSETS {
                add(&mut renderer, rx, 1.0, rz, 0.3, 1.0, 0.45, mat_rack);
            }
        }

        // Hot-aisle containment walls
        add(&mut renderer, -5.0, 1.5, 0.0, 0.05, 1.5, 5.0, mat_wall);
        add(&mut renderer,  5.0, 1.5, 0.0, 0.05, 1.5, 5.0, mat_wall);

        // Cable trays overhead
        for &(rx, _) in RACK_ROWS {
            add(&mut renderer, rx, 3.55, 0.0, 0.25, 0.08, 5.5, mat_tray);
        }

        // Ceiling fluorescent panel bodies
        for &(px, pz) in CEILING_PANEL_XZ {
            add(&mut renderer, px, 3.92, pz, 0.3, 0.04, 0.8, mat_panel);
        }

        // Rear cooling units
        for &cx in &[-9.0_f32, -3.0, 3.0, 9.0] {
            add(&mut renderer, cx, 1.5, -5.75, 1.1, 1.5, 0.25, mat_cooling);
        }

        // Entry door
        add(&mut renderer, 0.0, 1.2, 5.90, 0.70, 1.2, 0.08, mat_door);
        add(&mut renderer, 0.0, 1.0, 5.95, 0.55, 1.0, 0.04, mat_door);

        // ── Lights ───────────────────────────────────────────────────────────────
        let mut light_ids: Vec<LightId> = Vec::new();

        // Overhead fluorescent panel spots (8)
        for &(px, pz) in CEILING_PANEL_XZ {
            light_ids.push(renderer.insert_light(spot_light(
                [px, 3.78, pz], [0.0, -1.0, 0.0], [0.88, 0.93, 1.0], 4.5, 7.0, 1.22, 1.48,
            )));
        }

        // Per-row status LED strips
        for &(rx, tag) in RACK_ROWS {
            let col = row_color(tag);
            light_ids.push(renderer.insert_light(point_light([rx, 2.1,  0.0], col, 2.5, 6.0)));
            light_ids.push(renderer.insert_light(point_light([rx, 2.1, -4.5], col, 1.0, 3.5)));
            light_ids.push(renderer.insert_light(point_light([rx, 2.1,  4.5], col, 1.0, 3.5)));
        }

        // Cooling unit indicators
        for (i, &cx) in [-9.0_f32, -3.0, 3.0, 9.0].iter().enumerate() {
            let col: [f32; 3] = if i == 1 { [0.0, 1.0, 0.3] } else { [0.0, 0.6, 1.0] };
            light_ids.push(renderer.insert_light(point_light([cx, 2.8, -5.6], col, 0.8, 3.0)));
        }

        self.state = Some(AppState {
            window, surface, device, surface_format: format, renderer,
            last_frame: std::time::Instant::now(),
            cam_pos: glam::Vec3::new(0.0, 1.75, 5.0),
            cam_yaw: std::f32::consts::PI, cam_pitch: -0.05,
            keys: HashSet::new(), cursor_grabbed: false, mouse_delta: (0.0, 0.0),
            _light_ids: light_ids,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event: KeyEvent {
                state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::Escape), ..
            }, .. } => {
                if state.cursor_grabbed {
                    state.cursor_grabbed = false;
                    let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                    state.window.set_cursor_visible(true);
                } else { event_loop.exit(); }
            }
            WindowEvent::KeyboardInput { event: KeyEvent {
                state: ks, physical_key: PhysicalKey::Code(key), ..
            }, .. } => {
                match ks {
                    ElementState::Pressed  => { state.keys.insert(key); }
                    ElementState::Released => { state.keys.remove(&key); }
                }
            }
            WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. } => {
                if !state.cursor_grabbed {
                    let ok = state.window.set_cursor_grab(CursorGrabMode::Confined)
                        .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked)).is_ok();
                    if ok { state.window.set_cursor_visible(false); state.cursor_grabbed = true; }
                }
            }
            WindowEvent::Resized(s) if s.width > 0 && s.height > 0 => {
                state.surface.configure(&state.device, &wgpu::SurfaceConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format: state.surface_format,
                    width: s.width, height: s.height, present_mode: wgpu::PresentMode::AutoVsync,
                    alpha_mode: wgpu::CompositeAlphaMode::Auto, view_formats: vec![],
                    desired_maximum_frame_latency: 2,
                });
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
        if let Some(s) = &self.state { s.window.request_redraw(); }
    }
}

impl AppState {
    fn render(&mut self, dt: f32) {
        const SPEED: f32 = 4.0;
        const SENS:  f32 = 0.002;

        self.cam_yaw   += self.mouse_delta.0 * SENS;
        self.cam_pitch  = (self.cam_pitch - self.mouse_delta.1 * SENS).clamp(-1.4, 1.4);
        self.mouse_delta = (0.0, 0.0);

        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let forward = glam::Vec3::new(sy * cp, sp, -cy * cp);
        let right   = glam::Vec3::new(cy, 0.0, sy);

        if self.keys.contains(&KeyCode::KeyW)      { self.cam_pos += forward * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyS)      { self.cam_pos -= forward * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyA)      { self.cam_pos -= right   * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyD)      { self.cam_pos += right   * SPEED * dt; }
        if self.keys.contains(&KeyCode::Space)     { self.cam_pos += glam::Vec3::Y * SPEED * dt; }
        if self.keys.contains(&KeyCode::ShiftLeft) { self.cam_pos -= glam::Vec3::Y * SPEED * dt; }

        let size   = self.window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;
        let camera = Camera::perspective_look_at(
            self.cam_pos, self.cam_pos + forward, glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4, aspect, 0.1, 80.0,
        );

        let output = match self.surface.get_current_texture() {
            Ok(t) => t, Err(e) => { log::warn!("surface: {:?}", e); return; }
        };
        let view = output.texture.create_view(&Default::default());

        if let Err(e) = self.renderer.render(&camera, &view) {
            log::error!("render: {:?}", e);
        }
        output.present();
    }
}
