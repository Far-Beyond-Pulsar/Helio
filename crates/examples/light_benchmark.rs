//! Light Benchmark — helio v3
//!
//! 150 simultaneous point lights (10×15 grid) over a warehouse floor.
//! Tests v3 deferred + radiance-cascade GI with high light count.
//!
//! Controls:
//!   WASD / Space / Shift — fly
//!   +/-                  — increase/decrease light intensity
//!   Escape               — release cursor / exit

mod v3_demo_common;
use v3_demo_common::{box_mesh, insert_object, make_material, point_light};

use helio::{
    required_wgpu_features, required_wgpu_limits, Camera, LightId, Renderer, RendererConfig,
};

use std::collections::HashSet;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

// ── Light grid ──────────────────────────────────────────────────────────────
const LIGHT_COLS: usize = 10;
const LIGHT_ROWS: usize = 15;
const LIGHT_HEIGHT: f32 = 2.5;
const LIGHT_SPACING_X: f32 = 3.8;
const LIGHT_SPACING_Z: f32 = 2.6;
const LIGHT_RANGE: f32 = 7.0;
const LIGHT_INTENSITY: f32 = 6.0;

/// Build 150 light params (position, color, intensity, range) from grid.
fn build_light_params() -> Vec<([f32; 3], [f32; 3], f32, f32)> {
    let mut out = Vec::with_capacity(LIGHT_COLS * LIGHT_ROWS);
    let half_x = (LIGHT_COLS as f32 - 1.0) * 0.5 * LIGHT_SPACING_X;
    let half_z = (LIGHT_ROWS as f32 - 1.0) * 0.5 * LIGHT_SPACING_Z;
    for row in 0..LIGHT_ROWS {
        for col in 0..LIGHT_COLS {
            let x = col as f32 * LIGHT_SPACING_X - half_x;
            let z = row as f32 * LIGHT_SPACING_Z - half_z;
            let hue = (col * LIGHT_ROWS + row) as f32 / (LIGHT_COLS * LIGHT_ROWS) as f32;
            let color = hsv_to_rgb(hue, 0.75, 1.0);
            out.push(([x, LIGHT_HEIGHT, z], color, LIGHT_INTENSITY, LIGHT_RANGE));
        }
    }
    out
}

/// Simple HSV → linear-RGB conversion (no gamma).
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h6 = h * 6.0;
    let i = h6.floor() as u32 % 6;
    let f = h6 - h6.floor();
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match i {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}
fn main() {
    env_logger::init();
    log::info!(
        "Starting Light Benchmark ({} lights)",
        LIGHT_COLS * LIGHT_ROWS
    );
    EventLoop::new()
        .expect("event loop")
        .run_app(&mut App { state: None })
        .expect("run");
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
    frame_count: u64,

    light_ids: Vec<LightId>,
    base_lights: Vec<([f32; 3], [f32; 3], f32, f32)>,

    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

    light_intensity_multiplier: f32,

    time_render_end: Option<std::time::Instant>,
    time_about_to_wait_start: Option<std::time::Instant>,
    time_redraw_requested: Option<std::time::Instant>,
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
                        .with_title(format!(
                            "Helio — Light Benchmark ({} lights)",
                            LIGHT_COLS * LIGHT_ROWS
                        ))
                        .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
                )
                .expect("window"),
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
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
        let fmt = caps
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
                format: fmt,
                width: size.width,
                height: size.height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 1,
            },
        );

        let mut renderer = Renderer::new(
            device.clone(),
            queue.clone(),
            RendererConfig::new(size.width, size.height, fmt),
        );
        renderer.set_ambient([0.03, 0.03, 0.04], 1.0);

        // Materials
        let mat_floor = renderer.insert_material(make_material(
            [0.55, 0.52, 0.45, 1.0],
            0.85,
            0.00,
            [0.0; 3],
            0.0,
        ));
        let mat_pillar = renderer.insert_material(make_material(
            [0.60, 0.60, 0.62, 1.0],
            0.50,
            0.30,
            [0.0; 3],
            0.0,
        ));
        let mat_crate = renderer.insert_material(make_material(
            [0.50, 0.38, 0.25, 1.0],
            0.70,
            0.00,
            [0.0; 3],
            0.0,
        ));

        let mut add =
            |r: &mut Renderer, cx: f32, cy: f32, cz: f32, hx: f32, hy: f32, hz: f32, mat| {
                let m = r.insert_mesh(box_mesh([cx, cy, cz], [hx, hy, hz]));
                let _ = insert_object(
                    r,
                    m,
                    mat,
                    glam::Mat4::IDENTITY,
                    (hx * hx + hy * hy + hz * hz).sqrt(),
                );
            };

        add(&mut renderer, 0.0, -0.05, 0.0, 20.0, 0.05, 20.0, mat_floor);

        for ix in -2..=3_i32 {
            for iz in -2..=3_i32 {
                add(
                    &mut renderer,
                    ix as f32 * 4.0,
                    2.0,
                    iz as f32 * 4.0,
                    0.4,
                    2.0,
                    0.4,
                    mat_pillar,
                );
            }
        }

        for &(pos, hs) in &[
            ([-8.0_f32, 0.3, -6.5], 0.30_f32),
            ([8.5, 0.3, 4.0], 0.30),
            ([-5.0, 0.3, 10.5], 0.25),
            ([7.0, 0.3, -9.5], 0.30),
            ([-13.0, 0.3, 2.0], 0.35),
            ([13.0, 0.3, -3.0], 0.35),
            ([-10.0, 0.3, -13.0], 0.30),
            ([10.0, 0.3, 13.0], 0.30),
            ([3.0, 0.3, -15.5], 0.25),
            ([-3.0, 0.3, 15.5], 0.25),
            ([0.0, 0.3, 8.0], 0.30),
            ([-1.0, 0.3, -8.0], 0.28),
            ([15.0, 0.3, 7.0], 0.30),
            ([-15.0, 0.3, -7.0], 0.30),
            ([6.0, 0.3, 17.0], 0.25),
            ([-6.0, 0.3, -17.0], 0.25),
        ] {
            add(&mut renderer, pos[0], pos[1], pos[2], hs, hs, hs, mat_crate);
        }

        let base_lights = build_light_params();
        let light_ids: Vec<LightId> = base_lights
            .iter()
            .map(|&(pos, col, intensity, range)| {
                renderer.insert_light(point_light(pos, col, intensity, range))
            })
            .collect();

        self.state = Some(AppState {
            window,
            surface,
            device,
            surface_format: fmt,
            renderer,
            last_frame: std::time::Instant::now(),
            frame_count: 0,
            light_ids,
            base_lights,
            cam_pos: glam::Vec3::new(0.0, 4.0, 22.0),
            cam_yaw: 0.0,
            cam_pitch: -0.18,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            light_intensity_multiplier: 1.0,
            time_render_end: None,
            time_about_to_wait_start: None,
            time_redraw_requested: None,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

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
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Minus),
                        ..
                    },
                ..
            } => {
                state.light_intensity_multiplier =
                    (state.light_intensity_multiplier - 0.1).max(0.1);
                eprintln!("Light intensity: {:.1}x", state.light_intensity_multiplier);
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Equal),
                        ..
                    },
                ..
            } => {
                state.light_intensity_multiplier =
                    (state.light_intensity_multiplier + 0.1).min(5.0);
                eprintln!("Light intensity: {:.1}x", state.light_intensity_multiplier);
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
                        desired_maximum_frame_latency: 1,
                    },
                );
                state.renderer.set_render_size(s.width, s.height);
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
                        desired_maximum_frame_latency: 1,
                    },
                );
                state.renderer.set_render_size(s.width, s.height);
            }

            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();

                if let Some(last_render_end) = state.time_render_end {
                    let full_cycle_ms = last_render_end.elapsed().as_secs_f32() * 1000.0;
                    if state.frame_count % 60 == 0 {
                        eprintln!("render_end -> next RedrawRequested: {:.2}ms", full_cycle_ms);
                    }
                }

                if let Some(about_to_wait_start) = state.time_about_to_wait_start {
                    let gap_ms = about_to_wait_start.elapsed().as_secs_f32() * 1000.0;
                    if gap_ms > 2.0 {
                        eprintln!("about_to_wait -> RedrawRequested: {:.2}ms", gap_ms);
                    }
                }

                state.time_redraw_requested = Some(now);
                let dt = (now - state.last_frame).as_secs_f32();
                state.last_frame = now;
                state.render(dt);
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
        if let Some(s) = &mut self.state {
            let now = std::time::Instant::now();
            if let Some(render_end) = s.time_render_end {
                let gap_ms = render_end.elapsed().as_secs_f32() * 1000.0;
                if gap_ms > 2.0 {
                    eprintln!("render_end -> about_to_wait: {:.2}ms", gap_ms);
                }
            }
            s.time_about_to_wait_start = Some(now);
            s.window.request_redraw();
        }
    }
}
impl AppState {
    fn render(&mut self, dt: f32) {
        if let Some(redraw_time) = self.time_redraw_requested {
            let gap_ms = redraw_time.elapsed().as_secs_f32() * 1000.0;
            if gap_ms > 2.0 {
                eprintln!("RedrawRequested -> render(): {:.2}ms", gap_ms);
            }
        }

        const SPEED: f32 = 8.0;
        const SENS: f32 = 0.002;
        self.cam_yaw += self.mouse_delta.0 * SENS;
        self.cam_pitch = (self.cam_pitch - self.mouse_delta.1 * SENS).clamp(-1.5, 1.5);
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
            self.cam_pos.y += SPEED * dt;
        }
        if self.keys.contains(&KeyCode::ShiftLeft) {
            self.cam_pos.y -= SPEED * dt;
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
            300.0,
        );

        let get_texture_start = std::time::Instant::now();
        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(e) => {
                log::warn!("Surface error: {:?}", e);
                return;
            }
        };
        let get_texture_ms = get_texture_start.elapsed().as_secs_f32() * 1000.0;
        if get_texture_ms > 10.0 {
            eprintln!("get_current_texture() blocked for {:.2}ms", get_texture_ms);
        }

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let scene_build_start = std::time::Instant::now();

        let fade_in_frames = 120.0_f32;
        let frame_age = (self.frame_count as f32).min(fade_in_frames);
        let time_fade = if frame_age < fade_in_frames {
            let t = frame_age / fade_in_frames;
            t * t * (3.0 - 2.0 * t)
        } else {
            1.0
        };

        let multiplier = self.light_intensity_multiplier * time_fade;
        for (i, &id) in self.light_ids.iter().enumerate() {
            let (pos, col, intensity, range) = self.base_lights[i];
            let _ = self
                .renderer
                .update_light(id, point_light(pos, col, intensity * multiplier, range));
        }

        let scene_build_ms = scene_build_start.elapsed().as_secs_f32() * 1000.0;
        if scene_build_ms > 10.0 {
            eprintln!("Scene construction took {:.2}ms", scene_build_ms);
        }

        if let Err(e) = self.renderer.render(&camera, &view) {
            log::error!("Render error: {:?}", e);
        }

        let present_start = std::time::Instant::now();
        output.present();
        let present_ms = present_start.elapsed().as_secs_f32() * 1000.0;

        self.time_render_end = Some(std::time::Instant::now());
        self.frame_count += 1;

        if self.frame_count % 60 == 0 {
            let total_render_ms = if let Some(redraw_time) = self.time_redraw_requested {
                redraw_time.elapsed().as_secs_f32() * 1000.0
            } else {
                0.0
            };
            eprintln!(
                "Full cycle: total_render={:.2}ms, present={:.2}ms",
                total_render_ms, present_ms
            );
        }
    }
}

