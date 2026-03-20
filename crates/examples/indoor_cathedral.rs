//! Indoor cathedral example – high complexity
//!
//! A large Gothic cathedral interior: a 60 m nave flanked by two side aisles,
//! 12 stone columns, a raised altar platform with a cross, carved stone pews
//! in 6 rows on each side, three ornate chandeliers, stained-glass window
//! shafts casting coloured light at intervals along both walls, and candle
//! clusters near the altar.
//!
//! No sky atmosphere — the scene relies entirely on the interplay of the
//! chandelier warm-white lights, the cool-coloured stained-glass fills, and
//! a very dim stone-cold ambient to create a moody sacred atmosphere. The
//! radiance cascades GI system bounces chandelier light deep into the side
//! aisles and onto the vaulted ceiling.
//!
//! Controls:
//!   WASD        — move forward/left/back/right
//!   Space/Shift — move up/down
//!   Mouse drag  — look around (click to grab cursor)
//!   Escape      — release cursor / exit



mod v3_demo_common;

use helio::{required_wgpu_features, required_wgpu_limits, Camera, LightId, MeshId, Renderer, RendererConfig};
use v3_demo_common::{box_mesh, make_material, plane_mesh, point_light};


use winit::{
    application::ApplicationHandler, event::*, event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey}, window::{Window, WindowId, CursorGrabMode},
};


use std::collections::HashSet;


use std::sync::Arc;

// ── Scene data ────────────────────────────────────────────────────────────────

// Column positions along the nave (Z axis), symmetric at x = ±5.5
const COLUMN_Z: &[f32] = &[-22.0, -14.0, -6.0, 2.0, 10.0, 18.0];

// Stained glass window lights: (x_wall_side, y, z, r, g, b)
// Positive x = right-side windows, negative = left-side; placed just inside the wall
const GLASS_LIGHTS: &[(f32, f32, f32, f32, f32, f32)] = &[
    // Left wall (x ≈ -10.5), windows between columns
    (-10.3,  9.0, -18.0,  0.8, 0.2, 1.0),  // violet
    (-10.3,  9.0,  -6.0,  0.2, 0.7, 1.0),  // sky blue
    (-10.3,  9.0,   6.0,  0.2, 1.0, 0.4),  // emerald
    (-10.3,  9.0,  18.0,  1.0, 0.7, 0.1),  // gold
    // Right wall (x ≈ +10.5)
    ( 10.3,  9.0, -18.0,  1.0, 0.2, 0.3),  // ruby
    ( 10.3,  9.0,  -6.0,  1.0, 0.5, 0.1),  // amber
    ( 10.3,  9.0,   6.0,  0.1, 0.8, 0.9),  // teal
    ( 10.3,  9.0,  18.0,  0.9, 0.1, 0.7),  // magenta
    // Rose window above entrance (back wall, z ≈ +28)
    (  0.0, 13.0,  27.0,  1.0, 0.75, 0.3), // warm gold
];

// Chandelier positions (x=0, hanging from y≈19.5, at z intervals)
const CHANDELIER_Z: &[f32] = &[-16.0, 0.0, 16.0];

// Candle cluster positions near the altar (z ≈ -24)
const CANDLES: &[(f32, f32, f32)] = &[
    (-3.0, 1.6, -23.5),
    (-1.5, 1.6, -23.0),
    ( 0.0, 1.6, -23.5),
    ( 1.5, 1.6, -23.0),
    ( 3.0, 1.6, -23.5),
];

// Pew rows: 6 per side, spaced 2.4 m apart starting at z = -20
const PEW_Z_START: f32 = -20.0;
const PEW_Z_STEP:  f32 =  3.2;
const PEW_COUNT:   usize = 6;

// ─────────────────────────────────────────────────────────────────────────────



fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("run");
}

struct App { state: Option<AppState> }

struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    surface_format: wgpu::TextureFormat,
    renderer: Renderer,
    last_frame: std::time::Instant,

    // Major structural surfaces
    _floor: MeshId,
    _nave_ceiling: MeshId,
    _aisle_ceil_l: MeshId,
    _aisle_ceil_r: MeshId,
    _wall_left_outer:  MeshId,
    _wall_right_outer: MeshId,
    _wall_front:  MeshId,
    _wall_back:   MeshId,
    // Colonnade arches (inner walls between nave and aisles, with gaps left for columns)
    _colonnade_l: Vec<MeshId>, // wall segments between columns
    _colonnade_r: Vec<MeshId>,
    // Columns
    _columns: Vec<MeshId>,
    // Altar
    _altar_plinth: MeshId,
    _altar_step:   MeshId,
    _cross_vert:   MeshId,
    _cross_horiz:  MeshId,
    // Pews
    _pews_left:  Vec<MeshId>,
    _pews_right: Vec<MeshId>,
    // Chandelier bodies (chain + ring)
    _chandelier_chains: Vec<MeshId>,
    _chandelier_rings:  Vec<MeshId>,

    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

    // Scene state
    chandelier_light_ids: Vec<LightId>,
    candle_light_ids: Vec<LightId>,
    start_time: std::time::Instant,
}

impl App {
    fn new() -> Self { Self { state: None } }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }

        let window = Arc::new(event_loop.create_window(
            Window::default_attributes()
                .with_title("Helio – Indoor Cathedral")
                .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
        ).expect("window"));

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::all(), ..Default::default() });
        let surface  = instance.create_surface(window.clone()).expect("surface");
        let adapter  = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface), force_fallback_adapter: false,
        })).expect("adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Device"),
            required_features: required_wgpu_features(adapter.features()),
            required_limits: required_wgpu_limits(adapter.limits()),
            ..Default::default()
        }, None)).expect("device");
        device.on_uncaptured_error(Box::new(|e| {
            panic!("[GPU UNCAPTURED ERROR] {:?}", e);
        }));
        let device = Arc::new(device);
        let queue  = Arc::new(queue);

        let caps   = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().find(|f| f.is_srgb()).copied().unwrap_or(caps.formats[0]);
        let size   = window.inner_size();
        surface.configure(&device, &wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format, width: size.width, height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![], desired_maximum_frame_latency: 2,
        });

        let mut renderer = Renderer::new(device.clone(), queue.clone(),
            RendererConfig::new(size.width, size.height, format),
        );

        let mat = renderer.insert_material(make_material([0.75, 0.72, 0.68, 1.0], 0.85, 0.0, [0.0, 0.0, 0.0], 0.0));

        // Nave + aisles: total width = 22m (x: -11..+11), length = 60m (z: -28..+28), height = 21m
        let _floor         = renderer.insert_mesh(plane_mesh([0.0, 0.0, 0.0], 11.0));
        let _nave_ceiling  = renderer.insert_mesh(box_mesh([0.0, 21.0,  0.0], [6.0, 0.18, 28.0]));
        let _aisle_ceil_l  = renderer.insert_mesh(box_mesh([-8.5, 11.0, 0.0], [2.5, 0.15, 28.0]));
        let _aisle_ceil_r  = renderer.insert_mesh(box_mesh([ 8.5, 11.0, 0.0], [2.5, 0.15, 28.0]));
        let _wall_left_outer  = renderer.insert_mesh(box_mesh([-11.0, 7.0,  0.0], [0.25, 7.0, 28.0]));
        let _wall_right_outer = renderer.insert_mesh(box_mesh([ 11.0, 7.0,  0.0], [0.25, 7.0, 28.0]));
        let _wall_front  = renderer.insert_mesh(box_mesh([0.0, 10.5, 28.0],  [11.0, 10.5, 0.25]));
        let _wall_back   = renderer.insert_mesh(box_mesh([0.0, 10.5, -28.0], [11.0, 10.5, 0.25]));
        let _ = v3_demo_common::insert_object(&mut renderer, _floor, mat, glam::Mat4::IDENTITY, 11.0);
        let _ = v3_demo_common::insert_object(&mut renderer, _nave_ceiling, mat, glam::Mat4::IDENTITY, 28.0);
        let _ = v3_demo_common::insert_object(&mut renderer, _aisle_ceil_l, mat, glam::Mat4::IDENTITY, 28.0);
        let _ = v3_demo_common::insert_object(&mut renderer, _aisle_ceil_r, mat, glam::Mat4::IDENTITY, 28.0);
        let _ = v3_demo_common::insert_object(&mut renderer, _wall_left_outer, mat, glam::Mat4::IDENTITY, 28.0);
        let _ = v3_demo_common::insert_object(&mut renderer, _wall_right_outer, mat, glam::Mat4::IDENTITY, 28.0);
        let _ = v3_demo_common::insert_object(&mut renderer, _wall_front, mat, glam::Mat4::IDENTITY, 11.0);
        let _ = v3_demo_common::insert_object(&mut renderer, _wall_back, mat, glam::Mat4::IDENTITY, 11.0);

        // Colonnade: short wall segments between columns (between column z-positions)
        // 7 segments per side: before first col, between each pair, after last col
        let col_z_all: Vec<f32> = {
            let mut v = vec![-28.0_f32]; // south wall
            v.extend_from_slice(COLUMN_Z);
            v.push(28.0); // north wall
            v
        };
        let _colonnade_l: Vec<MeshId> = col_z_all.windows(2).map(|w| {
            let mid_z = (w[0] + w[1]) * 0.5;
            let half_len = (w[1] - w[0]) * 0.5 - 0.9; // gap for column
            let id = renderer.insert_mesh(box_mesh([-5.5, 5.5, mid_z], [0.25, 5.5, half_len.max(0.1)]));
            let _ = v3_demo_common::insert_object(&mut renderer, id, mat, glam::Mat4::IDENTITY, 5.5);
            id
        }).collect();
        let _colonnade_r: Vec<MeshId> = col_z_all.windows(2).map(|w| {
            let mid_z = (w[0] + w[1]) * 0.5;
            let half_len = (w[1] - w[0]) * 0.5 - 0.9;
            let id = renderer.insert_mesh(box_mesh([ 5.5, 5.5, mid_z], [0.25, 5.5, half_len.max(0.1)]));
            let _ = v3_demo_common::insert_object(&mut renderer, id, mat, glam::Mat4::IDENTITY, 5.5);
            id
        }).collect();

        // Columns: 0.65 m square, 20 m tall, at x = ±5.5
        let _columns: Vec<MeshId> = COLUMN_Z.iter().flat_map(|&z| {
            let l = renderer.insert_mesh(box_mesh([-5.5, 10.0, z], [0.65, 10.0, 0.65]));
            let _ = v3_demo_common::insert_object(&mut renderer, l, mat, glam::Mat4::IDENTITY, 10.0);
            let r = renderer.insert_mesh(box_mesh([ 5.5, 10.0, z], [0.65, 10.0, 0.65]));
            let _ = v3_demo_common::insert_object(&mut renderer, r, mat, glam::Mat4::IDENTITY, 10.0);
            [l, r]
        }).collect();

        // Altar: at far end (z = -26)
        let _altar_step   = renderer.insert_mesh(box_mesh([0.0, 0.2,  -24.5], [5.5, 0.20, 3.0]));
        let _altar_plinth = renderer.insert_mesh(box_mesh([0.0, 0.65, -25.5], [3.0, 0.45, 1.5]));
        let _cross_vert   = renderer.insert_mesh(box_mesh([0.0, 3.2,  -25.8], [0.18, 2.2, 0.18]));
        let _cross_horiz  = renderer.insert_mesh(box_mesh([0.0, 4.5,  -25.8], [1.0,  0.18, 0.18]));
        let _ = v3_demo_common::insert_object(&mut renderer, _altar_step, mat, glam::Mat4::IDENTITY, 5.5);
        let _ = v3_demo_common::insert_object(&mut renderer, _altar_plinth, mat, glam::Mat4::IDENTITY, 3.0);
        let _ = v3_demo_common::insert_object(&mut renderer, _cross_vert, mat, glam::Mat4::IDENTITY, 2.2);
        let _ = v3_demo_common::insert_object(&mut renderer, _cross_horiz, mat, glam::Mat4::IDENTITY, 1.0);

        // Pews: long narrow rect3d per row, 6 rows each side
        let _pews_left: Vec<MeshId> = (0..PEW_COUNT).map(|i| {
            let z = PEW_Z_START + i as f32 * PEW_Z_STEP;
            let id = renderer.insert_mesh(box_mesh([-3.2, 0.45, z], [1.5, 0.45, 0.5]));
            let _ = v3_demo_common::insert_object(&mut renderer, id, mat, glam::Mat4::IDENTITY, 1.5);
            id
        }).collect();
        let _pews_right: Vec<MeshId> = (0..PEW_COUNT).map(|i| {
            let z = PEW_Z_START + i as f32 * PEW_Z_STEP;
            let id = renderer.insert_mesh(box_mesh([ 3.2, 0.45, z], [1.5, 0.45, 0.5]));
            let _ = v3_demo_common::insert_object(&mut renderer, id, mat, glam::Mat4::IDENTITY, 1.5);
            id
        }).collect();

        // Chandeliers: vertical chain + horizontal ring at each Z
        let chandelier_mat = renderer.insert_material(make_material([0.3, 0.28, 0.25, 1.0], 0.5, 0.8, [0.0, 0.0, 0.0], 0.0));
        let _chandelier_chains: Vec<MeshId> = CHANDELIER_Z.iter().map(|&z| {
            let id = renderer.insert_mesh(box_mesh([0.0, 17.5, z], [0.06, 2.0, 0.06]));
            let _ = v3_demo_common::insert_object(&mut renderer, id, chandelier_mat, glam::Mat4::IDENTITY, 2.0);
            id
        }).collect();
        let _chandelier_rings: Vec<MeshId> = CHANDELIER_Z.iter().map(|&z| {
            let id = renderer.insert_mesh(box_mesh([0.0, 15.2, z], [1.2, 0.12, 1.2]));
            let _ = v3_demo_common::insert_object(&mut renderer, id, chandelier_mat, glam::Mat4::IDENTITY, 1.2);
            id
        }).collect();

        // Register lights (chandelier & candle light_ids stored for per-frame flicker updates)
        let mut chandelier_light_ids = Vec::new();
        for &z in CHANDELIER_Z {
            chandelier_light_ids.push(renderer.insert_light(point_light([0.0_f32, 15.0, z], [1.0, 0.92, 0.78], 8.0, 22.0)));
        }
        // Stained glass shafts — static, no need to store ids
        for &(x, y, z, r, g, b) in GLASS_LIGHTS {
            let _ = renderer.insert_light(point_light([x, y, z], [r, g, b], 1.8, 8.0));
        }
        let mut candle_light_ids = Vec::new();
        for &(x, y, z) in CANDLES {
            candle_light_ids.push(renderer.insert_light(point_light([x, y, z], [1.0, 0.6, 0.15], 1.2, 4.0)));
        }
        renderer.set_ambient([0.65, 0.7, 0.85], 0.015);
        renderer.set_clear_color([0.0, 0.0, 0.0, 1.0]);

        self.state = Some(AppState {
            window, surface, device, surface_format: format, renderer,
            last_frame: std::time::Instant::now(),
            _floor, _nave_ceiling, _aisle_ceil_l, _aisle_ceil_r,
            _wall_left_outer, _wall_right_outer, _wall_front, _wall_back,
            _colonnade_l, _colonnade_r, _columns,
            _altar_plinth, _altar_step, _cross_vert, _cross_horiz,
            _pews_left, _pews_right,
            _chandelier_chains, _chandelier_rings,
            // Start at entrance, looking toward the altar
            cam_pos: glam::Vec3::new(0.0, 2.0, 24.0),
            cam_yaw: std::f32::consts::PI, cam_pitch: -0.05,
            keys: HashSet::new(), cursor_grabbed: false, mouse_delta: (0.0, 0.0),
            chandelier_light_ids,
            candle_light_ids,
            start_time: std::time::Instant::now(),
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
                    width: s.width, height: s.height, present_mode: wgpu::PresentMode::Fifo,
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
        const SPEED: f32 = 5.0;
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
        let time   = self.start_time.elapsed().as_secs_f32();

        let camera = Camera::perspective_look_at(
            self.cam_pos, self.cam_pos + forward, glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4, aspect, 0.1, 200.0,
        );

        // Chandeliers flicker slightly
        let flicker = 1.0 + (time * 9.1).sin() * 0.03 + (time * 5.7).cos() * 0.02;
        // Candle flicker — more pronounced
        let cflicker = 1.0 + (time * 14.3).sin() * 0.07 + (time * 8.9).cos() * 0.05;

        // Update flickering chandelier intensities
        for (i, &id) in self.chandelier_light_ids.iter().enumerate() {
            let z = CHANDELIER_Z[i];
            let _ = self.renderer.update_light(id, point_light([0.0_f32, 15.0, z], [1.0, 0.92, 0.78], 8.0 * flicker, 22.0));
        }
        // Update flickering candle intensities
        for (i, &id) in self.candle_light_ids.iter().enumerate() {
            let (x, y, z) = CANDLES[i];
            let _ = self.renderer.update_light(id, point_light([x, y, z], [1.0, 0.6, 0.15], 1.2 * cflicker, 4.0));
        }

        // Scene state is persistent — no per-frame setup needed.

        let output = match self.surface.get_current_texture() {
            Ok(t) => t, Err(e) => { log::warn!("Surface: {:?}", e); return; }
        };
        let view = output.texture.create_view(&Default::default());

        if let Err(e) = self.renderer.render(&camera, &view) {
            log::error!("Render: {:?}", e);
        }
        output.present();
    }
}
