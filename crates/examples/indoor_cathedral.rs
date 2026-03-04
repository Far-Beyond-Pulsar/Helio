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

use helio_render_v2::{Renderer, RendererConfig, Camera, GpuMesh, Scene, SceneLight};
use helio_render_v2::features::{
    FeatureRegistry, LightingFeature, BloomFeature, ShadowsFeature,
    BillboardsFeature, BillboardInstance, RadianceCascadesFeature,
};
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

fn load_sprite() -> (Vec<u8>, u32, u32) {
    let img = image::load_from_memory(include_bytes!("../../spotlight.png"))
        .unwrap_or_else(|_| image::DynamicImage::new_rgba8(1, 1))
        .into_rgba8();
    let (w, h) = img.dimensions();
    (img.into_raw(), w, h)
}

#[allow(dead_code)]
fn load_probe_sprite() -> (Vec<u8>, u32, u32) {
    let img = image::load_from_memory(include_bytes!("../../probe.png"))
        .unwrap_or_else(|_| image::DynamicImage::new_rgba8(1, 1))
        .into_rgba8();
    let (w, h) = img.dimensions();
    (img.into_raw(), w, h)
}

const RC_WORLD_MIN: [f32; 3] = [-12.0, -0.1, -30.0];
const RC_WORLD_MAX: [f32; 3] = [12.0, 22.0, 30.0];

fn probe_billboards(world_min: [f32; 3], world_max: [f32; 3]) -> Vec<helio_render_v2::features::BillboardInstance> {
    use helio_render_v2::features::radiance_cascades::PROBE_DIMS;
    const COLORS: [[f32; 4]; 4] = [
        [0.0, 1.0, 1.0, 0.85],
        [0.0, 1.0, 0.0, 0.80],
        [1.0, 1.0, 0.0, 0.75],
        [1.0, 0.35, 0.0, 0.70],
    ];
    // screen_scale=true: sizes are angular (multiplied by distance), giving constant apparent size
    const SIZES: [[f32; 2]; 4] = [
        [0.035, 0.035],  // cascade 0 — finest (4096 probes) — tiny dots
        [0.075, 0.075],  // cascade 1
        [0.140, 0.140],  // cascade 2
        [0.260, 0.260],  // cascade 3 — coarsest (8 probes) — large markers
    ];
    let mut out = Vec::new();
    for (c, &dim) in PROBE_DIMS.iter().enumerate() {
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    let x = world_min[0] + (i as f32 + 0.5) / dim as f32 * (world_max[0] - world_min[0]);
                    let y = world_min[1] + (j as f32 + 0.5) / dim as f32 * (world_max[1] - world_min[1]);
                    let z = world_min[2] + (k as f32 + 0.5) / dim as f32 * (world_max[2] - world_min[2]);
                    out.push(helio_render_v2::features::BillboardInstance::new([x, y, z], SIZES[c])
                        .with_color(COLORS[c])
                        .with_screen_scale(true));
                }
            }
        }
    }
    out
}

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
    floor: GpuMesh,
    nave_ceiling: GpuMesh,
    aisle_ceil_l: GpuMesh,
    aisle_ceil_r: GpuMesh,
    wall_left_outer:  GpuMesh,
    wall_right_outer: GpuMesh,
    wall_front:  GpuMesh,
    wall_back:   GpuMesh,
    // Colonnade arches (inner walls between nave and aisles, with gaps left for columns)
    colonnade_l: Vec<GpuMesh>, // wall segments between columns
    colonnade_r: Vec<GpuMesh>,
    // Columns
    columns: Vec<GpuMesh>,
    // Altar
    altar_plinth: GpuMesh,
    altar_step:   GpuMesh,
    cross_vert:   GpuMesh,
    cross_horiz:  GpuMesh,
    // Pews
    pews_left:  Vec<GpuMesh>,
    pews_right: Vec<GpuMesh>,
    // Chandelier bodies (chain + ring)
    chandelier_chains: Vec<GpuMesh>,
    chandelier_rings:  Vec<GpuMesh>,

    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

    probe_vis: bool,
    sprite_w: u32,
    sprite_h: u32,
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
            required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY | wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
            required_limits: wgpu::Limits::default().using_minimum_supported_acceleration_structure_values(),
            memory_hints: wgpu::MemoryHints::default(),
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            trace: wgpu::Trace::Off,
        })).expect("device");
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

        let (sprite_rgba, sprite_w, sprite_h) = load_sprite();
        let features = FeatureRegistry::builder()
            .with_feature(LightingFeature::new())
            .with_feature(BloomFeature::new().with_intensity(0.6).with_threshold(0.85))
            .with_feature(ShadowsFeature::new().with_atlas_size(2048).with_max_lights(4))
            .with_feature(BillboardsFeature::new().with_sprite(sprite_rgba, sprite_w, sprite_h).with_max_instances(5000))
            .with_feature(RadianceCascadesFeature::new()
                .with_world_bounds([-12.0, -0.1, -30.0], [12.0, 22.0, 30.0]))
            .build();

        let renderer = Renderer::new(device.clone(), queue.clone(),
            RendererConfig { width: size.width, height: size.height, surface_format: format, features },
        ).expect("renderer");

        // Nave + aisles: total width = 22m (x: -11..+11), length = 60m (z: -28..+28), height = 21m
        let floor         = GpuMesh::plane(&device, [0.0, 0.0, 0.0], 11.0);
        let nave_ceiling  = GpuMesh::rect3d(&device, [0.0, 21.0,  0.0], [6.0, 0.18, 28.0]);
        let aisle_ceil_l  = GpuMesh::rect3d(&device, [-8.5, 11.0, 0.0], [2.5, 0.15, 28.0]);
        let aisle_ceil_r  = GpuMesh::rect3d(&device, [ 8.5, 11.0, 0.0], [2.5, 0.15, 28.0]);
        let wall_left_outer  = GpuMesh::rect3d(&device, [-11.0, 7.0,  0.0], [0.25, 7.0, 28.0]);
        let wall_right_outer = GpuMesh::rect3d(&device, [ 11.0, 7.0,  0.0], [0.25, 7.0, 28.0]);
        let wall_front  = GpuMesh::rect3d(&device, [0.0, 10.5, 28.0],  [11.0, 10.5, 0.25]);
        let wall_back   = GpuMesh::rect3d(&device, [0.0, 10.5, -28.0], [11.0, 10.5, 0.25]);

        // Colonnade: short wall segments between columns (between column z-positions)
        // 7 segments per side: before first col, between each pair, after last col
        let col_z_all: Vec<f32> = {
            let mut v = vec![-28.0_f32]; // south wall
            v.extend_from_slice(COLUMN_Z);
            v.push(28.0); // north wall
            v
        };
        let colonnade_l: Vec<GpuMesh> = col_z_all.windows(2).map(|w| {
            let mid_z = (w[0] + w[1]) * 0.5;
            let half_len = (w[1] - w[0]) * 0.5 - 0.9; // gap for column
            GpuMesh::rect3d(&device, [-5.5, 5.5, mid_z], [0.25, 5.5, half_len.max(0.1)])
        }).collect();
        let colonnade_r: Vec<GpuMesh> = col_z_all.windows(2).map(|w| {
            let mid_z = (w[0] + w[1]) * 0.5;
            let half_len = (w[1] - w[0]) * 0.5 - 0.9;
            GpuMesh::rect3d(&device, [ 5.5, 5.5, mid_z], [0.25, 5.5, half_len.max(0.1)])
        }).collect();

        // Columns: 0.65 m square, 20 m tall, at x = ±5.5
        let columns: Vec<GpuMesh> = COLUMN_Z.iter().flat_map(|&z| {
            [
                GpuMesh::rect3d(&device, [-5.5, 10.0, z], [0.65, 10.0, 0.65]),
                GpuMesh::rect3d(&device, [ 5.5, 10.0, z], [0.65, 10.0, 0.65]),
            ]
        }).collect();

        // Altar: at far end (z = -26)
        let altar_step   = GpuMesh::rect3d(&device, [0.0, 0.2,  -24.5], [5.5, 0.20, 3.0]);
        let altar_plinth = GpuMesh::rect3d(&device, [0.0, 0.65, -25.5], [3.0, 0.45, 1.5]);
        let cross_vert   = GpuMesh::rect3d(&device, [0.0, 3.2,  -25.8], [0.18, 2.2, 0.18]);
        let cross_horiz  = GpuMesh::rect3d(&device, [0.0, 4.5,  -25.8], [1.0,  0.18, 0.18]);

        // Pews: long narrow rect3d per row, 6 rows each side
        let pews_left: Vec<GpuMesh> = (0..PEW_COUNT).map(|i| {
            let z = PEW_Z_START + i as f32 * PEW_Z_STEP;
            GpuMesh::rect3d(&device, [-3.2, 0.45, z], [1.5, 0.45, 0.5])
        }).collect();
        let pews_right: Vec<GpuMesh> = (0..PEW_COUNT).map(|i| {
            let z = PEW_Z_START + i as f32 * PEW_Z_STEP;
            GpuMesh::rect3d(&device, [ 3.2, 0.45, z], [1.5, 0.45, 0.5])
        }).collect();

        // Chandeliers: vertical chain + horizontal ring at each Z
        let chandelier_chains: Vec<GpuMesh> = CHANDELIER_Z.iter().map(|&z| {
            GpuMesh::rect3d(&device, [0.0, 17.5, z], [0.06, 2.0, 0.06])
        }).collect();
        let chandelier_rings: Vec<GpuMesh> = CHANDELIER_Z.iter().map(|&z| {
            GpuMesh::rect3d(&device, [0.0, 15.2, z], [1.2, 0.12, 1.2])
        }).collect();

        self.state = Some(AppState {
            window, surface, device, surface_format: format, renderer,
            last_frame: std::time::Instant::now(),
            floor, nave_ceiling, aisle_ceil_l, aisle_ceil_r,
            wall_left_outer, wall_right_outer, wall_front, wall_back,
            colonnade_l, colonnade_r, columns,
            altar_plinth, altar_step, cross_vert, cross_horiz,
            pews_left, pews_right,
            chandelier_chains, chandelier_rings,
            // Start at entrance, looking toward the altar
            cam_pos: glam::Vec3::new(0.0, 2.0, 24.0),
            cam_yaw: std::f32::consts::PI, cam_pitch: -0.05,
            keys: HashSet::new(), cursor_grabbed: false, mouse_delta: (0.0, 0.0),
            probe_vis: false,
            sprite_w,
            sprite_h,
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
                state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::Digit3), ..
            }, .. } => {
                state.probe_vis = !state.probe_vis;
                let raw: &[u8] = if state.probe_vis {
                    include_bytes!("../../probe.png")
                } else {
                    include_bytes!("../../spotlight.png")
                };
                let img = image::load_from_memory(raw)
                    .unwrap_or_else(|_| image::DynamicImage::new_rgba8(state.sprite_w, state.sprite_h))
                    .resize_exact(state.sprite_w, state.sprite_h, image::imageops::FilterType::Triangle)
                    .into_rgba8();
                if let Some(bb) = state.renderer.get_feature_mut::<BillboardsFeature>("billboards") {
                    bb.set_sprite(img.into_raw(), state.sprite_w, state.sprite_h);
                }
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
                state.renderer.resize(s.width, s.height);
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
        let time   = self.renderer.frame_count() as f32 * 0.016;

        let camera = Camera::perspective(
            self.cam_pos, self.cam_pos + forward, glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4, aspect, 0.1, 200.0, time,
        );

        // Chandeliers flicker slightly
        let flicker = 1.0 + (time * 9.1).sin() * 0.03 + (time * 5.7).cos() * 0.02;
        // Candle flicker — more pronounced
        let cflicker = 1.0 + (time * 14.3).sin() * 0.07 + (time * 8.9).cos() * 0.05;

        let output = match self.surface.get_current_texture() {
            Ok(t) => t, Err(e) => { log::warn!("Surface: {:?}", e); return; }
        };
        let view = output.texture.create_view(&Default::default());

        let mut scene = Scene::new()
            .with_sky([0.0, 0.0, 0.0])
            .with_ambient([0.65, 0.7, 0.85], 0.015); // very dim cold-stone fill

        // Chandeliers – warm white
        for &z in CHANDELIER_Z {
            let p = [0.0_f32, 15.0, z];
            scene = scene
                .add_light(SceneLight::point(p, [1.0, 0.92, 0.78], 8.0 * flicker, 22.0));
        }

        // Stained glass shafts – coloured point lights from windows
        for &(x, y, z, r, g, b) in GLASS_LIGHTS {
            // Low intensity: these colour the stone without overwhelming it
            scene = scene.add_light(SceneLight::point([x, y, z], [r, g, b], 1.8, 8.0));
        }

        // Candles near altar
        for &(x, y, z) in CANDLES {
            let p = [x, y, z];
            scene = scene
                .add_light(SceneLight::point(p, [1.0, 0.6, 0.15], 1.2 * cflicker, 4.0));
        }

        // Geometry
        scene = scene
            .add_object(self.floor.clone())
            .add_object(self.nave_ceiling.clone())
            .add_object(self.aisle_ceil_l.clone())
            .add_object(self.aisle_ceil_r.clone())
            .add_object(self.wall_left_outer.clone())
            .add_object(self.wall_right_outer.clone())
            .add_object(self.wall_front.clone())
            .add_object(self.wall_back.clone())
            .add_object(self.altar_plinth.clone())
            .add_object(self.altar_step.clone())
            .add_object(self.cross_vert.clone())
            .add_object(self.cross_horiz.clone());

        for m in &self.colonnade_l     { scene = scene.add_object(m.clone()); }
        for m in &self.colonnade_r     { scene = scene.add_object(m.clone()); }
        for m in &self.columns         { scene = scene.add_object(m.clone()); }
        for m in &self.pews_left       { scene = scene.add_object(m.clone()); }
        for m in &self.pews_right      { scene = scene.add_object(m.clone()); }
        for m in &self.chandelier_chains { scene = scene.add_object(m.clone()); }
        for m in &self.chandelier_rings  { scene = scene.add_object(m.clone()); }

        if self.probe_vis {
            for b in probe_billboards(RC_WORLD_MIN, RC_WORLD_MAX) {
                scene = scene.add_billboard(b);
            }
        } else {
            for &z in CHANDELIER_Z {
                let p = [0.0_f32, 15.0, z];
                scene = scene
                    .add_billboard(BillboardInstance::new(p, [0.4, 0.4])
                        .with_color([1.0, 0.92, 0.78, 1.0]));
            }
            for &(x, y, z) in CANDLES {
                let p = [x, y, z];
                scene = scene
                    .add_billboard(BillboardInstance::new(p, [0.15, 0.15])
                        .with_color([1.0, 0.6, 0.15, 1.0]));
            }
        }

        if let Err(e) = self.renderer.render_scene(&scene, &camera, &view, dt) {
            log::error!("Render: {:?}", e);
        }
        output.present();
    }
}
