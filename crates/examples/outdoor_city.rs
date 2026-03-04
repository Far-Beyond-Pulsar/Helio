//! Outdoor city example – high complexity
//!
//! A dense downtown city block at dusk: 21 buildings of varying heights
//! arranged across two city blocks, 10 sodium streetlamps lining the main
//! avenue, 4 neon signs on landmark buildings, sidewalk strips, a central
//! plaza with a fountain base, and a physical sky with a low controllable sun
//! casting long building shadows.
//!
//! The RC global illumination bounces warm sunset colour between building
//! facades for realistic inter-building colour bleeding.
//!
//! Controls:
//!   WASD        — move forward/left/back/right
//!   Space/Shift — move up/down
//!   Q/E         — rotate sun (time of day)
//!   Mouse drag  — look around (click to grab cursor)
//!   Escape      — release cursor / exit

use helio_render_v2::{
    Renderer, RendererConfig, Camera, GpuMesh, Scene, SceneLight,
    SkyAtmosphere, VolumetricClouds, Skylight,
};
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

/// Buildings: (center_x, center_z, half_w, half_d, half_h)
/// half_h also serves as Y center (so the base sits on the ground plane).
const BUILDINGS: &[(f32, f32, f32, f32, f32)] = &[
    // West city block
    (-16.0, -20.0,  5.0,  4.0, 13.0),
    (-10.0, -24.0,  3.5,  3.0,  9.0),
    (-20.0,  -5.0,  4.5,  7.0,  6.0),
    (-11.0,   7.0,  3.0,  4.0, 15.0),
    (-19.0,  17.0,  4.0,  3.5,  5.0),
    ( -7.0,  22.0,  2.5,  3.0,  8.0),
    // East city block
    ( 16.0, -20.0,  4.5,  4.5, 18.0), // tallest tower
    ( 10.0, -24.0,  3.0,  3.5,  8.0),
    ( 20.0,  -5.0,  4.0,  6.5, 10.0),
    ( 11.0,   7.0,  3.5,  4.0, 12.0),
    ( 19.0,  17.0,  4.5,  4.0,  5.0),
    (  8.0,  22.0,  2.5,  3.5,  7.0),
    // Background skyline towers
    ( -7.0, -33.0,  2.5,  2.5, 24.0),
    (  0.0, -30.0,  5.5,  4.5,  8.0),
    (  7.0, -33.0,  2.5,  2.5, 21.0),
    // Foreground low shops
    ( -6.0,  31.0,  3.5,  2.5,  4.0),
    (  0.0,  29.0,  4.5,  3.0,  3.0),
    (  6.0,  31.0,  3.0,  3.0,  5.5),
    // Central plaza features
    ( -2.5,  -7.0,  1.2,  1.2,  2.5), // kiosk A
    (  2.5,  -7.0,  1.2,  1.2,  2.5), // kiosk B
    (  0.0,   0.0,  1.8,  1.8,  0.45), // fountain plinth
];

/// Streetlamps: (x, z)
const LAMPS: &[(f32, f32)] = &[
    (-4.5, -22.0), ( 4.5, -22.0),
    (-4.5, -12.0), ( 4.5, -12.0),
    (-4.5,  -2.0), ( 4.5,  -2.0),
    (-4.5,   8.0), ( 4.5,   8.0),
    (-4.5,  18.0), ( 4.5,  18.0),
];

/// Neon signs: (x, y, z, r, g, b)
const NEONS: &[(f32, f32, f32, f32, f32, f32)] = &[
    ( 16.0, 14.5, -19.5,  1.0, 0.05, 0.85), // magenta on east tower
    ( -7.0, 20.0, -32.5,  0.05, 0.85, 1.0), // cyan on bg tower
    (-11.0, 12.0,   6.5,  0.1,  1.0,  0.2), // green on west tower
    (  7.0, 17.0, -32.5,  1.0,  0.5,  0.0), // amber on east bg tower
];

// ─────────────────────────────────────────────────────────────────────────────

fn load_sprite() -> (Vec<u8>, u32, u32) {
    let img = image::load_from_memory(include_bytes!("../../spotlight.png"))
        .unwrap_or_else(|_| image::DynamicImage::new_rgba8(1, 1))
        .into_rgba8();
    let (w, h) = img.dimensions();
    (img.into_raw(), w, h)
}

fn load_probe_sprite() -> (Vec<u8>, u32, u32) {
    let img = image::load_from_memory(include_bytes!("../../probe.png"))
        .unwrap_or_else(|_| image::DynamicImage::new_rgba8(1, 1))
        .into_rgba8();
    let (w, h) = img.dimensions();
    (img.into_raw(), w, h)
}

const RC_WORLD_MIN: [f32; 3] = [-35.0, -0.5, -40.0];
const RC_WORLD_MAX: [f32; 3] = [35.0, 30.0, 35.0];

fn probe_billboards(world_min: [f32; 3], world_max: [f32; 3]) -> Vec<helio_render_v2::features::BillboardInstance> {
    use helio_render_v2::features::radiance_cascades::PROBE_DIMS;
    const COLORS: [[f32; 4]; 4] = [
        [0.0, 1.0, 1.0, 0.85],
        [0.0, 1.0, 0.0, 0.80],
        [1.0, 1.0, 0.0, 0.75],
        [1.0, 0.35, 0.0, 0.70],
    ];
    const SIZES: [[f32; 2]; 4] = [
        [0.04, 0.04],
        [0.10, 0.10],
        [0.22, 0.22],
        [0.45, 0.45],
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
                        .with_color(COLORS[c]));
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

    ground: GpuMesh,
    buildings: Vec<GpuMesh>,
    lamp_poles: Vec<GpuMesh>,
    sidewalks: Vec<GpuMesh>,   // 4 strips bordering the main avenue
    road_center: GpuMesh,      // dark road surface

    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),
    sun_angle: f32,

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
                .with_title("Helio – Outdoor City")
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
            required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
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
            .with_feature(BloomFeature::new().with_intensity(0.55).with_threshold(0.95))
            .with_feature(ShadowsFeature::new().with_atlas_size(2048).with_max_lights(4))
            .with_feature(BillboardsFeature::new().with_sprite(sprite_rgba, sprite_w, sprite_h).with_max_instances(5000))
            .with_feature(RadianceCascadesFeature::new()
                .with_world_bounds([-35.0, -0.5, -40.0], [35.0, 30.0, 35.0]))
            .build();

        let renderer = Renderer::new(device.clone(), queue.clone(),
            RendererConfig { width: size.width, height: size.height, surface_format: format, features },
        ).expect("renderer");

        let ground = GpuMesh::plane(&device, [0.0, 0.0, 0.0], 40.0);
        // Buildings: Y-center = half_h so base sits on ground
        let buildings = BUILDINGS.iter().map(|&(cx, cz, hw, hd, hh)| {
            GpuMesh::rect3d(&device, [cx, hh, cz], [hw, hh, hd])
        }).collect();
        // Lamp poles: 5 m tall pole (half_h = 2.75)
        let lamp_poles = LAMPS.iter().map(|&(x, z)| {
            GpuMesh::rect3d(&device, [x, 2.75, z], [0.08, 2.75, 0.08])
        }).collect();
        // Sidewalk strips flanking the avenue (slightly raised)
        let sidewalks = vec![
            GpuMesh::rect3d(&device, [-4.2, 0.04, 0.0], [0.35, 0.04, 32.0]),
            GpuMesh::rect3d(&device, [ 4.2, 0.04, 0.0], [0.35, 0.04, 32.0]),
            GpuMesh::rect3d(&device, [0.0, 0.04, -32.0], [32.0, 0.04, 0.35]),
            GpuMesh::rect3d(&device, [0.0, 0.04,  32.0], [32.0, 0.04, 0.35]),
        ];
        let road_center = GpuMesh::rect3d(&device, [0.0, 0.01, 0.0], [4.0, 0.01, 32.0]);

        self.state = Some(AppState {
            window, surface, device, surface_format: format, renderer,
            last_frame: std::time::Instant::now(),
            ground, buildings, lamp_poles, sidewalks, road_center,
            cam_pos: glam::Vec3::new(0.0, 5.0, 30.0),
            cam_yaw: std::f32::consts::PI, cam_pitch: -0.1,
            keys: HashSet::new(), cursor_grabbed: false, mouse_delta: (0.0, 0.0),
            sun_angle: 0.38, // low golden-hour sun
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
        const SPEED: f32 = 8.0;
        const SENS:  f32 = 0.002;
        const SUN_SPEED: f32 = 0.4;

        if self.keys.contains(&KeyCode::KeyQ) { self.sun_angle -= SUN_SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyE) { self.sun_angle += SUN_SPEED * dt; }

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
            std::f32::consts::FRAC_PI_4, aspect, 0.1, 1000.0, time,
        );

        let sun_dir_v = glam::Vec3::new(self.sun_angle.cos() * 0.35, self.sun_angle.sin(), 0.45).normalize();
        let light_dir = [-sun_dir_v.x, -sun_dir_v.y, -sun_dir_v.z];
        let sun_elev  = sun_dir_v.y.clamp(-1.0, 1.0);
        let sun_lux   = (sun_elev * 3.0).clamp(0.0, 1.0);
        let warmth    = (1.0 - sun_elev).clamp(0.0, 1.0);
        let sun_color = [
            (1.0 + warmth * 0.55_f32).min(1.0),
            (0.72 + sun_elev * 0.28).clamp(0.0, 1.0),
            (0.50 + sun_elev * 0.38).clamp(0.0, 1.0),
        ];

        let output = match self.surface.get_current_texture() {
            Ok(t) => t, Err(e) => { log::warn!("Surface: {:?}", e); return; }
        };
        let view = output.texture.create_view(&Default::default());

        let mut scene = Scene::new()
            .with_sky_atmosphere(
                SkyAtmosphere::new()
                    .with_sun_intensity(22.0)
                    .with_exposure(3.8)
                    .with_mie_g(0.80)
                    .with_clouds(VolumetricClouds::new()
                        .with_coverage(0.25)
                        .with_density(0.6)
                        .with_layer(500.0, 1400.0)
                        .with_wind([1.0, 0.2], 0.06)),
            )
            .with_skylight(Skylight::new().with_intensity(0.08).with_tint([1.0, 0.9, 0.8]))
            .add_light(SceneLight::directional(light_dir, sun_color, (sun_lux * 0.45).max(0.005)));

        // Streetlamps – sodium orange, activate as sun goes down
        let lamp_on = (1.0 - sun_lux).clamp(0.0, 1.0);
        for &(x, z) in LAMPS {
            let p = [x, 5.55, z];
            scene = scene
                .add_light(SceneLight::point(p, [1.0, 0.72, 0.30], 5.5 * lamp_on, 14.0))
                .add_billboard(BillboardInstance::new(p, [0.35, 0.35])
                    .with_color([1.0, 0.72, 0.30, lamp_on]));
        }

        // Neon signs – always on, bloom harder at night
        let neon_boost = 0.6 + lamp_on * 0.4;
        for &(x, y, z, r, g, b) in NEONS {
            let p = [x, y, z];
            scene = scene
                .add_light(SceneLight::point(p, [r, g, b], 5.0 * neon_boost, 12.0))
                .add_billboard(BillboardInstance::new(p, [0.7, 0.25])
                    .with_color([r, g, b, 1.0]));
        }

        // Geometry
        scene = scene.add_object(self.ground.clone()).add_object(self.road_center.clone());
        for m in &self.sidewalks   { scene = scene.add_object(m.clone()); }
        for m in &self.buildings   { scene = scene.add_object(m.clone()); }
        for m in &self.lamp_poles  { scene = scene.add_object(m.clone()); }

        if let Err(e) = self.renderer.render_scene(&scene, &camera, &view, dt) {
            log::error!("Render: {:?}", e);
        }
        output.present();
    }
}
