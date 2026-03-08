//! Space Station — the most complex Helio example scene.
//!
//! A massive orbital station assembled from hundreds of axis-aligned primitives
//! arranged with trigonometry into rings, spokes, solar arrays, and engine pods.
//!
//! Station layout (Y-up, station forward = −Z):
//!   • Central hub cylinder (r=7, 40m tall) with command tower & engineering bay
//!   • Habitat Ring A (r=35, 20 modules) — crew quarters
//!   • Industrial Ring B (r=62, 16 modules) — labs and manufacturing
//!   • 4 Solar Array Arms (45°,135°,225°,315°) with truss, panels, radiators
//!   • Forward docking arm + node (along −Z)
//!   • Engine cluster, 4 pods, micro-thrusters (along +Z)
//!   • 8 attitude-control thruster pods on hub equator
//!
//! Controls:
//!   WASD / Space / Shift  — fly  (speed 40 m/s)
//!   Mouse drag            — look (click to grab cursor)
//!   Escape                — release cursor / exit
//!   3                     — toggle RC probe visualization



mod demo_portal;

use helio_render_v2::{Renderer, RendererConfig, Camera, GpuMesh, SceneLight, SceneEnv};


use helio_render_v2::features::{
    FeatureRegistry, LightingFeature, BloomFeature, ShadowsFeature,
    BillboardsFeature, BillboardInstance, RadianceCascadesFeature,
};


use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId, CursorGrabMode},
};


use std::collections::HashSet;


use std::sync::Arc;

const RC_WORLD_MIN: [f32; 3] = [-130.0, -45.0, -75.0];
const RC_WORLD_MAX: [f32; 3] = [ 130.0,  65.0,  70.0];

const PI:  f32 = std::f32::consts::PI;
const TAU: f32 = std::f32::consts::TAU;

// ── Sprite helpers ────────────────────────────────────────────────────────────

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

fn probe_billboards(world_min: [f32; 3], world_max: [f32; 3]) -> Vec<BillboardInstance> {
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
                    out.push(BillboardInstance::new([x, y, z], SIZES[c])
                        .with_color(COLORS[c])
                        .with_screen_scale(true));
                }
            }
        }
    }
    out
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    env_logger::init();
    log::info!("Starting Space Station example");
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("event loop run");
}

struct App { state: Option<AppState> }
impl App { fn new() -> Self { Self { state: None } } }

struct AppState {
    window:         Arc<Window>,
    surface:        wgpu::Surface<'static>,
    device:         Arc<wgpu::Device>,
    surface_format: wgpu::TextureFormat,
    renderer:       Renderer,
    last_frame:     std::time::Instant,
    meshes:         Vec<GpuMesh>,
    // camera
    cam_pos:        glam::Vec3,
    cam_yaw:        f32,
    cam_pitch:      f32,
    keys:           HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta:    (f32, f32),
    // probe vis
    probe_vis:      bool,
    sprite_w:       u32,
    sprite_h:       u32,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }

        let window = Arc::new(
            event_loop.create_window(
                Window::default_attributes()
                    .with_title("Helio — Space Station  |  WASD fly  |  3: probes")
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
                label: Some("Main Device"),
                required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY | wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
                required_limits: wgpu::Limits::default()
                    .using_minimum_supported_acceleration_structure_values(),
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                trace: wgpu::Trace::Off,
            },
        )).expect("device");
        let device = Arc::new(device);
        let queue  = Arc::new(queue);

        let caps   = surface.get_capabilities(&adapter);
        let fmt    = caps.formats.iter().find(|f| f.is_srgb()).copied().unwrap_or(caps.formats[0]);
        let size   = window.inner_size();
        surface.configure(&device, &wgpu::SurfaceConfiguration {
            usage:  wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: fmt, width: size.width, height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![], desired_maximum_frame_latency: 2,
        });

        let (sprite_rgba, sprite_w, sprite_h) = load_sprite();
        let features = FeatureRegistry::builder()
            .with_feature(LightingFeature::new())
            .with_feature(BloomFeature::new().with_intensity(0.7).with_threshold(0.8))
            .with_feature(ShadowsFeature::new().with_atlas_size(2048).with_max_lights(4))
            .with_feature(BillboardsFeature::new()
                .with_sprite(sprite_rgba, sprite_w, sprite_h)
                .with_max_instances(5000))
            .with_feature(RadianceCascadesFeature::new()
                .with_world_bounds(RC_WORLD_MIN, RC_WORLD_MAX))
            .build();

        let mut renderer = Renderer::new(device.clone(), queue.clone(), RendererConfig::new(
            size.width, size.height, fmt, features
        )).expect("renderer");

        let meshes = build_station(&device);
        log::info!("Space station: {} meshes", meshes.len());
        demo_portal::enable_live_dashboard(&mut renderer);

        for mesh in &meshes { renderer.add_object(mesh, None, glam::Mat4::IDENTITY); }

        self.state = Some(AppState {
            window, surface, device, surface_format: fmt, renderer,
            last_frame: std::time::Instant::now(),
            meshes,
            cam_pos:   glam::Vec3::new(0.0, 55.0, 175.0),
            cam_yaw:   0.0,
            cam_pitch: -0.18,
            keys:           HashSet::new(),
            cursor_grabbed: false,
            mouse_delta:    (0.0, 0.0),
            probe_vis: false, sprite_w, sprite_h,
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
                event: KeyEvent { state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::Digit3), .. }, ..
            } => {
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

            WindowEvent::KeyboardInput {
                event: KeyEvent { state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::Digit4), .. }, ..
            } => { state.renderer.debug_key_pressed(); }

            WindowEvent::KeyboardInput {
                event: KeyEvent { state: ks, physical_key: PhysicalKey::Code(key), .. }, ..
            } => { match ks {
                ElementState::Pressed  => { state.keys.insert(key); }
                ElementState::Released => { state.keys.remove(&key); }
            }}

            WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. } => {
                if !state.cursor_grabbed {
                    let ok = state.window.set_cursor_grab(CursorGrabMode::Confined)
                        .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked))
                        .is_ok();
                    if ok { state.window.set_cursor_visible(false); state.cursor_grabbed = true; }
                }
            }

            WindowEvent::Resized(sz) if sz.width > 0 && sz.height > 0 => {
                state.surface.configure(&state.device, &wgpu::SurfaceConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format: state.surface_format,
                    width: sz.width, height: sz.height,
                    present_mode: wgpu::PresentMode::Fifo,
                    alpha_mode: wgpu::CompositeAlphaMode::Auto,
                    view_formats: vec![], desired_maximum_frame_latency: 2,
                });
                state.renderer.resize(sz.width, sz.height);
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

// ── Per-frame render ──────────────────────────────────────────────────────────

impl AppState {
    fn render(&mut self, dt: f32) {
        const SPEED: f32 = 40.0;
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
        let time   = self.renderer.frame_count() as f32 * 0.016;

        let camera = Camera::perspective(
            self.cam_pos, self.cam_pos + fwd, glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4, aspect, 0.5, 1000.0, time,
        );

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(e) => { log::warn!("surface: {:?}", e); return; }
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Animated engine flicker
        let flicker = 1.0 + 0.18 * (time * 8.7).sin() * (time * 13.3).cos();
        // Slow station-wide pulse
        let pulse   = 1.0 + 0.06 * (time * 0.7).sin();
        // Red warning beacon (1 Hz strobe)
        let beacon  = (0.5 + 0.5 * (time * 1.0 * TAU).sin()).max(0.0);

        let billboards = if self.probe_vis {
            probe_billboards(RC_WORLD_MIN, RC_WORLD_MAX)
        } else {
            let mut bb = Vec::new();
            // Engine glow (4 pods)
            for &[ex, ey] in &[[5.0f32, 5.0], [-5.0, 5.0], [5.0, -5.0], [-5.0, -5.0]] {
                bb.push(BillboardInstance::new([ex, ey, 60.5], [2.2, 2.2]).with_color([1.0, 0.52, 0.10, 1.0]));
            }
            // Hub lights
            bb.push(BillboardInstance::new([0.0,  14.0, 0.0], [0.5, 0.5]).with_color([0.82, 0.90, 1.0, 0.9]));
            bb.push(BillboardInstance::new([0.0,  -9.0, 0.0], [0.4, 0.4]).with_color([0.70, 0.80, 1.0, 0.9]));
            // Docking floodlight
            bb.push(BillboardInstance::new([0.0, 0.0, -55.0], [0.7, 0.7]).with_color([1.0, 1.0, 0.92, 0.9]));
            // Hab ring A cardinal lights
            for &[bx, bz] in &[[35.0f32, 0.0], [-35.0, 0.0], [0.0, 35.0], [0.0, -35.0]] {
                bb.push(BillboardInstance::new([bx, 7.0, bz], [0.4, 0.4]).with_color([0.78, 0.88, 1.0, 0.85]));
            }
            // Warning beacons (red strobe)
            let b_alpha = beacon.max(0.05);
            bb.push(BillboardInstance::new([0.0, 7.5,  65.0], [0.55, 0.55]).with_color([1.0, 0.08, 0.08, b_alpha]));
            bb.push(BillboardInstance::new([0.0, 7.5, -65.0], [0.55, 0.55]).with_color([1.0, 0.08, 0.08, b_alpha]));
            bb
        };

        let env = SceneEnv {
            lights: vec![
                // ── Distant cold starlight ─────────────────────────────────────
                SceneLight::directional([0.35, -0.65, 0.25], [0.72, 0.82, 1.0], 0.10),
                // ── Hub interior ───────────────────────────────────────────────
                SceneLight::point([0.0,  14.0, 0.0], [0.82, 0.90, 1.0], 8.0 * pulse, 28.0),
                SceneLight::point([0.0,  -9.0, 0.0], [0.70, 0.80, 1.0], 6.0 * pulse, 22.0),
                // ── Hab Ring A — 4 cardinal lights ────────────────────────────
                SceneLight::point([ 35.0, 6.0,  0.0], [0.78, 0.88, 1.0], 5.5 * pulse, 20.0),
                SceneLight::point([-35.0, 6.0,  0.0], [0.78, 0.88, 1.0], 5.5 * pulse, 20.0),
                SceneLight::point([  0.0, 6.0,  35.0], [0.78, 0.88, 1.0], 5.5 * pulse, 20.0),
                SceneLight::point([  0.0, 6.0, -35.0], [0.78, 0.88, 1.0], 5.5 * pulse, 20.0),
                // ── Industrial Ring B — 4 cardinal lights ─────────────────────
                SceneLight::point([ 62.0, -3.0,  0.0], [0.62, 0.76, 1.0], 7.0, 28.0),
                SceneLight::point([-62.0, -3.0,  0.0], [0.62, 0.76, 1.0], 7.0, 28.0),
                SceneLight::point([  0.0, -3.0,  62.0], [0.62, 0.76, 1.0], 7.0, 28.0),
                SceneLight::point([  0.0, -3.0, -62.0], [0.62, 0.76, 1.0], 7.0, 28.0),
                // ── Engine pods — orange glow ──────────────────────────────────
                SceneLight::point([ 5.0,  5.0, 58.0], [1.0, 0.42, 0.06], 10.0 * flicker, 22.0),
                SceneLight::point([-5.0,  5.0, 58.0], [1.0, 0.42, 0.06], 10.0 * flicker, 22.0),
                SceneLight::point([ 5.0, -5.0, 58.0], [1.0, 0.42, 0.06], 10.0 * flicker, 22.0),
                SceneLight::point([-5.0, -5.0, 58.0], [1.0, 0.42, 0.06], 10.0 * flicker, 22.0),
                // ── Docking node floodlight ────────────────────────────────────
                SceneLight::point([0.0, 0.0, -54.0], [1.0, 1.0, 0.92], 7.5, 26.0),
                // ── Warning beacons on ring B (strobe) ────────────────────────
                SceneLight::point([0.0, 6.0,  65.0], [1.0, 0.04, 0.04], 6.0 * beacon, 14.0),
                SceneLight::point([0.0, 6.0, -65.0], [1.0, 0.04, 0.04], 6.0 * beacon, 14.0),
            ],
            ambient_color: [0.08, 0.10, 0.18],
            ambient_intensity: 0.035,
            sky_color: [0.003, 0.005, 0.015],
            billboards,
            ..Default::default()
        };
        self.renderer.set_scene_env(env);
        if let Err(e) = self.renderer.render(&camera, &view, dt) {
            log::error!("render error: {:?}", e);
        }
        output.present();
    }
}

// ── Station geometry builder ──────────────────────────────────────────────────

fn build_station(dev: &Arc<wgpu::Device>) -> Vec<GpuMesh> {
    let mut m = Vec::<GpuMesh>::new();

    // helper: push a rect3d
    macro_rules! box3 {
        ($cx:expr, $cy:expr, $cz:expr, $hx:expr, $hy:expr, $hz:expr) => {
            m.push(GpuMesh::rect3d(dev, [$cx, $cy, $cz], [$hx, $hy, $hz]))
        };
    }

    // ── 1. PRIMARY HUB CYLINDER  (r=7, Y: −15 → +25) ─────────────────────
    let hub_r      = 7.0_f32;
    let hub_mid_y  = 5.0_f32;     // centre of 40-unit tall cylinder
    let hub_half_h = 20.0_f32;
    let n_hub      = 16_u32;
    for i in 0..n_hub {
        let a     = i as f32 * TAU / n_hub as f32;
        let cx    = hub_r * a.cos();
        let cz    = hub_r * a.sin();
        let chord = 2.0 * hub_r * (PI / n_hub as f32).sin();
        box3!(cx, hub_mid_y, cz,  chord * 0.5 + 0.2, hub_half_h, 0.55);
    }
    // top/bottom caps
    box3!(0.0, hub_mid_y + hub_half_h + 0.6, 0.0,  6.5, 0.6, 6.5);
    box3!(0.0, hub_mid_y - hub_half_h - 0.6, 0.0,  7.5, 0.6, 7.5);
    // interior deck plates
    for deck in 0..4_i32 {
        let dy = -12.0 + deck as f32 * 9.5;
        box3!(0.0, dy, 0.0,  5.2, 0.18, 5.2);
    }

    // ── 2. COMMAND TOWER (Y=25 upward) ────────────────────────────────────
    let cmd_y0 = hub_mid_y + hub_half_h; // 25.0
    // 4 tapering octagonal sections
    for i in 0..4_u32 {
        let s  = 4.0 - i as f32 * 0.75;
        let cy = cmd_y0 + 4.5 + i as f32 * 5.5;
        box3!(0.0, cy, 0.0,  s, 2.2, s);
        // bevel cut each side (chamfered-feel)
        box3!(s * 0.7, cy, s * 0.7,  s * 0.25, 2.0, s * 0.25);
        box3!(-s * 0.7, cy, s * 0.7,  s * 0.25, 2.0, s * 0.25);
    }
    // antenna stalk
    box3!(0.0, cmd_y0 + 38.0, 0.0,  0.22, 7.0, 0.22);
    // comm dish (large flat disc approx)
    box3!(0.0, cmd_y0 + 46.0, 0.0,  5.0, 0.22, 5.0);
    // 8 dish ribs
    for i in 0..8_u32 {
        let a = i as f32 * TAU / 8.0;
        let dr = 4.0_f32;
        box3!(dr * a.cos(), cmd_y0 + 45.8, dr * a.sin(),  0.28, 0.28, 0.28);
    }
    // 4 lateral sensor masts
    for i in 0..4_u32 {
        let a  = i as f32 * PI * 0.5;
        let ax = 2.8 * a.cos();
        let az = 2.8 * a.sin();
        box3!(ax, cmd_y0 + 16.0, az,  0.12, 5.5, 0.12);
        box3!(4.5 * a.cos(), cmd_y0 + 18.0, 4.5 * a.sin(),  1.5, 0.12, 0.12);
    }

    // ── 3. ENGINEERING SECTION (below hub, Y ≤ −15) ───────────────────────
    let bot_y = hub_mid_y - hub_half_h; // −15.0
    box3!(0.0, bot_y - 5.0, 0.0,  11.5, 4.5, 11.5);
    // 4 large heat-radiator fins (rotated 45° for visual variety)
    for i in 0..4_u32 {
        let a  = i as f32 * PI * 0.5 + PI * 0.25;
        let rx = 15.5 * a.cos();
        let rz = 15.5 * a.sin();
        box3!(rx, bot_y - 5.0, rz,  0.16, 4.5, 7.0);
    }
    // 3 coolant-conduit rings (12-gon each, shrinking outward)
    for ring in 0..3_u32 {
        let rr   = 8.5 + ring as f32 * 2.0;
        let ry   = bot_y - 2.0 - ring as f32 * 3.5;
        let n    = 12_u32;
        let half = 2.0 * rr * (PI / n as f32).sin() * 0.45;
        for i in 0..n {
            let a  = i as f32 * TAU / n as f32;
            let cx = rr * a.cos();
            let cz = rr * a.sin();
            box3!(cx, ry, cz,  half, 0.4, 0.4);
        }
    }

    // ── 4. ATTITUDE-CONTROL PODS on hub equator (8 pods) ─────────────────
    for i in 0..8_u32 {
        let a  = i as f32 * TAU / 8.0;
        let r  = hub_r + 1.5;
        let px = r * a.cos();
        let pz = r * a.sin();
        box3!(px, hub_mid_y + 2.0,  pz,  1.0, 0.8, 1.0);
        box3!(px, hub_mid_y - 2.0, pz,  1.0, 0.8, 1.0);
        // thruster nozzle
        let nx = (r + 1.2) * a.cos();
        let nz = (r + 1.2) * a.sin();
        box3!(nx, hub_mid_y + 2.0,  nz,  0.35, 0.35, 0.35);
        box3!(nx, hub_mid_y - 2.0, nz,  0.35, 0.35, 0.35);
    }

    // ── 5. HABITAT RING A  (r=35, Y=5, 20 modules) ────────────────────────
    let hab_a_r = 35.0_f32;
    let hab_a_y =  5.0_f32;
    let n_a     = 20_u32;
    for i in 0..n_a {
        let a  = i as f32 * TAU / n_a as f32;
        let cx = hab_a_r * a.cos();
        let cz = hab_a_r * a.sin();
        // Main crew module
        box3!(cx, hab_a_y, cz,  3.0, 2.4, 3.0);
        // Outer window blister
        let wo = hab_a_r + 3.4;
        box3!(wo * a.cos(), hab_a_y + 0.4, wo * a.sin(),  1.0, 0.9, 1.0);
        // Under-module docking ring
        box3!(cx, hab_a_y - 3.0, cz,  1.6, 0.4, 1.6);
    }
    // Outer ring rail (continuous tube approx)
    let rail_a = hab_a_r + 5.2;
    for i in 0..n_a {
        let a0 = i as f32 * TAU / n_a as f32;
        let a1 = (i + 1) as f32 * TAU / n_a as f32;
        let am = (a0 + a1) * 0.5;
        let cx = rail_a * am.cos();
        let cz = rail_a * am.sin();
        let len = (
            (rail_a * a1.cos() - rail_a * a0.cos()).powi(2)
          + (rail_a * a1.sin() - rail_a * a0.sin()).powi(2)
        ).sqrt() * 0.45;
        box3!(cx, hab_a_y, cz,  len, 0.25, 0.25);
    }
    // 8 spokes hub→ring A (3 nodes + cross-braces)
    for i in 0..8_u32 {
        let a = i as f32 * TAU / 8.0;
        for node in 1..4_u32 {
            let r  = node as f32 * hab_a_r / 3.6;
            let s  = 0.65 - node as f32 * 0.12;
            box3!(r * a.cos(), hab_a_y, r * a.sin(),  s, s, s);
        }
        // diagonal cross-brace at mid-span
        let mr   = hab_a_r * 0.55;
        let mx   = mr * a.cos();
        let mz   = mr * a.sin();
        let perp = a + PI * 0.5;
        for sign in [-2.2_f32, 2.2] {
            box3!(mx + sign * perp.cos(), hab_a_y + 1.8, mz + sign * perp.sin(),  0.22, 1.8, 0.22);
        }
    }

    // ── 6. INDUSTRIAL RING B  (r=62, Y=−5, 16 modules) ───────────────────
    let ind_r  = 62.0_f32;
    let ind_y  = -5.0_f32;
    let n_b    = 16_u32;
    for i in 0..n_b {
        let a  = i as f32 * TAU / n_b as f32;
        let cx = ind_r * a.cos();
        let cz = ind_r * a.sin();
        // Large industrial module
        box3!(cx, ind_y, cz,  5.5, 3.5, 5.5);
        // Outer bay extension
        let ox = (ind_r + 7.0) * a.cos();
        let oz = (ind_r + 7.0) * a.sin();
        box3!(ox, ind_y + 1.0, oz,  2.5, 2.5, 2.5);
        // Top utility cupola
        box3!(cx, ind_y + 4.8, cz,  2.8, 0.9, 2.8);
        // Mounting collar
        box3!(cx, ind_y - 4.2, cz,  3.2, 0.4, 3.2);
    }
    // 4 main spokes, 5 nodes each + diagonal braces
    for i in 0..4_u32 {
        let a = i as f32 * PI * 0.5;
        for node in 1..6_u32 {
            let r = node as f32 * ind_r / 5.8;
            box3!(r * a.cos(), ind_y, r * a.sin(),  1.05, 1.05, 1.05);
            if node % 2 == 0 {
                let perp = a + PI * 0.5;
                let bx = r * a.cos();
                let bz = r * a.sin();
                for sign in [-2.8_f32, 2.8] {
                    box3!(bx + sign * perp.cos(), ind_y + 2.2, bz + sign * perp.sin(),
                          0.28, 2.2, 0.28);
                }
            }
        }
    }
    // Outer ring rail for ring B
    let rail_b = ind_r + 6.5;
    for i in 0..n_b {
        let a0 = i as f32 * TAU / n_b as f32;
        let a1 = (i + 1) as f32 * TAU / n_b as f32;
        let am = (a0 + a1) * 0.5;
        let len = (
            (rail_b * a1.cos() - rail_b * a0.cos()).powi(2)
          + (rail_b * a1.sin() - rail_b * a0.sin()).powi(2)
        ).sqrt() * 0.44;
        box3!(rail_b * am.cos(), ind_y, rail_b * am.sin(),  len, 0.32, 0.32);
    }

    // ── 7. SOLAR POWER ARRAYS  (4 arms at 45°, 135°, 225°, 315°) ─────────
    // Each arm starts just outside ring B and extends outward.
    // Structure: truss spine → perpendicular solar panels above/below truss.
    let sol_base = ind_r + 8.0;
    let n_seg    = 5_u32;          // truss nodes per arm
    let seg_step = 14.0_f32;      // metres between nodes
    for arm in 0..4_u32 {
        let arm_a = arm as f32 * PI * 0.5 + PI * 0.25; // 45° offset
        let perp  = arm_a + PI * 0.5;

        for seg in 0..n_seg {
            let r  = sol_base + seg as f32 * seg_step;
            let cx = r * arm_a.cos();
            let cz = r * arm_a.sin();

            // Truss node box
            box3!(cx, 0.0, cz,  1.3, 0.55, 1.3);

            // Upper truss chord
            box3!(cx, 3.5, cz,  0.35, 0.35, 0.35);
            box3!(cx, -3.5, cz, 0.35, 0.35, 0.35);

            // 3 solar panel rows along perp direction (above truss)
            for pi in -1_i32..=1 {
                let pd = pi as f32 * 9.5;
                let px = cx + pd * perp.cos();
                let pz = cz + pd * perp.sin();
                // Panel above arm
                box3!(px,  6.5, pz,  6.5, 0.11, 3.8);
                // Panel below arm
                box3!(px, -6.5, pz,  6.5, 0.11, 3.8);
            }
        }

        // Truss longerons (connecting nodes)
        for seg in 0..(n_seg - 1) {
            let r0 = sol_base + seg as f32 * seg_step + seg_step * 0.5;
            let cx = r0 * arm_a.cos();
            let cz = r0 * arm_a.sin();
            // Main spine longeron
            box3!(cx, 0.0, cz,  0.45, 0.45, 0.45);
            // Upper/lower chord longerons
            box3!(cx,  3.5, cz,  0.28, 0.28, 0.28);
            box3!(cx, -3.5, cz,  0.28, 0.28, 0.28);
        }

        // Radiator panels (heat rejection, thin vertical fins)
        for seg in 0..4_u32 {
            let r  = sol_base + seg as f32 * seg_step + seg_step * 0.5;
            let cx = r * arm_a.cos();
            let cz = r * arm_a.sin();
            // Two vertical radiator panels bracketing the truss
            for sign in [-13.0_f32, 13.0] {
                box3!(cx + sign * perp.cos(), -10.0, cz + sign * perp.sin(),
                      0.14, 8.0, 5.5);
            }
        }

        // Arm tip navigation box + antenna
        let tip_r = sol_base + n_seg as f32 * seg_step;
        let tx = tip_r * arm_a.cos();
        let tz = tip_r * arm_a.sin();
        box3!(tx, 0.0, tz,  2.2, 2.2, 2.2);
        box3!(tx, 4.5, tz,  0.14, 3.0, 0.14);
    }

    // ── 8. FORWARD DOCKING ARM & NODE  (along −Z) ─────────────────────────
    // Three-segment articulated approach arm
    for seg in 0..3_u32 {
        let z = -(13.0 + seg as f32 * 10.5);
        box3!(0.0, 0.0, z,  1.9, 1.9, 4.8);
    }
    // Docking node (large boxy airlock hub)
    box3!(0.0, 0.0, -50.0,  9.5, 9.5, 6.5);
    // 4 lateral docking ports (N/S/E/W orientation)
    for i in 0..4_u32 {
        let a  = i as f32 * PI * 0.5;
        let px = 11.5 * a.cos();
        let py = 11.5 * a.sin();
        box3!(px, py, -50.0,  2.4, 2.4, 4.5);
        // Docking collar ring
        box3!(px, py, -55.0,  3.2, 3.2, 0.55);
        // Approach light bar
        box3!(px * 1.15, py * 1.15, -55.5,  0.3, 0.3, 0.3);
    }
    // Forward sensor boom
    box3!(0.0, 0.0, -59.5,  0.55, 0.55, 5.5);
    // Sensor disc at tip
    box3!(0.0, 0.0, -65.5,  2.8, 0.18, 2.8);

    // ── 9. ENGINE SECTION  (along +Z) ─────────────────────────────────────
    // Central spine
    box3!(0.0, 0.0, 16.0,  2.8, 2.8, 6.5);
    box3!(0.0, 0.0, 28.5,  3.8, 3.8, 6.0);
    // Engine manifold plate
    box3!(0.0, 0.0, 38.0,  9.5, 9.5, 3.8);
    // 4 engine pods (2×2)
    const ENG: [[f32; 2]; 4] = [[5.2, 5.2], [-5.2, 5.2], [5.2, -5.2], [-5.2, -5.2]];
    for [ex, ey] in ENG {
        // Pod body
        box3!(ex, ey, 46.5,  3.2, 3.2, 6.0);
        // Nozzle bell
        box3!(ex, ey, 53.5,  4.5, 4.5, 2.5);
        // 6 micro-thrusters arranged in ring around pod
        for tr in 0..6_u32 {
            let ta = tr as f32 * TAU / 6.0;
            let tx = ex + 5.2 * ta.cos();
            let ty = ey + 5.2 * ta.sin();
            box3!(tx, ty, 49.5,  0.55, 0.55, 2.5);
        }
        // Engine fairing (connecting manifold to pod)
        box3!(ex * 0.55, ey * 0.55, 40.5,  1.6, 1.6, 1.0);
    }
    // Engine cowl rings (4 rings at increasing Z)
    for ring in 0..4_u32 {
        let rz = 42.0 + ring as f32 * 2.5;
        let rr = 8.5 + ring as f32 * 0.8;
        let n  = 16_u32;
        let half = 2.0 * rr * (PI / n as f32).sin() * 0.44;
        for i in 0..n {
            let a  = i as f32 * TAU / n as f32;
            let cx = rr * a.cos();
            let cz = rr * a.sin();
            box3!(cx, cz, rz,  half, 0.32, 0.32); // note: cz in Y-slot here is intentional ring cross-section
        }
    }

    // ── 10. SPINE TRUSS  (long Z-axis structural backbone) ────────────────
    // Connects hub bottom to engine spine, passes engineering section
    for seg in 0..6_u32 {
        let sz = -8.0 + seg as f32 * 4.5;
        box3!(3.5, 0.0, sz,  0.35, 0.35, 0.35);
        box3!(-3.5, 0.0, sz, 0.35, 0.35, 0.35);
        box3!(0.0, 3.5, sz,  0.35, 0.35, 0.35);
        box3!(0.0, -3.5, sz, 0.35, 0.35, 0.35);
    }

    // ── 11. OBSERVATION DECK  (top of command tower) ──────────────────────
    let obs_y = cmd_y0 + 29.0;
    box3!(0.0, obs_y, 0.0,  5.0, 0.9, 5.0);
    // 8 viewport blisters around perimeter
    for i in 0..8_u32 {
        let a  = i as f32 * TAU / 8.0;
        let vr = 5.2_f32;
        box3!(vr * a.cos(), obs_y + 0.9, vr * a.sin(),  0.65, 0.55, 0.65);
    }

    m
}
