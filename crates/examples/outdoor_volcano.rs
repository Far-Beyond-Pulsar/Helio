//! Outdoor volcano example – high complexity
//!
//! An active volcanic island: a multi-layered cone built from five stacked
//! rect3d slabs of decreasing radius, a glowing crater pit, two lava-flow
//! channels snaking down the slopes, scattered boulders and rock formations,
//! and a ring of lava-pool planes at the base.
//!
//! Eight fire/lava glow lights in deep red-orange fill the scene with
//! hellish warmth.  A cool blue "ocean ambient" directional light provides
//! just enough contrast to read the dark rock silhouettes.  No physical sky
//! — the sky is replaced with a deep red-black volcanic haze.
//!
//! Controls:
//!   WASD        — move forward/left/back/right
//!   Space/Shift — move up/down
//!   Mouse drag  — look around (click to grab cursor)
//!   Escape      — release cursor / exit



mod demo_portal;

use helio_render_v2::{Renderer, RendererConfig, Camera, GpuMesh, SceneLight, LightId, BillboardId};


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

// Lava/fire lights: (x, y, z, r, g, b, intensity, range)
const LAVA_LIGHTS: &[(f32, f32, f32, f32, f32, f32, f32, f32)] = &[
    // Crater eruption — hottest, most intense
    (  0.0, 33.5, -10.0,  1.0, 0.35, 0.05, 18.0, 35.0),
    // Lava lake surface glow
    (  0.0, 30.2, -10.0,  1.0, 0.20, 0.02,  8.0, 20.0),
    // Left lava flow channel
    (-12.0,  3.0,   4.0,  1.0, 0.30, 0.04,  5.0, 14.0),
    (-18.0,  0.8,  14.0,  1.0, 0.25, 0.03,  4.0, 12.0),
    // Right lava flow channel
    ( 14.0,  2.5,   2.0,  1.0, 0.28, 0.04,  5.0, 13.0),
    ( 20.0,  0.8,  10.0,  0.9, 0.22, 0.03,  4.0, 11.0),
    // Fumarole vents on mid-slope
    ( -6.0, 14.0,  -4.0,  1.0, 0.45, 0.1,   3.0,  8.0),
    (  6.0, 12.0,  -6.0,  1.0, 0.40, 0.08,  3.0,  8.0),
];

// Boulder/rock formations: (x, y_half, z, half_size)
const BOULDERS: &[(f32, f32, f32, f32)] = &[
    (-22.0, 1.4,  12.0, 1.4),
    ( 25.0, 1.1,   8.0, 1.1),
    (-16.0, 0.8,  20.0, 0.8),
    ( 18.0, 1.6,  18.0, 1.6),
    ( -8.0, 0.9,  24.0, 0.9),
    (  6.0, 1.2,  22.0, 1.2),
    (-28.0, 0.7,  -4.0, 0.7),
    ( 28.0, 1.0, -12.0, 1.0),
    ( -4.0, 2.1,  -2.0, 2.1), // large rock near base
    ( 10.0, 1.8, -18.0, 1.8),
];

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

const RC_WORLD_MIN: [f32; 3] = [-50.0, -0.5, -50.0];
const RC_WORLD_MAX: [f32; 3] = [50.0, 40.0, 50.0];

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

    island_ground: GpuMesh,
    // Volcano cone layers (bottom to top)
    cone_l1: GpuMesh,
    cone_l2: GpuMesh,
    cone_l3: GpuMesh,
    cone_l4: GpuMesh,
    cone_l5: GpuMesh, // near-summit
    // Crater rim cap
    crater_rim: GpuMesh,
    // Lava lake (glowing plane at crater)
    lava_lake: GpuMesh,
    // Lava flow channels
    flow_left:  Vec<GpuMesh>, // 3 segments
    flow_right: Vec<GpuMesh>,
    // Lava pool puddles at base
    lava_pools: Vec<GpuMesh>,
    // Boulders and rocks
    boulders: Vec<GpuMesh>,
    // Ash/scorch terrain patches around base
    scorch_patches: Vec<GpuMesh>,

    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

    // Scene state
    ocean_light_id: LightId,
    lava_light_ids: Vec<LightId>,
    billboard_ids: Vec<BillboardId>,

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
                .with_title("Helio – Outdoor Volcano")
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
        device.on_uncaptured_error(std::sync::Arc::new(|e| {
            panic!("[GPU UNCAPTURED ERROR] {:?}", e);
        }));
        let info = adapter.get_info();
        println!("[WGPU] Backend: {:?}, Device: {}, Driver: {}", info.backend, info.name, info.driver);
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
            .with_feature(BloomFeature::new().with_intensity(1.0).with_threshold(0.75))
            .with_feature(ShadowsFeature::new().with_atlas_size(2048).with_max_lights(4))
            .with_feature(BillboardsFeature::new().with_sprite(sprite_rgba, sprite_w, sprite_h).with_max_instances(5000))
            .with_feature(RadianceCascadesFeature::new()
                .with_world_bounds([-50.0, -0.5, -50.0], [50.0, 40.0, 50.0]))
            .build();

        let mut renderer = Renderer::new(device.clone(), queue.clone(),
            RendererConfig::new(size.width, size.height, format, features),
        ).expect("renderer");

        let island_ground = renderer.create_mesh_plane([0.0, 0.0, 0.0], 55.0);

        // Volcano cone: offset back (-10 on Z) so the erupting face is visible
        // Layers are stepped: each shrinks by ~30% and rises by 6-8m
        let cone_l1 = renderer.create_mesh_rect3d([0.0,  5.0, -10.0], [22.0,  5.0, 20.0]);
        let cone_l2 = renderer.create_mesh_rect3d([0.0, 11.5, -10.0], [15.5, 6.5, 14.0]);
        let cone_l3 = renderer.create_mesh_rect3d([0.0, 18.0, -10.0], [10.0, 6.5,  9.5]);
        let cone_l4 = renderer.create_mesh_rect3d([0.0, 24.0, -10.0], [ 5.5, 6.0,  5.5]);
        let cone_l5 = renderer.create_mesh_rect3d([0.0, 28.5, -10.0], [ 2.8, 4.5,  2.8]); // near summit
        let crater_rim = renderer.create_mesh_rect3d([0.0, 30.5, -10.0], [3.2, 0.4, 3.2]);
        // Lava lake: glowing plane inside crater
        let lava_lake = renderer.create_mesh_rect3d([0.0, 30.1, -10.0], [2.2, 0.05, 2.2]);

        // Left lava flow: 3 rect3d segments snaking down the slope
        let flow_left = vec![
            renderer.create_mesh_rect3d([ -5.5, 23.0, -6.0], [1.0, 0.12, 2.5]),
            renderer.create_mesh_rect3d([ -9.0, 16.0, -2.0], [1.2, 0.12, 3.5]),
            renderer.create_mesh_rect3d([-13.0,  6.5,  3.0], [1.4, 0.12, 5.0]),
            renderer.create_mesh_rect3d([-17.0,  1.5, 10.0], [1.5, 0.1,  6.0]),
        ];
        // Right lava flow
        let flow_right = vec![
            renderer.create_mesh_rect3d([  5.0, 22.0, -7.0], [1.0, 0.12, 2.5]),
            renderer.create_mesh_rect3d([  9.0, 15.5, -3.0], [1.2, 0.12, 3.5]),
            renderer.create_mesh_rect3d([ 13.5,  6.0,  2.0], [1.4, 0.12, 4.5]),
            renderer.create_mesh_rect3d([ 19.0,  1.5,  8.0], [1.5, 0.1,  5.5]),
        ];

        // Lava pools at base
        let lava_pools = vec![
            renderer.create_mesh_rect3d([-18.0, 0.06, 16.0], [4.0, 0.06, 3.0]),
            renderer.create_mesh_rect3d([ 22.0, 0.06, 12.0], [3.5, 0.06, 2.5]),
            renderer.create_mesh_rect3d([  0.0, 0.06, 22.0], [2.5, 0.06, 2.0]),
        ];

        // Boulders
        let boulders: Vec<GpuMesh> = BOULDERS.iter().map(|&(x, yh, z, hs)| {
            renderer.create_mesh_cube([x, yh, z], hs)
        }).collect();

        // Scorch/ash patches (dark flat planes)
        let scorch_patches = vec![
            renderer.create_mesh_rect3d([-10.0, 0.02,  8.0], [4.5, 0.02, 3.5]),
            renderer.create_mesh_rect3d([ 12.0, 0.02,  6.0], [3.5, 0.02, 3.0]),
            renderer.create_mesh_rect3d([  2.0, 0.02, 16.0], [3.0, 0.02, 4.0]),
            renderer.create_mesh_rect3d([-20.0, 0.02, -2.0], [3.0, 0.02, 2.5]),
            renderer.create_mesh_rect3d([ 22.0, 0.02, -8.0], [3.5, 0.02, 2.5]),
        ];
        demo_portal::enable_live_dashboard(&mut renderer);

        renderer.add_object(&island_ground, None, glam::Mat4::IDENTITY);
        renderer.add_object(&cone_l1,       None, glam::Mat4::IDENTITY);
        renderer.add_object(&cone_l2,       None, glam::Mat4::IDENTITY);
        renderer.add_object(&cone_l3,       None, glam::Mat4::IDENTITY);
        renderer.add_object(&cone_l4,       None, glam::Mat4::IDENTITY);
        renderer.add_object(&cone_l5,       None, glam::Mat4::IDENTITY);
        renderer.add_object(&crater_rim,    None, glam::Mat4::IDENTITY);
        renderer.add_object(&lava_lake,     None, glam::Mat4::IDENTITY);
        for m in &flow_left      { renderer.add_object(m, None, glam::Mat4::IDENTITY); }
        for m in &flow_right     { renderer.add_object(m, None, glam::Mat4::IDENTITY); }
        for m in &lava_pools     { renderer.add_object(m, None, glam::Mat4::IDENTITY); }
        for m in &boulders       { renderer.add_object(m, None, glam::Mat4::IDENTITY); }
        for m in &scorch_patches { renderer.add_object(m, None, glam::Mat4::IDENTITY); }

        // Ocean/sky directional fill — cool blue, very dim
        let ocean_dir = glam::Vec3::new(-0.3, -0.6, 0.2).normalize();
        let ocean_light_id = renderer.add_light(SceneLight::directional(
            [ocean_dir.x, ocean_dir.y, ocean_dir.z], [0.3, 0.5, 1.0], 0.04,
        ));
        let mut lava_light_ids = Vec::new();
        for &(x, y, z, r, g, b, intensity, range) in LAVA_LIGHTS {
            let p = [x, y, z];
            lava_light_ids.push(renderer.add_light(SceneLight::point(p, [r, g, b], intensity, range)));
        }
        renderer.set_ambient([0.5, 0.1, 0.02], 0.04);
        renderer.set_sky_color([0.06, 0.01, 0.01]);

        let mut billboard_ids = Vec::new();
        for &(x, y, z, r, g, b, _intensity, _range) in LAVA_LIGHTS {
            let p = [x, y, z];
            billboard_ids.push(renderer.add_billboard(BillboardInstance::new(p, [0.6, 0.6]).with_color([r, g, b, 0.9])));
        }

        self.state = Some(AppState {
            window, surface, device, surface_format: format, renderer,
            last_frame: std::time::Instant::now(),
            island_ground, cone_l1, cone_l2, cone_l3, cone_l4, cone_l5,
            crater_rim, lava_lake, flow_left, flow_right, lava_pools,
            boulders, scorch_patches,
            cam_pos: glam::Vec3::new(0.0, 8.0, 38.0),
            cam_yaw: std::f32::consts::PI, cam_pitch: -0.15,
            keys: HashSet::new(), cursor_grabbed: false, mouse_delta: (0.0, 0.0),
            ocean_light_id,
            lava_light_ids,
            billboard_ids,
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
                for id in state.billboard_ids.drain(..) { state.renderer.remove_billboard(id); }
                if state.probe_vis {
                    for b in probe_billboards(RC_WORLD_MIN, RC_WORLD_MAX) {
                        state.billboard_ids.push(state.renderer.add_billboard(b));
                    }
                } else {
                    for &(x, y, z, r, g, b, _intensity, _range) in LAVA_LIGHTS {
                        let p = [x, y, z];
                        state.billboard_ids.push(state.renderer.add_billboard(BillboardInstance::new(p, [0.6, 0.6]).with_color([r, g, b, 0.9])));
                    }
                }
            }
            WindowEvent::KeyboardInput { event: KeyEvent {
                state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::Digit4), ..
            }, .. } => {
                let _ = state.renderer.start_live_portal_default();
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
        const SPEED: f32 = 10.0;
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
            std::f32::consts::FRAC_PI_4, aspect, 0.1, 1000.0, time,
        );

        // Per-light flicker: each light gets a unique phase
        let f = |phase: f32, freq: f32, amp: f32| 1.0 + (time * freq + phase).sin() * amp;

        let output = match self.surface.get_current_texture() {
            Ok(t) => t, Err(e) => { log::warn!("Surface: {:?}", e); return; }
        };
        let view = output.texture.create_view(&Default::default());

        // Update lava lights with per-light flicker
        for (i, &id) in self.lava_light_ids.iter().enumerate() {
            let (x, y, z, r, g, b, intensity, range) = LAVA_LIGHTS[i];
            let phase = i as f32 * 1.37;
            let fi = f(phase, 8.0 + i as f32 * 1.1, 0.06 + (i % 3) as f32 * 0.03);
            let p = [x, y, z];
            self.renderer.update_light(id, SceneLight::point(p, [r, g, b], intensity * fi, range));
        }
        if let Err(e) = self.renderer.render(&camera, &view, dt) {
            log::error!("Render: {:?}", e);
        }
        output.present();
    }
}
