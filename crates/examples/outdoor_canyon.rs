//! Outdoor canyon example – high complexity
//!
//! A dramatic desert canyon at golden hour.  The sun is low on the horizon
//! (controllable with Q/E), casting long shadows across layered rock terraces.
//! Volumetric clouds drift overhead.  Three campfire-orange point lights sit
//! in a valley camp; one cool-blue moonlight-style fill light provides
//! contrast in shaded alcoves.  The RC global-illumination system bounces the
//! warm sun colour into the rock faces.
//!
//! Scene geometry:
//!   - Large flat valley floor
//!   - Four canyon wall slabs at varying heights and depths
//!   - Three rock terrace shelves stepping up each side
//!   - A mesa/butte plateau in the background
//!   - Small camp: three tent prisms (rect3d) + a central firepit cube
//!
//! Controls:
//!   WASD        — move forward/left/back/right
//!   Space/Shift — move up/down
//!   Q/E         — rotate sun (time of day)
//!   Mouse drag  — look around (click to grab cursor)
//!   Escape      — release cursor / exit



mod demo_portal;

use helio_render_v2::{
    Renderer, RendererConfig, Camera, GpuMesh, SceneLight, LightId, BillboardId,
    SkyAtmosphere, VolumetricClouds, Skylight,
};


use helio_render_v2::features::{
    FeatureRegistry,
    LightingFeature,
    BloomFeature, ShadowsFeature,
    BillboardsFeature, BillboardInstance,
    RadianceCascadesFeature,
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

const RC_WORLD_MIN: [f32; 3] = [-40.0, -0.5, -60.0];
const RC_WORLD_MAX: [f32; 3] = [40.0, 30.0, 40.0];

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
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("Event loop error");
}

struct App { state: Option<AppState> }

struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    surface_format: wgpu::TextureFormat,
    renderer: Renderer,
    last_frame: std::time::Instant,

    // Terrain
    valley_floor: GpuMesh,
    // Canyon walls – left side
    wall_l1: GpuMesh,
    wall_l2: GpuMesh,
    wall_l3: GpuMesh,
    // Canyon walls – right side
    wall_r1: GpuMesh,
    wall_r2: GpuMesh,
    wall_r3: GpuMesh,
    // Terraces stepping up the walls
    terrace_l1: GpuMesh,
    terrace_l2: GpuMesh,
    terrace_r1: GpuMesh,
    terrace_r2: GpuMesh,
    // Background mesa
    mesa: GpuMesh,
    // Camp
    tent_a: GpuMesh,
    tent_b: GpuMesh,
    tent_c: GpuMesh,
    firepit: GpuMesh,

    cam_pos:        glam::Vec3,
    cam_yaw:        f32,
    cam_pitch:      f32,
    keys:           HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta:    (f32, f32),
    sun_angle:      f32,

    // Scene state
    sun_light_id:   LightId,
    fire_light_id:  LightId,
    ember_a_id:     LightId,
    ember_b_id:     LightId,
    moon_light_id:  LightId,
    billboard_ids:  Vec<BillboardId>,

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

        let window = Arc::new(
            event_loop.create_window(
                Window::default_attributes()
                    .with_title("Helio – Outdoor Canyon")
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

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Device"),
            required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY | wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
            required_limits: wgpu::Limits::default()
                .using_minimum_supported_acceleration_structure_values(),
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
            .with_feature(BloomFeature::new().with_intensity(0.4).with_threshold(1.1))
            .with_feature(ShadowsFeature::new().with_atlas_size(2048).with_max_lights(4))
            .with_feature(BillboardsFeature::new().with_sprite(sprite_rgba, sprite_w, sprite_h).with_max_instances(5000))
            .with_feature(
                RadianceCascadesFeature::new()
                    .with_world_bounds([-40.0, -0.5, -60.0], [40.0, 30.0, 40.0]),
            )
            .build();

        let mut renderer = Renderer::new(
            device.clone(), queue.clone(),
            RendererConfig::new(size.width, size.height, format, features),
        ).expect("renderer");

        // Valley floor
        let valley_floor = renderer.create_mesh_plane([0.0, 0.0, 0.0], 35.0);

        // Canyon walls – left side, stairstepping upward away from center
        let wall_l1 = renderer.create_mesh_rect3d([-12.0,  4.0, 0.0], [3.0,  4.0, 30.0]);
        let wall_l2 = renderer.create_mesh_rect3d([-18.0,  8.0, 0.0], [3.0,  8.0, 25.0]);
        let wall_l3 = renderer.create_mesh_rect3d([-24.0, 14.0, 0.0], [3.0, 14.0, 20.0]);
        // Canyon walls – right side
        let wall_r1 = renderer.create_mesh_rect3d([ 12.0,  4.0, 0.0], [3.0,  4.0, 30.0]);
        let wall_r2 = renderer.create_mesh_rect3d([ 18.0,  8.0, 0.0], [3.0,  8.0, 25.0]);
        let wall_r3 = renderer.create_mesh_rect3d([ 24.0, 14.0, 0.0], [3.0, 14.0, 20.0]);
        // Terraces (horizontal slabs on top of wall tiers)
        let terrace_l1 = renderer.create_mesh_rect3d([-13.5, 8.1, -2.0],  [1.5, 0.2, 12.0]);
        let terrace_l2 = renderer.create_mesh_rect3d([-19.5, 16.1, -4.0], [1.5, 0.2,  8.0]);
        let terrace_r1 = renderer.create_mesh_rect3d([ 13.5, 8.1, -2.0],  [1.5, 0.2, 12.0]);
        let terrace_r2 = renderer.create_mesh_rect3d([ 19.5, 16.1, -4.0], [1.5, 0.2,  8.0]);
        // Background mesa
        let mesa = renderer.create_mesh_rect3d([3.0, 12.0, -38.0], [10.0, 12.0, 8.0]);
        // Camp tents
        let tent_a = renderer.create_mesh_rect3d([-2.5, 0.6, 8.0],  [0.8, 0.6, 1.2]);
        let tent_b = renderer.create_mesh_rect3d([ 0.0, 0.7, 7.5],  [0.9, 0.7, 1.3]);
        let tent_c = renderer.create_mesh_rect3d([ 2.8, 0.55, 8.5], [0.7, 0.55, 1.1]);
        let firepit = renderer.create_mesh_cube( [0.0, 0.15, 9.5], 0.2);
        demo_portal::enable_live_dashboard(&mut renderer);

        renderer.add_object(&valley_floor, None, glam::Mat4::IDENTITY);
        renderer.add_object(&wall_l1,      None, glam::Mat4::IDENTITY);
        renderer.add_object(&wall_l2,      None, glam::Mat4::IDENTITY);
        renderer.add_object(&wall_l3,      None, glam::Mat4::IDENTITY);
        renderer.add_object(&wall_r1,      None, glam::Mat4::IDENTITY);
        renderer.add_object(&wall_r2,      None, glam::Mat4::IDENTITY);
        renderer.add_object(&wall_r3,      None, glam::Mat4::IDENTITY);
        renderer.add_object(&terrace_l1,   None, glam::Mat4::IDENTITY);
        renderer.add_object(&terrace_l2,   None, glam::Mat4::IDENTITY);
        renderer.add_object(&terrace_r1,   None, glam::Mat4::IDENTITY);
        renderer.add_object(&terrace_r2,   None, glam::Mat4::IDENTITY);
        renderer.add_object(&mesa,         None, glam::Mat4::IDENTITY);
        renderer.add_object(&tent_a,       None, glam::Mat4::IDENTITY);
        renderer.add_object(&tent_b,       None, glam::Mat4::IDENTITY);
        renderer.add_object(&tent_c,       None, glam::Mat4::IDENTITY);
        renderer.add_object(&firepit,      None, glam::Mat4::IDENTITY);

        // Register initial lights (sun & moon updated per-frame; fire has flicker)
        let fire_pos = [0.0f32, 0.5, 9.5];
        let ember_a  = [-0.4f32, 0.4, 9.2];
        let ember_b  = [ 0.4f32, 0.4, 9.8];
        let moon_dir_v = glam::Vec3::new(0.4, -0.7, 0.3).normalize();
        let initial_sun_dir: [f32; 3] = [-0.0, -1.0, -0.5];
        let sun_light_id  = renderer.add_light(SceneLight::directional(initial_sun_dir, [1.0, 0.9, 0.7], 0.005));
        let fire_light_id = renderer.add_light(SceneLight::point(fire_pos, [1.0, 0.45, 0.1], 5.0, 12.0));
        let ember_a_id    = renderer.add_light(SceneLight::point(ember_a,  [1.0, 0.35, 0.05], 1.5, 5.0));
        let ember_b_id    = renderer.add_light(SceneLight::point(ember_b,  [1.0, 0.35, 0.05], 1.5, 5.0));
        let moon_light_id = renderer.add_light(SceneLight::directional(
            [moon_dir_v.x, moon_dir_v.y, moon_dir_v.z], [0.5, 0.65, 1.0], 0.05,
        ));
        renderer.set_sky_atmosphere(Some(
            SkyAtmosphere::new()
                .with_sun_intensity(20.0)
                .with_exposure(3.5)
                .with_mie_g(0.78)
                .with_clouds(
                    VolumetricClouds::new()
                        .with_coverage(0.20)
                        .with_density(0.5)
                        .with_layer(600.0, 1600.0)
                        .with_wind([0.7, 0.3], 0.05),
                ),
        ));
        renderer.set_skylight(Some(
            Skylight::new().with_intensity(0.10).with_tint([1.0, 0.92, 0.82]),
        ));

        // Register fire billboard marker
        let mut billboard_ids = Vec::new();
        billboard_ids.push(renderer.add_billboard(BillboardInstance::new(fire_pos, [0.5, 0.5]).with_color([1.0, 0.45, 0.1, 1.0])));

        self.state = Some(AppState {
            window, surface, device, surface_format: format, renderer,
            last_frame: std::time::Instant::now(),
            valley_floor,
            wall_l1, wall_l2, wall_l3,
            wall_r1, wall_r2, wall_r3,
            terrace_l1, terrace_l2, terrace_r1, terrace_r2,
            mesa,
            tent_a, tent_b, tent_c, firepit,
            cam_pos: glam::Vec3::new(0.0, 4.0, 25.0),
            cam_yaw: 0.0, cam_pitch: -0.15,
            keys: HashSet::new(), cursor_grabbed: false, mouse_delta: (0.0, 0.0),
            sun_angle: 0.45, // golden-hour low sun
            sun_light_id,
            fire_light_id,
            ember_a_id,
            ember_b_id,
            moon_light_id,
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
                physical_key: PhysicalKey::Code(KeyCode::Digit3), ..
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
                // Remove existing billboards and register new set
                for id in state.billboard_ids.drain(..) {
                    state.renderer.remove_billboard(id);
                }
                if state.probe_vis {
                    for inst in probe_billboards(RC_WORLD_MIN, RC_WORLD_MAX) {
                        state.billboard_ids.push(state.renderer.add_billboard(inst));
                    }
                } else {
                    let fire_pos = [0.0f32, 0.5, 9.5];
                    state.billboard_ids.push(state.renderer.add_billboard(BillboardInstance::new(fire_pos, [0.5, 0.5]).with_color([1.0, 0.45, 0.1, 1.0])));
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
                    ElementState::Pressed  => {
                        if key == KeyCode::F3 {
                            state.renderer.debug_viz_mut().enabled ^= true;
                        }
                        state.keys.insert(key);
                    }
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
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format: state.surface_format,
                    width: s.width, height: s.height,
                    present_mode: wgpu::PresentMode::Fifo,
                    alpha_mode: wgpu::CompositeAlphaMode::Auto,
                    view_formats: vec![], desired_maximum_frame_latency: 2,
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

        // Sun direction
        let sun_dir = glam::Vec3::new(
            self.sun_angle.cos() * 0.3,
            self.sun_angle.sin(),
            0.5,
        ).normalize();
        let light_dir = [-sun_dir.x, -sun_dir.y, -sun_dir.z];
        let sun_elev  = sun_dir.y.clamp(-1.0, 1.0);
        let sun_lux   = (sun_elev * 3.0).clamp(0.0, 1.0);
        // Warm golden-hour tint when sun is low
        let warmth = (1.0 - sun_elev).clamp(0.0, 1.0);
        let sun_color = [
            1.0_f32.min(1.0 + warmth * 0.5),
            (0.75 + sun_elev * 0.25).clamp(0.0, 1.0),
            (0.55 + sun_elev * 0.35).clamp(0.0, 1.0),
        ];

        // Campfire flicker
        let flicker = 1.0 + (time * 13.1).sin() * 0.08 + (time * 7.3).cos() * 0.05;
        let fire_pos = [0.0f32, 0.5, 9.5];

        // Update dynamic lights
        self.renderer.update_light(self.sun_light_id, SceneLight::directional(light_dir, sun_color, (sun_lux * 0.4).max(0.005)));
        self.renderer.update_light(self.fire_light_id, SceneLight::point(fire_pos, [1.0, 0.45, 0.1], 5.0 * flicker, 12.0));

        // Scene state is persistent — no per-frame setup needed.

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(e) => { log::warn!("Surface: {:?}", e); return; }
        };
        let view = output.texture.create_view(&Default::default());

        if let Err(e) = self.renderer.render(&camera, &view, dt) {
            log::error!("Render: {:?}", e);
        }
        output.present();
    }
}
