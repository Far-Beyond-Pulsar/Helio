//! Outdoor night plaza example – medium complexity
//!
//! A city plaza at night: a large paved ground plane, several building-block
//! structures of varying heights, four streetlamps with warm sodium-orange
//! glow, and two neon sign lights (magenta + cyan) on the taller buildings.
//! A deep-blue night sky with subtle ambient.
//!
//! Controls:
//!   WASD        — move forward/left/back/right
//!   Space/Shift — move up/down
//!   Mouse drag  — look around (click to grab cursor)
//!   Escape      — release cursor / exit

use helio_render_v2::{Renderer, RendererConfig, Camera, GpuMesh, Scene, SceneLight};
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

const RC_WORLD_MIN: [f32; 3] = [-20.0, -0.1, -20.0];
const RC_WORLD_MAX: [f32; 3] = [20.0, 20.0, 20.0];

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

    ground:    GpuMesh,
    // Buildings of varying heights
    bld_a:     GpuMesh, // tall
    bld_b:     GpuMesh, // medium-left
    bld_c:     GpuMesh, // medium-right
    bld_d:     GpuMesh, // short squat
    bld_e:     GpuMesh, // background tower
    // Streetlamp poles
    lamp_pole_a: GpuMesh,
    lamp_pole_b: GpuMesh,
    lamp_pole_c: GpuMesh,
    lamp_pole_d: GpuMesh,

    cam_pos:        glam::Vec3,
    cam_yaw:        f32,
    cam_pitch:      f32,
    keys:           HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta:    (f32, f32),

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
                    .with_title("Helio – Outdoor Night Plaza")
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
            required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
            required_limits: wgpu::Limits::default()
                .using_minimum_supported_acceleration_structure_values(),
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
            .with_feature(BloomFeature::new().with_intensity(0.8).with_threshold(0.9))
            .with_feature(ShadowsFeature::new().with_atlas_size(2048).with_max_lights(4))
            .with_feature(BillboardsFeature::new().with_sprite(sprite_rgba, sprite_w, sprite_h).with_max_instances(5000))
            .with_feature(
                RadianceCascadesFeature::new()
                    .with_world_bounds([-20.0, -0.1, -20.0], [20.0, 20.0, 20.0]),
            )
            .build();

        let renderer = Renderer::new(
            device.clone(), queue.clone(),
            RendererConfig { width: size.width, height: size.height, surface_format: format, features },
        ).expect("renderer");

        let ground = GpuMesh::plane(&device, [0.0, 0.0, 0.0], 20.0);

        // Buildings placed asymmetrically around the plaza
        let bld_a = GpuMesh::rect3d(&device, [ 8.0, 7.0, -6.0],  [2.5, 7.0, 2.5]); // tall
        let bld_b = GpuMesh::rect3d(&device, [-7.0, 4.5, -5.0],  [3.0, 4.5, 2.0]); // medium-left
        let bld_c = GpuMesh::rect3d(&device, [ 6.0, 3.0,  6.0],  [2.0, 3.0, 3.0]); // medium-right
        let bld_d = GpuMesh::rect3d(&device, [-5.0, 1.5,  5.0],  [3.5, 1.5, 2.5]); // short squat
        let bld_e = GpuMesh::rect3d(&device, [ 0.0, 9.5, -14.0], [4.0, 9.5, 3.0]); // bg tower

        // Thin poles + small cap for each streetlamp
        let lamp_pole_a = GpuMesh::rect3d(&device, [-5.0, 2.5,  -5.0], [0.08, 2.5, 0.08]);
        let lamp_pole_b = GpuMesh::rect3d(&device, [ 5.0, 2.5,  -5.0], [0.08, 2.5, 0.08]);
        let lamp_pole_c = GpuMesh::rect3d(&device, [-5.0, 2.5,   5.0], [0.08, 2.5, 0.08]);
        let lamp_pole_d = GpuMesh::rect3d(&device, [ 5.0, 2.5,   5.0], [0.08, 2.5, 0.08]);

        self.state = Some(AppState {
            window, surface, device, surface_format: format, renderer,
            last_frame: std::time::Instant::now(),
            ground, bld_a, bld_b, bld_c, bld_d, bld_e,
            lamp_pole_a, lamp_pole_b, lamp_pole_c, lamp_pole_d,
            cam_pos: glam::Vec3::new(0.0, 3.0, 12.0),
            cam_yaw: std::f32::consts::PI, cam_pitch: -0.15,
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
        const SPEED: f32 = 6.0;
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

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(e) => { log::warn!("Surface: {:?}", e); return; }
        };
        let view = output.texture.create_view(&Default::default());

        // Streetlamp heads at top of each pole
        let lamp_a = [-5.0f32, 5.1, -5.0];
        let lamp_b = [ 5.0f32, 5.1, -5.0];
        let lamp_c = [-5.0f32, 5.1,  5.0];
        let lamp_d = [ 5.0f32, 5.1,  5.0];
        // Neon signs: magenta on tall building, cyan on bg tower
        let neon_m = [ 8.0f32, 12.0, -5.8];
        let neon_c = [ 0.0f32, 16.5, -14.0];

        let mut scene = Scene::new()
            .with_sky([0.005, 0.005, 0.025])
            .with_ambient([0.1, 0.15, 0.3], 0.06)
            // Streetlamps – warm sodium orange
            .add_light(SceneLight::point(lamp_a, [1.0, 0.72, 0.3], 6.0, 14.0))
            .add_light(SceneLight::point(lamp_b, [1.0, 0.72, 0.3], 6.0, 14.0))
            .add_light(SceneLight::point(lamp_c, [1.0, 0.72, 0.3], 6.0, 14.0))
            .add_light(SceneLight::point(lamp_d, [1.0, 0.72, 0.3], 6.0, 14.0))
            // Neon: magenta on tall building
            .add_light(SceneLight::point(neon_m, [1.0, 0.05, 0.8], 5.0, 12.0))
            // Neon: cyan on bg tower
            .add_light(SceneLight::point(neon_c, [0.05, 0.9, 1.0], 4.0, 10.0))
            // Geometry
            .add_object(self.ground.clone())
            .add_object(self.bld_a.clone())
            .add_object(self.bld_b.clone())
            .add_object(self.bld_c.clone())
            .add_object(self.bld_d.clone())
            .add_object(self.bld_e.clone())
            .add_object(self.lamp_pole_a.clone())
            .add_object(self.lamp_pole_b.clone())
            .add_object(self.lamp_pole_c.clone())
            .add_object(self.lamp_pole_d.clone());

        if self.probe_vis {
            for b in probe_billboards(RC_WORLD_MIN, RC_WORLD_MAX) {
                scene = scene.add_billboard(b);
            }
        } else {
            // Billboards for all lights
            scene = scene
                .add_billboard(BillboardInstance::new(lamp_a, [0.4, 0.4]).with_color([1.0, 0.72, 0.3, 1.0]))
                .add_billboard(BillboardInstance::new(lamp_b, [0.4, 0.4]).with_color([1.0, 0.72, 0.3, 1.0]))
                .add_billboard(BillboardInstance::new(lamp_c, [0.4, 0.4]).with_color([1.0, 0.72, 0.3, 1.0]))
                .add_billboard(BillboardInstance::new(lamp_d, [0.4, 0.4]).with_color([1.0, 0.72, 0.3, 1.0]))
                .add_billboard(BillboardInstance::new(neon_m, [0.6, 0.25]).with_color([1.0, 0.05, 0.8, 1.0]))
                .add_billboard(BillboardInstance::new(neon_c, [0.6, 0.25]).with_color([0.05, 0.9, 1.0, 1.0]));
        }

        if let Err(e) = self.renderer.render_scene(&scene, &camera, &view, dt) {
            log::error!("Render: {:?}", e);
        }
        output.present();
    }
}
