//! Indoor corridor example – medium complexity
//!
//! A long hallway lit by a row of evenly-spaced overhead fluorescent lights
//! (cool white), with red emergency-exit signs at both ends and a pair of
//! decorative wall sconces with warm amber glow partway along.
//!
//! The camera starts at one end looking down the corridor.
//!
//! Controls:
//!   WASD        — move forward/left/back/right
//!   Space/Shift — move up/down
//!   Mouse drag  — look around (click to grab cursor)
//!   Escape      — release cursor / exit



mod demo_portal;

use helio_render_v2::{Renderer, RendererConfig, Camera, GpuMesh, SceneLight, SceneEnv};


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

const RC_WORLD_MIN: [f32; 3] = [-2.0, -0.1, -20.0];
const RC_WORLD_MAX: [f32; 3] = [2.0, 3.5, 20.0];

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

    floor:    GpuMesh,
    ceiling:  GpuMesh,
    wall_l:   GpuMesh,
    wall_r:   GpuMesh,
    wall_far: GpuMesh,
    wall_near: GpuMesh,
    // Sconce brackets on each side wall
    sconce_l: GpuMesh,
    sconce_r: GpuMesh,

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
                    .with_title("Helio – Indoor Corridor")
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
            .with_feature(BloomFeature::new().with_intensity(0.35).with_threshold(1.1))
            .with_feature(ShadowsFeature::new().with_atlas_size(2048).with_max_lights(4))
            .with_feature(BillboardsFeature::new().with_sprite(sprite_rgba, sprite_w, sprite_h).with_max_instances(5000))
            .with_feature(
                RadianceCascadesFeature::new()
                    .with_world_bounds([-2.0, -0.1, -20.0], [2.0, 3.5, 20.0]),
            )
            .build();

        let mut renderer = Renderer::new(
            device.clone(), queue.clone(),
            RendererConfig::new(size.width, size.height, format, features),
        ).expect("renderer");

        // Corridor: 4 m wide (X), 3 m tall (Y), 36 m long (Z: -18..+18)
        let floor     = GpuMesh::rect3d(&device, [0.0, 0.0,  0.0], [2.0, 0.02, 18.0]);
        let ceiling   = GpuMesh::rect3d(&device, [0.0, 3.0,  0.0], [2.0, 0.02, 18.0]);
        let wall_l    = GpuMesh::rect3d(&device, [-2.0, 1.5, 0.0], [0.02, 1.5, 18.0]);
        let wall_r    = GpuMesh::rect3d(&device, [ 2.0, 1.5, 0.0], [0.02, 1.5, 18.0]);
        let wall_far  = GpuMesh::rect3d(&device, [0.0, 1.5, -18.0], [2.0, 1.5, 0.02]);
        let wall_near = GpuMesh::rect3d(&device, [0.0, 1.5,  18.0], [2.0, 1.5, 0.02]);
        // Sconce brackets midway along the corridor, mounted on each side wall
        let sconce_l  = GpuMesh::rect3d(&device, [-1.85, 1.8, 0.0], [0.12, 0.08, 0.25]);
        let sconce_r  = GpuMesh::rect3d(&device, [ 1.85, 1.8, 0.0], [0.12, 0.08, 0.25]);
        demo_portal::enable_live_dashboard(&mut renderer);

        renderer.add_object(&floor,     None, glam::Mat4::IDENTITY);
        renderer.add_object(&ceiling,   None, glam::Mat4::IDENTITY);
        renderer.add_object(&wall_l,    None, glam::Mat4::IDENTITY);
        renderer.add_object(&wall_r,    None, glam::Mat4::IDENTITY);
        renderer.add_object(&wall_far,  None, glam::Mat4::IDENTITY);
        renderer.add_object(&wall_near, None, glam::Mat4::IDENTITY);
        renderer.add_object(&sconce_l,  None, glam::Mat4::IDENTITY);
        renderer.add_object(&sconce_r,  None, glam::Mat4::IDENTITY);

        self.state = Some(AppState {
            window, surface, device, surface_format: format, renderer,
            last_frame: std::time::Instant::now(),
            floor, ceiling, wall_l, wall_r, wall_far, wall_near,
            sconce_l, sconce_r,
            cam_pos: glam::Vec3::new(0.0, 1.6, 16.0),
            cam_yaw: std::f32::consts::PI, cam_pitch: 0.0,
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
                state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::Digit4), ..
            }, .. } => {
                state.renderer.debug_key_pressed();
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
            std::f32::consts::FRAC_PI_4, aspect, 0.1, 100.0, time,
        );

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(e) => { log::warn!("Surface: {:?}", e); return; }
        };
        let view = output.texture.create_view(&Default::default());

        let overhead_z: &[f32] = &[-14.0, -7.0, 0.0, 7.0, 14.0];
        // Emergency-exit red lights at both ends
        let exit_near = [0.0f32, 2.4,  17.5];
        let exit_far  = [0.0f32, 2.4, -17.5];
        // Warm sconce pair in the middle
        let sconce_lp = [-1.7f32, 1.85, 0.0];
        let sconce_rp = [ 1.7f32, 1.85, 0.0];

        let mut lights = Vec::new();
        for &z in overhead_z {
            let p = [0.0f32, 2.88, z];
            lights.push(SceneLight::spot(
                p, [0.0, -1.0, 0.0],
                [0.9, 0.95, 1.0], 3.5, 9.0,
                1.22, /* inner ~70° */ 1.48, /* outer ~85° */
            ));
        }
        lights.push(SceneLight::point(exit_near, [1.0, 0.08, 0.08], 1.5, 4.0));
        lights.push(SceneLight::point(exit_far,  [1.0, 0.08, 0.08], 1.5, 4.0));
        lights.push(SceneLight::point(sconce_lp, [1.0, 0.65, 0.3],  2.0, 4.5));
        lights.push(SceneLight::point(sconce_rp, [1.0, 0.65, 0.3],  2.0, 4.5));

        let billboards = if self.probe_vis {
            probe_billboards(RC_WORLD_MIN, RC_WORLD_MAX)
        } else {
            let mut bb = Vec::new();
            for &z in overhead_z {
                let p = [0.0f32, 2.88, z];
                bb.push(BillboardInstance::new(p, [0.2, 0.2]).with_color([0.9, 0.95, 1.0, 1.0]));
            }
            bb.push(BillboardInstance::new(exit_near, [0.3, 0.2]).with_color([1.0, 0.08, 0.08, 1.0]));
            bb.push(BillboardInstance::new(exit_far,  [0.3, 0.2]).with_color([1.0, 0.08, 0.08, 1.0]));
            bb.push(BillboardInstance::new(sconce_lp, [0.2, 0.2]).with_color([1.0, 0.65, 0.3, 1.0]));
            bb.push(BillboardInstance::new(sconce_rp, [0.2, 0.2]).with_color([1.0, 0.65, 0.3, 1.0]));
            bb
        };

        let env = SceneEnv {
            lights,
            ambient_color: [0.85, 0.9, 1.0],
            ambient_intensity: 0.04,
            sky_color: [0.0, 0.0, 0.0],
            billboards,
            ..Default::default()
        };
        self.renderer.set_scene_env(env);
        if let Err(e) = self.renderer.render(&camera, &view, dt) {
            log::error!("Render: {:?}", e);
        }
        output.present();
    }
}
