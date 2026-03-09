//! Radiance Cascades benchmark — GI performance test
//!
//! Modified Cornell box with colored walls (red left, green right, white others)
//! and a few bright area lights to showcase multi-bounce global illumination.
//! This scene is designed to stress-test RC probe tracing and merge passes.
//!
//! Controls:
//!   WASD        — move forward/left/back/right
//!   Space/Shift — move up/down
//!   Mouse drag  — look around (click to grab cursor)
//!   3           — toggle RC probe visualization ↔ light markers
//!   4           — toggle GPU timing printout (stderr)
//!   +/-         — increase/decrease light intensity
//!   Escape      — release cursor / exit



mod demo_portal;

use helio_render_v2::{Renderer, RendererConfig, Camera, GpuMesh, SceneLight, LightId, BillboardId};


use helio_render_v2::features::{
    FeatureRegistry, LightingFeature, BloomFeature,
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

const RC_WORLD_MIN: [f32; 3] = [-6.0, -0.1, -6.0];
const RC_WORLD_MAX: [f32; 3] = [ 6.0,  6.0,  6.0];

fn load_sprite() -> (Vec<u8>, u32, u32) {
    let img = image::load_from_memory(include_bytes!("../../spotlight.png"))
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
    const SIZES: [[f32; 2]; 4] = [
        [0.035, 0.035],
        [0.075, 0.075],
        [0.140, 0.140],
        [0.260, 0.260],
    ];
    let mut out = Vec::new();
    for (c, &dim) in PROBE_DIMS.iter().enumerate() {
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    let [xmin, ymin, zmin] = world_min;
                    let [xmax, ymax, zmax] = world_max;
                    let t = 1.0 / (dim as f32 - 1.0).max(1.0);
                    let x = xmin + (xmax - xmin) * (i as f32 * t);
                    let y = ymin + (ymax - ymin) * (j as f32 * t);
                    let z = zmin + (zmax - zmin) * (k as f32 * t);
                    out.push(BillboardInstance::new([x, y, z], SIZES[c])
                        .with_color(COLORS[c]).with_screen_scale(true));
                }
            }
        }
    }
    out
}

fn main() {
    env_logger::init();
    log::info!("Starting RC Benchmark — Radiance Cascades GI test");
    EventLoop::new()
        .expect("event loop")
        .run_app(&mut App::new())
        .expect("run");
}

struct App {
    state: Option<AppState>,
}

struct AppState {
    window:         Arc<Window>,
    surface:        wgpu::Surface<'static>,
    device:         Arc<wgpu::Device>,
    surface_format: wgpu::TextureFormat,
    renderer:       Renderer,
    last_frame:     std::time::Instant,

    // Geometry
    floor:   GpuMesh,
    ceiling: GpuMesh,
    wall_n:  GpuMesh,  // -Z (back)
    wall_s:  GpuMesh,  // +Z (front)
    wall_e:  GpuMesh,  // +X (right, green)
    wall_w:  GpuMesh,  // -X (left, red)
    cubes:   Vec<GpuMesh>,

    // Camera
    cam_pos:        glam::Vec3,
    cam_yaw:        f32,
    cam_pitch:      f32,
    keys:           HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta:    (f32, f32),

    // Scene state
    light_ids: Vec<LightId>,
    base_lights: Vec<SceneLight>,
    billboard_ids: Vec<BillboardId>,

    // Debug
    probe_vis: bool,
    light_intensity_multiplier: f32,
    sprite_w: u32,
    sprite_h: u32,
}

impl App {
    fn new() -> Self {
        Self { state: None }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }

        let window = Arc::new(
            event_loop.create_window(
                Window::default_attributes()
                    .with_title("Helio — RC Benchmark (Radiance Cascades GI)")
                    .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
            ).expect("window"),
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
        })).expect("adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_features:
                    wgpu::Features::EXPERIMENTAL_RAY_QUERY |
                    wgpu::Features::TIMESTAMP_QUERY |
                    wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
                required_limits: wgpu::Limits::default()
                    .using_minimum_supported_acceleration_structure_values(),
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                trace: wgpu::Trace::Off,
            },
        )).expect("device");
        let device = Arc::new(device);
        let queue  = Arc::new(queue);

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let fmt  = caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
        surface.configure(&device, &wgpu::SurfaceConfiguration {
            usage:        wgpu::TextureUsages::RENDER_ATTACHMENT,
            format:       fmt,
            width:        size.width,
            height:       size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 1,
            alpha_mode:   caps.alpha_modes[0],
            view_formats: vec![],
        });

        let (sprite_rgba, sprite_w, sprite_h) = load_sprite();
        let features = FeatureRegistry::builder()
            .with_feature(LightingFeature::new())
            .with_feature(BloomFeature::new().with_intensity(0.6).with_threshold(0.8))
            .with_feature(
                BillboardsFeature::new()
                    .with_sprite(sprite_rgba, sprite_w, sprite_h)
                    .with_max_instances(5000),
            )
            .with_feature(
                RadianceCascadesFeature::new()
                    .with_world_bounds(RC_WORLD_MIN, RC_WORLD_MAX),
            )
            .build();

        let mut renderer = Renderer::new(device.clone(), queue.clone(), RendererConfig::new(
            size.width, size.height, fmt, features
        )).expect("renderer");

        // ── Cornell-box geometry (12 m room) ──────────────────────────────────
        let floor   = GpuMesh::plane(&device, [0.0, 0.0, 0.0], 5.0);
        let ceiling = GpuMesh::rect3d(&device, [0.0, 5.0, 0.0], [5.0, 0.0, 5.0]);

        // Walls: 5×5 m quads using rect3d with rotations
        let wall_n = GpuMesh::rect3d(&device, [0.0, 2.5, -5.0], [5.0, 2.5, 0.0]);  // back
        let wall_s = GpuMesh::rect3d(&device, [0.0, 2.5,  5.0], [5.0, 2.5, 0.0]);  // front
        let wall_e = GpuMesh::rect3d(&device, [ 5.0, 2.5, 0.0], [0.0, 2.5, 5.0]);  // right
        let wall_w = GpuMesh::rect3d(&device, [-5.0, 2.5, 0.0], [0.0, 2.5, 5.0]);  // left

        // A few cubes scattered in the room to show off light bounces
        let cubes = vec![
            GpuMesh::cube(&device, [-2.0, 0.5, -2.0], 0.5),
            GpuMesh::cube(&device, [ 2.0, 0.5,  2.0], 0.5),
            GpuMesh::cube(&device, [ 0.0, 0.7,  0.0], 0.7),
            GpuMesh::cube(&device, [-3.0, 1.0,  1.5], 1.0),
            GpuMesh::cube(&device, [ 3.0, 0.6, -1.5], 0.6),
        ];
        demo_portal::enable_live_dashboard(&mut renderer);

        renderer.add_object(&floor,   None, glam::Mat4::IDENTITY);
        renderer.add_object(&ceiling, None, glam::Mat4::IDENTITY);
        renderer.add_object(&wall_n,  None, glam::Mat4::IDENTITY);
        renderer.add_object(&wall_s,  None, glam::Mat4::IDENTITY);
        renderer.add_object(&wall_e,  None, glam::Mat4::IDENTITY);
        renderer.add_object(&wall_w,  None, glam::Mat4::IDENTITY);
        for c in &cubes { renderer.add_object(c, None, glam::Mat4::IDENTITY); }

        // Three bright area lights to showcase multi-bounce GI
        let base_lights = vec![
            SceneLight::point([0.0, 4.8, 0.0], [1.0, 1.0, 1.0], 18.0, 6.0),
            SceneLight::point([-3.5, 3.0, -2.0], [1.0, 0.7, 0.4], 12.0, 5.0),
            SceneLight::point([3.5, 3.0, 2.0], [0.4, 0.7, 1.0], 12.0, 5.0),
        ];
        let mut light_ids = Vec::new();
        for &ref l in &base_lights {
            light_ids.push(renderer.add_light(l.clone()));
        }
        renderer.set_ambient([0.02, 0.02, 0.03], 1.0);

        let mut billboard_ids = Vec::new();
        for l in &base_lights {
            let col = l.color;
            billboard_ids.push(renderer.add_billboard(
                BillboardInstance::new(l.position, [0.2, 0.2])
                    .with_color([col[0], col[1], col[2], 0.9])
                    .with_screen_scale(true)
            ));
        }

        self.state = Some(AppState {
            window, surface, device, surface_format: fmt, renderer,
            last_frame: std::time::Instant::now(),
            floor, ceiling, wall_n, wall_s, wall_e, wall_w, cubes,
            cam_pos:        glam::Vec3::new(0.0, 2.5, 8.0),
            cam_yaw:        0.0,
            cam_pitch:      0.0,
            keys:           HashSet::new(),
            cursor_grabbed: false,
            mouse_delta:    (0.0, 0.0),
            light_ids,
            base_lights,
            billboard_ids,
            probe_vis: false,
            light_intensity_multiplier: 1.0,
            sprite_w, sprite_h,
        });
    }

    fn window_event(
        &mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent,
    ) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => { event_loop.exit(); }

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

            // Toggle probe visualization
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
                for id in state.billboard_ids.drain(..) { state.renderer.remove_billboard(id); }
                if state.probe_vis {
                    for b in probe_billboards(RC_WORLD_MIN, RC_WORLD_MAX) {
                        state.billboard_ids.push(state.renderer.add_billboard(b));
                    }
                } else {
                    for l in &state.base_lights {
                        let col = l.color;
                        state.billboard_ids.push(state.renderer.add_billboard(
                            BillboardInstance::new(l.position, [0.2, 0.2])
                                .with_color([col[0], col[1], col[2], 0.9])
                                .with_screen_scale(true)
                        ));
                    }
                }
            }

            // Live profiler portal
            WindowEvent::KeyboardInput { event: KeyEvent {
                state: ElementState::Pressed,
                physical_key: PhysicalKey::Code(KeyCode::Digit4), ..
            }, .. } => { let _ = state.renderer.start_live_portal_default(); }

            // Decrease light intensity
            WindowEvent::KeyboardInput { event: KeyEvent {
                state: ElementState::Pressed,
                physical_key: PhysicalKey::Code(KeyCode::Minus), ..
            }, .. } => {
                state.light_intensity_multiplier = (state.light_intensity_multiplier - 0.1).max(0.1);
                eprintln!("Light intensity: {:.1}x", state.light_intensity_multiplier);
            }

            // Increase light intensity
            WindowEvent::KeyboardInput { event: KeyEvent {
                state: ElementState::Pressed,
                physical_key: PhysicalKey::Code(KeyCode::Equal), ..
            }, .. } => {
                state.light_intensity_multiplier = (state.light_intensity_multiplier + 0.1).min(5.0);
                eprintln!("Light intensity: {:.1}x", state.light_intensity_multiplier);
            }

            WindowEvent::KeyboardInput { event: KeyEvent {
                state: ks, physical_key: PhysicalKey::Code(key), ..
            }, .. } => {
                match ks {
                    ElementState::Pressed  => { state.keys.insert(key); }
                    ElementState::Released => { state.keys.remove(&key); }
                }
            }

            WindowEvent::MouseInput {
                state: ElementState::Pressed, button: MouseButton::Left, ..
            } => {
                if !state.cursor_grabbed {
                    let ok = state.window.set_cursor_grab(CursorGrabMode::Confined)
                        .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked))
                        .is_ok();
                    if ok {
                        state.window.set_cursor_visible(false);
                        state.cursor_grabbed = true;
                    }
                }
            }

            WindowEvent::Resized(new_size) => {
                log::info!("Window resized to {:?}", new_size);
                state.surface.configure(&state.device, &wgpu::SurfaceConfiguration {
                    usage:  wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format: state.surface_format,
                    width:  new_size.width,
                    height: new_size.height,
                    present_mode: wgpu::PresentMode::AutoVsync,
                    desired_maximum_frame_latency: 1,
                    alpha_mode: wgpu::CompositeAlphaMode::Auto,
                    view_formats: vec![],
                });
                state.renderer.resize(new_size.width, new_size.height);
            }

            WindowEvent::RedrawRequested => { state.render(); }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }

    fn device_event(
        &mut self, _: &ActiveEventLoop, _: winit::event::DeviceId, event: DeviceEvent,
    ) {
        let Some(state) = &mut self.state else { return };
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if state.cursor_grabbed {
                state.mouse_delta.0 += dx as f32;
                state.mouse_delta.1 += dy as f32;
            }
        }
    }
}

impl AppState {
    fn render(&mut self) {
        const SPEED: f32 = 4.0;
        const MOUSE_SENS: f32 = 0.002;

        let now = std::time::Instant::now();
        let dt  = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        // Mouse look
        self.cam_yaw   += self.mouse_delta.0 * MOUSE_SENS;
        self.cam_pitch -= self.mouse_delta.1 * MOUSE_SENS;
        self.cam_pitch  = self.cam_pitch.clamp(-1.5, 1.5);
        self.mouse_delta = (0.0, 0.0);

        // Movement
        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let forward  = glam::Vec3::new(sy * cp, sp, -cy * cp);
        let right    = glam::Vec3::new(cy, 0.0, sy);

        if self.keys.contains(&KeyCode::KeyW)      { self.cam_pos += forward * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyS)      { self.cam_pos -= forward * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyA)      { self.cam_pos -= right   * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyD)      { self.cam_pos += right   * SPEED * dt; }
        if self.keys.contains(&KeyCode::Space)     { self.cam_pos.y += SPEED * dt; }
        if self.keys.contains(&KeyCode::ShiftLeft) { self.cam_pos.y -= SPEED * dt; }

        let size   = self.window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;
        let time   = self.renderer.frame_count() as f32 * 0.016;

        let camera = Camera::perspective(
            self.cam_pos,
            self.cam_pos + forward,
            glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            aspect,
            0.1,
            300.0,
            time,
        );

        let output = match self.surface.get_current_texture() {
            Ok(t)  => t,
            Err(e) => { log::warn!("Surface error: {:?}", e); return; }
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Apply intensity multiplier to all lights per-frame
        for (i, &id) in self.light_ids.iter().enumerate() {
            let mut light = self.base_lights[i].clone();
            light.intensity *= self.light_intensity_multiplier;
            self.renderer.update_light(id, light);
        }
        if let Err(e) = self.renderer.render(&camera, &view, dt) {
            log::error!("Render error: {:?}", e);
        }
        output.present();
    }
}
