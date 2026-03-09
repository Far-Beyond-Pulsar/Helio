//! Sky atmosphere example using helio-render-v2
//!
//! Demonstrates the volumetric sky system working together with colored scene
//! lights: a sun directional light drives the sky's sun direction and casts
//! shadows; a sky-based ambient (Skylight) fills shadowed areas with the sky
//! colour; and three point lights with distinct hues (warm amber, cool blue,
//! deep red) show how colored scene lights interact with the sky-lit scene.
//! Optional volumetric clouds drift overhead.
//!
//! Controls:
//!   WASD        — move forward/left/back/right
//!   Space/Shift — move up/down
//!   Q/E         — rotate sun left/right (changes time of day)
//!   Mouse drag  — look around (click to grab cursor)
//!   3           — toggle RC probe visualisation
//!   4           — toggle GPU timing printout (stderr)
//!   Escape      — release cursor / exit



mod demo_portal;

use helio_render_v2::{
    Renderer, RendererConfig, Camera, GpuMesh, SceneLight, SceneEnv,
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

const RC_WORLD_MIN: [f32; 3] = [-10.0, -0.3, -10.0];
const RC_WORLD_MAX: [f32; 3] = [10.0, 8.0, 10.0];

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
    log::info!("Starting Helio Sky Example");

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("Event loop error");
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

    cube1: GpuMesh,
    cube2: GpuMesh,
    cube3: GpuMesh,
    ground: GpuMesh,
    roof: GpuMesh,

    // Free-camera state
    cam_pos:   glam::Vec3,
    cam_yaw:   f32,
    cam_pitch: f32,
    keys:      HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

    // Time-of-day: sun_angle=0 → noon, PI → midnight
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

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Helio – Volumetric Sky")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
                )
                .expect("Failed to create window"),
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance
            .create_surface(window.clone())
            .expect("Failed to create surface");

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("Failed to find adapter");

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
        ))
        .expect("Failed to create device");

        let device = Arc::new(device);
        let queue  = Arc::new(queue);

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let (sprite_rgba, sprite_w, sprite_h) = load_sprite();
        let feature_registry = FeatureRegistry::builder()
            .with_feature(LightingFeature::new())
            .with_feature(BloomFeature::new().with_intensity(0.3).with_threshold(1.2))
            .with_feature(ShadowsFeature::new().with_atlas_size(1024).with_max_lights(4))
            .with_feature(BillboardsFeature::new().with_sprite(sprite_rgba, sprite_w, sprite_h).with_max_instances(5000))
            .with_feature(
                RadianceCascadesFeature::new()
                    .with_world_bounds([-10.0, -0.3, -10.0], [10.0, 8.0, 10.0]),
            )
            .build();

        let mut renderer = Renderer::new(
            device.clone(),
            queue.clone(),
            RendererConfig::new(size.width, size.height, surface_format, feature_registry),
        )
        .expect("Failed to create renderer");

        let cube1  = GpuMesh::cube(&device, [ 0.0, 0.5,  0.0], 0.5);
        let cube2  = GpuMesh::cube(&device, [-2.0, 0.4, -1.0], 0.4);
        let cube3  = GpuMesh::cube(&device, [ 2.0, 0.3,  0.5], 0.3);
        let ground = GpuMesh::plane(&device, [0.0, 0.0, 0.0], 20.0);
        // Thin slab roof sitting just above the colored lights (y=2.7..3.0).
        // rect3d gives independent extents: wide/deep but only 0.15 thick.
        let roof   = GpuMesh::rect3d(&device, [0.0, 2.85, 0.0], [4.5, 0.15, 4.5]);
        demo_portal::enable_live_dashboard(&mut renderer);

        renderer.add_object(&cube1,  None, glam::Mat4::IDENTITY);
        renderer.add_object(&cube2,  None, glam::Mat4::IDENTITY);
        renderer.add_object(&cube3,  None, glam::Mat4::IDENTITY);
        renderer.add_object(&ground, None, glam::Mat4::IDENTITY);
        renderer.add_object(&roof,   None, glam::Mat4::IDENTITY);

        self.state = Some(AppState {
            window, surface, device, surface_format, renderer,
            last_frame: std::time::Instant::now(),
            cube1, cube2, cube3, ground, roof,
            cam_pos:   glam::Vec3::new(0.0, 2.5, 7.0),
            cam_yaw:   0.0,
            cam_pitch: -0.2,
            keys:      HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            // Start at a nice afternoon angle (sun ~50° above horizon)
            sun_angle: 1.0,
            probe_vis: false,
            sprite_w,
            sprite_h,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };

        match event {
            WindowEvent::CloseRequested => { event_loop.exit(); }

            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    state: ElementState::Pressed,
                    physical_key: PhysicalKey::Code(KeyCode::Escape), ..
                }, ..
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
                event: KeyEvent {
                    state: ElementState::Pressed,
                    physical_key: PhysicalKey::Code(KeyCode::Digit3),
                    ..
                },
                ..
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
                event: KeyEvent {
                    state: ElementState::Pressed,
                    physical_key: PhysicalKey::Code(KeyCode::Digit4),
                    ..
                },
                ..
            } => { let _ = state.renderer.start_live_portal_default(); }

            WindowEvent::KeyboardInput {
                event: KeyEvent { state: ks, physical_key: PhysicalKey::Code(key), .. }, ..
            } => {
                match ks {
                    ElementState::Pressed  => { state.keys.insert(key); }
                    ElementState::Released => { state.keys.remove(&key); }
                }
            }

            WindowEvent::MouseInput {
                state: ElementState::Pressed, button: MouseButton::Left, ..
            } => {
                if !state.cursor_grabbed {
                    let grabbed = state.window.set_cursor_grab(CursorGrabMode::Confined)
                        .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked))
                        .is_ok();
                    if grabbed {
                        state.window.set_cursor_visible(false);
                        state.cursor_grabbed = true;
                    }
                }
            }

            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                let cfg = wgpu::SurfaceConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format: state.surface_format,
                    width: size.width, height: size.height,
                    present_mode: wgpu::PresentMode::Fifo,
                    alpha_mode: wgpu::CompositeAlphaMode::Auto,
                    view_formats: vec![],
                    desired_maximum_frame_latency: 2,
                };
                state.surface.configure(&state.device, &cfg);
                state.renderer.resize(size.width, size.height);
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
        const LOOK_SENS: f32 = 0.002;
        const SUN_SPEED: f32 = 0.5; // radians/sec

        // Sun rotation (Q/E keys)
        if self.keys.contains(&KeyCode::KeyQ) { self.sun_angle -= SUN_SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyE) { self.sun_angle += SUN_SPEED * dt; }

        // Camera look
        self.cam_yaw   += self.mouse_delta.0 * LOOK_SENS;
        self.cam_pitch  = (self.cam_pitch - self.mouse_delta.1 * LOOK_SENS).clamp(-1.5, 1.5);
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
            self.cam_pos,
            self.cam_pos + forward,
            glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            aspect, 0.1, 1000.0, time,
        );

        // Sun direction: orbits in the XY plane (rotate sun_angle around Z axis)
        let sun_dir = glam::Vec3::new(
            self.sun_angle.cos() * 0.3,
            self.sun_angle.sin(),
            0.5,
        ).normalize();
        // SceneLight direction = "ray direction" (toward scene), so negate the toward-sun vector
        let light_dir = [-sun_dir.x, -sun_dir.y, -sun_dir.z];

        // Sun intensity dims at horizon/night
        let sun_elev = sun_dir.y.clamp(-1.0, 1.0);
        let sun_lux  = (sun_elev * 3.0).clamp(0.0, 1.0);
        let sun_color = [
            1.0_f32.min(1.0 + (1.0 - sun_elev) * 0.3),  // warmer at horizon
            (0.85 + sun_elev * 0.15).clamp(0.0, 1.0),
            (0.7  + sun_elev * 0.3 ).clamp(0.0, 1.0),
        ];

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(e) => { log::warn!("Surface error: {:?}", e); return; }
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let billboards = if self.probe_vis {
            probe_billboards(RC_WORLD_MIN, RC_WORLD_MAX)
        } else {
            vec![
                BillboardInstance::new([ 0.0, 2.5,  0.0], [0.35, 0.35]).with_color([1.0, 0.85, 0.6, 1.0]),
                BillboardInstance::new([-2.5, 2.0, -1.5], [0.35, 0.35]).with_color([0.4, 0.6, 1.0, 1.0]),
                BillboardInstance::new([ 2.5, 1.8,  1.5], [0.35, 0.35]).with_color([1.0, 0.3, 0.3, 1.0]),
            ]
        };

        let env = SceneEnv {
            lights: vec![
                SceneLight::directional(light_dir, sun_color, (sun_lux * 0.35).max(0.01)),
                SceneLight::point([ 0.0, 2.5,  0.0], [1.0, 0.85, 0.6],  4.0, 8.0),
                SceneLight::point([-2.5, 2.0, -1.5], [0.4, 0.6,  1.0],  3.5, 7.0),
                SceneLight::point([ 2.5, 1.8,  1.5], [1.0, 0.3,  0.3],  3.0, 6.0),
            ],
            sky_atmosphere: Some(
                SkyAtmosphere::new()
                    .with_sun_intensity(22.0)
                    .with_exposure(4.0)
                    .with_mie_g(0.76)
                    .with_clouds(
                        VolumetricClouds::new()
                            .with_coverage(0.30)
                            .with_density(0.7)
                            .with_layer(800.0, 1800.0)
                            .with_wind([1.0, 0.0], 0.08),
                    ),
            ),
            skylight: Some(Skylight::new().with_intensity(0.08).with_tint([1.0, 1.0, 1.0])),
            billboards,
            ..Default::default()
        };
        self.renderer.set_scene_env(env);
        if let Err(e) = self.renderer.render(&camera, &view, dt) {
            log::error!("Render error: {:?}", e);
        }

        output.present();
    }
}
