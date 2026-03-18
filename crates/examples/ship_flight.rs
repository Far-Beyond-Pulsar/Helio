//! Ship Flight — pilot the embedded FBX through a deep-space asteroid field.
//!
//! The FBX model is placed as your ship in an open-space scene. A third-person
//! chase camera tracks it from behind and above. Fly through a scattered
//! asteroid field with Newtonian-lite physics: thrust builds speed, which slowly
//! decays in the vacuum.
//!
//! Run with:
//!   cargo run -p examples --bin ship_flight
//!
//! Controls:
//!   Mouse drag        — pitch / yaw the ship   (left-click to grab cursor)
//!   Q / E             — roll left / right
//!   W / S             — forward / reverse thrust
//!   A / D             — lateral (strafe) thrust
//!   Space / Shift     — vertical thrust
//!   F3                — toggle debug overlay
//!   Escape            — release cursor / exit

use glam::{EulerRot, Mat4, Quat, Vec3};
use helio_asset_compat::{load_scene_bytes_with_config, AssetError, ConvertedScene, LoadConfig};
use helio_render_v2::features::{BloomFeature, FeatureRegistry, LightingFeature, ShadowsFeature};
use helio_render_v2::{Camera, LightId, Material, ObjectId, Renderer, RendererConfig, SceneLight};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

const EMBEDDED_SCENE_BYTES: &[u8] = include_bytes!("../../test.fbx");

const ASTEROID_COUNT: usize = 90;
const LOOK_SENS: f32 = 0.0025;
const ROLL_SPEED: f32 = 1.2; // rad/sec

// The FBX mesh points up (+Y). Rotate it so the nose aligns with ship forward (−Z).
const MESH_BASE_ROT: Quat = Quat::from_xyzw(
    -std::f32::consts::FRAC_1_SQRT_2, // sin(-π/4)
    0.0,
    0.0,
    std::f32::consts::FRAC_1_SQRT_2,  // cos(-π/4)
);

fn main() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    log::info!("Helio — Ship Flight");

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("Event loop error");
}

fn base_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn load_ship() -> Result<(ConvertedScene, f32), AssetError> {
    let dir = base_dir();
    let scene = load_scene_bytes_with_config(
        EMBEDDED_SCENE_BYTES,
        "fbx",
        Some(dir.as_path()),
        LoadConfig::default().with_uv_flip(false),
    )?;
    let radius = scene_radius(&scene);
    Ok((scene, radius))
}

fn scene_radius(scene: &ConvertedScene) -> f32 {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    let mut found = false;
    for mesh in &scene.meshes {
        for v in &mesh.vertices {
            let p = Vec3::from(v.position);
            min = min.min(p);
            max = max.max(p);
            found = true;
        }
    }
    if found {
        ((max - min).length() * 0.5).max(1.0)
    } else {
        2.0
    }
}

// ── Deterministic pseudo-random (no external dep) ─────────────────────────────

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*seed >> 33) as f32) / (u32::MAX as f32)
}

fn rand_s(seed: &mut u64) -> f32 {
    lcg(seed) * 2.0 - 1.0
}

// ── Scene construction ─────────────────────────────────────────────────────────

fn build_asteroid_field(renderer: &mut Renderer, field_radius: f32, min_size: f32) {
    let rocky = renderer.upload_material(
        &Material::new()
            .with_base_color([0.15, 0.12, 0.09, 1.0])
            .with_roughness(0.90)
            .with_metallic(0.0),
    );
    let dark = renderer.upload_material(
        &Material::new()
            .with_base_color([0.09, 0.09, 0.11, 1.0])
            .with_roughness(0.70)
            .with_metallic(0.25),
    );

    let mut seed: u64 = 0xCAFE_BABE_1234_5678;

    for i in 0..ASTEROID_COUNT {
        // Distribute across the full shell (inner 25 % to outer 100 %)
        let dist = field_radius * (0.25 + lcg(&mut seed) * 0.75);
        let theta = lcg(&mut seed) * std::f32::consts::TAU;
        let phi = rand_s(&mut seed).asin();
        let pos = Vec3::new(
            dist * phi.cos() * theta.cos(),
            dist * phi.sin(),
            dist * phi.cos() * theta.sin(),
        );

        // Non-uniform scale to look boulder-like
        let base = min_size * (1.0 + lcg(&mut seed) * 9.0);
        let scale = Vec3::new(
            base * (0.6 + lcg(&mut seed) * 0.8),
            base * (0.5 + lcg(&mut seed) * 0.7),
            base * (0.6 + lcg(&mut seed) * 0.8),
        );

        let rot = Quat::from_euler(
            EulerRot::XYZ,
            rand_s(&mut seed) * std::f32::consts::PI,
            rand_s(&mut seed) * std::f32::consts::PI,
            rand_s(&mut seed) * std::f32::consts::PI,
        );

        let transform = Mat4::from_scale_rotation_translation(scale, rot, pos);
        let mat = if i % 3 == 0 { &dark } else { &rocky };
        let mesh = renderer.create_mesh_cube([0.0, 0.0, 0.0], 0.5);
        renderer.add_object(&mesh, Some(mat), transform);
    }
}

fn upload_ship_meshes(
    renderer: &mut Renderer,
    scene: &ConvertedScene,
    ship_radius: f32,
) -> Vec<ObjectId> {
    if scene.meshes.is_empty() {
        // Fallback: simple metallic box
        let mat = renderer.upload_material(
            &Material::new()
                .with_base_color([0.25, 0.40, 0.70, 1.0])
                .with_metallic(0.85)
                .with_roughness(0.25),
        );
        let mesh = renderer.create_mesh_cube([0.0, 0.0, 0.0], ship_radius);
        return vec![renderer.add_object(&mesh, Some(&mat), Mat4::IDENTITY)];
    }

    let gpu_mats: Vec<_> = scene
        .materials
        .iter()
        .map(|m| renderer.upload_material(m))
        .collect();

    scene
        .meshes
        .iter()
        .map(|mesh| {
            let gpu_mesh = renderer.create_mesh(&mesh.vertices, &mesh.indices);
            let mat = mesh.material_index.and_then(|i| gpu_mats.get(i));
            renderer.add_object(&gpu_mesh, mat, Mat4::IDENTITY)
        })
        .collect()
}

// ── Ship state ─────────────────────────────────────────────────────────────────

struct Ship {
    ids: Vec<ObjectId>,
    radius: f32,
    pos: Vec3,
    quat: Quat,
    velocity: Vec3,
    engine_light: LightId,
    thrusting: bool,
    thrust_accel: f32,
    max_speed: f32,
}

impl Ship {
    fn forward(&self) -> Vec3 {
        self.quat * -Vec3::Z
    }

    fn right(&self) -> Vec3 {
        self.quat * Vec3::X
    }

    fn up(&self) -> Vec3 {
        self.quat * Vec3::Y
    }

    fn engine_pos(&self) -> Vec3 {
        self.pos - self.forward() * self.radius * 0.8
    }

    fn push_transforms(&self, renderer: &mut Renderer) {
        let xform = Mat4::from_rotation_translation(self.quat * MESH_BASE_ROT, self.pos);
        for &id in &self.ids {
            renderer.update_transform(id, xform);
        }
    }

    fn chase_cam_pos(&self) -> Vec3 {
        self.pos - self.forward() * self.radius * 1.5 + self.up() * self.radius * 0.35
    }

    fn chase_cam_target(&self) -> Vec3 {
        self.pos + self.forward() * self.radius * 0.5
    }
}

// ── App ────────────────────────────────────────────────────────────────────────

struct App {
    state: Option<AppState>,
}

struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    surface_format: wgpu::TextureFormat,
    renderer: Renderer,
    start_time: Instant,
    last_frame: Instant,

    ship: Ship,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),
}

impl AppState {
    fn update(&mut self, dt: f32) {
        // ── Orientation ──────────────────────────────────────────────────────
        let yaw_delta = self.mouse_delta.0 * LOOK_SENS;
        let pitch_delta = self.mouse_delta.1 * LOOK_SENS;
        self.mouse_delta = (0.0, 0.0);

        let mut roll_delta = 0.0_f32;
        if self.keys.contains(&KeyCode::KeyQ) {
            roll_delta += ROLL_SPEED * dt;
        }
        if self.keys.contains(&KeyCode::KeyE) {
            roll_delta -= ROLL_SPEED * dt;
        }

        // Rotate in ship-local space so all axes feel consistent
        let rot = Quat::from_axis_angle(self.ship.up(), -yaw_delta)
            * Quat::from_axis_angle(self.ship.right(), -pitch_delta)
            * Quat::from_axis_angle(self.ship.forward(), roll_delta);
        self.ship.quat = (rot * self.ship.quat).normalize();

        // ── Thrust ───────────────────────────────────────────────────────────
        let mut thrust = Vec3::ZERO;
        if self.keys.contains(&KeyCode::KeyW) {
            thrust += self.ship.forward();
        }
        if self.keys.contains(&KeyCode::KeyS) {
            thrust -= self.ship.forward();
        }
        if self.keys.contains(&KeyCode::KeyA) {
            thrust -= self.ship.right();
        }
        if self.keys.contains(&KeyCode::KeyD) {
            thrust += self.ship.right();
        }
        if self.keys.contains(&KeyCode::Space) {
            thrust += self.ship.up();
        }
        if self.keys.contains(&KeyCode::ShiftLeft) {
            thrust -= self.ship.up();
        }

        self.ship.thrusting = thrust.length_squared() > 0.01;
        if self.ship.thrusting {
            let accel = thrust.normalize_or_zero() * self.ship.thrust_accel * dt;
            self.ship.velocity += accel;
        }

        let speed = self.ship.velocity.length();
        if speed > self.ship.max_speed {
            self.ship.velocity *= self.ship.max_speed / speed;
        }

        // Light space-drag so the ship gradually drifts to a stop
        self.ship.velocity *= 1.0 - (0.55 * dt);
        self.ship.pos += self.ship.velocity * dt;

        // ── Push GPU updates ─────────────────────────────────────────────────
        self.ship.push_transforms(&mut self.renderer);

        let glow = if self.ship.thrusting { 9.0 } else { 1.8 };
        self.renderer.update_light(
            self.ship.engine_light,
            SceneLight::point(
                self.ship.engine_pos().to_array(),
                [0.35, 0.65, 1.0],
                glow,
                self.ship.radius * 3.5,
            ),
        );
    }
}

impl App {
    fn new() -> Self {
        Self { state: None }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Helio — Ship Flight")
                        .with_inner_size(winit::dpi::PhysicalSize::new(1280, 720)),
                )
                .expect("Failed to create window"),
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::VALIDATION,
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
        .expect("No suitable GPU adapter found");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Ship Flight Device"),
                required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY
                    | wgpu::Features::TIMESTAMP_QUERY
                    | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
                required_limits: wgpu::Limits::default()
                    .using_minimum_supported_acceleration_structure_values(),
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                trace: wgpu::Trace::Off,
            },
        ))
        .expect("No suitable GPU device found");

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);

        let size = window.inner_size();
        surface.configure(
            &device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: surface_format,
                width: size.width,
                height: size.height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            },
        );

        let feature_registry = FeatureRegistry::builder()
            .with_feature(LightingFeature::new())
            .with_feature(BloomFeature::new().with_intensity(1.1).with_threshold(0.80))
            .with_feature(ShadowsFeature::new().with_atlas_size(2048).with_max_lights(8))
            .build();

        let mut renderer = Renderer::new(
            device.clone(),
            queue.clone(),
            RendererConfig::new(size.width, size.height, surface_format, feature_registry),
        )
        .expect("Failed to create renderer");

        // ── Load ship FBX ──────────────────────────────────────────────────────
        let (ship_radius, ship_ids) = match load_ship() {
            Ok((scene, radius)) => {
                log::info!(
                    "Ship '{}' — {} mesh(es), radius ≈ {:.2}",
                    scene.name,
                    scene.meshes.len(),
                    radius
                );
                let ids = upload_ship_meshes(&mut renderer, &scene, radius);
                (radius, ids)
            }
            Err(e) => {
                log::warn!("Failed to load embedded FBX: {}. Using fallback.", e);
                let fallback_radius = 2.0_f32;
                let mat = renderer.upload_material(
                    &Material::new()
                        .with_base_color([0.25, 0.40, 0.70, 1.0])
                        .with_metallic(0.85)
                        .with_roughness(0.25),
                );
                let mesh = renderer.create_mesh_cube([0.0, 0.0, 0.0], fallback_radius);
                let id = renderer.add_object(&mesh, Some(&mat), Mat4::IDENTITY);
                (fallback_radius, vec![id])
            }
        };

        // Scale environment to ship size
        let field_radius = (ship_radius * 100.0).clamp(250.0, 2500.0);
        let thrust_accel = (ship_radius * 4.5).clamp(8.0, 250.0);
        let max_speed = (ship_radius * 22.0).clamp(25.0, 700.0);

        // ── Space environment ──────────────────────────────────────────────────
        renderer.set_ambient([0.0, 0.0, 0.0], 0.0);
        renderer.set_sky_atmosphere(None);
        renderer.set_skylight(None);
        renderer.set_sky_color([0.0, 0.0, 0.008]); // near-black deep blue

        // Primary sun — warm-white, slightly above and to the right
        renderer.add_light(SceneLight::directional(
            [-0.55, -0.38, -0.74],
            [1.0, 0.97, 0.88],
            4.2,
        ));
        // Distant secondary star — cool blue from the opposite side
        renderer.add_light(SceneLight::directional(
            [0.72, 0.18, 0.68],
            [0.50, 0.70, 1.0],
            0.65,
        ));

        // ── Asteroid field ─────────────────────────────────────────────────────
        let min_asteroid = (ship_radius * 0.3).clamp(0.5, 8.0);
        build_asteroid_field(&mut renderer, field_radius, min_asteroid);

        // Engine thruster glow (starts dim, brightens when thrusting)
        let engine_light = renderer.add_light(SceneLight::point(
            [0.0, 0.0, ship_radius * 0.8],
            [0.35, 0.65, 1.0],
            1.8,
            ship_radius * 3.5,
        ));

        let now = Instant::now();
        let ship = Ship {
            ids: ship_ids,
            radius: ship_radius,
            pos: Vec3::ZERO,
            quat: Quat::IDENTITY,
            velocity: Vec3::ZERO,
            engine_light,
            thrusting: false,
            thrust_accel,
            max_speed,
        };

        self.state = Some(AppState {
            window,
            surface,
            device,
            surface_format,
            renderer,
            start_time: now,
            last_frame: now,
            ship,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                ..
            } => {
                if state.cursor_grabbed {
                    state.cursor_grabbed = false;
                    state.window.set_cursor_visible(true);
                    let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                } else {
                    event_loop.exit();
                }
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::F3),
                        ..
                    },
                ..
            } => {
                let on = !state.renderer.debug_viz().enabled;
                state.renderer.debug_viz_mut().enabled = on;
                log::info!("Debug overlay: {}", if on { "ON" } else { "OFF" });
            }

            WindowEvent::Resized(new_size) => {
                state.surface.configure(
                    &state.device,
                    &wgpu::SurfaceConfiguration {
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        format: state.surface_format,
                        width: new_size.width,
                        height: new_size.height,
                        present_mode: wgpu::PresentMode::Fifo,
                        alpha_mode: wgpu::CompositeAlphaMode::Opaque,
                        view_formats: vec![],
                        desired_maximum_frame_latency: 2,
                    },
                );
                state.renderer.resize(new_size.width, new_size.height);
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => match key_state {
                ElementState::Pressed => {
                    state.keys.insert(code);
                }
                ElementState::Released => {
                    state.keys.remove(&code);
                }
            },

            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if !state.cursor_grabbed {
                    let grabbed = state
                        .window
                        .set_cursor_grab(CursorGrabMode::Confined)
                        .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked))
                        .is_ok();
                    if grabbed {
                        state.cursor_grabbed = true;
                        state.window.set_cursor_visible(false);
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                // Cap dt to avoid large jumps if the window was minimised / hidden.
                let dt = now.duration_since(state.last_frame).as_secs_f32().min(0.05);
                state.last_frame = now;

                state.update(dt);

                let cam_pos = state.ship.chase_cam_pos();
                let cam_target = state.ship.chase_cam_target();
                let cam_up = state.ship.up();

                let size = state.window.inner_size();
                let aspect = size.width as f32 / size.height.max(1) as f32;

                let camera = Camera::perspective(
                    cam_pos,
                    cam_target,
                    cam_up,
                    std::f32::consts::FRAC_PI_4,
                    aspect,
                    0.05,
                    3000.0,
                    state.start_time.elapsed().as_secs_f32(),
                );

                let output = match state.surface.get_current_texture() {
                    Ok(t) => t,
                    Err(e) => {
                        log::warn!("Surface error: {:?}", e);
                        return;
                    }
                };
                let view = output
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                if let Err(e) = state.renderer.render(&camera, &view, dt) {
                    log::error!("Render error: {:?}", e);
                }

                output.present();
            }

            _ => {}
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        let Some(state) = &mut self.state else {
            return;
        };
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if state.cursor_grabbed {
                state.mouse_delta.0 += dx as f32;
                state.mouse_delta.1 += dy as f32;
            }
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}
