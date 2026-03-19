//! Ship Flight — pilot the embedded FBX through a sparse asteroid field with `helio`.

mod v3_demo_common;

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use glam::{EulerRot, Mat4, Quat, Vec3};
use helio::{required_wgpu_features, required_wgpu_limits, Camera, Renderer, RendererConfig};
use helio_asset_compat::{
    load_scene_bytes_with_config, upload_scene_materials, AssetError, ConvertedScene, LoadConfig,
};
use v3_demo_common::{cube_mesh, directional_light, make_material, point_light};
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

const EMBEDDED_SCENE_BYTES: &[u8] = include_bytes!("../../test.fbx");
const ASTEROID_COUNT: usize = 900;
const LOCAL_ASTEROID_COUNT: usize = 320;
const ASTEROID_FIELD_SCALE: f32 = 180.0;
const ASTEROID_FIELD_MIN_RADIUS: f32 = 900.0;
const ASTEROID_FIELD_MAX_RADIUS: f32 = 7000.0;
const LOOK_SENS: f32 = 0.0024;
const ROLL_SPEED: f32 = 1.9;
const SHIP_POSITION_LAG: f32 = 12.0;
const SHIP_ROTATION_LAG: f32 = 14.0;
const CAMERA_POSITION_LAG: f32 = 8.5;
const CAMERA_TARGET_LAG: f32 = 9.5;
const CAMERA_UP_LAG: f32 = 10.5;
const YAW_THRUST: f32 = 9.0;
const PITCH_THRUST: f32 = 8.0;
const ROLL_THRUST: f32 = 2.6;
const ANGULAR_DAMPING: f32 = 9.0;
const FORWARD_THRUST_SCALE: f32 = 0.9;
const REVERSE_THRUST_SCALE: f32 = 0.5;
const STRAFE_THRUST_SCALE: f32 = 0.62;
const LIFT_THRUST_SCALE: f32 = 0.58;
const FORWARD_DRAG: f32 = 0.28;
const LATERAL_DRAG: f32 = 3.4;
const VERTICAL_DRAG: f32 = 2.9;
const MESH_BASE_ROT: Quat = Quat::from_xyzw(
    -std::f32::consts::FRAC_1_SQRT_2,
    0.0,
    0.0,
    std::f32::consts::FRAC_1_SQRT_2,
);

fn base_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

#[derive(Clone, Copy, Debug)]
struct ShipBounds {
    center: Vec3,
    radius: f32,
}

impl ShipBounds {
    fn from_scene(scene: &ConvertedScene) -> Option<Self> {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        let mut found = false;
        for mesh in &scene.meshes {
            for vertex in &mesh.vertices {
                let position = Vec3::from_array(vertex.position);
                min = min.min(position);
                max = max.max(position);
                found = true;
            }
        }

        if !found {
            return None;
        }

        let extents = (max - min).max(Vec3::splat(0.1));
        Some(Self {
            center: (min + max) * 0.5,
            radius: (extents.length() * 0.5).max(1.0),
        })
    }
}

fn load_ship() -> Result<(ConvertedScene, ShipBounds), AssetError> {
    let dir = base_dir();
    let scene = load_scene_bytes_with_config(
        EMBEDDED_SCENE_BYTES,
        "fbx",
        Some(dir.as_path()),
        LoadConfig::default().with_uv_flip(false),
    )?;
    let bounds = ShipBounds::from_scene(&scene)
        .ok_or_else(|| AssetError::InvalidData("embedded ship scene did not contain any vertices".to_string()))?;
    Ok((scene, bounds))
}

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*seed >> 33) as f32) / (u32::MAX as f32)
}

fn rand_s(seed: &mut u64) -> f32 {
    lcg(seed) * 2.0 - 1.0
}

fn follow_factor(strength: f32, dt: f32) -> f32 {
    1.0 - (-strength * dt).exp()
}

fn build_asteroid_field(renderer: &mut Renderer, ship_radius: f32, field_radius: f32, min_size: f32) {
    let rocky = renderer.insert_material(make_material([0.15, 0.12, 0.09, 1.0], 0.90, 0.0, [0.0, 0.0, 0.0], 0.0));
    let dark = renderer.insert_material(make_material([0.09, 0.09, 0.11, 1.0], 0.70, 0.25, [0.0, 0.0, 0.0], 0.0));
    let cube = renderer.insert_mesh(cube_mesh([0.0, 0.0, 0.0], 0.5));

    let local_radius = (ship_radius * 40.0).clamp(120.0, 420.0);
    let spawn_asteroid = |renderer: &mut Renderer, seed: &mut u64, i: usize, dist: f32, size_bias: f32| {
        let theta = lcg(seed) * std::f32::consts::TAU;
        let phi = rand_s(seed).asin();
        let pos = Vec3::new(
            dist * phi.cos() * theta.cos(),
            dist * phi.sin(),
            dist * phi.cos() * theta.sin(),
        );
        let base = min_size * size_bias * (1.0 + lcg(seed) * 9.0);
        let scale = Vec3::new(
            base * (0.6 + lcg(seed) * 0.8),
            base * (0.5 + lcg(seed) * 0.7),
            base * (0.6 + lcg(seed) * 0.8),
        );
        let rot = Quat::from_euler(
            EulerRot::XYZ,
            rand_s(seed) * std::f32::consts::PI,
            rand_s(seed) * std::f32::consts::PI,
            rand_s(seed) * std::f32::consts::PI,
        );
        let material = if i % 3 == 0 { dark } else { rocky };
        let transform = Mat4::from_scale_rotation_translation(scale, rot, pos);
        let _ = v3_demo_common::insert_object(renderer, cube, material, transform, base);
    };

    let mut seed: u64 = 0xCAFE_BABE_1234_5678;
    for i in 0..LOCAL_ASTEROID_COUNT {
        let dist = ship_radius * 10.0 + lcg(&mut seed) * local_radius;
        spawn_asteroid(renderer, &mut seed, i, dist, 0.85);
    }
    for i in 0..ASTEROID_COUNT {
        let dist = field_radius * (0.12 + lcg(&mut seed) * 0.88);
        spawn_asteroid(renderer, &mut seed, i + LOCAL_ASTEROID_COUNT, dist, 1.0);
    }
}

fn upload_ship_meshes(renderer: &mut Renderer, scene: &ConvertedScene, ship_bounds: ShipBounds) -> Vec<helio::ObjectId> {
    if scene.meshes.is_empty() {
        let mat = renderer.insert_material(make_material([0.25, 0.40, 0.70, 1.0], 0.25, 0.85, [0.0, 0.0, 0.0], 0.0));
        let mesh = renderer.insert_mesh(cube_mesh([0.0, 0.0, 0.0], ship_bounds.radius));
        return vec![v3_demo_common::insert_object(renderer, mesh, mat, Mat4::IDENTITY, ship_bounds.radius).expect("fallback object")];
    }

    let material_ids = upload_scene_materials(renderer, scene).expect("upload ship materials");
    scene.meshes
        .iter()
        .map(|mesh| {
            let mut vertices = mesh.vertices.clone();
            for vertex in &mut vertices {
                let centered = Vec3::from_array(vertex.position) - ship_bounds.center;
                vertex.position = centered.to_array();
            }
            let mesh_id = renderer.insert_mesh(helio::MeshUpload {
                vertices,
                indices: mesh.indices.clone(),
            });
            let material = mesh.material_index.and_then(|i| material_ids.get(i).copied()).unwrap_or(material_ids[0]);
            v3_demo_common::insert_object(renderer, mesh_id, material, Mat4::IDENTITY, ship_bounds.radius).expect("ship object")
        })
        .collect()
}

struct Ship {
    ids: Vec<helio::ObjectId>,
    radius: f32,
    pos: Vec3,
    quat: Quat,
    render_pos: Vec3,
    render_quat: Quat,
    velocity: Vec3,
    angular_velocity: Vec3,
    engine_light: helio::LightId,
    thrusting: bool,
    thrust_accel: f32,
    max_speed: f32,
}

impl Ship {
    fn forward(&self) -> Vec3 { self.quat * -Vec3::Z }
    fn right(&self) -> Vec3 { self.quat * Vec3::X }
    fn up(&self) -> Vec3 { self.quat * Vec3::Y }
    fn render_forward(&self) -> Vec3 { self.render_quat * -Vec3::Z }
    fn render_up(&self) -> Vec3 { self.render_quat * Vec3::Y }
    fn engine_pos(&self) -> Vec3 { self.render_pos - self.render_forward() * self.radius * 0.8 }
    fn update_visual_follow(&mut self, dt: f32) {
        self.render_pos = self.render_pos.lerp(self.pos, follow_factor(SHIP_POSITION_LAG, dt));
        self.render_quat = self
            .render_quat
            .slerp(self.quat, follow_factor(SHIP_ROTATION_LAG, dt))
            .normalize();
    }
    fn push_transforms(&self, renderer: &mut Renderer) {
        let transform = Mat4::from_rotation_translation(self.render_quat * MESH_BASE_ROT, self.render_pos);
        for &id in &self.ids {
            let _ = renderer.update_object_transform(id, transform);
        }
    }
    fn desired_cam_pos(&self) -> Vec3 {
        self.render_pos - self.render_forward() * self.radius * 3.2 + self.render_up() * self.radius * 0.95
    }
    fn desired_cam_target(&self) -> Vec3 {
        self.render_pos + self.render_forward() * self.radius * 1.15 + self.render_up() * self.radius * 0.18
    }
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
    last_frame: Instant,
    ship: Ship,
    camera_pos: Vec3,
    camera_target: Vec3,
    camera_up: Vec3,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),
}

impl AppState {
    fn update(&mut self, dt: f32) {
        let yaw_delta = self.mouse_delta.0 * LOOK_SENS;
        let pitch_delta = self.mouse_delta.1 * LOOK_SENS;
        self.mouse_delta = (0.0, 0.0);

        let mut roll_input = 0.0;
        if self.keys.contains(&KeyCode::KeyQ) { roll_input += ROLL_SPEED; }
        if self.keys.contains(&KeyCode::KeyE) { roll_input -= ROLL_SPEED; }

        self.ship.angular_velocity += Vec3::new(
            -pitch_delta * PITCH_THRUST,
            -yaw_delta * YAW_THRUST,
            roll_input * ROLL_THRUST * dt,
        );
        self.ship.angular_velocity /= 1.0 + ANGULAR_DAMPING * dt;

        let local_rot = Quat::from_euler(
            EulerRot::XYZ,
            self.ship.angular_velocity.x * dt,
            self.ship.angular_velocity.y * dt,
            self.ship.angular_velocity.z * dt,
        );
        self.ship.quat = (self.ship.quat * local_rot).normalize();

        let mut local_velocity = self.ship.quat.conjugate() * self.ship.velocity;
        let mut thrusting = false;

        if self.keys.contains(&KeyCode::KeyW) {
            local_velocity.z -= self.ship.thrust_accel * FORWARD_THRUST_SCALE * dt;
            thrusting = true;
        }
        if self.keys.contains(&KeyCode::KeyS) {
            local_velocity.z += self.ship.thrust_accel * REVERSE_THRUST_SCALE * dt;
            thrusting = true;
        }
        if self.keys.contains(&KeyCode::KeyA) {
            local_velocity.x -= self.ship.thrust_accel * STRAFE_THRUST_SCALE * dt;
            thrusting = true;
        }
        if self.keys.contains(&KeyCode::KeyD) {
            local_velocity.x += self.ship.thrust_accel * STRAFE_THRUST_SCALE * dt;
            thrusting = true;
        }
        if self.keys.contains(&KeyCode::Space) {
            local_velocity.y += self.ship.thrust_accel * LIFT_THRUST_SCALE * dt;
            thrusting = true;
        }
        if self.keys.contains(&KeyCode::ShiftLeft) {
            local_velocity.y -= self.ship.thrust_accel * LIFT_THRUST_SCALE * dt;
            thrusting = true;
        }

        local_velocity.x /= 1.0 + LATERAL_DRAG * dt;
        local_velocity.y /= 1.0 + VERTICAL_DRAG * dt;
        local_velocity.z /= 1.0 + FORWARD_DRAG * dt;

        self.ship.velocity = self.ship.quat * local_velocity;
        self.ship.thrusting = thrusting;

        let speed = self.ship.velocity.length();
        if speed > self.ship.max_speed {
            self.ship.velocity *= self.ship.max_speed / speed;
        }
        self.ship.pos += self.ship.velocity * dt;

        self.ship.update_visual_follow(dt);
        self.ship.push_transforms(&mut self.renderer);
        self.camera_pos = self
            .camera_pos
            .lerp(self.ship.desired_cam_pos(), follow_factor(CAMERA_POSITION_LAG, dt));
        self.camera_target = self
            .camera_target
            .lerp(self.ship.desired_cam_target(), follow_factor(CAMERA_TARGET_LAG, dt));
        self.camera_up = self
            .camera_up
            .lerp(self.ship.render_up(), follow_factor(CAMERA_UP_LAG, dt))
            .normalize_or_zero();
        if self.camera_up.length_squared() < 1.0e-4 {
            self.camera_up = Vec3::Y;
        }
        let glow = if self.ship.thrusting { 9.0 } else { 1.8 };
        let _ = self.renderer.update_light(
            self.ship.engine_light,
            point_light(self.ship.engine_pos().to_array(), [0.35, 0.65, 1.0], glow, self.ship.radius * 3.5),
        );
    }
}

impl App {
    fn new() -> Self { Self { state: None } }
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
                .expect("failed to create window"),
        );
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).expect("failed to create surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("no adapter");
        let (device, queue) = pollster::block_on(
            adapter.request_device(
                &wgpu::DeviceDescriptor {
                    required_features: required_wgpu_features(adapter.features()),
                    required_limits: required_wgpu_limits(adapter.limits()),
                    ..Default::default()
                },
                None,
            ),
        )
        .expect("no device");
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps.formats.iter().find(|f| f.is_srgb()).copied().unwrap_or(caps.formats[0]);
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

        let mut renderer = Renderer::new(device.clone(), queue, RendererConfig::new(size.width, size.height, surface_format));
        renderer.set_clear_color([0.0, 0.0, 0.01, 1.0]);
        renderer.set_ambient([0.0, 0.0, 0.0], 0.0);
        renderer.insert_light(directional_light([-0.55, -0.38, -0.74], [1.0, 0.97, 0.88], 4.2));
        renderer.insert_light(directional_light([0.72, 0.18, 0.68], [0.50, 0.70, 1.0], 0.65));

        let (ship_radius, ship_ids) = match load_ship() {
            Ok((scene, bounds)) => (bounds.radius, upload_ship_meshes(&mut renderer, &scene, bounds)),
            Err(error) => {
                log::warn!("failed to load embedded ship: {}. using fallback cube.", error);
                let material = renderer.insert_material(make_material([0.25, 0.40, 0.70, 1.0], 0.25, 0.85, [0.0, 0.0, 0.0], 0.0));
                let mesh = renderer.insert_mesh(cube_mesh([0.0, 0.0, 0.0], 2.0));
                (2.0, vec![v3_demo_common::insert_object(&mut renderer, mesh, material, Mat4::IDENTITY, 2.0).expect("fallback ship")])
            }
        };

        let field_radius = (ship_radius * ASTEROID_FIELD_SCALE)
            .clamp(ASTEROID_FIELD_MIN_RADIUS, ASTEROID_FIELD_MAX_RADIUS);
        let thrust_accel = (ship_radius * 4.8).clamp(10.0, 220.0);
        let max_speed = (ship_radius * 22.0).clamp(35.0, 520.0);
        build_asteroid_field(&mut renderer, ship_radius, field_radius, (ship_radius * 0.3).clamp(0.5, 8.0));

        let engine_light = renderer.insert_light(point_light([0.0, 0.0, ship_radius * 0.8], [0.35, 0.65, 1.0], 1.8, ship_radius * 3.5));
        let mut ship = Ship {
            ids: ship_ids,
            radius: ship_radius,
            pos: Vec3::ZERO,
            quat: Quat::IDENTITY,
            render_pos: Vec3::ZERO,
            render_quat: Quat::IDENTITY,
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            engine_light,
            thrusting: false,
            thrust_accel,
            max_speed,
        };
        ship.push_transforms(&mut renderer);
        let camera_pos = ship.desired_cam_pos();
        let camera_target = ship.desired_cam_target();
        let camera_up = ship.render_up();

        self.state = Some(AppState {
            window,
            surface,
            device,
            surface_format,
            renderer,
            last_frame: Instant::now(),
            ship,
            camera_pos,
            camera_target,
            camera_up,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event: KeyEvent {
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
            WindowEvent::Resized(size) => {
                state.surface.configure(
                    &state.device,
                    &wgpu::SurfaceConfiguration {
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        format: state.surface_format,
                        width: size.width,
                        height: size.height,
                        present_mode: wgpu::PresentMode::Fifo,
                        alpha_mode: wgpu::CompositeAlphaMode::Opaque,
                        view_formats: vec![],
                        desired_maximum_frame_latency: 2,
                    },
                );
                state.renderer.set_render_size(size.width, size.height);
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    physical_key: PhysicalKey::Code(code),
                    state: key_state,
                    ..
                },
                ..
            } => match key_state {
                ElementState::Pressed => { state.keys.insert(code); }
                ElementState::Released => { state.keys.remove(&code); }
            },
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if !state.cursor_grabbed {
                    let grabbed = state.window.set_cursor_grab(CursorGrabMode::Confined)
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
                let dt = now.duration_since(state.last_frame).as_secs_f32().min(0.05);
                state.last_frame = now;
                state.update(dt);

                let size = state.window.inner_size();
                let camera = Camera::perspective_look_at(
                    state.camera_pos,
                    state.camera_target,
                    state.camera_up,
                    std::f32::consts::FRAC_PI_4,
                    size.width as f32 / size.height.max(1) as f32,
                    0.05,
                    3000.0,
                );
                let output = match state.surface.get_current_texture() {
                    Ok(texture) => texture,
                    Err(error) => {
                        log::warn!("surface error: {:?}", error);
                        return;
                    }
                };
                let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
                if let Err(error) = state.renderer.render(&camera, &view) {
                    log::error!("render error: {:?}", error);
                }
                output.present();
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        let Some(state) = &mut self.state else { return };
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

fn main() {
    env_logger::Builder::from_default_env().filter_level(log::LevelFilter::Info).init();
    let event_loop = EventLoop::new().expect("failed to create event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("event loop error");
}
