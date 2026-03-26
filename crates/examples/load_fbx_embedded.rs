//! Embedded FBX showcase using the `helio` wrapper.

mod v3_demo_common;

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use glam::Vec3;
use helio::{required_wgpu_features, required_wgpu_limits, Camera, Renderer, RendererConfig};
use helio_asset_compat::{
    load_scene_bytes_with_config, upload_scene_materials, AssetError, ConvertedScene, LoadConfig,
};
use v3_demo_common::{box_mesh, directional_light, make_material, plane_mesh, spot_light};
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

const EMBEDDED_SCENE_BYTES: &[u8] = include_bytes!("../../test.fbx");

#[derive(Clone, Copy, Debug)]
struct SceneBounds {
    min: Vec3,
    max: Vec3,
    center: Vec3,
    radius: f32,
}

impl SceneBounds {
    fn from_scene(scene: &ConvertedScene) -> Option<Self> {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        let mut found = false;
        for mesh in &scene.meshes {
            for vertex in &mesh.vertices {
                let p = Vec3::from_array(vertex.position);
                min = min.min(p);
                max = max.max(p);
                found = true;
            }
        }
        if !found {
            return None;
        }
        let center = (min + max) * 0.5;
        let extents = (max - min).max(Vec3::splat(0.1));
        Some(Self {
            min,
            max,
            center,
            radius: extents.length().max(2.5),
        })
    }

    fn floor_y(self) -> f32 {
        self.min.y - self.radius * 0.08
    }

    fn focus_point(self) -> Vec3 {
        self.center + Vec3::new(0.0, (self.max.y - self.min.y) * 0.18, 0.0)
    }

    fn camera_start(self) -> Vec3 {
        self.center + Vec3::new(self.radius * 0.55, self.radius * 0.28, self.radius * 1.55)
    }

    fn movement_speed(self) -> f32 {
        (self.radius * 0.85).clamp(8.0, 42.0)
    }

    fn stage_extent(self) -> f32 {
        self.radius * 1.55
    }
}

fn embedded_scene_base_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn load_embedded_scene() -> Result<(ConvertedScene, SceneBounds), AssetError> {
    let base_dir = embedded_scene_base_dir();
    let scene = load_scene_bytes_with_config(
        EMBEDDED_SCENE_BYTES,
        "fbx",
        Some(base_dir.as_path()),
        LoadConfig::default().with_uv_flip(false),
    )?;
    let bounds = SceneBounds::from_scene(&scene).ok_or_else(|| {
        AssetError::InvalidData("embedded FBX scene did not contain any vertices".to_string())
    })?;
    Ok((scene, bounds))
}

fn look_angles(direction: Vec3) -> (f32, f32) {
    let dir = direction.normalize_or_zero();
    (dir.x.atan2(-dir.z), dir.y.asin())
}

fn add_showcase_stage(renderer: &mut Renderer, bounds: SceneBounds) {
    let floor_mesh = renderer.scene_mut().insert_mesh(plane_mesh(
        [bounds.center.x, bounds.floor_y(), bounds.center.z],
        bounds.stage_extent(),
    ));
    let floor_material = renderer.scene_mut().insert_material(make_material(
        [0.07, 0.08, 0.10, 1.0],
        0.16,
        0.02,
        [0.0, 0.0, 0.0],
        0.0,
    ));
    let _ = v3_demo_common::insert_object(
        renderer,
        floor_mesh,
        floor_material,
        glam::Mat4::IDENTITY,
        bounds.stage_extent(),
    );

    let pedestal_mesh = renderer.scene_mut().insert_mesh(box_mesh(
        [
            bounds.center.x,
            bounds.floor_y() + bounds.radius * 0.05,
            bounds.center.z,
        ],
        [
            bounds.radius * 0.62,
            bounds.radius * 0.05,
            bounds.radius * 0.62,
        ],
    ));
    let pedestal_material = renderer.scene_mut().insert_material(make_material(
        [0.11, 0.12, 0.15, 1.0],
        0.28,
        0.04,
        [0.0, 0.0, 0.0],
        0.0,
    ));
    let _ = v3_demo_common::insert_object(
        renderer,
        pedestal_mesh,
        pedestal_material,
        glam::Mat4::IDENTITY,
        bounds.radius,
    );

    let backdrop_mesh = renderer.scene_mut().insert_mesh(box_mesh(
        [
            bounds.center.x,
            bounds.floor_y() + bounds.radius * 0.62,
            bounds.center.z - bounds.radius * 1.35,
        ],
        [
            bounds.radius * 1.35,
            bounds.radius * 0.62,
            bounds.radius * 0.05,
        ],
    ));
    let backdrop_material = renderer.scene_mut().insert_material(make_material(
        [0.04, 0.05, 0.08, 1.0],
        0.82,
        0.0,
        [0.04, 0.06, 0.12],
        0.03,
    ));
    let _ = v3_demo_common::insert_object(
        renderer,
        backdrop_mesh,
        backdrop_material,
        glam::Mat4::IDENTITY,
        bounds.radius * 1.5,
    );
}

fn add_showcase_lighting(renderer: &mut Renderer, bounds: SceneBounds) {
    let focus = bounds.focus_point();
    let radius = bounds.radius;
    let elevated_focus = focus + Vec3::new(0.0, radius * 0.08, 0.0);
    let upper_focus = focus + Vec3::new(0.0, radius * 0.18, 0.0);

    let key_pos = focus + Vec3::new(radius * 0.22, radius * 0.34, radius * 0.24);
    let key_dir = (elevated_focus - key_pos).normalize_or_zero();
    renderer.scene_mut().insert_light(spot_light(
        key_pos.to_array(),
        key_dir.to_array(),
        [1.0, 0.80, 0.62],
        18.0,
        radius * 0.62,
        0.20,
        0.38,
    ));

    let fill_pos = focus + Vec3::new(-radius * 0.26, radius * 0.14, radius * 0.28);
    let fill_dir = (focus - fill_pos).normalize_or_zero();
    renderer.scene_mut().insert_light(spot_light(
        fill_pos.to_array(),
        fill_dir.to_array(),
        [0.52, 0.66, 1.0],
        6.5,
        radius * 0.59,
        0.28,
        0.46,
    ));

    let rim_pos = focus + Vec3::new(-radius * 0.30, radius * 0.22, -radius * 0.32);
    let rim_dir = (upper_focus - rim_pos).normalize_or_zero();
    renderer.scene_mut().insert_light(spot_light(
        rim_pos.to_array(),
        rim_dir.to_array(),
        [0.36, 0.55, 1.0],
        14.0,
        radius * 0.57,
        0.22,
        0.40,
    ));

    renderer.scene_mut().insert_light(directional_light(
        [0.15, -1.0, 0.1],
        [0.07, 0.09, 0.14],
        0.3,
    ));
    renderer.set_ambient([0.0, 0.0, 0.0], 0.0);
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
    movement_speed: f32,
    last_frame: Instant,
    cam_pos: Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),
}

impl App {
    fn new() -> Self {
        Self { state: None }
    }
}

impl AppState {
    fn update_camera(&mut self, dt: f32) -> Vec3 {
        const LOOK_SENS: f32 = 0.002;
        self.cam_yaw += self.mouse_delta.0 * LOOK_SENS;
        self.cam_pitch = (self.cam_pitch - self.mouse_delta.1 * LOOK_SENS).clamp(-1.5, 1.5);
        self.mouse_delta = (0.0, 0.0);

        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let forward = Vec3::new(sy * cp, sp, -cy * cp);
        let right = Vec3::new(cy, 0.0, sy);
        let up = Vec3::Y;

        if self.keys.contains(&KeyCode::KeyW) {
            self.cam_pos += forward * self.movement_speed * dt;
        }
        if self.keys.contains(&KeyCode::KeyS) {
            self.cam_pos -= forward * self.movement_speed * dt;
        }
        if self.keys.contains(&KeyCode::KeyA) {
            self.cam_pos -= right * self.movement_speed * dt;
        }
        if self.keys.contains(&KeyCode::KeyD) {
            self.cam_pos += right * self.movement_speed * dt;
        }
        if self.keys.contains(&KeyCode::Space) {
            self.cam_pos += up * self.movement_speed * dt;
        }
        if self.keys.contains(&KeyCode::ShiftLeft) {
            self.cam_pos -= up * self.movement_speed * dt;
        }
        forward
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
                        .with_title("Helio - Embedded FBX Showcase")
                        .with_inner_size(winit::dpi::PhysicalSize::new(1280, 720)),
                )
                .expect("failed to create window"),
        );
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .expect("failed to create surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("no adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: required_wgpu_features(adapter.features()),
                required_limits: required_wgpu_limits(adapter.limits()),
                ..Default::default()
            },
        ))
        .expect("no device");
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
        let mut renderer = Renderer::new(
            device.clone(),
            queue,
            RendererConfig::new(size.width, size.height, surface_format),
        );
        renderer.set_clear_color([0.01, 0.01, 0.02, 1.0]);

        let (scene, bounds) = match load_embedded_scene() {
            Ok(result) => result,
            Err(error) => {
                log::error!("failed to load embedded scene: {}", error);
                let fallback_bounds = SceneBounds {
                    min: Vec3::new(-0.75, 0.0, -0.75),
                    max: Vec3::new(0.75, 1.5, 0.75),
                    center: Vec3::new(0.0, 0.75, 0.0),
                    radius: 3.0,
                };
                let mesh = renderer.scene_mut().insert_mesh(box_mesh([0.0, 0.75, 0.0], [0.75, 0.75, 0.75]));
                let material = renderer.scene_mut().insert_material(make_material(
                    [0.65, 0.72, 0.9, 1.0],
                    0.35,
                    0.1,
                    [0.0, 0.0, 0.0],
                    0.0,
                ));
                let _ = v3_demo_common::insert_object(
                    &mut renderer,
                    mesh,
                    material,
                    glam::Mat4::IDENTITY,
                    1.5,
                );
                add_showcase_stage(&mut renderer, fallback_bounds);
                add_showcase_lighting(&mut renderer, fallback_bounds);
                let camera_start = fallback_bounds.camera_start();
                let focus = fallback_bounds.focus_point();
                let (cam_yaw, cam_pitch) = look_angles(focus - camera_start);
                self.state = Some(AppState {
                    window,
                    surface,
                    device,
                    surface_format,
                    renderer,
                    movement_speed: fallback_bounds.movement_speed(),
                    last_frame: Instant::now(),
                    cam_pos: camera_start,
                    cam_yaw,
                    cam_pitch,
                    keys: HashSet::new(),
                    cursor_grabbed: false,
                    mouse_delta: (0.0, 0.0),
                });
                return;
            }
        };

        let material_ids =
            upload_scene_materials(&mut renderer, &scene).expect("upload scene materials");
        for mesh in scene.meshes {
            let radius = mesh
                .vertices
                .iter()
                .map(|v| Vec3::from_array(v.position).distance(bounds.center))
                .fold(0.5, f32::max);
            let mesh_id = renderer.scene_mut().insert_mesh(helio::MeshUpload {
                vertices: mesh.vertices,
                indices: mesh.indices,
            });
            if let Some(material) = mesh
                .material_index
                .and_then(|index| material_ids.get(index).copied())
            {
                let _ = v3_demo_common::insert_object(
                    &mut renderer,
                    mesh_id,
                    material,
                    glam::Mat4::IDENTITY,
                    radius,
                );
            }
        }
        add_showcase_stage(&mut renderer, bounds);
        add_showcase_lighting(&mut renderer, bounds);
        let camera_start = bounds.camera_start();
        let focus = bounds.focus_point();
        let (cam_yaw, cam_pitch) = look_angles(focus - camera_start);

        self.state = Some(AppState {
            window,
            surface,
            device,
            surface_format,
            renderer,
            movement_speed: bounds.movement_speed(),
            last_frame: Instant::now(),
            cam_pos: camera_start,
            cam_yaw,
            cam_pitch,
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
                    let _ = state
                        .window
                        .set_cursor_grab(winit::window::CursorGrabMode::None);
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
                        .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                        .or_else(|_| {
                            state
                                .window
                                .set_cursor_grab(winit::window::CursorGrabMode::Locked)
                        })
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
                let forward = state.update_camera(dt);
                let size = state.window.inner_size();
                let camera = Camera::perspective_look_at(
                    state.cam_pos,
                    state.cam_pos + forward,
                    Vec3::Y,
                    std::f32::consts::FRAC_PI_4,
                    size.width as f32 / size.height.max(1) as f32,
                    0.1,
                    200.0,
                );
                let output = match state.surface.get_current_texture() {
                    Ok(texture) => texture,
                    Err(error) => {
                        log::warn!("surface error: {:?}", error);
                        return;
                    }
                };
                let view = output
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
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
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();
    let event_loop = EventLoop::new().expect("failed to create event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("event loop error");
}

