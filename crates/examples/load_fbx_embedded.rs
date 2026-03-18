//! Embedded FBX showcase scene with curated lighting.
//!
//! This example embeds the sample FBX directly into the binary and presents it
//! in a small staged environment with a fixed cinematic light rig.
//!
//! Run with:
//!   cargo run -p examples --bin load_fbx_embedded

use glam::Vec3;
use helio_asset_compat::{load_scene_bytes_with_config, AssetError, ConvertedScene, LoadConfig};
use helio_render_v2::features::{BloomFeature, FeatureRegistry, LightingFeature, ShadowsFeature};
use helio_render_v2::{
    Camera, Material, Renderer, RendererConfig, SceneLight, SkyAtmosphere, Skylight,
    VolumetricClouds,
};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

const EMBEDDED_SCENE_BYTES: &[u8] = include_bytes!("../../test.fbx");

fn main() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    log::info!("Helio Embedded FBX Showcase");

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("Event loop error");
}

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
        let mut found_vertex = false;

        for mesh in &scene.meshes {
            for vertex in &mesh.vertices {
                let position = Vec3::from(vertex.position);
                min = min.min(position);
                max = max.max(position);
                found_vertex = true;
            }
        }

        if !found_vertex {
            return None;
        }

        let center = (min + max) * 0.5;
        let extents = (max - min).max(Vec3::splat(0.1));
        let radius = extents.length().max(2.5);

        Some(Self {
            min,
            max,
            center,
            radius,
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
        AssetError::InvalidData("Embedded FBX scene did not contain any vertices".to_string())
    })?;
    Ok((scene, bounds))
}

fn look_angles(direction: Vec3) -> (f32, f32) {
    let dir = direction.normalize_or_zero();
    let yaw = dir.x.atan2(-dir.z);
    let pitch = dir.y.asin();
    (yaw, pitch)
}

fn add_showcase_stage(renderer: &mut Renderer, bounds: SceneBounds) {
    let floor = renderer.create_mesh_plane([bounds.center.x, bounds.floor_y(), bounds.center.z], bounds.stage_extent());
    let pedestal = renderer.create_mesh_rect3d(
        [bounds.center.x, bounds.floor_y() + bounds.radius * 0.05, bounds.center.z],
        [bounds.radius * 0.62, bounds.radius * 0.05, bounds.radius * 0.62],
    );
    let backdrop = renderer.create_mesh_rect3d(
        [
            bounds.center.x,
            bounds.floor_y() + bounds.radius * 0.62,
            bounds.center.z - bounds.radius * 1.35,
        ],
        [bounds.radius * 1.35, bounds.radius * 0.62, bounds.radius * 0.05],
    );

    let floor_material = renderer.upload_material(
        &Material::new()
            .with_base_color([0.07, 0.08, 0.10, 1.0])
            .with_roughness(0.16)
            .with_metallic(0.02),
    );
    let pedestal_material = renderer.upload_material(
        &Material::new()
            .with_base_color([0.11, 0.12, 0.15, 1.0])
            .with_roughness(0.28)
            .with_metallic(0.04),
    );
    let backdrop_material = renderer.upload_material(
        &Material::new()
            .with_base_color([0.04, 0.05, 0.08, 1.0])
            .with_roughness(0.82)
            .with_emissive([0.04, 0.06, 0.12], 0.2),
    );

    renderer.add_object(&floor, Some(&floor_material), glam::Mat4::IDENTITY);
    renderer.add_object(&pedestal, Some(&pedestal_material), glam::Mat4::IDENTITY);
    renderer.add_object(&backdrop, Some(&backdrop_material), glam::Mat4::IDENTITY);
}

fn add_showcase_lighting(renderer: &mut Renderer, bounds: SceneBounds) {
    let focus = bounds.focus_point();
    let radius = bounds.radius;

    let sun_dir = Vec3::new(-0.35, -0.82, -0.45).normalize();
    renderer.add_light(SceneLight::directional(
        sun_dir.to_array(),
        [0.62, 0.70, 0.95],
        0.85,
    ));

    let key_pos = focus + Vec3::new(radius * 0.95, radius * 0.75, radius * 1.05);
    let key_dir = (focus - key_pos).normalize_or_zero();
    renderer.add_light(SceneLight::spot(
        key_pos.to_array(),
        key_dir.to_array(),
        [1.0, 0.78, 0.58],
        20.0,
        radius * 4.2,
        0.42,
        0.78,
    ));

    let rim_pos = focus + Vec3::new(-radius * 1.15, radius * 0.45, -radius * 1.35);
    let rim_dir = (focus + Vec3::new(0.0, radius * 0.08, 0.0) - rim_pos).normalize_or_zero();
    renderer.add_light(SceneLight::spot(
        rim_pos.to_array(),
        rim_dir.to_array(),
        [0.36, 0.55, 1.0],
        15.0,
        radius * 4.4,
        0.48,
        0.88,
    ));

    let bounce_pos = Vec3::new(
        bounds.center.x,
        bounds.floor_y() + radius * 0.18,
        bounds.center.z + radius * 0.55,
    );
    renderer.add_light(SceneLight::point(
        bounce_pos.to_array(),
        [1.0, 0.30, 0.16],
        5.5,
        radius * 2.3,
    ));

    renderer.set_ambient([0.02, 0.025, 0.04], 0.22);
    renderer.set_sky_atmosphere(Some(
        SkyAtmosphere::new()
            .with_sun_intensity(10.0)
            .with_exposure(2.8)
            .with_mie_g(0.78)
            .with_clouds(
                VolumetricClouds::new()
                    .with_coverage(0.18)
                    .with_density(0.45)
                    .with_layer(900.0, 1700.0)
                    .with_wind([0.8, 0.15], 0.03),
            ),
    ));
    renderer.set_skylight(Some(
        Skylight::new()
            .with_intensity(0.05)
            .with_tint([0.68, 0.78, 1.0]),
    ));
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
    start_time: Instant,
    last_frame: Instant,
    movement_speed: f32,

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
        .expect("No adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Embedded Showcase Device"),
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
        .expect("No device");

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps
            .formats
            .iter()
            .find(|format| format.is_srgb())
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
            .with_feature(BloomFeature::new().with_intensity(0.85).with_threshold(0.95))
            .with_feature(ShadowsFeature::new().with_atlas_size(2048).with_max_lights(4))
            .build();

        let mut renderer = Renderer::new(
            device.clone(),
            queue.clone(),
            RendererConfig::new(size.width, size.height, surface_format, feature_registry),
        )
        .expect("Failed to create renderer");

        let (scene, bounds) = match load_embedded_scene() {
            Ok(result) => result,
            Err(error) => {
                log::error!("Failed to load embedded FBX scene: {}", error);
                let fallback_mesh = renderer.create_mesh_cube([0.0, 0.75, 0.0], 0.75);
                renderer.add_object(&fallback_mesh, None, glam::Mat4::IDENTITY);
                let fallback_bounds = SceneBounds {
                    min: Vec3::new(-0.75, 0.0, -0.75),
                    max: Vec3::new(0.75, 1.5, 0.75),
                    center: Vec3::new(0.0, 0.75, 0.0),
                    radius: 3.0,
                };
                add_showcase_stage(&mut renderer, fallback_bounds);
                add_showcase_lighting(&mut renderer, fallback_bounds);
                let camera_start = fallback_bounds.camera_start();
                let focus = fallback_bounds.focus_point();
                let (cam_yaw, cam_pitch) = look_angles(focus - camera_start);
                let now = Instant::now();

                self.state = Some(AppState {
                    window,
                    surface,
                    device,
                    surface_format,
                    renderer,
                    start_time: now,
                    last_frame: now,
                    movement_speed: fallback_bounds.movement_speed(),
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

        log::info!(
            "Loaded embedded '{}' scene with {} meshes, {} materials, {} source lights",
            scene.name,
            scene.meshes.len(),
            scene.materials.len(),
            scene.lights.len()
        );
        if !scene.lights.is_empty() {
            log::info!("Using curated showcase lighting instead of embedded scene lights");
        }

        let gpu_materials: Vec<_> = scene
            .materials
            .iter()
            .enumerate()
            .map(|(index, material)| {
                println!("Uploading material #{}", index);
                renderer.upload_material(material)
            })
            .collect();

        for mesh in &scene.meshes {
            println!("Uploading mesh '{}' with {} vertices and {} indices", mesh.name, mesh.vertices.len(), mesh.indices.len());
            let gpu_mesh = renderer.create_mesh(&mesh.vertices, &mesh.indices);
            let material = mesh
                .material_index
                .and_then(|index| gpu_materials.get(index));
            renderer.add_object(&gpu_mesh, material, glam::Mat4::IDENTITY);
        }

        add_showcase_stage(&mut renderer, bounds);
        add_showcase_lighting(&mut renderer, bounds);

        let camera_start = bounds.camera_start();
        let focus = bounds.focus_point();
        let (cam_yaw, cam_pitch) = look_angles(focus - camera_start);
        let now = Instant::now();

        self.state = Some(AppState {
            window,
            surface,
            device,
            surface_format,
            renderer,
            start_time: now,
            last_frame: now,
            movement_speed: bounds.movement_speed(),
            cam_pos: camera_start,
            cam_yaw,
            cam_pitch,
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
                    let _ = state
                        .window
                        .set_cursor_grab(winit::window::CursorGrabMode::None);
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

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::KeyU),
                        ..
                    },
                ..
            } => {
                let mode = (state.renderer.debug_mode() + 1) % 6;
                state.renderer.set_debug_mode(mode);
                let mode_name = match mode {
                    0 => "Normal (with normal mapping)",
                    1 => "UV Grid",
                    2 => "Texture Direct (G-buffer write)",
                    3 => "Lit without normal mapping",
                    4 => "G-buffer Readback Test",
                    5 => "World Normals (RGB = XYZ remapped)",
                    _ => "Unknown",
                };
                log::info!("Debug mode {}: {}", mode, mode_name);
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
                state.renderer.resize(size.width, size.height);
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
                let dt = now.duration_since(state.last_frame).as_secs_f32();
                state.last_frame = now;

                let forward = state.update_camera(dt);
                let size = state.window.inner_size();
                let aspect = size.width as f32 / size.height.max(1) as f32;

                let camera = Camera::perspective(
                    state.cam_pos,
                    state.cam_pos + forward,
                    Vec3::Y,
                    std::f32::consts::FRAC_PI_4,
                    aspect,
                    0.1,
                    400.0,
                    state.start_time.elapsed().as_secs_f32(),
                );

                let output = match state.surface.get_current_texture() {
                    Ok(texture) => texture,
                    Err(error) => {
                        log::warn!("Surface error: {:?}", error);
                        return;
                    }
                };

                let view = output
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                if let Err(error) = state.renderer.render(&camera, &view, dt) {
                    log::error!("Render error: {:?}", error);
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
