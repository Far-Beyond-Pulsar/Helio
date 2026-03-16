//! Load and display a 3D model from FBX/glTF/OBJ using helio-asset-compat
//!
//! This example demonstrates loading external 3D model files using SolidRS.
//!
//! Usage: Place a file named "test.fbx" in the working directory and run:
//!   cargo run --bin load_fbx

use helio_render_v2::{Renderer, RendererConfig, Camera, ObjectId, LightId};
use helio_render_v2::features::{
    FeatureRegistry, LightingFeature, BloomFeature, ShadowsFeature
};
use helio_render_v2::scene::SceneLight;
use helio_asset_compat::load_scene_file;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};
use std::sync::Arc;
use std::time::Instant;
use std::collections::HashSet;
use glam::Vec3;

fn main() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    log::info!("Helio Asset Loading Example");

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
    objects: Vec<ObjectId>,
    lights: Vec<LightId>,
    start_time: Instant,
    last_frame: Instant,

    // Free-camera state
    cam_pos: Vec3,
    cam_yaw: f32,   // radians, horizontal rotation
    cam_pitch: f32, // radians, vertical rotation (clamped)
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
        // Camera movement (exact same as render_v2_basic)
        const SPEED: f32 = 5.0;
        const LOOK_SENS: f32 = 0.002;

        // Apply mouse look — yaw left/right, pitch up/down (non-inverted)
        self.cam_yaw   += self.mouse_delta.0 * LOOK_SENS;
        self.cam_pitch  = (self.cam_pitch - self.mouse_delta.1 * LOOK_SENS).clamp(-1.5, 1.5);
        self.mouse_delta = (0.0, 0.0);

        // Standard FPS basis: yaw=0 looks down -Z
        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let forward = Vec3::new(sy * cp, sp, -cy * cp);
        let right   = Vec3::new(cy, 0.0, sy);
        let up      = Vec3::Y;

        if self.keys.contains(&KeyCode::KeyW)      { self.cam_pos += forward * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyS)      { self.cam_pos -= forward * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyA)      { self.cam_pos -= right   * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyD)      { self.cam_pos += right   * SPEED * dt; }
        if self.keys.contains(&KeyCode::Space)     { self.cam_pos += up * SPEED * dt; }
        if self.keys.contains(&KeyCode::ShiftLeft) { self.cam_pos -= up * SPEED * dt; }

        forward
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Helio - Asset Loading")
                        .with_inner_size(winit::dpi::PhysicalSize::new(1280, 720)),
                )
                .expect("Failed to create window"),
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
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
                label: Some("Main Device"),
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
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);

        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Create feature registry
        let feature_registry = FeatureRegistry::builder()
            .with_feature(LightingFeature::new())
            .with_feature(BloomFeature::new().with_intensity(0.3).with_threshold(1.5))
            .with_feature(ShadowsFeature::new().with_atlas_size(1024).with_max_lights(4))
            .build();

        let mut renderer = Renderer::new(
            device.clone(),
            queue.clone(),
            RendererConfig::new(size.width, size.height, surface_format, feature_registry),
        )
        .expect("Failed to create renderer");

        let mut objects = Vec::new();
        let mut lights = Vec::new();

        // Load scene from test.fbx
        log::info!("Current directory: {:?}", std::env::current_dir().unwrap());
        log::info!("Looking for test.fbx...");

        // Configure UV handling:
        // The FBX loader already flips V coordinates (DirectX → OpenGL)
        // So we use flip_uv_y = false to avoid double-flipping
        // If textures look wrong, try true (some exporters may vary)
        let config = helio_asset_compat::LoadConfig::default().with_uv_flip(false);

        match helio_asset_compat::load_scene_file_with_config("test.fbx", config) {
            Ok(scene) => {
                log::info!("✓ Loaded '{}'", scene.name);
                log::info!("  {} meshes, {} materials, {} lights",
                    scene.meshes.len(), scene.materials.len(), scene.lights.len());

                // Upload all materials to GPU
                let gpu_materials: Vec<_> = scene.materials.iter()
                    .enumerate()
                    .map(|(idx, mat)| {
                        log::info!("  Uploading material {}: base_color={:?}, metallic={}, roughness={}",
                            idx, mat.base_color, mat.metallic, mat.roughness);
                        renderer.upload_material(mat)
                    })
                    .collect();

                log::info!("  Uploaded {} materials to GPU", gpu_materials.len());

                // Upload all meshes to GPU and register as objects with materials
                for mesh in scene.meshes.iter() {
                    let mat_status = match mesh.material_index {
                        Some(idx) => format!("material {}", idx),
                        None => "NO MATERIAL".to_string(),
                    };

                    log::info!("  Mesh '{}': {} verts, {} indices, {}",
                        mesh.name, mesh.vertices.len(), mesh.indices.len(), mat_status);

                    let gpu_mesh = renderer.create_mesh(&mesh.vertices, &mesh.indices);

                    // Use the mesh's material if it has one
                    let material = mesh.material_index
                        .and_then(|idx| {
                            if idx >= gpu_materials.len() {
                                log::error!("    ⚠️  Material index {} out of bounds (have {} materials)", idx, gpu_materials.len());
                                None
                            } else {
                                Some(&gpu_materials[idx])
                            }
                        });

                    let object_id = renderer.add_object(&gpu_mesh, material, glam::Mat4::IDENTITY);
                    objects.push(object_id);
                }

                // Add all lights from the scene
                for light in scene.lights.iter() {
                    let light_id = renderer.add_light(light.clone());
                    lights.push(light_id);
                    log::debug!("  Added light {:?}", light_id);
                }

                // If no lights in the scene, add default lighting
                if scene.lights.is_empty() {
                    log::info!("  No lights in scene, adding default lighting");

                    let key = renderer.add_light(SceneLight::directional(
                        [-0.5, -1.0, -0.3],
                        [1.0, 0.95, 0.9],
                        3.0,
                    ));
                    let fill = renderer.add_light(SceneLight::directional(
                        [0.5, 0.3, 0.5],
                        [0.6, 0.7, 1.0],
                        0.8,
                    ));
                    lights.push(key);
                    lights.push(fill);
                }

                // Set ambient lighting
                renderer.set_ambient([0.05, 0.05, 0.08], 1.0);

                log::info!("✓ Scene loaded: {} objects, {} lights", objects.len(), lights.len());
            }
            Err(e) => {
                log::error!("Failed to load test.fbx: {}", e);
                log::info!("Place test.fbx in the working directory");
                log::info!("Creating fallback cube for demonstration");

                // Fallback: create a simple cube
                let cube = renderer.create_mesh_cube([0.0, 0.0, 0.0], 0.5);
                let cube_id = renderer.add_object(&cube, None, glam::Mat4::IDENTITY);
                objects.push(cube_id);

                // Add default lights
                let key = renderer.add_light(SceneLight::directional(
                    [-0.5, -1.0, -0.3],
                    [1.0, 0.95, 0.9],
                    3.0,
                ));
                lights.push(key);

                renderer.set_ambient([0.05, 0.05, 0.08], 1.0);
            }
        }

        let now = Instant::now();
        self.state = Some(AppState {
            window,
            surface,
            device,
            surface_format,
            renderer,
            objects,
            lights,
            start_time: now,
            last_frame: now,

            // Start camera looking at the scene
            cam_pos: Vec3::new(0.0, 2.0, 7.0),
            cam_yaw: 0.0,         // yaw=0 looks down -Z
            cam_pitch: -0.2,      // slight downward angle
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            // Escape key: release cursor or exit
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
                    let _ = state.window.set_cursor_grab(winit::window::CursorGrabMode::None);
                } else {
                    event_loop.exit();
                }
            }

            // F3: Toggle debug visualization
            WindowEvent::KeyboardInput {
                event: KeyEvent {
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

            // U: Toggle UV debug mode
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    state: ElementState::Pressed,
                    physical_key: PhysicalKey::Code(KeyCode::KeyU),
                    ..
                },
                ..
            } => {
                let mode = (state.renderer.debug_mode() + 1) % 3;
                state.renderer.set_debug_mode(mode);
                let mode_name = match mode {
                    0 => "Normal",
                    1 => "UV Grid",
                    2 => "Texture Direct",
                    _ => "Unknown",
                };
                log::info!("Debug mode: {} ({})", mode, mode_name);
            }

            WindowEvent::Resized(size) => {
                let config = wgpu::SurfaceConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format: state.surface_format,
                    width: size.width,
                    height: size.height,
                    present_mode: wgpu::PresentMode::Fifo,
                    alpha_mode: wgpu::CompositeAlphaMode::Opaque,
                    view_formats: vec![],
                    desired_maximum_frame_latency: 2,
                };
                state.surface.configure(&state.device, &config);
                state.renderer.resize(size.width, size.height);
            }

            // Keyboard input for movement
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    physical_key: PhysicalKey::Code(code),
                    state: key_state,
                    ..
                },
                ..
            } => {
                match key_state {
                    ElementState::Pressed => { state.keys.insert(code); }
                    ElementState::Released => { state.keys.remove(&code); }
                }
            }

            // Mouse button — grab cursor on click
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if !state.cursor_grabbed {
                    // Try confined first, fall back to locked
                    let grabbed = state.window.set_cursor_grab(winit::window::CursorGrabMode::Confined)
                        .or_else(|_| state.window.set_cursor_grab(winit::window::CursorGrabMode::Locked))
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

                // Update camera from mouse and keyboard input (returns forward vector)
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
                    200.0,
                    state.start_time.elapsed().as_secs_f32(),
                );

                // Acquire surface texture
                let output = match state.surface.get_current_texture() {
                    Ok(t) => t,
                    Err(e) => {
                        log::warn!("Surface error: {:?}", e);
                        return;
                    }
                };

                let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

                // Render
                if let Err(e) = state.renderer.render(&camera, &view, dt) {
                    log::error!("Render error: {:?}", e);
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
