//! AO and AA Demo
//! 
//! Demonstrates ambient occlusion and anti-aliasing features.
//! Press 1-5 to cycle through AA modes.
//! Press A to toggle SSAO on/off.



mod demo_portal;

use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};


use std::sync::Arc;


use std::collections::HashSet;


use helio_render_v2::{
    Renderer, RendererConfig,
    passes::{AntiAliasingMode, SsaoConfig, MsaaSamples},
    features::{FeatureRegistry, LightingFeature, ShadowsFeature},
    mesh::GpuMesh,
    camera::Camera,
    SceneLight, LightId,
};


use glam::{Vec3, Mat4};

struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    surface_format: wgpu::TextureFormat,
    renderer: Renderer,

    // Geometry
    floor: GpuMesh,
    spheres: Vec<GpuMesh>,
    cubes: Vec<GpuMesh>,

    // Scene state
    light_ids: Vec<LightId>,

    // Camera
    cam_pos: Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f64, f64),

    // Settings
    aa_mode: AntiAliasingMode,
    ssao_enabled: bool,

    last_frame: std::time::Instant,
}

struct App {
    state: Option<AppState>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }

        let window_attrs = Window::default_attributes()
            .with_title("Helio AO & AA Demo")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));
        let window = Arc::new(event_loop.create_window(window_attrs).unwrap());

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })).expect("adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_features: wgpu::Features::TIMESTAMP_QUERY 
                    | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: unsafe { wgpu::ExperimentalFeatures::disabled() },
                trace: wgpu::Trace::Off,
            },
        )).expect("device");

        device.on_uncaptured_error(std::sync::Arc::new(|e| {
            panic!("[GPU UNCAPTURED ERROR] {:?}", e);
        }));
        let info = adapter.get_info();
        println!("[WGPU] Backend: {:?}, Device: {}, Driver: {}", info.backend, info.name, info.driver);
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().find(|f| f.is_srgb()).copied().unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        surface.configure(&device, &wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        });

        // Build features
        let features = FeatureRegistry::builder()
            .with_feature(LightingFeature::new())
            .with_feature(ShadowsFeature::new().with_atlas_size(2048).with_max_lights(4))
            .build();

        // Create renderer with SSAO and TAA enabled
        let config = RendererConfig::new(size.width, size.height, format, features)
            .with_ssao_config(SsaoConfig {
                radius: 0.5,
                bias: 0.025,
                power: 2.0,
                samples: 16,
            })
            .with_aa(AntiAliasingMode::Taa);

        let mut renderer = Renderer::new(device.clone(), queue.clone(), config)
            .expect("Failed to create renderer");

        // Create geometry - a Cornell box-like scene with various objects
        let floor = renderer.create_mesh_plane([0.0, 0.0, 0.0], 10.0);
        
        // Create spheres at different heights to showcase AO
        let spheres = vec![
            renderer.create_mesh_sphere([0.0, 0.5, 0.0], 0.5, 32),
            renderer.create_mesh_sphere([-2.0, 0.3, 1.0], 0.3, 24),
            renderer.create_mesh_sphere([2.0, 0.4, -1.0], 0.4, 24),
        ];
        
        // Create cubes
        let cubes = vec![
            renderer.create_mesh_cube([-1.5, 0.25, -1.5], 0.5),
            renderer.create_mesh_cube([1.5, 0.5, 1.5], 1.0),
        ];
        demo_portal::enable_live_dashboard(&mut renderer);

        renderer.add_object(&floor, None, glam::Mat4::IDENTITY);
        for sphere in &spheres { renderer.add_object(sphere, None, glam::Mat4::IDENTITY); }
        for cube   in &cubes   { renderer.add_object(cube,   None, glam::Mat4::IDENTITY); }

        let mut light_ids = Vec::new();
        light_ids.push(renderer.add_light(SceneLight::point([0.0,  3.0,  0.0], [1.0, 0.95, 0.9], 10.0, 8.0)));
        light_ids.push(renderer.add_light(SceneLight::point([-3.0, 1.5, -3.0], [1.0, 0.3,  0.3],  5.0, 5.0)));
        light_ids.push(renderer.add_light(SceneLight::point([ 3.0, 1.5,  3.0], [0.3, 0.5,  1.0],  5.0, 5.0)));
        renderer.set_ambient([0.05, 0.05, 0.08], 1.0);

        self.state = Some(AppState {
            window,
            surface,
            device,
            surface_format: format,
            renderer,
            floor,
            spheres,
            cubes,
            light_ids,
            cam_pos: Vec3::new(0.0, 3.0, 8.0),
            cam_yaw: 0.0,
            cam_pitch: -0.3,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            aa_mode: AntiAliasingMode::Taa,
            ssao_enabled: true,
            last_frame: std::time::Instant::now(),
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                let dt = (now - state.last_frame).as_secs_f32();
                state.last_frame = now;

                // Update camera from input
                let speed = 3.0 * dt;
                let forward = Vec3::new(state.cam_yaw.cos(), 0.0, -state.cam_yaw.sin());
                let right = Vec3::new(forward.z, 0.0, -forward.x);

                if state.keys.contains(&KeyCode::KeyW) { state.cam_pos += forward * speed; }
                if state.keys.contains(&KeyCode::KeyS) { state.cam_pos -= forward * speed; }
                if state.keys.contains(&KeyCode::KeyA) { state.cam_pos -= right * speed; }
                if state.keys.contains(&KeyCode::KeyD) { state.cam_pos += right * speed; }
                if state.keys.contains(&KeyCode::Space) { state.cam_pos.y += speed; }
                if state.keys.contains(&KeyCode::ShiftLeft) { state.cam_pos.y -= speed; }

                if state.cursor_grabbed {
                    state.cam_yaw += (state.mouse_delta.0 as f32) * 0.002;
                    state.cam_pitch += (state.mouse_delta.1 as f32) * 0.002;
                    state.cam_pitch = state.cam_pitch.clamp(-1.5, 1.5);
                    state.mouse_delta = (0.0, 0.0);
                }

                // Build camera
                let view = Mat4::look_at_rh(
                    state.cam_pos,
                    state.cam_pos + Vec3::new(
                        state.cam_yaw.cos() * state.cam_pitch.cos(),
                        state.cam_pitch.sin(),
                        -state.cam_yaw.sin() * state.cam_pitch.cos(),
                    ),
                    Vec3::Y,
                );
                let size = state.window.inner_size();
                let aspect = size.width as f32 / size.height as f32;
                let proj = Mat4::perspective_rh(60.0_f32.to_radians(), aspect, 0.1, 100.0);
                let camera = Camera::from_matrices(view, proj, state.cam_pos);

                // Scene state is persistent — no per-frame setup needed.

                let frame = state.surface.get_current_texture().unwrap();
                let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
                state.renderer.render(&camera, &view, dt).ok();
                frame.present();

                state.window.request_redraw();
            }
            WindowEvent::KeyboardInput { event: KeyEvent { physical_key, state: key_state, .. }, .. } => {
                if let PhysicalKey::Code(code) = physical_key {
                    match key_state {
                        ElementState::Pressed => {
                            state.keys.insert(code);
                            
                            // Toggle settings
                            match code {
                                KeyCode::Escape => {
                                    if state.cursor_grabbed {
                                        state.cursor_grabbed = false;
                                        let _ = state.window.set_cursor_grab(winit::window::CursorGrabMode::None);
                                        state.window.set_cursor_visible(true);
                                    } else {
                                        event_loop.exit();
                                    }
                                }
                                KeyCode::KeyA => {
                                    state.ssao_enabled = !state.ssao_enabled;
                                    println!("SSAO: {}", if state.ssao_enabled { "ON" } else { "OFF" });
                                }
                                KeyCode::Digit1 => {
                                    state.aa_mode = AntiAliasingMode::None;
                                    println!("AA: None");
                                }
                                KeyCode::Digit2 => {
                                    state.aa_mode = AntiAliasingMode::Fxaa;
                                    println!("AA: FXAA");
                                }
                                KeyCode::Digit3 => {
                                    state.aa_mode = AntiAliasingMode::Smaa;
                                    println!("AA: SMAA");
                                }
                                KeyCode::Digit4 => {
                                    state.aa_mode = AntiAliasingMode::Taa;
                                    println!("AA: TAA");
                                }
                                KeyCode::Digit5 => {
                                    state.aa_mode = AntiAliasingMode::Msaa(MsaaSamples::X4);
                                    println!("AA: MSAA 4x");
                                }
                                _ => {}
                            }
                        }
                        ElementState::Released => { state.keys.remove(&code); }
                    }
                }
            }
            WindowEvent::MouseInput { state: ElementState::Pressed, .. } => {
                if !state.cursor_grabbed {
                    state.cursor_grabbed = true;
                    let _ = state.window.set_cursor_grab(winit::window::CursorGrabMode::Confined);
                    state.window.set_cursor_visible(false);
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

fn main() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App { state: None };
    event_loop.run_app(&mut app).unwrap();
}
